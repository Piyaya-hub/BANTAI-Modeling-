"""Standalone adapter regression fixtures; never patch installed Transformers."""
import copy
import importlib.util
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from scripts.training_adapter import (EncoderOutputAdapter,EXPECTED_CONFIG,config_signature,
    selected_proposals,validate_labels,adapter_source_digest)
from scripts.training_smoke import SNAPSHOT,PROMPT
from scripts.qualify_training_adapter import main,compare_predictions


class ToyModel(torch.nn.Module):
    """Small tensor graph with pinned output shapes, not a pretrained model."""
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(**EXPECTED_CONFIG)
        self.proposal_logits=torch.nn.Parameter(torch.linspace(-2.,-1.,901)[:,None].expand(901,30).clone())
        self.proposal_coordinates=torch.nn.Parameter(torch.full((901,4),-1.))
        self.decoder_logits=torch.nn.Parameter(torch.full((900,30),-2.))
        self.decoder_coordinates=torch.nn.Parameter(torch.full((900,4),-1.))
        self.detach_encoder=False;self.break_reference=False
        self.received_labels=None

    def forward(self,input_ids,attention_mask,**kwargs):
        self.received_labels='labels' in kwargs
        count=len(input_ids)
        tail=torch.full((count,901,226),float('-inf'))
        encoder=torch.cat([self.proposal_logits[None].expand(count,-1,-1),tail],dim=-1)
        coords=self.proposal_coordinates[None].expand(count,-1,-1)
        if self.detach_encoder:coords=coords.detach()
        ids=encoder.max(-1).values.topk(900,dim=1).indices
        selected=coords.gather(1,ids[:,:,None].expand(-1,-1,4)).sigmoid()
        decoder_tail=torch.full((count,900,226),float('-inf'))
        return SimpleNamespace(logits=torch.cat([self.decoder_logits[None].expand(count,-1,-1),decoder_tail],dim=-1),
            pred_boxes=self.decoder_coordinates.sigmoid()[None].expand(count,-1,-1),
            enc_outputs_class=encoder,enc_outputs_coord_logits=coords,encoder_pred_boxes=selected.detach(),
            init_reference_points=selected if self.break_reference else selected.detach())


class AdapterValidationTests(unittest.TestCase):
    def test_source_identity_survives_only_line_ending_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            first,second=Path(folder)/'lf.py',Path(folder)/'crlf.py'
            first.write_bytes(b'x = 1\n');second.write_bytes(b'x = 1\r\n')
            self.assertEqual(adapter_source_digest(first),adapter_source_digest(second))
            second.write_bytes(b'x = 2\r\n')
            self.assertNotEqual(adapter_source_digest(first),adapter_source_digest(second))

    def test_config_change_is_rejected_not_silently_adopted(self):
        for key,value in [('num_queries',300),('bbox_loss_coefficient',1.),('auxiliary_loss',True),('two_stage',False)]:
            config=SimpleNamespace(**dict(EXPECTED_CONFIG,**{key:value}))
            with self.subTest(key=key),self.assertRaises(ValueError):config_signature(config)

    def test_normalized_target_contract_and_empty(self):
        empty=[{'class_labels':torch.empty(0,dtype=torch.int64),'boxes':torch.empty(0,4)}]
        validate_labels(empty,1)
        for ids,box in [(torch.tensor([8]),[[.5,.5,.2,.2]]),(torch.tensor([0],dtype=torch.int32),[[.5,.5,.2,.2]]),
                        (torch.tensor([0]),[[.9,.9,.5,.5]]),(torch.tensor([0]),[[.5,.5,0.,.2]]),
                        (torch.tensor([0]),[[float('nan'),.5,.2,.2]])]:
            with self.assertRaises(ValueError):validate_labels([{'class_labels':ids,'boxes':torch.tensor(box)}],1)

    def test_detached_decoder_invariant_is_enforced(self):
        model=ToyModel();model.break_reference=True
        with self.assertRaisesRegex(ValueError,'detach boundary'):
            selected_proposals(model(torch.ones(1,30,dtype=torch.long),torch.ones(1,30)),900)

    def test_decoder_comparison_rejects_a_changed_box(self):
        left=SimpleNamespace(logits=torch.zeros(1,1,2),pred_boxes=torch.zeros(1,1,4))
        right=SimpleNamespace(logits=left.logits.clone(),pred_boxes=torch.ones(1,1,4))
        with self.assertRaises(ValueError):compare_predictions(left,right)

    def test_cli_no_real_inputs_and_step_bounds(self):
        for args in (['--input','data/input'],['--steps','4'],['--cases','empty']):
            with patch('sys.stderr'),self.assertRaises(SystemExit) as raised:main(args)
            self.assertEqual(raised.exception.code,2)


@unittest.skipUnless(importlib.util.find_spec('scipy') and (SNAPSHOT/'tokenizer_config.json').exists(),
                     'Use isolated environment with cached tokenizer and SciPy')
class AdapterForwardTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer=AutoTokenizer.from_pretrained(SNAPSHOT,local_files_only=True)
        self.model=ToyModel()
        self.adapter=EncoderOutputAdapter(self.model,self.tokenizer)
        tokens=self.tokenizer([PROMPT],return_tensors='pt')
        self.batch={'input_ids':tokens['input_ids'],'attention_mask':tokens['attention_mask'],
            'labels':[{'class_labels':torch.tensor([0]),'boxes':torch.tensor([[.25,.25,.2,.2]])}]}

    def test_existing_loss_routing_and_encoder_gradients(self):
        from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps,build_text_mask
        from transformers.loss.loss_grounding_dino import GroundingDinoForObjectDetectionLoss
        result=self.adapter(self.model,self.batch)
        raw=result.outputs
        expected,components,_=GroundingDinoForObjectDetectionLoss(raw.logits,self.batch['labels'],raw.logits.device,
            raw.pred_boxes,self.model.config,build_label_maps(raw.logits,self.batch['input_ids']),
            build_text_mask(raw.logits,self.batch['attention_mask']),encoder_logits=result.encoder_logits,encoder_pred_boxes=result.encoder_boxes)
        self.assertTrue(torch.equal(result.loss,expected))
        self.assertTrue(all(torch.equal(v,components[k]) for k,v in result.loss_dict.items()))
        self.assertFalse(self.model.received_labels)
        self.assertFalse(raw.init_reference_points.requires_grad)
        self.assertTrue(torch.equal(raw.init_reference_points,result.encoder_boxes))
        result.loss.backward()
        self.assertGreater(float(self.model.proposal_logits.grad.norm()),0)
        self.assertGreater(float(self.model.proposal_coordinates.grad.norm()),0)

    def test_forward_preserves_parameters_decoder_outputs_and_inputs(self):
        state={k:v.clone() for k,v in self.model.state_dict().items()}
        targets=copy.deepcopy(self.batch['labels'])
        direct=self.model(**{k:v for k,v in self.batch.items() if k!='labels'})
        result=self.adapter(self.model,self.batch)
        compare_predictions(direct,result.outputs)
        for k,v in self.model.state_dict().items():self.assertTrue(torch.equal(v,state[k]))
        for k,v in self.batch['labels'][0].items():self.assertTrue(torch.equal(v,targets[0][k]))

    def test_empty_targets_have_finite_classification_zero_regression(self):
        self.batch['labels']=[{'class_labels':torch.empty(0,dtype=torch.int64),'boxes':torch.empty(0,4)}]
        result=self.adapter(self.model,self.batch)
        self.assertEqual(float(result.loss_dict['loss_bbox_enc']),0)
        self.assertEqual(float(result.loss_dict['loss_giou_enc']),0)
        result.loss.backward()
        self.assertGreater(float(self.model.proposal_logits.grad.norm()),0)
        self.assertEqual(float(self.model.proposal_coordinates.grad.norm()),0)

    def test_manifest_rejects_changed_version_mapping_source_or_configuration(self):
        manifest=self.adapter.manifest(self.model)
        self.adapter.verify_manifest(self.model,manifest)
        for key in ('adapter_version','prompt','adapter_source_sha256','configuration'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.adapter.verify_manifest(self.model,dict(manifest,**{key:'changed'}))

    def test_changed_text_or_detached_encoder_path_fails(self):
        changed=copy.deepcopy(self.batch);changed['input_ids'][0,1]=0
        with self.assertRaisesRegex(ValueError,'Prompt'):self.adapter(self.model,changed)
        self.model.detach_encoder=True
        with self.assertRaisesRegex(ValueError,'regression path'):self.adapter(self.model,self.batch)


if __name__=='__main__':unittest.main()
