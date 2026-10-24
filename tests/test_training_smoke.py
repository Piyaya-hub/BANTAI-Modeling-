"""Software fixtures; no real photographs, pretrained model construction or training."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import GroundingDinoImageProcessor, AutoTokenizer, GroundingDinoProcessor, GroundingDinoConfig

from scripts.training_smoke import (LABELS, PROMPT, SNAPSHOT, checked_loss, loading_ok, move_batch,
    prepare_batch, require_training_environment, state_hash, synthetic_batch, validate_targets, validate_taxonomy,
    validate_encoder_capacity,validate_predictions,main)


class SyntheticTargetsTests(unittest.TestCase):
    def setUp(self):
        self.images,self.targets = synthetic_batch()

    def tearDown(self):
        for image in self.images: image.close()

    def test_taxonomy_order_and_synthetic_coverage(self):
        self.assertEqual([r['name'] for r in validate_taxonomy()['labels']],list(LABELS))
        validate_targets(self.images,self.targets)
        self.assertEqual(sorted(o['category_id'] for o in self.targets[0]['annotations']),list(range(8)))
        self.assertEqual([len(t['annotations']) for t in self.targets],[8,3])
        self.assertTrue(PROMPT.endswith('.'))

    def test_malformed_targets_fail_before_processor_can_clip(self):
        cases = ([1,2,0,3],[-1,2,3,4],[319,239,4,4],[1,2,float('nan'),4],[1,2,3])
        for box in cases:
            target = copy.deepcopy(self.targets); target[0]['annotations'][0]['bbox'] = box
            with self.subTest(box=box),self.assertRaises(ValueError): validate_targets(self.images,target)
        for key,value in [('category_id',8),('iscrowd',1),('ignore',1),('area',-1),('image_id',999)]:
            target = copy.deepcopy(self.targets); target[0]['annotations'][0][key] = value
            with self.subTest(key=key),self.assertRaises(ValueError): validate_targets(self.images,target)

    def test_empty_targets_are_valid_but_not_real_negative_evidence(self):
        for target in self.targets: target['annotations'] = []
        validate_targets(self.images,self.targets)

    def test_inference_environment_is_rejected(self):
        with patch.object(sys,'prefix',str(Path('.venv').resolve())):
            with self.assertRaisesRegex(ValueError,'inference .venv is prohibited'): require_training_environment()

    def test_state_hash_and_batch_dtype_preservation(self):
        model = torch.nn.Linear(2,1)
        model.bias.requires_grad_(False)
        frozen = state_hash(model,True); all_state = state_hash(model)
        with torch.no_grad(): model.weight.add_(1)
        self.assertEqual(state_hash(model,True),frozen)
        self.assertNotEqual(state_hash(model),all_state)
        moved = move_batch({'input_ids':torch.tensor([[1,2]]),'labels':[{'class_labels':torch.tensor([7]),'boxes':torch.zeros(1,4)}]},'cpu')
        self.assertEqual(moved['input_ids'].dtype,torch.int64)
        self.assertEqual(moved['labels'][0]['class_labels'].dtype,torch.int64)

    def test_load_mismatches_are_not_ignored(self):
        loading_ok({'missing_keys':[],'unexpected_keys':[],'mismatched_keys':[],'error_msgs':[]})
        with self.assertRaises(ValueError): loading_ok({'missing_keys':['model.head']})

    def test_small_input_does_not_have_900_encoder_positions(self):
        with self.assertRaisesRegex(ValueError,'encoder positions'): validate_encoder_capacity(170,170)
        self.assertEqual(validate_encoder_capacity(256,256),1360)
        self.assertEqual(validate_encoder_capacity(192,256),1020)

    def test_sentinels_do_not_hide_invalid_predictions(self):
        mask = torch.tensor([[1,1,0]])
        logits = torch.tensor([[[.1,.2,float('-inf'),float('-inf')]]])
        boxes = torch.full((1,1,4),.5)
        validate_predictions(logits,boxes,mask)
        for value in (float('nan'),float('inf'),float('-inf')):
            invalid = logits.clone(); invalid[0,0,0] = value
            with self.assertRaises(ValueError): validate_predictions(invalid,boxes,mask)
        invalid = logits.clone(); invalid[0,0,2] = 0
        with self.assertRaises(ValueError): validate_predictions(invalid,boxes,mask)
        with self.assertRaises(ValueError): validate_predictions(logits,boxes*3,mask)

    def test_cli_has_no_photo_input_or_unbounded_steps(self):
        for args in (['--input','data/input'],['--steps','6'],['--device','auto']):
            with patch('sys.stderr'),self.assertRaises(SystemExit) as raised: main(args)
            self.assertEqual(raised.exception.code,2)

    @unittest.skipUnless((SNAPSHOT/'tokenizer_config.json').exists(),'Pinned cached tokenizer unavailable')
    def test_processor_padding_geometry_spans_batch_one_two_and_empty(self):
        processor = GroundingDinoProcessor(image_processor=GroundingDinoImageProcessor(),
            tokenizer=AutoTokenizer.from_pretrained(SNAPSHOT,local_files_only=True))
        original = copy.deepcopy(self.targets)
        for count in (1,2):
            batch, evidence = prepare_batch(processor,self.images[:count],self.targets[:count])
            self.assertEqual(len(batch['labels']),count)
            self.assertEqual(evidence['token_count'],30)
            for row in evidence['spans']: self.assertEqual(row,[label.replace('_',' ') for label in LABELS])
            self.assertEqual(evidence['geometry'][0]['original_wh'],[320,240])
            # Opposite aspect ratios force actual batch padding; tiny edge box survives.
            if count==2:
                self.assertLess(evidence['geometry'][0]['resized_wh'][1],evidence['geometry'][0]['padded_wh'][1])
                self.assertLess(evidence['geometry'][1]['resized_wh'][0],evidence['geometry'][1]['padded_wh'][0])
            self.assertEqual(batch['labels'][0]['boxes'].shape,(8,4))
        self.assertEqual(self.targets,original)
        empty = copy.deepcopy(self.targets)
        empty[1]['annotations'] = []
        batch,_ = prepare_batch(processor,self.images,empty)
        self.assertEqual(batch['labels'][1]['boxes'].shape,(0,4))


@unittest.skipUnless(importlib.util.find_spec('scipy'),'SciPy absent: run loss fixtures in isolated training environment')
class NativeLossTests(unittest.TestCase):
    def test_native_matcher_loss_values_gradients_and_empty(self):
        from transformers.loss.loss_grounding_dino import GroundingDinoForObjectDetectionLoss
        config = GroundingDinoConfig(two_stage=False,auxiliary_loss=False)
        # Eight disjoint class spans plus two masked tokens; each class gets one target.
        logits = torch.zeros(2,8,10,requires_grad=True)
        maps = (torch.eye(8,10),torch.eye(8,10))
        boxes = torch.full((2,8,4),.25,requires_grad=True)
        labels = [{'class_labels':torch.arange(8),'boxes':torch.full((8,4),.25)},
                  {'class_labels':torch.empty(0,dtype=torch.int64),'boxes':torch.empty(0,4)}]
        mask = torch.ones_like(logits,dtype=torch.bool); mask[:,:,8:] = False
        total,components,_ = GroundingDinoForObjectDetectionLoss(logits,labels,logits.device,boxes,config,maps,mask)
        result = SimpleNamespace(loss=total,loss_dict=components)
        audited = checked_loss(result,config)
        self.assertAlmostEqual(audited['loss_bbox'],0.)
        self.assertAlmostEqual(audited['loss_giou'],0.)
        # Focal alpha .25/gamma 2: 8 positives, 120 negatives / 8 target boxes.
        expected_ce = torch.log(torch.tensor(2.)).item()*.25*(8*.25+120*.75)/8
        self.assertAlmostEqual(audited['loss_ce'],expected_ce,places=5)
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0.)
        self.assertEqual(float(logits.grad[:,:,8:].abs().sum()),0.)
        self.assertTrue(torch.isfinite(boxes.grad).all())

    def test_missing_or_nonfinite_loss_fails(self):
        config = GroundingDinoConfig(two_stage=False,auxiliary_loss=False)
        for value in (None,torch.tensor(float('nan'))):
            with self.assertRaises(ValueError): checked_loss(SimpleNamespace(loss=value,loss_dict={}),config)
        with self.assertRaises(ValueError): checked_loss(SimpleNamespace(loss=torch.tensor(1.),loss_dict={}),config)


if __name__ == '__main__': unittest.main()
