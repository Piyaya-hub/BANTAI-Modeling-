"""Regress diagnostic identities, not a replacement training objective."""
import copy
import importlib.util
from types import SimpleNamespace
import unittest

import torch

from scripts.loss_investigation import (targets_for_case, selected_encoder_tensors,
    focal_decomposition, gradient_paths, make_criterion)
from scripts.training_smoke import synthetic_batch


class DiagnosticHelpersTests(unittest.TestCase):
    def test_target_count_controls_preserve_source(self):
        images,targets=synthetic_batch(2)
        try:
            original=copy.deepcopy(targets)
            expected={'baseline_11':[8,3],'two_targets':[1,1],'one_target':[1,0],'empty':[0,0]}
            for case,counts in expected.items():
                self.assertEqual([len(t['annotations']) for t in targets_for_case(targets,case)],counts)
            self.assertEqual(targets,original)
            with self.assertRaises(ValueError): targets_for_case(targets,'real_data')
        finally:
            for image in images:image.close()

    def test_selected_coordinates_keep_expected_detach_boundary(self):
        scores=torch.tensor([[[.1,.2],[1.,.3],[.2,.4]]],requires_grad=True)
        coords=torch.zeros(1,3,4,requires_grad=True)
        selected,boxes,indices=selected_encoder_tensors(SimpleNamespace(enc_outputs_class=scores,enc_outputs_coord_logits=coords),2)
        self.assertEqual(indices.tolist(),[[1,2]])
        self.assertTrue(boxes.requires_grad)
        decoder_reference=boxes.detach()
        self.assertTrue(torch.equal(boxes,decoder_reference))
        self.assertFalse(decoder_reference.requires_grad)
        grad=torch.autograd.grad(boxes.sum(),coords)[0]
        self.assertEqual(grad[0,0].abs().sum(),0)
        self.assertGreater(grad[0,1:].abs().sum(),0)
        self.assertTrue(selected.requires_grad)

    def test_focal_normalizer_and_mask_are_accounted_for(self):
        logits=torch.tensor([[[0.,0.,float('-inf')],[0.,0.,float('-inf')]]])
        targets=torch.tensor([[[1,0,0],[0,0,0]]])
        mask=torch.isfinite(logits)
        a=focal_decomposition(logits,targets,mask,1)
        b=focal_decomposition(logits,targets,mask,2)
        expected=torch.log(torch.tensor(2.)).item()*.25*(.25+3*.75)
        self.assertAlmostEqual(a['total'],expected,places=6)
        self.assertAlmostEqual(a['positive']+a['negative'],a['total'],places=6)
        self.assertAlmostEqual(b['total']*2,a['total'],places=6)
        self.assertEqual(a['per_token'][2],0)

    def test_gradient_audit_does_not_accumulate_update_gradients(self):
        p=torch.nn.Parameter(torch.tensor([2.]))
        q=torch.nn.Parameter(torch.tensor([3.]))
        audit=gradient_paths((p*p).sum(),{'connected':p,'unused':q})
        self.assertEqual(audit['parameters']['connected']['l2'],4.)
        self.assertIsNone(audit['parameters']['unused'])
        self.assertIsNone(p.grad);self.assertIsNone(q.grad)
        self.assertFalse(gradient_paths(torch.tensor(0.),{'p':p})['loss_requires_grad'])


@unittest.skipUnless(importlib.util.find_spec('scipy'),'Native diagnostic fixtures require isolated SciPy')
class NativeCriterionTests(unittest.TestCase):
    def test_detach_preserves_regression_value_but_blocks_head_training(self):
        from transformers import GroundingDinoConfig
        criterion=make_criterion(GroundingDinoConfig())
        head=torch.nn.Linear(4,4)
        features=torch.ones(1,2,4)
        boxes=head(features).sigmoid()
        logits=torch.zeros(1,2,4,requires_grad=True)
        labels=[{'class_labels':torch.tensor([0]),'boxes':torch.tensor([[.3,.4,.2,.2]])}]
        shared={'logits':logits,'label_maps':(torch.eye(4),),'text_mask':torch.ones_like(logits,dtype=torch.bool)}
        detached=criterion(dict(shared,pred_boxes=boxes.detach()),labels)
        connected=criterion(dict(shared,pred_boxes=boxes),labels)
        for key in ('loss_bbox','loss_giou'):
            self.assertTrue(torch.equal(detached[key],connected[key]))
            self.assertFalse(detached[key].requires_grad)
            self.assertTrue(connected[key].requires_grad)
        grad=torch.autograd.grad(connected['loss_bbox']+connected['loss_giou'],head.weight)[0]
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0)

    def test_unmatched_large_positive_logits_explain_large_focal_loss(self):
        mask=torch.ones(1,900,30,dtype=torch.bool)
        onehot=torch.zeros_like(mask)
        saturated=focal_decomposition(torch.full(mask.shape,50.),onehot,mask,1)
        negative=focal_decomposition(torch.full(mask.shape,-5.),onehot,mask,1)
        self.assertAlmostEqual(saturated['total'],900*30*.75*50,places=2)
        self.assertGreater(saturated['total'],negative['total']*1000)

    def test_two_images_and_fully_empty_targets_supported(self):
        from transformers import GroundingDinoConfig
        criterion=make_criterion(GroundingDinoConfig())
        logits=torch.zeros(2,3,4,requires_grad=True)
        boxes=torch.full((2,3,4),.25,requires_grad=True)
        labels=[{'class_labels':torch.empty(0,dtype=torch.int64),'boxes':torch.empty(0,4)} for _ in range(2)]
        outputs={'logits':logits,'pred_boxes':boxes,'label_maps':(torch.eye(4),torch.eye(4)),
                 'text_mask':torch.ones_like(logits,dtype=torch.bool)}
        losses=criterion(outputs,labels)
        self.assertEqual(float(losses['loss_bbox']),0.)
        self.assertEqual(float(losses['loss_giou']),0.)
        self.assertTrue(torch.isfinite(losses['loss_ce']))
        losses['loss_ce'].backward()
        self.assertGreater(float(logits.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
