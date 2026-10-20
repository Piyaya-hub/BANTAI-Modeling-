"""Independent native-frame oracles; cached processor only, no weights/updates."""
import copy
import importlib.util
from types import SimpleNamespace
import unittest

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from transformers.models.grounding_dino.modeling_grounding_dino import (
    GroundingDinoModel, build_label_maps)

from scripts.training_smoke import SNAPSHOT, LABELS, prepare_batch
from scripts.taco_training_data import collate
from scripts.taco_evaluation import decode_predictions, coco_metrics


@unittest.skipUnless((SNAPSHOT/'preprocessor_config.json').exists(), 'Pinned processor unavailable')
class NativeCoordinateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True)

    def sample(self, size, objects):
        image = Image.new('RGB', size, 'black')
        self.addCleanup(image.close)
        target = {'image_id': 1, 'annotations': [
            {'id': i+1, 'image_id': 1, 'category_id': category, 'bbox': box,
             'area': box[2]*box[3], 'iscrowd': 0}
            for i, (category, box) in enumerate(objects)]}
        return image, target

    def outputs(self, batch, boxes, classes):
        logits = torch.full((1, 900, 256), float('-inf'))
        logits[:, :, :batch['input_ids'].shape[1]] = -30
        mapping = build_label_maps(logits, batch['input_ids'])[0]
        predictions = torch.zeros(1, 900, 4)  # Remaining queries invalid, not thresholded.
        for q, (box, category) in enumerate(zip(boxes, classes)):
            predictions[0, q] = torch.tensor(box)
            logits[0, q, mapping[category].bool()] = 10
        return SimpleNamespace(logits=logits, pred_boxes=predictions)

    def test_landscape_portrait_and_padded_square_targets(self):
        # Independent original boxes share this fixed native-frame answer.
        for size, box, resized in [((640,320), [160,80,160,80], [384,192]),
                                   ((320,640), [80,160,80,160], [192,384]),
                                   ((640,640), [160,160,160,160], [256,256])]:
            with self.subTest(size=size):
                image, target = self.sample(size, [(0, box)])
                original = copy.deepcopy(target)
                batch, frames = collate(self.processor, [(image,target)])
                self.assertEqual(frames[0]['resized_wh'], resized)
                self.assertEqual(frames[0]['padded_wh'], [384,384])
                self.assertEqual(int(batch['pixel_mask'].sum()), resized[0]*resized[1])
                torch.testing.assert_close(batch['labels'][0]['boxes'],
                    torch.tensor([[.375,.375,.25,.25]]))
                self.assertEqual(target, original)
                # Smoke preprocessing uses dynamic batch padding: same native answer.
                smoke, _ = prepare_batch(self.processor, [image], [target], 256, 384)
                torch.testing.assert_close(smoke['labels'][0]['boxes'],
                    torch.tensor([[.375,.375,.25,.25]]))

    def test_decode_matches_native_postprocessor_and_literal_original_boxes(self):
        for size, expected in [((640,320), [160,80,160,80]),
                               ((320,640), [80,160,80,160]),
                               ((640,640), [160,160,160,160])]:
            with self.subTest(size=size):
                image,target = self.sample(size, [(0,expected)])
                batch,frames = collate(self.processor, [(image,target)])
                out = self.outputs(batch, [[.375,.375,.25,.25]], [0])
                predictions,_ = decode_predictions(out,batch,frames,self.processor.tokenizer)
                torch.testing.assert_close(torch.tensor(predictions[0]['bbox']),torch.tensor(expected).float())
                native = self.processor.post_process_grounded_object_detection(out,
                    input_ids=batch['input_ids'],threshold=.1,text_threshold=.1,
                    target_sizes=[(size[1],size[0])])[0]['boxes'][0]
                x,y,w,h = expected
                torch.testing.assert_close(native,torch.tensor([x,y,x+w,y+h]).float())

    def test_all_classes_small_bottom_right_edge_boxes_and_clipping(self):
        # Literal original and normalized boxes, never copied from processed labels.
        objects = [(0,[160,80,160,80]), (1,[632,100,8,4]), (2,[8,316,4,4]),
                   (3,[0,0,8,4]), (4,[288,280,64,16]), (5,[600,280,32,32]),
                   (6,[320,160,32,16]), (7,[636,318,4,2])]
        oracle = [[.375,.375,.25,.25], [.99375,.31875,.0125,.0125],
                  [.015625,.99375,.00625,.0125], [.00625,.00625,.0125,.0125],
                  [.5,.9,.1,.05], [.9625,.925,.05,.1],
                  [.525,.525,.05,.05], [.996875,.996875,.00625,.00625]]
        image,target = self.sample((640,320),objects)
        batch,frames = collate(self.processor,[(image,target)])
        self.assertEqual(batch['labels'][0]['class_labels'].tolist(),list(range(8)))
        torch.testing.assert_close(batch['labels'][0]['boxes'],torch.tensor(oracle))
        out = self.outputs(batch,oracle,list(range(8)))
        predictions,raw = decode_predictions(out,batch,frames,self.processor.tokenizer)
        self.assertEqual(len(raw),900)
        self.assertEqual(len(predictions),8)
        for prediction in predictions:
            expected = objects[prediction['category_id']][1]
            torch.testing.assert_close(torch.tensor(prediction['bbox']),torch.tensor(expected).float(),atol=5e-5,rtol=1e-5)
        # Partially outside remains clipped/valid; zero-width edge/interior reject.
        out = self.outputs(batch,[[1.,.5,.1,.1],[1.,.5,0.,.1],[.5,.5,0.,.1]],[0,1,2])
        predictions,_ = decode_predictions(out,batch,frames,self.processor.tokenizer)
        self.assertEqual(len(predictions),1)
        torch.testing.assert_close(torch.tensor(predictions[0]['bbox']),torch.tensor([608.,144.,32.,32.]),atol=5e-5,rtol=1e-5)

    def test_pixels_remain_aligned_and_padding_stays_bottom_right(self):
        image,target = self.sample((640,320),[(0,[160,80,160,80])])
        ImageDraw.Draw(image).rectangle([160,80,319,159],fill='red')
        batch,_ = collate(self.processor,[(image,target)])
        proc = self.processor.image_processor
        pixels = batch['pixel_values'][0]*torch.tensor(proc.image_std)[:,None,None]+torch.tensor(proc.image_mean)[:,None,None]
        red = (pixels[0]>.6)&(pixels[1]<.2)&(pixels[2]<.2)
        yy,xx = red.nonzero(as_tuple=True)
        self.assertEqual([int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],[96,48,191,95])
        mask = batch['pixel_mask'][0]
        self.assertTrue(mask[:192,:384].all())
        self.assertFalse(mask[192:,:].any())

    def test_empty_targets_and_changed_identity_still_reject(self):
        image,target = self.sample((640,320),[])
        batch,_ = collate(self.processor,[(image,target)])
        self.assertEqual(batch['labels'][0]['boxes'].shape,(0,4))
        from scripts.training_smoke import valid_image_loss_targets
        damaged = copy.deepcopy(batch['labels'][0]);damaged['image_id'] += 1
        with self.assertRaisesRegex(ValueError,'identity/count'):
            valid_image_loss_targets(damaged,target,image.size,(384,192),(384,384))

    def test_native_proposals_use_valid_image_reference_frame(self):
        # Native source helper, independent of both target and decoding formulas.
        stub = SimpleNamespace(enc_output=torch.nn.Identity(),enc_output_norm=torch.nn.Identity())
        pad = torch.zeros(1,4,8,dtype=torch.bool);pad[:,2:]=True
        _, logits = GroundingDinoModel.generate_encoder_output_proposals(
            stub,torch.zeros(1,32,4),pad.flatten(1),torch.tensor([[4,8]]))
        torch.testing.assert_close(logits.sigmoid()[0,11],torch.tensor([.4375,.75,.05,.05]))
        torch.testing.assert_close(GroundingDinoModel.get_valid_ratio(stub,~pad),torch.tensor([[1.,.5]]))

    @unittest.skipUnless(importlib.util.find_spec('scipy'),'Native matcher requires SciPy')
    def test_native_decoder_and_encoder_box_losses_share_corrected_targets(self):
        from transformers import GroundingDinoConfig
        from transformers.loss.loss_grounding_dino import GroundingDinoForObjectDetectionLoss
        from transformers.models.grounding_dino.modeling_grounding_dino import build_text_mask
        image,target = self.sample((640,320),[(0,[160,80,160,80])])
        batch,_ = collate(self.processor,[(image,target)])
        out = self.outputs(batch,[[.375,.375,.25,.25]],[0])
        total,components,_ = GroundingDinoForObjectDetectionLoss(
            out.logits,batch['labels'],'cpu',out.pred_boxes,
            GroundingDinoConfig(auxiliary_loss=False),
            build_label_maps(out.logits,batch['input_ids']),
            build_text_mask(out.logits,batch['attention_mask']),
            encoder_logits=out.logits,encoder_pred_boxes=out.pred_boxes)
        self.assertTrue(torch.isfinite(total))
        for key in ('loss_bbox','loss_giou','loss_bbox_enc','loss_giou_enc'):
            self.assertAlmostEqual(float(components[key]),0.,places=6)

    @unittest.skipUnless(importlib.util.find_spec('pycocotools'),'COCO evaluator unavailable')
    def test_known_perfect_and_wrong_geometry_ap50(self):
        image,target = self.sample((640,320),[(0,[160,80,160,80])])
        batch,frames = collate(self.processor,[(image,target)])
        out = self.outputs(batch,[[.375,.375,.25,.25]],[0])
        predictions,_ = decode_predictions(out,batch,frames,self.processor.tokenizer)
        data = {'info':{},'images':[{'id':1,'width':640,'height':320}],
                'categories':[{'id':i,'name':name} for i,name in enumerate(LABELS)],
                'annotations':target['annotations']}
        predictions[0]['score'] = .01  # AP is independent of diagnostic threshold .25.
        self.assertAlmostEqual(coco_metrics(data,predictions)['AP50'],1.)
        wrong = [dict(predictions[0],bbox=[160,160,160,160])]
        self.assertEqual(coco_metrics(data,wrong)['AP50'],0.)


if __name__ == '__main__':
    unittest.main()
