"""Resolution geometry/decoding/binning fixtures; no real inference or updates."""
import unittest
from types import SimpleNamespace
from pathlib import Path
import importlib.util
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
from scripts.training_smoke import SNAPSHOT,validate_encoder_capacity
from scripts.taco_training_data import collate
from scripts.taco_evaluation import decode_predictions
from scripts.configuration import ROOT,strict_load

def private_support(name,relative):
    path=ROOT/relative
    if not path.is_file():
        raise unittest.SkipTest('Optional local resolution audit helpers are not published; work/ remains private')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class ResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.processor=AutoProcessor.from_pretrained(SNAPSHOT,local_files_only=True)

    def sample(self,size):
        image=Image.new('RGB',size,'black');self.addCleanup(image.close)
        w,h=size;box=[w/4,h/4,w/4,h/4]
        return image,{'image_id':1,'annotations':[{'id':1,'image_id':1,'category_id':0,'bbox':box,'area':box[2]*box[3],'iscrowd':0}]}

    def test_576_collate_rectangular_mask_capacity_and_valid_targets(self):
        self.assertEqual(validate_encoder_capacity(576,576),6885)
        for original,expected in [((640,320),[576,288]),((320,640),[288,576]),((1600,100),[576,36])]:
            with self.subTest(original=original):
                image,target=self.sample(original);batch,frames=collate(self.processor,[(image,target)],384,576)
                self.assertEqual(frames[0]['resized_wh'],expected);self.assertEqual(frames[0]['padded_wh'],[576,576])
                mask=batch['pixel_mask'][0];oracle=torch.zeros_like(mask);oracle[:expected[1],:expected[0]]=1
                self.assertTrue(torch.equal(mask,oracle));self.assertEqual(int(mask.sum()),expected[0]*expected[1])
                torch.testing.assert_close(batch['labels'][0]['boxes'],torch.tensor([[.375,.375,.25,.25]]))

    def test_identical_normalized_predictions_decode_identically_across_scales(self):
        for size in ((640,320),(320,640),(1600,100)):
            image,target=self.sample(size);decoded=[]
            for shortest,longest in ((256,384),(384,576)):
                batch,frames=collate(self.processor,[(image,target)],shortest,longest)
                logits=torch.full((1,900,256),float('-inf'));logits[:,:,:batch['input_ids'].shape[1]]=-30
                mapping=build_label_maps(logits,batch['input_ids'])[0]
                logits[0,0,mapping[0].bool()]=10
                boxes=torch.zeros(1,900,4);boxes[0,0]=torch.tensor([.375,.375,.25,.25])
                predictions,raw=decode_predictions(SimpleNamespace(logits=logits,pred_boxes=boxes),batch,frames,self.processor.tokenizer)
                self.assertEqual(len(raw),900);decoded.append(predictions)
                torch.testing.assert_close(torch.tensor(predictions[0]['bbox']),torch.tensor(target['annotations'][0]['bbox']))
            self.assertEqual(decoded[0],decoded[1])

    def test_bins_lock_baseline_denominators_despite_new_pixel_sizes(self):
        support=private_support('resolution_support','work/resolution_experiment/support.py')
        widths=[3]*14+[6]*10+[12]*15+[20]*36
        objects=[{'annotation_id':i,'resized_bbox_width':w,'resized_bbox_height':w*2} for i,w in enumerate(widths)]
        high={i:{'any_class_coverage':True,'correct_class_coverage':True,'resized_bbox_width':w*1.5,'resized_bbox_height':w*3} for i,w in enumerate(widths)}
        locked=support.summarize_locked(objects,high)
        self.assertEqual([locked[g]['N'] for g in support.GROUPS],[14,10,15,36])
        self.assertEqual([locked[g]['same'] for g in support.GROUPS],[14,10,15,36])
        self.assertNotEqual(support.bin_name(3),support.bin_name(4.5))
        pointer=ROOT/'work/small-object-audit/study.json'
        if pointer.is_file():
            audit=Path(strict_load(pointer)['study'])
            actual=support.lock_bins(strict_load(audit/'preprocessing.json')['objects'])
            self.assertEqual([sum(v==g for v in actual.values()) for g in support.GROUPS],[14,10,15,36])

    def test_multi_model_locked_bins_and_transition_identity(self):
        multi=private_support('resolution_replication_support','work/resolution_replication/replication_support.py')
        widths=[3]*14+[6]*10+[12]*15+[20]*36
        objects=[{'annotation_id':i,'resized_bbox_width':w,'resized_bbox_height':w} for i,w in enumerate(widths)]
        missing={'any_class_coverage':False,'correct_class_coverage':False}
        wrong={'any_class_coverage':True,'correct_class_coverage':False}
        correct={'any_class_coverage':True,'correct_class_coverage':True}
        models={}
        for name in ('pretrained','reference','seed123'):
            low={i:dict(missing) for i in range(75)};high={i:dict(missing) for i in range(75)}
            low[0]=correct;high[0]=wrong
            low[14]=wrong;high[14]=correct
            high[24]=wrong
            high[40]=correct
            if name=='seed123':high[41]=correct
            models[name]={'256/384':low,'384/576':high}
        result=multi.aggregate_models(objects,models)
        self.assertEqual(result['consistent_all_models'],{'gain':[14,40],'loss':[0],'newly_localized_wrong':[24]})
        for r in result['models'].values():
            self.assertEqual([r['bins']['384/576'][g]['N'] for g in multi.GROUPS],[14,10,15,36])
            self.assertEqual(sum(sum(row.values()) for row in r['matrix_counts'].values()),75)
            self.assertEqual(r['matrix_counts']['correct_class']['wrong_class_only'],1)
            self.assertEqual(r['matrix_GT_ids']['wrong_class_only']['correct_class'],[14])
        bad=dict(models);bad['bad']={'256/384':{},'384/576':{}}
        with self.assertRaises(ValueError):multi.aggregate_models(objects,bad)

if __name__=='__main__':unittest.main()
