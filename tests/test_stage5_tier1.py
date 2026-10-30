"""D074 metadata/safety tests; no real photo decoding or model execution."""
import copy
import unittest
from types import SimpleNamespace
from scripts.configuration import ROOT, strict_load
from scripts.taco_training_data_v2 import dataset_context,require_rights,CocoTrainingDatasetV2
from scripts.prepare_stage5_tier1_export import reconstruct_frozen, serialized
from scripts.train_taco_stage5 import require_tier1_contract

class Tier1Tests(unittest.TestCase):
    def test_bound_corpus_and_old_rights_remain_usable(self):
        for name,rights,images in [('strict-clean-stage5-v1','rights.json',307),
                  ('strict-clean-stage5-tier1-v1','rights_tier1.json',336)]:
            context=dataset_context(ROOT/'exports/taco'/name)
            record=require_rights(ROOT/'work/stage5-paper'/rights,context)
            self.assertEqual(context['splits']['train']['images'],images)
            self.assertEqual(context['splits']['validation']['annotations'],109)
            self.assertFalse(record['record']['photo_rights_cleared'])
            ds=CocoTrainingDatasetV2(ROOT/'exports/taco'/name,'train',ROOT/'work/stage5-paper'/rights)
            self.assertEqual(len(ds),images)
        self.assertEqual(len(ds.photo_hashes),401)
        self.assertNotIn(1173,{r['id'] for r in ds.images})

    def test_reconstructs_test_metadata_without_split_reads(self):
        image={'id':7,'file_name':'not-opened.jpg','width':10,'height':10}
        ann={'id':3,'image_id':7,'category_id':9,'bbox':[0,0,2,2],'area':4,'iscrowd':0}
        ledger=[{'eligible':True,'source_image_id':7,'source_image':image,'image_record_index':0,
                 'split':'test','fingerprint':{'file_sha256':'file','pixel_sha256':'pixel'}}]
        template={'info':{},'licenses':[],'categories':[{'id':0,'name':'cigarette'}]}
        raw={'images':[image],'annotations':[ann],'categories':[{'id':9,'name':'Cigarette'}]}
        taxonomy={'labels':[{'id':0,'taco_categories':['Cigarette']}]}
        result,next_id=reconstruct_frozen(raw,ledger,template,taxonomy)
        self.assertEqual(result['test']['images'][0]['id'],7)
        self.assertEqual(result['test']['annotations'][0]['source_taco']['annotation_id'],3)
        self.assertEqual(next_id,2)
        self.assertIn(b'\r\n',serialized(result['test']))

    def test_bounded_tier1_settings_reject_changes(self):
        args=SimpleNamespace(epochs=1,learning_rate=2e-6,shortest=384,longest=576,
             seed=42,evaluate_baseline=False,warmstart='stage5_highres_epoch1')
        require_tier1_contract(args)
        for key,value in [('epochs',2),('learning_rate',1e-5),('evaluate_baseline',True),('seed',123)]:
            other=copy.copy(args);setattr(other,key,value)
            with self.assertRaises(ValueError):require_tier1_contract(other)

    def test_tier1_validation_is_byte_identical_and_test_stays_unavailable(self):
        old=ROOT/'exports/taco/strict-clean-stage5-v1'
        new=ROOT/'exports/taco/strict-clean-stage5-tier1-v1'
        self.assertEqual((old/'validation.json').read_bytes(),(new/'validation.json').read_bytes())
        self.assertEqual(strict_load(old/'preparation_manifest.json')['artifacts']['test.json'],
                         strict_load(new/'preparation_manifest.json')['artifacts']['test.json'])
        with self.assertRaises(ValueError):CocoTrainingDatasetV2(new,'test',ROOT/'work/stage5-paper/rights_tier1.json')

class Tier1DiagnosticsTests(unittest.TestCase):
    def test_cached_nms_views_equal_direct_calculation(self):
        from scripts.taco_stage5_tier1_diagnostics import evaluate_views
        from scripts.taco_stage5_material_hierarchy import evaluate_case
        data={'info':{},'images':[{'id':1,'width':100,'height':100},{'id':2,'width':100,'height':100}],
          'categories':[{'id':i,'name':str(i)} for i in range(9)],
          'annotations':[{'id':1,'image_id':1,'category_id':6,'bbox':[0,0,50,50],'area':2500,'iscrowd':0}]}
        rows=[{'query_id':0,'image_id':1,'category_id':1,'score':.9,'bbox':[0,0,50,50]},
              {'query_id':1,'image_id':1,'category_id':6,'score':.8,'bbox':[0,0,50,50]}]
        cases=list(evaluate_views(data,rows,{1}));self.assertEqual(len(cases),27)
        for c in cases:
            if c['view']=='combined_65':
                direct=evaluate_case(data,rows,c['taxonomy'],c['nms'])
                self.assertEqual(c['metrics'],direct['metrics'])
                self.assertEqual(c['coverage'],direct['coverage'])

    def test_gui_exposes_saved_tier1_state_selector(self):
        import inspect
        from scripts.taco_stage5_visual_review import Stage5ViewerGUI,load_review_data
        source=inspect.getsource(load_review_data)
        for name in ('soup_tier1_hr_blend2575.json','soup_tier1_hr_5050.json'):
            self.assertIn(name,source)
        self.assertIn('review_state_var',inspect.getsource(Stage5ViewerGUI))

if __name__=='__main__':unittest.main()
