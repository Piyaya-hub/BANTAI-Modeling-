"""D076 controlled training-only mapping and fixed-boundary qualification."""
import copy
import unittest
from types import SimpleNamespace
from scripts.configuration import ROOT,strict_load
from scripts.prepare_stage5_tier2_export import training_mapping
from scripts.taco_training_labels_v2 import validate_taxonomy
from scripts.taco_training_data_v2 import dataset_context,require_rights,CocoTrainingDatasetV2
from scripts.train_taco_stage5 import require_tier2_contract

class Tier2Tests(unittest.TestCase):
    def test_training_only_mapping_does_not_mutate_taxonomy(self):
        taxonomy=validate_taxonomy();original=copy.deepcopy(taxonomy);mapping=training_mapping(taxonomy)
        self.assertEqual(taxonomy,original)
        expected={'Polypropylene bag':2,'Six pack rings':3,'Food Can':5,'Aerosol':5,
          'Disposable food container':6,'Spread tub':6,'Other plastic container':6,
          'Foam cup':7,'Foam food container':7,'Plastified paper bag':8}
        for name,cid in expected.items():self.assertEqual(mapping[name],cid)
        self.assertNotIn('Other plastic',mapping)

    def test_exact_bound_export_and_all_prior_rights(self):
        for name,rights,count in [('strict-clean-stage5-v1','rights.json',307),
            ('strict-clean-stage5-tier1-v1','rights_tier1.json',336),
            ('strict-clean-stage5-tier2-v1','rights_tier2.json',384)]:
            folder=ROOT/'exports/taco'/name;path=ROOT/'work/stage5-paper'/rights
            context=dataset_context(folder);record=require_rights(path,context)
            self.assertEqual(context['splits']['train']['images'],count)
            self.assertFalse(record['record']['photo_rights_cleared'])
        ds=CocoTrainingDatasetV2(folder,'train',path)
        self.assertEqual(len(ds.photo_hashes),449)
        self.assertEqual([sum(a['category_id']==i for a in ds.data['annotations']) for i in range(9)],
                         [147,49,23,74,134,64,54,54,145])
        self.assertNotIn(1173,{i['id'] for i in ds.images})
        self.assertEqual(record['record']['limits'],{'training_updates':384,'validation_forwards':336,'test_access':False})

    def test_frozen_validation_alias_and_test_metadata_hash(self):
        old=ROOT/'exports/taco/strict-clean-stage5-v1';new=ROOT/'exports/taco/strict-clean-stage5-tier2-v1'
        expected=(old/'validation.json').read_bytes()
        self.assertEqual((new/'validation.json').read_bytes(),expected)
        self.assertEqual((new/'val.json').read_bytes(),expected)
        self.assertEqual(strict_load(old/'preparation_manifest.json')['artifacts']['test.json'],
                         strict_load(new/'preparation_manifest.json')['artifacts']['test.json'])
        with self.assertRaises(ValueError):CocoTrainingDatasetV2(new,'test',ROOT/'work/stage5-paper/rights_tier2.json')

    def test_bounds_require_exactly_two_tier2_soups(self):
        args=SimpleNamespace(epochs=1,learning_rate=2e-6,shortest=384,longest=576,seed=42,
          evaluate_baseline=False,warmstart='stage5_highres_epoch1',tier1_soups=False,tier2_soups=True)
        require_tier2_contract(args)
        for key,value in [('epochs',2),('seed',123),('learning_rate',1e-5),('tier2_soups',False),('tier1_soups',True)]:
            other=copy.copy(args);setattr(other,key,value)
            with self.assertRaises(ValueError):require_tier2_contract(other)

    def test_tier2_discovery_falls_back_and_preserves_explicit_override(self):
        import tempfile
        from pathlib import Path
        from scripts.taco_stage5_visual_review import discover_tier2_run
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.assertIsNone(discover_tier2_run(root=root))
            chosen=root/'explicit';self.assertEqual(discover_tier2_run(chosen,root),chosen)
            run=root/'runs/training/stage5-tier2-fixture';run.mkdir(parents=True)
            for name in ('report.json','validation-epoch-1.json','soup_tier2_hr_blend2575.json','soup_tier2_d074_champion_5050.json'):
                (run/name).write_text('{}')
            self.assertEqual(discover_tier2_run(root=root),run)
            self.assertEqual(discover_tier2_run(chosen,root),chosen)

if __name__=='__main__':unittest.main()
