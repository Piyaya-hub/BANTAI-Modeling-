"""D075 CPU fixtures for private discovery and replay-only integrity."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.persist_stage5_soup import verify_bundle
from scripts.taco_stage5_visual_review import discover_tier1_run, DEFAULT_TIER1_RUN
from scripts.training_smoke import digest

class SoupPersistenceTests(unittest.TestCase):
    def test_discovery_respects_override_missing_and_complete_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.assertIsNone(discover_tier1_run(root=root))
            explicit=root/'explicit';self.assertEqual(discover_tier1_run(explicit,root),explicit)
            candidate=root/DEFAULT_TIER1_RUN;candidate.mkdir(parents=True)
            (candidate/'report.json').write_text('{}')
            self.assertIsNone(discover_tier1_run(root=root))
            for name in ('validation-epoch-1.json','soup_tier1_hr_blend2575.json','soup_tier1_hr_5050.json'):
                (candidate/name).write_text('{}')
            self.assertEqual(discover_tier1_run(root=root),candidate)
            self.assertEqual(discover_tier1_run(explicit,root),explicit)

    def test_manifest_requires_replay_only_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            files=('model.safetensors','config.json','preprocessor_config.json','training_adapter.json')
            for name in files:(root/name).write_text('fixture')
            manifest={'schema_version':'bantai.stage5-soup.v1','status':'complete',
              'resume_supported':False,'training_resume_supported':False,'files':{n:digest(root/n) for n in files}}
            path=root/'soup_manifest.json';path.write_text(json.dumps(manifest))
            self.assertFalse(verify_bundle(root)['resume_supported'])
            manifest['resume_supported']=True;path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_bundle(root)
            manifest['resume_supported']=False;path.write_text(json.dumps(manifest))
            (root/'model.safetensors').write_text('tampered')
            with self.assertRaises(ValueError):verify_bundle(root)

    def test_serialized_parameter_hash_matches_named_parameter_contract(self):
        import torch
        from safetensors.torch import save_model
        from scripts.persist_stage5_soup import saved_parameter_hash
        from scripts.training_smoke import state_hash
        with tempfile.TemporaryDirectory() as folder:
            model=torch.nn.Linear(2,1)
            path=Path(folder)/'model.safetensors';save_model(model,str(path))
            self.assertEqual(saved_parameter_hash(model,path),state_hash(model))
            with torch.no_grad():model.weight.add_(1.)
            self.assertNotEqual(saved_parameter_hash(model,path),state_hash(model))

if __name__=='__main__':unittest.main()
