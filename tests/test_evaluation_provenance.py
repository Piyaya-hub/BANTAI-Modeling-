"""Generated files only; fixture authority never represents real TACO rights."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.configuration import strict_load
from scripts.evaluation_provenance import (HISTORICAL_SOURCES, SUPPORTING_SOURCES,
    REVIEWED_DIFFERENCES, verify_evaluation_checkpoint)
from scripts.training_adapter import adapter_source_digest
from scripts.training_checkpoint import verify_checkpoint
from scripts.training_smoke import digest
from scripts.reevaluate_taco import evaluate_checkpoint, validate_bundle, main, configuration_semantics
from test_taco_training import fixture


class EvaluationProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root/'dataset';self.dataset.mkdir()
        self.rights,self.context = fixture(self.dataset)
        self.sources = self.root/'implementation'
        old = hashlib.sha256(b'# historical fixture\n').hexdigest()
        for name in HISTORICAL_SOURCES+SUPPORTING_SOURCES:
            p = self.sources/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('# reviewed correction\n' if name in REVIEWED_DIFFERENCES else '# historical fixture\n')
        historical = dict.fromkeys(HISTORICAL_SOURCES,old)
        current = {name:adapter_source_digest(self.sources/name) for name in HISTORICAL_SOURCES}
        self.contract = {'dataset':self.context,'implementation_sha256':historical,
                         'rights_sha256':digest(self.rights),'adapter_context':{'synthetic':True}}
        self.report = self.root/'report.json'
        self.report.write_text(json.dumps({'contract':self.contract,
            'implementation_sha256':historical,'rights':{'record':strict_load(self.rights)}}))
        self.folder = self.root/'epoch-01';self.folder.mkdir()
        for name in ('model.safetensors','training_state.pt','config.json','preprocessor_config.json',
                     'training_adapter.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json'):
            (self.folder/name).write_bytes(b'Generated integrity fixture; NOT serialized model/training state')
        files = {p.name:digest(p) for p in self.folder.iterdir()}
        self.manifest = self.folder/'checkpoint_manifest.json'
        self.manifest.write_text(json.dumps({'schema_version':'bantai.training-checkpoint.v1',
            'status':'complete','context':self.contract,'files':files,'position':{'epoch':1}}))
        self.policy = {'schema_version':'bantai.evaluation-compatibility.v1',
            'purpose':'validation_only_no_training','historical_report_sha256':digest(self.report),
            'checkpoint_manifests':{'epoch-01':digest(self.manifest)},
            'historical_sources':historical,'current_sources':current,
            'supporting_sources':{name:adapter_source_digest(self.sources/name) for name in SUPPORTING_SOURCES},
            'validation_counts':{'images':1,'annotations':1},
            'approved_differences':{name:{'historical_sha256':old,'evaluation_sha256':current[name],
                'reason':'Reviewed synthetic coordinate correction'} for name in REVIEWED_DIFFERENCES}}

    def verify(self, **kwargs):
        return verify_evaluation_checkpoint(self.folder,self.report,self.policy,
            self.dataset,self.rights,source_root=self.sources,**kwargs)

    def test_valid_artifacts_retain_both_identities_without_mutation(self):
        before = copy.deepcopy(self.policy),self.report.read_bytes(),self.manifest.read_bytes()
        result = self.verify()
        self.assertEqual(result['historical_context'],self.contract)
        self.assertEqual(result['sources']['historical_sources'],self.policy['historical_sources'])
        self.assertEqual(result['sources']['current_sources'],self.policy['current_sources'])
        self.assertEqual(set(result['sources']['approved_differences']),REVIEWED_DIFFERENCES)
        self.assertFalse(result['training_resume_authorized'])
        self.assertEqual(before,(self.policy,self.report.read_bytes(),self.manifest.read_bytes()))

    def test_tampered_weights_training_state_and_tokenizer_reject(self):
        for name in ('model.safetensors','training_state.pt','tokenizer.json'):
            p=self.folder/name;before=p.read_bytes()
            with self.subTest(name=name):
                p.write_bytes(before+b' altered')
                with self.assertRaisesRegex(ValueError,'artifact hash mismatch'):self.verify()
                p.write_bytes(before)

    def test_missing_artifact_rejects(self):
        (self.folder/'model.safetensors').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):self.verify()

    def test_modified_manifest_and_report_reject(self):
        for p in (self.manifest,self.report):
            before=p.read_bytes();p.write_bytes(before+b' ')
            with self.subTest(path=p.name),self.assertRaisesRegex(ValueError,'anchor mismatch'):self.verify()
            p.write_bytes(before)

    def test_unmanifested_loader_artifact_rejects(self):
        (self.folder/'added_tokens.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Unmanifested'):self.verify()

    def test_any_current_source_change_rejects(self):
        for name in HISTORICAL_SOURCES+SUPPORTING_SOURCES:
            p=self.sources/name;before=p.read_bytes();p.write_bytes(before+b'# unexpected\n')
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'Unexpected current'):self.verify()
            p.write_bytes(before)

    def test_filename_only_allowlist_and_wrong_pairs_reject(self):
        for field,value in [('historical_sha256','invented'),('evaluation_sha256','invented'),('reason','')]:
            entry=self.policy['approved_differences']['scripts/taco_evaluation.py']
            before=copy.deepcopy(entry);entry[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'exact reviewed'):self.verify()
            entry.clear();entry.update(before)

    def test_unreviewed_change_or_omitted_review_rejects(self):
        del self.policy['approved_differences']['scripts/taco_evaluation.py']
        with self.assertRaisesRegex(ValueError,'Unapproved'):self.verify()

    def test_strict_resume_remains_rejected(self):
        verify_checkpoint(self.folder,self.contract)
        changed=dict(self.contract,implementation_sha256=self.policy['current_sources'])
        with self.assertRaisesRegex(ValueError,'changed dataset/rights/settings/runtime/source'):
            verify_checkpoint(self.folder,changed)
        self.verify()  # Explicitly different evaluation contract, not a resume exception.

    def test_test_and_train_reject_before_metadata_or_photo_access(self):
        with patch('scripts.evaluation_provenance.digest',side_effect=AssertionError('must not read')):
            for split in ('test','train'):
                with self.subTest(split=split),self.assertRaisesRegex(ValueError,'Only validation'):
                    self.verify(split=split)
        with patch('sys.stderr'),self.assertRaises(SystemExit):
            main(['--historical-run','unused','--rights','unused','--baseline-record','unused','--split','test'])

    def test_changed_dataset_and_invalid_rights_reject_before_photos(self):
        with patch('PIL.Image.open',side_effect=AssertionError('must not open')):
            before=self.rights.read_bytes()
            self.rights.write_text('{}')
            with self.assertRaises(ValueError):self.verify()
            self.rights.write_bytes(before)
            path=self.dataset/'validation.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Split hash mismatch'):self.verify()

    def test_missing_rights_rejects(self):
        self.rights.unlink()
        with self.assertRaises(ValueError):self.verify()

    def test_valid_but_different_dataset_identity_rejects(self):
        path=self.dataset/'validation.json'
        value=strict_load(path);value['annotations'][0]['area']=28.
        path.write_text(json.dumps(value))
        manifest=self.dataset/'preparation_manifest.json'
        value=strict_load(manifest);value['artifacts']['validation.json']=digest(path)
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'dataset identity differs'):self.verify()

    def test_forged_historical_source_identity_rejects(self):
        self.policy['historical_sources']['scripts/train_taco.py']='invented'
        with self.assertRaisesRegex(ValueError,'historical identity'):self.verify()

    def test_verification_never_deserializes_or_restores_state(self):
        with (patch('torch.load',side_effect=AssertionError('no deserialization')),
              patch('scripts.training_checkpoint.restore_training',side_effect=AssertionError('no resume')),
              patch('scripts.training_checkpoint.restore_rng',side_effect=AssertionError('no RNG restore')),
              patch('torch.optim.AdamW.step',side_effect=AssertionError('no updates'))):
            proof=self.verify()
        for name in ('training_state_deserialized','optimizer_restored','rng_restored','training_position_restored'):
            self.assertFalse(proof[name])
        self.assertEqual(proof['optimizer_updates'],0)

    def test_execution_freezes_parameters_and_uses_inference_mode_without_updates(self):
        # Orchestration fixture only; real HF compatibility is checked in actual runs.
        model=torch.nn.Linear(2,1);before=copy.deepcopy(model.state_dict())
        seen=[]
        def evaluation(m,*args):
            self.assertFalse(m.training);self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertTrue(all(not p.requires_grad for p in m.parameters()))
            seen.append(True)
            return {'metrics':{'AP':0.},'mean_loss_per_image':1.}
        with (patch('scripts.reevaluate_taco.load_model',return_value=(model,None,None)),
              patch('scripts.reevaluate_taco.validate_bundle',return_value={'fixture':True}),
              patch('scripts.reevaluate_taco.evaluate',side_effect=evaluation),
              patch('torch.load',side_effect=AssertionError('no training state')),
              patch('torch.optim.AdamW.step',side_effect=AssertionError('no updates')),
              patch('scripts.training_checkpoint.restore_rng',side_effect=AssertionError('no RNG restore'))):
            result=evaluate_checkpoint(self.folder,self.report,self.policy,self.dataset,self.rights,
                'cpu',source_root=self.sources)
        self.assertEqual(seen,[True]);self.assertTrue(result['parameter_values_unchanged'])
        self.assertTrue(all(torch.equal(before[k],model.state_dict()[k]) for k in before))

    def test_incompatible_architecture_rejects(self):
        with self.assertRaisesRegex(ValueError,'pinned architecture'):
            validate_bundle(SimpleNamespace(config=SimpleNamespace()),None,None,{})

    def test_hf_autoset_bookkeeping_is_not_an_architecture_change(self):
        from transformers import GroundingDinoConfig
        reference=GroundingDinoConfig();loaded=copy.deepcopy(reference)
        loaded._attn_implementation_autoset=True
        self.assertEqual(configuration_semantics(reference),configuration_semantics(loaded))

    def test_resolved_attention_backend_and_architecture_changes_remain_visible(self):
        from transformers import GroundingDinoConfig
        reference=GroundingDinoConfig();changed=copy.deepcopy(reference)
        changed._attn_implementation='sdpa'
        self.assertNotEqual(configuration_semantics(reference),configuration_semantics(changed))
        changed=copy.deepcopy(reference);changed.num_queries=123
        self.assertNotEqual(configuration_semantics(reference),configuration_semantics(changed))


if __name__ == '__main__':
    unittest.main()
