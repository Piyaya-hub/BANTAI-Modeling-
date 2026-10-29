"""Stage 5 qualification tests: 9-class waste taxonomy, loader, adapter, and rights."""
from __future__ import annotations

import unittest
from pathlib import Path
import torch
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from scripts.configuration import ROOT
from scripts.taco_training_labels_v2 import LABELS, PROMPT, validate_taxonomy
from scripts.taco_training_data_v2 import dataset_context, require_rights, CocoTrainingDatasetV2
from scripts.training_adapter_v2 import EncoderOutputAdapterV2, ADAPTER_VERSION


class TestStage5Pipeline(unittest.TestCase):
    def test_06_coco_result_load_keeps_info(self):
        from scripts.taco_evaluation_v3 import coco_metrics
        data={"info":{},"images":[{"id":1,"width":100,"height":100}],
            "categories":[{"id":i,"name":name} for i,name in enumerate(LABELS)],
            "annotations":[{"id":1,"image_id":1,"category_id":8,"bbox":[0,0,20,20],"area":400,"iscrowd":0}]}
        metrics=coco_metrics(data,[{"image_id":1,"category_id":8,"score":.9,"bbox":[0,0,20,20]}])
        self.assertAlmostEqual(metrics['AP'],1.)

    def test_07_second_checkpoint_replaces_pointer_and_uses_resume_position(self):
        from scripts import train_taco_stage5 as runner
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            def save(path,*args):path.mkdir(parents=True)
            with patch.object(runner,'save_checkpoint',side_effect=save) as saving,patch.object(runner,'verify_checkpoint'),patch.object(runner,'state_hash',return_value='frozen'):
                for step in (100,200):runner.save_stage5_checkpoint(folder,'step-'+str(step),None,None,1,step,{'training_images':307},None,None)
            from scripts.configuration import strict_load
            self.assertEqual(strict_load(folder/'latest_checkpoint.json')['latest'],'step-200')
            self.assertEqual(saving.call_args.args[6],{'epoch':0,'next_batch':200,'global_step':200})

    def test_08_legacy_uses_original_prompt_evaluation_not_filtered_nine_class(self):
        from scripts import taco_evaluation_v3 as evaluator
        data={'images':[{'id':i} for i in range(65)],'annotations':[],'categories':[]}
        old={'images':data['images'][:47],'annotations':[],'categories':[]}
        ds=SimpleNamespace(data=data,images=data['images'],root=Path('deliberately-not-project'))
        class Dataset:
            data=ds.data;images=ds.images;root=ds.root
            def __len__(self):return 65
        model=SimpleNamespace(training=True,eval=lambda:None,train=lambda value:None,config=None)
        with (patch('scripts.configuration.strict_load',return_value=old) as read,
              patch.object(evaluator,'get_batch',return_value=({},[])),
              patch('scripts.training_smoke.move_batch',return_value={}),patch('scripts.training_smoke.checked_loss'),
              patch.object(evaluator,'decode_predictions',return_value=([],[])),
              patch.object(evaluator,'coco_metrics',return_value={'AP':.9}),
              patch('scripts.training_adapter.EncoderOutputAdapter',return_value=None),
              patch('scripts.taco_evaluation.evaluate',return_value={'metrics':{'AP':.8}}) as legacy):
            value=evaluator.evaluate_stage5(model,SimpleNamespace(tokenizer=None),lambda *v:SimpleNamespace(loss=torch.tensor(1.),outputs=None),Dataset(),'cpu')
        read.assert_called_once_with(ROOT/'exports/taco/strict-clean-stage2-v1/validation.json')
        self.assertEqual(value['legacy_47']['AP'],.8)
        self.assertEqual(value['legacy_47_nine_class_conditioning']['AP'],.9)
        self.assertEqual(value['validation_forwards'],112);self.assertEqual(legacy.call_count,1)

    def test_01_taxonomy_v2(self):
        tax = validate_taxonomy()
        self.assertEqual(len(tax["labels"]), 9)
        self.assertEqual(tax["labels"][8]["name"], "paper")
        self.assertEqual([l["name"] for l in tax["labels"]], list(LABELS))
        self.assertTrue(PROMPT.endswith("paper."))

    def test_02_export_manifest_and_splits(self):
        export_dir = ROOT / "exports/taco/strict-clean-stage5-v1"
        self.assertTrue(export_dir.is_dir(), "Export directory must exist")
        ctx = dataset_context(export_dir)
        self.assertEqual(ctx["splits"]["train"]["images"], 307)
        self.assertEqual(ctx["splits"]["validation"]["images"], 65)

    def test_03_rights_gate(self):
        export_dir = ROOT / "exports/taco/strict-clean-stage5-v1"
        rights_path = ROOT / "work/stage5-paper/rights.json"
        self.assertTrue(rights_path.is_file(), "Rights record must exist")
        ctx = dataset_context(export_dir)
        rights = require_rights(rights_path, ctx)
        self.assertEqual(rights["record"]["schema_version"], "bantai.training-rights.v2")
        self.assertTrue(rights["record"]["workflow_authorized"])

    def test_04_dataset_loaders(self):
        export_dir = ROOT / "exports/taco/strict-clean-stage5-v1"
        rights_path = ROOT / "work/stage5-paper/rights.json"
        train_ds = CocoTrainingDatasetV2(export_dir, "train", rights_path)
        self.assertEqual(len(train_ds), 307)
        val_ds = CocoTrainingDatasetV2(export_dir, "validation", rights_path)
        self.assertEqual(len(val_ds), 65)

        # Confirm test split cannot be loaded
        with self.assertRaises(ValueError):
            CocoTrainingDatasetV2(export_dir, "test", rights_path)

    def test_05_paper_counts(self):
        export_dir = ROOT / "exports/taco/strict-clean-stage5-v1"
        rights_path = ROOT / "work/stage5-paper/rights.json"
        train_ds = CocoTrainingDatasetV2(export_dir, "train", rights_path)
        paper_train = sum(1 for ann in train_ds.data["annotations"] if ann["category_id"] == 8)
        self.assertEqual(paper_train, 120)

        val_ds = CocoTrainingDatasetV2(export_dir, "validation", rights_path)
        paper_val = sum(1 for ann in val_ds.data["annotations"] if ann["category_id"] == 8)
        self.assertEqual(paper_val, 20)


if __name__ == "__main__":
    unittest.main()
