"""Learning-rate wiring and immutable context fixtures; no optimizer updates."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch

import torch
from scripts import train_taco as trainer
from scripts.training_checkpoint import verify_checkpoint


class LearningRateTests(unittest.TestCase):
    def test_optimizer_uses_selected_lr_preserves_defaults_and_frozen_parameters(self):
        model=torch.nn.Linear(2,1);model.bias.requires_grad_(False)
        for rate in (None,5e-6):
            optimizer=trainer.make_optimizer(model) if rate is None else trainer.make_optimizer(model,rate)
            group=optimizer.param_groups[0]
            self.assertEqual(group['lr'],1e-5 if rate is None else rate)
            self.assertEqual(group['weight_decay'],1e-4)
            self.assertIs(group['foreach'],False)
            self.assertEqual([id(p) for p in group['params']],[id(model.weight)])

    def test_invalid_learning_rates_reject_before_dataset_access(self):
        model=torch.nn.Linear(1,1)
        for rate in (0,-1,float('nan'),float('inf')):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):trainer.make_optimizer(model,rate)
                with (patch.object(trainer,'dataset_context',side_effect=AssertionError('No data reads')),
                      contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit)):
                    trainer.main(['--learning-rate='+str(rate)])

    def test_cli_delivers_lr_to_both_optimizers_and_immutable_checkpoint_context(self):
        for selected in (None,5e-6):
            with self.subTest(selected=selected),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);base=root/'base';base.mkdir();(base/'fixture.txt').write_text('generated')
                dataset=MagicMock();dataset.__len__.return_value=1;dataset.photo_reads=0
                dataset.images=[{'id':1}];dataset.objects={1:[{}]}
                adapter=MagicMock();adapter.manifest.return_value={'fixture':True}
                model=torch.nn.Linear(2,1);identity=trainer.source_identity();contexts=[]
                def checkpoint(folder,model,processor,adapter,optimizer,context,position,evidence):
                    self.assertEqual(optimizer.param_groups[0]['lr'],context['optimizer']['lr'])
                    contexts.append(copy.deepcopy(context))
                argv=['--device','cpu']+([] if selected is None else ['--learning-rate',str(selected)])
                patches=[patch.object(trainer,'ROOT',root),patch.object(trainer,'SNAPSHOT',base),
                      patch.object(trainer,'source_identity',return_value=identity),
                      patch.object(trainer,'dataset_context',return_value={'fixture':True}),
                      patch.object(trainer,'require_rights',return_value={'record_sha256':'fixture'}),
                      patch.object(trainer,'require_training_environment',return_value={'packages':{},'python':'fixture','platform':'fixture'}),
                      patch.object(trainer,'load_model',return_value=(model,MagicMock(),adapter)),
                      patch.object(trainer,'CocoTrainingDataset',return_value=dataset),
                      patch.object(trainer,'preflight',return_value=[]),
                      patch.object(trainer,'state_hash',return_value='fixture'),
                      patch.object(trainer,'get_batch',return_value=({},[{'image_id':1}])),
                      patch.object(trainer,'move_batch',side_effect=lambda batch,device:batch),
                      patch.object(trainer,'evaluate',return_value={'metrics':{'AP':0.},'mean_loss_per_image':1.}),
                      patch.object(trainer,'train_step',return_value={'loss':1.}),
                      patch.object(trainer,'save_checkpoint',side_effect=checkpoint),
                      patch.object(trainer,'make_optimizer',wraps=trainer.make_optimizer),
                      patch('torch.optim.AdamW.step',side_effect=AssertionError('No fixture updates')),
                      contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO())]
                with contextlib.ExitStack() as stack:
                    entered=[stack.enter_context(p) for p in patches]
                    make=entered[-4]
                    self.assertEqual(trainer.main(argv),0)
                rate=1e-5 if selected is None else selected
                self.assertEqual(make.call_count,2)
                self.assertTrue(all(c.args[1]==rate for c in make.call_args_list))
                self.assertEqual(len(contexts),2)
                self.assertTrue(all(c['optimizer']['lr']==rate and c['settings']['learning_rate']==rate for c in contexts))
                report=json.loads(next((root/'runs/training').glob('*/report.json')).read_text())
                self.assertEqual(report['settings']['learning_rate'],rate)

    def test_strict_checkpoint_context_rejects_changed_lr_before_state_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);context={'optimizer':{'lr':5e-6}}
            manifest={'schema_version':'bantai.training-checkpoint.v1','status':'complete','context':context,'files':{}}
            (path/'checkpoint_manifest.json').write_text(json.dumps(manifest))
            with (patch('torch.load',side_effect=AssertionError('No optimizer restoration')),
                  self.assertRaisesRegex(ValueError,'changed dataset/rights/settings/runtime/source')):
                verify_checkpoint(path,{'optimizer':{'lr':1e-5}})


if __name__=='__main__':unittest.main()
