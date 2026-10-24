"""Seed propagation/default/order fixtures; no model training or optimizer steps."""
import contextlib
import copy
import io
import json
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch
import numpy as np
import torch
from scripts import train_taco as trainer
from scripts.training_checkpoint import epoch_batches


class SeedTests(unittest.TestCase):
    def test_python_numpy_and_torch_sequences_repeat_for_each_seed(self):
        for seed in (42,123):
            trainer.seed_rngs(seed,'cpu')
            first=(random.random(),float(np.random.rand()),float(torch.rand(())))
            trainer.seed_rngs(seed,'cpu')
            self.assertEqual(first,(random.random(),float(np.random.rand()),float(torch.rand(()))))
            self.assertEqual(first[0],random.Random(seed).random())
            self.assertEqual(first[1],float(np.random.RandomState(seed).rand()))
            self.assertEqual(first[2],float(torch.rand((),generator=torch.Generator().manual_seed(seed))))

    def test_cuda_seed_is_explicitly_propagated(self):
        with (patch.object(trainer.random,'seed') as py,patch('numpy.random.seed') as np_seed,
              patch('torch.manual_seed') as cpu,patch('torch.cuda.manual_seed_all') as cuda):
            trainer.seed_rngs(123,'cuda')
        for call in (py,np_seed,cpu,cuda):call.assert_called_once_with(123)

    def test_fixed_sampler_oracles_and_epoch_offsets(self):
        self.assertEqual(sum(epoch_batches(5,1,42,0),[]),[3,1,2,4,0])
        self.assertEqual(sum(epoch_batches(5,1,123,0),[]),[3,1,4,2,0])
        self.assertNotEqual(epoch_batches(221,1,123,0),epoch_batches(221,1,123,1))
        self.assertEqual(epoch_batches(221,1,123,1),epoch_batches(221,1,123,1))

    def test_invalid_seed_rejects_before_dataset_access(self):
        for seed in ('-1',str(2**32)):
            with (self.subTest(seed=seed),patch.object(trainer,'dataset_context',side_effect=AssertionError('No data reads')),
                  contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit)):
                trainer.main(['--seed',seed])

    def test_cli_default_and_selected_seed_control_restart_sampler_and_context(self):
        for selected in (None,123):
            seed=42 if selected is None else selected
            with self.subTest(selected=selected),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);base=root/'base';base.mkdir();(base/'fixture').write_text('generated')
                model=torch.nn.Linear(2,1);adapter=MagicMock();adapter.manifest.return_value={'fixture':True}
                ds=MagicMock();ds.__len__.return_value=5;ds.photo_reads=0
                ds.images=[{'id':i} for i in range(1,6)];ds.objects={i:[{}] for i in range(1,6)}
                contexts=[];draws=[];loaded_with_seed=[];identity=trainer.source_identity()
                def load(path,device):
                    loaded_with_seed.append((random.getstate()==random.Random(seed).getstate(),
                        np.array_equal(np.random.get_state()[1],np.random.RandomState(seed).get_state()[1]),
                        torch.equal(torch.get_rng_state(),torch.Generator().manual_seed(seed).get_state())))
                    return model,MagicMock(),adapter
                def batch(dataset,indexes,*args):return {},[{'image_id':i+1} for i in indexes]
                def step(*args):
                    draws.append((random.random(),float(np.random.rand()),float(torch.rand(()))))
                    return {'loss':1.}
                def save(folder,model,processor,adapter,optimizer,context,position,evidence):contexts.append(copy.deepcopy(context))
                argv=['--device','cpu','--learning-rate','5e-6']+([] if selected is None else ['--seed',str(seed)])
                sampler=MagicMock(wraps=epoch_batches);seeder=MagicMock(wraps=trainer.seed_rngs)
                patches=[patch.object(trainer,'ROOT',root),patch.object(trainer,'SNAPSHOT',base),
                    patch.object(trainer,'source_identity',return_value=identity),
                    patch.object(trainer,'dataset_context',return_value={'fixture':True}),
                    patch.object(trainer,'require_rights',return_value={'record_sha256':'fixture'}),
                    patch.object(trainer,'require_training_environment',return_value={'packages':{},'python':'fixture','platform':'fixture'}),
                    patch.object(trainer,'load_model',side_effect=load),patch.object(trainer,'CocoTrainingDataset',return_value=ds),
                    patch.object(trainer,'preflight',return_value=[]),patch.object(trainer,'state_hash',return_value='fixture'),
                    patch.object(trainer,'get_batch',side_effect=batch),patch.object(trainer,'move_batch',side_effect=lambda b,d:b),
                    patch.object(trainer,'evaluate',return_value={'metrics':{'AP':0.},'mean_loss_per_image':1.}),
                    patch.object(trainer,'train_step',side_effect=step),patch.object(trainer,'save_checkpoint',side_effect=save),
                    patch.object(trainer,'epoch_batches',sampler),patch.object(trainer,'seed_rngs',seeder),
                    patch('torch.optim.AdamW.step',side_effect=AssertionError('No fixture optimizer updates')),
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO())]
                with contextlib.ExitStack() as stack:
                    for item in patches:stack.enter_context(item)
                    self.assertEqual(trainer.main(argv),0)
                self.assertEqual(loaded_with_seed,[(True,True,True)]*2)
                self.assertEqual(seeder.call_count,3)
                self.assertTrue(all(c.args==(seed,'cpu') for c in seeder.call_args_list))
                self.assertEqual([c.args for c in sampler.call_args_list],[(5,1,seed,0),(5,1,seed,1)])
                self.assertEqual(draws[0],draws[3])  # Discarded pilot stream versus reset full-run stream.
                self.assertTrue(all(c['seed']==seed and c['settings']['seed']==seed for c in contexts))
                self.assertTrue(all(c['settings']['learning_rate']==5e-6 and c['optimizer']['lr']==5e-6 for c in contexts))
                report=json.loads(next((root/'runs/training').glob('*/report.json')).read_text())
                self.assertEqual(report['contract']['seed'],seed)


if __name__=='__main__':unittest.main()
