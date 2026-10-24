"""Scale CLI propagation with generated fixtures and zero optimizer updates."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch
from scripts import train_taco as trainer


class Scale576Tests(unittest.TestCase):
    def test_invalid_scale_rejected_before_dataset_access(self):
        for short,long in ((384,577),(577,576),(191,576)):
            with self.subTest(short=short,long=long), patch.object(trainer,'dataset_context') as data:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    trainer.main(['--shortest-edge',str(short),'--longest-edge',str(long)])
                data.assert_not_called()

    def test_selected_and_default_scale_reaches_all_training_consumers(self):
        for scale in ((384,576),(256,384)):
            with self.subTest(scale=scale),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);base=root/'base';base.mkdir();(base/'fixture').write_text('generated')
                model=torch.nn.Linear(2,1);adapter=MagicMock();adapter.manifest.return_value={}
                ds=MagicMock();ds.__len__.return_value=3;ds.photo_reads=0
                ds.images=[{'id':i} for i in range(3)];ds.objects={i:[{}] for i in range(3)}
                contexts=[];seen={'preflight':[],'evaluate':[],'get_batch':[]}
                def preflight(ds,processor,short,long):seen['preflight'].append((short,long));return []
                def evaluate(m,p,a,ds,device,short,long):
                    seen['evaluate'].append((short,long));return {'metrics':{},'mean_loss_per_image':1.}
                def batch(ds,indexes,p,short,long):
                    seen['get_batch'].append((short,long));return {},[{'image_id':i} for i in indexes]
                def save(folder,m,p,a,o,contract,position,evidence):contexts.append(copy.deepcopy(contract))
                mocks={'ROOT':root,'SNAPSHOT':base}
                returns={'source_identity':{},'dataset_context':{},'require_rights':{'record_sha256':'fixture'},
                    'require_training_environment':{'packages':{},'python':'fixture','platform':'fixture'},
                    'load_model':(model,MagicMock(),adapter),'CocoTrainingDataset':ds,'state_hash':'fixture',
                    'make_optimizer':MagicMock(),'train_step':{'loss':1.}}
                effects={'preflight':preflight,'evaluate':evaluate,'get_batch':batch,
                         'save_checkpoint':save,'move_batch':lambda b,d:b}
                with contextlib.ExitStack() as stack:
                    for name,value in mocks.items():stack.enter_context(patch.object(trainer,name,value))
                    for name,value in returns.items():stack.enter_context(patch.object(trainer,name,return_value=value))
                    for name,value in effects.items():stack.enter_context(patch.object(trainer,name,side_effect=value))
                    stack.enter_context(patch('torch.optim.AdamW.step',side_effect=AssertionError('No updates')))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                    args=['--device','cpu']
                    if scale==(384,576):args+=['--shortest-edge','384','--longest-edge','576']
                    self.assertEqual(trainer.main(args),0)
                for values in seen.values():self.assertTrue(values and all(v==scale for v in values))
                self.assertTrue(contexts)
                for contract in contexts:
                    self.assertEqual((contract['settings']['shortest_edge'],contract['settings']['longest_edge']),scale)
