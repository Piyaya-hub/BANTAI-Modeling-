"""D062 schedule/orchestration fixtures; no real model, photos or updates."""
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import copy
from unittest.mock import patch
import torch
from scripts import taco_lr_decay_epoch2 as decay


class CosineContinuationTests(unittest.TestCase):
    def test_saved_review_loading_and_audit_guards(self):
        from scripts import taco_lr_decay_review as review
        contract={'continuation':{'schedule':decay.SCHEDULE}}
        ev={'metrics':{'AP':.1}}
        refs={'constant-350':{'path':'reference.json','sha256':'reference'}}
        proofs={str(s):{'path':f'eval{s}.json','sha256':str(s),'forwards':47,'RNG_restored_exactly':True,
                        'parameters_unchanged':True,'frozen_parameter_sha256':'frozen','metrics':ev['metrics']} for s in (350,442)}
        audit={'authorization':'D-062','status':'completed','contract':contract,'protocol':review.PROTOCOL,
            'schedule':decay.SCHEDULE,'new_optimizer_updates':221,'new_backward_passes':221,'training_forwards':221,
            'validation_forwards':94,'reference_GPU_forwards':0,'test_split_access':0,
            'original_checkpoint_bundles_before':50,'original_checkpoint_bundles_after':50,'new_checkpoint_bundles_after':5,
            'protected_after':{'unchanged':True},'core_source_bytes_before':{'a':'x'},'core_source_bytes_after':{'a':'x'},
            'restore_proof':{'exact_model_optimizer_RNG':True},'frozen_parameter_sha256':'frozen',
            'references':refs,'evaluations':proofs,'checkpoints':{
                'step-000350':{'path':'cp350','manifest_sha256':'cp350'},'epoch-02':{'path':'cp442','manifest_sha256':'cp442'}}}
        entry={'lr_decay':{'path':'audit.json'},'panel_labels':{'pretrained':'Constant350','epoch-01':'Cosine350','epoch-02':'Cosine442'},
            'panels':{'pretrained':{'evaluation':refs['constant-350'],'reference':'constant-350'},
                **{slot:{'evaluation':{k:proofs[str(s)][k] for k in ('path','sha256')},'decay_step':s} for slot,s in [('epoch-01',350),('epoch-02',442)]}}}
        for mutation in (None,'counts','schedule','frozen','metrics','position','core'):
            current=copy.deepcopy(audit)
            if mutation=='counts':current['new_optimizer_updates']=222
            if mutation=='schedule':current['schedule']={}
            if mutation=='frozen':current['evaluations']['350']['frozen_parameter_sha256']='changed'
            if mutation=='metrics':current['evaluations']['442']['metrics']={}
            if mutation=='core':current['core_source_bytes_after']={}
            def read(p):
                name=str(p)
                if name=='audit.json':return current
                if name.startswith('cp'):
                    s=350 if name.startswith('cp350') else 442
                    return {'context':contract,'position':{} if mutation=='position' else decay.position(s),'status':'complete'}
                return ev
            with (self.subTest(mutation=mutation),patch('scripts.taco_saved_review.checked_file',side_effect=lambda a:Path(a['path'])),
                  patch.object(review,'strict_load',side_effect=read),patch.object(review,'digest',return_value='manifest'),
                  patch.object(review,'saved_rows',return_value={777:[]})):
                args=('manifest','choice',{'images':[{'image_id':777}]},entry,SimpleNamespace(images=[{'id':777}],data={}),{'contract':contract})
                if mutation:
                    with self.assertRaises(ValueError):review.load_lr_decay_review(*args)
                else:
                    _,models,record=review.load_lr_decay_review(*args)
                    self.assertEqual(set(models),{'pretrained','epoch-01','epoch-02'})
                    self.assertEqual(record['saved_configuration']['cosine_panel_learning_rates']['epoch-02'],1e-6)

    def test_exact_endpoints_midpoint_monotonic_all_groups_and_no_state_reset(self):
        values=[decay.cosine_lr(k) for k in range(221)]
        self.assertAlmostEqual(values[0],5e-6,places=20)
        self.assertAlmostEqual(values[-1],1e-6,places=20)
        self.assertAlmostEqual(values[110],3e-6,places=20)
        self.assertTrue(all(a>b for a,b in zip(values,values[1:])))
        opt=SimpleNamespace(param_groups=[{'lr':1e-5,'other':1},{'lr':1e-5}],state={'moment':'preserved'})
        self.assertEqual(decay.apply_lr(opt,128),decay.cosine_lr(128))
        self.assertTrue(all(g['lr']==values[128] for g in opt.param_groups));self.assertEqual(opt.state,{'moment':'preserved'})
        for k in (-1,221,True,1.5):
            with self.assertRaises(ValueError):decay.cosine_lr(k)

    def test_full_bounded_epoch_checkpoint_positions_and_rng_isolation(self):
        class Dataset:
            def __init__(self,n):self.n=n;self.images=[{'id':i} for i in range(n)]
            def __len__(self):return self.n
        train=Dataset(221);val=Dataset(47);opt=SimpleNamespace(param_groups=[{'lr':1e-5}])
        stages=[];rows=[];checkpoints=[];evaluations=[];saved={}
        def save(cp,*args):saved[cp]={'position':args[5]}
        snapshot={'torch':torch.tensor([1,2,3])}
        with (patch.object(decay,'state_hash',return_value='unchanged'),
              patch.object(decay,'get_batch',return_value=({},[])),patch.object(decay,'move_batch',side_effect=lambda b,d:b),
              patch.object(decay,'train_step',return_value={'loss':1.}),
              patch.object(decay,'save_checkpoint',side_effect=save),
              patch.object(decay,'verify_checkpoint',side_effect=lambda cp,c:saved[cp]),
              patch.object(decay,'evaluate',return_value={'metrics':{}}) as ev,
              patch.object(decay,'rng_state',return_value=snapshot),patch.object(decay,'restore_rng') as restore):
            decay.run_epoch2(None,None,None,opt,train,val,{},Path('new'),
                on_stage=lambda *v:stages.append(v),on_step=rows.append,
                on_checkpoint=lambda cp,p:checkpoints.append(p['position']),
                on_evaluation=lambda *v:evaluations.append(v))
        self.assertEqual(len(rows),221);self.assertEqual([r['global_step'] for r in rows],list(range(222,443)))
        self.assertEqual([r['dataset_indexes'] for r in rows],decay.epoch_batches(221,1,42,1))
        self.assertEqual([r['lr'] for r in rows],[decay.cosine_lr(k) for k in range(221)])
        self.assertEqual(checkpoints,[decay.position(s) for s in decay.SAVE_STEPS])
        self.assertEqual([e[0] for e in evaluations],[350,442]);self.assertEqual(ev.call_count,2)
        self.assertEqual(restore.call_count,2);self.assertEqual(stages[-1],('completed',442))

    def test_optimizer_rng_digest_detects_tampering_and_position_boundaries(self):
        self.assertEqual(decay.tree_digest({'x':torch.tensor([1.,2.])}),decay.tree_digest({'x':torch.tensor([1.,2.])}))
        self.assertNotEqual(decay.tree_digest({'x':torch.tensor([1.,2.])}),decay.tree_digest({'x':torch.tensor([1.,3.])}))
        self.assertEqual(decay.position(350),{'epoch':1,'next_batch':129,'global_step':350})
        self.assertEqual(decay.position(442),{'epoch':2,'next_batch':0,'global_step':442})
        for step in (221,443):
            with self.assertRaises(ValueError):decay.position(step)
