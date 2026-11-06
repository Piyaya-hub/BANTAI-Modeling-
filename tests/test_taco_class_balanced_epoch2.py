"""D064 metadata/LR/orchestration fixtures, never actual training or photos."""
import unittest
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from scripts import taco_class_balanced_epoch2 as cb


class Dataset:
    def __init__(self,n=221):
        self.images=[{'id':i} for i in range(n)]
        self.objects={i:[{'category_id':4 if i<61 else 0 if i<91 else 2}] for i in range(n)}
    def __len__(self):return len(self.images)


class ClassBalancedTests(unittest.TestCase):
    def test_saved_replay_counts_rule_blend_and_checkpoint_guards(self):
        from scripts import taco_class_balanced_review as review
        contract={'continuation':{'lr_rule':cb.RULE}};ev={'metrics':{'AP':.2}}
        evaluations={n:{'path':n+'.json','sha256':n,'forwards':47,'parameters_unchanged':True,'metrics':ev['metrics'],
            'RNG_restored_exactly':True,'frozen_parameter_sha256':'frozen'} for n in ('350','442','blend_2575')}
        evaluations['blend_2575'].update(weights={'350':.25,'442':.75},frozen_parameters_identical=True,trainable_tensors=558)
        audit={'authorization':'D-064','status':'completed','contract':contract,'protocol':review.PROTOCOL,'lr_rule':cb.RULE,
            'new_optimizer_updates':221,'new_backward_passes':221,'training_forwards':221,'validation_forwards':141,
            'reference_GPU_forwards':0,'test_split_access':0,'original_checkpoint_bundles_before':55,'original_checkpoint_bundles_after':55,
            'new_checkpoint_bundles_after':5,'lr_assignment_counts':{'pure_plastic_film':61,'other':160,'cigarette_full_lr':30},
            'protected_after':{'unchanged':True},'restore_proof':{'exact_model_optimizer_RNG':True},
            'core_source_bytes_before':{'a':'x'},'core_source_bytes_after':{'a':'x'},'evaluations':evaluations,
            'frozen_parameter_sha256':'frozen','checkpoints':{'step-000350':{'path':'cp350','manifest_sha256':'350'},'epoch-02':{'path':'cp442','manifest_sha256':'442'}}}
        entry={'class_balanced':{'path':'audit.json'},'panel_labels':{},'panels':{
            slot:{'evaluation':{k:evaluations[n][k] for k in ('path','sha256')},'cb_state':n}
            for slot,n in [('pretrained','350'),('epoch-01','442'),('epoch-02','blend_2575')]}}
        for mutation in (None,'counts','blend','metrics','position','rule'):
            current=copy.deepcopy(audit)
            if mutation=='counts':current['lr_assignment_counts']['pure_plastic_film']=60
            if mutation=='blend':current['evaluations']['blend_2575']['weights']={'350':.5,'442':.5}
            if mutation=='metrics':current['evaluations']['350']['metrics']={}
            if mutation=='rule':current['lr_rule']={}
            def read(p):
                name=str(p)
                if name=='audit.json':return current
                if name.startswith('cp'):
                    return {'status':'complete','context':contract,'position':{} if mutation=='position' else cb.position(350 if name.startswith('cp350') else 442)}
                return ev
            with (self.subTest(mutation=mutation),patch('scripts.taco_saved_review.checked_file',side_effect=lambda a:Path(a['path'])),
                  patch.object(review,'strict_load',side_effect=read),patch.object(review,'digest',return_value='manifest'),
                  patch.object(review,'saved_rows',return_value={777:[]})):
                args=('manifest','choice',{'images':[{'image_id':777}]},entry,SimpleNamespace(images=[{'id':777}],data={}),{'contract':contract})
                if mutation:
                    with self.assertRaises(ValueError):review.load_class_balanced_review(*args)
                else:
                    _,models,record=review.load_class_balanced_review(*args)
                    self.assertEqual(set(models),{'pretrained','epoch-01','epoch-02'});self.assertEqual(record['provenance']['new_inference_calls'],0)

    def test_exact_pure_only_rule_and_all_groups_without_moment_reset(self):
        opt=SimpleNamespace(param_groups=[{'lr':1e-5},{'lr':1e-5}],state={'moment':'preserved'})
        for cats,expected in [([4],1e-6),([4,4],1e-6),([4,0],1e-5),([0],1e-5),([1,2,3,5,6,7],1e-5)]:
            self.assertEqual(cb.apply_batch_lr(opt,[{'category_id':c} for c in cats]),expected)
            self.assertTrue(all(g['lr']==expected for g in opt.param_groups))
            self.assertEqual(opt.state,{'moment':'preserved'})
        for bad in ([],[{'category_id':True}],[{'category_id':8}],[{}]):
            with self.assertRaises(ValueError):cb.batch_lr(bad)

    def test_locked_counts_and_every_cigarette_image_full_lr(self):
        train=Dataset();inventory=cb.training_inventory(train)
        self.assertEqual(inventory['counts'],{'pure_plastic_film':61,'other':160,'cigarette':30})
        self.assertTrue(all(cb.batch_lr(train.objects[i])==1e-5 for i in inventory['cigarette_image_ids']))
        train.objects[0].append({'category_id':0})
        with self.assertRaises(ValueError):cb.training_inventory(train)

    def test_exact_order_221_assignments_five_positions_two_rng_isolated_evaluations(self):
        train=Dataset();val=Dataset(47);opt=SimpleNamespace(param_groups=[{'lr':1e-5}])
        rows=[];cp=[];evaluations=[];saved={};snapshot={'torch':torch.tensor([2,3])}
        def save(path,*args):saved[path]={'position':args[5]}
        with (patch.object(cb,'state_hash',return_value='frozen'),patch.object(cb,'get_batch',return_value=({},[])),
              patch.object(cb,'move_batch',side_effect=lambda b,d:b),patch.object(cb,'train_step',return_value={'loss':1.}),
              patch.object(cb,'save_checkpoint',side_effect=save),patch.object(cb,'verify_checkpoint',side_effect=lambda p,c:saved[p]),
              patch.object(cb,'evaluate',return_value={'metrics':{}}) as ev,patch.object(cb,'rng_state',return_value=snapshot),
              patch.object(cb,'restore_rng') as restore):
            cb.run_epoch2(None,None,None,opt,train,val,{},Path('new'),on_stage=lambda *v:None,on_step=rows.append,
                on_checkpoint=lambda p,v:cp.append(v['position']),on_evaluation=lambda *v:evaluations.append(v))
        self.assertEqual(len(rows),221)
        self.assertEqual([v['dataset_indexes'] for v in rows],cb.epoch_batches(221,1,42,1))
        self.assertEqual([v['global_step'] for v in rows],list(range(222,443)))
        self.assertEqual(sum(v['lr']==1e-6 for v in rows),61);self.assertEqual(sum(v['lr']==1e-5 for v in rows),160)
        self.assertTrue(all(v['lr']==1e-5 for v in rows if 0 in v['annotation_category_ids']))
        self.assertEqual(cp,[cb.position(s) for s in cb.SAVE_STEPS]);self.assertEqual(ev.call_count,2)
        self.assertEqual([v[0] for v in evaluations],[350,442]);self.assertEqual(restore.call_count,2)
