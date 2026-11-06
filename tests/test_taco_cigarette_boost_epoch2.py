"""Bounded late continuation fixtures: no live training or photo reads."""
import unittest
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from scripts import taco_cigarette_boost_epoch2 as cb
from tests.test_taco_class_balanced_epoch2 import Dataset


class CigaretteBoostTests(unittest.TestCase):
    def test_external_display_transition_requires_exact_hash_reconstruction(self):
        import hashlib
        from scripts.taco_cigarette_boost_review import integrity_preserved
        from scripts.persistence import strict_payload
        closed={'display_status':'closed','source_verified_photo_reads':16,'saved_prediction_sha':'unchanged'}
        pending=copy.deepcopy(closed);pending.pop('source_verified_photo_reads');pending['display_status']='pending'
        sha=hashlib.sha256(strict_payload(pending).encode('utf-8')).hexdigest()
        audit={'protected_after':{'unchanged':False,'changed':['history.json'],'model_data_checkpoint_files_unchanged':True,
                                 'only_change_is_verified_external_display_transition':True},
               'external_display_transition':{'path':'history.json','after_sha256':'after','before_sha256':sha,'reconstructed_pending_sha256':sha,
                    'changed_fields':{'display_status':['pending','closed'],'source_verified_photo_reads':['absent',16]},
                    'historical_file_not_modified_by_reconciliation':True,'extra_training_updates':0,'extra_model_forwards':0}}
        for mutation in (None,'predictions','updates','other_file'):
            current=copy.deepcopy(audit);value=copy.deepcopy(closed)
            if mutation=='predictions':value['saved_prediction_sha']='tampered'
            if mutation=='updates':current['external_display_transition']['extra_training_updates']=1
            if mutation=='other_file':current['protected_after']['changed'].append('weights')
            with (self.subTest(mutation=mutation),patch('scripts.taco_saved_review.checked_file',return_value=Path('history.json')),
                  patch('scripts.taco_cigarette_boost_review.strict_load',return_value=value)):
                self.assertEqual(integrity_preserved(current),mutation is None)
    def test_explicit_top15_cli_propagation_and_bounds(self):
        from scripts import taco_visual_review as view
        with (patch('scripts.taco_saved_review.load_saved_review',return_value=(None,{},{})),
              patch.object(Path,'mkdir'),patch.object(view,'atomic_json'),patch.object(view,'show_review') as gui):
            self.assertEqual(view.main(['--saved-review','manifest','--run-choice','choice','--gui','--image-id','777','--box-top-k','15']),0)
            self.assertEqual(gui.call_args.kwargs,{'box_top_k':15})
        for args in (['--box-top-k','15'],['--gui','--box-top-k','0'],['--gui','--box-top-k','101']):
            with self.assertRaises(SystemExit) as error:view.main(args)
            self.assertEqual(error.exception.code,2)
    def test_cigarette_priority_pure_film_and_other_all_groups(self):
        opt=SimpleNamespace(param_groups=[{'lr':0},{'lr':0}],state={'moments':'preserved'})
        for ids,expected in [([0],1.5e-5),([0,4],1.5e-5),([4,4],1e-7),([4,3],5e-6),([1,2,3,5,6,7],5e-6)]:
            self.assertEqual(cb.apply_batch_lr(opt,[{'category_id':i} for i in ids]),expected)
            self.assertTrue(all(g['lr']==expected for g in opt.param_groups))
            self.assertEqual(opt.state,{'moments':'preserved'})
        for bad in ([],[{'category_id':True}],[{'category_id':8}]):
            with self.assertRaises(ValueError):cb.batch_lr(bad)

    def test_late_inventory_uses_only_remaining_sampler_suffix(self):
        train=Dataset();inv=cb.late_inventory(train)
        ids=[train.images[i[0]]['id'] for i in cb.epoch_batches(221,1,42,1)[129:]]
        self.assertEqual(sum(inv['late_counts'].values()),92)
        self.assertEqual(set(sum(inv['late_tier_image_ids'].values(),[])),set(ids))
        self.assertEqual(inv['full_epoch']['counts']['cigarette'],30)

    def test_exact_92_steps_two_checkpoints_and_one_rng_isolated_validation(self):
        train=Dataset();val=Dataset(47);opt=SimpleNamespace(param_groups=[{'lr':1e-5}])
        rows=[];checkpoints=[];evaluations=[];saved={};snapshot={'torch':torch.tensor([2,3])}
        def save(path,*args):saved[path]={'position':args[5]}
        with (patch.object(cb,'state_hash',return_value='frozen'),patch.object(cb,'get_batch',return_value=({},[])),
              patch.object(cb,'move_batch',side_effect=lambda b,d:b),patch.object(cb,'train_step',return_value={'loss':1.}),
              patch.object(cb,'save_checkpoint',side_effect=save),patch.object(cb,'verify_checkpoint',side_effect=lambda p,c:saved[p]),
              patch.object(cb,'evaluate',return_value={'metrics':{}}) as ev,patch.object(cb,'rng_state',return_value=snapshot),
              patch.object(cb,'restore_rng') as restore):
            cb.run_epoch2(None,None,None,opt,train,val,{},Path('new'),on_stage=lambda *v:None,on_step=rows.append,
                on_checkpoint=lambda p,v:checkpoints.append(v['position']),on_evaluation=lambda *v:evaluations.append(v))
        self.assertEqual([r['global_step'] for r in rows],list(range(351,443)))
        self.assertEqual([r['batch_index'] for r in rows],list(range(129,221)))
        self.assertEqual([r['dataset_indexes'] for r in rows],cb.epoch_batches(221,1,42,1)[129:])
        self.assertTrue(all(r['lr']==cb.batch_lr(train.objects[r['image_ids'][0]]) for r in rows))
        self.assertEqual(checkpoints,[cb.position(400),cb.position(442)])
        self.assertEqual([v[0] for v in evaluations],[442]);self.assertEqual(ev.call_count,1);self.assertEqual(restore.call_count,1)

    def test_saved_replay_counts_rule_blend_and_checkpoint_guards(self):
        from scripts import taco_cigarette_boost_review as review
        contract={'continuation':{'lr_rule':cb.RULE}};ev={'metrics':{'AP':.2}}
        evaluations={n:{'path':n+'.json','sha256':n,'forwards':47,'parameters_unchanged':True,'metrics':ev['metrics'],
            'RNG_restored_exactly':True,'frozen_parameter_sha256':'frozen'} for n in ('442','442','blend')}
        evaluations['blend'].update(weights={'new442':.50,'constant442':.50},frozen_parameters_identical=True,trainable_tensors=558)
        audit={'authorization':'D-066','status':'completed','contract':contract,'protocol':review.PROTOCOL,'lr_rule':cb.RULE,
            'new_optimizer_updates':92,'new_backward_passes':92,'training_forwards':92,'validation_forwards':94,
            'reference_GPU_forwards':0,'test_split_access':0,'original_checkpoint_bundles_before':60,'original_checkpoint_bundles_after':60,
            'new_checkpoint_bundles_after':2,'lr_assignment_counts':{'cigarette':11,'pure_film':25,'other':56},
            'protected_after':{'unchanged':True},'restore_proof':{'exact_model_optimizer_RNG':True},
            'core_source_bytes_before':{'a':'x'},'core_source_bytes_after':{'a':'x'},'evaluations':evaluations,
            'frozen_parameter_sha256':'frozen','checkpoints':{'step-000350':{'path':'cp350','manifest_sha256':'442'},'epoch-02':{'path':'cp442','manifest_sha256':'442'}}}
        entry={'cigarette_boost':{'path':'audit.json'},'panel_labels':{},'panels':{
            slot:{'evaluation':{k:evaluations[n][k] for k in ('path','sha256')},'boost_state':n}
            for slot,n in [('pretrained','442'),('epoch-01','442'),('epoch-02','blend')]}}
        for mutation in (None,'counts','blend','metrics','position','rule'):
            current=copy.deepcopy(audit)
            if mutation=='counts':current['lr_assignment_counts']['pure_film']=24
            if mutation=='blend':current['evaluations']['blend']['weights']={'442':.5,'442':.5}
            if mutation=='metrics':current['evaluations']['442']['metrics']={}
            if mutation=='rule':current['lr_rule']={}
            def read(p):
                name=str(p)
                if name=='audit.json':return current
                if name.startswith('cp'):
                    return {'status':'complete','context':contract,'position':{} if mutation=='position' else cb.position(442)}
                return ev
            with (self.subTest(mutation=mutation),patch('scripts.taco_saved_review.checked_file',side_effect=lambda a:Path(a['path'])),
                  patch.object(review,'strict_load',side_effect=read),patch.object(review,'digest',return_value='manifest'),
                  patch.object(review,'saved_rows',return_value={777:[]})):
                args=('manifest','choice',{'images':[{'image_id':777}]},entry,SimpleNamespace(images=[{'id':777}],data={}),{'contract':contract})
                if mutation:
                    with self.assertRaises(ValueError):review.load_cigarette_boost_review(*args)
                else:
                    _,models,record=review.load_cigarette_boost_review(*args)
                    self.assertEqual(set(models),{'pretrained','epoch-01','epoch-02'});self.assertEqual(record['provenance']['new_inference_calls'],0)
