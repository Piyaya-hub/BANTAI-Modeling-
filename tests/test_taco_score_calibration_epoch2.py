"""D067 exact auxiliary arithmetic and bounded orchestration; no live updates."""
import unittest,math
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from scripts import taco_score_calibration_epoch2 as cal
cb=cal
from scripts import taco_score_calibration_loss as objective
from tests.test_taco_class_balanced_epoch2 import Dataset


def fixture():
    ids=torch.tensor([[101]+sum(([10+c,1012] for c in range(8)),[])+[102]])
    logits=torch.full((1,900,20),math.log(.01/.99))
    boxes=torch.tensor([.2,.2,.1,.1]).repeat(1,900,1)
    target={'class_labels':torch.tensor([0]),'boxes':torch.tensor([[.5,.5,.2,.2]])}
    return logits,boxes,ids,target


class ScoreCalibrationTests(unittest.TestCase):
    def test_exact_positive_selection_weight_sum_film_zero_and_detached_matching(self):
        logits,boxes,ids,target=fixture();boxes[0,1]=target['boxes'][0]
        logits[0,0,1]=math.log(.6/.4)
        pos,absent,audit=objective.calibration_losses(logits.requires_grad_(),boxes.requires_grad_(),ids,target)
        self.assertEqual(audit['positive_assignments'][0]['query_id'],1)
        self.assertAlmostEqual(float(pos),-1.5*math.log(.01+1e-6),places=5)
        self.assertEqual(float(absent),0.);self.assertTrue(pos.requires_grad)
        self.assertEqual(audit['class_probs_shape'],[1,900,8]);self.assertEqual(audit['class_logits_shape'],[1,900,8])
        target['class_labels']=torch.tensor([4]);pos,_,_=objective.calibration_losses(logits,boxes,ids,target)
        self.assertEqual(float(pos),0.)
        target['class_labels']=torch.tensor([1,1]);target['boxes']=target['boxes'].repeat(2,1)
        pos,_,audit=objective.calibration_losses(logits,boxes,ids,target)
        self.assertAlmostEqual(float(pos),-1.5*math.log(.01+1e-6),places=5)
        self.assertEqual([v['query_id'] for v in audit['positive_assignments']],[1,1])

    def test_absent_mask_selected_mean_present_exclusion_and_epsilon(self):
        logits,boxes,ids,target=fixture()
        logits[0,3,5]=math.log(.1/.9)  # class2 token at index5, absent
        logits[0,4,1]=10.             # present cigarette: never absent penalty
        _,absent,audit=objective.calibration_losses(logits,boxes,ids,target)
        self.assertAlmostEqual(float(absent),-10*math.log(.9+1e-6)*.1**2,places=6)
        self.assertEqual(audit['absent_class_passing_queries']['2'],1)
        self.assertNotIn('0',audit['absent_class_passing_queries'])
        logits[0,3,5]=100.
        pos,absent,_=objective.calibration_losses(logits,boxes,ids,target)
        self.assertTrue(torch.isfinite(pos+absent))

    def test_mean_sigmoid_is_not_sigmoid_mean_and_masked_tail_is_excluded(self):
        ids=torch.tensor([[101,10,11,1012]+sum(([20+c,1012] for c in range(7)),[])+[102]])
        logits=torch.zeros(1,900,24);logits[:,:,1]=-4.;logits[:,:,19:]=float('-inf')
        probs,means=objective.class_scores(logits,ids)
        self.assertAlmostEqual(float(probs[0,0,0]),(torch.sigmoid(torch.tensor(-4.)).item()+.5)/2,places=6)
        self.assertEqual(float(means[0,0,0]),-2.)
        self.assertNotAlmostEqual(float(probs[0,0,0]),float(means[0,0,0].sigmoid()))
        with self.assertRaises(ValueError):objective.class_scores(logits[:,:899],ids)

    def test_LR_and_92_batch_suffix_two_saves_one_isolated_validation(self):
        train=Dataset();val=Dataset(47);opt=SimpleNamespace(param_groups=[{'lr':0}]);rows=[];cps=[];evaluations=[];saved={}
        def save(p,*args):saved[p]={'position':args[5]}
        snapshot={'torch':torch.tensor([1,2])}
        with (patch.object(cal,'state_hash',return_value='frozen'),patch.object(cal,'get_batch',return_value=({},[])),
              patch.object(cal,'move_batch',side_effect=lambda b,d:b),patch.object(cal,'train_step',return_value={'loss':1.}),
              patch.object(cal,'save_checkpoint',side_effect=save),patch.object(cal,'verify_checkpoint',side_effect=lambda p,c:saved[p]),
              patch.object(cal,'evaluate',return_value={}) as ev,patch.object(cal,'rng_state',return_value=snapshot),patch.object(cal,'restore_rng') as restore):
            cal.run_epoch2(None,None,None,opt,train,val,{},Path('new'),on_stage=lambda *v:None,on_step=rows.append,
                on_checkpoint=lambda p,v:cps.append(v['position']),on_evaluation=lambda *v:evaluations.append(v))
        self.assertEqual([v['global_step'] for v in rows],list(range(351,443)))
        self.assertEqual([v['dataset_indexes'] for v in rows],cal.epoch_batches(221,1,42,1)[129:])
        self.assertTrue(all(v['lr']==(1e-6 if v['annotation_category_ids']==[4] else 8e-6) for v in rows))
        self.assertEqual(cps,[cal.position(400),cal.position(442)]);self.assertEqual(ev.call_count,1);self.assertEqual(restore.call_count,1)

    def test_saved_replay_counts_rule_blend_and_checkpoint_guards(self):
        from scripts import taco_score_calibration_review as review
        contract={'continuation':{'lr_rule':cb.RULE}};ev={'metrics':{'AP':.2}}
        evaluations={n:{'path':n+'.json','sha256':n,'forwards':47,'parameters_unchanged':True,'metrics':ev['metrics'],
            'RNG_restored_exactly':True,'frozen_parameter_sha256':'frozen'} for n in ('442','442','blend')}
        evaluations['blend'].update(weights={'source350':.25,'cal442':.75},frozen_parameters_identical=True,trainable_tensors=558)
        audit={'authorization':'D-067','status':'completed','contract':contract,'protocol':review.PROTOCOL,'lr_rule':cb.RULE,
            'new_optimizer_updates':92,'new_backward_passes':92,'training_forwards':92,'validation_forwards':94,
            'reference_GPU_forwards':0,'test_split_access':0,'original_checkpoint_bundles_before':62,'original_checkpoint_bundles_after':62,
            'new_checkpoint_bundles_after':2,'lr_assignment_counts':{'pure_film':25,'other':67},
            'protected_after':{'unchanged':True},'restore_proof':{'exact_model_optimizer_RNG':True},
            'core_source_bytes_before':{'a':'x'},'core_source_bytes_after':{'a':'x'},'evaluations':evaluations,
            'frozen_parameter_sha256':'frozen','checkpoints':{'step-000350':{'path':'cp350','manifest_sha256':'442'},'epoch-02':{'path':'cp442','manifest_sha256':'442'}}}
        entry={'score_calibration':{'path':'audit.json'},'panel_labels':{},'panels':{
            slot:{'evaluation':{k:evaluations[n][k] for k in ('path','sha256')},'cal_state':n}
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
                    with self.assertRaises(ValueError):review.load_score_calibration_review(*args)
                else:
                    _,models,record=review.load_score_calibration_review(*args)
                    self.assertEqual(set(models),{'pretrained','epoch-01','epoch-02'});self.assertEqual(record['provenance']['new_inference_calls'],0)
