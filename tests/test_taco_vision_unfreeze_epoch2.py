"""CPU structure/state fixtures; no detector execution or optimization updates."""
from collections import OrderedDict
from types import SimpleNamespace
import unittest
import copy
from pathlib import Path
from unittest.mock import patch
import torch
from scripts import taco_vision_unfreeze_epoch2 as vu


class Model:
    def __init__(self):
        self.named=OrderedDict((f'neck.p{i}',torch.nn.Parameter(torch.ones(1))) for i in range(558))
        for suffix in ('embeddings.weight','encoder.layers.0.weight','encoder.layers.1.weight',
                       'encoder.layers.2.weight','encoder.layers.3.weight','hidden_states_norms.weight'):
            self.named[vu.VISION_ROOT+suffix]=torch.nn.Parameter(torch.ones(1),requires_grad=False)
        self.named['model.text_backbone.weight']=torch.nn.Parameter(torch.ones(1),requires_grad=False)
    def named_parameters(self):return iter(self.named.items())


class VisionUnfreezeTests(unittest.TestCase):
    def test_bounded_epoch_uses_two_lr_rules_and_exact_checkpoint_positions(self):
        class Data:
            def __init__(self,n):
                self.images=[{'id':i} for i in range(n)]
                self.objects={i:[{'category_id':4 if i<61 else 0 if i<91 else 2}] for i in range(n)}
            def __len__(self):return len(self.images)
        train=Data(221);validation=Data(47);optimizer=SimpleNamespace(param_groups=[{},{}])
        rows=[];checkpoints=[];evaluations=[];saved={};rng={'x':torch.tensor([1,2])}
        def save(cp,*args):saved[cp]={'position':args[5]}
        with (patch.object(vu,'state_hash',return_value='frozen'),patch.object(vu,'get_batch',return_value=({},[])),
              patch.object(vu,'move_batch',side_effect=lambda b,d:b),patch.object(vu,'audited_train_step',return_value={'loss':1.}),
              patch.object(vu,'save_checkpoint',side_effect=save),patch.object(vu,'verify_checkpoint',side_effect=lambda p,c:saved[p]),
              patch.object(vu,'rng_state',return_value=rng),patch.object(vu,'restore_rng') as restore,
              patch.object(vu,'evaluate',return_value={'metrics':{}})):
            vu.run_epoch2(None,None,None,optimizer,train,validation,{},Path('new'),on_stage=lambda *v:None,on_step=rows.append,
                on_checkpoint=lambda p,v:checkpoints.append(v['position']),on_evaluation=lambda *v:evaluations.append(v))
        self.assertEqual(len(rows),221)
        self.assertEqual([v['dataset_indexes'] for v in rows],vu.epoch_batches(221,1,42,1))
        self.assertEqual(sum((v['lr'],v['vision_lr'])==(1e-6,5e-7) for v in rows),61)
        self.assertEqual(sum((v['lr'],v['vision_lr'])==(1e-5,2e-6) for v in rows),160)
        self.assertEqual(checkpoints,[vu.position(s) for s in vu.SAVE_STEPS]);self.assertEqual([v[0] for v in evaluations],[350,442])
        self.assertEqual(restore.call_count,2)

    def test_saved_vision_review_audit_and_blend_guards(self):
        from scripts import taco_vision_unfreeze_review as review
        contract={'continuation':{'lr_rule':vu.RULE}};ev={'metrics':{'AP':.1}}
        evaluations={n:{'path':n+'.json','sha256':n,'forwards':47,'parameters_unchanged':True,'metrics':ev['metrics'],
            'RNG_restored_exactly':True,'frozen_parameter_sha256':'frozen'} for n in ('350','442','blend_2575')}
        evaluations['blend_2575'].update(weights={'350':.25,'442':.75},frozen_parameters_identical=True,trainable_tensors=697)
        audit={'authorization':'D-065','status':'completed','contract':contract,'protocol':review.PROTOCOL,'lr_rule':vu.RULE,
            'new_optimizer_updates':221,'new_backward_passes':221,'training_forwards':221,'validation_forwards':141,
            'reference_GPU_forwards':0,'test_split_access':0,'original_checkpoint_bundles_before':60,'original_checkpoint_bundles_after':60,
            'new_checkpoint_bundles_after':5,'lr_assignment_counts':{'pure_plastic_film':61,'other':160,'cigarette_full_lr':30},
            'protected_after':{'unchanged':True},'restore_proof':{'exact_model_optimizer_RNG':True},
            'core_source_bytes_before':{'a':'x'},'core_source_bytes_after':{'a':'x'},'evaluations':evaluations,
            'trainable_extension':{'original_tensors':558,'vision_tensors':139,'original_optimizer_preserved':True,'RNG_preserved':True},
            'frozen_parameter_sha256':'frozen','checkpoints':{'step-000350':{'path':'cp350','manifest_sha256':'350'},'epoch-02':{'path':'cp442','manifest_sha256':'442'}}}
        entry={'vision_unfreeze':{'path':'audit.json'},'panel_labels':{},'panels':{
            slot:{'evaluation':{k:evaluations[n][k] for k in ('path','sha256')},'vu_state':n}
            for slot,n in [('pretrained','350'),('epoch-01','442'),('epoch-02','blend_2575')]}}
        for mutation in (None,'group','count','blend','position'):
            current=copy.deepcopy(audit)
            if mutation=='group':current['trainable_extension']['original_optimizer_preserved']=False
            if mutation=='count':current['trainable_extension']['vision_tensors']=140
            if mutation=='blend':current['evaluations']['blend_2575']['trainable_tensors']=558
            def read(p):
                name=str(p)
                if name=='audit.json':return current
                if name.startswith('cp'):
                    return {'status':'complete','context':contract,'position':{} if mutation=='position' else vu.position(350 if name.startswith('cp350') else 442)}
                return ev
            with (self.subTest(mutation=mutation),patch('scripts.taco_saved_review.checked_file',side_effect=lambda a:Path(a['path'])),
                  patch.object(review,'strict_load',side_effect=read),patch.object(review,'saved_rows',return_value={777:[]}),
                  patch.object(review,'digest',return_value='manifest')):
                args=('manifest','choice',{'images':[{'image_id':777}]},entry,SimpleNamespace(images=[{'id':777}],data={}),{'contract':contract})
                if mutation:
                    with self.assertRaises(ValueError):review.load_vision_unfreeze_review(*args)
                else:
                    _,models,record=review.load_vision_unfreeze_review(*args)
                    self.assertEqual(set(models),{'pretrained','epoch-01','epoch-02'});self.assertEqual(record['provenance']['new_inference_calls'],0)

    def test_only_allowed_stages_and_optional_layernorm_are_unfrozen(self):
        model=Model();model.named[vu.VISION_ROOT+'layernorm.weight']=torch.nn.Parameter(torch.ones(1),requires_grad=False)
        result=vu.unfreeze_top_vision(model)
        self.assertEqual(result['original_tensors'],558);self.assertEqual(result['vision_tensors'],3)
        self.assertTrue(result['optional_layernorm_present'])
        for n,p in model.named_parameters():
            self.assertEqual(p.requires_grad,n.startswith('neck.') or n.startswith(vu.VISION_PREFIXES))
        with self.assertRaises(ValueError):vu.unfreeze_top_vision(model)

    def test_existing_adam_moments_and_rng_preserved_when_group_added(self):
        model=Model();params=[p for _,p in model.named_parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-4,foreach=False)
        optimizer.state[params[0]]={'step':torch.tensor(221.),'exp_avg':torch.tensor([.3]),'exp_avg_sq':torch.tensor([.05])}
        before=vu.tree_digest(optimizer.state_dict());rng=vu.tree_digest(vu.rng_state())
        result=vu.extend_optimizer(model,optimizer)
        self.assertEqual(result['original_optimizer_sha256'],before)
        self.assertEqual(vu.tree_digest(vu.rng_state()),rng)
        self.assertEqual(len(optimizer.param_groups),2)
        self.assertEqual(optimizer.param_groups[1]['lr'],2e-6)
        self.assertFalse(optimizer.param_groups[1]['foreach'])
        self.assertTrue(all(p not in optimizer.state for p in optimizer.param_groups[1]['params']))

    def test_exact_two_group_pure_film_and_mixed_rules(self):
        optimizer=SimpleNamespace(param_groups=[{},{}])
        for ids,lrs in [([4],(1e-6,5e-7)),([4,4],(1e-6,5e-7)),([4,0],(1e-5,2e-6)),([0],(1e-5,2e-6))]:
            self.assertEqual(vu.apply_batch_lrs(optimizer,[{'category_id':c} for c in ids]),lrs)
            self.assertEqual(tuple(g['lr'] for g in optimizer.param_groups),lrs)
        with self.assertRaises(ValueError):vu.apply_batch_lrs(SimpleNamespace(param_groups=[{}]),[{'category_id':4}])
