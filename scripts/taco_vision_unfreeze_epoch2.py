"""D-065 image-composition LR damping; unchanged model, loss and batch order."""
import hashlib
import math

from .train_taco import train_step
from .training_checkpoint import epoch_batches, rng_state, restore_rng, save_checkpoint, verify_checkpoint
from .training_smoke import state_hash, move_batch
from .taco_training_data import get_batch
from .taco_evaluation import evaluate

from .taco_lr_decay_epoch2 import position,tree_digest

from .taco_class_balanced_epoch2 import category_ids,training_inventory
from .taco_lr_decay_epoch2 import rng_state,tree_digest

VISION_ROOT='model.backbone.conv_encoder.model.'
VISION_PREFIXES=(VISION_ROOT+'encoder.layers.2.',VISION_ROOT+'encoder.layers.3.',VISION_ROOT+'layernorm.')
VISION_PROBES=(VISION_ROOT+'encoder.layers.2.blocks.0.attention.self.query.weight',
               VISION_ROOT+'encoder.layers.3.blocks.0.attention.self.query.weight')
RULE={'type':'top_vision_unfreeze_plus_pure_film_damping','pure_category_id':4,
      'original_full_lr':1e-5,'original_pure_film_lr':1e-6,
      'vision_full_lr':2e-6,'vision_pure_film_lr':5e-7,
      'expected_pure_images':61,'expected_other_images':160,'expected_cigarette_images':30,
      'vision_prefixes':list(VISION_PREFIXES),'weight_decay':1e-4,'foreach':False,'clip_norm':1.0}
SAVE_STEPS=(250,300,350,400,442)
EVALUATE_STEPS=(350,442)


def batch_lrs(annotations):
    return (1e-6,5e-7) if category_ids(annotations)==[4] else (1e-5,2e-6)


def apply_batch_lrs(optimizer,annotations):
    if len(optimizer.param_groups)!=2:raise ValueError('Require original and vision groups')
    values=batch_lrs(annotations)
    for group,lr in zip(optimizer.param_groups,values):group['lr']=lr
    return values


def unfreeze_top_vision(model):
    named=dict(model.named_parameters());original=[n for n,p in named.items() if p.requires_grad]
    if len(original)!=558 or any(n.startswith(('model.backbone.','model.text_backbone.')) for n in original):
        raise ValueError('Require exact original558 trainable tensors before unfreezing')
    new=[n for n in named if n.startswith(VISION_PREFIXES)]
    if any(not any(n.startswith(prefix) for n in new) for prefix in VISION_PREFIXES[:2]):
        raise ValueError('Both allowed top vision stages must be present')
    if any(named[n].requires_grad for n in new):raise ValueError('Vision stage already trainable')
    for n in new:named[n].requires_grad_(True)
    expected=set(original)|set(new)
    if {n for n,p in named.items() if p.requires_grad}!=expected:raise ValueError('Unexpected trainable parameter set')
    return {'original_names':original,'vision_names':new,'original_tensors':558,'vision_tensors':len(new),
            'vision_elements':sum(named[n].numel() for n in new),
            'optional_layernorm_present':any(n.startswith(VISION_PREFIXES[2]) for n in new)}


def extend_optimizer(model,optimizer):
    before=optimizer.state_dict();before_hash=tree_digest(before);rng_before=tree_digest(rng_state())
    metadata=unfreeze_top_vision(model);named=dict(model.named_parameters())
    if len(optimizer.param_groups)!=1 or [id(p) for p in optimizer.param_groups[0]['params']]!=[id(named[n]) for n in metadata['original_names']]:
        raise ValueError('Original optimizer group/order differs from restored model')
    optimizer.add_param_group({'params':[named[n] for n in metadata['vision_names']],
                              'lr':2e-6,'weight_decay':1e-4,'foreach':False})
    after=optimizer.state_dict();ids=set(before['param_groups'][0]['params'])
    subset={'state':{k:v for k,v in after['state'].items() if k in ids},'param_groups':after['param_groups'][:1]}
    if tree_digest(subset)!=before_hash or set(after['state'])!=set(before['state']):
        raise ValueError('Original optimizer state changed while adding vision group')
    if tree_digest(rng_state())!=rng_before:raise ValueError('RNG changed while adding vision group')
    return dict(metadata,original_optimizer_sha256=before_hash,original_optimizer_preserved=True,RNG_preserved=True)


def audited_train_step(model,adapter,optimizer,batch):
    import torch
    named=dict(model.named_parameters());seen={};handles=[]
    snapshots={n:named[n].detach().clone() for n in VISION_PROBES}
    def capture(name):
        def hook(gradient):
            value=float(gradient.norm())
            if not torch.isfinite(gradient).all() or not value>0:raise ValueError('Missing/nonfinite/zero vision gradient: '+name)
            seen[name]=value
        return hook
    try:
        for n in VISION_PROBES:handles.append(named[n].register_hook(capture(n)))
        row=train_step(model,adapter,optimizer,batch)
        updates={n:float((named[n].detach()-snapshots[n]).abs().max()) for n in VISION_PROBES}
        if set(seen)!=set(VISION_PROBES) or any(v<=0 for v in updates.values()):raise ValueError('Vision probe gradient/update missing')
        return dict(row,vision_gradient_norms=seen,vision_update_max_abs=updates)
    finally:
        for handle in handles:handle.remove()


def run_epoch2(model,processor,adapter,optimizer,train,validation,contract,folder,
               *, on_stage,on_step,on_checkpoint,on_evaluation):
    """One fixed epoch, five complete checkpoints, two RNG-isolated evaluations."""
    import torch
    if len(train)!=221 or len(validation)!=47:raise ValueError('Require221 train /47 validation images')
    batches=epoch_batches(221,1,42,1)
    frozen=state_hash(model,True)
    for k,indexes in enumerate(batches):
        step=222+k
        on_stage('training',step)
        annotations=train.objects[train.images[indexes[0]]['id']]
        ids=category_ids(annotations)
        lr,vision_lr=apply_batch_lrs(optimizer,annotations)
        batch,geometry=get_batch(train,indexes,processor,384,576)
        batch=move_batch(batch,'cuda')
        row=audited_train_step(model,adapter,optimizer,batch)
        del batch
        if tuple(group['lr'] for group in optimizer.param_groups)!=(lr,vision_lr):raise ValueError('Optimizer LR changed unexpectedly')
        on_step(dict(row,lr=lr,vision_lr=vision_lr,global_step=step,epoch=1,batch_index=k,
                     annotation_category_ids=ids,pure_plastic_film=(ids==[4]),
                     dataset_indexes=indexes,image_ids=[train.images[i]['id'] for i in indexes],geometry=geometry))
        if step in SAVE_STEPS:
            on_stage('checkpoint',step)
            if state_hash(model,True)!=frozen:raise ValueError('Frozen parameters changed before checkpoint')
            name='epoch-02' if step==442 else f'step-{step:06d}'
            cp=folder/'checkpoints'/name
            save_checkpoint(cp,model,processor,adapter,optimizer,contract,position(step),
                {'authorization':'D-065','lr_rule':RULE,'lr':lr,'frozen_parameter_sha256':frozen})
            proof=verify_checkpoint(cp,contract)
            if proof['position']!=position(step) or state_hash(model,True)!=frozen:
                raise ValueError('Checkpoint position/frozen parameters changed')
            on_checkpoint(cp,proof)
        if step in EVALUATE_STEPS:
            on_stage('validation',step)
            snapshot=rng_state();rng_before=tree_digest(snapshot);weights=state_hash(model)
            try:
                with torch.inference_mode():value=evaluate(model,processor,adapter,validation,'cuda',384,576)
            finally:restore_rng(snapshot)
            if (tree_digest(rng_state())!=rng_before or state_hash(model)!=weights
                    or state_hash(model,True)!=frozen):
                raise ValueError('Validation changed RNG or parameters')
            on_evaluation(step,value,{'RNG_restored_exactly':True,'RNG_sha256':rng_before,
                                      'parameters_unchanged':True,'frozen_parameter_sha256':frozen})
    on_stage('completed',442)
