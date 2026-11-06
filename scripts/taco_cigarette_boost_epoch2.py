"""D-066 bounded late-epoch cigarette boost; unchanged model, loss and batch order."""
import hashlib
import math

from .train_taco import train_step
from .training_checkpoint import epoch_batches, rng_state, restore_rng, save_checkpoint, verify_checkpoint
from .training_smoke import state_hash, move_batch
from .taco_training_data import get_batch
from .taco_evaluation import evaluate

from .taco_lr_decay_epoch2 import position,tree_digest

RULE = {'type':'late_epoch2_cigarette_boost', 'cigarette_lr':1.5e-5,
        'pure_plastic_film_lr':1e-7,'other_image_lr':5e-6,
        'start_step':350,'first_batch':129,'updates':92,'batch_size':1,
        'resampling':False,'loss_reweighting':False,
        'definition':'cigarette present first; otherwise pure category4; otherwise other'}
SAVE_STEPS=(400,442)
EVALUATE_STEPS=(442,)


from .taco_class_balanced_epoch2 import category_ids,training_inventory


def batch_lr(annotations):
    cats=category_ids(annotations)
    return 1.5e-5 if 0 in cats else 1e-7 if cats==[4] else 5e-6


def late_inventory(train):
    full=training_inventory(train)
    tiers={'cigarette':[],'pure_film':[],'other':[]}
    for indexes in epoch_batches(221,1,42,1)[129:]:
        image_id=train.images[indexes[0]]['id'];cats=category_ids(train.objects[image_id])
        tier='cigarette' if 0 in cats else 'pure_film' if cats==[4] else 'other'
        tiers[tier].append(image_id)
    return {'full_epoch':full,'late_tier_image_ids':tiers,'late_counts':{k:len(v) for k,v in tiers.items()}}


def apply_batch_lr(optimizer,annotations):
    value=batch_lr(annotations)
    if not optimizer.param_groups:raise ValueError('Optimizer has no parameter groups')
    for group in optimizer.param_groups:group['lr']=value
    return value


def training_inventory(train):
    if len(train)!=221 or len({i['id'] for i in train.images})!=221:
        raise ValueError('Require221 unique training images')
    pure=[];other=[];cigarettes=[]
    for image in train.images:
        ids=category_ids(train.objects[image['id']])
        (pure if ids==[4] else other).append(image['id'])
        if 0 in ids:cigarettes.append(image['id'])
    if (len(pure),len(other),len(cigarettes))!=(61,160,30):raise ValueError('Training composition differs from D064')
    return {'pure_plastic_film_image_ids':sorted(pure),'full_lr_image_ids':sorted(other),
            'cigarette_image_ids':sorted(cigarettes),'counts':{'pure_plastic_film':61,'other':160,'cigarette':30}}


def run_epoch2(model,processor,adapter,optimizer,train,validation,contract,folder,
               *, on_stage,on_step,on_checkpoint,on_evaluation):
    """Remaining92 batches only; two checkpoints and one RNG-isolated endpoint validation."""
    import torch
    if len(train)!=221 or len(validation)!=47:raise ValueError('Require221 train /47 validation images')
    batches=epoch_batches(221,1,42,1)
    frozen=state_hash(model,True)
    for k,indexes in enumerate(batches[129:],start=129):
        step=222+k
        on_stage('training',step)
        annotations=train.objects[train.images[indexes[0]]['id']]
        ids=category_ids(annotations)
        lr=apply_batch_lr(optimizer,annotations)
        batch,geometry=get_batch(train,indexes,processor,384,576)
        batch=move_batch(batch,'cuda')
        row=train_step(model,adapter,optimizer,batch)
        del batch
        if any(group['lr']!=lr for group in optimizer.param_groups):raise ValueError('Optimizer LR changed unexpectedly')
        on_step(dict(row,lr=lr,global_step=step,epoch=1,batch_index=k,
                     annotation_category_ids=ids,pure_plastic_film=(ids==[4]),
                     dataset_indexes=indexes,image_ids=[train.images[i]['id'] for i in indexes],geometry=geometry))
        if step in SAVE_STEPS:
            on_stage('checkpoint',step)
            if state_hash(model,True)!=frozen:raise ValueError('Frozen parameters changed before checkpoint')
            name='epoch-02' if step==442 else f'step-{step:06d}'
            cp=folder/'checkpoints'/name
            save_checkpoint(cp,model,processor,adapter,optimizer,contract,position(step),
                {'authorization':'D-066','lr_rule':RULE,'lr':lr,'frozen_parameter_sha256':frozen})
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
