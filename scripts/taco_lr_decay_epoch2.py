"""D-062 bounded Epoch-2 continuation using the unchanged Stage-4 core."""
import hashlib
import math

from .train_taco import train_step
from .training_checkpoint import epoch_batches, rng_state, restore_rng, save_checkpoint, verify_checkpoint
from .training_smoke import state_hash, move_batch
from .taco_training_data import get_batch
from .taco_evaluation import evaluate

SCHEDULE = {'type':'cosine_per_optimizer_step','lr_max':5e-6,'lr_min':1e-6,
            'first_global_step':222,'last_global_step':442,'updates':221,
            'formula':'1e-6 + 0.5 * (5e-6 - 1e-6) * (1 + cos(pi * k / 220))',
            'k_definition':'global_step - 222, integer 0..220'}
SAVE_STEPS = (250,300,350,400,442)
EVALUATE_STEPS = (350,442)


def cosine_lr(k):
    if type(k) is not int or not 0 <= k <= 220:
        raise ValueError('Cosine step must be an integer in 0..220')
    return 1e-6 + .5 * (5e-6 - 1e-6) * (1 + math.cos(math.pi*k/220))


def apply_lr(optimizer, k):
    value=cosine_lr(k)
    if not optimizer.param_groups:raise ValueError('Optimizer has no parameter groups')
    for group in optimizer.param_groups:group['lr']=value
    return value


def position(global_step):
    if type(global_step) is not int or not 222 <= global_step <= 442:
        raise ValueError('Outside the authorized Epoch-2 continuation')
    return {'epoch':2,'next_batch':0,'global_step':442} if global_step==442 else {
        'epoch':1,'next_batch':global_step-221,'global_step':global_step}


def tree_digest(value):
    """Exact tensor/primitive identity, independent of tensor device/dict order."""
    import torch
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            t=v.detach().cpu().contiguous()
            h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
        elif isinstance(v,dict):
            h.update(b'dict')
            for key in sorted(v,key=lambda k:(type(k).__name__,repr(k))):visit(key);visit(v[key])
        elif isinstance(v,(tuple,list)):
            h.update(type(v).__name__.encode());h.update(str(len(v)).encode())
            for item in v:visit(item)
        else:h.update((type(v).__name__+':'+repr(v)).encode())
    visit(value)
    return h.hexdigest()


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
        lr=apply_lr(optimizer,k)
        batch,geometry=get_batch(train,indexes,processor,384,576)
        batch=move_batch(batch,'cuda')
        row=train_step(model,adapter,optimizer,batch)
        del batch
        if any(group['lr']!=lr for group in optimizer.param_groups):raise ValueError('Optimizer LR changed unexpectedly')
        on_step(dict(row,lr=lr,global_step=step,epoch=1,batch_index=k,
                     dataset_indexes=indexes,image_ids=[train.images[i]['id'] for i in indexes],geometry=geometry))
        if step in SAVE_STEPS:
            on_stage('checkpoint',step)
            if state_hash(model,True)!=frozen:raise ValueError('Frozen parameters changed before checkpoint')
            name='epoch-02' if step==442 else f'step-{step:06d}'
            cp=folder/'checkpoints'/name
            save_checkpoint(cp,model,processor,adapter,optimizer,contract,position(step),
                {'authorization':'D-062','schedule':SCHEDULE,'lr':lr,'frozen_parameter_sha256':frozen})
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
