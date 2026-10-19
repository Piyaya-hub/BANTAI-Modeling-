"""Stage 4 trainer/checkpoint qualification on generated images ONLY."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    __package__='scripts'

import gc
import argparse
import os
import random
import subprocess
import time
import uuid

from .configuration import ROOT, strict_load
from .persistence import atomic_json
from .training_smoke import SNAPSHOT, synthetic_batch, move_batch, state_hash, digest, require_training_environment
from .train_taco import load_model, make_optimizer, train_step, memory, source_identity
from .taco_training_data import collate
from .training_checkpoint import save_checkpoint, verify_checkpoint, restore_training, rng_state


def tensor_identity(value):
    """Stable exact state hash; ignores object identity and device placement."""
    import hashlib
    import torch
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str((str(x.dtype),list(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=str):h.update(repr(key).encode());visit(x[key])
        elif isinstance(x,(tuple,list)):
            h.update(str(type(x)).encode())
            for item in x:visit(item)
        else:h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()
from .taco_evaluation import decode_predictions, coco_metrics


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--determinism-probe',action='store_true',
                        help='Separate process: identify unsupported deterministic backward, no updates')
    args=parser.parse_args(argv)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    run=ROOT/'runs/training'/('stage4-synthetic-'+uuid.uuid4().hex)
    run.mkdir(parents=True)
    report={'schema_version':'bantai.stage4-synthetic.v1','status':'running','synthetic_only':True,
            'real_photo_reads':0,'source_identity':source_identity(),
            'qualification_source_sha256':digest(__file__)}
    atomic_json(run/'report.json',report);print(run,flush=True)
    images=[];model=optimizer=adapter=batch=None;started=time.perf_counter()
    source={p.name:digest(p) for p in SNAPSHOT.iterdir() if p.is_file()}
    try:
        report['environment']=require_training_environment()
        if not torch.cuda.is_available():raise RuntimeError('CUDA qualification requires actual GPU')
        report['gpu']={'name':torch.cuda.get_device_name(),'total_bytes':torch.cuda.get_device_properties(0).total_memory}
        report['gpu']['nvidia_smi']=subprocess.run(
            ['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],
            capture_output=True,text=True,check=True).stdout.strip()
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        model,processor,adapter=load_model(SNAPSHOT,'cuda')
        frozen=state_hash(model,True)
        images,targets=synthetic_batch(1)
        batch,geometry=collate(processor,list(zip(images,targets)),256,384)
        batch=move_batch(batch,'cuda')
        optimizer=make_optimizer(model)
        if args.determinism_probe:
            # A failed backward can retain engine allocations until process exit.
            # Never combine this diagnostic with memory/continuation qualification.
            torch.use_deterministic_algorithms(True)
            try:
                probe=adapter(model,batch)
                probe.loss.backward()
                report['strict_determinism_probe']='backward supported'
            except RuntimeError as exc:
                if 'grid_sampler_2d_backward_cuda does not have a deterministic implementation' not in str(exc):raise
                report['strict_determinism_probe']=str(exc)
            report['optimizer_steps']=0
            report['status']='diagnostic_completed'
            return 0
        report['step1']=train_step(model,adapter,optimizer,batch)
        first_hash=state_hash(model)
        context={'synthetic_only':True,'source':source_identity(),'adapter':adapter.manifest(model)}
        position={'epoch':0,'next_batch':1,'global_step':1}
        folder=run/'checkpoint'
        save_checkpoint(folder,model,processor,adapter,optimizer,context,position)
        reference_rng=tensor_identity(rng_state())
        reference_optimizer=tensor_identity(optimizer.state_dict())
        reference_batch=tensor_identity(batch)
        report['uninterrupted_step2']=train_step(model,adapter,optimizer,batch)
        expected_hash=state_hash(model)
        expected={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        # Control: restore tensors into the SAME model object, without HF reconstruction.
        from safetensors.torch import load_model as load_shared_state
        # Native HF serialization omits shared decoder-head aliases; respect them.
        load_shared_state(model,str(folder/'model.safetensors'),strict=True,device='cpu')
        optimizer=make_optimizer(model)
        restore_training(folder,model,processor,adapter,optimizer,context)
        report['same_model_repeat_step2']=train_step(model,adapter,optimizer,batch)
        report['same_model_repeat_max_difference']=max(float((p.detach().cpu()-expected[n]).abs().max())
                                                       for n,p in model.named_parameters() if p.requires_grad)
        model.eval()
        with torch.no_grad():
            direct=model(**{k:v for k,v in batch.items() if k!='labels'})
            adapted=adapter(model,batch)
            report['decoder_equivalence']=bool(torch.equal(direct.logits,adapted.outputs.logits)
                                               and torch.equal(direct.pred_boxes,adapted.outputs.pred_boxes))
            pred,raw=decode_predictions(direct,batch,geometry,processor.tokenizer)
            gt={'info':{},'images':[{'id':1,'width':320,'height':240}],
                'categories':[{'id':i,'name':n} for i,n in enumerate(context['adapter']['labels'])],
                'annotations':targets[0]['annotations']}
            report['synthetic_metrics']=coco_metrics(gt,pred)
            report['prediction_counts']={'COCO':len(pred),'raw_queries':len(raw)}
        del direct,adapted
        report['initial_memory']=memory('cuda')
        model=optimizer=adapter=batch=None;gc.collect();torch.cuda.empty_cache()
        verify_checkpoint(folder,context)
        model,processor,adapter=load_model(folder,'cuda')
        if state_hash(model)!=first_hash:raise ValueError('Saved weights did not reload exactly')
        optimizer=make_optimizer(model)
        batch,replay_geometry=collate(processor,list(zip(images,targets)),256,384)
        batch=move_batch(batch,'cuda')
        restored=restore_training(folder,model,processor,adapter,optimizer,context)
        if restored!=position or replay_geometry!=geometry:raise ValueError('Resume position/processor frame differs')
        report['restored_rng_exact']=tensor_identity(rng_state())==reference_rng
        report['restored_optimizer_exact']=tensor_identity(optimizer.state_dict())==reference_optimizer
        report['restored_batch_exact']=tensor_identity(batch)==reference_batch
        report['resumed_step2']=train_step(model,adapter,optimizer,batch)
        report['resume_loss_difference']=abs(report['resumed_step2']['loss']-report['uninterrupted_step2']['loss'])
        report['resume_parameter_max_difference']=max(float((p.detach().cpu()-expected[n]).abs().max())
                                                       for n,p in model.named_parameters() if p.requires_grad)
        report['resume_weights_bitwise_equal']=state_hash(model)==expected_hash
        report['resume_tolerance']={'absolute':2e-7,'relative':1e-6,
            'reason':'FP32 CUDA backward numerical continuation; exact restored states checked independently'}
        report['resume_weights_close']=all(torch.allclose(p.detach().cpu(),expected[n],atol=2e-7,rtol=1e-6)
                                          for n,p in model.named_parameters() if p.requires_grad)
        report['frozen_unchanged']=state_hash(model,True)==frozen
        report['memory']=memory('cuda')
        if (report['resume_loss_difference']>1e-5 or not report['resume_weights_close']
                or not all(report[k] for k in ('restored_rng_exact','restored_optimizer_exact','restored_batch_exact',
                                               'frozen_unchanged','decoder_equivalence'))):
            raise ValueError('Stage 4 checkpoint/decoder qualification failed; inspect report')
        report['status']='passed'
        return 0
    except Exception as exc:
        import traceback
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        print(report['error'],flush=True);return 1
    finally:
        report['base_snapshot_unchanged']=all(digest(SNAPSHOT/n)==v for n,v in source.items())
        if not report['base_snapshot_unchanged']:report['status']='failed_preservation'
        report['elapsed_seconds']=time.perf_counter()-started
        atomic_json(run/'report.json',report,replace=True)
        for image in images:image.close()
        torch.use_deterministic_algorithms(False)
        model=optimizer=adapter=batch=None;gc.collect();torch.cuda.empty_cache()
        print(report['status'],flush=True)


if __name__=='__main__':raise SystemExit(main())
