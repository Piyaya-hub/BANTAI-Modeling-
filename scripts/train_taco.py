"""Stage 4 real-image pilot, bounded training and fixed validation. Rights gated."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'scripts'

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
import uuid

from .configuration import ROOT, MODEL_ID, REVISION, strict_load
from .persistence import atomic_json
from .training_smoke import (SNAPSHOT, digest, require_training_environment, move_batch,
                             checked_loss, loading_ok, state_hash, PROMPT, LABELS)
from .training_adapter import EncoderOutputAdapter, adapter_source_digest
from .training_checkpoint import save_checkpoint, verify_checkpoint, restore_training, epoch_batches
from .taco_training_data import (dataset_context, require_rights, RightsRequired,
                                CocoTrainingDataset, get_batch)
from .taco_evaluation import PROTOCOL, evaluate

SOURCES = ('scripts/train_taco.py','scripts/taco_training_data.py','scripts/taco_evaluation.py',
           'scripts/training_checkpoint.py','scripts/training_adapter.py','scripts/training_smoke.py')


def source_identity():
    return {name:adapter_source_digest(ROOT/name) for name in SOURCES}


def freeze_backbones(model):
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith(('model.backbone.', 'model.text_backbone.')))


def load_model(path, device):
    from transformers import GroundingDinoForObjectDetection, AutoProcessor
    model, info = GroundingDinoForObjectDetection.from_pretrained(path,local_files_only=True,
        trust_remote_code=False,use_safetensors=True,disable_custom_kernels=True,output_loading_info=True)
    loading_ok(info)
    processor = AutoProcessor.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    freeze_backbones(model)
    return model.to(device), processor, EncoderOutputAdapter(model,processor.tokenizer)


def make_optimizer(model):
    import torch
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-5,weight_decay=1e-4,foreach=False)


def memory(device):
    import torch
    if device != 'cuda':
        return {'device':'cpu','peak_allocated_bytes':None,'peak_reserved_bytes':None}
    torch.cuda.synchronize()
    return {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'current_allocated_bytes':torch.cuda.memory_allocated()}


def train_step(model, adapter, optimizer, batch):
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    result = adapter(model,batch)  # Never invoke model(labels=...).
    components = checked_loss(result,model.config)
    result.loss.backward()
    probes = ('model.enc_output.weight','model.encoder_output_bbox_embed.layers.2.weight')
    named = dict(model.named_parameters())
    gradients={}
    for name,p in named.items():
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
            raise ValueError('Missing/nonfinite training gradient: '+name)
    before={name:named[name].detach().clone() for name in probes}
    for name in probes:
        gradients[name] = float(named[name].grad.norm())
        if gradients[name] <= 0:
            raise ValueError('Expected nonempty encoder gradient: '+name)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step()
    updates={name:float((named[name].detach()-before[name]).abs().max()) for name in probes}
    if not all(value>0 for value in updates.values()):
        raise ValueError('Encoder optimizer update missing')
    if any(not torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
        raise ValueError('Nonfinite updated parameter')
    loss=float(result.loss.detach())
    del result
    return {'loss':loss,'components':components,'gradient_norm_before_clip':float(norm),
            'encoder_gradient_norms':gradients,'encoder_update_max_abs':updates,
            'seconds_including_gradient_audit':time.perf_counter()-start}


def preflight(dataset, processor, shortest, longest):
    from tqdm import tqdm
    evidence=[]
    for i in tqdm(range(len(dataset)),desc='actual-photo coordinate preflight',unit='image'):
        _, geometry = get_batch(dataset,[i],processor,shortest,longest)
        evidence.extend(geometry)
    return evidence


def append_step(path, row):
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')
        stream.flush()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'exports/taco/strict-clean-stage2-v1')
    parser.add_argument('--rights',type=Path)
    parser.add_argument('--write-context',type=Path,help='Write metadata-only rights binding; no photos/model opened')
    parser.add_argument('--resume',type=Path,help='Complete Stage 4 checkpoint folder; continuation creates a new run')
    parser.add_argument('--epochs',type=int,choices=(2,3),default=2)
    parser.add_argument('--batch-size',type=int,choices=(1,2),default=1)
    parser.add_argument('--shortest-edge',type=int,default=256)
    parser.add_argument('--longest-edge',type=int,default=384)
    parser.add_argument('--pilot-steps',type=int,choices=range(2,6),default=3)
    parser.add_argument('--checkpoint-every',type=int,default=50)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args(argv)
    if not 192<=args.shortest_edge<=args.longest_edge<=512 or args.checkpoint_every<1:
        parser.error('Require 192 <= shortest <= longest <= 512, positive checkpoint interval')
    context=dataset_context(args.dataset)
    if args.write_context:
        atomic_json(args.write_context,context)
        print(args.write_context); return 0
    run=ROOT/'runs/training'/('taco-'+uuid.uuid4().hex)
    run.mkdir(parents=True)
    report={'schema_version':'bantai.taco-training.v1','run_id':run.name,'status':'running',
            'stage':'rights_gate','dataset':context,'real_photo_reads':0,'optimizer_steps':0,
            'settings':{k:v for k,v in vars(args).items() if k not in ('dataset','rights','resume','write_context')},
            'protocol':PROTOCOL,'epochs':[],'implementation_sha256':source_identity()}
    report_path=run/'report.json'
    atomic_json(report_path,report); print(report_path,flush=True)
    started=time.perf_counter(); model=optimizer=adapter=train=validation=None; source={}; torch=None
    def publish(stage):
        report['stage']=stage;atomic_json(report_path,report,replace=True);print(stage,flush=True)
    try:
        rights=require_rights(args.rights,context)
        report['rights']=rights
        report['environment']=require_training_environment()
        if importlib.metadata.version('pycocotools')!='2.0.10':
            raise ValueError('Stage 4 requires isolated pycocotools==2.0.10')
        # Configure before initializing CUDA; do not silently enable AMP/TF32.
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import numpy as np
        import torch as torch_module
        torch=torch_module
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        if args.device=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
            torch.cuda.manual_seed_all(42);torch.cuda.reset_peak_memory_stats()
            report['gpu']={'name':torch.cuda.get_device_name(),'total_bytes':torch.cuda.get_device_properties(0).total_memory,
                           'capability':list(torch.cuda.get_device_capability()),'torch_cuda':torch.version.cuda}
            smi=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],
                               capture_output=True,text=True,check=True)
            report['gpu']['nvidia_smi']=smi.stdout.strip()
        source={p.name:digest(p) for p in SNAPSHOT.iterdir() if p.is_file()}
        # Bind native library, all code, dataset, runtime, rights and fixed hyperparameters.
        contract={'dataset':context,'rights_sha256':rights['record_sha256'],'settings':report['settings'],
                  'implementation_sha256':report['implementation_sha256'],'base_snapshot_sha256':source,
                  'model_id':MODEL_ID,'revision':REVISION,'prompt':PROMPT,'labels':list(LABELS),
                  'protocol':PROTOCOL,'packages':report['environment']['packages'],
                  'runtime':{key:report['environment'][key] for key in ('python','platform')},
                  'gpu':report.get('gpu'),
                  'optimizer':{'name':'AdamW','lr':1e-5,'weight_decay':1e-4,'foreach':False,'clip_norm':1.},
                  'precision':'fp32','frozen':'vision_and_text_backbones','augmentation':'none',
                  'scheduler':'none','seed':42,'workers':0,'padding':'fixed_longest_edge_square'}
        if args.resume:
            candidate=dict(contract,adapter_context=strict_load(args.resume/'training_adapter.json'))
            verify_checkpoint(args.resume,candidate)
        publish('load_model')
        model,processor,adapter=load_model(args.resume or SNAPSHOT,args.device)
        contract['adapter_context']=adapter.manifest(model)
        # Include adapter context on both verification passes.
        if args.resume:
            verify_checkpoint(args.resume,contract)
        report['contract']=contract
        frozen=state_hash(model,True)
        train=CocoTrainingDataset(args.dataset,'train',args.rights)
        validation=CocoTrainingDataset(args.dataset,'validation',args.rights)
        publish('coordinate_preflight_before_any_optimizer_updates')
        report['geometry']={'train':preflight(train,processor,args.shortest_edge,args.longest_edge),
                            'validation':preflight(validation,processor,args.shortest_edge,args.longest_edge)}
        report['real_photo_reads']=len(train)+len(validation)
        if args.resume:
            evidence=strict_load(args.resume/'experiment_evidence.json')
            if evidence['baseline']['base_snapshot_sha256']!=source or evidence['pilot']['status']!='passed':
                raise ValueError('Missing qualified pilot or unchanged baseline evidence')
            optimizer=make_optimizer(model)
            position=restore_training(args.resume,model,processor,adapter,optimizer,contract)
            total_batches=(len(train)+args.batch_size-1)//args.batch_size
            if (not 0<=position['epoch']<args.epochs or not 0<=position['next_batch']<=total_batches
                    or position['global_step']!=position['epoch']*total_batches+position['next_batch']):
                raise ValueError('Resume sampler position invalid or experiment already complete')
            report['resumed_from']=str(args.resume.resolve())
        else:
            publish('unchanged_pretrained_validation')
            baseline=evaluate(model,processor,adapter,validation,args.device,args.shortest_edge,args.longest_edge)
            baseline['base_snapshot_sha256']=source
            atomic_json(run/'baseline.json',baseline)
            publish('real_image_memory_pilot')
            optimizer=make_optimizer(model)
            if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
            # Prefer object-dense training photos for this bounded wiring/memory pilot.
            order=sorted(range(len(train)),key=lambda i:(-len(train.objects[train.images[i]['id']]),train.images[i]['id']))
            pilot={'status':'running','steps':[],'selection':'most_annotations_then_image_id'}
            for step in range(args.pilot_steps):
                indexes=[order[(step*args.batch_size+j)%len(order)] for j in range(args.batch_size)]
                batch,geometry=get_batch(train,indexes,processor,args.shortest_edge,args.longest_edge)
                row=train_step(model,adapter,optimizer,move_batch(batch,args.device))
                row.update({'image_ids':[g['image_id'] for g in geometry],'memory':memory(args.device)})
                pilot['steps'].append(row);report['optimizer_steps']+=1
                atomic_json(run/'pilot.json',pilot,replace=(run/'pilot.json').exists())
                print(f'pilot {step+1}/{args.pilot_steps} loss={row["loss"]:.6f} memory={row["memory"]}',flush=True)
            if state_hash(model,True)!=frozen:raise ValueError('Frozen parameters changed in pilot')
            pilot.update(status='passed',memory=memory(args.device),checkpoint_not_promoted=True)
            atomic_json(run/'pilot.json',pilot,replace=True)
            evidence={'baseline':baseline,'pilot':pilot}
            # Discard pilot optimizer AND weights. Full experiment starts from pinned base.
            del batch
            model=optimizer=adapter=None;gc.collect()
            if args.device=='cuda':torch.cuda.empty_cache()
            model,processor,adapter=load_model(SNAPSHOT,args.device)
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            if args.device=='cuda':torch.cuda.manual_seed_all(42)
            optimizer=make_optimizer(model)
            position={'epoch':0,'next_batch':0,'global_step':0}
        atomic_json(run/'baseline.json',evidence['baseline'],replace=(run/'baseline.json').exists())
        atomic_json(run/'pilot.json',evidence['pilot'],replace=(run/'pilot.json').exists())
        if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
        from tqdm import tqdm
        publish('bounded_training')
        for epoch in range(position['epoch'],args.epochs):
            batches=epoch_batches(len(train),args.batch_size,42,epoch)
            start_batch=position['next_batch'] if epoch==position['epoch'] else 0
            if not 0<=start_batch<=len(batches):raise ValueError('Invalid resume sampler position')
            rows=[]
            progress=tqdm(range(start_batch,len(batches)),desc=f'epoch {epoch+1}/{args.epochs}',initial=start_batch,total=len(batches))
            for b in progress:
                batch,geometry=get_batch(train,batches[b],processor,args.shortest_edge,args.longest_edge)
                row=train_step(model,adapter,optimizer,move_batch(batch,args.device))
                position={'epoch':epoch,'next_batch':b+1,'global_step':position['global_step']+1}
                row.update(position,image_ids=[g['image_id'] for g in geometry],memory=memory(args.device))
                report['optimizer_steps']+=1;rows.append(row['loss'])
                append_step(run/'steps.jsonl',row)
                progress.set_postfix(loss=f'{row["loss"]:.4f}',gpu_MiB=round((row['memory'].get('peak_allocated_bytes') or 0)/2**20))
                if position['global_step']%args.checkpoint_every==0:
                    folder=run/'checkpoints'/f'step-{position["global_step"]:06d}'
                    save_checkpoint(folder,model,processor,adapter,optimizer,contract,position,evidence)
                    atomic_json(run/'latest_checkpoint.json',{'path':str(folder),'position':position},replace=(run/'latest_checkpoint.json').exists())
            publish(f'validation_epoch_{epoch+1}')
            evaluation=evaluate(model,processor,adapter,validation,args.device,args.shortest_edge,args.longest_edge)
            atomic_json(run/f'validation-epoch-{epoch+1}.json',evaluation)
            summary={'epoch':epoch+1,'mean_loss_observed_batches':sum(rows)/len(rows) if rows else None,
                     'observed_batches':len(rows),'validation':evaluation['metrics'],
                     'validation_loss':evaluation['mean_loss_per_image'],'memory':memory(args.device)}
            report['epochs'].append(summary);print(json.dumps(summary,allow_nan=False),flush=True)
            if state_hash(model,True)!=frozen:raise ValueError('Frozen parameters changed')
            position={'epoch':epoch+1,'next_batch':0,'global_step':position['global_step']}
            folder=run/'checkpoints'/f'epoch-{epoch+1:02d}'
            save_checkpoint(folder,model,processor,adapter,optimizer,contract,position,evidence)
            atomic_json(run/'latest_checkpoint.json',{'path':str(folder),'position':position},replace=(run/'latest_checkpoint.json').exists())
        report['status']='completed'
        report['comparison']={'baseline':evidence['baseline']['metrics'],
                              'final':report['epochs'][-1]['validation'] if report['epochs'] else 'resume already complete'}
        report['memory']=memory(args.device)
        report['acceptance_gaps']=['human source/box and representative error review pending',
                                   'rare-class estimates unstable; no test evaluation or checkpoint promotion']
        return 0
    except RightsRequired as exc:
        report.update(status='blocked_rights',error=str(exc));print(str(exc),flush=True);return 2
    except (Exception,KeyboardInterrupt) as exc:
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                      error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        print(report['error'],flush=True);return 130 if isinstance(exc,KeyboardInterrupt) else 1
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        report['real_photo_reads']=sum(d.photo_reads for d in (train,validation) if d is not None)
        if source:
            report['base_snapshot_unchanged']=all(digest(SNAPSHOT/name)==value for name,value in source.items())
            if not report['base_snapshot_unchanged']:report['status']='failed_preservation'
        atomic_json(report_path,report,replace=True)
        model=optimizer=adapter=None;gc.collect()
        if torch is not None and args.device=='cuda' and torch.cuda.is_initialized():torch.cuda.empty_cache()
        if report.get('base_snapshot_unchanged') is False:
            raise RuntimeError('Pinned pretrained snapshot changed; see failed preservation report')


if __name__=='__main__':
    raise SystemExit(main())
