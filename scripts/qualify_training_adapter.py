"""Stage 3D synthetic-only corrected-objective qualification; no real-data loader."""
import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time
import traceback
import uuid

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts.configuration import ROOT,MODEL_ID,REVISION,strict_load
from scripts.persistence import atomic_json
from scripts.training_adapter import EncoderOutputAdapter,config_signature
from scripts.training_smoke import (SNAPSHOT,require_training_environment,digest,synthetic_batch,prepare_batch,
    move_batch,loading_ok,validate_predictions,validate_encoder_capacity,state_hash,checked_loss)
from scripts.loss_investigation import CASES,targets_for_case,gradient_paths


def architecture_hash(model):
    import hashlib
    data=[(name,list(tensor.shape),str(tensor.dtype)) for name,tensor in model.state_dict().items()]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def compare_predictions(left,right):
    import torch
    if not torch.equal(left.logits,right.logits) or not torch.equal(left.pred_boxes,right.pred_boxes):
        raise ValueError('Decoder outputs changed at identical weights')
    active=torch.isfinite(left.logits)
    return {'exact_equal':True,'max_active_logit_difference':float((left.logits[active]-right.logits[active]).abs().max()),
            'max_box_difference':float((left.pred_boxes-right.pred_boxes).abs().max())}


def eval_equivalence(model,adapter,batch):
    import torch
    model.eval()
    with torch.no_grad():
        direct=model(**{k:v for k,v in batch.items() if k!='labels'})
        adapted=adapter(model,batch)
        repeated=adapter(model,batch)
    validate_predictions(direct.logits,direct.pred_boxes,batch['attention_mask'])
    result=compare_predictions(direct,adapted.outputs)
    compare_predictions(adapted.outputs,repeated.outputs)
    if not torch.equal(adapted.loss,repeated.loss): raise ValueError('Repeated adapter loss differs in eval')
    result['repeated_loss_equal']=True
    result['loss']=float(adapted.loss)
    return result


def checkpoint_roundtrip(model,adapter,processor,batch,images,targets,run,device):
    import torch
    from transformers import GroundingDinoForObjectDetection,AutoProcessor
    model.eval()
    with torch.no_grad(): reference=adapter(model,batch)
    weights=state_hash(model);architecture=architecture_hash(model)
    folder=run/'checkpoint'
    model.save_pretrained(folder,safe_serialization=True)
    processor.save_pretrained(folder)
    manifest=adapter.manifest(model)
    atomic_json(folder/'training_adapter.json',manifest)
    atomic_json(run/'training_context.json',{'synthetic_only':True,'base_revision':REVISION,
        'resume_supported':False,'optimizer_scheduler_scaler_rng_saved':False,
        'preprocessing':{'shortest_edge':192,'longest_edge':256},'prompt':manifest['prompt'],
        'seed':42,'purpose':'Stage 3D qualification; never automatically promoted'})
    restored,info=GroundingDinoForObjectDetection.from_pretrained(folder,local_files_only=True,
        trust_remote_code=False,use_safetensors=True,output_loading_info=True)
    loading_ok(info)
    restored_processor=AutoProcessor.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    restored_adapter=EncoderOutputAdapter(restored,restored_processor.tokenizer)
    restored_adapter.verify_manifest(restored,strict_load(folder/'training_adapter.json'))
    if state_hash(restored)!=weights or architecture_hash(restored)!=architecture:
        raise ValueError('Checkpoint architecture/weight values changed')
    replay_batch,replay_details=prepare_batch(restored_processor,images,targets,192,256)
    for key,tensor in replay_batch.items():
        if key!='labels' and not torch.equal(tensor,batch[key].cpu()): raise ValueError('Processor replay differs')
    for expected,actual in zip(batch['labels'],replay_batch['labels']):
        if any(not torch.equal(value.cpu(),actual[key]) for key,value in expected.items()):
            raise ValueError('Reloaded processor target tensors differ')
    replay_batch=move_batch(replay_batch,device)
    restored.to(device).eval()
    with torch.no_grad(): replay=restored_adapter(restored,replay_batch)
    predictions=compare_predictions(reference.outputs,replay.outputs)
    if not torch.equal(reference.loss,replay.loss): raise ValueError('Reloaded adapter loss differs')
    if any(not torch.equal(value,replay.loss_dict[key]) for key,value in reference.loss_dict.items()):
        raise ValueError('Reloaded loss components differ')
    tampered=dict(manifest,adapter_version='unsupported')
    try: restored_adapter.verify_manifest(restored,tampered)
    except ValueError: pass
    else: raise ValueError('Tampered adapter context accepted')
    return {'loading':info,'weight_hash_equal':True,'architecture_hash_equal':True,'processor_equal':True,
        'adapter_manifest_equal':True,'loss_and_components_exact_equal':True,'tampered_context_rejected':True,
        'decoder':predictions,'sha256':{p.name:digest(p) for p in folder.iterdir() if p.is_file()},
        'resume_supported':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--steps',type=int,choices=(1,2,3),default=2)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    args=parser.parse_args(argv)
    if len(set(args.cases))!=len(args.cases) or args.cases[0]!='baseline_11':
        parser.error('Start with baseline_11 and do not repeat cases')
    run=ROOT/'runs/training'/('adapter-'+uuid.uuid4().hex)
    run.mkdir(parents=True,exist_ok=False)
    path=run/'report.json'
    report={'schema_version':'bantai.adapter-qualification.v1','run_id':run.name,'status':'running',
        'assessment':'PENDING','stage':'environment','synthetic_only':True,'real_data_training_ready':False,
        'model_id':MODEL_ID,'revision':REVISION,'settings':{'device':args.device,'steps_per_case':args.steps,
        'cases':args.cases,'seed':42,'precision':'fp32','batch_size':2,'shortest_edge':192,'longest_edge':256,
        'trainable':'fusion-decoder','optimizer':'AdamW','lr':1e-5,'weight_decay':1e-4,'clip_norm':1.},'cases':[]}
    atomic_json(path,report);print(path,flush=True)
    torch=None;images=[];source={};library={}

    def publish(stage):
        report['stage']=stage;atomic_json(path,report,replace=True);print(stage,flush=True)

    def memory():
        if torch is not None and args.device=='cuda' and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            return {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        return {'cuda_memory':'not_applicable'}

    started=time.perf_counter()
    try:
        report['environment']=require_training_environment()
        import torch as torch_module
        torch=torch_module
        from transformers import GroundingDinoForObjectDetection,AutoProcessor
        import transformers
        package=Path(transformers.__file__).parent
        library={p:digest(package/p) for p in ('models/grounding_dino/modeling_grounding_dino.py',
            'loss/loss_grounding_dino.py','loss/loss_for_object_detection.py')}
        source={p.name:digest(p) for p in SNAPSHOT.iterdir() if p.is_file()}
        report['base_snapshot_sha256']=source;report['library_sha256']=library
        report['implementation_sha256']={p:digest(ROOT/p) for p in ('scripts/training_adapter.py','scripts/qualify_training_adapter.py')}
        if args.device=='cuda':
            if not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; no fallback')
            report['gpu']={'name':torch.cuda.get_device_name(),'total_bytes':torch.cuda.get_device_properties(0).total_memory,
                           'capability':list(torch.cuda.get_device_capability()),'torch_cuda':torch.version.cuda}
        torch.set_num_threads(4);random.seed(42);torch.manual_seed(42)
        publish('load')
        model,info=GroundingDinoForObjectDetection.from_pretrained(SNAPSHOT,local_files_only=True,
            trust_remote_code=False,use_safetensors=True,disable_custom_kernels=True,output_loading_info=True)
        loading_ok(info);config_signature(model.config)
        processor=AutoProcessor.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=False)
        for name,p in model.named_parameters():
            if name.startswith(('model.backbone.','model.text_backbone.')):p.requires_grad_(False)
        baseline={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        frozen=state_hash(model,True);architecture=architecture_hash(model)
        adapter=EncoderOutputAdapter(model,processor.tokenizer)
        report['adapter_manifest']=adapter.manifest(model)
        report['parameters']={'total':sum(p.numel() for p in model.parameters()),
                              'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}
        model.to(args.device)
        images,original=synthetic_batch(2)
        params=dict(model.named_parameters())
        names=['model.enc_output.weight','model.enc_output_norm.weight','model.encoder_output_bbox_embed.layers.2.weight',
               'model.query_position_embeddings.weight','model.text_projection.weight','model.decoder.bbox_embed.0.layers.0.weight']
        probes={n:params[n] for n in names}
        for case in args.cases:
            publish(case)
            model.load_state_dict(baseline,strict=True);model.zero_grad(set_to_none=True)
            gc.collect()
            if args.device=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.manual_seed_all(42)
            torch.manual_seed(42)
            targets=targets_for_case(original,case)
            batch,geometry=prepare_batch(processor,images,targets,192,256)
            validate_encoder_capacity(*batch['pixel_values'].shape[-2:])
            batch=move_batch(batch,args.device)
            case_report={'name':case,'target_counts':[len(t['annotations']) for t in targets],
                         'preprocessing':geometry,'steps':[]}
            weights_before=state_hash(model)
            case_report['decoder_equivalence_before_updates']=eval_equivalence(model,adapter,batch)
            if state_hash(model)!=weights_before or architecture_hash(model)!=architecture:
                raise ValueError('Adapter changed model weights or architecture during equivalence check')
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-4,foreach=False)
            nonempty=any(t['annotations'] for t in targets)
            for step in range(args.steps):
                model.train();model.model.backbone.eval();model.model.text_backbone.eval()
                optimizer.zero_grad(set_to_none=True)
                if args.device=='cuda':torch.cuda.synchronize()
                tick=time.perf_counter()
                output=adapter(model,batch)
                components=checked_loss(output,model.config)
                validate_predictions(output.outputs.logits,output.outputs.pred_boxes,batch['attention_mask'])
                class_paths=gradient_paths(output.loss_dict['loss_ce_enc'],probes)
                box_paths=gradient_paths(output.loss_dict['loss_bbox_enc']+output.loss_dict['loss_giou_enc'],probes)
                if any(p.grad is not None for p in model.parameters()):raise ValueError('Diagnostic gradients accumulated')
                if class_paths['parameters']['model.enc_output.weight']['l2']<=0:
                    raise ValueError('Encoder proposal classification path has no gradient')
                regression=box_paths['parameters']['model.encoder_output_bbox_embed.layers.2.weight']
                if regression is None or (regression['l2']<=0 if nonempty else regression['l2']!=0):
                    raise ValueError('Unexpected encoder box gradient for annotation count')
                before={n:p.detach().cpu().clone() for n,p in probes.items()}
                output.loss.backward()
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Nonfinite/missing corrected gradients')
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step()
                updates={n:{'has_gradient':p.grad is not None,'gradient_l2_after_clip':None if p.grad is None else float(p.grad.norm()),
                    'max_abs_update':float((p.detach().cpu()-before[n]).abs().max()),'adam_state_present':p in optimizer.state}
                    for n,p in probes.items()}
                for n in names[:2]+([names[2]] if nonempty else []):
                    if not updates[n]['adam_state_present'] or updates[n]['max_abs_update']==0:
                        raise ValueError('Encoder probe did not receive an actual optimizer update: '+n)
                if not nonempty and (components['loss_bbox_enc']!=0 or components['loss_giou_enc']!=0):
                    raise ValueError('Empty annotation regression must be zero')
                if state_hash(model,True)!=frozen:raise ValueError('Frozen parameters changed')
                case_report['steps'].append({'step':step+1,'loss':float(output.loss.detach()),'components':components,
                    'encoder_class_paths':class_paths,'encoder_box_paths':box_paths,'optimizer_probe_updates':updates,
                    'trainable_without_gradient':[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None],
                    'gradient_norm_before_clip':float(norm),'memory':memory(),'seconds_including_audits':time.perf_counter()-tick})
                del output,grads
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            gc.collect()
            case_report['decoder_equivalence_after_updates']=eval_equivalence(model,adapter,batch)
            case_report['memory']=memory()
            case_report['frozen_parameters_unchanged']=True
            report['cases'].append(case_report);publish(case+'_complete')
            if case=='baseline_11':
                publish('checkpoint_roundtrip')
                if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
                report['checkpoint_roundtrip']=checkpoint_roundtrip(model,adapter,processor,batch,images,targets,run,args.device)
                report['checkpoint_roundtrip']['memory_with_original_model_also_resident']=memory()
                publish('checkpoint_complete')
            del batch
            gc.collect()
        if architecture_hash(model)!=architecture:raise ValueError('Architecture changed')
        report['assessment']='PASS';report['status']='passed'
        report['scope']='Recorded small FP32 frozen-backbone synthetic corrected objective only'
    except Exception as exc:
        report['assessment']='FAIL';report['status']='failed'
        report['error']={'type':type(exc).__name__,'message':str(exc),'stage':report['stage']}
        report['memory_at_failure']=memory()
        (run/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');print(traceback.format_exc(),file=sys.stderr)
    finally:
        for image in images:image.close()
        report['base_snapshot_unchanged']=bool(source) and all(digest(SNAPSHOT/p)==h for p,h in source.items())
        report['library_unchanged']=bool(library) and all(digest(package/p)==h for p,h in library.items())
        if not report['base_snapshot_unchanged'] or not report['library_unchanged']:
            report['status']='failed';report['assessment']='FAIL'
        report['total_seconds']=time.perf_counter()-started
        atomic_json(path,report,replace=True)
        print(json.dumps({'assessment':report['assessment'],'report':str(path)},allow_nan=False),flush=True)
    return 0 if report['assessment']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
