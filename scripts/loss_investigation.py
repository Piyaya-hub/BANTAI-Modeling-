"""Observe unmodified native loss; counterfactual tensor audits NEVER drive updates."""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import traceback
import uuid

from .configuration import ROOT, REVISION, MODEL_ID
from .persistence import atomic_json
from .training_smoke import (SNAPSHOT, PROMPT, digest, require_training_environment, synthetic_batch,
    prepare_batch, move_batch, checked_loss, loading_ok, validate_encoder_capacity, state_hash, validate_predictions)

CASES = ('baseline_11', 'two_targets', 'one_target', 'empty')


def targets_for_case(targets, case):
    result = copy.deepcopy(targets)
    if case not in CASES: raise ValueError('Unknown bounded diagnostic case')
    if case != 'baseline_11':
        for i,row in enumerate(result):
            row['annotations'] = row['annotations'][:1] if case=='two_targets' or (case=='one_target' and i==0) else []
    return result


def selected_encoder_tensors(outputs, num_queries):
    """Read-only hypothesis audit: gather published pre-detach outputs; no model patch."""
    import torch
    indices = torch.topk(outputs.enc_outputs_class.max(-1).values,num_queries,dim=1).indices
    logits = outputs.enc_outputs_class.gather(1,indices[:,:,None].expand(-1,-1,outputs.enc_outputs_class.shape[-1]))
    boxes = outputs.enc_outputs_coord_logits.gather(1,indices[:,:,None].expand(-1,-1,4)).sigmoid()
    return logits,boxes,indices


def make_criterion(config):
    from transformers.loss.loss_grounding_dino import GroundingDinoHungarianMatcher, GroundingDinoImageLoss
    matcher = GroundingDinoHungarianMatcher(class_cost=config.class_cost,bbox_cost=config.bbox_cost,giou_cost=config.giou_cost)
    return GroundingDinoImageLoss(matcher,config.focal_alpha,['labels','boxes','cardinality'])


def focal_decomposition(logits, onehot, mask, normalizer, alpha=.25):
    """Independent elementwise calculation, for auditing only; no backward objective."""
    import torch
    import torch.nn.functional as F
    active_logits = logits[mask].detach()
    targets = onehot[mask].float().detach()
    p = active_logits.sigmoid()
    pt = p*targets+(1-p)*(1-targets)
    terms = F.binary_cross_entropy_with_logits(active_logits,targets,reduction='none')*(1-pt).square()
    terms *= alpha*targets+(1-alpha)*(1-targets)
    full = torch.zeros_like(logits)
    full[mask] = terms
    return {'positive':float(terms[targets.bool()].sum()/normalizer),
            'negative':float(terms[~targets.bool()].sum()/normalizer),
            'total':float(terms.sum()/normalizer),'sum_before_normalization':float(terms.sum()),
            'active_cells':int(targets.numel()),'positive_cells':int(targets.sum()),
            'per_token':(full.sum(dim=(0,1))/normalizer).cpu().tolist()}


def logit_stats(logits, mask):
    import torch
    values = logits.detach()[mask]
    return {'min':float(values.min()),'max':float(values.max()),'mean':float(values.mean()),
            'positive_fraction':float((values>0).float().mean()),
            'quantiles':torch.quantile(values,torch.tensor([0.,.5,.95,1.],device=values.device)).cpu().tolist()}


def branch_audit(criterion, logits, boxes, maps, mask, targets, ids, tokenizer, alpha):
    import torch
    data = {'logits':logits,'pred_boxes':boxes,'label_maps':maps,'text_mask':mask}
    losses = criterion(data,targets)
    indices = criterion.matcher(data,targets)
    onehot = criterion._get_target_classes_one_hot(data,targets,indices)
    n = max(sum(len(t['class_labels']) for t in targets),1)
    decomp = focal_decomposition(logits,onehot,mask,n,alpha)
    if not torch.isclose(losses['loss_ce'].detach(),torch.tensor(decomp['total'],device=logits.device),rtol=1e-5,atol=1e-5):
        raise ValueError('Independent focal decomposition differs from native criterion')
    ordinary = maps[0].bool().any(dim=0)
    per_token = decomp.pop('per_token')
    decomp['ordinary_token_loss'] = sum(v for v,is_ordinary in zip(per_token,ordinary.tolist()) if is_ordinary)
    decomp['delimiter_special_loss'] = sum(v for v,is_ordinary in zip(per_token,ordinary.tolist()) if not is_ordinary)
    tokens = [{'position':i,'token':tokenizer.convert_ids_to_tokens(int(ids[0,i])),
               'ordinary':bool(ordinary[i]),'focal_loss':per_token[i]} for i in range(ids.shape[1])]
    audit = {'losses':{k:float(v.detach()) for k,v in losses.items()},'normalizer':n,
             'matches':[{'queries':a.cpu().tolist(),'target_indexes':b.cpu().tolist()} for a,b in indices],
             'focal':decomp,'tokens':tokens,'logits':logit_stats(logits,mask),
             'requires_grad':{k:v.requires_grad for k,v in losses.items()}}
    return audit,losses


def gradient_paths(loss, probes):
    import torch
    if not loss.requires_grad: return {'loss_requires_grad':False,'parameters':{name:None for name in probes}}
    gradients = torch.autograd.grad(loss,list(probes.values()),retain_graph=True,allow_unused=True)
    result = {}
    for name,grad in zip(probes,gradients):
        if grad is None: result[name] = None
        else:
            if not torch.isfinite(grad).all(): raise ValueError('Nonfinite diagnostic gradient')
            result[name] = {'l2':float(grad.norm()),'max_abs':float(grad.abs().max())}
    return {'loss_requires_grad':True,'parameters':result}


def comparison_flags(logits, boxes, outputs, captures):
    import torch
    result = {'selected_boxes_equal_native':torch.equal(boxes.detach(),outputs.encoder_pred_boxes.detach()),
              'selected_boxes_require_grad':boxes.requires_grad,'native_boxes_require_grad':outputs.encoder_pred_boxes.requires_grad,
              'native_logits_equal_between_images':torch.equal(outputs.encoder_logits[0],outputs.encoder_logits[1]),
              'selected_logits_equal_between_images':torch.equal(logits[0],logits[1])}
    # Two calls: selected proposal scoring, then learned-query scoring for reported encoder loss.
    if len(captures)!=2: raise ValueError(f'Unexpected encoder classifier call count: {len(captures)}')
    query,text,mask = captures[1]
    reproduced = query @ text.transpose(-1,-2)
    active = mask[:,None,:].expand_as(reproduced)
    result['native_logits_equal_learned_query_dot_text'] = torch.equal(reproduced[active],outputs.encoder_logits[:,:,:text.shape[1]][active])
    result['query_inputs_equal_between_images'] = torch.equal(query[0],query[1])
    result['preencoder_text_equal_between_images'] = torch.equal(text[0],text[1])
    result['classifier_input_mean_norms'] = [{'visual_or_query':float(q.detach().norm(dim=-1).mean()),
                                            'text':float(t.detach().norm(dim=-1).mean())} for q,t,_ in captures]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    args = parser.parse_args(argv)
    if len(set(args.cases))!=len(args.cases): parser.error('Do not repeat cases')
    run = ROOT/'runs/training'/('investigation-'+uuid.uuid4().hex)
    run.mkdir(parents=True,exist_ok=False)
    report = {'schema_version':'bantai.loss-investigation.v1','status':'running','run_id':run.name,
              'model_id':MODEL_ID,'revision':REVISION,'synthetic_only':True,'prompt':PROMPT,'device':args.device,
              'settings':{'batch_size':2,'shortest_edge':192,'longest_edge':256,'precision':'fp32',
                'seed':42,'trainable':'fusion-decoder','optimizer_objective':'native_total_only',
                'candidate_audit':'same native criterion on gathered encoder tensors; never optimized',
                'native_steps_per_case':1,'reset_to_pretrained_before_each_case':True},'cases':[],'stage':'environment'}
    path = run/'report.json'
    atomic_json(path,report); print(path,flush=True)
    torch = None; images=[]; source_hashes={}; lib_hashes={}

    def publish(stage):
        report['stage']=stage; atomic_json(path,report,replace=True); print(stage,flush=True)

    try:
        report['environment']=require_training_environment()
        import torch as torch_module
        torch = torch_module
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
        from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps,build_text_mask
        import transformers
        lib = Path(transformers.__file__).parent
        lib_hashes = {str(lib/p):digest(lib/p) for p in ('models/grounding_dino/modeling_grounding_dino.py',
                      'loss/loss_grounding_dino.py','loss/loss_for_object_detection.py')}
        report['library_sha256']=lib_hashes
        report['implementation_sha256']={p:digest(ROOT/p) for p in ('scripts/loss_investigation.py','scripts/investigate_training_loss.py','scripts/training_smoke.py')}
        source_hashes={p.name:digest(p) for p in SNAPSHOT.iterdir() if p.is_file()}
        report['base_snapshot_sha256']=source_hashes
        if args.device=='cuda':
            if not torch.cuda.is_available(): raise RuntimeError('Requested CUDA unavailable; no fallback')
            report['gpu']={'name':torch.cuda.get_device_name(),'total_bytes':torch.cuda.get_device_properties(0).total_memory,'cuda':torch.version.cuda}
        torch.set_num_threads(4)
        torch.manual_seed(42); random.seed(42)
        publish('load')
        model,info = GroundingDinoForObjectDetection.from_pretrained(SNAPSHOT,local_files_only=True,use_safetensors=True,
                          trust_remote_code=False,disable_custom_kernels=True,output_loading_info=True)
        loading_ok(info)
        if model.config.num_queries!=900 or model.config.auxiliary_loss or not model.config.two_stage:
            raise ValueError('Unexpected pinned query/loss configuration')
        report['config']={k:getattr(model.config,k) for k in ('num_queries','max_text_len','auxiliary_loss','two_stage',
                        'embedding_init_target','two_stage_bbox_embed_share','class_cost','bbox_cost','giou_cost','focal_alpha',
                        'bbox_loss_coefficient','giou_loss_coefficient')}
        for name,p in model.named_parameters():
            if name.startswith(('model.backbone.','model.text_backbone.')): p.requires_grad_(False)
        baseline={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        frozen = state_hash(model,True)
        model.to(args.device)
        processor=AutoProcessor.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=False)
        images,original = synthetic_batch(2)
        criterion=make_criterion(model.config).to(args.device)
        params=dict(model.named_parameters())
        probes={}
        for prefix in ('model.query_position_embeddings.','model.text_projection.','model.enc_output.',
                       'model.enc_output_norm.','model.encoder_output_bbox_embed.layers.2.','model.encoder.',
                       'model.decoder.','model.decoder.bbox_embed.0.'):
            matches=[(n,p) for n,p in params.items() if n.startswith(prefix) and p.requires_grad]
            if not matches: raise ValueError('Diagnostic probe prefix not found: '+prefix)
            n,p = matches[0]
            probes[n]=p
        reference_logits=None
        for case in args.cases:
            publish(case)
            model.load_state_dict(baseline,strict=True)
            model.zero_grad(set_to_none=True)
            gc.collect()
            if args.device=='cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.manual_seed_all(42)
            torch.manual_seed(42)
            model.train(); model.model.backbone.eval(); model.model.text_backbone.eval()
            targets=targets_for_case(original,case)
            batch,geometry=prepare_batch(processor,images,targets,192,256)
            validate_encoder_capacity(*batch['pixel_values'].shape[-2:])
            batch=move_batch(batch,args.device)
            captures=[]
            handle=model.model.encoder_output_class_embed.register_forward_pre_hook(lambda module,inputs:captures.append(inputs))
            started=time.perf_counter()
            try: outputs=model(**batch)
            finally: handle.remove()
            components=checked_loss(outputs,model.config)
            validate_predictions(outputs.logits,outputs.pred_boxes,batch['attention_mask'])
            maps=build_label_maps(outputs.logits,batch['input_ids'])
            mask=build_text_mask(outputs.logits,batch['attention_mask'])
            decoder,decoder_losses=branch_audit(criterion,outputs.logits,outputs.pred_boxes,maps,mask,batch['labels'],
                                                    batch['input_ids'],processor.tokenizer,model.config.focal_alpha)
            native,native_losses=branch_audit(criterion,outputs.encoder_logits,outputs.encoder_pred_boxes,maps,mask,batch['labels'],
                                                    batch['input_ids'],processor.tokenizer,model.config.focal_alpha)
            candidate_logits,candidate_boxes,indices=selected_encoder_tensors(outputs,model.config.num_queries)
            candidate,candidate_losses=branch_audit(criterion,candidate_logits,candidate_boxes,maps,mask,batch['labels'],
                                                    batch['input_ids'],processor.tokenizer,model.config.focal_alpha)
            for key,value in decoder['losses'].items():
                if abs(value-components[key])>1e-4: raise ValueError('Decoder audit loss mismatch')
            for key,value in native['losses'].items():
                if abs(value-components[key+'_enc'])>1e-4: raise ValueError('Encoder audit loss mismatch')
            flags=comparison_flags(candidate_logits,candidate_boxes,outputs,captures)
            if reference_logits is None: reference_logits=outputs.encoder_logits.detach().cpu()
            flags['native_logits_same_as_first_case']=torch.equal(reference_logits,outputs.encoder_logits.detach().cpu())
            paths={}
            for name,loss in [('native_encoder_ce',native_losses['loss_ce']),('native_encoder_box',native_losses['loss_bbox']+native_losses['loss_giou']),
                              ('decoder_ce',decoder_losses['loss_ce']),('decoder_box',decoder_losses['loss_bbox']+decoder_losses['loss_giou']),
                              ('hypothesis_encoder_ce',candidate_losses['loss_ce']),('hypothesis_encoder_box',candidate_losses['loss_bbox']+candidate_losses['loss_giou'])]:
                paths[name]=gradient_paths(loss,probes)
            del loss
            ablations={}
            if case=='baseline_11':
                # Isolate the two hypotheses; these losses also NEVER enter optimizer.backward.
                same_scores,undetched_losses=branch_audit(criterion,outputs.encoder_logits,candidate_boxes,maps,mask,batch['labels'],
                                                batch['input_ids'],processor.tokenizer,model.config.focal_alpha)
                for key,value in same_scores['losses'].items():
                    if abs(value-native['losses'][key])>1e-4: raise ValueError('Detachment changed loss values')
                ablations['only_box_detach_boundary']={'losses':same_scores['losses'],
                    'box_gradient_paths':gradient_paths(undetched_losses['loss_bbox']+undetched_losses['loss_giou'],probes)}
                same_boxes,rewired_class_losses=branch_audit(criterion,candidate_logits,outputs.encoder_pred_boxes,maps,mask,batch['labels'],
                                                batch['input_ids'],processor.tokenizer,model.config.focal_alpha)
                ablations['only_classification_inputs']={'losses':same_boxes['losses'],
                    'box_loss_requires_grad':rewired_class_losses['loss_bbox'].requires_grad}
                del undetched_losses,rewired_class_losses
            if any(p.grad is not None for p in model.parameters()):
                raise ValueError('Diagnostic autograd calls must not accumulate optimizer gradients')
            before={n:p.detach().cpu().clone() for n,p in probes.items()}
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-4,foreach=False)
            optimizer.zero_grad(set_to_none=True)
            # Only the original unmodified model-returned total is differentiated for an update.
            outputs.loss.backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            if not all(torch.isfinite(g).all() for g in grads): raise ValueError('Nonfinite native gradient')
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step()
            changes={n:{'has_native_grad':p.grad is not None,'native_grad_l2_after_clip':None if p.grad is None else float(p.grad.norm()),
                        'max_abs_update':float((p.detach().cpu()-before[n]).abs().max()),'adam_state_present':p in optimizer.state}
                     for n,p in probes.items()}
            if state_hash(model,True)!=frozen: raise ValueError('Frozen parameters changed')
            memory={}
            if args.device=='cuda':
                torch.cuda.synchronize()
                memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            report['cases'].append({'name':case,'target_counts':[len(t['annotations']) for t in targets], 'preprocessing':geometry,
                'total_native_loss':float(outputs.loss.detach()),'native_components':components,
                'decoder':decoder,'native_encoder':native,'hypothesis_only_selected_encoder':candidate,
                'tensor_checks':flags,'gradient_paths':paths,'native_optimizer_probe_updates':changes,'native_gradient_norm':float(norm),
                'baseline_factor_isolation':ablations,
                'frozen_unchanged':True,'memory_scope':'forward, diagnostic autograd.grad calls, native backward/optimizer; no warmup',
                'memory':memory,'seconds_including_audits':time.perf_counter()-started})
            publish(case+'_complete')
            optimizer.zero_grad(set_to_none=True)
            del outputs,captures,decoder_losses,native_losses,candidate_losses,candidate_logits,candidate_boxes,optimizer,grads,batch
            gc.collect()
        report['status']='completed'
        report['real_data_training_ready']=False
    except Exception as exc:
        report['status']='failed'; report['error']={'type':type(exc).__name__,'message':str(exc),'stage':report['stage']}
        (run/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print(traceback.format_exc(),file=sys.stderr)
    finally:
        for image in images: image.close()
        report['base_snapshot_unchanged']=bool(source_hashes) and all(digest(SNAPSHOT/name)==value for name,value in source_hashes.items())
        report['library_unchanged']=bool(lib_hashes) and all(digest(name)==value for name,value in lib_hashes.items())
        if not report['base_snapshot_unchanged'] or not report['library_unchanged']: report['status']='failed'
        atomic_json(path,report,replace=True)
        print(json.dumps({'status':report['status'],'report':str(path)},allow_nan=False),flush=True)
    return 0 if report['status']=='completed' else 1
