"""Synthetic-only, offline Grounding DINO training qualification (not a dataset trainer)."""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
import uuid

from .configuration import MODEL_ID, REVISION, ROOT, strict_load
from .persistence import atomic_json

LABELS = ('cigarette', 'plastic_bottle', 'plastic_bag', 'plastic_wrapper',
          'plastic_film', 'drink_can', 'plastic_cup', 'styrofoam_piece')
PROMPT = '. '.join(label.replace('_', ' ') for label in LABELS) + '.'
SNAPSHOT = ROOT/'.cache/huggingface'/('models--'+MODEL_ID.replace('/', '--'))/'snapshots'/REVISION


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def require_training_environment():
    """Refuse to run training from inference .venv or inherited site-packages."""
    expected = (ROOT/'.venv-training').resolve()
    if Path(sys.prefix).resolve() != expected or sys.prefix == sys.base_prefix:
        raise ValueError('Run with .venv-training/Scripts/python.exe; inference .venv is prohibited')
    if 'include-system-site-packages = false' not in (expected/'pyvenv.cfg').read_text():
        raise ValueError('Training environment must not inherit system site-packages')
    versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    for file in ('requirements-windows.lock.txt', 'requirements-training-windows.lock.txt'):
        for line in (ROOT/file).read_text().splitlines():
            if line.strip() and not line.startswith(('#', '-')):
                name, version = line.strip().split('==')
                if importlib.metadata.version(name) != version:
                    raise ValueError(f'Training dependency mismatch: {name} must be {version}')
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    if check.returncode:
        raise ValueError('Training pip check failed: '+check.stdout+check.stderr)
    return {'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
            'platform': platform.platform(), 'packages': dict(sorted(versions.items())), 'pip_check': check.stdout.strip()}


def validate_taxonomy():
    taxonomy = strict_load(ROOT/'config/taco_taxonomy.v1.json')
    if [(r['id'],r['name']) for r in taxonomy['labels']] != list(enumerate(LABELS)):
        raise ValueError('Approved eight-class taxonomy/order changed')
    return taxonomy


def synthetic_batch(batch_size=2):
    """Created in memory; no photo loader, real-data path or image export exists."""
    from PIL import Image, ImageDraw
    if batch_size not in (1,2): raise ValueError('Synthetic batch size must be 1 or 2')
    specs = [((320,240), [(0,[8,8,20,24]), (1,[80,8,28,40]), (2,[160,8,50,35]),
               (3,[240,8,60,28]), (4,[8,120,70,18]), (5,[96,100,22,40]),
               (6,[168,140,24,30]), (7,[316,236,4,4])]),
             ((240,320), [(3,[35,35,80,100]), (0,[40,40,5,5]), (7,[0,270,20,50])])]
    images, targets = [], []
    for index, (size, objects) in enumerate(specs[:batch_size]):
        image = Image.new('RGB', size, (40,60,80)); draw = ImageDraw.Draw(image)
        annotations = []
        for j, (label, box) in enumerate(objects):
            x,y,w,h = box
            draw.rectangle([x,y,x+w-1,y+h-1], fill=((label*29+35)%256,(label*53+90)%256,(label*71+130)%256))
            annotations.append({'id':index*100+j+1,'image_id':index+1,'category_id':label,
                                'bbox':box,'area':w*h,'iscrowd':0})
        images.append(image); targets.append({'image_id':index+1, 'annotations':annotations})
    return images, targets


def validate_targets(images, targets):
    if not images or len(images) != len(targets): raise ValueError('Image/target batch mismatch')
    seen = set()
    for image, target in zip(images, targets):
        for obj in target['annotations']:
            label, box = obj['category_id'], obj['bbox']
            if type(label) is not int or not 0 <= label < len(LABELS): raise ValueError('Invalid target class')
            if len(box) != 4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in box):
                raise ValueError('Invalid target box')
            x,y,w,h = box
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > image.width or y+h > image.height:
                raise ValueError('Out-of-bounds or nonpositive box; no clipping allowed')
            if (obj['id'] in seen or obj.get('image_id') != target['image_id']
                    or obj.get('iscrowd') != 0 or obj.get('ignore',0) != 0 or obj.get('area') != w*h):
                raise ValueError('Ambiguous target metadata; no ignore-region semantics')
            seen.add(obj['id'])


def valid_image_loss_targets(label, target, original_wh, resized_wh, padded_wh):
    """Undo HF 4.51.3 annotation padding, preserving native prediction coordinates.

    The processor normalizes annotations to the padded canvas, while Grounding
    DINO references/predictions are normalized to the valid image. Check the
    processor result before converting; never repair changed annotation identities.
    Only boxes change: retain all other processed annotation metadata.
    """
    import torch
    ow, oh = original_wh
    rw, rh = resized_wh
    pw, ph = padded_wh
    expected = label['boxes'].new_tensor([
        [(x+w/2)/ow, (y+h/2)/oh, w/ow, h/oh]
        for x,y,w,h in (obj['bbox'] for obj in target['annotations'])
    ]).reshape(-1, 4)
    scale = expected.new_tensor([pw/rw, ph/rh, pw/rw, ph/rh])
    if (label['image_id'].item() != target['image_id']
            or label['class_labels'].tolist() != [obj['category_id'] for obj in target['annotations']]
            or label['boxes'].shape != expected.shape
            or not torch.allclose(label['boxes'], expected/scale, atol=2e-6, rtol=1e-5)):
        raise ValueError('Preprocessing changed target identity/count or box frame')
    label['boxes'] = label['boxes'] * scale
    if not torch.allclose(label['boxes'], expected, atol=2e-6, rtol=1e-5):
        raise ValueError('Loss targets differ from native valid-image coordinates')


def prepare_batch(processor, images, targets, shortest_edge=192, longest_edge=256):
    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
    validate_targets(images, targets)
    validate_taxonomy()
    batch = processor(images=images, text=[PROMPT]*len(images), annotations=copy.deepcopy(targets),
                      padding=True, return_tensors='pt',
                      size={'shortest_edge':shortest_edge,'longest_edge':longest_edge})
    maps = build_label_maps(torch.zeros(len(images),1,256), batch['input_ids'])
    spans = []
    for ids, mapping in zip(batch['input_ids'], maps):
        decoded = [processor.tokenizer.decode(ids[row[:len(ids)].bool()]) for row in mapping]
        if decoded != [label.replace('_',' ') for label in LABELS]:
            raise ValueError(f'Prompt span/category mismatch: {decoded}')
        spans.append(decoded)
    if batch['input_ids'].shape[1] > 256: raise ValueError('Prompt truncation risk')
    geometry = []
    padded_h,padded_w = batch['pixel_values'].shape[-2:]
    for i, (image, target, label) in enumerate(zip(images, targets, batch['labels'])):
        mask = batch['pixel_mask'][i]
        resized_w, resized_h = int(mask[0].sum()), int(mask[:,0].sum())
        valid_image_loss_targets(label, target, image.size, (resized_w,resized_h), (padded_w,padded_h))
        geometry.append({'image_id':target['image_id'], 'original_wh':list(image.size),
                         'resized_wh':[resized_w,resized_h], 'padded_wh':[padded_w,padded_h],
                         'class_labels':label['class_labels'].tolist(), 'boxes_cxcywh':label['boxes'].tolist()})
    return batch, {'spans':spans,'token_count':batch['input_ids'].shape[1],'geometry':geometry}


def validate_encoder_capacity(height, width, queries=900):
    # Pinned Swin-T feature levels: stride 8, 16, 32 plus projected stride 64.
    positions = sum(math.ceil(height/stride)*math.ceil(width/stride) for stride in (8,16,32,64))
    if positions < queries:
        raise ValueError(f'Input has only {positions} encoder positions for {queries} queries; use a larger explicit input size')
    return positions


def checked_loss(outputs, config):
    """Audit native loss, not a substitute criterion or threshold-filtered objective."""
    import torch
    if outputs.loss is None or not torch.isfinite(outputs.loss).all(): raise ValueError('Nonfinite total loss')
    if config.auxiliary_loss: raise ValueError('Smoke retains cached auxiliary_loss=false')
    weights = {'loss_ce':2.0,'loss_bbox':config.bbox_loss_coefficient,'loss_giou':config.giou_loss_coefficient}
    if config.two_stage: weights.update({k+'_enc':v for k,v in list(weights.items())})
    if not set(weights).issubset(outputs.loss_dict): raise ValueError('Native loss components missing')
    for value in outputs.loss_dict.values():
        if not torch.isfinite(value).all(): raise ValueError('Nonfinite loss component')
    expected = sum(outputs.loss_dict[k]*v for k,v in weights.items())
    if not torch.allclose(outputs.loss, expected): raise ValueError('Unexpected weighted native loss')
    return {k:float(v.detach().cpu()) for k,v in outputs.loss_dict.items()}


def state_hash(model, frozen_only=False):
    """Exact frozen parameter preservation, including values across optimizer steps."""
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not frozen_only or not parameter.requires_grad:
            result.update(name.encode()); result.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def move_batch(batch, device):
    return {k:([{name:value.to(device) for name,value in row.items()} for row in v] if k=='labels' else v.to(device))
            for k,v in batch.items()}


def loading_ok(info):
    failures = {k:v for k,v in info.items() if v}
    if failures: raise ValueError(f'Checkpoint loading mismatch: {failures}')


def validate_predictions(logits, boxes, attention_mask):
    import torch
    active = torch.zeros(logits.shape, dtype=torch.bool, device=logits.device)
    active[:,:,:attention_mask.shape[1]] = attention_mask[:,None,:].bool()
    if not torch.isfinite(logits[active]).all() or not torch.isneginf(logits[~active]).all():
        raise ValueError('Invalid active logits or text-padding sentinels')
    if not torch.isfinite(boxes).all() or not ((boxes >= 0) & (boxes <= 1)).all():
        raise ValueError('Invalid normalized prediction boxes')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--steps',type=int,choices=range(1,6),default=2)
    parser.add_argument('--batch-size',type=int,choices=(1,2),default=2)
    parser.add_argument('--shortest-edge',type=int,default=192)
    parser.add_argument('--longest-edge',type=int,default=256)
    parser.add_argument('--trainable',choices=('fusion-decoder','all'),default='fusion-decoder')
    parser.add_argument('--precision',choices=('fp32','fp16'),default='fp32')
    args = parser.parse_args(argv)
    if not 64 <= args.shortest_edge <= args.longest_edge <= 1333:
        parser.error('Require 64 <= shortest edge <= longest edge <= 1333')
    if args.precision=='fp16' and args.device!='cuda': parser.error('FP16 smoke requires explicit CUDA')
    run = ROOT/'runs/training'/uuid.uuid4().hex
    run.mkdir(parents=True, exist_ok=False)
    report = {'schema_version':'bantai.synthetic-training-smoke.v1','run_id':run.name,'status':'running',
              'synthetic_only':True,'model_id':MODEL_ID,'revision':REVISION,'settings':vars(args),
              'prompt':PROMPT,'labels':list(LABELS),'seed':42,'steps':[], 'stage':'environment',
              'checks':{},'memory':{},'timings_seconds':{}}
    report['implementation_sha256'] = {name:digest(ROOT/name) for name in
        ('scripts/training_smoke.py','scripts/smoke_train_grounding_dino.py',
         'requirements-training-windows.lock.txt','requirements-training-windows.hashes.txt')}
    report_path = run/'report.json'
    atomic_json(report_path, report)
    print(str(report_path),flush=True)
    images = []; torch = None; model = None; optimizer = None; batch = None; outputs = None
    source_hashes = {}
    started = time.perf_counter()

    def publish(stage):
        report['stage'] = stage
        atomic_json(report_path,report,replace=True)
        print(stage,flush=True)

    def sync():
        if args.device=='cuda': torch.cuda.synchronize()

    def memory():
        if torch is not None and args.device=='cuda' and torch.cuda.is_initialized():
            return {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'current_allocated_bytes':torch.cuda.memory_allocated(),
                    'current_reserved_bytes':torch.cuda.memory_reserved()}
        return {'cuda_peak_memory':'not_applicable_cpu_or_uninitialized'}

    try:
        report['environment'] = require_training_environment()
        import torch as torch_module
        torch = torch_module
        from transformers import GroundingDinoForObjectDetection, AutoProcessor
        random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
        if args.device=='cuda':
            if not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
            torch.cuda.manual_seed_all(42)
            report['environment']['gpu'] = {'name':torch.cuda.get_device_name(), 'capability':list(torch.cuda.get_device_capability()),
                                           'total_bytes':torch.cuda.get_device_properties(0).total_memory,'torch_cuda':torch.version.cuda}
            smi = subprocess.run(['nvidia-smi'],capture_output=True,text=True)
            (run/'nvidia-smi-before.txt').write_text(smi.stdout+smi.stderr,encoding='utf-8')
            torch.cuda.reset_peak_memory_stats()
        publish('source_verification')
        if not SNAPSHOT.is_dir(): raise FileNotFoundError('Pinned cached snapshot missing; downloads prohibited')
        source_hashes = {p.name:digest(p) for p in SNAPSHOT.iterdir() if p.is_file()}
        report['base_snapshot_sha256'] = source_hashes
        report['taxonomy_sha256'] = digest(ROOT/'config/taco_taxonomy.v1.json')
        validate_taxonomy()
        publish('load')
        before = time.perf_counter()
        model, info = GroundingDinoForObjectDetection.from_pretrained(SNAPSHOT,local_files_only=True,
            use_safetensors=True,trust_remote_code=False,disable_custom_kernels=True,output_loading_info=True)
        loading_ok(info)
        processor = AutoProcessor.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=False)
        report['checks']['base_loading'] = info
        if model.config.num_queries != 900 or model.config.max_text_len != 256 or model.config.auxiliary_loss:
            raise ValueError('Cached architecture/loss configuration differs from assessed baseline')
        if args.trainable=='fusion-decoder':
            for name, parameter in model.named_parameters():
                if name.startswith(('model.backbone.','model.text_backbone.')): parameter.requires_grad_(False)
        else:
            for parameter in model.parameters(): parameter.requires_grad_(True)
        report['parameters'] = {'total':sum(p.numel() for p in model.parameters()),
                                'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
                                'frozen_names':[n for n,p in model.named_parameters() if not p.requires_grad]}
        frozen_before = state_hash(model, frozen_only=True)
        train_name, train_probe = next((n,p) for n,p in model.named_parameters() if p.requires_grad)
        before_probe = train_probe.detach().clone()
        model.to(args.device); sync()
        report['timings_seconds']['load_and_transfer'] = time.perf_counter()-before
        publish('preprocessing')
        images, targets = synthetic_batch(args.batch_size)
        batch, report['preprocessing'] = prepare_batch(processor,images,targets,args.shortest_edge,args.longest_edge)
        report['preprocessing']['encoder_positions'] = validate_encoder_capacity(*batch['pixel_values'].shape[-2:],model.config.num_queries)
        report['synthetic_targets'] = targets
        batch = move_batch(batch,args.device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-4,foreach=False)
        scaler = torch.amp.GradScaler('cuda',enabled=args.precision=='fp16')
        publish('training')
        for step in range(args.steps):
            model.train()
            if args.trainable=='fusion-decoder':
                model.model.backbone.eval(); model.model.text_backbone.eval()
            optimizer.zero_grad(set_to_none=True)
            sync(); before = time.perf_counter()
            with torch.autocast(device_type=args.device,dtype=torch.float16,enabled=args.precision=='fp16'):
                outputs = model(**batch)
            components = checked_loss(outputs,model.config)
            validate_predictions(outputs.logits,outputs.pred_boxes,batch['attention_mask'])
            native_loss_observations = {
                'component_requires_grad':{k:v.requires_grad for k,v in outputs.loss_dict.items()},
                'encoder_box_predictions_require_grad':outputs.encoder_pred_boxes.requires_grad,
                'encoder_logits_require_grad':outputs.encoder_logits.requires_grad}
            value = float(outputs.loss.detach().cpu())
            scaler.scale(outputs.loss).backward(); scaler.unscale_(optimizer)
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not all(torch.isfinite(g).all().item() for g in grads): raise ValueError('Missing/nonfinite gradients')
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
            if float(norm)==0: raise ValueError('All gradients are zero')
            scale_before = scaler.get_scale()
            scaler.step(optimizer); scaler.update(); sync()
            if scaler.get_scale() < scale_before: raise ValueError('AMP optimizer update skipped')
            report['steps'].append({'step':step+1,'loss':value,'components':components,'gradient_norm_before_clip':float(norm),
                'parameters_with_gradients':len(grads),'native_loss_observations':native_loss_observations,
                'trainable_parameters_without_gradients':[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None],
                'seconds_forward_backward_step':time.perf_counter()-before,'memory':memory()})
            outputs = None; del grads
            publish('training')
        report['memory']['training'] = memory()
        report['checks']['optimizer_state_nonempty'] = bool(optimizer.state)
        changed = not torch.equal(before_probe, dict(model.named_parameters())[train_name].detach().cpu())
        report['checks']['changed_trainable_probe'] = {'name':train_name,'changed':changed}
        if not changed: raise ValueError('Optimizer did not change the trainable probe')
        if state_hash(model, frozen_only=True) != frozen_before: raise ValueError('Frozen parameters changed')
        report['checks']['frozen_parameters_unchanged'] = True
        optimizer.zero_grad(set_to_none=True)
        optimizer = None; gc.collect()
        if args.device=='cuda': torch.cuda.empty_cache()
        publish('save')
        model.eval()
        inputs = {k:v for k,v in batch.items() if k!='labels'}
        with torch.no_grad():
            reference = model(**inputs)
        validate_predictions(reference.logits,reference.pred_boxes,inputs['attention_mask'])
        reference_logits, reference_boxes = reference.logits.cpu(), reference.pred_boxes.cpu()
        del reference
        if not torch.isfinite(reference_boxes).all(): raise ValueError('Invalid evaluation boxes')
        full_state = state_hash(model)
        checkpoint = run/'checkpoint'
        model.save_pretrained(checkpoint,safe_serialization=True)
        processor.save_pretrained(checkpoint)
        atomic_json(run/'smoke_context.json', {'base_revision':REVISION,'prompt':PROMPT,'taxonomy':validate_taxonomy(),
                    'settings':vars(args),'seed':42,'completed_steps':len(report['steps']),
                    'resume_supported':False,'purpose':'synthetic compatibility only; not a resumable dataset trainer'})
        report['checkpoint'] = {'directory':'checkpoint','resume_supported':False,
                                'sha256':{p.name:digest(p) for p in checkpoint.iterdir() if p.is_file()}}
        model = None; gc.collect()
        if args.device=='cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        publish('reload')
        model, info = GroundingDinoForObjectDetection.from_pretrained(checkpoint,local_files_only=True,
            use_safetensors=True,trust_remote_code=False,output_loading_info=True)
        loading_ok(info)
        if state_hash(model) != full_state: raise ValueError('Reloaded parameter values differ')
        restored = AutoProcessor.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
        replay_batch, replay_details = prepare_batch(restored,images,targets,args.shortest_edge,args.longest_edge)
        replay_details['encoder_positions'] = validate_encoder_capacity(*replay_batch['pixel_values'].shape[-2:],model.config.num_queries)
        for key,value in replay_batch.items():
            if key!='labels' and not torch.equal(value,batch[key].cpu()): raise ValueError('Reloaded processor inputs differ')
        if replay_details != report['preprocessing']: raise ValueError('Reloaded label geometry differs')
        replay_batch = move_batch(replay_batch,args.device)
        model.to(args.device).eval()
        with torch.no_grad():
            replay = model(**{k:v for k,v in replay_batch.items() if k!='labels'})
        validate_predictions(replay.logits,replay.pred_boxes,replay_batch['attention_mask'])
        logits, boxes = replay.logits.cpu(), replay.pred_boxes.cpu()
        if not torch.equal(torch.isneginf(logits),torch.isneginf(reference_logits)):
            raise ValueError('Reloaded text-padding sentinel mask differs')
        active = torch.isfinite(reference_logits)
        if not torch.isfinite(logits[active]).all() or not torch.allclose(logits[active],reference_logits[active],atol=1e-5,rtol=1e-5):
            raise ValueError('Reloaded active logits differ')
        if not torch.allclose(boxes,reference_boxes,atol=1e-5,rtol=1e-5): raise ValueError('Reloaded boxes differ')
        report['checks']['reload'] = {'loading':info,'parameter_hash_equal':True,'processor_equal':True,
            'atol':1e-5,'rtol':1e-5,'max_active_logit_difference':float((logits[active]-reference_logits[active]).abs().max()),
            'max_box_difference':float((boxes-reference_boxes).abs().max())}
        report['memory']['reload'] = memory()
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = {'type':type(exc).__name__,'message':str(exc),'stage':report['stage']}
        report['memory']['at_failure'] = memory()
        (run/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
    finally:
        for image in images: image.close()
        if source_hashes:
            intact = all((SNAPSHOT/name).is_file() and digest(SNAPSHOT/name)==value for name,value in source_hashes.items())
            report['checks']['base_snapshot_unchanged'] = intact
            if not intact: report['status'] = 'failed'
        report['timings_seconds']['total_including_hashes_load_save_reload'] = time.perf_counter()-started
        atomic_json(report_path,report,replace=True)
        print(json.dumps({'status':report['status'],'report':str(report_path),'error':report.get('error')},allow_nan=False),flush=True)
    return 0 if report['status']=='passed' else 1
