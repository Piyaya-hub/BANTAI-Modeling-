"""Validation-only historical reevaluation; no optimizer/resume entry point."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'scripts'

import argparse
import gc
import json
import os
from pathlib import Path
import time
import uuid

from .configuration import ROOT, strict_load
from .evaluation_provenance import verify_evaluation_checkpoint
from .persistence import atomic_json
from .taco_evaluation import PROTOCOL, evaluate
from .taco_training_data import CocoTrainingDataset
from .train_taco import load_model, memory
from .training_adapter import config_signature
from .training_smoke import SNAPSHOT, digest, require_training_environment, state_hash

POLICY = ROOT/'config/taco_evaluation_compatibility.v1.json'


def configuration_semantics(config):
    # HF configuration_utils initializes this flag false; modeling_utils sets it
    # true after backend selection. Compare the selected backend itself instead.
    metadata = {'_name_or_path', 'transformers_version', '_attn_implementation_autoset'}
    value = {k:v for k,v in config.to_dict().items() if k not in metadata}
    value['_resolved_attention_implementation'] = config._attn_implementation
    return value


def validate_bundle(model, processor, adapter, historical_context):
    """Check saved semantics against the unchanged pinned pretrained processor."""
    import torch
    from transformers import AutoConfig, AutoProcessor
    config_signature(model.config)
    adapter.verify_manifest(model, historical_context['adapter_context'])
    reference = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    reference_config = AutoConfig.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    # Existing load_model explicitly disables custom kernels for both model states.
    reference_config.disable_custom_kernels = True
    actual = configuration_semantics(model.config)
    expected = configuration_semantics(reference_config)
    if actual != expected:
        differences = sorted(k for k in set(actual)|set(expected) if actual.get(k) != expected.get(k))
        raise ValueError('Checkpoint architecture/configuration mismatch: '+str(differences))
    if processor.image_processor.to_dict() != reference.image_processor.to_dict():
        raise ValueError('Checkpoint image processor differs from pinned preprocessing')
    a,b = processor.tokenizer, reference.tokenizer
    def backend(tokenizer):
        value = json.loads(tokenizer.backend_tokenizer.to_str())
        # Transient call state is explicitly replaced by collate(padding=True).
        value.pop('padding', None); value.pop('truncation', None)
        return value
    if (a.get_vocab() != b.get_vocab() or a.special_tokens_map != b.special_tokens_map
            or a.get_added_vocab() != b.get_added_vocab() or backend(a) != backend(b)):
        raise ValueError('Checkpoint tokenizer differs from pinned text semantics')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Evaluation requires unchanged FP32 parameters')
    return {'architecture': 'pinned_configuration_and_resolved_attention_backend_equal',
            'ignored_bookkeeping':['_name_or_path','transformers_version','_attn_implementation_autoset'],
            'image_processor': 'pinned_configuration_equal',
            'tokenizer': 'vocabulary_special_tokens_backend_equal',
            'adapter': 'historical_manifest_equal', 'precision': 'fp32'}


def evaluate_checkpoint(folder, report_path, policy, dataset, rights, device,
                        *, split='validation', source_root=ROOT):
    """Verify, load safetensors only, freeze all parameters, evaluate and reverify."""
    import torch
    proof = verify_evaluation_checkpoint(folder, report_path, policy, dataset, rights,
                                         split=split, source_root=source_root)
    model = processor = adapter = None
    try:
        model,processor,adapter = load_model(folder,device)
        compatibility = validate_bundle(model,processor,adapter,proof['historical_context'])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before = state_hash(model,True)
        validation = CocoTrainingDataset(dataset,'validation',rights)
        counts = {'images':len(validation), 'annotations':len(validation.data['annotations'])}
        if counts != policy['validation_counts']:
            raise ValueError('Validation corpus counts differ from reviewed policy')
        with torch.inference_mode():
            result = evaluate(model,processor,adapter,validation,device,256,384)
        if state_hash(model,True) != before:
            raise ValueError('Evaluation changed model parameters')
        after = verify_evaluation_checkpoint(folder,report_path,policy,dataset,rights,
                                             split=split,source_root=source_root)
        if proof != after:
            raise ValueError('Provenance changed during evaluation')
        return {'provenance':proof, 'compatibility':compatibility,
                'parameter_values_unchanged':True, 'validation_photo_reads':validation.photo_reads,
                'loss_semantics':'recalculated_with_corrected_valid_image_targets_not_historical_objective',
                'evaluation':result}
    finally:
        model = processor = adapter = None
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()


def score_distribution(evaluation):
    import numpy as np
    values = np.array([r['score'] for r in evaluation['raw_queries']])
    return {'queries':len(values), 'quantiles':dict(zip(
        ('min','p25','median','p75','p90','p99','max'),
        map(float,np.quantile(values,[0,.25,.5,.75,.9,.99,1])))),
        'queries_ge_025':int((values>=.25).sum()),
        'invalid_clipped_boxes':sum(not r['valid_clipped_box'] for r in evaluation['raw_queries'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-run',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=ROOT/'exports/taco/strict-clean-stage2-v1')
    parser.add_argument('--rights',type=Path,required=True)
    parser.add_argument('--baseline-record',type=Path,required=True)
    parser.add_argument('--split',choices=['validation'],default='validation')
    parser.add_argument('--device',choices=['cuda'],default='cuda')
    args = parser.parse_args(argv)
    policy = strict_load(POLICY)
    historical = strict_load(args.historical_run/'report.json')
    environment = require_training_environment()
    if environment['packages'] != historical['contract']['packages']:
        raise ValueError('Historical training/evaluation environment changed')
    if digest(args.baseline_record) != policy['baseline_record_sha256']:
        raise ValueError('Corrected pretrained baseline record anchor mismatch')
    baseline = strict_load(args.baseline_record)
    if (baseline['protocol'] != PROTOCOL or baseline['implementation_sha256'] != policy['current_sources']
            or baseline['dataset'] != historical['contract']['dataset']
            or baseline['rights_record_sha256'] != digest(args.rights)):
        raise ValueError('Corrected baseline protocol/source/dataset/rights mismatch')
    snapshot = {p.name:digest(p) for p in SNAPSHOT.iterdir() if p.is_file()}
    if snapshot != historical['contract']['base_snapshot_sha256']:
        raise ValueError('Pinned pretrained snapshot changed')
    import transformers
    for name, expected in policy['processor_source_sha256'].items():
        if digest(Path(transformers.__file__).parent/name) != expected:
            raise ValueError('Native processor source changed')
    # Verify BOTH historical bundles before loading either model or photograph.
    for epoch in ('epoch-01','epoch-02'):
        verify_evaluation_checkpoint(args.historical_run/'checkpoints'/epoch,
            args.historical_run/'report.json',policy,args.dataset,args.rights)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent fallback')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run = ROOT/'runs/training'/('step1c-evaluation-'+uuid.uuid4().hex)
    run.mkdir()
    record = {'schema_version':'bantai.historical-evaluation.v1','status':'running',
        'purpose':'validation_only_no_training','optimizer_updates':0,'test_photo_reads':0,
        'training_resume_authorized':False,'protocol':PROTOCOL,'policy':policy,
        'policy_sha256':digest(POLICY),'environment':environment,
        'historical_run':args.historical_run.name,'pretrained_baseline_record_sha256':digest(args.baseline_record),
        'pretrained_corrected':baseline['corrected_baseline'],
        'gpu':{'name':torch.cuda.get_device_name(), 'torch_cuda':torch.version.cuda,
               'total_bytes':torch.cuda.get_device_properties(0).total_memory},
        'settings':{'shortest_edge':256,'longest_edge':384,'padding':'384_square','batch_size':1,'precision':'fp32'},
        'models':{}}
    atomic_json(run/'record.json',record)
    print(run,flush=True)
    start = time.perf_counter()
    try:
        for epoch in ('epoch-01','epoch-02'):
            print('validation-only '+epoch,flush=True)
            torch.cuda.reset_peak_memory_stats()
            value = evaluate_checkpoint(args.historical_run/'checkpoints'/epoch,
                args.historical_run/'report.json',policy,args.dataset,args.rights,args.device)
            filename = epoch+'-corrected.json'
            atomic_json(run/filename,value)
            historical_file = 'validation-epoch-'+str(int(epoch[-2:]))+'.json'
            old = strict_load(args.historical_run/historical_file)
            prior_queries = {(r['image_id'],r['query_id']):r for r in old['raw_queries']}
            new_queries = {(r['image_id'],r['query_id']):r for r in value['evaluation']['raw_queries']}
            if prior_queries.keys() != new_queries.keys():
                raise ValueError('Historical/current validation query identities differ')
            record['models'][epoch] = {'metrics':value['evaluation']['metrics'],
                'corrected_target_loss':value['evaluation']['mean_loss_per_image'],
                'historical_target_loss':old['mean_loss_per_image'],
                'historical_metrics':old['metrics'],
                'scores':score_distribution(value['evaluation']),
                'historical_scores':score_distribution(old),
                'maximum_class_score_difference_from_historical':max(
                    abs(prior_queries[k]['score']-new_queries[k]['score']) for k in prior_queries),
                'category_differences_from_historical':sum(
                    prior_queries[k]['category_id']!=new_queries[k]['category_id'] for k in prior_queries),
                'validation_photo_reads':value['validation_photo_reads'],
                'memory':memory(args.device),'evaluation_file':filename,
                'evaluation_sha256':digest(run/filename)}
            atomic_json(run/'record.json',record,replace=True)
            print(json.dumps(record['models'][epoch]),flush=True)
        record['status']='completed'
    except Exception as error:
        record.update(status='failed',error=repr(error))
        raise
    finally:
        record['elapsed_seconds_including_verification_loading_evaluation_and_publication']=time.perf_counter()-start
        atomic_json(run/'record.json',record,replace=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
