"""D075 CPU-only durable inference bundle. No detector forwards or training resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def verify_bundle(folder):
    """Verify a replay-only soup manifest and all named files, without model loading."""
    from scripts.training_smoke import digest
    from scripts.configuration import strict_load
    folder=Path(folder);m=strict_load(folder/'soup_manifest.json')
    if (m.get('schema_version')!='bantai.stage5-soup.v1' or m.get('status')!='complete'
        or m.get('resume_supported') is not False or m.get('training_resume_supported') is not False):
        raise ValueError('Not a complete replay-only soup bundle')
    required={'model.safetensors','config.json','preprocessor_config.json','training_adapter.json'}
    if not required<=set(m['files']) or 'training_state.pt' in m['files']:
        raise ValueError('Missing replay files or misleading optimizer state')
    for name,h in m['files'].items():
        if Path(name).name!=name or digest(folder/name)!=h:raise ValueError('Soup artifact hash mismatch: '+name)
    return m


def saved_parameter_hash(model, path):
    from safetensors import safe_open
    from scripts.taco_weight_integration import canonical_saved_keys
    h=hashlib.sha256()
    with safe_open(str(path),framework='pt',device='cpu') as saved:
        inverse={n:k for k,n in canonical_saved_keys(model,saved.keys()).items()}
        for name,_ in model.named_parameters():
            value=saved.get_tensor(inverse[name]);h.update(name.encode());h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    import torch
    from transformers import GroundingDinoForObjectDetection,AutoProcessor
    from scripts.train_taco_stage5 import freeze_backbones,load_verified_stage5_checkpoint
    from scripts.taco_weight_integration import average_trainable
    from scripts.training_checkpoint import verify_checkpoint
    from scripts.training_smoke import SNAPSHOT,digest,state_hash,loading_ok
    from scripts.configuration import strict_load
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337');args=parser.parse_args()
    run=args.run_dir.resolve();target=run/'checkpoints/soup-tier1-hr-blend2575'
    if target.exists():raise FileExistsError('Existing bundle is immutable; use verify_bundle to inspect it')
    evidence=run/('d075-persistence-'+uuid.uuid4().hex);evidence.mkdir()
    temporary=target.with_name('.soup-tier1-hr-blend2575-'+uuid.uuid4().hex);temporary.mkdir()
    def guard(event,values):
        if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(values[0])).resolve();flags=values[2] if len(values)>2 else 0
            if p.name=='test.json' or p.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'):
                raise AssertionError('D075 never opens photos or test JSON')
            if isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
                if not any(p.is_relative_to(x) for x in (temporary,target,evidence)):
                    raise AssertionError('Only new soup/evidence paths may be written')
        elif event in ('os.mkdir','os.remove','os.rmdir','os.rename'):
            paths=values[:2] if event=='os.rename' else values[:1]
            for value in paths:
                if isinstance(value,(str,bytes,os.PathLike)):
                    p=Path(os.fsdecode(value)).resolve()
                    if not any(p.is_relative_to(x) for x in (temporary,target,evidence)):
                        raise AssertionError('Existing artifact mutation prohibited')
    sys.addaudithook(guard)
    def blocked(*a,**kw):raise AssertionError('D075 prohibits detector forward, CUDA and training')
    with (patch.object(GroundingDinoForObjectDetection,'forward',blocked),
          patch('torch.cuda._lazy_init',blocked),patch('torch.Tensor.backward',blocked),
          patch('torch.optim.AdamW.step',blocked)):
        torch.set_num_threads(4)
        report=strict_load(run/'report.json');evaluation=strict_load(run/'soup_tier1_hr_blend2575.json')
        proof=report['soups']['soup_tier1_hr_blend2575']
        for view in ('combined_65','legacy_47','annex_18'):
            if proof[view]!=evaluation[view]:raise ValueError('Saved evaluation/report mismatch')
        expected=proof['parameter_sha256']
        if (proof['reference_weight'],proof['tier1_weight'])!=(.2,.8):raise ValueError('Unexpected saved recipe')
        previous=ROOT/'runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/checkpoints'
        sources={'e1':previous/'epoch-01','e2':previous/'epoch-02','tier1':run/'checkpoints/epoch-01'}
        identities={}
        for key,cp in sources.items():
            context=strict_load(cp/'checkpoint_manifest.json')['context'];m=verify_checkpoint(cp,context)
            identities[key]={'path':str(cp.relative_to(ROOT)),'manifest_sha256':digest(cp/'checkpoint_manifest.json'),'files':m['files']}
        inputs={str(run/'report.json'):digest(run/'report.json'),str(run/'soup_tier1_hr_blend2575.json'):digest(run/'soup_tier1_hr_blend2575.json')}
        model,info=GroundingDinoForObjectDetection.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=False,
          use_safetensors=True,disable_custom_kernels=True,output_loading_info=True,torch_dtype=torch.float32)
        loading_ok(info);freeze_backbones(model)
        frozen=state_hash(model,True);states={}
        def copy_trainable(values):
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if p.requires_grad:p.copy_(values[name])
        for key,cp in sources.items():
            load_verified_stage5_checkpoint(model,cp)
            states[key]={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        if len(states['tier1'])!=558:raise ValueError('Trainable tensor contract changed')
        blend=average_trainable(states,{'e1':.25,'e2':.75});copy_trainable(blend)
        if state_hash(model)!='5a8b832754cfbf16fa56e4742299c61a8f60cdba6cba9d6c9758ec85ad7b3e98':raise ValueError('Reference blend mismatch')
        soup=average_trainable({'blend':blend,'tier1':states['tier1']},{'blend':.2,'tier1':.8});copy_trainable(soup)
        if state_hash(model)!=expected or state_hash(model,True)!=frozen:raise ValueError('Soup/frozen parameter SHA-256 mismatch')
        model.eval();model.save_pretrained(temporary,safe_serialization=True)
        processor=AutoProcessor.from_pretrained(sources['tier1'],local_files_only=True,trust_remote_code=False)
        processor.save_pretrained(temporary)
        shutil.copyfile(sources['tier1']/'training_adapter.json',temporary/'training_adapter.json')
        if saved_parameter_hash(model,temporary/'model.safetensors')!=expected:raise ValueError('Serialized soup parameter SHA-256 mismatch')
        manifest={'schema_version':'bantai.stage5-soup.v1','status':'complete','authorization':'D-075',
          'purpose':'private_local_inference_replay_only_not_promoted','resume_supported':False,'training_resume_supported':False,
          'weight_averaged_soup':True,'parameter_sha256':expected,'frozen_parameter_sha256':frozen,
          'recipe':{'blend2575':{'e1':.25,'e2':.75},'soup':{'blend2575':.20,'tier1_e1':.80},'arithmetic':'CPU FP32 average_trainable; nested averaging order preserved'},
          'sources':identities,'input_sha256':inputs,'parameter_hash_source':'report.json:soups.soup_tier1_hr_blend2575.parameter_sha256',
          'evaluation_binding':'soup_tier1_hr_blend2575.json metrics match report; original evaluation has no parameter-hash field',
          'saved_metrics':{v:proof[v] for v in ('combined_65','legacy_47','annex_18')},
          'precision':'FP32','trainable_tensors_averaged':558,'model_forwards':0,'GPU_execution':False,
          'files':{p.name:digest(p) for p in temporary.iterdir() if p.is_file()}}
        (temporary/'soup_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        verify_bundle(temporary)
        for key,cp in sources.items():
            context=strict_load(cp/'checkpoint_manifest.json')['context'];verify_checkpoint(cp,context)
            if digest(cp/'checkpoint_manifest.json')!=identities[key]['manifest_sha256']:raise ValueError('Source manifest changed')
        for path,h in inputs.items():
            if digest(path)!=h:raise ValueError('Saved evaluation input changed')
        if torch.cuda.is_initialized():raise AssertionError('CUDA was initialized')
        temporary.rename(target);verify_bundle(target)
        (evidence/'verification.json').write_text(json.dumps({'complete':True,'bundle':str(target),'parameter_sha256':expected,
           'serialized_parameter_hash_matches':True,'source_bundles_unchanged':True,'frozen_parameters_unchanged':True,
           'resume_supported':False,'GPU_initialized':False,'model_forwards':0,'test_json_reads':0,'photo_reads':0,
           'manifest_sha256':digest(target/'soup_manifest.json')},indent=2),encoding='utf-8')
        print(json.dumps({'bundle':str(target),'parameter_sha256':expected,'verification':str(evidence/'verification.json')}))

if __name__=='__main__':main()
