"""Verified saved epoch scores; explicit --gui replay without inference/training."""
if __package__ in (None,''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    __package__='scripts'

import argparse
from pathlib import Path
import uuid

from .configuration import ROOT,strict_load
from .evaluation_provenance import verify_evaluation_checkpoint
from .persistence import atomic_json
from .taco_error_analysis import MODELS,saved_rows,analyze,representative_sample
from .taco_evaluation import PROTOCOL
from .taco_training_data import CocoTrainingDataset
from .training_adapter import adapter_source_digest
from .training_smoke import digest

REPLAY_POLICY=ROOT/'config/taco_visual_review.v1.json'


def load_review(evaluation_record, baseline_record, historical_run, dataset, rights, *, split='validation'):
    if split!='validation':raise ValueError('Diagnostic replay permits validation only')
    anchors=strict_load(REPLAY_POLICY)
    if (anchors.get('schema_version')!='bantai.visual-review-anchors.v1'
            or anchors.get('purpose')!='saved_validation_replay_no_inference'):
        raise ValueError('Unsupported visual replay anchor policy')
    if (digest(evaluation_record)!=anchors['step1c_record_sha256']
            or digest(baseline_record)!=anchors['baseline_record_sha256']):
        raise ValueError('Saved evaluation record anchor mismatch')
    policy=strict_load(ROOT/'config/taco_evaluation_compatibility.v1.json')
    record=strict_load(evaluation_record);baseline=strict_load(baseline_record)
    if (record['status']!='completed' or record['policy']!=policy
            or record['policy_sha256']!=digest(ROOT/'config/taco_evaluation_compatibility.v1.json')
            or record['protocol']!=PROTOCOL or baseline['protocol']!=PROTOCOL
            or digest(baseline_record)!=policy['baseline_record_sha256']):
        raise ValueError('Saved evaluation source/protocol is not the approved corrected replay')
    proofs={name:verify_evaluation_checkpoint(Path(historical_run)/'checkpoints'/name,
        Path(historical_run)/'report.json',policy,dataset,rights) for name in MODELS[1:]}
    ds=CocoTrainingDataset(dataset,'validation',rights)
    if (ds.context!=baseline['dataset'] or len(ds)!=47 or len(ds.data['annotations'])!=75
            or baseline['implementation_sha256']!=policy['current_sources']):
        raise ValueError('Replay corpus/source identity mismatch')
    base_path=Path(baseline_record).parent/'baseline-corrected.json'
    if digest(base_path)!=anchors['baseline_evaluation_sha256']:
        raise ValueError('Pretrained prediction artifact hash mismatch')
    evaluations={'pretrained':strict_load(base_path)}
    if evaluations['pretrained']['metrics']!=baseline['corrected_baseline']['metrics']:
        raise ValueError('Pretrained metrics/evaluation identity mismatch')
    files={'pretrained':{'path':str(base_path),'sha256':digest(base_path)}}
    for name in MODELS[1:]:
        entry=record['models'][name];filename=entry['evaluation_file']
        if filename!=name+'-corrected.json':raise ValueError('Unexpected evaluation artifact path')
        path=Path(evaluation_record).parent/filename
        if digest(path)!=entry['evaluation_sha256']:raise ValueError('Epoch prediction artifact hash mismatch')
        value=strict_load(path)
        if (value['provenance']!=proofs[name] or not value['parameter_values_unchanged']
                or value['evaluation']['metrics']!=entry['metrics']):
            raise ValueError('Epoch saved/current provenance mismatch')
        evaluations[name]=value['evaluation'];files[name]={'path':str(path),'sha256':digest(path)}
    models={name:saved_rows(ds.data,evaluations[name]) for name in MODELS}
    return ds,models,{'schema_version':'bantai.visual-replay-provenance.v1',
        'step1c_record_sha256':digest(evaluation_record),'baseline_record_sha256':digest(baseline_record),
        'replay_policy_sha256':digest(REPLAY_POLICY),'saved_evaluations':files,'checkpoint_proofs':proofs,
        'dataset':ds.context,'new_inference_calls':0,'optimizer_updates':0,'test_photo_reads':0,
        'model_parameters_loaded':False,'human_visual_confirmation':'pending'}


def prepare_review(evaluation_record,baseline_record,historical_run,dataset,rights,limit=16):
    ds,models,proof=load_review(evaluation_record,baseline_record,historical_run,dataset,rights)
    analysis=analyze(ds.data,models)
    sample=representative_sample(ds.data,analysis,limit)
    folder=ROOT/'runs/training'/('step2-review-'+uuid.uuid4().hex);folder.mkdir()
    record={'schema_version':'bantai.visual-error-review.v1','provenance':proof,
        'analysis':analysis,'representative_sample':sample,
        'selection_method':'deterministic class-size and annotation-error contrasts; not a prevalence sample',
        'visual_defaults':{'threshold':.05,'top_k':5,'source':'official top100','label_font_points':5},
        'implementation_sha256':{name:adapter_source_digest(ROOT/name) for name in
            ('scripts/taco_visual_review.py','scripts/taco_error_analysis.py','scripts/taco_comparison_gui.py','scripts/gui_viewer.py')},
        'human_visual_confirmation':'pending; annotation matching is not visual review',
        'display_status':'not_attempted','annotated_image_exports':0}
    atomic_json(folder/'analysis.json',record)
    return ds,models,record,folder


def show_review(dataset,models,record,folder,image_id=None,box_top_k=None):
    """Run the existing viewer and persist its separate display outcome."""
    from .taco_comparison_gui import ComparisonWindow
    if record.get('per_object_diagnostic'):
        from .taco_object_gui import PerObjectWindow
        ComparisonWindow=PerObjectWindow
    window=None
    record['display_status']='pending'
    atomic_json(folder/'analysis.json',record,replace=True)
    try:
        if image_id is not None and image_id not in {r['id'] for r in dataset.images}:
            raise ValueError(f'Image ID {image_id} is not in the validation split')
        window=ComparisonWindow(dataset,models,record['representative_sample'],folder)
        if box_top_k is not None:
            if type(box_top_k) is not int or not 1<=box_top_k<=100:raise ValueError('Box top K must be 1..100')
            window.top_k.set(str(box_top_k))
            record['initial_box_top_k']=box_top_k
        if 'saved_run_choice' in record:
            window.set_saved_configuration(record['saved_run_choice'],record['saved_configuration'])
        if image_id is not None:
            window.scope.set('all validation')
            window.selected.set(str(image_id))
            window.set_scope()
        window.root.mainloop()
        if window.error:
            raise RuntimeError(window.error)
        record['display_status']='closed'
    except Exception as error:
        record['display_status']='error'
        record['display_error']=repr(error)
        raise
    finally:
        if window is not None:window.close()
        record['source_verified_photo_reads']=dataset.photo_reads
        atomic_json(folder/'analysis.json',record,replace=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-record',type=Path,default=ROOT/'runs/training/step1c-evaluation-3d24a67e1fe041e5a8acf56a26406db4/record.json')
    p.add_argument('--baseline-record',type=Path,default=ROOT/'runs/training/step1b-evaluation-c8b127d8bf364c7dbb735795cd81c06d/record.json')
    p.add_argument('--historical-run',type=Path,default=ROOT/'runs/training/taco-1698bcb46b85435786dd2a199910910f')
    p.add_argument('--dataset',type=Path,default=ROOT/'exports/taco/strict-clean-stage2-v1')
    p.add_argument('--rights',type=Path,default=ROOT/'work/stage4-real/rights.json')
    p.add_argument('--split',choices=['validation'],default='validation')
    p.add_argument('--sample-size',type=int,default=16)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--analyze-only',action='store_true')
    mode.add_argument('--gui',action='store_true',help='Open the existing four-panel saved validation viewer')
    p.add_argument('--image-id',type=int,help='Start GUI at this validation image ID (requires --gui)')
    p.add_argument('--saved-review',type=Path,help='Explicit cross-seed saved-review manifest; leaves historical defaults unchanged')
    p.add_argument('--run-choice',help='Run key from --saved-review (seed and learning rate)')
    p.add_argument('--per-object',action='store_true',help='Explicit saved-review GUI: inspect each GT using all eligible saved queries')
    p.add_argument('--box-top-k',type=int,help='Initial GUI box cap, 1..100; diagnostic drawing only')
    args=p.parse_args(argv)
    if args.per_object and not (args.gui and args.saved_review):
        p.error('--per-object requires --gui and --saved-review')
    if bool(args.saved_review) != bool(args.run_choice):
        p.error('--saved-review and --run-choice must be used together')
    if args.image_id is not None and not args.gui:
        p.error('--image-id requires --gui')
    if args.box_top_k is not None and (not args.gui or not 1<=args.box_top_k<=100):
        p.error('--box-top-k requires --gui and a value in 1..100')
    if args.saved_review:
        from .taco_saved_review import load_saved_review
        dataset,models,record=load_saved_review(args.saved_review,args.run_choice,args.dataset,args.rights)
        if args.per_object:record['per_object_diagnostic']=True
        folder=ROOT/'runs/training'/('saved-review-'+uuid.uuid4().hex);folder.mkdir()
        atomic_json(folder/'analysis.json',record)
        print(folder,flush=True)
        print('Verified saved run: '+args.run_choice+'; no new inference or training.',flush=True)
        if args.gui:
            if args.box_top_k is None:show_review(dataset,models,record,folder,args.image_id)
            else:show_review(dataset,models,record,folder,args.image_id,box_top_k=args.box_top_k)
    elif args.analyze_only or args.gui:
        dataset,models,record,folder=prepare_review(args.evaluation_record,args.baseline_record,args.historical_run,
            args.dataset,args.rights,args.sample_size)
        print(folder,flush=True)
        if args.gui:
            if args.box_top_k is None:show_review(dataset,models,record,folder,args.image_id)
            else:show_review(dataset,models,record,folder,args.image_id,box_top_k=args.box_top_k)
    else:
        load_review(args.evaluation_record,args.baseline_record,args.historical_run,args.dataset,args.rights)
        print(epoch_score_text(strict_load(args.evaluation_record)),flush=True)
    return 0


def epoch_score_text(record):
    """Separate ranked detection metrics, model scores and corrected-target losses."""
    lines=['Saved validation scores (47 images / 75 annotations; no new training).',
           'AP and model scores below are fractions, not accuracy.']
    states=[('Pretrained baseline',record['pretrained_corrected']['metrics'],
        record['pretrained_corrected']['mean_loss_per_image'],record['pretrained_corrected']['scores'])]
    states += [(name,value['metrics'],value['corrected_target_loss'],value['scores'])
               for name,value in record['models'].items()]
    for name,metrics,loss,scores in states:
        quantiles=scores['quantiles']
        lines += ['',name,
            '  AP={AP:.6f}  AP50={AP50:.6f}  AP75={AP75:.6f}  AR100={AR100:.6f}'.format(**metrics),
            f'  Validation loss (corrected targets): {loss:.6f}',
            f"  Model scores: median={quantiles['median']:.6f}  P99={quantiles['p99']:.6f}  max={quantiles['max']:.6f}",
            f"  Queries scoring >=0.25: {scores['queries_ge_025']} / {scores['queries']}"]
    return '\n'.join(lines)


if __name__=='__main__':raise SystemExit(main())
