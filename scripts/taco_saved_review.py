"""Explicit, locally hash-anchored replay of completed corrected training runs."""
from pathlib import Path

from .configuration import ROOT, strict_load
from .taco_error_analysis import saved_rows
from .taco_evaluation import PROTOCOL
from .taco_training_data import CocoTrainingDataset
from .training_checkpoint import verify_checkpoint
from .training_smoke import digest


def checked_file(entry):
    path = Path(entry['path']).resolve()
    if not path.is_relative_to(ROOT.resolve()) or digest(path) != entry['sha256']:
        raise ValueError('Saved review artifact path/hash mismatch')
    return path


def load_integrated_review(manifest_path, choice, manifest, entry, ds, report):
    """Verify saved integrated predictions; never reconstruct or load weights."""
    from .taco_weight_integration import RECIPES
    audit_path=checked_file(entry['integration'])
    audit=strict_load(audit_path)
    if (audit['authorization']!='D-061' or audit['status']!='completed'
            or audit['contract']!=report['contract'] or audit['protocol']!=PROTOCOL
            or audit['new_model_forwards']!=188 or audit['optimizer_updates']!=0
            or audit['backward_passes']!=0 or audit['test_split_access']!=0
            or audit['training_photo_reads']!=0 or audit['reference_GPU_forwards']!=0
            or audit['checkpoint_bundles_before']!=50 or audit['checkpoint_bundles_after']!=50
            or audit['weight_checks']['trainable_parameter_tensors']!=558
            or not audit['weight_checks']['all_frozen_parameters_and_buffers_bitwise_identical']
            or not audit['protected_after']['unchanged']):
        raise ValueError('Saved integration audit/contract mismatch')
    if set(audit['states'])!=set(RECIPES):raise ValueError('Integration recipe set mismatch')
    evaluations={}
    for slot,panel in entry['panels'].items():
        anchor=panel['evaluation'];path=checked_file(anchor)
        if 'reference_step' in panel:
            proof=audit['references'][str(panel['reference_step'])]
            if anchor!=proof:raise ValueError('Saved integration reference anchor mismatch')
            metrics=None
        else:
            name=panel['integration_state'];proof=audit['states'][name]
            if (proof['weights']!={str(k):v for k,v in RECIPES[name].items()}
                    or proof['evaluation_sha256']!=anchor['sha256']
                    or path!=audit_path.parent/proof['evaluation_file']
                    or proof['forwards']!=47 or not proof['unchanged_after_evaluation']):
                raise ValueError('Saved integrated state binding mismatch')
            metrics=proof['metrics']
        evaluations[slot]=strict_load(path)
        if metrics is not None and evaluations[slot]['metrics']!=metrics:
            raise ValueError('Saved integrated evaluation metrics mismatch')
    if set(evaluations)!={'pretrained','epoch-01','epoch-02'}:
        raise ValueError('Require three explicit integration replay panels')
    valid={i['id'] for i in ds.images};sample=manifest['images']
    if not sample or any(s['image_id'] not in valid for s in sample):
        raise ValueError('Review selection contains non-validation images')
    return ds,{k:saved_rows(ds.data,v) for k,v in evaluations.items()},{
        'schema_version':'bantai.visual-error-review.v1','representative_sample':sample,
        'display_status':'not_attempted','saved_run_choice':choice,
        'saved_configuration':dict(report['contract'],replay_panels=entry['panel_labels'],
            saved_prediction_panels=entry['panels'],
            weight_integration={p['integration_state']:audit['states'][p['integration_state']]['weights']
                                for p in entry['panels'].values() if 'integration_state' in p}),
        'provenance':{'manifest_path':str(Path(manifest_path).resolve()),'manifest_sha256':digest(manifest_path),
            'integration':entry['integration'],'panels':entry['panels'],
            'new_inference_calls':0,'optimizer_updates':0,'test_photo_reads':0},
        'human_visual_confirmation':'pending','annotated_image_exports':0}


def load_saved_review(manifest_path, choice, dataset, rights):
    manifest = strict_load(manifest_path)
    if manifest.get('schema_version') != 'bantai.cross-seed-review.v1':
        raise ValueError('Unsupported saved review manifest')
    entry = manifest['runs'][choice]
    for anchor in entry.get('supporting_records', []):
        checked_file(anchor)
    ds = CocoTrainingDataset(dataset, 'validation', rights)
    if ds.context != manifest['dataset'] or len(ds) != 47 or len(ds.data['annotations']) != 75:
        raise ValueError('Saved review corpus mismatch')
    report_path = checked_file(entry['report'])
    report = strict_load(report_path)
    if (report['protocol'] != PROTOCOL or report['contract']['dataset'] != ds.context
            or report['contract']['rights_sha256'] != ds.rights['record_sha256']
            or report['contract']['seed'] != entry['seed']
            or report['contract']['optimizer']['lr'] != entry['learning_rate']):
        raise ValueError('Saved review protocol/configuration mismatch')
    if 'integration' in entry:
        return load_integrated_review(manifest_path,choice,manifest,entry,ds,report)
    if 'lr_decay' in entry:
        from .taco_lr_decay_review import load_lr_decay_review
        return load_lr_decay_review(manifest_path,choice,manifest,entry,ds,report)
    if 'score_calibration' in entry:
        from .taco_score_calibration_review import load_score_calibration_review
        return load_score_calibration_review(manifest_path,choice,manifest,entry,ds,report)
    if 'cigarette_boost' in entry:
        from .taco_cigarette_boost_review import load_cigarette_boost_review
        return load_cigarette_boost_review(manifest_path,choice,manifest,entry,ds,report)
    if 'class_balanced' in entry:
        from .taco_class_balanced_review import load_class_balanced_review
        return load_class_balanced_review(manifest_path,choice,manifest,entry,ds,report)
    if 'vision_unfreeze' in entry:
        from .taco_vision_unfreeze_review import load_vision_unfreeze_review
        return load_vision_unfreeze_review(manifest_path,choice,manifest,entry,ds,report)
    baseline = entry['baseline'] if 'baseline' in entry else manifest['baseline']
    evaluations = {'pretrained': strict_load(checked_file(baseline))}
    run_baseline = strict_load(checked_file(entry['run_baseline']) if 'run_baseline' in entry
                               else report_path.parent/'baseline.json')
    if any(run_baseline[k] != evaluations['pretrained'][k] for k in ('metrics','predictions','raw_queries')):
        raise ValueError('Saved review pretrained baseline mismatch')
    for epoch in (1, 2):
        key = f'epoch-{epoch:02d}'
        anchor = entry['checkpoints'][key]
        cp = checked_file(anchor).parent
        if entry.get('json_only'):
            # Replay verifies saved JSON identity, never opens parameter/optimizer files.
            proof = strict_load(checked_file(anchor))
            if (proof['status'] != 'complete' or proof['context'] != report['contract']):
                raise ValueError('Saved review checkpoint context mismatch')
        else:
            proof = verify_checkpoint(cp, report['contract'])
        expected = entry.get('positions', {}).get(key,
            {'epoch': epoch, 'next_batch': 0, 'global_step': epoch * 221})
        if proof['position'] != expected:
            raise ValueError('Saved review checkpoint position mismatch')
        evaluations[key] = strict_load(checked_file(entry['evaluations'][key]))
        metrics = report['epochs'][epoch-1]['validation']
        if key in entry.get('trajectory', {}):
            audit = strict_load(checked_file(entry['trajectory'][key]))
            step = str(expected['global_step'])
            if (audit['status'] != 'completed' or audit['contract'] != report['contract']
                    or audit['checkpoints'][step]['position'] != expected):
                raise ValueError('Saved review trajectory binding mismatch')
            metrics = audit['checkpoints'][step]['metrics']
        if evaluations[key]['metrics'] != metrics:
            raise ValueError('Saved review evaluation/report metric mismatch')
    models = {name: saved_rows(ds.data, value) for name, value in evaluations.items()}
    sample = manifest['images']
    valid_ids = {i['id'] for i in ds.images}
    if not sample or any(s['image_id'] not in valid_ids for s in sample):
        raise ValueError('Review selection contains non-validation images')
    return ds, models, {
        'schema_version': 'bantai.visual-error-review.v1',
        'representative_sample': sample, 'display_status': 'not_attempted',
        'saved_run_choice': choice, 'saved_configuration': dict(report['contract'],
            **({'replay_panels': entry['panel_labels']} if 'panel_labels' in entry else {})),
        'provenance': {'manifest_path': str(Path(manifest_path).resolve()),
                       'manifest_sha256': digest(manifest_path), 'run': str(report_path.parent),
                       'saved_evaluations': entry['evaluations'],
                       'checkpoint_manifests': entry['checkpoints'],
                       'new_inference_calls': 0, 'optimizer_updates': 0, 'test_photo_reads': 0},
        'human_visual_confirmation': 'pending', 'annotated_image_exports': 0}
