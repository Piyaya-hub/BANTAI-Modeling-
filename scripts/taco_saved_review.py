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


def load_saved_review(manifest_path, choice, dataset, rights):
    manifest = strict_load(manifest_path)
    if manifest.get('schema_version') != 'bantai.cross-seed-review.v1':
        raise ValueError('Unsupported saved review manifest')
    entry = manifest['runs'][choice]
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
    evaluations = {'pretrained': strict_load(checked_file(manifest['baseline']))}
    run_baseline = strict_load(report_path.parent/'baseline.json')
    if any(run_baseline[k] != evaluations['pretrained'][k] for k in ('metrics','predictions','raw_queries')):
        raise ValueError('Saved review pretrained baseline mismatch')
    for epoch in (1, 2):
        key = f'epoch-{epoch:02d}'
        anchor = entry['checkpoints'][key]
        cp = checked_file(anchor).parent
        proof = verify_checkpoint(cp, report['contract'])
        if proof['position'] != {'epoch': epoch, 'next_batch': 0, 'global_step': epoch * 221}:
            raise ValueError('Saved review checkpoint position mismatch')
        evaluations[key] = strict_load(checked_file(entry['evaluations'][key]))
        if evaluations[key]['metrics'] != report['epochs'][epoch-1]['validation']:
            raise ValueError('Saved review evaluation/report metric mismatch')
    models = {name: saved_rows(ds.data, value) for name, value in evaluations.items()}
    sample = manifest['images']
    valid_ids = {i['id'] for i in ds.images}
    if not sample or any(s['image_id'] not in valid_ids for s in sample):
        raise ValueError('Review selection contains non-validation images')
    return ds, models, {
        'schema_version': 'bantai.visual-error-review.v1',
        'representative_sample': sample, 'display_status': 'not_attempted',
        'saved_run_choice': choice, 'saved_configuration': report['contract'],
        'provenance': {'manifest_path': str(Path(manifest_path).resolve()),
                       'manifest_sha256': digest(manifest_path), 'run': str(report_path.parent),
                       'new_inference_calls': 0, 'optimizer_updates': 0, 'test_photo_reads': 0},
        'human_visual_confirmation': 'pending', 'annotated_image_exports': 0}
