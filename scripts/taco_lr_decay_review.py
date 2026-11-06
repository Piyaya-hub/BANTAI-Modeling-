"""Saved-only D062 replay binding; never opens model or optimizer weights."""
from pathlib import Path
from .configuration import strict_load
from .training_smoke import digest
from .taco_evaluation import PROTOCOL
from .taco_lr_decay_epoch2 import SCHEDULE,position,cosine_lr
from .taco_error_analysis import saved_rows


def load_lr_decay_review(manifest_path,choice,manifest,entry,ds,report):
    from .taco_saved_review import checked_file
    audit=strict_load(checked_file(entry['lr_decay']))
    if (audit['authorization']!='D-062' or audit['status']!='completed'
            or audit['contract']!=report['contract'] or audit['protocol']!=PROTOCOL
            or audit['schedule']!=SCHEDULE or audit['contract']['continuation']['schedule']!=SCHEDULE
            or audit['new_optimizer_updates']!=221 or audit['new_backward_passes']!=221
            or audit['training_forwards']!=221 or audit['validation_forwards']!=94
            or audit['reference_GPU_forwards']!=0 or audit['test_split_access']!=0
            or audit['original_checkpoint_bundles_before']!=50 or audit['original_checkpoint_bundles_after']!=50
            or audit['new_checkpoint_bundles_after']!=5 or not audit['protected_after']['unchanged']
            or audit['core_source_bytes_before']!=audit['core_source_bytes_after']
            or not audit['restore_proof']['exact_model_optimizer_RNG']):
        raise ValueError('Saved cosine continuation audit mismatch')
    evaluations={};lrs={}
    for slot,panel in entry['panels'].items():
        anchor=panel['evaluation'];path=checked_file(anchor)
        if 'reference' in panel:
            if anchor!=audit['references'][panel['reference']]:raise ValueError('Cosine reference anchor mismatch')
            metrics=None
        else:
            step=panel['decay_step']
            if step not in (350,442):raise ValueError('Unauthorized cosine evaluation step')
            proof=audit['evaluations'][str(step)]
            if (anchor!={k:proof[k] for k in ('path','sha256')} or proof['forwards']!=47
                    or not proof['RNG_restored_exactly'] or not proof['parameters_unchanged']
                    or proof['frozen_parameter_sha256']!=audit['frozen_parameter_sha256']):
                raise ValueError('Cosine saved evaluation binding mismatch')
            name='epoch-02' if step==442 else 'step-000350'
            cp=audit['checkpoints'][name]
            saved=strict_load(checked_file({'path':str(Path(cp['path'])/'checkpoint_manifest.json'),
                                          'sha256':cp['manifest_sha256']}))
            if saved['context']!=report['contract'] or saved['position']!=position(step) or saved['status']!='complete':
                raise ValueError('Cosine checkpoint context/position mismatch')
            metrics=proof['metrics'];lrs[slot]=cosine_lr(step-222)
        evaluations[slot]=strict_load(path)
        if metrics is not None and evaluations[slot]['metrics']!=metrics:raise ValueError('Cosine metrics mismatch')
    if set(evaluations)!={'pretrained','epoch-01','epoch-02'}:raise ValueError('Require three explicit cosine panels')
    valid={i['id'] for i in ds.images};sample=manifest['images']
    if not sample or any(s['image_id'] not in valid for s in sample):raise ValueError('Non-validation review selection')
    return ds,{k:saved_rows(ds.data,v) for k,v in evaluations.items()},{
        'schema_version':'bantai.visual-error-review.v1','representative_sample':sample,'display_status':'not_attempted',
        'saved_run_choice':choice,'saved_configuration':dict(report['contract'],replay_panels=entry['panel_labels'],
            saved_prediction_panels=entry['panels'],cosine_panel_learning_rates=lrs),
        'provenance':{'manifest_path':str(Path(manifest_path).resolve()),'manifest_sha256':digest(manifest_path),
            'lr_decay':entry['lr_decay'],'panels':entry['panels'],'new_inference_calls':0,'optimizer_updates':0,'test_photo_reads':0},
        'human_visual_confirmation':'pending','annotated_image_exports':0}
