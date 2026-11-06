"""Hash-anchored D064 prediction replay; no checkpoint parameter/state loading."""
from pathlib import Path
from .configuration import strict_load
from .training_smoke import digest
from .taco_evaluation import PROTOCOL
from .taco_error_analysis import saved_rows
from .taco_class_balanced_epoch2 import RULE,position


def load_class_balanced_review(manifest_path,choice,manifest,entry,ds,report):
    from .taco_saved_review import checked_file
    audit=strict_load(checked_file(entry['class_balanced']))
    if (audit['authorization']!='D-064' or audit['status']!='completed' or audit['contract']!=report['contract']
            or audit['protocol']!=PROTOCOL or audit['lr_rule']!=RULE or audit['contract']['continuation']['lr_rule']!=RULE
            or audit['new_optimizer_updates']!=221 or audit['new_backward_passes']!=221
            or audit['training_forwards']!=221 or audit['validation_forwards']!=141 or audit['reference_GPU_forwards']!=0
            or audit['test_split_access']!=0 or audit['original_checkpoint_bundles_before']!=55
            or audit['original_checkpoint_bundles_after']!=55 or audit['new_checkpoint_bundles_after']!=5
            or audit['lr_assignment_counts']!={'pure_plastic_film':61,'other':160,'cigarette_full_lr':30}
            or not audit['protected_after']['unchanged'] or not audit['restore_proof']['exact_model_optimizer_RNG']
            or audit['core_source_bytes_before']!=audit['core_source_bytes_after']):
        raise ValueError('Saved class-balanced audit mismatch')
    evaluations={}
    for slot,panel in entry['panels'].items():
        anchor=panel['evaluation'];path=checked_file(anchor)
        if 'reference' in panel:
            if anchor!=audit['references'][panel['reference']]:raise ValueError('Class-balanced reference mismatch')
            metrics=None
        else:
            name=panel['cb_state'];proof=audit['evaluations'][name]
            if (anchor!={k:proof[k] for k in ('path','sha256')} or proof['forwards']!=47 or not proof['parameters_unchanged']):
                raise ValueError('Class-balanced evaluation binding mismatch')
            if name=='blend_2575':
                if (proof['weights']!={'350':.25,'442':.75} or not proof['frozen_parameters_identical']
                        or proof['trainable_tensors']!=558):raise ValueError('Class-balanced blend mismatch')
            else:
                if name not in ('350','442') or not proof['RNG_restored_exactly'] or proof['frozen_parameter_sha256']!=audit['frozen_parameter_sha256']:
                    raise ValueError('Class-balanced checkpoint validation mismatch')
            for step in ((350,442) if name=='blend_2575' else (int(name),)):
                cp=audit['checkpoints']['epoch-02' if step==442 else 'step-000350']
                saved=strict_load(checked_file({'path':str(Path(cp['path'])/'checkpoint_manifest.json'),'sha256':cp['manifest_sha256']}))
                if saved['status']!='complete' or saved['context']!=report['contract'] or saved['position']!=position(step):
                    raise ValueError('Class-balanced checkpoint context/position mismatch')
            metrics=proof['metrics']
        evaluations[slot]=strict_load(path)
        if metrics is not None and evaluations[slot]['metrics']!=metrics:raise ValueError('Class-balanced metric mismatch')
    if set(evaluations)!={'pretrained','epoch-01','epoch-02'}:raise ValueError('Require three explicit CB replay panels')
    sample=manifest['images'];valid={v['id'] for v in ds.images}
    if not sample or any(v['image_id'] not in valid for v in sample):raise ValueError('Non-validation review selection')
    return ds,{k:saved_rows(ds.data,v) for k,v in evaluations.items()},{
        'schema_version':'bantai.visual-error-review.v1','representative_sample':sample,'display_status':'not_attempted',
        'saved_run_choice':choice,'saved_configuration':dict(report['contract'],replay_panels=entry['panel_labels'],saved_prediction_panels=entry['panels']),
        'provenance':{'manifest_path':str(Path(manifest_path).resolve()),'manifest_sha256':digest(manifest_path),
            'class_balanced':entry['class_balanced'],'panels':entry['panels'],'new_inference_calls':0,'optimizer_updates':0,'test_photo_reads':0},
        'human_visual_confirmation':'pending','annotated_image_exports':0}
