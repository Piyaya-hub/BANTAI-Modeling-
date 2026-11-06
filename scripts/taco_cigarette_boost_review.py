"""Hash-anchored D066 prediction replay; no checkpoint parameter/state loading."""
from pathlib import Path
from .configuration import strict_load
from .training_smoke import digest
from .taco_evaluation import PROTOCOL
from .taco_error_analysis import saved_rows
from .taco_cigarette_boost_epoch2 import RULE,position


def integrity_preserved(audit):
    """Allow only a hash-proven concurrent GUI pending->closed transition."""
    if audit['protected_after']['unchanged']:return True
    import copy,hashlib
    from .persistence import strict_payload
    from .taco_saved_review import checked_file
    proof=audit.get('external_display_transition',{})
    if (not audit['protected_after'].get('model_data_checkpoint_files_unchanged')
            or not audit['protected_after'].get('only_change_is_verified_external_display_transition')
            or audit['protected_after']['changed']!=[proof.get('path')]
            or proof.get('changed_fields')!={'display_status':['pending','closed'],'source_verified_photo_reads':['absent',16]}
            or not proof.get('historical_file_not_modified_by_reconciliation')
            or proof.get('extra_training_updates')!=0 or proof.get('extra_model_forwards')!=0):return False
    value=strict_load(checked_file({'path':proof['path'],'sha256':proof['after_sha256']}))
    if value.get('display_status')!='closed' or value.get('source_verified_photo_reads')!=16:return False
    prior=copy.deepcopy(value);prior['display_status']='pending';prior.pop('source_verified_photo_reads')
    actual=hashlib.sha256(strict_payload(prior).encode('utf-8')).hexdigest()
    return actual==proof['before_sha256']==proof['reconstructed_pending_sha256']


def load_cigarette_boost_review(manifest_path,choice,manifest,entry,ds,report):
    from .taco_saved_review import checked_file
    audit=strict_load(checked_file(entry['cigarette_boost']))
    if (audit['authorization']!='D-066' or audit['status']!='completed' or audit['contract']!=report['contract']
            or audit['protocol']!=PROTOCOL or audit['lr_rule']!=RULE or audit['contract']['continuation']['lr_rule']!=RULE
            or audit['new_optimizer_updates']!=92 or audit['new_backward_passes']!=92
            or audit['training_forwards']!=92 or audit['validation_forwards']!=94 or audit['reference_GPU_forwards']!=0
            or audit['test_split_access']!=0 or audit['original_checkpoint_bundles_before']!=60
            or audit['original_checkpoint_bundles_after']!=60 or audit['new_checkpoint_bundles_after']!=2
            or audit['lr_assignment_counts']!={'cigarette':11,'pure_film':25,'other':56}
            or not integrity_preserved(audit) or not audit['restore_proof']['exact_model_optimizer_RNG']
            or audit['core_source_bytes_before']!=audit['core_source_bytes_after']):
        raise ValueError('Saved cigarette-boost audit mismatch')
    evaluations={}
    for slot,panel in entry['panels'].items():
        anchor=panel['evaluation'];path=checked_file(anchor)
        if 'reference' in panel:
            if anchor!=audit['references'][panel['reference']]:raise ValueError('Class-balanced reference mismatch')
            metrics=None
        else:
            name=panel['boost_state'];proof=audit['evaluations'][name]
            if (anchor!={k:proof[k] for k in ('path','sha256')} or proof['forwards']!=47 or not proof['parameters_unchanged']):
                raise ValueError('Class-balanced evaluation binding mismatch')
            if name=='blend':
                if (proof['weights']!={'new442':.50,'constant442':.50} or not proof['frozen_parameters_identical']
                        or proof['trainable_tensors']!=558):raise ValueError('Cigarette-boost blend mismatch')
            else:
                if name!='442' or not proof['RNG_restored_exactly'] or proof['frozen_parameter_sha256']!=audit['frozen_parameter_sha256']:
                    raise ValueError('Class-balanced checkpoint validation mismatch')
            for step in (442,):
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
            'cigarette_boost':entry['cigarette_boost'],'panels':entry['panels'],'new_inference_calls':0,'optimizer_updates':0,'test_photo_reads':0},
        'human_visual_confirmation':'pending','annotated_image_exports':0}
