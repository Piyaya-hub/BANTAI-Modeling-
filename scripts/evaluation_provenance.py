"""Historical artifact verification plus explicit evaluation compatibility.

Never returns a resume contract or deserializes training_state.pt. The unchanged
training verifier is used ONLY with the independently anchored historical context.
"""
import copy
from pathlib import Path

from .configuration import ROOT, strict_load
from .training_adapter import adapter_source_digest
from .training_checkpoint import verify_checkpoint
from .training_smoke import digest
from .taco_training_data import dataset_context, require_rights

HISTORICAL_SOURCES = ('scripts/train_taco.py', 'scripts/taco_training_data.py',
    'scripts/taco_evaluation.py', 'scripts/training_checkpoint.py',
    'scripts/training_adapter.py', 'scripts/training_smoke.py')
REVIEWED_DIFFERENCES = {'scripts/taco_training_data.py',
    'scripts/training_smoke.py', 'scripts/taco_evaluation.py'}
SUPPORTING_SOURCES = ('scripts/evaluation_provenance.py', 'scripts/reevaluate_taco.py',
    'scripts/configuration.py', 'scripts/persistence.py', 'scripts/taco_dataset.py')


def verify_sources(policy, historical, root=ROOT):
    """Accept exact reviewed old/new pairs, never a filename-only allowlist."""
    if (policy.get('schema_version') != 'bantai.evaluation-compatibility.v1'
            or policy.get('purpose') != 'validation_only_no_training'
            or policy.get('historical_sources') != historical
            or set(historical) != set(HISTORICAL_SOURCES)
            or set(policy.get('current_sources', {})) != set(HISTORICAL_SOURCES)
            or set(policy.get('supporting_sources', {})) != set(SUPPORTING_SOURCES)):
        raise ValueError('Invalid evaluation source policy or historical identity')
    current = {name: adapter_source_digest(Path(root)/name) for name in HISTORICAL_SOURCES}
    support = {name: adapter_source_digest(Path(root)/name) for name in SUPPORTING_SOURCES}
    if current != policy['current_sources'] or support != policy['supporting_sources']:
        raise ValueError('Unexpected current evaluation implementation change')
    changed = {name for name in historical if historical[name] != current[name]}
    reviewed = policy.get('approved_differences', {})
    if changed != set(reviewed) or not changed.issubset(REVIEWED_DIFFERENCES):
        raise ValueError('Unapproved historical/current source differences')
    for name, entry in reviewed.items():
        if (entry.get('historical_sha256') != historical[name]
                or entry.get('evaluation_sha256') != current[name]
                or not isinstance(entry.get('reason'), str) or not entry['reason'].strip()):
            raise ValueError('Evaluation difference lacks exact reviewed provenance')
    return {'historical_sources': copy.deepcopy(historical), 'current_sources': current,
            'supporting_sources': support, 'approved_differences': copy.deepcopy(reviewed)}


def verify_evaluation_checkpoint(folder, report_path, policy, dataset, rights_path,
                                 *, split='validation', source_root=ROOT):
    """Fail closed before any model/photo loading. Hash training state, never load it."""
    if split != 'validation':
        raise ValueError('Only validation split is authorized; train/test access rejected')
    folder, report_path = Path(folder).resolve(), Path(report_path).resolve()
    if digest(report_path) != policy.get('historical_report_sha256'):
        raise ValueError('Historical report anchor mismatch')
    report = strict_load(report_path)
    manifest_path = folder/'checkpoint_manifest.json'
    if digest(manifest_path) != policy.get('checkpoint_manifests', {}).get(folder.name):
        raise ValueError('Historical checkpoint manifest anchor mismatch')
    original = report['contract']
    if report.get('implementation_sha256') != original['implementation_sha256']:
        raise ValueError('Historical report source identities disagree')
    provenance = verify_sources(policy, original['implementation_sha256'], source_root)
    # This is original-context artifact verification, NOT permission to resume
    # under a different implementation. No training verifier semantics are changed.
    manifest = verify_checkpoint(folder, original)
    expected_files = set(manifest['files']) | {'checkpoint_manifest.json'}
    if {p.name for p in folder.iterdir()} != expected_files:
        raise ValueError('Unmanifested checkpoint artifacts or missing files')
    for name in expected_files:
        p = folder/name
        if not p.is_file() or p.is_symlink() or p.resolve().parent != folder:
            raise ValueError('Unsafe checkpoint artifact path')
    required = {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'}
    if not required.issubset(manifest['files']):
        raise ValueError('Checkpoint tokenizer artifacts missing')
    actual_dataset = dataset_context(dataset)
    if actual_dataset != original['dataset']:
        raise ValueError('Evaluation dataset identity differs from historical corpus')
    rights = require_rights(rights_path, actual_dataset)
    if (rights['record'] != report['rights']['record']
            or rights['record_sha256'] != original['rights_sha256']):
        raise ValueError('Rights record differs from original authorized scope')
    return {'schema_version': 'bantai.evaluation-provenance.v1',
        'purpose': 'validation_only_no_training', 'training_resume_authorized': False,
        'historical_report_sha256': digest(report_path),
        'checkpoint_manifest_sha256': digest(manifest_path),
        'checkpoint_artifacts_sha256': copy.deepcopy(manifest['files']),
        'historical_context': copy.deepcopy(original), 'sources': provenance,
        'dataset': actual_dataset, 'rights_file_sha256': digest(rights_path),
        'historical_rights_gate_sha256': rights['record_sha256'],
        'training_state_deserialized': False, 'optimizer_restored': False,
        'rng_restored': False, 'training_position_restored': False,
        'optimizer_updates': 0, 'split': split}
