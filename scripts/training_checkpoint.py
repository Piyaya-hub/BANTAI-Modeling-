"""Complete, immutable Stage 4 checkpoints; commit manifest is published last."""
import random
from pathlib import Path

from .configuration import strict_load
from .persistence import atomic_json
from .training_smoke import digest


def rng_state():
    import numpy as np
    import torch
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed on resume')
        torch.cuda.set_rng_state_all(state['cuda'])


def epoch_batches(length, batch_size, seed, epoch):
    indexes = list(range(length))
    random.Random(seed+epoch).shuffle(indexes)
    return [indexes[i:i+batch_size] for i in range(0, length, batch_size)]


def save_checkpoint(folder, model, processor, adapter, optimizer, context, position, evidence=None):
    import torch
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(folder, safe_serialization=True)
    processor.save_pretrained(folder)
    atomic_json(folder/'training_adapter.json', adapter.manifest(model))
    if evidence is not None:
        atomic_json(folder/'experiment_evidence.json', evidence)
    # Only tensors and Python primitives; weights_only=True loading is required.
    state = {'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'position': position,
             'trainable_names': [n for n,p in model.named_parameters() if p.requires_grad]}
    with (folder/'training_state.pt').open('wb') as out:
        torch.save(state, out)
        out.flush()
        import os
        os.fsync(out.fileno())
    files = {p.name: digest(p) for p in folder.iterdir() if p.is_file()}
    manifest = {'schema_version': 'bantai.training-checkpoint.v1', 'status': 'complete',
                'training_resume_supported': True, 'context': context, 'position': position, 'files': files}
    atomic_json(folder/'checkpoint_manifest.json', manifest)
    return folder


def verify_checkpoint(folder, context):
    folder = Path(folder)
    manifest = strict_load(folder/'checkpoint_manifest.json')
    if (manifest.get('schema_version') != 'bantai.training-checkpoint.v1'
            or manifest.get('status') != 'complete' or manifest.get('context') != context):
        raise ValueError('Incomplete checkpoint or changed dataset/rights/settings/runtime/source')
    required = {'model.safetensors','config.json','preprocessor_config.json','training_state.pt','training_adapter.json'}
    if not required.issubset(manifest['files']):
        raise ValueError('Checkpoint required files missing')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or digest(folder/name) != expected:
            raise ValueError('Checkpoint artifact hash mismatch: '+name)
    return manifest


def restore_training(folder, model, processor, adapter, optimizer, context):
    """Model/processor must already be loaded from this verified HF folder."""
    import torch
    manifest = verify_checkpoint(folder, context)
    adapter.verify_manifest(model, strict_load(Path(folder)/'training_adapter.json'))
    state = torch.load(Path(folder)/'training_state.pt', map_location='cpu', weights_only=True)
    if (state['trainable_names'] != [n for n,p in model.named_parameters() if p.requires_grad]
            or state['position'] != manifest['position']):
        raise ValueError('Resume position or trainable parameter order changed')
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    return state['position']
