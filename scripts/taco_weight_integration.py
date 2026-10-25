"""CPU FP32 weight arithmetic only; no model execution or persistence."""
import math

RECIPES = {
    'blend_350_442_5050': {350: .5, 442: .5},
    'blend_350_442_2575': {350: .25, 442: .75},
    'blend_221_442_5050': {221: .5, 442: .5},
    'soup_221_350_442': {221: 1/3, 350: 1/3, 442: 1/3},
}


def canonical_saved_keys(model, saved_keys):
    """Resolve HF safetensors' saved aliases without breaking tied parameters."""
    persisted=set(model.state_dict())
    canonical={id(t):n for n,t in list(model.named_parameters())+list(model.named_buffers()) if n in persisted}
    aliases={n:canonical[id(t)] for n,t in list(model.named_parameters(remove_duplicate=False))
             +list(model.named_buffers(remove_duplicate=False)) if n in persisted}
    if any(n not in aliases for n in saved_keys):raise ValueError('Unknown saved state key')
    mapping={n:aliases[n] for n in saved_keys}
    if len(mapping)!=len(canonical) or set(mapping.values())!=set(canonical.values()):
        raise ValueError('Missing/duplicate canonical saved state keys')
    return mapping


def bitwise_equal(a, b):
    import torch
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def average_trainable(states, weights):
    """Return fresh tensors; frozen parameters/buffers must never enter states."""
    import torch
    if (not weights or any(not math.isfinite(w) or w <= 0 for w in weights.values())
            or not math.isclose(sum(weights.values()), 1., abs_tol=1e-12)):
        raise ValueError('Require positive weights summing to one')
    selected=[states[step] for step in weights]
    names=set(selected[0])
    if not names or any(set(s) != names for s in selected):
        raise ValueError('Trainable state keys differ')
    result={}
    for name in sorted(names):
        tensors=[s[name] for s in selected]
        if any(t.dtype != torch.float32 or t.device.type != 'cpu'
               or t.shape != tensors[0].shape or not torch.isfinite(t).all() for t in tensors):
            raise ValueError('Require finite matching CPU FP32 tensors')
        value=torch.zeros_like(tensors[0])
        for tensor,weight in zip(tensors,weights.values()):
            value.add_(tensor, alpha=weight)
        if not torch.isfinite(value).all():raise ValueError('Nonfinite average')
        result[name]=value
    return result
