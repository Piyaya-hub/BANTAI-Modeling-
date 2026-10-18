"""Deterministic photograph fingerprints and multi-label group assignment."""
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random

from PIL import Image

SPLITS = ('train', 'validation', 'test')


def fingerprint(row):
    payload = Path(row['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != row['sha256']:
        raise ValueError(f"Source image changed since audit: {row['path']}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.size != (row['actual_width'], row['actual_height']):
            raise ValueError('Source dimensions changed since audit')
        with image.convert('RGBA') as rgba:
            digest = hashlib.sha256(json.dumps(image.size).encode()+rgba.tobytes()).hexdigest()
        with image.convert('L') as gray:
            with gray.resize((9, 8), Image.Resampling.LANCZOS) as small:
                pixels = list(small.getdata())
                dhash = sum((pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
            with gray.resize((8, 8), Image.Resampling.LANCZOS) as small:
                pixels = list(small.getdata()); mean = sum(pixels)/64
                ahash = sum((v > mean) << i for i, v in enumerate(pixels))
    return {'file_sha256': row['sha256'], 'pixel_sha256': digest,
            'dhash64': f'{dhash:016x}', 'ahash64': f'{ahash:016x}',
            'aspect_ratio': row['actual_width']/row['actual_height']}


class Groups:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, ident):
        while self.parent[ident] != ident:
            self.parent[ident] = self.parent[self.parent[ident]]
            ident = self.parent[ident]
        return ident

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def members(self):
        groups = defaultdict(list)
        for ident in sorted(self.parent): groups[self.find(ident)].append(ident)
        return list(groups.values())


def duplicate_groups(fingerprints, policy):
    exact = Groups(fingerprints)
    by_file, by_pixel = {}, {}
    for ident, row in sorted(fingerprints.items()):
        for index, key in ((by_file, 'file_sha256'), (by_pixel, 'pixel_sha256')):
            value = row[key]
            if value in index: exact.join(ident, index[value])
            else: index[value] = ident
    candidates = []
    rows = sorted(fingerprints.items())
    for position, (a, x) in enumerate(rows):
        for b, y in rows[position+1:]:
            if exact.find(a) == exact.find(b): continue
            aspect = abs(x['aspect_ratio']-y['aspect_ratio'])/max(x['aspect_ratio'], y['aspect_ratio'])
            if aspect > policy['max_relative_aspect_difference']: continue
            dh = (int(x['dhash64'], 16) ^ int(y['dhash64'], 16)).bit_count()
            if dh > policy['max_dhash_distance']: continue
            ah = (int(x['ahash64'], 16) ^ int(y['ahash64'], 16)).bit_count()
            if ah <= policy['max_ahash_distance']:
                candidates.append({'image_id_a': a, 'image_id_b': b, 'dhash_distance': dh, 'ahash_distance': ah,
                                   'relative_aspect_difference': aspect, 'identity_confirmed': False})
    return [g for g in exact.members() if len(g) > 1], candidates


def allocate(total, ratios, minimum=False):
    """Largest-remainder image targets; optional one per nonzero evaluation split."""
    expected = {s: total*ratios[s] for s in SPLITS}
    counts = {s: int(expected[s]) for s in SPLITS}
    remainder = total-sum(counts.values())
    order = sorted(SPLITS, key=lambda s: (-(expected[s]-int(expected[s])), SPLITS.index(s)))
    for s in order[:remainder]: counts[s] += 1
    if minimum and total >= len(SPLITS):
        for s in SPLITS:
            if not counts[s]:
                donor = max(SPLITS, key=lambda name: counts[name])
                counts[donor] -= 1; counts[s] += 1
    return counts


def assign_splits(image_labels, groups, ratios, seed):
    """Rare-label-first heuristic with image capacity targets and indivisible groups.

    Class targets count images, not boxes. Small-class minimum targets are goals,
    not a guarantee: grouping and multi-label conflicts may make them infeasible.
    """
    rng = random.Random(seed)
    groups = sorted([sorted(g) for g in groups], key=lambda g: g[0])
    targets = allocate(len(image_labels), ratios)
    total = Counter(label for labels in image_labels.values() for label in labels)
    label_groups = Counter(label for group in groups for label in set().union(*(image_labels[i] for i in group)))
    desired = {label: allocate(n, ratios, minimum=label_groups[label] >= 3) for label, n in total.items()}
    pending = [{'ids': g, 'counts': Counter(label for i in g for label in image_labels[i]), 'tie': rng.random()} for g in groups]
    actual = dict.fromkeys(SPLITS, 0); counts = {s: Counter() for s in SPLITS}; assignments = {}
    while pending:
        remaining = Counter()
        for group in pending: remaining.update(group['counts'])
        focus = min(remaining, key=lambda label: (remaining[label], label))
        chosen = min((g for g in pending if focus in g['counts']),
                     key=lambda g: (-len(g['counts']), -len(g['ids']), g['tie']))
        feasible = [s for s in SPLITS if actual[s]+len(chosen['ids']) <= targets[s]] or list(SPLITS)
        def score(s):
            return (desired[focus][s]-counts[s][focus],
                    sum((desired[label][s]-counts[s][label])/max(1, desired[label][s]) for label in sorted(chosen['counts'])),
                    targets[s]-actual[s])
        best = max(score(s) for s in feasible)
        split = rng.choice([s for s in feasible if score(s) == best])
        for ident in chosen['ids']: assignments[ident] = split
        actual[split] += len(chosen['ids']); counts[split].update(chosen['counts'])
        pending.remove(chosen)
    return assignments, {'target_images': targets, 'actual_images': actual, 'class_image_targets': desired}
