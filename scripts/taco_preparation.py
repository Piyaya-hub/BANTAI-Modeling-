"""Strict-clean COCO conversion: whole-image quarantine and source traceability."""
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path

from .configuration import ROOT, finite_number, resolve_path, strict_load
from .taco_dataset import audit_dataset, bbox_issues, load_report, sha256, valid_id
from .taco_splits import SPLITS, Groups, assign_splits, duplicate_groups, fingerprint

DEFAULT_POLICY = ROOT/'config/taco_preparation_policy.v1.json'
RULES = ['missing_or_unreadable', 'invalid_image_metadata', 'dimension_mismatch', 'ambiguous_exif_orientation', 'invalid_annotation',
         'crowd_or_ignore_annotation', 'unlabeled_annotation', 'excluded_annotation',
         'no_mapped_annotations', 'exact_duplicate_conflict', 'exact_duplicate_quarantine']


def load_policy(path=DEFAULT_POLICY):
    policy = strict_load(path)
    required = {'schema_version': 'bantai.taco-eligibility.v1', 'subset': 'strict_clean',
                'coordinate_frame': 'stored_raster_without_exif_transpose', 'box_repair': 'none',
                'allowed_exif_orientations': [1],
                'duplicate_annotation_id_repair': 'none', 'empty_images': 'quarantine',
                'exact_duplicates': 'coassign_and_propagate_quarantine',
                'split_algorithm': 'rare_label_first_group_greedy_v1'}
    if not isinstance(policy, dict) or any(policy.get(k) != v for k, v in required.items()):
        raise ValueError('Unsupported eligibility policy; strict-clean safeguards cannot be relaxed')
    if (policy.get('image_rules_in_priority_order') != RULES
            or not isinstance(policy.get('version'), str) or not policy['version']):
        raise ValueError('Unsupported eligibility rules/version')
    near = policy.get('near_duplicates', {})
    if not isinstance(near, dict): raise ValueError('Near-duplicate settings must be an object')
    if near.get('algorithm') != 'dhash64_ahash64_aspect_v1' or near.get('action') != 'coassign_eligible_candidates':
        raise ValueError('Unsupported near-duplicate policy')
    for key in ('max_dhash_distance', 'max_ahash_distance'):
        if type(near.get(key)) is not int or not 0 <= near[key] <= 64:
            raise ValueError(f'Invalid {key}')
    aspect = near.get('max_relative_aspect_difference')
    if not finite_number(aspect) or not 0 <= aspect <= 1:
        raise ValueError('Invalid near-duplicate aspect tolerance')
    check_settings(policy.get('default_ratios'), policy.get('default_seed'))
    return policy


def check_settings(ratios, seed):
    if (not isinstance(ratios, dict) or set(ratios) != set(SPLITS)
            or not all(finite_number(v) and 0 < v < 1 for v in ratios.values())
            or abs(sum(ratios.values())-1) > 1e-9 or type(seed) is not int):
        raise ValueError('Use integer seed and positive train/validation/test ratios summing to one')


def fresh_audit(report):
    """Never trust editable review ledgers to decide training eligibility."""
    recorded = load_report(report)
    source = recorded['summary']['source']
    fresh = audit_dataset(source['dataset_root'], source['annotations'], source['taxonomy_path'])
    if fresh['summary']['source'] != source:
        raise ValueError('Source identity changed; regenerate the audit')
    return fresh


def annotation_issues(row):
    issues = list(row['issues']); original = row['original']
    if not finite_number(original.get('area')) or original['area'] <= 0: issues.append('invalid_source_area')
    if type(original.get('iscrowd')) is not int or original['iscrowd'] not in (0, 1): issues.append('invalid_source_iscrowd')
    return issues


def make_plan(audit, policy, seed=None, ratios=None, progress=None):
    seed = policy['default_seed'] if seed is None else seed
    ratios = dict(policy['default_ratios'] if ratios is None else ratios)
    check_settings(ratios, seed)
    by_image = defaultdict(list)
    ids = {r['original'].get('id') for r in audit['images'] if isinstance(r['original'], dict) and valid_id(r['original'].get('id'))}
    orphans = []
    for row in audit['annotations']:
        source = row['original']
        if isinstance(source, dict) and valid_id(source.get('image_id')) and source['image_id'] in ids:
            by_image[source['image_id']].append(row)
        else:
            orphans.append(row['index'])
    # Unassignable annotations cannot safely be made background on any image.
    if orphans:
        raise ValueError(f'Unassignable annotation records block strict export: indexes {orphans}')
    fingerprints = {}
    for position, row in enumerate(audit['images'], 1):
        if row['status'] == 'readable':
            fingerprints[row['original']['id']] = fingerprint(row)
        if progress and (position % 100 == 0 or position == len(audit['images'])):
            progress(f'Fingerprint records: {position}/{len(audit["images"])}')
    exact, near = duplicate_groups(fingerprints, policy['near_duplicates'])
    rows = []
    for image in audit['images']:
        original = image['original']; ident = original.get('id') if isinstance(original, dict) else None
        annotations = by_image[ident] if valid_id(ident) else []
        reasons = []
        if image['status'] in ('missing', 'unreadable'): reasons.append('missing_or_unreadable')
        if image['status'] == 'invalid' or 'invalid_declared_dimensions' in image['issues']:
            reasons.append('invalid_image_metadata')
        if 'declared_actual_dimension_mismatch' in image['issues']: reasons.append('dimension_mismatch')
        if image['status'] == 'readable' and image['exif_orientation'] not in policy['allowed_exif_orientations']:
            reasons.append('ambiguous_exif_orientation')
        invalid = [a['index'] for a in annotations if annotation_issues(a)]
        if invalid: reasons.append('invalid_annotation')
        if any(a['original'].get('iscrowd') == 1 or a['original'].get('ignore', 0) != 0 for a in annotations):
            reasons.append('crowd_or_ignore_annotation')
        if any(a['handling'] == 'unlabeled_requires_decision' for a in annotations): reasons.append('unlabeled_annotation')
        if any(a['bantai_label'] is None for a in annotations): reasons.append('excluded_annotation')
        if not any(a['bantai_label'] is not None for a in annotations): reasons.append('no_mapped_annotations')
        rows.append({'image_record_index': image['index'], 'source_image': original,
                     'source_image_id': ident, 'fingerprint': fingerprints.get(ident) if valid_id(ident) else None,
                     'annotation_indexes': [a['index'] for a in annotations], 'invalid_annotation_indexes': invalid,
                     'annotation_count': len(annotations),
                     'mapped_annotation_count': sum(a['bantai_label'] is not None for a in annotations),
                     'labels': sorted({a['bantai_label'] for a in annotations if a['bantai_label'] is not None}),
                     'reasons': reasons, 'eligible': not reasons, 'split': None, 'split_group': None})
    indexed = {r['source_image_id']: r for r in rows if valid_id(r['source_image_id'])}
    conflicts = []
    for group in exact:
        if all(indexed[i]['eligible'] for i in group):
            signatures = {json.dumps(sorted((a['bantai_label_id'], a['original']['bbox']) for a in by_image[i])) for i in group}
            if len(signatures) > 1:
                conflicts.append(group)
                for ident in group:
                    indexed[ident]['reasons'].append('exact_duplicate_conflict'); indexed[ident]['eligible'] = False
        if any(not indexed[i]['eligible'] for i in group):
            for ident in group:
                if indexed[ident]['eligible']:
                    indexed[ident]['reasons'].append('exact_duplicate_quarantine'); indexed[ident]['eligible'] = False
    eligible = {r['source_image_id']: set(r['labels']) for r in rows if r['eligible']}
    groups = Groups(eligible)
    for group in exact:
        members = [i for i in group if i in eligible]
        for member in members[1:]: groups.join(members[0], member)
    for pair in near:
        a, b = pair['image_id_a'], pair['image_id_b']
        if a in eligible and b in eligible: groups.join(a, b)
    assignments, targets = assign_splits(eligible, groups.members(), ratios, seed) if eligible else ({}, {})
    if eligible and assign_splits(eligible, groups.members(), ratios, seed)[0] != assignments:
        raise ValueError('Non-reproducible split assignment')
    for row in rows:
        row['primary_reason'] = next((r for r in RULES if r in row['reasons']), None)
        if row['eligible']:
            ident = row['source_image_id']; row['split'] = assignments[ident]; row['split_group'] = groups.find(ident)
    exports = {s: {'info': {'description': 'BANTAI strict-clean TACO box-detection preparation',
                           'image_root': audit['summary']['source']['dataset_root'],
                           'coordinate_frame': policy['coordinate_frame'], 'policy_version': policy['version']},
                   'licenses': copy.deepcopy(audit['source_document'].get('licenses', [])),
                   'images': [], 'annotations': [],
                   'categories': [{'id': l['id'], 'name': l['name'], 'supercategory': 'trash_object',
                                   'source_taco_categories': l['taco_categories']} for l in audit['taxonomy']['labels']]}
               for s in SPLITS}
    for row in rows:
        if row['eligible']:
            original = row['source_image']
            # Source license strings are provenance, not valid COCO license IDs.
            image = {key: original[key] for key in ('id', 'file_name', 'width', 'height')}
            image['source_taco'] = {'image': copy.deepcopy(original), 'image_record_index': row['image_record_index'],
                                    'file_sha256': row['fingerprint']['file_sha256'],
                                    'pixel_sha256': row['fingerprint']['pixel_sha256']}
            exports[row['split']]['images'].append(image)
    decisions = []
    next_id = 1
    for annotation in audit['annotations']:
        original = annotation['original']; image = indexed[original['image_id']]
        decision = {'annotation_record_index': annotation['index'], 'source_annotation': original,
                    'source_category': annotation['original_category'], 'bantai_label': annotation['bantai_label'],
                    'annotation_issues': annotation_issues(annotation), 'image_reasons': image['reasons'],
                    'split': image['split'], 'exported_annotation_id': None}
        if image['eligible']:
            converted = copy.deepcopy(original)
            converted.update(id=next_id, category_id=annotation['bantai_label_id'],
                             source_taco={'annotation_id': original['id'], 'category': annotation['original_category'],
                                          'annotation_record_index': annotation['index'], 'annotation': original})
            exports[image['split']]['annotations'].append(converted)
            decision['exported_annotation_id'] = next_id; next_id += 1
        decisions.append(decision)
    return {'eligibility': rows, 'annotation_decisions': decisions, 'exports': exports,
            'fingerprints': fingerprints, 'exact_duplicate_groups': exact, 'near_duplicate_candidates': near,
            'exact_duplicate_annotation_conflicts': conflicts,
            'split_groups': groups.members(), 'targets': targets, 'seed': seed, 'ratios': ratios}


def statistics(audit, plan):
    rows = plan['eligibility']; source_anns = {a['index']: a for a in audit['annotations']}
    remaining_images = len(rows); remaining_anns = set(source_anns); waterfall = []
    for reason in RULES:
        removed = [r for r in rows if r['primary_reason'] == reason]
        indexes = set(i for r in removed for i in r['annotation_indexes']) & remaining_anns
        remaining_images -= len(removed); remaining_anns -= indexes
        waterfall.append({'rule': reason, 'images_removed': len(removed), 'annotations_removed': len(indexes),
                          'mapped_annotations_removed': sum(source_anns[i]['bantai_label'] is not None for i in indexes),
                          'images_remaining': remaining_images, 'annotations_remaining': len(remaining_anns)})
    classes = []; warnings = []
    for label in audit['taxonomy']['labels']:
        row = {'label': label['name'], 'category_id': label['id']}
        for split in SPLITS:
            annotations = [a for a in plan['exports'][split]['annotations'] if a['category_id'] == label['id']]
            row[split+'_objects'] = len(annotations)
            row[split+'_images'] = len({a['image_id'] for a in annotations})
            row[split+'_unique_photographs'] = len({plan['fingerprints'][a['image_id']]['pixel_sha256'] for a in annotations})
        row['total_objects'] = sum(row[s+'_objects'] for s in SPLITS)
        row['total_images'] = sum(row[s+'_images'] for s in SPLITS)
        row['total_unique_photographs'] = sum(row[s+'_unique_photographs'] for s in SPLITS)
        row['eligible_split_groups'] = sum(any(label['name'] in next(r['labels'] for r in rows if r['source_image_id'] == i) for i in group) for group in plan['split_groups'])
        for split in SPLITS:
            if row[split+'_images'] < 2:
                warnings.append(f"{label['name']}: {split} has {row[split+'_images']} images; inadequate evaluation/training coverage needs review")
        classes.append(row)
    sizes = {s: {'images': len(plan['exports'][s]['images']), 'annotations': len(plan['exports'][s]['annotations']),
                 'unique_exact_photographs': len({plan['fingerprints'][i['id']]['pixel_sha256'] for i in plan['exports'][s]['images']})}
             for s in SPLITS}
    counts = [r['total_objects'] for r in classes if r['total_objects']]
    return {'schema_version': 'bantai.taco-preparation-stats.v1', 'source_audit_counts': audit['summary']['counts'],
            'eligibility_waterfall': waterfall, 'classes': classes, 'splits': sizes, 'split_targets': plan['targets'],
            'eligible_images': sum(r['eligible'] for r in rows), 'eligible_annotations': len(remaining_anns),
            'quarantined_images': sum(not r['eligible'] for r in rows),
            'reason_counts_overlapping': dict(Counter(reason for r in rows for reason in r['reasons'])),
            'exact_duplicate_groups': len(plan['exact_duplicate_groups']),
            'near_duplicate_candidate_pairs': len(plan['near_duplicate_candidates']),
            'largest_split_group': max((len(g) for g in plan['split_groups']), default=0),
            'object_imbalance_max_to_min_nonzero': max(counts)/min(counts) if counts else None,
            'zero_object_classes': [r['label'] for r in classes if not r['total_objects']],
            'coverage_warnings': warnings,
            'limitations': ['Strict-clean selection can bias scene/class coverage; it is not evidence that photos are fully annotated.',
                           'Multi-label/group stratification is heuristic, not guaranteed or optimal.',
                           'Perceptual hashes flag candidates, not confirmed identity; crops, rotations and viewpoint changes may evade them.',
                           'Detected eligible near-duplicate candidates are coassigned; unresolved undetected leakage needs visual/source review.',
                           'Licensing and eventual trainer compatibility remain unverified; no fine-tuning or ignore-region strategy was tested.']}


def validate_coco(exports, plan, taxonomy):
    errors = []; all_images = set(); all_annotations = set(); file_splits = {}; pixel_splits = {}; group_splits = {}
    expected_categories = [{'id': l['id'], 'name': l['name'], 'supercategory': 'trash_object',
                            'source_taco_categories': l['taco_categories']} for l in taxonomy['labels']]
    indexed = {r['source_image_id']: r for r in plan['eligibility'] if r['eligible']}
    for split in SPLITS:
        coco = exports[split]
        if coco.get('categories') != expected_categories: errors.append(f'{split}: taxonomy/order mismatch')
        images = {}
        for image in coco.get('images', []):
            ident = image.get('id')
            if not valid_id(ident) or ident in all_images: errors.append(f'{split}: duplicate/invalid image ID {ident}')
            all_images.add(ident); images[ident] = image
            row = indexed.get(ident)
            if row is None or row['split'] != split: errors.append(f'{split}: ineligible or wrong-split image {ident}'); continue
            for key, lookup in (('file_sha256', file_splits), ('pixel_sha256', pixel_splits)):
                value = row['fingerprint'][key]
                if value in lookup and lookup[value] != split: errors.append('Exact photograph leakage across splits')
                lookup[value] = split
            group = row['split_group']
            if group in group_splits and group_splits[group] != split: errors.append('Split-group leakage')
            group_splits[group] = split
        allowed = {l['id'] for l in taxonomy['labels']}
        for ann in coco.get('annotations', []):
            if not valid_id(ann.get('id')) or ann['id'] in all_annotations: errors.append('Duplicate/invalid annotation ID')
            all_annotations.add(ann.get('id'))
            image = images.get(ann.get('image_id'))
            if image is None: errors.append('Unknown annotation image reference'); continue
            if ann.get('category_id') not in allowed: errors.append('Unknown category ID')
            if bbox_issues(ann.get('bbox'), image['width'], image['height']): errors.append('Invalid exported bbox')
        if coco != plan['exports'][split]: errors.append(f'{split}: export differs from independently derived eligible source records')
    if not indexed: errors.append('Strict-clean subset is empty')
    return errors


def write_preparation(audit, policy, policy_path, plan, output, input_audit=None):
    output = resolve_path(str(output)); root = Path(audit['summary']['source']['dataset_root'])
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Preparation output cannot overlap source dataset')
    if load_policy(policy_path) != policy: raise ValueError('Policy source changed during preparation')
    errors = validate_coco(plan['exports'], plan, audit['taxonomy'])
    if errors: raise ValueError('Export gates failed: '+'; '.join(errors))
    output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False, sort_keys=True)+'\n', encoding='utf-8')
    stats = statistics(audit, plan)
    write('policy_snapshot.json', policy); write('taxonomy_snapshot.json', audit['taxonomy'])
    for split in SPLITS: write(split+'.json', plan['exports'][split])
    write('statistics.json', stats)
    write('duplicate_review.json', {'exact_groups': plan['exact_duplicate_groups'],
                                  'exact_annotation_conflicts': plan['exact_duplicate_annotation_conflicts'],
                                  'near_candidates': plan['near_duplicate_candidates'], 'split_groups': plan['split_groups']})
    write('source_metadata.json', {k: v for k, v in audit['source_document'].items() if k not in ('images', 'annotations')})
    for name, rows in (('image_eligibility.jsonl', plan['eligibility']), ('annotation_decisions.jsonl', plan['annotation_decisions'])):
        with (output/name).open('w', encoding='utf-8', newline='\n') as stream:
            for row in rows: stream.write(json.dumps(row, allow_nan=False, ensure_ascii=False, sort_keys=True)+'\n')
    for name, rows in (('class_counts.csv', stats['classes']), ('eligibility_waterfall.csv', stats['eligibility_waterfall']),
                       ('image_eligibility.csv', [{'image_record_index': r['image_record_index'], 'image_id': r['source_image_id'],
                                                   'file_name': r['source_image'].get('file_name') if isinstance(r['source_image'], dict) else None,
                                                   'eligible': r['eligible'], 'split': r['split'], 'primary_reason': r['primary_reason'],
                                                   'all_reasons': r['reasons'], 'annotations': r['annotation_count'],
                                                   'mapped_annotations': r['mapped_annotation_count']} for r in plan['eligibility']])):
        with (output/name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
            for row in rows: writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    source = audit['summary']['source']
    verify_source_integrity(audit)
    if load_policy(policy_path) != policy: raise ValueError('Policy source changed during preparation')
    manifest = {'schema_version': 'bantai.taco-preparation.v1', 'status': 'validated', 'training_ready': False,
                'source': source, 'input_audit': str(resolve_path(str(input_audit))) if input_audit else None,
                'policy_version': policy['version'], 'policy_source_sha256': sha256(policy_path),
                'taxonomy_version': audit['taxonomy']['version'], 'seed': plan['seed'], 'ratios': plan['ratios'],
                'split_algorithm': policy['split_algorithm'], 'image_root': source['dataset_root'],
                'images_copied': False, 'validation': {'coco_source_consistency': 'passed', 'source_integrity': 'passed',
                                                     'exact_duplicate_leakage': 'passed', 'reproducible_assignment': 'passed'},
                'pending_requirements': ['license_review', 'trainer_compatibility', 'near_duplicate_visual_review', 'class_coverage_acceptance'],
                'artifacts': {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write('preparation_manifest.json', manifest)
    return stats


def verify_source_integrity(audit):
    source = audit['summary']['source']
    if sha256(source['annotations']) != source['sha256'] or sha256(source['taxonomy_path']) != source['taxonomy_sha256']:
        raise ValueError('Source metadata/taxonomy changed during preparation')
    for row in audit['images']:
        if row['sha256'] and sha256(row['path']) != row['sha256']:
            raise ValueError(f"Source photo changed: {row['path']}")
        if row['status'] == 'missing' and row['path'] and Path(row['path']).exists():
            raise ValueError(f"Previously missing source appeared during preparation: {row['path']}")


def validate_preparation(folder):
    folder = resolve_path(str(folder)); manifest = strict_load(folder/'preparation_manifest.json')
    if manifest.get('schema_version') != 'bantai.taco-preparation.v1' or manifest.get('status') != 'validated':
        raise ValueError('Incomplete or unsupported preparation manifest')
    for name, digest in manifest['artifacts'].items():
        path = (folder/name).resolve()
        if path.parent != folder or sha256(path) != digest: raise ValueError(f'Artifact checksum mismatch: {name}')
    source = manifest['source']
    if sha256(source['annotations']) != source['sha256'] or sha256(source['taxonomy_path']) != source['taxonomy_sha256']:
        raise ValueError('Current source no longer matches preparation manifest')
    policy = load_policy(folder/'policy_snapshot.json')
    audit = audit_dataset(source['dataset_root'], source['annotations'], source['taxonomy_path'])
    if audit['summary']['source'] != source: raise ValueError('Source identity mismatch')
    if strict_load(folder/'taxonomy_snapshot.json') != audit['taxonomy'] or manifest['policy_version'] != policy['version']:
        raise ValueError('Taxonomy/policy snapshot identity mismatch')
    plan = make_plan(audit, policy, manifest['seed'], manifest['ratios'])
    actual = {s: strict_load(folder/(s+'.json')) for s in SPLITS}
    errors = validate_coco(actual, plan, audit['taxonomy'])
    if strict_load(folder/'statistics.json') != statistics(audit, plan): errors.append('Statistics/reproducibility mismatch')
    with (folder/'image_eligibility.jsonl').open(encoding='utf-8') as stream:
        if [json.loads(line) for line in stream] != plan['eligibility']: errors.append('Eligibility manifest mismatch')
    with (folder/'annotation_decisions.jsonl').open(encoding='utf-8') as stream:
        if [json.loads(line) for line in stream] != plan['annotation_decisions']: errors.append('Annotation traceability mismatch')
    verify_source_integrity(audit)
    return {'schema_version': 'bantai.taco-validation.v1', 'passed': not errors, 'errors': errors,
            'recomputed_from_current_sources': True, 'training_ready': False,
            'splits': statistics(audit, plan)['splits'], 'pending_requirements': manifest['pending_requirements']}
