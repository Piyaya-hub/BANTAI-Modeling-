"""Read-only COCO audit and taxonomy views; never creates training examples."""

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PureWindowsPath

from PIL import Image

from .configuration import ROOT, finite_number, resolve_path, strict_load

DEFAULT_DATASET = ROOT.parent / 'BANTAI-Datasets' / 'TACO'
DEFAULT_TAXONOMY = ROOT / 'config/taco_taxonomy.v1.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_id(value):
    return type(value) is int and value >= 0


def load_taxonomy(path=DEFAULT_TAXONOMY):
    taxonomy = strict_load(path)
    if (not isinstance(taxonomy, dict) or taxonomy.get('schema_version') != 'bantai.taxonomy.v1'
            or taxonomy.get('matching') != 'exact_original_category_name'
            or taxonomy.get('excluded_annotation_policy') != 'requires_explicit_decision_not_background'
            or not isinstance(taxonomy.get('version'), str) or not taxonomy['version']):
        raise ValueError('Unsupported taxonomy schema, matching or exclusion policy')
    names, ids, original = set(), set(), set()
    labels = taxonomy.get('labels')
    if not isinstance(labels, list) or not labels:
        raise ValueError('Taxonomy labels must be a nonempty list')
    for label in labels:
        if (not isinstance(label, dict) or not valid_id(label.get('id'))
                or not isinstance(label.get('name'), str) or not label['name']
                or not isinstance(label.get('taco_categories'), list) or not label['taco_categories']):
            raise ValueError('Malformed taxonomy label')
        if label['id'] in ids or label['name'] in names:
            raise ValueError('Duplicate taxonomy label ID/name')
        ids.add(label['id']); names.add(label['name'])
        for category in label['taco_categories']:
            if not isinstance(category, str) or not category or category in original:
                raise ValueError('Duplicate or invalid original category mapping')
            original.add(category)
    flagged = taxonomy.get('flagged_categories')
    if not isinstance(flagged, list) or not all(isinstance(n, str) for n in flagged):
        raise ValueError('flagged_categories must be a list of original names')
    if original.intersection(flagged):
        raise ValueError('An explicitly flagged category cannot also be mapped')
    return taxonomy


def category_mapping(category, taxonomy):
    name = category.get('name') if isinstance(category, dict) else None
    for label in taxonomy['labels']:
        if name in label['taco_categories']:
            return label['name'], label['id'], 'mapped_review_only'
    handling = ('unlabeled_requires_decision' if name in taxonomy['flagged_categories']
                else 'excluded_requires_decision' if name else 'unknown_requires_decision')
    return None, None, handling


def image_path(root, file_name):
    """Reject absolute/traversal/symlink escape paths before reading a source."""
    if not isinstance(file_name, str) or not file_name or '\x00' in file_name:
        raise ValueError('invalid_file_name')
    win = PureWindowsPath(file_name)
    if win.drive or win.root or Path(file_name).is_absolute():
        raise ValueError('unsafe_image_path')
    path = (root / file_name.replace('\\', '/')).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('unsafe_image_path')
    return path


def bbox_issues(box, width, height):
    if not isinstance(box, list) or len(box) != 4:
        return ['bbox_requires_four_xywh_values']
    if not all(finite_number(v) for v in box):
        return ['bbox_nonfinite_or_nonnumeric']
    x, y, w, h = box
    try:
        right, bottom = x+w, y+h
    except OverflowError:
        return ['bbox_endpoint_overflow']
    if not finite_number(right) or not finite_number(bottom):
        return ['bbox_endpoint_overflow']
    issues = []
    if w <= 0 or h <= 0:
        issues.append('bbox_nonpositive_size')
    if x < 0 or y < 0:
        issues.append('bbox_negative_origin')
    if width is not None and height is not None and (right > width or bottom > height):
        issues.append('bbox_out_of_bounds')
    return issues


def unique_index(records, kind, issues):
    counts = Counter(r.get('id') for r in records if isinstance(r, dict) and valid_id(r.get('id')))
    index = {}
    for position, record in enumerate(records):
        ident = record.get('id') if isinstance(record, dict) else None
        reason = ('invalid_record_or_id' if not valid_id(ident)
                  else 'duplicate_id' if counts[ident] > 1 else None)
        if reason:
            issues.append({'kind': kind, 'index': position, 'id': ident, 'reason': reason})
        else:
            index[ident] = record
    return index


def audit_dataset(dataset_root=DEFAULT_DATASET, annotations=None, taxonomy_path=DEFAULT_TAXONOMY):
    root = resolve_path(str(dataset_root))
    annotation_path = resolve_path(str(annotations)) if annotations else root / 'annotations.json'
    taxonomy_path = resolve_path(str(taxonomy_path))
    if not root.is_dir():
        raise ValueError(f'Dataset directory unavailable: {root}')
    source_hash, taxonomy_hash = sha256(annotation_path), sha256(taxonomy_path)
    document = strict_load(annotation_path)
    taxonomy = load_taxonomy(taxonomy_path)
    if not isinstance(document, dict) or any(not isinstance(document.get(k), list)
                                            for k in ('images', 'annotations', 'categories')):
        raise ValueError('COCO requires images, annotations and categories arrays')
    structural = []
    categories = unique_index(document['categories'], 'category', structural)
    images = unique_index(document['images'], 'image', structural)
    annotations_by_id = unique_index(document['annotations'], 'annotation', structural)
    for ident, category in list(categories.items()):
        if not isinstance(category.get('name'), str) or not category['name']:
            structural.append({'kind': 'category', 'id': ident, 'reason': 'invalid_category_name'})
            del categories[ident]
    image_rows, image_lookup = [], {}
    for position, original in enumerate(document['images']):
        row = {'index': position, 'original': original, 'status': 'invalid', 'issues': [],
               'path': None, 'sha256': None, 'actual_width': None, 'actual_height': None,
               'exif_orientation': None, 'exif_width': None, 'exif_height': None}
        image_rows.append(row)
        if not isinstance(original, dict) or not valid_id(original.get('id')) or original['id'] not in images:
            row['issues'].append('invalid_or_duplicate_image_id'); continue
        image_lookup[original['id']] = row
        width, height = original.get('width'), original.get('height')
        if not (type(width) is int and type(height) is int and width > 0 and height > 0):
            row['issues'].append('invalid_declared_dimensions')
        try:
            path = image_path(root, original.get('file_name'))
            row['path'] = str(path)
            if not path.is_file():
                row['status'] = 'missing'; row['issues'].append('missing_image'); continue
            payload = path.read_bytes()
            row['sha256'] = hashlib.sha256(payload).hexdigest()
            # COCO boxes refer to the stored raster. Do not apply EXIF transpose.
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                row.update(actual_width=image.width, actual_height=image.height,
                           exif_orientation=image.getexif().get(274, 1))
            row['status'] = 'readable'
            row['exif_width'], row['exif_height'] = ((row['actual_height'], row['actual_width'])
                if row['exif_orientation'] in (5, 6, 7, 8) else (row['actual_width'], row['actual_height']))
            if (width, height) != (row['actual_width'], row['actual_height']):
                row['issues'].append('declared_actual_dimension_mismatch')
        except ValueError as exc:
            row['issues'].append(str(exc))
        except (OSError, Image.DecompressionBombError) as exc:
            row['status'] = 'unreadable'; row['issues'].append(type(exc).__name__)
    annotation_rows = []
    for position, original in enumerate(document['annotations']):
        row = {'index': position, 'original': original, 'original_category': None,
               'bantai_label': None, 'bantai_label_id': None, 'handling': 'unknown_requires_decision',
               'issues': [], 'image_status': None, 'image_issues': [], 'usable': False, 'valid_annotation': False}
        annotation_rows.append(row)
        if not isinstance(original, dict):
            row['issues'].append('annotation_not_object'); continue
        if not valid_id(original.get('id')) or original['id'] not in annotations_by_id:
            row['issues'].append('invalid_or_duplicate_annotation_id')
        category_id, image_id = original.get('category_id'), original.get('image_id')
        category = categories.get(category_id) if valid_id(category_id) else None
        image_row = image_lookup.get(image_id) if valid_id(image_id) else None
        if category is None:
            row['issues'].append('unknown_or_ambiguous_category_id')
        else:
            row['original_category'] = category
            row['bantai_label'], row['bantai_label_id'], row['handling'] = category_mapping(category, taxonomy)
        if image_row is None:
            row['issues'].append('unknown_or_ambiguous_image_id')
        else:
            row['image_status'] = image_row['status']
            row['image_issues'] = list(image_row['issues'])
        declared = image_row['original'] if image_row else {}
        w, h = declared.get('width'), declared.get('height')
        w = w if type(w) is int and w > 0 else None
        h = h if type(h) is int and h > 0 else None
        row['issues'].extend(bbox_issues(original.get('bbox'), w, h))
        row['valid_annotation'] = not row['issues']
        row['usable'] = bool(row['valid_annotation'] and image_row
                             and image_row['status'] == 'readable' and not image_row['issues'])
    class_rows = []
    readable_ids = {r['original']['id'] for r in image_rows if r['status'] == 'readable'}
    for label in taxonomy['labels']:
        rows = [a for a in annotation_rows if a['bantai_label'] == label['name']]
        usable = [a for a in rows if a['usable']]
        class_rows.append({'label_id': label['id'], 'label': label['name'],
                           'original_categories': label['taco_categories'], 'total_annotations': len(rows),
                           'annotations_on_readable_images': sum(valid_id(a['original'].get('image_id')) and a['original']['image_id'] in readable_ids for a in rows),
                           'usable_objects': len(usable),
                           'usable_unique_images': len({a['original']['image_id'] for a in usable}),
                           'invalid_annotations': sum(not a['valid_annotation'] for a in rows),
                           'unavailable_or_unaligned_annotations': sum(a['valid_annotation'] and not a['usable'] for a in rows)})
    original_classes = []
    for ident, category in sorted(categories.items()):
        rows = [a for a in annotation_rows if a['original_category'] is category]
        usable = [a for a in rows if a['usable']]
        label, _, handling = category_mapping(category, taxonomy)
        original_classes.append({'category_id': ident, 'original_name': category['name'], 'bantai_label': label,
                                 'handling': handling, 'total_annotations': len(rows), 'usable_objects': len(usable),
                                 'usable_unique_images': len({a['original']['image_id'] for a in usable})})
    usable = [a for a in annotation_rows if a['usable']]
    excluded_images = {a['original']['image_id'] for a in usable if a['bantai_label'] is None}
    mapped_images = {a['original']['image_id'] for a in usable if a['bantai_label'] is not None}
    expected_names = {n for label in taxonomy['labels'] for n in label['taco_categories']}
    observed_names = {c['name'] for c in categories.values()}
    summary = {'schema_version': 'bantai.taco-audit.v1',
               'source': {'dataset_root': str(root), 'annotations': str(annotation_path), 'sha256': source_hash,
                          'taxonomy_path': str(taxonomy_path), 'taxonomy_sha256': taxonomy_hash,
                          'taxonomy_version': taxonomy['version']},
               'coordinate_frame': 'stored_raster_without_exif_transpose', 'bbox_format': 'coco_xywh_pixels',
               'training_export_created': False, 'excluded_policy': taxonomy['excluded_annotation_policy'],
               'counts': {'image_records': len(image_rows), 'image_status': dict(Counter(r['status'] for r in image_rows)),
                          'image_issue_records': sum(bool(r['issues']) for r in image_rows),
                          'annotations': len(annotation_rows), 'invalid_annotations': sum(not a['valid_annotation'] for a in annotation_rows),
                          'annotations_on_readable_images': sum(isinstance(a['original'], dict) and valid_id(a['original'].get('image_id')) and a['original']['image_id'] in readable_ids for a in annotation_rows),
                          'dimension_mismatch_images': sum('declared_actual_dimension_mismatch' in r['issues'] for r in image_rows),
                          'usable_annotations_all_categories': len(usable),
                          'usable_mapped_annotations': sum(a['bantai_label'] is not None for a in usable),
                          'usable_excluded_annotations': sum(a['bantai_label'] is None for a in usable),
                          'usable_mapped_unique_images': len(mapped_images),
                          'mapped_images_with_usable_excluded_annotations': len(mapped_images & excluded_images)},
               'classes': class_rows, 'original_classes': original_classes,
               'missing_taxonomy_categories': sorted(expected_names-observed_names), 'structural_issues': structural,
               'warning': 'Review only. Excluded/unlabeled annotations remain explicit; no implicit background, splits or training dataset is produced.'}
    if sha256(annotation_path) != source_hash or sha256(taxonomy_path) != taxonomy_hash:
        raise ValueError('Source annotations or taxonomy changed during audit; rerun on a stable source')
    return {'summary': summary, 'images': image_rows, 'annotations': annotation_rows,
            'source_document': document, 'taxonomy': taxonomy}


def write_reports(audit, output):
    output = resolve_path(str(output))
    root = Path(audit['summary']['source']['dataset_root'])
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Report output must not overlap the external dataset')
    output.mkdir(parents=True, exist_ok=False)  # Never replace an earlier audit.
    def write_json(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False)+'\n', encoding='utf-8')
    write_json('audit.json', audit['summary'])
    write_json('taxonomy_snapshot.json', audit['taxonomy'])
    write_json('source_metadata.json', {k: v for k, v in audit['source_document'].items() if k not in ('images', 'annotations')})
    for name, rows in (('images.jsonl', audit['images']), ('annotation_review.jsonl', audit['annotations'])):
        with (output/name).open('w', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False, ensure_ascii=False)+'\n')
    for name, rows in (('class_counts.csv', audit['summary']['classes']),
                       ('original_classes.csv', audit['summary']['original_classes']),
                       ('image_issues.csv', [dict(image_id=r['original'].get('id') if isinstance(r['original'], dict) else None,
                                                  file_name=r['original'].get('file_name') if isinstance(r['original'], dict) else None,
                                                  status=r['status'], issues=r['issues']) for r in audit['images'] if r['issues']]),
                       ('annotation_issues.csv', [dict(index=r['index'], annotation_id=r['original'].get('id') if isinstance(r['original'], dict) else None,
                                                       issues=r['issues'], image_status=r['image_status'], image_issues=r['image_issues'])
                                                  for r in audit['annotations'] if not r['usable']])):
        with (output/name).open('w', encoding='utf-8', newline='') as stream:
            empty_headers = {'image_issues.csv': ['image_id', 'file_name', 'status', 'issues'],
                             'annotation_issues.csv': ['index', 'annotation_id', 'issues', 'image_status', 'image_issues']}
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else empty_headers.get(name, ['category_id']))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    return output


def load_report(folder):
    """Reuse an audit; source metadata must still match, photos checked on viewing."""
    folder = resolve_path(str(folder))
    summary = strict_load(folder/'audit.json')
    if summary.get('schema_version') != 'bantai.taco-audit.v1':
        raise ValueError('Unsupported audit report schema')
    source = summary['source']
    if sha256(source['annotations']) != source['sha256'] or sha256(source['taxonomy_path']) != source['taxonomy_sha256']:
        raise ValueError('Annotations/taxonomy changed since audit; generate a new report')
    def rows(name):
        with (folder/name).open(encoding='utf-8') as stream:
            return [json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v))) for line in stream]
    return {'summary': summary, 'taxonomy': load_taxonomy(source['taxonomy_path']),
            'images': rows('images.jsonl'), 'annotations': rows('annotation_review.jsonl')}
