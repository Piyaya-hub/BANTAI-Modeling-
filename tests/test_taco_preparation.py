"""Synthetic strict-clean conversion, leakage prevention and reproducibility checks."""
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from PIL import Image

from scripts.configuration import ROOT
from scripts.taco_dataset import audit_dataset, sha256, write_reports
from scripts.taco_preparation import (DEFAULT_POLICY, fresh_audit, load_policy, make_plan,
                                      statistics, validate_coco, validate_preparation, write_preparation)
from scripts.taco_splits import SPLITS, allocate, assign_splits, duplicate_groups


class PreparationTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'work').mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT/'work')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.dataset = self.root/'photos'; self.dataset.mkdir()
        self.document = {'info': {'fixture': True}, 'licenses': [], 'images': [], 'annotations': [],
                         'categories': [{'id': 59, 'name': 'Cigarette'}, {'id': 5, 'name': 'Clear plastic bottle'},
                                        {'id': 58, 'name': 'Unlabeled litter'}, {'id': 6, 'name': 'Glass bottle'}]}
        self.policy = load_policy()
        self.source = self.dataset/'annotations.json'

    def add(self, ident, category=59, duplicate=None, bbox=None, orientation=None, missing=False):
        path = self.dataset/f'{ident}.png'
        if not missing:
            if duplicate is not None:
                path.write_bytes((self.dataset/f'{duplicate}.png').read_bytes())
            else:
                rng = random.Random(ident)
                with Image.frombytes('RGB', (40, 30), bytes(rng.randrange(256) for _ in range(40*30*3))) as image:
                    if orientation is None: image.save(path)
                    else:
                        exif = Image.Exif(); exif[274] = orientation; image.save(path, exif=exif)
        self.document['images'].append({'id': ident, 'file_name': path.name, 'width': 40, 'height': 30, 'license': None})
        self.document['annotations'].append({'id': ident*10+1, 'image_id': ident, 'category_id': category,
                                             'bbox': [2, 3, 5, 7] if bbox is None else bbox, 'area': 35.0, 'iscrowd': 0,
                                             'segmentation': [[2, 3, 7, 3, 7, 10]], 'custom': {'retained': ident}})

    def audit(self):
        self.source.write_text(json.dumps(self.document), encoding='utf-8')
        return audit_dataset(self.dataset)

    def plan(self):
        audit = self.audit()
        return audit, make_plan(audit, self.policy)

    def test_all_mapping_order_and_traceability_with_unique_export_ids(self):
        for i in range(12): self.add(i, 59 if i % 2 else 5)
        audit, plan = self.plan(); before = sha256(self.source)
        self.assertEqual(validate_coco(plan['exports'], plan, audit['taxonomy']), [])
        ids = []
        for coco in plan['exports'].values():
            self.assertEqual([c['name'] for c in coco['categories']], [l['name'] for l in audit['taxonomy']['labels']])
            for ann in coco['annotations']:
                ids.append(ann['id'])
                original = ann['source_taco']['annotation']
                self.assertEqual(ann['bbox'], original['bbox'])
                self.assertEqual(ann['custom'], original['custom'])
                self.assertEqual(ann['source_taco']['annotation_id'], original['id'])
                self.assertEqual(ann['category_id'], 0 if original['category_id'] == 59 else 1)
        self.assertEqual(sorted(ids), list(range(1, 13))); self.assertEqual(sha256(self.source), before)

    def test_mixed_unlabeled_excluded_and_invalid_whole_images_quarantined(self):
        for i in range(6): self.add(i)
        for ident, category in ((0, 58), (1, 6)):
            self.document['annotations'].append(dict(self.document['annotations'][ident], id=900+ident, category_id=category))
        self.document['annotations'][2]['bbox'] = [-1, 0, 5, 7]
        self.document['annotations'][3]['id'] = self.document['annotations'][4]['id']
        audit, plan = self.plan(); rows = {r['source_image_id']: r for r in plan['eligibility']}
        self.assertEqual([i for i in rows if rows[i]['eligible']], [5])
        self.assertEqual(rows[0]['primary_reason'], 'unlabeled_annotation')
        self.assertEqual(rows[1]['primary_reason'], 'excluded_annotation')
        self.assertTrue(all(rows[i]['primary_reason'] == 'invalid_annotation' for i in (2, 3, 4)))
        self.assertEqual(sum(len(e['annotations']) for e in plan['exports'].values()), 1)
        stats = statistics(audit, plan)
        self.assertEqual(sum(s['annotations_removed'] for s in stats['eligibility_waterfall'])+stats['eligible_annotations'], len(self.document['annotations']))

    def test_missing_mismatched_and_ambiguous_orientation_held(self):
        self.add(0, missing=True); self.add(1); self.add(2, orientation=3); self.add(3)
        self.document['images'][1]['width'] = 39
        audit, plan = self.plan(); rows = plan['eligibility']
        self.assertEqual([r['primary_reason'] for r in rows], ['missing_or_unreadable', 'dimension_mismatch', 'ambiguous_exif_orientation', None])

    def test_exact_duplicates_coassigned_and_dirty_peer_propagates(self):
        self.add(0); self.add(1, duplicate=0)
        for i in range(2, 9): self.add(i)
        audit, plan = self.plan(); rows = {r['source_image_id']: r for r in plan['eligibility']}
        self.assertEqual(rows[0]['split'], rows[1]['split'])
        self.assertEqual(plan['exact_duplicate_groups'], [[0, 1]])
        self.document['annotations'].append(dict(self.document['annotations'][1], id=999, category_id=58))
        audit, plan = self.plan(); rows = {r['source_image_id']: r for r in plan['eligibility']}
        self.assertFalse(rows[0]['eligible']); self.assertEqual(rows[0]['primary_reason'], 'exact_duplicate_quarantine')

    def test_exact_same_pixels_different_encoding_and_annotation_conflict(self):
        self.add(0)
        self.add(1, category=5)
        with Image.open(self.dataset/'0.png') as image: image.save(self.dataset/'1.png', compress_level=0)
        audit, plan = self.plan()
        self.assertNotEqual(plan['fingerprints'][0]['file_sha256'], plan['fingerprints'][1]['file_sha256'])
        self.assertEqual(plan['fingerprints'][0]['pixel_sha256'], plan['fingerprints'][1]['pixel_sha256'])
        self.assertEqual(plan['exact_duplicate_annotation_conflicts'], [[0, 1]])
        self.assertTrue(all(r['primary_reason'] == 'exact_duplicate_conflict' for r in plan['eligibility']))

    def test_seeded_multilabel_assignment_and_small_class_targets(self):
        labels = {i: {'rare'} if i < 3 else {'common', 'second'} if i % 2 else {'common'} for i in range(30)}
        groups = [[i] for i in labels]
        ratios = self.policy['default_ratios']
        a, stats = assign_splits(labels, groups, ratios, 42)
        b, _ = assign_splits(dict(reversed(list(labels.items()))), list(reversed(groups)), ratios, 42)
        self.assertEqual(a, b)
        self.assertEqual(set(a[i] for i in range(3)), set(SPLITS))
        self.assertEqual(CounterLike(a), stats['target_images'])
        self.assertNotEqual(a, assign_splits(labels, groups, ratios, 43)[0])
        self.assertEqual(allocate(3, ratios, minimum=True), dict.fromkeys(SPLITS, 1))

    def test_near_duplicate_candidates_are_coassigned_not_asserted_identity(self):
        self.add(0); self.add(1)
        audit = self.audit()
        from scripts import taco_preparation as preparation
        from unittest.mock import patch
        fp = {i: {'file_sha256': str(i), 'pixel_sha256': str(i), 'dhash64': f'{i:016x}', 'ahash64': f'{i:016x}', 'aspect_ratio': 4/3} for i in (0, 1)}
        exact, pairs = duplicate_groups(fp, self.policy['near_duplicates'])
        self.assertEqual(exact, []); self.assertFalse(pairs[0]['identity_confirmed'])
        with patch.object(preparation, 'fingerprint', side_effect=lambda row: fp[row['original']['id']]):
            plan = make_plan(audit, self.policy)
        self.assertEqual(plan['eligibility'][0]['split'], plan['eligibility'][1]['split'])
        self.assertEqual(plan['split_groups'], [[0, 1]])

    def test_class_coverage_losses_and_imbalance_explicit(self):
        for i in range(5): self.add(i, 5 if i == 0 else 59)
        audit, plan = self.plan(); stats = statistics(audit, plan)
        self.assertIn('plastic_bag', stats['zero_object_classes'])
        self.assertTrue(any('plastic_bottle:' in w and '0 images' in w for w in stats['coverage_warnings']))
        self.assertEqual(stats['object_imbalance_max_to_min_nonzero'], 4)

    def test_validation_detects_omissions_foreign_refs_bad_boxes_taxonomy_and_leakage(self):
        for i in range(12): self.add(i)
        audit, plan = self.plan()
        for change in ('omit', 'category', 'box', 'duplicate'):
            bundle = copy.deepcopy(plan['exports'])
            split = next(s for s in SPLITS if bundle[s]['annotations'])
            if change == 'omit': bundle[split]['annotations'].pop()
            elif change == 'category': bundle[split]['annotations'][0]['category_id'] = 999
            elif change == 'box': bundle[split]['annotations'][0]['bbox'] = [39, 0, 5, 7]
            else:
                other = next(s for s in SPLITS if s != split)
                bundle[other]['images'].append(copy.deepcopy(bundle[split]['images'][0]))
            self.assertTrue(validate_coco(bundle, plan, audit['taxonomy']), change)

    def test_write_validate_source_integrity_and_edited_audit_is_not_trusted(self):
        for i in range(9): self.add(i)
        audit, plan = self.plan(); before = {p: sha256(p) for p in self.dataset.iterdir()}
        report = write_reports(audit, self.root/'audit')
        # Mark an audit ledger arbitrarily; fresh source reconstruction ignores it.
        ledger = report/'annotation_review.jsonl'
        ledger.write_text(ledger.read_text().replace('mapped_review_only', 'changed_review_text'), encoding='utf-8')
        self.assertEqual(fresh_audit(report)['annotations'], audit['annotations'])
        output = self.root/'export'
        write_preparation(audit, self.policy, DEFAULT_POLICY, plan, output, report)
        self.assertTrue(validate_preparation(output)['passed'])
        repeat = self.root/'repeat'
        write_preparation(audit, self.policy, DEFAULT_POLICY, make_plan(audit, self.policy), repeat, report)
        self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, {p.name: p.read_bytes() for p in repeat.iterdir()})
        self.assertFalse(any(p.suffix == '.png' for p in output.rglob('*')))
        self.assertEqual({p: sha256(p) for p in before}, before)
        with self.assertRaises(FileExistsError): write_preparation(audit, self.policy, DEFAULT_POLICY, plan, output)
        with self.assertRaises(ValueError): write_preparation(audit, self.policy, DEFAULT_POLICY, plan, self.dataset/'export')
        path = output/'train.json'; path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'checksum'): validate_preparation(output)

    def test_independent_validation_rejects_wrong_valid_category_even_with_updated_checksums(self):
        for i in range(9): self.add(i)
        audit, plan = self.plan(); output = self.root/'export'
        write_preparation(audit, self.policy, DEFAULT_POLICY, plan, output)
        split = next(s for s in SPLITS if plan['exports'][s]['annotations'])
        path = output/(split+'.json'); data = json.loads(path.read_text())
        data['annotations'][0]['category_id'] = 1  # Valid taxonomy ID, incorrect source mapping.
        path.write_text(json.dumps(data), encoding='utf-8')
        manifest_path = output/'preparation_manifest.json'; manifest = json.loads(manifest_path.read_text())
        manifest['artifacts'][path.name] = sha256(path); manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        result = validate_preparation(output)
        self.assertFalse(result['passed'])
        self.assertTrue(any('source records' in error for error in result['errors']))

    def test_source_change_or_orphan_blocks_export_and_settings_validated(self):
        self.add(0); audit, plan = self.plan()
        (self.dataset/'0.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'): make_plan(audit, self.policy)
        self.document['annotations'][0]['image_id'] = 900
        with self.assertRaisesRegex(ValueError, 'Unassignable'): make_plan(self.audit(), self.policy)
        for ratios in ({'train': .7, 'validation': .2, 'test': .2}, {'train': 1., 'validation': 0., 'test': 0.}):
            with self.assertRaises(ValueError): make_plan(audit, self.policy, ratios=ratios)

    def test_versioned_policy_cannot_relax_exclusion_safeguards(self):
        for key, value in (('box_repair', 'clip'), ('duplicate_annotation_id_repair', 'renumber'),
                           ('exact_duplicates', 'ignore'), ('version', ''), ('near_duplicates', []), ('default_seed', True)):
            policy = copy.deepcopy(self.policy); policy[key] = value
            path = self.root/'policy.json'; path.write_text(json.dumps(policy), encoding='utf-8')
            with self.assertRaises(ValueError): load_policy(path)


def CounterLike(assignments):
    return {s: sum(v == s for v in assignments.values()) for s in SPLITS}


if __name__ == '__main__':
    unittest.main()
