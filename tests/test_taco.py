"""Synthetic COCO audit/view fixtures; no external dataset or model required."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scripts.configuration import ROOT
from scripts.taco_dataset import (audit_dataset, bbox_issues, category_mapping, image_path,
                                  load_report, load_taxonomy, sha256, write_reports)
from scripts.view_taco import aligned, details_text, draw_annotations, open_verified_image, select_images


class TacoTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'work').mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=ROOT/'work')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.dataset = self.root/'dataset'; self.dataset.mkdir()
        with Image.new('RGB', (40, 30), 'blue') as image:
            image.save(self.dataset/'photo.png')
        self.source = self.dataset/'annotations.json'
        self.document = {
            'info': {'version': 'fixture'}, 'licenses': [{'id': 1}],
            'scene_annotations': [{'image_id': 1, 'category_id': 0}],
            'images': [{'id': 1, 'file_name': 'photo.png', 'width': 40, 'height': 30, 'license': 1},
                       {'id': 2, 'file_name': 'missing.png', 'width': 40, 'height': 30}],
            'categories': [{'id': 59, 'name': 'Cigarette', 'supercategory': 'Cigarette'},
                           {'id': 58, 'name': 'Unlabeled litter'},
                           {'id': 99, 'name': 'Glass bottle'}],
            'annotations': [
                {'id': 10, 'image_id': 1, 'category_id': 59, 'bbox': [2, 3, 5, 7],
                 'area': 30.5, 'segmentation': [[2, 3, 7, 3, 7, 10]], 'iscrowd': 0, 'custom': {'preserve': True}},
                {'id': 11, 'image_id': 1, 'category_id': 58, 'bbox': [15, 10, 5, 5]},
                {'id': 12, 'image_id': 1, 'category_id': 99, 'bbox': [25, 10, 5, 5]},
                {'id': 13, 'image_id': 2, 'category_id': 59, 'bbox': [2, 3, 5, 7]}]}

    def audit(self):
        self.source.write_text(json.dumps(self.document), encoding='utf-8')
        return audit_dataset(self.dataset)

    def test_exact_versioned_mapping_and_no_inference_category_change(self):
        taxonomy = load_taxonomy()
        self.assertEqual(len(taxonomy['labels']), 8)
        self.assertEqual(sum(len(c['taco_categories']) for c in taxonomy['labels']), 11)
        for category, name in [('Cigarette', 'cigarette'), ('Other plastic bottle', 'plastic_bottle'),
                               ('Clear plastic bottle', 'plastic_bottle'), ('Garbage bag', 'plastic_bag'),
                               ('Single-use carrier bag', 'plastic_bag'), ('Other plastic wrapper', 'plastic_wrapper'),
                               ('Crisp packet', 'plastic_wrapper'), ('Plastic film', 'plastic_film'),
                               ('Drink can', 'drink_can'), ('Disposable plastic cup', 'plastic_cup'),
                               ('Styrofoam piece', 'styrofoam_piece')]:
            self.assertEqual(category_mapping({'name': category}, taxonomy)[0], name)
        self.assertIsNone(category_mapping({'name': 'cigarette'}, taxonomy)[0])
        self.assertEqual(category_mapping({'name': 'Unlabeled litter'}, taxonomy)[2], 'unlabeled_requires_decision')
        self.assertEqual(category_mapping({'name': 'Glass bottle'}, taxonomy)[2], 'excluded_requires_decision')
        taxonomy['labels'][1]['taco_categories'].append('Cigarette')
        path = self.root/'taxonomy.json'; path.write_text(json.dumps(taxonomy))
        with self.assertRaisesRegex(ValueError, 'Duplicate'): load_taxonomy(path)

    def test_counts_missing_files_and_all_metadata_preserved(self):
        audit = self.audit(); counts = audit['summary']['counts']
        self.assertEqual(counts['image_status'], {'readable': 1, 'missing': 1})
        self.assertEqual(counts['usable_annotations_all_categories'], 3)
        self.assertEqual(counts['usable_mapped_annotations'], 1)
        self.assertEqual(counts['usable_excluded_annotations'], 2)
        self.assertEqual(counts['mapped_images_with_usable_excluded_annotations'], 1)
        row = audit['summary']['classes'][0]
        self.assertEqual((row['total_annotations'], row['usable_objects'], row['usable_unique_images']), (2, 1, 1))
        before = sha256(self.source); photo = sha256(self.dataset/'photo.png')
        out = write_reports(audit, self.root/'report')
        saved = [json.loads(line) for line in (out/'annotation_review.jsonl').read_text().splitlines()]
        self.assertEqual([r['original'] for r in saved], self.document['annotations'])
        self.assertEqual(json.loads((out/'source_metadata.json').read_text())['scene_annotations'], self.document['scene_annotations'])
        self.assertEqual(sha256(self.source), before); self.assertEqual(sha256(self.dataset/'photo.png'), photo)
        self.assertEqual(len(load_report(out)['annotations']), 4)
        with self.assertRaises(FileExistsError): write_reports(audit, out)
        with self.assertRaises(ValueError): write_reports(audit, self.dataset/'report')

    def test_malformed_bbox_and_references_continue_without_clipping(self):
        bad = [None, [1, 2, 3], [1, 2, 0, 4], [-1, 2, 3, 4], [39, 2, 5, 4], [True, 2, 3, 4], ['1', 2, 3, 4]]
        self.document['annotations'] = [dict(self.document['annotations'][0], id=i, bbox=b) for i, b in enumerate(bad)]
        self.document['annotations'] += [dict(self.document['annotations'][0], id=100, category_id=1234),
                                         dict(self.document['annotations'][0], id=101, image_id=[1]),
                                         dict(self.document['annotations'][0], id=102, image_id=1234)]
        audit = self.audit()
        self.assertEqual(audit['summary']['counts']['invalid_annotations'], len(bad)+3)
        self.assertEqual(audit['summary']['counts']['usable_annotations_all_categories'], 0)
        self.assertEqual(audit['annotations'][3]['original']['bbox'], [-1, 2, 3, 4])
        self.assertTrue(bbox_issues([1, float('nan'), 3, 4], 40, 30))
        self.assertTrue(bbox_issues([float('inf'), 1, 3, 4], 40, 30))
        self.assertEqual(bbox_issues([10**400, 1, 1.0, 4], 40, 30), ['bbox_endpoint_overflow'])

    def test_duplicate_ids_ambiguous_images_categories_and_structural_schema(self):
        for key in ('annotations', 'images', 'categories'):
            document = copy.deepcopy(self.document)
            self.document[key].append(copy.deepcopy(self.document[key][0]))
            audit = self.audit()
            self.assertTrue(audit['summary']['structural_issues'])
            self.assertFalse(audit['annotations'][0]['usable'])
            self.document = document
        self.document.pop('annotations')
        with self.assertRaisesRegex(ValueError, 'COCO requires'): self.audit()

    def test_corrupt_image_and_unsafe_paths_are_reported(self):
        (self.dataset/'bad.png').write_bytes(b'corrupt-image')
        for i, name in enumerate(('bad.png', '../escape.png', 'C:/outside.png', '/outside.png'), 3):
            self.document['images'].append({'id': i, 'file_name': name, 'width': 40, 'height': 30})
        audit = self.audit()
        self.assertEqual(audit['images'][2]['status'], 'unreadable')
        self.assertTrue(all(r['issues'] for r in audit['images'][3:]))
        for path in ('../escape.png', 'C:/outside.png', '/outside.png'):
            with self.assertRaises(ValueError): image_path(self.dataset, path)

    def test_coco_xywh_original_labels_excluded_and_table_on_real_canvas(self):
        audit = self.audit(); rows = audit['annotations'][:3]
        figure = Figure(); ax = figure.subplots()
        with open_verified_image(audit['images'][0]) as image:
            pixels = image.tobytes()
            with patch('PIL.Image.Image.save', side_effect=AssertionError('export')), patch.object(figure, 'savefig', side_effect=AssertionError('export')):
                draw_annotations(ax, image, rows, {'cigarette'}); FigureCanvasAgg(figure).draw()
            self.assertEqual(image.tobytes(), pixels)
        patch0 = ax.patches[0]
        self.assertEqual((patch0.get_x(), patch0.get_y(), patch0.get_width(), patch0.get_height()), (2, 3, 5, 7))
        self.assertEqual(len(ax.patches), 3)  # Sampling one class does not hide excluded objects.
        self.assertEqual([p.get_linestyle() for p in ax.patches], ['-', ':', '--'])
        labels = ' '.join(t.get_text() for t in ax.texts)
        self.assertIn('Cigarette -> cigarette', labels); self.assertIn('Unlabeled litter', labels)
        self.assertIn('EXCLUDED', labels)
        table = details_text(audit['images'][0], rows)
        self.assertIn('iscrowd=0', table); self.assertIn('area=30.5', table)
        self.assertIn('coco_xywh_pixels', audit['summary']['bbox_format'])

    def test_deterministic_sampling_and_empty_or_invalid_selections(self):
        audit = self.audit()
        self.assertEqual(select_images(audit, ['cigarette'], seed=42), [1])
        self.assertEqual(select_images(audit, ['excluded'], seed=42), [1])
        self.assertEqual(select_images(audit, ['unlabeled'], seed=42), [1])
        self.assertEqual(select_images(audit, ['plastic_bottle']), [])
        for args in ({'classes': ['unknown']}, {'image_ids': [2]}, {'sample_per_class': 0}):
            with self.assertRaises(ValueError): select_images(audit, **args)

    def test_explicit_exif_preview_and_source_hash_identity(self):
        exif = Image.Exif(); exif[274] = 6
        with Image.new('RGB', (30, 40), 'red') as image: image.save(self.dataset/'rotated.jpg', exif=exif)
        self.document['images'][0]['file_name'] = 'rotated.jpg'
        audit = self.audit(); row = audit['images'][0]
        self.assertFalse(aligned(row)); self.assertTrue(aligned(row, 'exif'))
        self.assertFalse(audit['annotations'][0]['usable'])
        self.assertEqual(select_images(audit, ['cigarette'], frame='exif'), [1])
        with self.assertRaises(ValueError): open_verified_image(row)
        with open_verified_image(row, 'exif') as image: self.assertEqual(image.size, (40, 30))
        (self.dataset/'rotated.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'): open_verified_image(row, 'exif')

    def test_cli_report_reuse_rejects_changed_metadata_and_no_model_import(self):
        from scripts.audit_taco import main as audit_main
        from scripts.view_taco import main as view_main
        self.audit()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit_main(['--dataset-root', str(self.dataset), '--output', str(self.root/'report')]), 1)
            self.assertEqual(view_main(['--report', str(self.root/'report'), '--classes', 'cigarette', '--list']), 0)
        self.source.write_text(self.source.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'changed'): load_report(self.root/'report')


if __name__ == '__main__':
    unittest.main()
