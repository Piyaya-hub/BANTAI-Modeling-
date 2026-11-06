"""Saved-only hierarchy invariants, duplicate accounting and GUI integration."""
import copy
import unittest
from scripts.taco_material_hierarchy import map_row,MAPS,TAXONOMY_VIEWS,greedy_matching,mapped_coco_metrics
from scripts.taco_diagnostic_nms import NMS_MODES,diagnostic_nms,io_min
from scripts.taco_comparison_gui import visible_rows,ComparisonWindow
from scripts.taco_object_gui import PerObjectWindow,object_view_review
from scripts.taco_object_review import annotation_review
from tests import test_taco_diagnostic_nms as nms_tests
row=nms_tests.row
Var=nms_tests.Var


class MaterialHierarchyTests(unittest.TestCase):
    def test_exact_mapping_metadata_immutability_and_reversibility(self):
        for scheme,ids in MAPS.items():
            for specific,expected in enumerate(ids):
                original=row(0,category=specific);before=copy.deepcopy(original)
                mapped=map_row(original,scheme)
                self.assertEqual(mapped['category_id'],expected);self.assertEqual(original,before)
                if scheme!='8-specific':
                    self.assertEqual(mapped['hazard_flag'],specific==0)
                    self.assertIn(mapped['specific_class'],mapped['display_label'])
                    self.assertEqual(map_row(mapped,'8-specific')['category_id'],specific)
                    self.assertEqual(map_row(mapped,scheme),mapped)
        self.assertEqual(map_row(row(0,category=7),'4-material')['material'],'polystyrene_foam')
        with self.assertRaises(ValueError):map_row(row(0,category=8),'4-material')
        with self.assertRaises(ValueError):map_row(row(0),'invalid')

    def test_remap_before_class_aware_suppression_and_matching(self):
        ann={'id':259,'image_id':777,'category_id':3,'bbox':[0,0,10,10]}
        rows=[row(0,.9,category=1),row(1,.8,category=3)]
        self.assertEqual(len(visible_rows(rows,.05,'all 900 (diagnostic)',NMS_MODES[2])),2)
        kept=visible_rows(rows,.05,'all 900 (diagnostic)',NMS_MODES[2],'4-material')
        self.assertEqual([r['query_id'] for r in kept],[0])
        result=object_view_review(ann,rows,kept,.05,'all 900 (diagnostic)',NMS_MODES[2],'4-material')
        self.assertTrue(result['correct_class_coverage']);self.assertEqual(result['best_same']['specific_class'],'plastic_bottle')
        self.assertFalse(result['best_same']['hazard_flag'])
        self.assertTrue(annotation_review(ann,rows,taxonomy='3-material')['correct_class_coverage'])

    def test_class_aware_containment_hazard_separation_and_exact_boundary(self):
        mode='class-aware IoU 0.50 + containment 0.80'
        rows=[row(0,.9,category=1,box=[2,0,100,100]),row(1,.8,category=3)]
        self.assertEqual(io_min(rows[0]['bbox'],rows[1]['bbox']),.8)
        self.assertEqual(len(diagnostic_nms(rows,mode)),2)
        mapped=[map_row(r,'4-material') for r in rows]
        self.assertEqual(len(diagnostic_nms(mapped,mode)),1)
        mapped[1]=map_row(row(1,category=0),'4-material')
        self.assertEqual(len(diagnostic_nms(mapped,mode)),2)
        self.assertEqual(len(diagnostic_nms(mapped,NMS_MODES[3])),1)

    def test_one_to_one_counts_duplicates_conflicts_ties_and_empty(self):
        anns=[{'id':1,'category_id':0,'bbox':[0,0,10,10]}, {'id':2,'category_id':0,'bbox':[0,0,10,10]}]
        single=[row(9)]
        self.assertEqual(sum(annotation_review(dict(a,image_id=777),single)['correct_class_coverage'] for a in anns),2)
        value=greedy_matching(anns,single)
        self.assertEqual((value['TP'],value['FP'],value['FN']),(1,0,1));self.assertEqual(value['assignments'][0]['annotation_id'],1)
        value=greedy_matching(anns,[row(2),row(1),row(3)])
        self.assertEqual((value['TP'],value['FP'],value['FN']),(2,1,0));self.assertEqual(value['assignments'][0]['query_id'],1)
        self.assertEqual(greedy_matching([],[])['F1'],0)
        self.assertEqual(greedy_matching(anns,[row(0,.049)])['FN'],2)

    def test_both_gui_renderers_switch_taxonomy_and_restore_saved_rows(self):
        factory=nms_tests.DiagnosticNMSTests()
        for cls in (ComparisonWindow,PerObjectWindow):
            view=factory.fixture_view(cls);view.taxonomy_view=Var(TAXONOMY_VIEWS[0])
            # Deliberate wrapper GT / bottle prediction at identical geometry.
            old_ds=view.dataset
            class DS:
                def __getitem__(self,index):
                    image,target=old_ds[index]
                    target['annotations'][0]['category_id']=3
                    return image,target
            view.dataset=DS()
            before=copy.deepcopy(view.models)
            try:
                for label in TAXONOMY_VIEWS+(TAXONOMY_VIEWS[0],):
                    view.taxonomy_view.set(label);view.render()
                    if cls is PerObjectWindow:
                        expected=label!=TAXONOMY_VIEWS[0]
                        self.assertTrue(all(v['correct_class_coverage']==expected for v in view.object_results.values()))
                self.assertEqual(view.models,before)
            finally:
                if view.current_image:view.current_image.close()
                view.figure.clear()

    def test_mapped_coco_perfect_empty_and_class_support(self):
        data={'info':{},'images':[{'id':777,'width':100,'height':100}],
              'annotations':[{'id':1,'image_id':777,'category_id':3,'bbox':[0,0,10,10],'area':100,'iscrowd':0}],
              'categories':[{'id':i,'name':str(i)} for i in range(8)]}
        predictions=[{'image_id':777,'category_id':1,'score':.9,'bbox':[0,0,10,10]}]
        mapped=mapped_coco_metrics(data,predictions,'4-material')
        self.assertAlmostEqual(mapped['AP'],1.);self.assertEqual(mapped['classes']['plastic_waste']['objects'],1)
        self.assertEqual(mapped_coco_metrics(data,[],'3-material')['AP'],0.)
