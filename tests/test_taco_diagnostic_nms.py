"""Deterministic saved-query NMS and real Agg rendering, no model/photos/GPU."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from PIL import Image
from scripts.taco_diagnostic_nms import NMS_MODES,diagnostic_nms,nms_with_proof,intersection_over_minimum_area
from scripts.taco_comparison_gui import ComparisonWindow,visible_rows
from scripts.taco_object_gui import PerObjectWindow,object_view_review
from scripts.taco_error_analysis import MODELS


def row(q,score=.8,category=0,box=None,retained=True):
    return {'query_id':q,'image_id':777,'score':score,'category_id':category,
            'bbox':box or [0,0,10,10],'valid_clipped_box':True,'retained':retained,
            'rank':q+1,'raw_phrase':'saved fragment'}


class Var:
    def __init__(self,value):self.value=value
    def get(self):return self.value
    def set(self,value):self.value=value


class DiagnosticNMSTests(unittest.TestCase):
    def test_containment_example_boundary_symmetry_and_old_modes_preserved(self):
        a=row(0,.9,box=[0,0,100,100]);b=row(1,.8,category=1,box=[10,10,10,10])
        for mode in NMS_MODES[:3]:self.assertEqual(len(diagnostic_nms([a,b],mode)),2)
        kept,proof=nms_with_proof([a,b],NMS_MODES[3])
        self.assertEqual(len(kept),1);self.assertEqual(proof[0]['iomin'],1.)
        self.assertEqual(intersection_over_minimum_area([0,0,10,10],[2,0,10,10]),.8)
        self.assertEqual(len(diagnostic_nms([row(0,.9,box=[2,0,100,100]),row(1,.8)],NMS_MODES[3])),1)
        self.assertEqual(len(diagnostic_nms([row(0,.9,box=[2.1,0,100,100]),row(1,.8)],NMS_MODES[3])),2)
        small=[982,79,53.6,57.4];large=[976.7,80.4,116.8,269.3]
        self.assertGreater(intersection_over_minimum_area(small,large),.8)
        self.assertEqual(intersection_over_minimum_area(small,large),intersection_over_minimum_area(large,small))
        self.assertEqual(intersection_over_minimum_area([0,0,0,10],large),0.)
        self.assertEqual(len(diagnostic_nms([row(0,.9,box=small),row(1,.8,box=large)],NMS_MODES[3])),1)
    def test_raw_mode_order_identity_and_no_input_mutation(self):
        rows=[row(8,.7),row(2,.9)];before=copy.deepcopy(rows)
        self.assertEqual(diagnostic_nms(rows),rows)
        self.assertIs(diagnostic_nms(rows)[0],rows[0])
        for mode in NMS_MODES:diagnostic_nms(rows,mode)
        self.assertEqual(rows,before)
        with self.assertRaises(ValueError):diagnostic_nms(rows,'unknown')

    def test_exact_inclusive_iou_boundary_and_cross_class_semantics(self):
        rows=[row(0,.9),row(1,.8,category=1,box=[0,0,5,10])]
        self.assertEqual([r['query_id'] for r in diagnostic_nms(rows,NMS_MODES[1])],[0])
        self.assertEqual([r['query_id'] for r in diagnostic_nms(rows,NMS_MODES[2])],[0,1])
        rows[1]['category_id']=0
        self.assertEqual(len(diagnostic_nms(rows,NMS_MODES[2])),1)
        kept,proof=nms_with_proof(rows,NMS_MODES[2])
        self.assertEqual(proof,[{'query_id':1,'suppressed_by_query_id':0,'iou':.5,'same_predicted_class':True}])

    def test_ties_are_score_then_query_id_and_suppression_is_not_transitive(self):
        a,b,c=row(0,.9),row(1,.8,box=[3,0,10,10]),row(2,.7,box=[6,0,10,10])
        for mode in NMS_MODES[1:]:
            self.assertEqual([r['query_id'] for r in diagnostic_nms([c,b,a],mode)],[0,2])
            self.assertEqual([r['query_id'] for r in diagnostic_nms([row(9),row(1)],mode)],[1])
            self.assertEqual([r['query_id'] for r in diagnostic_nms([row(1),row(9)],mode)],[1])

    def test_source_threshold_and_invalid_filtering_precede_nms_and_topk(self):
        rows=[row(0,.9,retained=False),row(1,.8),dict(row(2,1.),valid_clipped_box=False),row(3,.05,box=[30,0,10,10])]
        self.assertEqual([r['query_id'] for r in visible_rows(rows,.05,'official top100',NMS_MODES[1])],[1,3])
        self.assertEqual([r['query_id'] for r in visible_rows(rows,.05,'all 900 (diagnostic)',NMS_MODES[1])],[0,3])
        self.assertEqual(visible_rows(rows,.85,'official top100',NMS_MODES[2]),[])
        self.assertEqual([r['query_id'] for r in visible_rows(rows,.05,'all 900 (diagnostic)',NMS_MODES[1])[:2]],[0,3])

    def fixture_view(self,cls):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        view=object.__new__(cls)
        ann={'id':9,'image_id':777,'category_id':0,'bbox':[0,0,10,10]}
        class Dataset:
            def __getitem__(self,index):return Image.new('RGB',(80,60),'white'),{'image_id':777,'annotations':[ann]}
        rows=[row(0,.9,category=1),row(1,.8),row(2,.07,box=[40,20,10,10])]
        view.models={n:{777:copy.deepcopy(rows)} for n in MODELS};view.dataset=Dataset();view.index={777:0}
        view.threshold=Var('.05');view.top_k=Var('10');view.font=Var('5');view.source=Var('all 900 (diagnostic)')
        view.nms_mode=Var(NMS_MODES[0]);view.selected=Var('777');view.current_image=None
        view.figure=Figure(figsize=(12,8));view.canvas=FigureCanvasAgg(view.figure)
        view.axes=list(view.figure.subplots(2,2).flat);view.tables={n:MagicMock() for n in ('ground_truth',)+MODELS}
        view.toolbar=MagicMock();view.status=Var('');view.sample=[{'image_id':777,'reasons':[]}]
        for table in view.tables.values():table.get_children.return_value=[]
        if cls is PerObjectWindow:
            view.object_mode=True;view.object_id=Var('9');view.object_zoom=Var(False)
            view.object_selector={};view.object_table=MagicMock();view.object_table.get_children.return_value=[]
        return view

    def test_both_gui_renderers_consume_nms_control_and_restore_raw_view(self):
        for cls in (ComparisonWindow,PerObjectWindow):
            view=self.fixture_view(cls);before=copy.deepcopy(view.models)
            try:
                for mode in (NMS_MODES[0],NMS_MODES[2],NMS_MODES[1],NMS_MODES[0]):
                    with self.subTest(view=cls.__name__,mode=mode):
                        view.nms_mode.set(mode)
                        for table in view.tables.values():table.insert.reset_mock()
                        view.render()
                        self.assertIn(mode,view.status.get())
                        if cls is ComparisonWindow:
                            expected=2 if mode==NMS_MODES[1] else 3
                            self.assertTrue(all(len(ax.patches)==expected for ax in view.axes[1:]))
                            self.assertTrue(all(view.tables[n].insert.call_count==expected for n in MODELS))
                        else:
                            self.assertTrue(all(v['correct_class_coverage']==(mode!=NMS_MODES[1]) for v in view.object_results.values()))
                            self.assertTrue(all(v['same_class_coverage_all900_no_score_floor'] for v in view.object_results.values()))
                self.assertEqual(view.models,before)
            finally:
                if view.current_image is not None:view.current_image.close()
                view.figure.clear()

    def test_per_object_nms_is_global_before_gt_matching_and_preserves_raw_diagnostics(self):
        ann={'id':9,'category_id':0,'image_id':777,'bbox':[0,0,10,10]}
        rows=[row(0,.9,category=1),row(1,.8)]
        kept=visible_rows(rows,.05,'all 900 (diagnostic)',NMS_MODES[1])
        result=object_view_review(ann,rows,kept,.05,'all 900 (diagnostic)',NMS_MODES[1])
        self.assertFalse(result['correct_class_coverage']);self.assertTrue(result['any_class_coverage'])
        self.assertEqual(result['stored_queries'],2)
        self.assertTrue(result['same_class_coverage_all900_no_score_floor'])
