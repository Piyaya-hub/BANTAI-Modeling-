"""Independent per-object coverage, filtering and explicit CLI guards."""
import contextlib,io,copy,unittest
from scripts.taco_object_review import annotation_review,eligible_rows,object_limits
from scripts.taco_visual_review import main


def row(q=0,category=0,score=.1,box=None,retained=True):
    return {'query_id':q,'category_id':category,'score':score,'bbox':box or [0,0,10,10],
            'valid_clipped_box':True,'rank':q+1,'retained':retained,'raw_phrase':'raw fragment'}


class PerObjectTests(unittest.TestCase):
    def setUp(self):self.ann={'id':9,'image_id':777,'category_id':0,'bbox':[0,0,10,10]}

    def test_best_iou_does_not_mean_highest_score_or_first_five(self):
        rows=[row(q=i,score=.9,box=[50,50,5,5]) for i in range(5)]
        rows += [row(q=6,category=3,score=.2),row(q=7,score=.05,box=[0,0,5,10])]
        before=copy.deepcopy(rows);result=annotation_review(self.ann,rows)
        self.assertEqual(result['best_any']['query_id'],6)
        self.assertEqual(result['best_any']['iou'],1)
        self.assertEqual(result['best_same']['query_id'],7)
        self.assertEqual(result['best_same']['iou'],.5)
        self.assertTrue(result['correct_class_coverage']);self.assertTrue(result['any_class_coverage'])
        self.assertEqual(rows,before)

    def test_score_floor_and_top100_exclusion_are_distinct(self):
        rows=[row(score=.049),row(q=101,score=.06,retained=False)]
        self.assertTrue(annotation_review(self.ann,rows)['correct_class_coverage'])
        official=annotation_review(self.ann,rows,source='official top100')
        self.assertFalse(official['correct_class_coverage'])
        self.assertTrue(official['same_class_coverage_all900_no_score_floor'])
        self.assertTrue(annotation_review(self.ann,rows,0,'official top100')['correct_class_coverage'])

    def test_wrong_class_absence_and_low_iou_are_not_detection_impossibility(self):
        wrong=annotation_review(self.ann,[row(category=4)])
        self.assertEqual(wrong['kind'],'localized_without_correct_class_at_threshold')
        self.assertIsNone(wrong['best_same'])
        no_overlap=annotation_review(self.ann,[row(box=[20,20,10,10])])
        self.assertIsNone(no_overlap['best_same']);self.assertIsNone(no_overlap['best_any'])
        low=annotation_review(self.ann,[row(box=[5,0,10,10])])
        self.assertAlmostEqual(low['best_same']['iou'],1/3);self.assertFalse(low['correct_class_coverage'])

    def test_duplicate_candidates_and_shared_gt_are_nonexclusive(self):
        rows=[row(),row(q=2)]
        result=annotation_review(self.ann,rows)
        self.assertEqual(result['qualifying_same_class_candidates'],2)
        self.assertTrue(annotation_review(dict(self.ann,id=10),rows)['correct_class_coverage'])

    def test_ties_zero_score_and_invalid_candidates(self):
        rows=[row(q=2,score=0),row(q=1,score=0),dict(row(q=3),valid_clipped_box=False)]
        result=annotation_review(self.ann,rows,0)
        self.assertEqual(result['best_any']['query_id'],1);self.assertEqual(result['eligible_queries'],2)
        for score in (-1,1.1,float('nan')):
            with self.assertRaises(ValueError):eligible_rows(rows,score)
        with self.assertRaises(ValueError):eligible_rows(rows,source='unknown')

    def test_zoom_limits_keep_original_frame_and_minimum_context(self):
        x,y=object_limits(self.ann,100,80)
        self.assertEqual(x,(0,34));self.assertEqual(y,(34,0))

    def test_per_object_flag_requires_explicit_saved_gui(self):
        for args in (['--per-object'],['--per-object','--gui'],['--per-object','--saved-review','m','--run-choice','x']):
            with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(args)


if __name__=='__main__':unittest.main()
