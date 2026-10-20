"""Generated diagnostics only. Native desktop/photo evidence is recorded separately."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock
from types import SimpleNamespace

from PIL import Image

from scripts.taco_error_analysis import MODELS,iou,saved_rows,diagnose,analyze,representative_sample
from scripts.taco_visual_review import load_review,epoch_score_text,main,show_review
from scripts.taco_comparison_gui import visible_rows,draw_panel,ComparisonWindow
from scripts.training_smoke import digest


def row(query=0,category=0,score=.1,box=None,retained=True):
    return {'image_id':1,'query_id':query,'category_id':category,'score':score,
            'bbox':box or [10,10,10,10],'raw_phrase':'plastic wrap',
            'valid_clipped_box':True,'retained':retained,'rank':query+1}


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.ann={'id':1,'image_id':1,'category_id':0,'bbox':[10,10,10,10],'area':100,'iscrowd':0}

    def test_independent_iou_known_values(self):
        self.assertEqual(iou([0,0,10,10],[0,0,10,10]),1.)
        self.assertEqual(iou([0,0,10,10],[20,20,10,10]),0.)
        self.assertAlmostEqual(iou([0,0,10,10],[5,0,10,10]),1/3)

    def test_threshold_hidden_is_distinct_from_missing_box(self):
        low=diagnose([self.ann],[row(score=.1)],.25)
        self.assertEqual(low['annotation_counts'],{'threshold_hidden':1})
        self.assertEqual(low['annotations'][0]['best_compatible_score_top100'],.1)
        self.assertEqual(diagnose([self.ann],[row(score=.1)],.05)['annotation_counts'],{'matched':1})
        miss=diagnose([self.ann],[row(box=[90,90,5,5])],.05)
        self.assertEqual(miss['annotation_counts'],{'no_substantial_overlap':1})

    def test_rank_exclusion_is_not_threshold_suppression(self):
        value=diagnose([self.ann],[row(score=.9,retained=False)],.05)
        self.assertEqual(value['annotation_counts'],{'outside_top100':1})

    def test_wrong_class_and_localization_separated(self):
        wrong=diagnose([self.ann],[row(category=4)],.05)
        self.assertEqual(wrong['annotation_counts'],{'wrong_class_only':1})
        self.assertEqual(wrong['prediction_counts'],{'wrong_class':1})
        poor=diagnose([self.ann],[row(box=[15,10,10,10])],.05)
        self.assertEqual(poor['annotation_counts'],{'poor_localization_candidate':1})
        self.assertEqual(poor['prediction_counts'],{'localization_overlap_010_to_050':1})

    def test_duplicates_and_shared_annotation_candidate_do_not_invent_membership(self):
        result=diagnose([self.ann],[row(),row(query=1,score=.09)])
        self.assertEqual(result['prediction_counts'],{'matched':1,'duplicate':1})
        peer=dict(self.ann,id=2)
        result=diagnose([self.ann,peer],[row()])
        self.assertEqual(result['annotation_counts'],{'matched':1,'assignment_conflict':1})

    def test_empty_annotations_are_not_automatic_visual_false_positives(self):
        result=diagnose([],[row()])
        self.assertEqual(result['prediction_counts'],{'unmatched_prediction':1})
        self.assertEqual(result['annotation_counts'],{})

    def evaluation(self):
        data={'images':[{'id':1,'width':100,'height':100}],'annotations':[self.ann]}
        raw=[{k:v for k,v in row(query=q,score=.9-q*.001).items() if k not in ('rank','retained')} for q in range(101)]
        predictions=[{k:r[k] for k in ('image_id','category_id','score','bbox')} for r in raw[:100]]
        return data,{'raw_queries':raw,'predictions':predictions}

    def test_saved_query_identity_exact_membership_and_raw_phrase_preserved(self):
        data,value=self.evaluation();before=copy.deepcopy(value)
        rows=saved_rows(data,value,queries=101)[1]
        self.assertTrue(rows[99]['retained']);self.assertFalse(rows[100]['retained'])
        self.assertEqual(rows[0]['raw_phrase'],'plastic wrap')
        self.assertEqual(rows[0]['category_id'],0)
        self.assertEqual(value,before)

    def test_saved_result_identity_coordinate_and_prediction_tampering_rejected(self):
        for change in ('image','query','category','box','prediction','rank'):
            data,value=self.evaluation()
            if change=='image':value['raw_queries'][0]['image_id']=2
            elif change=='query':value['raw_queries'][0]['query_id']=1
            elif change=='category':value['raw_queries'][0]['category_id']=8
            elif change=='box':value['raw_queries'][0]['bbox']=[95,95,10,10]
            elif change=='prediction':value['predictions'][0]['score']=.2
            else:value['raw_queries'].reverse()
            with self.subTest(change=change),self.assertRaises(ValueError):saved_rows(data,value,queries=101)

    def test_visual_filters_preserve_zero_scores_and_official_membership(self):
        rows=[row(score=0),row(query=1,score=.03,retained=False)]
        self.assertEqual(len(visible_rows(rows,0,'official top100')),1)
        self.assertEqual(len(visible_rows(rows,0,'all 900 (diagnostic)')),2)
        self.assertEqual(visible_rows(rows,.25,'official top100'),[])
        with self.assertRaises(ValueError):visible_rows(rows,-.1,'official top100')

    def test_test_split_rejects_before_files_models_or_photos(self):
        with patch('scripts.taco_visual_review.digest',side_effect=AssertionError('no reads')):
            with self.assertRaisesRegex(ValueError,'validation only'):
                load_review(None,None,None,None,None,split='test')

    def test_changed_saved_record_anchor_rejects_before_checkpoint_or_photo_access(self):
        for changed in ('record','baseline'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);record=root/'record.json';baseline=root/'baseline.json';policy=root/'anchors.json'
                record.write_text('{}');baseline.write_text('{}')
                policy.write_text(json.dumps({'schema_version':'bantai.visual-review-anchors.v1',
                    'purpose':'saved_validation_replay_no_inference','step1c_record_sha256':digest(record),
                    'baseline_record_sha256':digest(baseline)}))
                (record if changed=='record' else baseline).write_text('{"changed":true}')
                with (patch('scripts.taco_visual_review.REPLAY_POLICY',policy),
                      patch('scripts.taco_visual_review.verify_evaluation_checkpoint',side_effect=AssertionError('no checkpoint reads')),
                      patch('PIL.Image.open',side_effect=AssertionError('no photos'))):
                    with self.assertRaisesRegex(ValueError,'anchor mismatch'):
                        load_review(record,baseline,None,None,None)

    def test_sampling_is_reproducible_covers_classes_and_keeps_human_pending(self):
        data={'images':[{'id':i,'width':100,'height':100} for i in range(1,9)],
              'annotations':[dict(self.ann,id=i,image_id=i,category_id=i-1) for i in range(1,9)]}
        models={name:{i:[dict(row(category=i-1),image_id=i)] for i in range(1,9)} for name in MODELS}
        diagnostic=analyze(data,models);sample=representative_sample(data,diagnostic,16)
        self.assertEqual(sample,representative_sample(data,diagnostic,16))
        self.assertEqual({r['image_id'] for r in sample},set(range(1,9)))
        self.assertTrue(all(r['human_visual_confirmation']=='pending' for r in sample))

    def test_render_coordinates_aspect_labels_and_no_image_exports(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        figure=Figure(figsize=(6,4));canvas=FigureCanvasAgg(figure);ax=figure.subplots()
        image=Image.new('RGB',(640,320),'white')
        try:
            with patch.object(image,'save',side_effect=AssertionError('no exports')),patch.object(figure,'savefig',side_effect=AssertionError('no exports')):
                draw_panel(ax,image,[row(box=[632,316,8,4])])
                canvas.draw()
            self.assertEqual(tuple(ax.images[0].get_extent()),(0,640,320,0))
            patch_box=ax.patches[0]
            self.assertEqual(patch_box.get_xy(),(632,316))
            self.assertEqual((patch_box.get_width(),patch_box.get_height()),(8,4))
            self.assertEqual(ax.get_aspect(),1.)
            self.assertEqual(ax.artists[0].raw_phrase,'cigarette s=0.1000')
        finally:image.close();figure.clear()

    def test_callback_failure_closes_stale_view_and_retains_error(self):
        view=object.__new__(ComparisonWindow);closed=[]
        view.close=lambda:closed.append(True)
        view.callback_error(ValueError,ValueError('Source photograph hash mismatch'),None)
        self.assertEqual(closed,[True])
        self.assertIn('Source photograph hash mismatch',view.error)

    def test_terminal_epoch_scores_use_corrected_metrics_without_gui_or_analysis_export(self):
        metrics={'AP':.12,'AP50':.2,'AP75':.1,'AR100':.3}
        scores={'quantiles':{'median':.03,'p99':.07,'max':.18},'queries_ge_025':0,'queries':42300}
        record={'pretrained_corrected':{'metrics':metrics,'mean_loss_per_image':10.,'scores':scores},
            'models':{'epoch-01':{'metrics':metrics,'corrected_target_loss':9.,'historical_target_loss':888.,'scores':scores}}}
        output=io.StringIO()
        with (patch('scripts.taco_visual_review.load_review') as verification,
              patch('scripts.taco_visual_review.strict_load',return_value=record),
              patch('scripts.taco_visual_review.prepare_review',side_effect=AssertionError('no report export')),
              patch('scripts.taco_comparison_gui.ComparisonWindow',side_effect=AssertionError('no GUI')),
              patch('PIL.Image.open',side_effect=AssertionError('no photographs')),
              contextlib.redirect_stdout(output)):
            self.assertEqual(main([]),0)
        verification.assert_called_once()
        text=output.getvalue()
        self.assertIn('epoch-01',text);self.assertIn('AP=0.120000',text)
        self.assertIn('max=0.180000',text);self.assertIn('9.000000',text)
        self.assertNotIn('888.',text)

    def test_gui_cli_dispatches_saved_review_and_image_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared=(object(),object(),{},Path(tmp))
            with (patch('scripts.taco_visual_review.prepare_review',return_value=prepared),
                  patch('scripts.taco_visual_review.show_review') as show,
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(main(['--gui','--image-id','42']),0)
            show.assert_called_once_with(*prepared,42)

    def test_conflicting_modes_reject_before_loading(self):
        for args in (['--gui','--analyze-only'],['--image-id','42']):
            with (self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()),
                  patch('scripts.taco_visual_review.prepare_review') as prepare,
                  self.assertRaises(SystemExit)):
                main(args)
            prepare.assert_not_called()

    def test_gui_lifecycle_and_invalid_id_preserve_separate_status(self):
        for failure in (None,'callback failure','startup failure','invalid id'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                dataset=SimpleNamespace(images=[{'id':42}],photo_reads=1)
                record={'representative_sample':[]}
                window=MagicMock()
                window.error='callback failure' if failure=='callback failure' else None
                with patch('scripts.taco_comparison_gui.ComparisonWindow',
                           return_value=window,
                           side_effect=ValueError('startup failure') if failure=='startup failure' else None) as factory:
                    if failure:
                        with self.assertRaises((RuntimeError,ValueError)):
                            show_review(dataset,{},record,Path(tmp),999 if failure=='invalid id' else 42)
                    else:
                        show_review(dataset,{},record,Path(tmp),42)
                        window.selected.set.assert_called_once_with('42')
                        window.scope.set.assert_called_once_with('all validation')
                        window.set_scope.assert_called_once()
                    if failure=='invalid id':factory.assert_not_called()
                    if failure in (None,'callback failure'):window.close.assert_called_once()
                saved=json.loads((Path(tmp)/'analysis.json').read_text())
                self.assertEqual(saved['display_status'],'error' if failure else 'closed')
                self.assertEqual(saved['source_verified_photo_reads'],1)


if __name__=='__main__':unittest.main()
