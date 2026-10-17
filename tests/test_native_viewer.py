"""Software drawing/lifecycle checks; Agg and mock Tk are not visual evidence."""

import copy
import io
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scripts.configuration import load_config
from scripts.results import candidate,make_result
from scripts.viewer import draw_overlays,show_image,show_synthetic,NativeDisplayError,DisplayBatch
from scripts.gui_viewer import section_data,table_text,wheel_steps,open_results_window,covers_border


class NativeViewerTests(unittest.TestCase):
    def result(self,empty=False):
        config=load_config()
        config.update(threshold=.35,text_threshold=.25)
        source=dict(image_id="fixture.png",sha256="a"*64,width=60,height=40,
                    coordinate_frame="exif_oriented_original")
        ds=[] if empty else [candidate("q0","a garbage pile",[2.25,3.5,51.75,35.25],.86123456789,60,40,
                                      category="garbage_pile",mapping_reason="exact_alias"),
                            candidate("q1","a group",[4,4,20,20],.8,60,40),
                            candidate("bad","invalid",[0,0,0,0],.9,60,40),
                            candidate("below","below",[1,1,5,5],.1,60,40)]
        duplicate=candidate("duplicate","a group",[4,4,20,20],.7,60,40)
        duplicate.update(state="duplicate",duplicate_of="q1")
        if not empty:ds.append(duplicate)
        return make_result("synthetic",source,config,detections=ds,inference_completed=True,
                           library_version="4.51.3",execution={"actual_device":"cpu",
                           "environment_manifest":"synthetic","preprocessing_manifest":"synthetic"})

    def test_original_pixels_boxes_and_dynamic_phrase_labels(self):
        result=self.result();before=copy.deepcopy(result)
        figure=Figure(figsize=(12,8));canvas=FigureCanvasAgg(figure);ax=figure.subplots()
        with Image.new("RGB",(60,40),"blue") as image:
            pixels=image.tobytes()
            with patch("PIL.Image.Image.save",side_effect=AssertionError("export")),patch("matplotlib.figure.Figure.savefig",side_effect=AssertionError("export")):
                draw_overlays(ax,image,result);canvas.draw()
            self.assertEqual(image.tobytes(),pixels)
        self.assertEqual(tuple(ax.images[0].get_extent()),(0,60,40,0))
        self.assertEqual(ax.images[0].get_array().shape,(40,60,3))
        self.assertEqual(len(ax.patches),2)
        rect=ax.patches[0]
        self.assertEqual((rect.get_x(),rect.get_y(),rect.get_width(),rect.get_height()),(2.25,3.5,49.5,31.75))
        layouts=[artist.last_layout for artist in ax.artists if artist.last_layout]
        self.assertTrue(layouts)
        for layout in layouts:
            x0,y0,x1,y1=layout["bounds"];left,right=layout["box_span"]
            self.assertGreaterEqual(x0,left);self.assertLessEqual(x1,right)
            self.assertGreater(y0,layout["box_top"]+2)
            self.assertGreaterEqual(layout["font_size"],8)
            self.assertLessEqual(layout["font_size"],8.5)
            self.assertNotIn("86.1",layout["text"])
            self.assertNotIn("garbage_pile",layout["text"])
        self.assertIn("a garbage pile"," ".join(layouts[0]["text"].split()))
        self.assertEqual(result,before)

    def test_edge_and_narrow_boxes_never_clip_or_shrink_below_readable_size(self):
        result=self.result();result["detections"]=[
            candidate("edge","discarded item with incomplete words",[0,0,25,30],.8,60,40),
            candidate("tiny","a very long incomplete phrase",[59.999,0,60,1],.8,60,40)]
        figure=Figure(figsize=(10,7));canvas=FigureCanvasAgg(figure);ax=figure.subplots()
        with Image.new("RGB",(60,40)) as image:draw_overlays(ax,image,result)
        canvas.draw()
        edge,tiny=ax.artists
        self.assertIsNotNone(edge.last_layout)
        self.assertIsNone(tiny.last_layout)
        self.assertIn('"a very long incomplete phrase"',str(section_data(result)["summary"]))
        bounds=edge.last_layout["bounds"]
        self.assertGreaterEqual(bounds[0],ax.bbox.x0);self.assertLessEqual(bounds[3],ax.bbox.y1)
        self.assertEqual(ax.get_title(),"")
        ax.set_xlim(0,30);ax.set_ylim(35,-12);canvas.draw()
        self.assertEqual(tuple(ax.images[0].get_extent()),(0,60,40,0))
        self.assertTrue(ax.xaxis.majorTicks[0].label2.get_visible())
        self.assertTrue(ax.yaxis.majorTicks[0].label2.get_visible())

    def test_empty_image_and_all_four_sections_have_honest_messages(self):
        result=self.result(empty=True);data=section_data(result)
        self.assertEqual(data["summary"],[]);self.assertEqual(data["tokens"],[])
        figure=Figure();ax=figure.subplots()
        with Image.new("RGB",(60,40)) as image:draw_overlays(ax,image,result)
        FigureCanvasAgg(figure).draw()
        self.assertEqual(len(ax.images),1);self.assertEqual(len(ax.patches),0)
        self.assertEqual(len(ax.artists),0)

    def test_saved_prompt_threshold_model_device_and_passing_tokens(self):
        result=self.result();result["query_diagnostics"]={
            "tokens":[{"token":token,"special":special,"position":i,"token_id":100+i}
                      for i,(token,special) in enumerate([("[CLS]",True),("litter",False),("below",False),("[PAD]",False)])],
            "queries":[{"id":d["id"],"threshold_pass":True,"numeric_diagnostics":[],"invalid_token_positions":[],
                        "model_score":d["model_score"],"token_scores":[.99,.8,.1,.99],"token_threshold_pass":[True,True,False,True]}
                       for d in result["detections"] if d["id"] in ("q0","q1")]}
        before=copy.deepcopy(result);data=section_data(result)
        self.assertEqual([r[0] for r in data["summary"]],["q0","q1"])
        self.assertEqual([r[:2] for r in data["tokens"]],[("q0",'"litter"'),("q1",'"litter"')])
        self.assertEqual(data["summary"][1][1:3],('"a group"',"unmapped"))
        config=dict(data["configuration"])
        self.assertEqual(config["Prompt"],result["inference"]["prompt"])
        self.assertEqual(config["Box threshold"],"0.35")
        self.assertEqual(config["Revision"],result["model"]["revision"])
        self.assertEqual(config["Device used"],"cpu")
        self.assertEqual(result,before)

    def test_batch_owns_copies_and_shows_one_window_for_multiple_images(self):
        batch=DisplayBatch()
        with Image.new("RGB",(60,40)) as image:
            show_image(image,self.result(),batch=batch)
            show_image(image,self.result(empty=True),batch=batch)
        with patch("scripts.gui_viewer.open_results_window") as window:
            batch.show();window.assert_called_once()
            self.assertEqual(len(window.call_args.args[0]),2)
            self.assertEqual(window.call_args.args[0][0][0].getpixel((0,0)),(0,0,0))
        batch.close();self.assertFalse(batch.entries)

    def test_batch_failure_and_cleanup_do_not_mutate_predictions(self):
        result=self.result();before=copy.deepcopy(result)
        with Image.new("RGB",(60,40)) as image,patch("scripts.gui_viewer.open_results_window",side_effect=RuntimeError("fixture failure")),self.assertRaisesRegex(NativeDisplayError,"fixture failure"):
            show_image(image,result)
        self.assertEqual(result,before)

    def test_mock_tab_order_scrolling_copyable_text_and_clean_closure(self):
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        import contextlib
        root=MagicMock();root.winfo_screenwidth.return_value=1920;root.winfo_screenheight.return_value=1080
        callbacks={};bindings={}
        root.protocol.side_effect=lambda name,callback:callbacks.update({name:callback})
        root.bind.side_effect=lambda name,callback:bindings.update({name:callback})
        root.mainloop.side_effect=lambda:callbacks["WM_DELETE_WINDOW"]()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("tkinter.Tk",return_value=root))
            widgets={name:stack.enter_context(patch(name)) for name in ("tkinter.ttk.Notebook","tkinter.ttk.Frame","tkinter.ttk.Label","tkinter.ttk.Scrollbar","tkinter.Canvas","tkinter.Text","matplotlib.backends.backend_tkagg.FigureCanvasTkAgg")}
            for name in ("__init__","update","pack"):stack.enter_context(patch.object(NavigationToolbar2Tk,name,return_value=None))
            with Image.new("RGB",(60,40)) as image:open_results_window([(image,self.result()),(image,self.result(empty=True))])
            headings=[call.kwargs["text"] for call in widgets["tkinter.ttk.Label"].call_args_list if call.kwargs["text"] in ("Image #1","Image #2","Detection Summary","Text-Token Details","Configuration")]
            self.assertEqual(headings,["Image #1","Detection Summary","Text-Token Details","Configuration","Image #2","Detection Summary","Text-Token Details","Configuration"])
            notebook=widgets["tkinter.ttk.Notebook"].return_value
            self.assertEqual(notebook.add.call_count,1)
            self.assertTrue(widgets["tkinter.Text"].return_value.configure.called)
            from types import SimpleNamespace
            bindings["<MouseWheel>"](SimpleNamespace(delta=-120,state=0))
            widgets["tkinter.Canvas"].return_value.yview_scroll.assert_called_with(1,"units")
            bindings["<MouseWheel>"](SimpleNamespace(delta=120,state=1))
            widgets["tkinter.Canvas"].return_value.xview_scroll.assert_called_with(-1,"units")
        root.destroy.assert_called_once()

    def test_window_init_and_callback_failures(self):
        with patch("tkinter.Tk",side_effect=RuntimeError("no desktop")),self.assertRaisesRegex(RuntimeError,"no desktop"):
            open_results_window([])
        root=MagicMock();root.winfo_screenwidth.return_value=1000;root.winfo_screenheight.return_value=800
        root.mainloop.side_effect=lambda:root.report_callback_exception(RuntimeError,RuntimeError("callback fixture"),None)
        import contextlib
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("tkinter.Tk",return_value=root))
            for name in ("tkinter.ttk.Notebook","tkinter.ttk.Frame","tkinter.ttk.Scrollbar","tkinter.Canvas"):stack.enter_context(patch(name))
            with self.assertRaisesRegex(NativeDisplayError,"callback fixture"):open_results_window([])
        root.destroy.assert_called_once()

    def test_long_selectable_table_content_and_wheel_units(self):
        phrase="incomplete "*100
        text=table_text(("Query","Phrase"),[("q0",phrase)])
        self.assertIn(phrase.rstrip(),text)
        self.assertEqual(wheel_steps(120),-1);self.assertEqual(wheel_steps(-240),2)
        self.assertEqual(wheel_steps(1),-1);self.assertEqual(wheel_steps(0),0)

    def test_label_backgrounds_cannot_cover_other_box_outlines(self):
        result=self.result();figure=Figure(figsize=(10,7));ax=figure.subplots()
        with Image.new("RGB",(60,40)) as image:draw_overlays(ax,image,result)
        FigureCanvasAgg(figure).draw()
        d=result["detections"][0]
        left,top=ax.transData.transform(d["box_xyxy"][:2])
        self.assertTrue(covers_border((left-4,top-4,left+4,top+4),ax,[d]))
        self.assertFalse(covers_border((left+10,top+10,left+30,top+30),ax,[d]))

    def test_photo_shaped_narrow_region_still_displays_complete_phrase(self):
        # Software geometry regression for the observed missing q2 label.
        result=self.result();result["source"].update(width=3000,height=4000)
        result["detections"]=[candidate(ident,phrase,box,.8,3000,4000) for ident,phrase,box in (
            ("q0","a group",[489.86,968.05,2331.79,2404.26]),
            ("q2","discarded item",[1325.28,2815.03,1779.48,3167.18]),
            ("q4","discarded item",[504.78,1322.53,1557.03,2242.48]))]
        figure=Figure(figsize=(11,7.5),dpi=100,layout="constrained");ax=figure.subplots()
        with Image.new("RGB",(3000,4000)) as image:draw_overlays(ax,image,result)
        FigureCanvasAgg(figure).draw()
        self.assertTrue(all(a.last_layout for a in ax.artists))
        label=ax.artists[1].last_layout
        self.assertIn("discarded item","".join(label["text"].splitlines()))
        self.assertGreaterEqual(label["font_size"],8)
        self.assertFalse(covers_border(label["bounds"],ax,result["detections"]))

    def test_synthetic_dispatch_and_legacy_config_migrate(self):
        from scripts.detect_images import main
        from scripts.configuration import validate_config
        import contextlib
        config=load_config();config["display"]["backend"]="terminal"
        self.assertEqual(validate_config(config)["display"]["layout"],"scrollable_tab")
        with patch("scripts.viewer.show_synthetic") as show,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--mode","display-check"]),0)
        show.assert_called_once()
