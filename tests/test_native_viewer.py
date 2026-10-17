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
from scripts.viewer import draw_overlays,show_image,show_synthetic,open_native_window,NativeDisplayError,native_details


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

    def test_full_source_pixel_alignment_labels_and_no_exports(self):
        result=self.result(); before=copy.deepcopy(result)
        figure=Figure();canvas=FigureCanvasAgg(figure);ax=figure.subplots()
        with Image.new("RGB",(60,40),"blue") as image:
            pixels=image.tobytes()
            with patch("PIL.Image.Image.save",side_effect=AssertionError("no image files")), \
                    patch("matplotlib.figure.Figure.savefig",side_effect=AssertionError("no exports")):
                draw_overlays(ax,image,result);canvas.draw()
            self.assertEqual(image.tobytes(),pixels)
        self.assertEqual(tuple(ax.images[0].get_extent()),(0,60,40,0))
        self.assertEqual(ax.images[0].get_array().shape,(40,60,3))
        self.assertEqual(tuple(ax.get_ylim()),(40,0))
        self.assertEqual(len(ax.patches),2)
        rect=ax.patches[0]
        self.assertEqual((rect.get_x(),rect.get_y(),rect.get_width(),rect.get_height()),(2.25,3.5,49.5,31.75))
        self.assertEqual(rect.get_linewidth(),2)
        labels=native_details(result)
        for value in ("q0","q1","86.1% model score","unmapped","a group","a garbage pile"):
            self.assertIn(value,labels)
        self.assertIn("Box xyxy (original px):\n[2.25, 3.50, 51.75, 35.25]",labels)
        self.assertEqual([t.get_text() for t in ax.texts],["garbage_pile","unmapped"])
        self.assertNotIn("duplicate",labels)
        self.assertNotIn("below",labels)
        self.assertTrue(all(t.get_bbox_patch().get_facecolor()[:3]==(0,0,0) for t in ax.texts))
        self.assertEqual(result,before)

    def test_empty_image_and_navigation_keep_original_coordinates(self):
        figure=Figure();ax=figure.subplots()
        with Image.new("RGB",(60,40)) as image:draw_overlays(ax,image,self.result(empty=True))
        self.assertEqual(len(ax.images),1)
        self.assertEqual(len(ax.patches),0)
        self.assertIn("Original image displayed without boxes",native_details(self.result(empty=True)))
        ax.set_xlim(10,30);ax.set_ylim(30,10)  # Navigation changes view, not source coordinates.
        self.assertEqual(tuple(ax.images[0].get_extent()),(0,60,40,0))

    def test_passing_console_and_native_failure_cleanup(self):
        result=self.result();result["display"]["terminal_mode"]="full"
        out=io.StringIO()
        with Image.new("RGB",(60,40)) as image,patch("scripts.viewer.open_native_window") as window:
            show_image(image,result,stream=out)
            fig=window.call_args.args[0]
            self.assertIn("Box xyxy",window.call_args.kwargs["details"])
            self.assertFalse(fig.axes)  # Figure cleared after normal return.
        self.assertIn("IMAGE: fixture.png",out.getvalue())
        self.assertIn("DETECTION q0",out.getvalue())
        self.assertNotIn("DETECTION below",out.getvalue())
        self.assertNotIn("\x1b",out.getvalue())
        with Image.new("RGB",(60,40)) as image,patch("scripts.viewer.open_native_window",side_effect=RuntimeError("fixture failure")), \
                self.assertRaisesRegex(NativeDisplayError,"fixture failure"):
            show_image(image,result,stream=io.StringIO())

    def test_window_normal_close_callback_failure_and_initialization_failure(self):
        # Actual backend classes with mocked OS objects. Not native visual evidence.
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        root=MagicMock();root.winfo_screenwidth.return_value=1920;root.winfo_screenheight.return_value=1080
        callbacks={}
        root.protocol.side_effect=lambda name,callback:callbacks.update({name:callback})
        for failure in (False,True):
            figure=Figure()
            figure.canvas.toolbar=MagicMock()
            root.destroy.side_effect=lambda:self.assertIsNone(figure.canvas.toolbar)
            def loop():
                if failure:root.report_callback_exception(RuntimeError,RuntimeError("fixture callback"),None)
                else:callbacks["WM_DELETE_WINDOW"]()
            root.mainloop.side_effect=loop
            with patch("tkinter.Tk",return_value=root),patch("matplotlib.backends.backend_tkagg.FigureCanvasTkAgg"), \
                    patch.object(NavigationToolbar2Tk,"__init__",return_value=None), \
                    patch.object(NavigationToolbar2Tk,"update"),patch.object(NavigationToolbar2Tk,"pack"):
                if failure:
                    with self.assertRaisesRegex(NativeDisplayError,"callback failed"):open_native_window(figure,"fixture")
                else:open_native_window(figure,"fixture")
            figure.clear()  # Must not touch the destroyed Tk toolbar.
            root.destroy.assert_called()
        with patch("tkinter.Tk",side_effect=RuntimeError("no desktop")),self.assertRaisesRegex(RuntimeError,"no desktop"):
            open_native_window(Figure(),"fixture")

    def test_synthetic_check_uses_native_window_without_files(self):
        out=io.StringIO()
        with patch("scripts.viewer.open_native_window") as window,patch("PIL.Image.Image.save",side_effect=AssertionError("export")):
            show_synthetic(stream=out)
        window.assert_called_once()
        self.assertIn("closed normally",out.getvalue())

    def test_cli_synthetic_native_dispatch_and_legacy_backend_migration(self):
        import contextlib
        from scripts.detect_images import main
        from scripts.configuration import validate_config
        config=load_config();config["display"]["backend"]="terminal"
        self.assertEqual(validate_config(config)["display"]["backend"],"TkAgg")
        with patch("scripts.viewer.show_synthetic") as show,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--mode","display-check"]),0)
        show.assert_called_once()
