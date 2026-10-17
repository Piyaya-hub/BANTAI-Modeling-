"""Display-only fixtures: never pretrained inference or visual proof."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.configuration import ROOT, load_config
from scripts.detect_images import main
from scripts.inputs import load_image
from scripts.persistence import atomic_json
from scripts.replay import run_replay
from scripts.results import candidate, make_result, save_result
from scripts.viewer import NativeDisplayError, show_synthetic


class ReplayTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "work").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "work")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config(overrides={"input": str(self.root / "unused-input"), "results": str(self.root / "replays")})
        self.config.update(threshold=.35,text_threshold=.25)  # Saved fixture, independent of owner edits.
        self.image_path = self.root / "original.png"
        with Image.new("RGB", (12,8), "blue") as image:
            image.save(self.image_path)
        image, source, error = load_image(self.image_path)
        image.close()
        self.assertIsNone(error)
        d = candidate("q0", "a garbage pile", [1,1,10,6], .8, 12,8, category="garbage_pile")
        self.result = make_result("saved-run", source, self.config, detections=[d],
                                  inference_completed=True, library_version="4.51.3",
                                  execution={"actual_device":"cpu", "environment_manifest":"saved/environment.json",
                                             "preprocessing_manifest":"saved/preprocessing.json"})
        saved = self.root / "saved-run"
        saved.mkdir()
        self.result_path = save_result(saved,self.result)
        self.original_json = self.result_path.read_bytes()
        self.original_photo = self.image_path.read_bytes()

    def replay_records(self):
        return [json.loads(p.read_text()) for p in (self.root / "replays").glob("*/replay.json")]

    def invoke(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_replay(self.config,self.result_path,self.image_path)

    def assert_originals(self):
        self.assertEqual(self.result_path.read_bytes(), self.original_json)
        self.assertEqual(self.image_path.read_bytes(), self.original_photo)

    def test_replay_persists_before_view_and_never_changes_saved_thresholds(self):
        self.config["threshold"] = .99
        def viewer(image,result):
            self.assertEqual(self.replay_records()[0]["display"]["status"],"pending")
            self.assertEqual(result["inference"]["threshold"],.35)
            self.assertEqual(image.size,(12,8))
        with patch("scripts.replay.show_image",side_effect=viewer) as show:
            self.assertEqual(self.invoke(),0)
        self.assertEqual(show.call_count,1)
        record=self.replay_records()[0]
        self.assertFalse(record["inference_executed"])
        self.assertEqual(record["display"]["status"],"shown")
        self.assertEqual(record["display"]["visual_verification"],"pending")
        self.assertEqual(record["thresholds_from_saved_result"]["threshold"],.35)
        self.assert_originals()

    def test_replay_rejects_different_source_identity_even_same_bytes(self):
        other=self.root / "copy.png"
        other.write_bytes(self.original_photo)
        with patch("scripts.replay.show_image") as show, self.assertRaisesRegex(ValueError,"image_id"):
            run_replay(self.config,self.result_path,other)
        show.assert_not_called()
        self.assertFalse((self.root / "replays").exists())
        self.assert_originals()

    def test_replay_rejects_changed_hash_and_dimensions(self):
        for field, value in (("sha256","0"*64),("width",13),("height",9)):
            altered=copy.deepcopy(self.result)
            altered["source"][field]=value
            self.result_path.write_text(json.dumps(altered),encoding="utf-8")
            with patch("scripts.replay.show_image") as show, self.assertRaisesRegex(ValueError,field):
                self.invoke()
            show.assert_not_called()
        self.assertFalse((self.root / "replays").exists())

    def test_replay_renderer_failure_and_interrupt_keep_originals(self):
        with patch("scripts.replay.show_image",side_effect=NativeDisplayError("unsupported fixture stream")):
            self.assertEqual(self.invoke(),1)
        self.assertEqual(self.replay_records()[0]["display"]["status"],"error")
        with patch("scripts.replay.show_image",side_effect=KeyboardInterrupt()):
            self.assertEqual(self.invoke(),130)
        self.assertIn("pending",{r["display"]["status"] for r in self.replay_records()})
        self.assert_originals()

    def test_replay_initial_and_final_write_failures(self):
        with patch("scripts.replay.atomic_json",side_effect=OSError("initial fixture failure")), patch("scripts.replay.show_image") as show:
            with self.assertRaises(OSError):
                self.invoke()
        show.assert_not_called()
        def write(path,record,**kwargs):
            if kwargs.get("replace"):
                raise OSError("final fixture failure")
            return atomic_json(path,record,**kwargs)
        with patch("scripts.replay.atomic_json",side_effect=write), patch("scripts.replay.show_image"):
            with self.assertRaises(OSError):
                self.invoke()
        self.assertEqual(self.replay_records()[0]["display"]["status"],"pending")
        self.assert_originals()

    def test_cli_replay_needs_no_input_discovery_and_no_display_is_honest(self):
        config_path=self.root / "config.json"
        config_path.write_text(json.dumps(self.config),encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), patch("scripts.replay.show_image") as show:
            code=main(["--mode","replay","--config",str(config_path),"--result",str(self.result_path),
                       "--image",str(self.image_path),"--no-display"])
        self.assertEqual(code,0)
        show.assert_not_called()
        self.assertEqual(self.replay_records()[0]["display"]["status"],"disabled")
        self.assert_originals()


if __name__ == "__main__":
    unittest.main()
