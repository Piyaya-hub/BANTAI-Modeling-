"""Software fixtures only: every image/prediction here is synthetic, never model evidence."""

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.configuration import ConfigError, ROOT, load_config, validate_config
from scripts.detect_images import main
from scripts.inputs import InputError, discover_inputs, load_image
from scripts.persistence import atomic_json, create_run, source_key
from scripts.results import (candidate, make_result, prepare_result, save_display_outcome,
                             save_result, valid_candidates, validate_result)


class FoundationTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "work").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "work")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config()
        self.source = {"image_id": "synthetic/software-fixture.png", "sha256": "a" * 64,
                       "width": 6, "height": 4, "coordinate_frame": "exif_oriented_original"}
        self.execution = {"actual_device": "cpu", "environment_manifest": "synthetic-test-only",
                          "preprocessing_manifest": "synthetic-test-only"}

    def prediction(self, box=(-1, 1, 7, 3), score=0.7, name="synthetic-d1"):
        return candidate(name, "synthetic unknown phrase", box, score, 6, 4)

    def result(self, detections=None, config=None):
        return make_result("synthetic-run", self.source, config or self.config,
                           detections=[self.prediction()] if detections is None else detections,
                           inference_completed=True, execution=self.execution)

    def test_config_relative_paths_independent_of_cwd(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            c = load_config()
        finally:
            os.chdir(previous)
        self.assertEqual(c["paths"]["input"], str(ROOT / "data/input"))
        c = load_config(overrides={"input": str(self.root / "photos")})
        self.assertEqual(c["paths"]["input"], str(self.root / "photos"))

    def test_bad_configuration_fields(self):
        for field, value in (("threshold", True), ("threshold", float("nan")),
                             ("text_threshold", float("inf")), ("threshold", -0.1),
                             ("text_threshold", 1.1), ("device", "gpu"), ("dtype", "float16"),
                             ("prompt", ""), ("display", {"enabled": "yes"})):
            with self.subTest(field=field, value=value):
                c = copy.deepcopy(self.config)
                c[field] = value
                with self.assertRaises(ConfigError):
                    validate_config(c)
        for field in ("revision", "processor_revision", "id", "disable_custom_kernels", "use_safetensors"):
            c = copy.deepcopy(self.config)
            c["model"][field] = "wrong"
            with self.assertRaises(ConfigError):
                validate_config(c)
        with self.assertRaises(ConfigError):
            validate_config({})

    def test_config_strict_json_and_missing(self):
        for raw in ('{"threshold":NaN}', '{"a":1,"a":2}', '[1]', 'broken'):
            p = self.root / "bad.json"
            p.write_text(raw)
            with self.assertRaises(ConfigError):
                load_config(p)
        with self.assertRaises(ConfigError):
            load_config(self.root / "absent.json")

    def test_overlapping_paths_and_symlink(self):
        for output in (self.root, self.root / "nested"):
            c = copy.deepcopy(self.config)
            c["paths"].update(input=str(self.root), results=str(output))
            with self.assertRaises(ConfigError):
                validate_config(c)
        alias = self.root / "alias"
        try:
            alias.symlink_to(self.root, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"Symlink fixture unavailable on this platform/account: {exc}")
        c = copy.deepcopy(self.config)
        c["paths"].update(input=str(self.root), results=str(alias))
        with self.assertRaises(ConfigError):
            validate_config(c)

    def test_discovery_missing_empty_not_directory(self):
        (self.root / "nested").mkdir()
        for path, code in ((self.root / "missing", "missing_input_folder"),
                           (self.root, "empty_input_folder")):
            with self.assertRaises(InputError) as caught:
                discover_inputs(path)
            self.assertEqual(caught.exception.code, code)
        p = self.root / "file"
        p.touch()
        with self.assertRaises(InputError) as caught:
            discover_inputs(p)
        self.assertEqual(caught.exception.code, "not_input_folder")

    def test_sorted_nonrecursive_and_distinct_errors(self):
        (self.root / "nested").mkdir()
        (self.root / "nested/hidden.png").touch()
        for name in ("b.JPG", "a.txt", "c.png"):
            (self.root / name).write_bytes(b"synthetic corrupt content")
        self.assertEqual([p.name for p in discover_inputs(self.root)], ["a.txt", "b.JPG", "c.png"])
        _, _, error = load_image(self.root / "a.txt")
        self.assertEqual(error["code"], "unsupported_file")
        _, source, error = load_image(self.root / "b.JPG")
        self.assertEqual(error["code"], "decode_error")
        self.assertIsNotNone(source["sha256"])
        self.assertIsNone(source["width"])
        _, _, error = load_image(self.root / "missing.png")
        self.assertEqual(error["code"], "source_read_error")
        with patch("scripts.inputs.Path.read_bytes", side_effect=PermissionError("synthetic denial")):
            self.assertEqual(load_image(self.root / "c.png")[2]["code"], "source_read_error")

    def test_exif_all_orientations_rgb_coordinates_and_unchanged_bytes(self):
        # Pixel-exact PNG EXIF fixture: matrix [[1,2,3],[4,5,6]].
        expected = {1: [1,2,3,4,5,6], 2: [3,2,1,6,5,4], 3: [6,5,4,3,2,1],
                    4: [4,5,6,1,2,3], 5: [1,4,2,5,3,6], 6: [4,1,5,2,6,3],
                    7: [6,3,5,2,4,1], 8: [3,6,2,5,1,4]}
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                p = self.root / f"synthetic-{orientation}.png"
                with Image.new("L", (3, 2)) as im:
                    im.putdata([1,2,3,4,5,6])
                    exif = Image.Exif()
                    exif[274] = orientation
                    im.save(p, exif=exif)
                original = p.read_bytes()
                image, source, error = load_image(p)
                self.assertIsNone(error)
                self.addCleanup(image.close)
                self.assertEqual(image.mode, "RGB")
                self.assertEqual([pixel[0] for pixel in image.getdata()], expected[orientation])
                self.assertEqual(image.size, (3,2) if orientation <= 4 else (2,3))
                self.assertEqual((source["width"], source["height"]), image.size)
                d = candidate("synthetic", "unknown", [-1,-1,9,9], .5, *image.size)
                self.assertEqual(d["box_xyxy"], [0,0,*image.size])
                self.assertEqual(source["sha256"], hashlib.sha256(original).hexdigest())
                self.assertEqual(p.read_bytes(), original)
        self.assertEqual(len(list(self.root.iterdir())), 8)

    def test_jpeg_rgba_and_wrong_content(self):
        for fmt, mode, suffix in (("JPEG", "CMYK", ".JPEG"), ("PNG", "RGBA", ".png")):
            p = self.root / (mode + suffix)
            with Image.new(mode, (5,3)) as im:
                im.save(p, format=fmt)
            image, _, error = load_image(p)
            self.assertIsNone(error)
            self.assertEqual(image.mode, "RGB")
            image.close()
        p = self.root / "disguised.png"
        with Image.new("RGB", (2,2)) as im:
            im.save(p, format="BMP")
        self.assertEqual(load_image(p)[2]["code"], "decode_error")

    def test_raw_box_sanitation_and_geometry(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, "1", None):
            d = self.prediction([value,1,5,3])
            self.assertEqual(d["raw_box_xyxy"], [None,1,5,3])
            self.assertIsNone(d["box_xyxy"])
            self.assertEqual(d["state"], "invalid")
            self.assertEqual(d["model_score"], .7)
            self.assertEqual(d["numeric_diagnostics"][0]["index"], 0)
            self.result([d])
        for box in ([3,1,2,3], [1,1,1,3], [-5,1,-1,3], [1,8,4,9], [1,2,3]):
            d = self.prediction(box)
            self.assertEqual(d["state"], "invalid")
            self.assertIsNone(d["box_xyxy"])
        self.assertEqual(self.prediction()["box_xyxy"], [0,1,6,3])

    def test_invalid_scores_preserve_finite_raw(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.5", None):
            d = self.prediction(score=value)
            self.assertIsNone(d["model_score"])
            self.assertEqual(d["state"], "invalid")
            if type(value) is float and value in (-0.1, 1.1):
                self.assertEqual(d["numeric_diagnostics"][-1]["finite_raw_value"], value)
        for value in (0, 1):
            self.assertEqual(self.prediction(score=value)["state"], "retained")

    def test_statuses_unmapped_and_unknown_counts(self):
        good, bad = self.prediction(), self.prediction(score=float("nan"), name="synthetic-bad")
        mixed = self.result([good,bad])
        self.assertEqual(mixed["status"], "ok")
        self.assertEqual(valid_candidates(mixed), [good])
        self.assertIsNone(good["category"])
        self.assertEqual(good["raw_phrase"], "synthetic unknown phrase")
        self.assertIsNone(mixed["review"]["physical_object_count"])
        self.assertIsNone(mixed["review"]["independent_region_count"])
        invalid = self.result([bad])
        self.assertEqual(invalid["status"], "error")
        self.assertEqual(invalid["error"]["code"], "invalid_predictions")
        self.assertEqual(invalid["display"]["status"], "not_attempted")
        self.assertEqual(self.result([])["status"], "no_detections")
        with self.assertRaisesRegex(ValueError, "preparation"):
            make_result("synthetic", self.source, self.config)

    def test_processing_failures_and_disabled_display(self):
        source = dict(self.source, width=None, height=None)
        r = make_result("synthetic", source, self.config,
                        error={"stage":"decode", "code":"decode_error", "message":"synthetic failure"})
        self.assertEqual(r["display"]["status"], "not_attempted")
        r = make_result("synthetic", source, self.config, unsupported=True)
        self.assertEqual(r["status"], "skipped_unsupported")
        c = copy.deepcopy(self.config)
        c["display"]["enabled"] = False
        self.assertEqual(self.result(config=c)["display"]["status"], "disabled")

    def test_validation_rejects_inconsistent_contracts(self):
        mutations = [("status", "no_detections"), ("error", {}), ("run_id", "../unsafe"),
                     ("schema_version", "other"), ("detections", []), ("display", {"enabled":True,"status":"disabled","error":None})]
        for key, value in mutations:
            r = self.result()
            r[key] = value
            with self.assertRaises(ValueError):
                validate_result(r)
        for section, key, value in (("source","width",True), ("source","sha256","wrong"),
                                    ("model","revision","main"), ("inference","completed",False),
                                    ("inference","threshold",float("nan")), ("inference","actual_device",None),
                                    ("review","physical_object_count",5)):
            r = self.result()
            r[section][key] = value
            with self.assertRaises(ValueError):
                prepare_result(r)
        for key, value in (("box_xyxy", [0,0,999,999]), ("state", "invalid"),
                           ("duplicate_of", "missing"), ("origin", "derived_spatial_group")):
            r = self.result()
            r["detections"][0][key] = value
            with self.assertRaises(ValueError):
                validate_result(r)

    def test_numeric_metadata_sanitation_is_strict_and_nonmutating(self):
        r = self.result()
        r["timing"]["inference_ms"] = float("nan")
        r["timing"]["extra"] = [float("inf"), 2]
        prepared = prepare_result(r)
        self.assertIsNone(prepared["timing"]["inference_ms"])
        self.assertIsNone(prepared["timing"]["extra"][0])
        self.assertEqual(len(prepared["numeric_diagnostics"]), 2)
        self.assertNotEqual(r["timing"]["inference_ms"], r["timing"]["inference_ms"])
        p = self.root / "strict.json"
        atomic_json(p, prepared)
        json.loads(p.read_text(), parse_constant=lambda s: self.fail(f"Invalid JSON: {s}"))

    def test_collisions_runs_sources_and_no_overwrite(self):
        a, fa = create_run(self.root)
        b, fb = create_run(self.root)
        self.assertNotEqual(a,b)
        self.assertNotEqual(source_key(self.source), source_key(dict(self.source,image_id="other/software-fixture.png")))
        self.assertNotEqual(source_key(self.source), source_key(dict(self.source,sha256="b"*64)))
        p = fa / "record.json"
        atomic_json(p, {"old":True})
        with self.assertRaises(FileExistsError):
            atomic_json(p, {"new":True})
        self.assertEqual(json.loads(p.read_text()), {"old":True})
        class FixedID:
            hex = a
        with patch("scripts.persistence.uuid.uuid4", return_value=FixedID()):
            with self.assertRaises(FileExistsError):
                create_run(self.root)
        with self.assertRaises(ValueError):
            save_result(fb, self.result())

    def test_atomic_failures_cleanup_and_preserve_old(self):
        p = self.root / "result.json"
        atomic_json(p, {"old":True})
        for operation in ("os.fsync", "os.replace"):
            with self.subTest(operation=operation), patch("scripts.persistence."+operation, side_effect=OSError("synthetic failure")):
                with self.assertRaises(OSError):
                    atomic_json(p, {"new":True}, replace=True)
            self.assertEqual(json.loads(p.read_text()), {"old":True})
            self.assertEqual(list(self.root.glob(".bantai-*.tmp")), [])
        with patch("scripts.persistence.os.link", side_effect=OSError("synthetic initial failure")):
            with self.assertRaises(OSError):
                atomic_json(self.root / "new.json", {"new":True})
        self.assertFalse((self.root / "new.json").exists())
        with self.assertRaises(TypeError):
            atomic_json(p, {"bad":object()}, replace=True)
        self.assertEqual(list(self.root.glob(".bantai-*.tmp")), [])

    def test_atomic_write_and_flush_failures(self):
        p = self.root / "result.json"
        atomic_json(p, {"old": True})
        import scripts.persistence as persistence
        factory = persistence.tempfile.NamedTemporaryFile
        for operation in ("write", "flush"):
            @contextlib.contextmanager
            def failing_temp(**kwargs):
                with factory(**kwargs) as real:
                    with patch.object(real, operation, side_effect=OSError("synthetic I/O failure")):
                        yield real
            with patch("scripts.persistence.tempfile.NamedTemporaryFile", failing_temp):
                with self.assertRaises(OSError):
                    atomic_json(p, {"new": True}, replace=True)
            self.assertEqual(json.loads(p.read_text()), {"old": True})
            self.assertEqual(list(self.root.glob(".bantai-*.tmp")), [])

    def test_supplied_categories_and_duplicate_references(self):
        a = candidate("synthetic-a", "synthetic garbage pile", [0,0,4,3], .7, 6, 4,
                      category="garbage_pile", mapping_reason="synthetic_supplied_category")
        b = copy.deepcopy(a)
        b.update(id="synthetic-b", state="duplicate", duplicate_of="synthetic-a")
        r = self.result([a,b])
        self.assertEqual(valid_candidates(r), [a])
        self.assertEqual(a["level"], "region")
        r["detections"][1]["duplicate_of"] = "missing"
        with self.assertRaises(ValueError):
            validate_result(r)
        # This tests supplied contract metadata, not an implemented mapping/duplicate algorithm.
        r = self.result()
        r["timing"]["inference_ms"] = -1
        clean = prepare_result(r)
        self.assertIsNone(clean["timing"]["inference_ms"])
        self.assertEqual(clean["numeric_diagnostics"][-1]["reason"], "invalid_timing")

    def test_display_lifecycle_pending_interruption_and_update_failure(self):
        folder = self.root / "synthetic-run"
        folder.mkdir()
        result = self.result()
        p = save_result(folder,result)
        initial = p.read_bytes()
        self.assertEqual(json.loads(initial)["display"]["status"], "pending")
        with patch("scripts.persistence.os.replace", side_effect=OSError("synthetic final failure")):
            with self.assertRaises(OSError):
                save_display_outcome(p,"shown")
        self.assertEqual(p.read_bytes(),initial)
        final = save_display_outcome(p,"error",error={"stage":"display","code":"synthetic_failure","message":"synthetic viewer failure"})
        self.assertEqual(final["status"], "ok")
        self.assertEqual(final["detections"],result["detections"])
        self.assertIsNone(final["error"])
        with self.assertRaises(ValueError):
            save_display_outcome(p,"shown")
        with self.assertRaises(FileExistsError):
            save_result(folder,result)

    def test_display_shown_and_no_detections_lifecycle(self):
        folder = self.root / "synthetic-run"
        folder.mkdir()
        p = save_result(folder,self.result([]))
        final = save_display_outcome(p,"shown")
        self.assertEqual(final["status"],"no_detections")
        self.assertEqual(final["display"]["status"],"shown")

    def test_cli_preparation_only_continues_after_decode_error(self):
        photos = self.root / "photos"
        photos.mkdir()
        p = photos / "synthetic.png"
        with Image.new("RGB", (3,2)) as im:
            im.save(p)
        original = p.read_bytes()
        (photos / "broken.jpg").write_bytes(b"synthetic broken image")
        (photos / "unsupported.txt").write_text("synthetic")
        output = self.root / "output"
        with contextlib.redirect_stdout(io.StringIO()) as console:
            code = main(["--input",str(photos),"--results",str(output)])
        self.assertEqual(code,1)
        self.assertIn("inference not run", console.getvalue())
        run = next(output.iterdir())
        records = [json.loads(f.read_text()) for f in run.glob("*.preparation.json")]
        self.assertEqual({r["status"] for r in records}, {"prepared","error","skipped_unsupported"})
        for r in records:
            self.assertFalse(r["inference_executed"])
            self.assertNotIn("detections",r)
            self.assertEqual(r["display"]["status"],"not_attempted")
        self.assertEqual(p.read_bytes(),original)
        self.assertTrue(all(f.suffix==".json" for f in run.iterdir()))

    def test_cli_missing_empty_unsupported_and_persistence_failure(self):
        for source in (self.root / "missing", self.root):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--input",str(source),"--results",str(self.root.parent / "unused-output")]),2)
        photos = self.root / "photos"
        photos.mkdir()
        (photos / "file.txt").touch()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--input",str(photos),"--results",str(self.root / "output")]),2)
            with patch("scripts.detect_images.atomic_json", side_effect=OSError("synthetic persistence failure")):
                self.assertEqual(main(["--input",str(photos),"--results",str(self.root / "output")]),2)


if __name__ == "__main__":
    unittest.main()
