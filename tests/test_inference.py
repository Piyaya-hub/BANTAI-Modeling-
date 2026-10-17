"""Synthetic software tests only. No pretrained weights, GPU or model predictions."""

import contextlib
import copy
import io
import json
import math
import os
import struct
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from scripts.configuration import ROOT, ConfigError, load_config, validate_config, REVISION
from scripts.inference import (GroundingDinoDetector, InferenceError, DIRECT_VERSIONS,
                               extract_candidates, runtime_versions)
from scripts.pipeline import run_inference
from scripts.postprocessing import apply_policies, map_phrase, normalize_phrase
from scripts.results import candidate, make_result, prepare_result, save_result, validate_result
from scripts.viewer import candidate_label, show_image, summary, selected_queries, presentation_blocks, decision, passing_candidates, float32_pass
from scripts.viewer import detection_fields, table_pages


class SyntheticTensor:
    """Only the adapter's data-export calls; deliberately does not imitate model execution."""
    def __init__(self, data):
        self.data = data

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.data)

    def __gt__(self, value):
        # Model a float32 scalar comparison for boundary fixtures only.
        threshold = struct.unpack("f", struct.pack("f", value))[0]
        return SyntheticTensor([[v > threshold for v in row] for row in self.data])

    def max(self, dim):
        assert dim == -1
        return SimpleNamespace(values=SyntheticTensor([[float("nan") if any(math.isnan(v) for v in row) else max(row)
                                                      for row in batch] for batch in self.data]))


class SyntheticCudaOOM(RuntimeError):
    pass


class InferenceTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "work").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "work")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config(overrides={"input": str(self.root / "photos"), "results": str(self.root / "results")})
        self.config.update(threshold=.35,text_threshold=.25)  # Stable software-fixture boundaries.
        self.config["paths"]["cache"] = str(self.root / "cache")
        self.source = {"image_id": "synthetic-fixture.png", "sha256": "a"*64, "width": 6, "height": 4,
                       "coordinate_frame": "exif_oriented_original"}

    def prediction(self, ident="synthetic-q1", phrase="a discarded bag", box=(0,0,4,3), score=.7):
        category, reason = map_phrase(phrase)
        return candidate(ident, phrase, box, score, 6,4, category=category, mapping_reason=reason)

    def functional(self, predictions=None):
        return make_result("synthetic-run", self.source, self.config,
                           detections=[self.prediction()] if predictions is None else predictions,
                           inference_completed=True, library_version="4.51.3",
                           execution={"actual_device":"cpu", "environment_manifest":"synthetic-test-only",
                                      "preprocessing_manifest":"synthetic-test-only"})

    def image_file(self, name="synthetic.png"):
        folder = Path(self.config["paths"]["input"])
        folder.mkdir(exist_ok=True)
        p = folder / name
        with Image.new("RGB", (6,4)) as image:
            image.save(p)
        return p

    def detector(self, predictions=None):
        detector = MagicMock()
        detector.versions = {"transformers":"4.51.3"}
        detector.device = "cpu"
        detector.fallbacks = []
        detector.preprocessing = {"test_fixture":"synthetic software processor; no model"}
        detector.predict.return_value = ([self.prediction()] if predictions is None else predictions,
                                         {"scope":"synthetic software fixture; not measured", "inference_ms":None})
        detector.environment.return_value = {"test_fixture":"synthetic software environment; no model"}
        return detector

    def run_pipeline(self, files, detector, viewer=None):
        with patch("scripts.pipeline.GroundingDinoDetector", return_value=detector) as factory, \
                patch("scripts.pipeline.show_image", side_effect=viewer) as show, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = run_inference(self.config, files, local_files_only=True)
        run = next(Path(self.config["paths"]["results"]).iterdir())
        records = [json.loads(p.read_text()) for p in run.glob("*.json")
                   if p.name not in ("run.json", "environment.json", "preprocessing.json")]
        return code, run, records, factory, show

    def test_exact_aliases_do_not_force_partial_or_combined_phrases(self):
        for phrase in (" A   Garbage Pile. ", "an garbage pile!", "garbage pile"):
            self.assertEqual(map_phrase(phrase), ("garbage_pile","exact_alias"))
        self.assertEqual(map_phrase("a group of scattered litter")[0],"group_of_individual_litter")
        self.assertEqual(map_phrase("a discarded bottle"), (None,"no_exact_alias"))
        for phrase in ("bottle", "garbage pile and bag", "", "individual discarded item group"):
            self.assertIsNone(map_phrase(phrase)[0])
        self.assertEqual(normalize_phrase("An  discarded bag."),"discarded bag")

    def test_duplicates_keep_highest_score_and_preserve_near_and_other_phrases(self):
        raw = [self.prediction("synthetic-a", score=.6),
               self.prediction("synthetic-b", phrase="Discarded Bag.", score=.8),
               self.prediction("synthetic-near", box=(0,0,4.01,3)),
               self.prediction("synthetic-other", phrase="a garbage pile")]
        processed, flags = apply_policies(raw)
        self.assertEqual(processed[0]["state"],"duplicate")
        self.assertEqual(processed[0]["duplicate_of"],"synthetic-b")
        self.assertEqual(processed[2]["state"],"retained")
        self.assertEqual(processed[3]["state"],"retained")
        self.assertEqual(raw[0]["state"],"retained")
        validate_result(self.functional(processed))
        self.assertTrue(flags)
        processed[0]["duplicate_of"]="synthetic-near"
        with self.assertRaises(ValueError):
            validate_result(self.functional(processed))

    def test_overlaps_flag_without_suppression_or_memberships(self):
        obj = self.prediction("synthetic-object", box=(1,1,2,2))
        region = self.prediction("synthetic-region", phrase="a group of scattered litter", box=(0,0,5,4))
        near = self.prediction("synthetic-near", phrase="a group of individual litter", box=(0,0,5.1,4))
        processed, flags = apply_policies([obj,region,near])
        self.assertEqual([p["state"] for p in processed],["retained"]*3)
        self.assertIn("same_category_overlap",{f["kind"] for f in flags})
        self.assertIn("cross_category_overlap",{f["kind"] for f in flags})
        self.assertIn("object_within_region_membership_unresolved",{f["kind"] for f in flags})
        result = self.functional(processed)
        self.assertEqual(result["review"]["memberships"],[])
        self.assertIsNone(result["review"]["physical_object_count"])

    def test_invalid_candidates_are_not_deduplicated_or_summarized_as_valid(self):
        predictions = [self.prediction("synthetic-bad",score=float("nan")),self.prediction()]
        processed, _ = apply_policies(predictions)
        self.assertEqual(processed[0]["state"],"invalid")
        text = summary(self.functional(processed))
        self.assertNotIn("synthetic-bad",text)
        self.assertNotIn("INVALID",text)
        self.assertIn("0.7000",text)
        self.assertIn("individual_litter=1",text)
        self.assertIn("Physical object count: unknown",text)

    def test_postprocessing_configuration_rejects_invalid_policy_and_numbers(self):
        for key,value in (("mapping_version","other"),("duplicate_policy","nms"),("overlap_iou",float("nan")),("overlap_iou",True)):
            c=copy.deepcopy(self.config)
            c["postprocessing"][key]=value
            with self.assertRaises(ConfigError):
                validate_config(c)

    def test_pinned_runtime_guard_missing_or_changed_packages(self):
        with patch("scripts.inference.importlib.metadata.version", side_effect=lambda p:DIRECT_VERSIONS[p]):
            self.assertEqual(runtime_versions(),DIRECT_VERSIONS)
        with patch("scripts.inference.importlib.metadata.version", return_value="99.0"):
            with self.assertRaises(InferenceError) as caught:
                runtime_versions()
            self.assertEqual(caught.exception.code,"dependency_version_mismatch")
        import importlib.metadata
        with patch("scripts.inference.importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError("synthetic")):
            with self.assertRaises(InferenceError) as caught:
                runtime_versions()
            self.assertEqual(caught.exception.code,"missing_dependency")

    def test_grounded_adapter_threshold_target_sizes_and_invalid_raw_queries(self):
        # Scores here are synthetic post-sigmoid fixtures, deliberately not torch/model output.
        logits = [[[.8,.2],[float("nan"),.4],[.1,.2],[.1,.2]]]
        boxes = [[[.5,.5,.5,.5],[.5,.5,.5,.5],[.5,.5,-.5,.5],[.5,.5,.5,.5]]]
        outputs = SimpleNamespace(logits=SyntheticTensor(logits),pred_boxes=SyntheticTensor(boxes))
        processor=MagicMock()
        processor.post_process_grounded_object_detection.return_value=[{
            "text_labels":["a discarded bag"],"boxes":SyntheticTensor([[1.5,1,4.5,3]]),"scores":SyntheticTensor([.8])}]
        torch=SimpleNamespace(sigmoid=lambda x:x)
        result=extract_candidates(outputs,{"input_ids":SyntheticTensor([[10,11]])},processor,self.config,6,4,torch)
        self.assertEqual([d["id"] for d in result],["q0","q1","q2"])
        self.assertEqual(result[0]["box_xyxy"],[1.5,1,4.5,3])
        self.assertEqual(result[1]["raw_phrase"],"")
        self.assertEqual(result[1]["state"],"invalid")
        self.assertEqual(result[2]["state"],"invalid")
        self.assertIsNone(result[1]["model_score"])
        kwargs=processor.post_process_grounded_object_detection.call_args.kwargs
        self.assertEqual(kwargs["target_sizes"],[(4,6)])
        self.assertEqual(kwargs["threshold"],.35)
        self.assertEqual(kwargs["text_threshold"],.25)
        self.assertNotIn("box_threshold",kwargs)
        validate_result(self.functional(result))

    def test_grounded_adapter_rejects_mismatched_results(self):
        outputs=SimpleNamespace(logits=SyntheticTensor([[[.8]]]),pred_boxes=SyntheticTensor([[[.5,.5,.5,.5]]]))
        processor=MagicMock()
        processor.post_process_grounded_object_detection.return_value=[{
            "text_labels":[],"boxes":SyntheticTensor([]),"scores":SyntheticTensor([])}]
        with self.assertRaises(InferenceError):
            extract_candidates(outputs,{"input_ids":SyntheticTensor([[10]])},processor,self.config,6,4,SimpleNamespace(sigmoid=lambda x:x))

    def test_grounded_adapter_uses_tensor_threshold_mask_at_float32_boundary(self):
        value=struct.unpack("f",struct.pack("f",.1))[0]
        self.assertGreater(value,.1)  # Exported Python float comparison differs.
        self.config["threshold"]=.1
        outputs=SimpleNamespace(logits=SyntheticTensor([[[value]]]),pred_boxes=SyntheticTensor([[[.5,.5,.5,.5]]]))
        processor=MagicMock()
        processor.post_process_grounded_object_detection.return_value=[{
            "text_labels":[],"boxes":SyntheticTensor([]),"scores":SyntheticTensor([])}]
        self.assertEqual(extract_candidates(outputs,{"input_ids":SyntheticTensor([[10]])},processor,self.config,6,4,SimpleNamespace(sigmoid=lambda x:x)),[])

    def test_expected_negative_infinity_padding_preserves_valid_predictions(self):
        # Upstream fills both attention-mask zeros and the fixed-width text tail
        # with -inf. This mirrors that layout; every value is synthetic.
        for with_mask in (True, False):
            logits = [.8,.2,float("-inf"),float("-inf")]
            ids = [10,11,0] if with_mask else [10,11]
            inputs = {"input_ids":SyntheticTensor([ids])}
            if with_mask:
                inputs["attention_mask"] = SyntheticTensor([[1,1,0]])
            outputs = SimpleNamespace(logits=SyntheticTensor([[logits]]),
                                      pred_boxes=SyntheticTensor([[[.5,.5,.5,.5]]]))
            processor = MagicMock()
            processor.post_process_grounded_object_detection.return_value = [{
                "text_labels":["a discarded bag"], "boxes":SyntheticTensor([[1.5,1,4.5,3]]),
                "scores":SyntheticTensor([.8])}]
            result = extract_candidates(outputs,inputs,processor,self.config,6,4,SimpleNamespace(sigmoid=lambda x:x))
            self.assertEqual(len(result),1)
            self.assertEqual(result[0]["state"],"retained")
            self.assertEqual(result[0]["model_score"],.8)
            self.assertEqual(result[0]["numeric_diagnostics"],[])
            self.assertEqual(self.functional(result)["status"],"ok")

    def test_padding_does_not_turn_below_threshold_queries_into_errors(self):
        outputs = SimpleNamespace(logits=SyntheticTensor([[[.1,.2,float("-inf")]]]),
                                  pred_boxes=SyntheticTensor([[[.5,.5,.5,.5]]]))
        processor = MagicMock()
        processor.post_process_grounded_object_detection.return_value = [{
            "text_labels":[],"boxes":SyntheticTensor([]),"scores":SyntheticTensor([])}]
        result = extract_candidates(outputs,{"input_ids":SyntheticTensor([[10,11]])},processor,self.config,6,4,
                                    SimpleNamespace(sigmoid=lambda x:x))
        self.assertEqual(result,[])
        self.assertEqual(self.functional(result)["status"],"no_detections")

    def test_active_nonfinite_and_unexpected_padding_nonfinite_remain_invalid(self):
        for position,value in ((0,float("-inf")),(0,float("inf")),(0,float("nan")),
                               (1,float("inf")),(1,float("nan")),(2,float("inf")),(2,float("nan"))):
            with self.subTest(position=position,value=value):
                tokens=[.2,float("-inf"),float("-inf")]
                tokens[position]=value
                outputs=SimpleNamespace(logits=SyntheticTensor([[tokens]]),pred_boxes=SyntheticTensor([[[.5,.5,.5,.5]]]))
                torch=SimpleNamespace(sigmoid=lambda x:x)
                selected=not any(math.isnan(v) for v in tokens) and max(tokens)>.35
                processor=MagicMock()
                processor.post_process_grounded_object_detection.return_value=[{
                    "text_labels":["a discarded bag"] if selected else [],
                    "boxes":SyntheticTensor([[1.5,1,4.5,3]] if selected else []),
                    "scores":SyntheticTensor([max(tokens)] if selected else [])}]
                result=extract_candidates(outputs,{"input_ids":SyntheticTensor([[10,0]]),
                                          "attention_mask":SyntheticTensor([[1,0]])},processor,self.config,6,4,torch)
                self.assertEqual(len(result),1)
                self.assertEqual(result[0]["state"],"invalid")
                self.assertIsNone(result[0]["model_score"])
                self.assertIn("nonfinite_model_logit",[d["reason"] for d in result[0]["numeric_diagnostics"]])
                self.assertEqual(self.functional(result)["error"]["code"],"invalid_predictions")

    def test_invalid_attention_masks_are_not_guessed(self):
        outputs=SimpleNamespace(logits=SyntheticTensor([[[.2,.1,float("-inf")]]]),
                                pred_boxes=SyntheticTensor([[[.5,.5,.5,.5]]]))
        for mask in ([0,0],[1],[1,2]):
            with self.assertRaises(InferenceError) as caught:
                extract_candidates(outputs,{"input_ids":SyntheticTensor([[10,11]]),
                                            "attention_mask":SyntheticTensor([mask])},MagicMock(),self.config,6,4,
                                   SimpleNamespace(sigmoid=lambda x:x))
            self.assertEqual(caught.exception.code,"invalid_text_mask")

    def loader_fixtures(self):
        config=SimpleNamespace(_commit_hash=REVISION,model_type="grounding-dino",
                               backbone_config=SimpleNamespace(model_type="swin",depths=[2,2,6,2],num_heads=[3,6,12,24]))
        model=MagicMock()
        model.config=config
        processor=MagicMock()
        processor.image_processor.to_dict.return_value={"synthetic_processor":True}
        module=SimpleNamespace(AutoConfig=MagicMock(),GroundingDinoProcessor=MagicMock(),AutoModelForZeroShotObjectDetection=MagicMock())
        module.AutoConfig.from_pretrained.return_value=config
        module.GroundingDinoProcessor.from_pretrained.return_value=processor
        module.AutoModelForZeroShotObjectDetection.from_pretrained.return_value=model
        torch=SimpleNamespace(float32="synthetic-float32",cuda=SimpleNamespace(is_available=lambda:False,OutOfMemoryError=SyntheticCudaOOM))
        return module,torch,model,processor

    def test_loader_pins_config_model_processor_safetensors_and_evaluation(self):
        module,torch,model,processor=self.loader_fixtures()
        with patch("scripts.inference.runtime_versions",return_value=DIRECT_VERSIONS), \
                patch.dict("sys.modules",{"torch":torch,"transformers":module}):
            d=GroundingDinoDetector(self.config,local_files_only=True)
        for api in (module.AutoConfig,module.GroundingDinoProcessor,module.AutoModelForZeroShotObjectDetection):
            kwargs=api.from_pretrained.call_args.kwargs
            self.assertEqual(kwargs["revision"],REVISION)
            self.assertEqual(kwargs["cache_dir"],self.config["paths"]["cache"])
            self.assertIs(kwargs["trust_remote_code"],False)
            self.assertIs(kwargs["local_files_only"],True)
        model_kwargs=module.AutoModelForZeroShotObjectDetection.from_pretrained.call_args.kwargs
        self.assertIs(model_kwargs["use_safetensors"],True)
        self.assertTrue(model_kwargs["config"].disable_custom_kernels)
        model.eval.assert_called_once()
        model.to.assert_called_once_with(device="cpu",dtype="synthetic-float32")
        self.assertEqual(d.preprocessing["processor_revision"],REVISION)
        d.close()

    def test_loader_rejects_unresolved_or_wrong_revision_before_weights(self):
        module,torch,_,_=self.loader_fixtures()
        module.AutoConfig.from_pretrained.return_value._commit_hash="wrong"
        with patch("scripts.inference.runtime_versions",return_value=DIRECT_VERSIONS), \
                patch.dict("sys.modules",{"torch":torch,"transformers":module}):
            with self.assertRaises(InferenceError) as caught:
                GroundingDinoDetector(self.config)
        self.assertEqual(caught.exception.code,"revision_mismatch")
        module.AutoModelForZeroShotObjectDetection.from_pretrained.assert_not_called()

    def test_forward_uses_inference_context_and_oriented_rgb_input(self):
        d=self.bare_detector(actual="cpu")
        calls=[]
        @contextlib.contextmanager
        def inference_context():
            calls.append("enter")
            yield
            calls.append("exit")
        d.torch.inference_mode=inference_context
        class SyntheticInputs(dict):
            def to(self,device):
                calls.append(device)
                return self
        inputs=SyntheticInputs(input_ids="synthetic-tokens",pixel_values="synthetic-pixels")
        d.processor=MagicMock(return_value=inputs)
        d.forward_attempts=d.completed_forwards=0
        with Image.new("RGB",(6,4)) as image, patch("scripts.inference.extract_candidates",return_value=[]) as extract:
            self.assertEqual(d._once(image),[])
            d.processor.assert_called_once_with(images=image,text=self.config["prompt"],return_tensors="pt")
            self.assertEqual(extract.call_args.args[4:6],(6,4))
        self.assertEqual(calls,["cpu","enter","exit"])
        self.assertEqual(d.completed_forwards,1)

    def raw_fixture(self):
        import torch
        # Actual tensor operations on synthetic data, never pretrained predictions.
        logits = torch.tensor([[[0., -2., float("-inf"), float("-inf")],
                                [-4., -3., float("-inf"), float("-inf")],
                                [float("nan"), -2., float("-inf"), float("-inf")]]])
        outputs = SimpleNamespace(logits=logits, pred_boxes=torch.tensor([[[.5,.5,.5,.5]]*3]))
        inputs = {"input_ids": torch.tensor([[101, 123, 0]]), "attention_mask": torch.tensor([[1,1,0]])}
        processor = MagicMock()
        processor.tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "litter", "[PAD]"]
        processor.tokenizer.all_special_ids = [101, 0]
        processor.tokenizer.pad_token_id = 0
        processor.post_process_grounded_object_detection.return_value = [{
            "text_labels": ["a discarded bag"], "boxes": torch.tensor([[1.5,1.,4.5,3.]]), "scores": torch.tensor([.5])}]
        return torch, outputs, inputs, processor

    def test_raw_scores_alignment_padding_invalid_and_filter_equivalence(self):
        torch, outputs, inputs, processor = self.raw_fixture()
        raw = {}
        args = (outputs, inputs, processor, self.config, 6, 4, torch)
        with_raw = extract_candidates(*args, diagnostics=raw)
        self.assertEqual(with_raw, extract_candidates(*args))
        self.assertEqual([t["position"] for t in raw["tokens"]], [0,1])
        self.assertEqual([t["token"] for t in raw["tokens"]], ["[CLS]", "litter"])
        self.assertEqual(len(raw["queries"]), 3)
        self.assertAlmostEqual(raw["queries"][0]["token_scores"][1], torch.sigmoid(torch.tensor(-2.)).item())
        rejected = raw["queries"][1]
        self.assertFalse(rejected["threshold_pass"])
        self.assertIsNone(rejected["raw_phrase"])
        self.assertIsNone(rejected["category"])
        self.assertTrue(all(v > 0 for v in rejected["token_scores"]))
        self.assertIsNone(raw["queries"][2]["token_scores"][0])
        self.assertEqual(raw["queries"][2]["invalid_token_positions"], [0])
        result = self.functional(with_raw)
        result["query_diagnostics"] = raw
        json.dumps(prepare_result(result), allow_nan=False)
        full = summary(result, "full", width=120, include_special=True)
        self.assertEqual(full.count("SPECIAL"),3)
        self.assertEqual(full.count("litter [pos=1 id=123]"),3)
        self.assertIn("not decoded",full)
        self.assertIn("BELOW THRESHOLD",full)
        self.assertNotIn("SPECIAL", summary(result))
        result["query_diagnostics"]["queries"][0]["token_scores"].pop()
        with self.assertRaises(ValueError):
            prepare_result(result)

    def test_raw_strict_threshold_boundary_and_no_attention_mask(self):
        torch, outputs, inputs, processor = self.raw_fixture()
        outputs.logits = torch.zeros((1,1,4))
        outputs.logits[0,0,3] = -float("inf")
        outputs.pred_boxes = outputs.pred_boxes[:,:1]
        inputs.pop("attention_mask")
        self.config.update(threshold=.5, text_threshold=.5)
        processor.post_process_grounded_object_detection.return_value = [{
            "text_labels": [], "boxes": torch.empty((0,4)), "scores": torch.empty(0)}]
        raw = {}
        self.assertEqual(extract_candidates(outputs, inputs, processor, self.config, 6,4,torch,diagnostics=raw), [])
        self.assertEqual(len(raw["tokens"]), 2)
        self.assertEqual(raw["queries"][0]["token_scores"], [.5]*2)
        self.assertFalse(raw["queries"][0]["threshold_pass"])
        self.assertEqual(raw["queries"][0]["token_threshold_pass"], [False]*2)


    def test_pipeline_persists_complete_diagnostics_even_without_display(self):
        torch, outputs, inputs, processor = self.raw_fixture()
        raw = {}
        predictions = extract_candidates(outputs, inputs, processor, self.config, 6,4,torch,diagnostics=raw)
        detector = self.detector(predictions)
        detector.query_diagnostics = raw
        self.config["display"]["enabled"] = False
        code, _, records, _, show = self.run_pipeline([self.image_file()], detector)
        self.assertEqual(code, 0)
        self.assertEqual(records[0]["query_diagnostics"], raw)
        show.assert_not_called()

    def test_legacy_backend_migrates_and_invalid_terminal_mode_rejected(self):
        self.config["display"]["backend"] = "TkAgg"
        self.assertEqual(validate_config(self.config)["display"]["backend"], "TkAgg")
        self.config["display"]["terminal_mode"] = "invalid"
        with self.assertRaises(ConfigError):
            validate_config(self.config)

    def presentation_fixture(self):
        torch, outputs, inputs, processor = self.raw_fixture()
        raw = {}
        predictions = extract_candidates(outputs, inputs, processor, self.config, 6,4,torch,diagnostics=raw)
        result = self.functional(predictions)
        result["query_diagnostics"] = raw
        return result

    def test_summary_passing_only_and_full_restores_raw_coverage(self):
        result = self.presentation_fixture()
        raw = result["query_diagnostics"]
        raw["queries"][0]["token_scores"][1] = .8123456789
        raw["queries"][0]["token_threshold_pass"][1] = True
        # Passing raw duplicates are still excluded from normal output/overlays.
        duplicate = copy.deepcopy(result["detections"][0])
        duplicate.update(id="q3",model_score=.49,state="duplicate",duplicate_of="q0")
        result["detections"].append(duplicate)
        q3 = copy.deepcopy(raw["queries"][0]); q3.update(id="q3",query_index=3,model_score=.49)
        raw["queries"].append(q3)
        before = copy.deepcopy(result)
        for mode in ("summary",):
            self.assertEqual([q["id"] for q in selected_queries(result,mode,900)],["q0"])
            text = summary(result,mode,width=120,below_count=900,include_special=True)
            self.assertEqual(text.count("litter [pos=1 id=123]"),1)
            self.assertIn("0.8123",text)
            for excluded in ("q1", "q2", "q3", "SPECIAL", "not decoded", "INVALID", "BELOW THRESHOLD"):
                self.assertNotIn(excluded,text)
        self.assertEqual([q["id"] for q in selected_queries(result,"full")],["q0","q1","q2","q3"])
        full=summary(result,"full",width=120,include_special=True)
        for included in ("QUERY q0", "QUERY q1", "QUERY q2", "QUERY q3", "BELOW THRESHOLD", "INVALID", "SPECIAL", "duplicate"):
            self.assertIn(included,full)
        self.assertEqual(full.count("litter [pos=1 id=123]"),4)
        self.assertEqual(result,before)  # Raw precision and excluded diagnostics intact.

    def test_token_console_uses_stored_strict_mask_and_excludes_invalid_values(self):
        result=self.presentation_fixture()
        q=result["query_diagnostics"]["queries"][0]
        for score,mask,invalid in ((.25,False,[]),(float("nan"),True,[]),(.8,True,[1])):
            q["token_scores"][1]=score
            q["token_threshold_pass"][1]=mask
            q["invalid_token_positions"]=invalid
            self.assertNotIn("litter [pos=1",summary(result,width=120))
        q["token_scores"][1]=.2500000298023224
        q["token_threshold_pass"][1]=True
        q["invalid_token_positions"]=[]
        self.assertIn("litter [pos=1",summary(result,width=120))
        self.assertEqual(q["token_scores"][1],.2500000298023224)

    def test_empty_passing_results_render_original_and_empty_tables(self):
        result = self.presentation_fixture()
        result["detections"] = []
        text = summary(result,width=120)
        self.assertIn("No valid retained queries pass",text)
        self.assertIn("No ordinary text tokens pass",text)
        self.assertEqual(passing_candidates(result),[])

    def test_legacy_float32_boundary_without_model_import(self):
        self.assertFalse(float32_pass(.35,.35))
        self.assertFalse(float32_pass(.35000000001,.35))
        self.assertTrue(float32_pass(.3500000238418579,.35))
        result=self.functional([self.prediction(score=.35)])
        self.assertEqual(passing_candidates(result),[])
        self.assertNotIn("0.3500 | PASS",summary(result))

    def test_presentation_decimal_units_and_status_boundaries(self):
        result = self.presentation_fixture()
        text = summary(result,"full",width=120)
        self.assertIn("0.5000",text)
        self.assertIn("Box threshold: 0.3500",text)
        self.assertIn("Text threshold: 0.2500",text)
        self.assertNotIn("%",text)
        self.assertNotIn("FAIL",text)
        self.assertEqual(decision(.35,False),"BELOW THRESHOLD")
        self.assertEqual(decision(.35000001,True),"PASS")
        self.assertEqual(decision(None,False),"INVALID")
        self.assertEqual(decision(.8,True,True),"INVALID")
        self.assertEqual(result["query_diagnostics"]["queries"][0]["model_score"],.5)

    def test_tables_fit_narrow_width_and_pages_are_bounded(self):
        result = self.presentation_fixture()
        for width in (32,50,79,120):
            text = summary(result,"full",width=width,page_rows=2,include_special=True)
            self.assertTrue(all(len(line)<=width for line in text.splitlines()),(width,text))
        blocks = list(presentation_blocks(result,"full",width=79,page_rows=2))
        tables = [b for b in blocks if b[0] in ("DETECTION SUMMARY","TEXT-TOKEN SCORE DETAILS")]
        self.assertGreaterEqual(len(tables),2)
        self.assertTrue(all(len(b)<=6 for b in tables))


    def test_token_groups_and_unwrapped_status_header(self):
        result=self.presentation_fixture()
        for width in (49,79,119):
            blocks=list(presentation_blocks(result,"full",width=width,page_rows=2))
            groups=[b for b in blocks if b[0].startswith("QUERY ")]
            self.assertEqual([b[0].split()[1] for b in groups],["q0","q1","q2"])
            for block in groups:
                self.assertIn("Text Threshold Status",block[1])
                header_positions=[i for i,c in enumerate(block[1]) if c=="|"]
                for row in block[3:]:
                    if "|" in row:
                        self.assertEqual([i for i,c in enumerate(row) if c=="|"],header_positions)
                self.assertTrue(all(len(line)<=width for line in block))

    def test_terminal_presentation_configuration_limits(self):
        for key,value in (("summary_below_threshold",-1),("summary_below_threshold",True),("page_rows",0),("include_special_tokens",1),("preview_width",0),("preview_width",True),("paginate",1)):
            bad=copy.deepcopy(self.config)
            bad["display"][key]=value
            with self.assertRaises(ConfigError):
                validate_config(bad)

    def test_detection_records_preserve_mapping_and_round_only_presentation(self):
        result=self.functional([self.prediction("q0",phrase="a garbage pile",score=.419058620929718),
                                self.prediction("q1",phrase="a group",score=.49413442611694336)])
        before=copy.deepcopy(result)
        records=[dict(detection_fields(result,d)) for d in result["detections"]]
        self.assertEqual(list(records[0]),list(records[1]))
        self.assertEqual(records[0]["Mapped category"],"garbage_pile")
        self.assertEqual(records[1]["Raw decoded phrase"],'"a group"')
        self.assertEqual(records[1]["Mapped category"],"unmapped")
        self.assertEqual(records[1]["Mapping reason"],"no_exact_alias")
        self.assertEqual(records[1]["Threshold status"],"PASS")
        self.assertEqual(records[1]["Model score"],"0.4941")
        self.assertEqual(records[1]["Box xyxy (pixels)"],"[0.00, 0.00, 4.00, 3.00]")
        for mode in ("summary","full"):
            self.assertIn("not a failed detection",summary(result,mode))
        self.assertEqual(result,before)

    def test_complete_detection_fields_wrap_without_clipping_or_false_columns(self):
        result=self.functional([self.prediction("q0",phrase="a group | with a very long incomplete phrase\n"*4)])
        result["source"]["image_id"]="nested/"*30+"photo.jpg"
        fields=detection_fields(result,result["detections"][0])
        for width in (24,32,49,79,119):
            pages=list(table_pages(("Field","Value"),fields,width=width,page_rows=50))
            self.assertTrue(all(len(line)<=width for page in pages for line in page))
            body="\n".join(line for page in pages for line in page)
            self.assertIn(r"\u007c",body)
            self.assertNotIn("...",body)
            separators=[i for i,c in enumerate(pages[0][0]) if c=="|"]
            for page in pages:
                for line in page:
                    if "|" in line:
                        self.assertEqual([i for i,c in enumerate(line) if c=="|"],separators)


    def bare_detector(self, requested="auto", actual="cuda"):
        d=GroundingDinoDetector.__new__(GroundingDinoDetector)
        d.config=copy.deepcopy(self.config)
        d.config["device"]=requested
        d.device=actual
        d.fallbacks=[]
        d.model=MagicMock()
        d.torch=SimpleNamespace(float32="synthetic-float32",cuda=SimpleNamespace(
            OutOfMemoryError=SyntheticCudaOOM,empty_cache=MagicMock(),synchronize=MagicMock(),is_available=lambda:False))
        return d

    def test_device_selection_explicit_cuda_and_auto_cpu(self):
        d=self.bare_detector()
        self.assertEqual(d.select_device("auto"),"cpu")
        self.assertEqual(d.fallbacks[0]["code"],"cuda_unavailable")
        with self.assertRaises(InferenceError):
            d.select_device("cuda")
        self.assertEqual(d.select_device("cpu"),"cpu")

    def test_auto_oom_retries_cpu_once_and_records_provenance(self):
        d=self.bare_detector()
        d._warm_and_measure=MagicMock(side_effect=[SyntheticCudaOOM("synthetic CUDA OOM"),([], {"scope":"synthetic"})])
        self.assertEqual(d.predict(None),([],{"scope":"synthetic"}))
        self.assertEqual(d.device,"cpu")
        self.assertEqual(d._warm_and_measure.call_count,2)
        self.assertEqual(d.fallbacks[0]["code"],"cuda_out_of_memory")
        d.model.to.assert_called_once_with(device="cpu",dtype="synthetic-float32")

    def test_explicit_cuda_and_unrelated_failures_never_silently_fallback(self):
        for error in (SyntheticCudaOOM("synthetic OOM"),RuntimeError("synthetic other failure")):
            d=self.bare_detector("cuda")
            d._warm_and_measure=MagicMock(side_effect=error)
            with self.assertRaises(InferenceError):
                d.predict(None)
            self.assertEqual(d.fallbacks,[])
            self.assertEqual(d._warm_and_measure.call_count,1)

    def test_warmup_once_per_actual_device_and_timing_scope(self):
        d=self.bare_detector(actual="cpu")
        d.warmed_devices=set()
        d.load_ms=12
        d._once=MagicMock(return_value=[])
        _,first=d._warm_and_measure(None)
        _,second=d._warm_and_measure(None)
        self.assertEqual(d._once.call_count,3)
        self.assertTrue(first["warmup_performed_for_device"])
        self.assertFalse(second["warmup_performed_for_device"])
        self.assertIsNone(second["warmup_ms"])
        self.assertIn("excludes decode/model_load/warmup/duplicate_overlap_review/display",first["scope"])

    def test_pipeline_persists_pending_before_viewer_and_then_shown(self):
        p=self.image_file()
        original=p.read_bytes()
        def viewer(image,result,**kwargs):
            run=next(Path(self.config["paths"]["results"]).iterdir())
            saved=[json.loads(f.read_text()) for f in run.glob("*.json") if len(f.stem)==64]
            self.assertEqual(saved[0]["display"]["status"],"pending")
            self.assertEqual(image.size,(6,4))
        detector=self.detector()
        code,run,records,factory,show=self.run_pipeline([p],detector,viewer)
        self.assertEqual(code,0)
        self.assertEqual(records[0]["status"],"ok")
        self.assertEqual(records[0]["display"]["status"],"shown")
        self.assertEqual(records[0]["model"]["library_version"],"4.51.3")
        factory.assert_called_once_with(self.config,local_files_only=True)
        self.assertEqual(p.read_bytes(),original)
        detector.close.assert_called_once()
        self.assertTrue(all(f.suffix==".json" for f in run.iterdir()))

    def test_pipeline_display_error_preserves_inference_and_continues(self):
        files=[self.image_file("synthetic-a.png"),self.image_file("synthetic-b.png")]
        code,run,records,_,show=self.run_pipeline(files,self.detector(),RuntimeError("synthetic viewer failed"))
        self.assertEqual(code,1)
        self.assertEqual(show.call_count,2)
        for r in records:
            self.assertEqual(r["status"],"ok")
            self.assertIsNone(r["error"])
            self.assertEqual(r["display"]["status"],"error")
            self.assertTrue(r["detections"])

    def test_shared_batch_waits_for_all_saved_results_before_display(self):
        from scripts.viewer import show_image as queue_image
        files=[self.image_file("batch-a.png"),self.image_file("batch-b.png")]
        originals=[p.read_bytes() for p in files]
        def window(entries):
            self.assertEqual(len(entries),2)
            folder=next(Path(self.config["paths"]["results"]).iterdir())
            records=[json.loads(p.read_text()) for p in folder.glob("*.json") if len(p.stem)==64]
            self.assertEqual([r["display"]["status"] for r in records],["pending","pending"])
        with patch("scripts.gui_viewer.open_results_window",side_effect=window) as show:
            code,_,records,_,_=self.run_pipeline(files,self.detector(),queue_image)
        self.assertEqual(code,0);show.assert_called_once()
        self.assertEqual([r["display"]["status"] for r in records],["shown","shown"])
        self.assertEqual([p.read_bytes() for p in files],originals)

    def test_shared_batch_failure_preserves_every_prediction(self):
        from scripts.viewer import show_image as queue_image
        files=[self.image_file("batch-a.png"),self.image_file("batch-b.png")]
        with patch("scripts.gui_viewer.open_results_window",side_effect=RuntimeError("batch fixture failure")):
            code,_,records,_,_=self.run_pipeline(files,self.detector(),queue_image)
        self.assertEqual(code,1)
        for r in records:
            self.assertEqual(r["status"],"ok")
            self.assertEqual(r["display"]["status"],"error")
            self.assertTrue(r["detections"])

    def test_pipeline_disabled_display_and_no_detections(self):
        self.config["display"]["enabled"]=False
        code,_,records,_,show=self.run_pipeline([self.image_file()],self.detector([]))
        self.assertEqual(code,0)
        self.assertEqual(records[0]["status"],"no_detections")
        self.assertEqual(records[0]["display"]["status"],"disabled")
        show.assert_not_called()

    def test_pipeline_all_invalid_is_processing_error_without_viewer(self):
        invalid=self.prediction(score=float("nan"))
        code,_,records,_,show=self.run_pipeline([self.image_file()],self.detector([invalid]))
        self.assertEqual(code,1)
        self.assertEqual(records[0]["status"],"error")
        self.assertEqual(records[0]["error"]["code"],"invalid_predictions")
        self.assertEqual(records[0]["display"]["status"],"not_attempted")
        show.assert_not_called()

    def test_pipeline_prediction_failure_continues_without_reloading_model(self):
        files=[self.image_file("synthetic-a.png"),self.image_file("synthetic-b.png")]
        detector=self.detector()
        detector.predict.side_effect=[InferenceError("inference","prediction_failed","synthetic failure"),
                                      ([self.prediction()],{"scope":"synthetic software fixture", "inference_ms":None})]
        code,_,records,factory,show=self.run_pipeline(files,detector)
        self.assertEqual(code,1)
        self.assertEqual({r["status"] for r in records},{"error","ok"})
        self.assertEqual(detector.predict.call_count,2)
        factory.assert_called_once()
        show.assert_called_once()

    def test_interrupted_viewer_keeps_saved_pending_and_returns_130(self):
        code,_,records,_,_=self.run_pipeline([self.image_file()],self.detector(),KeyboardInterrupt())
        self.assertEqual(code,130)
        self.assertEqual(records[0]["display"]["status"],"pending")
        self.assertEqual(records[0]["status"],"ok")

    def test_pipeline_decode_and_unsupported_never_load_model(self):
        folder=Path(self.config["paths"]["input"])
        folder.mkdir()
        bad=folder/"broken.png"
        bad.write_bytes(b"synthetic decode failure")
        unsupported=folder/"file.txt"
        unsupported.touch()
        code,_,records,factory,show=self.run_pipeline([bad,unsupported],self.detector())
        self.assertEqual(code,1)
        factory.assert_not_called()
        show.assert_not_called()
        self.assertEqual({r["status"] for r in records},{"error","skipped_unsupported"})
        self.assertTrue(all(r["model"]["library_version"] is None for r in records))

    def test_pipeline_startup_error_persisted_and_not_retried_for_each_image(self):
        files=[self.image_file("synthetic-a.png"),self.image_file("synthetic-b.png")]
        with patch("scripts.pipeline.GroundingDinoDetector",side_effect=InferenceError("runtime","missing_dependency","synthetic missing dependency")) as factory, \
                contextlib.redirect_stdout(io.StringIO()):
            code=run_inference(self.config,files)
        self.assertEqual(code,1)
        self.assertEqual(factory.call_count,1)
        run=next(Path(self.config["paths"]["results"]).iterdir())
        records=[json.loads(p.read_text()) for p in run.glob("*.json") if len(p.stem)==64]
        self.assertEqual(len(records),2)
        self.assertTrue(all(r["error"]["code"]=="missing_dependency" and not r["inference"]["completed"] for r in records))

    def test_pipeline_failed_initial_persistence_prevents_viewer(self):
        p=self.image_file()
        with patch("scripts.pipeline.save_result",side_effect=OSError("synthetic initial failure")):
            code,_,_,_,show=self.run_pipeline([p],self.detector())
        self.assertEqual(code,2)
        show.assert_not_called()

    def test_pipeline_failed_final_persistence_leaves_pending(self):
        p=self.image_file()
        with patch("scripts.pipeline.save_display_outcome",side_effect=OSError("synthetic final failure")):
            code,_,records,_,show=self.run_pipeline([p],self.detector())
        self.assertEqual(code,2)
        show.assert_called_once()
        self.assertEqual(records[0]["display"]["status"],"pending")
        self.assertEqual(records[0]["status"],"ok")


if __name__ == "__main__":
    unittest.main()
