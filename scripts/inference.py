"""Pinned Transformers baseline, imported lazily only by explicit inference mode."""

import gc
import importlib.metadata
from pathlib import Path
import platform
import struct
import time

from .configuration import REVISION, finite_number
from .postprocessing import map_phrase
from .results import candidate

DIRECT_VERSIONS = {"torch": "2.7.1", "torchvision": "0.22.1", "transformers": "4.51.3",
                   "Pillow": "11.2.1", "matplotlib": "3.10.3"}


class InferenceError(RuntimeError):
    def __init__(self, stage, code, message):
        super().__init__(message)
        self.stage, self.code = stage, code

    def record(self):
        return {"stage": self.stage, "code": self.code, "message": str(self)}


def runtime_versions():
    versions = {}
    for package, expected in DIRECT_VERSIONS.items():
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise InferenceError("runtime", "missing_dependency", f"{package} is missing; run with the existing verified target .venv") from exc
        if observed.split("+")[0] != expected:
            raise InferenceError("runtime", "dependency_version_mismatch", f"{package}={observed}; expected {expected}; preserve/review the verified environment")
        versions[package] = observed
    return versions


def package_snapshot():
    """Only installed package names/versions, never environment variables or secrets."""
    return dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                       if d.metadata.get("Name")))


def active_text_tokens(inputs):
    """Match the pinned model's attention-mask semantics before logit auditing.

    GroundingDinoContrastiveEmbedding fills masked positions and the tail up to
    max_text_len with -inf deliberately. Only those positions may use that
    sentinel; active tokens must still be finite. The model defaults a missing
    attention mask to ones_like(input_ids).
    """
    ids = inputs["input_ids"].detach().cpu().tolist()
    if len(ids) != 1 or not isinstance(ids[0], list) or not ids[0]:
        raise InferenceError("postprocessing", "invalid_text_mask", "Expected one nonempty input-token sequence")
    mask_tensor = inputs.get("attention_mask")
    if mask_tensor is None:
        return [True] * len(ids[0])
    mask = mask_tensor.detach().cpu().tolist()
    if (len(mask) != 1 or not isinstance(mask[0], list) or len(mask[0]) != len(ids[0])
            or any(v not in (0, 1) for v in mask[0])):
        raise InferenceError("postprocessing", "invalid_text_mask", "Attention mask must match input tokens with 0/1 values")
    return [bool(v) for v in mask[0]]


def extract_candidates(outputs, inputs, processor, config, width, height, torch):
    """Audit raw queries before upstream threshold comparisons can hide invalids.

    Grounded postprocessing uses the pinned API. Invalid raw queries below the
    score threshold remain diagnostic candidates, with an empty phrase if the
    upstream filter did not decode it. No phrase/score is invented.
    """
    logits = outputs.logits.detach().cpu().tolist()
    normalized = outputs.pred_boxes.detach().cpu().tolist()
    score_tensor = torch.sigmoid(outputs.logits).max(dim=-1).values
    scores = score_tensor.detach().cpu().tolist()
    # Use torch's dtype-aware comparison exactly as the pinned processor does;
    # comparing exported float32 values to Python floats can differ at a boundary.
    keep = (score_tensor > config["threshold"]).detach().cpu().tolist()
    if len(logits) != 1 or len(normalized) != 1 or len(scores) != 1:
        raise InferenceError("postprocessing", "invalid_output_shape", "Expected a single-image output batch")
    rows, boxes, query_scores = logits[0], normalized[0], scores[0]
    if not (len(rows) == len(boxes) == len(query_scores)):
        raise InferenceError("postprocessing", "invalid_output_shape", "Inconsistent query counts")
    active = active_text_tokens(inputs)
    if rows and not any(active[:len(rows[0])]):
        raise InferenceError("postprocessing", "invalid_text_mask", "No active text token in the model's output frame")
    selected = [i for i, retained in enumerate(keep[0]) if retained]
    grounded = processor.post_process_grounded_object_detection(
        outputs, input_ids=inputs["input_ids"], threshold=config["threshold"],
        text_threshold=config["text_threshold"], target_sizes=[(height, width)])
    if len(grounded) != 1:
        raise InferenceError("postprocessing", "invalid_output_shape", "Expected one grounded result")
    detected = grounded[0]
    phrases = detected["text_labels"]
    pixel_boxes = detected["boxes"].detach().cpu().tolist()
    selected_scores = detected["scores"].detach().cpu().tolist()
    if not (len(selected) == len(phrases) == len(pixel_boxes) == len(selected_scores)):
        raise InferenceError("postprocessing", "invalid_output_shape", "Grounded results do not match raw threshold selection")
    by_query = {q: (phrase, box, score) for q, phrase, box, score in zip(selected, phrases, pixel_boxes, selected_scores)}
    candidates = []
    for i, (tokens, box, score) in enumerate(zip(rows, boxes, query_scores)):
        if not isinstance(tokens, list) or not tokens or not isinstance(box, list) or len(box) != 4:
            raise InferenceError("postprocessing", "invalid_output_shape", "Malformed logits or normalized box")
        bad_logits = any(not finite_number(v) and not (
            v == float("-inf") and (j >= len(active) or not active[j]))
            for j, v in enumerate(tokens))
        # -inf is expected only in known masked/tail padding slots. NaN/+inf
        # anywhere and -inf on active tokens remain invalid, even after sigmoid.
        phrase = ""
        if i in by_query:
            phrase, pixel_box, score = by_query[i]
        else:
            cx, cy, bw, bh = box
            pixel_box = [(cx - bw/2)*width, (cy - bh/2)*height,
                         (cx + bw/2)*width, (cy + bh/2)*height]
        if not isinstance(phrase, str):
            raise InferenceError("postprocessing", "invalid_phrase", "Grounded phrase must be a string")
        category, reason = map_phrase(phrase)
        d = candidate(f"q{i}", phrase, pixel_box, None if bad_logits else score,
                      width, height, category=category, mapping_reason=reason)
        d["query_index"] = i
        if bad_logits:
            d["numeric_diagnostics"].append({"field": "logits", "reason": "nonfinite_model_logit"})
        if d["state"] == "invalid" or i in by_query:
            candidates.append(d)
    return candidates


class GroundingDinoDetector:
    def __init__(self, config, *, local_files_only=False):
        self.config = config
        self.versions = runtime_versions()  # Fail before downloads if the runtime is missing/mismatched.
        try:
            import torch
            from transformers import AutoConfig, AutoModelForZeroShotObjectDetection, AutoProcessor
        except Exception as exc:
            raise InferenceError("runtime", "dependency_import_failed", f"Verified library import failed ({type(exc).__name__}); inspect the target environment") from exc
        self.torch = torch
        self.model = None
        self.fallbacks = []
        self.warmed_devices = set()
        self.forward_attempts = 0
        self.completed_forwards = 0
        try:
            self.device = self.select_device(config["device"])
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError("device", "device_probe_failed", f"Device selection failed ({type(exc).__name__}); no fallback is inferred") from exc
        cache = Path(config["paths"]["cache"])
        cache.mkdir(parents=True, exist_ok=True)
        kwargs = {"revision": REVISION, "cache_dir": str(cache),
                  "local_files_only": local_files_only, "trust_remote_code": False}
        start = time.perf_counter()
        try:
            model_config = AutoConfig.from_pretrained(config["model"]["id"], **kwargs)
            if getattr(model_config, "_commit_hash", None) != REVISION:
                raise InferenceError("model_load", "revision_mismatch", "Model config did not resolve to the pinned revision")
            if (model_config.model_type != "grounding-dino" or model_config.backbone_config.model_type != "swin"
                    or model_config.backbone_config.depths != [2, 2, 6, 2]
                    or model_config.backbone_config.num_heads != [3, 6, 12, 24]):
                raise InferenceError("model_load", "configuration_mismatch", "Expected Grounding DINO with Swin backbone")
            model_config.disable_custom_kernels = True
            self.processor = AutoProcessor.from_pretrained(config["model"]["id"], **kwargs)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                config["model"]["id"], config=model_config, use_safetensors=True,
                torch_dtype=torch.float32, **kwargs)
            if getattr(self.model.config, "_commit_hash", None) != REVISION:
                raise InferenceError("model_load", "revision_mismatch", "Loaded model did not retain the pinned revision")
            self.model.eval()
            self._place_model()
            self._synchronize()
            self.preprocessing = {"processor_revision": REVISION, "exif_orientation": "in_memory_before_processor",
                                  "color_mode": "RGB", "coordinate_frame": "exif_oriented_original",
                                  "target_sizes_order": "height,width", "image_processor": self.processor.image_processor.to_dict(),
                                  "logit_audit": "finite active tokens; -inf allowed only at masked/text-tail padding positions"}
        except InferenceError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise InferenceError("model_load", "checkpoint_load_failed", f"Pinned checkpoint/processor load failed ({type(exc).__name__}); check cache, Hub access and model memory") from exc
        self.load_ms = (time.perf_counter() - start) * 1000

    def select_device(self, requested):
        if requested == "cpu":
            return "cpu"
        available = self.torch.cuda.is_available()
        if requested == "cuda" and not available:
            raise InferenceError("device", "cuda_unavailable", "Explicit CUDA request is unavailable; choose --device cpu or diagnose the target CUDA runtime")
        if available:
            return "cuda"
        self.fallbacks.append({"stage": "device", "code": "cuda_unavailable", "from": "auto", "to": "cpu"})
        return "cpu"

    def _cpu_fallback(self, stage):
        self.fallbacks.append({"stage": stage, "code": "cuda_out_of_memory", "from": "cuda", "to": "cpu"})
        self.device = "cpu"
        self.model.to(device="cpu", dtype=self.torch.float32)
        gc.collect()
        self.torch.cuda.empty_cache()

    def _place_model(self):
        retry = False
        try:
            self.model.to(device=self.device, dtype=self.torch.float32)
        except self.torch.cuda.OutOfMemoryError:
            if self.config["device"] != "auto" or self.device != "cuda":
                raise
            retry = True
        if retry:
            self._cpu_fallback("model_load")

    def _synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def _once(self, image):
        inputs = self.processor(images=image, text=self.config["prompt"], return_tensors="pt")
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            self.forward_attempts += 1
            outputs = self.model(**inputs)
            self.completed_forwards += 1
        return extract_candidates(outputs, inputs, self.processor, self.config, image.width, image.height, self.torch)

    def _warm_and_measure(self, image):
        warmup_ms = None
        if self.device not in self.warmed_devices:
            self._synchronize()
            start = time.perf_counter()
            self._once(image)
            self._synchronize()
            warmup_ms = (time.perf_counter() - start) * 1000
            self.warmed_devices.add(self.device)
        self._synchronize()
        start = time.perf_counter()
        predictions = self._once(image)
        self._synchronize()
        timing = {"scope": "processor_transfer_forward_grounded_postprocess_candidate_extraction; excludes decode/model_load/warmup/duplicate_overlap_review/display",
                  "inference_ms": (time.perf_counter() - start) * 1000,
                  "warmup_ms": warmup_ms, "warmup_performed_for_device": warmup_ms is not None,
                  "repetitions": 1, "model_load_ms": self.load_ms}
        return predictions, timing

    def predict(self, image):
        retry = False
        try:
            return self._warm_and_measure(image)
        except self.torch.cuda.OutOfMemoryError:
            if self.config["device"] != "auto" or self.device != "cuda":
                raise InferenceError("inference", "cuda_out_of_memory", "CUDA out of memory; explicit CUDA does not silently fall back") from None
            retry = True
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError("inference", "prediction_failed", f"Inference/postprocessing failed ({type(exc).__name__}); no successful result is claimed") from exc
        if retry:
            # Leave the exception handler first so failed CUDA-frame references can release.
            try:
                self._cpu_fallback("inference")
                return self._warm_and_measure(image)
            except Exception as exc:
                raise InferenceError("inference", "cpu_retry_failed", f"Single CPU retry failed ({type(exc).__name__}) after CUDA OOM") from exc

    def environment(self):
        cuda = self.torch.cuda
        device = {"actual_device": self.device, "cuda_build": self.torch.version.cuda}
        if self.device == "cuda":
            device.update(name=cuda.get_device_name(0), capability=list(cuda.get_device_capability(0)),
                          memory_free_total_bytes=list(cuda.mem_get_info()))
        return {"python": platform.python_version(), "python_bits": struct.calcsize("P")*8,
                "platform": platform.platform(), "direct_versions": self.versions,
                "packages": package_snapshot(), "device": device,
                "driver": None, "driver_note": "Not queried in this run; 2A driver observations are historical",
                "cuda_install_source_from_2a": "https://download.pytorch.org/whl/cu118",
                "resolved_revision": REVISION, "model_load_ms": self.load_ms, "fallbacks": self.fallbacks,
                "forward_attempts_including_warmup": self.forward_attempts,
                "completed_forwards_including_warmup": self.completed_forwards}

    def close(self):
        self.model = None
        gc.collect()
