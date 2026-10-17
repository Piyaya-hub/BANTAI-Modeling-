"""Section 7 result utilities. Callers supply predictions; this module runs no model.

Synthetic software tests are separate from observed model outputs.
"""

import copy
from pathlib import Path
import re

from .configuration import MODEL_ID, REVISION, finite_number, strict_load, validate_config
from .persistence import atomic_json, sanitize_numbers, source_key

CATEGORIES = {"garbage_pile": "region", "individual_litter": "object",
              "group_of_individual_litter": "region"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_size(width, height):
    return all(type(v) is int and v > 0 for v in (width, height))


def candidate(candidate_id, raw_phrase, box, score, width, height, *, category=None,
              mapping_reason="no_exact_alias"):
    """Validate/clamp supplied pixel-edge coordinates; never guess invalid values."""
    require(valid_size(width, height), "oriented positive integer dimensions required")
    require(isinstance(candidate_id, str) and candidate_id, "candidate ID required")
    require(isinstance(raw_phrase, str), "raw phrase must be a string")
    require(category is None or category in CATEGORIES, "unknown category")
    require(isinstance(mapping_reason, str) and mapping_reason, "mapping reason required")
    diagnostics = []
    values = list(box) if isinstance(box, (list, tuple)) else []
    if len(values) != 4:
        diagnostics.append({"field": "raw_box_xyxy", "reason": "expected_four_coordinates"})
        values = [None] * 4
    raw = []
    for i, value in enumerate(values):
        if not finite_number(value):
            raw.append(None)
            diagnostics.append({"field": "raw_box_xyxy", "index": i, "reason": "nonfinite_or_invalid_coordinate"})
        else:
            raw.append(value)
    clipped = None
    if all(v is not None for v in raw):
        x0, y0, x1, y1 = raw
        if x0 >= x1 or y0 >= y1:
            diagnostics.append({"field": "raw_box_xyxy", "reason": "invalid_geometry"})
        else:
            clipped = [max(0, min(width, x0)), max(0, min(height, y0)),
                       max(0, min(width, x1)), max(0, min(height, y1))]
            if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                clipped = None
                diagnostics.append({"field": "box_xyxy", "reason": "empty_after_clipping"})
    if not finite_number(score) or not 0 <= score <= 1:
        diagnostic = {"field": "model_score", "reason": "nonfinite_or_invalid_score"}
        if finite_number(score):
            diagnostic.update(reason="out_of_range_score", finite_raw_value=score)
        diagnostics.append(diagnostic)
        score = None
    return {"id": candidate_id, "origin": "model_prediction", "raw_phrase": raw_phrase,
            "category": category, "mapping_reason": mapping_reason,
            "level": CATEGORIES.get(category, "unknown"), "model_score": score,
            "raw_box_xyxy": raw, "box_xyxy": clipped,
            "state": "invalid" if diagnostics else "retained", "duplicate_of": None,
            "numeric_diagnostics": diagnostics}


def valid_candidates(result):
    """Candidates eligible for a later viewer/summary, including unmapped ones."""
    return [d for d in result["detections"] if d["state"] == "retained"]


def make_result(run_id, source, config, *, detections=(), inference_completed=False,
                execution=None, error=None, unsupported=False, library_version=None):
    """Build functional results only with explicit inference completion/evidence.

    A decoded/prepared photo is deliberately rejected here without an error or
    explicit completed inference. The CLI uses a separate preparation schema.
    """
    config = validate_config(config)
    detections = copy.deepcopy(list(detections))
    if unsupported:
        require(not inference_completed and not detections, "unsupported input cannot contain inference")
        status = "skipped_unsupported"
        error = None
    elif error is not None:
        require(not inference_completed and not detections, "pre-inference failure cannot contain predictions")
        status = "error"
    else:
        require(inference_completed is True, "preparation is not successful inference")
        require(isinstance(execution, dict), "completed inference requires execution metadata")
        status = "ok" if any(d["state"] in ("retained", "duplicate") for d in detections) else "no_detections"
        if detections and status == "no_detections":
            status = "error"
            error = {"stage": "postprocessing", "code": "invalid_predictions", "message": "All extracted candidates are invalid"}
    inference = {"requested_device": config["device"], "actual_device": None, "dtype": config["dtype"],
                 "environment_manifest": None, "preprocessing_manifest": None,
                 "disable_custom_kernels": True, "prompt": config["prompt"],
                 "threshold": config["threshold"], "text_threshold": config["text_threshold"],
                 "mapping_version": "not_applied", "duplicate_policy": "not_applied", "fallback": None,
                 "completed": inference_completed}
    if execution:
        # Only execution provenance may be supplied; configured settings stay authoritative.
        require(set(execution) <= {"actual_device", "environment_manifest", "preprocessing_manifest",
                                   "mapping_version", "duplicate_policy", "fallback"}, "unknown execution field")
        inference.update(copy.deepcopy(execution))
    enabled = config["display"]["enabled"]
    result = {"schema_version": "bantai.functional.v1", "run_id": run_id, "source": copy.deepcopy(source),
              "model": {"id": config["model"]["id"], "revision": config["model"]["revision"],
                        "library": "transformers", "library_version": library_version},
              "inference": inference, "status": status,
              "display": {"enabled": enabled, "status": ("pending" if enabled else "disabled")
                          if status in ("ok", "no_detections") else "not_attempted", "error": None},
              "score_definition": {"range": [0, 1], "meaning": "model score; not accuracy"},
              "box_format": "xyxy_original_display_pixel_edges", "detections": detections,
              "review": {"flags": [], "memberships": [], "physical_object_count": None,
                         "independent_region_count": None, "notes": "Membership and counts unresolved"},
              "timing": {"scope": "not_measured", "inference_ms": None}, "error": error}
    return prepare_result(result)


def validate_result(r):
    """Reject inconsistent/malformed records before persistence; no repair by guessing."""
    try:
        required = {"schema_version", "run_id", "source", "model", "inference", "status", "display",
                    "score_definition", "box_format", "detections", "review", "timing", "error"}
        require(isinstance(r, dict) and required <= r.keys(), "missing required result fields")
        require(r["schema_version"] == "bantai.functional.v1", "unknown result schema")
        require(isinstance(r["run_id"], str) and re.fullmatch(r"[A-Za-z0-9_-]+", r["run_id"]), "unsafe run ID")
        source, model, inference = r["source"], r["model"], r["inference"]
        require(isinstance(source["image_id"], str) and source["image_id"], "source identity required")
        require(source["coordinate_frame"] == "exif_oriented_original", "wrong coordinate frame")
        require(source["sha256"] is None or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]), "invalid SHA-256")
        size = valid_size(source["width"], source["height"])
        require(size or (source["width"] is None and source["height"] is None), "invalid dimensions")
        require(model["library"] == "transformers" and (model["library_version"] is None
                or isinstance(model["library_version"], str)), "invalid model library metadata")
        require(model["id"] == MODEL_ID and model["revision"] == REVISION, "2A-verified model/revision required")
        for key in ("threshold", "text_threshold"):
            require(finite_number(inference[key]) and 0 <= inference[key] <= 1, "invalid required threshold")
        require(inference["requested_device"] in ("cpu", "cuda", "auto") and inference["dtype"] == "float32", "invalid device/dtype")
        require(inference["disable_custom_kernels"] is True and isinstance(inference["prompt"], str)
                and inference["prompt"].strip(), "invalid inference settings")
        require(type(inference["completed"]) is bool, "inference completion must be explicit")
        for key in ("environment_manifest", "preprocessing_manifest", "mapping_version", "duplicate_policy", "fallback"):
            require(key in inference, f"missing inference metadata: {key}")
        status, display = r["status"], r["display"]
        require(status in ("ok", "no_detections", "error", "skipped_unsupported"), "invalid processing status")
        require(type(display["enabled"]) is bool, "display enabled must be boolean")
        success = status in ("ok", "no_detections")
        allowed = ({"pending", "shown", "error"} if display["enabled"] else {"disabled"}) if success else {"not_attempted"}
        require(display["status"] in allowed, "display state contradicts processing/intent")
        for error, needed in ((r["error"], status == "error"), (display["error"], display["status"] == "error")):
            require((error is not None) == needed, "error object contradicts status")
            if needed:
                require(all(isinstance(error[k], str) and error[k] for k in ("stage", "code", "message")), "malformed error")
        if success or inference["completed"]:
            require(model["library_version"] == "4.51.3", "completed inference requires observed pinned library version")
            require(size and source["sha256"] is not None, "completed inference requires source dimensions/hash")
            require(inference["actual_device"] in ("cpu", "cuda"), "actual device required")
            require(all(isinstance(inference[k], str) and inference[k] for k in
                        ("environment_manifest", "preprocessing_manifest")), "execution manifests required")
        require(not success or inference["completed"], "success requires completed inference")
        require(r["box_format"] == "xyxy_original_display_pixel_edges", "invalid box format")
        require(r["score_definition"] == {"range": [0, 1], "meaning": "model score; not accuracy"}, "invalid score definition")
        detections = r["detections"]
        if "query_diagnostics" in r:
            raw = r["query_diagnostics"]
            require(raw["schema_version"] == "bantai.query-diagnostics.v1", "unknown query diagnostics schema")
            require(raw["threshold"] == inference["threshold"] and raw["text_threshold"] == inference["text_threshold"], "diagnostic thresholds differ")
            tokens = raw["tokens"]
            positions = [t["position"] for t in tokens]
            require(positions == sorted(set(positions)) and all(type(p) is int and p >= 0 for p in positions), "invalid token alignment")
            require(all(type(t["token_id"]) is int and isinstance(t["token"], str) and type(t["special"]) is bool for t in tokens), "invalid token metadata")
            for i, q in enumerate(raw["queries"]):
                require(q["id"] == f"q{i}" and q["query_index"] == i, "invalid query alignment")
                require(len(q["token_scores"]) == len(tokens) == len(q["token_threshold_pass"]), "invalid score alignment")
                require(all(v is None or (finite_number(v) and 0 <= v <= 1) for v in [q["model_score"], *q["token_scores"]]), "invalid diagnostic scores")
                require(type(q["threshold_pass"]) is bool and all(type(v) is bool for v in q["token_threshold_pass"]), "invalid diagnostic decisions")
                require(q["raw_phrase"] is None or isinstance(q["raw_phrase"], str), "invalid raw phrase")
                require(q["threshold_pass"] or (q["raw_phrase"] is None and q["category"] is None), "rejected queries cannot have invented labels")
        require(isinstance(detections, list), "detections must be a list")
        ids = set()
        for d in detections:
            require(d["id"] not in ids, "duplicate candidate ID")
            ids.add(d["id"])
            require(d["origin"] == "model_prediction", "derived groups are outside the foundation")
            expected = candidate(d["id"], d["raw_phrase"], d["raw_box_xyxy"], d["model_score"],
                                 source["width"], source["height"], category=d["category"], mapping_reason=d["mapping_reason"])
            require(isinstance(d["raw_box_xyxy"], list) and len(d["raw_box_xyxy"]) == 4
                    and all(v is None or finite_number(v) for v in d["raw_box_xyxy"]), "raw coordinates must be finite or null")
            require(d["model_score"] is None or (finite_number(d["model_score"]) and 0 <= d["model_score"] <= 1), "invalid scores must be null")
            require(d["level"] == expected["level"] and d["box_xyxy"] == expected["box_xyxy"], "inconsistent candidate geometry/level")
            require(isinstance(d["numeric_diagnostics"], list), "numeric diagnostics required")
            invalid = expected["state"] == "invalid"
            require(d["state"] in (("invalid",) if invalid else ("retained", "duplicate")), "invalid candidate state")
            require(not invalid or d["numeric_diagnostics"], "invalid candidates require diagnostics")
            for diag in d["numeric_diagnostics"]:
                require(all(isinstance(diag[k], str) and diag[k] for k in ("field", "reason")), "malformed diagnostic")
            if d["state"] != "duplicate":
                require(d["duplicate_of"] is None, "unexpected duplicate link")
        by_id = {d["id"]: d for d in detections}
        for d in detections:
            if d["state"] == "duplicate":
                target = by_id.get(d["duplicate_of"])
                require(target is not None and target["state"] == "retained" and target["id"] != d["id"], "invalid duplicate reference")
                from .postprocessing import normalize_phrase
                require(d["raw_box_xyxy"] == target["raw_box_xyxy"]
                        and normalize_phrase(d["raw_phrase"]) == normalize_phrase(target["raw_phrase"]), "duplicate must have identical raw box/normalized phrase")
                require(target["model_score"] >= d["model_score"], "duplicate target must retain highest score")
        require(status != "ok" or bool(valid_candidates(r)), "ok requires a retained valid candidate")
        require(status != "no_detections" or not detections, "no_detections cannot hide invalid predictions")
        if status in ("error", "skipped_unsupported") and detections:
            require(status == "error" and r["error"]["code"] == "invalid_predictions"
                    and inference["completed"] and all(d["state"] == "invalid" for d in detections), "failure predictions inconsistent")
        if status == "error" and inference["completed"]:
            require(bool(detections) and r["error"]["code"] == "invalid_predictions"
                    and all(d["state"] == "invalid" for d in detections), "completed error must preserve invalid predictions")
        if status == "skipped_unsupported":
            require(not inference["completed"], "unsupported input cannot have completed inference")
        review = r["review"]
        require(review["physical_object_count"] is None and review["independent_region_count"] is None,
                "foundation cannot resolve physical/independent-region counts")
        require(review["memberships"] == [] and isinstance(review["flags"], list), "memberships unresolved")
        timing = r["timing"]
        require(isinstance(timing["scope"], str) and timing["scope"], "timing scope required")
        require(timing["inference_ms"] is None or (finite_number(timing["inference_ms"]) and timing["inference_ms"] >= 0), "invalid inference timing")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed result: {exc}") from exc
    return r


def prepare_result(record):
    diagnostics = []
    result = sanitize_numbers(copy.deepcopy(record), diagnostics)
    if isinstance(result.get("timing"), dict):
        value = result["timing"].get("inference_ms")
        if value is not None and (not finite_number(value) or value < 0):
            result["timing"]["inference_ms"] = None
            diagnostics.append({"field": "$.timing.inference_ms", "reason": "invalid_timing"})
    if diagnostics:
        result.setdefault("numeric_diagnostics", []).extend(diagnostics)
    return validate_result(result)


def save_result(folder, record):
    result = prepare_result(record)
    folder = Path(folder)
    require(folder.name == result["run_id"], "result must belong to its reserved run directory")
    path = folder / (source_key(result["source"]) + ".json")
    atomic_json(path, result)
    return path


def save_display_outcome(path, status, *, error=None):
    """Persist a caller-observed display outcome; does not start/render a viewer.

    This is the sole functional-result update operation. The initial pending
    record must already exist. Failure leaves it intact. One writer per result.
    """
    require(status in ("shown", "error"), "final display outcome must be shown/error")
    result = validate_result(strict_load(path))
    path = Path(path)
    require(path.parent.name == result["run_id"] and path.name == source_key(result["source"]) + ".json",
            "saved path does not match run/source identity")
    require(result["display"]["status"] == "pending", "only a saved pending result can be finalized")
    result["display"].update(status=status, error=error)
    result = prepare_result(result)
    atomic_json(path, result, replace=True)
    return result
