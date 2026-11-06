"""CPU audit of saved Stage 5 winning-class queries and display profiles.

Reads only the 65-image validation annotation JSON and nine saved prediction
JSONs. It never loads checkpoint weights, test metadata/photos, or a model.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import uuid

from .taco_error_analysis import iou
from .taco_stage5_display_profiles import (NMS_CHOICES, THRESHOLD_CHOICES,
                                           visible_rows)
from .taco_stage5_material_hierarchy import map_row

ROOT = Path(__file__).resolve().parents[1]
RUN1 = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4"
STATES = {
    "baseline_256_384": RUN1 / "baseline.json",
    "epoch2_256_384": RUN1 / "validation-epoch-2.json",
    "blend2575_384_576": RUN1 / "eval_blend_2575_384_576.json",
    "highres_epoch1_384_576": ROOT / "runs/training/stage5-paper-558bcc18d6cf4556a966d8cfc320c968/validation-epoch-1.json",
    "tier1_epoch1": ROOT / "runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/validation-epoch-1.json",
    "tier1_soup_20_80": ROOT / "runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/soup_tier1_hr_blend2575.json",
    "tier2_epoch1": ROOT / "runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14/validation-epoch-1.json",
    "tier2_soup_20_80": ROOT / "runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14/soup_tier2_hr_blend2575.json",
    "tier2_soup_d074_50_50": ROOT / "runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14/soup_tier2_d074_champion_5050.json",
    "balanced40_60_tier1": ROOT / "runs/training/stage5-balanced-soup-eval-1453160757f14d6c89104989c9593a68/balanced40_60_tier1.json",
    "balanced40_60_tier2": ROOT / "runs/training/stage5-balanced-soup-eval-1453160757f14d6c89104989c9593a68/balanced40_60_tier2.json",
}
VALIDATION = ROOT / "exports/taco/strict-clean-stage5-v1/validation.json"
LEGACY_VALIDATION = ROOT / "exports/taco/strict-clean-stage2-v1/validation.json"


def greedy_matching(annotations, rows):
    """Strict one-to-one correct-class matching at IoU .50 after display filters."""
    matched = set()
    tp = fp = 0
    for row in sorted(rows, key=lambda r: (-r["score"], r.get("query_id", 0))):
        best = max(((iou(row["bbox"], ann["bbox"]), ann) for ann in annotations
                    if ann["id"] not in matched and ann["category_id"] == row["category_id"]),
                   key=lambda pair: (pair[0], -pair[1]["id"]), default=(0., None))
        if best[1] is not None and best[0] >= .50:
            matched.add(best[1]["id"])
            tp += 1
        else:
            fp += 1
    return {"TP": tp, "FP": fp, "FN": len(annotations) - tp}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    legacy = json.loads(LEGACY_VALIDATION.read_text(encoding="utf-8"))
    images = {row["id"]: row for row in validation["images"]}
    annotations = {image_id: [] for image_id in images}
    for ann in validation["annotations"]:
        annotations[ann["image_id"]].append(ann)
    states = {}
    for name, path in STATES.items():
        saved = json.loads(path.read_text(encoding="utf-8"))
        rows = saved["raw_queries"]
        if len(rows) != 65 * 900 or {r["image_id"] for r in rows} != set(images):
            raise ValueError(f"Incomplete 65x900 saved prediction set: {name}")
        grouped = {image_id: [] for image_id in images}
        for row in rows:
            grouped[row["image_id"]].append(row)
        for image_id in grouped:
            grouped[image_id].sort(key=lambda r: (-r["score"], r["query_id"]))
        states[name] = grouped
    if len(annotations[777]) < 7:
        raise ValueError("Expected Image 777 validation annotations are absent")
    stage5_by_source = {(a["image_id"], a.get("source_taco", {}).get("annotation_id")): a["id"]
                        for a in validation["annotations"]}
    historical_to_stage5 = {}
    for ann in legacy["annotations"]:
        key = (ann["image_id"], ann.get("source_taco", {}).get("annotation_id"))
        if key in stage5_by_source:
            historical_to_stage5[ann["id"]] = stage5_by_source[key]
    return validation, images, annotations, states, historical_to_stage5


def image777_ledger(annotations, states, historical_to_stage5):
    target = {a["id"]: a for a in annotations[777]}
    # Resolve review IDs through source TACO annotation provenance, not an
    # assumed contiguous-ID offset between the original and Stage 5 exports.
    export_ids = {historical: historical_to_stage5[historical]
                  for historical in (259, *range(278, 285)) if historical in historical_to_stage5}
    if len(export_ids) != 8 or not all(export_id in target for export_id in export_ids.values()):
        raise ValueError("Image 777 GT IDs 259 and 278..284 are required")
    result = {"gt259": [], "cigarettes": []}
    for state, by_image in states.items():
        rows = by_image[777]
        ranked = {r["query_id"]: rank for rank, r in enumerate(rows, 1)}
        for row in rows:
            if iou(row["bbox"], target[export_ids[259]]["bbox"]) >= 0.50:
                row_mapped = map_row(row, "5-material")
                result["gt259"].append({
                "state": state, "query_id": row["query_id"],
                "export_annotation_id": export_ids[259],
                    "specific_class": row_mapped["specific_class"],
                    "material_5": row_mapped["mapped_category"],
                    "material_4": map_row(row, "4-material")["mapped_category"],
                    "score": row["score"], "iou": iou(row["bbox"], target[export_ids[259]]["bbox"]),
                    "rank": ranked[row["query_id"]],
                })
        for gt_id in range(278, 285):
            candidates = [r for r in rows if r["category_id"] == 0]
            localized = [r for r in candidates
                         if iou(r["bbox"], target[export_ids[gt_id]]["bbox"]) >= .50]
            # Prefer the highest-scoring correctly-classified localized query;
            # if no IoU>=.50 cigarette query exists, expose the best-IoU one.
            matched = (max(localized, key=lambda r: (r["score"],
                        iou(r["bbox"], target[export_ids[gt_id]]["bbox"]), -r["query_id"]))
                       if localized else max(candidates, key=lambda r: (
                        iou(r["bbox"], target[export_ids[gt_id]]["bbox"]), r["score"],
                        -r["query_id"]), default=None))
            value = iou(matched["bbox"], target[export_ids[gt_id]]["bbox"]) if matched else 0.0
            result["cigarettes"].append({
                "state": state, "gt_id": gt_id, "export_annotation_id": export_ids[gt_id],
                "query_id": matched["query_id"] if matched else None,
                "iou": value, "score": matched["score"] if matched else None,
                "rank": ranked[matched["query_id"]] if matched else None,
                "score_band": ("0.05_to_lt_0.10" if matched and .05 <= matched["score"] < .10
                               else "at_least_0.10" if matched and matched["score"] >= .10
                               else "below_0.05_or_missing"),
            })
    result["gt259"].sort(key=lambda r: (r["state"], -r["score"], -r["iou"], r["query_id"]))
    return result


def profile_audit(annotations, states):
    cases = []
    for state, by_image in states.items():
        for threshold in (.05, .10, .15):
            for threshold_mode in THRESHOLD_CHOICES:
                for nms_mode in NMS_CHOICES:
                    for scheme in ("9-specific", "5-material"):
                        totals = {"TP": 0, "FP": 0, "FN": 0}
                        kept = 0
                        for image_id, raw in by_image.items():
                            rows = visible_rows(raw, scheme, nms_mode, threshold, threshold_mode)
                            kept += len(rows)
                            gt = [map_row(a, scheme) for a in annotations[image_id]]
                            match = greedy_matching(gt, rows)
                            for key in totals:
                                totals[key] += match[key]
                        tp, fp, fn = (totals[k] for k in ("TP", "FP", "FN"))
                        precision = tp / (tp + fp) if tp + fp else 0.
                        recall = tp / (tp + fn) if tp + fn else 0.
                        cases.append({"state": state, "taxonomy": scheme,
                                      "slider": threshold, "threshold_mode": threshold_mode,
                                      "nms": nms_mode, **totals, "Precision": precision,
                                      "Recall": recall,
                                      "F1": 2 * precision * recall / (precision + recall)
                                      if precision + recall else 0.,
                                      "retained_boxes_65": kept})
    return cases


def run_audit(out_dir=None):
    input_paths = list(STATES.values()) + [VALIDATION, LEGACY_VALIDATION]
    before = {str(p.relative_to(ROOT)): sha(p) for p in input_paths}
    validation, images, annotations, states, historical_to_stage5 = load_inputs()
    ledger = image777_ledger(annotations, states, historical_to_stage5)
    cases = profile_audit(annotations, states)
    after = {str(p.relative_to(ROOT)): sha(p) for p in input_paths}
    if before != after:
        raise RuntimeError("Saved inputs changed during read-only audit")
    if out_dir is None:
        out_dir = ROOT / "runs/training" / ("stage5-image777-calibration-" + uuid.uuid4().hex)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": "bantai.stage5-image777-calibration.v1",
        "diagnostic_only": True,
        "model_execution_during_this_saved_query_audit": False,
        "gpu_execution_during_this_saved_query_audit": False,
        "balanced_soup_prediction_states_include_bounded_prior_validation_forwards": True,
        "test_json_opened": False,
        "test_photo_access": False,
        "source_prediction_rows_per_state": 65 * 900,
        "score_vectors_available": False,
        "pooling_note": "Inputs contain only each query's winning class and scalar score; no class score/logit vector exists for subtype pooling or logit-prior calibration.",
        "cigarette_match_rule": "Highest-scoring predicted cigarette with IoU>=0.50; if none, highest-IoU predicted cigarette fallback.",
        "threshold_profiles": list(THRESHOLD_CHOICES),
        "slider_values": [.05, .10, .15],
        "nms_profiles": list(NMS_CHOICES),
        "image_count": len(images),
        "annotation_count": len(validation["annotations"]),
        "image777_historical_to_stage5_annotation_ids": {
            str(i): historical_to_stage5[i] for i in (259, *range(278, 285))},
        "input_sha256": before,
        "image777": ledger,
        "strict_matching_cases": cases,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = ("state", "taxonomy", "slider", "threshold_mode", "nms", "TP", "FP", "FN",
              "Precision", "Recall", "F1", "retained_boxes_65")
    with (out_dir / "strict-matching.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cases)
    return out_dir, summary


if __name__ == "__main__":
    folder, result = run_audit()
    print(json.dumps({"output": str(folder), "states": len(STATES),
                      "strict_matching_cases": len(result["strict_matching_cases"]),
                      "image777_gt259_rows": len(result["image777"]["gt259"]),
                      "image777_cigarette_rows": len(result["image777"]["cigarettes"])}))
