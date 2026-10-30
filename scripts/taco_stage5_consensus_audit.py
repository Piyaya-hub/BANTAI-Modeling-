"""CPU-only evaluation of D079 score-preserving consensus on saved D078 vectors."""
from __future__ import annotations

from collections import defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.configuration import strict_load
from scripts.taco_error_analysis import iou
from scripts.taco_evaluation_v3 import coco_metrics
from scripts.taco_stage5_consensus_tiebreak import (
    COACTIVATION_PROFILES, CONFIDENCE_MODES, apply_consensus_tiebreak,
)
from scripts.taco_stage5_material_hierarchy import NAMES
from scripts.taco_stage5_material_hierarchy import map_row
from scripts.taco_training_labels_v2 import LABELS

VECTOR_DIR = ROOT / "runs/training/stage5-score-vectors-5619e219fa264287aa0564c4565a2472"
VECTOR_MANIFEST = VECTOR_DIR / "manifest.json"
VECTOR_FILE = VECTOR_DIR / "query-scores.npz"
TIER2 = ROOT / "runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14"
SAVED_FILE = TIER2 / "soup_tier2_d074_champion_5050.json"
VAL_FILE = ROOT / "exports/taco/strict-clean-stage5-tier2-v1/validation.json"
RIGHTS_FILE = ROOT / "work/stage5-paper/rights_tier2.json"
DELTAS = (0.016, 0.020, 0.025)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_counts(data, rows_by_image, scheme, threshold):
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(map_row(ann, scheme))
    totals = {"TP": 0, "FP": 0, "FN": 0}
    for image in data["images"]:
        image_id = image["id"]
        gts = annotations[image_id]
        matched = set()
        rows = [map_row(row, scheme) for row in rows_by_image[image_id]
                if row.get("valid_clipped_box", True) and row["score"] >= threshold
                and row["bbox"][2] > 0 and row["bbox"][3] > 0]
        rows.sort(key=lambda row: (-row["score"], row["query_id"]))
        for row in rows:
            best = max(((iou(row["bbox"], ann["bbox"]), ann) for ann in gts
                        if ann["category_id"] == row["category_id"]
                        and ann["id"] not in matched),
                       key=lambda pair: (pair[0], -pair[1]["id"]),
                       default=(0.0, None))
            if best[1] is not None and best[0] >= 0.50:
                matched.add(best[1]["id"])
                totals["TP"] += 1
            else:
                totals["FP"] += 1
        totals["FN"] += len(gts) - len(matched)
    return totals


def official_ap(data, rows_by_image, scheme):
    category_names = NAMES[scheme]
    mapped_data = copy.deepcopy(data)
    mapped_data["annotations"] = [map_row(ann, scheme) for ann in data["annotations"]]
    mapped_data["categories"] = [{"id": cid, "name": name}
                                  for cid, name in enumerate(category_names)]
    predictions = []
    for image in data["images"]:
        rows = rows_by_image[image["id"]]
        valid = [row for row in rows if row.get("valid_clipped_box", True)
                 and row["bbox"][2] > 0 and row["bbox"][3] > 0]
        valid.sort(key=lambda row: (-row["score"], row["query_id"]))
        mapped = [map_row(row, scheme) for row in valid]
        predictions.extend({key: row[key] for key in
                            ("image_id", "category_id", "score", "bbox")}
                           for row in mapped[:100])
    return coco_metrics(mapped_data, predictions,
                        cat_ids=list(range(len(category_names))),
                        label_names=category_names)


def main():
    manifest = strict_load(VECTOR_MANIFEST)
    if manifest.get("state") != "soup_tier2_d074_champion_5050" or manifest.get("shape") != [65, 900, 9]:
        raise ValueError("D078 champion vector manifest mismatch")
    if manifest.get("parameter_sha256") != "cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422":
        raise ValueError("D076 checkpoint parameter hash mismatch")
    input_paths = [VECTOR_MANIFEST, VECTOR_FILE, SAVED_FILE, VAL_FILE, RIGHTS_FILE,
                   ROOT / "config/taco_taxonomy.v1.json",
                   ROOT / "config/taco_taxonomy.v2.json"]
    protected_core = [ROOT / "scripts" / name for name in (
        "train_taco.py", "taco_training_data.py", "taco_evaluation.py",
        "training_checkpoint.py", "training_adapter.py", "training_smoke.py")]
    all_inputs = input_paths + protected_core
    before = {str(path.relative_to(ROOT)): digest(path) for path in all_inputs}

    val = strict_load(VAL_FILE)
    saved = strict_load(SAVED_FILE)
    with np.load(VECTOR_FILE, allow_pickle=False) as archive:
        image_ids = archive["image_ids"].tolist()
        scores = archive["scores"].copy()
    if scores.shape != (65, 900, 9) or scores.dtype != np.float32:
        raise ValueError("Expected the saved float32 (65,900,9) score vectors")
    if set(image_ids) != {row["id"] for row in val["images"]}:
        raise ValueError("Vector image IDs differ from frozen validation membership")

    base_rows = defaultdict(list)
    for row in saved["raw_queries"]:
        base_rows[row["image_id"]].append(copy.deepcopy(row))
    for image_id, rows in base_rows.items():
        rows.sort(key=lambda row: row["query_id"])
        if len(rows) != 900 or [row["query_id"] for row in rows] != list(range(900)):
            raise ValueError("Saved raw query identities are incomplete")
    vector_index = {image_id: idx for idx, image_id in enumerate(image_ids)}

    # Reconstruct baseline official predictions and require exact saved AP.
    baseline_9 = official_ap(val, base_rows, "9-specific")
    expected_ap = saved["combined_65"]["AP"]
    if abs(baseline_9["AP"] - expected_ap) > 1e-12:
        raise ValueError("Raw saved-query reconstruction differs from official nine-class AP")
    baseline_5 = official_ap(val, base_rows, "bantai-5-category")
    baseline_9["strict"] = {str(t): strict_counts(val, base_rows, "9-specific", t)
                             for t in (0.05, 0.10, 0.15)}
    baseline_5["strict"] = {str(t): strict_counts(val, base_rows, "bantai-5-category", t)
                             for t in (0.05, 0.10, 0.15)}
    baseline = {"9-specific": baseline_9, "bantai-5-category": baseline_5}

    candidates = []
    query_audit = {}
    for delta in DELTAS:
        for profile, _ in COACTIVATION_PROFILES.items():
            for confidence_mode in CONFIDENCE_MODES:
                rows_by_image = defaultdict(list)
                changed_queries = []
                for image_id, rows in base_rows.items():
                    image_scores = scores[vector_index[image_id]]
                    for row in rows:
                        adjusted, changed = apply_consensus_tiebreak(
                            row, image_scores[row["query_id"]], delta=delta,
                            coactivation=profile, confidence_mode=confidence_mode)
                        rows_by_image[image_id].append(adjusted)
                        if changed:
                            changed_queries.append({"image_id": image_id,
                                "query_id": row["query_id"], "old_category_id": 8,
                                "new_category_id": adjusted["category_id"],
                                "old_score": row["score"], "new_score": adjusted["score"]})
                key = f"delta={delta:.3f}|{profile}|{confidence_mode}"
                result = {"delta": delta, "coactivation": profile,
                          "confidence_mode": confidence_mode,
                          "affected_queries": len(changed_queries),
                          "9-specific": official_ap(val, rows_by_image, "9-specific"),
                          "bantai-5-category": official_ap(val, rows_by_image, "bantai-5-category"),
                          "strict": {scheme: {str(threshold): strict_counts(
                              val, rows_by_image, scheme, threshold)
                              for threshold in (0.05, 0.10, 0.15)}
                              for scheme in ("9-specific", "bantai-5-category")}}
                candidates.append(result)
                query_audit[key] = changed_queries

    image777_idx = vector_index[777]
    q0 = base_rows[777][0]
    q0_vector = scores[image777_idx, 0]
    q0_variants = {}
    for delta in DELTAS:
        for profile in COACTIVATION_PROFILES:
            for confidence_mode in CONFIDENCE_MODES:
                adjusted, changed = apply_consensus_tiebreak(
                    q0, q0_vector, delta=delta, coactivation=profile,
                    confidence_mode=confidence_mode)
                q0_variants[f"{delta:.3f}|{profile}|{confidence_mode}"] = {
                    "changed": changed, "category_id": adjusted["category_id"],
                    "category": LABELS[adjusted["category_id"]],
                    "score": adjusted["score"]}

    after = {str(path.relative_to(ROOT)): digest(path) for path in all_inputs}
    if before != after:
        raise RuntimeError("D079 CPU audit modified or observed a changed protected input")

    out = ROOT / "runs/training" / ("stage5-score-consensus-audit-" + uuid.uuid4().hex)
    out.mkdir(parents=False, exist_ok=False)
    summary = {
        "schema_version": "bantai.stage5-consensus-tiebreak-audit.v1",
        "authorization": "D-079", "device": "CPU only",
        "gpu_forwards": 0, "training_updates": 0, "backward_passes": 0,
        "test_split_access": False,
        "vector_artifact": str(VECTOR_FILE.relative_to(ROOT)),
        "vector_sha256": digest(VECTOR_FILE), "vector_shape": list(scores.shape),
        "checkpoint_parameter_sha256": manifest["parameter_sha256"],
        "baseline_official_9_specific_AP": expected_ap,
        "baseline_bantai_5_category_AP": baseline_5["AP"],
        "bantai_5_category_classes": list(NAMES["bantai-5-category"]),
        "organic_ground_truth_objects": sum(
            map_row(ann, "bantai-5-category")["category_id"] == 2
            for ann in val["annotations"]),
        "input_sha256": before,
        "deltas": list(DELTAS), "coactivation_profiles": {
            name: {"minimum_subtypes": count, "subtype_score_floor": floor}
            for name, (count, floor) in COACTIVATION_PROFILES.items()},
        "confidence_modes": list(CONFIDENCE_MODES),
        "score_preserving_rule": "change only raw paper queries; retain raw top1 score, or compare non-increasing plastic_max mode",
        "baseline_metrics": baseline, "candidates": candidates,
        "affected_query_ids": query_audit,
        "image777_gt259_q0": {"score_vector": dict(zip(LABELS, map(float, q0_vector))),
                               "variants": q0_variants},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = ("delta", "coactivation", "confidence_mode", "affected_queries",
              "AP_9_specific", "AP_bantai_5_category",
              "TP_9_005", "FP_9_005", "FN_9_005",
              "TP_9_010", "FP_9_010", "FN_9_010",
              "TP_9_015", "FP_9_015", "FN_9_015",
              "TP_5_005", "FP_5_005", "FN_5_005",
              "TP_5_010", "FP_5_010", "FN_5_010",
              "TP_5_015", "FP_5_015", "FN_5_015")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for item in candidates:
            row = {"delta": item["delta"], "coactivation": item["coactivation"],
                   "confidence_mode": item["confidence_mode"],
                   "affected_queries": item["affected_queries"],
                   "AP_9_specific": item["9-specific"]["AP"],
                   "AP_bantai_5_category": item["bantai-5-category"]["AP"]}
            for prefix, scheme in (("9", "9-specific"), ("5", "bantai-5-category")):
                for threshold, suffix in (("0.05", "005"), ("0.1", "010"), ("0.15", "015")):
                    row.update({f"{metric}_{prefix}_{suffix}": item["strict"][scheme][threshold][metric]
                                for metric in ("TP", "FP", "FN")})
            writer.writerow(row)
    print(json.dumps({"output": str(out), "candidates": len(candidates),
                      "baseline_9_AP": baseline_9["AP"],
                      "baseline_bantai_5_AP": baseline_5["AP"],
                      "q0": q0_variants}, indent=2))


if __name__ == "__main__":
    main()
