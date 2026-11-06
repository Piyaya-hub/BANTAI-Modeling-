"""CPU-only material-pooling audit over D-078 saved per-query vectors."""
from __future__ import annotations

from collections import defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.configuration import strict_load
from scripts.taco_error_analysis import iou
from scripts.taco_stage5_material_hierarchy import MAPS, NAMES, map_row
from scripts.taco_evaluation_v3 import coco_metrics

RUN = ROOT / "runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14"
SAVED = RUN / "soup_tier2_d074_champion_5050.json"
VAL = ROOT / "exports/taco/strict-clean-stage5-tier2-v1/validation.json"
VECTORS_ROOT = ROOT / "runs/training"

# Each pooling option is prespecified and only combines the known subtype
# probabilities. Paper is left unchanged; no validation-fitted prior is used.
POOLERS = ("top1_argmax", "max_subtype", "noisy_or", "bounded_top2_sum")
THRESHOLDS = (.05, .10, .15)


def aggregate(scores, scheme, pooler):
    if scheme not in ("5-material", "4-material") or pooler not in POOLERS:
        raise ValueError("Unknown material pooling contract")
    groups = defaultdict(list)
    for cls, material in enumerate(MAPS[scheme]):
        groups[material].append(cls)
    out = np.empty((scores.shape[0], len(NAMES[scheme])), dtype=np.float32)
    for material, classes in groups.items():
        subset = scores[:, classes]
        if pooler == "top1_argmax":
            values = subset.max(axis=1)
        elif pooler == "max_subtype":
            values = subset.max(axis=1)
        elif pooler == "noisy_or":
            values = 1.0 - np.prod(1.0 - subset, axis=1)
        else:
            values = np.sort(subset, axis=1)[:, -min(2, len(classes)):].sum(axis=1)
            values = np.minimum(values, 1.0)
        out[:, material] = values
    return out


def build_predictions(data, scores_by_image, scheme, pooler):
    ids = list(range(len(NAMES[scheme])))
    preds = []
    raw_by_image = {}
    for image in data["images"]:
        image_id = image["id"]
        rows = scores_by_image[image_id]
        all_scores = rows["scores"]
        if pooler == "top1_argmax":
            specific = all_scores.argmax(axis=1)
            confidence = all_scores.max(axis=1)
            categories = np.asarray(MAPS[scheme], dtype=np.int8)[specific]
        else:
            scores = aggregate(all_scores, scheme, pooler)
            categories = scores.argmax(axis=1)
            confidence = scores.max(axis=1)
        query_order = np.lexsort((np.arange(900), -confidence))
        raw = []
        for q in query_order:
            row = rows["boxes"][q]
            cid = int(categories[q])
            value = float(confidence[q])
            raw.append({"image_id": image_id, "query_id": int(q), "category_id": cid,
                        "score": value, "bbox": row["bbox"],
                        "valid_clipped_box": row["valid_clipped_box"]})
        selected = [r for r in raw if r["valid_clipped_box"]][:100]
        preds.extend({k: r[k] for k in ("image_id", "category_id", "score", "bbox")}
                     for r in selected)
        raw_by_image[image_id] = raw
    gt = copy.deepcopy(data)
    gt["annotations"] = [map_row(a, scheme) for a in data["annotations"]]
    gt["categories"] = [{"id": i, "name": n} for i, n in enumerate(NAMES[scheme])]
    metrics = coco_metrics(gt, preds, cat_ids=ids, label_names=NAMES[scheme])
    return metrics, raw_by_image, preds


def strict_metrics(data, raw, scheme, threshold):
    gt_by_image = defaultdict(list)
    for ann in data["annotations"]:
        gt_by_image[ann["image_id"]].append(map_row(ann, scheme))
    tp = fp = fn = 0
    per_image = {}
    for image in data["images"]:
        anns = gt_by_image[image["id"]]
        candidates = [r for r in raw[image["id"]]
                      if r["score"] >= threshold and r["valid_clipped_box"]]
        matched = set()
        local_tp = local_fp = 0
        for row in candidates:
            best = max(((iou(row["bbox"], ann["bbox"]), ann) for ann in anns
                        if ann["category_id"] == row["category_id"] and ann["id"] not in matched),
                       key=lambda pair: (pair[0], -pair[1]["id"]), default=(0., None))
            if best[1] is not None and best[0] >= .50:
                matched.add(best[1]["id"]); tp += 1; local_tp += 1
            else:
                fp += 1; local_fp += 1
        fn += len(anns) - len(matched)
        per_image[image["id"]] = {"TP": local_tp, "FP": local_fp,
                                   "FN": len(anns)-len(matched)}
    return {"threshold": threshold, "TP": tp, "FP": fp, "FN": fn,
            "images": per_image}


def run(vectors_dir=None):
    if vectors_dir is None:
        candidates = sorted(VECTORS_ROOT.glob("stage5-score-vectors-*/manifest.json"),
                           key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError("No D-078 score-vector extraction found")
        vectors_dir = candidates[-1].parent
    vectors_dir = Path(vectors_dir)
    manifest = strict_load(vectors_dir / "manifest.json")
    if manifest["shape"] != [65, 900, 9] or manifest["parameter_sha256"] != "cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422":
        raise ValueError("Score vector artifact is not the authorized D-076 champion extraction")
    val = strict_load(VAL)
    saved = strict_load(SAVED)
    protected_files = [SAVED, VAL,
        ROOT / "exports/taco/strict-clean-stage5-tier2-v1/preparation_manifest.json",
        ROOT / "exports/taco/strict-clean-stage5-tier2-v1/validation.json",
        ROOT / "work/stage5-paper/rights_tier2.json",
        ROOT / "config/taco_taxonomy.v2.json"]
    before_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in protected_files}
    array = np.load(vectors_dir / manifest["artifact"])
    scores = array["scores"]
    ids = array["image_ids"].tolist()
    if scores.shape != (65, 900, 9) or scores.dtype != np.float32:
        raise ValueError("Invalid extracted vector shape/dtype")
    saved_by_image = defaultdict(dict)
    for row in saved["raw_queries"]:
        saved_by_image[row["image_id"]][row["query_id"]] = row
    row_by_image = {}
    for index, image_id in enumerate(ids):
        rows = [saved_by_image[image_id][q] for q in range(900)]
        row_by_image[image_id] = {"scores": scores[index], "boxes": rows}
    states = {}
    strict_rows = []
    for scheme in ("5-material", "4-material"):
        for pooler in POOLERS:
            metrics, raw, predictions = build_predictions(val, row_by_image, scheme, pooler)
            strict = {str(t): strict_metrics(val, raw, scheme, t) for t in THRESHOLDS}
            states[f"{scheme}:{pooler}"] = {"metrics": metrics, "strict": strict}
            for threshold in THRESHOLDS:
                outcome = strict[ str(threshold) ]
                strict_rows.append({"scheme": scheme, "pooler": pooler,
                                    **{k: outcome[k] for k in ("threshold", "TP", "FP", "FN")},
                                    "AP": metrics["AP"], "AP50": metrics["AP50"],
                                    "AR100": metrics["AR100"]})

    stage2 = strict_load(ROOT / "exports/taco/strict-clean-stage2-v1/validation.json")
    legacy = {(a["image_id"], a.get("source_taco", {}).get("annotation_id")): a
              for a in stage2["annotations"]}
    source_id = next((key[1] for key, ann in legacy.items()
                      if key[0] == 777 and ann["id"] == 259), None)
    if source_id is None: raise ValueError("Cannot resolve frozen Image 777 historical GT259")
    target = next(a for a in val["annotations"]
                  if a["image_id"] == 777 and a.get("source_taco", {}).get("annotation_id") == source_id)
    target_image_index = ids.index(777)
    target_rows = saved_by_image[777]
    query_details = []
    for q, raw in target_rows.items():
        overlap = iou(raw["bbox"], target["bbox"])
        if q == 0 or overlap >= .50:
            vector = scores[target_image_index, q].astype(float).tolist()
            detail = {"query_id": q, "iou": overlap, "official_top1_class_id": raw["category_id"],
                      "official_top1_class": NAMES["9-specific"][raw["category_id"]],
                      "official_top1_score": raw["score"],
                      "score_vector": dict(zip(NAMES["9-specific"], vector))}
            for scheme in ("5-material", "4-material"):
                detail[scheme] = {}
                for pooler in POOLERS[1:]:
                    probs = aggregate(scores[target_image_index, q:q+1], scheme, pooler)[0]
                    detail[scheme][pooler] = {name: float(probs[i]) for i, name in enumerate(NAMES[scheme])}
            query_details.append(detail)

    after_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in protected_files}
    if before_hashes != after_hashes:
        raise RuntimeError("Saved inputs/rights/export changed during the read-only audit")

    out = ROOT / "runs/training" / ("stage5-score-vector-audit-" + uuid.uuid4().hex)
    out.mkdir(parents=False, exist_ok=False)
    summary = {"schema_version": "bantai.stage5-material-score-pooling-audit.v1",
        "authorization": "D-078", "diagnostic_only": True,
        "model_execution": False, "gpu_execution": False, "test_split_access": False,
        "source_vectors_manifest_sha256": hashlib.sha256((vectors_dir / "manifest.json").read_bytes()).hexdigest(),
        "source_vectors_sha256": manifest["artifact_sha256"],
        "protected_input_sha256": before_hashes,
        "validation_images": 65, "query_count_per_image": 900,
        "completed_vector_extraction_forwards": manifest["completed_score_extraction_forwards"],
        "failed_prior_forward_attempts": manifest["failed_prior_forward_attempts"],
        "cumulative_detector_forward_invocations_for_d078": manifest["cumulative_detector_forward_invocations_for_d078"],
        "authorization_forward_cap": manifest["authorization_forward_cap"],
        "forward_cap_overrun": manifest["forward_cap_overrun"],
        "poolers": list(POOLERS), "thresholds": list(THRESHOLDS),
        "strict_matching": "one-to-one correct mapped class, score threshold, IoU>=0.50, all valid 900 queries; no NMS",
        "ap_protocol": "D-078 diagnostic COCO evaluation on transformed per-query material scores, top100 per image, no NMS",
        "official_9_specific_metrics_unchanged": True,
        "image777_gt259": {"source_taco_annotation_id": source_id,
            "stage5_annotation_id": target["id"], "image_id": 777,
            "ground_truth_category": "plastic_wrapper", "queries_q0_or_iou_ge_0.50": query_details},
        "states": states, "strict_rows": strict_rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("scheme", "pooler", "threshold", "TP", "FP", "FN", "AP", "AP50", "AR100"))
        writer.writeheader(); writer.writerows(strict_rows)
    print(out)
    print(json.dumps({k: states[k]["metrics"]["AP"] for k in states}, indent=2))
    print(json.dumps(summary["image777_gt259"], indent=2))
    return out, summary


if __name__ == "__main__":
    run()
