"""Multi-Scale Query Fusion + D-079 Consensus Integration (D-087).

CPU-only evaluation and artifact generator combining standard-resolution (384/576)
and high-resolution (448/672) validation queries and score vectors for 9 active
supervised classes (0..8) without GPU or model forwards.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.configuration import strict_load
from scripts.persistence import atomic_json
from scripts.taco_error_analysis import iou
from scripts.taco_stage5_material_hierarchy import evaluate_case, OBJECT_REGISTRY
from scripts.taco_stage5_consensus_tiebreak import apply_consensus_tiebreak

VAL_JSON = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/validation.json"
D083_RUN = ROOT / "runs/training/stage5-d083-core-20261031-a"
D086_RUN = ROOT / "runs/training/stage5-d086-highres-20261031-a"


def fuse_and_evaluate(val_data, std_queries, hr_queries, std_vecs, std_img_map, hr_vecs, hr_img_map,
                      consensus_mode="none"):
    """
    Fuses 900 standard-resolution and 900 high-resolution queries per image using
    class-aware IoU 0.55 NMS (keeping top 100 queries per image).
    
    consensus_mode:
      - 'none': raw multi-scale query fusion
      - 'pre_nms': D-079 consensus tiebreak applied before NMS
      - 'post_nms': D-079 consensus tiebreak applied to kept queries after NMS
    """
    by_img = defaultdict(lambda: {"std": [], "hr": []})
    for r in std_queries:
        by_img[r["image_id"]]["std"].append(dict(r))
    for r in hr_queries:
        by_img[r["image_id"]]["hr"].append(dict(r))

    fused_pool = []
    fused_vectors_by_img = {}
    changes = []

    for img_id in sorted(by_img.keys()):
        pools = by_img[img_id]
        std_v = std_vecs["scores"][std_img_map[img_id]]
        hr_v = hr_vecs["scores"][hr_img_map[img_id]]

        combined = []
        for r in pools["std"]:
            v = std_v[r["query_id"]]
            if consensus_mode == "pre_nms":
                r_adj, ch = apply_consensus_tiebreak(r, v, delta=0.016, coactivation="3 plastic subtypes >= 0.20", confidence_mode="top1_score")
                if ch:
                    changes.append({"image_id": img_id, "query_id": r["query_id"], "scale": "std", "from": r["category_id"], "to": r_adj["category_id"], "score": r["score"]})
                r = r_adj
            combined.append((r, v))

        for r in pools["hr"]:
            v = hr_v[r["query_id"]]
            if consensus_mode == "pre_nms":
                r_adj, ch = apply_consensus_tiebreak(r, v, delta=0.016, coactivation="3 plastic subtypes >= 0.20", confidence_mode="top1_score")
                if ch:
                    changes.append({"image_id": img_id, "query_id": r["query_id"], "scale": "hr", "from": r["category_id"], "to": r_adj["category_id"], "score": r["score"]})
                r = r_adj
            combined.append((r, v))

        combined.sort(key=lambda item: -item[0]["score"])
        kept = []
        for r, v in combined:
            suppress = False
            for kr, kv in kept:
                if kr["category_id"] != r["category_id"]:
                    continue
                if iou(kr["bbox"], r["bbox"]) >= 0.55:
                    suppress = True
                    break
            if not suppress:
                kept.append((r, v))
                if len(kept) == 100:
                    break

        if consensus_mode == "post_nms":
            final_kept = []
            for r, v in kept:
                r_adj, ch = apply_consensus_tiebreak(r, v, delta=0.016, coactivation="3 plastic subtypes >= 0.20", confidence_mode="top1_score")
                if ch:
                    changes.append({"image_id": img_id, "query_id": r["query_id"], "scale": "fused", "from": r["category_id"], "to": r_adj["category_id"], "score": r["score"]})
                final_kept.append((r_adj, v))
            kept = final_kept

        # Assign clean query_id 0..99 for the 100 fused queries per image
        clean_rows = []
        for new_qid, (r, v) in enumerate(kept):
            row_copy = dict(r)
            row_copy["fused_query_id"] = new_qid
            clean_rows.append(row_copy)

        fused_pool.extend(clean_rows)
        fused_vectors_by_img[img_id] = np.stack([item[1] for item in kept], axis=0).astype(np.float32)

    m9 = evaluate_case(val_data, fused_pool, "9-specific", "raw")
    m5 = evaluate_case(val_data, fused_pool, "bantai-5-category", "raw")

    return {
        "consensus_mode": consensus_mode,
        "fused_query_count": len(fused_pool),
        "changes_count": len(changes),
        "changes": changes,
        "9_specific": m9["metrics"],
        "bantai_5_category": m5["metrics"],
        "fused_pool": fused_pool,
        "fused_vectors_by_img": fused_vectors_by_img,
    }


def run_d087_multiscale_consensus(output_dir: Path = D086_RUN):
    start = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    val_data = strict_load(VAL_JSON)

    d083_json = strict_load(D083_RUN / "soup_d083_d074_5050.json")
    d083_vecs = np.load(D083_RUN / "best-score-vectors.npz")

    d086_s2575_json = strict_load(D086_RUN / "soup_d086_d083_2575_std384.json")
    d086_s2575_vecs = np.load(D086_RUN / "scores-soup_d086_d083_2575_std384.npz")

    d086_hr_json = strict_load(D086_RUN / "validation-d086-hr448-e1.json")
    d086_hr_vecs = np.load(D086_RUN / "scores-validation-d086-hr448-e1.npz")

    d083_img_map = {img_id: idx for idx, img_id in enumerate(d083_vecs["image_ids"])}
    s2575_img_map = {img_id: idx for idx, img_id in enumerate(d086_s2575_vecs["image_ids"])}
    hr_img_map = {img_id: idx for idx, img_id in enumerate(d086_hr_vecs["image_ids"])}

    pairings = {
        "D083_Champion_384_plus_D086_HR448": {
            "std_queries": d083_json["raw_queries"],
            "hr_queries": d086_hr_json["raw_queries"],
            "std_vecs": d083_vecs,
            "std_img_map": d083_img_map,
            "hr_vecs": d086_hr_vecs,
            "hr_img_map": hr_img_map,
        },
        "D086_Soup2575_384_plus_D086_HR448": {
            "std_queries": d086_s2575_json["raw_queries"],
            "hr_queries": d086_hr_json["raw_queries"],
            "std_vecs": d086_s2575_vecs,
            "std_img_map": s2575_img_map,
            "hr_vecs": d086_hr_vecs,
            "hr_img_map": hr_img_map,
        },
    }

    results = {}
    winning_pool = None
    winning_vectors = None
    winning_pair_name = None
    winning_mode = None
    highest_ap = -1.0

    for pair_name, cfg in pairings.items():
        results[pair_name] = {}
        for mode in ("none", "pre_nms", "post_nms"):
            eval_res = fuse_and_evaluate(
                val_data, cfg["std_queries"], cfg["hr_queries"],
                cfg["std_vecs"], cfg["std_img_map"],
                cfg["hr_vecs"], cfg["hr_img_map"],
                consensus_mode=mode,
            )
            pool = eval_res.pop("fused_pool")
            vecs = eval_res.pop("fused_vectors_by_img")
            results[pair_name][mode] = eval_res

            ap9 = eval_res["9_specific"]["AP"]
            if ap9 > highest_ap:
                highest_ap = ap9
                winning_pool = pool
                winning_vectors = vecs
                winning_pair_name = pair_name
                winning_mode = mode

    # Align images in array for export
    sorted_image_ids = sorted(winning_vectors.keys())
    scores_array = np.stack([winning_vectors[img_id] for img_id in sorted_image_ids], axis=0)
    image_ids_array = np.array(sorted_image_ids, dtype=np.int64)

    # Save score vectors npz
    scores_npz_path = output_dir / "scores-multiscale-consensus.npz"
    np.savez_compressed(scores_npz_path, image_ids=image_ids_array, scores=scores_array)

    # Save multiscale-consensus-replay.json
    replay_meta = {
        "authorization": "D-087",
        "winning_pairing": winning_pair_name,
        "winning_consensus_mode": winning_mode,
        "highest_9_specific_AP": highest_ap,
        "vector_artifact": scores_npz_path.name,
        "vector_shape": list(scores_array.shape),
        "vector_dtype": str(scores_array.dtype),
        "fused_query_count": len(winning_pool),
        "all_pairings_summary": {
            pname: {
                m: {
                    "9_specific_AP": pdata[m]["9_specific"]["AP"],
                    "9_specific_AP50": pdata[m]["9_specific"]["AP50"],
                    "9_specific_AR100": pdata[m]["9_specific"]["AR100"],
                    "bantai_5_category_AP": pdata[m]["bantai_5_category"]["AP"],
                    "bantai_5_category_AP50": pdata[m]["bantai_5_category"]["AP50"],
                    "bantai_5_category_AR100": pdata[m]["bantai_5_category"]["AR100"],
                    "changes_count": pdata[m]["changes_count"],
                }
                for m in pdata
            }
            for pname, pdata in results.items()
        },
        "winning_metrics": {
            "9_specific": results[winning_pair_name][winning_mode]["9_specific"],
            "bantai_5_category": results[winning_pair_name][winning_mode]["bantai_5_category"],
        },
        "raw_queries": winning_pool,
        "elapsed_seconds": time.perf_counter() - start,
    }
    atomic_json(output_dir / "multiscale-consensus-replay.json", replay_meta, replace=True)

    audit_summary = {
        "authorization": "D-087",
        "status": "complete",
        "winning_pairing": winning_pair_name,
        "winning_consensus_mode": winning_mode,
        "winning_9_specific_AP": highest_ap,
        "winning_bantai_5_category_AP": results[winning_pair_name][winning_mode]["bantai_5_category"]["AP"],
        "results": results,
    }
    atomic_json(output_dir / "multiscale-consensus-audit.json", audit_summary, replace=True)

    print(json.dumps({
        "status": "complete",
        "winning_pairing": winning_pair_name,
        "winning_consensus_mode": winning_mode,
        "9_specific_AP": highest_ap,
        "bantai_5_category_AP": results[winning_pair_name][winning_mode]["bantai_5_category"]["AP"],
        "scores_shape": list(scores_array.shape),
    }, indent=2))
    return audit_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=D086_RUN)
    args = parser.parse_args()
    run_d087_multiscale_consensus(args.output_dir)
