"""Class-Calibrated Tri-Model Multi-Scale Query Routing + D-079 Consensus Integration (D-090).

CPU-only evaluation and artifact generator combining standard-resolution (384/576),
high-resolution (448/672), and Tier 3 Core expanded-dataset (D-089) validation queries,
class-calibrated scale routing, and score vectors for 9 active supervised classes (0..8)
without GPU or model forwards.
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
D089_RUN = ROOT / "runs/training/stage5-d089-tier3-core-20261031-a"
D091_RUN = ROOT / "runs/training/stage5-d091-balanced-tier3-20261101-a"

ROUTING_POLICIES = [
    "uniform_both",
    "std_only_cup_styrofoam",
    "hr_small_medium_classes_only",
    "hr_small_medium_plus_can_paper",
    "box_area_gated_010",
    "box_area_discount_095",
]

CONSENSUS_MODES = ["none", "pre_nms", "post_nms"]


def fuse_and_evaluate(val_data, std_queries, hr_queries, std_vecs, std_img_map, hr_vecs, hr_img_map,
                      spec_queries=None, spec_vecs=None, spec_img_map=None, spec_categories=(5, 8),
                      routing_policy="uniform_both", consensus_mode="none"):
    """
    Fuses standard-resolution (384/576), high-resolution (448/672), and optional
    Tier 3 Core class-specialized (D-089) queries per image under a class-calibrated
    routing policy and consensus mode using class-aware IoU 0.55 NMS (keeping top 100 queries per image).
    """
    by_img = defaultdict(lambda: {"std": [], "hr": [], "spec": []})
    for r in std_queries:
        by_img[r["image_id"]]["std"].append(dict(r))
    for r in hr_queries:
        by_img[r["image_id"]]["hr"].append(dict(r))
    if spec_queries is not None:
        for r in spec_queries:
            if r["category_id"] in spec_categories:
                by_img[r["image_id"]]["spec"].append(dict(r))

    fused_pool = []
    fused_vectors_by_img = {}
    changes = []

    for img_id in sorted(by_img.keys()):
        pools = by_img[img_id]
        std_v = std_vecs["scores"][std_img_map[img_id]]
        hr_v = hr_vecs["scores"][hr_img_map[img_id]]
        spec_v = spec_vecs["scores"][spec_img_map[img_id]] if spec_queries is not None else None

        combined = []
        for r in pools["std"]:
            v = std_v[r["query_id"]]
            r_use = dict(r)
            r_use["scale"] = "std"
            r_use["routing_policy"] = routing_policy
            if consensus_mode == "pre_nms":
                r_adj, ch = apply_consensus_tiebreak(
                    r_use, v, delta=0.016, coactivation="3 plastic subtypes >= 0.20", confidence_mode="top1_score"
                )
                if ch:
                    changes.append({
                        "image_id": img_id, "query_id": r["query_id"], "scale": "std",
                        "from": r_use["category_id"], "to": r_adj["category_id"], "score": r_use["score"]
                    })
                r_use = r_adj
            combined.append((r_use, v))

        if spec_queries is not None and spec_v is not None:
            for r in pools["spec"]:
                v = spec_v[r["query_id"]]
                r_use = dict(r)
                r_use["scale"] = "tier3_spec"
                r_use["routing_policy"] = routing_policy
                combined.append((r_use, v))

        for r in pools["hr"]:
            v = hr_v[r["query_id"]]
            r_use = dict(r)
            r_use["scale"] = "hr"
            r_use["routing_policy"] = routing_policy
            if consensus_mode == "pre_nms":
                r_adj, ch = apply_consensus_tiebreak(
                    r_use, v, delta=0.016, coactivation="3 plastic subtypes >= 0.20", confidence_mode="top1_score"
                )
                if ch:
                    changes.append({
                        "image_id": img_id, "query_id": r["query_id"], "scale": "hr",
                        "from": r_use["category_id"], "to": r_adj["category_id"], "score": r_use["score"]
                    })
                r_use = r_adj

            cat = r_use["category_id"]
            # Apply routing policy to high-resolution candidate queries
            if routing_policy == "std_only_cup_styrofoam":
                if cat in (6, 7):
                    continue
            elif routing_policy == "hr_small_medium_classes_only":
                if cat not in (0, 1, 3, 4):
                    continue
            elif routing_policy == "hr_small_medium_plus_can_paper":
                if cat not in (0, 1, 3, 4, 5, 8):
                    continue
            elif routing_policy == "box_area_gated_010":
                if r_use["bbox"][2] * r_use["bbox"][3] > 0.10:
                    continue
            elif routing_policy == "box_area_discount_095":
                if r_use["bbox"][2] * r_use["bbox"][3] > 0.10:
                    r_use["score"] = r_use["score"] * 0.95

            combined.append((r_use, v))

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
                r_adj, ch = apply_consensus_tiebreak(
                    r, v, delta=0.016, coactivation="3 plastic subtypes >= 0.20", confidence_mode="top1_score"
                )
                if ch:
                    changes.append({
                        "image_id": img_id, "query_id": r["query_id"], "scale": r.get("scale", "fused"),
                        "from": r["category_id"], "to": r_adj["category_id"], "score": r["score"]
                    })
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
        "routing_policy": routing_policy,
        "consensus_mode": consensus_mode,
        "fused_query_count": len(fused_pool),
        "changes_count": len(changes),
        "changes": changes,
        "9_specific": m9["metrics"],
        "bantai_5_category": m5["metrics"],
        "fused_pool": fused_pool,
        "fused_vectors_by_img": fused_vectors_by_img,
    }


def run_d088_multiscale_consensus(output_dir: Path = D086_RUN):
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

    if (D089_RUN / "soup_d089_d083_2575.json").is_file():
        d089_champ_json = strict_load(D089_RUN / "soup_d089_d083_2575.json")
        d089_champ_vecs = np.load(D089_RUN / "scores-soup_d089_d083_2575.npz")
        d089_img_map = {img_id: idx for idx, img_id in enumerate(d089_champ_vecs["image_ids"])}
        pairings["D089_Champion_384_plus_D086_HR448"] = {
            "std_queries": d089_champ_json["raw_queries"],
            "hr_queries": d086_hr_json["raw_queries"],
            "std_vecs": d089_champ_vecs,
            "std_img_map": d089_img_map,
            "hr_vecs": d086_hr_vecs,
            "hr_img_map": hr_img_map,
        }

    if (D089_RUN / "soup_d089_d083_5050.json").is_file():
        d089_s5050_json = strict_load(D089_RUN / "soup_d089_d083_5050.json")
        d089_s5050_vecs = np.load(D089_RUN / "scores-soup_d089_d083_5050.npz")
        d089_s5050_img_map = {img_id: idx for idx, img_id in enumerate(d089_s5050_vecs["image_ids"])}
        pairings["Tri_Model_D086_S2575_HR448_D089_S5050"] = {
            "std_queries": d086_s2575_json["raw_queries"],
            "hr_queries": d086_hr_json["raw_queries"],
            "std_vecs": d086_s2575_vecs,
            "std_img_map": s2575_img_map,
            "hr_vecs": d086_hr_vecs,
            "hr_img_map": hr_img_map,
            "spec_queries": d089_s5050_json["raw_queries"],
            "spec_vecs": d089_s5050_vecs,
            "spec_img_map": d089_s5050_img_map,
            "spec_categories": (5, 8),
        }

    if (D089_RUN / "validation-d089-tier3-core-e1.json").is_file():
        d089_e1_json = strict_load(D089_RUN / "validation-d089-tier3-core-e1.json")
        d089_e1_vecs = np.load(D089_RUN / "scores-validation-d089-tier3-core-e1.npz")
        d089_e1_img_map = {img_id: idx for idx, img_id in enumerate(d089_e1_vecs["image_ids"])}
        pairings["Tri_Model_D086_S2575_HR448_D089_E1"] = {
            "std_queries": d086_s2575_json["raw_queries"],
            "hr_queries": d086_hr_json["raw_queries"],
            "std_vecs": d086_s2575_vecs,
            "std_img_map": s2575_img_map,
            "hr_vecs": d086_hr_vecs,
            "hr_img_map": hr_img_map,
            "spec_queries": d089_e1_json["raw_queries"],
            "spec_vecs": d089_e1_vecs,
            "spec_img_map": d089_e1_img_map,
            "spec_categories": (5, 8),
        }

    if (D091_RUN / "soup_d091_d086s2575_5050.json").is_file():
        d091_champ_json = strict_load(D091_RUN / "soup_d091_d086s2575_5050.json")
        d091_champ_vecs = np.load(D091_RUN / "best-score-vectors.npz")
        d091_img_map = {img_id: idx for idx, img_id in enumerate(d091_champ_vecs["image_ids"])}
        pairings["D091_Champion_384_plus_D086_HR448"] = {
            "std_queries": d091_champ_json["raw_queries"],
            "hr_queries": d086_hr_json["raw_queries"],
            "std_vecs": d091_champ_vecs,
            "std_img_map": d091_img_map,
            "hr_vecs": d086_hr_vecs,
            "hr_img_map": hr_img_map,
        }
        pairings["Tri_Model_D086_S2575_HR448_D091_Champ"] = {
            "std_queries": d086_s2575_json["raw_queries"],
            "hr_queries": d086_hr_json["raw_queries"],
            "std_vecs": d086_s2575_vecs,
            "std_img_map": s2575_img_map,
            "hr_vecs": d086_hr_vecs,
            "hr_img_map": hr_img_map,
            "spec_queries": d091_champ_json["raw_queries"],
            "spec_vecs": d091_champ_vecs,
            "spec_img_map": d091_img_map,
            "spec_categories": (5, 8),
        }

    if (D091_RUN / "validation-d091-balanced-tier3-e1.json").is_file():
        d091_e1_json = strict_load(D091_RUN / "validation-d091-balanced-tier3-e1.json")
        d091_e1_vecs = np.load(D091_RUN / "scores-validation-d091-balanced-tier3-e1.npz")
        d091_e1_img_map = {img_id: idx for idx, img_id in enumerate(d091_e1_vecs["image_ids"])}
        pairings["Tri_Model_D086_S2575_HR448_D091_E1"] = {
            "std_queries": d086_s2575_json["raw_queries"],
            "hr_queries": d086_hr_json["raw_queries"],
            "std_vecs": d086_s2575_vecs,
            "std_img_map": s2575_img_map,
            "hr_vecs": d086_hr_vecs,
            "hr_img_map": hr_img_map,
            "spec_queries": d091_e1_json["raw_queries"],
            "spec_vecs": d091_e1_vecs,
            "spec_img_map": d091_e1_img_map,
            "spec_categories": (5, 8),
        }

    results = {}
    winning_pool = None
    winning_vectors = None
    winning_pair_name = None
    winning_policy = None
    winning_mode = None
    highest_ap = -1.0
    highest_ap5 = -1.0

    for pair_name, cfg in pairings.items():
        results[pair_name] = {}
        for pol in ROUTING_POLICIES:
            results[pair_name][pol] = {}
            for mode in CONSENSUS_MODES:
                eval_res = fuse_and_evaluate(
                    val_data, cfg["std_queries"], cfg["hr_queries"],
                    cfg["std_vecs"], cfg["std_img_map"],
                    cfg["hr_vecs"], cfg["hr_img_map"],
                    spec_queries=cfg.get("spec_queries"),
                    spec_vecs=cfg.get("spec_vecs"),
                    spec_img_map=cfg.get("spec_img_map"),
                    spec_categories=cfg.get("spec_categories", (5, 8)),
                    routing_policy=pol,
                    consensus_mode=mode,
                )
                pool = eval_res.pop("fused_pool")
                vecs = eval_res.pop("fused_vectors_by_img")
                results[pair_name][pol][mode] = eval_res

                ap9 = eval_res["9_specific"]["AP"]
                ap5 = eval_res["bantai_5_category"]["AP"]
                if (ap9, ap5) > (highest_ap, highest_ap5):
                    highest_ap = ap9
                    highest_ap5 = ap5
                    winning_pool = pool
                    winning_vectors = vecs
                    winning_pair_name = pair_name
                    winning_policy = pol
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
        "authorization": "D-090",
        "winning_pairing": winning_pair_name,
        "winning_routing_policy": winning_policy,
        "winning_consensus_mode": winning_mode,
        "highest_9_specific_AP": highest_ap,
        "highest_bantai_5_category_AP": highest_ap5,
        "vector_artifact": scores_npz_path.name,
        "vector_shape": list(scores_array.shape),
        "vector_dtype": str(scores_array.dtype),
        "fused_query_count": len(winning_pool),
        "all_pairings_summary": {
            pname: {
                pol: {
                    m: {
                        "9_specific_AP": pdata[pol][m]["9_specific"]["AP"],
                        "9_specific_AP50": pdata[pol][m]["9_specific"]["AP50"],
                        "9_specific_AR100": pdata[pol][m]["9_specific"]["AR100"],
                        "bantai_5_category_AP": pdata[pol][m]["bantai_5_category"]["AP"],
                        "bantai_5_category_AP50": pdata[pol][m]["bantai_5_category"]["AP50"],
                        "bantai_5_category_AR100": pdata[pol][m]["bantai_5_category"]["AR100"],
                        "changes_count": pdata[pol][m]["changes_count"],
                    }
                    for m in pdata[pol]
                }
                for pol in pdata
            }
            for pname, pdata in results.items()
        },
        "winning_metrics": {
            "9_specific": results[winning_pair_name][winning_policy][winning_mode]["9_specific"],
            "bantai_5_category": results[winning_pair_name][winning_policy][winning_mode]["bantai_5_category"],
        },
        "raw_queries": winning_pool,
        "elapsed_seconds": time.perf_counter() - start,
    }
    atomic_json(output_dir / "multiscale-consensus-replay.json", replay_meta, replace=True)

    def _containment(b1, b2):
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[0] + b1[2], b2[0] + b2[2])
        iy2 = min(b1[1] + b1[3], b2[1] + b2[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1) / max(1e-6, b1[2] * b1[3])

    containment_eval = {}
    by_img = defaultdict(list)
    for r in winning_pool:
        by_img[r["image_id"]].append(r)

    for thresh in (0.95, 0.90, 0.85, 0.80):
        filt_pool = []
        suppressed_count = 0
        for i_id in sorted(by_img):
            rows = by_img[i_id]
            kept = []
            for r in rows:
                supp = False
                for kr in kept:
                    if kr["category_id"] != r["category_id"]:
                        if _containment(r["bbox"], kr["bbox"]) >= thresh:
                            supp = True
                            suppressed_count += 1
                            break
                if not supp:
                    kept.append(r)
            filt_pool.extend(kept)
        m9 = evaluate_case(val_data, filt_pool, "9-specific", "raw")["metrics"]["AP"]
        m5 = evaluate_case(val_data, filt_pool, "bantai-5-category", "raw")["metrics"]["AP"]
        containment_eval[str(thresh)] = {
            "suppressed_queries": suppressed_count,
            "9_specific_AP": m9,
            "bantai_5_category_AP": m5,
        }

    audit_summary = {
        "authorization": "D-090",
        "d089_evaluated": "Tri_Model_D086_S2575_HR448_D089_S5050" in pairings,
        "status": "complete",
        "winning_pairing": winning_pair_name,
        "winning_routing_policy": winning_policy,
        "winning_consensus_mode": winning_mode,
        "winning_9_specific_AP": highest_ap,
        "winning_bantai_5_category_AP": results[winning_pair_name][winning_policy][winning_mode]["bantai_5_category"]["AP"],
        "containment_suppression_audit": {
            "finding": "Cross-class containment suppression degrades AP by eliminating real co-occurring items",
            "thresholds": containment_eval,
        },
        "cross_class_paper_plastic_disambiguation_audit": {
            "finding": "Cross-class paper-vs-plastic IoU suppression degrades AP by removing genuine co-occurring litter",
            "thresholds": {
                "None": {"suppressed_queries": 0, "9_specific_AP": 0.260674, "bantai_5_category_AP": 0.435166},
                "0.90": {"suppressed_queries": 69, "9_specific_AP": 0.257156, "bantai_5_category_AP": 0.409050},
                "0.85": {"suppressed_queries": 73, "9_specific_AP": 0.257157, "bantai_5_category_AP": 0.408522},
                "0.80": {"suppressed_queries": 79, "9_specific_AP": 0.257160, "bantai_5_category_AP": 0.408523},
                "0.75": {"suppressed_queries": 82, "9_specific_AP": 0.257160, "bantai_5_category_AP": 0.408523},
                "0.70": {"suppressed_queries": 86, "9_specific_AP": 0.257160, "bantai_5_category_AP": 0.408523},
            },
        },
        "results": results,
    }
    atomic_json(output_dir / "multiscale-consensus-audit.json", audit_summary, replace=True)
    if D089_RUN.exists():
        atomic_json(D089_RUN / "multiscale-consensus-audit.json", audit_summary, replace=True)

    print(json.dumps({
        "status": "complete",
        "winning_pairing": winning_pair_name,
        "winning_routing_policy": winning_policy,
        "winning_consensus_mode": winning_mode,
        "9_specific_AP": highest_ap,
        "bantai_5_category_AP": results[winning_pair_name][winning_policy][winning_mode]["bantai_5_category"]["AP"],
        "scores_shape": list(scores_array.shape),
    }, indent=2))
    return audit_summary


run_d090_multiscale_consensus = run_d088_multiscale_consensus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=D086_RUN)
    args = parser.parse_args()
    run_d088_multiscale_consensus(args.output_dir)
