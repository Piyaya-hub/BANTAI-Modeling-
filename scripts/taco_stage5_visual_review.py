#!/usr/bin/env python3
"""
Stage 5 Visual Review: Interactive Tk/TkAgg local viewer for 9-class waste detections.

Replays verified predictions from Stage 5 run across the 65 validation images.
Compares:
  - Ground Truth
  - Baseline (Warm-start)
  - Epoch 1
  - Epoch 2
Zero model execution or GPU memory needed during review.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.configuration import strict_load
from scripts.tk_runtime import configure_tcl_tk_environment
from scripts.taco_training_labels_v2 import LABELS
from scripts.taco_stage5_material_hierarchy import (VIEWS, VIEW_SCHEMES, NAMES, COLORS, map_row,
                                                    prepare_rows, scheme_names)
from scripts.taco_stage5_display_profiles import (NMS_CHOICES, THRESHOLD_CHOICES,
                                                   visible_rows)
from scripts.taco_stage5_consensus_tiebreak import apply_to_query_rows, apply_consensus_tiebreak
import numpy as np

# 9 distinct high-contrast colors for waste categories
CLASS_COLORS = [
    "#e6194B",  # 0: cigarette (crimson red)
    "#3cb44b",  # 1: plastic_bottle (green)
    "#ffe119",  # 2: plastic_bag (yellow)
    "#4363d8",  # 3: plastic_wrapper (blue)
    "#f58231",  # 4: plastic_film (orange)
    "#911eb4",  # 5: drink_can (purple)
    "#42d4f4",  # 6: plastic_cup (cyan)
    "#f032e6",  # 7: styrofoam_piece (magenta)
    "#808000",  # 8: paper (olive gold / amber)
]


DEFAULT_TIER1_RUN = Path("runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337")
DEFAULT_SCORE_VECTOR_DIR = Path("runs/training/stage5-score-vectors-5619e219fa264287aa0564c4565a2472")
CONSENSUS_STATE = "Tier 2 + D074 champion soup"
MULTISCALE_STATE = "Multi-Scale Fused (384/576 + 448/672)"
D091_CHAMPION_STATE = "D-091 Champion (soup 50/50)"
D091_CONSENSUS_STATE = "D-091 Champion + D-079 consensus"
D093_CHAMPION_STATE = "D-093 Champion (soup 50/50)"
D093_CONSENSUS_STATE = "D-093 Champion + D-079 consensus"
PREDICTION_SOURCES = ["single_scale_champion", "multiscale_384_448"]
ORGANIC_SHOWCASE_FILTER = "Organic Showcase (4 Images: 68, 72, 92, 841)"
ORGANIC_9_STATE = "Organic Showcase 9-Prompt (384/576)"
ORGANIC_10_STATE = "Organic Showcase 10-Prompt (Hybrid)"


def discover_tier1_run(explicit=None, root=ROOT):
    """Explicit paths win; absent/incomplete private defaults fall back to old review."""
    if explicit is not None:
        return Path(explicit)
    candidate=Path(root)/DEFAULT_TIER1_RUN
    required=('report.json','validation-epoch-1.json','soup_tier1_hr_blend2575.json','soup_tier1_hr_5050.json')
    return candidate if all((candidate/name).is_file() for name in required) else None


def discover_tier2_run(explicit=None, root=ROOT):
    if explicit is not None:return Path(explicit)
    required=('report.json','validation-epoch-1.json','soup_tier2_hr_blend2575.json','soup_tier2_d074_champion_5050.json')
    candidates=[p for p in (Path(root)/'runs/training').glob('stage5-tier2-*')
                if all((p/name).is_file() for name in required)]
    return max(candidates,key=lambda p:(p/'report.json').stat().st_mtime) if candidates else None


def discover_d083_run(explicit=None, root=ROOT):
    """Discover only a completed local D-083 run with all three replay states."""
    if explicit is not None:
        return Path(explicit)
    candidates = []
    for candidate in (Path(root) / 'runs/training').glob('stage5-d083-core-*'):
        report_path = candidate / 'report.json'
        if not report_path.is_file():
            continue
        try:
            report = strict_load(report_path)
            if (report.get('authorization') == 'D-083' and report.get('status') == 'complete'
                    and report.get('training_updates') == 379 and report.get('validation_forwards') == 336
                    and (candidate / 'checkpoints/best-inference/soup_manifest.json').is_file()):
                candidates.append(candidate)
        except (OSError, ValueError, KeyError):
            continue
    return max(candidates, key=lambda p: (p / 'report.json').stat().st_mtime) if candidates else None


def discover_d086_run(explicit=None, root=ROOT):
    """Discover only a completed local D-086 run with multi-scale artifacts."""
    if explicit is not None:
        return Path(explicit)
    candidates = []
    for candidate in (Path(root) / 'runs/training').glob('stage5-d086-highres-*'):
        report_path = candidate / 'report.json'
        if not report_path.is_file():
            continue
        try:
            report = strict_load(report_path)
            if (report.get('authorization') == 'D-086' and report.get('status') == 'complete'
                    and (candidate / 'multiscale-consensus-replay.json').is_file()
                    and (candidate / 'scores-multiscale-consensus.npz').is_file()):
                candidates.append(candidate)
        except (OSError, ValueError, KeyError):
            continue
    return max(candidates, key=lambda p: (p / 'report.json').stat().st_mtime) if candidates else None


def discover_d091_run(explicit=None, root=ROOT):
    """Discover only a completed local D-091 run with balanced Tier 3 core artifacts."""
    if explicit is not None:
        return Path(explicit)
    candidates = []
    for candidate in (Path(root) / 'runs/training').glob('stage5-d091-balanced-tier3-*'):
        report_path = candidate / 'report.json'
        if not report_path.is_file():
            continue
        try:
            report = strict_load(report_path)
            if (report.get('authorization') == 'D-091' and report.get('status') == 'complete'
                    and report.get('training_updates') == 576
                    and (candidate / 'soup_d091_d086s2575_5050.json').is_file()
                    and (candidate / 'best-score-vectors.npz').is_file()):
                candidates.append(candidate)
        except (OSError, ValueError, KeyError):
            continue
    return max(candidates, key=lambda p: (p / 'report.json').stat().st_mtime) if candidates else None


def discover_d093_run(explicit=None, root=ROOT):
    """Discover only a completed local D-093 run with hard-negative multi-epoch artifacts."""
    if explicit is not None:
        return Path(explicit)
    candidates = []
    for candidate in (Path(root) / 'runs/training').glob('stage5-d093-multiepoch-hn-*'):
        report_path = candidate / 'report.json'
        if not report_path.is_file():
            continue
        try:
            report = strict_load(report_path)
            if (report.get('authorization') == 'D-093' and report.get('status') == 'complete'
                    and report.get('training_updates') == 1728
                    and (candidate / 'soup_d093best_d091_5050.json').is_file()
                    and (candidate / 'best-score-vectors.npz').is_file()):
                candidates.append(candidate)
        except (OSError, ValueError, KeyError):
            continue
    return max(candidates, key=lambda p: (p / 'report.json').stat().st_mtime) if candidates else None


def load_consensus_score_vectors(tier2_run_dir, images, root=ROOT):
    """Load only the verified D078 validation vectors matching the saved state."""
    if tier2_run_dir is None:
        return None
    import hashlib
    import numpy as np
    vector_dir = Path(root) / DEFAULT_SCORE_VECTOR_DIR
    manifest_path = vector_dir / "manifest.json"
    vector_path = vector_dir / "query-scores.npz"
    saved_path = Path(tier2_run_dir) / "soup_tier2_d074_champion_5050.json"
    if not all(path.is_file() for path in (manifest_path, vector_path, saved_path)):
        return None
    manifest = strict_load(manifest_path)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if (manifest.get("state") != "soup_tier2_d074_champion_5050"
            or manifest.get("shape") != [65, 900, 9]
            or manifest.get("parameter_sha256") != "cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422"
            or manifest.get("artifact") != vector_path.name
            or manifest.get("artifact_sha256") != sha(vector_path)
            or manifest.get("saved_evaluation_sha256") != sha(saved_path)):
        return None
    with np.load(vector_path, allow_pickle=False) as archive:
        image_ids = archive["image_ids"].tolist()
        scores = archive["scores"].copy()
    if (scores.shape != (65, 900, 9) or scores.dtype != np.float32
            or set(image_ids) != set(images)):
        return None
    return {"state": CONSENSUS_STATE,
            "scores_by_image": {image_id: scores[index]
                                for index, image_id in enumerate(image_ids)},
            "parameter_sha256": manifest["parameter_sha256"],
            "artifact_sha256": manifest["artifact_sha256"]}


def load_organic_showcase(root=ROOT):
    """Load a complete, hash-bound local D081 replay artifact when available."""
    import hashlib
    runs = Path(root) / "runs/training"
    candidates = sorted(runs.glob("stage5-organic-hybrid-*"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for run in candidates:
        payload_path, manifest_path = run / "gui-showcase.json", run / "manifest.json"
        if not payload_path.is_file() or not manifest_path.is_file():
            continue
        manifest = strict_load(manifest_path)
        digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        payload = strict_load(payload_path)
        if (manifest.get("authorization") != "D-081" or manifest.get("status") != "complete"
                or manifest.get("gui_showcase_sha256") != digest
                or manifest.get("checkpoint_parameter_sha256") != "cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422"
                or payload.get("image_ids") != [68, 72, 92, 841]
                or set(map(int, payload.get("images", {}).keys())) != {68, 72, 92, 841}
                or set(payload.get("predictions", {})) != {ORGANIC_9_STATE, ORGANIC_10_STATE}):
            continue
        payload["run_dir"] = run
        payload["manifest"] = manifest
        return payload
    return None


def apply_containment_nms(objects, iou_thresh: float = 0.50, iomin_thresh: float = 0.80):
    """
    Suppresses duplicate bounding boxes using IoU and Containment (IoMin).
    If a smaller box is >=80% contained in a larger box, or if IoU >= 50%, suppress it.
    """
    if not objects:
        return []
    sorted_objs = sorted(objects, key=lambda x: -x.get("score", 1.0))
    kept = []
    for obj in sorted_objs:
        bx, by, bw, bh = obj["bbox"]
        area = bw * bh
        if area <= 0:
            continue
        suppressed = False
        for k_obj in kept:
            if obj["category_id"] != k_obj["category_id"]:
                continue
            kx, ky, kw, kh = k_obj["bbox"]
            k_area = kw * kh
            if k_area <= 0:
                continue
            ix1 = max(bx, kx)
            iy1 = max(by, ky)
            ix2 = min(bx + bw, kx + kw)
            iy2 = min(by + bh, ky + kh)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            if inter > 0:
                union = area + k_area - inter
                iou = inter / union if union > 0 else 0
                iomin = inter / min(area, k_area) if min(area, k_area) > 0 else 0
                if iou >= iou_thresh or iomin >= iomin_thresh:
                    suppressed = True
                    break
        if not suppressed:
            kept.append(obj)
    return kept


def load_review_data(run_dir: Path, dataset_dir: Path, highres_run_dir: Path | None = None, tier1_run_dir: Path | None = None, tier2_run_dir: Path | None = None, d083_run_dir: Path | None = None, d086_run_dir: Path | None = None, d091_run_dir: Path | None = None, d093_run_dir: Path | None = None):
    val_json_path = dataset_dir / "validation.json"
    val_data = strict_load(val_json_path)

    images = {img["id"]: img for img in val_data["images"]}
    annotations_by_img = defaultdict(list)
    for ann in val_data["annotations"]:
        annotations_by_img[ann["image_id"]].append(ann)

    # Load prediction files from base run
    baseline_path = run_dir / "baseline.json"
    ep1_path = run_dir / "validation-epoch-1.json"
    ep2_path = run_dir / "validation-epoch-2.json"

    baseline_data = strict_load(baseline_path) if baseline_path.is_file() else {}
    ep1_data = strict_load(ep1_path) if ep1_path.is_file() else {}
    ep2_data = strict_load(ep2_path) if ep2_path.is_file() else {}

    # Check for high-res run
    highres_ep1_data = {}
    if highres_run_dir and (highres_run_dir / "validation-epoch-1.json").is_file():
        highres_ep1_data = strict_load(highres_run_dir / "validation-epoch-1.json")
    else:
        # Default high-res run location if available
        def_highres = ROOT / "runs/training/stage5-paper-558bcc18d6cf4556a966d8cfc320c968"
        if (def_highres / "validation-epoch-1.json").is_file():
            highres_ep1_data = strict_load(def_highres / "validation-epoch-1.json")

    preds_by_state = {
        "Baseline": defaultdict(list),
        "Stage 5 (256/384)": defaultdict(list),
        "Stage 5 High-Res (384/576)": defaultdict(list),
    }

    for p in baseline_data["raw_queries"]:
        preds_by_state["Baseline"][p["image_id"]].append(p)
    for p in ep2_data["raw_queries"]:
        preds_by_state["Stage 5 (256/384)"][p["image_id"]].append(p)
    for p in highres_ep1_data["raw_queries"]:
        preds_by_state["Stage 5 High-Res (384/576)"][p["image_id"]].append(p)

    if set(images) != {p["image_id"] for p in highres_ep1_data["raw_queries"]}:
        raise ValueError("High-resolution saved validation predictions are incomplete")

    if tier1_run_dir is not None:
        report = strict_load(tier1_run_dir / "report.json")
        if not report.get("completed") or report.get("training_updates") != 336:
            raise ValueError("Tier1 replay requires a completed bounded run")
        for label, filename in (
            ("Tier 1 High-Res (384/576)", "validation-epoch-1.json"),
            ("Tier 1 + blend2575 soup", "soup_tier1_hr_blend2575.json"),
            ("Tier 1 + HR E1 soup", "soup_tier1_hr_5050.json"),
        ):
            saved = strict_load(tier1_run_dir / filename)
            rows = saved['raw_queries']
            if len(rows) != 65*900 or {r['image_id'] for r in rows} != set(images):
                raise ValueError("Tier1 saved queries differ from frozen validation")
            preds_by_state[label] = defaultdict(list)
            for row in rows: preds_by_state[label][row['image_id']].append(row)

    if tier2_run_dir is not None:
        report=strict_load(tier2_run_dir/'report.json')
        if not report.get('completed') or report.get('training_updates')!=384 or report.get('validation_forwards')!=336:
            raise ValueError('Tier2 replay requires completed D076 bounds')
        for label,filename in (
            ('Tier 2 E1 (384/576)','validation-epoch-1.json'),
            ('Tier 2 + blend2575 soup','soup_tier2_hr_blend2575.json'),
            ('Tier 2 + D074 champion soup','soup_tier2_d074_champion_5050.json')):
            rows=strict_load(tier2_run_dir/filename)['raw_queries']
            if len(rows)!=65*900 or {r['image_id'] for r in rows}!=set(images):
                raise ValueError('Tier2 saved queries differ from frozen validation')
            preds_by_state[label]=defaultdict(list)
            for row in rows:preds_by_state[label][row['image_id']].append(row)

    if d083_run_dir is not None:
        d083_report = strict_load(d083_run_dir / 'report.json')
        if (d083_report.get('authorization') != 'D-083' or d083_report.get('status') != 'complete'
                or d083_report.get('training_updates') != 379
                or d083_report.get('validation_forwards') != 336):
            raise ValueError('D-083 replay requires completed bounded training/evaluation evidence')
        state_files = (
            ('D083-Core E1 (384/576)', 'validation-d083-core-e1.json'),
            ('D083 + D074 soup (50/50)', 'soup_d083_d074_5050.json'),
            ('D083 + D076 soup (50/50)', 'soup_d083_d076_5050.json'),
        )
        for label, filename in state_files:
            saved = strict_load(d083_run_dir / filename)
            rows = saved.get('raw_queries', [])
            if len(rows) != 65 * 900 or {r['image_id'] for r in rows} != set(images):
                raise ValueError('D-083 saved replay does not contain exactly 65 x 900 queries')
            preds_by_state[label] = defaultdict(list)
            for row in rows:
                preds_by_state[label][row['image_id']].append(row)

    multiscale_preds = None
    multiscale_vectors = None
    if d086_run_dir is not None:
        ms_replay_file = d086_run_dir / "multiscale-consensus-replay.json"
        ms_scores_file = d086_run_dir / "scores-multiscale-consensus.npz"
        if ms_replay_file.is_file() and ms_scores_file.is_file():
            ms_data = strict_load(ms_replay_file)
            ms_rows = ms_data.get("raw_queries", [])
            preds_by_state[MULTISCALE_STATE] = defaultdict(list)
            multiscale_preds = defaultdict(list)
            for row in ms_rows:
                preds_by_state[MULTISCALE_STATE][row["image_id"]].append(row)
                multiscale_preds[row["image_id"]].append(row)
            with np.load(ms_scores_file, allow_pickle=False) as ms_npz:
                ms_img_ids = ms_npz["image_ids"].tolist()
                ms_scores = ms_npz["scores"].copy()
            multiscale_vectors = {
                "scores_by_image": {img_id: ms_scores[idx] for idx, img_id in enumerate(ms_img_ids)},
                "shape": list(ms_scores.shape),
                "dtype": str(ms_scores.dtype),
            }

    d091_vectors = None
    if d091_run_dir is not None:
        d091_report = strict_load(d091_run_dir / "report.json")
        if (d091_report.get("authorization") != "D-091" or d091_report.get("status") != "complete"
                or d091_report.get("training_updates") != 576):
            raise ValueError("D-091 replay requires completed bounded training evidence")

        state_files = (
            ("D-091 Balanced Tier 3 E1 (384/576)", "validation-d091-balanced-tier3-e1.json"),
            (D091_CHAMPION_STATE, "soup_d091_d086s2575_5050.json"),
            ("D-091 + D083 soup (50/50)", "soup_d091_d083_5050.json"),
            ("D-091 + D083 soup (25/75)", "soup_d091_d083_2575.json"),
        )
        for label, filename in state_files:
            file_path = d091_run_dir / filename
            if not file_path.is_file():
                continue
            saved = strict_load(file_path)
            rows = saved.get("raw_queries", [])
            if len(rows) != 65 * 900 or {r["image_id"] for r in rows} != set(images):
                raise ValueError(f"D-091 saved replay {filename} does not contain exactly 65 x 900 queries")
            preds_by_state[label] = defaultdict(list)
            for row in rows:
                preds_by_state[label][row["image_id"]].append(row)

        vec_path = d091_run_dir / "best-score-vectors.npz"
        if vec_path.is_file():
            with np.load(vec_path, allow_pickle=False) as npz:
                v_ids = npz["image_ids"].tolist()
                v_scores = npz["scores"].copy()
            if v_scores.shape == (65, 900, 9) and set(v_ids) == set(images):
                d091_vectors = {
                    "scores_by_image": {img_id: v_scores[idx] for idx, img_id in enumerate(v_ids)},
                    "shape": list(v_scores.shape),
                    "dtype": str(v_scores.dtype),
                }
                if D091_CHAMPION_STATE in preds_by_state:
                    preds_by_state[D091_CONSENSUS_STATE] = defaultdict(list)
                    for img_id, rows in preds_by_state[D091_CHAMPION_STATE].items():
                        s_vec = d091_vectors["scores_by_image"][img_id]
                        c_rows = []
                        for r in rows:
                            r_c, _ = apply_consensus_tiebreak(
                                r, s_vec[r["query_id"]], delta=0.016,
                                coactivation="3 plastic subtypes >= 0.20",
                                confidence_mode="top1_score",
                            )
                            c_rows.append(r_c)
                        preds_by_state[D091_CONSENSUS_STATE][img_id] = c_rows

    d093_vectors = None
    if d093_run_dir is not None:
        d093_report = strict_load(d093_run_dir / "report.json")
        if (d093_report.get("authorization") != "D-093" or d093_report.get("status") != "complete"
                or d093_report.get("training_updates") != 1728):
            raise ValueError("D-093 replay requires completed bounded training evidence")

        state_files = (
            ("D-093 Hard-Negative E1 (384/576)", "validation-d093-hn-e1.json"),
            ("D-093 Hard-Negative E2 (384/576)", "validation-d093-hn-e2.json"),
            ("D-093 Hard-Negative E3 (384/576)", "validation-d093-hn-e3.json"),
            (D093_CHAMPION_STATE, "soup_d093best_d091_5050.json"),
            ("D-093 + D091 soup (25/75)", "soup_d093best_d091_2575.json"),
        )
        for label, filename in state_files:
            file_path = d093_run_dir / filename
            if not file_path.is_file():
                continue
            saved = strict_load(file_path)
            rows = saved.get("raw_queries", [])
            if len(rows) != 65 * 900 or {r["image_id"] for r in rows} != set(images):
                raise ValueError(f"D-093 saved replay {filename} does not contain exactly 65 x 900 queries")
            preds_by_state[label] = defaultdict(list)
            for row in rows:
                preds_by_state[label][row["image_id"]].append(row)

        vec_path = d093_run_dir / "best-score-vectors.npz"
        if vec_path.is_file():
            with np.load(vec_path, allow_pickle=False) as npz:
                v_ids = npz["image_ids"].tolist()
                v_scores = npz["scores"].copy()
            if v_scores.shape == (65, 900, 9) and set(v_ids) == set(images):
                d093_vectors = {
                    "scores_by_image": {img_id: v_scores[idx] for idx, img_id in enumerate(v_ids)},
                    "shape": list(v_scores.shape),
                    "dtype": str(v_scores.dtype),
                }
                if D093_CHAMPION_STATE in preds_by_state:
                    preds_by_state[D093_CONSENSUS_STATE] = defaultdict(list)
                    for img_id, rows in preds_by_state[D093_CHAMPION_STATE].items():
                        s_vec = d093_vectors["scores_by_image"][img_id]
                        c_rows = []
                        for r in rows:
                            r_c, _ = apply_consensus_tiebreak(
                                r, s_vec[r["query_id"]], delta=0.016,
                                coactivation="3 plastic subtypes >= 0.20",
                                confidence_mode="top1_score",
                            )
                            c_rows.append(r_c)
                        preds_by_state[D093_CONSENSUS_STATE][img_id] = c_rows

    # Sort predictions by score descending
    for state in preds_by_state:
        for img_id in preds_by_state[state]:
            preds_by_state[state][img_id].sort(key=lambda x: -x["score"])

    consensus = load_consensus_score_vectors(tier2_run_dir, images, ROOT)
    organic_showcase = load_organic_showcase(ROOT)
    if organic_showcase:
        for state, by_image in organic_showcase["predictions"].items():
            preds_by_state[state] = defaultdict(list)
            for image_id, rows in by_image.items():
                preds_by_state[state][int(image_id)] = rows
    return {
        "images": images,
        "dataset_root": Path(val_data["info"]["image_root"]).resolve(),
        "ground_truth": annotations_by_img,
        "predictions": preds_by_state,
        "consensus_score_vectors": consensus,
        "d091_score_vectors": d091_vectors,
        "d093_score_vectors": d093_vectors,
        "multiscale_predictions": multiscale_preds,
        "multiscale_score_vectors": multiscale_vectors,
        "organic_showcase": organic_showcase,
        "metrics": {
            "Baseline": baseline_data.get("combined_65", {}),
            "Stage 5 (256/384)": ep2_data.get("combined_65", {}),
            "Stage 5 High-Res (384/576)": highres_ep1_data.get("combined_65", {}),
        },
    }


def draw_panel(ax, image, objects, is_gt: bool = False, score_thresh: float = 0.10, top_k: int = 15):
    from matplotlib.patches import Rectangle

    w, h = image.size
    ax.clear()
    ax.imshow(image)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    rendered = 0
    for obj in objects:
        if not is_gt:
            if obj.get("score", 1.0) < score_thresh:
                continue
            if rendered >= top_k:
                break

        cid = obj["category_id"]
        color = CLASS_COLORS[cid] if 0 <= cid < len(CLASS_COLORS) else "#ffffff"
        label_name = obj.get("display_label", LABELS[cid] if 0 <= cid < len(LABELS) else f"cls_{cid}")
        color = COLORS.get(obj.get("mapped_category"), obj.get("bantai_color", color))

        bx, by, bw, bh = obj["bbox"]
        linestyle = "-" if is_gt else "--"
        linewidth = 1.8 if is_gt else 1.5

        rect = Rectangle(
            (bx, by), bw, bh,
            fill=False, edgecolor=color,
            linestyle=linestyle, linewidth=linewidth,
            zorder=3,
        )
        ax.add_patch(rect)

        # Label badge
        score_str = f" {obj['score']:.2f}" if not is_gt else ""
        text = f"{label_name}{score_str}"
        ax.text(
            bx, max(0, by - 4), text,
            fontsize=7, weight="bold", color="white",
            bbox=dict(boxstyle="square,pad=0.2", facecolor=color, edgecolor="none", alpha=0.85),
            zorder=4,
        )
        rendered += 1


class Stage5ViewerGUI:
    def __init__(self, data: dict, initial_image_id: int | None = None, default_threshold: float = 0.10, default_top_k: int = 15, prediction_source: str = PREDICTION_SOURCES[0]):
        configure_tcl_tk_environment()
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        self.data = data
        self.images = data["images"]
        self.gt = data["ground_truth"]
        self.preds = data["predictions"]
        self.consensus_vectors = data.get("consensus_score_vectors")
        self.multiscale_preds = data.get("multiscale_predictions")
        self.multiscale_vectors = data.get("multiscale_score_vectors")
        self.d091_vectors = data.get("d091_score_vectors")
        self.d093_vectors = data.get("d093_score_vectors")
        self.organic_showcase = data.get("organic_showcase")
        self.dataset_root = data["dataset_root"]
        self.prediction_source = prediction_source

        # Identify which images have paper
        self.paper_img_ids = set()
        for img_id, anns in self.gt.items():
            if any(a["category_id"] == 8 for a in anns):
                self.paper_img_ids.add(img_id)

        self.root = tk.Tk()
        self.root.title("BANTAI — Stage 5 9-Class Waste Detection Visual Review")
        self.root.geometry("1480x920")

        # Top Control Bar
        ctrl_frame = ttk.Frame(self.root, padding=6)
        ctrl_frame.pack(side="top", fill="x")

        # Filter mode
        ttk.Label(ctrl_frame, text="Filter:").pack(side="left", padx=(4, 2))
        self.filter_var = tk.StringVar(value="Paper Images (18)" if self.paper_img_ids else "All Images (65)")
        filter_values = ["Paper Images (18)", "All Images (65)"]
        if self.organic_showcase:
            filter_values.append(ORGANIC_SHOWCASE_FILTER)
        filter_cb = ttk.Combobox(ctrl_frame, textvariable=self.filter_var, values=filter_values, state="readonly", width=35)
        filter_cb.pack(side="left", padx=4)
        filter_cb.bind("<<ComboboxSelected>>", self.on_filter_change)

        # Image Dropdown
        ttk.Label(ctrl_frame, text="Image:").pack(side="left", padx=(10, 2))
        self.img_var = tk.StringVar()
        self.img_cb = ttk.Combobox(ctrl_frame, textvariable=self.img_var, state="readonly", width=26)
        self.img_cb.pack(side="left", padx=4)
        self.img_cb.bind("<<ComboboxSelected>>", self.on_image_change)

        # Prev / Next Buttons
        ttk.Button(ctrl_frame, text="◀ Prev", width=7, command=self.prev_image).pack(side="left", padx=2)
        ttk.Button(ctrl_frame, text="Next ▶", width=7, command=self.next_image).pack(side="left", padx=2)

        # Threshold Control
        ttk.Label(ctrl_frame, text="Score Threshold:").pack(side="left", padx=(14, 2))
        self.thresh_var = tk.DoubleVar(value=default_threshold)
        thresh_spin = ttk.Spinbox(ctrl_frame, from_=0.01, to=0.99, increment=0.02, textvariable=self.thresh_var, width=6, command=self.update_plots)
        thresh_spin.pack(side="left", padx=2)
        thresh_spin.bind("<Return>", lambda e: self.update_plots())

        # Top-K Control
        ttk.Label(ctrl_frame, text="Top-K:").pack(side="left", padx=(10, 2))
        self.topk_var = tk.IntVar(value=default_top_k)
        topk_spin = ttk.Spinbox(ctrl_frame, from_=1, to=100, increment=5, textvariable=self.topk_var, width=5, command=self.update_plots)
        topk_spin.pack(side="left", padx=2)
        topk_spin.bind("<Return>", lambda e: self.update_plots())

        self.nms_var = tk.StringVar(value='agnostic_iou_0.50')
        nms_cb = ttk.Combobox(ctrl_frame, textvariable=self.nms_var,
            values=NMS_CHOICES, state="readonly", width=34)
        nms_cb.pack(side="left", padx=4)
        nms_cb.bind("<<ComboboxSelected>>", lambda e: self.update_plots())

        ttk.Label(ctrl_frame, text="Taxonomy View:").pack(side="left", padx=4)
        self.taxonomy_var = tk.StringVar(value=VIEWS[0])
        taxonomy_cb = ttk.Combobox(ctrl_frame, textvariable=self.taxonomy_var,
                                  values=VIEWS, state="readonly", width=38)
        taxonomy_cb.pack(side="left")
        taxonomy_cb.bind("<<ComboboxSelected>>", lambda e: self.update_plots())
        state_frame = ttk.Frame(self.root)
        state_frame.pack(side="top", fill="x")
        ttk.Label(state_frame, text="Replay Source:").pack(side="left", padx=(10, 2))
        self.pred_source_var = tk.StringVar(value=prediction_source if prediction_source in PREDICTION_SOURCES else PREDICTION_SOURCES[0])
        source_cb = ttk.Combobox(state_frame, textvariable=self.pred_source_var,
                                 values=PREDICTION_SOURCES, state="readonly", width=22)
        source_cb.pack(side="left", padx=2)
        source_cb.bind("<<ComboboxSelected>>", self.on_prediction_source_change)

        ttk.Label(state_frame, text="Fourth panel state:").pack(side="left", padx=6)
        choices=[n for n in self.preds if n not in ("Baseline", "Stage 5 (256/384)")]
        if prediction_source == "multiscale_384_448" and MULTISCALE_STATE in choices:
            initial = MULTISCALE_STATE
        elif D093_CHAMPION_STATE in choices and prediction_source == "single_scale_champion":
            initial = D093_CHAMPION_STATE
        elif D091_CHAMPION_STATE in choices and prediction_source == "single_scale_champion":
            initial = D091_CHAMPION_STATE
        else:
            initial=(CONSENSUS_STATE if self.consensus_vectors and CONSENSUS_STATE in choices else
                     "Tier 1 + blend2575 soup" if "Tier 1 + blend2575 soup" in choices else choices[0])
        self.review_state_var=tk.StringVar(value=initial)
        state_cb=ttk.Combobox(state_frame, textvariable=self.review_state_var, values=choices,
                              state="readonly", width=38)
        state_cb.pack(side="left")
        state_cb.bind("<<ComboboxSelected>>", lambda e:self.update_plots())
        has_vectors = bool(self.consensus_vectors or self.multiscale_vectors or self.d091_vectors or self.d093_vectors)
        self.consensus_var = tk.BooleanVar(value=has_vectors)
        self.consensus_check = ttk.Checkbutton(
            state_frame, text="Score-preserving plastic consensus",
            variable=self.consensus_var, command=self.update_plots,
            state="normal" if has_vectors else "disabled")
        self.consensus_check.pack(side="left", padx=8)
        ttk.Label(state_frame, text="Threshold Mode:").pack(side="left", padx=(14, 2))
        self.threshold_mode_var = tk.StringVar(value=THRESHOLD_CHOICES[0])
        threshold_cb = ttk.Combobox(state_frame, textvariable=self.threshold_mode_var,
                                    values=THRESHOLD_CHOICES, state="readonly", width=43)
        threshold_cb.pack(side="left", padx=4)
        threshold_cb.bind("<<ComboboxSelected>>", lambda e: self.update_plots())
        self.legend_frame = ttk.Frame(self.root)
        self.legend_frame.pack(side="top", fill="x")

        # Main Paned Window (Plots on top, Details table on bottom, exactly like Stage 4)
        pane = ttk.Panedwindow(self.root, orient="vertical")
        pane.pack(fill="both", expand=True)

        plot_frame = ttk.Frame(pane)
        details_frame = ttk.Frame(pane)
        pane.add(plot_frame, weight=3)
        pane.add(details_frame, weight=1)

        # Matplotlib Figure (2x2 Grid)
        self.fig = Figure(figsize=(14, 7), dpi=100)
        self.axes = {
            "GT": self.fig.add_subplot(2, 2, 1),
            "Baseline": self.fig.add_subplot(2, 2, 2),
            "Stage 5 (256/384)": self.fig.add_subplot(2, 2, 3),
            "Stage 5 High-Res (384/576)": self.fig.add_subplot(2, 2, 4),
        }
        self.fig.tight_layout(pad=1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(side="bottom", fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Details Notebook (Tables for GT, Baseline, Stage 5 Run 1, Stage 5 High-Res)
        notebook = ttk.Notebook(details_frame)
        self.details_notebook = notebook
        notebook.pack(fill="both", expand=True)
        self.tables = {}
        columns = ("rank", "category", "specific_class", "hazard_flag", "score", "bbox_xywh")
        for tab_name in ("Ground Truth", "Baseline", "Stage 5 (256/384)", "Stage 5 High-Res (384/576)"):
            tab_f = ttk.Frame(notebook)
            notebook.add(tab_f, text=tab_name)
            tree = ttk.Treeview(tab_f, columns=columns, show="headings", height=5)
            tree.heading("rank", text="Rank")
            tree.heading("category", text="Category")
            tree.heading("specific_class", text="specific_class")
            tree.heading("hazard_flag", text="hazard_flag")
            tree.column("specific_class", width=150)
            tree.column("hazard_flag", width=80)
            tree.heading("score", text="Score")
            tree.heading("bbox_xywh", text="BBox [x, y, w, h]")
            tree.column("rank", width=60, stretch=False)
            tree.column("category", width=140, stretch=False)
            tree.column("score", width=100, stretch=False)
            tree.column("bbox_xywh", width=320, stretch=True)
            y_sb = ttk.Scrollbar(tab_f, orient="vertical", command=tree.yview)
            x_sb = ttk.Scrollbar(tab_f, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=y_sb.set, xscrollcommand=x_sb.set)
            tree.grid(row=0, column=0, sticky="nsew")
            y_sb.grid(row=0, column=1, sticky="ns")
            x_sb.grid(row=1, column=0, sticky="ew")
            tab_f.rowconfigure(0, weight=1)
            tab_f.columnconfigure(0, weight=1)
            self.tables[tab_name] = tree

        # Status Bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=4)
        status_bar.pack(side="bottom", fill="x")

        self.populate_image_list()
        if (initial_image_id is not None and self.organic_showcase
                and initial_image_id in {68, 72, 92, 841}):
            self.filter_var.set(ORGANIC_SHOWCASE_FILTER)
            if ORGANIC_10_STATE in self.preds:
                self.review_state_var.set(ORGANIC_10_STATE)
            self.populate_image_list()
            self.set_current_image(initial_image_id)
        elif initial_image_id is not None and initial_image_id in self.images:
            self.filter_var.set("All Images (65)")
            self.populate_image_list()
            self.set_current_image(initial_image_id)
        else:
            first_id = sorted(self.paper_img_ids)[0] if self.paper_img_ids else sorted(self.images.keys())[0]
            self.set_current_image(first_id)

    def on_prediction_source_change(self, event=None):
        src = self.pred_source_var.get()
        if src == "multiscale_384_448" and MULTISCALE_STATE in self.preds:
            self.review_state_var.set(MULTISCALE_STATE)
        elif src == "single_scale_champion":
            initial = (D093_CHAMPION_STATE if D093_CHAMPION_STATE in self.preds
                       else D091_CHAMPION_STATE if D091_CHAMPION_STATE in self.preds
                       else CONSENSUS_STATE if self.consensus_vectors and CONSENSUS_STATE in self.preds
                       else "D083 + D074 soup (50/50)" if "D083 + D074 soup (50/50)" in self.preds
                       else list(self.preds.keys())[0])
            self.review_state_var.set(initial)
        self.update_plots()

    def populate_image_list(self):
        f = self.filter_var.get()
        if f == ORGANIC_SHOWCASE_FILTER and self.organic_showcase:
            target_ids = [68, 72, 92, 841]
            source_images = {int(k): v for k, v in self.organic_showcase["images"].items()}
        elif "Paper" in f:
            target_ids = sorted(self.paper_img_ids)
            source_images = self.images
        else:
            target_ids = sorted(self.images.keys())
            source_images = self.images

        items = []
        for img_id in target_ids:
            img = source_images[img_id]
            tag = (" [ORGANIC]" if f == ORGANIC_SHOWCASE_FILTER else
                   " [PAPER]" if img_id in self.paper_img_ids else "")
            fname = Path(img["file_name"]).name
            items.append(f"Image {img_id}{tag} ({fname})")

        self.img_cb["values"] = items
        self.active_ids = target_ids

    def on_filter_change(self, event=None):
        if self.filter_var.get() == ORGANIC_SHOWCASE_FILTER and ORGANIC_10_STATE in self.preds:
            self.review_state_var.set(ORGANIC_10_STATE)
        elif (self.consensus_vectors and CONSENSUS_STATE in self.preds
              and self.review_state_var.get() in (ORGANIC_9_STATE, ORGANIC_10_STATE)):
            self.review_state_var.set(CONSENSUS_STATE)
        self.populate_image_list()
        if self.active_ids:
            self.set_current_image(self.active_ids[0])

    def on_image_change(self, event=None):
        selection = self.img_var.get()
        if not selection:
            return
        img_id_str = selection.split()[1]
        img_id = int(img_id_str)
        self.load_and_draw(img_id)

    def set_current_image(self, img_id: int):
        for val in self.img_cb["values"]:
            if val.startswith(f"Image {img_id} "):
                self.img_var.set(val)
                self.load_and_draw(img_id)
                break

    def prev_image(self):
        if not self.active_ids:
            return
        cur_id = self.current_image_id
        if cur_id in self.active_ids:
            idx = self.active_ids.index(cur_id)
            new_id = self.active_ids[(idx - 1) % len(self.active_ids)]
            self.set_current_image(new_id)

    def next_image(self):
        if not self.active_ids:
            return
        cur_id = self.current_image_id
        if cur_id in self.active_ids:
            idx = self.active_ids.index(cur_id)
            new_id = self.active_ids[(idx + 1) % len(self.active_ids)]
            self.set_current_image(new_id)

    def load_and_draw(self, img_id: int):
        self.current_image_id = img_id
        organic_mode = (self.organic_showcase is not None
                        and self.filter_var.get() == ORGANIC_SHOWCASE_FILTER
                        and img_id in (68, 72, 92, 841))
        img_entry = ({int(k): v for k, v in self.organic_showcase["images"].items()}[img_id]
                     if organic_mode else self.images[img_id])
        image_root = Path(self.organic_showcase["dataset_root"]).resolve() if organic_mode else self.dataset_root
        photo_path = (image_root / img_entry["file_name"]).resolve()
        if not photo_path.is_relative_to(image_root):
            raise ValueError("Validation photo path escapes dataset root")

        if not photo_path.is_file():
            self.status_var.set(f"ERROR: Photograph not found at {photo_path}")
            return

        import hashlib
        expected = (img_entry.get("source_taco", {}).get("file_sha256")
                    or img_entry.get("file_sha256"))
        if not expected:
            raise ValueError("Replay image has no bound SHA-256")
        if hashlib.sha256(photo_path.read_bytes()).hexdigest() != expected:
            raise ValueError("Validation source photo hash mismatch")
        with Image.open(photo_path) as pil_img:
            self.current_pil = pil_img.convert("RGB")

        self.update_plots()

    def update_plots(self):
        if not hasattr(self, "current_pil") or not hasattr(self, "current_image_id"):
            return

        img_id = self.current_image_id
        showcase_mode = (self.organic_showcase is not None
                         and self.filter_var.get() == ORGANIC_SHOWCASE_FILTER)
        self.details_notebook.tab(3, text=self.review_state_var.get())
        self.details_notebook.tab(1, text="9-Prompt" if showcase_mode else "Baseline")
        self.details_notebook.tab(2, text="10-Prompt Hybrid" if showcase_mode else "Stage 5 (256/384)")
        thresh = float(self.thresh_var.get())
        top_k = int(self.topk_var.get())

        scheme = VIEW_SCHEMES[self.taxonomy_var.get()]
        hierarchical = self.taxonomy_var.get() == VIEWS[3]
        import tkinter as tk
        for child in self.legend_frame.winfo_children(): child.destroy()
        gt_rows = (self.organic_showcase["ground_truth"].get(str(img_id), []) if showcase_mode
                   else self.gt.get(img_id, []))
        legend_names = list(scheme_names(scheme))
        if any(row.get("specific_category_id", row.get("category_id")) == 9 for row in gt_rows):
            organic_name = "food_waste" if scheme == "9-specific" else "Organic"
            if organic_name not in legend_names:
                legend_names.append(organic_name)
        for name in legend_names:
            tk.Label(self.legend_frame, text=name, bg=COLORS.get(name, CLASS_COLORS[LABELS.index(name)] if name in LABELS else "white"), fg="white").pack(side="left", padx=2)
        gt_boxes = [map_row(a, scheme, hierarchical) for a in gt_rows]
        base_preds = (self.preds[ORGANIC_9_STATE].get(img_id, []) if showcase_mode
                      else self.preds["Baseline"].get(img_id, []))
        run1_preds = (self.preds[ORGANIC_10_STATE].get(img_id, []) if showcase_mode
                      else self.preds["Stage 5 (256/384)"].get(img_id, []))
        highres_preds = self.preds[self.review_state_var.get()].get(img_id, [])

        is_multiscale = (self.review_state_var.get() == MULTISCALE_STATE
                         and self.multiscale_vectors is not None
                         and img_id in self.multiscale_vectors["scores_by_image"])
        is_d091_champ = (self.review_state_var.get() == D091_CHAMPION_STATE
                         and self.d091_vectors is not None
                         and img_id in self.d091_vectors["scores_by_image"])
        is_d093_champ = (self.review_state_var.get() == D093_CHAMPION_STATE
                         and self.d093_vectors is not None
                         and img_id in self.d093_vectors["scores_by_image"])

        use_consensus = (self.consensus_var.get()
                         and scheme == "bantai-5-category"
                         and self.taxonomy_var.get() != VIEWS[2]
                         and ((is_multiscale)
                              or (is_d091_champ)
                              or (is_d093_champ)
                              or (self.consensus_vectors is not None
                                  and self.review_state_var.get() == CONSENSUS_STATE
                                  and img_id in self.consensus_vectors["scores_by_image"])))

        if use_consensus:
            if is_multiscale:
                # Multi-scale replay consensus application
                s_vec = self.multiscale_vectors["scores_by_image"][img_id]
                adj_rows = []
                for r in highres_preds:
                    qid = r.get("fused_query_id", r.get("query_id", 0))
                    if 0 <= qid < len(s_vec):
                        r_adj, _ = apply_consensus_tiebreak(
                            r, s_vec[qid], delta=0.016,
                            coactivation="3 plastic subtypes >= 0.20",
                            confidence_mode="top1_score")
                        adj_rows.append(r_adj)
                    else:
                        adj_rows.append(r)
                highres_preds = adj_rows
            elif is_d093_champ:
                highres_preds, _ = apply_to_query_rows(
                    highres_preds, self.d093_vectors["scores_by_image"][img_id],
                    delta=0.016, coactivation="3 plastic subtypes >= 0.20",
                    confidence_mode="top1_score")
            elif is_d091_champ:
                highres_preds, _ = apply_to_query_rows(
                    highres_preds, self.d091_vectors["scores_by_image"][img_id],
                    delta=0.016, coactivation="3 plastic subtypes >= 0.20",
                    confidence_mode="top1_score")
            else:
                highres_preds, _ = apply_to_query_rows(
                    highres_preds, self.consensus_vectors["scores_by_image"][img_id],
                    delta=0.016, coactivation="3 plastic subtypes >= 0.20",
                    confidence_mode="top1_score")

        # Apply score threshold and optional containment NMS
        def filter_preds(plist):
            return visible_rows(plist, scheme, self.nms_var.get(), thresh,
                                self.threshold_mode_var.get(), hierarchical)[:top_k]

        base_filtered = filter_preds(base_preds)
        run1_filtered = filter_preds(run1_preds)
        highres_filtered = filter_preds(highres_preds)

        draw_panel(self.axes["GT"], self.current_pil, gt_boxes, is_gt=True)
        self.axes["GT"].set_title(f"Ground Truth ({len(gt_boxes)} objects)", fontsize=11, weight="bold")

        # Rows were already filtered by the selected uniform/calibrated profile.
        # A second uniform filter here would silently hide 0.06-0.10 cigarettes.
        draw_panel(self.axes["Baseline"], self.current_pil, base_filtered, is_gt=False, score_thresh=0.0, top_k=top_k)
        threshold_caption = (f"uniform >= {thresh:.2f}" if self.threshold_mode_var.get() == THRESHOLD_CHOICES[0]
                             else f"cigarette >= 0.06 / macro >= {max(thresh, .15):.2f}")
        self.axes["Baseline"].set_title((f"9-Prompt (384/576; {threshold_caption})" if showcase_mode
                                           else f"Warm-Start Baseline ({threshold_caption})"), fontsize=11)

        draw_panel(self.axes["Stage 5 (256/384)"], self.current_pil, run1_filtered, is_gt=False, score_thresh=0.0, top_k=top_k)
        self.axes["Stage 5 (256/384)"].set_title((f"10-Prompt Hybrid (384/576; {threshold_caption})" if showcase_mode
                                                   else f"Stage 5 Run 1 (256/384; {threshold_caption})"), fontsize=11)

        draw_panel(self.axes["Stage 5 High-Res (384/576)"], self.current_pil, highres_filtered, is_gt=False, score_thresh=0.0, top_k=top_k)
        self.axes["Stage 5 High-Res (384/576)"].set_title(self.review_state_var.get(), fontsize=11, color="#005500", weight="bold")

        self.canvas.draw_idle()

        # Update Details Tables
        for tree in self.tables.values():
            tree.delete(*tree.get_children())

        # Ground Truth table
        for r_idx, ann in enumerate(gt_boxes, 1):
            c_name = ann["display_label"]
            bx = [round(v, 1) for v in ann["bbox"]]
            self.tables["Ground Truth"].insert("", "end", values=(f"GT-{ann['id']}", c_name, ann["specific_class"], ann["hazard_flag"], "1.000", str(bx)))

        # Baseline, Stage 5 Run 1, Stage 5 High-Res tables
        for tab_name, passing in [
            ("Baseline", base_filtered),
            ("Stage 5 (256/384)", run1_filtered),
            ("Stage 5 High-Res (384/576)", highres_filtered),
        ]:
            for r_idx, p in enumerate(passing, 1):
                c_name = p["display_label"]
                bx = [round(v, 1) for v in p["bbox"]]
                self.tables[tab_name].insert("", "end", values=(f"q{p['query_id']}", c_name, p["specific_class"], p["hazard_flag"], f"{p['score']:.4f}", str(bx)))

        # Update status bar
        gt_types = [a["display_label"] for a in gt_boxes]
        hr_types = [f"{p['display_label']}({p['score']:.2f})" for p in highres_filtered]
        nms_str = " | NMS: " + self.nms_var.get() + " | " + self.threshold_mode_var.get()
        self.status_var.set(
            f"Image ID: {img_id} [{self.current_pil.size[0]}x{self.current_pil.size[1]}]{nms_str} | "
            f"GT: {len(gt_boxes)} {gt_types} | "
            f"High-Res Detections ({threshold_caption}, top-{top_k}): {len(highres_filtered)} {hr_types}"
        )

    def run(self):
        self.root.mainloop()


def print_cli_summary(data: dict, score_thresh: float, top_k: int, image_id: int | None = None):
    print("=== Stage 5 Visual Review Summary ===")
    target_ids = [image_id] if image_id else sorted(data["images"].keys())

    for img_id in target_ids:
        img = data["images"][img_id]
        gt_list = data["ground_truth"].get(img_id, [])
        gt_names = [LABELS[a["category_id"]] for a in gt_list]

        run1_preds = [p for p in data["predictions"]["Stage 5 (256/384)"].get(img_id, []) if p["score"] >= score_thresh][:top_k]
        hr_preds = [p for p in data["predictions"]["Stage 5 High-Res (384/576)"].get(img_id, []) if p["score"] >= score_thresh][:top_k]

        pred1_desc = [f"{LABELS[p['category_id']]}:{p['score']:.2f}" for p in run1_preds]
        hr_desc = [f"{LABELS[p['category_id']]}:{p['score']:.2f}" for p in hr_preds]

        paper_mark = " [PAPER]" if any(a["category_id"] == 8 for a in gt_list) else ""
        print(f"\nImage {img_id}{paper_mark} ({img['file_name']}) [{img['width']}x{img['height']}]:")
        print(f"  Ground Truth ({len(gt_list)}): {', '.join(gt_names) if gt_names else 'None'}")
        print(f"  Stage 5 Run 1  @ {score_thresh:.2f} ({len(run1_preds)}): {', '.join(pred1_desc) if pred1_desc else 'None'}")
        print(f"  Stage 5 HighRes @ {score_thresh:.2f} ({len(hr_preds)}): {', '.join(hr_desc) if hr_desc else 'None'}")
        if D091_CHAMPION_STATE in data["predictions"]:
            d091_preds = [p for p in data["predictions"][D091_CHAMPION_STATE].get(img_id, []) if p["score"] >= score_thresh][:top_k]
            d091_desc = [f"{LABELS[p['category_id']]}:{p['score']:.2f}" for p in d091_preds]
            print(f"  D-091 Champion  @ {score_thresh:.2f} ({len(d091_preds)}): {', '.join(d091_desc) if d091_desc else 'None'}")
        if D093_CHAMPION_STATE in data["predictions"]:
            d093_preds = [p for p in data["predictions"][D093_CHAMPION_STATE].get(img_id, []) if p["score"] >= score_thresh][:top_k]
            d093_desc = [f"{LABELS[p['category_id']]}:{p['score']:.2f}" for p in d093_preds]
            print(f"  D-093 Champion  @ {score_thresh:.2f} ({len(d093_preds)}): {', '.join(d093_desc) if d093_desc else 'None'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 5 Nine-Class Waste Detection Visual Review")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4",
        help="Path to Stage 5 Run 1 training directory",
    )
    parser.add_argument(
        "--highres-run-dir",
        type=Path,
        default=ROOT / "runs/training/stage5-paper-558bcc18d6cf4556a966d8cfc320c968",
        help="Path to Stage 5 High-Res training directory",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "exports/taco/strict-clean-stage5-v1",
        help="Path to Stage 5 dataset export",
    )
    parser.add_argument("--tier1-run-dir", type=Path, default=None, help="Explicit Tier1 replay directory; auto-discovers the completed local D074 run by default")
    parser.add_argument("--tier2-run-dir",type=Path,default=None,help="Explicit Tier2 run, otherwise newest complete local Tier2 run")
    parser.add_argument("--d083-run-dir", type=Path, default=None, help="Explicit D-083 replay run; otherwise discover newest completed local run")
    parser.add_argument("--d086-run-dir", type=Path, default=None, help="Explicit D-086 replay run; otherwise discover newest completed local run")
    parser.add_argument("--d091-run-dir", type=Path, default=None, help="Explicit D-091 replay run; otherwise discover newest completed local run")
    parser.add_argument("--d093-run-dir", type=Path, default=None, help="Explicit D-093 replay run; otherwise discover newest completed local run")
    parser.add_argument("--prediction-source", type=str, choices=PREDICTION_SOURCES, default=PREDICTION_SOURCES[0], help="Prediction source / replay mode")
    parser.add_argument("--gui", action="store_true", help="Launch interactive TkAgg GUI viewer")
    parser.add_argument("--image-id", type=int, default=None, help="Inspect specific Image ID")
    parser.add_argument("--score-threshold", type=float, default=0.10, help="Confidence score threshold (default: 0.10)")
    parser.add_argument("--box-top-k", type=int, default=15, help="Maximum boxes drawn per panel (default: 15)")
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        print(f"ERROR: Run directory not found: {args.run_dir}", file=sys.stderr)
        return 1

    args.tier1_run_dir = discover_tier1_run(args.tier1_run_dir)
    args.tier2_run_dir = discover_tier2_run(args.tier2_run_dir)
    args.d083_run_dir = discover_d083_run(args.d083_run_dir)
    args.d086_run_dir = discover_d086_run(args.d086_run_dir)
    args.d091_run_dir = discover_d091_run(args.d091_run_dir)
    args.d093_run_dir = discover_d093_run(args.d093_run_dir)
    if args.tier2_run_dir is not None and not args.tier2_run_dir.is_dir():
        print("ERROR: Explicit Tier2 replay directory is missing",file=sys.stderr)
        return 1
    if args.tier1_run_dir is not None and not args.tier1_run_dir.is_dir():
        print(f"ERROR: Explicit Tier1 saved replay directory is missing: {args.tier1_run_dir}", file=sys.stderr)
        return 1
    if args.d083_run_dir is not None and not args.d083_run_dir.is_dir():
        print(f"ERROR: Explicit D-083 replay directory is missing: {args.d083_run_dir}", file=sys.stderr)
        return 1
    if args.d086_run_dir is not None and not args.d086_run_dir.is_dir():
        print(f"ERROR: Explicit D-086 replay directory is missing: {args.d086_run_dir}", file=sys.stderr)
        return 1
    if args.d091_run_dir is not None and not args.d091_run_dir.is_dir():
        print(f"ERROR: Explicit D-091 replay directory is missing: {args.d091_run_dir}", file=sys.stderr)
        return 1
    if args.d093_run_dir is not None and not args.d093_run_dir.is_dir():
        print(f"ERROR: Explicit D-093 replay directory is missing: {args.d093_run_dir}", file=sys.stderr)
        return 1
    if not (args.dataset_dir / "validation.json").is_file():
        print("ERROR: Saved review needs the private validation export; no model will be run.", file=sys.stderr)
        return 1
    try:
        data = load_review_data(args.run_dir, args.dataset_dir, args.highres_run_dir, args.tier1_run_dir, args.tier2_run_dir, args.d083_run_dir, args.d086_run_dir, args.d091_run_dir, args.d093_run_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: Saved replay artifacts are unavailable/incomplete ({type(exc).__name__}); supply local run/export paths. No model will be run.", file=sys.stderr)
        return 1

    if args.gui:
        app = Stage5ViewerGUI(
            data,
            initial_image_id=args.image_id,
            default_threshold=args.score_threshold,
            default_top_k=args.box_top_k,
            prediction_source=args.prediction_source,
        )
        app.run()
    else:
        print_cli_summary(data, args.score_threshold, args.box_top_k, args.image_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
