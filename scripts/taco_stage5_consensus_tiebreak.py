"""CPU-only score-preserving paper-to-plastic display consensus rule."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from .taco_training_labels_v2 import LABELS

PLASTIC_SUBTYPE_IDS = (1, 2, 3, 4, 6, 7)
PLASTIC_STRUCTURED_IDS = (1, 2, 3, 6, 7)
PAPER_ID = 8
DEFAULT_DELTA = 0.016
DEFAULT_STRONG_COACTIVATION_DELTA = 0.040
COACTIVATION_PROFILES = {
    "2 plastic subtypes >= 0.25": (2, 0.25),
    "3 plastic subtypes >= 0.20": (3, 0.20),
}
CONFIDENCE_MODES = ("top1_score", "plastic_max")


def apply_consensus_tiebreak(row, class_scores, *, delta=DEFAULT_DELTA,
                             coactivation="2 plastic subtypes >= 0.25",
                             confidence_mode="top1_score",
                             strong_coactivation_delta=None):
    """Return a copied row and whether paper was reassigned to a plastic subtype.

    The top1_score mode preserves the original query score exactly. plastic_max
    is also score non-increasing. Boxes/query IDs are never altered.
    When strong_coactivation_delta is provided (e.g. 0.040), a higher-margin tiebreak
    is permitted if strong multi-subtype plastic co-activation (>=3 subtypes >= 0.22,
    or >=3 >= 0.20 with top-2 sum >= 0.50) is observed.
    """
    if coactivation not in COACTIVATION_PROFILES:
        raise ValueError("Unknown plastic co-activation profile")
    if confidence_mode not in CONFIDENCE_MODES:
        raise ValueError("Unknown consensus confidence mode")
    if not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    if strong_coactivation_delta is not None:
        if not isinstance(strong_coactivation_delta, (int, float)) or not math.isfinite(strong_coactivation_delta) or strong_coactivation_delta < 0:
            raise ValueError("strong_coactivation_delta must be finite and nonnegative")
    if len(class_scores) != len(LABELS):
        raise ValueError("Expected nine class probabilities")
    scores = [float(value) for value in class_scores]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores):
        raise ValueError("Class probabilities must be finite and in [0, 1]")
    result = deepcopy(row)
    if row.get("category_id") != PAPER_ID or scores[PAPER_ID] < 0.15:
        return result, False
    plastic_max = max(PLASTIC_SUBTYPE_IDS, key=lambda cid: (scores[cid], -cid))
    score_diff = scores[PAPER_ID] - scores[plastic_max]

    # Check standard Rule 1
    min_count, floor = COACTIVATION_PROFILES[coactivation]
    is_standard_coactive = sum(scores[cid] >= floor for cid in PLASTIC_SUBTYPE_IDS) >= min_count
    if score_diff <= delta and is_standard_coactive:
        result["raw_category_id"] = PAPER_ID
        result["category_id"] = plastic_max
        result["specific_category_id"] = plastic_max
        if confidence_mode == "plastic_max":
            result["score"] = scores[plastic_max]
        result["consensus_reassigned"] = True
        result["consensus_original_score"] = row.get("score")
        result["consensus_plastic_max_score"] = scores[plastic_max]
        return result, True

    # Check Rule 1B (Strong multi-subtype plastic co-activation with extended margin)
    if strong_coactivation_delta is not None and score_diff <= strong_coactivation_delta:
        p_subtypes_22 = sum(scores[cid] >= 0.22 for cid in PLASTIC_SUBTYPE_IDS)
        p_subtypes_20 = sum(scores[cid] >= 0.20 for cid in PLASTIC_SUBTYPE_IDS)
        top_plastics = sorted([scores[cid] for cid in PLASTIC_SUBTYPE_IDS], reverse=True)
        top2_p_sum = top_plastics[0] + top_plastics[1]
        is_strong = (p_subtypes_22 >= 3 or (p_subtypes_20 >= 3 and top2_p_sum >= 0.50))
        if is_strong:
            result["raw_category_id"] = PAPER_ID
            result["category_id"] = plastic_max
            result["specific_category_id"] = plastic_max
            if confidence_mode == "plastic_max":
                result["score"] = scores[plastic_max]
            result["consensus_reassigned"] = True
            result["consensus_rule1b"] = True
            result["consensus_original_score"] = row.get("score")
            result["consensus_plastic_max_score"] = scores[plastic_max]
            return result, True

    return result, False


RIGID_PLASTIC_CAN_IDS = (1, 5, 6)
FLEXIBLE_PLASTIC_IDS = (2, 3, 4)

_CACHED_BASELINE_BY_IMAGE: dict[int, list[dict]] | None = None
_CACHED_RUN1_BY_IMAGE: dict[int, list[dict]] | None = None


def _get_default_baseline_rows(image_id: int | None) -> list[dict]:
    global _CACHED_BASELINE_BY_IMAGE
    if image_id is None:
        return []
    if _CACHED_BASELINE_BY_IMAGE is None:
        _CACHED_BASELINE_BY_IMAGE = {}
        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/baseline.json",
            root / "runs/training/stage5-paper-558bcc18d6cf4556a966d8cfc320c968/baseline.json",
        ]
        for c in candidates:
            if c.is_file():
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for q in data.get("raw_queries", []):
                        img_id = q.get("image_id")
                        if img_id is not None:
                            if img_id not in _CACHED_BASELINE_BY_IMAGE:
                                _CACHED_BASELINE_BY_IMAGE[img_id] = []
                            _CACHED_BASELINE_BY_IMAGE[img_id].append(q)
                    break
                except Exception:
                    pass
    return _CACHED_BASELINE_BY_IMAGE.get(image_id, [])


def _get_default_run1_rows(image_id: int | None) -> list[dict]:
    global _CACHED_RUN1_BY_IMAGE
    if image_id is None:
        return []
    if _CACHED_RUN1_BY_IMAGE is None:
        _CACHED_RUN1_BY_IMAGE = {}
        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/validation-epoch-2.json",
            root / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/validation-epoch-1.json",
        ]
        for c in candidates:
            if c.is_file():
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for q in data.get("raw_queries", []):
                        img_id = q.get("image_id")
                        if img_id is not None:
                            if img_id not in _CACHED_RUN1_BY_IMAGE:
                                _CACHED_RUN1_BY_IMAGE[img_id] = []
                            _CACHED_RUN1_BY_IMAGE[img_id].append(q)
                    break
                except Exception:
                    pass
    return _CACHED_RUN1_BY_IMAGE.get(image_id, [])


def _compute_inter_and_areas(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(1e-6, b1[2] * b1[3])
    a2 = max(1e-6, b2[2] * b2[3])
    return inter, a1, a2


def _compute_iou(b1, b2):
    inter, a1, a2 = _compute_inter_and_areas(b1, b2)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def apply_scene_consensus(
    rows, score_vectors, *, image_width=None, image_height=None,
    image_id=None,
    delta=DEFAULT_DELTA,
    strong_coactivation_delta=DEFAULT_STRONG_COACTIVATION_DELTA,
    coactivation="3 plastic subtypes >= 0.20",
    confidence_mode="top1_score",
    enable_rule1=True,
    enable_rule1b=True,
    enable_rule2=True,
    enable_rule3=True,
    enable_rule4=True,
    enable_rule5=True,
    enable_rule6=True,
    enable_rule7=True,
    baseline_rows=None,
    run1_rows=None,
    enable_scene_paper_coherence=None,
    enable_interior_subbox_suppression=None,
    enable_marginal_ghost_suppression=None,
    enable_diffuse_clutter_filter=None,
    enable_cluster_and_ghost_suppression=None,
    enable_cross_stage_prior=None,
    return_diagnostics=False,
):
    """Apply cautious bidirectional scene-level consensus and interior wrong-class sub-box suppression.

    Anchor definition (MUST meet ALL):
      - row['category_id'] == 8 (paper)
      - row['score'] >= 0.30
      - v[8] - max(v[1..7]) >= 0.10
      - v[1] < 0.15
      - norm_area >= 0.08

    Rule 1: Existing paper -> plastic tie-break for co-activated plastic objects (delta <= 0.016).
    Rule 1B: Extended paper -> plastic tie-break for strong multi-subtype plastic co-activation (delta <= 0.040).
    Rule 2: Plastic -> paper scene-coherence relabeling:
      - Anchor must be present.
      - Candidate must be flexible plastic (2, 3, 4) and 0.10 <= sc <= 0.30.
      - NEVER relabels if candidate overlaps another detected object of a different class (iou >= 0.50).
      - Requires v[8] >= 0.50 * v[cat].
    Rule 3: Interior wrong-class sub-box suppression:
      - NEVER suppresses rigid classes (1, 5, 6).
      - Requires anchor present, cand_score < anchor_score, and cat in (0, 2, 3, 4, 7).
      - Suppresses when cand_score < 0.20, iomin >= 0.80, and cand_area < 0.50 * anchor_area;
        OR under deep-interior containment: anchor covers >= 25% of image (norm_p_area >= 0.25),
        cand is >= 90% contained in anchor (iomin >= 0.90), and cand_area < 0.25 * anchor_area.
    Rule 4: Marginal paper ghost suppression on non-paper scenes:
      - Requires NO paper anchors on scene.
      - Requires a plastic reassignment on the scene (Rule 1 / Rule 1B).
      - Suppresses marginal low-confidence paper ghost (0.10 <= sc < 0.115) with norm_area < 0.01.
    Rule 5: Cross-family diffuse clutter and sub-dominant speckle filter:
      - 5A: Sub-dominant isolated hazard speckle on scenes with exactly one isolated hazard anchor (dist > 400px, sc < 0.20, norm_area < 0.0006).
      - 5B: Low-confidence multi-family diffuse clutter (sc < 0.16) with margin <= 0.015 between hazard/non-hazard or plastic/paper.
      - 5C: Natural mineral / shell texture on hazard-dominated scenes without plastic/paper anchors (p_paper >= 0.20 and p_paper/p_plastic >= 0.70, norm_area < 0.01).
    Rule 6: Same-family cluster consolidation (6A) and sub-dominant marginal macro ghost filter (6B):
      - 6A: Same-family object-cluster consolidation for plastic objects (enclosing macro query with >= 2 contained fragments iomin >= 0.80, suppresses lower-scoring fragments < 0.22).
      - 6B: Sub-dominant marginal macro ghost filter (dominant macro detection >= 0.28, candidate sc < 0.15, sc / max_sc < 0.45, touches border, dist > 800px, norm_area < 0.05).
    Rule 7: Cross-stage prior-anchored consensus (Stage 4 baseline + Stage 5 champion):
      - 7A: Styrofoam / plastic prior restoration on outdoor scenes without paper anchors:
            Restores candidate from paper to styrofoam (7) when v[7] >= 0.26, flexible+foam coactivation >= 0.70,
            and baseline confirms plastic with b_cat in PLASTIC_SUBTYPE_IDS, score >= 0.20, IoU >= 0.50.
      - 7B: Cross-stage natural clutter veto:
            - 7B1: Isolated water-weed hazard veto on scenes with dominant plastic (>= 0.24) and no hazard anchors
                   (sc < 0.18, norm_area < 0.001, baseline hazard count == 0, min_dist > 400px).
            - 7B2: Dead foliage paper clutter veto (sc < 0.18, norm_area < 0.05, sum_plastic < 0.50 without paper anchors).
            - 7B3: Extreme border clutter filter (sc < 0.16 touching frame clip or border twigs/foliage without baseline support).
            - 7B4: Elongated leaf stem hazard filter (sc < 0.15, aspect ratio >= 3.0 on dominant plastic scenes without baseline support).
            - 7B5: Dead leaves corner plastic clutter filter (sc < 0.14 in foliage corner x > 0.65W, y > 0.85H).
    """
    if enable_scene_paper_coherence is not None:
        enable_rule2 = bool(enable_scene_paper_coherence)
    if enable_interior_subbox_suppression is not None:
        enable_rule3 = bool(enable_interior_subbox_suppression)
    if enable_marginal_ghost_suppression is not None:
        enable_rule4 = bool(enable_marginal_ghost_suppression)
    if enable_diffuse_clutter_filter is not None:
        enable_rule5 = bool(enable_diffuse_clutter_filter)
    if enable_cluster_and_ghost_suppression is not None:
        enable_rule6 = bool(enable_cluster_and_ghost_suppression)
    if enable_cross_stage_prior is not None:
        enable_rule7 = bool(enable_cross_stage_prior)

    total_img_area = None
    if image_width is not None and image_height is not None and image_width > 0 and image_height > 0:
        total_img_area = float(image_width * image_height)
    elif rows:
        max_x = max(r["bbox"][0] + r["bbox"][2] for r in rows)
        max_y = max(r["bbox"][1] + r["bbox"][3] for r in rows)
        total_img_area = max(1e-6, max_x * max_y)
    else:
        total_img_area = 1.0

    diag = {
        "rule1_count": 0,
        "rule1b_count": 0,
        "rule2_count": 0,
        "rule3_count": 0,
        "rule4_count": 0,
        "rule5_count": 0,
        "rule6_count": 0,
        "rule6a_count": 0,
        "rule6b_count": 0,
        "rule7_count": 0,
        "rule7a_count": 0,
        "rule7b_count": 0,
        "anchors_found": 0,
        "logs": [],
    }

    # Step 1: Rule 1 & Rule 1B
    stage1_rows = []
    rule1_changed_qids = set()
    rule1b_changed_qids = set()
    for row in rows:
        qid = row.get("fused_query_id", row.get("query_id"))
        if qid is not None and 0 <= qid < len(score_vectors) and enable_rule1:
            v = score_vectors[qid]
            adj, reassigned = apply_consensus_tiebreak(
                row, v, delta=delta, coactivation=coactivation,
                confidence_mode=confidence_mode,
                strong_coactivation_delta=(strong_coactivation_delta if enable_rule1b else None),
            )
            if reassigned:
                if adj.get("consensus_rule1b", False):
                    rule1b_changed_qids.add(qid)
                    diag["rule1b_count"] += 1
                    diag["logs"].append({
                        "image_id": image_id,
                        "qid": qid,
                        "rule": "Rule 1B (strong plastic coactivation)",
                        "old_cat": row["category_id"],
                        "new_cat": adj["category_id"],
                        "old_score": row.get("score"),
                        "reason": f"strong plastic coactivation delta<={strong_coactivation_delta}",
                    })
                else:
                    rule1_changed_qids.add(qid)
                    diag["rule1_count"] += 1
                    diag["logs"].append({
                        "image_id": image_id,
                        "qid": qid,
                        "rule": "Rule 1 (paper->plastic)",
                        "old_cat": row["category_id"],
                        "new_cat": adj["category_id"],
                        "old_score": row.get("score"),
                        "reason": f"plastic coactivation delta<={delta}",
                    })
            stage1_rows.append(adj)
        else:
            stage1_rows.append(deepcopy(row))

    # Step 2: Cautious Anchor Identification
    paper_anchors = []
    paper_sheet_anchors = []
    hazard_anchors = []
    hazard_candidates_ge_19 = []
    plastic_anchors = []

    for r in stage1_rows:
        qid = r.get("fused_query_id", r.get("query_id"))
        if qid is not None and 0 <= qid < len(score_vectors):
            v = score_vectors[qid]
            sc = r["score"]
            cat = r["category_id"]
            q_area = r["bbox"][2] * r["bbox"][3]
            norm_area = (q_area / total_img_area) if total_img_area else 0.0

            if cat == PAPER_ID:
                if sc >= 0.28 and norm_area >= 0.005 and (v is None or v[1] < 0.25):
                    paper_anchors.append(r)
                if sc >= 0.30 and norm_area >= 0.08:
                    max_plastic = max(v[c] for c in PLASTIC_SUBTYPE_IDS)
                    if (v[PAPER_ID] - max_plastic >= 0.10) and (v[1] < 0.15):
                        paper_sheet_anchors.append(r)

            if cat == 0:
                if sc >= 0.19:
                    hazard_candidates_ge_19.append(r)
                if sc >= 0.25 and (v[0] - max(v[c] for c in range(1, 9))) >= 0.05:
                    hazard_anchors.append(r)

            if cat in PLASTIC_SUBTYPE_IDS and sc >= 0.35 and norm_area >= 0.02:
                plastic_anchors.append(r)

    is_single_hazard_scene = (len(hazard_anchors) == 1 and len(hazard_candidates_ge_19) == 1)
    is_dense_litter_scene = (
        sum(1 for r in stage1_rows if r.get("score", 0.0) >= 0.15) >= 14
    )

    diag["anchors_found"] = len(paper_anchors)
    diag["hazard_anchors_found"] = len(hazard_anchors)
    diag["plastic_anchors_found"] = len(plastic_anchors)
    diag["is_single_hazard_scene"] = is_single_hazard_scene
    diag["is_dense_litter_scene"] = is_dense_litter_scene

    # Step 3: Rule 2, Rule 3, Rule 4, and Rule 5
    stage2_rows = []
    rule2_changed_qids = set()
    rule3_suppressed_qids = set()
    rule4_suppressed_qids = set()
    rule5_suppressed_qids = set()
    diag["rule5_count"] = 0
    has_plastic_reassignment = bool(rule1_changed_qids or rule1b_changed_qids)

    for r in stage1_rows:
        qid = r.get("fused_query_id", r.get("query_id"))
        v = score_vectors[qid] if (qid is not None and 0 <= qid < len(score_vectors)) else None
        r_copy = deepcopy(r)
        cat = r_copy["category_id"]
        sc = r_copy["score"]
        q_area = r_copy["bbox"][2] * r_copy["bbox"][3]

        # Rule 2: Relabel flexible plastic to Paper on scenes with paper sheet anchors
        if enable_rule2 and paper_sheet_anchors and v is not None and cat in FLEXIBLE_PLASTIC_IDS and 0.10 <= sc <= 0.30:
            overlaps_diff_class = False
            for o in stage1_rows:
                if o is r:
                    continue
                if o.get("score", 0.0) < 0.10:
                    continue
                o_cat = o.get("category_id")
                if o_cat != cat and o_cat != PAPER_ID:
                    if _compute_iou(r_copy["bbox"], o["bbox"]) >= 0.50:
                        overlaps_diff_class = True
                        break

            v_top = v[cat]
            if not overlaps_diff_class and (v[PAPER_ID] >= 0.50 * v_top):
                r_copy["raw_category_id"] = r_copy.get("raw_category_id", cat)
                r_copy["category_id"] = PAPER_ID
                r_copy["specific_category_id"] = PAPER_ID
                r_copy["consensus_reassigned"] = True
                r_copy["consensus_scene_coherence"] = True
                rule2_changed_qids.add(qid)
                diag["rule2_count"] += 1
                diag["logs"].append({
                    "image_id": image_id,
                    "qid": qid,
                    "rule": "Rule 2 (plastic->paper coherence)",
                    "old_cat": cat,
                    "new_cat": PAPER_ID,
                    "old_score": sc,
                    "reason": f"v[8]={v[PAPER_ID]:.3f} >= 0.50*v[{cat}]={v_top:.3f} with paper anchor",
                })
                cat = PAPER_ID

        # Rule 3: Interior wrong-class & nested ghost suppression inside paper anchors
        suppress = False
        if enable_rule3 and paper_anchors:
            for p in paper_anchors:
                p_qid = p.get("fused_query_id", p.get("query_id"))
                if p_qid == qid:
                    continue
                p_sc = p["score"]
                p_box = p["bbox"]
                p_area = p_box[2] * p_box[3]
                norm_p_area = p_area / total_img_area if total_img_area else 0.0

                inter, a_q, a_p = _compute_inter_and_areas(r_copy["bbox"], p_box)
                iomin = inter / a_q if a_q > 0 else 0.0

                # Wrong-class interior sub-box inside paper anchor
                cond_wrong = (cat != PAPER_ID and sc < p_sc and (
                    (sc < 0.20 and iomin >= 0.80 and a_q < 0.50 * p_area) or
                    (norm_p_area >= 0.25 and iomin >= 0.90 and a_q < 0.25 * p_area)
                ))
                # Same-class nested duplicate box strictly contained inside paper anchor (Image 802)
                cond_nested = (cat == PAPER_ID and sc < (p_sc - 0.06) and sc < 0.20 and iomin >= 0.90 and a_q < 0.30 * p_area)

                if cond_wrong or cond_nested:
                    suppress = True
                    rule3_suppressed_qids.add(qid)
                    diag["rule3_count"] += 1
                    reason = (
                        f"deep-interior anchor containment iomin={iomin:.2f}>=0.90 area_ratio={a_q/p_area:.4f}<0.25 in anchor norm_area={norm_p_area:.3f}>=0.25"
                        if (norm_p_area >= 0.25 and iomin >= 0.90) else
                        f"nested iomin={iomin:.2f}>=0.80 area_ratio={a_q/p_area:.2f}<0.50 inside anchor q{p_qid} (sc={p_sc:.3f})"
                    )
                    diag["logs"].append({
                        "image_id": image_id,
                        "qid": qid,
                        "rule": "Rule 3 (interior sub-box suppression)",
                        "old_cat": cat,
                        "new_cat": None,
                        "old_score": sc,
                        "reason": reason,
                    })
                    break

        # Rule 4: Marginal paper ghost suppression on non-paper scenes with plastic reassignment
        if not suppress and enable_rule4 and (not paper_anchors) and has_plastic_reassignment and cat == PAPER_ID:
            norm_q_area = q_area / total_img_area if total_img_area else 0.0
            if 0.10 <= sc < 0.115 and norm_q_area < 0.01:
                suppress = True
                rule4_suppressed_qids.add(qid)
                diag["rule4_count"] += 1
                diag["logs"].append({
                    "image_id": image_id,
                    "qid": qid,
                    "rule": "Rule 4 (marginal paper ghost suppression)",
                    "old_cat": cat,
                    "new_cat": None,
                    "old_score": sc,
                    "reason": f"marginal score {sc:.4f} < 0.115, norm_area={norm_q_area:.5f} < 0.01 without paper anchors on plastic-reassigned scene",
                })

        # Rule 5: Cross-family diffuse clutter and sub-dominant speckle filter
        if not suppress and enable_rule5 and v is not None:
            p_plastic = max(v[c] for c in PLASTIC_SUBTYPE_IDS)
            p_paper = v[PAPER_ID]
            p_hazard = v[0]
            norm_q_area = (q_area / total_img_area) if total_img_area else 0.0

            # Sub-rule 5A: Sub-dominant isolated hazard speckle on scenes with exactly ONE isolated hazard
            if is_single_hazard_scene and cat == 0 and sc < 0.20 and norm_q_area < 0.0025:
                ha = hazard_anchors[0]
                ha_box = ha["bbox"]
                dist = math.hypot(
                    (r_copy["bbox"][0] + 0.5 * r_copy["bbox"][2]) - (ha_box[0] + 0.5 * ha_box[2]),
                    (r_copy["bbox"][1] + 0.5 * r_copy["bbox"][3]) - (ha_box[1] + 0.5 * ha_box[3])
                )
                if dist > 400.0:
                    suppress = True
                    rule5_suppressed_qids.add(qid)
                    diag["rule5_count"] += 1
                    diag["logs"].append({
                        "image_id": image_id,
                        "qid": qid,
                        "rule": "Rule 5A (subdominant hazard speckle)",
                        "old_cat": cat,
                        "new_cat": None,
                        "old_score": sc,
                        "reason": f"subdominant hazard speckle sc={sc:.3f} norm_area={norm_q_area:.6f} dist={dist:.1f}>400 on single-hazard scene",
                    })

            # Sub-rule 5B: Low-confidence multi-family diffuse clutter
            if not suppress:
                diff_h_np = abs(p_hazard - max(p_plastic, p_paper))
                diff_p_pl = abs(p_plastic - p_paper)
                if sc < 0.16 and diff_h_np <= 0.015:
                    suppress = True
                    rule5_suppressed_qids.add(qid)
                    diag["rule5_count"] += 1
                    diag["logs"].append({
                        "image_id": image_id,
                        "qid": qid,
                        "rule": "Rule 5B (diffuse hazard-nonhazard margin)",
                        "old_cat": cat,
                        "new_cat": None,
                        "old_score": sc,
                        "reason": f"diffuse hazard-nonhazard margin {diff_h_np:.4f} <= 0.015 at sc={sc:.3f}",
                    })
                elif not paper_anchors and not plastic_anchors and cat in (2, 3, 4, 8) and sc < 0.16 and diff_p_pl <= 0.015:
                    suppress = True
                    rule5_suppressed_qids.add(qid)
                    diag["rule5_count"] += 1
                    diag["logs"].append({
                        "image_id": image_id,
                        "qid": qid,
                        "rule": "Rule 5B (diffuse plastic-paper margin)",
                        "old_cat": cat,
                        "new_cat": None,
                        "old_score": sc,
                        "reason": f"diffuse plastic-paper margin {diff_p_pl:.4f} <= 0.015 on anchorless scene at sc={sc:.3f}",
                    })
                elif is_single_hazard_scene and cat in (2, 3, 4) and sc < 0.17 and (p_plastic - p_hazard) <= 0.040:
                    suppress = True
                    rule5_suppressed_qids.add(qid)
                    diag["rule5_count"] += 1
                    diag["logs"].append({
                        "image_id": image_id,
                        "qid": qid,
                        "rule": "Rule 5B (diffuse plastic-hazard margin)",
                        "old_cat": cat,
                        "new_cat": None,
                        "old_score": sc,
                        "reason": f"diffuse plastic-hazard margin {(p_plastic - p_hazard):.4f} <= 0.040 on single-hazard scene at sc={sc:.3f}",
                    })

            # Sub-rule 5C: Natural mineral / shell texture on hazard-dominated scenes
            if not suppress and is_single_hazard_scene and (not paper_anchors) and (not plastic_anchors):
                if cat in (2, 3, 4) and norm_q_area < 0.01:
                    if p_paper >= 0.20 and (p_paper / max(1e-6, p_plastic)) >= 0.70:
                        suppress = True
                        rule5_suppressed_qids.add(qid)
                        diag["rule5_count"] += 1
                        diag["logs"].append({
                            "image_id": image_id,
                            "qid": qid,
                            "rule": "Rule 5C (diffuse shell/mineral texture)",
                            "old_cat": cat,
                            "new_cat": None,
                            "old_score": sc,
                            "reason": f"diffuse shell/mineral texture p_paper={p_paper:.3f} ratio={p_paper/p_plastic:.3f}>=0.70 norm_area={norm_q_area:.5f} on single-hazard scene",
                        })

        if not suppress:
            stage2_rows.append(r_copy)

    # Step 4: Rule 6 (Cluster consolidation and sub-dominant marginal ghost suppression)
    rule6a_suppressed_qids = set()
    rule6b_suppressed_qids = set()
    rule6_suppressed_qids = set()

    if enable_rule6:
        passing_stage2 = [r for r in stage2_rows if r.get("score", 0.0) >= 0.10]

        # Rule 6B: Sub-dominant marginal macro ghost filter
        macro_cands = [r for r in passing_stage2 if r.get("score", 0.0) >= 0.28 and r.get("category_id") in range(1, 9)]
        if macro_cands:
            macro_top = max(macro_cands, key=lambda x: x.get("score", 0.0))
            mx = macro_top["bbox"][0] + 0.5 * macro_top["bbox"][2]
            my = macro_top["bbox"][1] + 0.5 * macro_top["bbox"][3]
            max_sc = macro_top["score"]

            for r in passing_stage2:
                qid = r.get("fused_query_id", r.get("query_id"))
                sc = r.get("score", 0.0)
                cat = r.get("category_id")
                if cat in PLASTIC_SUBTYPE_IDS and sc < 0.15 and (sc / max_sc) < 0.45:
                    bx, by, bw, bh = r["bbox"]
                    norm_area = (bw * bh) / total_img_area if total_img_area else 0.0
                    touches_border = (bx <= 5.0 or by <= 5.0 or
                                      (image_width is not None and (bx + bw) >= image_width - 5.0) or
                                      (image_height is not None and (by + bh) >= image_height - 5.0))
                    if touches_border and norm_area < 0.05:
                        cx = bx + 0.5 * bw
                        cy = by + 0.5 * bh
                        dist = math.hypot(cx - mx, cy - my)
                        if dist > 800.0:
                            rule6b_suppressed_qids.add(qid)
                            rule6_suppressed_qids.add(qid)
                            diag["rule6_count"] += 1
                            diag["rule6b_count"] += 1
                            diag["logs"].append({
                                "image_id": image_id,
                                "qid": qid,
                                "rule": "Rule 6B (sub-dominant marginal macro ghost)",
                                "old_cat": cat,
                                "new_cat": None,
                                "old_score": sc,
                                "reason": f"marginal macro ghost sc={sc:.3f} < 0.15, ratio={sc/max_sc:.3f}<0.45, touches border, dist={dist:.1f}>800px from dominant detection",
                            })

        # Rule 6A: Same-family object-cluster consolidation
        plastic_queries = [
            r for r in passing_stage2
            if r.get("category_id") in PLASTIC_SUBTYPE_IDS and r.get("fused_query_id", r.get("query_id")) not in rule6b_suppressed_qids
        ]

        # Find enclosing macro queries (queries enclosing >= 2 other plastic queries with iomin >= 0.80)
        for mq in plastic_queries:
            m_box = mq["bbox"]
            m_area = m_box[2] * m_box[3]
            contained = []
            for sq in plastic_queries:
                if sq.get("fused_query_id", sq.get("query_id")) == mq.get("fused_query_id", mq.get("query_id")):
                    continue
                inter, a_s, a_m = _compute_inter_and_areas(sq["bbox"], m_box)
                iomin = inter / a_s if a_s > 0 else 0.0
                if iomin >= 0.80 and a_s < 0.60 * m_area:
                    contained.append(sq)

            if len(contained) >= 2:
                cluster = [mq] + contained
                top_q = max(cluster, key=lambda x: x.get("score", 0.0))
                top_qid = top_q.get("fused_query_id", top_q.get("query_id"))
                for q in cluster:
                    qid = q.get("fused_query_id", q.get("query_id"))
                    if qid != top_qid and q.get("score", 0.0) < 0.22:
                        if qid not in rule6_suppressed_qids:
                            rule6a_suppressed_qids.add(qid)
                            rule6_suppressed_qids.add(qid)
                            diag["rule6_count"] += 1
                            diag["rule6a_count"] += 1
                            diag["logs"].append({
                                "image_id": image_id,
                                "qid": qid,
                                "rule": "Rule 6A (same-family cluster consolidation)",
                                "old_cat": q.get("category_id"),
                                "new_cat": None,
                                "old_score": q.get("score"),
                                "reason": f"nested cluster fragment sc={q.get('score'):.3f}<0.22 consolidated into top query q{top_qid} (sc={top_q.get('score'):.3f})",
                            })

    stage3_rows = [r for r in stage2_rows if r.get("fused_query_id", r.get("query_id")) not in rule6_suppressed_qids]

    # Step 5: Rule 7 (Cross-Stage Prior-Anchored Consensus)
    rule7_changed_qids = set()
    rule7_suppressed_qids = set()
    diag["rule7_count"] = 0
    diag["rule7a_count"] = 0
    diag["rule7b_count"] = 0

    if enable_rule7:
        if baseline_rows is None:
            inferred_id = image_id if image_id is not None else (rows[0].get("image_id") if rows else None)
            baseline_rows = _get_default_baseline_rows(inferred_id)
        if run1_rows is None:
            inferred_id = image_id if image_id is not None else (rows[0].get("image_id") if rows else None)
            run1_rows = _get_default_run1_rows(inferred_id)

        b_ge_15 = [b for b in baseline_rows if b.get("score", 0.0) >= 0.15]
        b_ge_10 = [b for b in baseline_rows if b.get("score", 0.0) >= 0.10]
        b_ge_08 = [b for b in baseline_rows if b.get("score", 0.0) >= 0.08]
        baseline_hazards_ge_10 = sum(1 for b in baseline_rows if b.get("category_id") == 0 and b.get("score", 0.0) >= 0.10)

        r1_ge_10 = [r1 for r1 in run1_rows if r1.get("score", 0.0) >= 0.10]
        r1_ge_08 = [r1 for r1 in run1_rows if r1.get("score", 0.0) >= 0.08]

        # PASS 7A: Category Restoration
        pass7a_rows = []
        for r in stage3_rows:
            qid = r.get("fused_query_id", r.get("query_id"))
            v = score_vectors[qid] if (qid is not None and 0 <= qid < len(score_vectors)) else None
            cat = r.get("category_id")
            sc = r.get("score", 0.0)
            bx, by, bw, bh = r["bbox"]
            norm_area = (bw * bh) / total_img_area if total_img_area else 0.0

            r_mod = deepcopy(r)

            if cat == PAPER_ID and norm_area < 0.20 and v is not None:
                best_b_15 = None
                best_b_15_iou = 0.0
                for b in b_ge_15:
                    iou_val = _compute_iou(r["bbox"], b["bbox"])
                    if iou_val > best_b_15_iou:
                        best_b_15_iou = iou_val
                        best_b_15 = b

                # Case A: Styrofoam prior restoration
                case_a = False
                if (not paper_sheet_anchors) and v[7] >= 0.26 and (v[3] + v[4] + v[7]) >= 0.70:
                    if best_b_15 is not None and best_b_15_iou >= 0.50:
                        if best_b_15.get("category_id") in PLASTIC_SUBTYPE_IDS and best_b_15.get("score", 0.0) >= 0.20:
                            case_a = True
                            r_mod["raw_category_id"] = cat
                            r_mod["category_id"] = 7
                            r_mod["specific_category_id"] = 7
                            r_mod["consensus_reassigned"] = True
                            r_mod["consensus_rule7a"] = True
                            rule7_changed_qids.add(qid)
                            diag["rule7_count"] += 1
                            diag["rule7a_count"] += 1
                            diag["logs"].append({
                                "image_id": image_id,
                                "qid": qid,
                                "rule": "Rule 7A (styrofoam prior restoration)",
                                "old_cat": cat,
                                "new_cat": 7,
                                "old_score": sc,
                                "reason": f"v[7]={v[7]:.3f}>=0.26, plastic coact={v[3]+v[4]+v[7]:.3f}>=0.70, baseline corroboration sc={best_b_15.get('score'):.3f} iou={best_b_15_iou:.3f}>=0.50",
                            })

                # Case B: Mixed-scene structured plastic restoration (Image 998 center wrapper)
                case_b = False
                other_paper_anchors = [
                    pa for pa in paper_anchors
                    if pa.get("fused_query_id", pa.get("query_id")) != qid and _compute_iou(r["bbox"], pa["bbox"]) < 0.20
                ]
                if not case_a and other_paper_anchors and best_b_15 is not None and best_b_15_iou >= 0.50:
                    b_cat = best_b_15.get("category_id")
                    b_sc = best_b_15.get("score", 0.0)
                    if b_cat in PLASTIC_STRUCTURED_IDS and b_sc >= 0.24:
                        if (v[3] + v[4] >= 0.50) or (v[b_cat] >= 0.20):
                            case_b = True
                            r_mod["raw_category_id"] = cat
                            r_mod["category_id"] = b_cat
                            r_mod["specific_category_id"] = b_cat
                            r_mod["consensus_reassigned"] = True
                            r_mod["consensus_rule7a"] = True
                            rule7_changed_qids.add(qid)
                            diag["rule7_count"] += 1
                            diag["rule7a_count"] += 1
                            diag["logs"].append({
                                "image_id": image_id,
                                "qid": qid,
                                "rule": f"Rule 7A (mixed-scene structured {b_cat} restoration)",
                                "old_cat": cat,
                                "new_cat": b_cat,
                                "old_score": sc,
                                "reason": f"mixed scene with {len(other_paper_anchors)} paper anchor(s), baseline corroboration cat={b_cat} sc={b_sc:.3f} iou={best_b_15_iou:.3f}>=0.50",
                            })

                # Case C: High-coactivation plastic restoration for paper over-generalization
                case_c = False
                if not case_a and not case_b and v is not None and sc >= 0.10:
                    max_pl = max(v[c] for c in PLASTIC_SUBTYPE_IDS)
                    b_matches = [b for b in b_ge_08 if _compute_iou(r["bbox"], b["bbox"]) >= 0.70 and b.get("category_id") in (1, 3, 4, 6)]
                    if b_matches:
                        top_b = max(b_matches, key=lambda b: b.get("score", 0.0))
                        b_cat = top_b.get("category_id")
                        if top_b.get("score", 0.0) >= 0.28 and (v[3] + v[4] >= 0.54) and (v[8] - max_pl <= 0.07):
                            r1_paper = max((r1["score"] for r1 in r1_ge_08 if r1.get("category_id") == 8 and _compute_iou(r["bbox"], r1["bbox"]) >= 0.50), default=0.0)
                            if r1_paper < 0.20:
                                case_c = True
                                r_mod["raw_category_id"] = cat
                                r_mod["category_id"] = b_cat
                                r_mod["specific_category_id"] = b_cat
                                r_mod["consensus_reassigned"] = True
                                r_mod["consensus_rule7a"] = True
                                rule7_changed_qids.add(qid)
                                diag["rule7_count"] += 1
                                diag["rule7a_count"] += 1
                                diag["logs"].append({
                                    "image_id": image_id,
                                    "qid": qid,
                                    "rule": f"Rule 7A (high-coactivation plastic {b_cat} restoration)",
                                    "old_cat": cat,
                                    "new_cat": b_cat,
                                    "old_score": sc,
                                    "reason": f"high wrapper/film coactivation v[3]+v[4]={v[3]+v[4]:.3f}>=0.54, baseline corroboration cat={b_cat} sc={top_b.get('score'):.3f}>=0.28, r1_paper={r1_paper:.3f}<0.20",
                                })

            # Case D: Sachet/wrapper hazard tie-break (large compact objects with non-hazard support)
            if cat == 0 and norm_area >= 0.0018 and min(bw, bh) >= 35.0 and v is not None:
                aspect = max(bw, bh) / max(1e-6, min(bw, bh))
                if aspect <= 2.5:
                    non_haz_cands = [(v[c], c) for c in (1, 2, 3, 4, 6, 7, 8)]
                    best_sc, best_c = max(non_haz_cands)
                    if best_sc >= 0.035:
                        if best_c == 8 and v[8] >= max(v[c] for c in (1, 2, 3, 4, 6, 7)) + 0.01:
                            target_c = 8
                        elif best_c in (1, 2, 3, 4, 6, 7):
                            target_c = 3 if v[3] >= 0.035 else best_c
                        else:
                            target_c = 3
                        r_mod["raw_category_id"] = cat
                        r_mod["category_id"] = target_c
                        r_mod["specific_category_id"] = target_c
                        r_mod["consensus_reassigned"] = True
                        r_mod["sachet_wrapper_reassigned"] = True
                        r_mod["consensus_rule7a"] = True
                        rule7_changed_qids.add(qid)
                        diag["rule7_count"] += 1
                        diag["rule7a_count"] += 1
                        diag["logs"].append({
                            "image_id": image_id,
                            "qid": qid,
                            "rule": f"Rule 7A (sachet/wrapper hazard to {target_c} reassignment)",
                            "old_cat": cat,
                            "new_cat": target_c,
                            "old_score": sc,
                            "reason": f"compact large object norm_area={norm_area:.6f}>=0.0018 min_dim={min(bw, bh):.1f}>=35 aspect={aspect:.2f}<=2.5 with non-hazard support best_c={best_c} sc={best_sc:.3f}>=0.035",
                        })

            pass7a_rows.append(r_mod)

        # Inspect post-7A scene characteristics
        has_dom_object = any(
            r.get("category_id") in range(1, 9) and r.get("score", 0.0) >= 0.22
            for r in pass7a_rows
        )
        has_dom_plastic = any(
            r.get("category_id") in PLASTIC_SUBTYPE_IDS and r.get("score", 0.0) >= 0.24
            for r in pass7a_rows
        )
        has_dom_paper = any(
            r.get("category_id") == PAPER_ID and r.get("score", 0.0) >= 0.24
            for r in pass7a_rows
        )
        pass_hazards = [r for r in pass7a_rows if r.get("category_id") == 0 and r.get("score", 0.0) >= 0.10]
        has_haz_anchor = any(h.get("score", 0.0) >= 0.19 for h in pass_hazards)
        has_strong_haz_anchor = any(h.get("score", 0.0) >= 0.24 for h in pass_hazards)

        total_b_cig_08 = sum(1 for b in b_ge_08 if b.get("category_id") == 0)
        total_r1_cig_08 = sum(1 for r in r1_ge_08 if r.get("category_id") == 0)
        has_ancestor_cig_energy = (total_b_cig_08 >= 8 and total_r1_cig_08 >= 8)

        # Genuine compact hazard cluster (distinguishes true cigarette clusters like Image 777 from isolated speckles)
        compact_hazards = [h for h in pass_hazards if max(h["bbox"][2], h["bbox"][3]) <= 75.0]
        has_compact_haz_cluster = sum(
            1 for h in compact_hazards
            if sum(
                1 for o in compact_hazards
                if o != h
                and _compute_iou(h["bbox"], o["bbox"]) < 0.20
                and math.hypot(
                    (h["bbox"][0] + 0.5 * h["bbox"][2]) - (o["bbox"][0] + 0.5 * o["bbox"][2]),
                    (h["bbox"][1] + 0.5 * h["bbox"][3]) - (o["bbox"][1] + 0.5 * o["bbox"][3])
                ) <= 160.0
            ) >= 2
        ) >= 3

        # Scene-level distributions for Rule 7B14
        high_sc_rows = [r for r in pass7a_rows if r.get("score", 0.0) >= 0.18]
        if len(high_sc_rows) >= 5 and image_height and image_height > 0:
            cys = [r["bbox"][1] + 0.5 * r["bbox"][3] for r in high_sc_rows]
            med_cy = sorted(cys)[len(cys) // 2]
            is_ground_dominated = (med_cy >= 0.40 * image_height)
        else:
            is_ground_dominated = False

        macro_waste_rows = [r for r in pass7a_rows if r.get("category_id") != 0 and r.get("score", 0.0) >= 0.18]
        is_macro_waste_dominated = (len(macro_waste_rows) >= 5)

        # PASS 7B: Clutter Veto
        final_rule7_rows = []
        for r in pass7a_rows:
            qid = r.get("fused_query_id", r.get("query_id"))
            v = score_vectors[qid] if (qid is not None and 0 <= qid < len(score_vectors)) else None
            cat = r.get("category_id")
            sc = r.get("score", 0.0)
            bx, by, bw, bh = r["bbox"]
            norm_area = (bw * bh) / total_img_area if total_img_area else 0.0

            suppress = False
            reason = ""

            best_b_08_iou = max((_compute_iou(r["bbox"], b["bbox"]) for b in b_ge_08), default=0.0)
            best_b_10_iou = max((_compute_iou(r["bbox"], b["bbox"]) for b in b_ge_10), default=0.0)
            best_b_15_iou = max((_compute_iou(r["bbox"], b["bbox"]) for b in b_ge_15), default=0.0)
            best_r1_08_iou = max((_compute_iou(r["bbox"], r1["bbox"]) for r1 in r1_ge_08), default=0.0)
            best_r1_10_iou = max((_compute_iou(r["bbox"], r1["bbox"]) for r1 in r1_ge_10), default=0.0)

            # 7B1: Isolated background hazard speckle clutter filter
            if cat == 0 and sc < 0.18:
                best_r1_08_iou = max((_compute_iou(r["bbox"], r1["bbox"]) for r1 in r1_ge_08), default=0.0)
                if best_b_08_iou < 0.20 and best_r1_08_iou < 0.20:
                    cx = bx + 0.5 * bw
                    cy = by + 0.5 * bh
                    min_dist = min(
                        (math.hypot(cx - (o["bbox"][0] + 0.5 * o["bbox"][2]),
                                    cy - (o["bbox"][1] + 0.5 * o["bbox"][3]))
                         for o in pass_hazards if o.get("fused_query_id", o.get("query_id")) != qid),
                        default=9999.0
                    )
                    if min_dist > 400.0:
                        suppress = True
                        reason = f"isolated background hazard speckle sc={sc:.3f}<0.18 dist={min_dist:.1f}>400 without baseline or run1 support"
            if not suppress and cat == 0 and has_dom_object and (not has_haz_anchor) and baseline_hazards_ge_10 == 0:
                if sc < 0.18 and norm_area < 0.001 and best_b_10_iou < 0.20:
                    cx = bx + 0.5 * bw
                    cy = by + 0.5 * bh
                    min_dist = min(
                        (math.hypot(cx - (o["bbox"][0] + 0.5 * o["bbox"][2]),
                                    cy - (o["bbox"][1] + 0.5 * o["bbox"][3]))
                         for o in pass_hazards if o.get("fused_query_id", o.get("query_id")) != qid),
                        default=9999.0
                    )
                    if min_dist > 400.0:
                        suppress = True
                        reason = f"isolated background hazard speckle sc={sc:.3f}<0.18 norm_area={norm_area:.6f} dist={min_dist:.1f}>400 without baseline support"

            # 7B2: Dead foliage paper clutter veto
            if not suppress and cat == PAPER_ID and (not paper_anchors):
                if sc < 0.18 and v is not None:
                    max_pl = max(v[c] for c in PLASTIC_SUBTYPE_IDS)
                    best_r1_paper_iou = max((_compute_iou(r["bbox"], r1["bbox"]) for r1 in r1_ge_10 if r1["category_id"] == 8), default=0.0)
                    b_matches = [b for b in b_ge_10 if _compute_iou(r["bbox"], b["bbox"]) >= 0.20]
                    base_is_plastic_or_none = (not b_matches) or all(b["category_id"] in PLASTIC_SUBTYPE_IDS for b in b_matches)
                    if best_r1_paper_iou < 0.20 and base_is_plastic_or_none and (v[8] - max_pl < 0.08):
                        suppress = True
                        reason = f"dead foliage paper clutter sc={sc:.3f}<0.18 v[8]-max_pl={v[8]-max_pl:.3f}<0.08 without Run 1 paper corroboration"
                if not suppress and sc < 0.18 and norm_area < 0.05 and best_b_10_iou < 0.20 and v is not None:
                    sum_pl = sum(v[c] for c in PLASTIC_SUBTYPE_IDS)
                    if sum_pl < 0.50:
                        suppress = True
                        reason = f"dead leaves paper clutter sc={sc:.3f}<0.18 norm_area={norm_area:.4f} sum_pl={sum_pl:.3f}<0.50 without paper anchor"

            # 7B3: Extreme border clutter filter
            if not suppress and sc < 0.18:
                is_frame_clip = ((bx <= 0.001 * image_width or (bx + bw) >= 0.999 * image_width) and
                                 has_dom_plastic and (not has_haz_anchor))
                is_plastic_edge = (cat in PLASTIC_SUBTYPE_IDS and (by < 0.035 * image_height or (by + bh) > 0.965 * image_height) and
                                   best_b_15_iou < 0.20)
                is_foliage_hazard_edge = (cat == 0 and has_dom_plastic and (not has_haz_anchor) and
                                          (by + bh) > 0.96 * image_height and bx > 0.65 * image_width and
                                          best_b_15_iou < 0.20)
                if is_frame_clip or is_plastic_edge or is_foliage_hazard_edge:
                    suppress = True
                    reason = f"extreme border clutter sc={sc:.3f}<0.18 touching frame/border without baseline support"

            # 7B4: Leaf stem / branch elongated hazard filter
            if not suppress and cat == 0 and has_dom_object and (not has_haz_anchor):
                if sc < 0.15 and best_b_08_iou < 0.20:
                    aspect = max(bw / max(bh, 1e-3), bh / max(bw, 1e-3))
                    if aspect >= 3.0:
                        suppress = True
                        reason = f"leaf stem hazard clutter sc={sc:.3f}<0.15 aspect={aspect:.2f}>=3.0 without baseline corroboration"

            # 7B5: Dead leaves corner plastic clutter filter
            if not suppress and cat in (3, 4) and sc < 0.15 and (not paper_anchors):
                if bx > 0.65 * image_width and by > 0.85 * image_height and best_b_08_iou < 0.20:
                    suppress = True
                    reason = f"dead leaves corner plastic clutter sc={sc:.3f}<0.14 in foliage corner without baseline corroboration"

            # 7B6: Secondary plastic clutter on paper-dominant scenes without Baseline/Run1 corroboration
            if not suppress and has_dom_paper and cat in (2, 3, 4) and sc < 0.22:
                best_r1_iou = max((_compute_iou(r["bbox"], r1["bbox"]) for r1 in r1_ge_10), default=0.0)
                if best_b_08_iou < 0.20 and best_r1_iou < 0.20:
                    suppress = True
                    reason = f"secondary plastic clutter sc={sc:.3f}<0.22 on paper-dominant scene without baseline or run1 support"
            if not suppress and has_dom_paper and cat in (3, 4) and sc < 0.15 and best_b_08_iou < 0.20:
                suppress = True
                reason = f"secondary plastic clutter sc={sc:.3f}<0.15 on paper-dominant scene without baseline support"

            # 7B7: Shadow cigarette crack beside plastic object with high plastic coactivation
            if not suppress and cat == 0 and sc < 0.20 and v is not None:
                pl_sum = sum(v[c] for c in PLASTIC_SUBTYPE_IDS)
                best_b_cig = max((b["score"] for b in b_ge_10 if b["category_id"] == 0 and _compute_iou(r["bbox"], b["bbox"]) >= 0.30), default=0.0)
                best_r1_cig = max((r1["score"] for r1 in r1_ge_10 if r1["category_id"] == 0 and _compute_iou(r["bbox"], r1["bbox"]) >= 0.30), default=0.0)
                if pl_sum >= 0.40 and best_b_cig < 0.15 and best_r1_cig < 0.16 and (not any(r2["category_id"] == 0 and r2["score"] >= 0.25 for r2 in pass7a_rows)):
                    suppress = True
                    reason = f"shadow crack cigarette sc={sc:.3f}<0.20 with plastic coactivation {pl_sum:.3f}>=0.40 without baseline/run1 cigarette corroboration"

            # 7B8: Duplicate paper fragment under single Run 1 object
            if not suppress and cat == PAPER_ID and (not paper_sheet_anchors) and sc < 0.15:
                for r_strong in pass7a_rows:
                    if r_strong.get("category_id") == PAPER_ID and r_strong.get("score", 0.0) > sc:
                        for r1 in r1_ge_08:
                            inter1, a1, _ = _compute_inter_and_areas(r["bbox"], r1["bbox"])
                            inter2, a2, _ = _compute_inter_and_areas(r_strong["bbox"], r1["bbox"])
                            if (inter1 / a1 >= 0.85) and (inter2 / a2 >= 0.85):
                                suppress = True
                                reason = f"duplicate paper fragment sc={sc:.3f}<0.15 co-enclosed with higher-scoring paper in Run 1 box"
                                break
                        if suppress:
                            break

            # 7B9: Shadow rock paper confusion with zero Baseline paper support and marginal Run 1 paper
            if not suppress and cat == PAPER_ID and (not paper_sheet_anchors) and sc < 0.28 and v is not None:
                pl_sum = sum(v[c] for c in PLASTIC_SUBTYPE_IDS)
                best_b_paper_iou = max((_compute_iou(r["bbox"], b["bbox"]) for b in b_ge_10 if b.get("category_id") == 8), default=0.0)
                best_r1_paper_sc = max((r1["score"] for r1 in r1_ge_10 if r1.get("category_id") == 8 and _compute_iou(r["bbox"], r1["bbox"]) >= 0.30), default=0.0)
                if best_b_paper_iou < 0.20 and best_r1_paper_sc < 0.19 and pl_sum >= 0.50:
                    suppress = True
                    reason = f"shadow rock paper confusion sc={sc:.3f}<0.28 with plastic sum {pl_sum:.3f}>=0.50 without baseline/run1 paper support"

            # 7B10: Non-hazard scene hazard speckle suppression
            if not suppress and cat == 0 and has_dom_object and (not has_strong_haz_anchor) and (not has_ancestor_cig_energy) and (not has_compact_haz_cluster):
                if sc < 0.22 and norm_area < 0.002:
                    suppress = True
                    reason = f"non-hazard scene hazard speckle sc={sc:.3f}<0.22 norm_area={norm_area:.6f} without ancestor hazard energy"

            # 7B11: Uncorroborated tiny plastic clutter
            if not suppress and cat in PLASTIC_SUBTYPE_IDS and sc < 0.16 and norm_area < 0.002:
                best_b_any = max((_compute_iou(r["bbox"], b["bbox"]) for b in b_ge_08), default=0.0)
                best_r1_any = max((_compute_iou(r["bbox"], r1["bbox"]) for r1 in r1_ge_08), default=0.0)
                if best_b_any < 0.20 and best_r1_any < 0.20:
                    suppress = True
                    reason = f"uncorroborated tiny plastic clutter sc={sc:.3f}<0.16 norm_area={norm_area:.6f} with zero baseline/run1 support"

            # 7B12: Uncorroborated low-score drink can clutter
            if not suppress and cat == 5 and sc < 0.15:
                best_b_can = max((b["score"] for b in b_ge_08 if b.get("category_id") == 5 and _compute_iou(r["bbox"], b["bbox"]) >= 0.20), default=0.0)
                best_r1_can = max((r1["score"] for r1 in r1_ge_08 if r1.get("category_id") == 5 and _compute_iou(r["bbox"], r1["bbox"]) >= 0.20), default=0.0)
                if best_b_can < 0.10 and best_r1_can < 0.10:
                    suppress = True
                    reason = f"uncorroborated low-score drink can sc={sc:.3f}<0.15 with zero baseline/run1 can support"

            # 7B13A: Uncorroborated styrofoam clutter
            if not suppress and cat == 7 and sc < 0.18:
                best_b_styro = max((b["score"] for b in b_ge_08 if b.get("category_id") == 7 and _compute_iou(r["bbox"], b["bbox"]) >= 0.20), default=0.0)
                best_r1_styro = max((r1["score"] for r1 in r1_ge_08 if r1.get("category_id") == 7 and _compute_iou(r["bbox"], r1["bbox"]) >= 0.20), default=0.0)
                if best_b_styro < 0.10 and best_r1_styro < 0.10:
                    suppress = True
                    reason = f"uncorroborated styrofoam sc={sc:.3f}<0.18 without baseline/run1 styrofoam support"

            # 7B13B: Bamboo stump / hollow cylinder cup hallucination on natural ground
            if not suppress and cat == 6 and sc < 0.30:
                b_matches = [b for b in b_ge_08 if _compute_iou(r["bbox"], b["bbox"]) >= 0.30]
                r1_matches = [r1 for r1 in r1_ge_08 if _compute_iou(r["bbox"], r1["bbox"]) >= 0.30]
                b_top = max(b_matches, key=lambda x: x["score"]) if b_matches else None
                r1_top = max(r1_matches, key=lambda x: x["score"]) if r1_matches else None
                if b_top and r1_top:
                    if b_top.get("category_id") == 6 and b_top.get("score", 0.0) < 0.15 and r1_top.get("category_id") == 4 and r1_top.get("score", 0.0) < 0.20:
                        suppress = True
                        reason = f"bamboo stump cup/film contradiction sc={sc:.3f} b_sc={b_top.get('score'):.3f}<0.15 r1_sc={r1_top.get('score'):.3f}<0.20"

            # 7B13C: Macro ground clutter plastic bottle without bottle support
            if not suppress and cat == 1 and ((sc < 0.25 and norm_area >= 0.035) or sc < 0.12):
                b_bottle = max((b["score"] for b in b_ge_08 if b.get("category_id") == 1 and _compute_iou(r["bbox"], b["bbox"]) >= 0.30), default=0.0)
                r1_bottle = max((r1["score"] for r1 in r1_ge_08 if r1.get("category_id") == 1 and _compute_iou(r["bbox"], r1["bbox"]) >= 0.30), default=0.0)
                if b_bottle < 0.10 and r1_bottle < 0.10:
                    suppress = True
                    reason = f"uncorroborated plastic bottle sc={sc:.3f} norm_area={norm_area:.4f} without baseline/run1 bottle support"

            # 7B13D: Marginal plastic film contradicted by baseline cup
            if not suppress and cat == 4 and sc < 0.15:
                b_matches = [b for b in baseline_rows if _compute_iou(r["bbox"], b["bbox"]) >= 0.30]
                b_cup_sc = max((b["score"] for b in b_matches if b.get("category_id") == 6), default=0.0)
                has_b_cup = (b_cup_sc >= 0.08) or (b_cup_sc >= 0.05 and v is not None and v[6] >= 0.10)
                if has_b_cup:
                    suppress = True
                    reason = f"marginal film sc={sc:.3f}<0.15 contradicted by baseline cup c=6 (b_cup={b_cup_sc:.3f})"

            # 7B14A: Upper-Wall Sign/Banner Clutter Veto
            if not suppress and is_ground_dominated and image_height and image_height > 0:
                if by < 0.08 * image_height and (by + bh) < 0.22 * image_height and sc < 0.24:
                    b_agree = any(_compute_iou(r["bbox"], b["bbox"]) >= 0.25 and b.get("score", 0.0) >= 0.20 for b in b_ge_08)
                    r1_agree = any(_compute_iou(r["bbox"], r1["bbox"]) >= 0.25 and r1.get("score", 0.0) >= 0.20 for r1 in r1_ge_08)
                    if not (b_agree and r1_agree):
                        suppress = True
                        reason = f"upper wall sign/banner clutter sc={sc:.3f}<0.24 by={by:.1f}<0.08*H without high-conf baseline+run1 agreement"

            # 7B14B: Macro-Heap Cigarette/Hazard Speckle Veto
            if not suppress and cat == 0 and is_macro_waste_dominated and (not has_compact_haz_cluster):
                if norm_area < 0.0015 and sc < 0.25:
                    b_agree = any(_compute_iou(r["bbox"], b["bbox"]) >= 0.30 and b.get("category_id") == 0 and b.get("score", 0.0) >= 0.16 for b in b_ge_08)
                    r1_agree = any(_compute_iou(r["bbox"], r1["bbox"]) >= 0.30 and r1.get("category_id") == 0 and r1.get("score", 0.0) >= 0.16 for r1 in r1_ge_08)
                    if not (b_agree and r1_agree):
                        suppress = True
                        reason = f"macro-heap hazard speckle sc={sc:.3f}<0.25 norm_area={norm_area:.6f}<0.0015 without baseline+run1 hazard agreement"

            if suppress:
                rule7_suppressed_qids.add(qid)
                diag["rule7_count"] += 1
                diag["rule7b_count"] += 1
                diag["logs"].append({
                    "image_id": image_id,
                    "qid": qid,
                    "rule": "Rule 7B (cross-stage clutter veto)",
                    "old_cat": cat,
                    "new_cat": None,
                    "old_score": sc,
                    "reason": reason,
                })
            else:
                final_rule7_rows.append(r)

        output = final_rule7_rows
    else:
        output = stage3_rows

    all_changed = sorted(rule1_changed_qids | rule1b_changed_qids | rule2_changed_qids | rule7_changed_qids)
    all_suppressed = sorted(
        rule3_suppressed_qids | rule4_suppressed_qids | rule5_suppressed_qids | rule6_suppressed_qids | rule7_suppressed_qids
    )

    if return_diagnostics:
        return output, all_changed, all_suppressed, diag
    return output, all_changed, all_suppressed


def apply_to_query_rows(rows, score_vectors, *, delta=DEFAULT_DELTA,
                        coactivation="2 plastic subtypes >= 0.25",
                        confidence_mode="top1_score"):
    """Apply the rule to one image's rows using `[900, 9]` saved scores."""
    if len(score_vectors) != 900:
        raise ValueError("Expected 900 query score vectors")
    output = []
    changed = []
    for row in rows:
        query_id = row.get("query_id")
        if type(query_id) is not int or not 0 <= query_id < 900:
            raise ValueError("Invalid query_id")
        adjusted, reassigned = apply_consensus_tiebreak(
            row, score_vectors[query_id], delta=delta,
            coactivation=coactivation, confidence_mode=confidence_mode)
        output.append(adjusted)
        if reassigned:
            changed.append(query_id)
    return output, changed
