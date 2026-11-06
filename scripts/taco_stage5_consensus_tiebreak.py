"""CPU-only score-preserving paper-to-plastic display consensus rule."""
from __future__ import annotations

from copy import deepcopy
import math

from .taco_training_labels_v2 import LABELS

PLASTIC_SUBTYPE_IDS = (1, 2, 3, 4, 6, 7)
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
    enable_scene_paper_coherence=None,
    enable_interior_subbox_suppression=None,
    enable_marginal_ghost_suppression=None,
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
    """
    if enable_scene_paper_coherence is not None:
        enable_rule2 = bool(enable_scene_paper_coherence)
    if enable_interior_subbox_suppression is not None:
        enable_rule3 = bool(enable_interior_subbox_suppression)
    if enable_marginal_ghost_suppression is not None:
        enable_rule4 = bool(enable_marginal_ghost_suppression)

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

    # Step 2: Cautious Paper Anchor Identification
    paper_anchors = []
    for r in stage1_rows:
        qid = r.get("fused_query_id", r.get("query_id"))
        if qid is not None and 0 <= qid < len(score_vectors):
            v = score_vectors[qid]
            sc = r["score"]
            cat = r["category_id"]
            q_area = r["bbox"][2] * r["bbox"][3]
            norm_area = q_area / total_img_area

            if cat == PAPER_ID and sc >= 0.30 and norm_area >= 0.08:
                max_plastic = max(v[c] for c in PLASTIC_SUBTYPE_IDS)
                if (v[PAPER_ID] - max_plastic >= 0.10) and (v[1] < 0.15):
                    paper_anchors.append(r)

    diag["anchors_found"] = len(paper_anchors)

    # Step 3: Rule 2, Rule 3, and Rule 4
    output = []
    rule2_changed_qids = set()
    rule3_suppressed_qids = set()
    rule4_suppressed_qids = set()
    has_plastic_reassignment = bool(rule1_changed_qids or rule1b_changed_qids)

    for r in stage1_rows:
        qid = r.get("fused_query_id", r.get("query_id"))
        v = score_vectors[qid] if (qid is not None and 0 <= qid < len(score_vectors)) else None
        r_copy = deepcopy(r)
        cat = r_copy["category_id"]
        sc = r_copy["score"]
        q_area = r_copy["bbox"][2] * r_copy["bbox"][3]

        # Rule 2: Relabel flexible plastic to Paper
        if enable_rule2 and paper_anchors and v is not None and cat in FLEXIBLE_PLASTIC_IDS and 0.10 <= sc <= 0.30:
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

        # Rule 3: Interior wrong-class sub-box suppression
        suppress = False
        if enable_rule3 and paper_anchors and cat != PAPER_ID:
            if cat in (0, 2, 3, 4, 7):
                for p in paper_anchors:
                    if sc < p["score"]:
                        inter, a_q, a_p = _compute_inter_and_areas(r_copy["bbox"], p["bbox"])
                        iomin = inter / a_q
                        p_area = p["bbox"][2] * p["bbox"][3]
                        norm_p_area = p_area / total_img_area
                        cond_a = (sc < 0.20 and iomin >= 0.80 and a_q < 0.50 * a_p)
                        cond_b = (norm_p_area >= 0.25 and iomin >= 0.90 and a_q < 0.25 * a_p)
                        if cond_a or cond_b:
                            suppress = True
                            rule3_suppressed_qids.add(qid)
                            diag["rule3_count"] += 1
                            reason = (
                                f"deep-interior anchor containment iomin={iomin:.2f}>=0.90 area_ratio={a_q/a_p:.4f}<0.25 in anchor norm_area={norm_p_area:.3f}>=0.25"
                                if cond_b else
                                f"nested iomin={iomin:.2f}>=0.80 area_ratio={a_q/a_p:.2f}<0.50 inside anchor"
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
            norm_q_area = q_area / total_img_area
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

        if not suppress:
            output.append(r_copy)

    all_changed = sorted(rule1_changed_qids | rule1b_changed_qids | rule2_changed_qids)
    all_suppressed = sorted(rule3_suppressed_qids | rule4_suppressed_qids)

    if return_diagnostics:
        return output, all_changed, all_suppressed, diag
    return output, all_changed, all_suppressed

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
