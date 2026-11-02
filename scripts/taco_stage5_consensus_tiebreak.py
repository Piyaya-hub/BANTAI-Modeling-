"""CPU-only score-preserving paper-to-plastic display consensus rule."""
from __future__ import annotations

from copy import deepcopy
import math

from .taco_training_labels_v2 import LABELS

PLASTIC_SUBTYPE_IDS = (1, 2, 3, 4, 6, 7)
PAPER_ID = 8
DEFAULT_DELTA = 0.016
COACTIVATION_PROFILES = {
    "2 plastic subtypes >= 0.25": (2, 0.25),
    "3 plastic subtypes >= 0.20": (3, 0.20),
}
CONFIDENCE_MODES = ("top1_score", "plastic_max")


def apply_consensus_tiebreak(row, class_scores, *, delta=DEFAULT_DELTA,
                             coactivation="2 plastic subtypes >= 0.25",
                             confidence_mode="top1_score"):
    """Return a copied row and whether paper was reassigned to a plastic subtype.

    The top1_score mode preserves the original query score exactly. plastic_max
    is also score non-increasing. Boxes/query IDs are never altered.
    """
    if coactivation not in COACTIVATION_PROFILES:
        raise ValueError("Unknown plastic co-activation profile")
    if confidence_mode not in CONFIDENCE_MODES:
        raise ValueError("Unknown consensus confidence mode")
    if not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    if len(class_scores) != len(LABELS):
        raise ValueError("Expected nine class probabilities")
    scores = [float(value) for value in class_scores]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores):
        raise ValueError("Class probabilities must be finite and in [0, 1]")
    result = deepcopy(row)
    if row.get("category_id") != PAPER_ID or scores[PAPER_ID] < 0.15:
        return result, False
    plastic_max = max(PLASTIC_SUBTYPE_IDS, key=lambda cid: (scores[cid], -cid))
    if scores[PAPER_ID] - scores[plastic_max] > delta:
        return result, False
    min_count, floor = COACTIVATION_PROFILES[coactivation]
    if sum(scores[cid] >= floor for cid in PLASTIC_SUBTYPE_IDS) < min_count:
        return result, False

    result["raw_category_id"] = PAPER_ID
    result["category_id"] = plastic_max
    result["specific_category_id"] = plastic_max
    if confidence_mode == "plastic_max":
        result["score"] = scores[plastic_max]
    result["consensus_reassigned"] = True
    result["consensus_original_score"] = row.get("score")
    result["consensus_plastic_max_score"] = scores[plastic_max]
    return result, True


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


def apply_scene_consensus(rows, score_vectors, *, image_width=None, image_height=None,
                          delta=DEFAULT_DELTA,
                          coactivation="3 plastic subtypes >= 0.20",
                          confidence_mode="top1_score",
                          enable_scene_paper_coherence=True,
                          enable_interior_subbox_suppression=True):
    """
    Apply bidirectional scene-level consensus and interior wrong-class sub-box/speckle suppression.

    Rule 1: Existing paper -> plastic tie-break for genuine co-activated plastic objects.
    Rule 2: Bidirectional plastic -> paper scene-coherence tie-break on scenes with a dominant paper anchor.
    Rule 3: Interior wrong-class sub-box and micro-speckle suppression inside dominant paper regions.
    """
    total_img_area = None
    if image_width is not None and image_height is not None and image_width > 0 and image_height > 0:
        total_img_area = float(image_width * image_height)
    elif rows:
        max_x = max(r["bbox"][0] + r["bbox"][2] for r in rows)
        max_y = max(r["bbox"][1] + r["bbox"][3] for r in rows)
        total_img_area = max(1e-6, max_x * max_y)
    else:
        total_img_area = 1.0

    # Step 1: Run Rule 1 (Existing Paper -> Plastic tiebreak)
    stage1_rows = []
    rule1_changed = set()
    for row in rows:
        qid = row.get("fused_query_id", row.get("query_id"))
        if qid is not None and 0 <= qid < len(score_vectors):
            v = score_vectors[qid]
            adj, reassigned = apply_consensus_tiebreak(
                row, v, delta=delta, coactivation=coactivation, confidence_mode=confidence_mode
            )
            if reassigned:
                rule1_changed.add(qid)
            stage1_rows.append(adj)
        else:
            stage1_rows.append(deepcopy(row))

    # Step 2: Identify dominant Paper anchor queries
    paper_anchors = []
    if enable_scene_paper_coherence or enable_interior_subbox_suppression:
        for r in stage1_rows:
            qid = r.get("fused_query_id", r.get("query_id"))
            if qid is not None and 0 <= qid < len(score_vectors):
                v = score_vectors[qid]
                if r["category_id"] == PAPER_ID and r["score"] >= 0.25:
                    max_plastic = max(v[c] for c in PLASTIC_SUBTYPE_IDS)
                    if v[PAPER_ID] > max_plastic + 0.04:
                        paper_anchors.append(r)

    # Step 3: Run Rule 2 (Plastic -> Paper coherence) & Rule 3 (Interior suppression)
    output = []
    all_changed = list(rule1_changed)
    suppressed = []

    for r in stage1_rows:
        qid = r.get("fused_query_id", r.get("query_id"))
        v = score_vectors[qid] if (qid is not None and 0 <= qid < len(score_vectors)) else None
        r_copy = deepcopy(r)
        cat = r_copy["category_id"]
        sc = r_copy["score"]
        q_area = r_copy["bbox"][2] * r_copy["bbox"][3]
        norm_q_area = q_area / total_img_area

        if enable_scene_paper_coherence and paper_anchors and v is not None:
            if cat in FLEXIBLE_PLASTIC_IDS and 0.10 <= sc <= 0.30:
                if v[PAPER_ID] >= 0.05 or (v[PAPER_ID] / max(1e-6, sc)) >= 0.30:
                    r_copy["raw_category_id"] = r_copy.get("raw_category_id", cat)
                    r_copy["category_id"] = PAPER_ID
                    r_copy["specific_category_id"] = PAPER_ID
                    r_copy["consensus_reassigned"] = True
                    r_copy["consensus_scene_coherence"] = True
                    cat = PAPER_ID
                    all_changed.append(qid)

        suppress = False
        if enable_interior_subbox_suppression and paper_anchors and cat != PAPER_ID:
            for p in paper_anchors:
                p_area = p["bbox"][2] * p["bbox"][3]
                norm_p_area = p_area / total_img_area
                inter, a_q, a_p = _compute_inter_and_areas(r_copy["bbox"], p["bbox"])
                if norm_p_area >= 0.08 and (sc < p["score"] or (norm_p_area >= 0.25 and a_q < 0.20 * a_p and sc < p["score"] + 0.02)):
                    iomin = inter / a_q
                    if iomin >= 0.80 and a_q < 0.50 * a_p:
                        suppress = True
                        suppressed.append(qid)
                        break
                    if cat == 0 and norm_q_area < 0.0015 and sc < 0.26 and norm_p_area >= 0.25 and (v is None or v[0] < 0.26):
                        suppress = True
                        suppressed.append(qid)
                        break

        if not suppress:
            output.append(r_copy)

    return output, sorted(set(all_changed)), sorted(set(suppressed))


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
