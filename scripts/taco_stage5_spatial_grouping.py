"""Scale-normalized hierarchical spatial grouping module for BANTAI trash detection.

Layers the three BANTAI spatial region categories from docs/SCOPE.md:
  - `individual_litter` -> "Individual Litter": An isolated discarded object not assigned to a group.
  - `group_of_individual_litter` -> "Group of Individual Litter": Multiple separately distinguishable
    discarded objects that are spatially associated without necessarily forming a dense pile.
  - `garbage_pile` -> "Garbage Pile": Concentrated accumulation of discarded waste, typically
    overlapping, stacked, or densely collected.

Implements a Two-Tier Hierarchical DBSCAN Spatial Grouping Architecture:
  - Tier 1: Dense Garbage Pile Core Extraction
    Identifies compact, high-density overlapping piles using tight heap adjacency:
      g_ij == 0.0 or IoU_ij > 0.01 or (g_ij <= 0.35 * min(s_i, s_j) and bilateral neighborhood support)
    Clusters qualify as Garbage Pile if:
      (k >= 3 and (touching_pairs >= 1 or density >= 0.40)) or
      (k == 2 and density >= density_threshold and max_iou >= iou_pile_threshold)
  - Tier 2: Local Proximity Grouping with Anti-Chaining Constraint
    Clusters remaining non-pile items using scale-normalized proximity:
      g_ij <= max(scale_factor * max(s_i, s_j), diag_fraction * L)
    bounded by an anti-chaining cluster diagonal threshold:
      cluster_diag <= max_cluster_diag_fraction * L (default 0.42 * L)

Invariants:
  - Preserves underlying supervised object predictions (category_id, score, bbox).
  - Never invents fake model confidence scores for derived multi-object groups (model_score = None for n >= 2).
  - Individual litter retains the exact model score of its underlying detection.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, List, Optional, Tuple

SPATIAL_CATEGORIES = {
    "individual_litter": "Individual Litter",
    "group_of_individual_litter": "Group of Individual Litter",
    "garbage_pile": "Garbage Pile",
}

DEFAULT_SCALE_FACTOR = 2.0
DEFAULT_DIAG_FRACTION = 0.225
DEFAULT_DENSITY_THRESHOLD = 0.45
DEFAULT_IOU_PILE_THRESHOLD = 0.15
DEFAULT_MAX_CLUSTER_DIAG_FRACTION = 0.42


def compute_edge_gap(box_a: List[float], box_b: List[float]) -> float:
    """Compute the edge-to-edge Euclidean gap between two [x, y, w, h] boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    gx = max(0.0, max(ax, bx) - min(ax + aw, bx + bw))
    gy = max(0.0, max(ay, by) - min(ay + ah, by + bh))
    return math.hypot(gx, gy)


def compute_box_intersection_and_areas(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """Computes (intersection_area, area_a, area_b) for two xywh boxes."""
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    if wa <= 0 or ha <= 0 or wb <= 0 or hb <= 0:
        return 0.0, max(0.0, wa * ha), max(0.0, wb * hb)
    ix = max(0.0, min(xa + wa, xb + wb) - max(xa, xb))
    iy = max(0.0, min(ya + ha, yb + hb) - max(ya, yb))
    return ix * iy, wa * ha, wb * hb


def compute_box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute intersection-over-union between two [x, y, w, h] boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0.0 else 0.0


def compute_spatial_grouping(
    rows,
    image_width: float | None = None,
    image_height: float | None = None,
    scale_factor: float = 2.0,
    diag_fraction: float = 0.225,
    density_threshold: float = 0.45,
    iou_pile_threshold: float = 0.15,
    max_cluster_diag_fraction: float = 0.42,
):
    if not rows:
        return {
            "annotated_rows": [],
            "regions": [],
            "summary": {
                "total_physical_items": 0,
                "total_regions": 0,
                "individual_litter_count": 0,
                "group_of_individual_litter_count": 0,
                "garbage_pile_count": 0,
            },
        }

    if image_width and image_height and image_width > 0 and image_height > 0:
        diag_l = math.hypot(image_width, image_height)
    else:
        max_x = max(r["bbox"][0] + r["bbox"][2] for r in rows)
        max_y = max(r["bbox"][1] + r["bbox"][3] for r in rows)
        diag_l = math.hypot(max(100.0, max_x), max(100.0, max_y))

    n_items = len(rows)
    boxes = [r["bbox"] for r in rows]
    scales = [math.sqrt(max(1.0, b[2] * b[3])) for b in boxes]

    pairwise_gaps = {}
    pairwise_ious = {}
    for i in range(n_items):
        for j in range(i + 1, n_items):
            g = compute_edge_gap(boxes[i], boxes[j])
            pairwise_gaps[(i, j)] = g
            pairwise_gaps[(j, i)] = g
            u = compute_box_iou(boxes[i], boxes[j])
            pairwise_ious[(i, j)] = u
            pairwise_ious[(j, i)] = u

    # Tier 1: Dense core extraction
    tight_adj = {i: [] for i in range(n_items)}
    for i in range(n_items):
        for j in range(i + 1, n_items):
            g = pairwise_gaps[(i, j)]
            u = pairwise_ious[(i, j)]
            s_min = min(scales[i], scales[j])
            is_tight = (g == 0.0 or u > 0.01)
            if not is_tight and g <= 0.35 * s_min:
                i_nbr = any(k != j and pairwise_gaps[(i, k)] <= 0.80 * max(scales[i], scales[k]) for k in range(n_items) if k != i)
                j_nbr = any(k != i and pairwise_gaps[(j, k)] <= 0.80 * max(scales[j], scales[k]) for k in range(n_items) if k != j)
                if i_nbr and j_nbr:
                    is_tight = True
            if is_tight:
                tight_adj[i].append(j)
                tight_adj[j].append(i)

    visited_t1 = set()
    tight_comps = []
    for i in range(n_items):
        if i not in visited_t1:
            comp = []
            q = [i]
            visited_t1.add(i)
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for nbr in tight_adj[curr]:
                    if nbr not in visited_t1:
                        visited_t1.add(nbr)
                        q.append(nbr)
            tight_comps.append(sorted(comp))

    pile_comps = []
    assigned_indices = set()
    for comp in tight_comps:
        k = len(comp)
        if k < 2:
            continue
        m_boxes = [boxes[idx] for idx in comp]
        enc_x = min(b[0] for b in m_boxes)
        enc_y = min(b[1] for b in m_boxes)
        enc_w = max(b[0] + b[2] for b in m_boxes) - enc_x
        enc_h = max(b[1] + b[3] for b in m_boxes) - enc_y
        area_sum = sum(b[2] * b[3] for b in m_boxes)
        density = area_sum / max(1e-6, enc_w * enc_h)
        touch_count = sum(1 for ii in range(k) for jj in range(ii + 1, k) if pairwise_gaps[(comp[ii], comp[jj])] == 0.0)
        max_u = max((pairwise_ious[(comp[ii], comp[jj])] for ii in range(k) for jj in range(ii + 1, k)), default=0.0)

        is_pile = (
            (k >= 3 and (touch_count >= 1 or density >= 0.40))
            or (k == 2 and density >= density_threshold and max_u >= iou_pile_threshold)
        )
        if is_pile:
            pile_comps.append({
                "members": comp,
                "category": "garbage_pile",
                "label": SPATIAL_CATEGORIES["garbage_pile"],
                "bbox": [enc_x, enc_y, enc_w, enc_h],
                "packing_density": density,
                "touching_pairs": touch_count,
                "max_iou": max_u,
            })
            assigned_indices.update(comp)

    # Iterative consolidation of Step 1A & 1B
    changed_consolidation = True
    while changed_consolidation:
        changed_consolidation = False

        # Step 1A: Merge adjacent / overlapping pile cores
        changed_merge = True
        while changed_merge:
            changed_merge = False
            for p1_i in range(len(pile_comps)):
                for p2_i in range(p1_i + 1, len(pile_comps)):
                    p1 = pile_comps[p1_i]
                    p2 = pile_comps[p2_i]
                    b1 = p1["bbox"]
                    b2 = p2["bbox"]
                    u = compute_box_iou(b1, b2)
                    inter, a1, a2 = compute_box_intersection_and_areas(b1, b2)
                    iomin = inter / min(a1, a2) if min(a1, a2) > 0 else 0.0
                    min_gap = min(pairwise_gaps[(m1, m2)] for m1 in p1["members"] for m2 in p2["members"])
                    max_s = max(max(scales[m] for m in p1["members"]), max(scales[m] for m in p2["members"]))

                    if u > 0.02 or iomin >= 0.15 or min_gap <= max(0.85 * max_s, 0.045 * diag_l):
                        merged_members = sorted(list(set(p1["members"] + p2["members"])))
                        m_boxes = [boxes[idx] for idx in merged_members]
                        enc_x = min(b[0] for b in m_boxes)
                        enc_y = min(b[1] for b in m_boxes)
                        enc_w = max(b[0] + b[2] for b in m_boxes) - enc_x
                        enc_h = max(b[1] + b[3] for b in m_boxes) - enc_y
                        area_sum = sum(b[2] * b[3] for b in m_boxes)
                        p1["members"] = merged_members
                        p1["bbox"] = [enc_x, enc_y, enc_w, enc_h]
                        p1["packing_density"] = area_sum / max(1e-6, enc_w * enc_h)
                        k = len(merged_members)
                        p1["touching_pairs"] = sum(1 for ii in range(k) for jj in range(ii + 1, k) if pairwise_gaps[(merged_members[ii], merged_members[jj])] == 0.0)
                        p1["max_iou"] = max((pairwise_ious[(merged_members[ii], merged_members[jj])] for ii in range(k) for jj in range(ii + 1, k)), default=0.0)
                        pile_comps.pop(p2_i)
                        changed_merge = True
                        changed_consolidation = True
                        break
                if changed_merge:
                    break

        # Step 1B: Absorb pile-enclosed & pile-bordering items
        unassigned = [i for i in range(n_items) if i not in assigned_indices]
        for i in unassigned:
            b_i = boxes[i]
            s_i = scales[i]
            absorbed_into = None
            for p in pile_comps:
                b_p = p["bbox"]
                inter, a_i, a_p = compute_box_intersection_and_areas(b_i, b_p)
                enclosed = (a_i > 0 and (inter / a_i) >= 0.25)
                bordering = any(pairwise_gaps[(i, m)] <= max(1.15 * max(s_i, scales[m]), 0.05 * diag_l) for m in p["members"])
                if enclosed or bordering:
                    absorbed_into = p
                    break
            if absorbed_into is not None:
                absorbed_into["members"] = sorted(absorbed_into["members"] + [i])
                assigned_indices.add(i)
                m_boxes = [boxes[idx] for idx in absorbed_into["members"]]
                enc_x = min(b[0] for b in m_boxes)
                enc_y = min(b[1] for b in m_boxes)
                enc_w = max(b[0] + b[2] for b in m_boxes) - enc_x
                enc_h = max(b[1] + b[3] for b in m_boxes) - enc_y
                area_sum = sum(b[2] * b[3] for b in m_boxes)
                absorbed_into["bbox"] = [enc_x, enc_y, enc_w, enc_h]
                absorbed_into["packing_density"] = area_sum / max(1e-6, enc_w * enc_h)
                k = len(absorbed_into["members"])
                absorbed_into["touching_pairs"] = sum(1 for ii in range(k) for jj in range(ii + 1, k) if pairwise_gaps[(absorbed_into["members"][ii], absorbed_into["members"][jj])] == 0.0)
                absorbed_into["max_iou"] = max((pairwise_ious[(absorbed_into["members"][ii], absorbed_into["members"][jj])] for ii in range(k) for jj in range(ii + 1, k)), default=0.0)
                changed_consolidation = True

    # Tier 2: Non-pile items
    rem_indices = [i for i in range(n_items) if i not in assigned_indices]
    t2_adj = {i: [] for i in rem_indices}
    for i_idx, i in enumerate(rem_indices):
        for j in rem_indices[i_idx + 1:]:
            g = pairwise_gaps[(i, j)]
            thresh = max(scale_factor * max(scales[i], scales[j]), diag_fraction * diag_l)
            if g <= thresh:
                t2_adj[i].append(j)
                t2_adj[j].append(i)

    visited_t2 = set()
    t2_comps = []
    for i in rem_indices:
        if i not in visited_t2:
            comp = []
            q = [i]
            visited_t2.add(i)
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for nbr in t2_adj[curr]:
                    if nbr not in visited_t2:
                        test_boxes = [boxes[idx] for idx in comp + [nbr]]
                        min_x = min(b[0] for b in test_boxes)
                        min_y = min(b[1] for b in test_boxes)
                        mw = max(b[0] + b[2] for b in test_boxes) - min_x
                        mh = max(b[1] + b[3] for b in test_boxes) - min_y
                        if math.hypot(mw, mh) > max_cluster_diag_fraction * diag_l:
                            continue
                        c_box = [min_x, min_y, mw, mh]
                        overlaps_pile = any(
                            compute_box_intersection_and_areas(c_box, p["bbox"])[0] / p["bbox"][2] / p["bbox"][3] >= 0.20
                            for p in pile_comps if p["bbox"][2] * p["bbox"][3] > 0
                        )
                        if not overlaps_pile:
                            visited_t2.add(nbr)
                            q.append(nbr)
            t2_comps.append(sorted(comp))

    t2_comps.sort(key=lambda c: c[0])

    # Step 1C: Absorb any Tier 2 cluster overlapping a pile by >= 0.20 of either area
    final_t2_comps = []
    for comp in t2_comps:
        m_boxes = [boxes[idx] for idx in comp]
        min_x = min(b[0] for b in m_boxes)
        min_y = min(b[1] for b in m_boxes)
        mw = max(b[0] + b[2] for b in m_boxes) - min_x
        mh = max(b[1] + b[3] for b in m_boxes) - min_y
        c_box = [min_x, min_y, mw, mh]

        merged_pile = None
        for p in pile_comps:
            inter, a_c, a_p = compute_box_intersection_and_areas(c_box, p["bbox"])
            if (a_c > 0 and inter / a_c >= 0.20) or (a_p > 0 and inter / a_p >= 0.20):
                merged_pile = p
                break

        if merged_pile is not None:
            merged_pile["members"] = sorted(list(set(merged_pile["members"] + comp)))
            all_m_boxes = [boxes[idx] for idx in merged_pile["members"]]
            enc_x = min(b[0] for b in all_m_boxes)
            enc_y = min(b[1] for b in all_m_boxes)
            enc_w = max(b[0] + b[2] for b in all_m_boxes) - enc_x
            enc_h = max(b[1] + b[3] for b in all_m_boxes) - enc_y
            area_sum = sum(b[2] * b[3] for b in all_m_boxes)
            merged_pile["bbox"] = [enc_x, enc_y, enc_w, enc_h]
            merged_pile["packing_density"] = area_sum / max(1e-6, enc_w * enc_h)
            k = len(merged_pile["members"])
            merged_pile["touching_pairs"] = sum(1 for ii in range(k) for jj in range(ii + 1, k) if pairwise_gaps[(merged_pile["members"][ii], merged_pile["members"][jj])] == 0.0)
            merged_pile["max_iou"] = max((pairwise_ious[(merged_pile["members"][ii], merged_pile["members"][jj])] for ii in range(k) for jj in range(ii + 1, k)), default=0.0)
        else:
            final_t2_comps.append(comp)

    # Also merge any two Tier 2 clusters that overlap each other by >= 0.20
    changed_t2_merge = True
    while changed_t2_merge:
        changed_t2_merge = False
        for c1_i in range(len(final_t2_comps)):
            for c2_i in range(c1_i + 1, len(final_t2_comps)):
                comp1 = final_t2_comps[c1_i]
                comp2 = final_t2_comps[c2_i]
                b1 = [min(boxes[idx][0] for idx in comp1), min(boxes[idx][1] for idx in comp1),
                      max(boxes[idx][0] + boxes[idx][2] for idx in comp1) - min(boxes[idx][0] for idx in comp1),
                      max(boxes[idx][1] + boxes[idx][3] for idx in comp1) - min(boxes[idx][1] for idx in comp1)]
                b2 = [min(boxes[idx][0] for idx in comp2), min(boxes[idx][1] for idx in comp2),
                      max(boxes[idx][0] + boxes[idx][2] for idx in comp2) - min(boxes[idx][0] for idx in comp2),
                      max(boxes[idx][1] + boxes[idx][3] for idx in comp2) - min(boxes[idx][1] for idx in comp2)]
                inter, a1, a2 = compute_box_intersection_and_areas(b1, b2)
                if (a1 > 0 and inter / a1 >= 0.20) or (a2 > 0 and inter / a2 >= 0.20):
                    final_t2_comps[c1_i] = sorted(list(set(comp1 + comp2)))
                    final_t2_comps.pop(c2_i)
                    changed_t2_merge = True
                    break
            if changed_t2_merge:
                break

    # Build regions and annotated rows
    regions = []
    annotated_rows = [deepcopy(r) for r in rows]
    reg_id = 0
    pile_count = len(pile_comps)
    group_count = 0
    indiv_count = 0

    for p in pile_comps:
        comp = p["members"]
        p["group_id"] = reg_id
        p["item_count"] = len(comp)
        p["model_score"] = None
        p["member_indices"] = comp
        regions.append(p)
        for idx in comp:
            annotated_rows[idx]["spatial_group_id"] = reg_id
            annotated_rows[idx]["spatial_category"] = "garbage_pile"
            annotated_rows[idx]["spatial_label"] = SPATIAL_CATEGORIES["garbage_pile"]
        reg_id += 1

    for comp in final_t2_comps:
        k = len(comp)
        m_boxes = [boxes[idx] for idx in comp]
        enc_x = min(b[0] for b in m_boxes)
        enc_y = min(b[1] for b in m_boxes)
        enc_w = max(b[0] + b[2] for b in m_boxes) - enc_x
        enc_h = max(b[1] + b[3] for b in m_boxes) - enc_y
        area_sum = sum(b[2] * b[3] for b in m_boxes)
        density = area_sum / max(1e-6, enc_w * enc_h)
        touch_count = sum(1 for ii in range(k) for jj in range(ii + 1, k) if pairwise_gaps[(comp[ii], comp[jj])] == 0.0)
        max_u = max((pairwise_ious[(comp[ii], comp[jj])] for ii in range(k) for jj in range(ii + 1, k)), default=0.0)

        if k == 1:
            cat = "individual_litter"
            label = SPATIAL_CATEGORIES[cat]
            sc_val = float(rows[comp[0]]["score"])
            indiv_count += 1
        else:
            is_pile = (
                (density >= density_threshold and (max_u >= iou_pile_threshold or touch_count >= 1))
                or (k >= 3 and density >= 0.55)
            )
            if is_pile:
                cat = "garbage_pile"
                label = SPATIAL_CATEGORIES[cat]
                sc_val = None
                pile_count += 1
            else:
                cat = "group_of_individual_litter"
                label = SPATIAL_CATEGORIES[cat]
                sc_val = None
                group_count += 1

        regions.append({
            "group_id": reg_id,
            "category": cat,
            "label": label,
            "bbox": [enc_x, enc_y, enc_w, enc_h],
            "item_count": k,
            "model_score": sc_val,
            "member_indices": comp,
            "members": comp,
            "packing_density": density,
            "touching_pairs": touch_count,
            "max_iou": max_u,
        })
        for idx in comp:
            annotated_rows[idx]["spatial_group_id"] = reg_id
            annotated_rows[idx]["spatial_category"] = cat
            annotated_rows[idx]["spatial_label"] = label
        reg_id += 1

    summary = {
        "total_physical_items": n_items,
        "total_regions": len(regions),
        "individual_litter_count": indiv_count,
        "group_of_individual_litter_count": group_count,
        "garbage_pile_count": pile_count,
    }

    return {
        "annotated_rows": annotated_rows,
        "regions": regions,
        "summary": summary,
    }
