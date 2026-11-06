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
    rows: List[Dict[str, Any]],
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
    *,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    diag_fraction: float = DEFAULT_DIAG_FRACTION,
    density_threshold: float = DEFAULT_DENSITY_THRESHOLD,
    iou_pile_threshold: float = DEFAULT_IOU_PILE_THRESHOLD,
    max_cluster_diag_fraction: float = DEFAULT_MAX_CLUSTER_DIAG_FRACTION,
) -> Dict[str, Any]:
    """Cluster visible detections into spatial regions using Two-Tier Hierarchical DBSCAN.

    Parameters:
        rows: List of visible detection dictionaries, each containing 'bbox' [x, y, w, h]
              and 'score'.
        image_width: Image pixel width (optional, used for diagonal calculation).
        image_height: Image pixel height (optional, used for diagonal calculation).
        scale_factor: Proximity scale factor multiplier alpha (default 2.0).
        diag_fraction: Fraction of image diagonal beta (default 0.225).
        density_threshold: Packing density threshold for garbage pile (default 0.45).
        iou_pile_threshold: Pairwise IoU threshold indicating overlapping pile (default 0.15).
        max_cluster_diag_fraction: Bounding box diagonal ceiling for Tier 2 clusters (default 0.42).

    Returns:
        A dictionary with:
          - 'annotated_rows': Copy of rows with 'spatial_group_id', 'spatial_category',
            'spatial_label'.
          - 'regions': List of spatial region dictionaries.
          - 'summary': Summary counts of physical items and regional categories.
    """
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

    # Determine image diagonal L
    if image_width is not None and image_height is not None and image_width > 0 and image_height > 0:
        diag_l = math.hypot(image_width, image_height)
    else:
        # Fallback to bounding box span if image dimensions not provided
        max_x = max(r["bbox"][0] + r["bbox"][2] for r in rows)
        max_y = max(r["bbox"][1] + r["bbox"][3] for r in rows)
        diag_l = math.hypot(max(100.0, max_x), max(100.0, max_y))

    n_items = len(rows)
    boxes = [r["bbox"] for r in rows]
    scales = [math.sqrt(max(1.0, b[2] * b[3])) for b in boxes]

    # Precalculate pairwise gaps and IoUs
    pairwise_gaps: Dict[Tuple[int, int], float] = {}
    pairwise_ious: Dict[Tuple[int, int], float] = {}
    for i in range(n_items):
        for j in range(i + 1, n_items):
            gap = compute_edge_gap(boxes[i], boxes[j])
            pairwise_gaps[(i, j)] = gap
            pairwise_gaps[(j, i)] = gap

            iou_val = compute_box_iou(boxes[i], boxes[j])
            pairwise_ious[(i, j)] = iou_val
            pairwise_ious[(j, i)] = iou_val

    # =========================================================================
    # Tier 1: Dense Garbage Pile Core Extraction
    # =========================================================================
    tight_adj: Dict[int, List[int]] = {i: [] for i in range(n_items)}
    for i in range(n_items):
        for j in range(i + 1, n_items):
            g = pairwise_gaps[(i, j)]
            u = pairwise_ious[(i, j)]
            s_min = min(scales[i], scales[j])
            is_tight = (g == 0.0 or u > 0.01)
            if not is_tight and g <= 0.35 * s_min:
                i_nbr = any(
                    k != j and pairwise_gaps[(i, k)] <= 0.80 * max(scales[i], scales[k])
                    for k in range(n_items) if k != i
                )
                j_nbr = any(
                    k != i and pairwise_gaps[(j, k)] <= 0.80 * max(scales[j], scales[k])
                    for k in range(n_items) if k != j
                )
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
            queue = [i]
            visited_t1.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in tight_adj[curr]:
                    if neighbor not in visited_t1:
                        visited_t1.add(neighbor)
                        queue.append(neighbor)
            tight_comps.append(sorted(comp))

    tight_comps.sort(key=lambda c: c[0])

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

    # =========================================================================
    # Tier 2: Remaining Non-Pile Items (Local Proximity & Anti-Chaining)
    # =========================================================================
    rem_indices = [i for i in range(n_items) if i not in assigned_indices]
    t2_adj: Dict[int, List[int]] = {i: [] for i in rem_indices}
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
            queue = [i]
            visited_t2.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in t2_adj[curr]:
                    if neighbor not in visited_t2:
                        test_boxes = [boxes[idx] for idx in comp + [neighbor]]
                        min_x = min(b[0] for b in test_boxes)
                        min_y = min(b[1] for b in test_boxes)
                        mw = max(b[0] + b[2] for b in test_boxes) - min_x
                        mh = max(b[1] + b[3] for b in test_boxes) - min_y
                        if math.hypot(mw, mh) <= max_cluster_diag_fraction * diag_l:
                            visited_t2.add(neighbor)
                            queue.append(neighbor)
            t2_comps.append(sorted(comp))

    t2_comps.sort(key=lambda c: c[0])

    # =========================================================================
    # Build Unified Regions & Annotated Rows
    # =========================================================================
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

    for comp in t2_comps:
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
