"""Scale-normalized spatial grouping module for BANTAI trash detection.

Layers the three BANTAI spatial region categories from docs/SCOPE.md:
  - `individual_litter` -> "Individual Litter": An isolated discarded object not assigned to a group.
  - `group_of_individual_litter` -> "Group of Individual Litter": Multiple separately distinguishable
    discarded objects that are spatially associated without necessarily forming a dense pile.
  - `garbage_pile` -> "Garbage Pile": Concentrated accumulation of discarded waste, typically
    overlapping, stacked, or densely collected.

Uses scale-normalized pairwise proximity:
  g_ij <= max(alpha * max(s_i, s_j), beta * L)
where:
  - g_ij is the edge-to-edge gap between detection bounding boxes i and j.
  - s_i = sqrt(w_i * h_i) is the characteristic scale of detection i.
  - L = sqrt(W^2 + H^2) is the image diagonal.
  - alpha is scale_factor (default 2.4).
  - beta is diag_fraction (default 0.25).

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

DEFAULT_SCALE_FACTOR = 2.4
DEFAULT_DIAG_FRACTION = 0.25
DEFAULT_DENSITY_THRESHOLD = 0.35
DEFAULT_IOU_PILE_THRESHOLD = 0.15


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
) -> Dict[str, Any]:
    """Cluster visible detections into spatial regions and classify each region.

    Parameters:
        rows: List of visible detection dictionaries, each containing 'bbox' [x, y, w, h]
              and 'score'.
        image_width: Image pixel width (optional, used for diagonal calculation).
        image_height: Image pixel height (optional, used for diagonal calculation).
        scale_factor: Proximity scale factor multiplier alpha (default 2.4).
        diag_fraction: Fraction of image diagonal beta (default 0.25).
        density_threshold: Packing density threshold for garbage pile (default 0.35).
        iou_pile_threshold: Pairwise IoU threshold indicating overlapping pile (default 0.15).

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

    # Build adjacency graph
    adj = {i: [] for i in range(n_items)}
    pairwise_gaps = {}
    pairwise_ious = {}

    for i in range(n_items):
        for j in range(i + 1, n_items):
            gap = compute_edge_gap(boxes[i], boxes[j])
            pairwise_gaps[(i, j)] = gap
            pairwise_gaps[(j, i)] = gap

            iou_val = compute_box_iou(boxes[i], boxes[j])
            pairwise_ious[(i, j)] = iou_val
            pairwise_ious[(j, i)] = iou_val

            thresh = max(scale_factor * max(scales[i], scales[j]), diag_fraction * diag_l)
            if gap <= thresh:
                adj[i].append(j)
                adj[j].append(i)

    # Find connected components (BFS)
    visited = set()
    components = []
    for i in range(n_items):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(sorted(comp))

    # Sort components by minimum member index for determinism
    components.sort(key=lambda c: c[0])

    regions = []
    annotated_rows = [deepcopy(r) for r in rows]

    indiv_count = 0
    group_count = 0
    pile_count = 0

    for group_id, member_indices in enumerate(components):
        n_m = len(member_indices)
        m_boxes = [boxes[idx] for idx in member_indices]

        # Enclosing bounding box
        enc_x = min(b[0] for b in m_boxes)
        enc_y = min(b[1] for b in m_boxes)
        enc_max_x = max(b[0] + b[2] for b in m_boxes)
        enc_max_y = max(b[1] + b[3] for b in m_boxes)
        enc_w = enc_max_x - enc_x
        enc_h = enc_max_y - enc_y
        enc_area = enc_w * enc_h

        # Member area sum and packing density
        area_sum = sum(b[2] * b[3] for b in m_boxes)
        density = area_sum / max(1e-6, enc_area)

        # Touching pairs and max IoU within component
        touch_count = 0
        comp_max_iou = 0.0
        for ii_idx, i in enumerate(member_indices):
            for j in member_indices[ii_idx + 1:]:
                if pairwise_gaps.get((i, j), 1.0) == 0.0:
                    touch_count += 1
                iou_val = pairwise_ious.get((i, j), 0.0)
                if iou_val > comp_max_iou:
                    comp_max_iou = iou_val

        # Classification
        if n_m == 1:
            cat = "individual_litter"
            label = SPATIAL_CATEGORIES[cat]
            # Single item retains exact model score of underlying detection
            region_score = float(rows[member_indices[0]]["score"])
            indiv_count += 1
        else:
            # Derived multi-object region: check Garbage Pile criteria
            is_pile = (
                density >= density_threshold
                or comp_max_iou >= iou_pile_threshold
                or touch_count >= 1
            )
            if is_pile:
                cat = "garbage_pile"
                label = SPATIAL_CATEGORIES[cat]
                pile_count += 1
            else:
                cat = "group_of_individual_litter"
                label = SPATIAL_CATEGORIES[cat]
                group_count += 1
            # Never invent model confidence scores for multi-object derived groups
            region_score = None

        regions.append({
            "group_id": group_id,
            "category": cat,
            "label": label,
            "bbox": [enc_x, enc_y, enc_w, enc_h],
            "item_count": n_m,
            "model_score": region_score,
            "member_indices": member_indices,
            "packing_density": density,
            "touching_pairs": touch_count,
            "max_iou": comp_max_iou,
        })

        for idx in member_indices:
            annotated_rows[idx]["spatial_group_id"] = group_id
            annotated_rows[idx]["spatial_category"] = cat
            annotated_rows[idx]["spatial_label"] = label

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
