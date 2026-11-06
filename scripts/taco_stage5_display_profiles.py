"""Display-only Stage 5 NMS and hazard-aware score profiles.

These controls operate on saved winning-class queries. They do not alter saved
predictions, logits, official COCO metrics, or model outputs.
"""
from __future__ import annotations

from .taco_diagnostic_nms import NMS_MODES, diagnostic_nms
from .taco_stage5_material_hierarchy import map_row, MAPS

NMS_CHOICES = (
    "raw",
    "class_aware_iou_0.50",
    "class_aware_iou_0.50_containment_0.80",
    "agnostic_iou_0.50",
    "agnostic_iou_0.50_containment_0.80",
)
THRESHOLD_CHOICES = (
    "Uniform Threshold",
    "Calibrated Profile (Cigarette 0.06 / Macro 0.15)",
)
_NMS_MAP = {
    "raw": NMS_MODES[0],
    "class_aware_iou_0.50": NMS_MODES[2],
    "class_aware_iou_0.50_containment_0.80": NMS_MODES[4],
    "agnostic_iou_0.50": NMS_MODES[1],
    "agnostic_iou_0.50_containment_0.80": NMS_MODES[3],
}


def score_floor(row: dict, slider: float, threshold_mode: str) -> float:
    if threshold_mode not in THRESHOLD_CHOICES:
        raise ValueError("Unknown score-threshold profile")
    if threshold_mode == THRESHOLD_CHOICES[0]:
        return slider
    specific = row.get("specific_category_id", row["category_id"])
    return 0.06 if specific == 0 else max(slider, 0.15)


def visible_rows(rows, scheme="9-specific", nms_mode="raw", threshold=0.10,
                 threshold_mode=THRESHOLD_CHOICES[0], hierarchical=False):
    """Map, apply per-query display floor, then diagnostic NMS in active taxonomy."""
    if nms_mode not in _NMS_MAP:
        raise ValueError("Unknown diagnostic NMS mode")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    mapped = []
    for raw in rows:
        if not raw.get("valid_clipped_box", True) or min(raw["bbox"][2:]) <= 0:
            continue
        row = map_row(raw, scheme, hierarchical)
        if row["score"] >= score_floor(row, threshold, threshold_mode):
            mapped.append(row)
    mapped.sort(key=lambda row: (-row["score"], row.get("query_id", 0)))
    return diagnostic_nms(mapped, _NMS_MAP[nms_mode])


def mapped_predictions(rows, scheme="9-specific"):
    """Taxonomy-remapped rows without a score filter, for strict audit matching."""
    return [map_row(row, scheme) for row in rows
            if row.get("valid_clipped_box", True) and min(row["bbox"][2:]) > 0]


PLASTIC_SPECIFIC_IDS = frozenset(i for i, mapped in enumerate(MAPS["5-material"])
                                 if mapped == MAPS["5-material"][1])
