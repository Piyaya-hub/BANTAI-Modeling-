"""Exact aliases and conservative review heuristics; no grouping or general NMS."""

import copy
import re

MAPPING_VERSION = "aliases-v1"
DUPLICATE_POLICY = "exact-raw-box-normalized-phrase-highest-score-v1"
ALIASES = {
    "garbage pile": "garbage_pile",
    "individual discarded item": "individual_litter",
    "discarded plastic bottle": "individual_litter",
    "discarded wrapper": "individual_litter",
    "discarded aluminum can": "individual_litter",
    "discarded bag": "individual_litter",
    "group of individual litter": "group_of_individual_litter",
    "group of scattered litter": "group_of_individual_litter",
}


def normalize_phrase(phrase):
    phrase = " ".join(phrase.lower().strip().rstrip(".!?,;:").split())
    return re.sub(r"^(?:a|an)\s+", "", phrase)


def map_phrase(phrase):
    category = ALIASES.get(normalize_phrase(phrase))
    return category, "exact_alias" if category else "no_exact_alias"


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection), intersection > 0


def apply_policies(detections, *, overlap_iou=0.8):
    detections = copy.deepcopy(detections)
    groups = {}
    for d in detections:
        if d["state"] == "invalid":
            continue
        key = (tuple(d["raw_box_xyxy"]), normalize_phrase(d["raw_phrase"]))
        groups.setdefault(key, []).append(d)
    for group in groups.values():
        # Stable extraction order breaks score ties; preserve every raw candidate.
        winner = max(group, key=lambda d: d["model_score"])
        for d in group:
            if d is not winner:
                d.update(state="duplicate", duplicate_of=winner["id"])
    retained = [d for d in detections if d["state"] == "retained"]
    flags = []
    for i, a in enumerate(retained):
        for b in retained[i + 1:]:
            iou, intersects = overlap(a["box_xyxy"], b["box_xyxy"])
            if not intersects:
                continue
            pair = [a["id"], b["id"]]
            if a["category"] is not None and a["category"] == b["category"] and iou >= overlap_iou:
                flags.append({"kind": "same_category_overlap", "candidates": pair, "iou": iou})
            if a["category"] != b["category"]:
                flags.append({"kind": "cross_category_overlap", "candidates": pair, "iou": iou})
            for obj, region in ((a, b), (b, a)):
                if obj["level"] != "object" or region["level"] != "region":
                    continue
                x0, y0, x1, y1 = obj["box_xyxy"]
                r0, s0, r1, s1 = region["box_xyxy"]
                if r0 <= x0 and s0 <= y0 and x1 <= r1 and y1 <= s1:
                    flags.append({"kind": "object_within_region_membership_unresolved", "candidates": [obj["id"], region["id"]]})
    return detections, flags
