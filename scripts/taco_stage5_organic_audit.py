"""D-080 read-only Food waste eligibility audit; no held-out photos/test split."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import uuid

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.taco_dataset import bbox_issues, image_path, sha256
from scripts.taco_splits import Groups, duplicate_groups, fingerprint

ACCESSORIES = frozenset(("Plastic bottle cap", "Metal bottle cap", "Plastic lid",
                         "Metal lid", "Plastic straw", "Plastic utensils", "Pop tab",
                         "Aluminium foil", "Glass bottle"))
SIBLINGS = {"Polypropylene bag": 2, "Six pack rings": 3, "Food Can": 5,
            "Aerosol": 5, "Disposable food container": 6, "Spread tub": 6,
            "Other plastic container": 6, "Foam cup": 7, "Foam food container": 7,
            "Plastified paper bag": 8}


def mapping_for(categories, tier):
    """Map the unchanged D-073 base taxonomy and add Food waste as class 9."""
    names = {
        "Cigarette": 0,
        "Clear plastic bottle": 1, "Other plastic bottle": 1,
        "Single-use carrier bag": 2, "Garbage bag": 2,
        "Other plastic wrapper": 3, "Crisp packet": 3,
        "Plastic film": 4, "Drink can": 5, "Disposable plastic cup": 6,
        "Styrofoam piece": 7,
        **{name: 8 for name in ("Normal paper", "Magazine paper", "Wrapping paper", "Tissues",
                                  "Paper bag", "Paper cup", "Corrugated carton", "Other carton",
                                  "Drink carton", "Meal carton", "Egg carton", "Pizza box",
                                  "Paper straw", "Toilet tube")},
        "Food waste": 9,
    }
    if tier in ("tier2", "tier3"):
        names.update(SIBLINGS)
    return names


def disposition(image, annotations, categories, tier, duplicate_component=()):
    """Return exact first-failure strict-clean rule for one image under a tier."""
    names = mapping_for(categories, tier)
    usable = []
    invalid = []
    for ann in annotations:
        issues = bbox_issues(ann.get("bbox"), image.get("actual_width"), image.get("actual_height"))
        area = ann.get("area")
        if not isinstance(area, (int, float)) or isinstance(area, bool) or area <= 0:
            issues.append("invalid_source_area")
        crowd = ann.get("iscrowd")
        if type(crowd) is not int or crowd not in (0, 1):
            issues.append("invalid_source_iscrowd")
        if issues:
            invalid.append(ann["id"])
        else:
            usable.append(ann)
    all_names = [categories.get(a.get("category_id"), "<unknown category>") for a in annotations]
    reasons = []
    if image.get("status") in ("missing", "unreadable"):
        reasons.append("missing_or_unreadable")
    if image.get("status") == "invalid" or "invalid_declared_dimensions" in image.get("issues", []):
        reasons.append("invalid_image_metadata")
    if "declared_actual_dimension_mismatch" in image.get("issues", []):
        reasons.append("dimension_mismatch")
    if image.get("status") == "readable" and image.get("exif_orientation") != 1:
        reasons.append("ambiguous_exif_orientation")
    if invalid:
        reasons.append("invalid_annotation")
    if any(a.get("iscrowd") == 1 or a.get("ignore", 0) != 0 for a in annotations):
        reasons.append("crowd_or_ignore_annotation")
    if "Unlabeled litter" in all_names:
        reasons.append("unlabeled_annotation")
    omitted = []
    excluded = []
    for ann, name in zip(annotations, all_names):
        safe_accessory = (tier == "tier3" and name in ACCESSORIES and ann not in [a for a in annotations if a["id"] in invalid]
                          and ann.get("iscrowd") == 0 and ann.get("ignore", 0) == 0)
        if safe_accessory:
            omitted.append(ann["id"])
        elif name not in names:
            excluded.append(ann["id"])
    if excluded:
        reasons.append("excluded_annotation")
    if not any(categories.get(a.get("category_id")) in names for a in annotations):
        reasons.append("no_mapped_annotations")
    reasons = list(dict.fromkeys(reasons))
    first = reasons[0] if reasons else None
    return {"eligible_before_duplicate_rules": not reasons,
            "primary_reason": first, "reasons": reasons,
            "invalid_annotation_ids": invalid,
            "unlabeled_annotation_ids": [a["id"] for a, n in zip(annotations, all_names) if n == "Unlabeled litter"],
            "excluded_annotation_ids": excluded, "omitted_accessory_annotation_ids": omitted,
            "mapped_food_waste_annotation_ids": [a["id"] for a in annotations
                                                    if categories.get(a.get("category_id")) == "Food waste"],
            "duplicate_component": list(duplicate_component)}


def resized_box(box, width, height, shortest=384, longest=576):
    scale = min(shortest / min(width, height), longest / max(width, height))
    x, y, w, h = box
    return {"scale": scale, "xywh": [x * scale, y * scale, w * scale, h * scale],
            "width_px": w * scale, "height_px": h * scale}


def main():
    from scripts.configuration import strict_load

    stage = ROOT / "exports/taco/strict-clean-stage5-v1"
    prior = ROOT / "runs/training/stage5-data-expansion-audit-5412a13c0258467ba33a1f475cc4b65c/summary.json"
    review = ROOT / "reports/taco/local-review/images.jsonl"
    prep_manifest = strict_load(stage / "preparation_manifest.json")
    source_path = Path(prep_manifest["source"]["annotations"])
    dataset_root = Path(prep_manifest["source"]["dataset_root"])
    prior_path = prior
    cached_rows = [json.loads(line) for line in review.read_text(encoding="utf-8").splitlines()]
    cache_by_id = {row["original"]["id"]: row for row in cached_rows}
    frozen = {split: {row["source_image_id"] for row in
                      (json.loads(line) for line in (stage / "image_eligibility.jsonl").read_text(encoding="utf-8").splitlines())
                      if row.get("eligible") and row.get("split") == split}
              for split in ("validation", "test")}
    locked = frozen["validation"] | frozen["test"]
    assert len(frozen["validation"]) == len(frozen["test"]) == 65
    # Cached metadata is sufficient to block held-out paths before reading source annotations.
    heldout_paths = {Path(cache_by_id[i]["path"]).resolve() for i in locked}
    out = ROOT / "runs/training" / ("stage5-organic-audit-" + uuid.uuid4().hex)
    out.mkdir()

    def guard(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.name.lower() == "test.json" or path in heldout_paths:
                raise AssertionError("D-080 blocked a test JSON or frozen validation/test photo read")
            flags = args[2] if len(args) > 2 else 0
            if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                if not path.is_relative_to(out):
                    raise AssertionError("D-080 write outside new audit folder: " + str(path))
    sys.addaudithook(guard)

    raw = strict_load(source_path)
    prior_data = strict_load(prior_path)
    categories = {row["id"]: row["name"] for row in raw["categories"]}
    images = {row["id"]: row for row in raw["images"]}
    by_image = defaultdict(list)
    for ann in raw["annotations"]:
        by_image[ann.get("image_id")].append(ann)
    food_rows = [ann for ann in raw["annotations"] if categories.get(ann.get("category_id")) == "Food waste"]
    food_images = sorted({ann["image_id"] for ann in food_rows})
    assert len(raw["images"]) == 1500 and len(food_rows) == 8 and len(food_images) == 7
    assert not set(food_images) & locked, "Food waste image intersects frozen split; never open its photo"

    # Freshly inspect only the seven non-held-out Food waste rasters; no other photo is opened.
    physical = {}
    for image_id in food_images:
        original = images[image_id]
        path = image_path(dataset_root, original["file_name"])
        payload = path.read_bytes()
        file_hash = hashlib.sha256(payload).hexdigest()
        with Image.open(io.BytesIO(payload)) as decoded:
            decoded.load()
            orientation = decoded.getexif().get(274, 1)
            actual = (decoded.width, decoded.height)
        cached = cache_by_id[image_id]
        issues = []
        if not isinstance(original.get("width"), int) or not isinstance(original.get("height"), int) or min(original["width"], original["height"]) <= 0:
            issues.append("invalid_declared_dimensions")
        if actual != (original.get("width"), original.get("height")):
            issues.append("declared_actual_dimension_mismatch")
        physical[image_id] = {"status": "readable", "issues": issues, "actual_width": actual[0],
                              "actual_height": actual[1], "exif_orientation": orientation,
                              "file_sha256": file_hash,
                              "cached_exif_orientation": cached.get("exif_orientation"),
                              "cached_file_sha256": cached.get("sha256"),
                              "fresh_matches_cache": file_hash == cached.get("sha256")
                                  and orientation == cached.get("exif_orientation")
                                  and actual == (cached.get("actual_width"), cached.get("actual_height"))}

    # Rebuild duplicate graph from cached 1,500-image fingerprints, replacing Food waste
    # image fingerprints with a fresh fingerprint from the same seven decoded rasters.
    fingerprints = {row["id"]: row["fingerprint"] for row in prior_data["image_evidence"] if row.get("fingerprint")}
    for image_id in food_images:
        meta = physical[image_id]
        img = images[image_id]
        if meta["fresh_matches_cache"] and image_id in fingerprints:
            continue
        # Preserve the same pixel/hash algorithm as D-073. A mismatch here is audited,
        # but a prior fingerprint remains the duplicate reference only when it matches.
        path = image_path(dataset_root, img["file_name"])
        row = {"path": str(path), "sha256": meta["file_sha256"], "actual_width": meta["actual_width"],
               "actual_height": meta["actual_height"]}
        fingerprints[image_id] = fingerprint(row)
    policy = strict_load(ROOT / "config/taco_preparation_policy.v1.json")
    exact, near = duplicate_groups(fingerprints, policy["near_duplicates"])
    groups = Groups(fingerprints)
    for group in exact:
        for member in group[1:]: groups.join(group[0], member)
    for pair in near: groups.join(pair["image_id_a"], pair["image_id_b"])
    components = {ident: group for group in groups.members() for ident in group}
    assert all(components.get(i, [i]) == [i] for i in food_images), (
        "A Food waste image joined an exact/near duplicate component; expand component-level strict evaluation")

    ledger = []
    summaries = {}
    for image_id in food_images:
        original = images[image_id]
        image_record = dict(physical[image_id])
        anns = by_image[image_id]
        category_names = [categories.get(a.get("category_id"), "<unknown category>") for a in anns]
        for tier in ("tier1", "tier2", "tier3"):
            result = disposition(image_record, anns, categories, tier, components.get(image_id, [image_id]))
            if result["eligible_before_duplicate_rules"] and any(set(component) & locked for component in [components.get(image_id, [image_id])]):
                result.update(eligible=False, primary_reason="duplicate_component_touches_heldout")
            else:
                result["eligible"] = result["eligible_before_duplicate_rules"]
            summaries.setdefault(tier, {"image_ids": [], "annotation_ids": []})
            if result["eligible"]:
                summaries[tier]["image_ids"].append(image_id)
                summaries[tier]["annotation_ids"].extend(result["mapped_food_waste_annotation_ids"])
            ledger.append({"tier": tier, "image_id": image_id, "file_name": original["file_name"],
                           "image_size": [physical[image_id]["actual_width"], physical[image_id]["actual_height"]],
                           "frozen_split_membership": ("validation" if image_id in frozen["validation"] else
                                                        "test" if image_id in frozen["test"] else None),
                           "candidate_assignment_if_eligible": "train" if result["eligible"] else None,
                           "raster_exif": {k: physical[image_id][k] for k in
                                            ("status", "issues", "exif_orientation", "fresh_matches_cache")},
                           "cooccurring_categories": category_names,
                           "disposition": result,
                           "food_waste_annotations": [
                               {"annotation_id": ann["id"], "bbox_xywh_original_px": ann["bbox"],
                                "bbox_area_px": ann["bbox"][2] * ann["bbox"][3],
                                "source_segmentation_area_px": ann.get("area"),
                                "resized_384_576": resized_box(ann["bbox"], physical[image_id]["actual_width"],
                                                                 physical[image_id]["actual_height"]),
                                "bbox_issues": bbox_issues(ann.get("bbox"), physical[image_id]["actual_width"],
                                                           physical[image_id]["actual_height"])}
                               for ann in anns if categories.get(ann.get("category_id")) == "Food waste"]})
    # Tier counts are image-level distinct counts and raw Food waste box counts.
    for result in summaries.values():
        result["images"] = len(set(result["image_ids"]))
        result["boxes"] = len(result["annotation_ids"])
        result["image_ids"] = sorted(set(result["image_ids"]))
        result["annotation_ids"] = sorted(result["annotation_ids"])

    category_mentions = {"organic", "biodegradable", "compost", "food scrap", "food waste", "fruit waste", "vegetable waste"}
    organic_source_categories = [c for c in raw["categories"]
                                 if any(term in c["name"].lower() for term in category_mentions)]
    result = {"authorization": "D-080", "read_only": True, "device": "CPU only",
              "training_updates": 0, "detector_forwards": 0, "test_json_opened": False,
              "heldout_photos_opened": 0, "raw_image_count": len(raw["images"]),
              "raw_food_waste_annotations": len(food_rows), "raw_food_waste_images": len(food_images),
              "organic_related_category_name_search": sorted(category_mentions),
              "organic_related_source_categories": [{"id": c["id"], "name": c["name"],
                                                       "annotations": sum(a["category_id"] == c["id"] for a in raw["annotations"])}
                                                      for c in organic_source_categories],
              "frozen_splits": {s: {"count": len(ids), "ids_sha256": hashlib.sha256(
                  json.dumps(sorted(ids)).encode()).hexdigest()} for s, ids in frozen.items()},
              "physical_food_image_checks": physical, "tiers": summaries, "ledger": ledger,
              "duplicate_evidence": {"exact_groups": exact, "near_pairs": near,
                                     "food_image_components": {str(i): components.get(i, [i]) for i in food_images}},
              "input_sha256": {str(source_path): sha256(source_path), str(prior_path): sha256(prior_path),
                               str(review): sha256(review), str(stage / "image_eligibility.jsonl"): sha256(stage / "image_eligibility.jsonl")},
              "assessment": {"ten_class_taco_only_feasible": False,
                             "reason": "At most four Food waste boxes survive any tier, on four or fewer candidate training images, and none are in frozen validation/test; this yields no independent Organic validation support.",
                             "recommended_path": "Keep supervised IDs 0..8 fixed; assess a separately authorized zero-shot food waste prompt route and/or external permissioned Organic supplement. Do not claim supervised Organic performance from these counts."}}
    path = out / "summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "counts": summaries,
                      "organic_categories": result["organic_related_source_categories"]}, indent=2))


if __name__ == "__main__":
    main()
