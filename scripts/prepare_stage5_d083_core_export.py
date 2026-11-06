"""Create the D-083 core eight-object export; never opens test.json/test photos."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.configuration import strict_load
from scripts.taco_dataset import image_path, sha256
from scripts.taco_training_data_v2 import dataset_context

TARGET = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1"
RIGHTS = ROOT / "work/stage5-paper/rights_d083_core.json"
D073 = ROOT / "runs/training/stage5-data-expansion-audit-5412a13c0258467ba33a1f475cc4b65c/summary.json"
TIER1 = ROOT / "exports/taco/strict-clean-stage5-tier1-v1"
TIER1_RIGHTS = ROOT / "work/stage5-paper/rights_tier1.json"
STAGE5 = ROOT / "exports/taco/strict-clean-stage5-v1"
STAGE2 = ROOT / "exports/taco/strict-clean-stage2-v1"

ALLOWED_SIBLINGS = {
    "Polypropylene bag": 2,
    "Six pack rings": 3,
    "Food Can": 5,
    "Disposable food container": 6,
    "Spread tub": 6,
    "Other plastic container": 6,
    "Foam cup": 7,
    "Foam food container": 7,
}
EXCLUDED_SIBLINGS = frozenset(("Aerosol", "Plastified paper bag", "Other plastic"))
EXPECTED_CLASS_BOXES = (147, 48, 22, 74, 133, 59, 54, 54, 140)


def serialized(data):
    return (json.dumps(data, indent=2, allow_nan=False, ensure_ascii=False,
                       sort_keys=True) + "\n").replace("\n", "\r\n").encode("utf-8")


def snapshot_exports():
    """Hash existing export files except prohibited-to-open test.json files."""
    root = ROOT / "exports/taco"
    return {str(path.relative_to(root)): sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name.lower() != "test.json"
            and not path.is_relative_to(TARGET)}


def calculate_selection(raw, candidate_ids, base_ids, mapping):
    categories = {row["id"]: row["name"] for row in raw["categories"]}
    annotations = {ident: [] for ident in candidate_ids}
    for ann in raw["annotations"]:
        if ann["image_id"] in annotations:
            annotations[ann["image_id"]].append(ann)
    selected = set()
    dropped_reasons = {}
    for ident, rows in annotations.items():
        names = [categories[row["category_id"]] for row in rows]
        if any(name in EXCLUDED_SIBLINGS for name in names):
            dropped_reasons[ident] = sorted(set(names) & EXCLUDED_SIBLINGS)
            continue
        if not rows or any(name not in mapping for name in names):
            dropped_reasons[ident] = sorted(set(names) - set(mapping)) or ["no mapped boxes"]
            continue
        selected.add(ident)
    if not base_ids.issubset(selected):
        raise ValueError("D-083 rule set unexpectedly drops a Tier 1 training image")
    return selected, annotations, dropped_reasons


def main():
    if TARGET.exists() or RIGHTS.exists():
        raise FileExistsError("D-083 export/rights already exist; never overwrite")
    old_manifest = strict_load(TIER1 / "preparation_manifest.json")
    audit = strict_load(D073)
    if audit.get("authorization") != "D-073" or not audit.get("proposal_only"):
        raise ValueError("Expected the verified D-073 read-only expansion audit")
    raw_path = Path(old_manifest["source"]["annotations"])
    dataset_root = Path(old_manifest["source"]["dataset_root"]).resolve()
    raw = strict_load(raw_path)
    taxonomy = strict_load(ROOT / "config/taco_taxonomy.v2.json")
    base_train_bytes = (TIER1 / "train.json").read_bytes()
    base_train = json.loads(base_train_bytes)
    val_bytes = (STAGE5 / "validation.json").read_bytes()
    val_data = json.loads(val_bytes)
    stage2_val_bytes = (STAGE2 / "validation.json").read_bytes()
    val_before = {
        "stage5_validation_sha256": hashlib.sha256(val_bytes).hexdigest(),
        "stage2_legacy_validation_sha256": hashlib.sha256(stage2_val_bytes).hexdigest(),
    }
    if len(base_train["images"]) != 336 or len(base_train["annotations"]) != 637:
        raise ValueError("Verified Tier 1 base corpus counts differ from D-074")
    if len(val_data["images"]) != 65 or len(val_data["annotations"]) != 109:
        raise ValueError("Frozen Stage 5 validation must be 65 images / 109 boxes")
    legacy_val = json.loads(stage2_val_bytes)
    if len(legacy_val["images"]) != 47 or len(legacy_val["annotations"]) != 75:
        raise ValueError("Frozen Stage 2 benchmark must be 47 images / 75 boxes")

    mapping = {name: label["id"] for label in taxonomy["labels"]
               for name in label["taco_categories"]}
    mapping.update(ALLOWED_SIBLINGS)
    if any(name in mapping for name in EXCLUDED_SIBLINGS):
        raise ValueError("An explicitly excluded sibling entered the D-083 mapping")
    d073_train = set(audit["tiers"]["tier2"]["training_image_ids"])
    base_ids = {row["id"] for row in base_train["images"]}
    heldout = set(audit["frozen"]["validation"]["ids"]) | set(audit["frozen"]["test"]["ids"])
    if len(audit["frozen"]["validation"]["ids"]) != 65 or len(audit["frozen"]["test"]["ids"]) != 65:
        raise ValueError("D-073 frozen membership references are incomplete")
    selected, annotations_by_image, drop_reasons = calculate_selection(raw, d073_train, base_ids, mapping)
    if selected & heldout or 1173 in selected:
        raise ValueError("D-083 candidate overlaps frozen holdout or known duplicate quarantine")
    if (len(selected), len(selected - base_ids)) != (379, 43):
        raise ValueError(f"D-083 image count mismatch: {len(selected)} selected")

    raw_images = {image["id"]: image for image in raw["images"]}
    image_indexes = {image["id"]: index for index, image in enumerate(raw["images"])}
    fingerprint = {row["id"]: row["fingerprint"] for row in audit["image_evidence"]}
    train = {key: copy.deepcopy(base_train[key]) for key in ("info", "licenses", "categories")}
    train["images"] = copy.deepcopy(base_train["images"])
    train["annotations"] = copy.deepcopy(base_train["annotations"])
    next_ann_id = max(row["id"] for row in train["annotations"]) + 1
    raw_categories = {row["id"]: row for row in raw["categories"]}
    added_class_counts = [0] * 9
    new_image_records = []
    new_annotation_records = []
    for ident in sorted(selected - base_ids, key=lambda value: image_indexes[value]):
        image = raw_images[ident]
        fp = fingerprint[ident]
        new_image_records.append({
            "id": ident, "file_name": image["file_name"], "width": image["width"], "height": image["height"],
            "source_taco": {"image": copy.deepcopy(image), "image_record_index": image_indexes[ident],
                            "file_sha256": fp["file_sha256"], "pixel_sha256": fp["pixel_sha256"]},
        })
        for source_ann in annotations_by_image[ident]:
            category = raw_categories[source_ann["category_id"]]
            class_id = mapping[category["name"]]
            row = copy.deepcopy(source_ann)
            row.update(id=next_ann_id, category_id=class_id,
                       source_taco={"annotation_id": source_ann["id"],
                                    "annotation_record_index": next(index for index, ann in enumerate(raw["annotations"])
                                                                     if ann["id"] == source_ann["id"]),
                                    "category": copy.deepcopy(category), "annotation": copy.deepcopy(source_ann)})
            new_annotation_records.append(row)
            added_class_counts[class_id] += 1
            next_ann_id += 1
    train["images"].extend(new_image_records)
    train["annotations"].extend(new_annotation_records)
    class_counts = [sum(row["category_id"] == class_id for row in train["annotations"])
                    for class_id in range(9)]
    if len(train["images"]) != 379 or len(train["annotations"]) != 731 or tuple(class_counts) != EXPECTED_CLASS_BOXES:
        raise ValueError(f"D-083 exact corpus count mismatch: {len(train['images'])}, {len(train['annotations'])}, {class_counts}")
    if any(not annotations_by_image[ident] for ident in selected):
        raise ValueError("Selected D-083 image has no source annotations")

    before_exports = snapshot_exports()
    core_names = ("train_taco.py", "taco_training_data.py", "taco_evaluation.py",
                  "training_checkpoint.py", "training_adapter.py", "training_smoke.py")
    protected = [ROOT / "config/taco_taxonomy.v1.json", ROOT / "config/taco_taxonomy.v2.json"] + [ROOT / "scripts" / n for n in core_names]
    protected_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in protected}

    # New export files: deliberately no test.json, test image list, or test-photo data.
    TARGET.mkdir(parents=True)
    artifacts = {}
    def write(name, payload):
        path = TARGET / name
        with path.open("xb") as stream:
            stream.write(payload)
        artifacts[name] = hashlib.sha256(payload).hexdigest()
    write("train.json", serialized(train))
    write("validation.json", val_bytes)
    write("taxonomy_snapshot.json", serialized(taxonomy))
    sibling_payload = {"authorization": "D-083", "active_supervised_ids": list(range(9)),
                       "base_mapping": {name: cid for name, cid in mapping.items()
                                        if name not in ALLOWED_SIBLINGS},
                       "allowed_sibling_mapping": ALLOWED_SIBLINGS,
                       "excluded_siblings": sorted(EXCLUDED_SIBLINGS),
                       "excluded_candidate_images": drop_reasons}
    write("training-sibling-mapping.json", serialized(sibling_payload))
    source_sha = sha256(raw_path)
    manifest = {
        "schema_version": "bantai.taco-preparation.v1", "status": "validated",
        "authorization": "D-083", "image_root": str(dataset_root),
        "source": {"annotations": str(raw_path), "dataset_root": str(dataset_root),
                   "sha256": source_sha,
                   "taxonomy_sha256": sha256(ROOT / "config/taco_taxonomy.v2.json")},
        "artifacts": artifacts,
        "split_policy": "D083 Tier1 strict-clean base plus D073-audited selected siblings; only train and frozen validation exposed",
        "frozen_membership": {"validation_images": 65, "validation_annotations": 109,
                              "validation_sha256": val_before["stage5_validation_sha256"],
                              "test_images": 65, "test_json_read": False, "test_photos_opened": 0,
                              "test_id_source": "D073 frozen preparation metadata only"},
        "d083_core": {"train_images": 379, "train_annotations": 731,
                      "tier1_base_images": 336, "sibling_added_images": 43,
                      "class_box_counts": class_counts, "class_image_support": [
                          len({a["image_id"] for a in train["annotations"] if a["category_id"] == cid})
                          for cid in range(9)]},
    }
    photo_rows = {}
    permitted_ids = selected | {row["id"] for row in val_data["images"]}
    for split, images in (("train", train["images"]), ("validation", val_data["images"])):
        for image in images:
            path = image_path(dataset_root, image["file_name"])
            observed = sha256(path)
            expected = image["source_taco"]["file_sha256"]
            if observed != expected:
                raise ValueError(f"D-073/Tier1 photo binding changed for Image {image['id']}")
            photo_rows[str(image["id"])] = {"split": split, "file_name": image["file_name"], "sha256": observed}
    if set(map(int, photo_rows)) != permitted_ids or len(photo_rows) != 444:
        raise ValueError("D-083 train+validation photo binding set is incomplete")

    # Bind hashes and finish the export manifest before any raster is decoded.
    write("photo-verification.json", serialized(photo_rows))
    manifest["artifacts"] = artifacts.copy()
    (TARGET / "preparation_manifest.json").write_bytes(serialized(manifest))
    context = dataset_context(TARGET)
    tier1_rights = strict_load(TIER1_RIGHTS)
    rights = copy.deepcopy(tier1_rights)
    rights.update(authorization="D-083", approval_date="2026-10-31",
                  approving_authority="Project owner via D-083 authorization; no third-party authority claimed",
                  dataset=context, photo_sha256=photo_rows,
                  assessment={"path": "docs/TACO_STAGE_5_ACADEMIC_USE.md",
                              "sha256": sha256(ROOT / "docs/TACO_STAGE_5_ACADEMIC_USE.md")})
    rights["factors"]["amount"] = (
        "Exact D-083 core corpus: 379 training images / 731 boxes and 65 frozen validation images / 109 boxes; "
        "444 train+validation photographs bound by SHA-256; no test JSON/photos used.")
    rights["limits"] = {"training_updates": 379, "validation_forwards": 336,
                        "test_access": False, "taxonomy_ids": list(range(9))}
    with RIGHTS.open("xb") as stream:
        stream.write(serialized(rights))

    # Decode only the authorized training and frozen validation photos after rights binding exists.
    heldout_paths = {image_path(dataset_root, raw_images[ident]["file_name"]).resolve()
                     for ident in audit["frozen"]["test"]["ids"]}
    allowed_photo_paths = {image_path(dataset_root, image["file_name"]).resolve()
                           for image in train["images"] + val_data["images"]}
    if heldout_paths & allowed_photo_paths:
        raise ValueError("Photo binding overlaps frozen test set")
    def guard(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.lower() == "test.json" or path in heldout_paths:
            raise AssertionError("D-083 prohibits test metadata/photo access")
        if path.is_relative_to(dataset_root) and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            if path not in allowed_photo_paths:
                raise AssertionError("D-083 photo read outside train+validation corpus")
        flags = args[2] if len(args) > 2 else 0
        writing = isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if writing and path.is_relative_to(ROOT) and not (path.is_relative_to(TARGET) or path == RIGHTS):
            raise AssertionError("D-083 export may write only its versioned export and rights record")
    sys.addaudithook(guard)
    physical = {}
    for split, images in (("train", train["images"]), ("validation", val_data["images"])):
        for image in images:
            path = image_path(dataset_root, image["file_name"])
            payload = path.read_bytes()
            digest_value = hashlib.sha256(payload).hexdigest()
            if digest_value != photo_rows[str(image["id"])]["sha256"]:
                raise ValueError("Photo changed between hash binding and decode")
            with Image.open(io.BytesIO(payload)) as photo:
                photo.load()
                if photo.size != (image["width"], image["height"]) or photo.getexif().get(274, 1) != 1:
                    raise ValueError(f"D-083 raster/EXIF verification failed on Image {image['id']}")
            physical[str(image["id"])] = digest_value
    if len(physical) != 444:
        raise ValueError("Expected 444 physical train/validation checks")
    if physical != {ident: row["sha256"] for ident, row in photo_rows.items()}:
        raise ValueError("Post-decode photo verification differs from pre-decode rights binding")

    if snapshot_exports() != before_exports:
        raise AssertionError("An existing export changed")
    if {str(path.relative_to(ROOT)): sha256(path) for path in protected} != protected_hashes:
        raise AssertionError("Protected taxonomy/core Stage 4 source changed")
    if hashlib.sha256((STAGE5 / "validation.json").read_bytes()).hexdigest() != val_before["stage5_validation_sha256"]:
        raise AssertionError("Frozen Stage 5 validation bytes changed")
    if hashlib.sha256((STAGE2 / "validation.json").read_bytes()).hexdigest() != val_before["stage2_legacy_validation_sha256"]:
        raise AssertionError("Frozen Stage 2 validation bytes changed")
    print(json.dumps({"export": str(TARGET), "rights": str(RIGHTS),
                      "training_images": len(train["images"]), "training_annotations": len(train["annotations"]),
                      "class_box_counts": class_counts, "tier1_base_images": len(base_ids),
                      "sibling_added_images": len(selected - base_ids), "verified_train_validation_photos": 444,
                      "stage5_validation_unchanged": True, "legacy47_validation_unchanged": True,
                      "test_json_opened": False, "test_photos_opened": 0}))


if __name__ == "__main__":
    main()
