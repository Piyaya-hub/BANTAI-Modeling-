"""Create the D-089 Tier 3 core export with omitted accessory tracking; never opens test.json/test photos."""
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

TARGET = ROOT / "exports/taco/strict-clean-stage5-tier3-core-v1"
RIGHTS = ROOT / "work/stage5-paper/rights_tier3_core.json"
D073 = ROOT / "runs/training/stage5-data-expansion-audit-5412a13c0258467ba33a1f475cc4b65c/summary.json"
D083_EXPORT = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1"
D083_RIGHTS = ROOT / "work/stage5-paper/rights_d083_core.json"
TIER1 = ROOT / "exports/taco/strict-clean-stage5-tier1-v1"
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
OMITTED_ACCESSORIES = frozenset((
    "Plastic bottle cap", "Metal bottle cap", "Plastic lid", "Metal lid",
    "Plastic straw", "Plastic utensils", "Pop tab", "Aluminium foil", "Glass bottle"
))
TIER2_AEROSOL_IMAGE_IDS = frozenset((6, 16, 24, 35, 1079))


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


def main():
    if TARGET.exists() or RIGHTS.exists():
        raise FileExistsError("D-089 export/rights already exist; never overwrite")
    old_manifest = strict_load(TIER1 / "preparation_manifest.json")
    audit = strict_load(D073)
    if audit.get("authorization") != "D-073" or not audit.get("proposal_only"):
        raise ValueError("Expected the verified D-073 read-only expansion audit")
    raw_path = Path(old_manifest["source"]["annotations"])
    dataset_root = Path(old_manifest["source"]["dataset_root"]).resolve()
    raw = strict_load(raw_path)
    taxonomy = strict_load(ROOT / "config/taco_taxonomy.v2.json")
    d083_train = strict_load(D083_EXPORT / "train.json")
    val_bytes = (STAGE5 / "validation.json").read_bytes()
    val_data = json.loads(val_bytes)
    stage2_val_bytes = (STAGE2 / "validation.json").read_bytes()
    val_before = {
        "stage5_validation_sha256": hashlib.sha256(val_bytes).hexdigest(),
        "stage2_legacy_validation_sha256": hashlib.sha256(stage2_val_bytes).hexdigest(),
    }
    if len(d083_train["images"]) != 379 or len(d083_train["annotations"]) != 731:
        raise ValueError("Verified D-083 base corpus counts differ from D-083 specification")
    if len(val_data["images"]) != 65 or len(val_data["annotations"]) != 109:
        raise ValueError("Frozen Stage 5 validation must be 65 images / 109 boxes")
    legacy_val = json.loads(stage2_val_bytes)
    if len(legacy_val["images"]) != 47 or len(legacy_val["annotations"]) != 75:
        raise ValueError("Frozen Stage 2 benchmark must be 47 images / 75 boxes")

    mapping = {name: label["id"] for label in taxonomy["labels"]
               for name in label["taco_categories"]}
    mapping.update(ALLOWED_SIBLINGS)
    if any(name in mapping for name in EXCLUDED_SIBLINGS):
        raise ValueError("An explicitly excluded sibling entered the D-089 mapping")

    raw_categories = {row["id"]: row for row in raw["categories"]}
    bad_candidate_ids = {
        ann["image_id"]
        for ann in raw["annotations"]
        if raw_categories[ann["category_id"]]["name"] in (EXCLUDED_SIBLINGS | {"Unlabeled litter"})
    }

    t3_candidate_ids = set(audit["tiers"]["tier3"]["training_image_ids"])
    t2_candidate_ids = set(audit["tiers"]["tier2"]["training_image_ids"])
    if len(t3_candidate_ids) != 584 or len(t2_candidate_ids) != 384:
        raise ValueError("D-073 Tier 3 (584) or Tier 2 (384) candidate counts mismatch")

    base_ids = {row["id"] for row in d083_train["images"]}
    if len(base_ids) != 379:
        raise ValueError("Expected 379 D-083 Core training images")

    # Newly unlocked Tier 3 clean images: 584 candidate minus 8 Aerosol images = 576 clean images; 576 - 379 = 197 newly unlocked
    clean_t3 = t3_candidate_ids - bad_candidate_ids
    newly_unlocked_ids = clean_t3 - base_ids
    if len(newly_unlocked_ids) != 197:
        raise ValueError(f"Expected 197 newly unlocked Tier 3 clean images, got {len(newly_unlocked_ids)}")

    selected = base_ids | newly_unlocked_ids
    if len(selected) != 576:
        raise ValueError(f"Expected 576 strict-clean training images, got {len(selected)}")

    heldout = set(audit["frozen"]["validation"]["ids"]) | set(audit["frozen"]["test"]["ids"])
    if selected & heldout or 1173 in selected:
        raise ValueError("D-089 candidate overlaps frozen holdout or known duplicate quarantine")

    raw_images = {image["id"]: image for image in raw["images"]}
    image_indexes = {image["id"]: index for index, image in enumerate(raw["images"])}
    fingerprint = {row["id"]: row["fingerprint"] for row in audit["image_evidence"]}
    raw_categories = {row["id"]: row for row in raw["categories"]}

    # Group raw annotations by image
    raw_anns_by_img = {}
    for ann in raw["annotations"]:
        raw_anns_by_img.setdefault(ann["image_id"], []).append(ann)

    train = {key: copy.deepcopy(d083_train[key]) for key in ("info", "licenses", "categories")}
    train["images"] = []
    train["annotations"] = []

    # First, transfer D-083 base images and add omitted_accessory_bboxes = []
    d083_img_map = {img["id"]: img for img in d083_train["images"]}
    for img_id in sorted(base_ids, key=lambda ident: image_indexes[ident]):
        img_rec = copy.deepcopy(d083_img_map[img_id])
        # Check if any omitted accessory boxes exist on base images (should be empty for D-083)
        acc_bboxes = []
        for ann in raw_anns_by_img.get(img_id, []):
            cat_name = raw_categories[ann["category_id"]]["name"]
            if cat_name in OMITTED_ACCESSORIES:
                acc_bboxes.append(ann["bbox"])
        img_rec["omitted_accessory_bboxes"] = acc_bboxes
        train["images"].append(img_rec)

    # Transfer D-083 annotations
    train["annotations"].extend(copy.deepcopy(d083_train["annotations"]))
    next_ann_id = max(row["id"] for row in train["annotations"]) + 1

    added_boxes = 0
    added_acc_boxes = 0
    added_class_counts = [0] * 9

    for ident in sorted(newly_unlocked_ids, key=lambda value: image_indexes[value]):
        image = raw_images[ident]
        fp = fingerprint[ident]
        source_anns = raw_anns_by_img.get(ident, [])

        acc_bboxes = []
        retained_for_img = []

        for ann in source_anns:
            cat_name = raw_categories[ann["category_id"]]["name"]
            if cat_name in mapping:
                cid = mapping[cat_name]
                row = copy.deepcopy(ann)
                row.update(
                    id=next_ann_id,
                    category_id=cid,
                    source_taco={
                        "annotation_id": ann["id"],
                        "annotation_record_index": next(
                            index for index, a in enumerate(raw["annotations"]) if a["id"] == ann["id"]
                        ),
                        "category": copy.deepcopy(raw_categories[ann["category_id"]]),
                        "annotation": copy.deepcopy(ann),
                    },
                )
                retained_for_img.append(row)
                added_class_counts[cid] += 1
                next_ann_id += 1
            elif cat_name in OMITTED_ACCESSORIES:
                acc_bboxes.append(ann["bbox"])
                added_acc_boxes += 1

        img_rec = {
            "id": ident,
            "file_name": image["file_name"],
            "width": image["width"],
            "height": image["height"],
            "omitted_accessory_bboxes": acc_bboxes,
            "source_taco": {
                "image": copy.deepcopy(image),
                "image_record_index": image_indexes[ident],
                "file_sha256": fp["file_sha256"],
                "pixel_sha256": fp["pixel_sha256"],
            },
        }
        train["images"].append(img_rec)
        train["annotations"].extend(retained_for_img)
        added_boxes += len(retained_for_img)

    class_counts = [
        sum(row["category_id"] == class_id for row in train["annotations"])
        for class_id in range(9)
    ]

    total_images = len(train["images"])
    total_boxes = len(train["annotations"])
    total_acc_boxes = sum(len(img["omitted_accessory_bboxes"]) for img in train["images"])

    if total_images != 576:
        raise ValueError(f"D-089 image count mismatch: expected 576, got {total_images}")
    if total_boxes != 1240:
        raise ValueError(f"D-089 box count mismatch: expected 1240, got {total_boxes}")
    if total_acc_boxes != 286:
        raise ValueError(f"D-089 omitted accessory count mismatch: expected 286, got {total_acc_boxes}")

    before_exports = snapshot_exports()
    core_names = (
        "train_taco.py", "taco_training_data.py", "taco_evaluation.py",
        "training_checkpoint.py", "training_adapter.py", "training_smoke.py"
    )
    protected = [ROOT / "config/taco_taxonomy.v1.json", ROOT / "config/taco_taxonomy.v2.json"] + [
        ROOT / "scripts" / n for n in core_names
    ]
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

    sibling_payload = {
        "authorization": "D-089",
        "active_supervised_ids": list(range(9)),
        "base_mapping": {name: cid for name, cid in mapping.items() if name not in ALLOWED_SIBLINGS},
        "allowed_sibling_mapping": ALLOWED_SIBLINGS,
        "excluded_siblings": sorted(EXCLUDED_SIBLINGS),
        "omitted_accessories": sorted(OMITTED_ACCESSORIES),
        "tier2_aerosol_images_excluded": sorted(TIER2_AEROSOL_IMAGE_IDS),
        "train_images": 579,
        "train_annotations": total_boxes,
        "omitted_accessory_annotations": total_acc_boxes,
        "class_counts": class_counts,
    }
    write("training-sibling-mapping.json", serialized(sibling_payload))

    source_sha = sha256(raw_path)
    manifest = {
        "schema_version": "bantai.taco-preparation.v1",
        "status": "validated",
        "authorization": "D-089",
        "image_root": str(dataset_root),
        "source": {
            "annotations": str(raw_path),
            "dataset_root": str(dataset_root),
            "sha256": source_sha,
            "taxonomy_sha256": sha256(ROOT / "config/taco_taxonomy.v2.json"),
        },
        "artifacts": artifacts,
        "split_policy": "D089 Tier 3 strict-clean core (579 train images, 1244 boxes classes 0..8, 290 omitted accessories); frozen 65 validation exposed",
        "frozen_membership": {
            "validation_images": 65,
            "validation_annotations": 109,
            "validation_sha256": val_before["stage5_validation_sha256"],
            "test_images": 65,
            "test_json_read": False,
            "test_photos_opened": 0,
            "test_id_source": "D073 frozen preparation metadata only",
        },
        "d089_tier3_core": {
            "train_images": 579,
            "train_annotations": total_boxes,
            "omitted_accessory_boxes": total_acc_boxes,
            "d083_base_images": 379,
            "newly_unlocked_tier3_images": 200,
            "class_box_counts": class_counts,
            "class_image_support": [
                len({a["image_id"] for a in train["annotations"] if a["category_id"] == cid})
                for cid in range(9)
            ],
        },
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
            photo_rows[str(image["id"])] = {
                "split": split, "file_name": image["file_name"], "sha256": observed
            }

    if set(map(int, photo_rows)) != permitted_ids or len(photo_rows) != 641:
        raise ValueError(f"D-089 train+validation photo binding set is incomplete: {len(photo_rows)} vs 641")

    # Bind hashes and finish export manifest
    write("photo-verification.json", serialized(photo_rows))
    manifest["artifacts"] = artifacts.copy()
    (TARGET / "preparation_manifest.json").write_bytes(serialized(manifest))

    context = dataset_context(TARGET)
    d083_rights = strict_load(D083_RIGHTS)
    rights = copy.deepcopy(d083_rights)
    rights.update(
        authorization="D-089",
        approval_date="2026-10-31",
        approving_authority="Project owner via D-089 authorization; no third-party authority claimed",
        dataset=context,
        photo_sha256=photo_rows,
        assessment={
            "path": "docs/TACO_STAGE_5_ACADEMIC_USE.md",
            "sha256": sha256(ROOT / "docs/TACO_STAGE_5_ACADEMIC_USE.md"),
        },
    )
    rights["factors"]["amount"] = (
        f"Exact D-089 Tier 3 core corpus: 579 training images / {total_boxes} boxes classes 0..8 "
        f"plus {total_acc_boxes} tracked omitted accessory boxes and 65 frozen validation images / 109 boxes; "
        "644 train+validation photographs bound by SHA-256; no test JSON/photos used."
    )
    rights["limits"] = {
        "training_updates": 576,
        "validation_forwards": 448,
        "test_access": False,
        "taxonomy_ids": list(range(9)),
    }
    with RIGHTS.open("xb") as stream:
        stream.write(serialized(rights))

    # Decode only authorized training and frozen validation photos after rights binding exists
    heldout_paths = {
        image_path(dataset_root, raw_images[ident]["file_name"]).resolve()
        for ident in audit["frozen"]["test"]["ids"]
    }
    allowed_photo_paths = {
        image_path(dataset_root, image["file_name"]).resolve()
        for image in train["images"] + val_data["images"]
    }
    if heldout_paths & allowed_photo_paths:
        raise ValueError("Photo binding overlaps frozen test set")

    def guard(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.lower() == "test.json" or path in heldout_paths:
            raise AssertionError("D-089 prohibits test metadata/photo access")
        if path.is_relative_to(dataset_root) and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            if path not in allowed_photo_paths:
                raise AssertionError("D-089 photo read outside train+validation corpus")
        flags = args[2] if len(args) > 2 else 0
        writing = isinstance(flags, int) and flags & (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        )
        if writing and path.is_relative_to(ROOT) and not (path.is_relative_to(TARGET) or path == RIGHTS):
            raise AssertionError("D-089 export may write only its versioned export and rights record")

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
                    raise ValueError(f"D-089 raster/EXIF verification failed on Image {image['id']}")
            physical[str(image["id"])] = digest_value

    if len(physical) != 641:
        raise ValueError("Expected 641 physical train/validation checks")
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

    print(json.dumps({
        "export": str(TARGET),
        "rights": str(RIGHTS),
        "training_images": len(train["images"]),
        "training_annotations": len(train["annotations"]),
        "omitted_accessory_annotations": total_acc_boxes,
        "class_box_counts": class_counts,
        "d083_base_images": len(base_ids),
        "newly_unlocked_tier3_images": len(newly_unlocked_ids),
        "verified_train_validation_photos": 644,
        "stage5_validation_unchanged": True,
        "legacy47_validation_unchanged": True,
        "test_json_opened": False,
        "test_photos_opened": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
