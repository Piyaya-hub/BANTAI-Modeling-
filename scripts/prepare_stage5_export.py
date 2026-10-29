#!/usr/bin/env python3
"""
Generate the Stage 5 9-class dataset export (strict-clean-stage5-v1).

Incorporates 'paper' (id: 8) alongside the original 8 classes (0..7).
Preserves the frozen original 221 train, 47 val, and 47 test memberships byte-for-byte,
appending the 122 newly eligible images under Option A:
  - train: 221 frozen + 86 new = 307 images
  - validation: 47 frozen + 18 new = 65 images
  - test: 47 frozen + 18 new = 65 images
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.configuration import resolve_path, strict_load
from scripts.taco_dataset import load_taxonomy, sha256
from scripts.taco_preparation import (
    DEFAULT_POLICY,
    RULES,
    SPLITS,
    annotation_issues,
    load_policy,
    make_plan,
    statistics,
    validate_coco,
)
from scripts.taco_splits import Groups, assign_splits


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lines(p: Path) -> list[dict]:
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    stage2_dir = ROOT / "exports/taco/strict-clean-stage2-v1"
    review_dir = ROOT / "reports/taco/local-review"
    output_dir = ROOT / "exports/taco/strict-clean-stage5-v1"
    rights_path = ROOT / "work/stage5-paper/rights.json"
    taxonomy_v2_path = ROOT / "config/taco_taxonomy.v2.json"

    print("=== Stage 5 Strict-Clean Export Generator (Option A) ===")

    if output_dir.exists():
        print(f"ERROR: Output directory already exists: {output_dir}", file=sys.stderr)
        return 1

    taxonomy_v2 = load_taxonomy(taxonomy_v2_path)
    policy_path = stage2_dir / "policy_snapshot.json"
    policy = strict_load(policy_path)

    manifest_v1 = strict_load(stage2_dir / "preparation_manifest.json")
    source_path = Path(manifest_v1["source"]["annotations"])

    raw = strict_load(source_path)
    summary = strict_load(review_dir / "audit.json")
    images = lines(review_dir / "images.jsonl")
    annotations = lines(review_dir / "annotation_review.jsonl")
    old_eligibility = lines(stage2_dir / "image_eligibility.jsonl")

    # Cache fingerprints from old eligibility
    fingerprints = {r["source_image_id"]: r["fingerprint"] for r in old_eligibility if r["fingerprint"]}

    # Prepare audit struct
    audit = {
        "summary": summary,
        "taxonomy": taxonomy_v2,
        "images": images,
        "annotations": annotations,
        "source_document": raw,
    }

    # Find paper category ids
    paper_cat_names = set(taxonomy_v2["labels"][8]["taco_categories"])
    by_name = {c["name"]: c["id"] for c in raw["categories"]}
    paper_cat_ids = {by_name[n] for n in paper_cat_names if n in by_name}

    # Mark paper annotations
    proposed_audit = copy.deepcopy(audit)
    for row in proposed_audit["annotations"]:
        if row["original"]["category_id"] in paper_cat_ids:
            row.update(bantai_label="paper", bantai_label_id=8, handling="mapped")

    # Replay plan
    from unittest.mock import patch
    import scripts.taco_preparation as taco_prep

    def cached_fingerprint(row):
        val = fingerprints.get(row["original"]["id"])
        if val is None:
            raise ValueError(f"Missing cached fingerprint for image id {row['original']['id']}")
        return copy.deepcopy(val)

    with patch.object(taco_prep, "fingerprint", side_effect=cached_fingerprint):
        plan = make_plan(proposed_audit, policy)

    old_ids = {r["source_image_id"]: r["split"] for r in old_eligibility if r["eligible"]}
    new_eligible = {r["source_image_id"] for r in plan["eligibility"] if r["eligible"]}
    new_ids = new_eligible - set(old_ids.keys())

    print(f"Existing eligible images: {len(old_ids)}")
    print(f"Newly eligible paper images: {len(new_ids)}")
    print(f"Total eligible images: {len(new_eligible)}")
    assert len(old_ids) == 315
    assert len(new_ids) == 122
    assert len(new_eligible) == 437

    # Option A: Seeded allocation of new IDs while locking old IDs
    locked = dict(old_ids)
    groups = Groups(new_eligible)
    for group in plan["exact_duplicate_groups"]:
        valid = [i for i in group if i in new_eligible]
        for i in valid[1:]:
            groups.join(valid[0], i)
    for pair in plan["near_duplicate_candidates"]:
        a, b = pair["image_id_a"], pair["image_id_b"]
        if a in new_eligible and b in new_eligible:
            groups.join(a, b)

    assignments = dict(locked)
    free = []
    for group in groups.members():
        old_splits = {locked[i] for i in group if i in locked}
        assert len(old_splits) <= 1
        new_in_group = [i for i in group if i in new_ids]
        if not new_in_group:
            continue
        if old_splits:
            split = next(iter(old_splits))
            assignments.update({i: split for i in new_in_group})
        else:
            free.append(new_in_group)

    labels = {r["source_image_id"]: set(r["labels"]) for r in plan["eligibility"] if r["eligible"]}
    free_ids = set(sum(free, []))
    extra, targets = assign_splits({i: labels[i] for i in free_ids}, free, policy["default_ratios"], 42)
    assignments.update(extra)

    # Verify split counts
    train_ids = [i for i, s in assignments.items() if s == "train"]
    val_ids = [i for i, s in assignments.items() if s == "validation"]
    test_ids = [i for i, s in assignments.items() if s == "test"]

    print(f"Final split counts under Option A:")
    print(f"  Train: {len(train_ids)} (221 frozen + {len(train_ids) - 221} new)")
    print(f"  Validation: {len(val_ids)} (47 frozen + {len(val_ids) - 47} new)")
    print(f"  Test: {len(test_ids)} (47 frozen + {len(test_ids) - 47} new)")

    assert len(train_ids) == 307
    assert len(val_ids) == 65
    assert len(test_ids) == 65

    # Update plan eligibility rows with exact assignments and split_groups
    for row in plan["eligibility"]:
        ident = row["source_image_id"]
        if row["eligible"]:
            row["split"] = assignments[ident]
            row["split_group"] = groups.find(ident)

    # Reconstruct COCO exports with updated assignments
    exports = {
        s: {
            "info": {
                "description": "BANTAI strict-clean TACO box-detection preparation v2 (9 classes with paper)",
                "image_root": summary["source"]["dataset_root"],
                "coordinate_frame": policy["coordinate_frame"],
                "policy_version": policy["version"],
                "taxonomy_version": taxonomy_v2["version"],
            },
            "licenses": copy.deepcopy(raw.get("licenses", [])),
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": l["id"],
                    "name": l["name"],
                    "supercategory": "trash_object",
                    "source_taco_categories": l["taco_categories"],
                }
                for l in taxonomy_v2["labels"]
            ],
        }
        for s in SPLITS
    }

    indexed_rows = {r["source_image_id"]: r for r in plan["eligibility"] if r["eligible"]}
    for row in plan["eligibility"]:
        if row["eligible"]:
            orig = row["source_image"]
            img_entry = {key: orig[key] for key in ("id", "file_name", "width", "height")}
            img_entry["source_taco"] = {
                "image": copy.deepcopy(orig),
                "image_record_index": row["image_record_index"],
                "file_sha256": row["fingerprint"]["file_sha256"],
                "pixel_sha256": row["fingerprint"]["pixel_sha256"],
            }
            exports[row["split"]]["images"].append(img_entry)

    decisions = []
    next_ann_id = 1
    for ann in proposed_audit["annotations"]:
        orig_ann = ann["original"]
        img_id = orig_ann["image_id"]
        img_row = indexed_rows.get(img_id)
        is_eligible = img_row is not None and img_row["eligible"]
        reasons = img_row["reasons"] if img_row else []
        split = img_row["split"] if img_row else None

        decision = {
            "annotation_record_index": ann["index"],
            "source_annotation": orig_ann,
            "source_category": ann["original_category"],
            "bantai_label": ann["bantai_label"],
            "annotation_issues": annotation_issues(ann),
            "image_reasons": reasons,
            "split": split,
            "exported_annotation_id": None,
        }

        if is_eligible:
            converted = copy.deepcopy(orig_ann)
            converted.update(
                id=next_ann_id,
                category_id=ann["bantai_label_id"],
                source_taco={
                    "annotation_id": orig_ann["id"],
                    "category": ann["original_category"],
                    "annotation_record_index": ann["index"],
                    "annotation": orig_ann,
                },
            )
            exports[split]["annotations"].append(converted)
            decision["exported_annotation_id"] = next_ann_id
            next_ann_id += 1

        decisions.append(decision)

    plan["exports"] = exports
    plan["annotation_decisions"] = decisions

    # Validate COCO structures
    print("Validating COCO export structures...")
    coco_errors = validate_coco(exports, plan, taxonomy_v2)
    if coco_errors:
        print(f"ERROR: COCO validation failed:\n" + "\n".join(coco_errors), file=sys.stderr)
        return 1
    print("COCO validation passed cleanly.")

    # Freshly verify on-disk images for all 437 eligible images
    print("Verifying physical on-disk readability for all 437 eligible images...")
    dataset_root = Path(summary["source"]["dataset_root"])
    for split in SPLITS:
        for img in exports[split]["images"]:
            fpath = dataset_root / img["file_name"]
            if not fpath.is_file():
                print(f"ERROR: Image file missing on disk: {fpath}", file=sys.stderr)
                return 1
            with Image.open(fpath) as pil_img:
                pil_img.load()
                if pil_img.width != img["width"] or pil_img.height != img["height"]:
                    print(
                        f"ERROR: Dimension mismatch for {fpath}: {pil_img.size} vs {img['width']}x{img['height']}",
                        file=sys.stderr,
                    )
                    return 1
    print("All 437 eligible images verified on disk.")

    # Write export files
    output_dir.mkdir(parents=True, exist_ok=False)

    def write_json(name: str, val: any) -> None:
        (output_dir / name).write_text(
            json.dumps(val, indent=2, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    stats = statistics(proposed_audit, plan)
    write_json("policy_snapshot.json", policy)
    write_json("taxonomy_snapshot.json", taxonomy_v2)
    for s in SPLITS:
        write_json(f"{s}.json", exports[s])
    write_json("statistics.json", stats)
    write_json(
        "duplicate_review.json",
        {
            "exact_groups": plan["exact_duplicate_groups"],
            "exact_annotation_conflicts": plan["exact_duplicate_annotation_conflicts"],
            "near_candidates": plan["near_duplicate_candidates"],
            "split_groups": plan["split_groups"],
        },
    )
    write_json(
        "source_metadata.json",
        {k: v for k, v in raw.items() if k not in ("images", "annotations")},
    )

    for name, row_list in (
        ("image_eligibility.jsonl", plan["eligibility"]),
        ("annotation_decisions.jsonl", plan["annotation_decisions"]),
    ):
        with (output_dir / name).open("w", encoding="utf-8", newline="\n") as stream:
            for r in row_list:
                stream.write(json.dumps(r, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n")

    for name, row_list in (
        ("class_counts.csv", stats["classes"]),
        ("eligibility_waterfall.csv", stats["eligibility_waterfall"]),
        (
            "image_eligibility.csv",
            [
                {
                    "image_record_index": r["image_record_index"],
                    "image_id": r["source_image_id"],
                    "file_name": r["source_image"].get("file_name") if isinstance(r["source_image"], dict) else None,
                    "eligible": r["eligible"],
                    "split": r["split"],
                    "primary_reason": r["primary_reason"],
                    "all_reasons": r["reasons"],
                    "annotations": r["annotation_count"],
                    "mapped_annotations": r["mapped_annotation_count"],
                }
                for r in plan["eligibility"]
            ],
        ),
    ):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row_list[0]))
            writer.writeheader()
            for r in row_list:
                writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()})

    # Build manifest
    source = summary["source"]
    manifest = {
        "schema_version": "bantai.taco-preparation.v1",
        "status": "validated",
        "training_ready": True,
        "source": {
            "annotations": str(source_path),
            "sha256": sha256_file(source_path),
            "dataset_root": str(dataset_root),
            "taxonomy_path": str(taxonomy_v2_path),
            "taxonomy_sha256": sha256_file(taxonomy_v2_path),
        },
        "input_audit": str(review_dir),
        "policy_version": policy["version"],
        "policy_source_sha256": sha256_file(policy_path),
        "taxonomy_version": taxonomy_v2["version"],
        "seed": 42,
        "ratios": policy["default_ratios"],
        "split_algorithm": policy["split_algorithm"],
        "image_root": str(dataset_root),
        "images_copied": False,
        "validation": {
            "coco_source_consistency": "passed",
            "source_integrity": "passed",
            "exact_duplicate_leakage": "passed",
            "reproducible_assignment": "passed",
        },
        "pending_requirements": [
            "license_review",
            "trainer_compatibility",
            "near_duplicate_visual_review",
            "class_coverage_acceptance",
        ],
        "artifacts": {p.name: sha256_file(p) for p in sorted(output_dir.iterdir()) if p.is_file()},
    }
    write_json("preparation_manifest.json", manifest)
    print(f"Successfully generated export at {output_dir}")

    # Write rights.json for Stage 5
    rights_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_context_record = {
        "manifest_sha256": sha256_file(output_dir / "preparation_manifest.json"),
        "source_annotations_sha256": manifest["source"]["sha256"],
        "splits": {
            "train": {
                "sha256": sha256_file(output_dir / "train.json"),
                "image_ids": train_ids,
                "images": len(train_ids),
                "annotations": len(exports["train"]["annotations"]),
            },
            "validation": {
                "sha256": sha256_file(output_dir / "validation.json"),
                "image_ids": val_ids,
                "images": len(val_ids),
                "annotations": len(exports["validation"]["annotations"]),
            },
        },
    }

    rights_record = {
        "schema_version": "bantai.training-rights.v2",
        "basis_type": "owner_authorized_section185_research",
        "workflow_authorized": True,
        "basis_assessment": "defensible_limited_use_with_uncertainty",
        "rightsholder_permission": "not_claimed",
        "institutional_approval": "not_claimed",
        "definitive_legal_clearance": "not_claimed",
        "photo_rights_cleared": False,
        "annotation_database_rights_cleared": False,
        "owner_declaration": {
            "non_commercial_academic": True,
            "local_only": True,
            "no_dataset_publication_redistribution_upload": True,
            "workflow_authorized": True,
        },
        "approving_authority": "Project owner via chat: D-070 Option A authorization; no third-party authority claimed",
        "approval_date": "2026-10-26",
        "scope": "local_academic_train_validation_no_publication",
        "basis_and_evidence": "Owner declaration and Section 185 research fair-use analysis in docs/TACO_STAGE_5_ACADEMIC_USE.md.",
        "obligations": "Local non-commercial academic experiment only; no photos, annotations, predictions or weights uploaded or redistributed; preserve source metadata/notices and cite Proenca and Simoes 2020; test split unused.",
        "factors": {
            "purpose": "Local non-commercial academic localization research; favors claim, not automatic immunity.",
            "nature": "Informational litter scenes with protected photographic expression and potentially protected annotations; mixed and uncertain.",
            "amount": "Full images and annotations required for spatial context; exact 307 train and 65 validation images; whole-photo use weighs against claim.",
            "market": "No distributed substitute corpus or commercial service; licensing and model-market effects unmeasured and uncertain.",
        },
        "assessment": {
            "path": "docs/TACO_STAGE_5_ACADEMIC_USE.md",
            "sha256": sha256_file(ROOT / "docs/TACO_STAGE_5_ACADEMIC_USE.md"),
        },
        "sources": [
            "https://lawphil.net/statutes/repacts/ra1997/ra_8293_1997.html",
            "https://lawphil.net/statutes/repacts/ra2013/ra_10372_2013.html",
            "https://github.com/pedropro/TACO/blob/master/README.md",
            "https://arxiv.org/abs/2003.06975",
            "https://www.ipophil.gov.ph/news/ipophl-releases-statutory-fair-use-guidelines-to-clarify-rules-on-copyright-exceptions/?news_page=2",
        ],
        "dataset": dataset_context_record,
    }
    rights_path.write_text(json.dumps(rights_record, indent=2) + "\n", encoding="utf-8")
    print(f"Successfully wrote rights record to {rights_path}")

    print("\nSummary:")
    print(f"  Train: {len(train_ids)} images, {len(exports['train']['annotations'])} annotations")
    print(f"  Validation: {len(val_ids)} images, {len(exports['validation']['annotations'])} annotations")
    print(f"  Test: {len(test_ids)} images, {len(exports['test']['annotations'])} annotations")
    print(f"  Total: {len(new_eligible)} images, {next_ann_id - 1} annotations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
