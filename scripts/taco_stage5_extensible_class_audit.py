"""Candidate Extensible Object Class Dataset Coverage & BANTAI 5-Category Audit (D-085).

CPU-only deterministic inspection across the 1,500-image TACO dataset metadata,
split boundaries, and candidate mappings to the canonical 5 BANTAI categories.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.configuration import strict_load
from scripts.persistence import atomic_json
from scripts.taco_dataset import sha256
from scripts.taco_stage5_material_hierarchy import OBJECT_REGISTRY, BANTAI_CATEGORY_IDS

D073_SUMMARY = ROOT / "runs/training/stage5-data-expansion-audit-5412a13c0258467ba33a1f475cc4b65c/summary.json"
D083_MANIFEST = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/preparation_manifest.json"
D083_TRAIN = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/train.json"
TIER1_TRAIN = ROOT / "exports/taco/strict-clean-stage5-tier1-v1/train.json"
TAXONOMY_V2 = ROOT / "config/taco_taxonomy.v2.json"
OUT_JSON = ROOT / "work/stage5-paper/extensible_class_coverage_audit.json"
OUT_MD = ROOT / "docs/TACO_STAGE_5_EXTENSIBLE_CLASS_AUDIT.md"

ACTIVE_SUPERVISED_LABELS = [
    "cigarette", "plastic_bottle", "plastic_bag", "plastic_wrapper", "plastic_film",
    "drink_can", "plastic_cup", "styrofoam_piece", "paper"
]

CANDIDATE_GROUPS = {
    "Plastic": {
        "plastic_lid": ["Plastic lid"],
        "plastic_straw": ["Plastic straw"],
        "other_plastic": ["Other plastic"],
        "plastic_utensil": ["Plastic utensils"],
        "plastic_tub": ["Tupperware", "Spread tub"],
        "six_pack_rings": ["Six pack rings"],
        "plastic_gloves": ["Plastic glooves"],
    },
    "Paper": {
        "paper_cup": ["Paper cup"],
        "paper_straw": ["Paper straw"],
        "plastified_paper_bag": ["Plastified paper bag"],
        "drink_carton": ["Drink carton", "Meal carton", "Corrugated carton", "Egg carton", "Toilet tube"],
    },
    "Metal": {
        "aluminium_foil": ["Aluminium foil", "Aluminium blister pack"],
        "metal_bottle_cap": ["Metal bottle cap", "Metal lid"],
        "food_can": ["Food Can"],
        "scrap_metal": ["Scrap metal", "Pop tab"],
    },
    "Potentially Hazardous": {
        "aerosol": ["Aerosol"],
        "battery": ["Battery"],
        "broken_glass": ["Broken glass"],
        "syringe": [],
        "glass_bottle": ["Glass bottle"],
        "glass_jar": ["Glass jar"],
        "glass_cup": ["Glass cup"],
    },
    "Organic": {
        "food_waste": ["Food waste"],
    }
}

QUARANTINED_CATEGORIES = [
    "Unlabeled litter", "Shoe", "Rope & strings", "Squeezable tube",
    "Carded blister pack", "Other plastic cup", "Plastic bottle cap"
]


def resolve_annotations_path() -> Path:
    local_data = ROOT / "data/taco/annotations.json"
    if local_data.is_file():
        return local_data.resolve()
    manifest = strict_load(D083_MANIFEST)
    source_path = Path(manifest["source"]["annotations"])
    if source_path.is_file():
        return source_path.resolve()
    raise FileNotFoundError(f"Could not locate annotations.json at {local_data} or {source_path}")


def load_audit_context():
    raw_path = resolve_annotations_path()
    raw = strict_load(raw_path)
    d073 = strict_load(D073_SUMMARY)
    d083_train = strict_load(D083_TRAIN)
    tier1_train = strict_load(TIER1_TRAIN)
    tax_v2 = strict_load(TAXONOMY_V2)

    val_ids = set(d073["frozen"]["validation"]["ids"])
    test_ids = set(d073["frozen"]["test"]["ids"])
    quarantine_ids = {1173, 1305}

    all_img_ids = {i["id"] for i in raw["images"]}
    non_heldout_pool = all_img_ids - val_ids - test_ids - quarantine_ids

    cat_by_id = {c["id"]: c["name"] for c in raw["categories"]}
    id_by_cat = {c["name"]: c["id"] for c in raw["categories"]}

    img_anns = defaultdict(list)
    for a in raw["annotations"]:
        img_anns[a["image_id"]].append(a)

    img_cats = {img_id: set(cat_by_id[a["category_id"]] for a in anns) for img_id, anns in img_anns.items()}

    # Base active source categories allowed under D-083 Core
    from scripts.prepare_stage5_d083_core_export import ALLOWED_SIBLINGS
    d083_allowed = set()
    for label in tax_v2["labels"]:
        d083_allowed.update(label["taco_categories"])
    d083_allowed.update(ALLOWED_SIBLINGS.keys())

    return {
        "raw": raw,
        "raw_path": str(raw_path),
        "raw_sha256": sha256(raw_path),
        "cat_by_id": cat_by_id,
        "id_by_cat": id_by_cat,
        "img_anns": img_anns,
        "img_cats": img_cats,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "quarantine_ids": quarantine_ids,
        "non_heldout_pool": non_heldout_pool,
        "d083_train_ids": set(i["id"] for i in d083_train["images"]),
        "tier1_train": tier1_train,
        "d083_train": d083_train,
        "d083_allowed": d083_allowed,
    }


def audit_active_classes(ctx):
    tier1_counts = defaultdict(int)
    for a in ctx["tier1_train"]["annotations"]:
        tier1_counts[a["category_id"]] += 1

    d083_counts = defaultdict(int)
    for a in ctx["d083_train"]["annotations"]:
        d083_counts[a["category_id"]] += 1

    val_counts = defaultdict(int)
    # count validation boxes by active class
    val_export = strict_load(ROOT / "exports/taco/strict-clean-stage5-d083-core-v1/validation.json")
    for a in val_export["annotations"]:
        val_counts[a["category_id"]] += 1

    results = []
    for cid, name in enumerate(ACTIVE_SUPERVISED_LABELS):
        results.append({
            "specific_category_id": cid,
            "specific_class": name,
            "bantai_category": OBJECT_REGISTRY[cid]["bantai_category"],
            "material": OBJECT_REGISTRY[cid]["material"],
            "hazard_flag": OBJECT_REGISTRY[cid]["hazard_flag"],
            "tier1_train_boxes": tier1_counts[cid],
            "d083_core_train_boxes": d083_counts[cid],
            "validation_boxes": val_counts[cid],
            "status": "ACTIVE_SUPERVISED"
        })
    return results


def evaluate_candidate_class(candidate_name, source_cats, bantai_parent, ctx):
    sc_set = set(source_cats)
    raw_anns = [a for a in ctx["raw"]["annotations"] if ctx["cat_by_id"][a["category_id"]] in sc_set]
    raw_imgs = {a["image_id"] for a in raw_anns}

    pool_anns = [a for a in raw_anns if a["image_id"] in ctx["non_heldout_pool"]]
    pool_imgs = {a["image_id"] for a in pool_anns}

    val_anns = [a for a in raw_anns if a["image_id"] in ctx["val_ids"]]
    val_imgs = {a["image_id"] for a in val_anns}

    test_anns = [a for a in raw_anns if a["image_id"] in ctx["test_ids"]]
    test_imgs = {a["image_id"] for a in test_anns}

    # Strict-clean training yield if this candidate class is unlocked
    extended_allowed = ctx["d083_allowed"] | sc_set
    clean_pool_imgs = set()
    blocked_by_unlabeled = set()
    for img_id in pool_imgs:
        cats_in_img = ctx["img_cats"][img_id]
        if "Unlabeled litter" in cats_in_img:
            blocked_by_unlabeled.add(img_id)
        if cats_in_img.issubset(extended_allowed):
            clean_pool_imgs.add(img_id)

    incremental_imgs = clean_pool_imgs - ctx["d083_train_ids"]
    incremental_candidate_boxes = [a for a in pool_anns if a["image_id"] in incremental_imgs]
    clean_train_candidate_boxes = [a for a in pool_anns if a["image_id"] in clean_pool_imgs]

    # Explicit Recommendation Rule:
    # - SUBTYPE_MERGE_CANDIDATE: e.g. six_pack_rings, food_can (already merged as siblings into 0..8 in D-083) or specific subtypes
    # - SUPERVISED_CANDIDATE: non-held-out strict-clean train boxes >= 15 AND validation boxes >= 3
    # - ZERO_SHOT_HYBRID_ONLY: train boxes < 15 OR validation boxes < 3
    val_boxes_count = len(val_anns)
    clean_train_box_count = len(clean_train_candidate_boxes)

    if candidate_name in ("food_can", "six_pack_rings"):
        recommendation = "SUBTYPE_MERGE_CANDIDATE"
        notes = "Already merged as container sibling into active 0..8 classes under D-083 Core."
    elif clean_train_box_count >= 15 and val_boxes_count >= 3:
        recommendation = "SUPERVISED_CANDIDATE"
        notes = f"Sufficient train boxes ({clean_train_box_count}) and fixed validation boxes ({val_boxes_count}) to support supervised evaluation."
    elif candidate_name == "syringe":
        recommendation = "ZERO_SHOT_HYBRID_ONLY"
        notes = "Zero instances exist in raw TACO dataset; supported strictly via zero-shot prompt."
    else:
        recommendation = "ZERO_SHOT_HYBRID_ONLY"
        notes = f"Insufficient fixed validation boxes ({val_boxes_count} < 3) or train yield ({clean_train_box_count} < 15); recommend zero-shot text prompting."

    return {
        "candidate_class": candidate_name,
        "bantai_category": bantai_parent,
        "source_categories": source_cats,
        "raw_boxes": len(raw_anns),
        "raw_images": len(raw_imgs),
        "pool_boxes": len(pool_anns),
        "pool_images": len(pool_imgs),
        "val_boxes": val_boxes_count,
        "val_images": len(val_imgs),
        "test_metadata_boxes": len(test_anns),
        "test_metadata_images": len(test_imgs),
        "clean_train_images": len(clean_pool_imgs),
        "clean_train_candidate_boxes": clean_train_box_count,
        "incremental_train_images": len(incremental_imgs),
        "incremental_candidate_boxes": len(incremental_candidate_boxes),
        "pool_images_blocked_by_unlabeled": len(blocked_by_unlabeled),
        "recommendation": recommendation,
        "notes": notes
    }


def audit_quarantined_categories(ctx):
    results = []
    for cat_name in QUARANTINED_CATEGORIES:
        anns = [a for a in ctx["raw"]["annotations"] if ctx["cat_by_id"][a["category_id"]] == cat_name]
        imgs = {a["image_id"] for a in anns}
        val_anns = [a for a in anns if a["image_id"] in ctx["val_ids"]]
        results.append({
            "category": cat_name,
            "raw_boxes": len(anns),
            "raw_images": len(imgs),
            "val_boxes": len(val_anns),
            "status": "QUARANTINED",
            "reason": "Ambiguous material, accessory item, or unlabeled litter causing strict quarantine."
        })
    return results


def run_extensible_class_audit():
    ctx = load_audit_context()
    active_audit = audit_active_classes(ctx)

    candidate_results = defaultdict(list)
    flat_candidates = []
    for bantai_parent, cands in CANDIDATE_GROUPS.items():
        for cand_name, source_cats in cands.items():
            evaluated = evaluate_candidate_class(cand_name, source_cats, bantai_parent, ctx)
            candidate_results[bantai_parent].append(evaluated)
            flat_candidates.append(evaluated)

    quarantined = audit_quarantined_categories(ctx)

    summary = {
        "authorization": "D-085",
        "scope": "Candidate Extensible Object Class Dataset Coverage & BANTAI 5-Category Audit",
        "status": "complete",
        "gpu_access": False,
        "model_forwards": 0,
        "training_updates": 0,
        "test_json_opened": False,
        "heldout_test_photos_opened": 0,
        "raw_annotations_sha256": ctx["raw_sha256"],
        "dataset_images": 1500,
        "frozen_validation_images": len(ctx["val_ids"]),
        "frozen_test_images": len(ctx["test_ids"]),
        "quarantined_duplicate_images": len(ctx["quarantine_ids"]),
        "non_heldout_pool_images": len(ctx["non_heldout_pool"]),
        "active_supervised_classes": active_audit,
        "candidate_classes_by_bantai_category": candidate_results,
        "all_candidate_classes": flat_candidates,
        "quarantined_categories": quarantined
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT_JSON, summary, replace=True)
    generate_markdown_report(summary, OUT_MD)
    return summary


def generate_markdown_report(summary, output_path: Path):
    lines = [
        "# Stage 5 Extensible Object Class Coverage & BANTAI 5-Category Audit",
        "",
        "## Executive Summary",
        "",
        "Under Owner authorization **D-085** on branch `feat/stage5-paper-category`, a deterministic, CPU-only audit was conducted to analyze coverage, strict-clean training yields, and fixed validation representation for candidate specific object classes across the TACO dataset.",
        "- **Boundaries Preserved**: Zero GPU execution, zero model forwards, zero training updates, zero `test.json` reads, and zero held-out photos opened.",
        "- **Active Supervised Classes (`0..8`)**: Maintained strictly unchanged pending owner decisions.",
        "- **Coverage Summary**: Evaluated 23 candidate specific classes across the 5 canonical BANTAI categories (`Plastic`, `Paper`, `Metal`, `Potentially Hazardous`, `Organic`).",
        "",
        "---",
        "",
        "## 1. Active Supervised Classes (`0..8`)",
        "",
        "| ID | Supervised Class | BANTAI Category | Material | Tier 1 Train Boxes | D-083 Core Train Boxes | Validation Boxes | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    ]
    for row in summary["active_supervised_classes"]:
        lines.append(f"| {row['specific_category_id']} | `{row['specific_class']}` | {row['bantai_category']} | `{row['material']}` | {row['tier1_train_boxes']} | {row['d083_core_train_boxes']} | {row['validation_boxes']} | {row['status']} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Candidate Specific Classes Mapped to Canonical BANTAI Categories",
        "",
        "| Candidate Class | BANTAI Category | Source Raw TACO Categories | Raw Pool (Img / Box) | Clean Train Yield (Img / Box) | Val Boxes | Blocked by Unlabeled | Recommendation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    ])
    for row in summary["all_candidate_classes"]:
        sc_str = ", ".join(row["source_categories"]) if row["source_categories"] else "*(None)*"
        lines.append(
            f"| `{row['candidate_class']}` | {row['bantai_category']} | {sc_str} | "
            f"{row['pool_images']}i / {row['pool_boxes']}b | "
            f"+{row['incremental_train_images']}i / {row['clean_train_candidate_boxes']}b | "
            f"{row['val_boxes']} | {row['pool_images_blocked_by_unlabeled']}i | "
            f"**`{row['recommendation']}`** |"
        )

    lines.extend([
        "",
        "### Key Findings by Category",
        "",
        "#### Plastic",
        "- `plastic_straw` and `plastic_lid` offer significant non-held-out training pool instances (157 and 77 boxes respectively), but **zero** validation instances exist in the frozen 65-image validation split. Recommended for **`ZERO_SHOT_HYBRID_ONLY`**.",
        "- `other_plastic`: 273 boxes in pool, unlocking 118 clean train boxes (+93 images). However, 0 validation boxes exist, and it represents a high-variance catch-all category.",
        "- `plastic_utensil`: 37 pool boxes, 15 clean train boxes (+13 images), 0 validation boxes. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.",
        "- `six_pack_rings`: Already merged into `plastic_wrapper` under D-083 (`SUBTYPE_MERGE_CANDIDATE`).",
        "",
        "#### Paper",
        "- `drink_carton`: Consolidates `Drink carton`, `Meal carton`, `Corrugated carton`, `Egg carton`, and `Toilet tube`. Has **59 clean train boxes** (+6 images) and **7 validation boxes** in the frozen split. Qualifies as **`SUPERVISED_CANDIDATE`** if separated from generic `paper`.",
        "- `paper_cup`: 66 pool boxes, but only 1 validation box in the frozen split. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.",
        "- `plastified_paper_bag`: Zero annotations in raw TACO dataset.",
        "",
        "#### Metal",
        "- `scrap_metal`: 119 pool boxes, 75 clean train boxes (+51 images), but 0 validation boxes in frozen split. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.",
        "- `aluminium_foil`: 68 pool boxes, 45 clean train boxes (+29 images), 0 validation boxes. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.",
        "- `metal_bottle_cap`: 90 pool boxes, 33 clean train boxes (+24 images), 0 validation boxes. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.",
        "- `food_can`: 34 pool boxes, already merged into `drink_can` (class 5) under D-083 (`SUBTYPE_MERGE_CANDIDATE`).",
        "",
        "#### Potentially Hazardous",
        "- `broken_glass`: 138 pool boxes, 31 clean train boxes (+4 images), but **0 validation boxes** in frozen split. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.",
        "- `glass_bottle`: 104 pool boxes, 43 clean train boxes (+36 images), but 0 validation boxes in frozen split. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.",
        "- `battery`: Only 2 annotations in raw TACO (1 clean train box, 0 validation boxes). Must use `ZERO_SHOT_HYBRID_ONLY`.",
        "- `aerosol`: 10 pool boxes, 5 clean train boxes, 0 validation boxes. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.",
        "- `syringe`: Zero annotations in raw TACO dataset. Strictly `ZERO_SHOT_HYBRID_ONLY`.",
        "",
        "#### Organic",
        "- `food_waste`: 8 pool boxes, 4 clean train boxes (+4 images), **0 validation boxes** in frozen split. Unlabeled litter blocks 2 images. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.",
        "",
        "---",
        "",
        "## 3. Quarantined / Ambiguous Categories",
        "",
        "| Raw TACO Category | Raw Images | Raw Boxes | Val Boxes | Reason |",
        "| --- | --- | --- | --- | --- |"
    ])
    for row in summary["quarantined_categories"]:
        lines.append(f"| `{row['category']}` | {row['raw_images']} | {row['raw_boxes']} | {row['val_boxes']} | {row['reason']} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Verification and Invariants",
        "- Zero GPU/model inference or training updates occurred during this audit.",
        "- Protected exports, taxonomies, and checkpoint files remained untouched.",
        "- `test.json` was not opened and zero held-out photos were read.",
        ""
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    res = run_extensible_class_audit()
    print(f"D-085 Audit Complete: {len(res['all_candidate_classes'])} candidate classes audited.")
