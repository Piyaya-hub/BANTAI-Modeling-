# Stage 5 Extensible Object Class Coverage & BANTAI 5-Category Audit

## Executive Summary

Under Owner authorization **D-085** on branch `feat/stage5-paper-category`, a deterministic, CPU-only audit was conducted to analyze coverage, strict-clean training yields, and fixed validation representation for candidate specific object classes across the TACO dataset.
- **Boundaries Preserved**: Zero GPU execution, zero model forwards, zero training updates, zero `test.json` reads, and zero held-out photos opened.
- **Active Supervised Classes (`0..8`)**: Maintained strictly unchanged pending owner decisions.
- **Coverage Summary**: Evaluated 23 candidate specific classes across the 5 canonical BANTAI categories (`Plastic`, `Paper`, `Metal`, `Potentially Hazardous`, `Organic`).

---

## 1. Active Supervised Classes (`0..8`)

| ID | Supervised Class | BANTAI Category | Material | Tier 1 Train Boxes | D-083 Core Train Boxes | Validation Boxes | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `cigarette` | Potentially Hazardous | `cellulose_acetate_filter` | 139 | 147 | 26 | ACTIVE_SUPERVISED |
| 1 | `plastic_bottle` | Plastic | `plastic` | 48 | 48 | 9 | ACTIVE_SUPERVISED |
| 2 | `plastic_bag` | Plastic | `plastic` | 17 | 22 | 4 | ACTIVE_SUPERVISED |
| 3 | `plastic_wrapper` | Plastic | `plastic` | 72 | 74 | 13 | ACTIVE_SUPERVISED |
| 4 | `plastic_film` | Plastic | `plastic` | 125 | 133 | 17 | ACTIVE_SUPERVISED |
| 5 | `drink_can` | Metal | `metal` | 47 | 59 | 7 | ACTIVE_SUPERVISED |
| 6 | `plastic_cup` | Plastic | `plastic` | 24 | 54 | 6 | ACTIVE_SUPERVISED |
| 7 | `styrofoam_piece` | Plastic | `polystyrene_foam` | 41 | 54 | 7 | ACTIVE_SUPERVISED |
| 8 | `paper` | Paper | `paper_fiber` | 124 | 140 | 20 | ACTIVE_SUPERVISED |

---

## 2. Candidate Specific Classes Mapped to Canonical BANTAI Categories

| Candidate Class | BANTAI Category | Source Raw TACO Categories | Raw Pool (Img / Box) | Clean Train Yield (Img / Box) | Val Boxes | Blocked by Unlabeled | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `plastic_lid` | Plastic | Plastic lid | 69i / 77b | +29i / 31b | 0 | 19i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `plastic_straw` | Plastic | Plastic straw | 110i / 157b | +48i / 52b | 0 | 26i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `other_plastic` | Plastic | Other plastic | 171i / 273b | +93i / 118b | 0 | 45i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `plastic_utensil` | Plastic | Plastic utensils | 29i / 37b | +13i / 15b | 0 | 9i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `plastic_tub` | Plastic | Tupperware, Spread tub | 13i / 13b | +4i / 11b | 0 | 2i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `six_pack_rings` | Plastic | Six pack rings | 4i / 5b | +1i / 1b | 0 | 2i | **`SUBTYPE_MERGE_CANDIDATE`** |
| `plastic_gloves` | Plastic | Plastic glooves | 4i / 4b | +4i / 4b | 0 | 0i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `paper_cup` | Paper | Paper cup | 61i / 66b | +3i / 16b | 1 | 22i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `paper_straw` | Paper | Paper straw | 3i / 3b | +0i / 0b | 0 | 1i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `plastified_paper_bag` | Paper | Plastified paper bag | 0i / 0b | +0i / 0b | 0 | 0i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `drink_carton` | Paper | Drink carton, Meal carton, Corrugated carton, Egg carton, Toilet tube | 102i / 136b | +6i / 59b | 7 | 23i | **`SUPERVISED_CANDIDATE`** |
| `aluminium_foil` | Metal | Aluminium foil, Aluminium blister pack | 49i / 68b | +29i / 45b | 0 | 9i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `metal_bottle_cap` | Metal | Metal bottle cap, Metal lid | 59i / 90b | +24i / 33b | 0 | 14i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `food_can` | Metal | Food Can | 20i / 34b | +1i / 11b | 0 | 0i | **`SUBTYPE_MERGE_CANDIDATE`** |
| `scrap_metal` | Metal | Scrap metal, Pop tab | 83i / 119b | +51i / 75b | 0 | 14i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `aerosol` | Potentially Hazardous | Aerosol | 10i / 10b | +5i / 5b | 0 | 1i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `battery` | Potentially Hazardous | Battery | 2i / 2b | +1i / 1b | 0 | 0i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `broken_glass` | Potentially Hazardous | Broken glass | 15i / 138b | +4i / 31b | 0 | 8i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `syringe` | Potentially Hazardous | *(None)* | 0i / 0b | +0i / 0b | 0 | 0i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `glass_bottle` | Potentially Hazardous | Glass bottle | 74i / 104b | +36i / 43b | 0 | 23i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `glass_jar` | Potentially Hazardous | Glass jar | 4i / 6b | +0i / 0b | 0 | 0i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `glass_cup` | Potentially Hazardous | Glass cup | 4i / 6b | +2i / 2b | 0 | 0i | **`ZERO_SHOT_HYBRID_ONLY`** |
| `food_waste` | Organic | Food waste | 7i / 8b | +4i / 4b | 0 | 2i | **`ZERO_SHOT_HYBRID_ONLY`** |

### Key Findings by Category

#### Plastic
- `plastic_straw` and `plastic_lid` offer significant non-held-out training pool instances (157 and 77 boxes respectively), but **zero** validation instances exist in the frozen 65-image validation split. Recommended for **`ZERO_SHOT_HYBRID_ONLY`**.
- `other_plastic`: 273 boxes in pool, unlocking 118 clean train boxes (+93 images). However, 0 validation boxes exist, and it represents a high-variance catch-all category.
- `plastic_utensil`: 37 pool boxes, 15 clean train boxes (+13 images), 0 validation boxes. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.
- `six_pack_rings`: Already merged into `plastic_wrapper` under D-083 (`SUBTYPE_MERGE_CANDIDATE`).

#### Paper
- `drink_carton`: Consolidates `Drink carton`, `Meal carton`, `Corrugated carton`, `Egg carton`, and `Toilet tube`. Has **59 clean train boxes** (+6 images) and **7 validation boxes** in the frozen split. Qualifies as **`SUPERVISED_CANDIDATE`** if separated from generic `paper`.
- `paper_cup`: 66 pool boxes, but only 1 validation box in the frozen split. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.
- `plastified_paper_bag`: Zero annotations in raw TACO dataset.

#### Metal
- `scrap_metal`: 119 pool boxes, 75 clean train boxes (+51 images), but 0 validation boxes in frozen split. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.
- `aluminium_foil`: 68 pool boxes, 45 clean train boxes (+29 images), 0 validation boxes. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.
- `metal_bottle_cap`: 90 pool boxes, 33 clean train boxes (+24 images), 0 validation boxes. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.
- `food_can`: 34 pool boxes, already merged into `drink_can` (class 5) under D-083 (`SUBTYPE_MERGE_CANDIDATE`).

#### Potentially Hazardous
- `broken_glass`: 138 pool boxes, 31 clean train boxes (+4 images), but **0 validation boxes** in frozen split. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.
- `glass_bottle`: 104 pool boxes, 43 clean train boxes (+36 images), but 0 validation boxes in frozen split. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.
- `battery`: Only 2 annotations in raw TACO (1 clean train box, 0 validation boxes). Must use `ZERO_SHOT_HYBRID_ONLY`.
- `aerosol`: 10 pool boxes, 5 clean train boxes, 0 validation boxes. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.
- `syringe`: Zero annotations in raw TACO dataset. Strictly `ZERO_SHOT_HYBRID_ONLY`.

#### Organic
- `food_waste`: 8 pool boxes, 4 clean train boxes (+4 images), **0 validation boxes** in frozen split. Unlabeled litter blocks 2 images. Recommendation: `ZERO_SHOT_HYBRID_ONLY`.

---

## 3. Quarantined / Ambiguous Categories

| Raw TACO Category | Raw Images | Raw Boxes | Val Boxes | Reason |
| --- | --- | --- | --- | --- |
| `Unlabeled litter` | 269 | 517 | 0 | Ambiguous material, accessory item, or unlabeled litter causing strict quarantine. |
| `Shoe` | 6 | 7 | 0 | Ambiguous material, accessory item, or unlabeled litter causing strict quarantine. |
| `Rope & strings` | 28 | 29 | 0 | Ambiguous material, accessory item, or unlabeled litter causing strict quarantine. |
| `Squeezable tube` | 6 | 7 | 0 | Ambiguous material, accessory item, or unlabeled litter causing strict quarantine. |
| `Carded blister pack` | 1 | 1 | 0 | Ambiguous material, accessory item, or unlabeled litter causing strict quarantine. |
| `Other plastic cup` | 2 | 2 | 0 | Ambiguous material, accessory item, or unlabeled litter causing strict quarantine. |
| `Plastic bottle cap` | 185 | 209 | 0 | Ambiguous material, accessory item, or unlabeled litter causing strict quarantine. |

---

## 4. Verification and Invariants
- Zero GPU/model inference or training updates occurred during this audit.
- Protected exports, taxonomies, and checkpoint files remained untouched.
- `test.json` was not opened and zero held-out photos were read.
