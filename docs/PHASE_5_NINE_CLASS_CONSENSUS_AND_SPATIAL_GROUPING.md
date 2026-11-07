# BANTAI Phase 5: 9-Class Taxonomy Expansion, Three-Model Consensus & Spatial Hazard Grouping

> **Scope**: Addition of 9th supervised class (`8: paper`), BANTAI 5-category taxonomy, supervised tiered training, hard-negative box-contrastive fine-tuning (`D-095 Champion`), Three-Model Consensus Engine, Scale-Normalized Disjoint DBSCAN Spatial Grouping ($k \ge 8$ Garbage Pile rule), video frame extraction, workspace cleanup, and Argon website integration (`D-070` through `D-116`).  
> **Status**: Completed & Production-Ready.

---

## Table of Contents
1. [Executive Overview & 9-Class Taxonomy Expansion](#1-executive-overview--9-class-taxonomy-expansion)
2. [Active Supervised Object Taxonomy & Canonical BANTAI Categories](#2-active-supervised-object-taxonomy--canonical-bantai-categories)
3. [Supervised Tiered Training & Hard-Negative Box-Contrastive Fine-Tuning](#3-supervised-tiered-training--hard-negative-box-contrastive-fine-tuning)
4. [Three-Model Consensus Architecture & Clutter Suppression Engine](#4-three-model-consensus-architecture--clutter-suppression-engine)
5. [Scale-Normalized Disjoint DBSCAN Spatial Grouping (D-112 Standard)](#5-scale-normalized-disjoint-dbscan-spatial-grouping-d-112-standard)
6. [Shared Video Frame Extraction (1.0 FPS) & Still-Image Pipeline](#6-shared-video-frame-extraction-10-fps--still-image-pipeline)
7. [Workspace Cleanup & Production Model Checkpoint Preservation (D-113)](#7-workspace-cleanup--production-model-checkpoint-preservation-d-113)
8. [Final 65-Image Validation Benchmark Preservation](#8-final-65-image-validation-benchmark-preservation)
9. [Website & Dual-Model Integration Documentation (Argon)](#9-website--dual-model-integration-documentation-argon)
10. [Phase 5 Milestone Register (D-070 through D-116)](#10-phase-5-milestone-register-d-070-through-d-116)

---

## 1. Executive Overview & 9-Class Taxonomy Expansion

Phase 5 expanded the repository from the legacy 8-class system to the complete **9 active supervised classes** by introducing `8: paper`. This expansion enabled full coverage of street litter across the **5 canonical BANTAI categories** (`Plastic`, `Paper`, `Organic`, `Metal`, `Potentially Hazardous`).

Key Phase 5 Accomplishments:
- Peak validation benchmark records achieved: **Official 9-Specific AP = `0.301350`**, **BANTAI 5-Category AP = `0.497161`**, and **`paper` AP = `0.640610`**.
- Zero lost True Positives across all 65 validation benchmark images (`lost_tps == []`), 77 TPs, 63 FPs, and 36 / 65 clean images.
- Developed the **Three-Model Consensus Engine** fusing `Warm-Start Baseline` (384/576), `Stage 5 Run 1` (256/384), and `D-095 Champion` (384/576), suppressing environmental clutter (shadow rocks, vehicle tires, painted road stripes) while recovering small items like cigarette butts and paper towel balls.
- Implemented **Scale-Normalized Disjoint DBSCAN Spatial Grouping** with the strict $k \ge 8$ Garbage Pile cardinality rule (`D-112`).
- Integrated **1.0 FPS Video Frame Extraction** on `.mp4`/`.mov` media, sharing extracted JPEG frames with Model B (Road Defect Detector) without redundant video decoding.
- Surgically cleaned 128.08 GB of redundant testing files and optimizer states while bitwise preserving 100% of the active production model checkpoints (`D-113`).

---

## 2. Active Supervised Object Taxonomy & Canonical BANTAI Categories

The active supervised object taxonomy is strictly locked to **9 classes** (`0..8`):

| Category ID | Object Class Name | Canonical BANTAI Category | Typical Street Litter Objects |
| :---: | :--- | :--- | :--- |
| `0` | `cigarette` | `Potentially Hazardous` | Cigarette butts, filters, tobacco ends |
| `1` | `plastic_bottle` | `Plastic` | Water bottles, beverage bottles |
| `2` | `plastic_bag` | `Plastic` | Grocery shopping bags, trash bags |
| `3` | `plastic_wrapper` | `Plastic` | Candy wrappers, snack packaging sachets |
| `4` | `plastic_film` | `Plastic` | Clear plastic wrap, cling film, packaging foil |
| `5` | `drink_can` | `Metal` | Aluminum soda/beer cans, tin beverage cans |
| `6` | `plastic_cup` | `Plastic` | Disposable plastic cups, bubble tea cups |
| `7` | `styrofoam_piece`| `Potentially Hazardous` | Expanded polystyrene food containers, cup pieces |
| `8` | `paper` | `Paper` | Paper towels, napkins, receipts, cardboard pieces |

> [!CAUTION]
> **Strict Guardrail**: Do NOT inject, map, or prompt for any unsupervised classes (`battery`, `aerosol`, `broken_glass`, `syringe`, `food_waste`). These classes have insufficient verified training data in the supervised split and are strictly prohibited from active production inference.

---

## 3. Supervised Tiered Training & Hard-Negative Box-Contrastive Fine-Tuning

Training progressed through three disciplined data tiers on the strict-clean split:
1. **Tier 1 (307 train images / 637 boxes)**: Introduced `paper` using verified clean images.
2. **Tier 2 (343 train images / 685 boxes)**: Added sibling-category images without background contamination.
3. **Tier 3 Core (379 train images / 731 boxes)**: Added multi-object street scenes.

### Box-Contrastive Fine-Tuning (`D-095 Champion`)
In `D-095`, fine-tuning incorporated hard-negative box-contrastive loss, pulling matched object queries toward ground-truth features while pushing unmatched anchor queries away. A uniform 50/50 model soup between the best contrastive checkpoint and the Tier 3 checkpoint produced the **`D-095 Champion`** model:
- Parameter Hash: `best-inference`
- 9-Specific AP: **`0.301350`**
- BANTAI 5-Category AP: **`0.497161`**
- Paper AP: **`0.640610`**

---

## 4. Three-Model Consensus Architecture & Clutter Suppression Engine

To achieve high recall on small debris while eliminating false alarms on complex backgrounds, Model A fuses three models:

```
[Extracted Frame / Still Image]
         │
         ├──► Model 1: Warm-Start Baseline (384/576) ──► Conservative container detection
         ├──► Model 2: Stage 5 Run 1 (256/384)       ──► High-recall small-litter detection
         └──► Model 3: D-095 Champion (384/576)      ──► Primary detector & query confidence
                               │
                               ▼
               [Scene Consensus & Tie-Breaks]
               - Rule 7A: Material consistency (Case E: compact paper)
               - Rule 7B: Environmental clutter veto (rocks, tires, stripes)
               - Rule 7B9: Small-paper peer bypass (protects paper towels)
               - Pass 7C: Cross-model small-litter recall recovery
                               │
                               ▼
               [Agnostic NMS (IoU 0.50 + Containment 0.80) & Top-25]
```

---

## 5. Scale-Normalized Disjoint DBSCAN Spatial Grouping (D-112 Standard)

Detections are clustered into spatial hazard regions under the strict D-112 cardinality rule:

- **`Individual Litter` (`individual_litter`)**: $k = 1$ isolated bounding box (`model_score = float(score)`).
- **`Group of Individual Litter` (`group_of_individual_litter`)**: $2 \le k \le 7$ spatially clustered items ($k \in [2, 7]$, `model_score = null`).
- **`Garbage Pile` (`garbage_pile`)**: $k \ge 8$ bounding boxes in a dense, touching/consolidated heap (`model_score = null`).

**Guarantees**:
- **Zero Overlapping Multi-Item Regions**: Disjoint non-overlapping region boxes ($IoU \le 0.05$ and $IoMin < 0.20$).
- **Deterministic Cardinality**: Clusters with $< 8$ items are strictly classified as `Group of Individual Litter`. Only heaps with $k \ge 8$ qualify as `Garbage Pile`.

---

## 6. Shared Video Frame Extraction (1.0 FPS) & Still-Image Pipeline

- **Video Ingestion**: Uploaded videos (`.mp4`, `.mov`, `.avi`, `.mkv`) are sampled at 1.0 FPS into RGB JPEGs (`runs/video_detection/<run_id>/frames/`).
- **Shared Ingestion with Model B**: Both Model A (trash) and Model B (road defects) process the exact same extracted JPEG frames, avoiding redundant video decoding.
- **Still-Image Processing**: Direct evaluation on standalone photographs (e.g. `data/input/For_testing.jpg`).

---

## 7. Workspace Cleanup & Production Model Checkpoint Preservation (D-113)

Under `D-113`, a surgical disk cleanup freed **128.08 GB** across 499 non-production files (`runs/` reduced from 131.89 GB to 6.57 GB; `work/` from 2.72 GB to 4.1 MB) while preserving 100% of the active production model checkpoints:

1. **Primary Champion (`D-095 Champion`, soup 50/50)**:  
   `runs/training/stage5-d095-box-contrastive-20261102-a/checkpoints/best-inference/`
2. **Ancestor Corroborator #1 (`Stage 5 Run 1`, epoch 2)**:  
   `runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/checkpoints/epoch-02/`
3. **Ancestor Corroborator #2 (`Warm-Start Baseline`, step 350 + epoch 2 blend)**:  
   `runs/training/taco-896e2f44146d4d54aee518e6087908ca/checkpoints/` (`step-000350` and `epoch-02`)

---

## 8. Final 65-Image Validation Benchmark Preservation

Evaluated on the frozen 65-image validation benchmark:

| Metric | Measured Value | Benchmark Significance |
| :--- | :---: | :--- |
| **Official 9-Specific AP** | **`0.301350`** | Benchmark High Preserved |
| **BANTAI 5-Category AP** | **`0.497161`** | Benchmark High Preserved |
| **Paper Category AP** | **`0.640610`** | Stage 5 High Preserved |
| **True Positives** | **`77`** | 0 lost TPs across all 65 images |
| **False Positives** | **`63`** | Clean background suppression |
| **Clean Images** | **`36 / 65`** (55.4%) | 0 false alarms on background |

---

## 9. Website & Dual-Model Integration Documentation (Argon)

For complete backend integration recipes (FastAPI/Flask/Django/Celery), copy-pasteable Python classes (`BantaiTrashService`), CLI background invocation, unified JSON response schemas, and municipal hotspot priority scoring formulas, see the [Master Integration Guide](ARGON_WEBSITE_INTEGRATION_GUIDE.md).

---

## 10. Phase 5 Milestone Register (D-070 through D-116)

- **D-070**: Stage 5 authorization granted for 9th supervised class (`paper`) expansion and BANTAI 5-category taxonomy.
- **D-071 to D-076**: Tier 1 and Tier 2 training executed; established baseline paper recall.
- **D-077 to D-085**: Table-driven object registry implemented; extensible candidate class audits confirmed unsupervised classes excluded.
- **D-086 to D-094**: High-resolution training ($384 \times 576$) and multi-epoch hard negative mining.
- **D-095**: Box-contrastive fine-tuning and model soup integration producing `D-095 Champion` ($AP_9 = 0.301350$, $AP_5 = 0.497161$).
- **D-096 to D-103**: Three-Model Consensus engine established; environmental clutter veto heuristics tuned.
- **D-104 to D-105**: Scale-normalized DBSCAN spatial grouping foundation established.
- **D-106**: 1.0 FPS video frame extraction implemented (`Motion_Extract_Testing.mp4`).
- **D-107**: Two-Tier hierarchical spatial grouping implemented; sachet-vs-cigarette tie-break added.
- **D-108**: Dense-group intra-region recall recovery and distinct-item NMS preservation.
- **D-109**: Spatially disjoint pile/group region consolidation; wall/pavement hallucination suppression.
- **D-110**: Standalone input photo detection (`For_testing.jpg`) with camera watermark guardrails.
- **D-111**: Cross-model small-litter recall recovery: recovered 4 crumpled paper towels on `Motion_Extract_Testing_2.mp4` sidewalk.
- **D-112**: Garbage Pile minimum cardinality rule reworked to $k \ge 8$ bounding boxes (`min_pile_items = 8`).
- **D-113**: Surgical cleanup of 128.08 GB redundant experimental files; 100% production weights preserved.
- **D-114**: Comprehensive website & dual-model integration documentation authored for Argon.
- **D-115**: Initial documentation consolidation into master guide.
- **D-116**: Core architecture and governance documentation restored; Phase 1–5 progress organized into 5 consolidated phase documents.
