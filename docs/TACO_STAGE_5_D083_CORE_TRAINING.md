# Stage 5 D-083 Core Fine-Tuning and Resolution Evaluation Report

## Executive Summary

Under Owner authorization **D-083** on branch `feat/stage5-paper-category`, a controlled Stage 5 evaluation and fine-tuning sequence was executed:
1. **Taxonomy & Boundary Preservation**: Zero unaudited classes added. Active supervised classes remain strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can` [excluding Aerosol], `plastic_cup`, `styrofoam_piece`, and `paper` strictly mapped from Tier 1 baseline without any new paper source categories).
2. **Extensible Object -> BANTAI 5-Category Registry**: `scripts/taco_stage5_material_hierarchy.py` refactored to provide a table-driven `OBJECT_REGISTRY` and `register_object_class()` mechanism for future audited classes, while keeping the 9 active supervised classes and viewer/evaluator workflows fully compatible.
3. **Part A — High-Resolution Inference Evaluation (`448/672`) on Current Champion (`soup_tier2_d074_champion_5050`)**: Bounded 112 validation forwards (65 `combined_65` + 47 `legacy_47` 8-class prompt pass) at `shortest=448, longest=672` (672 square pad, FP32, TF32 disabled).
4. **Part B — 1-Epoch Controlled Training on `strict-clean-stage5-d083-core-v1`**:
   - Training export: 379 images / 731 boxes (336 Tier 1 strict-clean images + audited non-hazardous/non-aerosol/non-paper container siblings). Train and validation photo SHA-256 digests bound in `work/stage5-paper/rights_d083_core.json`.
   - Warm-start from D074/D076 champion parameter hash `cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422`.
   - 1 bounded epoch of 379 updates at `384/576`, batch size 1, AdamW `lr=1.5e-6, weight_decay=1e-4`, Swin-T and BERT frozen.
   - 3-state evaluation (336 validation forwards total: `D083-Core E1`, `soup_d083_d074_5050`, and `soup_d083_d076_5050`).
   - Champion selected: `soup_d083_d074_5050` achieves **0.236129** 9-specific AP and **0.252820** legacy 8-class AP.

---

## Part A: High-Resolution Evaluation (`448/672` vs `384/576`)

- **Model**: `soup_tier2_d074_champion_5050` (`best-inference`, parameter SHA-256 `cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422`).
- **Validation Budget**: Exactly 112 forwards (65 `combined_65` + 47 `legacy_47`).

### Metric Comparison

| Benchmark / Metric | 384/576 Resolution | 448/672 Resolution | Delta |
| --- | --- | --- | --- |
| **Official 9-Specific `combined_65` AP** | **0.235994** | 0.217783 | -0.018211 |
| 9-Specific AP50 | 0.282890 | 0.258814 | -0.024076 |
| 9-Specific AP75 | 0.258859 | 0.238478 | -0.020381 |
| 9-Specific AR100 | 0.379813 | 0.354020 | -0.025793 |
| **Official 8-Class `legacy_47` AP** | **0.250362** | 0.244212 | -0.006150 |
| Legacy 8-Class AP50 | 0.297426 | 0.291771 | -0.005655 |
| Legacy 8-Class AR100 | 0.407000 | 0.395000 | -0.012000 |
| **BANTAI 5-Category AP (raw)** | **0.400957** | 0.361760 | -0.039197 |
| BANTAI 5-Category AP50 | 0.473523 | 0.435728 | -0.037795 |
| BANTAI 5-Category AR100 | 0.540182 | 0.472727 | -0.067455 |

### Scale Bucket Progression (9-Specific `combined_65`)

At `448/672`, smaller objects benefited substantially from higher pixel sampling:
- `AP_small`: rose from **`0.059641`** (`384/576`) to **`0.073440`** (`448/672`)
- `AP_medium`: rose from **`0.063247`** (`384/576`) to **`0.184431`** (`448/672`)

### Per-Class AP Comparison (9-Specific)

| Class | 384/576 AP | 448/672 AP | Delta |
| --- | --- | --- | --- |
| `cigarette` | 0.111849 | 0.110532 | -0.001317 |
| `plastic_bottle` | 0.331400 | 0.453533 | +0.122132 |
| `plastic_bag` | 0.000000 | 0.000000 | 0.000000 |
| `plastic_wrapper` | 0.015842 | 0.019519 | +0.003678 |
| `plastic_film` | 0.042013 | 0.066377 | +0.024364 |
| `drink_can` | 0.656436 | 0.545545 | -0.110891 |
| `plastic_cup` | 0.395545 | 0.319802 | -0.075743 |
| `styrofoam_piece` | 0.148515 | 0.074257 | -0.074257 |
| `paper` | 0.422343 | 0.370481 | -0.051862 |

*Note on Recall & Legacy 8-Class*: For `cigarette`, 9-specific `AR100` rose from `0.380769` (`384/576`) to `0.396154` (`448/672`), and on `legacy_47` its AP rose from `0.135895` to `0.140744`. However, large container categories trained strictly at `384/576` experienced spatial/score miscalibration, leading to an overall AP drop. Training and inference baseline remains pinned to `384/576`.

---

## Part B: Controlled 1-Epoch Fine-Tuning & Weight Soup Evaluation

- **Dataset**: `exports/taco/strict-clean-stage5-d083-core-v1` (379 training images / 731 boxes; 65 validation images / 109 boxes).
- **Optimization**: 379 updates, AdamW `lr=1.5e-6, weight_decay=1e-4`, batch size 1, clip norm 1.0, FP32, TF32 off. Swin-T and BERT frozen (`frozen_parameter_sha256: 161c9814bffe1657ae0debd7e21b7ed1c41f29f3ba43fffe879ab31ba36a0b67`).
- **Peak VRAM**: 3641 MiB (GTX 1660 Ti).
- **Checkpoints**: Validated and verified at `step-000100`, `step-000200`, `step-000300`, and `epoch-01`.

### Evaluation Across Candidate States (336 Validation Forwards)

| Model State | Recipe | Combined 65 AP | Combined 65 AR100 | Legacy 47 AP | Legacy 47 AP50 | Legacy 47 AR100 | Mean Val Loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`D083-Core E1`** | Single epoch endpoint | 0.233825 | 0.375472 | **0.255879** | 0.301468 | 0.390262 | **3.949181** |
| **`soup_d083_d074_5050`** | `0.5*D074 + 0.5*D083-E1` | **0.236129** | **0.378128** | 0.252820 | 0.298086 | 0.389300 | 3.985499 |
| **`soup_d083_d076_5050`** | `0.5*D076 + 0.5*D083-E1` | 0.232746 | 0.373000 | 0.253044 | 0.297823 | 0.389741 | 3.963570 |

### Per-Class AP Comparison for Winning Champion (`soup_d083_d074_5050` vs D076 Reference)

| Supervised Class | D076 Reference AP | `soup_d083_d074_5050` AP | Delta | Legacy 47 AP (`soup_d083_d074_5050`) |
| --- | --- | --- | --- | --- |
| `cigarette` | 0.111849 | 0.111154 | -0.000695 | 0.131154 |
| `plastic_bottle` | 0.331400 | 0.331400 | 0.000000 | 0.292079 |
| `plastic_bag` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `plastic_wrapper` | 0.015842 | 0.015842 | 0.000000 | 0.202970 (`0.241584` in `D083-Core E1`) |
| `plastic_film` | 0.042013 | 0.045093 | +0.003080 | 0.056752 |
| `drink_can` | 0.656436 | 0.656436 | 0.000000 | 0.656436 |
| `plastic_cup` | 0.395545 | 0.395545 | 0.000000 | 0.475248 |
| `styrofoam_piece` | 0.148515 | 0.148515 | 0.000000 | 0.207921 |
| `paper` | 0.422343 | 0.421176 | -0.001167 | N/A (8-class legacy) |
| **Overall AP** | **0.235994** | **0.236129** | **+0.000135** | **0.252820** |

### BANTAI 5-Category & Score-Preserving Consensus Audit

Evaluated on `soup_d083_d074_5050` predictions (`runs/training/stage5-d083-core-20261031-a/d079-score-preserving-consensus.json`):

- **Raw BANTAI 5-Category**:
  - AP = **`0.401563`**, AP50 = **`0.495005`**, AR100 = **`0.594712`**
  - Per-category: `Plastic = 0.417487`, `Paper = 0.421176`, `Metal = 0.656436`, `Potentially Hazardous = 0.111154`, `Organic = null`
- **Consensus BANTAI 5-Category (Image 777 paper -> plastic subtype tie-break)**:
  - AP = **`0.405701`**, AP50 = **`0.500456`** (50.05% AP50), AR100 = **`0.598729`**
  - Per-category: `Plastic = 0.430832`, `Paper = 0.424384`, `Metal = 0.656436`, `Potentially Hazardous = 0.111154`, `Organic = null`
- **Consensus 9-Specific**:
  - AP = **`0.236393`**, AP50 = **`0.282824`**, AP75 = **`0.259271`**

- **Inference Bundle**: Persisted at `runs/training/stage5-d083-core-20261031-a/checkpoints/best-inference` with parameter SHA-256 `82c0a5bf3a104db5f5ff35e1cbfc1bd239595ec9946e22cf085986985d95fddc`.
- **Score Vectors**: Saved at `best-score-vectors.npz` (`(65, 900, 9)` float32 array).

---

## Verification & Integrity Proofs
- **Protected Snapshots**: Pre-run and post-run protected snapshots match identically (`integrity-after-training.json`). Zero protected core source or taxonomy files modified. Zero test split reads (`test.json` unopened, 0 held-out test photos accessed).
- **Guarded Test Suite**: 333 tests run, 317 passed, 16 documented skips, 0 failures, 0 errors. Zero unauthorized GPU/detector calls or optimizer updates during testing.
- **Native Tk Verification**: Passed 50 combinations on Image 777 with zero errors and unchanged source hashes.
- **Markdown Links & Git Check**: 64 markdown files / 515 local links verified with 0 broken links; `git diff --check` clean.
