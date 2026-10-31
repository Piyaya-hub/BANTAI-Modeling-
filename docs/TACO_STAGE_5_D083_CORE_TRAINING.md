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

### Per-Class AP Comparison (9-Specific)

| Class | 384/576 AP | 448/672 AP | Delta |
| --- | --- | --- | --- |
| `cigarette` | 0.111849 | 0.124503 | +0.012654 |
| `plastic_bottle` | 0.331400 | 0.316832 | -0.014568 |
| `plastic_bag` | 0.000000 | 0.000000 | 0.000000 |
| `plastic_wrapper` | 0.015842 | 0.009901 | -0.005941 |
| `plastic_film` | 0.042013 | 0.044569 | +0.002556 |
| `drink_can` | 0.656436 | 0.603960 | -0.052476 |
| `plastic_cup` | 0.222772 | 0.168317 | -0.054455 |
| `styrofoam_piece` | 0.000000 | 0.000000 | 0.000000 |
| `paper` | 0.743630 | 0.691964 | -0.051666 |

*Observation*: While small object localization (`cigarette`, `plastic_film`) modestly improved at `448/672`, larger and medium objects suffered significant confidence/spatial degradation due to training resolution mismatch (`384/576`), resulting in an overall AP drop of -0.0182 on 9-specific and -0.0392 on BANTAI 5-Category. The primary resolution for training and evaluation remains `384/576`.

---

## Part B: Controlled 1-Epoch Fine-Tuning & Weight Soup Evaluation

- **Dataset**: `exports/taco/strict-clean-stage5-d083-core-v1` (379 training images / 731 boxes; 65 validation images / 109 boxes).
- **Optimization**: 379 updates, AdamW `lr=1.5e-6, weight_decay=1e-4`, batch size 1, clip norm 1.0, FP32, TF32 off. Swin-T and BERT frozen (`frozen_parameter_sha256: 161c9814bffe1657ae0debd7e21b7ed1c41f29f3ba43fffe879ab31ba36a0b67`).
- **Peak VRAM**: 3641 MiB (GTX 1660 Ti).
- **Checkpoints**: Validated and verified at `step-000100`, `step-000200`, `step-000300`, and `epoch-01`.

### Evaluation Across Candidate States (336 Validation Forwards)

| Model State | Recipe | Combined 65 AP | Combined 65 AR100 | Legacy 47 AP | Legacy 47 AR100 | Mean Val Loss |
| --- | --- | --- | --- | --- | --- | --- |
| **`D083-Core E1`** | Single epoch endpoint | 0.233825 | 0.380720 | **0.255879** | 0.407000 | 4.96024 |
| **`soup_d083_d074_5050`** | `0.5*D074 + 0.5*D083-E1` | **0.236129** | **0.380720** | 0.252820 | 0.407000 | **4.80806** |
| **`soup_d083_d076_5050`** | `0.5*D076 + 0.5*D083-E1` | 0.232746 | 0.379813 | 0.253044 | 0.407000 | 4.82148 |

### Per-Class AP Comparison for Winning Champion (`soup_d083_d074_5050`)

| Supervised Class | D076 Reference AP | `soup_d083_d074_5050` AP | Delta |
| --- | --- | --- | --- |
| `cigarette` | 0.111849 | 0.111849 | 0.000000 |
| `plastic_bottle` | 0.331400 | 0.331400 | 0.000000 |
| `plastic_bag` | 0.000000 | 0.000000 | 0.000000 |
| `plastic_wrapper` | 0.015842 | 0.017059 | +0.001217 |
| `plastic_film` | 0.042013 | 0.042013 | 0.000000 |
| `drink_can` | 0.656436 | 0.656436 | 0.000000 |
| `plastic_cup` | 0.222772 | 0.222772 | 0.000000 |
| `styrofoam_piece` | 0.000000 | 0.000000 | 0.000000 |
| `paper` | 0.743630 | 0.743630 | 0.000000 |
| **Overall AP** | **0.235994** | **0.236129** | **+0.000135** |

### Material Hierarchy and Consensus Audit
- **Diagnostic BANTAI 5-Category AP (raw)**: `0.400957` (matches baseline).
- **D-079 Score-Preserving Plastic Consensus**: Verified on Image 777 (q0 remapped from Paper to `plastic_film`/Plastic while retaining score `0.307818`).
- **Inference Bundle**: Persisted at `runs/training/stage5-d083-core-20261031-a/checkpoints/best-inference` with parameter SHA-256 `82c0a5bf3a104db5f5ff35e1cbfc1bd239595ec9946e22cf085986985d95fddc`.
- **Score Vectors**: Saved at `best-score-vectors.npz` (`(65, 900, 9)` float32 array).

---

## Verification & Integrity Proofs
- **Protected Snapshots**: Pre-run and post-run protected snapshots match identically (`integrity-after-training.json`). Zero protected core source or taxonomy files modified. Zero test split reads (`test.json` unopened, 0 held-out test photos accessed).
- **Guarded Test Suite**: 333 tests run, 317 passed, 16 documented skips, 0 failures, 0 errors. Zero unauthorized GPU/detector calls or optimizer updates during testing.
- **Native Tk Verification**: Passed 50 combinations on Image 777 with zero errors and unchanged source hashes.
- **Markdown Links & Git Check**: 64 markdown files / 513 local links verified with 0 broken links; `git diff --check` clean.
