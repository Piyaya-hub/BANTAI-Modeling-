# Stage 5 Tri-Model Class-Specialized Query Routing & Paper-Plastic Disambiguation Audit (D-090)

## 1. Executive Summary & Authorization

Under owner authorization **D-090** on branch `feat/stage5-paper-category`, a strictly CPU-only **Tri-Model Class-Specialized Query Routing and Paper-vs-Plastic Disambiguation Audit** was executed. The task combined the complementary strengths of three distinct model states across our 9 active supervised classes (`0..8`):
1. **`D086_Soup2575` (`384/576`)**: Standard base queries delivering peak localization on `styrofoam_piece` (`0.148515`) and `plastic_cup` (`0.395545`).
2. **`D086_HR448` (`448/672`)**: High-resolution queries providing fine-grained detection for small/medium objects: `cigarette` (`0.123328`), `plastic_bottle` (`0.419554`), `plastic_wrapper` (`0.041867`), and `plastic_film` (`0.068693`).
3. **`D089_S5050` (`384/576`, Tier 3 Core 576 images / 1,240 boxes)**: Expanded-dataset queries specializing in non-plastic objects: elevating `drink_can` to **`0.680693`** and `paper` to **`0.467871`**!

Combining these models via class-calibrated routing and D-079 pre-NMS consensus tiebreak establishes an **all-time high across all metrics**:
- **Official 9-Specific Validation AP**: **`0.260674`** (AP50 `0.310330`, AR100 `0.418648`) — beating D-088 (`0.258410`) by `+0.002264` and D-083 single-scale champion (`0.236129`) by `+0.024545`.
- **Diagnostic BANTAI 5-Category Validation AP**: **`0.435166`** (AP50 `0.532717` / **`53.27% AP50`**, AR100 `0.653894`) — beating D-088 (`0.430072`) by `+0.005094`.

All operations were strictly CPU-only: zero GPU calls, zero model forwards, zero training updates, zero `test.json` reads, and zero held-out test photo opens occurred.

---

## 2. Quantitative Comparison Across Model Milestones

### Combined 65-Image Validation Results (9-Specific & BANTAI 5-Category)

| Metric | D-083 Champion (384) | D-086 HR448 (448) | D-088 Calibrated Fusion | D-089 Tier 3 Core (384) | D-090 Tri-Model Champion | D-090 vs D-083 | D-090 vs D-088 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **9-Specific AP** | `0.236129` | `0.207802` | `0.258410` | `0.233812` | **`0.260674`** | **+0.024545** | **+0.002264** |
| **9-Specific AP50** | `0.287235` | `0.264627` | `0.306843` | `0.278686` | **`0.310330`** | **+0.023095** | **+0.003487** |
| **9-Specific AR100** | `0.370594` | `0.368731` | `0.415315` | `0.371596` | **`0.418648`** | **+0.048054** | **+0.003333** |
| **BANTAI 5-Cat AP** | `0.401563` | `0.380755` | `0.430072` | `0.398774` | **`0.435166`** | **+0.033603** | **+0.005094** |
| **BANTAI 5-Cat AP50**| `0.500539` | `0.485304` | `0.524870` | `0.484210` | **`0.532717`** | **+0.032178** | **+0.007847** |
| **BANTAI 5-Cat AR100**|`0.589815` | `0.601951` | `0.646394` | `0.606277` | **`0.653894`** | **+0.064079** | **+0.007500** |

---

## 3. Per-Class 9-Specific AP Progression

| Class ID & Name | D-083 Core Champion | D-086 HR448 E1 | D-088 Calibrated Fusion | D-089 Tier 3 Core | D-090 Tri-Model Fusion | Primary Query Source |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `0: cigarette` | `0.111849` | `0.123328` | `0.123328` | `0.107777` | **`0.123328`** | `D086_HR448` |
| `1: plastic_bottle` | `0.406931` | `0.490219` | `0.419554` | `0.399587` | **`0.419554`** | `D086_HR448` + `std` |
| `2: plastic_bag` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | **`0.000000`** | (No distinct isolated GT) |
| `3: plastic_wrapper` | `0.038167` | `0.041867` | `0.041867` | `0.035414` | **`0.041867`** | `D086_HR448` |
| `4: plastic_film` | `0.050735` | `0.071569` | `0.068693` | `0.048839` | **`0.068693`** | `D086_HR448` |
| `5: drink_can` | `0.656436` | `0.485149` | `0.680693` | `0.670297` | **`0.680693`** | `D089_S5050` + `D086` |
| `6: plastic_cup` | `0.395545` | `0.357673` | `0.395545` | `0.380446` | **`0.395545`** | `D086_S2575` (std-only) |
| `7: styrofoam_piece` | `0.148515` | `0.074257` | `0.148515` | `0.133663` | **`0.148515`** | `D086_S2575` (std-only) |
| `8: paper` | `0.316982` | `0.226159` | `0.447498` | `0.429627` | **`0.467871`** | `D089_S5050` (Tier 3 Core) |
| **Macro Average (mAP)** | `0.236129` | `0.207802` | `0.258410` | `0.233812` | **`0.260674`** | **Tri-Model Champion** |

---

## 4. BANTAI 5-Category Breakdown

| BANTAI Category | Mapped Specific Classes | Objects / Images | AP | AR100 |
| :--- | :--- | :---: | :---: | :---: |
| **Plastic** | `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `plastic_cup` | 56 / 50 | **`0.468770`** | `0.751786` |
| **Paper** | `paper` | 20 / 18 | **`0.467871`** | `0.755000` |
| **Metal** | `drink_can` | 7 / 7 | **`0.680693`** | `0.685714` |
| **Potentially Hazardous** | `cigarette` | 26 / 9 | **`0.123328`** | `0.423077` |
| **Organic** | (Unsupervised in Stage 5) | 0 / 0 | `None` | `None` |
| **Overall BANTAI 5-Category** | All active categories | 109 / 65 | **`0.435166`** | **`0.653894`** |

---

## 5. Paper-vs-Plastic Disambiguation & Cross-Class Suppression Audit

During D-089 analysis, visual inspection of Image 1 (`000008.jpg`) and Image 777 (`000096.jpg`) revealed overlapping high-confidence predictions across paper and plastic categories (e.g. `paper` and `plastic_film` on the same physical object).

To test whether cross-class IoU suppression between paper and plastic improves performance, an exhaustive audit evaluated cross-class suppression across varying IoU thresholds:

| Cross-Class IoU Threshold | Suppressed Queries | 9-Specific AP | BANTAI 5-Category AP | Finding / Outcome |
| :---: | :---: | :---: | :---: | :--- |
| **None (Class-Aware NMS)** | **0** | **`0.260674`** | **`0.435166`** | **Optimal**: Preserves true positives for each independent class |
| `0.90` | 69 | `0.257156` | `0.409050` | Degradation: `-0.026116` BANTAI AP |
| `0.85` | 73 | `0.257157` | `0.408522` | Degradation: `-0.026644` BANTAI AP |
| `0.80` | 79 | `0.257160` | `0.408523` | Degradation: `-0.026643` BANTAI AP |
| `0.75` | 82 | `0.257160` | `0.408523` | Degradation: `-0.026643` BANTAI AP |
| `0.70` | 86 | `0.257160` | `0.408523` | Degradation: `-0.026643` BANTAI AP |

### Key Takeaway:
Cross-class IoU suppression between paper and plastic **degrades** validation AP by 2.6% (dropping BANTAI AP from `0.435166` to `0.409050`). In real-world litter scenes, plastic and paper frequently co-occur (e.g. plastic film inside a paper carton, or a paper label wrapped around a plastic container). Class-aware independent NMS allows each category to compete fairly against ground truth, yielding the highest accuracy.

---

## 6. Qualitative Inspections: Image 1 and Image 777

### Image 1 (`000008.jpg` - Two Paper Cartons)
- **Ground Truth**:
  - GT #1: `[1.0, 457.0, 1429.0, 1519.0]` (large paper box)
  - GT #2: `[531.0, 292.0, 1006.0, 672.0]` (medium paper box)
- **D-090 Tri-Model Detection**:
  - `qid=1`: `cat=8 (paper)`, score **`0.3381`**, bbox `[0.7, 458.1, 1432.7, 1516.9]` from `D089` source — **clean localization of GT #1 as paper**!
  - `qid=3`: `cat=3 (plastic_wrapper)`, score `0.1844`, bbox `[526.2, 280.6, 1007.9, 676.9]` from `std` source (co-activates `plastic_bag` 0.1509, `plastic_wrapper` 0.1908, `plastic_film` 0.1506, `paper` 0.1052).

### Image 777 (`000096.jpg` - Multi-Litter Scene)
- **Ground Truth**: 1 `plastic_wrapper` `[975, 81, 116, 263]` + 7 `cigarette` butts.
- **D-090 Tri-Model Detection**:
  - `qid=0`: `cat=4 (plastic_film)`, score **`0.3138`**, bbox `[977.4, 82.6, 115.6, 267.0]` from `HR448` source — **true positive under BANTAI 5-Category (Plastic)**!
  - `qid=2..5`: `cat=0 (cigarette)`, scores `0.1070 - 0.1250` from `std` source — **true positives for small cigarette butts**!

---

## 7. Replay Artifacts & Verification

The replay artifacts are persisted in `runs/training/stage5-d086-highres-20261031-a/`:
- `multiscale-consensus-replay.json`: Top 100 queries per image (6,500 total), bound to winning configuration `Tri_Model_D086_S2575_HR448_D089_S5050`.
- `scores-multiscale-consensus.npz`: Compressed `(65, 100, 9)` score array (`np.float32`).
- `multiscale-consensus-audit.json`: Complete 90-combination evaluation grid and cross-class disambiguation audit summary.

### Verification Commands
```powershell
# Replay unit test
.\.venv-training\Scripts\python.exe -m unittest tests\test_stage5_multiscale_consensus.py

# Native Tk visual review
.\.venv-training\Scripts\python.exe -B .\scripts\taco_stage5_visual_review.py --prediction-source multiscale_384_448 --image-id 1

# Guarded test suite
.\.venv-training\Scripts\python.exe -B .\scripts\verify_stage4_guarded.py
```
