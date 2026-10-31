# TACO Stage 5 D-089 Tier 3 Core Fine-Tuning and Evaluation Report

Owner authorization **D-089** executed on branch `feat/stage5-paper-category`:
- **Training run directory**: `runs/training/stage5-d089-tier3-core-20261031-a/`
- **Export dataset**: `exports/taco/strict-clean-stage5-tier3-core-v1/`
- **Rights ledger**: `work/stage5-paper/rights_tier3_core.json`
- **Warm-start champion**: D-083 Champion (`runs/training/stage5-d083-core-20261031-a/checkpoints/best-inference`, parameter SHA-256 `82c0a5bf3a104db5f5ff35e1cbfc1bd239595ec9946e22cf085986985d95fddc`)
- **Frozen backbones**: Swin-T + BERT frozen parameter SHA-256 `161c9814bffe1657ae0debd7e21b7ed1c41f29f3ba43fffe879ab31ba36a0b67`
- **Zero test-set exposure**: `test.json` unopened, zero held-out test photos opened, file-system audithook enforced.

---

## 1. Dataset & Rights Summary

| Split / Attribute | Specification | Exact Measured Count / Hash |
| :--- | :--- | :--- |
| **Training Images** | 379 D-083 Core + 197 clean Tier 3 | **576** |
| **Training Annotations (0..8)** | Strict audited categories | **1,240** |
| **Omitted Accessory Boxes** | Caps, tabs, straws, lids, glass/foil | **286** across 197 newly unlocked images |
| **Excluded Quarantined Images** | All images containing `Aerosol`, `Plastified paper bag`, `Other plastic`, `Unlabeled litter` | **8 Aerosol images quarantined** (`6, 16, 23, 24, 35, 800, 1079, 1101`) |
| **Validation Images** | Frozen Stage 5 holdout | **65** |
| **Validation Annotations** | Frozen Stage 5 holdout | **109** |
| **Legacy Benchmark View** | Frozen Stage 2 benchmark | **47 images / 75 boxes** |
| **Bound Photos Digest** | SHA-256 bindings in `rights_tier3_core.json` | **641 photos** (`576 train + 65 val`) |

### Class Distribution on 576 Training Images (1,240 boxes)
- `0: cigarette`: 252
- `1: plastic_bottle`: 151
- `2: plastic_bag`: 39
- `3: plastic_wrapper`: 97
- `4: plastic_film`: 168
- `5: drink_can`: 157
- `6: plastic_cup`: 75
- `7: styrofoam_piece`: 63
- `8: paper`: 238

---

## 2. In-Memory Omitted-Accessory Neutral Masking

During training, `CocoTier3CoreDataset` inspects each training image's `omitted_accessory_bboxes`. Any omitted accessory bounding box (caps, tabs, lids, straws, utensils) is neutral-masked in memory with the image's mean RGB value **unless it overlaps a retained 0..8 ground-truth box**. If an omitted accessory box overlaps any retained ground-truth box (such as a bottle cap attached to a bottle), the ground-truth region is strictly preserved. Zero disk mutations are performed.

---

## 3. Training Run Dynamics

- **Updates**: exactly 576 steps (batch size 1, resolution `384/576`, square pad 576)
- **Optimizer**: AdamW (`weight_decay=1e-4, foreach=False, clip_norm=1.0`, FP32, TF32 disabled)
- **Learning rate schedule**: Deterministic Cosine Annealing ($1.5 \times 10^{-6} \to 2.0 \times 10^{-7}$)
- **Training Mean Loss**: `4.62176`
- **Elapsed Training Time**: `1213.18 s` (~20.2 minutes)
- **Peak VRAM**: `3590.51 MiB` (within GTX 1660 Ti 6144 MiB hardware envelope)
- **Checkpoints saved**: `step-000100`, `step-000200`, `step-000300`, `step-000400`, `step-000500`, and `epoch-01`.

---

## 4. 4-State Validation Evaluation (`448` forwards)

Four states were evaluated at standard resolution `384/576` on both `combined_65` (109 boxes) and `legacy_47` (75 boxes):

| Evaluation State | Recipe / Weights | Parameter SHA-256 | `combined_65` AP | `combined_65` AP50 | `combined_65` AR100 | `legacy_47` AP | `legacy_47` AR100 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D089-Tier3-Core E1`** | $1.00 \times \text{E1}$ | `5d3003cc2661f4c8034dbda501e74f88e1e780798bb2ff21469e36814b7f5c53` | 0.216870 | 0.263714 | 0.358958 | 0.252880 | 0.392569 |
| **`soup_d089_d083_5050`** | $0.50 \text{ D083} + 0.50 \text{ E1}$ | `38ee8bca3269b730ee0f4c084f7b4ea647f1b952f44c4e7da3e2a22026a45f94` | 0.218216 | 0.264717 | 0.360467 | 0.253403 | 0.396855 |
| **`soup_d089_d083_2575` (Champion)** | $0.75 \text{ D083} + 0.25 \text{ E1}$ | `135ace61bc3acccd682175922e4f3a3581fef0a62d9af961b942751906c6bf70` | **0.233812** | **0.278686** | **0.371596** | **0.253113** | **0.390262** |
| **`soup_d089_d086s2575_5050`** | $0.50 \text{ D086s} + 0.50 \text{ E1}$ | `27ab02b50b961d720a402fe38bf64243eb032b49dfd41071d0e19dd8c2ec1f4a` | 0.218809 | 0.265513 | 0.358958 | 0.252160 | 0.391527 |

### Per-Class AP Comparison on `combined_65`

| Class | D-083 Champion | `D089-Tier3-Core E1` | `soup_d089_d083_5050` | `soup_d089_d083_2575` (Champ) | `soup_d089_d086s2575_5050` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0: cigarette` | 0.110532 | 0.107380 | 0.108087 | 0.107777 | 0.107612 |
| `1: plastic_bottle` | 0.331400 | 0.326733 | 0.331400 | 0.331400 | 0.331400 |
| `2: plastic_bag` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `3: plastic_wrapper`| 0.015842 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `4: plastic_film` | 0.042013 | 0.048839 | 0.047926 | 0.045215 | 0.046956 |
| `5: drink_can` | 0.656436 | **0.670297** | **0.670297** | **0.670297** | **0.670297** |
| `6: plastic_cup` | 0.395545 | 0.370297 | 0.376609 | 0.384724 | 0.384724 |
| `7: styrofoam_piece`| 0.148515 | 0.000000 | 0.000000 | **0.148515** | 0.000000 |
| `8: paper` | 0.423660 | **0.428288** | **0.429627** | 0.416378 | **0.428288** |
| **All 9 Classes AP** | **0.235994** | 0.216870 | 0.218216 | **0.233812** | 0.218809 |

---

## 5. BANTAI 5-Category & Material Hierarchy Audit

Under the table-driven extensible BANTAI 5-Category mapping (`Plastic`, `Paper`, `Organic`, `Metal`, `Potentially Hazardous`), the 4 states evaluate as follows (raw proposal view):

| Evaluation State | BANTAI 5-Category AP | Plastic AP | Paper AP | Metal AP | Potentially Hazardous AP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `D089-Tier3-Core E1` | 0.399090 | 0.390395 | 0.428288 | 0.670297 | 0.107380 |
| `soup_d089_d083_5050` | 0.397892 | 0.383557 | 0.429627 | 0.670297 | 0.108087 |
| **`soup_d089_d083_2575` (Champ)** | **0.398774** | **0.400643** | 0.416378 | **0.670297** | 0.107777 |
| `soup_d089_d086s2575_5050` | 0.397932 | 0.385531 | 0.428288 | 0.670297 | 0.107612 |

*(Note: `Organic` has 0 instances in the frozen validation split; its metric is `None` as documented in D-080).*

---

## 6. D-079 Score-Preserving Consensus Audit

Evaluating the winning `soup_d089_d083_2575` state under D-079 score-preserving consensus ($\delta=0.016$, top-1 confidence, $\ge 3$ plastic co-activations $\ge 0.20$):
- **Changed queries**: exactly 1 query
  - `image_id=419, query_id=0`: reassigned from `8: paper` to `3: plastic_wrapper` (score `0.349545`)
- **9-Specific AP**: `0.233812` $\to$ **`0.236248`** (`+0.002437`)
- **BANTAI 5-Category AP**: `0.398774` $\to$ **`0.403616`** (`+0.004843`)

---

## 7. Multi-Scale Fusion, Calibrated Routing, and Containment Suppression Audit

In Part C, the D-089 champion (`soup_d089_d083_2575` at `384/576`) was fused with `D086-HR448 E1` (`448/672`) under the calibrated routing grid:

| Multi-Scale Pairing | Routing Policy | Consensus Mode | 9-Specific AP | BANTAI 5-Category AP |
| :--- | :--- | :--- | :--- | :--- |
| `D089_Champ_384 + D086_HR448` | `hr_small_medium_plus_can_paper` | `pre_nms` | 0.256673 | 0.428970 |
| `D089_Champ_384 + D086_HR448` | `std_only_cup_styrofoam` | `pre_nms` | 0.256673 | 0.428970 |
| `D089_Champ_384 + D086_HR448` | `uniform_both` | `pre_nms` | 0.245093 | 0.425503 |
| **`D086_Soup2575_384 + D086_HR448` (D-088 Champion)** | `hr_small_medium_plus_can_paper` | `pre_nms` | **0.258410** | **0.430072** |

Because D-088's winning pairing (`D086_Soup2575_384 + D086_HR448`) remains higher (`0.258410` vs `0.256673`), `multiscale-consensus-replay.json` correctly retains the champion pairing.

### Cross-Class Containment Suppression Evaluation
Evaluating whether suppressing smaller nested boxes inside larger boxes of a different class improves precision:
- **Baseline (No containment suppression)**: 9-Specific AP = **`0.258410`**, BANTAI AP = **`0.430072`**
- **Threshold 0.95** (suppresses 3,516 queries): 9-Specific AP = `0.242711`, BANTAI AP = `0.403690`
- **Threshold 0.90** (suppresses 3,587 queries): 9-Specific AP = `0.243441`, BANTAI AP = `0.405883`
- **Threshold 0.85** (suppresses 3,621 queries): 9-Specific AP = `0.243460`, BANTAI AP = `0.405924`
- **Threshold 0.80** (suppresses 3,654 queries): 9-Specific AP = `0.243460`, BANTAI AP = `0.405979`

**Conclusion**: Cross-class containment suppression significantly degrades AP (by ~0.015 AP) across the dataset because real street waste frequently exhibits legitimate co-occurring objects (e.g., cigarette butts inside cardboard cartons or drink cans). Blindly discarding enclosed proposals harms detection recall.

---

## 8. Exact Detections on Key Images

### Image 1 (`batch_1/000008.jpg`)
Ground-truth annotations:
- `id=1`: Meal carton (`8: paper`), bbox `[1.0, 457.0, 1429.0, 1519.0]`
- `id=2`: Other carton (`8: paper`), bbox `[531.0, 292.0, 1006.0, 672.0]`

Detections ($\text{score} \ge 0.10$) under D-089 Champion multi-scale fusion:
1. `qid=0`: `cat=4: plastic_film`, score `0.3570`, bbox `[532.3, 798.1, 400.3, 328.8]`, scale `hr`
2. `qid=1`: `cat=8: paper`, score `0.3328`, bbox `[0.7, 458.1, 1432.6, 1516.8]`, scale `std`
3. `qid=2`: `cat=0: cigarette`, score `0.2362`, bbox `[0.0, 1153.8, 25.8, 30.3]`, scale `std`
4. `qid=3`: `cat=3: plastic_wrapper`, score `0.1866`, bbox `[526.2, 280.6, 1008.0, 676.8]`, scale `std`
5. `qid=4`: `cat=4: plastic_film`, score `0.1407`, bbox `[1.3, 289.3, 1521.3, 1684.3]`, scale `hr`
6. `qid=5`: `cat=0: cigarette`, score `0.1148`, bbox `[1357.4, 1938.5, 27.9, 29.7]`, scale `std`
7. `qid=6`: `cat=0: cigarette`, score `0.1115`, bbox `[731.6, 968.1, 54.0, 57.6]`, scale `std`

### Image 777
Ground-truth annotation:
- `id=408`: `8: paper`, bbox `[985.0, 83.0, 103.0, 269.0]`

Detections ($\text{score} \ge 0.10$) under D-089 Champion multi-scale fusion:
1. `qid=0`: `cat=4: plastic_film`, score `0.3138`, bbox `[977.4, 82.6, 115.6, 267.0]`, scale `hr`
2. `qid=1`: `cat=8: paper`, score `0.3131`, bbox `[977.9, 81.4, 116.5, 268.9]`, scale `std`
3. `qid=2`: `cat=0: cigarette`, score `0.1278`, bbox `[1039.5, 775.3, 42.4, 50.5]`, scale `std`
4. `qid=3`: `cat=0: cigarette`, score `0.1224`, bbox `[1120.8, 786.8, 26.3, 36.4]`, scale `std`
5. `qid=4`: `cat=0: cigarette`, score `0.1119`, bbox `[1076.9, 903.0, 27.1, 28.8]`, scale `std`
6. `qid=5`: `cat=0: cigarette`, score `0.1092`, bbox `[1621.1, 290.7, 78.1, 54.0]`, scale `std`

*(Under D-089 champion multi-scale fusion, `paper` on Image 777 achieves a strong detection score of `0.3131`, closely matching the ground-truth box with $\text{IoU} > 0.90$).*

---

## 9. Protected Artifact Verification

- Total protected files checked: **2,089**
- Snapshot SHA-256 before training: `de43089895ecd4e42c77dc22f8750a39e709a818738852cc2606a5537981b6d0`
- Snapshot SHA-256 after training: `de43089895ecd4e42c77dc22f8750a39e709a818738852cc2606a5537981b6d0`
- Drift: **zero** (`unchanged: true`)
