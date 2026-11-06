# TACO Stage 5 D-091 Class-Balanced & L2-Anchored Tier 3 Core Training Report

Owner authorization **D-091** executed on branch `feat/stage5-paper-category`:
- **Training run directory**: `runs/training/stage5-d091-balanced-tier3-20261101-a/`
- **Export dataset**: `exports/taco/strict-clean-stage5-tier3-core-v1/` (`576` train images / `1,240` boxes; `65` validation images / `109` boxes)
- **Rights ledger**: `work/stage5-paper/rights_tier3_core.json` (`641` photo SHA-256 digests)
- **Warm-start champion**: D-086 Soup2575 bundle (`runs/training/stage5-d086-highres-20261031-a/checkpoints/best-inference`, parameter SHA-256 `202ee007132838e381c43a7d9ed4b82d7dae728e0325e464c6c562a0b9ab92e7`)
- **Frozen backbones**: Swin-T + BERT frozen parameter SHA-256 `161c9814bffe1657ae0debd7e21b7ed1c41f29f3ba43fffe879ab31ba36a0b67`
- **Zero test-split exposure**: `test.json` unopened, zero held-out test photos opened, file-system audithook enforced.

---

## 1. Algorithmic Formulation & Design

In D-089, unweighted 576-step fine-tuning on Tier 3 Core achieved record paper detection (`0.429627`) but caused catastrophic forgetting on rare classes (`styrofoam_piece` dropped to `0.0000`, `plastic_wrapper` dropped to `0.0000`).

D-091 introduces two stabilizing mechanisms:

### A. Inverse-Square-Root Class Frequency Weighting
Each of the 9 active classes receives an individual sample weight scaled relative to the reference median category count ($N_{\text{ref}} = 151$ instances of `plastic_bottle`), clipped strictly to $[0.75, 1.80]$:
$$w_c = \text{clip}\!\left(\sqrt{\frac{151.0}{N_c}}, 0.75, 1.80\right)$$

| Category ID | Class Name | Box Count in Train Set | Raw Weight $\sqrt{151 / N_c}$ | Clipped Weight $w_c$ |
| :---: | :--- | :---: | :---: | :---: |
| `0` | `cigarette` | 252 | 0.7741 | **0.7741** |
| `1` | `plastic_bottle` | 151 | 1.0000 | **1.0000** |
| `2` | `plastic_bag` | 39 | 1.9678 | **1.8000** |
| `3` | `plastic_wrapper` | 97 | 1.2477 | **1.2477** |
| `4` | `plastic_film` | 168 | 0.9481 | **0.9481** |
| `5` | `drink_can` | 157 | 0.9807 | **0.9807** |
| `6` | `plastic_cup` | 75 | 1.4189 | **1.4189** |
| `7` | `styrofoam_piece` | 63 | 1.5482 | **1.5482** |
| `8` | `paper` | 238 | 0.7965 | **0.7965** |

For each training image $i$, the image-level loss scaling factor is the maximum weight among all annotated ground-truth boxes in that image:
$$w_{\text{img}} = \max_{c \in \text{image}} w_c$$
Across all 576 training images, $w_{\text{img}}$ ranges from $0.7741$ to $1.8000$ (mean $1.1162$).

### B. Proximal L2 Parameter Anchor
To prevent feature drift away from the multi-scale-qualified feature manifold while training on rare and newly unlocked images, a proximal L2 anchor is added to the objective:
$$\mathcal{L}_{\text{total}} = w_{\text{img}} \cdot \mathcal{L}_{\text{det}}(\theta; x, y) + \frac{\lambda_{\text{anchor}}}{2} \|\theta - \theta_0\|_2^2$$
where $\lambda_{\text{anchor}} = 0.005$ and $\theta_0$ represents the detached initial weights of the warm-start champion `soup_d086_d083_2575_std384`.

---

## 2. Training Run Dynamics

- **Updates**: exactly 576 steps (batch size 1, resolution `384/576`, square pad 576)
- **Optimizer**: AdamW (`weight_decay=1e-4, foreach=False, clip_norm=1.0`, FP32, TF32 disabled)
- **Learning rate schedule**: Deterministic Cosine Annealing ($8.0 \times 10^{-7} \to 1.5 \times 10^{-7}$)
- **Training Mean Loss**: `5.49067` (detection loss: `4.59431`, anchor loss: `0.000023`)
- **Training Mean Image Weight**: `1.11620`
- **Elapsed Training Time**: `960.0 s` (~16.0 minutes)
- **Peak VRAM**: `3860.0 MiB` (within GTX 1660 Ti 6144 MiB hardware envelope)
- **Checkpoints saved & verified**: `step-000100`, `step-000200`, `step-000300`, `step-000400`, `step-000500`, and `epoch-01`.

---

## 3. 4-State Validation Evaluation (`448` forwards)

Four states were evaluated at standard resolution `384/576` on both `combined_65` (109 boxes) and `legacy_47` (75 boxes):

| Evaluation State | Recipe / Weights | Parameter SHA-256 | `combined_65` AP | `combined_65` AP50 | `combined_65` AR100 | `legacy_47` AP | `legacy_47` AR100 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D091-Balanced-Tier3 E1`** | $1.00 \times \text{E1}$ | `86182c68d99ee68431720107db17e080946abd1b4b019cf22e8cddba01561dc0` | 0.234489 | 0.281339 | 0.371790 | 0.252616 | 0.389278 |
| **`soup_d091_d086s2575_5050` (Champion)** | $0.50 \text{ D086s} + 0.50 \text{ E1}$ | `4146517bd69837e2e11932e4c3ecee5951cd3e862f414e6f8ba45e1ed20212ee` | **0.236288** | **0.282958** | **0.378677** | **0.253725** | **0.392764** |
| **`soup_d091_d083_5050`** | $0.50 \text{ D083} + 0.50 \text{ E1}$ | `74d7430833c9ad0b29af06528adf744e31b835f5a598510fce43e695dd92b47a` | 0.235909 | 0.281352 | 0.374404 | 0.253557 | 0.390886 |
| **`soup_d091_d083_2575`** | $0.75 \text{ D083} + 0.25 \text{ E1}$ | `75271102ec893ced63716774183eb6cea2a1a3f6feb004aaae9f93564b1727d2` | 0.233640 | 0.277919 | 0.370942 | 0.251900 | 0.389332 |

---

## 4. Per-Class AP Comparison on `combined_65`

| Class | D-083 Champ | D-086 Soup2575 | D-089 Champ | D-091 E1 | **D-091 Soup5050 (Champ)** | D-091 vs D-089 Champ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0: cigarette` | 0.110532 | 0.108527 | 0.107777 | 0.107683 | **0.108946** | +0.001169 |
| `1: plastic_bottle` | 0.331400 | 0.331400 | 0.331400 | 0.331400 | **0.331400** | 0.000000 |
| `2: plastic_bag` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | **0.000000** | 0.000000 |
| `3: plastic_wrapper` | 0.015842 | 0.000000 | 0.000000 | 0.000000 | **0.000000** (0.015842 cons.) | 0.000000 |
| `4: plastic_film` | 0.042013 | 0.045093 | 0.045215 | 0.047368 | **0.047069** | +0.001854 |
| `5: drink_can` | 0.656436 | 0.656436 | 0.670297 | 0.670297 | **0.670297** | 0.000000 |
| `6: plastic_cup` | 0.395545 | 0.370297 | 0.384724 | 0.370297 | **0.384724** | 0.000000 |
| `7: styrofoam_piece` | 0.148515 | 0.148515 | 0.148515 | **0.148515** | **0.148515** | **100% PRESERVED** |
| `8: paper` | 0.423660 | 0.394982 | 0.416378 | **0.434842** | **0.435637** | **+0.019259** |
| **All 9 Classes AP** | 0.235994 | 0.228361 | 0.233812 | 0.234489 | **0.236288** | **+0.002476** |

### Key Takeaway on Class Balance
In D-089, unweighted fine-tuning on Tier 3 Core erased `styrofoam_piece` (`0.0000` in E1). In D-091, the class frequency weight ($w_7 = 1.5482$) and proximal L2 anchor completely protected `styrofoam_piece` at **`0.148515`** in E1 and all soup combinations, while pushing `paper` AP to an all-time record **`0.435637`** on single-pass inference and single-model AP to **`0.236288`** (exceeding D-083 Champion `0.235994`).

---

## 5. BANTAI 5-Category & Material Hierarchy Audit

Under the table-driven extensible BANTAI 5-Category mapping (`Plastic`, `Paper`, `Organic`, `Metal`, `Potentially Hazardous`), the D-091 states evaluate as follows (raw proposal view):

| Evaluation State | BANTAI 5-Category AP | AP50 | Plastic AP | Paper AP | Metal AP | Potentially Hazardous AP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `D091-Balanced-Tier3 E1` | 0.404602 | 0.497843 | 0.408082 | 0.434842 | 0.670297 | 0.107683 |
| **`soup_d091_d086s2575_5050` (Champ)** | **0.405054** | **0.497862** | **0.408688** | **0.435637** | **0.670297** | **0.108946** |
| `soup_d091_d083_5050` | 0.404135 | 0.497843 | 0.407519 | 0.435351 | 0.670297 | 0.107897 |
| `soup_d091_d083_2575` | 0.398487 | 0.490714 | 0.402283 | 0.416668 | 0.670297 | 0.106889 |

---

## 6. D-079 Score-Preserving Consensus Audit

Running the deterministic D-079 display consensus rule (`delta = 0.016`, coactivation `3 plastic subtypes >= 0.20`, confidence mode `top1_score`) on the champion D-091 bundle (`soup_d091_d086s2575_5050`):
- **Changed queries**: exactly 1 query (image `419`, query `0`: reclassified from `paper` [8] to `plastic_wrapper` [3] with score `0.350395` preserved).
- **All score, bbox, and query values preserved**: `True`.

| Metric Scheme | Raw Baseline AP | Consensus AP | Delta AP | Raw Baseline AP50 | Consensus AP50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **9-Specific Waste** | 0.236288 | **0.238786** | **+0.002498** | 0.282958 | **0.285456** |
| **BANTAI 5-Category** | 0.405054 | **0.409769** | **+0.004715** | 0.497862 | **0.503117** |

### Per-Class Gains Under Consensus on D-091 Champion
- `plastic_wrapper`: jumps from `0.000000` to **`0.015842`** (`+0.015842`).
- `paper`: rises from `0.435637` to **`0.442283`** (`+0.006646`).
- `Plastic` (5-Category): rises from `0.408688` to **`0.417548`** (`+0.008860`).

---

## 7. Multi-Scale Query Fusion & Routing Benchmark

Evaluating all model pairings across the 6 class-calibrated routing policies and consensus modes:

| Model Pairing | Best Policy | Consensus Mode | 9-Specific AP | BANTAI 5-Category AP | 5-Category AP50 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Tri-Model D086_S2575 + HR448 + D089_S5050** (All-Time Champion) | `hr_small_medium_plus_can_paper` | `pre_nms` | **0.260674** | **0.435166** | **53.27%** |
| **Tri-Model D086_S2575 + HR448 + D091_Champ** | `hr_small_medium_plus_can_paper` | `pre_nms` | **0.260171** | **0.434033** | **53.27%** |
| **D091_Champion_384 + D086_HR448** | `hr_small_medium_plus_can_paper` | `pre_nms` | **0.258014** | **0.431918** | **52.88%** |
| **D086_Soup2575_384 + D086_HR448** | `hr_small_medium_plus_can_paper` | `pre_nms` | 0.252874 | 0.428780 | 52.62% |
| **D083_Champion_384 + D086_HR448** | `hr_small_medium_plus_can_paper` | `pre_nms` | 0.251410 | 0.425178 | 51.52% |

### Evolution of Benchmark Across Stages

| Milestone | Single-Model 9-Spec AP | Multi-Scale 9-Spec AP | BANTAI 5-Cat AP | Consensus 5-Cat AP50 |
| :--- | :---: | :---: | :---: | :---: |
| **D-083 Core Champion** | 0.235994 | — | 0.395759 | 49.33% |
| **D-086 High-Res Fusion** | 0.228361 | 0.244349 | 0.424368 | 52.48% |
| **D-088 Calibrated Multi-Scale** | 0.228361 | 0.252874 | 0.428780 | 52.62% |
| **D-089 Tier 3 Core Expansion** | 0.233812 | 0.253018 | 0.428780 | 52.62% |
| **D-090 Tri-Model Routing** | 0.233812 | **0.260674** | **0.435166** | **53.27%** |
| **D-091 Class-Balanced Tier 3** | **0.236288** | **0.260171** | **0.434033** | **53.27%** |
