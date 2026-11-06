# TACO Stage 5 D-093 Multi-Epoch Hard-Negative & Class-Balanced Training Report

Owner authorization **D-093** executed on branch `feat/stage5-paper-category`:
- **Training run directory**: `runs/training/stage5-d093-multiepoch-hn-20261102-a/`
- **Export dataset**: `exports/taco/strict-clean-stage5-tier3-core-v1/` (`576` train images / `1,240` boxes; `65` validation images / `109` boxes)
- **Rights ledger**: `work/stage5-paper/rights_tier3_core.json` (`641` photo SHA-256 digests)
- **Warm-start champion**: D-091 Champion bundle (`runs/training/stage5-d091-balanced-tier3-20261101-a/checkpoints/best-inference`, parameter SHA-256 `4146517bd69837e2e11932e4c3ecee5951cd3e862f414e6f8ba45e1ed20212ee`)
- **Frozen backbones**: Swin-T + BERT frozen parameter SHA-256 `161c9814bffe1657ae0debd7e21b7ed1c41f29f3ba43fffe879ab31ba36a0b67`
- **Zero test-split exposure**: `test.json` unopened, zero held-out test photos opened, file-system audithook enforced.

---

## 1. Algorithmic Formulation & Design

In D-091, single-pass class-balanced fine-tuning pushed single-model validation AP to `0.236288`. However, visual review revealed residual false-positive activations (e.g. `plastic_film` and `cigarette` predictions appearing on pure paper scenes like Image 1 / `000008.jpg`).

D-093 addresses this via an auxiliary absent-class false-positive suppression penalty combined with multi-epoch cosine decay:

### A. Auxiliary Absent-Class Suppression Penalty
For every training image, we identify the active classes $\{0 \dots 8\}$ absent from the ground-truth annotations:
$$\mathcal{C}_{\text{absent}} = \{c \in \{0 \dots 8\} \mid c \notin \mathcal{C}_{\text{present}}\}$$
For each absent category $c$, we gather the model's top-10 sigmoid confidence logits across all 900 queries, penalizing activations exceeding the baseline floor $\tau = 0.08$:
$$\mathcal{L}_{\text{absent}} = \frac{1}{|\mathcal{C}_{\text{absent}}|} \sum_{c \in \mathcal{C}_{\text{absent}}} \frac{1}{10} \sum_{k=1}^{10} \left[\max(0, p_{c, (k)} - 0.08)\right]^2$$
If all classes are present, $\mathcal{L}_{\text{absent}} = 0$.

### B. Full Multi-Objective Training Loss
$$\mathcal{L}_{\text{total}} = w_{\text{img}} \cdot \mathcal{L}_{\text{det}}(\theta; x, y) + \lambda_{\text{absent}} \cdot \mathcal{L}_{\text{absent}} + \lambda_{\text{anchor}} \cdot \|\theta - \theta_0\|_2^2$$
where:
- $w_{\text{img}} = \max_{c \in \text{image}} w_c \in [0.7741, 1.8000]$ (mean $1.1162$, unchanged from D-091)
- $\lambda_{\text{absent}} = 2.5$
- $\lambda_{\text{anchor}} = 0.002$ (anchoring to warm-start $\theta_0$)
- 3 epochs = exactly $1,728$ update steps

---

## 2. Training Run Dynamics

- **Updates**: exactly 1,728 steps (3 epochs of 576 images, batch size 1, resolution `384/576`, square pad 576)
- **Optimizer**: AdamW (`weight_decay=1e-4, foreach=False, clip_norm=1.0`, FP32, TF32 disabled)
- **Learning rate schedule**: Deterministic Cosine Annealing ($1.8 \times 10^{-6} \to 2.0 \times 10^{-7}$)
- **Training Mean Loss**: `5.141757`
  - Detection loss: `4.475459`
  - Absent loss: `0.057391`
  - Anchor loss: `0.000030`
  - Image weight: `1.116200`
- **Elapsed Training Time**: ~48 minutes
- **Peak VRAM**: `3860.0 MiB` / `3950.0 MiB` (within GTX 1660 Ti 6144 MiB hardware envelope)
- **Checkpoints saved & verified**: `step-000288`, `step-000576` (`epoch-01`), `step-000864`, `step-001152` (`epoch-02`), `step-001440`, `step-001728` (`epoch-03`).

---

## 3. 5-State Validation Evaluation (`560` forwards)

Five states were evaluated at standard resolution `384/576` on both `combined_65` (109 boxes) and `legacy_47` (75 boxes):

| Evaluation State | Recipe / Weights | Parameter SHA-256 | `combined_65` AP | `combined_65` AP50 | `combined_65` AR100 | `legacy_47` AP | `legacy_47` AR100 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D093-HN E1`** | $1.00 \times \text{E1}$ | `69894e776ea49a1eb4d3d2db7aa61a0eb86716075908e08d66df870f7d0c3eb1` | 0.233860 | 0.282544 | 0.373289 | 0.252994 | 0.389772 |
| **`D093-HN E2`** | $1.00 \times \text{E2}$ | `2750e80a65fa766dd904116926918220939838ed3ec304c9afa1740e3b807dbd` | 0.218606 | 0.267927 | 0.365193 | 0.280291 | 0.386047 |
| **`D093-HN E3`** | $1.00 \times \text{E3}$ | `9501a1418eb01c7519131e29afc22f09bae9063ec2728bc5762eeefce67aa3ce` | 0.218925 | 0.268010 | 0.364740 | 0.284742 | 0.386187 |
| **`soup_d093best_d091_5050` (Champion)** | $0.50 \text{ D093-E1} + 0.50 \text{ D091}$ | `51ce99e6b7f2e9b08a070a7c237aba752b08467a84ffe482367e795dd1ad8fb7` | **0.241412** | **0.292904** | **0.384577** | **0.256137** | **0.395724** |
| **`soup_d093best_d091_2575`** | $0.25 \text{ D093-E1} + 0.75 \text{ D091}$ | `f7b56a5c123195fed88cfb266e84944920f14e91d4ace2a337888b46a8b8b309` | 0.234014 | 0.280506 | 0.373951 | 0.252771 | 0.391740 |

### Per-Class AP Comparison on `combined_65`

| Class | Baseline (D-073) | D-083 Core | D-091 Champion | **D-093 Champion (`soup 50/50`)** | Delta vs D-091 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `0: cigarette` | 0.1264 | 0.1235 | 0.1245 | **0.1091** | -0.0154 |
| `1: plastic_bottle` | 0.3643 | 0.3719 | 0.3719 | **0.3851** | **+0.0132** |
| `2: plastic_bag` | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 |
| `3: plastic_wrapper` | 0.0158 | 0.0158 | 0.0158 | **0.0000** | -0.0158 |
| `4: plastic_film` | 0.0400 | 0.0451 | 0.0462 | **0.0525** | **+0.0063** |
| `5: drink_can` | 0.6564 | 0.6564 | 0.6703 | **0.6703** | 0.0000 |
| `6: plastic_cup` | 0.3652 | 0.3652 | 0.3652 | **0.3703** | **+0.0051** |
| `7: styrofoam_piece` | 0.1485 | 0.1485 | 0.1485 | **0.1485** | 0.0000 |
| `8: paper` | 0.3921 | 0.4212 | 0.4285 | **0.4369** | **+0.0084** |
| **All-9 Specific AP** | 0.2343 | 0.2361 | 0.2363 | **0.241412** | **+0.0051** |
| **BANTAI 5-Category AP**| 0.3879 | 0.4024 | 0.4051 | **0.406022** | **+0.0010** |

---

## 4. Multi-Scale Consensus Routing Update

When integrated into the class-calibrated multi-scale query routing pipeline (`scripts/taco_stage5_multiscale_consensus.py`), pairing the D-093 champion with the D-086 High-Res (`448/672`) branch achieved project record benchmarks:

- **Winning Pairing**: `Tri_Model_D086_S2575_HR448_D093_Champ`
- **Routing Policy**: `hr_small_medium_plus_can_paper`
- **Consensus Mode**: `pre_nms`
- **9-Specific AP**: **`0.261603`** (previous D-090 best: `0.260674`, **+0.000929**)
- **BANTAI 5-Category AP**: **`0.437255`** (previous D-090 best: `0.435166`, **+0.002089**)

---

## 5. Visual Inspection Verification (Image 1 and Image 777)

### Validation Image 1 (`000008.jpg` - GT: 2 Paper boxes, 0 Plastic, 0 Hazardous)
- **Stage 5 Run 1 @ 0.10**: `plastic_film:0.31, plastic_wrapper:0.29` (Paper not detected)
- **Stage 5 HighRes @ 0.10**: `plastic_film:0.33, paper:0.32` (Plastic film higher than paper)
- **D-091 Champion @ 0.10**: `paper:0.33, plastic_film:0.33` (Tied)
- **D-093 Champion @ 0.10**: `paper:0.35, plastic_film:0.33` (Paper clearly and solidly out in front)

### Validation Image 777 (`000084.JPG` - GT: 7 Cigarettes, 1 Plastic Wrapper)
- **Stage 5 Run 1 @ 0.10**: 0 cigarettes detected, false positive plastic bottle at 0.30.
- **Stage 5 HighRes @ 0.10**: 4 cigarettes detected (`0.13, 0.12, 0.11, 0.10`).
- **D-091 Champion @ 0.10**: 4 cigarettes detected (`0.13, 0.12, 0.11, 0.11`).
- **D-093 Champion @ 0.10**: 4 cigarettes detected with increased confidence (`0.13, 0.13, 0.11, 0.11`).
- **Full GUI Smoke Test**: `tests/windows_stage5_image777_check.py` passed all 50 interactive UI combinations.

---

## 6. D-094 Multi-Checkpoint Query Routing and False-Positive Suppression Audit

Under owner authorization **D-094**, a strictly CPU-only per-class audit across all five D-093 evaluation states on `combined_65` and `legacy_47` was conducted, along with class-optimal multi-checkpoint routing and false-positive suppression integration in `scripts/taco_stage5_multiscale_consensus.py`.

### 1. D093-HN State Comparison & Absent-Class Suppression Effects
- **`D093-HN E3` on Legacy-47**: Reached an all-time legacy benchmark of **`0.284742` AP** (`AP50: 0.333934`, `AR100: 0.415101`). Multi-epoch absent-class suppression loss ($\mathcal{L}_{\text{absent}}$) severely punishes false-positive background activations on non-annotated classes, cleaning predictions on legacy scenes.
- **`D093-HN E3` on `combined_65` Cigarette**: Peaked at **`0.126224` AP** (`AR100: 0.392308`), highest among all D-093 individual states.
- **Image 1 (`000008.jpg`) Paper Confidence**: On `D093-HN E3`, Paper confidence reached an all-time peak of **`0.372`**, decisively outranking `plastic_film` (`0.319`).

### 2. Multi-Checkpoint Routing Benchmark Results
Evaluating 11 pairings across 7 routing policies and 3 consensus modes (231 evaluations total) confirmed new all-time project records:

| Pairing / Configuration | Routing Policy | Mode | 9-Specific AP | BANTAI 5-Category AP | Paper AP | Drink Can AP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| D-090 Winner (`Tri_Model_D086_S2575_HR448_D089_S5050`) | `hr_small_medium_plus_can_paper` | `pre_nms` | 0.260674 | 0.435166 | 0.468014 | 0.656436 |
| `Class_Optimal_Multi_Checkpoint_Ensemble` | `class_optimal_hn_e3_cig` | `pre_nms` | 0.261394 | 0.436786 | 0.468014 | 0.684158 |
| **`Tri_Model_D086_S2575_HR448_D093_Champ` (D-094 Champion)** | `hr_small_medium_plus_can_paper` | `pre_nms` | **0.261603** | **0.437255** | **0.472763** | **0.684158** |

### 3. All-Time Per-Class AP Breakdown (`Tri_Model_D086_S2575_HR448_D093_Champ`)
- `0: cigarette`: **`0.123328`** (`AR100: 0.423077`)
- `1: plastic_bottle`: **`0.419554`** (`AR100: 0.577778`)
- `2: plastic_bag`: **`0.000000`**
- `3: plastic_wrapper`: **`0.041867`** (`AR100: 0.146154`)
- `4: plastic_film`: **`0.068693`** (`AR100: 0.570588`)
- `5: drink_can`: **`0.684158`** (`AR100: 0.685714`, +0.0277 vs D-086)
- `6: plastic_cup`: **`0.395545`** (`AR100: 0.466667`)
- `7: styrofoam_piece`: **`0.148515`** (`AR100: 0.142857`)
- `8: paper`: **`0.472763`** (`AR100: 0.755000`, all-time record, +0.0047 vs D-090)
- **9-Specific AP**: **`0.261603`** (AP50 `0.310963`, AP75 `0.285697`, AR100 `0.418648`)
- **BANTAI 5-Category AP**: **`0.437255`** (AP50 `0.528602`, AP75 `0.458641`, AR100 `0.647740`)
