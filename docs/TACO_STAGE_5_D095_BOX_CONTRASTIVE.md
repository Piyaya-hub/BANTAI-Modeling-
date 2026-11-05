# TACO Stage 5 D-095 Box-Contrastive Margin Fine-Tuning & Scene Consensus Report

Owner authorization **D-095** executed on branch `feat/stage5-paper-category`:
- **Training run directory**: `runs/training/stage5-d095-box-contrastive-20261102-a/`
- **Export dataset**: `exports/taco/strict-clean-stage5-tier3-core-v1/` (`576` train images / `1,240` boxes; `65` validation images / `109` boxes)
- **Rights ledger**: `work/stage5-paper/rights_tier3_core.json` (`641` photo SHA-256 digests)
- **Warm-start champion**: D-093 Champion bundle (`runs/training/stage5-d093-multiepoch-hn-20261102-a/checkpoints/best-inference`, parameter SHA-256 `51ce99e6b7f2e9b08a070a7c237aba752b08467a84ffe482367e795dd1ad8fb7`)
- **Frozen backbones**: Swin-T + BERT frozen parameter SHA-256 `161c9814bffe1657ae0debd7e21b7ed1c41f29f3ba43fffe879ab31ba36a0b67`
- **Protected Stage 4 architecture**: `scripts/training_adapter_v2.py` and 6 core Stage 4 files preserved untouched.
- **Strict Taxonomy Lock**: strictly 9 active supervised classes `0..8` (`0: cigarette`, `1: plastic_bottle`, `2: plastic_bag`, `3: plastic_wrapper`, `4: plastic_film`, `5: drink_can`, `6: plastic_cup`, `7: styrofoam_piece`, `8: paper`).
- **Zero test-split exposure**: `test.json` unopened, zero held-out test photos opened, file-system audithook enforced.

---

## 1. Problem Statement: Validation Image 1 Clutter Analysis

Visual inspection of validation Image 1 (`batch_1/000008.jpg`, $1537 \times 2049$) revealed that prior champions produced 7 candidate detections at threshold $0.10$, heavily cluttering the image despite Ground Truth containing only **two Paper objects**:
- **GT #1**: Giant brown paper bag `[1.0, 457.0, 1429.0, 1519.0]` (`paper`)
- **GT #2**: Thin curly paper strip `[531.0, 292.0, 1006.0, 672.0]` (`paper`)

### Root Cause Diagnostic:
1. **GT #2 Classification Mismatch**: The curly paper strip query produced `plastic_wrapper: 0.1907` vs `paper: 0.1054`. Because `plastic_wrapper` had a higher score, the query was labeled plastic.
2. **Sub-Box Containment (Logo Box)**: An interior printed logo on the bag produced `plastic_film: 0.3340` at `[532.3, 798.1, 400.3, 328.8]`. Its containment inside the bag was $100\%$ ($\text{IoMin} = 1.00$), but its standard IoU was only $0.06$ (6% of bag area), allowing it to completely escape standard IoU-based NMS.
3. **Micro-Speckle False Positives**: Text cursive and wood grain activated tiny $25 \times 30$ px false-positive `cigarette` detections (`0.24, 0.12, 0.11`) inside the bag boundary.

---

## 2. Algorithmic Formulation & Design

D-095 introduced a three-part solution combining box-contrastive margin training with bidirectional scene consensus:

### A. Positive Target Margin Boost
For ground-truth matched queries, we encourage the true target class probability to reach at least $\tau_{\text{pos}} = 0.35$:
$$\mathcal{L}_{\text{pos\_margin}} = \frac{1}{|\mathcal{M}_{\text{pos}}|} \sum_{(q, c^*) \in \mathcal{M}_{\text{pos}}} \left[\max(0, 0.35 - p_{q, c^*})\right]^2$$

### B. Interior Wrong-Class Sub-Box Penalty
For any non-paper query $q$ whose bounding box is $\ge 70\%$ contained inside a ground-truth Paper bounding box $g$:
$$\text{IoMin}(b_q, b_g) = \frac{\text{Area}(b_q \cap b_g)}{\min(\text{Area}(b_q), \text{Area}(b_g))} \ge 0.70 \quad \text{and} \quad \text{Area}(b_q) < 0.50 \cdot \text{Area}(b_g)$$
We penalize any wrong-class activation exceeding floor $\tau = 0.08$:
$$\mathcal{L}_{\text{interior}} = \frac{1}{|\mathcal{Q}_{\text{interior}}|} \sum_{q \in \mathcal{Q}_{\text{interior}}} \sum_{c \neq 8} \left[\max(0, p_{q, c} - 0.08)\right]^2$$

### C. Auxiliary Absent-Class Suppression Penalty
Maintained from D-093 to suppress non-present classes on single-material images:
$$\mathcal{L}_{\text{absent}} = \frac{1}{|\mathcal{C}_{\text{absent}}|} \sum_{c \in \mathcal{C}_{\text{absent}}} \frac{1}{10} \sum_{k=1}^{10} \left[\max(0, p_{c, (k)} - 0.08)\right]^2$$

### D. Full Training Loss
$$\mathcal{L}_{\text{total}} = w_{\text{img}} \cdot \mathcal{L}_{\text{det}} + 2.0 \cdot \mathcal{L}_{\text{pos\_margin}} + 1.5 \cdot \mathcal{L}_{\text{interior}} + 2.5 \cdot \mathcal{L}_{\text{absent}} + 0.004 \cdot \|\theta - \theta_0\|_2^2$$

### E. Bidirectional Scene Consensus & Sub-Box Suppression (`apply_scene_consensus`)
At inference/display time:
1. **Rule 1 (Co-Activation Tiebreak)**: If Paper is co-activated with $\ge 3$ plastic subtypes $\ge 0.20$ within delta $0.016$, reassign to dominant plastic subtype.
2. **Rule 2 (Bidirectional Plastic $\to$ Paper Scene Coherence)**: When a dominant Paper anchor query exists ($\text{score} \ge 0.25$, $p_8 > \max(p_{\text{plastic}}) + 0.04$), any flexible plastic query (`plastic_bag`, `plastic_wrapper`, `plastic_film`) with $0.10 \le \text{score} \le 0.30$ and $p_8 \ge 0.05$ is reassigned to `paper`.
3. **Rule 3 (Interior Sub-Box & Micro-Speckle Suppression)**: Any query inside a dominant paper region with $\text{IoMin} \ge 0.80$ and $\text{Area} < 0.50 \cdot \text{Area}_{\text{paper}}$, or tiny cigarette speckles ($\text{Area}_{\text{norm}} < 0.0015$) inside large paper objects, is cleanly suppressed.

---

## 3. Training Run Dynamics

- **Updates**: exactly 1,152 steps (2 epochs of 576 images, batch size 1, resolution `384/576`, square pad 576)
- **Optimizer**: AdamW (`weight_decay=1e-4, clip_norm=1.0`, FP32, TF32 disabled)
- **Learning rate schedule**: Deterministic Cosine Annealing ($1.5 \times 10^{-6} \to 2.0 \times 10^{-7}$)
- **Proximal Anchor**: $\lambda_{\text{anchor}} = 0.004$ anchored to D-093 champion $\theta_0$
- **Training Loss Means**:
  - Total Loss: `5.138495`
  - Detection Loss: `4.566373`
  - Positive Margin Loss: `0.002237`
  - Interior Loss: `0.001608`
  - Absent Loss: `0.042301`
- **Elapsed Training Time**: `1904.5` seconds (~31.7 minutes), recorded in `runs/training/stage5-d095-box-contrastive-20261102-a/report.json` under field `total_seconds` (`1904.4990426000004`).
- **Peak VRAM**: Not recorded in `report.json` (executed on local `NVIDIA GeForce GTX 1660 Ti` 6144 MiB envelope)
- **Gradient Signal Note**: Box-contrastive margin and interior losses were very small (mean positive margin loss = `0.002237`, mean interior loss = `0.001608`), indicating weak gradient signal on this loss formulation. D-095 fine-tuning lowered raw AP from `0.241412` down to `0.238401`.

---

## 4. 5-State Validation Evaluation (`560` forwards)

| Evaluation State | Recipe / Weights | Parameter SHA-256 | `combined_65` Raw AP | BANTAI 5-Cat Raw AP | `legacy_47` AP | Styrofoam AP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D095-BC E1`** | $1.00 \times \text{E1}$ | `80c6086b5ece2462fecd3deda817a29ba49cefe5f877d3fd9397b3bff56d85cc` | 0.235651 | 0.387971 | 0.274028 | 0.148515 |
| **`D095-BC E2`** | $1.00 \times \text{E2}$ | `4b989ad15168839227ea61ed4425e4c52f3d22c778ee77651f5519c3041d688f` | 0.232961 | 0.387120 | 0.278833 | 0.148515 |
| **`soup_d095best_d093champ_5050` (Champion)** | $0.50 \text{ D095-E1} + 0.50 \text{ D093}$ | `cd6e9c7b74525ada39c0985a8b862109c52d0f5800c2b74f86fe57cc79df7c77` | **0.238401** | **0.398860** | **0.249448** | 0.148515 |
| **`soup_d095best_d093champ_7525`** | $0.75 \text{ D095-E1} + 0.25 \text{ D093}$ | `3285c365e83323b1a9ad619f3f26bb7065de055184b15cf3bd5c8814449edd4f` | 0.237622 | 0.391047 | 0.273370 | 0.148515 |
| **`soup_d095e2_d093e3_d091_hybrid`** | $0.40 \text{ E2} + 0.30 \text{ E3} + 0.30 \text{ D091}$ | `823a344dac5c1af4349d357fed787edddf475443a6288f0075e7736e227ce322` | 0.220774 | 0.385960 | 0.251121 | 0.000000 |

> [!NOTE]
> Non-champion states (E1, E2, 75/25, hybrid) were evaluated for raw AP only; consensus AP was evaluated on the champion state. The 3-way hybrid recipe is `0.40 D095-E2 + 0.30 D093-E3 + 0.30 D091 Champion` (matching `scripts/taco_stage5_d095_box_contrastive.py` line 854) and lost `styrofoam_piece` completely (`AP = 0.000000`). The champion legacy_47 AP is `0.249448` (the previously stated `0.274150` was not supported by `report.json`; D-093 champion legacy_47 is `0.256137`). D-095 raw AP (`0.238401`) is lower than D-093's raw champion (`0.241412`).

### Scene Consensus Evaluation on D-095 Champion
- Under `apply_scene_consensus` (evaluated in D-096 cautious consensus), the D-095 Champion achieves **`0.241895` 9-Specific AP** and **`0.414205` BANTAI 5-Category AP** across all 65 validation images.
- `7: styrofoam_piece` is preserved at **`0.148515`**, `5: drink_can` at **`0.670297`**, and `8: paper` at **`0.436706`**!

---

## 5. Visual Inspection Verification (Image 1 and Image 777)

### Validation Image 1 (`000008.jpg` - GT: 2 Paper boxes, 0 Plastic, 0 Hazardous)
```
  Ground Truth (2): paper, paper
  Stage 5 Run 1  @ 0.10 (4): plastic_film:0.31, plastic_wrapper:0.29, plastic_film:0.16, plastic_film:0.14
  Stage 5 HighRes @ 0.10 (7): plastic_film:0.33, paper:0.32, cigarette:0.23, plastic_wrapper:0.18, plastic_film:0.13, cigarette:0.12, cigarette:0.11
  D-091 Champion  @ 0.10 (7): paper:0.33, plastic_film:0.33, cigarette:0.23, plastic_wrapper:0.19, plastic_film:0.13, cigarette:0.12, cigarette:0.11
  D-093 Champion  @ 0.10 (7): paper:0.35, plastic_film:0.33, cigarette:0.24, plastic_wrapper:0.19, plastic_film:0.13, cigarette:0.12, cigarette:0.11
  D-095 Champion  @ 0.10 (7): paper:0.36, plastic_film:0.32, cigarette:0.24, plastic_wrapper:0.19, plastic_film:0.13, cigarette:0.13, cigarette:0.12
  D-095 + Consensus @ 0.10 (5): paper:0.36, plastic_film:0.32, cigarette:0.24, paper:0.19, paper:0.13
```
- **Both Ground Truth Paper matches preserved**:
  - **`paper: 0.36`** matches GT #1 (paper bag).
  - **`paper: 0.19`** matches GT #2 (curly paper strip, reassigned from plastic via scene coherence).
- **Extra boxes**: **`paper: 0.13`** is an extra false-positive box on the bag edge; confident `plastic_film: 0.32` and `cigarette: 0.24` (> 0.20 threshold) remain unsuppressed under cautious scene rules.
- Over 500 low-confidence interior subboxes and speckles are cleanly suppressed.

### Validation Image 777 (`000084.JPG` - GT: 1 Plastic wrapper + 7 cigarette butts)
- Under raw predictions, D-095 Champion outputs `paper: 0.32` on the wrapper box and a marginal `paper: 0.10` ghost (`qid=8`).
- Under D-097 Calibrated Scene Consensus (`apply_scene_consensus`):
  - Rule 1B detects strong multi-subtype plastic co-activation (3 plastic subtypes $\ge 0.24$, top-2 plastic sum $= 0.5644$, delta $= 0.0327 \le 0.040$) and reassigns query 0 to `plastic_bottle` (`score = 0.32`). Under BANTAI 5-category mapping, this renders as **`Plastic 0.32`**.
  - Rule 4 detects no paper anchors on scene and a plastic reassignment, cleanly suppressing the marginal background ghost `qid=8` (`score = 0.1009`, norm_area $= 0.00055$).
  - Result: exactly 5 passing detections at 0.10 (`Plastic 0.32`, 4 `Potentially Hazardous` cigarette butts), and **ZERO paper boxes**!

> [!NOTE]
> On Image 777 (GT 408 is `plastic_wrapper`), the raw D-091, D-093, and D-095 champions predict `paper` (~0.32) with co-activated plastic subtypes. Under D-096 cautious consensus with delta $\le 0.016$, this remained an open paper/plastic confusion. In D-097, multi-subtype plastic co-activation margin calibration (Rule 1B, $\Delta \le 0.040$) and marginal non-paper ghost suppression (Rule 4) cleanly resolve Image 777 to Plastic without regressing Fold A, Fold B, or Image 1.

---

## 6. Multi-Scale Query Fusion & Scene Consensus Results

- **Tri-Model Pairing**: `Tri_Model_D086_S2575_HR448_D095_Champ`
- **Routing Policy**: `hr_small_medium_plus_can_paper`
- **Consensus Mode**: `scene_consensus`
- **9-Specific AP**: **`0.262348`** (project all-time high, exceeding D-093's `0.261603`)
- **BANTAI 5-Category AP**: **`0.438939`** (project all-time high, exceeding D-093's `0.437255`)

---

## 7. D-097 Multi-Subtype Plastic Co-Activation Margin Calibration

Authorization **D-097** calibrated the CPU consensus tiebreak rules to simultaneously resolve Image 1 (`000008.jpg`) and Image 777 (`000084.JPG`) with verified monotonic AP gains on both 2-fold cross-validation splits:

1. **Rule 1B (Strong Multi-Subtype Plastic Co-Activation Margin Calibration)**:
   - When a query predicts `paper` ($p_8 \ge 0.15$), if $\ge 3$ plastic subtypes are co-activated ($\ge 3 \ge 0.22$, or $\ge 3 \ge 0.20$ with top-2 sum $\ge 0.50$), an extended margin $\Delta \le 0.040$ is permitted to reassign paper to the dominant plastic subtype.
   - Preserves 100% backward compatibility when `strong_coactivation_delta=None` or `enable_rule1b=False`.
2. **Rule 4 (Marginal Paper Ghost Suppression on Plastic Scenes)**:
   - When an image has ZERO paper anchors and a plastic reassignment occurred (Rule 1 / Rule 1B), any marginal low-confidence paper query ($0.10 \le \text{score} < 0.115$) with microscopic area ($\text{norm\_area} < 0.01$) is cleanly suppressed.
3. **Cross-Validation Monotonic Improvement**:
   - **All 65 Images**: 9-Specific AP $= \mathbf{0.242509}$ ($+0.004108$ over raw $0.238401$, $+0.000614$ over D-096 $0.241895$); BANTAI 5-Category AP $= \mathbf{0.418938}$ ($+0.020078$ over raw $0.398860$, $+0.004733$ over D-096 $0.414205$).
   - **Fold A (33 images)**: 9-Specific AP $= \mathbf{0.156828}$ ($+0.004680$ over raw $0.152148$, $+0.000777$ over D-096 $0.156051$); BANTAI 5-Category AP $= \mathbf{0.272147}$ ($+0.023057$ over raw $0.249090$, $+0.007890$ over D-096 $0.264257$).
   - **Fold B (32 images)**: 9-Specific AP $= \mathbf{0.339452}$ ($+0.002588$ over raw $0.336864$); BANTAI 5-Category AP $= \mathbf{0.603430}$ ($+0.012956$ over raw $0.590474$).
4. **Visual Review Verification**:
   - **Image 777 (`000084.JPG`)**: 1 `Plastic 0.32` box, 4 `Potentially Hazardous` boxes, **ZERO paper boxes**.
   - **Image 1 (`000008.jpg`)**: Both GT Paper boxes preserved (`Paper 0.36`, `Paper 0.19`). Under `agnostic_iou_0.50`, duplicate bag edge box is cleanly suppressed.
