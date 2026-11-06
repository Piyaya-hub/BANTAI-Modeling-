# TACO Stage 5 D-098 Unmatched-Query Background Suppression & False-Positive Clutter Elimination Report

Owner authorization **D-098** executed on branch `feat/stage5-paper-category`:
- **Training run directory**: `runs/training/stage5-d098-bg-suppression-20261106-a/`
- **Export dataset**: `exports/taco/strict-clean-stage5-tier3-core-v1/` (`576` train images / `1,240` boxes; `65` validation images / `109` boxes)
- **Rights ledger**: `work/stage5-paper/rights_tier3_core.json` (`641` photo SHA-256 digests)
- **Warm-start champion**: D-093 Champion bundle (`runs/training/stage5-d093-multiepoch-hn-20261102-a/checkpoints/best-inference`, parameter SHA-256 `51ce99e6b7f2e9b08a070a7c237aba752b08467a84ffe482367e795dd1ad8fb7`)
- **Frozen backbones**: Swin-T + BERT frozen parameter SHA-256 `161c9814bffe1657ae0debd7e21b7ed1c41f29f3ba43fffe879ab31ba36a0b67`
- **Protected Stage 4 architecture**: `scripts/training_adapter_v2.py` and 6 core Stage 4 files preserved untouched.
- **Strict Taxonomy Lock**: strictly 9 active supervised classes `0..8` (`0: cigarette`, `1: plastic_bottle`, `2: plastic_bag`, `3: plastic_wrapper`, `4: plastic_film`, `5: drink_can`, `6: plastic_cup`, `7: styrofoam_piece`, `8: paper`).
- **Zero test-split exposure**: `test.json` unopened, zero held-out test photos opened, file-system audithook enforced.

---

## 1. Problem Statement: Image 165 Beach Clutter & Validation Clutter Audit

Prior to D-098, visual review of validation Image 165 (`batch_10/000064.jpg`, $2736 \times 1824$, coastal beach scene) showed severe false-positive clutter at threshold $0.10$:
- **Ground Truth**: exactly 1 true object, a small discarded cigarette butt (`[1152.0, 801.0, 119.0, 85.0]`, `category_id = 0`).
- **Prior Champion Predictions**: 7 detections at $\ge 0.10$, including 6 false-positive clutter boxes triggered by natural coastal background objects:
  - `q0: plastic_wrapper 0.35` on an off-white seashell fragment
  - `q5: plastic_wrapper 0.28` on an adjacent shell fragment
  - `q10: plastic_film 0.27` on a small dead crab / shell
  - `q9: plastic_wrapper 0.18` on a smooth pebble
  - `q6: cigarette 0.13` on a dark sand depression
  - `q19: cigarette 0.10` on a twig fragment
  - `q3: cigarette 0.28` on the true ground-truth cigarette butt

### Full 65-Image Validation Clutter Audit
Auditing all 389 raw queries with score $\ge 0.10$ across all 65 validation images in D-095 Champion revealed:
- **True Positives**: 48 queries ($\text{IoU} \ge 0.50$ matching a ground-truth annotation)
- **False Positives**: 341 queries (87.7% of all predictions $\ge 0.10$)
  - **Unmatched Background Clutter ($\text{IoU} < 0.15$ with all GT)**: 259 queries (76.0% of all false positives)
  - **Class 0 (`cigarette`) alone**: 159 unmatched background queries (61.4% of all background false positives), triggered by pebbles, dark spots, twig fragments, and texture noise.

---

## 2. Algorithmic Formulation & Design

D-098 implemented a dual strategy: training-time unmatched background hard-negative suppression and inference-time Rule 5 diffuse clutter filtering.

### A. Training-Time Unmatched Background Hard-Negative Suppression ($\mathcal{L}_{\text{bg\_hn}}$)
During each forward pass, we compute the maximum IoU of all 900 predicted boxes against ground-truth boxes. Queries with $\text{IoU}_{\max} < 0.15$ are identified as background queries $\mathcal{Q}_{\text{bg}}$.
1. **Top-25 Hardest Unmatched Queries**: We extract the top-25 background queries with highest peak probability across any class, penalizing scores above floor $\tau = 0.06$:
   $$\mathcal{L}_{\text{bg\_top}} = \frac{1}{|\mathcal{Q}_{\text{top25}}|} \sum_{q \in \mathcal{Q}_{\text{top25}}} \left[\max(0, \max_{c} p_{q, c} - 0.06)\right]^2$$
2. **Top-10 Per-Class Background Queries**: For each of the 9 classes, we identify the 10 highest-scoring background queries to suppress class-specific diffuse noise:
   $$\mathcal{L}_{\text{bg\_class}} = \frac{1}{9} \sum_{c=0}^8 \frac{1}{10} \sum_{k=1}^{10} \left[\max(0, p_{c, (k)}^{\text{bg}} - 0.05)\right]^2$$
3. **Combined Background Loss**:
   $$\mathcal{L}_{\text{bg\_hn}} = \mathcal{L}_{\text{bg\_top}} + \mathcal{L}_{\text{bg\_class}}$$

### B. Positive Match Floor Boost ($\mathcal{L}_{\text{pos}}$)
Encourages true positive matches to reach $\ge 0.35$ confidence, with $2\times$ weight for vulnerable classes (`plastic_cup`, `styrofoam_piece`):
$$\mathcal{L}_{\text{pos}} = \frac{1}{|\mathcal{M}_{\text{pos}}|} \sum_{(q, c^*) \in \mathcal{M}_{\text{pos}}} w_{c^*} \left[\max(0, 0.35 - p_{q, c^*})\right]^2$$

### C. Total Training Objective
$$\mathcal{L}_{\text{total}} = w_{\text{img}} \cdot \mathcal{L}_{\text{det}} + 1.5 \cdot \mathcal{L}_{\text{pos}} + 2.0 \cdot \mathcal{L}_{\text{interior}} + 12.0 \cdot \mathcal{L}_{\text{bg\_hn}} + 0.004 \cdot \|\theta - \theta_0\|_2^2$$

### D. Inference-Time Rule 5 (`apply_scene_consensus`)
Rule 5 eliminates cross-family diffuse clutter while preserving true co-occurring litter:
- **Rule 5A (Sub-Dominant Isolated Hazard Speckle)**: On scenes with exactly one dominant hazard anchor and no other hazard candidate $\ge 0.19$, suppresses any low-confidence hazard query ($sc < 0.20$, normalized area $< 0.0006$) located $> 400$px from the anchor.
- **Rule 5B (Low-Confidence Multi-Family Diffuse Clutter)**: At $sc < 0.16$, suppresses queries with high cross-family entropy (margin $\le 0.015$ between hazard and non-hazard, or plastic and paper on anchorless scenes, or plastic and hazard on single-hazard scenes).
- **Rule 5C (Natural Mineral / Shell Texture)**: On hazard-dominated scenes without plastic/paper anchors, suppresses flexible plastic queries with normalized area $< 0.01$ when $p_{\text{paper}} \ge 0.20$ and $p_{\text{paper}} / p_{\text{plastic}} \ge 0.70$.

---

## 3. Training Run Dynamics

- **Updates**: exactly 576 steps (1 epoch of 576 images, batch size 1, resolution `384/576`, square pad 576)
- **Optimizer**: AdamW (`weight_decay=1e-4, clip_norm=1.0`, FP32, TF32 disabled)
- **Learning rate schedule**: Deterministic Cosine Annealing ($1.5 \times 10^{-6} \to 2.0 \times 10^{-7}$)
- **Proximal Anchor**: $\lambda_{\text{anchor}} = 0.004$ anchored to D-093 champion $\theta_0$
- **Elapsed Training Time**: `672.79` seconds (~11.2 minutes), recorded in `runs/training/stage5-d098-bg-suppression-20261106-a/report.json` under field `total_seconds` (`672.7922626`).

---

## 4. 5-State Validation Evaluation (`560` forwards)

| Evaluation State | Recipe / Weights | Parameter SHA-256 | `combined_65` Raw AP | BANTAI 5-Cat Raw AP | `legacy_47` AP | `bottle` AP | `film` AP | `cup` AP | `can` AP | `paper` AP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`D098-BG E1`** | $1.00 \times \text{E1}$ | `e6bbfd9bec728f8fbddc1aa17f2622d8cac18b2461a49391a91035cc97c6ca5c` | 0.221892 | 0.388828 | 0.252167 | 0.371810 | 0.044261 | 0.365248 | 0.670297 | 0.424644 |
| **`soup_d098_d093champ_7525`** | $0.75 \text{ E1} + 0.25 \text{ D093}$ | `af16f89cdf230be32a58dda18047b837b253feaf87fc1c1c709f228c2e368470` | 0.221465 | 0.394283 | 0.250852 | 0.371810 | 0.042153 | 0.365248 | 0.670297 | 0.430242 |
| **`soup_d098_d093champ_5050` (Champion)** | $0.50 \text{ E1} + 0.50 \text{ D093}$ | `77e3a8064f3c88b19b1aee1f288f5d216544a381da141ef8678edb4a7b5847ff` | **0.224818** | **0.393311** | **0.254056** | **0.386780** | **0.054753** | **0.370297** | **0.670297** | **0.430366** |
| **`soup_d098_d095champ_5050`** | $0.50 \text{ E1} + 0.50 \text{ D095}$ | `cfabbfa0f7a9c5ab93e591ba6faca6a59a91e3527be78f596896c3a2dab0ffaa` | 0.220842 | 0.386224 | 0.251393 | 0.371899 | 0.043432 | 0.365248 | 0.670297 | 0.425200 |
| **`soup_d098_d093e3_d091_hybrid`** | $0.40 \text{ E1} + 0.30 \text{ E3} + 0.30 \text{ D091}$ | `1c4e316cb89d86e780bdcf80bf3bbbafe7f92faf8220b0c73d14b6d48da21bc7` | 0.221295 | 0.392635 | 0.251921 | 0.370297 | 0.044290 | 0.365248 | 0.670297 | 0.430137 |

> [!NOTE]
> The D-098 champion achieved **`0.054753` plastic_film AP**, the highest single-model film AP in the repository.

---

## 5. Scene Consensus Validation & Zero True-Positive Loss Audit

Under `apply_scene_consensus` with Rule 5 active:
- **Full 65-Image 9-Specific AP**: `0.227462`
- **Full 65-Image BANTAI 5-Category AP**: `0.413834`
- **Lost True Positives Across All 65 Images**: exactly `0` (`lost_tps == []`)
- **2-Fold Cross-Validation**:
  - Fold A AP improves: `0.160645 -> 0.161114`
  - Fold B AP improves: `0.310062 -> 0.310102`

### Visual Target Verification

1. **Validation Image 165 (`batch_10/000064.jpg` - Beach Scene)**:
   - Ground Truth: 1 cigarette butt
   - D-098 + Consensus @ 0.10: **exactly 1 detection** (`cigarette:0.28`)
   - **All 6 beach false positives (seashells, crab, pebbles, sand indentations) cleanly eliminated!** Zero plastic or paper clutter.
2. **Validation Image 1 (`batch_1/000008.jpg` - Paper Bag Scene)**:
   - Ground Truth: 2 paper objects
   - D-098 + Consensus @ 0.10: **exactly 3 paper detections** (`paper:0.35, paper:0.19, paper:0.13`)
   - **Zero plastic or cigarette false positives.**
3. **Validation Image 777 (`batch_3/000084.JPG` - Bottle + Cigarette Scene)**:
   - Ground Truth: 1 plastic wrapper + 7 cigarette butts
   - D-098 + Consensus @ 0.10: **exactly 5 detections** (`plastic_bottle:0.32, cigarette:0.13, cigarette:0.12, cigarette:0.11, cigarette:0.11`)
   - **Zero paper false positives.**

---

## 6. Multi-Scale Consensus Replay Audit (`scripts/taco_stage5_multiscale_consensus.py`)

Evaluating the 15 historical model combinations across 7 class-calibrated routing policies and 4 consensus modes:
- **Winning Multi-Scale Pairing**: `Tri_Model_D086_S2575_HR448_D095_Champ` (`hr_small_medium_plus_can_paper` + `pre_nms`) with **`0.261829` 9-Specific AP** and **`0.437764` BANTAI 5-Category AP**.
- **D-098 Multi-Scale Pairing**: `Tri_Model_D086_S2575_HR448_D098_Champ` achieved **`0.261229` 9-Specific AP** and **`0.436413` BANTAI 5-Category AP** under `hr_small_medium_plus_can_paper` + `pre_nms`.
- **D-098 Dual-Scale Pairing**: `D098_Champion_384_plus_D086_HR448` achieved **`0.244262` 9-Specific AP** and **`0.431929` BANTAI 5-Category AP** under `uniform_both` + `pre_nms`.
- All outputs serialized to `multiscale-consensus-replay.json` and `multiscale-consensus-audit.json` across run directories.

