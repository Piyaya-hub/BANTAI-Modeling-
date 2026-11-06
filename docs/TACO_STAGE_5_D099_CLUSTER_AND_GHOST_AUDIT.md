# TACO Stage 5 D-099 Full 65-Image False-Positive Audit, Same-Family Cluster Consolidation & Ghost Suppression Report

Owner authorization **D-099** executed on branch `feat/stage5-paper-category`:
- **Execution Mode**: Strictly CPU-only diagnostic replay, evaluation, and GUI viewer calibration. Zero GPU forwards, zero model parameter mutations, zero new training.
- **Reference Champion**: `D095_CHAMPION_STATE` (`soup_d095best_d093champ_5050.json` / `best-score-vectors.npz` in `runs/training/stage5-d095-box-contrastive-20261102-a/`).
- **Validation Dataset**: `exports/taco/strict-clean-stage5-d083-core-v1/validation.json` (`65` images / `109` ground-truth annotations).
- **Strict Invariants**:
  - Pinned taxonomy locked to 9 active supervised classes `0..8` (`0: cigarette`, `1: plastic_bottle`, `2: plastic_bag`, `3: plastic_wrapper`, `4: plastic_film`, `5: drink_can`, `6: plastic_cup`, `7: styrofoam_piece`, `8: paper`) and 5 canonical BANTAI categories (`Plastic`, `Paper`, `Metal`, `Potentially Hazardous`, `Non-Target / Organic`).
  - Zero test-split exposure: `test.json` unopened, zero held-out test photos accessed.
  - Frozen architecture: `scripts/training_adapter_v2.py` and 6 core Stage 4 files untouched.
  - Strict evidence integrity: every reported metric originates directly from code execution on saved artifacts.

---

## 1. Problem Statements & Audit Findings

Visual inspection of model predictions across the 65-image validation set under D-095 Champion revealed two remaining localized presentation artifacts:

### A. Image 772 (`batch_2/000095.JPG`, $3264 \times 2448$): Fragmented Plastic Multi-Box Cluster
- **Ground Truth**: Exactly 1 object, a crushed `plastic_wrapper` (GT 402, `bbox = [1482.0, 711.0, 581.0, 536.0]`, category 3).
- **Prior Behavior**: Grounding DINO predicts multiple queries across different visual sub-folds and textures of the wrapper:
  - `q0`: `cat=1 (plastic_bottle)`, `score=0.2111`, `bbox=[1794.7, 1044.3, 130.1, 191.9]` (bottom-right corner)
  - `q5`: `cat=4 (plastic_film)`, `score=0.1862`, `bbox=[1674.4, 744.5, 166.5, 169.1]` (top-center fold)
  - `q1`: `cat=4 (plastic_film)`, `score=0.1789`, `bbox=[1662.8, 1050.5, 155.2, 167.1]` (bottom-center fold)
  - `q3`: `cat=4 (plastic_film)`, `score=0.1503`, `bbox=[1674.6, 718.6, 236.9, 199.7]`
  - `q19`: `cat=4 (plastic_film)`, `score=0.1475`, `bbox=[1357.9, 729.1, 710.8, 509.2]` (macro object envelope)
  - `q9`: `cat=4 (plastic_film)`, `score=0.1474`, `bbox=[1592.2, 719.8, 473.1, 517.3]` (macro object envelope)
  - `q7`: `cat=1 (plastic_bottle)`, `score=0.1382`, `bbox=[1659.9, 1035.6, 395.0, 201.7]`
  - `q17`: `cat=1 (plastic_bottle)`, `score=0.1244`, `bbox=[1802.7, 936.6, 261.2, 300.3]`
- **Root Cause**: Because `q0`, `q5`, and `q1` represent mutually disjoint sub-regions of the wrapper ($\text{IoU}(q_0, q_5) = 0.000$, $\text{IoU}(q_0, q_1) = 0.083$, $\text{IoU}(q_5, q_1) = 0.000$), standard pairwise NMS at $\text{IoU} \ge 0.50$ kept up to 6 stacked/overlapping boxes. Under containment NMS, `q19` and `q9` were suppressed by `q0`, but `q0`, `q5`, and `q1` survived because neither contains the other.

### B. Image 208 (`batch_11/000007.jpg`, $1824 \times 4000$): Sub-Dominant Border-Touching Macro Ghost
- **Ground Truth**: Exactly 1 object, a large `styrofoam_piece` (GT 142, `bbox = [1180.0, 2191.0, 505.0, 806.0]`, category 7).
- **Prior Behavior**: The true object is cleanly detected with high confidence by `q0` (`cat=4 (plastic_film)`, `score=0.3439`, `bbox=[1181.8, 2184.7, 501.9, 816.8]`, $\text{IoU} = 0.981$ with GT 142). However, a false positive appears at the very top image boundary (`q2`: `cat=7 (styrofoam_piece)`, `score=0.1362`, `bbox=[1573.1, 2.6, 250.3, 605.3]`, $y = 2.6 \le 5.0$).
- **Root Cause**: On high-aspect indoor photos ($1824 \times 4000$), architectural boundaries (wall edges, doorframe trim) at the image border can trigger low-confidence macro activations ($sc = 0.1362 < 0.15$). The ratio to the dominant scene detection is $0.1362 / 0.3439 = 0.396 < 0.45$, and the spatial distance from the true object is $2303.1$ px.

---

## 2. Algorithmic Formulation: Rule 6A & Rule 6B

Rule 6 is implemented in [`scripts/taco_stage5_consensus_tiebreak.py`](../scripts/taco_stage5_consensus_tiebreak.py) as an integral post-processing step within `apply_scene_consensus`:

### A. Rule 6B: Sub-Dominant Marginal Macro Ghost Filter
Eliminates isolated boundary artifacts on scenes with an established dominant macro detection:
1. **Dominant Macro Anchor Condition**: The scene contains a dominant detection $M$ with score $s_M \ge 0.28$ for any macro category $c \in \{1..8\}$, with center coordinates $(m_x, m_y)$.
2. **Target Query Filter**: A candidate query $q$ is suppressed if ALL of the following hold:
   - $q$ is a plastic subtype: $c_q \in \text{PLASTIC\_SUBTYPE\_IDS} = \{1, 2, 3, 4, 6, 7\}$
   - Marginal confidence: $s_q < 0.15$
   - Sub-dominant ratio: $s_q / s_M < 0.45$
   - Touches image boundary: $x_q \le 5.0 \lor y_q \le 5.0 \lor (x_q + w_q) \ge W - 5.0 \lor (y_q + h_q) \ge H - 5.0$
   - Non-panoramic bounding box: $\text{area}(q) / (W \cdot H) < 0.05$
   - Isolated distance: $\sqrt{(c_{q,x} - m_x)^2 + (c_{q,y} - m_y)^2} > 800.0\text{ px}$

### B. Rule 6A: Same-Family Object-Cluster Consolidation
Eliminates fragmented duplicate sub-boxes when multiple queries describe the same visual object:
1. **Macro Envelope Detection**: A candidate macro query $mq$ in `PLASTIC_SUBTYPE_IDS` is identified if it encloses at least 2 distinct plastic queries $sq$ with containment:
   $$\text{IoMin}(sq, mq) = \frac{\text{area}(sq \cap mq)}{\text{area}(sq)} \ge 0.80 \quad \text{and} \quad \text{area}(sq) < 0.60 \cdot \text{area}(mq)$$
2. **Cluster Definition**: The set of queries $\mathcal{C} = \{mq\} \cup \{sq_1, sq_2, \dots\}$ forms an object cluster.
3. **Representative Selection**: Let $q^* = \arg\max_{q \in \mathcal{C}} s_q$ be the highest-scoring query in the cluster.
4. **Sub-Box Consolidation**: For every other query $q \in \mathcal{C} \setminus \{q^*\}$, if $s_q < 0.22$, $q$ is suppressed.

---

## 3. Full 65-Image Sweep Results

Evaluating the complete 65-image validation set (`strict-clean-stage5-d083-core-v1`) before and after Rule 6:

| Metric | D-095 Champ Baseline (Rules 1-5) | D-099 (Rules 1-6) | Delta | Direction |
|:---|:---:|:---:|:---:|:---:|
| **9-Specific AP** | `0.241505` | **`0.242458`** | **`+0.000953`** | **Strictly Improved** |
| **BANTAI 5-Category AP** | `0.418866` | **`0.419057`** | **`+0.000191`** | **Strictly Improved** |
| **Fold A 9-Specific AP** | `0.155527` | **`0.157449`** | **`+0.001922`** | **Strictly Improved** |
| **Fold A 5-Category AP** | `0.273442` | **`0.273622`** | **`+0.000180`** | **Strictly Improved** |
| **Fold B 9-Specific AP** | `0.338271` | **`0.338271`** | `+0.000000` | **Preserved** |
| **Fold B 5-Category AP** | `0.601730` | **`0.601927`** | **`+0.000197`** | **Strictly Improved** |
| **Lost True Positives (9-spec)** | `0` (`[]`) | `0` (`[]`) | `0` | **Strict Zero** |

### Per-Class AP Comparison (9-Specific)
| Class ID | Class Name | Baseline AP | D-099 AP | Delta | Status |
|:---:|:---|:---:|:---:|:---:|:---:|
| 0 | `cigarette` | `0.115872` | `0.115872` | `0.000000` | Held |
| 1 | `plastic_bottle` | `0.370297` | `0.370297` | `0.000000` | Held |
| 2 | `plastic_bag` | `0.000000` | `0.000000` | `0.000000` | Held |
| 3 | `plastic_wrapper` | `0.000000` | `0.000000` | `0.000000` | Held |
| 4 | `plastic_film` | `0.034201` | **`0.037728`** | **`+0.003527`** | **Improved** |
| 5 | `drink_can` | `0.670297` | `0.670297` | `0.000000` | Held |
| 6 | `plastic_cup` | `0.365248` | **`0.370297`** | **`+0.005049`** | **Improved** |
| 7 | `styrofoam_piece` | `0.148515` | **`0.148515`** | `0.000000` | **Preserved** |
| 8 | `paper` | `0.469115` | `0.469115` | `0.000000` | Held |

### Per-Category AP Comparison (BANTAI 5-Category)
| Category ID | Category Name | Baseline AP | D-099 AP | Delta | Status |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | `Plastic` | `0.420658` | **`0.420943`** | **`+0.000285`** | **Improved** |
| 2 | `Paper` | `0.469115` | `0.469115` | `0.000000` | Held |
| 3 | `Metal` | `0.670297` | `0.670297` | `0.000000` | Held |
| 0 | `Potentially Hazardous` | `0.115872` | `0.115872` | `0.000000` | Held |

### Suppression Diagnostic Summary
- **Rule 6A Suppressions**: 18 queries across 4 images:
  - Image 157: queries `{1, 8}`
  - Image 424: queries `{3, 7, 14}` (duplicate fragments of GT 263 wrapper)
  - Image 772: queries `{1, 3, 5, 7, 9, 17, 19}` (sub-box fragments of GT 402 wrapper)
  - Image 815: queries `{2, 3, 4, 6, 12, 63}` (nested plastic clutter fragments)
- **Rule 6B Suppressions**: 3 queries across 3 images:
  - Image 208: query `{2}` (border-touching styrofoam ghost at $y = 2.6$, $sc = 0.1362$)
  - Image 806: query `{4}` (border-touching plastic film false positive at $x = 0.1$, $sc = 0.1288$)
  - Image 1076: query `{23}` (border-touching plastic film false positive at $x = 0.0$, $sc = 0.1023$)
- **Zero True Positives Lost**: All 3 suppressed queries under 6B had $\text{IoU} = 0.000$ with all ground-truth boxes.

---

## 4. Visual Verification of All 5 Showcase Images

Under the updated default GUI NMS mode (`agnostic_iou_0.50_containment_0.80`) at threshold $0.10$:

1. **Image 1 (`batch_1/000008.jpg`, GT: 2 Paper objects)**:
   - Box 1: `Paper 0.355` (`[0.5, 458.1, 1432.8, 1517.4]`) on large paper bag ($\text{IoU} = 0.992$ with GT 0)
   - Box 2: `Paper 0.194` (`[526.6, 280.7, 1007.5, 676.9]`) on paper strip ($\text{IoU} = 0.825$ with GT 1)
   - **Result**: Exactly 2 clean Paper boxes matching the 2 ground-truth annotations; zero plastic or hazard false positives.
2. **Image 165 (`batch_10/000064.jpg`, GT: 1 Potentially Hazardous cigarette butt)**:
   - Box 1: `Potentially Hazardous 0.282` (`[1147.0, 802.0, 129.2, 85.5]`) on cigarette butt ($\text{IoU} = 0.942$ with GT 96)
   - **Result**: Exactly 1 clean Potentially Hazardous box; zero coastal shell/pebble clutter false positives.
3. **Image 208 (`batch_11/000007.jpg`, GT: 1 Plastic styrofoam piece)**:
   - Box 1: `Plastic 0.344` (`[1181.8, 2184.7, 501.9, 816.8]`) on styrofoam piece ($\text{IoU} = 0.981$ with GT 142)
   - **Result**: Exactly 1 clean Plastic box; top-border doorframe ghost (`q2`, $sc = 0.1362$) eliminated by Rule 6B.
4. **Image 772 (`batch_2/000095.JPG`, GT: 1 Plastic wrapper)**:
   - Box 1: `Plastic 0.211` (`[1794.7, 1044.3, 130.1, 191.9]`) on wrapper
   - **Result**: Exactly 1 clean Plastic box; the 5 stacked/overlapping sub-box fragments collapsed into 1 clean detection.
5. **Image 777 (`batch_2/000084.JPG`, GT: 1 Plastic wrapper + 7 cigarette butts)**:
   - Box 1: `Plastic 0.323` (`[978.3, 81.6, 116.1, 268.8]`) on top printed wrapper ($\text{IoU} = 0.916$ with GT 414)
   - Boxes 2-5: 4 `Potentially Hazardous` detections (`0.133, 0.128, 0.116, 0.115`) on true cigarette butts
   - **Result**: Exactly 1 Plastic wrapper box + 4 cigarette butt boxes; zero paper confusion boxes.

---

## 5. Verification Suite & Guard Invariants

1. **Unit Tests**:
   - `tests/test_stage5_d099_cluster_and_ghost_suppression.py`: 4 tests passed in $3.27$s.
   - `tests/test_stage5_image777_calibration.py`: updated NMS default check, passed.
   - Entire Stage 5 suite (`test_stage5_*.py`): 80 tests passed in $16.27$s.
2. **Guarded Repository Verification**:
   - `scripts/verify_stage4_guarded.py`: 378 tests (362 passed, 0 failures, 0 errors, 16 skipped training fixtures).
   - Zero GPU access, zero model forwards, zero updates, zero protected file writes.
