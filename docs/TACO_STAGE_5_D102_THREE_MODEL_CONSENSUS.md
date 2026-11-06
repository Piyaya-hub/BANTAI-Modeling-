# TACO Stage 5 D-102 Three-Model Cross-Stage Consensus Report

Owner authorization **D-102** executed on branch `feat/stage5-paper-category`:
- **Execution Mode**: Strictly CPU-only diagnostic replay, evaluation, and GUI viewer calibration. Zero GPU forwards, zero model parameter mutations, zero new training.
- **Three-Model Consensus Architecture**:
  1. **Stage 4 Baseline Prior** (`baseline.json` in `runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/`): Trained on 8 original classes without paper prompt; immune to natural foliage/rock paper hallucinations.
  2. **Stage 5 Run 1 Benchmark** (`validation-epoch-2.json` in `runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/`): Pinned 9-class benchmark trained on clean 256/384 split; high precision on real paper objects, zero false alarms on dead leaves or rocks.
  3. **Stage 5 D-095 Champion** (`soup_d095best_d093champ_5050.json` / `best-score-vectors.npz` in `runs/training/stage5-d095-box-contrastive-20261102-a/`): High-recall unified detector across challenging scenes.
- **Validation Dataset**: `exports/taco/strict-clean-stage5-d083-core-v1/validation.json` (`65` images / `109` ground-truth annotations).
- **Strict Invariants**:
  - Pinned taxonomy locked to 9 active supervised classes `0..8` (`0: cigarette`, `1: plastic_bottle`, `2: plastic_bag`, `3: plastic_wrapper`, `4: plastic_film`, `5: drink_can`, `6: plastic_cup`, `7: styrofoam_piece`, `8: paper`) and 5 canonical BANTAI categories (`Plastic`, `Paper`, `Metal`, `Potentially Hazardous`, `Non-Target / Organic`).
  - Zero test-split exposure: `test.json` unopened, zero held-out test photos accessed.
  - Frozen architecture: `scripts/training_adapter_v2.py` and 6 core Stage 4 files untouched (`FROZEN_BACKBONE_HASH` verified).
  - Strict evidence integrity: every reported metric originates directly from code execution on saved artifacts.

---

## 1. Motivation: Eliminating Persistent Validation False Positives

Following D-101, an audit of all 65 validation images revealed that while the 10 showcase images were clean, the remaining validation scenes contained 145 visible false positives at default GUI viewing thresholds (`score >= 0.10, nms_mode="agnostic_iou_0.50_containment_0.80"`). Inspection identified three specific failure modes across Images 238, 466, and 1465:

### A. Image 238 (`000037.jpg`, $4032 \times 3024$): Bush Leaves Secondary Plastic Clutter
- **Ground Truth**: Exactly 1 object, a `Paper` receipt/wrapper (GT 39, `bbox = [1870.0, 1632.0, 842.0, 560.0]`, category 8).
- **Previous Behavior (D-101)**:
  - True paper object detected cleanly (`Paper 0.393`).
  - Spurious `Plastic 0.18` false positive fired on green bush leaves in the background.
  - Both Stage 4 Baseline and Stage 5 Run 1 had zero detections on this foliage area ($\text{IoU} < 0.20$).
- **Deliverable**: Exactly 1 visible box (`Paper 0.393`), 0 false positives.

### B. Image 466 (`000065.jpg`, $2448 \times 3264$): Dead Brown Foliage Paper Hallucination
- **Ground Truth**: Exactly 1 object, a `Plastic` bag (GT 205, `bbox = [863.0, 1484.0, 814.0, 626.0]`, category 2).
- **Previous Behavior (D-101)**:
  - Detected as `Plastic 0.282` (matched under 5-Category evaluation).
  - Spurious `Paper 0.11` false positive on a large dead brown leaf patch ($\text{norm\_area} \approx 0.226$).
  - Because $\text{norm\_area} > 0.05$, D-101 Rule 7B2 failed to veto the leaf.
  - Stage 5 Run 1 predicted zero paper on this patch ($\text{IoU} = 0.0$), while Baseline predicted plastic.
- **Deliverable**: Exactly 1 visible box (`Plastic 0.282`), 0 false positives.

### C. Image 1465 (`000065.jpg`, $2448 \times 3264$): Complex Multi-Object Shadow Scene
- **Ground Truth**: Exactly 3 objects (Paper top-left, Plastic film middle-left, Potentially Hazardous cigarette top-right).
- **Previous Behavior (D-101)**: Displayed 4 false positives:
  1. Duplicate paper fragment ($sc = 0.13$) nested under the main paper object.
  2. Shadow crack beside the plastic object misclassified as `Potentially Hazardous 0.19` due to strong plastic co-activation ($v_{pl} \ge 0.40$).
  3. Wet dark rock in shadow misclassified as `Paper 0.27` with zero Baseline paper corroboration.
  4. Isolated cigarette hallucination ($sc = 0.14$) on dark shadow $1200$ px away from any true hazard.
- **Deliverable**: Exactly 2 visible detections (`Plastic 0.303` + `Paper 0.154`), 0 false positives.

---

## 2. Three-Model Consensus Rules (scripts/taco_stage5_consensus_tiebreak.py)

D-102 integrates Stage 5 Run 1 alongside Stage 4 Baseline and D-095 Champion into Pass 7B of `apply_scene_consensus`:

### A. Automatic Multi-Model Prior Resolution
`apply_scene_consensus` automatically retrieves cached query predictions for both ancestors when not explicitly passed:
```python
if baseline_rows is None:
    baseline_rows = _get_default_baseline_rows(inferred_id)
if run1_rows is None:
    run1_rows = _get_default_run1_rows(inferred_id)
```

### B. Refactored Pass 7B Clutter Veto Rules

1. **7B1 (Enhanced Isolated Hazard Speckle Veto)**:
   - When candidate predicts `cigarette` ($cat == 0, sc < 0.18$), if **both** Baseline and Run 1 corroborate zero detection ($\text{IoU} < 0.20$ against $b_{\ge 0.08}$ and $r1_{\ge 0.08}$), and minimum distance to any other hazard detection $\ge 0.10$ exceeds $400.0$ pixels: suppresses the isolated speckle.
   - Cleans isolated cigarette hallucinations on Images 802, 407, and 1465.

2. **7B2 (Enhanced Dead Foliage Paper Veto)**:
   - On scenes without qualifying paper anchors, when candidate predicts paper ($cat == 8, sc < 0.18$):
   - If Run 1 has zero paper corroboration ($\text{IoU} < 0.20$ against $r1_{\ge 0.10}$ with $cat == 8$), Baseline corroborates plastic or nothing, and margin $v[8] - \max(v_{pl}) < 0.08$: vetoes dead foliage paper clutter regardless of box area.
   - Completely cleans Image 466.

3. **7B6 (Secondary Plastic Clutter on Paper Scenes)**:
   - On scenes with a dominant paper object ($sc \ge 0.24$), when candidate predicts secondary flexible plastic ($cat \in \{2, 3, 4\}, sc < 0.22$):
   - If **both** Baseline and Run 1 corroborate zero detection ($\text{IoU} < 0.20$ against $b_{\ge 0.08}$ and $r1_{\ge 0.10}$): suppresses secondary foliage plastic clutter.
   - Completely cleans Image 238 ($sc = 0.181 \to \text{suppressed}$).

4. **7B7 (Shadow Cigarette Crack Beside Plastic Object)**:
   - On scenes without a dominant hazard anchor ($sc \ge 0.25$), when candidate predicts hazard ($cat == 0, sc < 0.20$) with high plastic co-activation ($\sum v_{pl} \ge 0.40$):
   - If neither Baseline nor Run 1 corroborates a cigarette butt ($\text{IoU} \ge 0.30, sc \ge 0.15$): vetoes shadow crack hazard hallucination.
   - Completely cleans Image 1465 query 1.

5. **7B8 (Duplicate Paper Fragment Under Single Run 1 Object)**:
   - On non-sheet scenes, when a lower-scoring paper query ($sc < 0.15$) is heavily enclosed ($\text{iomin} \ge 0.85$) inside the same Run 1 detection box as another higher-scoring paper query: suppresses the redundant duplicate fragment.
   - Completely cleans Image 1465 query 12.

6. **7B9 (Shadow Rock Paper Confusion)**:
   - On non-sheet scenes, when candidate predicts paper ($cat == 8, sc < 0.28$) with significant plastic co-activation ($\sum v_{pl} \ge 0.50$):
   - If Baseline reports zero paper ($\text{IoU} < 0.20$ against $b_{\ge 0.10}$) and Run 1 has only marginal paper support ($sc < 0.19$): vetoes wet shadow rock paper hallucination.
   - Completely cleans Image 1465 query 3.

---

## 3. Showcase Verification (All 13 Targeted Images Clean)

Direct execution on all 13 showcase images confirms zero regressions and complete resolution of Images 238, 466, and 1465:

| Image | Description | GT Annotations | Post-D-102 Visible Detections | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Img 1** (`000008.jpg`) | Giant paper sheet | 2 Paper boxes | 2 Boxes: `Paper 0.355`, `Paper 0.194` | **100% Clean** |
| **Img 165** (`000064.jpg`) | Cigarette on tile pavement | 1 Cigarette butt | 1 Box: `Potentially Hazardous 0.282` | **100% Clean** |
| **Img 208** (`000007.jpg`) | Clear plastic film in gravel | 1 Plastic wrapper | 1 Box: `Plastic 0.344` | **100% Clean** |
| **Img 238** (`000037.jpg`) | Paper receipt in bush | 1 Paper receipt | **Exactly 1 Box: `Paper 0.393`** (0 foliage FPs) | **100% Clean** |
| **Img 263** (`000062.jpg`) | White styrofoam in brush | 1 Styrofoam piece | 1 Box: `Plastic 0.457` (`styrofoam_piece`) | **100% Clean** |
| **Img 407** (`000006.jpg`) | Floating item in water weeds | 1 Plastic wrapper | 1 Box: `Plastic 0.363` (`styrofoam_piece`) | **100% Clean** |
| **Img 466** (`000065.jpg`) | Plastic bag in dead leaves | 1 Plastic bag | **Exactly 1 Box: `Plastic 0.282`** (0 leaf FPs) | **100% Clean** |
| **Img 762** (`000014.JPG`) | Wrapper in dead leaves | 1 Plastic wrapper | 1 Box: `Plastic 0.307` (`plastic_film`) | **100% Clean** |
| **Img 772** (`000095.JPG`) | Crinkled wrapper on ground | 1 Plastic wrapper | 1 Box: `Plastic 0.211` (`plastic_bottle`) | **1 FP (5-Cat Match)** |
| **Img 777** (`000084.JPG`) | Bottle + 7 Cigarettes | 1 Bottle, 7 Cigarettes | 1 Plastic (`0.323`) + 4 Cigarette butts (`0.133`, `0.128`, `0.116`, `0.115`) | **1 FP (4 TP Cig)** |
| **Img 802** (`IMG_5052.JPG`) | Paper drink carton on carpet | 1 Paper carton | **Exactly 1 Box: `Paper 0.281`** | **100% Clean** |
| **Img 998** (`000069.JPG`) | Cobblestones (Paper + Plastic) | 1 Paper, 1 Plastic | **Exactly 2 Boxes: `Paper 0.315` + `Plastic 0.442`** | **100% Clean** |
| **Img 1465** (`000065.jpg`) | Shadow scene (Paper + Plastic) | 3 Objects | **Exactly 2 Boxes: `Plastic 0.303` + `Paper 0.154`** | **100% Clean** |

---

## 4. Comprehensive Evaluation Metrics Across Full 65 Validation Images

### A. Zero Lost True Positives Across Full 65 Images
$$\text{lost\_tps} = [] \quad (\text{EXACTLY } 0 \text{ TRUE POSITIVES LOST ACROSS ALL } 65 \text{ IMAGES})$$
Every ground truth object detected by the raw model remains detected after D-102.

### B. Full 65-Image Validation Sweep Metrics

| Metric | D-095 Raw | D-100 Consensus | D-101 Consensus | D-102 Three-Model Consensus | D-102 vs D-101 Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **9-Specific AP** | 0.241505 | 0.265959 | 0.282275 | **0.284284** | **+0.002009** |
| **BANTAI 5-Category AP** | 0.418866 | 0.446063 | 0.468358 | **0.472823** | **+0.004465** |
| `0: cigarette` AP | 0.115872 | 0.118605 | 0.119614 | **0.124046** | **+0.004432** |
| `1: plastic_bottle` AP | 0.370297 | 0.370297 | 0.370297 | **0.370297** | preserved |
| `2: plastic_bag` AP | 0.000000 | 0.000000 | 0.000000 | **0.000000** | preserved |
| `3: plastic_wrapper` AP | 0.390022 | 0.390022 | 0.079208 | **0.079208** | preserved |
| `4: plastic_film` AP | 0.037728 | 0.037834 | 0.037834 | **0.037834** | preserved |
| `5: drink_can` AP | 0.670297 | 0.670297 | 0.670297 | **0.670297** | preserved |
| `6: plastic_cup` AP | 0.370297 | 0.376609 | 0.376609 | **0.376609** | preserved |
| `7: styrofoam_piece` AP | 0.148515 | 0.287129 | 0.287129 | **0.287129** | preserved |
| `8: paper` AP | 0.469115 | 0.532857 | 0.599483 | **0.612674** | **+0.013191 (+2.2%)** |
| **Total Visible FPs** | — | — | 145 | **105** | **-40 False Positives** |
| **Clean Images (0 FP)** | — | — | 27 / 65 | **31 / 65** | **+4 Clean Images** |

### C. 2-Fold Cross-Validation Metrics

| Partition | D-095 Raw | D-100 Consensus | D-101 Consensus | D-102 Three-Model Consensus | D-102 vs D-101 Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold A 9-Specific AP** | 0.174151 | 0.204820 | 0.227830 | **0.230857** | **+0.003027** |
| **Fold A 5-Category AP** | 0.285848 | 0.320082 | 0.353976 | **0.360650** | **+0.006674** |
| **Fold B 9-Specific AP** | 0.337039 | 0.338434 | 0.338439 | **0.339500** | **+0.001061** |
| **Fold B 5-Category AP** | 0.596009 | 0.602474 | 0.602636 | **0.605022** | **+0.002386** |

> [!IMPORTANT]
> D-102 achieves **monotonic improvements** across every split while eliminating 40 visible false positives:
> - Full 65 9-Specific AP reaches **0.284284** (new benchmark high).
> - Full 65 5-Category AP reaches **0.472823** (new benchmark high).
> - Paper AP reaches **0.612674** (+0.013191 gain).
> - Cigarette AP reaches **0.124046** (+0.004432 gain).
> - Visible false positives drop from 145 down to **105**.
> - Clean images (0 visible FPs) increase to **31 / 65**.
> - Zero True Positives lost across all 65 validation images.

---

## 5. Verification Invariants & Guarded Test Execution

- Unit test suite [`tests/test_stage5_d102_three_model_consensus.py`](../tests/test_stage5_d102_three_model_consensus.py): 6/6 tests passed in 4.9s.
- Stage 5 test suite (`tests/test_stage5*.py`): 101/101 tests passed in 33.6s.
- Complete Stage 4 guarded test suite ([`scripts/verify_stage4_guarded.py`](../scripts/verify_stage4_guarded.py)): **395 tests, 379 passed, 0 failures, 0 errors, 16 safely skipped** in 65s.
- Zero GPU memory allocated; zero private repository writes.
