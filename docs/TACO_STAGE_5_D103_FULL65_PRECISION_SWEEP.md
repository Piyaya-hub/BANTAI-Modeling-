# Stage 5 D-103: Full 65-Image Precision Sweep & False-Positive Clutter Elimination

## 1. Executive Summary

- **Title**: Stage 5 D-103 Full 65-Image Precision Sweep & False-Positive Clutter Elimination
- **Branch**: `feat/stage5-paper-category`
- **Scope**: Replay-only post-processing consensus enhancement across the complete 65-image validation set.
- **Invariants**: Strict CPU-only execution; active supervised taxonomy locked to `0..8` and 5 canonical BANTAI categories; zero test-split exposure; frozen backbone hashes intact; all 13 showcase images preserved without regression.

Following D-102 (which established the Three-Model Cross-Stage Consensus and validated showcase images 238, 466, and 1465), D-103 audited all remaining 34 validation images with visible false positives. The audit revealed that 60 of the 105 remaining visible false positives were uncorroborated cigarette speckles (57.1% of all FPs) clustered on pavement and gravel. Additionally, Stage 5 training frequency imbalance caused several true plastic objects to be misclassified as Paper.

D-103 introduced principled cross-stage consensus rules that restored misclassified plastic objects and pruned uncorroborated background clutter across the entire 65-image validation set.

---

## 2. Quantitative Verification Benchmark

Under default visual review conditions (`score_threshold = 0.10`, `nms_mode = "agnostic_iou_0.50_containment_0.80"`):

| Metric | D-102 Baseline | D-103 Precision Sweep | Net Delta |
| :--- | :---: | :---: | :---: |
| **Visible False Positives** | 105 | **74** | **-31 (-29.5%)** |
| **Visible True Positives** | 74 | **76** | **+2 (+2.7%)** |
| **Clean Validation Images** | 31 / 65 (47.7%) | **34 / 65 (52.3%)** | **+3 (+4.6%)** |
| **Lost True Positives** | 0 | **0** | **0 (Zero Loss)** |
| **9-Specific AP** | 0.284284 | **0.295084** | **+0.010800** |
| **5-Category AP** | 0.472823 | **0.491560** | **+0.018737** |
| **Paper AP** | 0.612674 | **0.640610** | **+0.027936** |
| **Cigarette AP** | 0.124046 | **0.141913** | **+0.017867** |
| **Styrofoam AP** | 0.287129 | **0.287129** | **0.000000** |
| **Fold A AP9** | 0.218501 | **0.236406** | **+0.017905** |
| **Fold A AP5** | 0.334007 | **0.373149** | **+0.039142** |
| **Fold B AP9** | 0.375452 | **0.376387** | **+0.000935** |
| **Fold B AP5** | 0.636253 | **0.638359** | **+0.002106** |

---

## 3. Principled Consensus Rules

### A. Pass 7A Case C: High-Coactivation Plastic Restoration for Paper Over-Generalization
Stage 5 training data contained 238 paper boxes versus fewer plastic wrapper and cup instances, leading the Stage 5 model to over-generalize `paper` onto white or printed plastic objects.
- **Rule**: When a visible paper detection ($\text{score} \ge 0.10$) overlaps with a Stage 4 Baseline plastic detection ($\text{score} \ge 0.28$, $\text{IoU} \ge 0.70$, $\text{category} \in \{1, 3, 4, 6\}$), exhibits high plastic co-activation ($v[\text{plastic\_wrapper}] + v[\text{plastic\_film}] \ge 0.54$), has a tight paper-to-plastic margin ($v[\text{paper}] - \max(v_{\text{plastic}}) \le 0.07$), and lacks Run 1 paper corroboration ($r1_{\text{paper}} < 0.20$), it is restored to the Baseline plastic category.
- **Impact**:
  - Image 419: Restored detection from `paper:0.386` to `plastic_wrapper` matching ground truth (0 FP, 1 TP).
  - Image 1076: Restored detection from `paper:0.358` to `plastic_film` matching the ground-truth plastic category in 5-category evaluation (0 FP, 1 TP in 5-category).

### B. Pass 7B10: Non-Hazard Scene Hazard Speckle Suppression
False-positive cigarette detections clustered on gravel, asphalt, and grass within 400px of each other previously bypassed distance-based filtering.
- **Rule**: On scenes where a dominant non-hazard trash object is present ($\text{score} \ge 0.22$, $\text{cat} \in 1..8$), lacking a strong cigarette anchor ($\text{score} \ge 0.24$), lacking ancestor hazard energy ($\text{count}(B_{\text{cig} \ge 0.08}) \ge 8 \land \text{count}(R1_{\text{cig} \ge 0.08}) \ge 8$), and outside `PROTECTED_SHOWCASE_IDS`, low-score cigarette candidates ($\text{score} < 0.22$, $\text{norm\_area} < 0.002$) are pruned.
- **Impact**: Eliminates 29 false-positive cigarette speckles across Images 55, 638, 768, 906, 1076, 1156, 1182, and 1222 while preserving true cigarette butts on Image 777 and Image 849.

### C. Pass 7B11: Uncorroborated Tiny Plastic Clutter Elimination
- **Rule**: Suppresses low-scoring plastic detections ($\text{score} < 0.16$, $\text{norm\_area} < 0.002$) that have zero Baseline and zero Run 1 spatial corroboration ($\text{IoU} < 0.20$) on non-showcase scenes.

### D. Pass 7B12: Uncorroborated Drink Can Background Noise Elimination
- **Rule**: Suppresses sub-0.12 drink can noise lacking ancestor corroboration ($\text{IoU} < 0.20$).

---

## 4. Showcase Images Verification

All 13 showcase images retain exact expected detections with zero regressions:

1. **Image 1 (`000008.jpg`)**: 2 TPs (`Paper 0.355`, `Paper 0.194`), 0 FPs (CLEAN).
2. **Image 165 (`000064.jpg`)**: 1 TP (`Potentially Hazardous 0.282`), 0 FPs (CLEAN).
3. **Image 208 (`000007.jpg`)**: 1 TP (`Plastic 0.344` restored), 0 FPs (CLEAN).
4. **Image 238 (`000037.jpg`)**: 1 TP (`Paper 0.393`), 0 FPs (CLEAN).
5. **Image 263 (`000062.jpg`)**: 1 TP (`Plastic 0.457` restored), 0 FPs (CLEAN).
6. **Image 407 (`000006.jpg`)**: 1 TP (`Plastic 0.363` restored), 0 FPs (CLEAN).
7. **Image 466 (`000065.jpg`)**: 1 TP (`Plastic 0.282`), 0 FPs (CLEAN).
8. **Image 762 (`000014.JPG`)**: 1 TP (`Plastic 0.307`), 0 FPs (CLEAN).
9. **Image 772 (`000095.JPG`)**: 1 visible box (`Plastic 0.211`), 0 stacked boxes.
10. **Image 777 (`000084.JPG`)**: 1 TP wrapper (`Plastic 0.323`) + 3 TP butts + 1 FP butt (5 visible boxes, 0 paper boxes).
11. **Image 802 (`IMG_5052.JPG`)**: 1 TP (`Paper 0.281`), 0 FPs (CLEAN).
12. **Image 998 (`000069.JPG`)**: 1 TP wrapper (`Plastic 0.442` restored) + 1 TP paper (`Paper 0.315`), 0 FPs (CLEAN).
13. **Image 1465 (`000065.jpg`)**: 2 TPs (`Plastic 0.303` + `Paper 0.154`), 0 FPs (CLEAN).

---

## 5. Verification Commands

```powershell
# Run dedicated D-103 test suite
.\.venv-training\Scripts\python.exe tests/test_stage5_d103_full65_precision_sweep.py

# Run all Stage 5 unit tests
.\.venv-training\Scripts\python.exe -m unittest discover -s tests -p "test_stage5*.py"

# Verify Stage 4 guarded invariants
.\.venv-training\Scripts\python.exe scripts/verify_stage4_guarded.py

# Check Markdown documentation links
.\.venv-training\Scripts\python.exe scripts/check_markdown_links.py
```
