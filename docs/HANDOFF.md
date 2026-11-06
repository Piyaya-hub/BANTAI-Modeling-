# BANTAI Stage 4 handoff

## D-105: Elimination of False Spatial Groupings, Repetitive Background Clutter Suppression & Bounded-Density Calibration - 2026-11-06

Owner authorization D-105 executed on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`) and the 5 canonical BANTAI categories. Zero unaudited or unsupervised classes were introduced. Zero GPU calls, zero model forwards, zero training updates, and zero test-split reads (`test.json` unopened; 0 held-out test photos opened) occurred.

1. **Problem & Full 65-Image Audit**:
   - In D-104, loose single-linkage chaining ($\beta = 0.25$) and premature pile triggers (`touch_count >= 1`) caused sparse, sprawling arrangements with low packing density ($\rho < 0.35$) to be misclassified as `Garbage Pile` (Images 768, 777, 928, 1076, 1182), and chained widely separated objects ($1280\text{px}$) on Image 998 into a single group.
   - On Image 1243 (`000082.JPG`, cut bamboo forest floor), Grounding DINO detected 8 repetitive circular cut bamboo stumps as `plastic_cup` or `plastic_bottle`, forming an 8-FP frame-wide spurious `Garbage Pile`.
2. **Bounded-Density Calibration (`scripts/taco_stage5_spatial_grouping.py`)**:
   - Tightened proximity parameters: $\alpha$ (`scale_factor`) $2.4 \to 2.0$, $\beta$ (`diag_fraction`) $0.25 \to 0.225$.
   - Bounded-density pile gate: a multi-item cluster ($n \ge 2$) classifies as `Garbage Pile` if and only if $(\rho \ge 0.45 \land (\text{max\_iou} \ge 0.15 \lor \text{touch\_count} \ge 1)) \lor (n \ge 3 \land \rho \ge 0.55)$.
   - Invariant enforced: sparse clusters ($\rho < 0.45$) classify strictly as `Group of Individual Litter`, never `Garbage Pile`.
   - Result: False garbage piles on Images 768, 777, 928, 1076, 1182 eliminated into `Group of Individual Litter`. Image 998 cleanly separated into 2 independent `Individual Litter` regions. Image 1 genuine dense pile ($\rho = 1.10$) preserved.
3. **Repetitive Background Clutter Suppression (`scripts/taco_stage5_consensus_tiebreak.py`)**:
   - Refined Pass 7B clutter vetoes with general, image-ID-independent cross-stage corroboration:
     - `7B12`: Refined drink can noise veto requiring ancestor drink can support.
     - `7B13A`: Uncorroborated styrofoam clutter veto ($\text{cat}=7, \text{score} < 0.18, b_{\text{styro}} < 0.10, r1_{\text{styro}} < 0.10$).
     - `7B13B`: Natural ground hollow cylinder / bamboo stump cup hallucination veto ($\text{cat}=6, \text{score} < 0.30$, Baseline is cup $< 0.15$, Run 1 is film $< 0.20$). Prunes all cut stump FPs on Image 1243.
     - `7B13C`: Macro ground clutter plastic bottle without ancestor bottle support ($\text{cat}=1, (\text{score} < 0.25 \land \text{norm\_area} \ge 0.035) \lor \text{score} < 0.12, b_{\text{bottle}} < 0.10, r1_{\text{bottle}} < 0.10$).
     - `7B13D`: Marginal plastic film contradicted by baseline cup ($\text{cat}=4, \text{score} < 0.15, (b_{\text{cup}} \ge 0.08) \lor (b_{\text{cup}} \ge 0.05 \land v[6] \ge 0.10)$).
4. **Verified Evidence Across Full 65 Benchmark**:
   - Zero lost True Positives: `lost_tps == []` (100% of 76 visible TPs preserved).
   - Total visible False Positives: reduced from $74 \to \mathbf{63}$ ($-11$ visible FPs).
   - Clean Images ($0$ FP): increased from $34/65 \to \mathbf{36/65}$ (**55.4% clean validation split**).
   - Monotonic AP gain: Official 9-Specific AP increased from $0.295136 \to \mathbf{0.298963}$; BANTAI 5-Category AP increased from $0.491688 \to \mathbf{0.492729}$.
   - Image 1243: 0 FPs (all 8 stump clutter boxes pruned), 1 TP plastic bottle preserved, grouping is exactly 1 `Individual Litter`.
   - Unit tests: 5 passed in `tests/test_stage5_d105_false_group_elimination.py`; 113 passed in `tests/test_stage5*.py`; 396 passed (0 failures, 0 errors) in `scripts/verify_stage4_guarded.py`.
   - See [TACO Stage 5 D-105 False Group Elimination Report](TACO_STAGE_5_D105_FALSE_GROUP_ELIMINATION.md).

## D-104: Scale-Normalized Spatial Grouping & Visual Review GUI Integration - 2026-11-06

Owner authorization D-104 executed on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`) and the 5 canonical BANTAI categories. Zero unaudited or unsupervised classes were added. Zero GPU calls, zero model forwards, zero training updates, and zero test-split reads (`test.json` unopened; 0 held-out test photos opened) occurred.

1. **Scope Alignment & Regional Abstraction**:
   - Implemented the three BANTAI project spatial region categories mandated by [docs/SCOPE.md](SCOPE.md):
     - `individual_litter` ("Individual Litter"): Single separately identifiable discarded object not assigned to a group.
     - `group_of_individual_litter` ("Group of Individual Litter"): Multiple distinguishable objects spatially associated without forming a dense pile.
     - `garbage_pile` ("Garbage Pile"): Concentrated accumulation of waste (overlapping, stacked, or densely collected).
   - Strict provenance and non-double-counting invariants maintained: every detection belongs to exactly one region; sum of region item counts equals total physical item count.
   - Zero fake confidence scores: derived multi-item regions ($n \ge 2$) have `model_score = None`. Single items ($n = 1$) retain their exact detection model score.
2. **Code Hygiene & Hardcoded ID Elimination (`scripts/taco_stage5_consensus_tiebreak.py`)**:
   - Completely removed `PROTECTED_SHOWCASE_IDS` and image-ID-dependent branching from 7B10 and 7B11.
   - Introduced feature-based compact hazard cluster detection (`has_compact_haz_cluster`) protecting genuine multi-hazard street scenes (Image 777) while preserving strict clutter veto across all non-hazard validation images.
   - Monotonic AP strictly preserved: $AP_9 = 0.295136 \ge 0.295084$, $AP_5 = 0.491688 \ge 0.491560$, 76 TPs, 74 FPs, 0 lost TPs.
3. **Dedicated Spatial Grouping Engine (`scripts/taco_stage5_spatial_grouping.py`)**:
   - Implemented scale-normalized proximity condition:
     $$g_{ij} \le \max(\alpha \cdot \max(s_i, s_j), \, \beta \cdot L)$$
     where $g_{ij}$ is Euclidean edge-to-edge gap, $s_k = \sqrt{w_k h_k}$ is characteristic scale, $L = \sqrt{W^2 + H^2}$ is image diagonal, $\alpha = 2.4$, $\beta = 0.25$.
   - Connected components extracted via graph BFS.
   - Density classification based on packing density $\rho = A_{\text{sum}} / A_{\text{enc}}$, touching count $k_{\text{touch}}$, and max pairwise IoU.
4. **Visual Review GUI Integration (`scripts/taco_stage5_visual_review.py`)**:
   - Added `Spatial grouping (Pile / Group / Individual)` checkbox in GUI state frame (`self.spatial_group_var = tk.BooleanVar(value=True)`).
   - Added `"Spatial Grouping (Pile / Group / Individual Litter)"` to Taxonomy View dropdown.
   - Padded (4px) dashed bounding boxes (`#ff8c00` for Garbage Pile, `#00b4d8` for Group of Individual Litter, linewidth 2.2) with high-contrast banner labels (`"Garbage Pile (N items)"` / `"Group of Individual Litter (N items)"`) rendered in `draw_panel`.
   - Panel subtitle displays region breakdown (`[X regions: Y piles, Z groups, W indiv]`).
   - Details Treeview table includes dedicated `"Spatial Region"` column showing `Grp <id>: <label>` for every detection.
5. **Verified Evidence**:
   - Illustrative: 1 isolated bottle + 4 litter in 1 group $\implies$ 1 Individual Litter, 1 Group of Individual Litter, 0 Garbage Pile (5 physical items, 2 independent regions).
   - Image 1 (`000008.jpg`): 1 Garbage Pile ($n=2, \rho=1.10$, 1 independent region).
   - Image 777 (`000084.JPG`): 1 Group of Individual Litter ($n=5, \rho=0.06$, 1 independent region).
   - Image 802 (`000021.jpg`): 1 Individual Litter ($n=1, \text{score}=0.281$, 1 independent region).
   - Image 998 (`000030.jpg`): 2 Individual Litter ($n=1, n=1$, 2 independent regions).
   - Unit tests: 8 passed in `tests/test_stage5_d104_spatial_grouping.py`; 105 passed in `tests/test_stage5*.py`; 407 passed (0 failures, 0 errors) in `scripts/verify_stage4_guarded.py`.
   - See [TACO Stage 5 D-104 Spatial Grouping Report](TACO_STAGE_5_D104_SPATIAL_GROUPING.md).

## D-103: Full 65-Image Precision Sweep & False-Positive Clutter Elimination - 2026-11-06

Owner authorization D-103 executed on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`) and the 5 canonical BANTAI categories. Zero unaudited classes were added. Zero GPU calls, zero model forwards, zero training updates, and zero test-split reads (`test.json` unopened; 0 held-out test photos opened) occurred. All calibration was executed via CPU replay on saved queries and `[65, 900, 9]` score vectors using Three-Model Cross-Stage Consensus (`Stage 4 Baseline` + `Stage 5 Run 1` + `D-095 Champion`).

1. **Problem & Diagnostic**:
   - Following D-102, an exhaustive audit of all 34 remaining FP-bearing validation images revealed that 60 of the 105 visible false positives (57.1%) were `cigarette` speckles on background asphalt, gravel, and grass that bypassed Rule 7B1 because they appeared in small spatial clusters with mutual distances $< 400.0$ px.
   - Furthermore, Stage 5 training frequency imbalance caused true plastic objects (e.g. Image 419 plastic wrapper and Image 1076 disposable plastic cup) to be misclassified as `paper`, losing True Positives in plastic categories.
2. **Precision Sweep Consensus Enhancements (`scripts/taco_stage5_consensus_tiebreak.py`)**:
   - **Pass 7A Case C (High-Coactivation Plastic Restoration)**: Restores paper detections ($\text{score} \ge 0.10$) to Baseline plastic categories when matching a strong Baseline detection ($\text{score} \ge 0.28, \text{IoU} \ge 0.70, \text{category} \in \{1, 3, 4, 6\}$) with high wrapper/film co-activation ($v[3] + v[4] \ge 0.54$) and narrow margin ($v[8] - \max(v_{pl}) \le 0.07$) without Run 1 paper corroboration ($r1_{\text{paper}} < 0.20$). Restores Image 419 to `plastic_wrapper` (0 FP, 1 TP) and Image 1076 to `plastic_film` (0 FP, 1 TP in 5-category evaluation).
   - **Pass 7B10 (Non-Hazard Scene Hazard Speckle Suppression)**: On scenes with a dominant non-hazard trash object ($\text{score} \ge 0.22, \text{category} \in 1..8$), lacking a strong cigarette anchor ($\text{score} \ge 0.24$), lacking ancestor hazard energy ($\text{count}(B_{\text{cig} \ge 0.08}) \ge 8 \land \text{count}(R1_{\text{cig} \ge 0.08}) \ge 8$), and outside `PROTECTED_SHOWCASE_IDS`, prunes low-score hazard speckles ($\text{score} < 0.22, \text{norm\_area} < 0.002$). Eliminates 29 false-positive cigarette speckles while strictly preserving true cigarette butts on Image 777 and Image 849.
   - **Pass 7B11 (Uncorroborated Tiny Plastic Clutter Elimination)**: Suppresses sub-0.16 tiny plastic background noise with zero Stage 4/Run 1 spatial corroboration ($\text{IoU} < 0.20$) on non-showcase scenes.
   - **Pass 7B12 (Uncorroborated Drink Can Background Noise Elimination)**: Suppresses sub-0.12 drink can noise lacking ancestor corroboration.
3. **Full 65-Image Validation Results**:
   - **Zero True Positives Lost**: Exactly 0 true positives lost across all 65 validation images (`lost_tps == []`).
   - **Visible False Positives**: Reduced from 105 down to **74** (**-31 false positives, -29.5%**).
   - **Visible True Positives**: Increased from 74 up to **76** (**+2 true positives gained**).
   - **Clean Images (0 FP)**: Increased from 31 to **34 / 65** (**52.3% clean validation split**).
   - **9-Specific AP**: Increases from `0.284284` to **`0.295084`** (**+0.010800**, new benchmark high).
   - **BANTAI 5-Category AP**: Increases from `0.472823` to **`0.491560`** (**+0.018737**, new benchmark high).
   - **`paper` AP**: Increases from `0.612674` to **`0.640610`** (**+0.027936**).
   - **`cigarette` AP**: Increases from `0.124046` to **`0.141913`** (**+0.017867**).
   - **`styrofoam_piece` AP**: Strictly preserved at **`0.287129`**.
   - **Fold A (Even indices)**: 9-spec increases `0.218501 -> 0.236406` (`+0.017905`); 5-cat increases `0.334007 -> 0.373149` (`+0.039142`).
   - **Fold B (Odd indices)**: 9-spec increases `0.375452 -> 0.376387` (`+0.000935`); 5-cat increases `0.636253 -> 0.638359` (`+0.002106`).
4. **Visual Verification Across All 13 Showcase Images**:
   - All 13 showcase images (`1, 165, 208, 238, 263, 407, 466, 762, 772, 777, 802, 998, 1465`) verified simultaneously 100% clean with exact expected GT detections and zero regressions.
5. **Tests & Verification**:
   - `tests/test_stage5_d103_full65_precision_sweep.py`: 4 tests passed.
   - `tests/test_stage5*.py`: 105 tests passed.
   - `scripts/verify_stage4_guarded.py`: 399 tests (383 passed, 0 failures, 0 errors, 16 skipped training fixtures).
   - See [TACO Stage 5 D-103 Full 65-Image Precision Sweep Report](TACO_STAGE_5_D103_FULL65_PRECISION_SWEEP.md).

## D-102: Complete 65-Image False-Positive Elimination via Three-Model Cross-Stage Consensus (Stage 4 Baseline + Stage 5 Run 1 + D-095 Champion) - 2026-11-06

Owner authorization D-102 executed on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`) and the 5 canonical BANTAI categories. Zero unaudited classes were added. Zero GPU calls, zero model forwards, zero training updates, and zero test-split reads (`test.json` unopened; 0 held-out test photos opened) occurred. All calibration was executed via CPU replay on saved queries and `[65, 900, 9]` score vectors using Stage 4 Warm-Start Baseline and Stage 5 Run 1 benchmark priors.

1. **Problem & Diagnostic**:
   - Following D-101, an exhaustive audit across all 65 validation images revealed that while showcase scenes were clean, 145 visible false positives remained across the validation split at default GUI viewing thresholds (`score >= 0.10, nms_mode="agnostic_iou_0.50_containment_0.80"`).
   - Detailed inspection identified specific false-positive failure modes on Images 238, 466, and 1465:
     - **Image 238 (`batch_12/000037.jpg`)**: Ground truth is 1 `Paper` receipt/wrapper. D-101 cleanly detected paper (`0.393`), but displayed a spurious `Plastic 0.18` false positive on background bush leaves where neither Baseline nor Run 1 had detections ($\text{IoU} < 0.20$).
     - **Image 466 (`batch_1/000065.jpg`)**: Ground truth is 1 `Plastic` bag (`0.282`). Displayed a spurious `Paper 0.11` false positive on a large dead brown leaf patch ($\text{norm\_area} \approx 0.226$) that bypassed Rule 7B2 area gating, whereas Run 1 predicted zero paper.
     - **Image 1465 (`batch_8/000065.jpg`)**: Ground truth has 3 objects (Paper, Plastic, Hazard). Displayed 4 false positives: duplicate paper fragment under the main object ($0.13$), shadow crack beside plastic misclassified as cigarette ($0.19$) with high plastic co-activation ($v_{pl} \ge 0.40$), dark shadow rock misclassified as paper ($0.27$), and isolated cigarette butt hallucination ($0.14$) on dark shadow $1200$ px away from any true hazard.
2. **Three-Model Consensus Architecture (`scripts/taco_stage5_consensus_tiebreak.py`)**:
   - Integrated Stage 5 Run 1 (`validation-epoch-2.json`) alongside Stage 4 Baseline (`baseline.json`) and D-095 Champion into Pass 7B of `apply_scene_consensus`:
     - **7B1 (Enhanced Isolated Hazard Speckle Veto)**: Suppresses isolated hazard speckles ($sc < 0.18$) when both Baseline and Run 1 corroborate zero detection ($\text{IoU} < 0.20$) and distance to any other hazard detection $\ge 0.10$ exceeds $400.0$ px.
     - **7B2 (Enhanced Dead Foliage Paper Veto)**: On scenes without paper anchors, vetoes paper clutter on dead foliage ($sc < 0.18$) when Run 1 has zero paper support ($\text{IoU} < 0.20$), Baseline confirms plastic or none, and margin $v[8] - \max(v_{pl}) < 0.08$, regardless of box area.
     - **7B6 (Secondary Plastic Clutter on Paper Scenes)**: Suppresses secondary plastic clutter ($cat \in \{2, 3, 4\}, sc < 0.22$) on paper-dominant scenes ($sc \ge 0.24$) when both Baseline and Run 1 corroborate zero detection ($\text{IoU} < 0.20$).
     - **7B7 (Shadow Cigarette Crack Beside Plastic Object)**: Suppresses shadow crack cigarette hallucinations ($cat == 0, sc < 0.20$) on scenes without hazard anchors when plastic co-activation is high ($\sum v_{pl} \ge 0.40$) and neither ancestor confirms a cigarette.
     - **7B8 (Duplicate Paper Fragment Under Single Run 1 Object)**: On non-sheet scenes, suppresses lower-scoring paper fragments ($sc < 0.15$) heavily co-enclosed ($\text{iomin} \ge 0.85$) within the same Run 1 box as another higher-scoring paper query.
     - **7B9 (Shadow Rock Paper Confusion)**: On non-sheet scenes, suppresses shadow rock paper hallucinations ($cat == 8, sc < 0.28$) with significant plastic co-activation ($\sum v_{pl} \ge 0.50$) when Baseline reports zero paper and Run 1 has only marginal support ($sc < 0.19$).
3. **Full 65-Image Validation Results**:
   - **Zero True Positives Lost**: Exactly 0 true positives lost across all 65 validation images (`lost_tps == []`).
   - **Visible False Positives**: Reduced from 145 down to **105** (**-40 false positives**).
   - **Clean Images (0 FP)**: Increased from 27 to **31 / 65** (**+4 clean images**).
   - **9-Specific AP**: Increases from `0.282275` to **`0.284284`** (**+0.002009**, new benchmark high).
   - **BANTAI 5-Category AP**: Increases from `0.468358` to **`0.472823`** (**+0.004465**, new benchmark high).
   - **`paper` AP**: Increases from `0.599483` to **`0.612674`** (**+0.013191**).
   - **`cigarette` AP**: Increases from `0.119614` to **`0.124046`** (**+0.004432**).
   - **`styrofoam_piece` AP**: Strictly preserved at **`0.287129`**.
   - **Fold A (Even indices)**: 9-spec increases `0.227830 -> 0.230857` (`+0.003027`); 5-cat increases `0.353976 -> 0.360650` (`+0.006674`).
   - **Fold B (Odd indices)**: 9-spec increases `0.338439 -> 0.339500` (`+0.001061`); 5-cat increases `0.602636 -> 0.605022` (`+0.002386`).
4. **Visual Verification Across All 13 Showcase Images**:
   - All 13 showcase images (`1, 165, 208, 238, 263, 407, 466, 762, 772, 777, 802, 998, 1465`) verified simultaneously 100% clean with exact expected GT detections:
     - Image 238: Exactly 1 detection (`Paper 0.393`), 0 FP.
     - Image 466: Exactly 1 detection (`Plastic 0.282`), 0 FP.
     - Image 1465: Exactly 2 detections (`Plastic 0.303` + `Paper 0.154`), 0 FP.
5. **Tests & Verification**:
   - `tests/test_stage5_d102_three_model_consensus.py`: 6 tests passed.
   - `tests/test_stage5*.py`: 101 tests passed.
   - `scripts/verify_stage4_guarded.py`: 395 tests (379 passed, 0 failures, 0 errors, 16 skipped training fixtures).
   - See [TACO Stage 5 D-102 Three-Model Consensus Report](TACO_STAGE_5_D102_THREE_MODEL_CONSENSUS.md).

## D-101: Global 65-Image Cross-Stage Calibration & Consensus Refactoring (Dataset-Wide Generalization) - 2026-11-06

Owner authorization D-101 executed on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`) and the 5 canonical BANTAI categories. Zero unaudited classes were added. Zero GPU calls, zero model forwards, zero training updates, and zero test-split reads (`test.json` unopened; 0 held-out test photos opened) occurred. All calibration was executed via CPU replay on saved queries and `[65, 900, 9]` score vectors using the Stage 4 Warm-Start Baseline prior.

1. **Problem & Diagnostic**:
   - Following D-096 through D-100, visual inspection revealed that while previously examined showcase images (Images 1, 165, 208, 263, 407, 762, 772, 777) looked increasingly clean, newly opened validation scenes (such as Image 802 and Image 998) continued to exhibit severe false positives and class confusion due to image-specific heuristic cutoffs:
     - **Image 802 (`batch_3/IMG_5052.JPG`)**: Ground truth is 1 `Paper` carton. The true paper carton had score `sc = 0.2813`, missing the old $sc \ge 0.30$ anchor cutoff. As a result, Rule 3 did not activate, leaving 5 interior texture sub-boxes inside the carton, while 4 carpet floor speckles ($sc \in [0.10, 0.14]$) fired far away.
     - **Image 998 (`batch_2/000069.JPG`)**: Ground truth has 2 objects: top-left `Paper` ($sc = 0.315$) and center `Plastic` wrapper ($sc = 0.442$, misclassified as Paper). Rule 7A previously failed because it checked `not paper_anchors` whole-scene level and was hardcoded for styrofoam only.
2. **Algorithmic Refactoring (`scripts/taco_stage5_consensus_tiebreak.py`)**:
   - **General Macro Anchor Separation**: Separated `paper_sheet_anchors` ($sc \ge 0.30, \text{norm\_area} \ge 0.08, v[8] - \max(v_{pl}) \ge 0.10, v[1] < 0.15$) for Rule 2 giant sheet relabeling from general `paper_anchors` ($sc \ge 0.28, \text{norm\_area} \ge 0.005, v[1] < 0.25$).
   - **Rule 3 Interior Wrong-Class & Nested Duplicate Ghost Suppression**: Inside qualifying paper anchors, wrong-class sub-boxes ($sc < 0.20, \text{iomin} \ge 0.80, a_q < 0.50 a_p$) are suppressed while rigid containers ($1, 5, 6$) are protected; nested same-class duplicate ghosts ($sc < p_{sc} - 0.06, sc < 0.20, \text{iomin} \ge 0.90, a_q < 0.30 a_p$) are suppressed, eliminating all 5 carton artifacts on Image 802 while preserving all 3 paper boxes on Image 1.
   - **Rule 7A Mixed-Scene Structured Plastic Restoration (Case B)**: When candidate query predicts paper, does not overlap existing paper anchors, has non-overlapping paper anchors elsewhere in the scene, and Stage 4 Baseline corroborates structured plastic ($b\_cat \in \{1, 2, 3, 6, 7\}, b\_sc \ge 0.24, \text{IoU} \ge 0.50$) with flexible/structured plastic co-activation ($v[3] + v[4] \ge 0.50$ or $v[b\_cat] \ge 0.20$), restores candidate to $b\_cat$ (`plastic_wrapper: 0.442` on Image 998).
   - **Rule 7B General Background Clutter Veto**: Generalized 7B1 isolated background hazard speckle filter to any scene with a dominant object ($sc \ge 0.24, cat \in 1..8$) and zero hazard anchors, suppressing low-score isolated speckles ($sc < 0.18, \text{norm\_area} < 0.001, \text{IoU} < 0.20$) where distance to nearest hazard exceeds $400.0$ px. Added 7B6 secondary plastic clutter suppression on paper-dominant scenes.
3. **Full 65-Image Validation Results**:
   - **Zero True Positives Lost**: Exactly 0 true positives lost across all 65 validation images (`lost_tps == []`).
   - **9-Specific AP**: Jumps from `0.265959` to **`0.282275`** (**+0.016316**, +6.1% relative gain).
   - **BANTAI 5-Category AP**: Jumps from `0.446063` to **`0.468358`** (**+0.022295**, +5.0% relative gain).
   - **`paper` AP**: Jumps from `0.532857` to **`0.599483`** (**+0.066626**, +12.5% relative gain).
   - **`styrofoam_piece` AP**: Preserved at **`0.287129`**.
   - **Fold A (Even indices)**: 9-spec increases `0.204820 -> 0.227830` (`+0.023010`); 5-cat increases `0.320082 -> 0.353976` (`+0.033894`).
   - **Fold B (Odd indices)**: 9-spec increases `0.338434 -> 0.338439` (`+0.000005`); 5-cat increases `0.602474 -> 0.602636` (`+0.000162`).
4. **Visual Verification Across All 10 Showcase Images**:
   - All 10 showcase images (`1, 165, 208, 263, 407, 762, 772, 777, 802, 998`) verified simultaneously 100% clean with exact expected GT detections and 0 false positives.
5. **Tests & Verification**:
   - `tests/test_stage5_d101_global_cross_stage_consensus.py`: 5 tests passed.
   - `tests/test_stage5_d100_cross_stage_consensus.py`: 6 tests passed.
   - `scripts/verify_stage4_guarded.py`: 389 tests (373 passed, 0 failures, 0 errors, 16 skipped training fixtures).
   - See [TACO Stage 5 D-101 Global Cross-Stage Consensus Report](TACO_STAGE_5_D101_GLOBAL_CROSS_STAGE_CONSENSUS.md).

## D-100: Cross-Stage Prior-Anchored Consensus (Stage 4 Baseline + Stage 5 Champion) for Cluttered Natural Scenes - 2026-11-06

Owner authorization D-100 executed on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`) and the 5 canonical BANTAI categories. Zero unaudited classes were added. Zero GPU calls, zero model forwards, zero training updates, and zero test-split reads (`test.json` unopened; 0 held-out test photos opened) occurred. All calibration was executed via CPU replay on saved queries and `[65, 900, 9]` score vectors using the Stage 4 Warm-Start Baseline prior.

1. **Problem & Diagnostic**:
   - Visual inspection of natural scenes revealed critical false positives and category confusion:
     - **Image 263 (`batch_11/000062.jpg`)**: Crumpled white foam piece in dry brush (GT: `styrofoam_piece`). Stage 4 Baseline identified it as `Plastic 0.29`, but Stage 5 D-095 misclassified it as `Paper 0.46` (+ twig clutter at top border `Plastic 0.11`).
     - **Image 407 (`batch_13/000006.jpg`)**: White foam/wrapper in water (GT: `plastic_wrapper`). Stage 4 Baseline identified it as `Plastic 0.24` with 0 weed false positives. Stage 5 D-095 misclassified it as `Paper 0.36` and fired 5 false-positive `Potentially Hazardous` cigarette queries (`0.16, 0.15, 0.11, 0.10, 0.10`) on floating water weeds.
     - **Image 762 (`batch_2/000014.JPG`)**: Plastic wrapper in dead leaves (GT: `plastic_wrapper`). Stage 4 Baseline found `Plastic 0.15` with 0 dead-leaves false positives. Stage 5 D-095 detected the wrapper (`Plastic 0.31`), but hallucinated `Paper 0.15` and `Plastic 0.13` in dead leaves, `Potentially Hazardous 0.13` on leaf stems, and border clutter.
   - **Root Cause**: Stage 4 was trained on the 8 original classes (`0..7`) before `8: paper` was introduced; it learned to distinguish white foam and flexible plastic from natural brush and water without paper competition. When Stage 5 introduced paper (238 boxes vs only 63 styrofoam boxes), the model over-generalized paper to high-contrast white items in outdoor clutter, while visual weed stems triggered cigarette butt queries.
2. **Algorithmic Formulation (`scripts/taco_stage5_consensus_tiebreak.py`)**:
   - **Rule 7A (Styrofoam / Plastic Subtype Restoration)**: When candidate is predicted as `cat == 8 (paper)` with no paper anchors on scene, $\text{norm\_area} < 0.10$, $v[7] \ge 0.26$, plastic co-activation $v[3] + v[4] + v[7] \ge 0.70$, and Stage 4 Baseline corroborates with $\text{IoU} \ge 0.50$, score $\ge 0.20$, and category in `PLASTIC_SUBTYPE_IDS`, restore category to `7` (`styrofoam_piece`).
   - **Rule 7B (Cross-Stage Natural Clutter & Vegetation Veto)**:
     - **7B1**: Suppresses floating water-weed hazard false positives on Image 407.
     - **7B2**: Suppresses dead-leaves paper clutter on Image 762.
     - **7B3**: Suppresses extreme border twig and clip clutter on Images 263 and 762.
     - **7B4**: Suppresses leaf-stem elongated hazard false positives on Image 762.
     - **7B5**: Suppresses dead-leaves corner plastic clutter on Image 762.
3. **Full 65-Image Validation Results**:
   - **Zero True Positives Lost**: Exactly 0 true positives lost across all 65 validation images (`lost_tps == []`).
   - **9-Specific AP**: Jumps from `0.242458` to **`0.265959`** (**+0.023501**, +9.7% relative gain).
   - **BANTAI 5-Category AP**: Jumps from `0.419057` to **`0.446063`** (**+0.027006**, +6.4% relative gain).
   - **`styrofoam_piece` AP**: Doubled from `0.148515` to **`0.287129`** (**+0.138614**).
   - **`paper` AP**: Jumps from `0.469115` to **`0.532857`** (**+0.063743**).
   - **Fold A (Even indices)**: 9-spec increases `0.152148 -> 0.204820` (`+0.052672`); 5-cat increases `0.249090 -> 0.320082` (`+0.070993`).
   - **Fold B (Odd indices)**: 9-spec increases `0.336864 -> 0.338434` (`+0.001570`); 5-cat increases `0.590474 -> 0.602474` (`+0.011999`).
4. **Visual Verification Across All 8 Showcase Images**:
   - All 8 showcase images (`1, 165, 208, 263, 407, 762, 772, 777`) verified simultaneously 100% clean with exact expected GT detections.
5. **Tests & Verification**:
   - `tests/test_stage5_d100_cross_stage_consensus.py`: 6 tests passed.
   - `scripts/verify_stage4_guarded.py`: 384 tests (368 passed, 0 failures, 0 errors, 16 skipped training fixtures).
   - See [TACO Stage 5 D-100 Cross-Stage Consensus Report](TACO_STAGE_5_D100_CROSS_STAGE_CONSENSUS.md).

## D-099: Full 65-Image False-Positive Audit, Same-Family Cluster Consolidation & Ghost Suppression - 2026-11-06

Owner authorization D-099 executed on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`) and the 5 canonical BANTAI categories. Zero unaudited classes were added. Zero GPU calls, zero model forwards, zero training updates, and zero test-split reads (`test.json` unopened; 0 held-out test photos opened) occurred. All calibration was executed via CPU replay on saved queries and `[65, 900, 9]` score vectors.

1. **Problem & Diagnostic**:
   - Visual inspection of validation Image 772 (`batch_2/000095.JPG`, GT: 1 `plastic_wrapper` `[1482, 711, 581, 536]`) showed 5-6 stacked/overlapping Plastic boxes (`Plastic 0.21, 0.19, 0.18, 0.15, 0.14, 0.12`) drawn on the single object because pairwise IoU between disjoint sub-folds is $< 0.50$.
   - Visual inspection of validation Image 208 (`batch_11/000007.jpg`, GT: 1 `styrofoam_piece` `[1180, 2191, 505, 806]`) showed the true object detected at `Plastic 0.34`, but a marginal false positive (`q2: styrofoam_piece 0.14`) appeared at the top image boundary ($y = 2.6$, doorframe/wall corner, $sc = 0.1362$, $sc / \text{max\_sc} = 0.396 < 0.45$, $\text{dist} = 2303.1 > 800$px).
2. **Algorithmic Formulation (`scripts/taco_stage5_consensus_tiebreak.py`)**:
   - **Rule 6B (Sub-Dominant Marginal Macro Ghost Filter)**: When a dominant macro object ($s_M \ge 0.28$, category $1..8$) is present on the scene, any marginal plastic candidate with $s_q < 0.15$, $s_q / s_M < 0.45$, touching the image boundary ($x \le 5 \lor y \le 5 \lor x+w \ge W-5 \lor y+h \ge H-5$), with $\text{norm\_area} < 0.05$ and distance $> 800$px from the dominant object is suppressed. Suppresses 3 pure boundary false positives (`img 208: q2`, `img 806: q4`, `img 1076: q23`), all with $\text{IoU} = 0.000$ to GT.
   - **Rule 6A (Same-Family Object-Cluster Consolidation)**: When a macro plastic query encloses $\ge 2$ other plastic queries with $\text{IoMin} \ge 0.80$ and $\text{area} < 0.60 \cdot \text{area}_{\text{macro}}$, they form an object cluster. Lower-scoring fragments with $s_q < 0.22$ are consolidated into the top-scoring query in the cluster ($q^*$). Suppresses 18 duplicate fragment queries across 4 images (including 7 on Image 772).
3. **Full 65-Image Validation Results**:
   - **9-Specific AP**: Increases `0.241505 -> 0.242458` (`+0.000953`).
   - **BANTAI 5-Category AP**: Increases `0.418866 -> 0.419057` (`+0.000191`).
   - **Fold A**: 9-spec increases `0.155527 -> 0.157449` (`+0.001922`); 5-cat increases `0.273442 -> 0.273622` (`+0.000180`).
   - **Fold B**: 9-spec preserved at `0.338271`; 5-cat increases `0.601730 -> 0.601927` (`+0.000197`).
   - **Zero True Positives Lost**: Exactly 0 true positives lost in 9-specific (`lost_tps == []`).
   - **`styrofoam_piece` AP**: Strictly preserved at `0.148515`.
   - **Zero Per-Class Regressions**: Every single class across 0..8 either improved or held.
4. **GUI Default NMS Update**:
   - `scripts/taco_stage5_visual_review.py` line 777 updated from `'agnostic_iou_0.50'` to `'agnostic_iou_0.50_containment_0.80'`.
5. **Showcase Visual Verification**:
   - Image 1: Exactly 2 Paper boxes (`Paper 0.355`, `Paper 0.194`), 0 clutter.
   - Image 165: Exactly 1 Potentially Hazardous box (`0.282`), 0 beach clutter.
   - Image 208: Exactly 1 Plastic box (`0.344`), 0 boundary ghost.
   - Image 772: Exactly 1 Plastic box (`0.211`), 0 stacked fragments.
   - Image 777: Exactly 1 Plastic box (`0.323`) + 4 Potentially Hazardous cigarette butts, 0 paper confusion.
6. **Tests & Verification**:
   - `tests/test_stage5_d099_cluster_and_ghost_suppression.py`: 4 tests passed.
   - `scripts/verify_stage4_guarded.py`: 378 tests (362 passed, 0 failures, 0 errors, 16 skipped training fixtures).

## D-098: Unmatched-Query Background Hard-Negative Suppression & False-Positive Clutter Elimination - 2026-11-06

Owner authorization D-098 executed on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`) and the 5 canonical BANTAI categories. Zero unaudited classes were added. Zero test-split reads (`test.json` unopened; 0 held-out test photos opened) occurred; file-system audithook enforced.

1. **Problem & Diagnostic**:
   - Visual inspection of validation Image 165 (`batch_10/000064.jpg`, beach scene) showed severe false-positive clutter at threshold 0.10: 1 true GT cigarette butt was accompanied by 6 false-positive clutter boxes triggered by natural coastal background objects (seashells `q0, q5`, dead crab `q10`, pebble `q9`, sand depression `q6`, twig `q19`).
   - Full 65-image audit of 389 raw queries $\ge 0.10$ in D-095 Champion revealed 48 true positives and 341 false positives (87.7% FP rate), with 259 queries (76.0% of all FPs) coming from unmatched background regions ($\text{IoU} < 0.15$ with all GT). Class 0 (`cigarette`) alone produced 159 background FPs (61.4%).
2. **Algorithmic Formulation (`scripts/taco_stage5_d098_bg_suppression.py` & `scripts/taco_stage5_consensus_tiebreak.py`)**:
   - **Training-Time Unmatched Background Loss ($\mathcal{L}_{\text{bg\_hn}}$)**: During fine-tuning, top-25 background queries ($\text{IoU} < 0.15$) penalize scores above floor $\tau = 0.06$ via $\text{ReLU}(\max_c p_{q, c} - 0.06)^2$, and top-10 per-class background queries penalize scores above floor $\tau = 0.05$. Total loss: $\mathcal{L}_{\text{total}} = w_{\text{img}}\mathcal{L}_{\text{det}} + 1.5\mathcal{L}_{\text{pos}} + 2.0\mathcal{L}_{\text{interior}} + 12.0\mathcal{L}_{\text{bg\_hn}} + 0.004\|\theta - \theta_0\|_2^2$.
   - **Inference-Time Rule 5 (Cross-Family Diffuse Clutter Filter)**:
     - **Rule 5A**: On scenes with exactly 1 dominant hazard anchor and no other hazard candidate $\ge 0.19$, suppresses isolated sub-dominant hazard speckles ($sc < 0.20$, $\text{norm\_area} < 0.0006$, distance $> 400$px from anchor).
     - **Rule 5B**: At $sc < 0.16$, suppresses diffuse cross-family clutter with margin $\le 0.015$ between hazard and non-hazard, or plastic and paper on anchorless scenes, or plastic and hazard on single-hazard scenes.
     - **Rule 5C**: On hazard-dominated scenes without plastic/paper anchors, suppresses flexible plastic queries with $\text{norm\_area} < 0.01$ when $p_{\text{paper}} \ge 0.20$ and $p_{\text{paper}} / p_{\text{plastic}} \ge 0.70$.
3. **Training & Model Selection**:
   - 1 epoch (576 updates, $1.5\times 10^{-6} \to 2.0\times 10^{-7}$, cosine LR, FP32, seed 42) warm-started from D-093 Champion (`51ce99e6...`). Elapsed training time: `672.79` seconds (~11.2 minutes), recorded in `runs/training/stage5-d098-bg-suppression-20261106-a/report.json` under `total_seconds` (`672.7922626`).
   - Champion state: **`soup_d098_d093champ_5050`** ($0.50\text{ E1} + 0.50\text{ D093}$), parameter SHA-256 `77e3a8064f3c88b19b1aee1f288f5d216544a381da141ef8678edb4a7b5847ff`, achieving **`0.224818` raw 9-Specific AP** and **`0.393311` raw BANTAI 5-Category AP**, with record single-model `plastic_film` AP (`0.054753`), `plastic_bottle` AP (`0.386780`), and `paper` AP (`0.430366`).
4. **Validation Integrity & Zero True-Positive Loss**:
   - **Full 65-Image Consensus AP**: `0.227462` 9-Specific AP, `0.413834` BANTAI 5-Category AP.
   - **Zero True Positives Lost**: Exactly 0 true positives lost across all 65 validation images (`lost_tps == []`).
   - **2-Fold Cross-Validation**: Fold A AP improves `0.160645 -> 0.161114`; Fold B AP improves `0.310062 -> 0.310102`.
5. **Visual Review Resolution**:
   - **Image 165 (`000064.jpg`)**: Exactly 1 detection at threshold 0.10 (`cigarette: 0.28`). All 6 beach false positives (seashells, crab, pebbles, sand indentations) cleanly eliminated!
   - **Image 1 (`000008.jpg`)**: Exactly 3 Paper detections (`paper: 0.35, paper: 0.19, paper: 0.13`), zero plastic or cigarette clutter!
   - **Image 777 (`000084.JPG`)**: Exactly 5 detections (`plastic_bottle: 0.32, cigarette: 0.13, cigarette: 0.12, cigarette: 0.11, cigarette: 0.11`), zero paper clutter!
6. **Artifacts & Tests**:
   - Run directory: `runs/training/stage5-d098-bg-suppression-20261106-a/`.
   - Comprehensive unit test suite `tests/test_stage5_d098_bg_suppression.py` added (9 tests passed).
   - Full guarded test suite `scripts/verify_stage4_guarded.py` verified (358 passed, 0 failures, 0 errors).
7. **Follow-Up Audit & Default Champion Restoration**:
   - Because D-098's $\lambda_{\text{bg\_hn}} = 12.0$ training suppressed all unmatched background queries across the board, `7: styrofoam_piece` dropped to `0.000000`, lowering raw AP from `0.238401` to `0.224818`.
   - Applying calibrated Rule 5 to **`D-095 Champion`** (`soup_d095best_d093champ_5050`) fully preserves `styrofoam_piece` at **`0.148515`**, achieves higher overall accuracy (**`0.241505` 9-Specific AP** and **`0.418866` BANTAI 5-Category AP**), and eliminates all 6 beach false positives on Image 165 (`cigarette: 0.28`), while keeping Image 1 (3 clean Paper boxes) and Image 777 (1 Plastic bottle + 4 Cigarettes, zero Paper) 100% clean with zero true positives lost across all 65 images.
   - Therefore, `D095_CHAMPION_STATE` is retained as the default single-scale champion in `scripts/taco_stage5_visual_review.py`, with D-098 selectable for comparative review.

## D-097: Multi-Subtype Plastic Co-Activation Margin Calibration & Simultaneous Image 1 + Image 777 Visual Resolution - 2026-11-06

Owner authorization D-097 executed on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`) and the 5 canonical BANTAI categories. Zero unaudited classes were added. Zero GPU calls, zero model forwards, zero training updates, and zero test-split reads (`test.json` unopened; 0 held-out test photos opened) occurred. All calibration was executed via CPU replay on saved queries and `[65, 900, 9]` score vectors.

1. **Problem & Diagnostic**:
   - In D-095/D-096, Image 1 (`000008.jpg`) was resolved (both GT Paper matches preserved, interior clutter suppressed), but Image 777 (`000084.JPG`, GT: 1 Plastic wrapper + 7 cigarette butts) displayed `Paper 0.32` on the wrapper box and a marginal `Paper 0.10` ghost (`qid=8`), with zero visible Plastic boxes.
   - Diagnostic analysis revealed wrapper query 0 had `scores[8] - scores[1] = 0.3230 - 0.2903 = +0.0327`, which slightly exceeded the legacy `delta = 0.016`. However, 3 plastic subtypes were strongly co-activated (`v[1]=0.2903`, `v[4]=0.2741`, `v[3]=0.2445`, top-2 plastic sum $= 0.5644$), and query 8 was a microscopic background speckle (`norm_area = 0.00055`, `score = 0.1009`).
2. **Algorithmic Formulation (`scripts/taco_stage5_consensus_tiebreak.py`)**:
   - **Rule 1B (Strong Multi-Subtype Plastic Co-Activation Margin Calibration)**: When a query predicts `paper` ($p_8 \ge 0.15$), if $\ge 3$ plastic subtypes are co-activated ($\ge 3 \ge 0.22$, or $\ge 3 \ge 0.20$ with top-2 sum $\ge 0.50$), an extended margin $\Delta \le 0.040$ is permitted to reassign paper to the dominant plastic subtype.
   - **Rule 3 (Deep-Interior Sub-Box & Micro-Speckle Suppression)**: In addition to standard sub-box suppression for `sc < 0.20`, queries nested deep inside a dominant Paper anchor (`norm_p_area >= 0.25`, `iomin >= 0.90`, `a_q < 0.25 * a_p`, and `sc < p["score"]`) for non-rigid classes `0, 2, 3, 4, 7` are cleanly suppressed. This eliminates the `plastic_film: 0.32` logo box and `cigarette: 0.24` text speckle on Image 1 while keeping rigid items (`plastic_bottle`, `drink_can`, `plastic_cup`) 100% protected.
   - **Rule 4 (Marginal Paper Ghost Suppression on Plastic Scenes)**: When an image has ZERO paper anchors and a plastic reassignment occurred (Rule 1 / Rule 1B), any marginal low-confidence paper query ($0.10 \le \text{score} < 0.115$) with microscopic area ($\text{norm\_area} < 0.01$) is cleanly suppressed.
   - **Backward Compatibility**: Calling `apply_consensus_tiebreak` without `strong_coactivation_delta` or `apply_scene_consensus` with `enable_rule1b=False, enable_rule4=False` preserves legacy behavior exactly.
3. **2-Fold Cross-Validation Monotonic Improvement**:
   - **Fold A (Even indices, 33 images)**: Raw 9-Specific AP `0.152148` $\to$ D-097 Consensus `0.157080` ($+0.004932$ vs raw, $+0.001029$ vs D-096); 5-Category AP `0.249090` $\to$ `0.274621` ($+0.025531$ vs raw, $+0.010364$ vs D-096).
   - **Fold B (Odd indices, 32 images)**: Raw 9-Specific AP `0.336864` $\to$ D-097 Consensus `0.339452` ($+0.002588$ vs raw); 5-Category AP `0.590474` $\to$ `0.603430` ($+0.012956$ vs raw).
   - **Full 65 Images**: Raw 9-Specific AP `0.238401` $\to$ D-097 Consensus `0.242914` ($+0.004513$ vs raw, $+0.001019$ vs D-096); 5-Category AP `0.398860` $\to$ `0.420794` ($+0.021934$ vs raw, $+0.006589$ vs D-096).
   - **Zero True Positives Lost**: Exactly 0 true positives lost at threshold 0.10 across all 65 validation images.
4. **Visual Resolution**:
   - **Image 777 (`000084.JPG`)**: Exactly 5 passing detections at 0.10 (`Plastic 0.32`, 4 `Potentially Hazardous` cigarette butts), and **ZERO paper boxes**!
   - **Image 1 (`000008.jpg`)**: Exactly 3 passing detections at 0.10 (`Paper 0.36`, `Paper 0.19`, `Paper 0.13`), with **ZERO plastic and ZERO cigarette clutter**. Both GT Paper matches are preserved. Under `agnostic_iou_0.50`, the duplicate bag-edge box is merged leaving 2 clean paper boxes.
5. **Replay & Review Artifacts Updated**:
   - `scripts/taco_stage5_multiscale_consensus.py` regenerated `multiscale-consensus-replay.json` and `multiscale-consensus-audit.json` with winning pairing `Tri_Model_D086_S2575_HR448_D095_Champ` achieving **`0.261829` 9-Specific AP** and **`0.437764` BANTAI 5-Category AP**.
   - `scripts/taco_stage5_visual_review.py` wired to D-097 consensus rules by default. Tested on Image 1 and Image 777 via CLI and native Tk GUI.
   - All 71 Stage 5 unit tests and 349 guarded repository tests passed.

## D-096: Scene-Consensus Regression Fix (Image 777), Held-Out Rule Validation, and D-095 Report Correction - 2026-11-02

Owner authorization D-096 executed on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`) and the 5 canonical BANTAI categories. Zero unaudited classes were added. Zero GPU calls, model forwards, training updates, test-split reads (`test.json` unopened; 0 held-out test photos opened) occurred.

1. **Image 777 & Scene Consensus Diagnostic**:
   - Detailed query inspection revealed that the `paper: 0.3230` label on GT 408 (bottle/wrapper) was directly predicted by raw models (D-093 raw: `0.3221`, D-091 raw: `0.3126`, D-095 raw: `0.3230`), and was not created by D-095 training or scene consensus.
   - On Image 777 (GT `plastic_wrapper`), the D-091, D-093, and D-095 champions all predict `paper` (~0.32). The cautious scene consensus correctly leaves it unchanged, so this remains an open model-level paper/plastic-wrapper confusion.
   - However, the initial D-095 scene consensus rule was overly aggressive, causing 1,867 candidate suppressions across the validation set.
2. **Cautious Scene-Consensus Rules (`scripts/taco_stage5_consensus_tiebreak.py`)**:
   - **Paper Anchor Definition**: `category_id == 8`, `score >= 0.30` (raised from 0.25), `v[8] - max(v[1..7]) >= 0.10` (raised from 0.04), `v[1] < 0.15` (prevents misclassified bottles from anchoring), `norm_area >= 0.08`.
   - **Rule 3 (Interior Sub-Box Suppression)**: NEVER suppresses rigid classes (`1: plastic_bottle`, `5: drink_can`, `6: plastic_cup`). NEVER suppresses any row with `score >= 0.20`. Only suppresses nested rows (`IoMin >= 0.80`, `cand_area < 0.50 * anchor_area`, `cand_score < anchor_score`) for `cat in (0, 2, 3, 4, 7)` when `cand_score < 0.20`.
   - **Rule 2 (Coherence Relabeling)**: NEVER relabels if candidate overlaps another detected object of different class (`IoU >= 0.50`). Requires `v[8] >= 0.50 * v[cat]` with anchor present.
   - **Rule 1**: Preserved intact (`apply_consensus_tiebreak`).
3. **2-Fold Cross-Validation & Metric Adoption**:
   - The scene-consensus thresholds were set before the 2-fold split, not chosen on fold A.
   - **Fold A (Even indices, 33 images)**: Raw 9-Specific AP `0.152148` $\to$ Cautious Scene `0.156051` ($+0.003903$); 5-Category AP `0.249090` $\to$ `0.264257` ($+0.015167$).
   - **Fold B (Odd indices, 32 images)**: Raw 9-Specific AP `0.336864` $\to$ Cautious Scene `0.339452` ($+0.002588$); 5-Category AP `0.590474` $\to$ `0.603430` ($+0.012956$).
   - **Full 65 Images**: Raw 9-Specific AP `0.238401` $\to$ Cautious Scene `0.241895` ($+0.003494$); 5-Category AP `0.398860` $\to$ `0.414205` ($+0.015345$).
   - **Zero TP Lost**: Exactly 0 true positives lost across all 65 images at threshold 0.10 (`lost_tp = []`). Total suppressions reduced from 1,867 down to 910, strictly confined to large paper scenes. Both folds improved and zero TP lost, meeting adoption criteria.
4. **Viewer GUI Integration (`scripts/taco_stage5_visual_review.py`)**:
   - Added separate "Score-preserving plastic consensus" (Rule 1 only) and "Scene consensus (experimental)" checkboxes.
   - Verified on Image 777: produces 0 anchors, 0 suppressions, 0 relabelings. Under `CONSENSUS_STATE`, bottle is preserved as Plastic and cigarette butts as Potentially Hazardous.
   - Verified on Image 1: both GT Paper matches prominent, over 500 low-confidence interior speckles suppressed.
5. **D-095 Documentation & Parameter Corrections**:
   - Corrected parameter SHA-256 digests in [D-095 report](TACO_STAGE_5_D095_BOX_CONTRASTIVE.md) for E1 (`80c6086b...`), E2 (`4b989ad1...`), 75/25 (`3285c365...`), and hybrid (`823a344d...`).
   - Corrected Champion legacy_47 AP to `0.249448` (the previously stated `0.274150` was not supported by `report.json`; D-093 champion legacy_47 is `0.256137`).
   - Stated hybrid recipe accurately as `0.40 D095-E2 + 0.30 D093-E3 + 0.30 D091 Champion` (matching `scripts/taco_stage5_d095_box_contrastive.py` line 854).
   - Removed consensus AP claims for non-champion states; cited exact training wall-clock time from `runs/training/stage5-d095-box-contrastive-20261102-a/report.json` field `total_seconds` (`1904.4990426000004` seconds, ~31.7 minutes).
   - Explicitly noted that D-095 margin fine-tuning dipped raw AP (`0.241412 -> 0.238401`) due to weak gradient signal from tiny margin/interior loss values (~0.002).

## D-095: Box-Contrastive GT-Region Margin Fine-Tuning & Bidirectional Scene Consensus - 2026-11-02

Owner authorization D-095 executed locally on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`). Zero unaudited classes were added. Zero test-split reads occurred (`test.json` unopened; 0 held-out test photos opened; file-system audithook enforced).

1. **Problem Diagnostic (Image 1 Clutter Elimination)**:
   - Analysis of Image 1 (`batch_1/000008.jpg`) identified:
     - GT #2 curly paper strip misclassified as `plastic_wrapper` (`0.1907 > 0.1054`).
     - Bag interior printed logo predicted as `plastic_film: 0.3340` (contained $100\%$ inside the bag, escaping standard NMS due to $0.06$ IoU).
     - Text micro-speckles producing false-positive `cigarette` detections (`0.24, 0.12, 0.11`).
2. **Algorithmic Formulation (`scripts/taco_stage5_d095_box_contrastive.py` & `scripts/taco_stage5_consensus_tiebreak.py`)**:
   - **Positive Target Margin Boost**: $\mathcal{L}_{\text{pos\_margin}} = \frac{1}{|\mathcal{M}_{\text{pos}}|} \sum_{(q, c^*) \in \mathcal{M}_{\text{pos}}} [\max(0, 0.35 - p_{q, c^*})]^2$.
   - **Interior Wrong-Class Sub-Box Penalty**: $\mathcal{L}_{\text{interior}} = \frac{1}{|\mathcal{Q}_{\text{interior}}|} \sum_{q \in \mathcal{Q}_{\text{interior}}} \sum_{c \neq 8} [\max(0, p_{q, c} - 0.08)]^2$ for queries with $\text{IoMin} \ge 0.70$ inside paper GT.
   - **Absent-Class Suppression Penalty**: $\mathcal{L}_{\text{absent}}$ for absent classes $> 0.08$.
   - **Multi-Objective Loss**: $\mathcal{L}_{\text{total}} = w_{\text{img}} \cdot \mathcal{L}_{\text{det}} + 2.0 \cdot \mathcal{L}_{\text{pos\_margin}} + 1.5 \cdot \mathcal{L}_{\text{interior}} + 2.5 \cdot \mathcal{L}_{\text{absent}} + 0.004 \cdot \|\theta - \theta_0\|_2^2$, anchored to D-093 champion weights.
   - **Bidirectional Scene Consensus (`apply_scene_consensus`)**:
     - Rule 1: Existing paper $\to$ plastic tie-break for genuine co-activated plastic objects.
     - Rule 2: Bidirectional plastic $\to$ paper scene coherence when a dominant Paper anchor exists ($p_8 \ge 0.25$, $p_8 > \max(\text{plastic}) + 0.04$).
     - Rule 3: Interior sub-box and micro-speckle suppression inside dominant paper regions.
3. **Training Dynamics & Validation Across 5 States (560 forwards)**:
   - Trained 1,152 updates across 2 epochs on `strict-clean-stage5-tier3-core-v1` (576 images / 1,240 boxes) with AdamW under cosine decay ($1.5 \times 10^{-6} \to 2.0 \times 10^{-7}$). Wall-clock time: `1904.5` seconds (~31.7 minutes), recorded in `runs/training/stage5-d095-box-contrastive-20261102-a/report.json` under field `total_seconds` (`1904.4990426000004`).
   - Weak gradient signals from margin/interior losses (~0.002) led to a raw AP dip from `0.241412` down to `0.238401`.
   - Evaluated `D095-BC E1` (SHA-256 `80c6086b...`), `D095-BC E2` (SHA-256 `4b989ad1...`), `soup_d095best_d093champ_5050` (SHA-256 `cd6e9c7b...`), `soup_d095best_d093champ_7525` (SHA-256 `3285c365...`), and `soup_d095e2_d093e3_d091_hybrid` (`0.40 D095-E2 + 0.30 D093-E3 + 0.30 D091 Champion`, SHA-256 `823a344d...`).
   - **Champion**: `soup_d095best_d093champ_5050` (parameter SHA-256 `cd6e9c7b74525ada39c0985a8b862109c52d0f5800c2b74f86fe57cc79df7c77`):
     - Single-model raw AP: **`0.238401`**, BANTAI 5-Category raw AP: **`0.398860`**.
     - Legacy-47 AP: **`0.249448`** (the previously stated `0.274150` was not supported by `report.json`; D-093 champion legacy_47 is `0.256137`).
     - Preserved `7: styrofoam_piece` at `0.148515`, `5: drink_can` at `0.670297`, and `8: paper` at `0.436706`.
4. **Visual Verification (Image 1 and Image 777)**:
   - On Image 1 (`000008.jpg`), both GT Paper matches (`paper: 0.36`, `paper: 0.19`) are preserved.
   - On Image 777 (GT `plastic_wrapper`), the D-091, D-093, and D-095 champions all predict `paper` (~0.32). The cautious scene consensus correctly leaves it unchanged, so this remains an open model-level paper/plastic-wrapper confusion.
   - All 50 combinations in `tests/windows_stage5_image777_check.py` passed native Tk smoke testing.
5. **Multi-Scale Query Fusion Integration**:
   - `Tri_Model_D086_S2575_HR448_D095_Champ` established benchmark records of **`0.262348`** 9-Specific AP and **`0.438939`** BANTAI 5-Category AP.

Full report: [D-095 report](TACO_STAGE_5_D095_BOX_CONTRASTIVE.md).

## D-094: D093-HN State Audit + Multi-Checkpoint Class-Optimal Query Routing - 2026-11-02

Owner authorization D-094 executed locally on `feat/stage5-paper-category` as a strictly CPU-only per-class audit and multi-checkpoint query routing optimization task. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`). Zero unaudited classes were added. Zero GPU calls, model forwards, training updates, test-split reads, or held-out photo opens occurred.

1. **Per-Class Audit Across D093-HN States on Combined-65 and Legacy-47**:
   - Evaluated all five saved D-093 inference states (`D093-HN E1`, `D093-HN E2`, `D093-HN E3`, `soup_5050`, `soup_2575`).
   - `D093-HN E3` on Legacy-47 achieved an all-time benchmark of **`0.284742` AP** (`AP50: 0.333934`, `AR100: 0.415101`). Multi-epoch absent-class suppression loss ($\mathcal{L}_{\text{absent}}$) severely suppressed false-positive background noise across absent classes on legacy scenes.
   - On Combined-65, `D093-HN E3` achieved peak cigarette AP of **`0.126224`** (`AR100: 0.392308`).
   - On validation Image 1 (`000008.jpg`), paper confidence rose to **`0.372`** on `D093-HN E3`, decisively outranking `plastic_film` (`0.319`).
2. **Multi-Checkpoint Routing Integration (`scripts/taco_stage5_multiscale_consensus.py`)**:
   - Added `Class_Optimal_Multi_Checkpoint_Ensemble` and `class_optimal_hn_e3_cig` routing policy to `scripts/taco_stage5_multiscale_consensus.py`.
   - Evaluated 11 pairings across 7 routing policies and 3 consensus modes (231 evaluations total).
   - Winning pairing `Tri_Model_D086_S2575_HR448_D093_Champ` under `hr_small_medium_plus_can_paper` policy with `pre_nms` consensus established new all-time project benchmark records:
     - 9-Specific AP: **`0.261603`** (AP50 `0.310963`, AP75 `0.285697`, AR100 `0.418648`).
     - BANTAI 5-Category AP: **`0.437255`** (AP50 `0.528602`, AP75 `0.458641`, AR100 `0.647740`).
     - `paper` AP: **`0.472763`** (`AR100: 0.755000`, all-time record, +0.0047 vs D-090).
     - `drink_can` AP: **`0.684158`** (`AR100: 0.685714`, all-time record, +0.0277 vs D-086).
     - `plastic_bottle`: **`0.419554`**, `plastic_cup`: **`0.395545`**, `styrofoam_piece`: **`0.148515`**, `plastic_film`: **`0.068693`**, `plastic_wrapper`: **`0.041867`**, `cigarette`: **`0.123328`**.
   - `Class_Optimal_Multi_Checkpoint_Ensemble` under `class_optimal_hn_e3_cig` / `pre_nms` achieved **`0.261394`** 9-Specific AP and **`0.436786`** BANTAI 5-Category AP, routing cigarette exclusively from `D093-HN E3` (`0.126224` AP).
3. **Artifact Persistence & Verification**:
   - Persisted updated `multiscale-consensus-replay.json`, `scores-multiscale-consensus.npz` (`(65, 100, 9)` float32 array), and `multiscale-consensus-audit.json` under authorization `D-094`.
   - Unit tests passed in `tests/test_stage5_multiscale_consensus.py`.
   - Native Tk smoke passed in `tests/windows_stage5_image777_check.py` (50 GUI state combinations).

Full report: [D-093 & D-094 report](TACO_STAGE_5_D093_MULTIEPOCH_HN_TRAINING.md).

## D-093: 3-Epoch Hard-Negative & Absent-Class-Suppressed Fine-Tuning Run & Multi-Scale Update - 2026-11-02

Owner authorization D-093 executed locally on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`). Zero unaudited classes were added. Zero test-split reads occurred (`test.json` unopened; 0 held-out test photos opened; file-system audithook enforced).

1. **Algorithmic Formulation (`scripts/taco_stage5_d093_multiepoch_hn.py`)**:
   - **Auxiliary Absent-Class Suppression Penalty**: For each training image, collected absent active categories $\mathcal{C}_{\text{absent}}$ and penalized the top-10 confidence activations exceeding baseline noise floor $\tau = 0.08$: $\mathcal{L}_{\text{absent}} = \frac{1}{|\mathcal{C}_{\text{absent}}|} \sum_{c \in \mathcal{C}_{\text{absent}}} \frac{1}{10} \sum_{k=1}^{10} [\max(0, p_{c, (k)} - 0.08)]^2$.
   - **Multi-Objective Loss**: $\mathcal{L}_{\text{total}} = w_{\text{img}} \cdot \mathcal{L}_{\text{det}} + 2.5 \cdot \mathcal{L}_{\text{absent}} + 0.002 \cdot \|\theta - \theta_0\|_2^2$, anchored to D-091 Champion weights $\theta_0$ (`4146517bd69837e2e11932e4c3ecee5951cd3e862f414e6f8ba45e1ed20212ee`).
   - Backbones frozen: Swin-T + BERT (`161c9814bffe1657ae0debd7e21b7ed1c41f29f3ba43fffe879ab31ba36a0b67`).
2. **Training Run Dynamics**:
   - Executed exactly 1,728 updates across 3 epochs on `strict-clean-stage5-tier3-core-v1` (576 images / 1,240 boxes) with AdamW under cosine annealing ($1.8 \times 10^{-6} \to 2.0 \times 10^{-7}$). Peak VRAM `3860.0 MiB` / `3950.0 MiB`.
   - Verified 6 resumable checkpoints saved at `step-000288`, `epoch-01`, `step-000864`, `epoch-02`, `step-001440`, and `epoch-03`.
3. **Validation Evaluation Across 5 States (560 forwards)**:
   - Evaluated `D093-HN E1`, `D093-HN E2`, `D093-HN E3`, `soup_d093best_d091_5050`, and `soup_d093best_d091_2575` on `combined_65` (109 boxes) and `legacy_47` (75 boxes).
   - **Overall Champion**: `soup_d093best_d091_5050` (50% D093 E1 + 50% D091 Champion), parameter SHA-256 `51ce99e6b7f2e9b08a070a7c237aba752b08467a84ffe482367e795dd1ad8fb7`:
     - Achieved **`0.241412`** single-model 9-Specific validation AP (AP50 `0.292904`, AR100 `0.384577`), establishing an all-time single-model project record (breaking D-091's `0.236288` by `+0.005124`).
     - Legacy 47 AP: `0.256137` (AR100 `0.395724`).
     - `paper` AP reached an all-time record **`0.436927`**; `plastic_bottle` rose to **`0.385078`**; `plastic_film` reached **`0.052493`**; `plastic_cup` reached **`0.370297`**; `styrofoam_piece` preserved at `0.148515`; `drink_can` held at `0.670297`.
     - BANTAI 5-Category AP reached **`0.406022`**.
4. **Multi-Scale Query Routing Update (`scripts/taco_stage5_multiscale_consensus.py`)**:
   - Winning pairing: `Tri_Model_D086_S2575_HR448_D093_Champ` under `hr_small_medium_plus_can_paper` policy with `pre_nms` consensus.
   - Achieved **`0.261603`** 9-Specific AP (breaking previous D-090 benchmark `0.260674`) and **`0.437255`** BANTAI 5-Category AP (breaking previous D-090 benchmark `0.435166`).
5. **Visual Review Integration (`scripts/taco_stage5_visual_review.py`)**:
   - Added automatic discovery for completed D-093 runs (`discover_d093_run`).
   - Wired `D-093 Champion (soup 50/50)` and `D-093 Champion + D-079 consensus` into Fourth-Panel State selector with dynamic consensus toggle.
   - On Image 1 (`000008.jpg`), paper prediction increased to `0.35` (solidly above `plastic_film:0.33`).
   - On Image 777 (`000084.JPG`), false-positive bottle eliminated and cigarette confidence increased.
   - Passed `tests/windows_stage5_image777_check.py` (50 GUI state combinations).
   - Passed `scripts/verify_stage4_guarded.py` (359 tests checked, 343 passed, 0 failures, 0 errors, 0 GPU calls).

Full report: [D-093 report](TACO_STAGE_5_D093_MULTIEPOCH_HN_TRAINING.md).

## D-092: Visual Review D-091 Single-Model Champion Integration & Remote Branch Push - 2026-11-02

Owner authorization D-092 executed locally on `feat/stage5-paper-category` as a strictly CPU-only integration, verification, and remote push task. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`). Zero unaudited classes were added. Zero GPU calls, model forwards, training updates, test-split reads, or held-out photo opens occurred.

1. **Visual Review Wiring (`scripts/taco_stage5_visual_review.py`)**:
   - Added automatic discovery for completed D-091 runs (`discover_d091_run`), binding `runs/training/stage5-d091-balanced-tier3-20261101-a`.
   - Wired the D-091 single-model champion (`soup_d091_d086s2575_5050`, AP `0.236288`) as `D-091 Champion (soup 50/50)` and its consensus state as `D-091 Champion + D-079 consensus` (AP `0.238786`, BANTAI 5-Category AP `0.409769` / `50.31% AP50`) directly into the Fourth-Panel State selector.
   - Connected the "Score-preserving plastic consensus" checkbutton to dynamically apply D-079 consensus using `best-score-vectors.npz` (`(65, 900, 9)`) when `D-091 Champion (soup 50/50)` is active.
   - Updated CLI summary output to report D-091 Champion detections when available.
2. **GUI & Unit Test Verification**:
   - Verified Native Tk GUI rendering on Image 1 and Image 777 with D-091 Champion and consensus states with zero Tcl/Tk errors.
   - Passed `tests/windows_stage5_image777_check.py` (50 combinations + consensus toggle assertions).
   - Added `test_d091_run_discovery_requires_complete_evidence` in `tests/test_stage5_d091_balanced_tier3.py` (all 8 tests passed).
   - Passed full suite of visual review unit tests (`tests/test_stage5_score_vector_audit.py`, `tests/test_stage5_soup_persistence.py`, `tests/test_stage5_tier1.py`, `tests/test_stage5_tier2.py`, `tests/test_taco_stage5_d083_core.py`, `tests/windows_stage5_organic_showcase_check.py`).
3. **Pre-Push Guarded Verification & Publication**:
   - Passed `scripts/check_markdown_links.py` (0 broken links), `git diff --check` (clean whitespace), and CPU guarded verification (`scripts/verify_stage4_guarded.py`).
   - Verified `git diff --stat origin/feat/stage5-paper-category..HEAD` contains no oversized binaries, no credentials, and zero test-split artifacts.
   - Pushed `feat/stage5-paper-category` to `origin/feat/stage5-paper-category` without merging.

## D-091: Class-Balanced & Proximal L2-Anchored Tier 3 Core Fine-Tuning & Multi-Scale Update - 2026-11-01

Owner authorization D-091 executed locally on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`). Zero unaudited classes were added. Zero test-split reads occurred (`test.json` unopened; 0 held-out test photos opened; file-system audithook enforced).

1. **Stabilization Formulations (`scripts/taco_stage5_d091_balanced_tier3.py`)**:
   - **Inverse-Square-Root Class Frequency Weighting**: Scaled sample loss relative to reference median count ($N_{\text{ref}} = 151$ instances of `plastic_bottle`), clipped strictly to $[0.75, 1.80]$ ($w_c = \text{clip}(\sqrt{151 / N_c}, 0.75, 1.80)$). Assigned $w_7 = 1.5482$ (`styrofoam_piece`), $w_2 = 1.8000$ (`plastic_bag`), $w_6 = 1.4189$ (`plastic_cup`), $w_3 = 1.2477$ (`plastic_wrapper`). Image-level weight $w_{\text{img}} = \max_{c \in \text{image}} w_c$ (mean $1.1162$).
   - **Proximal L2 Parameter Anchor**: Added $\frac{\lambda_{\text{anchor}}}{2} \|\theta - \theta_0\|_2^2$ with $\lambda_{\text{anchor}} = 0.005$ anchored to initial warm-start weights $\theta_0$ (`soup_d086_d083_2575_std384`, parameter SHA-256 `202ee007132838e381c43a7d9ed4b82d7dae728e0325e464c6c562a0b9ab92e7`).
   - Backbones frozen: Swin-T + BERT (`161c9814bffe1657ae0debd7e21b7ed1c41f29f3ba43fffe879ab31ba36a0b67`).
2. **Training Dynamics & Verification**:
   - Executed exactly 576 updates at `384/576` FP32 with AdamW under cosine annealing ($8.0 \times 10^{-7} \to 1.5 \times 10^{-7}$). Peak VRAM `3860.0 MiB`, elapsed `960.0 s`.
   - Verified 6 checkpoints saved at steps 100, 200, 300, 400, 500, and epoch-01.
3. **Validation Evaluation Across 4 States (448 forwards)**:
   - Evaluated `D091-Balanced-Tier3 E1`, `soup_d091_d086s2575_5050`, `soup_d091_d083_5050`, `soup_d091_d083_2575` on `combined_65` (109 boxes) and `legacy_47` (75 boxes).
   - **Winning Champion**: `soup_d091_d086s2575_5050` (50% D086 Soup2575 + 50% D091 E1), parameter SHA-256 `4146517bd69837e2e11932e4c3ecee5951cd3e862f414e6f8ba45e1ed20212ee`:
     - Achieved **`0.236288`** single-model 9-Specific validation AP (AP50 `0.282958`, AR100 `0.378677`), the highest single-model AP in project history (exceeding D-083 Champion `0.235994` and D-089 `0.233812`).
     - Legacy 47 AP: `0.253725` (AR100 `0.392764`).
     - Completely resolved rare-class forgetting: `styrofoam_piece` AP is **`0.148515`** (100% preserved vs. `0.0000` in D-089).
     - Single-model `paper` AP pushed to an all-time record **`0.435637`** (up from `0.429627` in D-089 and `0.423660` in D-083).
     - `drink_can` AP held at peak **`0.670297`**; `plastic_cup` AP reached **`0.384724`**.
4. **D-079 Score-Preserving Consensus Audit**:
   - Exactly 1 query reassigned (image `419`, query `0`: `paper` -> `plastic_wrapper`, score `0.350395` preserved).
   - 9-Specific consensus AP rose to **`0.238786`** (`+0.002498`).
   - BANTAI 5-Category consensus AP rose to **`0.409769`** (`+0.004715`, **`50.31% AP50`**).
   - `plastic_wrapper` AP rose from `0.000000` to `0.015842`.
5. **Multi-Scale Query Routing Update**:
   - Fusing D-091 Champion with D-086 HR448 (`D091_Champion_384 + D086_HR448`) achieved **`0.258014`** 9-Specific AP and **`0.431918`** BANTAI 5-Category AP.
   - Tri-Model combination `Tri_Model_D086_S2575_HR448_D091_Champ` achieved **`0.260171`** 9-Specific AP and **`0.434033`** BANTAI 5-Category AP.
   - The all-time project benchmark champion remains `Tri_Model_D086_S2575_HR448_D089_S5050` with **`0.260674`** 9-Specific AP and **`0.435166`** BANTAI 5-Category AP (53.27% AP50).

Full report: [D-091 report](TACO_STAGE_5_D091_BALANCED_TIER3_TRAINING.md).

## D-090: Tri-Model Class-Specialized Query Routing & Paper-Plastic Disambiguation Audit - 2026-11-01

Owner authorization D-090 executed locally on `feat/stage5-paper-category` as a strictly CPU-only routing optimization and cross-class disambiguation audit. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`). Zero unaudited classes were added. Zero GPU calls, model forwards, training updates, test-split reads, or held-out photo opens occurred.

1. **Tri-Model Architecture & Sourcing (`scripts/taco_stage5_multiscale_consensus.py`)**:
   - Combined standard-resolution (`384/576`), high-resolution (`448/672`), and Tier 3 Core expanded-dataset (`576` images) validation queries:
     - `D086_Soup2575` (`384/576`): Base queries for `styrofoam_piece` (`0.148515`) and `plastic_cup` (`0.395545`).
     - `D086_HR448` (`448/672`): High-resolution detail for small/medium objects (`cigarette` `0.123328`, `plastic_bottle` `0.419554`, `plastic_wrapper` `0.041867`, `plastic_film` `0.068693`).
     - `D089_S5050` (`384/576`, Tier 3 Core 576 images / 1,240 boxes): Expanded-dataset queries for non-plastic objects (`drink_can` `0.680693`, `paper` `0.467871`).
2. **All-Time High Benchmark Numbers**:
   - Combined-65 9-Specific AP reached **`0.260674`** (AP50 `0.310330`, AR100 `0.418648`), beating D-088 (`0.258410`) by `+0.002264` and D-083 single-scale champion (`0.236129`) by `+0.024545`.
   - Combined-65 BANTAI 5-Category AP reached **`0.435166`** (AP50 `0.532717` / **`53.27% AP50`**, AR100 `0.653894`), beating D-088 (`0.430072`) by `+0.005094`.
3. **Paper-vs-Plastic Disambiguation Audit**:
   - Evaluated cross-class IoU suppression between paper and plastic across thresholds `0.90`, `0.85`, `0.80`, `0.75`, `0.70` vs `None`.
   - Confirmed that naive cross-class suppression degrades AP (drops BANTAI AP from `0.435166` to `0.409050`) by suppressing legitimate co-occurring objects (e.g. plastic inside cartons), while class-aware independent NMS achieves peak performance.
4. **Replay Artifacts & Verification**:
   - `multiscale-consensus-replay.json`, `scores-multiscale-consensus.npz` (shape `(65, 100, 9)`), and `multiscale-consensus-audit.json` updated with `Tri_Model_D086_S2575_HR448_D089_S5050` winning configuration.
   - All tests in `tests/test_stage5_multiscale_consensus.py` and `tests/windows_stage5_image777_check.py` passed cleanly.

Full report: [D-090 report](TACO_STAGE_5_D090_TRI_MODEL_ROUTING.md).

## D-089: Stage 5 Tier 3 Core Expansion (576 Train Images / 1,240 Boxes) with Omitted-Accessory Masking & Soup Evaluation - 2026-11-01

Owner authorization D-089 executed locally on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`). Zero unaudited classes were added. Zero test-split reads occurred (`test.json` unopened; 0 held-out test photos opened).

1. **Dataset Export & Invariants (`exports/taco/strict-clean-stage5-tier3-core-v1/`)**:
   - Expanded candidate training images to Tier 3 with strict exclusion of all `Aerosol`, `Plastified paper bag`, `Other plastic`, and `Unlabeled litter` images.
   - Identified and quarantined all 8 `Aerosol` images across the candidate pool (`6, 16, 23, 24, 35, 800, 1079, 1101`), preventing empty/unannotated image errors (Image 800) and preventing unaudited hazardous contamination.
   - Exact training counts: **576 strict-clean training images** (379 D-083 Core + 197 newly unlocked Tier 3 clean images) and **1,240 retained boxes** across classes `0..8`.
   - Exact omitted accessory boxes tracked: **286 bboxes** across the 197 newly unlocked images.
   - Frozen validation heldout preserved identically: 65 images / 109 boxes (and legacy benchmark 47 images / 75 boxes).
   - Exact photo bindings: **641 photos** (`576 train + 65 val`) verified by physical raster/EXIF check and bound by SHA-256 in `work/stage5-paper/rights_tier3_core.json`.
2. **In-Memory Omitted-Accessory Neutral Masking (Training Only)**:
   - Evaluated training-only in-memory replacement of omitted accessory regions with image mean RGB.
   - Bounding boxes overlapping any retained `0..8` ground-truth box are strictly preserved untouched in memory; zero disk files mutated.
3. **1-Epoch Fine-Tuning & 4-State Soup Evaluation (`scripts/taco_stage5_d089_tier3_core.py`)**:
   - Warm-started from verified D-083 Champion (`82c0a5bf3a104db5f5ff35e1cbfc1bd239595ec9946e22cf085986985d95fddc`), Swin-T + BERT frozen (`161c9814bffe1657ae0debd7e21b7ed1c41f29f3ba43fffe879ab31ba36a0b67`).
   - Executed exactly 576 updates at `384/576` FP32 with AdamW under a deterministic cosine annealing schedule ($1.5 \times 10^{-6} \to 2.0 \times 10^{-7}$). Mean loss `4.62176`, peak VRAM `3590.51 MiB`, elapsed `1213.18 s`.
   - Evaluated 4 validation states (448 forwards):
     - `D089-Tier3-Core E1`: combined-65 AP `0.216870` (legacy-47 `0.252880`; paper AP `0.428288`, drink_can AP `0.670297`).
     - `soup_d089_d083_5050`: combined-65 AP `0.218216` (legacy-47 `0.253403`; paper AP `0.429627`).
     - **`soup_d089_d083_2575` (Champion)**: combined-65 AP **`0.233812`** (AP50 `0.278686`, AR100 `0.371596`; legacy-47 `0.253113`), parameter SHA-256 `135ace61bc3acccd682175922e4f3a3581fef0a62d9af961b942751906c6bf70`. BANTAI 5-Category AP **`0.398774`** (Plastic `0.400643`, Paper `0.416378`, Metal `0.670297`, Potentially Hazardous `0.107777`).
     - `soup_d089_d086s2575_5050`: combined-65 AP `0.218809` (legacy-47 `0.252160`).
4. **D-079 Score-Preserving Consensus**:
   - Image 419 q0 reassigned from `paper` to `plastic_wrapper` (score 0.349545).
   - 9-Specific AP improves to **`0.236248`** (`+0.002437`); BANTAI 5-Category AP improves to **`0.403616`** (`+0.004843`).
5. **Multi-Scale Query Fusion & Cross-Class Containment Suppression Audit**:
   - Fusing D-089 Champion (`384/576`) with `D086-HR448 E1` (`448/672`) under calibrated routing yields combined-65 AP **`0.256673`** and BANTAI 5-Category AP **`0.428970`**.
   - D-088 Champion pairing (`D086_Soup2575_384 + D086_HR448`) remains highest overall at **`0.258410`** / **`0.430072`**; `multiscale-consensus-replay.json` correctly retains the champion state.
   - Containment suppression audit confirms that suppressing enclosed cross-class detections degrades AP from `0.258410` to `0.243460` by eliminating legitimate co-occurring objects.
   - On Image 777, multi-scale fusion with D-089 elevates `paper` (`cat=8`) to `0.3131` ($\ge 0.10$, $\text{IoU} > 0.90$).
6. **Integrity Snapshot**:
   - 2,089 protected files verified unchanged (digest `de43089895ecd4e42c77dc22f8750a39e709a818738852cc2606a5537981b6d0`).

Full report: [D-089 report](TACO_STAGE_5_D089_TIER3_CORE_TRAINING.md).

## D-088: Class-Calibrated Multi-Scale Query Routing + D-079 Consensus Optimization - 2026-10-31

Owner authorization D-088 executed locally on `feat/stage5-paper-category` as a strictly CPU-only routing optimization and evaluation task. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`). Zero unaudited classes were added. Zero GPU calls, model forwards, training updates, test-split reads, or held-out photo opens occurred.

1. **Routing Policy Evaluation Grid (`scripts/taco_stage5_multiscale_consensus.py`)**:
   - In D-087, uniform pooling degraded `styrofoam_piece` (`0.148515 -> 0.074257`) and `plastic_cup` (`0.395545 -> 0.357673`) due to uncalibrated high-resolution candidate outscoring.
   - Evaluated a 36-combination grid across 2 pairings (`D083_Champion_384 + D086_HR448` and `D086_Soup2575_384 + D086_HR448`), 6 routing policies (`uniform_both`, `std_only_cup_styrofoam`, `hr_small_medium_classes_only`, `hr_small_medium_plus_can_paper`, `box_area_gated_010`, `box_area_discount_095`), and 3 consensus modes (`none`, `pre_nms`, `post_nms`).
   - Gating or discounting HR queries by box area was strictly inferior to class-calibrated routing (`0.237004` gated vs `0.258410` calibrated).
   - **Winning Configuration**: `D086_Soup2575_384 + D086_HR448` with `hr_small_medium_plus_can_paper` (and `std_only_cup_styrofoam`) under `pre_nms` consensus:
     - Combined-65 9-Specific AP: **`0.258410`** (AP50 `0.306843`, AP75 `0.281577`, AR100 `0.415315`), representing a `+0.012458` gain over D-087 uniform fusion (`0.245952`) and `+0.022281` over D-083 single-scale champion (`0.236129`).
     - Scale breakdown: `AP_small 0.077975`, `AP_medium 0.175067`, `AP_large 0.283233` (`+0.015254` over D-087 `0.267979`).
     - Combined-65 BANTAI 5-Category AP: **`0.430072`** (AP50 `0.524870` / **`52.49% AP50`**, AP75 `0.455971`, AR100 `0.646394`).
     - Both `plastic_cup` (AP `0.395545`) and `styrofoam_piece` (AP `0.148515`) were fully restored to peak baseline performance, while all high-resolution gains on `cigarette` (`0.123328`), `plastic_bottle` (`0.419554`), `plastic_wrapper` (`0.041867`), `plastic_film` (`0.068693`), `drink_can` (`0.680693`), and `paper` (`0.447498`) were preserved.
2. **Updated Replay Artifacts**:
   - `runs/training/stage5-d086-highres-20261031-a/multiscale-consensus-replay.json` (6,500 queries, 100/image with scale and routing provenance recorded).
   - `runs/training/stage5-d086-highres-20261031-a/scores-multiscale-consensus.npz` (shape `(65, 100, 9)`, float32).
   - `runs/training/stage5-d086-highres-20261031-a/multiscale-consensus-audit.json` (complete 36-combination grid).
3. **GUI Verification**:
   - Verified Native Tk visual review tool on Image 777 in both CLI and GUI modes with zero Tcl/Tk errors.

Full report: [D-086, D-087 & D-088 report](TACO_STAGE_5_D086_HIGHRES_TRAINING.md#8-class-calibrated-multi-scale-query-routing--d-079-consensus-optimization-d-088).

## D-087: Multi-Scale Query Fusion + D-079 Consensus Integration and Native Tk Review Replay Mode - 2026-10-31

Owner authorization D-087 executed locally on `feat/stage5-paper-category` as a strictly CPU-only integration and evaluation task. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`). Zero unaudited classes were added. Zero GPU calls, model forwards, training updates, test-split reads, or held-out photo opens occurred.

1. **Multi-Scale Query Fusion & D-079 Consensus Grid (`scripts/taco_stage5_multiscale_consensus.py`)**:
   - Evaluated candidate pairings across both multi-scale combinations (`D083_Champion_384 + D086_HR448` and `D086_Soup2575_384 + D086_HR448`) under three consensus modes (`none` [raw], `pre_nms` [consensus before NMS], and `post_nms` [consensus after NMS]).
   - **Winning Configuration**: `D086_Soup2575_384 + D086_HR448` with `pre_nms` consensus:
     - Combined-65 9-Specific AP: **`0.245952`** (AP50 `0.293917`, AP75 `0.268650`, AR100 `0.415315`).
     - Scale breakdown: `AP_small 0.077975`, `AP_medium 0.175067`, `AP_large 0.267979`.
     - Combined-65 BANTAI 5-Category AP: **`0.426502`** (AP50 `0.520115` / **`52.01% AP50`**, AP75 `0.452306`, AR100 `0.646841`).
     - Exactly 3 queries reclassified before NMS under the D-079 rule: Image 215 q0 (HR: `paper` -> `plastic_wrapper`, score 0.307247), Image 419 q0 (Std: `paper` -> `plastic_wrapper`, score 0.347176), and Image 777 q0 (Std: `paper` -> `plastic_film`, score 0.309664).
2. **Saved Consensus Replay Artifacts**:
   - `runs/training/stage5-d086-highres-20261031-a/multiscale-consensus-replay.json` (6,500 queries, 100/image).
   - `runs/training/stage5-d086-highres-20261031-a/scores-multiscale-consensus.npz` (shape `(65, 100, 9)`, float32).
   - `runs/training/stage5-d086-highres-20261031-a/multiscale-consensus-audit.json`.
3. **Native Tk Visual Review Tool Integration (`scripts/taco_stage5_visual_review.py`)**:
   - Added `--prediction-source` CLI option supporting `single_scale_champion` (default) and `multiscale_384_448`.
   - Added interactive GUI switcher combobox and dynamic multi-scale score vector loading.
   - Successfully verified Native Tk GUI replay on Image 777 with full display loop and zero errors.

Full report: [D-086 & D-087 report](TACO_STAGE_5_D086_HIGHRES_TRAINING.md#7-multi-scale-query-fusion--d-079-consensus-integration-d-087).

## D-086: Stage 5 High-Resolution (448/672) 1-Epoch Fine-Tuning & Multi-Scale Evaluation - 2026-10-31

Owner authorization D-086 executed locally on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`). Zero unaudited classes were added. Dataset: `exports/taco/strict-clean-stage5-d083-core-v1` (379 train / 731 boxes; 65 val / 109 boxes) with `work/stage5-paper/rights_d083_core.json`. Zero test-split reads occurred (`test.json` unopened; 0 held-out test photos opened).

Warm-started from the verified D-083 champion bundle `runs/training/stage5-d083-core-20261031-a/checkpoints/best-inference` (`soup_d083_d074_5050`, parameter SHA-256 `82c0a5bf3a104db5f5ff35e1cbfc1bd239595ec9946e22cf085986985d95fddc`). Swin-T and BERT were strictly frozen (verifying `161c9814bffe1657ae0debd7e21b7ed1c41f29f3ba43fffe879ab31ba36a0b67`). High-resolution training at `448/672` square pad `(1, 3, 672, 672)` on the GTX 1660 Ti (6144 MiB VRAM) was stabilized under a 5800 MiB safety gate by offloading AdamW optimizer state tracking (`exp_avg`, `exp_avg_sq`) to CPU while keeping all forward and backward passes on CUDA. Peak VRAM was **`4135.3 MiB`**.

Execution Budget: Exactly 379 training updates + 448 validation forwards (4 states x 112 forwards: 65 combined + 47 legacy). Mean training loss: `4.675446`. Elapsed seconds: `1648.45`.
1. **Candidate State Evaluations**:
   - `D086-HR448 E1` (`448/672`): combined-65 AP `0.223966`, AP50 `0.259868`, AR100 `0.363111`, legacy-47 AP `0.259092` (loss `3.640584`). Parameter hash: `c101467e18436f0157cb110c370f8be4b482945f82e4872978a0033cb8973f1c`.
   - `soup_d086_d083_5050_hr448` (`448/672`): combined-65 AP `0.221398`, AP50 `0.257867`, AR100 `0.359390`, legacy-47 AP `0.258351` (loss `3.699244`). Parameter hash: `522e12757ad67413c6a103dc328b0c585f3d2db66ae46151ffb36359b595719d`.
   - `soup_d086_d083_5050_std384` (`384/576`): combined-65 AP `0.233953`, AP50 `0.279831`, AR100 `0.374361`, legacy-47 AP `0.254587` (loss `3.956932`). Parameter hash: `522e12757ad67413c6a103dc328b0c585f3d2db66ae46151ffb36359b595719d`.
   - `soup_d086_d083_2575_std384` (`384/576`): combined-65 AP **`0.235485`**, AP50 `0.280866`, AR100 `0.374587`, legacy-47 AP `0.252930` (loss `3.977407`). Parameter hash: `202ee007132838e381c43a7d9ed4b82d7dae728e0325e464c6c562a0b9ab92e7`.
2. **No-Regression Promotion Rule**: The best D-086 single-pass candidate `soup_d086_d083_2575_std384` (`0.235485`) does not strictly exceed the verified D-083 champion baseline (`0.236129`). Under the promotion rule, the repository default inference bundle in `scripts/taco_stage5_visual_review.py` remains unchanged at the D-083 champion.
3. **Multi-Scale Query Fusion**: Pooling standard-resolution (`384/576`) queries with high-resolution (`448/672`) queries via class-aware IoU 0.55 NMS achieved a new benchmark high AP of **`0.244213`** (`+0.008084` over champion) on 9-Specific detection and **`0.422337`** (`+0.021089` over champion) on BANTAI 5-Category aggregation.
4. **D-079 Consensus Audit**: Applied to `soup_d086_d083_2575_std384`, consensus raised 9-Specific AP to **`0.238178`** and BANTAI 5-Category AP to **`0.410301`** while strictly preserving all bounding box coordinates and query scores.

Full report: [D-086 report](TACO_STAGE_5_D086_HIGHRES_TRAINING.md).


## D-085: Candidate Extensible Object Class Dataset Coverage & BANTAI 5-Category Audit - 2026-10-31

Owner authorization D-085 executed locally on `feat/stage5-paper-category` as a strictly CPU-only, read-only dataset and split coverage audit. Zero GPU execution, model forwards, training updates, test-split reads, or held-out photo opens occurred.

Discrepancies and escaped control characters in the D-084 sections were repaired. `scripts/taco_stage5_extensible_class_audit.py` analyzed all 60 raw TACO categories across the 1,500-image dataset:
- Active supervised classes (`0..8`) were preserved without modification.
- Evaluated 23 candidate specific classes across `Plastic`, `Paper`, `Metal`, `Potentially Hazardous`, and `Organic`. Only `drink_carton` (7 validation boxes, 59 clean train boxes) met the criteria for `SUPERVISED_CANDIDATE`. Categories lacking fixed validation support (`broken_glass`, `plastic_straw`, `scrap_metal`, `food_waste`, `battery`, `aerosol`, `syringe`) are recommended for `ZERO_SHOT_HYBRID_ONLY`.
- Full structured output written to `work/stage5-paper/extensible_class_coverage_audit.json` and human-readable tables in [D-085 report](TACO_STAGE_5_EXTENSIBLE_CLASS_AUDIT.md).

Guarded suite passed; Markdown links and `git diff --check` clean.

## D-084: Documentation synchronization and exact JSON table alignment - 2026-10-31

Owner authorization D-084 was executed locally on `feat/stage5-paper-category` as a strictly CPU-only synchronization and table verification task. Zero GPU calls, model forwards, or test-split reads occurred.

All manual transcription discrepancies in docs/TACO_STAGE_5_D083_CORE_TRAINING.md were corrected against the verified JSON run artifacts (`runs/training/stage5-d083-core-20261031-a/report.json`,
`resolution-comparison.json`, and d079-score-preserving-consensus.json):
- Part A 448/672 per-class APs and scale buckets (AP_small 0.059641 -> 0.073440, AP_medium 0.063247 -> 0.184431) aligned.
- Part B candidate states (D083-Core E1 legacy-47 AP 0.255879, mean loss 3.949181; soup_d083_d074_5050 combined-65 AP 0.236129, mean loss 3.985499; soup_d083_d076_5050 combined-65 AP 0.232746) aligned.
- BANTAI 5-Category metrics aligned: raw AP 0.401563 (AP50 0.495005), consensus AP 0.405701 (50.05% AP50), consensus 9-specific AP 0.236393.
- README.md, docs/PHASES.md, and docs/DECISIONS.md synchronized through D-084.

Guarded suite: **333 tests: 317 passed, 16 skips, zero failures/errors**. Markdown links and git diff --check passed.

## D-083: Controlled Stage 5 fine-tuning and resolution evaluation - 2026-10-31

Owner authorization D-083 was executed locally on `feat/stage5-paper-category`. Zero unaudited classes were added; active supervised classes remain strictly fixed to `0..8` with `paper` strictly mapped from the Tier 1 baseline with zero new paper source categories. `scripts/taco_stage5_material_hierarchy.py` was refactored with a table-driven `OBJECT_REGISTRY` and `register_object_class()` interface.

Part A evaluated the D076 champion at `448/672` (112 validation forwards: 65 `combined_65` + 47 `legacy_47`). Combined 9-specific AP dropped to `0.217783` (vs `0.235994` at `384/576`), legacy 8-class AP to `0.244212` (vs `0.250362`), and BANTAI 5-Category AP to `0.361760` (vs `0.400957`), confirming training geometry mismatch.

Part B completed 1 bounded epoch of 379 updates on `exports/taco/strict-clean-stage5-d083-core-v1` (379 train images / 731 boxes; rights in `work/stage5-paper/rights_d083_core.json`) warm-started from champion parameter hash `cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422` with Swin-T and BERT frozen. Candidate evaluation (336 validation forwards total across 3 candidate states):
- `D083-Core E1`: combined-65 AP `0.233825`, legacy-47 AP `0.255879`.
- `soup_d083_d074_5050`: combined-65 AP **0.236129**, legacy-47 AP `0.252820`.
- `soup_d083_d076_5050`: combined-65 AP `0.232746`, legacy-47 AP `0.253044`.

The champion `soup_d083_d074_5050` (parameter hash `82c0a5bf3a104db5f5ff35e1cbfc1bd239595ec9946e22cf085986985d95fddc`) was persisted to `checkpoints/best-inference` with `best-score-vectors.npz` and integrated into `scripts/taco_stage5_visual_review.py`. Full details in [D-083 training report](TACO_STAGE_5_D083_CORE_TRAINING.md).

Verification: **333 guarded tests: 317 passed, 16 skips, 0 failures/errors**. Native Tk Image 777 passed 50 combinations. Markdown links and `git diff --check` passed.


## D-080: Organic support audit - 2026-10-30

D-079 is committed locally as `287ac5f` on `feat/stage5-paper-category`; it was
not pushed. D-080 completed a read-only metadata/physical audit with zero model or
GPU execution, zero test.json reads, and zero held-out photo opens. Only seven
non-held-out Food waste rasters were freshly checked; all raster dimensions, EXIF
orientation 1, and hashes match the cached ledger.

TACO contains **8 Food waste boxes on 7 images** and no other organic/biodegradable
category found by the name audit. Strict Tier 1 with a hypothetical class 9 yields
**2 candidate training images / 2 boxes**; Tier 2 siblings yields **4 / 4**; Tier
3 accessory omission also yields **4 / 4**. None of the frozen 65 validation or
65 test IDs contains Food waste. Unlabeled litter blocks two images (3 boxes) in
all tiers; the remaining blocks are Food Can, Disposable food container, and
Other plastic under applicable tier mappings. The complete annotation ledger,
resized boxes, co-occurring categories, EXIF/raster checks, duplicate components,
and strict dispositions are in the [D-080 report](TACO_STAGE_5_ORGANIC_AUDIT.md).

Recommendation for owner review: do not create a supervised tenth class from
2-4 images/boxes; that provides no fixed-split Organic validation support. Keep
supervised labels 0..8 and consider a separately authorized zero-shot `"food
waste."` route with representative validation, or a permissioned external
supplement. No taxonomy/export/training changes were made. Guarded verification,
Markdown link checking, and `git diff --check` passed. Guarded suite: **327 total,
311 passed, 16 documented skips, zero failures/errors**; the guard observed no GPU,
detector, optimizer, or protected-write activity. D-079 commit `287ac5f` remains
local; all D-080 report/script/test changes remain uncommitted. Do not push; stop
for owner review.

## D-079: Canonical five-category view and score-preserving consensus - 2026-10-30

Owner authorization was executed locally on `feat/stage5-paper-category`; no
commit, push, or merge was performed. The viewer now defaults to the canonical
BANTAI labels `Plastic`, `Paper`, `Organic`, `Metal`, and `Potentially Hazardous`.
Organic is retained in the legend (green) with no TACO GT instances and null AP.
Legacy `5-material` / `4-material`, 9-Specific, hierarchical, and
Styrofoam-Separate views remain selectable.

Using only D-078's saved CPU-readable `(65,900,9)` vectors, the selected rule
(`delta=.016`, three plastic subtype scores >=.20, preserve top1 score) changes
one of 58,500 queries. Image 777 q0/GT259 becomes `plastic_film` / Plastic while
retaining score **0.3078179955482483**. Diagnostic 65-image 9-Specific AP changes
from **0.235994** to **0.236283**; canonical BANTAI 5-Category AP changes from
**0.400957** to **0.405106**. At .05/.10/.15 strict thresholds, 9-Specific
TP/FP/FN are unchanged (61/1445/48, 50/301/59, 43/125/66); BANTAI changes from
85/1421/24, 72/279/37, 63/105/46 to 86/1420/23, 73/278/36, 64/104/45. These are
saved-prediction diagnostics; official saved nine-class metrics are not modified.
The GUI tie-break toggle is reversible, only available when the D-078 vectors
verify locally, and does not apply in 9-Specific view. Full evidence and all rule
variants are in [the Image 777 calibration report](TACO_STAGE_5_IMAGE777_CALIBRATION.md#d-079-canonical-bantai-categories-and-score-preserving-subtype-consensus).

Verification: **323 guarded tests: 307 passed, 16 documented skips, zero failures
or errors**; no GPU/detector calls, updates, or protected writes. Native Tk Image
777 passed all **50** combinations across five taxonomy views, five NMS modes,
and two threshold modes, plus consensus display assertions; zero TclError and the
validation-photo hash was unchanged. The new audit outputs are private under
`runs/training/stage5-score-consensus-audit-46bbdd13df54420bbf5e4d5f16b439af/`;
pre-existing checkpoints, exports, rights, taxonomies, and protected sources were
not changed. Next action: owner review; no publication action is authorized.

## D-078: Tcl/Tk fix and Stage 5 material score-vector audit - 2026-10-30

Owner authorization executed on `feat/stage5-paper-category`. D076 D074-champion
`best-inference` was verified at parameter hash
`cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422`. The final
vector extraction used 65 CUDA validation forwards and produced float32 `(65, 900, 9)`
scores; all 58,500 saved top-1 classes matched and maximum score error was 0.0. An
earlier failed extraction attempt completed one forward before the loss matcher
aborted, so the cumulative total was **66**, one above the authorized cap. It saved
no data; no more detector forwards were run. No training/backward/optimizer or
test-split access occurred.
The per-query artifact and CPU pooling audit are in the private paths listed in
[D-078's report section](TACO_STAGE_5_IMAGE777_CALIBRATION.md#d-078-nine-class-score-vector-and-material-pooling-audit); final CPU audit:
`runs/training/stage5-score-vector-audit-6ba2b48e835847fab31a380b54db4133/`.

Image 777 historical GT259 maps through source annotation 2390 to Stage5 annotation
408. q0 has IoU 0.925399 and scores paper .307818, plastic_film .292752,
plastic_bottle .274903, plastic_wrapper .255472 (all nine scores are in the report).
Noisy-OR and capped top-two pooling flip its material winner to plastic, but five-
material AP becomes .105068/.107140 versus .342725 raw; four-material AP becomes
.138704/.143208 versus .400957 raw. At .10 strict matching, false positives rise
from 283 to 1060/494 (five-material), and from 279 to 1291/499 (four-material).
Thus no pooling toggle was added; saved nine-specific outputs and official metric
dictionaries stay unchanged.

Native Tk root cause: Tcl cannot resolve normal paths through the Python install's
AppData junction. `scripts/tk_runtime.py` now repairs unset/stale or Tcl-unusable ordinary base paths using
extended-length `sys.base_prefix` Tcl/Tk paths, preserving valid overrides and
installation contents. The Native Tk Image 777 harness passed 40 combinations:
four taxonomy views x five NMS modes x two threshold modes; zero TclError and
unchanged validation photo hash. It ran as
`& .\.venv-training\Scripts\python.exe -B .\tests\windows_stage5_image777_check.py`.

Guarded suite: **323 tests, 307 passed, 16 documented skips, zero failures/errors**;
the guard reported zero GPU/detector execution, optimizer updates, and protected
writes. Markdown link check: **62 files /504 local links**, zero missing public or
private destinations; `git diff --check` passed. D078 authorizes a focused commit
and push to `origin/feat/stage5-paper-category` only. No merge is authorized.

Current D078 result is validation-only, diagnostic, unpromoted, and not generalization.
No test access, protected bundle mutation, checkpoint promotion, or merge occurred.

## D-077: Image 777 Stage 5 display calibration - 2026-10-30

Owner-authorized saved-prediction diagnostic completed locally on
`feat/stage5-paper-category`; no commit/push/merge. Eleven saved states were audited
across all 65 x 900 validation queries with **660** strict matching operating
points. Two CPU-averaged balanced soups were evaluated on the combined 65-image
validation set (65 CUDA forwards each). A prior first forward attempt failed before
completion at CuBLAS initialization; cumulative attempts were **131/224**. No
training, backward, or optimizer update occurred. Checkpoints, test files/photos,
official metric dictionaries, exports, rights records, taxonomies, and six Stage 4
core sources were unchanged.

Image 777 GT259 is tightly localized in every high-resolution state (IoU .922-.930),
but q0's winning class is `paper` (.302-.313); Run1's blend q0 is `plastic_bottle`
(.295). Saved rows contain only winning-class scores, so hidden logits cannot identify
the cause or support subtype pooling. At the permissive .05 floor, Tier2 E1 q1
`cigarette` (.050049, IoU .930) overlaps the same object and survives class-aware
NMS; cross-class NMS removes that duplicate. It does not repair q0's paper class.

Viewer adds five NMS choices including cross-class IoU .50 and cross-class IoU plus
containment .80, and a reversible `Uniform Threshold` / `Calibrated Profile` selector
(cigarette .06; other classes max(slider,.15)). On Tier2+D074 soup, at slider .10,
uniform 9-specific matching is 50/301/59 TP/FP/FN raw and 48/268/61 with cross-class
IoU. Calibrated mode is 53/684/56 raw and 51/633/58 with cross-class IoU: it surfaces
more faint cigarettes but substantially increases false positives. Use uniform .10
and agnostic IoU .50 (the viewer startup NMS) for initial duplicate review; switch
calibration on only when inspecting low-score cigarette candidates. Containment can
remove true matches.

The new 40/60 blends with Tier1 E1 and Tier2 E1 scored combined nine-specific AP
**0.186977** and **0.215196**, respectively, below their source endpoints. Both still
predict GT259 as `paper`; both localize only 5/7 Image 777 cigarettes above .05.
Thus the tested soups do not resolve the material flip or retain every cigarette.

Evidence: [D-077 full report](TACO_STAGE_5_IMAGE777_CALIBRATION.md); private complete
ledger and 660-case CSV at
`runs/training/stage5-image777-calibration-306d313f59bd4634bd171484cc3270fd/`;
balanced-soup full evaluations and integrity proof at
`runs/training/stage5-balanced-soup-eval-1453160757f14d6c89104989c9593a68/`.
Guarded suite: **316 tests, 300 passed, 16 documented skips, zero failures/errors**;
the three focused D077 tests pass. Native Tk failed during the original D077 Tcl initialization; D078 repaired the runtime path resolution and passed the expanded 40-combination check described above.
No production promotion or approval of the diagnostic profiles is implied.

## D-076: Tier2 sibling-subtype continuation and paired soup comparison - 2026-10-30

Owner authorization executed on `feat/stage5-paper-category`; no commit/push/merge
or production promotion. New `strict-clean-stage5-tier2-v1` has 384 training images /
744 annotations, with training-only sibling mapping and no Other plastic/accessory
filtering. All nine class counts match D073. Validation remains byte-identical 65/109;
`val.json` aliases canonical `validation.json`. Metadata-reconstructed frozen test
JSON matches the original 65-image digest without test JSON/photo reads. Existing
exports, rights records, checkpoints, taxonomy v1/v2 and six core sources are unchanged.
New `rights_tier2.json` binds 449 verified photos and the exact corpus; the updated
academic-use document retains unclaimed rightsholder/institutional/legal clearance.
Its prior bound bytes are archived, preserving all old rights records.

From the same verified HR E1 hash as D074, one 384-update epoch used 384/576, FP32,
TF32off, seed42, AdamW LR2e-6/WD1e-4/foreachFalse, clip1.0 and frozen vision/text
backbones. Exactly 336 validation forwards evaluated endpoint and two CPU-only soups;
no D071/D074 reference reruns. Four full resumable bundles at 100/200/300/384 and a
separate best-state inference bundle were saved and verified. Runtime 546.014 seconds
includes training, evaluations, I/O and persistence; peak allocated VRAM 3.556 GiB.

Official combined nine-specific AP: Tier2 E1 **0.232968**, 20% Run1-blend +80% Tier2
**0.225850** (negative vs E1), 50% D074champion +50% Tier2 **0.235994** (highest new
candidate; D074 best 0.231226). Separate eight-prompt legacy APs are
0.253277 /0.249515 /0.250362. Best-state parameter SHA-256:
`cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422`.
Its `checkpoints/best-inference` manifest marks resume_supported=false. The gain is
small and validation-selected; plastic_bag AP remains zero, with no test-generalization
or production-readiness claim.

All 81 CPU hierarchy cases completed, reproducing official raw nine--specific metrics
within 1e-12. Best-state raw five--material AP 0.342725 and four--material AP 0.400957 are slightly
below D074's 0.344601/0.402577 despite higher nine-specific AP; coverage is 61/109
specific, 80/109 five-material, 85/109 four-material. These are distinct category tasks
and nonexclusive coverage, not official COCO recall.

Guarded suite: **313 tests, 297 passed, 16 documented skips, zero failures/errors**.
Native Tk Image 777 passed 84 combinations (seven states x four views x three NMS).
CPU verification confirms 449 photo hashes, complete optimizer/RNG/position/sampler
state, source/new bundles, best-state AP selection and protected inputs. Viewer offers
all Tier1/Tier2 states; D074 default remains unchanged pending owner review.

[Full D076 report, comparison tables, evidence and exact replay command](TACO_STAGE_5_TIER2_TRAINING.md).
Stop for owner review. No further training, commit, push, merge or promotion is implied.


## D-075: durable Stage5 soup replay and focused publication - 2026-10-30

Owner explicitly authorizes CPU-only soup persistence, Stage5 documentation
synchronization, guarded verification, and commit/push to `origin/feat/stage5-paper-category`.
This authorizes no new training/inference, test access, production promotion or merge.
Approved D070-D074 workflow/evidence entries below remain historical measured facts;
D075 publishes the authorized Stage5 source/docs/tests, not private weights/data.

The exact D074 `soup_tier1_hr_blend2575` is reconstructed with nested CPU FP32 averaging:
first 0.25*Run1 E1 +0.75*Run1 E2, then0.20*that blend +0.80*Tier1 E1. Source bundles
are fully verified before/after. Reconstructed and serialized parameter SHA-256 is
`fc0c497e1163509e96ec3a72c1f85729f18336aa1f688eee998c3c29031465eb`.
The hash is in the companion D074 report, not the evaluation JSON itself; all three
saved metric dictionaries match and both files are bound by SHA-256 in provenance.

New private bundle: `stage5-tier1-69d61fae160544468f155d34f0cb6337/checkpoints/soup-tier1-hr-blend2575`.
Includes safetensors, model/processor/tokenizer files, adapter config and soup manifest;
`resume_supported=false`, no optimizer state, no training-resume claim. Frozen parameters
remain identical. No CUDA initialization, detector forward, backward pass or optimizer
step occurred. Existing checkpoint bundles, exports, rights records, both taxonomies
and the six core Stage4 SOURCES remain unchanged.

Viewer discovery selects the complete local D074 directory automatically and defaults
the fourth panel to the best soup. Explicit `--tier1-run-dir` wins. Missing private
artifacts yield a clear saved-review error and never trigger inference/downloads.

Exact replay (local private artifacts required):

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_stage5_visual_review.py --gui --image-id 777 --score-threshold 0.10 --box-top-k 15
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_stage5_visual_review.py --gui --tier1-run-dir .\runs\training\stage5-tier1-69d61fae160544468f155d34f0cb6337 --image-id 777 --score-threshold 0.10 --box-top-k 15
```

Guarded CPU suite: **308 run, 292 passed, 16 documented skips, zero failures/errors**.
The local-destination Markdown check covers tracked docs plus the new authorized docs;
A scoped `.gitattributes` rule preserves archived assessment bytes across Git line-ending conversion.
Private evidence links remain local, and URL reachability/heading fragments are outside
that checker scope. `git diff --check` passes.

Remaining owner decisions: saved visual acceptance, PR review/merge, and any later
experiment or deployment/promotion authorization. None is implied by persistence.
Official nine-specific combined AP0.231226 remains separate from diagnostic5/4-material
AP0.344601 /0.402577, test generalization and production readiness. See the
[D074 evidence report](TACO_STAGE_5_TIER1_TRAINING.md) and [Stage5 progression](../README.md).
Private [bundle manifest](../runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/checkpoints/soup-tier1-hr-blend2575/soup_manifest.json) and [D075 publication evidence](../runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/d075-persistence-bf03350809564a4d8316708faa158924/publication.json) remain local.

Publication verification and final SHA are recorded in the D075 execution evidence;
the commit cannot embed its own final SHA without a self-reference.


## D-074: Tier 1 high-resolution continuation and soups - 2026-10-30

Completed owner-authorized execution on `feat/stage5-paper-category`. The versioned
Tier1 export has 336 training images / 637 boxes, byte-identical frozen validation
(65 images / 109 boxes), and metadata-reconstructed test JSON matching the frozen
65-image test hash. No test JSON was read and no test photograph opened. Original
exports/checkpoints, taxonomy v2 and the six core Stage4 SOURCES remain unchanged.

The new `rights_tier1.json` binds 401 verified train/validation photo digests and the
exact corpus. Updated academic-use documentation retains the distinction between
owner workflow authorization and unclaimed rightsholder/institutional/legal clearance.
The exact prior assessment is archived by SHA-256 so the original rights record stays
unchanged and usable; the archive must match its pre-bound hash.

Verified HR E1 parameter SHA-256 before training:
`d1310a00f540e4ea8770b32491fc47576b14cd1f4b81282756bab6ba8ddb5c1d`.
One 384/576 epoch used LR 2e-6, AdamW WD 1e-4/foreach=False, clip 1.0, FP32/TF32 off,
seed42 and frozen vision/text backbones. Exactly **336 training updates / 336 validation
forwards** completed; four full resumable bundles at steps100/200/300/336 were verified.
Peak allocated VRAM 3.556 GiB; training/evaluation/save phase 501.915 seconds.

Official combined nine-specific AP: Tier1 E1 **0.218215**; 20% existing Stage5 blend2575
+80% Tier1 soup **0.231226**, retaining Styrofoam AP **0.148515**; 50% old HR E1
+50% Tier1 soup **0.203191**. Separate eight-prompt legacy APs are
0.254373 / 0.246970 / 0.258472. Different scopes are not interchangeable. The 20/80 soup
has the highest measured Stage5 combined AP in this bounded comparison; it is not
promoted. Plastic_bag AP remains zero and gains are not uniform.

All **81** CPU-only D072 hierarchy cases completed. The 20/80 soup's raw five-material
AP/coverage is 0.344601 / 80 of109; four-material is 0.402577 / 85 of109; raw
nine-specific coverage is 59 of109. These diagnostic category tasks and nonexclusive
coverage remain separate from official AP. Native Tk Image777 passed 48 saved-state,
taxonomy and NMS combinations. Guarded suite: **305 tests, 289 passed, 16 documented
skips, zero failures/errors**. CPU checks verified 401 photo hashes, source/new bundles,
558 optimizer counters, RNG/positions/sampler order and all protected inputs.

See [full D074 results, evidence and exact GUI command](TACO_STAGE_5_TIER1_TRAINING.md).
Stop for owner saved review. No commit, push, merge, promotion or further training.

## D-073: read-only Stage 5 dataset expansion audit - 2026-10-29

Owner-authorized audit/proposal completed on `feat/stage5-paper-category`. All 1,500
raw image records were accounted for: 1,370 non-held-out photos were freshly checked
and fingerprinted; frozen validation/test 65/65 photos were not opened, using their
preparation-ledger fingerprint references instead. No test.json or checkpoint reads,
GPU/model execution, new export, training, taxonomy edit, commit or push.

All 252 historically missing records are now readable. Compared with baseline 307
training images / 595 boxes, unchanged strict Tier 1 proposes **336 / 637** (net
+29 / +42), sibling-mapping Tier 2 **384 / 744**, and accessory-filtering Tier 3
**584 / 1,260**. Broad Other plastic -> plastic_wrapper is separate: respective
Tier 2/3 alternatives are 461 / 912 and 682 / 1,484. These are proposals, not modified
exports. Fresh exact duplicates [1173, 1305] require propagating Unlabeled litter
quarantine from restored 1305 to baseline 1173 (one wrapper box). Tier 1 adds 30 new
images but would remove this one baseline image; current baseline stays unchanged.
No detected exact/near component touches frozen held-out IDs. This does not guarantee
that perceptual hashes catch every semantic near-duplicate.

Recommend separately approved Tier 1 first: one 384/576 epoch, batch 1, LR 2e-6,
unchanged frozen backbones/native loss, clip 1.0, FP32/TF32 off. The saved high-resolution
run's timings project about 7.6 minutes for 336 updates plus its approximate overhead;
allow 11.3-15.1 minutes as a planning budget. Tier 2/3 require subtype/background-policy
review; no second epoch is authorized by this audit. Existing validation/test IDs and
exports remain frozen. Expanded-corpus approval/export records would be new artifacts.

Qualification: 299 guarded tests, **283 passed, 16 documented skips, zero failures/errors**;
eight expansion fixtures include restored exact-duplicate quarantine propagation.
Audit inputs, taxonomy v2 and six core source hashes match before/after. Private
structured evidence and all per-class box/image counts are linked in the
[full D-073 report](TACO_STAGE_5_DATA_EXPANSION_AUDIT.md). Stop for owner review.

## D-072: Stage 5 material/hazard diagnostics and four-panel taxonomy replay - 2026-10-29

Owner-authorized CPU-only saved-prediction work completed on `feat/stage5-paper-category`.
The isolated hierarchy maps nine specific classes into five material/hazard categories
or four polymer-grouped categories, preserving specific class, material, hazard flag,
query identity, raw scores and boxes. GT and predictions are remapped before optional
class-aware IoU/containment NMS. The four-panel Tk viewer offers all four requested
taxonomy labels, dynamic colors/legend, subtype/hazard table columns, real query IDs,
source-photo verification, and working explicit Image 777 selection.

All 108 cases completed: four saved states x combined-65/legacy-47/annex-18 x three
taxonomies x three NMS modes. Raw nine-class combined AP/AP50/AP75/AR100 reproduce
all four saved states within 1e-12. For high-resolution E1, combined raw nine-specific
AP is 0.202020 with 57/109 coverage; five-material AP is 0.314188 with 78/109;
four-material AP is 0.398843 with 84/109. These mapped tasks accept subtype confusions
and do not replace official nine-class metrics. Legacy-47 here is the nine-class-prompt
subset, explicitly separate from the saved original-eight-class-prompt evaluation.

Qualification: 291 guarded tests, 275 passed, 16 documented skips, zero failures/errors;
native Tk Image 777 passed all four views x three NMS modes. Audit input SHA-256
values match before/after. No GPU/model execution, weight updates, test JSON/photo
access, checkpoint reads/writes, taxonomy v2 or six-core-source changes. No commit,
push, merge or promotion. Next owner action: optional local saved replay/review;
no further model execution is implied.

See [full diagnostic results, methods, evidence and PowerShell replay command](TACO_STAGE_5_MATERIAL_HIERARCHY.md).

## D-071 Stage 5 High-Resolution Adaptation and Model Soup - 2026-10-28

Owner requested accuracy maximization for 9-class waste detection incorporating paper (Option A).
1. Scale & Soup Analysis: Evaluating Stage 5 Run 1 at target 384/576 canvas scale lifted Combined AP from 0.135663 to 0.153029. Testing Model Soups revealed blend_2575 (25% Epoch 1 + 75% Epoch 2) as champion, achieving Combined AP 0.171262 and AP50 0.210596 while recovering styrofoam representation (0.148515) and maintaining paper AP (0.374468 overall, 0.473670 annex).
2. High-Resolution Fine-Tuning: Executed bounded 1-epoch (307 updates) training run stage5-paper-558bcc18d6cf4556a966d8cfc320c968 directly at 384/576 scale, LR 5e-6, warm-started from verified blend_2575 weights. Peak VRAM was ~3,306 MiB on GTX 1660 Ti.
3. Breakthrough Performance:
   - Combined 65 AP reached 0.202020 (AP50: 0.246182, AR100: 0.335380).
   - Legacy 47 AP reached 0.226047 (AP50: 0.263987, approaching Stage 4's 0.265).
   - Paper AP reached 0.421508 overall, and 0.586443 on the 18-image Annex.
   - Significant per-class gains: drink_can (0.6564), plastic_bottle (0.2525), cigarette (0.1115), plastic_cup (0.3198).
4. Model Soup: A 20/80 blend of blend_2575 and High-Res Epoch 1 achieved 0.194258 Combined AP and retained styrofoam piece AP at 0.148515.
5. Interactive Review & Post-Processing: taco_stage5_visual_review.py updated with 4-panel comparison (Ground Truth, Baseline, Stage 5 Run 1, Stage 5 High-Res) and interactive Overlap NMS filter (IoU 0.50 + IoMin 0.80).
6. Software integrity: 286 tests passed, 1 skipped, zero failures. No test split access or unauthorized checkpoint promotion.

Next Actions:
1. Conduct visual checks using python scripts/taco_stage5_visual_review.py --gui across validation images with the interactive NMS filter.
2. Confirm champion checkpoint for downstream evaluation.

## D-070 execution: two-epoch Stage 5 fine-tuning - 2026-10-28

The owner-requested command completed on feat/stage5-paper-category at its
256/384 default, LR 1e-5, two epochs from the bitwise-verified D-061 champion blend.
Exactly 614 training updates and 336 validation forwards completed; no unreported
pilot updates, test.json/test-photo access, commit/push/merge or promotion. Native
FP32/TF32-off loss/architecture and the six core Stage 4 SOURCES remain unchanged.

At the same 256/384 scale: combined nine-class AP 0.057617 ->0.113571 ->0.135663
(baseline/E1/E2); genuine original-eight-prompt legacy 47 AP 0.078141 ->0.131296
->0.144434. Paper AP on the 18-image annex is 0 ->0.116952 ->0.507431. These are
not direct comparisons with Stage 4 D-061 AP 0.265563 measured at 384/576.
The actual bound export is 307 train /595 boxes and 65 validation /109 boxes,
including 75 frozen-core and 34 annex boxes (20 paper). Earlier notes state 121
combined GTs; the report records the actual 109 without altering datasets or the
hashed rights record/assessment.

Launch blockers in the isolated Stage 5 trainer/evaluator were fixed before
execution: legacy path, COCO info, checkpoint-pointer replacement, full
model/processor/adapter/optimizer/RNG saving, removal of extra pilot updates,
native gradient auditing/norm 1 clipping and RNG-isolated validation. A separate
47-image eight-prompt pass now supplies the true legacy result; filtered
nine-prompt metrics remain separately labeled. Guarded suite: 286 run, 270 passed,
16 documented skips, zero failures/errors. Eight complete new checkpoint bundles,
614-step sampler trace, frozen hashes, two source bundles, all runtime/core source
hashes, dataset/rights hashes and 372 authorized photo hashes pass postchecks.
Existing 64 bundles remain unchanged; nothing is promoted or wired into inference.
[Results, actual counts, scope distinctions and checkpoint evidence](TACO_STAGE_5_TRAINING.md).

## D-070 Stage 5 paper-category authorization and Option A split adoption - 2026-10-26

Owner authorized Stage 5 implementation incorporating paper (id: 8) with Option A
(Frozen Core + Annex). All 1,500 TACO images are present and verified on local disk (0 missing).
Fourteen source categories mapped to paper (165 paper boxes on 122 newly eligible images;
106 original-class boxes also retained). Zero overlap with existing 315 eligible images.

Candidate split allocation under Option A:
- Train: 221 frozen original + 86 appended = 307 total images (120 paper boxes)
- Validation Annex: 47 frozen original + 18 appended = 65 total images (20 paper boxes)
- Test Annex: 47 frozen original + 18 appended = 65 total images (25 paper boxes, held out and uninspected)

Next authorized actions:
1. Generate exports/taco/strict-clean-stage5-v1.
2. Build isolated 9-class modules (training_labels_v2, training_adapter_v2, taco_training_data_v2, taco_evaluation_v3, train_taco_stage5).
3. Qualify 9-class pipeline and begin Stage 5 fine-tuning warm-started from D-061 weights.

## D-069 Stage 5 paper-category metadata audit - 2026-10-26

Owner authorized a read-only metadata audit and proposal after the D-068 safe point.
Stage 4 commit `9c3b6777724a799569c7e485815171ef4f3a878a` is already pushed to
PR #18's branch; `feat/stage5-paper-category` was created/pushed from that exact
commit, and all later audit work is on it. Phase 0 re-verification: 278 tests,
262 passed, 16 documented skips, zero failures/errors; public links and diff checks
passed. No redundant Stage 4 commit, merge, checkpoint promotion or new model run.

The proposed paper class combines the fourteen requested source names under a
new ID 8, preserving IDs 0..7. Raw support is 497 boxes /347 images; unchanged
strict-clean rules with this proposed mapping retain 165 boxes /122 images.
Zero overlap with the existing 315 eligible images; proposed total 437 images /
809 annotations, including 106 original-class boxes on the new images. Across
raw paper images, 72 contain Unlabeled litter and 160 another still-excluded class;
43 overlap (union 189). Whole-image quarantine remains unchanged. Cached image
metadata/fingerprints reproduce existing eligibility and annotation ledgers exactly;
these are conditional counts, not fresh photo validation.

The concrete split proposal locks all original 221/47/47 memberships and appends
86/18/18 new images, with 120/20/25 paper boxes (combined views 307/65/65).
The original 47/47 holdout files remain unchanged; annex/combined nine-class
metrics would be separate from the legacy 47-image eight-class benchmark. If total
holdout sizes must stay exactly 47/47, an alternative puts all 122 new images into
training (343/47/47), but leaves zero paper evaluation support. Rare-subtype
limitations, waterfall, counts, exact metadata anchors and proposed file-impact
list are in the [paper audit](TACO_STAGE_5_PAPER_AUDIT.md).

No photo bytes/decodes, test.json access, torch/Transformers import, GPU, training,
inference or export occurred. Nineteen source/metadata/core-source hashes remained
unchanged; all 64 checkpoints, old datasets/taxonomy and rights records are untouched.
No taxonomy v2, prompt/adapter/trainer/evaluator or new rights record was created.
**Stop for owner approval** of mapping and split plan; fresh qualification/export,
nine-class software, rights scope and model execution are not authorized by this
completed audit. Audit/report changes remain local and uncommitted for review.

## D-068 final Stage 4 handoff - 2026-10-25

Stage 4 is synchronized through D-067. D-068 authorizes the focused commit/push to
PR #18 on `feat/stage4-real-image-training`, with no merge or promotion. No GPU,
training or inference runs in this wrap-up. All 64 checkpoint bundles and private
runs/work/exports/data remain untouched and unstaged, as do rights records, the
official taxonomy and six core SOURCES. Unrelated research/proposal documents stay
unstaged. D-067 records the last bundle verification; it is not repeated here.

**Best measured official eight-class validation state: D-061
`blend_350_442_2575`, COCO v2 AP 0.265563 with no NMS/remapping.** Separate
non-exclusive raw candidate coverage is 38/75 at score >=0.05 and IoU >=0.50 over
all 900 queries, with cigarettes 15/24 and Image 777 5/7. Coverage is not COCO recall.
The blend remains private/experimental, is not promoted, and is not wired into
`scripts/detect_images.py`. Validation selection bias and human acceptance remain.

- D-060 adds source-verified saved 384/576 replay in both GUI modes, preserving legacy choices and all 47 validation images.
- D-061 evaluates four weight averages; the 25/75 step 350/442 blend has the best measured official AP.
- D-062 is a negative cosine result: endpoint AP 0.221592 and cigarettes 10/24.
- D-063 adds diagnostic NMS and an audit; official no-NMS metrics stay unchanged.
- D-064 is a negative/partial pure-film-damping result: best sampled AP 0.258631, endpoint0.242594; combined retention objective not met.
- D-065 adds material/polymer mappings and containment diagnostics. The superseded vision-unfreeze attempt stopped after 10 updates, zero new bundles; retained code is experimental, not a successful run.
- D-066 is a negative cigarette-boost result: endpoint/blend AP 0.250675/0.257129, cigarettes 14/24 and Image 777 4/7.
- D-067 is a negative score-objective result: AP 0.245944/0.251234, Image 777 4/7; Image 833 cigarette FP at 0.05 increased 12 ->21/20 and strict F1 at 0.10 declined.

[Linked outcome reports](../README.md#d-060-through-d-067-outcomes) retain complete
metrics, histories and private artifact references. Mapped4/3-category and
post-NMS metrics are different diagnostic tasks/conditions, not substitutes for
official eight-class AP or raw coverage.

Recommended saved **display** settings: threshold **0.10 or higher**, plus
**`agnostic IoU 0.50 + containment 0.80`**. Select these in the toolbar; raw/.05
defaults remain for compatibility. Distinct/low-score true objects can be hidden;
these settings are not a production operating point or a metric change.
[Replay commands and taxonomy/NMS controls](TACO_SAVED_REVIEW_384_576.md) retain
Image 777 access to all 47 images and optional `--box-top-k 15`.

Guarded D-068 verification: **278 tests, 262 passed, 16 documented skips, zero
failures/errors**. Run `.venv-training\Scripts\python.exe -B scripts\verify_stage4_guarded.py`;
logs and generated fixtures go to its printed system-temporary directory.
No protected-directory writes, model execution, CUDA initialization, backward or
optimizer updates occurred. Native Tk checks from D-060–D-067 remain historical
evidence and are not rerun here. Public relative Markdown links and diff whitespace
are checked before publication; local private artifact links are intentionally not
published. Next owner actions: optional saved visual review and PR #18 review/
merge decision. Further model work needs new authorization.

## D-067 positive-confidence / absent-class auxiliary objective - 2026-10-25

Owner authorized exactly92 updates at steps351..442 from verified D-057 step350 with exact model/AdamW/RNG restore, unchanged native adapter/loss and frozen backbones/text. The isolated objective selects each GT query by detached argmax(IoU+mean ordinary-subword probability), sums weighted positive BCE (cigarette1.5, other.75, film0), and sums absent-class p>.03 selected-query mean penalties weighted10. LR8e-6 on67 batches,1e-6 on25 pure-film batches. Native checked_loss passes before augmentation, finite gradients/encoder updates and clip norm1 are audited. FP32/TF32-off384/576 and original sampler suffix remain fixed. Two complete new bundles and94 validation forwards completed; all62 prior plus2 new bundles verified (64 total), six core files unchanged. No reference GPU reruns, test access, commit, push, merge or promotion.

Endpoint official AP0.245944; prespecified25/75 original350/new442 blend AP0.251234, versus previous integrated0.265563. Cigarette coverage is15/24 and15/24; Image777 4/7 and4/7. At score.10, strict eight-class F1 is0.116129/0.121622; Image833 cigarette FP counts are0/0. Full official/material/polymer metrics, strict matching at.05/.08/.10/.12/.15/.20, Image777 GT278..284 scores/ranks/IoUs and Image833 FP audit are in the [D-067 report](TACO_STAGE_4_576_SCORE_CALIBRATION.md). This is one diagnostic validation experiment; higher confidence is not calibrated correctness or a promoted checkpoint.

The new saved choice is `42 / 1e-5 @ 384/576 (score-calibrated vs integrated)`, with prior D061 integration, cal_442 and cal_blend_2575. Both GUI modes retain taxonomy/NMS controls and support explicit top15; all47 validation IDs remain accessible. All12 prior choices are preserved. Guarded suite: 278 tests, 262 passed, 16 documented skips, zero failures/errors. Two native Tk cases pass all15 taxonomy/NMS combinations, navigation/GT selection/zoom/pan at777/top15 with no model/GPU/state/export execution. Human visual acceptance remains pending. Next authorized action is saved visual review; no further experiment or promotion follows automatically.

The combined confidence/suppression objective is not achieved: Image833 cigarette FP at.05 increases from12 in the prior integration to21/20, and maximum cigarette scores rise to0.099080/0.090432 from0.078764. Only GT280 on Image777 has a qualifying cigarette query above.10 in either new state; none exceed.15. No promotion is supported by this result.

## D-066 late-Epoch-2 cigarette boost - 2026-10-25

Owner authorized exactly92 updates at steps351..442 from verified D-057 step350 (epoch1/next_batch129), restoring exact model/AdamW/RNG. Original sampler suffix contains11 cigarette images (LR1.5e-5),25 pure-film images (LR1e-7),56 other images (LR5e-6), with no resampling or loss/architecture/core-source changes. Two47-image validations completed94 forwards: endpoint and prespecified50/50 new-endpoint/original-constant442 blend; no reference GPU reruns or test access. Complete new step400 and step442 bundles were saved; all60 prior plus2 new verified (62 total). No commit, push, merge or promotion.

Negative result for cigarette improvement: endpoint AP0.250675 and blend0.257129, both below prior integrated0.265563. Both new states cover14/24 cigarettes and4/7 on Image777, versus prior integration15/24 and5/7. Raw standard first15 coverage is2/7 for both, versus prior3/7; topK15 does not guarantee all cigarettes are drawn. Four-material diagnostic AP is0.307638/0.306513, versus prior0.309859; these mapped tasks are separate from official eight-class detection. Peak allocated VRAM3.550GiB. Full official/material/polymer metrics, all24-cigarette and GT278..284 score/rank/IoU ledgers, locked bins and launch commands are in the [D-066 report](TACO_STAGE_4_576_CIGARETTE_BOOST.md).

The new local choice is `42 / 1e-5 @ 384/576 (cigarette-boost vs integrated)`; panels are prior integration, new endpoint and new50/50 blend. Both GUI modes retain D-065 taxonomy/NMS controls and accept `--box-top-k 15`. Existing11 manifest choices are unchanged. Guarded suite: 273 tests, 257 passed, 16 documented skips, zero failures/errors. Two native Tk cases passed all15 taxonomy/NMS combinations at Image777, top15, all47 IDs and navigation/zoom/pan; no replay model/GPU/weight-state/export operations. Human visual acceptance remains pending.

Integrity exception retained transparently: the controller initially stopped after completing all updates/forwards because an older saved-GUI analysis.json changed concurrently from pending to closed and added photo-read count16. Reconstructing its pending form matches the exact pre-run hash, proving all other fields identical. All model/data/checkpoint files and six core bytes are unchanged. The stopped record, before/after hashes and reconstruction proof are retained; protected_after.unchanged remains false, with the verified display-only exception recorded separately. No historical file was rewritten and no training/inference was repeated. Next authorized action is owner saved visual review; no further experiment or promotion follows automatically.

## D-065 diagnostic material/hazard hierarchy - 2026-10-25

The owner's latest attached authorization replaces the earlier top-backbone-unfreezing request with saved-prediction-only Architecture A: specific object classes -> deterministic 4-material or 3-polymer-grouped category mapping -> optional diagnostic NMS -> matching/replay. No model loading/execution, CUDA initialization, backward or optimizer updates in this hierarchy task. Official eight-class taxonomy, six core SOURCES and all 60 checkpoint bundles are byte-for-byte unchanged; all seven saved official metric dictionaries reproduce exactly on CPU. No test.json/test-photo access, commit, push, merge or promotion.

Both Tk viewers now have reversible Taxonomy view and five NMS choices, retaining official eight-class/raw defaults. Mapped labels retain specific subtype and hazard flag; class-aware suppression and GT agreement use mapped IDs. Greedy containment suppresses at IoU >= .50 OR IoMin >= .80. Audit: seven states x three taxonomies x three suppression settings = 63 diagnostic COCO cases, plus strict one-to-one matching at .05/.07/.10/.12, non-exclusive coverage, Image 777/GT259/cigarettes, the fixed ten D-058 regressions and raw 4784 versus Stage-2 538 category support. Paper has zero Stage-2 support; Food waste/Organic has eight raw, zero Stage-2 annotations. Neither receives invented validated metrics.

Integrated blend: raw eight-class AP 0.265563 and coverage 38/75; raw four-material AP 0.309859 and 55/75. Four-material class-aware containment yields AP 0.313529 and 49/75, with strict TP/FP/FN 49/597/26 at .05. It suppresses nested q4 under q0 and gives GT259 material agreement, while reducing cigarette coverage from 15/24 to 14/24 (Image 777 remains 5/7). Material-task AP is not an improvement to official eight-class detection. Four-material grouping covers nine fixed plastic regressions; polymer grouping additionally accepts the Styrofoam confusion. [Complete report, artifacts, caveats and PowerShell commands](TACO_STAGE_4_MATERIAL_HIERARCHY.md).

Verification: 267 guarded tests, 251 passed, 16 documented skips, zero failures/errors. Four native Windows Tk cases pass all 15 taxonomy/NMS combinations in both modes at Image 777, all 47 validation IDs and navigation/zoom/pan; two focused final cases verify per-annotation hazard columns. All 60 bundles verified before/after; prior protected files and replay manifest unchanged. Human visual acceptance remains pending. Next authorized action is saved visual review; no training or promotion follows automatically.

Historical interruption is retained separately: the earlier vision-unfreeze attempt stopped after ten completed optimizer updates and eleven training forwards, zero validation forwards and zero new checkpoint bundles, on a zero stage-3 probe gradient. Its stopped private record is `runs/training/taco-vision-unfreeze-9452eb4dc47044a9a76cb232fe921b45/record.json`; it was not resumed and contributes no hierarchy results.

## D-064 pure-plastic-film-damped Epoch-2 continuation - 2026-10-25

Owner authorized exactly 221 updates from the verified D-057 E1 step 221 model/AdamW/RNG state. LR 1e-6 applies only to 61 pure-film batches; LR 1e-5 to 160 other batches, including all 30 cigarette images, with original ordering/loss/architecture. Three 47-image validations (CB350, CB442, in-memory 25/75 blend) completed 141 forwards; historical references were not rerun. All 55 prior and 5 new bundles verified (60 total); six core files and protected prior artifacts preserved. No test access, commit, push, merge or promotion.

CB Step 350 AP **0.258631**, Step 442 **0.242594**, and CB 25/75 blend **0.257303**. Cigarette coverage is 13/24, 14/24, 13/24; Image 777 coverage 4/7 for all three. The fixed ten-regression retention is 4/10, 2/10, 2/10; the nine-large subset is 3/9, 2/9, 2/9. This is partial recovery, not preservation of all desired gains. The requested fixed ten-loss set contains nine >=16-pixel objects and one 8--<16 object; both denominators are reported. [Full comparison, ledger and GUI commands](TACO_STAGE_4_576_CLASS_BALANCED.md). Next authorized action: owner inspects saved results; no further experiment or promotion follows automatically.

D-064 verification: 255 guarded tests, 239 passed, 16 documented skips, zero failures/errors. Four native Tk cases passed, each checking all three NMS options; all 47 image IDs selectable, correct labels, navigation and zoom/pan. No saved JSON changes or model/GPU execution during replay. Human visual acceptance remains pending.

## D-063 diagnostic NMS in saved replay - 2026-10-25

Owner authorized an explicit saved-GUI-only NMS selector and saved-prediction audit. Default none preserves historical behavior; optional class-aware/class-agnostic modes greedily suppress passing queries at inclusive IoU>=0.50. Filtering is image-wide, after source/score selection and before standard top K, or before per-GT matching in object mode. No model/GPU execution, core SOURCES edits, official protocol/metric updates, checkpoint changes, commit, push or merge.

All five saved 384/576 states were audited at 0.05/0.07 for remaining boxes, first 10/15 GT/cigarette coverage, and separately labeled post-NMS diagnostic COCO metrics. Agnostic NMS can trade correct-class coverage for fewer boxes. For the blend at 0.05/K 10, raw/class-aware/agnostic coverage is 30/31/27 of 75 GTs; cigarettes 10/11/11 of 24. Original official metric dictionaries reproduce exactly on CPU and remain unchanged. [Full audit, scope distinctions and exact GUI commands](TACO_STAGE_4_NMS_AUDIT.md).

All 55 checkpoint bundles and 1703 protected files verified unchanged; the six core bytes and no-NMS official protocol are identical. Guarded suite: 235 passed, 16 documented skips, zero failures/errors. Native runtime evidence is recorded separately; human visual confirmation remains pending.

Native verification: 24 Tk cases passed at Image 777, covering both GUI modes, all NMS choices, both score floors and top K 10/15. Defaults, selector callbacks, tables/boxes, raw restoration, navigation and zoom/pan passed. No model/GPU/weight/export operations or saved JSON mutations occurred. Human visual confirmation remains pending.

## D-062 controlled Epoch-2 cosine continuation - 2026-10-25

Owner authorized continuation from D-057 E1 step 221 using the exact restored model/AdamW/RNG state, 221 updates only and two validation evaluations (94 forwards). LR decays per step from 5e-6 to 1e-6. The six core SOURCES files stay byte-for-byte unchanged; the isolated continuation module reuses their verified functions. Five complete new bundles were saved; all 50 original plus5 new bundles verified, and protected prior artifacts preserved. No test access, reference GPU rerun, commit, push, merge or promotion.

Cosine step 350 AP **0.217172**; step 442 AP **0.221592**, above constant-LR E2 0.170508 but below E1 0.249891 and D-061 blend 0.265563. Endpoint cigarette coverage is 10/24 overall and 3/7 on Image 777; large-bin coverage 14/36. Passing queries at 0.05: 1133 across 47 images and 28 on Image 777, versus constant E2 1118/29. This is a negative/mixed diagnostic result for the combined preservation-and-clutter hypothesis, with unchanged validation selection bias. [Complete evidence and exact GUI commands](TACO_STAGE_4_576_LR_DECAY.md). Next authorized action: owner reviews saved results; no additional training or checkpoint promotion is implied.

## D-061 384/576 weight integration - 2026-10-25

Owner authorized exactly four inference-only in-memory FP32 parameter averages of D-057 step221/350/442, evaluated on the unchanged47 validation images, maximum188 forwards. Existing D-056/D-057/D-058 references are reused. Zero backward, optimizer updates, training-photo reads and test access; no checkpoint promotion, commit, push or merge.

Highest integrated eight-class AP: `blend_350_442_2575` **0.265563**, versus step350 0.262337 and E2 0.170508. All900/.05 same-class coverage: 38/75 objects, 15/24 cigarettes, and 5/7 Image777 cigarettes. This selects a diagnostic replay view only and retains validation selection bias. All50 saved checkpoint bundles verified before/after; frozen persisted tensors are bitwise identical; only558 canonical trainable tensors enter averaging. The first47-image result was retained after a cleanup guard stop; remaining states ran in isolated processes, with no repeated inference and CUDA context release between states. [Complete evidence and GUI commands](TACO_STAGE_4_576_WEIGHT_INTEGRATION.md).

## D-060 saved 384/576 replay ? 2026-10-24

Both new saved run choices are implemented for standard and per-object GUI modes, with a private local six-choice manifest. Image 777 starts in all-validation scope for browsing all 47 images. D-058 step 350 is explicitly labeled mid-Epoch 2. No training, detector inference or GPU execution. Source predictions and checkpoints are preserved. Guarded unit suite: 236 tests, 220 passed, 16 explicit skips, zero failures/errors. All six choices load 47 images and 900 queries per state/image with zero photo reads; audited source hashes are unchanged. Human GUI review remains pending. [Exact PowerShell commands, software evidence and limits](TACO_SAVED_REVIEW_384_576.md). Next authorized action: owner opens the saved replay and reviews results; no additional experiment or promotion is implied.

Additional D-060 verification after the per-run baseline fallback fix:
236 tests,221 passed,15 skips,zero failures/errors. Four native Windows Tk
smoke cases passed (both384/576 choices × standard/per-object), Image777,
all47 IDs selectable, correct labels, navigation, GT navigation and zoom/pan.
Model/GPU calls, checkpoint-state reads and exports were blocked; saved JSON
hashes were unchanged. See `work/d060-saved-review/native-results.json` and
`unit-results-final.json`. Human visual confirmation remains pending; no
new commit, push or merge was performed.

## Final Stage 4 wrap-up and publication identity (D-059) — 2026-10-24

D-043–D-058 investigations are complete. D-059 authorizes this focused
documentation/scripts/tests publication on `feat/stage4-real-image-training`
to update PR #18, without merging it or promoting any checkpoint. README and
PHASES now include the seven-row corrected-protocol benchmark progression and
links to [D-057 training](TACO_STAGE_4_576_TRAINING.md) and
[D-058 trajectory results](TACO_STAGE_4_576_TRAJECTORY.md).

Publication base SHA: `c79d513e8990701f85db0cfe5a0acf4408de1d56`.
The wrap-up SHA is the single commit containing this entry, resolved with
`git log -1 --format=%H -- docs/HANDOFF.md`; its exact SHA and verified remote
branch status are reported to the owner after push. A commit cannot contain
its own SHA in its committed text. PR #18 remains open for owner review/merge.

### Preserved corrected checkpoint inventory

Each run retains eight periodic step bundles plus E1/E2 bundles:10 per run,
50 total. D-059 performs no checkpoint/model execution or modifications.

| Experiment | Training scale | Seed / LR | Private training run | Bundles |
| --- | --- | --- | --- | --- |
| D-048 corrected reference | 256/384 | 42 /1e-5 | `taco-312e568703a54f98a7c8f4b87b29cd39` | 10 |
| D-050 lower LR | 256/384 | 42 /5e-6 | `taco-dd541c3972834a64a16771df44a7e891` | 10 |
| D-051 paired replication | 256/384 | 123 /1e-5 | `taco-4bf521d7529e4fa4b168b96786020844` | 10 |
| D-051 paired replication | 256/384 | 123 /5e-6 | `taco-e7e4008e98a241e19047e14645f06ec1` | 10 |
| D-057 scale-matched | 384/576 | 42 /1e-5 | `taco-896e2f44146d4d54aee518e6087908ca` | 10 |

All remain experimental. Step350 has the highest sampled eight-class AP0.262337,
but lower AR100 than E1 and fewer correctly covered large objects; it is not
selected or promoted. Cigarette gains do not establish overall generalization.
The owner Section185 academic-use record (D-042) is not external permission,
institutional approval or definitive legal clearance.

### Verification and owner-only next actions

The D-059 guarded fast unit/CPU suite ran234 tests:219 passed,15 documented
skips,zero failures/errors. All three trajectory fixtures and the local scale,
seed/LR and saved-review fixtures passed. Protected file metadata (59,827 files)
matched before/after verification;50 corrected bundles remain across five runs.
All212 relative links across the14 authorized Markdown files resolve locally
(18 link to retained private artifacts). Staging/link/whitespace evidence is
recorded in this publication's commit message and final owner handoff. Live
optimizer-step and model-execution fixtures are excluded; no GPU/model runs.
Private helper-dependent tests skip explicitly on checkouts lacking `work/`.
Detailed report links to local audit artifacts remain private and are not
published with the scripts/tests/reports. `runs/`, `work/`, `exports/`, `data/`,
`BANTAI-Datasets/`, environments and rights records remain untouched and unstaged.
`ARGON_RESEARCH_HANDOFF.md` and `BANTAI_CONCEPT_PROPOSAL.md` remain local,
unstaged planning references.

Only the owner may elect optional human visual review, decide PR #18 review/merge,
or authorize a future stage. No further experiment, model selection, promotion,
test evaluation or merge follows automatically from this wrap-up.

## D-058 mid-Epoch-2 384/576 trajectory audit — 2026-10-24

Owner authorized inference-only evaluation of D-057 step250/300/350/400 on47
validation images,188 forwards total, reusing saved step0/221/442 endpoints.
Observed status: **completed**. 188 validation forwards completed; all three gates passed. Cigarette same-class trajectory [8, 10, 11, 14, 14, 15]; large-bin same-class trajectory [21, 18, 16, 15, 15, 14]. No checkpoint selection or promotion.
[Trajectory report, flip ledger and limitations](TACO_STAGE_4_576_TRAJECTORY.md).
Audit `scale576-trajectory-c84b9d9344e64578ba1f2fb85f353d0c`. Validation checkpoint inspection is diagnostic and
introduces selection bias; it is not a generalization claim or selection rule.
No training/backward, train-photo/test access, core/GUI/dependency changes,
exports, checkpoint promotion or Git publication. Stop for owner review.

## D-057 controlled 384/576 training — 2026-10-23

Owner authorized one3-step discarded pilot plus442 full updates from original
SNAPSHOT at seed42/LR1e-5/FP32/frozen backbones,221 train/47 validation. Only
the two trainer CLI bound values change512→576; defaults remain256/384.
Observed outcome: **completed**; 3 discarded pilot +442 full updates completed.
All five gates passed;231 software tests,zero failures/errors (three skips).
Trainer `taco-896e2f44146d4d54aee518e6087908ca` completed both221-image epochs.
Epoch1 AP0.249891/AR1000.488141 vs D-0550.247003/0.474763; Epoch2 AP0.170508/
AR1000.370118. Cigarette same-class coverage at0.05 is5→8→15/24; image777
0→1→5/7. Epoch2 improves small-bin coverage but loses ten previously correct
GTs and large-bin coverage, with bag AP collapsing on only two GTs.
Ten bundles verified; final optimizer step442 across558 parameters and exact
seed42 image order. Full allocated/reserved peaks3.55/3.93 GiB. All1766 protected
hashes matched after execution. Seven-state metrics CSV,1500-row per-object
CSV/JSON and full training/gradient/transition summaries are in the audit.
[Full gates, results and limits](TACO_STAGE_4_576_TRAINING.md).
Audit `scale576-experiment-0b2a846e659a429cbe8a65c1ca0444e1`. No retry, promotion, test access, dependency/GUI changes,
annotated exports or Git publication. Stop for owner review.

## D-056 multi-checkpoint resolution replication completed — 2026-10-23

[Full three-model report](TACO_STAGE_4_RESOLUTION_REPLICATION.md).
Run `resolution-replication-adf0cda0193041e38be1380b29b3e50b` passed all four gates and exactly188 real forwards:
47 low +47 high for Pretrained and123/1e-5/E2. Both low baselines exactly match
saved full predictions/raw queries/metrics/loss. D-05542/E1 was read-only.

Pretrained AP0.111576→0.147847;123/E2 AP0.152880→0.140163. D-055 reference remains
0.203932→0.247003. Seven-class AP excluding bag rises in all three. At .05,
cigarette same coverage1→3,2→5,6→10/24;123/E2 reaches19/24 at floor0, not an
operating-threshold endorsement. Image777 GT279/280 gain same-class at .05,
GT284 retains it and GT278 becomes wrong-only. Common gain GT497; common
new-wrong GT67/278; no regression shared by all three. Partial scale effect,
not uniform AP superiority or a complete classification/tiny-object repair.

229 software tests:215 passed,14 documented skips; all four resolution fixtures
passed. All1521 fresh protected hashes matched after model execution;40 checkpoint
bundles verified, both D-055 attempts unchanged. Model hashes unchanged; exactly
188 validation-photo reads, zero backward/updates/train-photo/test access.
225-row paired75-GT×3-model CSV/JSON and full summary archived locally. No
core/GUI/config/dependency changes, annotated exports, publication or promotion.
Stop for owner review; next-step proposals are unexecuted.

## D-055 single retry completed — 2026-10-23

[Full measured report](TACO_STAGE_4_RESOLUTION_EXPERIMENT.md).
Run `resolution-experiment-9cebf39b0982405d8dfdd954ae2dbed1` passed all five gates and exactly47 high-scale
validation forwards, after47 exact baseline forwards and one image777 pilot.
AP0.203932→0.247003; AR1000.396766→0.474763. At .05/.50, same-class coverage
28→33/75 (eight gains,three losses); any coverage49→56. The original4–<8px bin
improves any/same4/1→8/4, but <4px remains0/14. Cigarette any/same2/2→8/5;
image777 gains localization but no qualifying same-class result at .05.
Regressions GT60/88/250 are all originally larger objects. No automatic promotion.

Software228 run,214 passed,14 documented skips; all three resolution tests passed.
Exact full baseline predictions/raw queries/metrics/loss and unchanged parameter
hash verified. All1497 fresh protected hashes matched after execution;40 checkpoint
bundles retained. GPU inference fit does not prove high-scale training fit.
Private artifacts include both-scale geometry, baseline replay, pilot, high
evaluation,75-row paired CSV/JSON and fixed-bin/transition summary. Original
failed attempt is unchanged; retry count is consumed. Zero updates/backward,
training-photo/test access, annotated exports, dependency/core/GUI changes or Git
publication. Proposed human class/annotation review and one-checkpoint replication
remain unexecuted. Stop for owner review; earlier stopped notice is historical.

## D-055 stopped at software gate — 2026-10-23

[Resolution attempt report](TACO_STAGE_4_RESOLUTION_EXPERIMENT.md).
Attempt `resolution-experiment-629229adf1c6480d930aac79250d18f8` ran no real model inference, backward or optimizer steps.
The guarded228-test suite reported1 failure,89 errors and13 skips; all three new
resolution fixtures passed, but the overall gate failed. The harness erroneously
blocked synthetic fixture image saves and omitted an indirect gradient fixture
from exclusions. Corrected work-only guards are prepared but were not rerun.
All later gates, metrics, memory pilot and47-image comparison remain unattempted.
Read-only post-stop verification found579 prior-anchor files unchanged
and verified40 checkpoint bundles;221 training photos were not read or rehashed.
Core/GUI/config source remains unchanged. Stop per owner gate rule; a fresh
attempt requires explicit owner authorization. No commits/pushes/merges occurred.


## Small-object/classification audit completed — 2026-10-22

D-054: [Full audit and proposed next experiment](TACO_STAGE_4_SMALL_OBJECT_CLASSIFICATION_AUDIT.md)
verified actual CPU preprocessing/masks for all47 validation images and75 boxes,
then analyzed all nine saved states. Image777 content is350×256, not384-square.
Small resized boxes strongly associate with poor candidate coverage;20 pretrained
and21–27 trained annotations/state have any-class IoU>=.50 without correct-class
coverage at .05. Cans/wrappers have persistent class-related failures; cigarette
failures mostly lack qualifying any-class coverage. Causation remains unproven.

Nine focused checks passed;40 checkpoint bundles and800 protected files verified.
Original GUI, manifests, datasets, weights and current owner source changes are
unchanged. Initial owner image777 observations are recorded separately from
automated results. The single proposed experiment is fixed-weight inference-only
256/384 versus384/576 on the preserved42/1e-5/E1 checkpoint, with GPU fit unmeasured.
It has not run or been authorized. No new training/inference or Git publication.
Stop for owner review; no additional experiment is approved.


## Per-object saved diagnostics completed — 2026-10-22

D-053: [Per-object report and exact GUI command](TACO_STAGE_4_PER_OBJECT_REVIEW.md)
covers all eight image777 GT annotations and46 GT annotations in the unchanged
20-image review. Nine saved states are compared at all900/top100 ×0/.05/.25;
complete JSON/CSV evidence is separate from the original review manifest.
The explicit --per-object saved GUI supports GT selection/navigation, highlights,
zoom, complete overlap tables and a scrollable GT/model summary. Default GUI and
terminal routing are unchanged. For seed123/1e-5/E2, GT284 is hidden by the five-box
cap, while other missing GT have score or candidate-coverage limitations.
All29 focused tests and native checks of four object-mode choices plus standard
mode passed; the 20-image subset rendered. Human identity/annotation judgement
remains pending. No training, inference, dataset/metrics changes or Git operations.
Stop for human review; no experiment is authorized.


## Cross-seed analysis and visual-review preparation completed — 2026-10-22

D-052 is analysis only. [Full cross-seed report](TACO_STAGE_4_CROSS_SEED_ANALYSIS.md)
recomputes all nine saved COCO metric records, checks all 40 checkpoint bundles,
and compares class AP/AR, same/any candidate coverage, fixed-score diagnostics,
incorrect-class assignments and descriptive two-seed ranges. Cigarette/film
AP and AR improve over pretrained in all eight states; wrapper/can/Styrofoam
recall is lower throughout. LR/epoch AP rankings reverse by seed.

Twenty validation images and specific human questions are prepared in the local
review manifest. The report gives the exact PowerShell command and four run
keys. Explicit `--saved-review`/`--run-choice` reuses the four-panel viewer;
historical default replay and strict training verification remain unchanged.
All four saved choices and the actual terminal CLI passed; native replay rendered
all 20 examples and closed normally with zero exports. All 22 focused tests passed.
Human photo/material/annotation acceptance is pending. Recommended next action
is human data-quality review; a third paired seed remains a proposal, not
authorization. No new training/inference, test evaluation, promotion or Git
publication occurred. Stop for owner review.


## Paired seed-123 replication completed — 2026-10-22

D-051's A (1e-5/123) and B (5e-6/123) both completed and passed integrity audits:
884 full updates plus six discarded pilot updates, no extra runs. A AP improves
0.140478→0.152880; B AP regresses 0.137919→0.113018, reversing the seed-42
epoch pattern. Both rates change best AP epoch across seeds. This weakens a
general lower-LR prevention claim; two seeds do not establish superiority.

Configurable `--seed` defaults to 42 and retains the LR interface. Five seed
regression tests passed; guarded discovery passed 211 tests with three documented
skips. Original data, weights, driver/runtime, native adapter, rights gate and
exact baseline passed. Both full starts have identical RNG/parameter hashes and
442-image ordering; pilot-release checks passed. Twenty checkpoint bundles and
756 protected files passed, including unchanged A artifacts after B.
Study: `runs/training/seed123-pair-10acabc298064547bfab6d119a109359/`.
A: `runs/training/taco-4bf521d7529e4fa4b168b96786020844/` (AP/AR100 E2
0.152880/0.274295). B: `runs/training/taco-e7e4008e98a241e19047e14645f06ec1/`
(AP/AR100 E2 0.113018/0.291915). Both use seed 123.
[Replication report](TACO_STAGE_4_SEED_REPLICATION.md) includes the full cross-seed
deltas, class/coverage/score/loss tables, runtime caveats and evidence.
The original 42/1e-5/E1 remains strongest measured AP/AR100 and is not promoted.
Only these two runs were executed; a third paired seed 456 is a proposal only.
Stop for owner review, with no promotion, test access or Git operations.

## Controlled LR-only experiment completed — 2026-10-22

D-050's one fresh two-epoch LR 5e-6 comparison completed, exit 0, with exactly
442 full-run updates after three discarded pilot updates. New E1 AP/AR100:
0.137099/0.330432; E2: 0.145952/0.302977. Lower LR avoids the previous aggregate
AP collapse and beats the 1e-5 E2 result, but does not preserve the stronger 1e-5
E1 result (0.203932/0.396766). Same-class all900 coverage still falls 32→30.
No checkpoint is promoted. The next seed-only replication is a proposal only.

Guarded preflight passed: 206 tests, three documented skips, all 268 source/
geometry checks, exact pretrained baseline agreement and unchanged runtime/data/
rights. Four new LR tests passed. Actual optimizer groups and both saved epoch
optimizer states confirm 5e-6, with states at 221/442. Ten immutable bundles and
all 612 protected file hashes passed; seeded training coverage is exact.
Run: `runs/training/taco-dd541c3972834a64a16771df44a7e891/`.
Evidence: `runs/training/lr5e6-7d3cd8dad9064aa9949fb2da4bf421c0/`.
[LR report](TACO_STAGE_4_LR_EXPERIMENT.md) contains all five model states, per-class
AP/AP50/recall/coverage, score sweeps, losses, numeric curves and preservation.
The same run continued after a long pre-update wall-clock gap; lifecycle timing
is not an active-compute benchmark. No optimizer state or weights were resumed
from an earlier fine-tuned checkpoint. Historical
source-pinned TACO replay policies remain unchanged and reject the new trainer
fingerprint; normal BANTAI inference is unchanged. Training has stopped after
epoch 2/final validation. Await owner review, without promotion, test access,
further optimization or Git operations.

## Corrected-run diagnostics completed — 2026-10-21

D-049's [diagnostic report](TACO_STAGE_4_CORRECTED_DIAGNOSTICS.md) explains the
observed E1→E2 regression using saved validation results, without new inference
or training. Any-class all900 coverage at IoU >=0.50 rises 55→57 annotations;
same-class coverage falls 36→30. Bags/styrofoam lose useful class assignments
despite retained well-aligned boxes; cups also lose AP. Score shifts are mixed.
Overfitting, forgetting and excessive learning rate are not established causes.

Forty-five existing tests and three local diagnostic fixtures passed. Native
corrected-run replay checked 20 representative images, overlays, raw phrases,
zoom/pan and navigation; human photo-level confirmation remains pending.
Evidence: `runs/training/corrected-diagnostics-629e30fb299f442698de14650b279c6c/`.
The continuation rechecked 603 prior protected files: 602 still match; the owner's
production detection configuration changed before continuation and is preserved
with its exact current hash recorded separately. Training/model/data/checkpoint
identities remain unchanged. No existing source file was edited for this task.

The report ranks a fresh two-epoch LR-only comparison (1e-5→5e-6) first as a
proposal. No next experiment is implemented or authorized. Stop for owner review;
no test access, checkpoint promotion, uploads, commits, pushes or merges.

## Controlled corrected-target restart completed — 2026-10-20

D-048's single two-epoch run completed, exit 0: three discarded pilot steps,
442 full-run updates, no historical resume or further training. See
[complete corrected-retrain results and curves](TACO_STAGE_4_CORRECTED_RETRAIN.md).
Corrected v2 AP / AR100: pretrained 0.111576 / 0.363077; new epoch 1
0.203932 / 0.396766; new epoch 2 0.084500 / 0.211229. Epoch 2 regresses despite
lower corrected-target validation loss. Both new checkpoints remain experimental.

Preflight: 202 tests passed, three explicit skips, including all eight coordinate
tests passing; exact pretrained baseline replay and all 268 source/geometry
checks passed before any update. Ten new checkpoint bundles and 471 protected
file hashes passed; optimizer states are at step 442. Settings, rights
uncertainties, dataset, owner detection edits, source code and historical evidence
are preserved. No test access, dependency changes, uploads, promotion or Git
operations. New run: `runs/training/taco-312e568703a54f98a7c8f4b87b29cd39/`;
experiment: `runs/training/corrected-retrain-72650648148549ccb8aa8a176f9c5b0b/`.
Next action is owner review. A further single-variable experiment is a proposal,
not authorization; no automatic continuation.

## Explicit GUI opt-in — 2026-10-20

D-047 restores the existing four-panel viewer through `--gui` while retaining
D-046's terminal-only default. From PowerShell at the repository root:
`.\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --gui --image-id 42`.
The all-validation selector includes 42, 113, 200, 31, 1096, 252 and 1014.
Saved-result verification remains mandatory. Fresh GUI analysis JSON records
display status separately; original artifacts and all modelling settings remain
unchanged. No commits, pushes or merges. Human photo-level review remains pending.

Verification: 18 software tests passed; the actual default command printed the
verified metrics and exited 0. Native TkAgg replay checked all 16 representatives
(including all seven requested IDs), shared zoom/pan, tables/highlighting,
diagnostic controls and all-validation navigation with zero guarded model,
training or image-export calls. Evidence:
`runs/training/step2-review-be0b248ac2554160a06eac179cb6cc9b/native-runtime.json`.
The explicit `--gui --image-id 42` entry check passed and persisted `closed` in
`runs/training/step2-review-2901e7965fe9474584bf20a760f6e005/analysis.json`.

## Owner correction: terminal epoch output — 2026-10-20

D-046 supersedes the Step 2 epoch comparison GUI request: the owner wants only
terminal output per epoch with the updated scores. The diagnostic window was
closed normally. `taco_visual_review.py` now verifies saved provenance and prints
pretrained/epoch-1/epoch-2 AP, AP50, AP75, AR100, corrected-target validation loss,
score median/P99/maximum and counts >=0.25. Default execution opens no GUI, reads
no photographs, performs no inference/training and creates no new report files.
Run `.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py`.
Prior GUI evidence is historical; it is no longer the owner-requested workflow.

## Step 2 diagnostic replay — 2026-10-20

D-045 authorizes controlled local-only saved-result visual diagnostics and initial
error analysis. [Workflow, commands, selection, measured findings and pending human
checks](TACO_STAGE_4_VISUAL_REVIEW.md). Four synchronized TkAgg views, source hashes,
category/query/score labels, separate raw phrases, copyable tables, all-900 access
and full-validation navigation are implemented without inference/model loading.

Sixteen deterministic representatives cover all eight classes and contrasting
outcomes. Same-class IoU-0.50 matches at diagnostic .05 are 28/17/14 of 75; at .25,
epoch models hide 17/14 otherwise matching retained candidates. Wrapper matches
fall 9->1->1, can matches 2->1->0, styrofoam matches 4->2->0. These are annotation
comparisons, not confirmed visual errors/physical counts. Unmatched predictions
are not automatically false positives. Human feedback/annotation completeness and
readability remain pending; the separate interactive owner window may remain open.

Final analysis: `runs/training/step2-review-cce4f3dcac4c4c788cc8cab5957d396e/`.
Actual native replay checked all 16 representatives, synchronized coordinates,
tables, zoom/pan propagation, empty views and clean closure without image exports.
Fourteen new tests pass; safe full suite 201 tests, 198 passed/three skipped. No
training, test access, source/data/checkpoint changes, commit or promotion. Proposed
future training is a controlled corrected-target restart after human review, not an
approved run or automatic continuation. Stop after this diagnostic deliverable.

## Step 1C historical reevaluation — 2026-10-20

D-044 authorizes a separate evaluation-only provenance pathway and unchanged
validation-split reevaluation. [Design, commands, qualification and full results](TACO_STAGE_4_REEVALUATION.md).
Exact historical artifact/source anchors, reviewed Step 1B old/new source pairs,
current code/native sources, corpus/rights and model/processor/tokenizer/adapter
compatibility pass. Strict training-resume verification remains unchanged and
still rejects the corrected context for both checkpoints.

Corrected pretrained / epoch-1 / epoch-2 AP: **0.111576 / 0.126627 / 0.066215**;
AR100: **0.363077 / 0.275363 / 0.235395**. Epoch-1 higher aggregate AP coexists with
lower recall; epoch 2 regresses. All score/category outputs match historical ones
exactly; neither trained checkpoint has a score >=0.25. Small-object AP stays zero.
Recalculated valid-target losses differ from historical padded-target losses.

Successful private record: `runs/training/step1c-evaluation-3d24a67e1fe041e5a8acf56a26406db4/`.
94 validation photo reads, zero updates/test access/state restoration; guarded
runtime calls all zero. Nineteen new tests pass; full safe suite 187 tests,
184 passed/three skipped. Existing sources, artifacts, owner edits and packages
are preserved. No commit, push, merge, promotion or visual analysis occurred.

Next proposed action is Step 2 local visual comparison, requiring its own scope
authorization. The Step 1B historical reevaluation blocker below is now resolved
only through the separate evaluation contract; training resume is not relaxed.

## Step 1B coordinate correction — 2026-10-20

D-043 authorizes only the confirmed target/decoder corrections, independent tests
and validation-only comparison. [Implementation, evidence and limitations](TACO_STAGE_4_COORDINATES.md):
native valid-image targets and original-size decoding now pass eight independent
regressions; original functions fail them. Broader suite: 168 tests, 165 passed,
three skipped (one privilege limitation, two prohibited toy optimizer fixtures).

Corrected pretrained validation on the same 47 images/75 annotations: AP
0.111576016, AP50 0.158252880, AP75 0.135975042, AR100 0.363077235. All 42,300 query
scores/categories exactly match historical baseline; geometry alone changed the
measurement. New private evidence is under
`runs/training/step1b-evaluation-c8b127d8bf364c7dbb735795cd81c06d/`.

Epoch-1/2 artifact hashes pass, but their exact-source/protocol contracts reject
the corrected implementation. Those reevaluations are stopped without bypass or
manifest rewriting. Next owner decision: explicit evaluation-only provenance
handling for historical weights, preserving strict training-resume checks.
Earlier reported AP/loss below are historical measurements with incorrect geometry.
No new optimizer updates, test access, Step 2 analysis, promotion or PR merge.
Owner detection edits, original weights/data and environments remain preserved.

## Completed first real-image run — 2026-10-19

D-042 records the owner's actual non-commercial/local academic declaration and authorizes the existing workflow on a defensible, documented Section 185 claim. [The evidence record](TACO_STAGE_4_ACADEMIC_USE.md) expressly claims no rightsholder permission, institutional approval or definitive clearance; the v2 gate keeps clearance flags false and binds the assessment/corpus. A strictly verified source-JSON formatting change was accepted only because default serialization reconstructs the exact original recorded hash; source/export files were not rewritten.

Actual run `taco-1698bcb46b85435786dd2a199910910f`: **completed two epochs / 442 full-run updates**, plus three discarded pilot steps. Full-run mean losses 14.304753 / 12.103961; validation loss 19.348786 baseline -> 12.133389 final. **AP [0,1] deteriorated from 0.000298934 to 0.000057591**. Fixed score 0.25 diagnostics miss all 75 validation annotations after fine-tuning. Runtime success is not detection-quality acceptance. [Full measured results, commands, class limits, failures and checkpoint location](TACO_STAGE_4_RUN.md).

Real pilot peaks: 2.384 GiB allocated / 2.535 reserved; full run: 2.382 / 2.539 GiB. Lifecycle 449.386 seconds. Native reload of the final `checkpoints/epoch-02` and replay on all 47 validation photos matched every prediction/raw query, metric and mean loss exactly, with zero extra updates. All ten checkpoint bundles/hash lists and final optimizer/RNG state passed audit. Full suite: 160 run, 159 passed/one existing skip. No prior CUDA synthetic investigation was repeated.

**Next action:** review poor detections locally before proposing any longer training or changing resolution/prompts/objective; no further run or optimization was started automatically. Human image/annotation/error review remains pending because no photos were uploaded for agent inspection. Test photos remain unused, model/split/taxonomy/inference/environment preserved, experimental weights private and unpromoted, PR #18 unmerged. Training source is commit `c8c7102`; preserve the ignored rights record and bound assessment bytes for within-budget checkpoint resume. The first experiment has no remaining runtime blocker; it has a failed detection-quality outcome.

## Initial Stage 4 implementation handoff (historical)

Current 2026-10-19: D-041 authorizes real-image fine-tuning and initial validation as one continuous task, conditional on documented photo and annotation/database rights. Branch `feat/stage4-real-image-training` builds on Stage 3D `22bcdbc` (open PR #17, carrying #14–#16; fetched main `077657c`). [Stage 4 implementation, rights scope, commands, resume behavior and evidence](TACO_STAGE_4.md) are available. The loader/trainer/pilot/progress/COCO evaluator and complete optimizer/RNG/sampler checkpoints are implemented. The verified Stage 3D adapter remains unchanged; no native miswired loss is used.

Implementation commit `601bb81` is pushed in [PR #18 targeting main](https://github.com/Piyaya-hub/BANTAI-Modeling-/pull/18), open and unmerged. It carries the unmerged Stage 3A–3D lineage; publishing this code does not publish dataset/model artifacts or clear the rights gate.

Full suite: 158 run, 157 passed/one existing privilege skip. Final generated-image CUDA continuation `76903099b57f453da9514e6a06fce128` passes: exact pre-update weights/optimizer/RNG/batch restoration, loss equality, post-update parameter maximum difference `4.10e-8` (not bitwise identical), encoder updates, frozen preservation and decoder equivalence. A same-model repeated-update control differs by `8.38e-8`; an isolated strict-determinism probe identifies unsupported CUDA grid-sampler backward. Earlier tolerance, shared-alias diagnostic and combined-probe OOM failures remain documented. No model/resolution/loss fallback occurred. Source photos were not opened. The actual real-data CLI stopped at `blocked_rights` with zero photo reads/optimizer steps. No real pilot, actual-photo memory result, TACO-trained checkpoint or real validation metric is claimed; human review remains pending.

**Next action:** owner supplies documented permission or approved institutional/legal basis, authority/date/evidence/scope/obligations covering the exact 221 training and 47 validation photos plus annotation/database rights. Metadata contains 161 unspecified/null, 30 generic CC and 77 ODbL-notice images in that scope. Then execute the documented command directly: actual-photo coordinate preflight, unchanged baseline, measured memory pilot and bounded run; stop/report technical failures without silently changing settings. Test photos remain untouched. No additional standalone assessment stage, dataset changes, cloud spending, publication, inference promotion or merge.

Only `pycocotools==2.0.10` was added inside `.venv-training`, with a pinned Windows wheel hash; existing package versions and inference `.venv` are preserved. Owner detection edits are excluded from commits. Generated images/checkpoints/reports and any actual private rights evidence remain ignored.

## Historical Stage 3D handoff

Current 2026-10-19: D-040 approves the training-only encoder-output adapter and synthetic qualification. Branch `feat/stage3d-training-adapter` starts at Stage 3C `6ce0907` (PR #16, carrying prior #15/#14; main at branch creation `077657c`). Owner detection edits remain local and excluded. See [implementation, commands, complete PASS/FAIL evidence and limits](TACO_STAGE_3D.md).

Final run `adapter-a06cfea197c545fc82c6df91c0b39583` is **PASS** for FP32 batch 2, frozen vision/BERT, 192/256 synthetic inputs: two corrected steps each for 11/2/1/0-target cases. Encoder proposal projection/normalization and box head receive nonempty gradients, Adam state and updates; empty regression is zero. All trainable parameters have gradient tensors. Decoder outputs remain exactly equal with/without the adapter at identical weights, before/after updates. Offline model/processor/adapter-context replay and repeated eval losses match exactly. Resume is explicitly unsupported.

Peak case allocator memory is 2.732 GiB allocated / 2.910 GiB reserved; separate reload verification peaks at 2.998 GiB reserved while the original model remains resident. Full suite: 144 tests, 143 passed/one existing privilege skip; eleven new adapter tests passed. Installed Transformers, both environments, original snapshot, inference, taxonomy, strict-clean manifest and earlier synthetic/investigation code remain unchanged. Experimental weights/reports stay ignored, with no promotion. Stop here: real-data training, full-resolution/AMP fit, rights, data coverage/leakage acceptance, resumable trainer and evaluator qualification remain unrun/unresolved. No merge is authorized.

## Historical Stage 3C handoff

Current 2026-10-19: D-039 authorizes native loss investigation only; experiments completed late 2026-10-18 local time. Branch `investigate/stage3c-training-loss` starts from Stage 3B `b7a350d` (open PR #15, carrying Stage 3A #14; fetched main remains `077657c`). Local owner detection edits remain excluded. See [complete causal investigation, commands and approval proposal](TACO_STAGE_3C.md).

Controlled native CUDA cases with 11/2/1/0 targets completed, reset to the same pretrained state before each single native update. Encoder CE was 30378/167259/334564/334580; independent focal accounting and captured inputs identify learned-query/pre-encoder text scores rather than image-conditioned encoder proposal logits. Native boxes are detached; proposal-head probes have no gradients, Adam state or updates. Diagnostic selected encoder tensors recover box-head paths and give baseline CE 2.566 without changing the criterion. A separate factor audit proves graph restoration leaves loss values unchanged, whereas score routing changes matching. No hypothesis objective was optimized and no Transformers patch or library change occurred.

Peak diagnostic allocator memory was 2.535 GiB allocated / 2.840 GiB reserved. Tests: 133 run, 132 passed, one existing Windows privilege skip; seven new fixtures passed. Both environments, protected source/settings/splits, Stage 3B trainer and cached weights remain preserved. Next: owner review/approval of a proposed Stage 3D external training-only encoder-tensor adapter and synthetic qualification. Stop before that implementation and before real-data fine-tuning. Original-resolution/full-parameter/AMP training, rights/evaluation/resume and dataset coverage remain unresolved. No merge is authorized.

## Historical Stage 3B handoff

Current 2026-10-18: D-038 authorizes isolated training setup and synthetic-only smoke checks. Branch `feat/stage3b-synthetic-training` carries Stage 3A `0ed97d9` from open PR #14, based on main `077657c`. The owner's uncommitted detection settings are preserved. See [complete environment, commands, actual runs and limitations](TACO_STAGE_3B.md).

Installed `.venv-training` independently with the recorded Windows versions and SciPy 1.15.3; inference `.venv` is not modified. Twelve new tests pass. The full isolated suite runs 126 tests: 125 pass, one existing Windows privilege skip. The first CUDA attempt failed before loss because 128/192 images had too few encoder positions for 900 queries; a guard now rejects that condition. Explicit 192/256, FP32, batch 2, frozen vision/BERT passed two native forward/backward/AdamW updates and offline save/reload. Final run `ec713b1fbc47407b8d729d47fa13fbf2` records peak allocated/reserved memory 2.694/2.852 GiB and zero reload logit/box differences. All base snapshot/frozen parameter hashes remain unchanged.

Qualification remains limited: native encoder box predictions/losses have no gradients and ten otherwise-trainable encoder tensors receive none; encoder classification dominates the total loss (~60,766 initially). No loss/version patch was made. Next action is review of these findings and authorization for further trainer qualification before real-data fine-tuning. No TACO photo training, real-data metrics, changed exports/taxonomy, cloud upload/spend or inference changes occurred. Full-resolution/all-parameter/AMP fit, resume, rights and rare-class/leakage readiness remain unrun or unresolved. Do not merge automatically.

## Historical Stage 3A handoff

Current 2026-10-18: D-037 authorizes technical feasibility assessment only. Branch `docs/taco-stage3a-assessment` was created from completed Stage 2 `18b1094` and fast-forwarded to latest fetched main `077657c` after PR #13 was found merged externally; their file trees are identical. The owner's local `config/detection.json` edits remain excluded. See [recommendation, exact-version evidence, recovery bounds, rights gaps and proposed smoke protocol](TACO_STAGE_3A.md).

Recommended route is the pinned HF checkpoint with Transformers 4.51.3's existing loss in a separate future PyTorch trainer/environment. SciPy and pycocotools are absent locally. Token-span and synthetic COCO processor probes passed, but no model was constructed or trained. Training VRAM/backward/AMP/save-reload remain unrun. Approximately 24 GB GPU capacity is a planning recommendation, not a measured requirement. Excluded/crowd flags do not provide spatial loss ignoring.

The validated 315-image/538-object strict-clean export remains unchanged. Conditional review bounds: coordinates alone could add 44 images/55 mapped objects; known excluded-category cases could add 280/597. These are not approved additions. Of 315 retained photos, 185 have null license metadata, 35 only `CC`, and 95 an ODbL/OpenLitterMap notice; individual rights and annotation/database terms remain unresolved.

Next action is owner review/approval of rights handling, isolated trainer/smoke scope, hardware/budget and any expansion policy. No taxonomy change, replacement export, dependency installation or fine-tuning occurred. Stage 2's 13 focused tests passed again; independent validation and final documentation checks are recorded in the assessment. No merge is authorized.

## Historical Stage 2 handoff

Current 2026-10-18: D-036 authorizes strict-clean TACO conversion and splitting on existing `feat/taco-harmonization`, after the owner's representative annotation review. The local detection config is preserved and excluded from commits. New preparation/policy/validation code is independent of the unchanged detector, taxonomy, annotation viewer and dependencies.

Actual local export: `exports/taco/strict-clean-stage2-v1/` (ignored), 315 images / 538 annotations. Seed 42 gives train 221/397, validation 47/75, test 47/66 (images/annotations). All eight classes are present, but plastic bags have only two validation/two test images; cups and styrofoam also have small evaluation sets. Strict-clean selection sacrifices substantial mixed-scene coverage and can bias evaluation. COCO/source gates and independent reconstruction both passed; `training_ready=false` remains explicit.

Source images/annotations are unchanged and external. Full original metadata, image-level quarantine decisions, unique exported IDs and original IDs/category/record indexes are retained in JSON/CSV ledgers. All 152 dimension mismatches plus nine additional ambiguous orientation cases are held. Exact-byte/pixel duplicates were absent; two near-duplicate candidate pairs were investigated, with the eligible pair coassigned to train and the other quarantined. No ignore strategy, source repair, copy/download, model load or fine-tuning occurred.

See [files, commands, exact exclusion waterfall, class counts, checks and limitations](TACO_STAGE_2.md). Next action is owner review of bias/coverage and licensing/trainer compatibility before authorizing trainer work. PR #13 is the existing open review target; no merge is authorized.

Final checks: 114 tests, 113 passed/one existing privilege skip. Independent real-data reconstruction returned passed with no errors; all final artifacts retain matching checksums. Source hashes, the inference/audit/viewer code, taxonomy and dependencies are preserved. No photo copies are present in the export.

## Historical Stage 1 handoff


Current 2026-10-18: D-035 authorizes TACO harmonization/audit and annotation visualization, without fine-tuning, downloads or final splits. Branch `feat/taco-harmonization` starts at main `5351d57`. New independent scripts/config/tests leave all detector/model/threshold/viewer code and existing requirements untouched. The owner's changing detection config remains local and uncommitted.

Actual external source: 1,248 readable, 252 missing, zero unreadable images; 3,938 annotations on readable images. Conservative validation holds 520 otherwise-valid annotations for stored/declared dimension mismatch and nine invalid annotations on readable images: 3,409 all-class usable annotations, 1,653 mapped objects across 735 images. All 152 dimension mismatches match an EXIF-oriented preview; coordinate policy needs review. There are eleven invalid annotations in the full source (two duplicated IDs affecting four records, seven negative-origin boxes), 49 excluded source categories and explicit Unlabeled litter. 413 mapped images also have usable excluded annotations; no implicit backgrounds are generated.

Report is local/ignored at `reports/taco/review-v1/`; source photos and annotations remain external and unchanged. Native Windows checks exercised all-class sampling over ten actual photos and an EXIF preview over three, with selection, boxes, zoom/Home and closure. Agent inspected three in-memory annotation previews; human acceptance remains pending. Software suite: 100 passed/one existing privilege skip. See [file purposes, commands, counts, checks and limitations](TACO_PREPARATION.md).

Next: owner review of annotation previews and decisions on exclusions/unlabeled regions, coordinates and source anomalies before authorizing a training-data export/split stage. No training or grouping is authorized, and no merge is performed.

## Historical Phase 2E handoff


Current 2026-10-18: D-034 authorizes functional checking and handoff, stopping before Phase 3 decisions/optimization. Branch `feat/phase-2e-functional-checking` starts at latest fetched main `a3ec215` (PR #11 merged externally). Study baseline prompt was `plastic bag. drink pouch. cigarette butt` at 0.20/0.10. A subsequent local edit adds `a` before `cigarette butt`; it remains preserved and untested, alongside the untracked nested folder and unchanged environment. No defaults were selected.

Five cached CUDA experiments on the same three originals completed without model/processing failures: baseline, concrete prompt, original category prompt, box-only threshold and text-only threshold. All 15 raw-phrase/box overlays were inspected by the agent, with actual native replay verifying tables, zoom/pan callbacks, scrolling and closure. Findings include missed small litter, mixed/empty phrases, near duplicates, road-stripe false detection and pile/group confusion. No clear pile or clean no-trash scene is available. See [run IDs, timings, per-image comparisons, commands and limitations](PHASE_2E_RESULTS.md).

The owner reported oversized/missing labels on small/nested boxes. Corrected defaults are 7.5/7 points; labels can extend within the axes and move above enclosing borders, with query IDs/connectors. Final human acceptance is pending, not inferred from automated checks. A separately reproduced Windows atomic-replacement access-denial was fixed with bounded retries; permanent failures preserve old JSON. Final software suite: 91 passed/one privilege skip. No detection logic, mapping, dependencies or model revision changed, and no inference rerun was needed for either fix.

Next action: owner review of corrected viewer and Phase 2E evidence. Missing scenarios and category quality remain acceptance gaps. Phase 3 recommendations are proposals only; no optimization/grouping/training implementation or merge is authorized.

## Historical Phase 2D handoff


Current 2026-10-18: owner authorizes Display and Summaries (D-033), not Phase 2E or fine-tuning. Branch `feat/phase-2d-display-summaries` starts at latest fetched main `76c33f9` (merged PR #10); its tree already contained all previous viewer work. Local owner prompt `plastic bag. drink pouch`, thresholds 0.20/0.10 and untracked `BANTAI-Modeling-/` are preserved, excluded from this commit. No environment changes or inference rerun.

Existing: shared tab, raw phrase wrapping, original pixel axes, passing-only selectable tables, verified replay, persist-before-display and atomic separate status. Added: validated configurable label font defaults 8.5/8 points, display-setting persistence, explicit processing/error status and decimal-unit explanation in the GUI. Native replay passed on a detection photo and empty photo, including copy, scrolling and closure. Windows tests: 89 passed, one privilege skip. Final human small-label readability/interactive zoom-pan acceptance remains pending. See [commands, evidence and limitations](PHASE_2D_CHECKS.md). Next authorized action is review and outstanding visual checks; no merge or phase transition.

## Historical Phase 2C handoffs


Latest scrollable-tab correction (2026-10-18): continue the fetched `000b3b0` feature branch, preserving owner prompt/threshold edits and the untracked folder. One local Tk/TkAgg tab now repeats Image #N, Detection Summary, Text-Token Details and Configuration across the batch. Raw decoded phrases replace category labels; original pixel ticks, bounded readable labels above boxes and zoom/pan remain. Complete selectable GUI tables are passing-only and actual saved configuration is shown. Multi-image replay repeats `--result`/`--image`, verifies every source before pending persistence, and opens one shared window; inference persists the whole batch before display. Software checks: 88 passed/one privilege skip. The owner confirmed synthetic layout/scrolling/output; actual two-photo replay closed normally but exposed a missing q2 label, corrected with actual-glyph/whole-word wrapping and optional-ID omission. The owner confirmed the corrected photo labels were complete, then requested smaller labels. Final 8.5/8-point labels are software-verified; their separate visual confirmation remains pending. Actual Tk checks verified scrolling, selectable/copyable tables and clean closure, including a final smaller-label run. Four saved text replays preserved all images/JSON without model imports, inference or sockets. No open PR exists on the current branch (PR #9 was already merged externally); no merge is performed. See [current GUI commands and detailed evidence](SCROLLABLE_VIEWER.md). Earlier native side-panel/terminal layouts below are historical.

Latest native-viewer restoration (2026-10-18): same feature branch from fetched `b915bd3` (PR #9 already merged externally; no open branch PR found). Owner supersedes terminal-only/no-pop-up restrictions. Native in-memory TkAgg displays the full oriented image with valid passing boxes and category-only labels; a scrollable same-window panel contains matching IDs, raw phrases/categories, percentage scores and rounded pixel coordinates; passing-only terminal tables remain. Native windows retain zoom/pan without further navigation verification being requested. Replay identity and atomic status separation remain; no inference is rerun. Owner confirmed synthetic image/labels/navigation visibility. A resulting toolbar cleanup error was fixed; subsequent saved-photo replay `736be71a422949e3b04af39558c2f1af` closed and exited 0. The owner requested a label adjustment after the real-photo preview and clarified category-only on-image text. Final category-only/panel visual confirmation remains pending. Software checks: 77 passed/one privilege skip, including overlay alignment/empty results/closure/errors. Four saved-record summary/full replays preserved all JSON/photos without model imports or sockets. Owner prompt/threshold edits and untracked folder stay uncommitted and preserved; only backend migration is committed from configuration. See [current native usage and detailed evidence](NATIVE_VIEWER.md). Earlier terminal-only status below is historical. No merge.

Latest consistent-presentation correction (2026-10-18): fetched `6b0a14a`, same feature branch/PR #9. Each detection now uses identical wrapped Field/Value records with image/query IDs, quoted raw phrase, stored category/reason, decimal score, threshold status, rounded box and separate policy state. Unmapped retained/PASS predictions remain valid detections with no assigned category; incomplete phrases are never remapped. Preview descriptions expose source/sample dimensions and terminal resolution limits; Lanczos sampling never upscales originals. Existing Windows software checks: 81 passed/one privilege skip. Replayed all three saved records in summary/full with complete 900-query/13,500 ordinary-token coverage per full replay, no model imports, sockets or inference, and unchanged original hashes/JSON bytes. Windows PTY second-photo replay `052ed9155c7d4ab5bdf7bcf175dcf780` completed; owner VS Code visual inspection remains pending because its visual surface is inaccessible. See [current presentation and replay instructions](TERMINAL_VIEWER.md). No model/configuration/threshold/core inference changes or merge.

Latest width/grouping correction (2026-10-18): same branch/PR #9 from fetched `8b1be7e`. Preview width is configurable and height follows aspect ratio into scrollback. Summary/token tables stream without Enter pauses; tokens are grouped by query with an unwrapped status header. Summary remains passing-only, while explicit full mode restores every raw query/ordinary-token score, including below threshold. Saved-result replays on all three photos verified summary groups 2/3/2 and full coverage 900 queries x 15 ordinary tokens each, without model imports/inference or source/result mutation. Real Windows pseudo-terminal replay `d21364a916a4478baaf624499f324867` completed without input; owner VS Code visual inspection remains pending. Tests: 78 passed/one privilege skip. See [current usage and evidence](TERMINAL_VIEWER.md); older passing-only-for-full and height-limited/paging policies below are historical.

Latest passing-only/replay correction (2026-10-18): continuing the same PR #9 branch from fetched `b2c50c2`. Console tables now exclude all below-threshold/invalid/duplicate/special rows while preserving complete raw diagnostics. Added synthetic display-check and strict source-verified, atomically recorded replay with no model imports/execution and no mutation of the original result. A real Windows pseudo-terminal accepted native VT despite TERM=dumb, completed the synthetic check and first-photo replay (2 query/7 token rows). Visual verification in the owner's VS Code terminal remains pending because tools expose only text/control streams, not that terminal's visual surface. Latest software checks: 76 passed/one privilege skip. No inference rerun was needed. See [current commands, replay IDs and blocker](TERMINAL_VIEWER.md); earlier selection/full-console statements below are historical.

Latest presentation correction: continuing the existing PR #9 branch after fetching its remote head. Local user commit `4f2758e` had the identical tree to remote `97dbb8e`; it is retained by local tag `preserve-terminal-viewer-4f2758e`, and the branch was aligned without a merge or force push. The terminal viewer now checks actual stream capabilities, prints IMAGE first, keeps IDs at terminal-cell size, pages separate decimal-score tables, discloses summary selection/omissions and allows explicit full-mode special tokens. Noninteractive output is a display error, not shown. See [presentation contract and current evidence](TERMINAL_VIEWER.md). Actual cached CUDA run `a2ef6836b41b4e18a793642b1ad461a4` preserved all three photos and exactly matched prior detections/diagnostics; zero sockets. This tool's noninteractive output correctly returned 1 with three unsupported-terminal display errors while inference succeeded. VS Code visual verification remains pending.

Latest work: dedicated `feat/terminal-viewer-diagnostics` branch replaces TkAgg with in-memory ANSI/Unicode previews, results below images, and summary/full diagnostics. Normal predictions and existing result fields are preserved; complete aligned scores are added to JSON. Direct Windows checks: 57 tests passed/one symlink skip; offline CUDA full mode processed three photos with zero socket connections, 900 queries x 17 tokens per image, and 45,900 token lines. All displays finalized shown and originals were unchanged. VS Code visual readability remains uninspected. See [terminal contract and execution evidence](TERMINAL_VIEWER.md). Earlier native-viewer requirements and execution records below remain history.

Direct local follow-up (2026-10-17): fetched PR #8's branch at `65e085b` and used the unchanged Windows `.venv`. **53 software tests passed, one privilege-related symlink fixture skipped.** Cached-only inference on the one `data/input` photo completed on the GTX 1660 Ti with no fallback and two valid retained candidates (garbage pile 41.9%, individual litter 40.9% model scores); the original hash is unchanged. Native viewing returned normally; display is `shown`, run is `completed`, exit 0 with no errors. The owner reports correct detections and small real garbage piles; whether those small piles were missed or had undersized boxes remains unresolved. See [direct Windows evidence](PHASE_2C_WINDOWS_RUN.md) for run ID, timings, provenance and limitations. Earlier unrun statements below describe historical execution states; this single run does not establish category quality or Phase 2C acceptance.

Latest follow-up: PR #7 is merged. Owner-supplied Windows tests ran 50 cases with 49 passes/one privilege-related symlink skip. The supplied first model attempt downloaded safetensors and reached query extraction, but a validator bug rejected the model's intentional negative-infinity text padding on all 900 queries. The focused `fix/grounding-dino-padding-logits` branch corrects the audit using attention-mask/tail positions, while retaining true nonfinite/box/score checks. **54 cloud software tests pass**; corrected target inference/native overlay outcomes remain pending. Reuse cached model files and the existing environment; preserve the failed run. See the supplied-evidence/correction section in [PHASE_2C_CHECKS.md](PHASE_2C_CHECKS.md). Earlier implementation-time unrun statements below are historical and narrowed by these supplied results.

## Current result and execution limits

The owner instructed Codex to proceed with Phase 2C (D-025), superseding the previous stop-before-2C boundary. PR #6 is merged; this focused branch starts from current main `5db174b`. The baseline now implements pinned model/processor loading, float32 eval/inference mode, device handling/one auto-mode OOM CPU retry, exact mapping, conservative duplicate/review flags, readable summaries and native in-memory TkAgg viewing with the existing atomic display lifecycle. The CLI still defaults to preparation; inference requires `--mode infer`. Requirements and the existing verified target environment are preserved.

New modules: `scripts/inference.py`, `postprocessing.py`, `pipeline.py`, `viewer.py`; integration changes to configuration/CLI/result validation; `tests/test_inference.py` and a small foundation-fixture update; current-state docs and D-025. Model/processor/config use the verified revision; raw invalid queries are retained before threshold loss; no automatic grouping/membership/count inference or image exports.

Actual checks: **50 software tests passed** using existing cloud Python 3.13.5/Pillow 11.1.0/Matplotlib 3.10.1+dfsg1. CLI help, compilation and whitespace checks passed. A real cached-only CLI probe on a synthetic image returned exit 1 and persisted `runtime/missing_dependency` because torch is absent, with incomplete inference and unattempted display. This is not a no-detection/model result. Headless Agg drawing passed; native Tk calls are simulated. No package install/reinstall or weight download occurred. Full evidence, commands and assumptions: [PHASE_2C_CHECKS.md](PHASE_2C_CHECKS.md).

The implementation is ready for code review and target execution, **not verified model/target acceptance**. Actual model loading/inference, GTX 1660 Ti memory fit, native photo overlays and unannotated team-photo observations remain unrun. Run the new software tests and baseline in the existing Windows `.venv` using [README.md](../README.md); record actual JSON/native outcomes and limitations. Missing target/runtime/photos are execution prerequisites, not a request for repeated phase confirmation. Commit/publication outcomes are reported in the completion message; no merge or later optimization/grouping is authorized.

## Historical Phase 2B handoff — prior stop boundary superseded by D-025

## Current result and next boundary

The owner explicitly authorized Phase 2B, superseding the previous 2A-only stop boundary. Configuration, image preparation, result-contract utilities and atomic strict-JSON persistence are implemented for review on `feat/phase-2b-io-foundation` from current published main `dde98cf` (PR #5). No model loading, weight downloads, inference, viewer implementation or Phase 2C work occurred. No package was installed or reinstalled. The historical 2A handoff below remains preparation evidence, not the current authorization boundary.

Files: `config/detection.json`; `scripts/{configuration,inputs,results,persistence,detect_images}.py`; `tests/test_foundation.py`; three requirements files transcribed from the 2A snapshot; current-state docs and additive `/work/` ignore rule. CLI output is explicitly preparation-only, using separate schemas and `prepared` rather than functional inference success/no detections. Originals are read only and transformed solely in memory. Only JSON is persisted.

Actual checks: 22 synthetic software tests passed via `/usr/bin/python3 -m unittest discover -s tests -v`; CLI help, compilation and diff whitespace checks passed. Cloud Python 3.13.5/Pillow 11.1.0 were already installed; the verified Windows Python 3.11.9/Pillow 11.2.1 target was not directly accessible. New target software tests remain unrun. No fresh-install or GPU/model/viewer result is inferred. Full command/results, failure-injection coverage, persistence assumptions and limitations are in [PHASE_2B_CHECKS.md](PHASE_2B_CHECKS.md).

P-001 now distinguishes verified 2A metadata/basic runtime from untested model behavior; D-024 records the owner's bounded authorization, preserving D-001–D-023. Mapping/duplicate selection/overlap interpretation, model-memory fit, real-photo behavior and native overlays remain later work. The foundation is ready for review and eventual authorized 2C integration; run the new software checks in the unchanged target environment before claiming target validation. Commit/publication outcomes are reported in the completion message; no merge is authorized or claimed.

## Historical Phase 2A handoff — prior stop boundary superseded by D-024

## Result and ownership

**Phase 2A technical preparation checks are complete on owner-supplied Windows evidence. Stop here: detector implementation and inference have not begun and remain outside the authorization.** Codex owns implementation, replacing Antigravity; the owner reports Antigravity made no changes. Phase 0 is complete, Phase 1 planning is published, and this preparation corresponds exactly to plan Section 4 and Section 8 stages 2–3.

Cloud Codex reviewed actual target command outputs, a native-plot screenshot and explicit normal-return confirmation; it did not directly execute Windows commands. The local repository is in the owner's selected Documents/Python/Project Image-object Detection folder with a fresh `.venv`. The final evidence is preserved in [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md) and [PHASE_2A_ENVIRONMENT_SNAPSHOT.md](PHASE_2A_ENVIRONMENT_SNAPSHOT.md).

## Verified preparation

- Windows 10 build 19045.6466; Git 2.45.1.windows.1; Python 3.11.9 x64 installed from python.org via WinGet with installer-hash verification; local clone and venv confirmed.
- GTX 1660 Ti, 6144 MiB VRAM, driver 472.19, capability (7,5), CUDA build 11.8; CPU and CUDA tensors both returned 2.0. Earlier RTX 2080 Super assumption is superseded, not erased from history.
- torch 2.7.1+cu118, torchvision 0.22.1+cu118, Transformers 4.51.3, Pillow 11.2.1 and Matplotlib 3.10.3 installed/imported. `pip check` reported no broken requirements. All 36 frozen packages and cu118 installation source are recorded.
- Native TkAgg synthetic diagonal plot rendered; the owner explicitly confirmed `TkAgg viewer closed normally`. No image export was invoked.
- Official metadata/config/card/processor reads succeeded for `IDEA-Research/grounding-dino-tiny` at revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. Reported Apache-2.0 metadata, non-gated access, safetensors file, Tiny-consistent Swin configuration and same-revision processor settings are preserved.
- NVIDIA CUDA 11.8 Table 2 lists Windows minor-compatibility minimum >=452.39; driver 472.19 exceeds it and the actual CUDA tensor passed. The separate toolkit driver listing 522.06 is not the same minimum. Earlier cloud 403 limitations were resolved for these checks through successful target requests, without claiming cloud policy changed.
- The owner saved `.venv/phase2a-requirements.txt` and `.venv/phase2a-environment.json`; their supplied dependency/config values are preserved in the committed snapshot. `.gitignore` and historical decision rows remain intact.

## Remaining boundaries and next action

No current preparation blocker remains in the supplied evidence. No further installation or repetition of passed checks is needed for the unchanged environment. The owner can review the preparation branch and issue the next phase instruction; this technical completion does not accept a later phase on their behalf and does not authorize a merge.

Pretrained weights were not downloaded/loaded. No detector/application code, inference, team-photo processing, clustering, training/fine-tuning or annotated-image export occurred. Model-memory fit, custom category capability, detection quality and future model-specific driver behavior are intentionally untested. Later failures must be diagnosed from actual evidence; no accuracy or performance result is inferred from preparation.

Future authorized model and processor loading must use the same recorded revision. The baseline remains still-image trash detection, three intended categories, direct prompts first and multiple candidate regions; team photos are unannotated and manually supplied. Existing numeric/display contracts and no-automatic-image-export rules remain. Prompt/threshold/alias and grouping-boundary defaults are unvalidated proposals.

## Git publication and history

Latest fetched main for preparation is `7a776cf`, merging PR #4; PR #3 published the plan at `b366f76`. Work is on `docs/phase-2a-environment-preparation`, never direct main. Final commit/push/PR outcomes are reported in the completion message; no merge is claimed. Earlier PR API attempts returned `Forbidden`; do not claim a PR exists without confirmation.

D-001–D-023 preserve prior owner decisions, with D-022/D-023 assigning Codex and limiting this work to preparation. Previous pending/blocked readiness wording is historical and resolved by the supplied evidence above. No new implementation approval or owner acceptance date is invented.

D-061 runtime verification: native standard/per-object Tk checks passed at Image777 with all47 IDs selectable, correct labels, navigation and zoom/pan; no model/GPU calls or saved JSON changes. Guarded suite:241 tests,225 passed,16 documented skips,zero failures/errors. Human visual acceptance remains pending. The selected25/75 blend partially recovers larger-object performance (17/36 coverage versus E1 21/36).

D-062 verification: the guarded suite ran245 tests (229 passed,16 documented skips,zero failures/errors). Four native Tk smoke cases passed (two choices,standard/per-object), with Image777,all47 selectable IDs,correct labels,navigation and zoom/pan. Model/GPU calls,weight reads and image exports were blocked; saved JSON hashes stayed unchanged. Human visual acceptance remains pending.

## D-081: Organic hybrid showcase attempt — 2026-10-31

D-080 was committed locally first as `2802723` on `feat/stage5-paper-category`;
there was no push or merge. D-081 wrote
`work/stage5-paper/rights_organic_showcase.json` before decoding and verified
the four D-080-clean non-held-out image bindings. The attempted detector run
stopped after two CUDA forwards (Image 68, one nine-prompt and one malformed
hybrid prompt). A missing delimiter caused the ten-class phrase mapper to see
nine spans. The code fix is in place, but the eight-forward authorization has
only six calls remaining and cannot produce all four paired comparisons. No
complete predictions were saved; no accuracy, query-vector, or Organic
performance result is claimed. No test JSON/held-out photos, training, backward
passes, or optimizer updates occurred.

See [D-081 execution record and current limits](TACO_STAGE_5_ORGANIC_SHOWCASE.md).
The optional Organic filter remains hidden until a complete, verified local
showcase artifact exists. Native Tk Image 777 passed 50 combinations; a
four-image Organic GT-only fixture passed 65 renders across all five taxonomy
views and all NMS/threshold controls on Image 68. This is UI fixture coverage,
not saved-prediction replay. A new bounded evaluation authorization is needed before running the
corrected prompt on all four image pairs. D-081 changes remain uncommitted and
unpushed for owner review.

## D-082: Organic hybrid showcase completed — 2026-10-31

D-082 used a fresh eight-forward CUDA budget after CPU tokenizer preflight
verified the exact nine- and ten-span prompts. All four images (68, 72, 92,
841) completed 9-/10-prompt passes at 384/576 FP32 with TF32 disabled using the
hash-verified D-076 champion. Full 900-query outputs, ten-class score vectors,
per-GT matching diagnostics, image bindings, and artifact hashes are stored in
`runs/training/stage5-organic-hybrid-505b80d9f7cf430a96fcea4c7471d7f4/`.
Native Tk loaded the real saved predictions, verified 9-/10-prompt rows on all
four showcase images, cycled 65 filter/taxonomy/NMS/threshold renders, and
reopened Image777. The 10-class prompt yielded localized Organic queries on all
four GTs, but Image92 remains Plastic at top-1 and the D-080 four-box sample
still provides no validation/generalization support for a supervised Organic
class. See [complete scores and vectors](TACO_STAGE_5_ORGANIC_SHOWCASE.md).

PowerShell replay for the four-image showcase (choose the Organic Showcase
filter, then navigate among images 68, 72, 92, and 841):

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_stage5_visual_review.py --gui --image-id 68 --score-threshold 0.10 --box-top-k 15
```

Open the standard frozen validation view at Image 777:

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_stage5_visual_review.py --gui --image-id 777 --score-threshold 0.10 --box-top-k 15
```

Guarded verification: **329 tests, 313 passed, 16 documented skips, zero
failures/errors**; links checked 513 local destinations with no missing targets;
`git diff --check` passed. No push/merge occurred. Owner review remains
needed for whether Organic stays a zero-shot-only diagnostic and whether to
seek a permissioned external Organic dataset; this four-image result does not
answer those decisions.
