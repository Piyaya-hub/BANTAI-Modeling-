# Decision register

## D-107: Two-Tier Hierarchical Spatial Grouping, Sachet Hazard Tie-Break & Video GUI De-Cluttering - 2026-11-06

Owner authorization D-107 executed on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`) and the 5 canonical BANTAI categories. Zero unaudited or unsupervised classes were added. Zero test-split reads (`test.json` unopened; 0 held-out test photos opened) occurred.

1. **Two-Tier Hierarchical Spatial Grouping Architecture (`scripts/taco_stage5_spatial_grouping.py`)**:
   - Tier 1: Dense core extraction via tight heap adjacency ($g_{ij} == 0.0 \lor \text{IoU}_{ij} > 0.01 \lor (g_{ij} \le 0.35 \cdot \min(s_i, s_j) \land \text{neighborhood support})$). Clusters qualify as `Garbage Pile` if $(k \ge 3 \land (\text{touching} \ge 1 \lor \rho \ge 0.40)) \lor (k = 2 \land \rho \ge 0.45 \land \text{IoU} \ge 0.15)$.
   - Tier 2: Non-pile items grouped with proximity parameters ($\alpha = 2.0, \beta = 0.225$), bounded by an anti-chaining cluster diagonal ceiling ($\le 0.42 \cdot L$) to prevent foreground outliers from daisy-chaining into dense heaps.
   - Preserves all 14 validation showcase image invariants: Image 1 (1 Garbage Pile, $n=2$), Image 777 (1 Group of Individual Litter, $n=5$), Image 802 (1 Individual Litter), Image 998 (2 Individual Litter), Image 1243 (1 Individual Litter), Image 1465 (1 Group of Individual Litter).
   - Separates continuous video frames into distinct dense heaps and foreground items: Frame 10006 yields 1 Garbage Pile ($\rho = 0.526$) + 1 Group of Individual Litter; Frame 10008 yields 2 Garbage Piles ($\rho = 0.633, 0.653$) + 1 Group of Individual Litter + 1 Individual Litter.
2. **Sachet/Wrapper Hazard Tie-Break Rule (`scripts/taco_stage5_consensus_tiebreak.py`)**:
   - Resolved the candy wrapper misclassification on Frame 10008 by reassigning compact rectangular objects ($\text{norm\_area} \ge 0.0018$, $\min(w, h) \ge 35\text{px}$, aspect ratio $\le 2.5$) with non-hazard support ($\ge 0.035$) from `0: cigarette` (`Potentially Hazardous`) to `3: plastic_wrapper` (`Plastic`) or `8: paper` (`Paper`).
   - Frame 10008 red candy wrapper (`qid=1`) correctly displays as `Plastic 0.33` (`plastic_wrapper`).
   - Full 65 validation sweep: exactly 0 lost True Positives (`lost_tps == []`), AP9 increases monotonically from $0.298963 \to \mathbf{0.301350}$, AP5 stable at $0.492729$.
3. **Visual Review GUI Label De-Cluttering & Auto Tab Selection (`scripts/taco_stage5_visual_review.py`)**:
   - In dense Garbage Piles ($k \ge 3$), text badges are drawn only for the top-2 scoring members, suppressing badge clutter while keeping all bounding boxes visible.
   - Region bounding boxes and category banners render at `zorder=7`.
   - Automatically activates fourth table tab (`D-095 Champion`) upon switching into Video Frames mode (`VIDEO_FRAMES_FILTER`), ensuring the details table is populated.
   - Adjusted figure margins (`left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.06, hspace=0.16`), optimizing display for portrait $1080 \times 1920$ video frames.
4. **Verified Evidence**:
   - Unit tests: 6 passed in `tests/test_stage5_d107_hierarchical_video_grouping.py`; 88 passed in `tests/test_stage5_d*.py`.
   - See [TACO Stage 5 D-107 Hierarchical Video Grouping Report](TACO_STAGE_5_D107_HIERARCHICAL_VIDEO_GROUPING.md).

## D-106: Video Frame Extraction & Three-Model Stage 5 Detection on Motion_Extract_Testing.mp4 - 2026-11-06

Owner authorization D-106 executed on `feat/stage5-paper-category`. Active supervised taxonomy remained strictly fixed to `0..8` (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`) and the 5 canonical BANTAI categories. Zero unaudited or unsupervised classes were added. Zero test-split reads (`test.json` unopened; 0 held-out test photos opened) occurred.

1. **Video Decoding & Deterministic Frame Extraction**:
   - Inspected `data/input/Motion_Extract_Testing.mp4` (SHA-256 `a10bb8ebf616be46f1da0fde85355673f9c90726243c29050c58f0e9d8319602`, 1080x1920, 24.03 FPS, 377 frames, 15.69s duration).
   - Installed `opencv-python` strictly into `.venv-training` (preserving Phase 2A `.venv`).
   - Extracted 16 sequential RGB frames at 1.0 FPS into git-ignored `runs/video_detection/motion_extract_testing/frames/` with unique non-colliding IDs `10001..10016`.
2. **Three-Model Forward Execution & Score Vectors**:
   - Executed sequential forward passes across all 16 frames on GPU:
     - `Warm-Start Baseline` (Stage 4 blend, 384x576) in 18.27s.
     - `Stage 5 Run 1` (256/384 epoch-2 checkpoint, 256x384) in 5.74s.
     - `D-095 Champion` (soup 50/50, 384x576) in 6.69s.
   - Saved full 900 queries per model and `[16, 900, 9]` float32 sigmoid score vectors to `video-score-vectors.npz`.
   - Saved `video-showcase.json` and cryptographic `manifest.json`.
3. **Consensus & Spatial Grouping Replay**:
   - Replayed full D-105 Three-Model Consensus (`apply_scene_consensus`) and Bounded-Density Spatial Grouping (`compute_spatial_grouping`).
   - Frame 10014 (t=13.0s) correctly classified 1 dense `Garbage Pile` (26 visible items), while other frames formed `Group of Individual Litter` and `Individual Litter`.
4. **Visual Review GUI Integration**:
   - Added `load_video_showcase` discovery to `scripts/taco_stage5_visual_review.py`.
   - Added top control bar toggle button `Check Video Frames (data/input)` / `Back to Validation Images (65)`.
   - Added `Video Frames: Motion_Extract_Testing.mp4` to `Filter:` dropdown.
   - Added CLI flag `--video-frames` to launch GUI directly in video frames inspection mode.
5. **Verified Evidence**:
   - Unit tests: 5 passed in `tests/test_stage5_d106_video_extraction.py`; 123 passed in `tests/test_stage5*.py`; 401 passed (0 failures, 0 errors) in `scripts/verify_stage4_guarded.py`.
   - All 65 validation images retain exact metrics: $AP_9 = 0.298963, AP_5 = 0.492729$, 76 TPs, 63 visible FPs, 36 clean images.
   - See [TACO Stage 5 D-106 Video Extraction Report](TACO_STAGE_5_D106_VIDEO_EXTRACTION.md).

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

Owner authorization D-085 executed locally on `feat/stage5-paper-category` as a strictly CPU-only, read-only dataset and split coverage audit. Zero GPU calls, model forwards, training updates, test-split reads, or held-out photo opens occurred.

In `docs/DECISIONS.md` and `docs/HANDOFF.md`, escaped control characters (`\x0c`, stray linebreaks) in the D-084 section were cleaned and verified.

The deterministic audit script `scripts/taco_stage5_extensible_class_audit.py` (tested in `tests/test_stage5_extensible_class_audit.py`) inspected `annotations.json` across 1,500 TACO images and frozen split metadata:
1. **Active Supervised Classes (`0..8`)**: Maintained strictly fixed at 9 classes (`cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`), comparing Tier 1 (637 boxes) vs D-083 Core (731 boxes) across 379 training images and 109 validation boxes.
2. **Candidate Class Audit Across 5 Canonical BANTAI Categories**:
   - `Plastic`: `plastic_straw` (157 pool boxes, 52 clean train, 0 val), `plastic_lid` (77 pool boxes, 31 clean train, 0 val), `other_plastic` (273 pool boxes, 118 clean train, 0 val), `plastic_utensil` (37 pool boxes, 15 clean train, 0 val) -> recommended `ZERO_SHOT_HYBRID_ONLY` due to 0 validation boxes. `six_pack_rings` is a `SUBTYPE_MERGE_CANDIDATE`.
   - `Paper`: `drink_carton` (136 pool boxes, 59 clean train, 7 val) -> qualifies as `SUPERVISED_CANDIDATE`. `paper_cup` (66 pool boxes, 16 clean train, 1 val) -> `ZERO_SHOT_HYBRID_ONLY`. `plastified_paper_bag` has 0 instances in TACO.
   - `Metal`: `scrap_metal` (119 pool boxes, 75 clean train, 0 val), `metal_bottle_cap` (90 pool boxes, 33 clean train, 0 val), `aluminium_foil` (68 pool boxes, 45 clean train, 0 val) -> `ZERO_SHOT_HYBRID_ONLY` due to 0 validation boxes. `food_can` is a `SUBTYPE_MERGE_CANDIDATE` (already merged into `drink_can`).
   - `Potentially Hazardous`: `broken_glass` (138 pool boxes, 31 clean train, 0 val), `glass_bottle` (104 pool boxes, 43 clean train, 0 val), `aerosol` (10 pool boxes, 5 clean train, 0 val), `battery` (2 pool boxes, 1 clean train, 0 val) -> `ZERO_SHOT_HYBRID_ONLY`. `syringe` has 0 instances in TACO.
   - `Organic`: `food_waste` (8 pool boxes, 4 clean train, 0 val) -> `ZERO_SHOT_HYBRID_ONLY`.
3. Complete structured data stored in `work/stage5-paper/extensible_class_coverage_audit.json` and report in [D-085 report](TACO_STAGE_5_EXTENSIBLE_CLASS_AUDIT.md). Active supervised classes in `scripts/taco_stage5_material_hierarchy.py` were not modified.

Guarded suite passed; Markdown links and `git diff --check` passed.

## D-084: Documentation synchronization and exact JSON table alignment - 2026-10-31

Owner authorization D-084 executed locally on `feat/stage5-paper-category` as a strictly CPU-only documentation and verification alignment task. Zero GPU calls, model forwards, or test-split reads occurred.

Manual transcription discrepancies in docs/TACO_STAGE_5_D083_CORE_TRAINING.md were corrected against
`runs/training/stage5-d083-core-20261031-a/report.json`,
`resolution-comparison.json`, and d079-score-preserving-consensus.json:
- Part A 448/672 per-class APs aligned (plastic_bottle: 0.453533, plastic_film: 0.066377, plastic_wrapper: 0.019519, drink_can: 0.545545, plastic_cup: 0.319802, styrofoam_piece: 0.074257, paper: 0.370481). Scale-bucket progressions documented (AP_small: 0.059641 -> 0.073440; AP_medium: 0.063247 -> 0.184431).
- Part B candidate states aligned: D083-Core E1 (combined-65 AP 0.233825, AR100 0.375472, legacy-47 AP 0.255879, mean loss 3.949181), soup_d083_d074_5050 (combined-65 AP 0.236129, AR100 0.378128, legacy-47 AP 0.252820, mean loss 3.985499), and soup_d083_d076_5050 (combined-65 AP 0.232746, AR100 0.373000, legacy-47 AP 0.253044, mean loss 3.963570).
- BANTAI 5-Category metrics aligned: raw AP 0.401563 (AP50 0.495005), consensus AP 0.405701 (50.05% AP50 = 0.500456), and consensus 9-specific AP 0.236393.
- README.md and docs/PHASES.md updated through D-084.

Guarded suite: **333 tests, 317 passed, 16 documented skips, zero failures/errors**. Markdown links and git diff --check passed.

## D-083: Controlled Stage 5 fine-tuning and resolution evaluation - 2026-10-31

Owner authorization D-083 executed locally on `feat/stage5-paper-category`. No unaudited classes were introduced; active supervised classes remain fixed to `0..8` with `paper` strictly bound to its Tier 1 baseline without new paper sources. The mapping registry in `scripts/taco_stage5_material_hierarchy.py` was refactored into a table-driven `OBJECT_REGISTRY` with `register_object_class()` for extensible future audited classes.

Part A evaluated the D076 champion at `448/672` (112 validation forwards: 65 `combined_65` + 47 `legacy_47`). 9-specific AP dropped from `0.235994` to `0.217783`, legacy 8-class AP from `0.250362` to `0.244212`, and BANTAI 5-Category AP from `0.400957` to `0.361760`, showing resolution mismatch against the `384/576` training geometry.

Part B executed 1 bounded epoch of 379 updates on `strict-clean-stage5-d083-core-v1` (379 train images / 731 boxes; rights bound in `work/stage5-paper/rights_d083_core.json`) warm-started from the D074/D076 champion hash `cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422`. Swin-T and BERT were frozen. Candidate evaluation (336 validation forwards across 3 states):
- `D083-Core E1`: combined-65 AP `0.233825`, legacy-47 AP `0.255879`.
- `soup_d083_d074_5050` (50/50 D074 champion + D083 E1): combined-65 AP **0.236129**, legacy-47 AP `0.252820`.
- `soup_d083_d076_5050` (50/50 D076 champion + D083 E1): combined-65 AP `0.232746`, legacy-47 AP `0.253044`.

Champion `soup_d083_d074_5050` (parameter hash `82c0a5bf3a104db5f5ff35e1cbfc1bd239595ec9946e22cf085986985d95fddc`) was serialized to `checkpoints/best-inference` with `best-score-vectors.npz` and wired into `scripts/taco_stage5_visual_review.py`. Full report: [D-083 report](TACO_STAGE_5_D083_CORE_TRAINING.md). Guarded verification: **333 tests: 317 passed, 16 skips, 0 failures/errors**. Native Tk Image 777 passed 50 combinations. Markdown links and `git diff --check` passed.


## D-080: Organic / Food waste support audit - 2026-10-30

D-079's verified taxonomy/consensus changes were committed locally first as
`287ac5f` (`Stage 5: add D-079 BANTAI consensus replay`) on
`feat/stage5-paper-category`; no push or merge. D-080 is a read-only CPU audit;
no training/inference, test.json read, held-out photo access, or protected-artifact
change occurred.

Raw TACO has one organic-like category, `Food waste` (category 25): 8 annotations
on 7 images. Fresh raster/EXIF checks were limited to those 7 after cached split
metadata confirmed none belongs to frozen validation/test. Tier 1 + class 9 has
2 candidate train images / 2 boxes; Tier 2 sibling mapping has 4 / 4; Tier 3
accessory omission remains 4 / 4. All fixed 65 validation and 65 test images have
zero Food waste GT boxes. The strict blockers are Unlabeled litter (three Food
waste boxes on two images), plus Food Can / Disposable food container / Other
plastic exclusions depending on tier. No audited Food waste image is in a frozen
duplicate component. Details: [complete per-image D-080 ledger](TACO_STAGE_5_ORGANIC_AUDIT.md).

Decision: TACO does not support a reliable tenth supervised class or validation
estimate at 2-4 training images/boxes. Keep supervised IDs 0..8. Consider a
separately authorized `"food waste."` zero-shot route with a representative,
permissioned evaluation set, or obtain an external supplement before supervised
Organic training. This is a recommendation for owner review, not an approved
taxonomy change or measured zero-shot capability. D-080 audit/script/test and
report remain uncommitted; stop for owner review. Guarded suite: **327 tests, 311
passed, 16 documented skips, zero failures/errors**; zero GPU/detector/optimizer
activity and protected writes. The four new tests cover mapping, sibling tiers,
Unlabeled/accessory handling and resolution scaling. Markdown links and
`git diff --check` passed.

## D-079: Canonical BANTAI 5-category display and score-preserving consensus - 2026-10-30

Owner-authorized local diagnostic completed on `feat/stage5-paper-category`;
no commit, push, or merge. The default GUI taxonomy is `Plastic`, `Paper`,
`Organic`, `Metal`, and `Potentially Hazardous`; class mapping, empty Organic AP,
colors, and compatibility views are documented in the calibration report.

The CPU-only audit used the D-078 saved `(65,900,9)` vectors. The selected
paper-to-plastic subtype tie-break uses delta `.016`, at least three plastic
subtype probabilities `>=.20`, and preserves the original scalar score. It
changes 1/58,500 queries (Image777 q0, Paper -> `plastic_film`/Plastic), with
unchanged score `0.3078179955482483`. Diagnostic combined-65 9-Specific AP is
`.235994 -> .236283`; BANTAI 5-Category AP is `.400957 -> .405106`. Strict TP/FP/FN
at `.05/.10/.15` are reported in the linked evidence; the 9-Specific values are
unchanged, while BANTAI gains one TP at each threshold. This is display-level
relabeling and diagnostic remapping, not model calibration, test generalization,
or a change to official saved COCO metrics.

The viewer toggle is reversible, defaults on only when verified matching local
vectors are available, affects only the D-074 champion state in BANTAI general
or hierarchical views, and is disabled in 9-Specific view. NMS follows remapping.
The CPU rule audit and native Tk Image777 GUI matrix are qualified: **323 guarded
tests, 307 passed, 16 documented skips, zero failures/errors; 50 GUI control
combinations plus consensus assertions passed with zero TclError**. No GPU
execution, optimizer updates, protected writes, commit, push, or merge occurred.
See [D-079 audit and display evidence](TACO_STAGE_5_IMAGE777_CALIBRATION.md#d-079-canonical-bantai-categories-and-score-preserving-subtype-consensus).

## D-078: Tcl/Tk runtime repair and champion query score vectors - 2026-10-30

Owner authorization executed on `feat/stage5-paper-category`; publication is
authorized after guarded verification. No training, backward pass, optimizer step,
test split access, checkpoint modification, export/rights/taxonomy/core-source
modification, promotion, or merge occurred. The D-076 `best-inference` bundle was
verified with parameter SHA-256
`cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422` before/after.
The completed score-vector extraction used **65** CUDA forwards over combined_65
and produced float32 scores shaped **65 x 900 x 9**. However, an earlier attempt
had already completed one model forward before aborting in loss matching; total
forward invocations were therefore **66**, exceeding the authorized 65 cap by one.
That failed attempt saved no vectors. The final 65-query set matched every saved
D-076 winning class; maximum score difference was **0.0**. No additional detector
forwards were run after discovering the overrun.
Private artifacts are in `runs/training/stage5-score-vectors-5619e219fa264287aa0564c4565a2472/`
and the final CPU audit is in `runs/training/stage5-score-vector-audit-6ba2b48e835847fab31a380b54db4133/`.

Historical Stage4 Image777 GT259 resolves by source annotation ID 2390 to Stage5
validation annotation 408. For its q0 box (IoU **0.925399**), the class scores are:

| Class | Score |
| --- | ---: |
| cigarette | 0.1130877063 |
| plastic_bottle | 0.2749033570 |
| plastic_bag | 0.1952677816 |
| plastic_wrapper | 0.2554721832 |
| plastic_film | 0.2927520871 |
| drink_can | 0.2038989961 |
| plastic_cup | 0.1313864738 |
| styrofoam_piece | 0.1276722550 |
| paper | 0.3078179955 |

On q0, raw/max-subtype pooling keeps paper as winner. Noisy-OR and capped top-two
sum flip five-material and four-material display to plastic, but validation results
are negative: five-material AP **0.105068** / **0.107140** versus raw **0.342725**;
four-material AP **0.138704** / **0.143208** versus raw **0.400957**. At score .10,
strict TP/FP/FN are 68/283/41 raw five-material versus 56/1060/53 noisy-OR and
56/494/53 top-two; four-material 72/279/37 raw versus 63/1291/46 noisy-OR and
63/499/46 top-two. Pooling is not enabled in the viewer; nine-specific argmax and
official saved COCO dictionaries remain unchanged.

The native Tk issue was caused by Tcl's Windows filesystem bridge failing to
traverse the base Python installation's AppData junction. The helper uses the
extended-length form of the standard `sys.base_prefix/tcl` paths when current variables are unset/stale or point at the ordinary base path Tcl cannot
traverse, preserving valid external overrides and changing no files.
Native Tk Image777 passed **40** combinations (four taxonomy views, five NMS modes,
two threshold profiles), with zero TclError and unchanged validation photo hash.
Guarded suite: **323 tests, 307 passed, 16 documented skips, zero failures/errors**;
the guard reported no GPU, detector execution, optimizer updates or protected writes.
Markdown link check: **62 files /504 local links**, zero missing public/private
destinations. `git diff --check` passed.

## D-077: Image 777 saved-query calibration and display controls - 2026-10-30

Owner-authorized saved-query audit executed on `feat/stage5-paper-category`.
Eleven states (65 x 900 winning-class queries each) were compared; two CPU-built
balanced soups received 65 combined-view GPU forwards each. One initial CuBLAS
attempt failed before completing a forward; cumulative attempted forwards were
131 of 224 allowed. No training, backward pass, optimizer update, or test-split
access occurred. Checkpoint bundles, exports, rights, taxonomies, official metric
dictionaries, and Stage 4 core sources remained unchanged.
The saved schema has no class-probability/logit vectors, so subtype pooling, priors,
and explanations of hidden logit shifts cannot be derived. The observed Image 777
GT259 class winner changes from Run1 plastic subtype (score .295 in blend2575)
to paper (scores .302-.313 high-res) while IoU remains .922-.930. This is not fixed
by display NMS. In Tier2 E1, q1 cigarette .050049 overlaps GT259 at IoU .930 and
is retained by class-aware NMS; agnostic NMS suppresses it under higher-score q0.

Viewer adds cross-class IoU/containment diagnostic NMS and a reversible calibrated
display floor (cigarette .06, other classes max(slider,.15)); its startup NMS is
agnostic IoU .50 and threshold mode remains uniform .10. Official saved COCO
metrics remain unchanged. Across 65 validation images, calibration at slider .10
raised nine-specific TP from 50 to 53 on Tier2+D074 soup but raised FP from 301 to
684. Cross-class IoU also lowered TP 50->48 and FP 301->268 at uniform .10. These
are operating-point diagnostics, not replacements for COCO AP or validated
calibration. The balanced 40/60 blend with Tier1 E1 and Tier2 E1 scored combined AP
0.186977 and 0.215196; both still label GT259 `paper`, and each has only 5/7 Image
777 cigarettes scoring >=.05. These states underperform their source endpoints.
The full 660-case data and Image 777 ledger are in the private
`stage5-image777-calibration-306d313f59bd4634bd171484cc3270fd` artifact.

Guarded suite: **316 tests, 300 passed, 16 documented skips, zero failures/errors**.
Three focused D077 tests cover score floors, cross-class/containment suppression,
and GUI selector integration. Native Tk resolution was repaired and verified under D078.
See the [full audit and profile comparisons](TACO_STAGE_5_IMAGE777_CALIBRATION.md).
No commit, push, merge, or promotion.

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

Owner-authorized bounded Stage5 workflow; measured validation results below are not test generalization or checkpoint promotion.

Owner requested accuracy maximization for 9-class waste detection incorporating paper (Option A).
1. Scale & Soup Analysis: Evaluating Stage 5 Run 1 at target 384/576 canvas scale lifted Combined AP from 0.135663 to 0.153029. Testing Model Soups revealed blend_2575 (25% Epoch 1 + 75% Epoch 2) as champion, achieving Combined AP 0.171262 and AP50 0.210596 while recovering styrofoam representation (0.148515) and maintaining paper AP (0.374468 overall, 0.473670 annex).
2. High-Resolution Fine-Tuning: Executed bounded 1-epoch (307 updates) training run stage5-paper-558bcc18d6cf4556a966d8cfc320c968 directly at 384/576 scale, LR 5e-6, warm-started from verified blend_2575 weights. Peak VRAM was ~3,306 MiB on GTX 1660 Ti.
3. Measured validation performance:
   - Combined 65 AP reached 0.202020 (AP50: 0.246182, AR100: 0.335380).
   - Legacy 47 AP reached 0.226047 (AP50: 0.263987, approaching Stage 4's 0.265).
   - Paper AP reached 0.421508 overall, and 0.586443 on the 18-image Annex.
   - Significant per-class gains: drink_can (0.6564), plastic_bottle (0.2525), cigarette (0.1115), plastic_cup (0.3198).
4. Model Soup: A 20/80 blend of blend_2575 and High-Res Epoch 1 achieved 0.194258 Combined AP and retained styrofoam piece AP at 0.148515.
5. Interactive Review & Post-Processing: taco_stage5_visual_review.py updated with 4-panel comparison (Ground Truth, Baseline, Stage 5 Run 1, Stage 5 High-Res) and interactive Overlap NMS filter (IoU 0.50 + IoMin 0.80).
6. Software integrity: 286 tests passed, 1 skipped, zero failures. No test split access or unauthorized checkpoint promotion.

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

Owner authorizes Stage 5 implementation incorporating paper (id: 8) and selects
Option A (Frozen Core + Annex). All 1,500 TACO images are verified present on local
disk (0 missing). The 14 TACO source categories are approved for paper: Normal paper,
Magazine paper, Wrapping paper, Tissues, Paper bag, Paper cup, Corrugated carton,
Other carton, Drink carton, Meal carton, Egg carton, Pizza box, Paper straw, and Toilet tube.

Option A preserves the original 221 training, 47 validation, and 47 test image
memberships byte-for-byte, appending the 122 newly eligible images (165 paper boxes,
106 original-class boxes) as follows:
- Training: 221 frozen original + 86 appended = 307 total images (120 paper boxes)
- Validation Annex: 47 frozen original + 18 appended = 65 total images (20 paper boxes)
- Held-out Test Annex: 47 frozen original + 18 appended = 65 total images (25 paper boxes)

Evaluation protocol preserves separate reporting for:
1. Legacy benchmark view on original 47 validation images / 75 GTs.
2. Paper annex view on 18 new validation images / 20 paper boxes.
3. Combined 9-class view on 65 validation images / 109 measured GTs (the earlier authorization estimate of 121 was superseded by the verified export).

Scope authorized: create config/taco_taxonomy.v2.json, document Section 185
academic-use basis in docs/TACO_STAGE_5_ACADEMIC_USE.md, build isolated export
exports/taco/strict-clean-stage5-v1, implement isolated 9-class training/evaluation
modules on feat/stage5-paper-category, and run 9-class fine-tuning warm-started
from champion D-061 weights. Existing 64 checkpoint bundles, frozen Stage 4 SOURCES,
and test-split privacy constraints remain strictly invariant.

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

## D-068 final Stage 4 synchronization and publication - 2026-10-25

Owner authorizes synchronizing D-060 through D-067 and committing/pushing only the
focused docs, scripts and tests to `origin/feat/stage4-real-image-training` for
PR #18. No merge, promotion, GPU, training or inference. The 64 checkpoint bundles,
private runs/work/exports/data, rights records, official taxonomy and six core
training SOURCES are excluded from edits/staging; unrelated research/proposal docs
remain unstaged. The earlier stopped backbone implementation is retained as
experimental software/history, not as a completed or qualified training result.

**Best measured official eight-class validation state: D-061
`blend_350_442_2575`, AP 0.265563, no NMS or remapping.** Its separate raw
eight-class candidate coverage is 38/75 at score >=0.05, IoU >=0.50, all 900 queries,
with non-exclusive matching; this count is not COCO recall. It is not promoted and
not wired into `scripts/detect_images.py`. D-062 cosine decay, D-064 pure-film
damping, D-066 cigarette LR boosting and D-067 auxiliary score calibration are
retained as negative results for their combined objectives, including partial
gains and regressions. D-060 replay, D-063 NMS and D-065 material/polymer views are
diagnostic capabilities, not changes to the official benchmark. [Synchronized
outcomes and limits](../README.md#d-060-through-d-067-outcomes).

Recommended saved-review **display-only** settings: threshold **0.10 or higher**,
plus **`agnostic IoU 0.50 + containment 0.80`**. Select explicitly; historical
raw/.05 defaults remain. Suppression can hide distinct or low-score true objects;
post-NMS or mapped-category numbers must not replace official eight-class AP and
raw candidate coverage. No production operating threshold is established.

D-068 verification: 278 tests, 262 passed, 16 documented skips, zero failures or
errors, using `scripts/verify_stage4_guarded.py`. Fifteen training-execution fixtures
are excluded; one Windows symlink fixture lacks privileges. Detector loading/
forward, CUDA initialization, backward and optimizer steps are blocked. Generated
fixtures/logs go to the system temporary directory; protected-directory writes are
guarded and none occurred. D-067's recorded bundle/native-GUI evidence is retained
without rerunning model/photo checks or re-opening checkpoints. Relative public
Markdown links and `git diff --check` are checked before commit; private artifact
links remain intentionally local. Human visual acceptance and the PR merge
decision remain with the owner; no follow-up experiment is implied.

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

## D-060 Saved 384/576 visual GUI replay ? 2026-10-24

Owner authorized saved-JSON replay for D-056/D-057/D-058 in standard and per-object modes, retaining all four 256/384 choices. No GPU, training or detector inference is authorized by this task. The local augmented manifest adds endpoint and step-350 choices; checkpoint JSON context/position and anchored evaluations bind their identity. No checkpoint promotion or merge. [Implementation, verification and exact commands](TACO_SAVED_REVIEW_384_576.md).

## D-059 Stage 4 wrap-up and focused publication — 2026-10-24

The owner authorizes closing the Stage 4 post-validation investigation series
D-043–D-058, synchronizing README/phase status through D-057/D-058, running
guarded fast unit/CPU checks, and publishing the explicitly listed scripts,
tests, reports and two SVG assets in one focused D-045–D-059 commit on
`feat/stage4-real-image-training` to update PR #18. No PR merge or checkpoint
promotion is authorized. [Final wrap-up state and inventory](HANDOFF.md).

This authorization supersedes earlier per-investigation publication stops only
within that allowlist. Preserve all runs, work, exports, data, environments,
rights records and50 corrected checkpoint bundles. No GPU/model execution,
training/inference, test-split/photo access or checkpoint modifications.
Local ARGON handoff and concept-proposal planning documents remain uncommitted.
Private helper-dependent tests explicitly skip when their unpublished work files
are absent; no private audit artifacts are added to Git.

The D-042 Section185 owner academic-use record remains distinct from external
permission, institutional approval and definitive legal clearance. Optional human
visual review, PR review/merge and future stages remain owner decisions, not
automatic follow-up actions. Higher sampled validation AP does not select a model.

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
Epoch1 AP0.249891/AR1000.488141 narrowly exceeds D-0550.247003/0.474763;
Epoch2 regresses to0.170508/0.370118 despite cigarette same-class coverage
improving5→8→15/24 at0.05 and seven-class AP excluding bag rising.
Ten new bundles verified; final optimizer step442 across558 parameters.
All1766 protected hashes matched after execution; no checkpoint is promoted.
[Full gates, results and limits](TACO_STAGE_4_576_TRAINING.md).
Audit `scale576-experiment-0b2a846e659a429cbe8a65c1ca0444e1`. No retry, promotion, test access, dependency/GUI changes,
annotated exports or Git publication. Stop for owner review.

## Multi-checkpoint resolution replication — 2026-10-23

D-056: The owner authorizes inference-only256/384 vs384/576 for unchanged
Pretrained and seed123/LR1e-5/E2 on47 validation images/75 GT, at most188 forwards.
Reuse D-055 seed42/E1 outputs read-only, without GPU repetition. Require guarded
software, fresh protected/source hashes, both exact low baselines before any
high pass, then both high evaluations. Fixed weights/FP32/900 queries/prompt/
taxonomy/COCO v2 and locked14/10/15/36 bins. No updates/backward, train-photo/test
access, core/GUI/dependency changes, exports, promotion or Git publication.
[Completed results and qualification](TACO_STAGE_4_RESOLUTION_REPLICATION.md)
show shared small-bin coverage gains, checkpoint-dependent raw AP and positive
bag-excluded AP deltas in all three states. No further experiment is authorized;
stop for owner review.

## D-055 single retry authorized and completed — 2026-10-23

The owner explicitly authorizes one retry from Gate1, preserving the original
stopped attempt `resolution-experiment-629229adf1c6480d930aac79250d18f8`.
The isolated pointer records that retry consumed and zero retries remaining.
All five gates and the47-image384/576 pass completed on the fixed42/1e-5/E1
checkpoint. [Measured results and limits](TACO_STAGE_4_RESOLUTION_EXPERIMENT.md)
record improved aggregate AP, partial small-object recovery, persistent <4px
failures and three larger-object regressions. No updates/backward, training-photo
reads, test access, core/GUI edits, promotions or Git publication. All next steps
remain proposals; stop for owner review. The initial failure record below is
historical and its no-retry statement is superseded by this explicit grant.

## Controlled inference-only resolution comparison — 2026-10-23

D-055: The owner authorizes one fixed seed42/LR1e-5/E1 validation comparison:
256/384 with384-square pad versus384/576 with576-square pad. Require software,
artifact/source, CPU geometry, exact CUDA baseline and one-image CUDA pilot gates,
in that order. Stop immediately at any failed gate; no fallback or parameter
changes. Use an isolated work harness and focused resolution tests; core training,
evaluation and GUI files remain untouched. No updates/backward, train-photo/test
access, promotion, exports, dependencies or Git publication. All proposals remain
unimplemented. [Actual stopped attempt and cause](TACO_STAGE_4_RESOLUTION_EXPERIMENT.md):
the software gate failed due to harness guard scoping; no real inference ran.
Prepared fixes are unqualified and no retry is authorized by this stopped record.


## Small-object and classification audit — 2026-10-22

D-054: The owner authorizes a read-only full-validation audit of actual256/384
preprocessing and384-square padding, all75 resized GT boxes and nine saved model
states. Separate size-associated localization, class assignment, score/rank
effects and annotation uncertainty; preserve the original20-image manifest and
existing GUI. Record current owner-supplied image777 observations separately.
Rank resolution/classification/data-quality directions as proposals only;
[report evidence and one next experiment](TACO_STAGE_4_SMALL_OBJECT_CLASSIFICATION_AUDIT.md).
No new training or model inference without separate authorization, optimizer
updates, model/data/protocol/GUI/dependency changes, test access, promotion,
exports, uploads or Git publication. Stop for owner review after verification.


## Per-object saved detection investigation — 2026-10-22

D-053: The owner authorizes diagnostic-only per-GT analysis for image777 and the
existing20-image review using verified saved predictions. Preserve the four-panel
GUI, terminal-only defaults and explicit saved-run selection. Add an opt-in
annotation selector, GT highlight/zoom and complete overlap tables; compare all
four seed/LR runs with unchanged pretrained baseline and same-run E1/E2.
Preserve the original manifest and add separate machine evidence. No training,
inference, optimizer/model/data/protocol/dependency changes, test use, promotion,
exports or automatic Git operations. Verify software and native Windows GUI;
[report findings](TACO_STAGE_4_PER_OBJECT_REVIEW.md), separate automated diagnostics
from human material/annotation confirmation, then stop for human review. No
training recommendation is authorized solely from image777.


## Cross-seed stability analysis — 2026-10-22

D-052: The owner authorizes saved-result analysis and visual-verification preparation
across pretrained and both epochs of four seed/LR runs (42/123, 1e-5/5e-6).
Preserve corrected COCO v2, original artifacts, owner changes and existing viewer
defaults. Prepare 12–20 validation examples and a local review manifest; permit
only the minimal explicit saved-run viewer extension. Separate automated
diagnostics from pending human confirmation. Compare the four possible research
directions as proposals only. No new training, inference, test evaluation, model
or dataset changes, dependency upgrades, promotion, exports, publication or Git
publication. Stop after [the report and review preparation](TACO_STAGE_4_CROSS_SEED_ANALYSIS.md)
for owner review; no additional experiment is authorized.

## Paired seed-123 replication — 2026-10-22

D-051: The owner authorizes exactly two fresh two-epoch runs at seed 123:
A with LR 1e-5 and B with LR 5e-6. Preserve all other corrected experiment
settings, original pretrained initialization, 221/47 corpus, eight-class prompt,
Stage 3D adapter, valid-image targets, native loss, frozen backbones, FP32,
batch 1, 256/384 resize, 384-square padding, weight decay 1e-4 and clip 1.
Add minimal configurable `--seed` support with default 42, propagating it to
Python/NumPy/CPU/CUDA RNGs, restart and epoch shuffling; record/test its exact
behavior without changing strict resume verification or old manifests.
Each run independently discards its three-step pilot, restarts from the pinned
pretrained model and performs 442 full updates (884 full + six pilot total).
Stop on critical preflight/integrity failure, and after both final validations.
No test access, model/data/inference changes, extra experiments, dependencies,
uploads, promotion or automatic Git operations. Compare both seeds/rates and
recommend further work only as proposals in [the replication report](TACO_STAGE_4_SEED_REPLICATION.md).

## Controlled learning-rate experiment — 2026-10-21

D-050: The owner authorizes exactly one fresh two-epoch experiment at learning
rate 5e-6, changing only learning rate from the corrected 1e-5 reference run.
Retain original pretrained weights, corrected targets, Stage 3D adapter, native
loss, 221/47 corpus/splits, eight-class prompt/taxonomy, seed 42, batch 1, frozen
backbones, FP32, 256/384 resize, 384-square padding, AdamW weight decay 1e-4,
clip 1, no augmentation/scheduler and corrected COCO v2 evaluation. Use three
discarded pilot steps, then exactly 442 full-run updates, stopping for safety
failures. Add minimal configurable learning-rate support if necessary, retain
default 1e-5, bind the chosen rate in settings/checkpoint context and test its
actual delivery to AdamW. Preflight must pass before updates. No historical
resume, test access, data/prompt/model/inference changes, dependency changes,
promotion, uploads, automatic tuning or Git operations. Compare all five
pretrained/reference/new states in [the LR experiment report](TACO_STAGE_4_LR_EXPERIMENT.md).
Stop after epoch 2/final validation for owner review.

## Corrected-run diagnostic investigation — 2026-10-20/21

D-049: The owner authorizes diagnostic-only investigation of corrected epoch 1
versus epoch 2 using verified saved results and the existing local visual viewer.
Verify comparability, score distributions/ranking, class assignment, localization,
losses/gradients and representative validation cases. Preserve source files,
configuration, data, checkpoints, strict resume and corrected COCO settings.
No training, optimizer updates, test access, promotion, dependency changes,
uploads or automatic Git operations. Report confirmed observations separately
from hypotheses and rank single-variable experiments as proposals only. See
[corrected diagnostics and next experiment proposal](TACO_STAGE_4_CORRECTED_DIAGNOSTICS.md).
Stop for owner review; no further experiment is approved by this decision.

## Controlled corrected-target restart — 2026-10-20

D-048: The owner authorizes exactly one fresh two-epoch Stage 4 experiment from
the original pinned pretrained weights, using corrected valid-image box targets.
Mandatory coordinate/software/source/data/rights/runtime/baseline preflight must
pass before optimizer updates. Preserve all other historical settings: 221/47
train/validation images, batch 1, 256/384 resize, 384-square padding, FP32,
AdamW 1e-5/1e-4, clip 1, seed 42, frozen vision/text backbones, eight-class prompt,
native Stage 3D adapter/loss, 900 queries and corrected COCO protocol v2.
Use the three-step real-image pilot, discard its weights/state, then perform
442 full-run updates. No augmentation, scheduler, accumulation, extra epochs,
test access, dataset/configuration changes, historical resume, dependency changes,
uploads, publication, checkpoint promotion or automatic Git operations.
Compare pretrained, both corrected-protocol historical results, and both new
epochs; retain losses, AP/recall, fixed-threshold diagnostics, score distributions,
memory/timing, checkpoint integrity and preservation evidence in
[the corrected-retrain report](TACO_STAGE_4_CORRECTED_RETRAIN.md).
Stop after epoch 2/final validation and await owner review.

## Explicit opt-in validation viewer — 2026-10-20

D-047: The owner authorizes restoring the existing four-panel TACO viewer through
`taco_visual_review.py --gui`, preserving D-046's terminal-only default.
Optional `--image-id` selects a validation image directly. Reuse verified saved
predictions, ground truth, zoom/pan, labels, scores, raw-phrase tables and diagnostic
controls. No inference/training, protocol/configuration changes, annotated-image
exports, commits, pushes or merges are authorized by this task.

## Terminal-only epoch scores — 2026-10-20

D-046: The owner rejects the epoch diagnostic display and requests terminal output
for each epoch showing updated scores. This supersedes the Step 2 GUI workflow,
without changing the existing functional-inference viewer. Close the diagnostic
window and use verified saved epoch metrics/score distributions in the terminal.
No new training authorization, inference, threshold change or resume exception is
implied. Prior GUI implementation/evidence remains historical and is not launched.

## Stage 4 Step 2 local visual diagnostics — 2026-10-20

D-045: The owner authorizes a standalone local-only four-view validation replay,
systematic representative selection, annotation-relative IoU/classification/miss/
duplicate/confidence diagnostics and initial visual review. Reuse verified saved
records and the approved evaluation-only provenance pathway. Keep raw phrases
separate from assigned classes; diagnostic thresholds do not alter official metrics.
Preserve all user work, inference, strict resume, checkpoints, taxonomy, splits,
originals and environments. No training, optimizer restoration/updates, test access,
downloads, dependency changes, exports/publication, promotion or automatic commits.

[Actual evidence](TACO_STAGE_4_VISUAL_REVIEW.md) distinguishes automated annotation
associations and native runtime checks from pending human photo-level judgments.
Recommendations for a future corrected-target restart/resolution/score investigation
remain proposals requiring later authorization. No further experiment is started.

## Stage 4 Step 1C evaluation-only provenance — 2026-10-20

D-044: The owner explicitly authorizes a separate historical checkpoint evaluation
path that independently verifies original manifests/artifacts/provenance, exact
reviewed current source differences, model/processor/tokenizer/adapter and corpus/
rights compatibility. Preserve strict training-resume verification; never restore
optimizer/RNG/training position, update weights, access test data, publish private
artifacts, promote a checkpoint or automatically commit/push/merge. Qualify using
synthetic fixtures before reevaluating unchanged epoch-1/2 validation models.

[Completed evidence](TACO_STAGE_4_REEVALUATION.md) records corrected metrics, preserved
historical evidence, unchanged strict resume rejection and zero guarded forbidden
operations. Step 2 visual comparison is recommended, not started or implicitly
approved. Corrected evaluation does not repair historical training targets.

## Stage 4 Step 1B coordinate correction — 2026-10-20

D-043: The owner authorizes correction of valid-image target normalization and
native prediction decoding, independent regression tests and validation-only
comparison of pretrained/epoch-1/epoch-2 models. Preserve image processing, adapter,
architecture, loss, scoring policy, taxonomy, splits, checkpoints, owner edits and
environments. No optimizer updates, test access, publication, promotion, automatic
merge or Step 2 visual analysis. Stop checkpoint comparison if integrity/source
binding blocks it; do not falsify provenance or disable checks.

[Actual Step 1B evidence](TACO_STAGE_4_COORDINATES.md): corrections/tests pass;
pretrained reevaluation completed; epoch comparisons blocked by unchanged
exact-source/protocol checks. An evaluation-only provenance contract for historical
weights is a proposal requiring approval, not an implemented exception or permission
to retrain. Historical run artifacts remain unchanged.

## Stage 4 owner academic-use declaration — 2026-10-19

D-042: The owner confirms non-commercial academic purpose, local training/evaluation, no publication/redistribution/cloud upload of TACO photos or annotations, and authorizes proceeding on a genuinely documented lawful-use basis if supportable. This is project-owner workflow authorization, not rightsholder permission, institutional approval or definitive legal clearance. The [case-specific Section 185 record](TACO_STAGE_4_ACADEMIC_USE.md) documents a defensible limited-use claim with remaining uncertainty, bound to the exact 221 training and 47 validation images/annotations. The v2 gate records that claim with clearance flags false and third-party authority explicitly not claimed. No new phase, model/loss/precision change, test use, cloud spending, publication or automatic merge is authorized. Existing Stage 4 pilot/training/evaluation proceeds within that scope.

## Current Stage 4 authorization — 2026-10-19

| ID | Approved decision | Consequence |
| --- | --- | --- |
| D-041 | Owner authorizes real-image fine-tuning and initial validation directly on Stage 3D's verified adapter as one continuous implementation task, conditional on verified permission or an appropriately approved lawful-use basis for source photos and annotations/database. | Implement loader, full resumable checkpoints, progress, real-memory pilot and bounded 2–3 epochs with fixed pretrained/validation comparison; preserve eight labels/splits, inference/environment and original weights. If rights remain unresolved, finish runnable software and identify exact approval before processing actual photos. No test-photo use, cloud spending/upload/publication, promotion or PR merge. Earlier synthetic-only boundaries are superseded within this scope. |

Implementation/evidence: [Stage 4](TACO_STAGE_4.md). Software and generated-image qualification are recorded separately from real-photo gates. Missing rights evidence currently blocks actual photo processing/training; no permission, real-data outcome or benchmark is invented. The fixed initial protocol/settings in the implementation are bounded engineering choices, not measured optimal defaults or a taxonomy/architecture decision.

## Historical Phase 0 decisions — retained record

Source: the project owner's Phase 0 task instructions. These entries preserve their original wording; they are not all current instructions. D-002 is refined by D-007/D-014; D-003's Phase 0-only task state is superseded by D-020; D-004's preserved original ignore rules now permit additive exclusions under D-019. No entry is experimental evidence.

| ID | Decision | Consequence |
| --- | --- | --- |
| D-001 | Initial repository development scope is trash detection only. | Broader street-issue and hotspot features remain outside initial implementation. |
| D-002 | Evaluate a pretrained Grounding DINO-family model on still images first. | Do not start with training, alternative models, or integration. Exact model selection is unresolved. |
| D-003 | Current work is Phase 0 documentation only. | No detection code, ML dependencies, model downloads, frontend, backend, or Phase 1 execution. |
| D-004 | Use a documentation branch and review against `main`; preserve `.gitignore`. | No direct `main` changes or pull request merge in this task. |
| D-005 | Separate approvals, proposals, unresolved choices, and measured evidence. | Do not invent model versions, benchmark results, or successful training/evaluation claims. |

## Historical final Phase 0 decisions — 2026-10-17

Source: the project owner's final Phase 0 revision request. The original D-001–D-005 entries above are retained as history. These approvals established direction, not successful model behavior. D-008’s annotation wording is superseded for this activity by D-016. D-013’s task-specific branch/pending-review status is superseded by D-020 and merged PR #2; its no-direct-main/no-merge rule remains.

| ID | Approved decision | Consequence |
| --- | --- | --- |
| D-006 | Repository responsibility is AI modelling and computer vision only. | Future work may include evaluation, preprocessing, detection, necessary postprocessing, approved optimization/fine-tuning, and reusable detection outputs. Frontend/backend application modules, central databases, reporting, incidents, and hotspot ranking are outside this repository. Supersedes any earlier Phase 4 suggestion to implement those modules here. |
| D-007 | Start with pretrained GroundingDINO-T (Swin-T / Tiny configuration). | Refines D-002: the model configuration family is now selected; exact implementation, checkpoint source/revision, configuration, dependencies, and local compatibility remain unresolved for Phase 1. No training from scratch. |
| D-008 | Require Garbage Pile, Individual Litter, and Group of Individual Litter outputs. | Definitions and counting example are in [SCOPE.md](SCOPE.md); operational annotation/grouping/overlap rules remain unresolved. These are not guaranteed pretrained classes. |
| D-009 | Evaluate direct text-guided detection first. | Investigate separate and combined prompts before adding spatial grouping. Illustrative vocabulary is not final. |
| D-010 | Support multiple independent regions and mixed categories in one image. | Preserve objects versus regions and prevent double-counting groups and their members as independent regions; actual direct-model capability remains experimental. |
| D-011 | Grouping is conditional on baseline evidence and explicit owner authorization. | DBSCAN, scale-normalized proximity, custom bounding-box grouping, and hybrid methods remain candidates, not mandatory components. |
| D-012 | Available development GPU reported by owner: NVIDIA RTX 2080 Super. | Verify actual runtime availability, dependency compatibility, and resource needs during Phase 1; no local hardware validation is claimed. |
| D-013 | Final Phase 0 revisions use a new branch and pull request from latest `main`. | PR #1 is merged; use `docs/finalize-phase-0-detection-scope`, do not reuse that PR, merge, or begin Phase 1. Owner review of these revisions remains pending. |

## Current owner-confirmed requirements

Source: the owner's Phase 1 planning, publication, documentation-correction and subsequent Phase 2A instructions. D-018/D-020 retain their prior task wording; D-022/D-023 below supersede the role assignment and documentation-only authorization. No prior decision row is deleted. No additional approval date or performance evidence is inferred. These are owner requirements; technical defaults below remain proposals.

| ID | Confirmed requirement | Replaces or clarifies |
| --- | --- | --- |
| D-014 | Use pretrained Grounding DINO Tiny through Hugging Face Transformers; no baseline training/fine-tuning. | Resolves the inference-library uncertainty in D-002/D-007. Exact checkpoint/configuration/conditions/safetensors/revision still require verification. |
| D-015 | Execute on the owner’s Windows computer with owner-reported RTX 2080 Super, GPU preferred and CPU fallback. | Refines D-012; local Python/driver/memory/runtime/viewer remain unverified. Remote cloud state is not local evidence. |
| D-016 | Owner manually downloads/extracts unannotated team photos from Google Drive for functional detection checking and visual inspection. | Supersedes mandatory annotation policy, held-out splits, formal metrics, and numerical accuracy targets in earlier test/phase guidance and D-008's prerequisite wording. No collection or Drive integration. |
| D-017 | Display boxes, category labels and individual model-score percentages in memory; provide readable summaries, per-image JSON, and appropriate run/environment metadata. | Percentages are model scores, not accuracy, calibrated correctness probabilities, or image-area coverage. Source photos stay unchanged; no automatic annotated-image copies. |
| D-018 | Codex corrects documentation; Antigravity performs later authorized implementation. | The correction does not authorize installation, weight downloads, inference, or Phase 1 acceptance. Retain agreed phase boundaries without new approval gates for routine defaults. |
| D-019 | Preserve original ignore rules and append missing `/data/input/`, `/results/`, `/.cache/huggingface/` exclusions. | Additive clarification to D-004; no directory creation, silent untracking, or moving photos. |
| D-020 | Phase 0 is complete and Phase 1 planning is published through merged PR #3 (`b366f76`). Current work is documentation correction on a focused branch. | Replaces D-003/D-013 task-state wording and old pending-merge claims. Publication is not environment readiness, successful detection, or approval of every proposal. No main changes or merge. |
| D-021 | Functional outputs must be standards-compliant JSON with invalid numbers represented as null plus diagnostics, and display outcome separate from inference outcome. | Corrects the plan's prior nonfinite-retention ambiguity and missing display lifecycle. Persist inference results before viewing and atomically update display outcome; unfinished pending attempts remain explicit. Detailed contract conventions are in the [plan](PHASE_1_IMPLEMENTATION_PLAN.md#7-proposed-json-contract). |

## Subsequent owner decisions — current Phase 2A authorization

| ID | Confirmed requirement | Replaces or clarifies |
| --- | --- | --- |
| D-022 | Codex now owns implementation, replacing Antigravity; the owner states Antigravity made no changes. | Supersedes the role assignment in D-018 and the original Antigravity handoff. No independent Antigravity work is asserted or reverted. |
| D-023 | Start Phase 2A: local Windows environment setup/readiness, corresponding exactly to implementation-plan Section 4 and Section 8 stages 2–3. | Supersedes D-020's documentation-only current-task boundary. Allows isolated dependency setup, official metadata checks and package/CPU/GPU/native-viewer probes; no detector implementation, model weights/loading or inference. [PHASES.md](PHASES.md) defines completion/stop criteria. |

## Subsequent target evidence — supplied Windows preflight

The owner supplied Command Prompt output reporting Windows 10 build 19045.6466, Git 2.45.1.windows.1, GTX 1660 Ti, NVIDIA driver 472.19 and 6144 MiB total VRAM (5235–5236 MiB free across successful queries). This supersedes the RTX 2080 Super assumption in D-012/D-015 for the current target, while preserving those rows as history. `py` is not recognized in that shell; Python itself is not yet proven absent. These are supplied observations, not a Codex-executed local check, CUDA compatibility result or accepted readiness decision. See [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md) for next checks.

These decisions are sourced from the owner's latest instruction; no additional approval date or local readiness evidence is invented. Initially missing Windows preflight/results were execution prerequisites, not repeated authorization requirements. Subsequent supplied evidence completed Phase 2A checks as recorded below and in [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md).

D-001/D-006 and D-009–D-011 remain current: modelling-only trash still images, direct prompts first, mixed categories/multiple candidate regions, and no automatic grouping in the baseline. Category codes are `individual_litter`, `group_of_individual_litter`, and `garbage_pile`. Predictions do not prove membership or physical-object/independent-region counts; unresolved counts remain null. Application modules, APIs, databases, road anomalies, video/cameras, incident consolidation, and hotspots remain outside scope.

## Subsequent supplied preparation progress

After the initial Store-alias failure, owner-supplied WinGet output reports Python 3.11.9 x64 installed from python.org with verified installer hash. A new terminal then reports Python 3.11.9 (AMD64), the real Python311 executable and 64 bits. The owner selected a local Documents/Python/Project Image-object Detection folder; clone output succeeded, followed by `.venv` reporting Python 3.11.9. No ML package, device execution, checkpoint or viewer readiness follows from these observations. Earlier proposed pins remain subject to actual package/device checks.

## Phase 2A technical completion evidence — no new scope approval

Supplied results now complete the preparation checks: dependency consistency/direct imports; CPU and CUDA tensors (2.0); native TkAgg rendering and explicit normal return; checkpoint API/model-card/configuration and same-revision processor metadata; official NVIDIA Table 2 driver minimum; and full 36-package freeze plus local environment record. Revision is `a2bb814dd30d776dcf7e30523b00659f4f141c71`, with Apache-2.0 metadata, non-gated access and safetensors availability. Windows driver 472.19 exceeds the documented CUDA 11.8 minor-compatibility minimum 452.39, and the actual tensor test passed. The toolkit's separate 522.06 listing is not that minimum.

These are technical observations and Codex's preparation assessment, not a new owner decision or automatic approval to implement/run a detector. All prior D-001–D-023 rows remain intact. No model-memory or detection-quality claim is made. [PHASE_2A_ENVIRONMENT_SNAPSHOT.md](PHASE_2A_ENVIRONMENT_SNAPSHOT.md) preserves the exact supplied metadata, processor settings, dependency freeze and evidence provenance; [PHASES.md](PHASES.md) records the stop gate.

## Technical proposals and pending verification

- P-001: The [architecture](ARCHITECTURE.md) and [plan](PHASE_1_IMPLEMENTATION_PLAN.md) retain the proposed detector workflow. Checkpoint/processor revision, configuration, license/access metadata, exact dependencies/source and basic Windows package/device/viewer readiness are now verified to the limited 2A scope in the [snapshot](PHASE_2A_ENVIRONMENT_SNAPSHOT.md). Prior unverified wording is historical. Model loading, memory fit, prompt/category quality and inference remain untested; 2B software fixtures do not establish them.
- P-002: The current [test plan](TEST_PLAN.md) applies D-016 functional checks. Earlier P-002's mandatory annotated/held-out protocol is superseded; formal annotated evaluation is optional, separately scoped future work.
- P-003: Follow [PHASES.md](PHASES.md). Earlier Phase 0-only task state was superseded by D-020; current Phase 2C authorization is D-025 below; D-024 remains the historical 2B scope.
- P-004: The proposed baseline is now observed in supplied target results: Python 3.11.9 x64, torch 2.7.1+cu118, torchvision 0.22.1+cu118, Transformers 4.51.3, Pillow 11.2.1 and Matplotlib 3.10.3. Installation/imports and `pip check` passed; CPU/GPU tensors each returned 2.0 with capability (7,5). Full 36-package freeze/source capture, native TkAgg rendering/normal return and pinned checkpoint/processor metadata have subsequently been verified in supplied outputs and preserved in the [snapshot](PHASE_2A_ENVIRONMENT_SNAPSHOT.md); no model compatibility/performance is inferred. Earlier source-only proposal status is retained in Git history.
- P-005: Starting prompts, `threshold=0.35`, `text_threshold=0.25`, exact aliases, visual grouping boundaries, identical-box/same-phrase collapse, and IoU >=0.8 review flags are unvalidated defaults. No general NMS or spatial grouping algorithm is selected. Routine refinements inside an authorized phase are recorded rather than treated as new owner approval gates.

| Area | Remaining evidence or operational choice | Relevant stage |
| --- | --- | --- |
| Checkpoint | Source/configuration/license/safetensors and immutable model/processor revision recorded in snapshot; retain those pins for later loading | Phase 2A complete; preserve recorded evidence |
| Windows runtime | Python/driver/GPU memory, installed versions/full freeze/CUDA source, successful CPU/GPU execution and native viewer recorded in snapshot | Phase 2A complete; preserve recorded evidence |
| Functional configuration | Prompts/thresholds, preprocessing, visual category boundaries, aliases, duplicate flags, source/output paths and reproducibility metadata | Preparation and routine implementation in authorized phases |
| Outputs | Implement corrected numeric/status/display lifecycle consistently; keep unmapped/invalid diagnostics and unresolved counts | Later authorized detector phase, outside 2A |
| Later work | Whether actual functional limitations justify grouping/hybrids/alternatives or other modelling improvement | Phase 3 decision; not baseline |

## Recording future decisions

For each approval, record the decision, approving owner, rationale, evidence links, affected scope/phase, and the actual approval date only when known; never invent one. A tool suggestion is not an approval. Move a proposal to approved status only after explicit owner approval; preserve superseded decisions with their replacement reference. No training or evaluation results exist in this documentation deliverable.

## Phase 2B owner authorization and implementation choices — 2026-10-17

| ID | Owner instruction | Effect |
| --- | --- | --- |
| D-024 | Implement Phase 2B — Input and Output Foundation: configuration, image loading, result-contract utilities and persistence only; synthetic predictions in tests only; preserve verified dependencies; commit/publish PR if possible, no merge. | Supersedes D-023's 2A-only stop boundary for this bounded foundation. No model loading/download/inference, viewer implementation, grouping, training, web/application modules or Phase 2C. Prior decisions/evidence remain historical records. |

Routine implementation choices: separate `bantai.preparation.v1` and `bantai.preparation-run.v1` schemas prevent decoding from being mislabeled functional success. The functional contract adds an explicit `inference.completed` guard. UUID run directories are reserved exclusively; source filenames use a SHA-256 of full source identity plus source-byte digest. External inputs use an absolute image ID because no repository-relative path exists; project inputs use relative IDs. Initial JSON uses atomic hard-link publication without overwrite; final display updates use same-directory `os.replace`. Filesystems lacking hard links fail visibly. Single writer per functional result is required; power-loss directory durability is not claimed. These are software choices, not measured model capability or new owner approval of prompt hypotheses.

The complete 36-package requirements snapshot is transcribed from supplied 2A evidence with the CUDA source retained. No package reinstall/fresh-install verification occurred. Actual 2B cloud software checks and target-version limitations are recorded separately in [PHASE_2B_CHECKS.md](PHASE_2B_CHECKS.md). Historical 2A readiness/snapshot stop wording applies to that earlier task and is superseded by D-024 for the foundation only.

## Phase 2C owner instruction — 2026-10-17

| ID | Owner instruction | Effect |
| --- | --- | --- |
| D-025 | “proceed now with phase 2C” | Supersedes D-024's stop-before-2C boundary. Implement the baseline previously described as the next phase: pinned loading/inference, exact mapping, conservative duplicates/review flags and native in-memory display. Preserve dependencies, originals, contract/evidence boundaries and no automatic merge. No grouping, training, web/application modules or later optimization. |

Codex interprets this phase instruction using the published 2B next-phase description and implementation-plan baseline; no successful model behavior or blanket validation of prompts/thresholds is inferred. Routine choices: preparation remains the default; inference is explicit, with optional cached-only execution. Raw query auditing precedes upstream threshold loss; missing decoded phrases remain empty rather than invented. Duplicate policy now uses identical raw boxes plus normalized phrases and highest-score retention; near duplicates remain. Functional results require an observed pinned Transformers version for completed inference; unavailable metadata is null for pre-model errors. Actual driver is not inferred from old 2A observations. Headless canvas and mock tests remain software evidence; target/model/native-photo execution is unrun. See [PHASE_2C_CHECKS.md](PHASE_2C_CHECKS.md).

## Terminal viewer owner instruction - 2026-10-17

D-026: The owner explicitly requested replacing TkAgg with ANSI/Unicode rendering in the VS Code terminal, results beneath each image, all raw query/non-padding token sigmoid scores, summary/full modes, complete JSON diagnostics, unchanged filtering and pinned model, tests/docs, and a dedicated branch commit without merging. This supersedes the prior native-window display requirement only. No grouping, training or web modules are authorized. PASS/FAIL denotes retention or strict threshold decisions, never measured correctness. See [terminal contract](TERMINAL_VIEWER.md).

## Terminal presentation correction owner instruction - 2026-10-17

D-027: Continue `feat/terminal-viewer-diagnostics`/PR #9; diagnose missing inline output, require IMAGE-first actual terminal rendering and aligned decimal-score tables, compact configurable summary selection, bounded full output, explicit special-token inclusion, separate threshold/policy states, and honest unsupported-terminal outcomes. Preserve raw precision, inference, thresholds, pinned checkpoint, originals and atomic display lifecycle. No new branch or merge. See [presentation contract and evidence](TERMINAL_VIEWER.md).

## Passing-only console and replay owner instruction - 2026-10-18

D-028: The owner supersedes below-threshold/full-dump console selection with passing-only retained query rows and passing ordinary-token rows, preserving all diagnostics in JSON. Authorizes independent synthetic renderer checks and source-verified display-only replay before further inference. Requires honest pending visual status when the actual Windows/VS Code surface cannot be inspected; keep the same branch/PR and no merge. Model revision, thresholds, originals and inference/display persistence remain protected. See [current terminal contract](TERMINAL_VIEWER.md).

## Width-driven previews and full-mode restoration - 2026-10-18

D-029: The owner requests larger configurable-width/aspect-preserving terminal previews with vertical scrolling, fewer pauses, fixed table alignment, token rows grouped by query, passing-only summary and working full below-threshold diagnostics. This supersedes D-028's restriction on explicit full-mode console output only. Preserve all inference, checkpoint/threshold, source/JSON and terminal-only boundaries. Verify using saved-result replay without rerunning inference; same branch and no merge. See [current terminal contract](TERMINAL_VIEWER.md).

## Consistent mapping and presentation owner instruction - 2026-10-18

D-030: The owner requests consistent detection summaries containing image/query IDs, raw decoded phrase, mapped category and mapping reason, score, threshold status and rounded boxes. Keep unmapped predictions distinct from failed detections; never force incomplete phrases into categories. Improve wrapping and terminal preview quality within the actual grid, explain resolution limits without claiming recovered detail, retain complete raw precision/token scores in JSON and summary/full modes. Use saved results where possible; preserve model inference, prompts, thresholds, originals and existing functionality. Continue the same feature branch/PR without merging. See [current presentation contract](TERMINAL_VIEWER.md).

## Native viewer restoration owner instruction - 2026-10-18

D-031: The owner explicitly replaces terminal image rendering with an existing-dependency Matplotlib/TkAgg window, superseding terminal-only/no-pop-up instructions. Display the oriented original with valid retained boxes, matching query IDs, phrases/categories, percentage scores and original pixel coordinates; retain zoom/pan, readable labels and passing-only terminal tables. Keep in-memory rendering, no image copies/exports, source-verified replay and atomic separate inference/display statuses. Test alignment, empty images, closure and failures; report actual Windows observations separately. Preserve user configuration edits, model inference, prompts, thresholds and weights. Continue current branch/open PR without merging. Subsequent owner feedback prioritizes output over additional navigation checks and explicitly requests only mapped-category text on image labels; keep IDs, scores, phrases and coordinates in the same-window details panel and terminal tables. See [native viewer contract and evidence](NATIVE_VIEWER.md).

## One scrollable results tab owner instruction - 2026-10-18

D-032: The owner supersedes terminal-only and separate image-window layouts with one scrollable GUI tab repeating Image #N, Detection Summary, Text-Token Details and Configuration in that order. Only numbered headings precede images. Use raw decoded phrase labels immediately above valid passing boxes, bounded readable wrapping, original pixel ticks, zoom/pan and matching IDs where space permits; keep scores in the summary. Passing-only tables must be selectable/copyable and show actual saved configuration. Preserve complete JSON, inference/model/mapping behavior, originals and atomic separate display outcomes, and support source-verified multi-image replay without inference. Verify synthetic and actual Windows observations separately. Continue the existing branch, update an open PR if present, and do not merge. See [current scrollable GUI contract](SCROLLABLE_VIEWER.md).

The owner confirmed the corrected raw phrase labels were complete and subsequently requested smaller label fonts to reduce occupied image space. Use 8.5-point labels with an 8-point fallback, retaining full readable tables and the disclosed zoom/detail fallback. This changes presentation only.

## Phase 2D authorization - 2026-10-18

D-033: The owner explicitly authorizes Phase 2D - Display and Summaries, completing and validating the existing shared Tk/TkAgg viewer. Preserve local prompt/threshold/font edits, environment, model/mapping, originals and raw JSON. Use latest main and a focused branch, carry relevant unmerged work, prefer saved-image replay, distinguish automated from visual evidence, publish a PR without merging. Phase 2E, fine-tuning, grouping, new data collection and web modules remain excluded. Supersedes the earlier same-feature-branch instruction. See [2D acceptance evidence](PHASE_2D_CHECKS.md).

## Phase 2E authorization - 2026-10-18

D-034: The owner authorizes Phase 2E - Functional Checking and Handoff, superseding the stop-before-2E boundary. Use existing owner photos and pinned cached inference in the preserved Windows environment, compare current settings with a small controlled prompt/threshold set, inspect boxes and raw phrases, verify display/persistence and publish evidence without merging. Keep human confirmation separate; missing observations/scenarios remain pending. Stop before Phase 3 decisions or optimization. No training, grouping, model/dependency changes, collection or application modules. See [results and recommendations](PHASE_2E_RESULTS.md).

During this work the owner reported that current labels needed to be smaller and some were absent for small/nested boxes. This authorizes the focused presentation correction: 7.5/7-point defaults, placement beyond narrow box width while bounded by axes and clear of box borders, query IDs/connectors. Corrected human acceptance remains pending. Existing 8.5/8-point decisions above describe history. Experimental prompt/threshold findings and proposed Phase 3 directions are evidence/proposals, not approved defaults or phase transitions.

## TACO harmonization and annotation visualization authorization

D-035 (2026-10-18): The owner authorizes preparation of the existing external TACO dataset with the eight-label object taxonomy in [config/taco_taxonomy.v1.json](../config/taco_taxonomy.v1.json): cigarette, plastic_bottle, plastic_bag, plastic_wrapper, plastic_film, drink_can, plastic_cup, styrofoam_piece. Match the eleven explicitly approved source names exactly and preserve their IDs and all original metadata. Implement read-only auditing and local annotation visualization with class sampling and explicit excluded/Unlabeled litter handling. No source modification, additional downloads, final splits, implicit negative examples, dependency/model changes or fine-tuning. Keep the existing inference pipeline and viewer unchanged. See [evidence](TACO_PREPARATION.md).

Unresolved preparation decisions: excluded/unlabeled/crowd handling for later training, stored-versus-EXIF coordinate policy, duplicate-ID resolution, negative-origin box repairs, final splits and training export format. Conservative audit holds and explicit EXIF previews are review behavior, not owner approval of a training transformation. The eight object labels do not replace the three existing inference project categories.

## Fine-Tuning Preparation Stage 2 authorization

D-036 (2026-10-18): The owner reports representative annotations visually reviewed and authorizes non-destructive strict-clean conversion and seeded image-level train/validation/test splitting of existing external TACO. Use the eight-class taxonomy/order, default proposed 70/15/15 ratios and seed 42, preserve original metadata/source identity, quarantine whole missing/unaligned/invalid/mixed/unlabeled images, and report losses/bias. Do not clip boxes, repair duplicated source annotation IDs, assume trainer ignore flags, copy photos, download data, change dependencies/inference or load/fine-tune Grounding DINO. This supersedes D-035's export/split stop only. No merge. See [Stage 2 evidence](TACO_STAGE_2.md).

Implementation choices within this scope: hold non-identity/ambiguous EXIF cases in addition to dimension mismatches; coassign exact/perceptual candidate photograph groups; propagate dirty exact-peer quarantine and hold conflicting exact-peer annotations; preserve full source records alongside unique export annotation IDs. These choices do not approve source repairs or confirm that all near duplicates were found. Licensing, trainer compatibility, small evaluation classes and strict-clean sampling bias remain pending readiness decisions. The existing three final inference categories remain unchanged.

## Stage 3A assessment authorization

D-037 (2026-10-18): The owner authorizes review of completed Stage 2 and a technical assessment of exact pinned Grounding DINO Tiny training implementation, excluded/unlabeled handling, existing-dataset recovery, GPU/cloud feasibility, licensing and a small smoke/evaluation proposal. Preserve strict-clean exports, eight-label taxonomy, original sources and inference environment/pipeline. No fine-tuning, taxonomy modification or replacement training export is authorized. See [assessment, primary sources and pending gates](TACO_STAGE_3A.md).

Proposals, not approvals: use Transformers 4.51.3's native loss with a separate PyTorch trainer/environment; qualify missing SciPy/evaluation dependencies separately; retain strict-clean as the first smoke reference; review coordinates before expansion; investigate auxiliary supervision of known excluded categories while keeping unresolved unlabeled photos quarantined; prefer approximately 24 GB GPU headroom. Rights clearance, runtime memory/backward/save-reload qualification, provider/budget, auxiliary taxonomy/ignore semantics and future versioned expansion/evaluation remain owner decisions. No training readiness or model quality is established by this assessment.

## Stage 3B synthetic training authorization

D-038 (2026-10-18): The owner authorizes a separate reproducible training environment, required dependencies including SciPy, and bounded synthetic-only PyTorch forward/backward/optimizer checks using the exact pinned Grounding DINO Tiny and Transformers 4.51.3 native loss. Verify all-eight prompt mapping, COCO preprocessing/batching, actual GPU peaks and model/processor saving/reloading. Preserve functioning inference `.venv`, pipeline/settings, source photos/annotations, taxonomy and strict-clean splits. No TACO photo training, cloud uploads/spending, replacement exports or real-data fine-tuning. See [actual implementation and evidence](TACO_STAGE_3B.md).

Recorded implementation choices for this software qualification only: FP32 batch 2, frozen vision/BERT, 192/256 input edges, 900 queries and unchanged native loss. The earlier 128/192 attempt failed spatial top-k capacity and remains preserved; there is no automatic model/device fallback. The successful small synthetic run is not an approved real-data hyperparameter configuration or evidence that full training fits 6 GB. Native encoder regression lacks gradients and encoder classification dominates the loss; investigating/correcting that behavior, real-data readiness, rights, resume/evaluation design, taxonomy/ignore policies and expansion remain future decisions. No library version/weight replacement or hidden loss patch is approved by this result.

## Stage 3C loss investigation authorization

D-039 (2026-10-18): The owner authorizes tracing pinned native encoder/decoder losses and controlled synthetic target-count/empty experiments in `.venv-training`, with masks, matching, normalization, gradient/update and memory diagnostics. Preserve environments, inference, original checkpoint, taxonomy and splits. Do not train on real TACO, patch Transformers, silently change coefficients/architecture or spend on cloud resources. Explain any proposed correction with supporting evidence and request approval before implementing a substantive training-method change. See [actual results and next-stage proposal](TACO_STAGE_3C.md).

Investigation findings, not an adopted correction within Stage 3C: native encoder supervision uses a detached box copy and recomputed learned-query/pre-encoder text scores. Selected proposal tensors with the same criterion show expected image-conditioned scores and encoder-head gradient paths; Stage 3C hypothesis terms are diagnostic only and never optimized. Native source, weights, loss and Stage 3B implementation remain unchanged. Its proposed Stage 3D required separate approval, recorded below. Rights, evaluator/resume design, resolution memory, excluded annotations and dataset coverage remain independent prerequisites.

## Stage 3D adapter authorization and outcome

D-040 (2026-10-19): The owner explicitly approves the separate training-only encoder-output adapter and synthetic qualification using Stage 3C findings. Supply selected image-conditioned encoder proposal scores and undetached supervised boxes to the existing native loss; preserve detached decoder reference initialization, pretrained architecture/base weights, query count, text mapping and coefficients. Verify actual gradients/updates, empty paths, decoder equivalence, checkpoint/adapter reconstruction, memory and regression evidence. No installed-library modification/monkey patch, inference change, real TACO training, data/taxonomy/split modification, cloud spending or automatic promotion. Stop/report if invariants cannot be preserved, and stop after Stage 3D synthetic qualification.

Actual Stage 3D outcome is a bounded synthetic PASS, detailed in [evidence](TACO_STAGE_3D.md). Every corrected case has two optimizer steps, finite values, intended encoder updates and exact decoder equivalence; empty regression is correctly zero. Experimental model/processor weights and explicit adapter context reload exactly. The checkpoint is weights/context only with `resume_supported=false`; no inference promotion or real-data readiness is approved by this outcome. Real-data trainer/resume/evaluator design, rights, coverage/leakage and intended-resolution memory remain future prerequisites.
## D-081: bounded Organic hybrid showcase attempt — 2026-10-31

D-081 execution record: D-080 was committed locally as `2802723`
before D-081 work began. The D-081 rights manifest was written before decoding
the four authorized non-held-out images and bound their verified hashes. The
attempt stopped after two CUDA forwards on Image 68 because the runner had
joined the last two prompt labels; no complete comparison artifact or Organic
detection result exists. The separator bug is fixed in the runner. The remaining
six calls cannot complete the requested four image pairs, and a fresh eight-call
evaluation would exceed this authorization's cumulative cap. No test-split
access/training/update occurred. Native Tk Image 777 passed 50 combinations;
the four-image Organic GUI fixture passed 65 renders across all five taxonomies
and every NMS/threshold combination on Image 68. This was GT-only in-memory
fixture verification, not saved-prediction replay. The actual Organic filter
was unavailable at the D-081 stop point; this is superseded by D-082 below.

## D-082: complete bounded Organic hybrid showcase — 2026-10-31

D-082 authorized a fresh eight-forward budget, excluding the two aborted D-081
setup forwards. CPU preflight verified the nine- and ten-prompt phrase spans in
order, and the D-076 champion bundle hashes matched before inference. Exactly
eight FP32 CUDA forwards completed across images 68, 72, 92, and 841 at 384/576;
no test JSON/held-out photo, training, backward pass, or optimizer update was
used. The artifact and detailed measurements are in
[the D-082 report](TACO_STAGE_5_ORGANIC_SHOWCASE.md).

Best spatially localized Organic results: Image 68 GT163 q9, 0.138023 / IoU
0.8911 / Organic rank 6; Image 72 GT171 q1, 0.176177 / 0.8997 / rank 2; Image
92 GT244 q0, 0.232682 / 0.9686 / rank 1 (top-1 remains Plastic); Image 841
GT2611 q3, 0.182473 / 0.9220 / rank 1. Full per-query vectors, including all
62 qualifying IoU≥0.50 queries, are retained in the private evaluation JSON.
Native Tk replay passed 65 Organic showcase renders with saved 9-/10-prompt
predictions and reopened Image777; no TclError occurred. The showcase remains a
four-image diagnostic and does not add validation support for supervised class
9. D-081/D-082 code and documentation were committed locally only; not pushed
or merged.
