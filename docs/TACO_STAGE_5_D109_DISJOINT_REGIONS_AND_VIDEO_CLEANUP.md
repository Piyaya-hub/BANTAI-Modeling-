# Stage 5 D-109: Spatially Disjoint Pile/Group Region Consolidation, Wall/Pavement Hallucination Suppression & Clean Overlay Rendering

## 1. Overview & Objectives

Stage 5 D-109 addresses the visual clutter, region bounding box entanglement, and background false positives observed when reviewing dense video frames (`data/input/Motion_Extract_Testing.mp4`, e.g., Frame 10001 `t=00:00.0s`):
1. **Spatially Disjoint Region Consolidation (`scripts/taco_stage5_spatial_grouping.py`)**:
   - **Step 1A (Adjacent & Overlapping Pile Core Merging)**: After Tier 1 extraction, merges adjacent or overlapping pile components ($u > 0.02 \lor \text{IoMin} \ge 0.15 \lor \text{gap} \le \max(0.85 \cdot s_{\max}, 0.045 \cdot L)$).
   - **Step 1B (Pile-Enclosed & Bordering Item Absorption)**: Iteratively absorbs unassigned items overlapping a pile bbox ($\text{inter} / \text{area}_i \ge 0.25$) or bordering pile members within $\le \max(1.15 \cdot s_{\max}, 0.05 \cdot L)$.
   - **Step 1C (Non-Overlapping Regional Partitioning)**: Prevents Tier 2 `group_of_individual_litter` boxes from wrapping around or overlapping `garbage_pile` boxes ($\ge 0.20$ area overlap triggers direct absorption into the pile). Also merges overlapping Tier 2 groups ($\ge 0.20$ area overlap).
   - **Result**: Exactly **0 overlapping multi-region pairs** across all 16 video frames and the 65 validation images. On Frame 10001, the roadside heap forms **1 clean, unified `Garbage Pile` (16 items)** rather than 3 small nested piles inside 1 group.
2. **Elimination of Wall Banner & Pavement Crack Hallucinations (`scripts/taco_stage5_consensus_tiebreak.py` & `scripts/taco_diagnostic_nms.py`)**:
   - Removed D-108's noisy Rule 7C ($[0.075, 0.10)$ score boost) and restored containment NMS clutter protection in `scripts/taco_diagnostic_nms.py`.
   - **Rule 7B14A (Upper-Wall Sign/Banner Clutter Veto)**: In ground-dominated litter scenes ($\ge 5$ items at $sc \ge 0.18$ with median $cy \ge 0.40 \cdot H$), suppresses uncorroborated detections located high on the upper wall border ($by < 0.08 \cdot H \land (by+bh) < 0.22 \cdot H \land sc < 0.24$), eliminating the `Paper 0.12` / `Paper 0.23` wall banner false positives on Frames 10001 and 10002.
   - **Rule 7B14B (Macro-Heap Cigarette/Hazard Speckle Veto)**: In macro-waste scenes ($\ge 5$ non-hazard items at $sc \ge 0.18$), suppresses tiny isolated `0: cigarette` speckles ($norm\_area < 0.0015 \land sc < 0.25$) lacking high-confidence Baseline + Run 1 agreement, eliminating the `Potentially Hazardous 0.11 / 0.19 / 0.23` pavement crack false positives on Frame 10001.
3. **Clean Visual Review Overlay & Startup Notebook Tab Selection (`scripts/taco_stage5_visual_review.py`)**:
   - **Banner Stacking Elimination**: Regional envelopes track placed banner rectangles and shift any colliding banners to prevent text overlap.
   - **Clean Member Badging**: In multi-item regions ($item\_count \ge 3$), individual text badges are suppressed while keeping all color-coded dashed bounding boxes visible, eliminating text collisions while retaining the bottom table for granular inspection.
   - **Startup Tab Selection**: On `--video-frames` launch, automatically selects the 4th tab (`D-095 Champion`) to prevent a blank startup table, defaulting to `score_threshold = 0.15` and `top_k = 25`.

---

## 2. Strict Scope & Taxonomy Guardrails

- **Active Supervised Taxonomy**: Strictly locked to the **9 active supervised classes** (`0: cigarette`, `1: plastic_bottle`, `2: plastic_bag`, `3: plastic_wrapper`, `4: plastic_film`, `5: drink_can`, `6: plastic_cup`, `7: styrofoam_piece`, `8: paper`) and the **5 canonical BANTAI categories** (`Plastic`, `Paper`, `Organic`, `Metal`, `Potentially Hazardous`). Zero unsupervised classes (`battery`, `aerosol`, `broken_glass`, `syringe`, `food_waste`) were introduced.
- **Regional Group Taxonomy**: Strictly layers the 3 spatial region categories from [docs/SCOPE.md](SCOPE.md):
  - `individual_litter` -> `"Individual Litter"`
  - `group_of_individual_litter` -> `"Group of Individual Litter"`
  - `garbage_pile` -> `"Garbage Pile"`
- **Score Integrity**: Individual litter regions retain the exact model score of their underlying detection (`model_score = float(score)`). Multi-object derived groups (`item_count >= 2`) have `model_score = None`, never inventing artificial confidence scores.
- **Zero Test-Split Access**: Held-out test split `test.json` remains completely untouched (zero test images opened).
- **Zero Hardcoded Identifiers**: All rules are formulated purely on physical box geometry (normalized area, aspect ratio, min dimensions) and model probability distributions. No image or frame IDs are hardcoded.

---

## 3. Full 65 Validation Evaluation Sweep

| Metric | D-107 Baseline | D-109 Clean Disjoint | Status |
| :--- | :---: | :---: | :---: |
| **Full 65 AP9** | $0.301350$ | **$0.301350$** | Benchmark high preserved |
| **BANTAI 5-Category AP** | $0.492729$ | **$0.497161$** | **$+0.004432$ Gain** |
| **`paper` AP** | $0.640610$ | **$0.640610$** | Maintained |
| **True Positives** | $76$ | **$77$** | **$+1$ TP** |
| **Lost True Positives** | $0$ | **$0$** (`lost_tps == []`) | Invariant preserved |
| **Total False Positives** | $63$ | **$63$** | Maintained |
| **Clean Images ($\text{FP}=0$)** | $36 / 65$ | **$36 / 65$** | Maintained |
| **All 14 Validation Showcase** | $14 / 14$ pass | **$14 / 14$ pass** | Invariant preserved |

---

## 4. Video Detection Verification (16-Frame Sequence)

- **Frame 10001 (`frame_0001_t00.0s.jpg`)**:
  - $0$ upper wall banner false positives ($y < 0.22 \cdot H$).
  - $0$ false `Potentially Hazardous` pavement-crack detections.
  - Main roadside heap forms **1 clean, unified `Garbage Pile` (16 items)** + **1 foreground `Individual Litter`**.
  - $0$ nested or overlapping regions.
- **Frame 10002 (`frame_0002_t01.0s.jpg`)**:
  - Wall banner detection eliminated.
  - Heap forms **1 unified `Garbage Pile` (18 items)** + **1 `Individual Litter`**.
- **Across All 16 Frames**:
  - Exactly **0 overlapping multi-region pairs**.
  - Visual output is crisp, readable, and free of banner stacking.

---

## 5. Verification Commands

```powershell
# 1. Run D-109 Disjoint Regions & Video Cleanup Unit Tests
.\.venv-training\Scripts\python.exe -m unittest tests/test_stage5_d109_disjoint_regions_and_video_cleanup.py

# 2. Run All Stage 5 Unit Tests
.\.venv-training\Scripts\python.exe -m unittest discover -s tests -p "test_stage5_d*.py"

# 3. Run Guarded Verification Suite
.\.venv-training\Scripts\python.exe scripts/verify_stage4_guarded.py

# 4. Check Markdown Link Integrity
.\.venv-training\Scripts\python.exe scripts/check_markdown_links.py
```
