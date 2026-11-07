# TACO Stage 5 D-112: Rework `Garbage Pile` Minimum Cardinality Rule to $k \ge 8$ Bounding Boxes (`min_pile_items = 8`)

**Date**: 2026-11-07  
**Branch**: `feat/stage5-paper-category`  
**Authorization**: D-112  

---

## 1. Executive Summary & Authorization Context

Under owner authorization **D-112**, the minimum cardinality threshold separating `Group of Individual Litter` from `Garbage Pile` in the scale-normalized hierarchical spatial grouping pipeline ([`scripts/taco_stage5_spatial_grouping.py`](file:///c:/Users/Justine/Documents/Python/Project%20Image-object%20Detection/scripts/taco_stage5_spatial_grouping.py)) was reworked to strictly require **$k \ge 8$ bounding boxes** (`DEFAULT_MIN_PILE_ITEMS = 8`).

### Exact Cardinality & Regional Category Contract
- **`Individual Litter` (`individual_litter`)**:
  - $k = 1$ isolated bounding box.
  - Retains the exact model confidence score of the underlying object detection (`model_score = float(score)`).
- **`Group of Individual Litter` (`group_of_individual_litter`)**:
  - $2 \le k < 8$ spatially associated bounding boxes ($k \in \{2, 3, 4, 5, 6, 7\}$), or any cluster of $k \ge 2$ items that does not meet the `Garbage Pile` contact/density criteria.
  - Multi-item derived region has `model_score = None` (no invented fake confidence).
- **`Garbage Pile` (`garbage_pile`)**:
  - Strictly requires **$k \ge 8$ bounding boxes** in a consolidated spatial heap (`item_count >= min_pile_items`, `model_score = None`).
  - Must meet contact/density criteria: `touching_pairs >= 1 or packing_density >= 0.25`.
  - If a Tier 1 candidate heap core still has $< 8$ items after Step 1A/1B/1C contiguous consolidation, it is demoted/classified as a `group_of_individual_litter` ($2 \le k \le 7$) rather than `garbage_pile`.

---

## 2. Problem Statement & Motivation

Prior to D-112, the spatial grouping module allowed clusters with $k \ge 2$ or $k \ge 3$ items to qualify as `Garbage Pile` if they exhibited touching boxes or high packing density. Consequently:
1. **Over-Classification of Small Clusters**: On validation Image 1 (`000008.jpg`), 2 overlapping sheets of paper were classified as `Garbage Pile (2 items)`.
2. **Curb Clusters in Still Photos**: On standalone test photo `data/input/For_testing.jpg` (Image 10000), 4 pieces of trash lying against the curb gutter were classified as `Garbage Pile (4 items)`.
3. **Semantic Realism**: In municipal trash inspection and street-level monitoring, 2 to 4 discarded items represent small clusters of litter (`Group of Individual Litter`) rather than full-fledged heaps or garbage piles (`Garbage Pile`). A true garbage pile represents a large, dense accumulation requiring significant cleanup effort ($k \ge 8$ items).

---

## 3. Algorithmic Implementation (`scripts/taco_stage5_spatial_grouping.py`)

### 3.1 Parameterization & Invariants
Added `DEFAULT_MIN_PILE_ITEMS = 8` and parameter `min_pile_items: int = DEFAULT_MIN_PILE_ITEMS` to `compute_spatial_grouping`:
```python
DEFAULT_SCALE_FACTOR = 2.0
DEFAULT_DIAG_FRACTION = 0.225
DEFAULT_DENSITY_THRESHOLD = 0.45
DEFAULT_IOU_PILE_THRESHOLD = 0.15
DEFAULT_MAX_CLUSTER_DIAG_FRACTION = 0.42
DEFAULT_MIN_PILE_ITEMS = 8

def compute_spatial_grouping(
    rows: List[Dict[str, Any]],
    image_width: float | None = None,
    image_height: float | None = None,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    diag_fraction: float = DEFAULT_DIAG_FRACTION,
    density_threshold: float = DEFAULT_DENSITY_THRESHOLD,
    iou_pile_threshold: float = DEFAULT_IOU_PILE_THRESHOLD,
    max_cluster_diag_fraction: float = DEFAULT_MAX_CLUSTER_DIAG_FRACTION,
    min_pile_items: int = DEFAULT_MIN_PILE_ITEMS,
) -> Dict[str, Any]:
```

### 3.2 Two-Tier Consolidation with Cardinality Enforcement
1. **Tier 1 (Heap Core Extraction)**: Compact, touching candidate cores are extracted.
2. **Step 1A & Step 1B (Contiguous Heap Consolidation)**: Adjacent and overlapping cores are merged (Step 1A), and bordering/enclosed unassigned items are absorbed (Step 1B).
3. **Tier 2 (Proximity Grouping with Anti-Chaining)**: Remaining items are grouped using scale-normalized proximity bounded by `max_cluster_diag_fraction = 0.42`.
4. **Step 1C & Multi-Item Disjointness Guarantee**: Any Tier 2 cluster overlapping a heap core by $\ge 0.15$ or $IoU > 0.05$ is absorbed. An iterative merging loop over all candidate multi-item regions ensures $IoU \le 0.05$ and $IoMin < 0.20$ across the entire image.
5. **Final Cardinality & Category Assignment**:
   ```python
   is_pile = (k >= min_pile_items) and (touch_count >= 1 or density >= 0.25)

   if is_pile:
       cat = "garbage_pile"
       label = SPATIAL_CATEGORIES[cat]
       pile_count += 1
   else:
       cat = "group_of_individual_litter"
       label = SPATIAL_CATEGORIES[cat]
       group_count += 1
   ```

---

## 4. Replay Artifacts & Video Verification

Replay artifacts in `runs/video_detection/motion_extract_testing/` (`video-showcase.json` and `manifest.json`) were updated to authorization `"D-112"` with valid SHA-256 hashes.

### Key Frame Verification:
1. **`For_testing.jpg` (`Image 10000`)**:
   - Total detections: 6 items.
   - Center Curb Cluster ($k = 4 < 8$): Classified as **`Group of Individual Litter (4 items)`** (bbox: $[2085.0, 1564.5, 1154.3, 878.9]$).
   - Lower-Left Cigarette Butt ($k = 1$): **`Individual Litter`** ($sc = 0.2427$).
   - Upper Curb Litter ($k = 1$): **`Individual Litter`** ($sc = 0.1987$).
   - Summary: **`0 Garbage Piles, 1 Group of Individual Litter, 2 Individual Litter`** (3 total regions).
2. **`Motion_Extract_Testing_2.mp4` (`Frames 10005, 10006, 10007`)**:
   - 4 crumpled white paper towel balls on concrete pavement.
   - Re-classified as **`Group of Individual Litter (4 items)`**.
   - Summary: **`0 Garbage Piles, 1 Group of Individual Litter, 0 Individual Litter`**.
3. **Dense Roadside Heaps ($k \ge 8$)**:
   - Any cluster with $k \ge 8$ items meeting density/contact criteria (such as the 16-item heap on `Motion_Extract_Testing.mp4` Frame 10001) qualifies as **`Garbage Pile (k items)`**.

---

## 5. Frozen 65-Image Validation Invariants

Object-level consensus detections and validation metrics remain bitwise identical to the benchmark highs:

| Metric | Frozen Invariant | Achieved Result | Status |
|---|---|---|---|
| **Official 9-Specific AP** | $\ge 0.301350$ | **`0.301350`** | Bitwise Identical |
| **BANTAI 5-Category AP** | $\ge 0.497161$ | **`0.497161`** | Bitwise Identical |
| **`paper` Class AP** | $\ge 0.640610$ | **`0.640610`** | Bitwise Identical |
| **True Positives (TP)** | $\ge 77$ | **`77`** | Bitwise Identical |
| **Lost True Positives** | $== 0$ (`[]`) | **`0` (`[]`)** | Clean |
| **False Positives (FP)** | $\le 63$ | **`63`** | Clean |
| **Clean Validation Images** | $\ge 36$ | **`36 / 65`** | Clean |

---

## 6. Unit Testing Suite

Created [`tests/test_stage5_d112_garbage_pile_min8_rule.py`](file:///c:/Users/Justine/Documents/Python/Project%20Image-object%20Detection/tests/test_stage5_d112_garbage_pile_min8_rule.py) verifying:
1. `test_01_code_hygiene_and_guardrails`: Active taxonomy locked to 9 classes; zero hardcoded identifiers; no `frame` token in spatial grouping module.
2. `test_02_synthetic_cardinality_boundary`: Tests $k=1 \to$ indiv; $k \in \{2, 3, 4, 5, 6, 7\} \to$ group; $k \in \{8, 12, 16\} \to$ pile; parameterization override with `min_pile_items=4`.
3. `test_03_standalone_image_for_testing_grouping`: Verifies `For_testing.jpg` produces 0 piles, 1 group, 2 indiv.
4. `test_04_video_frames_paper_towel_grouping`: Verifies Frames 10005..10007 produce 1 group (4 items).
5. `test_05_zero_overlapping_multi_item_regions_across_sweep`: Verifies 0 overlapping multi-item regions across all video frames and all 65 validation images.
6. `test_06_replay_artifact_integrity_and_authorization`: Confirms SHA-256 integrity and authorization `D-112`.
7. `test_07_full_65_benchmark_invariants_preserved`: Confirms bitwise preservation of AP9, AP5, paper AP, 77 TPs, 63 FPs, 36 clean images.

**Test Results**:
- `tests/test_stage5_d112_garbage_pile_min8_rule.py`: **7/7 passed**.
- Full Stage 5 discovery (`test_stage5_d*.py`): **120/120 passed (6 skipped)** in 89.4s.
- Full guarded test suite (`scripts/verify_stage4_guarded.py`): **433/455 passed (22 skipped, 0 failures, 0 errors)**.
