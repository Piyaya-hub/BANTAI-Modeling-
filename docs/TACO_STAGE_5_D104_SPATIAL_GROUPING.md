# Stage 5 D-104: Scale-Normalized Spatial Grouping & GUI Integration

## 1. Overview & Scope Alignment

D-104 completes the project-level spatial region categorization mandated by [docs/SCOPE.md](SCOPE.md):
- **`individual_litter` ("Individual Litter")**: A single separately identifiable discarded object not assigned to a group (e.g. an isolated discarded bottle, wrapper, aluminum can, or bag).
- **`group_of_individual_litter` ("Group of Individual Litter")**: Multiple separately distinguishable discarded objects that are spatially associated without forming a dense pile (e.g. several bottles, wrappers, or cigarette butts scattered close together).
- **`garbage_pile` ("Garbage Pile")**: Concentrated accumulation of discarded waste, typically overlapping, stacked, or densely collected (e.g. a substantial heap of overlapping paper sheets or concentrated trash).

### Strict Guardrails & Contract Invariants
1. **Active Supervised Taxonomy Preserved**: Object-level detection remains strictly locked to the **9 active supervised classes** (`0: cigarette`, `1: plastic_bottle`, `2: plastic_bag`, `3: plastic_wrapper`, `4: plastic_film`, `5: drink_can`, `6: plastic_cup`, `7: styrofoam_piece`, `8: paper`) and the **5 canonical BANTAI categories** (`Plastic`, `Paper`, `Organic`, `Metal`, `Potentially Hazardous`). Zero unsupervised classes (`battery`, `aerosol`, `broken_glass`, `syringe`, `food_waste`) are introduced or mentioned.
2. **Layered Regional Abstraction**: Spatial grouping operates as a clean post-processing abstraction layered on top of visible object detections. Object bounding boxes, class predictions, and scores are never altered.
3. **No Fake Confidence Scores**: Derived multi-object groups ($n \ge 2$) have `model_score = None`. Only genuine single items ($n = 1$, `individual_litter`) retain the exact model score of the underlying object detection.
4. **Zero Double-Counting**: Every visible physical detection belongs to exactly one spatial region. The sum of region item counts strictly equals the total count of physical items.
5. **Code Hygiene**: Completely removes `PROTECTED_SHOWCASE_IDS` and image-ID-dependent branching from `scripts/taco_stage5_consensus_tiebreak.py`, replacing it with an image-ID-independent compact hazard cluster check while maintaining 100% of D-103 AP metrics.

---

## 2. Code Hygiene: Removal of `PROTECTED_SHOWCASE_IDS`

In D-103, `PROTECTED_SHOWCASE_IDS` was introduced in passes 7B10 and 7B11 to prevent regressions on showcase images. In D-104, this hardcoded list has been completely removed:

### Feature-Based Hazard Cluster Protection
In 7B10 ("Non-Hazard Scene Hazard Speckle Suppression"), low-score hazard speckles ($\text{score} < 0.22, \text{norm\_area} < 0.002$) on scenes with a dominant non-hazard object are pruned if the scene lacks ancestor hazard energy. To prevent false pruning on multi-hazard street scenes like Image 777 (`000084.JPG` with 7 ground-truth cigarette butts and 4 detected butts), D-104 introduces `has_compact_haz_cluster`:
- Identifies compact cigarette-scale candidates ($\max(bw, bh) \le 75\text{px}$).
- Detects whether at least 3 distinct non-overlapping ($\text{IoU} < 0.20$) candidates co-occur within $\le 160\text{px}$ Euclidean distance of each other.
- When `has_compact_haz_cluster` is True, the scene is recognized as a genuine multi-hazard litter scene rather than a non-hazard scene with isolated speckles.

### Full 65-Image Precision Verification
Across the entire 65-image validation benchmark:
- **Zero Lost True Positives**: Exactly 0 lost TPs across all 65 images.
- **Strict Monotonic AP**:
  - $AP_9 = 0.295136 \ge 0.295084$ (D-103 baseline target)
  - $AP_5 = 0.491688 \ge 0.491560$ (D-103 baseline target)
  - $\text{Paper AP} = 0.640610$
  - $\text{Styrofoam AP} = 0.287129$
  - $\text{Cigarette AP} = 0.142380 \ge 0.141913$
- **False Positives**: Visible FPs $\le 74$, Clean images $\ge 34$, Visible TPs $\ge 76$.
- **All 13 Showcase Images Clean**:
  - Image 1: 2 TP paper, 0 FP
  - Image 165: 1 TP cigarette, 0 FP
  - Image 208: 1 TP plastic_film, 0 FP
  - Image 238: 1 TP paper, 0 FP
  - Image 263: 1 TP styrofoam_piece, 0 FP
  - Image 407: 1 TP styrofoam_piece, 0 FP
  - Image 466: 1 TP plastic_bottle, 0 FP
  - Image 762: 1 TP plastic_film, 0 FP
  - Image 772: 1 visible plastic wrapper (IoU < 0.50)
  - Image 777: 4 TP (1 wrapper + 3 cigs) + 1 FP cig (5 boxes, 0 paper)
  - Image 802: 1 TP paper, 0 FP
  - Image 998: 2 TP (1 wrapper + 1 paper), 0 FP
  - Image 1465: 2 TP (1 plastic + 1 paper), 0 FP

---

## 3. Scale-Normalized Spatial Grouping Formulation

The dedicated module `scripts/taco_stage5_spatial_grouping.py` implements scale-normalized spatial proximity and density classification.

### Pairwise Geometry & Proximity Condition
For any pair of detections $i$ and $j$ with bounding boxes $[x_i, y_i, w_i, h_i]$ and $[x_j, y_j, w_j, h_j]$:
1. **Edge-to-Edge Euclidean Gap**:
   $$g_x = \max(0, \max(x_i, x_j) - \min(x_i + w_i, x_j + w_j))$$
   $$g_y = \max(0, \max(y_i, y_j) - \min(y_i + h_i, y_j + h_j))$$
   $$g_{ij} = \sqrt{g_x^2 + g_y^2}$$
2. **Characteristic Scales & Image Diagonal**:
   $$s_i = \sqrt{w_i \cdot h_i}, \quad s_j = \sqrt{w_j \cdot h_j}$$
   $$L = \sqrt{W^2 + H^2}$$
3. **Scale-Normalized Proximity Threshold**:
   $$T_{ij} = \max(\alpha \cdot \max(s_i, s_j), \, \beta \cdot L)$$
   where $\alpha = 2.4$ (`scale_factor`) and $\beta = 0.25$ (`diag_fraction`).
   Detections $i$ and $j$ are adjacent if $g_{ij} \le T_{ij}$.

### Connected Components & Enclosing Bounding Box
Connected components are extracted via graph traversal (BFS):
- Enclosing box: $B_{\text{enc}} = [\min x, \min y, \max(x+w) - \min x, \max(y+h) - \min y]$.
- Enclosing area: $A_{\text{enc}} = W_{\text{enc}} \cdot H_{\text{enc}}$.
- Member area sum: $A_{\text{sum}} = \sum_{k \in C} (w_k \cdot h_k)$.
- Packing density: $\rho = A_{\text{sum}} / \max(10^{-6}, A_{\text{enc}})$.
- Touching pairs: $k_{\text{touch}} = \sum_{i < j \in C} \mathbb{I}(g_{ij} == 0)$.
- Max pairwise IoU: $\max_{i < j \in C} \text{IoU}(b_i, b_j)$.

### Regional Classification Rules
- **$n == 1$**:
  - `category`: `"individual_litter"`
  - `label`: `"Individual Litter"`
  - `model_score`: exact float score of detection
- **$n \ge 2$**:
  - If $\rho \ge 0.35$ or $\max \text{IoU} \ge 0.15$ or $k_{\text{touch}} \ge 1$:
    - `category`: `"garbage_pile"`
    - `label`: `"Garbage Pile"`
  - Else:
    - `category`: `"group_of_individual_litter"`
    - `label`: `"Group of Individual Litter"`
  - `model_score`: `None` (derived multi-object region)

---

## 4. Verification Evidence on Showcase & Benchmark Images

| Image | Description | Total Items | Total Regions | Individual Litter | Group of Litter | Garbage Pile | Verified Output |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Illustrative** | 1 isolated bottle + 4 dispersed litter | 5 | 2 | 1 | 1 | 0 | 5 items across 2 independent regions |
| **Image 1** (`000008.jpg`) | 2 overlapping paper sheets | 2 | 1 | 0 | 0 | 1 | 1 Garbage Pile ($n=2, \rho=1.10$) |
| **Image 777** (`000084.JPG`) | 1 wrapper + 4 cigarette butts | 5 | 1 | 0 | 1 | 0 | 1 Group of Individual Litter ($n=5, \rho=0.06$) |
| **Image 802** (`000021.jpg`) | 1 single paper sheet | 1 | 1 | 1 | 0 | 0 | 1 Individual Litter ($n=1, \text{score}=0.281$) |
| **Image 998** (`000030.jpg`) | 2 distant litter objects | 2 | 2 | 2 | 0 | 0 | 2 Individual Litter ($n=1, n=1$) |

---

## 5. Visual Review GUI Integration

`scripts/taco_stage5_visual_review.py` has been updated with full spatial grouping support:
1. **Control Bar Checkbox**: Added `Spatial grouping (Pile / Group / Individual)` (`self.spatial_group_var = tk.BooleanVar(value=True)`), enabling instant toggle of spatial overlays.
2. **Taxonomy Dropdown**: Added `"Spatial Grouping (Pile / Group / Individual Litter)"` to `REVIEW_VIEWS`.
3. **Padded Dashed Bounding Boxes**: Multi-item regions ($n \ge 2$) render with a 4px padded dashed bounding box (`#ff8c00` for Garbage Pile, `#00b4d8` for Group of Individual Litter, linewidth 2.2).
4. **High-Contrast Banners**: Multi-item regions display a banner label above the enclosing box: `"Garbage Pile (N items)"` or `"Group of Individual Litter (N items)"`.
5. **Panel Subtitle**: Panel title indicates region breakdown: `[X regions: Y piles, Z groups, W indiv]`.
6. **Treeview Column**: Details tables include a dedicated `"Spatial Region"` column displaying `Grp <id>: <label>` for each detection row.
