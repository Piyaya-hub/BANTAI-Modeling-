# Stage 5 D-107: Two-Tier Hierarchical Spatial Grouping, Sachet Hazard Tie-Break & Video GUI De-Cluttering

## 1. Overview & Objectives

Stage 5 D-107 addresses the key spatial grouping and classification challenges discovered on dense roadside heaps and high-resolution video frames (`data/input/Motion_Extract_Testing.mp4`):
1. **Two-Tier Hierarchical DBSCAN Spatial Grouping**:
   - **Tier 1 (Dense Core Extraction)**: Identifies tightly overlapping and physically contacting objects ($g_{ij} == 0.0 \lor \text{IoU}_{ij} > 0.01 \lor (g_{ij} \le 0.35 \cdot \min(s_i, s_j) \land \text{neighborhood support})$) and classifies compact accumulations ($k \ge 3$ with touching $\ge 1$ or density $\ge 0.40$; $k=2$ with density $\ge 0.45$ and IoU $\ge 0.15$) directly into `Garbage Pile`.
   - **Tier 2 (Anti-Chaining Proximity Grouping)**: Clusters remaining non-pile items using scale-normalized proximity ($\alpha=2.0, \beta=0.225$) bounded by an anti-chaining cluster diagonal ceiling ($\le 0.42 \cdot L$), preventing foreground outliers from single-linkage chaining into dense wall heaps.
2. **Sachet/Wrapper Hazard Tie-Break Rule**:
   - On Frame 10008, a prominent foreground red candy wrapper ($w=74.8\text{px}, h=60.4\text{px}, \text{norm\_area}=0.00218, \text{aspect}=1.24$) was misclassified as `0: cigarette` (`Potentially Hazardous 0.33`) despite strong plastic co-activation ($v_3=0.035, v_1=0.024$).
   - Reassigns compact rectangular objects ($\text{norm\_area} \ge 0.0018$, $\min(w, h) \ge 35\text{px}$, aspect ratio $\le 2.5$) with non-hazard support ($\ge 0.035$) from hazard to `Plastic` (`plastic_wrapper`) or `Paper`.
   - Increases full 65 validation AP9 monotonically from $0.298963$ to **$0.301350$** with **0 lost True Positives** (`lost_tps == []`).
3. **Video GUI De-Cluttering & Auto-Tab Selection**:
   - **Badge De-Cluttering**: In dense Garbage Piles ($k \ge 3$), badge text labels are drawn only for the top-2 scoring members, while all bounding boxes remain fully visible and distinct.
   - **Banner Positioning**: Regional bounding boxes and category banners render at `zorder=7` to avoid obscuring detection boxes.
   - **Auto Tab Selection**: Automatically activates the fourth table tab (`D-095 Champion`) upon switching into Video Frames mode (`VIDEO_FRAMES_FILTER`), ensuring the details table is never blank.
   - **Spacious Subplot Margins**: Replaced tight cropping with explicit figure margins (`left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.06, hspace=0.16`), optimizing display for vertical $1080 \times 1920$ portrait frames.

---

## 2. Strict Scope & Taxonomy Guardrails

- **Active Supervised Taxonomy Preserved**: Object-level detection remains strictly locked to the **9 active supervised classes** (`0: cigarette`, `1: plastic_bottle`, `2: plastic_bag`, `3: plastic_wrapper`, `4: plastic_film`, `5: drink_can`, `6: plastic_cup`, `7: styrofoam_piece`, `8: paper`) and the **5 canonical BANTAI categories** (`Plastic`, `Paper`, `Organic`, `Metal`, `Potentially Hazardous`). Zero unsupervised classes (`battery`, `aerosol`, `broken_glass`, `syringe`, `food_waste`) were introduced or proposed.
- **Regional Group Taxonomy**: Strictly layers the three project-level spatial region categories from [docs/SCOPE.md](SCOPE.md):
  - `individual_litter` -> `"Individual Litter"`
  - `group_of_individual_litter` -> `"Group of Individual Litter"`
  - `garbage_pile` -> `"Garbage Pile"`
- **Score Integrity**: Individual litter regions retain the exact model score of their underlying detection (`model_score = float(score)`). Multi-object derived groups (`item_count >= 2`) have `model_score = None`, never inventing artificial confidence scores.
- **Zero Test-Split Access**: Held-out test split `test.json` remains completely untouched.
- **Zero Hardcoded Identifiers**: All rules are formulated purely on physical box geometry (normalized area, aspect ratio, min dimensions) and model probability vectors. No image or frame IDs are hardcoded.

---

## 3. Two-Tier Hierarchical Spatial Grouping Architecture

### 3.1 Tier 1: Dense Garbage Pile Core Extraction
- **Tight Adjacency Condition**:
  $$\text{tight}(i, j) \iff (g_{ij} == 0.0) \lor (\text{IoU}_{ij} > 0.01) \lor (g_{ij} \le 0.35 \cdot \min(s_i, s_j) \land \mathcal{N}_i \land \mathcal{N}_j)$$
  where $\mathcal{N}_i$ requires detection $i$ to have another neighbor within $0.80 \cdot \max(s_i, s_k)$.
- **Core Classification**:
  A connected component of the tight graph qualifies as `garbage_pile` if:
  $$(k \ge 3 \land (\text{touching} \ge 1 \lor \rho \ge 0.40)) \lor (k = 2 \land \rho \ge 0.45 \land \text{IoU} \ge 0.15)$$
  where $\rho = \frac{\sum_m \text{area}_m}{\text{area}(\text{enclosing\_box})}$ is packing density.

### 3.2 Tier 2: Anti-Chaining Proximity Grouping
- Non-pile detections are grouped using scale-normalized proximity:
  $$g_{ij} \le \max(\alpha \cdot \max(s_i, s_j), \beta \cdot L)$$
  with $\alpha = 2.0, \beta = 0.225$.
- **Anti-Chaining Diagonal Bound**:
  A candidate item is only added to an expanding cluster if the resulting cluster bounding box diagonal satisfies:
  $$\text{diag}(\text{cluster}) \le 0.42 \cdot L$$
  preventing runaway single-linkage chaining across the image diagonal.

### 3.3 Verification on Video Frames (Frames 10006 & 10008)
- **Frame 10006** ($15$ visible items):
  - **Garbage Pile** ($4$ items along the wall, $\rho = 0.526$)
  - **Group of Individual Litter** ($11$ items, $\rho = 0.188$)
- **Frame 10008** ($15$ visible items):
  - **Garbage Pile #1** ($3$ items along wall, $\rho = 0.633$)
  - **Garbage Pile #2** ($3$ items along wall, $\rho = 0.653$)
  - **Group of Individual Litter** ($8$ items on street, $\rho = 0.115$)
  - **Individual Litter** ($1$ item, $\rho = 1.000$)

---

## 4. Full 65 Validation Evaluation Sweep

| Metric | D-105 Baseline | D-107 Hierarchical & Tie-Break | Monotonic Gain |
| :--- | :---: | :---: | :---: |
| **Full 65 AP9** | $0.298963$ | **$0.301350$** | **$+0.002387$** |
| **Full 65 AP5** | $0.492729$ | **$0.492729$** | Stable |
| **True Positives** | $76$ | **$76$** | Exact match |
| **Lost True Positives** | $0$ | **$0$** (`lost_tps == []`) | Invariant preserved |
| **Clean Images ($\text{FP}=0$)** | $36 / 65$ | **$36 / 65$** | Invariant preserved |
| **All 14 Validation Showcase** | $14 / 14$ pass | **$14 / 14$ pass** | Invariant preserved |

---

## 5. Verification Commands

```powershell
# 1. Run D-107 Hierarchical Spatial Grouping & Video Tests
.\.venv-training\Scripts\python.exe -m unittest tests/test_stage5_d107_hierarchical_video_grouping.py

# 2. Run All Stage 5 Unit Tests
.\.venv-training\Scripts\python.exe -m unittest discover -s tests -p "test_stage5_d*.py"

# 3. Run Guarded Verification Suite
.\.venv-training\Scripts\python.exe scripts/verify_stage4_guarded.py

# 4. Check Markdown Link Integrity
.\.venv-training\Scripts\python.exe scripts/check_markdown_links.py
```
