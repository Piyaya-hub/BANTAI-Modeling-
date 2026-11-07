# Stage 5 D-110: Standalone Input Image (`data/input/For_testing.jpg`) Three-Model Detection, Distractor Guardrails & GUI Integration

## 1. Overview & Objectives

Stage 5 D-110 expands the input media processing pipeline to support standalone still photographs alongside extracted video sequences. The primary test image `data/input/For_testing.jpg` ($5136 \times 3984$ RGB) was evaluated with the full Three-Model Stage 5 consensus ensemble (`Warm-Start Baseline`, `Stage 5 Run 1`, and `D-095 Champion`) combined with scene consensus and scale-normalized disjoint spatial grouping:

1. **True Litter Detection in the Street Gutter (Center, Curb & Roadway)**:
   - **Center Curb Cluster** ($x \sim 40\%..61\%, y \sim 39\%..60\%$): 4 physical items corroborated across models:
     * Blue rectangular snack wrapper/box: `qid=5`, `cat=8` (`paper` $\to$ `Paper`), $sc = 0.2923$.
     * White cigarette butt in curb seam: `qid=0`, `cat=0` (`cigarette` $\to$ `Potentially Hazardous`), $sc = 0.2617$.
     * Clear crumpled plastic wrapper/bag: `qid=6`, `cat=1` (`plastic_bottle`/film $\to$ `Plastic`), $sc = 0.2583$.
     * Red-and-white cup / container: `qid=17`, `cat=1` (plastic/composite $\to$ `Plastic`), $sc = 0.2580$.
     * Forms a single, compact **`Garbage Pile (4 items)`** with bounding box $[2085.0, 1564.5, 1154.3, 878.9]$ and packing density $0.319$.
   - **Lower-Left Cigarette Butt** ($x = 27.6\%, y = 85.2\%$):
     * Distinct orange filter cigarette butt lying on dark wet asphalt: `qid=1`, `cat=0` (`cigarette` $\to$ `Potentially Hazardous`), $sc = 0.2427$, bbox $[1417.2, 3395.9, 95.4, 102.7]$.
     * Correctly isolated as **`Individual Litter`** with $sc = 0.2427$.
   - **Upper Curb Plastic Litter** ($x = 67.3\%, y = 14.5\%$):
     * Small discarded plastic piece nestled against upper curb: `qid=7`, `cat=1` (`plastic` $\to$ `Plastic`), $sc = 0.1987$, bbox $[3458.2, 577.5, 90.1, 85.9]$.
     * Correctly isolated as **`Individual Litter`** with $sc = 0.1987$.

2. **Camera Border & Distractor Guardrails (0 False Positives)**:
   - **Two Black Leather Dress Shoes / Trousers** ($x > 70\% \cdot W, y > 45\% \cdot H$): **0 FPs**.
   - **Concrete Pillar / Sack Wrap** ($x > 75\% \cdot W, y < 25\% \cdot H$): **0 FPs**.
   - **White Painted Road Stripe**: **0 FPs**.
   - **Yellow POCO Camera Watermark Border** (margins: top/bottom $6.5\%$, left/right $5\%$): **0 FPs**.

3. **Media Pipeline & GUI Integration**:
   - `For_testing.jpg` is copied into `runs/video_detection/motion_extract_testing/frames/For_testing.jpg` and registered with `id = 10000`, placed first in `video-showcase.json` and `video-score-vectors.npz` (shape `(17, 900, 9)`).
   - In `scripts/taco_stage5_visual_review.py`:
     * Added `--for-testing` and `--input-image` CLI flags to open the viewer directly focused on `For_testing.jpg`.
     * Title displays `Input Still Image — data/input/For_testing.jpg` in panel 1.
     * For low-density scenes ($\le 8$ visible items), individual item badges remain visible, readable, and shifted to prevent label overlap.
     * Manifest authorization updated to `"D-110"` with matching SHA-256 hashes.

---

## 2. Strict Scope & Taxonomy Guardrails

- **Active Supervised Taxonomy**: Strictly locked to the **9 active supervised classes** (`0: cigarette`, `1: plastic_bottle`, `2: plastic_bag`, `3: plastic_wrapper`, `4: plastic_film`, `5: drink_can`, `6: plastic_cup`, `7: styrofoam_piece`, `8: paper`) and the **5 canonical BANTAI categories** (`Plastic`, `Paper`, `Organic`, `Metal`, `Potentially Hazardous`). Zero unsupervised classes (`battery`, `aerosol`, `broken_glass`, `syringe`, `food_waste`) are permitted or referenced.
- **Regional Group Taxonomy**: Strictly layers the 3 spatial region categories from [docs/SCOPE.md](SCOPE.md):
  - `individual_litter` $\to$ `"Individual Litter"`
  - `group_of_individual_litter` $\to$ `"Group of Individual Litter"`
  - `garbage_pile` $\to$ `"Garbage Pile"`
- **Score Integrity**: Individual litter regions retain the exact model score of their underlying detection (`model_score = float(score)`). Multi-object derived groups (`item_count >= 2`) have `model_score = None`, never inventing artificial confidence scores.
- **Zero Test-Split Access**: Held-out test split `test.json` remains completely untouched (zero test images opened).
- **Zero Hardcoded Identifiers**: All rules are formulated purely on physical box geometry (normalized area, aspect ratio, min dimensions) and model probability distributions. No image or frame IDs are hardcoded.

---

## 3. Full 65 Validation Evaluation Sweep

| Metric | D-109 Clean Disjoint | D-110 Standalone Image | Status |
| :--- | :---: | :---: | :---: |
| **Full 65 AP9** | $0.301350$ | **$0.301350$** | Benchmark high preserved |
| **BANTAI 5-Category AP** | $0.497161$ | **$0.497161$** | Maintained |
| **`paper` AP** | $0.640610$ | **$0.640610$** | Maintained |
| **True Positives** | $77$ | **$77$** | Maintained |
| **Lost True Positives** | $0$ | **$0$** (`lost_tps == []`) | Invariant preserved |
| **Total False Positives** | $63$ | **$63$** | Maintained |
| **Clean Images ($\text{FP}=0$)** | $36 / 65$ | **$36 / 65$** | Maintained |
| **All 14 Validation Showcase** | $14 / 14$ pass | **$14 / 14$ pass** | Invariant preserved |

---

## 4. Detection & Regional Summary on `For_testing.jpg` (Image 10000)

| Region ID | Category | Item Count | Model Score | Packing Density | Enclosing Bounding Box | Underlying Detections |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Region 0** | `Garbage Pile` | 4 | `None` | $0.319$ | $[2085.0, 1564.5, 1154.3, 878.9]$ | `Paper 0.2923`, `Hazard 0.2617`, `Plastic 0.2583`, `Plastic 0.2580` |
| **Region 1** | `Individual Litter` | 1 | $0.2427$ | $1.000$ | $[1417.2, 3395.9, 95.4, 102.7]$ | `Potentially Hazardous 0.2427` (Cigarette Butt) |
| **Region 2** | `Individual Litter` | 1 | $0.1987$ | $1.000$ | $[3458.2, 577.5, 90.1, 85.9]$ | `Plastic 0.1987` (Upper Curb Litter) |

---

## 5. GUI Launch Verification

To view the standalone test photograph in the Stage 5 review GUI:

```bash
# Launch directly to For_testing.jpg
.\.venv-training\Scripts\python.exe scripts/taco_stage5_visual_review.py --for-testing

# Or using the alias
.\.venv-training\Scripts\python.exe scripts/taco_stage5_visual_review.py --input-image
```
