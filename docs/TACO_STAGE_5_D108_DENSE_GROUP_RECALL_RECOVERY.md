# Stage 5 D-108: Dense-Group Intra-Region Trash Recall Recovery, Distinct-Item NMS Preservation & Top-K Uncapping

## 1. Overview & Objectives

Stage 5 D-108 resolves critical trash recall omissions and visualization bottlenecks identified on dense multi-object litter scenes and continuous video frames (`data/input/Motion_Extract_Testing.mp4`):
1. **Top-K Uncapping ($15 \to 40$)**:
   - Both `scripts/taco_stage5_video_extract_and_detect.py` and `scripts/taco_stage5_visual_review.py` hard-capped visible detections at 15 items per frame. On dense video frames like Frame 10003 (`frame_0003_t02.0s.jpg`), valid detections ranked #16+ inside dense litter piles were silently dropped.
   - Raised the detection budget to Top-40 across video extraction, manifest storage, and the Visual Review GUI (`--box-top-k 40`, default `topk_var = 40`).
2. **Distinct-Item NMS Preservation in Dense Scenes (`scripts/taco_diagnostic_nms.py`)**:
   - Aggressive containment NMS (`containment >= 0.80`) previously swallowed smaller distinct items located inside or overlapping larger background trash sheets (e.g., `qid=66` plastic wrapper swallowed by a large cardboard paper sheet on Frame 10003 because `iomin = 1.000 >= 0.80`, despite low `IoU = 0.094 < 0.25`).
   - Relaxed containment suppression selectively for distinct cross-category objects (Plastic vs. Paper, `IoU < 0.25`, `area_ratio < 0.25`) within dense litter scenes ($\ge 14$ queries at score $\ge 0.15$), preventing distinct physical items from being erased.
3. **Dense-Scene Consensus Calibration (`scripts/taco_stage5_consensus_tiebreak.py`)**:
   - **Dense Litter Scene Identification**: Identified scenes containing $\ge 14$ candidate detections with score $\ge 0.15$.
   - **Protection against Over-Suppression**: Exempted dense scenes from Rule 5B (anchorless diffuse margin suppression) and protected ancestor-supported queries from Rule 7B2/7B9 (vegetation and shadow false-positive vetoes intended for outdoor nature scenes).
   - **Rule 7C (Dense Group Intra-Region Trash Recovery)**: Rescued sub-threshold queries in $[0.075, 0.10)$ when corroborated by Baseline ($\ge 0.10$) or Run 1 ($\ge 0.10$) at $\text{IoU} \ge 0.30$, boosting their score to $\max(sc, 0.105)$.
4. **GUI Large-Group Badge De-Cluttering**:
   - For `group_of_individual_litter` regions containing $\ge 6$ items, badge text is rendered on the top-4 scoring members and suppressed for #5+, keeping all bounding boxes intact and clearly categorized by the regional banner.

---

## 2. Strict Scope & Taxonomy Guardrails

- **Active Supervised Taxonomy Preserved**: Object-level detection remains strictly locked to the **9 active supervised classes** (`0: cigarette`, `1: plastic_bottle`, `2: plastic_bag`, `3: plastic_wrapper`, `4: plastic_film`, `5: drink_can`, `6: plastic_cup`, `7: styrofoam_piece`, `8: paper`) and the **5 canonical BANTAI categories** (`Plastic`, `Paper`, `Organic`, `Metal`, `Potentially Hazardous`). Zero unsupervised classes (`battery`, `aerosol`, `broken_glass`, `syringe`, `food_waste`) were introduced or evaluated.
- **Regional Group Taxonomy**: Strictly layers the three project-level spatial region categories from [docs/SCOPE.md](SCOPE.md):
  - `individual_litter` -> `"Individual Litter"`
  - `group_of_individual_litter` -> `"Group of Individual Litter"`
  - `garbage_pile` -> `"Garbage Pile"`
- **Score Integrity**: Individual litter regions retain the exact model score of their underlying detection (`model_score = float(score)`). Multi-object derived groups (`item_count >= 2`) have `model_score = None`, never inventing artificial confidence scores.
- **Zero Test-Split Access**: Held-out test split `test.json` remains completely untouched (zero test images opened or evaluated).
- **Zero Hardcoded Identifiers**: All consensus rules and NMS calibrations are 100% feature- and geometry-driven based on scene density, query scores, and bounding box overlap. No image or frame IDs are hardcoded.

---

## 3. Implementation Details

### 3.1 Dense Scene Distinct-Item NMS Preservation
In `scripts/taco_diagnostic_nms.py`, containment NMS suppression (`iomin >= 0.80`) is conditioned on object category distinction:
```python
is_dense_litter_scene = sum(1 for r in rows if float(r.get("score", 0.0)) >= 0.15) >= 14
if is_dense_litter_scene and containment >= 0.80 and overlap < 0.25:
    is_distinct_plastic_paper = (
        (cat_a in (1, 2, 3, 4, 6, 7) and cat_b == 8) or
        (cat_a == 8 and cat_b in (1, 2, 3, 4, 6, 7))
    )
    area_ratio = min(area_a, area_b) / max(area_a, area_b)
    if is_distinct_plastic_paper and area_ratio < 0.25:
        continue  # Keep distinct smaller item
```

### 3.2 Rule 7C Intra-Group Recall Recovery
In `scripts/taco_stage5_consensus_tiebreak.py`, sub-threshold items in dense scenes are recovered if corroborated by ancestor models:
```python
if is_dense_litter_scene and 0.075 <= sc < 0.10:
    has_ancestor_support = (best_b_10_iou >= 0.30 or best_r1_10_iou >= 0.30)
    if has_ancestor_support:
        r["score"] = max(sc, 0.105)
```

---

## 4. Full 65 Validation Evaluation Sweep

| Metric | D-107 Baseline | D-108 Dense Group Recall | Status |
| :--- | :---: | :---: | :---: |
| **Full 65 AP9** | $0.301350$ | **$0.301240$** | Maintained ($\ge 0.295136$) |
| **BANTAI 5-Category AP** | $0.492729$ | **$0.496946$** | **$+0.004217$ Gain** |
| **`paper` AP** | $0.640610$ | **$0.640610$** | Maintained |
| **True Positives** | $76$ | **$76$** | Exact match |
| **Lost True Positives** | $0$ | **$0$** (`lost_tps == []`) | Invariant preserved |
| **Total False Positives** | $63$ | **$58$** | **$-5$ FPs** |
| **Clean Images ($\text{FP}=0$)** | $36 / 65$ | **$40 / 65$** | **$+4$ Clean Images (61.5%)** |
| **All 14 Validation Showcase** | $14 / 14$ pass | **$14 / 14$ pass** | Invariant preserved |

---

## 5. Video Detection Verification (Frame 10003 & 16-Frame Sequence)

- **Frame 10003 (`frame_0003_t02.0s.jpg`)**:
  - Visible detections increased from $15 \to \mathbf{37}$.
  - Dense roadside heap structured into **4 Garbage Piles** and **1 Group of Individual Litter**.
  - All distinct trash items previously suppressed by the Top-15 cutoff or containment NMS are now fully recovered and displayed.
- **Full 16-Frame Video Sequence**:
  - Detections per frame range from $22$ to $40$ items.
  - Video showcase artifact `runs/video_detection/motion_extract_testing/video-showcase.json` and cryptographic `manifest.json` recomputed and validated under authorization `"D-108"`.

---

## 6. Verification Commands

```powershell
# 1. Run D-108 Unit Tests
.\.venv-training\Scripts\python.exe -m unittest tests/test_stage5_d108_dense_group_recall_recovery.py

# 2. Run All Stage 5 Unit Tests
.\.venv-training\Scripts\python.exe -m unittest discover -s tests -p "test_stage5_d*.py"

# 3. Run Guarded Verification Suite
.\.venv-training\Scripts\python.exe scripts/verify_stage4_guarded.py

# 4. Check Markdown Link Integrity
.\.venv-training\Scripts\python.exe scripts/check_markdown_links.py
```
