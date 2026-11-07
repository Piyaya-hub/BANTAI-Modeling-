# Stage 5 D-106: Video Frame Extraction & Three-Model Stage 5 Detection on `data/input/Motion_Extract_Testing.mp4`

## 1. Overview & Objectives

Stage 5 D-106 extends the BANTAI Stage 5 evaluation suite to real-world continuous video footage, processing `data/input/Motion_Extract_Testing.mp4` through the Three-Model Cross-Stage Consensus pipeline and integrating a dedicated one-click video review mode directly into the existing `Stage5ViewerGUI`:
1. **Video Inspection & Decoding**: Verified source video properties and installed `opencv-python` strictly into `.venv-training` (preserving the frozen Phase 2A `.venv`).
2. **Deterministic Frame Extraction**: Extracted 16 sequential RGB still frames at 1.0 FPS into `runs/video_detection/motion_extract_testing/frames/`, assigning distinct non-colliding integer IDs `10001..10016` with full SHA-256 digest provenance.
3. **Three-Model GPU Forward Execution**:
   - `Warm-Start Baseline` (Stage 4 blend checkpoint at $384 \times 576$)
   - `Stage 5 Run 1` (256/384 epoch-2 checkpoint at $256 \times 384$)
   - `D-095 Champion` (soup 50/50 checkpoint at $384 \times 576$), recording all 900 proposal queries per frame and `[16, 900, 9]` float32 sigmoid class score vectors.
4. **D-105 Consensus & Spatial Grouping Replay**: Applied `apply_scene_consensus` and `compute_spatial_grouping` across all 16 frames, classifying waste into `Individual Litter`, `Group of Individual Litter`, and `Garbage Pile`.
5. **Interactive GUI Integration**: Added a dedicated top-bar button (`Check Video Frames (data/input)` / `Back to Validation Images (65)`), updated `Filter:` dropdown options, and added `--video-frames` CLI flag for immediate video inspection.

---

## 2. Strict Scope & Taxonomy Guardrails

- **Active Supervised Taxonomy Preserved**: Object-level detection remains strictly locked to the **9 active supervised classes** (`0: cigarette`, `1: plastic_bottle`, `2: plastic_bag`, `3: plastic_wrapper`, `4: plastic_film`, `5: drink_can`, `6: plastic_cup`, `7: styrofoam_piece`, `8: paper`) and the **5 canonical BANTAI categories** (`Plastic`, `Paper`, `Organic`, `Metal`, `Potentially Hazardous`). Zero unsupervised classes (`battery`, `aerosol`, `broken_glass`, `syringe`, `food_waste`) were introduced or proposed.
- **Regional Group Taxonomy**: Strictly layers the three project-level spatial region categories from [docs/SCOPE.md](SCOPE.md):
  - `individual_litter` -> `"Individual Litter"`
  - `group_of_individual_litter` -> `"Group of Individual Litter"`
  - `garbage_pile` -> `"Garbage Pile"`
- **Original Source Integrity**: `data/input/Motion_Extract_Testing.mp4` remains strictly read-only and unmodified. Extracted frames and prediction artifacts are stored under `runs/video_detection/motion_extract_testing/`, which is git-ignored.
- **Zero Test-Split Access**: Held-out test split `test.json` remains completely untouched.
- **Frozen Benchmark Invariants**: All 65 validation images retain exact metrics ($AP_9 = 0.298963$, $AP_5 = 0.492729$, 76 TPs, 63 visible FPs, 36 clean images).

---

## 3. Video Properties & Extracted Frame Sequence

### 3.1 Source Video Metadata
- **File**: `data/input/Motion_Extract_Testing.mp4`
- **File Size**: $33,131,335$ bytes ($31.60$ MB)
- **SHA-256**: `a10bb8ebf616be46f1da0fde85355673f9c90726243c29050c58f0e9d8319602`
- **Resolution**: $1080 \times 1920$ (vertical / portrait orientation)
- **FPS**: $24.027$
- **Total Frames**: $377$ frames
- **Duration**: $15.691$ seconds

### 3.2 Extracted Frame Inventory (1.0 Sample FPS, IDs `10001..10016`)
| Image ID | File Name | Frame Index | Timestamp | Visible Detections ($\ge 0.10$) | Spatial Grouping Breakdown |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **10001** | `frame_0001_t00.0s.jpg` | 0 | 00:00.0s | 36 items | 2 regions (0 piles, 1 groups, 1 indiv) |
| **10002** | `frame_0002_t01.0s.jpg` | 24 | 00:01.0s | 39 items | 1 regions (0 piles, 1 groups, 0 indiv) |
| **10003** | `frame_0003_t02.0s.jpg` | 48 | 00:02.0s | 25 items | 1 regions (0 piles, 1 groups, 0 indiv) |
| **10004** | `frame_0004_t03.0s.jpg` | 72 | 00:03.0s | 32 items | 1 regions (0 piles, 1 groups, 0 indiv) |
| **10005** | `frame_0005_t04.0s.jpg` | 96 | 00:04.0s | 29 items | 1 regions (0 piles, 1 groups, 0 indiv) |
| **10006** | `frame_0006_t05.0s.jpg` | 120 | 00:05.0s | 31 items | 1 regions (0 piles, 1 groups, 0 indiv) |
| **10007** | `frame_0007_t06.0s.jpg` | 144 | 00:06.0s | 44 items | 1 regions (0 piles, 1 groups, 0 indiv) |
| **10008** | `frame_0008_t07.0s.jpg` | 168 | 00:07.0s | 46 items | 1 regions (0 piles, 1 groups, 0 indiv) |
| **10009** | `frame_0009_t08.0s.jpg` | 192 | 00:08.0s | 55 items | 1 regions (0 piles, 1 groups, 0 indiv) |
| **10010** | `frame_0010_t09.0s.jpg` | 216 | 00:09.0s | 33 items | 1 regions (0 piles, 1 groups, 0 indiv) |
| **10011** | `frame_0011_t10.0s.jpg` | 240 | 00:10.0s | 41 items | 1 regions (0 piles, 1 groups, 0 indiv) |
| **10012** | `frame_0012_t11.0s.jpg` | 264 | 00:11.0s | 11 items | 1 regions (0 piles, 1 groups, 0 indiv) |
| **10013** | `frame_0013_t12.0s.jpg` | 288 | 00:12.0s | 24 items | 1 regions (0 piles, 1 groups, 0 indiv) |
| **10014** | `frame_0014_t13.0s.jpg` | 312 | 00:13.0s | 26 items | **1 regions (1 piles, 0 groups, 0 indiv)** |
| **10015** | `frame_0015_t14.0s.jpg` | 336 | 00:14.0s | 23 items | 1 regions (0 piles, 1 groups, 0 indiv) |
| **10016** | `frame_0016_t15.0s.jpg` | 360 | 00:15.0s | 24 items | 1 regions (0 piles, 1 groups, 0 indiv) |

---

## 4. Replay Artifacts & Verification

### 4.1 Saved Files (`runs/video_detection/motion_extract_testing/`)
- `video-showcase.json`: Full metadata, image index, and raw proposal queries for all three models and consensus ($17.56$ MB).
- `video-score-vectors.npz`: Sigmoid class score vectors for D-095 Champion ($16 \times 900 \times 9$, float32).
- `manifest.json`: Cryptographic manifest binding video SHA-256, artifact hashes, and frame summary.

### 4.2 GUI Integration Features
- **One-Click Toggle Button**: `Check Video Frames (data/input)` seamlessly alternates between standard validation images and video frames without restarting the application.
- **Top-Left Axis Title**: Automatically renders `Extracted Video Frame #k (t = MM:SS.s) — Motion_Extract_Testing.mp4` on clean video frames.
- **CLI Direct Launch**: Launching `python scripts/taco_stage5_visual_review.py --gui --video-frames` initializes directly into video frame inspection mode.

---

## 5. Verification Results

- Unit tests: 5 passed in [`tests/test_stage5_d106_video_extraction.py`](file:///c:/Users/Justine/Documents/Python/Project%20Image-object%20Detection/tests/test_stage5_d106_video_extraction.py) in $16.2$s.
- Full Stage 5 discovery: 123 passed in [`tests/test_stage5*.py`](file:///c:/Users/Justine/Documents/Python/Project%20Image-object%20Detection/tests/) in $64.9$s.
- Guarded suite: 417 tests (401 passed, 0 failures, 0 errors, 16 expected skipped fixtures, 0 GPU accesses, 0 optimizer updates) in [`scripts/verify_stage4_guarded.py`](file:///c:/Users/Justine/Documents/Python/Project%20Image-object%20Detection/scripts/verify_stage4_guarded.py).
- Markdown link integrity: 0 broken links across 80 documentation files in [`scripts/check_markdown_links.py`](file:///c:/Users/Justine/Documents/Python/Project%20Image-object%20Detection/scripts/check_markdown_links.py).
