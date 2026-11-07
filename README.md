# BANTAI Modeling (Stage 5 Production)

> **Intelligent Agent Platform for Municipal Street-Issue Detection, Spatial Hazard Grouping & Hotspot Prioritization**  
> **Repository Scope**: AI modeling and computer vision only (Model A: Trash Detector & Spatial Hazard Grouping).  
> **Master Documentation**: For full architecture, benchmark reports, backend integration recipes (FastAPI/Flask/Django/Celery), and dual-model JSON schemas, see [Master Integration Guide](docs/ARGON_WEBSITE_INTEGRATION_GUIDE.md).

---

## 1. Executive Summary

This repository contains the fine-tuned Grounding DINO Swin-T object detection model, multi-model consensus pipeline, and scale-normalized disjoint spatial grouping algorithms for automated street-litter detection. It is engineered to operate seamlessly alongside a complementary road-surface hazard model (Model B: potholes, cracks) within a unified municipal web application.

- **Dual-Model Shared Ingestion**: Single video ingestion extracts 1.0 FPS RGB JPEG frames shared between Model A and Model B, eliminating redundant video decoding and duplicate storage.
- **Active Supervised Taxonomy**: Strictly locked to **9 classes** (`0: cigarette`, `1: plastic_bottle`, `2: plastic_bag`, `3: plastic_wrapper`, `4: plastic_film`, `5: drink_can`, `6: plastic_cup`, `7: styrofoam_piece`, `8: paper`).
- **Canonical Categories**: Mapped to the 5 municipal BANTAI categories (`Plastic`, `Paper`, `Organic`, `Metal`, `Potentially Hazardous`).
- **Zero Unsupervised Classes**: Zero support for unaudited classes (`battery`, `aerosol`, `broken_glass`, `syringe`, `food_waste`).
- **Zero Test-Split Access**: Strict verification against frozen validation data without ever opening held-out test splits.

---

## 2. Final Benchmark Performance

Evaluated on the frozen 65-image validation benchmark:

| Metric | Measured Value | Status |
| :--- | :---: | :--- |
| **Official 9-Specific AP** | **`0.301350`** | Benchmark High Preserved |
| **BANTAI 5-Category AP** | **`0.497161`** | Benchmark High Preserved |
| **Paper Category AP** | **`0.640610`** | Stage 5 High Preserved |
| **True Positives** | **`77`** | 0 Lost TPs across all 65 images |
| **False Positives** | **`63`** | Clean background suppression |
| **Clean Images** | **`36 / 65`** (55.4%) | 0 false alarms on background |

---

## 3. Scale-Normalized Disjoint Spatial Grouping (D-112 Standard)

Detections are clustered into municipal hazard regions with non-overlapping bounding box guarantees:

- **`Individual Litter` (`individual_litter`)**: $k = 1$ isolated bounding box (`model_score = float(score)`).
- **`Group of Individual Litter` (`group_of_individual_litter`)**: $2 \le k \le 7$ spatially clustered items ($k \in \{2, 3, 4, 5, 6, 7\}$, `model_score = null`).
- **`Garbage Pile` (`garbage_pile`)**: $k \ge 8$ bounding boxes in a dense, touching/consolidated heap (`model_score = null`).

---

## 4. Preserved Production Weights

Following the D-113 surgical cleanup (128.08 GB freed), production weights are preserved in `runs/training/`:

1. **`D-095 Champion` (box-contrastive soup 50/50)**:  
   `runs/training/stage5-d095-box-contrastive-20261102-a/checkpoints/best-inference/` (`model.safetensors`, `config.json`, `generation_config.json`, `soup_manifest.json`)
2. **`Stage 5 Run 1` (epoch-02, 256/384 high-recall small-litter detector)**:  
   `runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/checkpoints/epoch-02/` (`model.safetensors`, `config.json`)
3. **`Warm-Start Baseline` (Stage 4 blend, 384/576 resolution)**:  
   `runs/training/taco-896e2f44146d4d54aee518e6087908ca/checkpoints/` (`step-000350` and `epoch-02`)

---

## 5. Quick Start for Website & API Integration

### Command-Line Video / Photo Processing
```bash
# Process video or image: extracts frames, runs 3-model detection + consensus + spatial grouping
python scripts/taco_stage5_video_extract_and_detect.py \
  --input-media "data/input/uploaded_street_video.mp4" \
  --output-dir "runs/video_detection/incident_001" \
  --sample-fps 1.0 \
  --score-threshold 0.15 \
  --top-k 25 \
  --device "cuda"
```

### Python Backend Adapter Recipe
For complete copy-pasteable Python service classes (`BantaiTrashService`), FastAPI endpoints, and unified JSON response schemas combining Model A (trash) and Model B (potholes), see [Argon Website Integration Guide](docs/ARGON_WEBSITE_INTEGRATION_GUIDE.md).

---

## 6. Verification & Health Checks

Verify repository health and test discovery:

```powershell
# 1. Run core spatial grouping & cardinality tests
.\.venv-training\Scripts\python.exe -m unittest tests/test_stage5_d112_garbage_pile_min8_rule.py

# 2. Run full guarded test suite (455 tests)
.\.venv-training\Scripts\python.exe scripts/verify_stage4_guarded.py

# 3. Check markdown link integrity (0 broken links)
.\.venv-training\Scripts\python.exe scripts/check_markdown_links.py
```

---

## 7. Complete Documentation Directory

### Core Architecture & Governance
- **[Master Integration Guide for Argon](docs/ARGON_WEBSITE_INTEGRATION_GUIDE.md)**: Master engineering guide for full-stack engineer Argon, detailing video frame extraction, dual-model integration recipes, and municipal hotspot scoring.
- **[System Scope & Boundaries](docs/SCOPE.md)**: Official project boundaries, Model A vs Model B demarcation, non-commercial academic research basis, and taxonomy guardrails.
- **[System Architecture](docs/ARCHITECTURE.md)**: Detailed neural network architecture, Swin-T backbone, Three-Model Consensus engine, and scale-normalized DBSCAN spatial grouping.
- **[JSON Schemas & Contracts](docs/SCHEMA.md)**: Formal JSON contracts for object detections, regional spatial hazard groupings, diagnostic telemetry, and unified dual-model API responses.
- **[Evaluation Methodology](docs/EVALUATION.md)**: Standardized validation protocol, COCO AP metrics, 5-category aggregation rules, visual review protocols, and split hygiene.
- **[Phases Directory](docs/PHASES.md)**: Master roadmap linking all 5 research and development phases.
- **[Decision Register](docs/DECISIONS.md)**: Complete chronological register of all architectural decisions and milestones (`D-001` through `D-116`).
- **[Handoff Report](docs/HANDOFF.md)**: Operational handoff report and deployment guide for Argon, backend engineers, and municipal stakeholders.
- **[Agent Collaboration Rules](AGENTS.md)**: Multi-agent coordination protocols and repository operating standards.

### Consolidated Phase Progress Documents
- **[Phase 1: Planning, Environment Readiness & Data Preparation](docs/PHASE_1_PLANNING_AND_READINESS.md)**: Concept proposal, hardware readiness verification, and initial dataset analysis (`D-001` through `D-020`).
- **[Phase 2: Baseline Zero-Shot Evaluation & Visual Review](docs/PHASE_2_BASELINE_AND_VISUAL_REVIEW.md)**: Zero-shot prompt evaluation, interactive TkAgg review viewer, and unannotated photo inspection (`D-021` through `D-039`).
- **[Phase 3: Adapter Synthesis & Synthetic Qualification](docs/PHASE_3_ADAPTER_AND_SYNTHETIC_QUALIFICATION.md)**: Swin-T text/vision adapter design, encoder freezing, and synthetic gradient qualification (`D-040`).
- **[Phase 4: 8-Class Bounded Real-Image Fine-Tuning](docs/PHASE_4_EIGHT_CLASS_FINE_TUNING.md)**: Section 185 academic claim, coordinate bugfix, learning rate schedules, and high-resolution $384 \times 576$ training (`D-041` through `D-069`).
- **[Phase 5: 9-Class Consensus, Spatial Grouping & Integration](docs/PHASE_5_NINE_CLASS_CONSENSUS_AND_SPATIAL_GROUPING.md)**: 9th class (`paper`) expansion, D-095 Champion soup, three-model consensus, video extraction, and scale-normalized disjoint spatial grouping (`D-070` through `D-116`).
