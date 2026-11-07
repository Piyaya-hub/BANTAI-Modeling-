# BANTAI Project Development Phases

> **System Scope**: Real-time academic street-issue detection and hotspot prioritization. Initial repository implementation strictly bounded to **trash detection on still images and extracted video frames only**.  
> **Repository Boundary**: AI modelling and computer vision only. Application frontend/backend modules, incident management, central databases, reporting, and municipal hotspot ranking belong to the broader municipal proposal.  
> **Taxonomy Guardrails**: Strictly locked to 9 active supervised classes (`0: cigarette`, `1: plastic_bottle`, `2: plastic_bag`, `3: plastic_wrapper`, `4: plastic_film`, `5: drink_can`, `6: plastic_cup`, `7: styrofoam_piece`, `8: paper`) and 5 canonical BANTAI categories (`Plastic`, `Paper`, `Organic`, `Metal`, `Potentially Hazardous`). Zero unsupervised classes (`battery`, `aerosol`, `broken_glass`, `syringe`, `food_waste`).

---

## Master Phase Directory

The research, implementation, and optimization of the BANTAI trash detection engine progressed across five major phases. All detailed experimental findings, ablation tables, convergence curves, and diagnostic audits are consolidated into the five phase documents below:

| Phase | Phase Title & Focus | Scope & Milestone Range | Consolidated Documentation |
|---|---|---|---|
| **Phase 1** | **Planning, Environment Readiness & Data Preparation** | Project boundary formalization, hardware snapshot (`GTX 1660 Ti`, Python 3.11.9), coordinate system definition, and TACO dataset/split preparation (`D-001` through `D-020`). | [Phase 1 Guide](PHASE_1_PLANNING_AND_READINESS.md) |
| **Phase 2** | **Baseline Zero-Shot Evaluation & Visual Review** | Zero-shot Grounding DINO Swin-T baseline, text prompt optimization, local TkAgg visual review viewer, and unannotated street photo inspection (`D-021` through `D-039`). | [Phase 2 Guide](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md) |
| **Phase 3** | **Adapter Synthesis & Synthetic Qualification** | Swin-T text/vision adapter design, frozen encoder routing, detached decoder initialization, and synthetic gradient qualification without library tampering (`D-040`). | [Phase 3 Guide](PHASE_3_ADAPTER_AND_SYNTHETIC_QUALIFICATION.md) |
| **Phase 4** | **8-Class Bounded Real-Image Fine-Tuning** | Section 185 academic claim (`D-042`), coordinate inversion bugfix, memory pilot, learning rate schedules, and high-resolution $384 \times 576$ training (`D-041` through `D-069`). | [Phase 4 Guide](PHASE_4_EIGHT_CLASS_FINE_TUNING.md) |
| **Phase 5** | **9-Class Consensus, Spatial Grouping & Integration** | 9th class (`paper`) expansion, D-095 Champion soup ($AP_9 = 0.301350$, $AP_5 = 0.497161$), Three-Model Consensus engine, 1.0 FPS video extraction, D-112 disjoint spatial grouping, and Argon website integration (`D-070` through `D-116`). | [Phase 5 Guide](PHASE_5_NINE_CLASS_CONSENSUS_AND_SPATIAL_GROUPING.md) |

---

## Executive Phase Summaries

### [Phase 1: Planning, Environment Readiness & Data Preparation](PHASE_1_PLANNING_AND_READINESS.md)
Phase 1 established the engineering foundations and legal/scope boundaries:
- **Scope Demarcation**: Locked Model A (trash detection) to still images and video frames; excluded municipal backend services and Model B (potholes/road cracks).
- **Coordinate Conventions**: Standardized bounding box geometry to standard COCO format `[x_min, y_min, width, height]` relative to stored raster dimensions without EXIF auto-transposition.
- **Split Hygiene**: Formalized the zero-test-split access rule: `test.json` strictly held out.
- **Hardware Budget**: Targeted NVIDIA GeForce GTX 1660 Ti (6144 MiB VRAM), driver 472.19, on Windows 10/11 x64 with Python 3.11.9.

### [Phase 2: Baseline Zero-Shot Evaluation & Visual Review](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md)
Phase 2 evaluated off-the-shelf Grounding DINO Swin-T via zero-shot prompt engineering:
- **Prompt Formulation**: Evaluated direct text prompts (`garbage pile . individual litter . group of individual litter .`).
- **Interactive Verification**: Developed native TkAgg interactive visual review viewer with zoom/pan, selectable detection tables, and atomic display status tracking.
- **Baseline Findings**: Zero-shot inference exhibited severe category confusion between transparent plastics and background pavement, establishing the imperative for supervised fine-tuning.

### [Phase 3: Adapter Synthesis & Synthetic Qualification](PHASE_3_ADAPTER_AND_SYNTHETIC_QUALIFICATION.md)
Phase 3 designed the parameter-efficient adaptation architecture:
- **Architectural Routing**: Introduced custom text-guided projection layers while preserving frozen Hugging Face Swin-T backbone weights.
- **Synthetic Qualification**: Validated backward gradient propagation and loss convergence on synthetic image-text tensors before touching any real training data.
- **Invariance Rule**: Strict prohibition on patching installed Hugging Face Transformers library files.

### [Phase 4: 8-Class Bounded Real-Image Fine-Tuning](PHASE_4_EIGHT_CLASS_FINE_TUNING.md)
Phase 4 conducted the first real-image fine-tuning runs on TACO:
- **Legal Authorization**: Recorded owner non-commercial academic research declaration under Section 185 (`D-042`).
- **Coordinate Fix**: Diagnosed and repaired a critical coordinate normalization inversion bug (`D-044`/`D-045`).
- **Scale Experimentation**: Scaled input resolution from $256 \times 384$ to $384 \times 576$, dramatically recovering small-object recall on bottle caps and cigarette butts.
- **Convergence Verification**: Established cosine learning rate schedule with warm-up, achieving stable multi-epoch convergence across seeds 42, 43, and 44.

### [Phase 5: 9-Class Consensus, Spatial Grouping & Integration](PHASE_5_NINE_CLASS_CONSENSUS_AND_SPATIAL_GROUPING.md)
Phase 5 brought the engine to its production-ready state:
- **Taxonomy Expansion**: Expanded active supervised classes to 9 by adding `paper` (achieving $AP_{paper} = 0.640610$).
- **Model Soup Champion (D-095)**: Blended multi-epoch box-contrastive checkpoints (50/50 uniform weight interpolation), securing frozen validation highs of **$AP_9 = 0.301350$** and **$AP_5 = 0.497161$**.
- **Three-Model Consensus Engine**: Triangulated predictions across `D-095 Champion`, `Stage 5 Run 1`, and `Warm-Start Baseline` to veto background pavement false positives while preserving true litter.
- **Video Frame Extraction**: Implemented uniform 1.0 FPS RGB JPEG extraction pipeline (`scripts/taco_stage5_video_extract_and_detect.py`).
- **Scale-Normalized Disjoint DBSCAN**: Enforced strict non-overlapping multi-item hazard boundaries under D-112 cardinality:
  * `Individual Litter` ($k = 1$, `model_score = float(score)`)
  * `Group of Individual Litter` ($2 \le k \le 7$, `model_score = null`)
  * `Garbage Pile` ($k \ge 8$ contiguous heap, `model_score = null`)
- **Production Integration**: Published [Argon Website Integration Guide](ARGON_WEBSITE_INTEGRATION_GUIDE.md), preserved all production weights, and cleaned 128.08 GB of redundant testing files.

---

## Core Architecture & Governance Directory

For technical specifications, system architecture, API schemas, evaluation methodology, and integration handoff, consult the core documentation set:

- **[SCOPE.md](SCOPE.md)**: System boundaries, non-commercial academic basis, Model A vs Model B demarcation, and taxonomy rules.
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Neural network architecture, Swin-T backbone, Three-Model Consensus routing, and scale-normalized DBSCAN spatial grouping.
- **[SCHEMA.md](SCHEMA.md)**: Complete JSON schemas for object candidates, regional groupings, run metadata, and unified dual-model API responses.
- **[EVALUATION.md](EVALUATION.md)**: Evaluation methodology, COCO AP metrics, 5-category aggregation rules, visual review protocols, and split hygiene.
- **[DECISIONS.md](DECISIONS.md)**: Complete chronological decision register from `D-001` through `D-116`.
- **[HANDOFF.md](HANDOFF.md)**: Production deployment guide and operational handoff report for engineering teams.
- **[ARGON_WEBSITE_INTEGRATION_GUIDE.md](ARGON_WEBSITE_INTEGRATION_GUIDE.md)**: Master integration guide for full-stack engineer "Argon", detailing video extraction, FastAPI/Flask service adapters, CLI execution, and municipal hotspot index calculations.
