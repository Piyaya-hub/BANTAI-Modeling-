# BANTAI Modeling

BANTAI is an academic intelligent-agent project for street-issue detection and hotspot prioritization. Its broader proposal includes roadside garbage, potholes, road-surface damage, incident management, and frequency-based hotspot prioritization.

This repository's initial development scope is **trash detection on still images only**. The approved starting model is **GroundingDINO-T (Swin-T / Tiny configuration)**, evaluated through pretrained text-guided inference. Hugging Face Transformers is the selected inference library. The exact checkpoint source/revision, configuration, dependencies, Windows runtime compatibility, prompts, thresholds, and proposed processing rules still need verification; publication of the plan does not establish their acceptance or readiness. Training, alternative models, and integration require later decisions based on evaluation evidence.

This repository is dedicated exclusively to AI modelling and computer vision: model evaluation/selection, preprocessing, detection, necessary postprocessing, accuracy/performance evaluation, approved optimization or potential fine-tuning, and reusable detection outputs for future integration. Frontend, backend application modules, central databases, incident management, reporting, and hotspot ranking are outside its implementation responsibility.

The three required output categories are **Garbage Pile**, **Individual Litter**, and **Group of Individual Litter**. Images may contain multiple independent regions and mixed categories. These are intended project outputs, not guaranteed pretrained classes. Direct prompting must be evaluated before considering spatial grouping; DBSCAN and other grouping methods are conditional alternatives requiring evidence and explicit authorization. See [scope](docs/SCOPE.md) for definitions and counting requirements.

Execution is intended for the owner’s Windows computer with an owner-reported NVIDIA RTX 2080 Super, preferring GPU with CPU fallback. Local Python, NVIDIA driver, available memory, runtime compatibility, and native viewer operation remain unverified. Checkpoint conditions, safetensors availability, and immutable revision also require verification.

## Current state

Phase 0 is complete. The Phase 1 planning deliverable is published and merged through PR #3 at `b366f76`. Environment preparation and detection remain unverified; publication is not acceptance of every technical proposal. Current work is documentation correction only: Codex handles the corrections, and Antigravity will handle later authorized implementation.

The owner supplies unannotated team photographs, manually downloaded/extracted from Google Drive into a configurable project folder (proposed `data/input/`). The planned detection activity is functional checking and visual inspection, with no required annotation work, benchmark splits, precision/recall/mAP, or numeric accuracy targets.

The planned outputs are an in-memory display of original photographs with predicted boxes, category labels, and individual model-score percentages; readable summaries; per-image JSON; and run/environment metadata. Preserve originals and do not automatically save annotated copies. Model-score percentages are not accuracy, calibrated correctness probabilities, or image-area coverage. Inference and display outcomes are recorded separately under the [proposed output contract](docs/PHASE_1_IMPLEMENTATION_PLAN.md#7-proposed-json-contract). No model has been implemented or evaluated by this documentation work.

## Documentation

- [Agent and collaboration rules](AGENTS.md)
- [Scope and exclusions](docs/SCOPE.md)
- [Proposed model inference architecture](docs/ARCHITECTURE.md)
- [Phases and acceptance criteria](docs/PHASES.md)
- [Approved decisions, proposals, and unresolved choices](docs/DECISIONS.md)
- [Validation and evaluation plan](docs/TEST_PLAN.md)
- [Current state and handoff](docs/HANDOFF.md)
- [Phase 1 implementation plan for review](docs/PHASE_1_IMPLEMENTATION_PLAN.md)

## Next step

Phase 0 is complete. Review the [Phase 1 implementation plan](docs/PHASE_1_IMPLEMENTATION_PLAN.md) for Antigravity. The current task authorizes documentation correction only. The planning deliverable describes functional visual checks on unannotated team photos. Implementation, installation, and inference still require explicit authorization. There are no implemented application startup or model evaluation commands at this stage.
