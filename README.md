# BANTAI Modeling

BANTAI is an academic intelligent-agent project for street-issue detection and hotspot prioritization. Its broader proposal includes roadside garbage, potholes, road-surface damage, incident management, and frequency-based hotspot prioritization.

This repository's initial development scope is **trash detection on still images only**. The approved starting model is **GroundingDINO-T (Swin-T / Tiny configuration)**, evaluated through pretrained text-guided inference. The exact implementation, checkpoint source/revision, configuration, dependencies, runtime compatibility, prompts, thresholds, and evaluation dataset remain unresolved for Phase 1. Training, alternative models, and integration require later decisions based on evaluation evidence.

This repository is dedicated exclusively to AI modelling and computer vision: model evaluation/selection, preprocessing, detection, necessary postprocessing, accuracy/performance evaluation, approved optimization or potential fine-tuning, and reusable detection outputs for future integration. Frontend, backend application modules, central databases, incident management, reporting, and hotspot ranking are outside its implementation responsibility.

The three required output categories are **Garbage Pile**, **Individual Litter**, and **Group of Individual Litter**. Images may contain multiple independent regions and mixed categories. These are intended project outputs, not guaranteed pretrained classes. Direct prompting must be evaluated before considering spatial grouping; DBSCAN and other grouping methods are conditional alternatives requiring evidence and explicit authorization. See [scope](docs/SCOPE.md) for definitions and counting requirements.

The owner-reported available development GPU is an NVIDIA RTX 2080 Super; compatibility and availability in the execution environment must be verified during Phase 1.

## Current state

Phase 0 is documentation only. No object-detection implementation, model download, training, evaluation results, frontend, or backend is provided. No model is claimed to have been trained or successfully tested. Phase 1 has not begun.

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

Phase 0 is complete. Review the [Phase 1 implementation plan](docs/PHASE_1_IMPLEMENTATION_PLAN.md) for Antigravity. The latest owner requirements authorize planning and replace the older annotation/benchmark requirements with functional visual checks on unannotated team photos. Implementation, installation, and inference still require explicit authorization. There are no implemented application startup or model evaluation commands at this stage.
