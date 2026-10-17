# BANTAI Modeling

BANTAI is an academic intelligent-agent project for street-issue detection and hotspot prioritization. Its broader proposal includes roadside garbage, potholes, road-surface damage, incident management, and frequency-based hotspot prioritization.

This repository's initial development scope is **trash detection on still images only**. The intended first model experiment is evaluation of a pretrained Grounding DINO-family model. The exact model, checkpoint, runtime, prompts, thresholds, and evaluation dataset remain unresolved. Training, alternative models, and integration require later decisions based on evaluation evidence.

## Current state

Phase 0 is documentation only. No object-detection implementation, model download, training, evaluation results, frontend, or backend is provided. No model is claimed to have been trained or successfully tested. A local Python environment from environment onboarding does not establish model readiness.

## Documentation

- [Agent and collaboration rules](AGENTS.md)
- [Scope and exclusions](docs/SCOPE.md)
- [Proposed model inference architecture](docs/ARCHITECTURE.md)
- [Phases and acceptance criteria](docs/PHASES.md)
- [Approved decisions, proposals, and unresolved choices](docs/DECISIONS.md)
- [Validation and evaluation plan](docs/TEST_PLAN.md)
- [Current state and handoff](docs/HANDOFF.md)

## Next step

Review the Phase 0 documentation against its acceptance criteria. Phase 1 may begin only after explicit project-owner authorization. There are no application startup or model evaluation commands at this stage.
