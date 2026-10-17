# BANTAI Modeling

BANTAI is an academic intelligent-agent project for street-issue detection and hotspot prioritization. Its broader proposal includes roadside garbage, potholes, road-surface damage, incident management, and frequency-based hotspot prioritization.

This repository's initial development scope is **trash detection on still images only**. The approved starting model is **GroundingDINO-T (Swin-T / Tiny configuration)**, evaluated through pretrained text-guided inference. Hugging Face Transformers is the selected inference library. Phase 2A has recorded the exact checkpoint/processor revision, settings, installed dependencies and basic Windows runtime readiness. Prompts, thresholds, detector processing rules and actual model execution remain unvalidated; publication of the plan does not establish their acceptance or readiness. Training, alternative models, and integration require later decisions based on evaluation evidence.

This repository is dedicated exclusively to AI modelling and computer vision: model evaluation/selection, preprocessing, detection, necessary postprocessing, accuracy/performance evaluation, approved optimization or potential fine-tuning, and reusable detection outputs for future integration. Frontend, backend application modules, central databases, incident management, reporting, and hotspot ranking are outside its implementation responsibility.

The three required output categories are **Garbage Pile**, **Individual Litter**, and **Group of Individual Litter**. Images may contain multiple independent regions and mixed categories. These are intended project outputs, not guaranteed pretrained classes. Direct prompting must be evaluated before considering spatial grouping; DBSCAN and other grouping methods are conditional alternatives requiring evidence and explicit authorization. See [scope](docs/SCOPE.md) for definitions and counting requirements.

Execution is intended for the owner’s Windows computer, preferring GPU with CPU fallback. Owner-supplied Command Prompt output now reports Windows 10 build 19045.6466, a GTX 1660 Ti with 6144 MiB VRAM, driver 472.19, and about 5235–5236 MiB free at the time of the query. This replaces the earlier RTX 2080 Super assumption. Git works. After the initial Store-alias/launcher issue, supplied output confirms Python 3.11.9 x64 installation, a successful local clone and `.venv` reporting Python 3.11.9. Supplied checks now show CPU/CUDA tensor results of 2.0, CUDA available, capability (7,5), matching direct library imports and no broken requirements. Native TkAgg rendering/normal return, checkpoint/processor metadata, official driver-reference comparison and exact dependency/source capture are also confirmed. Phase 2A technical preparation checks are complete; model-memory fit, detector execution and accuracy remain untested; see the [preflight record](docs/PHASE_2A_READINESS.md). Checkpoint metadata reports Apache-2.0, non-gated access and safetensors at immutable revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`; future model and processor loading must use that same revision.

## Current state

Phase 0 is complete. The Phase 1 planning deliverable is published and merged through PR #3 at `b366f76`. Phase 2A technical preparation checks are complete on owner-supplied Windows evidence; detection remains unimplemented and unverified; publication is not acceptance of every technical proposal. Codex now owns implementation, replacing Antigravity, which the owner reports made no changes. Phase 2A is authorized for local Windows environment setup and readiness verification only (plan Section 4 and Section 8 stages 2–3); detector implementation and inference remain outside this authorization.

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
- [Phase 1 implementation plan](docs/PHASE_1_IMPLEMENTATION_PLAN.md)
- [Phase 2A Windows preparation and readiness record](docs/PHASE_2A_READINESS.md)
- [Verified Phase 2A environment snapshot](docs/PHASE_2A_ENVIRONMENT_SNAPSHOT.md)

## Next step

Review the completed [Phase 2A readiness record](docs/PHASE_2A_READINESS.md) and [exact environment snapshot](docs/PHASE_2A_ENVIRONMENT_SNAPSHOT.md). Local Windows commands were owner-run and their outputs reviewed by Codex from the separate cloud checkout. Stop after preparation: detector implementation, weight downloads/loading and inference require a later phase authorization. No further setup command is needed for the unchanged verified environment.
