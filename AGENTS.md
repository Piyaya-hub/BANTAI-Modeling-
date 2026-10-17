# BANTAI agent instructions

## Project boundary

BANTAI is an academic intelligent-agent project for street-issue detection and hotspot prioritization. The initial development scope of this repository is **trash detection on still images only**. Roadside garbage, potholes, road-surface damage, incident management, and frequency-based hotspot prioritization belong to the broader proposal; they are not all approved implementation work here.

Phase 0 is complete and Phase 1 planning is published (PR #3 merged). Planning publication does not prove environment readiness, successful detection, or owner acceptance of every proposal. The owner has authorized **Phase 2A: local Windows environment setup and readiness verification**, limited to Section 4 and Section 8 stages 2–3 of the [implementation plan](docs/PHASE_1_IMPLEMENTATION_PLAN.md). Codex now owns implementation, replacing Antigravity; the owner reports Antigravity made no changes. Phase 2A permits isolated local dependency setup, package/device/native-viewer probes, metadata verification and evidence recording. Do not implement detector/application code, load/download model weights or datasets, or run model inference. These remain outside this preparation-only authorization. Follow the agreed phase boundaries in [docs/PHASES.md](docs/PHASES.md); routine implementation defaults do not need additional approval checkpoints within an authorized phase.

## Approved modelling direction

This repository owns AI modelling and computer vision only, including future reusable detection outputs; application frontend/backend modules, incident management, central databases, reporting, and hotspot ranking are outside its implementation responsibility.

Start with pretrained **GroundingDINO-T (Swin-T / Tiny configuration)** through **Hugging Face Transformers** and direct text-guided detection of `garbage_pile` (Garbage Pile), `individual_litter` (Individual Litter), and `group_of_individual_litter` (Group of Individual Litter). Preserve multiple independent regions and mixed categories within one image. These outputs are not guaranteed pretrained classes. Evaluate direct prompting before any grouping implementation. DBSCAN, scale-normalized proximity, custom bounding-box grouping, and hybrid methods remain conditional candidates requiring baseline evidence and owner authorization.

Phase 2A preparation must verify exact checkpoint/configuration/conditions/safetensors/revision, local Windows Python/driver/memory/runtime compatibility, viewer operation, and the proposed prompts, thresholds, visual-review boundaries, and output rules. The owner reports an RTX 2080 Super; do not use the remote cloud environment as evidence about that computer.

The owner manually downloads/extracts unannotated team photos from Google Drive into a project folder. Functional visual checks do not require annotations, ground-truth boxes, benchmark splits, precision/recall/mAP, or numeric accuracy targets. Do not add data collection or Drive integration. Plan in-memory boxes/labels/individual model-score percentages, readable summaries, per-image JSON and run/environment metadata; preserve originals and disable automatic annotated-image exports. A score percentage is not accuracy, calibrated correctness probability, or area coverage.

Follow the [output contract](docs/PHASE_1_IMPLEMENTATION_PLAN.md#7-proposed-json-contract): invalid numeric values become null with diagnostics, invalid candidates are excluded from displayed boxes/valid summaries, and display status is separate from inference status. Preserve unmapped predictions and unresolved counts; do not invent model confidence or membership. Do not infer successful detection from a model selection or written requirement.

## Collaboration rules

- ChatGPT supports requirements analysis, explanations, and proposed decisions. Label suggestions as proposals; do not present them as approvals or measured results.
- Codex owns implementation and the currently authorized Phase 2A preparation: inspect the target, preserve existing environments, execute setup/probes only where the Windows target is accessible, record evidence and blockers, and stop before detector implementation/inference. Preserve existing files and user changes, including `.gitignore`; never commit credentials or unrelated artifacts.
- Antigravity is no longer the implementation owner. Per the owner it made no changes; preserve earlier assignments as history in the decision register rather than attributing unverified work to it. The owner authorizes later phases, not tool suggestions.
- All three tools must read this file and the relevant documents before work, distinguish facts from assumptions, and record unresolved choices rather than inventing answers.
- The project owner approves scope changes, phase transitions, and decisions. Tools may propose changes through a pull request but must not claim owner approval or merge without authorization.
- Use the existing isolated checkout; do not create a Git worktree unless requested. Do not change `main` directly. Use focused branches and pull requests targeting `main`.
- Do not claim that a model has been trained, evaluated successfully, or benchmarked without recorded evidence. Do not invent exact model versions, datasets, or benchmark results.

## Documentation maintenance

Keep approved decisions in [docs/DECISIONS.md](docs/DECISIONS.md) separate from proposals and unresolved questions. Update [docs/HANDOFF.md](docs/HANDOFF.md) with the actual state, evidence, blockers, and next authorized action. Check relative Markdown links and phase acceptance criteria before committing.

Start with [README.md](README.md), [scope](docs/SCOPE.md), and [phases](docs/PHASES.md). For future authorized model work, follow the [inference architecture](docs/ARCHITECTURE.md) and [test plan](docs/TEST_PLAN.md).
