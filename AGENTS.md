# BANTAI agent instructions

## Project boundary

BANTAI is an academic intelligent-agent project for street-issue detection and hotspot prioritization. The initial development scope of this repository is **trash detection on still images only**. Roadside garbage, potholes, road-surface damage, incident management, and frequency-based hotspot prioritization belong to the broader proposal; they are not all approved implementation work here.

Phase 0/Phase 1 planning, Phase 2A readiness and Phase 2B foundation are published; PR #6 merged the foundation. The owner's latest instruction explicitly authorizes **Phase 2C**, superseding the previous stop-before-2C boundary. Codex implements the baseline described in [docs/PHASES.md](docs/PHASES.md): pinned model/processor loading and inference, exact phrase mapping, conservative duplicates/review flags, native in-memory viewing and the persisted display lifecycle. Preserve the verified environment and requirements; do not reinstall packages or silently change versions. Keep default CLI preparation safe, with inference selected explicitly. Preserve originals, unresolved physical/region counts and strict JSON diagnostics. No automatic grouping, training/fine-tuning, web/application modules, annotated exports or later optimization phase. Commit and publish a focused PR if possible; do not merge. Routine authorized work needs no repeated confirmation.

## Approved modelling direction

This repository owns AI modelling and computer vision only, including future reusable detection outputs; application frontend/backend modules, incident management, central databases, reporting, and hotspot ranking are outside its implementation responsibility.

Start with pretrained **GroundingDINO-T (Swin-T / Tiny configuration)** through **Hugging Face Transformers** and direct text-guided detection of `garbage_pile` (Garbage Pile), `individual_litter` (Individual Litter), and `group_of_individual_litter` (Group of Individual Litter). Preserve multiple independent regions and mixed categories within one image. These outputs are not guaranteed pretrained classes. Evaluate direct prompting before any grouping implementation. DBSCAN, scale-normalized proximity, custom bounding-box grouping, and hybrid methods remain conditional candidates requiring baseline evidence and owner authorization.

Phase 2A preparation must verify exact checkpoint/configuration/conditions/safetensors/revision, local Windows Python/driver/memory/runtime compatibility, viewer operation, and the proposed prompts, thresholds, visual-review boundaries, and output rules. Owner-supplied Windows Command Prompt output now identifies a GTX 1660 Ti, 6144 MiB VRAM and driver 472.19; this supersedes the earlier RTX 2080 Super assumption. Initial `py`/Store-alias checks failed; subsequent supplied output confirms Python 3.11.9 x64 installed and a local `.venv` reporting 3.11.9. Supplied CPU/CUDA tensor, dependency-consistency and direct-library import checks have passed. Phase 2A technical preparation is now complete on supplied evidence: native TkAgg rendering/normal return, pinned checkpoint/processor metadata, official driver reference and exact dependency/source capture also passed; no model inference or memory-fit result is claimed. Record supplied observations separately from direct execution; do not use the cloud environment as Windows evidence or assume CUDA/model compatibility from hardware identification.

The owner manually downloads/extracts unannotated team photos from Google Drive into a project folder. Functional visual checks do not require annotations, ground-truth boxes, benchmark splits, precision/recall/mAP, or numeric accuracy targets. Do not add data collection or Drive integration. Plan in-memory boxes/labels/individual model-score percentages, readable summaries, per-image JSON and run/environment metadata; preserve originals and disable automatic annotated-image exports. A score percentage is not accuracy, calibrated correctness probability, or area coverage.

Follow the [output contract](docs/PHASE_1_IMPLEMENTATION_PLAN.md#7-proposed-json-contract): invalid numeric values become null with diagnostics, invalid candidates are excluded from displayed boxes/valid summaries, and display status is separate from inference status. Preserve unmapped predictions and unresolved counts; do not invent model confidence or membership. Do not infer successful detection from a model selection or written requirement.

## Collaboration rules

- ChatGPT supports requirements analysis, explanations, and proposed decisions. Label suggestions as proposals; do not present them as approvals or measured results.
- Codex owns authorized Phase 2C baseline implementation and checks. Separate software fixtures, cloud executions and actual target/model observations; report missing runtime/photo/desktop evidence explicitly. Preserve existing files and user changes, including `.gitignore`; never commit credentials or unrelated artifacts.
- Antigravity is no longer the implementation owner. Per the owner it made no changes; preserve earlier assignments as history in the decision register rather than attributing unverified work to it. The owner authorizes later phases, not tool suggestions.
- All three tools must read this file and the relevant documents before work, distinguish facts from assumptions, and record unresolved choices rather than inventing answers.
- The project owner approves scope changes, phase transitions, and decisions. Tools may propose changes through a pull request but must not claim owner approval or merge without authorization.
- Use the existing isolated checkout; do not create a Git worktree unless requested. Do not change `main` directly. Use focused branches and pull requests targeting `main`.
- Do not claim that a model has been trained, evaluated successfully, or benchmarked without recorded evidence. Do not invent exact model versions, datasets, or benchmark results.

## Documentation maintenance

Keep approved decisions in [docs/DECISIONS.md](docs/DECISIONS.md) separate from proposals and unresolved questions. Update [docs/HANDOFF.md](docs/HANDOFF.md) with the actual state, evidence, blockers, and next authorized action. Check relative Markdown links and phase acceptance criteria before committing.

Start with [README.md](README.md), [scope](docs/SCOPE.md), and [phases](docs/PHASES.md). For future authorized model work, follow the [inference architecture](docs/ARCHITECTURE.md) and [test plan](docs/TEST_PLAN.md).
