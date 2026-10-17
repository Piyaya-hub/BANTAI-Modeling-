# Decision register

## Historical Phase 0 decisions — retained record

Source: the project owner's Phase 0 task instructions. These entries preserve their original wording; they are not all current instructions. D-002 is refined by D-007/D-014; D-003's Phase 0-only task state is superseded by D-020; D-004's preserved original ignore rules now permit additive exclusions under D-019. No entry is experimental evidence.

| ID | Decision | Consequence |
| --- | --- | --- |
| D-001 | Initial repository development scope is trash detection only. | Broader street-issue and hotspot features remain outside initial implementation. |
| D-002 | Evaluate a pretrained Grounding DINO-family model on still images first. | Do not start with training, alternative models, or integration. Exact model selection is unresolved. |
| D-003 | Current work is Phase 0 documentation only. | No detection code, ML dependencies, model downloads, frontend, backend, or Phase 1 execution. |
| D-004 | Use a documentation branch and review against `main`; preserve `.gitignore`. | No direct `main` changes or pull request merge in this task. |
| D-005 | Separate approvals, proposals, unresolved choices, and measured evidence. | Do not invent model versions, benchmark results, or successful training/evaluation claims. |

## Historical final Phase 0 decisions — 2026-10-17

Source: the project owner's final Phase 0 revision request. The original D-001–D-005 entries above are retained as history. These approvals established direction, not successful model behavior. D-008’s annotation wording is superseded for this activity by D-016. D-013’s task-specific branch/pending-review status is superseded by D-020 and merged PR #2; its no-direct-main/no-merge rule remains.

| ID | Approved decision | Consequence |
| --- | --- | --- |
| D-006 | Repository responsibility is AI modelling and computer vision only. | Future work may include evaluation, preprocessing, detection, necessary postprocessing, approved optimization/fine-tuning, and reusable detection outputs. Frontend/backend application modules, central databases, reporting, incidents, and hotspot ranking are outside this repository. Supersedes any earlier Phase 4 suggestion to implement those modules here. |
| D-007 | Start with pretrained GroundingDINO-T (Swin-T / Tiny configuration). | Refines D-002: the model configuration family is now selected; exact implementation, checkpoint source/revision, configuration, dependencies, and local compatibility remain unresolved for Phase 1. No training from scratch. |
| D-008 | Require Garbage Pile, Individual Litter, and Group of Individual Litter outputs. | Definitions and counting example are in [SCOPE.md](SCOPE.md); operational annotation/grouping/overlap rules remain unresolved. These are not guaranteed pretrained classes. |
| D-009 | Evaluate direct text-guided detection first. | Investigate separate and combined prompts before adding spatial grouping. Illustrative vocabulary is not final. |
| D-010 | Support multiple independent regions and mixed categories in one image. | Preserve objects versus regions and prevent double-counting groups and their members as independent regions; actual direct-model capability remains experimental. |
| D-011 | Grouping is conditional on baseline evidence and explicit owner authorization. | DBSCAN, scale-normalized proximity, custom bounding-box grouping, and hybrid methods remain candidates, not mandatory components. |
| D-012 | Available development GPU reported by owner: NVIDIA RTX 2080 Super. | Verify actual runtime availability, dependency compatibility, and resource needs during Phase 1; no local hardware validation is claimed. |
| D-013 | Final Phase 0 revisions use a new branch and pull request from latest `main`. | PR #1 is merged; use `docs/finalize-phase-0-detection-scope`, do not reuse that PR, merge, or begin Phase 1. Owner review of these revisions remains pending. |

## Current owner-confirmed requirements

Source: the owner's Phase 1 planning request, publication request, and current documentation-correction request. No additional approval date or performance evidence is inferred. These are owner requirements; technical defaults below remain proposals.

| ID | Confirmed requirement | Replaces or clarifies |
| --- | --- | --- |
| D-014 | Use pretrained Grounding DINO Tiny through Hugging Face Transformers; no baseline training/fine-tuning. | Resolves the inference-library uncertainty in D-002/D-007. Exact checkpoint/configuration/conditions/safetensors/revision still require verification. |
| D-015 | Execute on the owner’s Windows computer with owner-reported RTX 2080 Super, GPU preferred and CPU fallback. | Refines D-012; local Python/driver/memory/runtime/viewer remain unverified. Remote cloud state is not local evidence. |
| D-016 | Owner manually downloads/extracts unannotated team photos from Google Drive for functional detection checking and visual inspection. | Supersedes mandatory annotation policy, held-out splits, formal metrics, and numerical accuracy targets in earlier test/phase guidance and D-008's prerequisite wording. No collection or Drive integration. |
| D-017 | Display boxes, category labels and individual model-score percentages in memory; provide readable summaries, per-image JSON, and appropriate run/environment metadata. | Percentages are model scores, not accuracy, calibrated correctness probabilities, or image-area coverage. Source photos stay unchanged; no automatic annotated-image copies. |
| D-018 | Codex corrects documentation; Antigravity performs later authorized implementation. | The correction does not authorize installation, weight downloads, inference, or Phase 1 acceptance. Retain agreed phase boundaries without new approval gates for routine defaults. |
| D-019 | Preserve original ignore rules and append missing `/data/input/`, `/results/`, `/.cache/huggingface/` exclusions. | Additive clarification to D-004; no directory creation, silent untracking, or moving photos. |
| D-020 | Phase 0 is complete and Phase 1 planning is published through merged PR #3 (`b366f76`). Current work is documentation correction on a focused branch. | Replaces D-003/D-013 task-state wording and old pending-merge claims. Publication is not environment readiness, successful detection, or approval of every proposal. No main changes or merge. |
| D-021 | Functional outputs must be standards-compliant JSON with invalid numbers represented as null plus diagnostics, and display outcome separate from inference outcome. | Corrects the plan's prior nonfinite-retention ambiguity and missing display lifecycle. Persist inference results before viewing and atomically update display outcome; unfinished pending attempts remain explicit. Detailed contract conventions are in the [plan](PHASE_1_IMPLEMENTATION_PLAN.md#7-proposed-json-contract). |

D-001/D-006 and D-009–D-011 remain current: modelling-only trash still images, direct prompts first, mixed categories/multiple candidate regions, and no automatic grouping in the baseline. Category codes are `individual_litter`, `group_of_individual_litter`, and `garbage_pile`. Predictions do not prove membership or physical-object/independent-region counts; unresolved counts remain null. Application modules, APIs, databases, road anomalies, video/cameras, incident consolidation, and hotspots remain outside scope.

## Technical proposals and pending verification

- P-001: The [architecture](ARCHITECTURE.md) and detailed [plan](PHASE_1_IMPLEMENTATION_PLAN.md) propose the local pipeline and output contract. GroundingDINO-T/Transformers is selected; its checkpoint revision, conditions, and local compatibility are not verified.
- P-002: The current [test plan](TEST_PLAN.md) applies D-016 functional checks. Earlier P-002's mandatory annotated/held-out protocol is superseded; formal annotated evaluation is optional, separately scoped future work.
- P-003: Follow [PHASES.md](PHASES.md). Earlier Phase 0-only task state is superseded by D-020; the present correction is still documentation only.
- P-004: Python 3.11 x64, torch 2.7.1/cu118, torchvision 0.22.1, Transformers 4.51.3, Pillow 11.2.1, Matplotlib 3.10.3 remain proposed pins, supported by prior metadata/source inspection but not locally tested. Resolve exact dependencies and CUDA installation source after authorized local verification.
- P-005: Starting prompts, `threshold=0.35`, `text_threshold=0.25`, exact aliases, visual grouping boundaries, identical-box/same-phrase collapse, and IoU >=0.8 review flags are unvalidated defaults. No general NMS or spatial grouping algorithm is selected. Routine refinements inside an authorized phase are recorded rather than treated as new owner approval gates.

| Area | Remaining evidence or operational choice | Relevant stage |
| --- | --- | --- |
| Checkpoint | Exact source/configuration/conditions/license/safetensors/immutable revision; model and processor at same recorded revision | Later Phase 1 preparation |
| Windows runtime | Python/driver/GPU memory, proposed package pins and exact resolved dependencies/CUDA source, GPU execution or diagnosed CPU fallback, native viewer | Later Phase 1 preparation |
| Functional configuration | Prompts/thresholds, preprocessing, visual category boundaries, aliases, duplicate flags, source/output paths and reproducibility metadata | Preparation and routine implementation in authorized phases |
| Outputs | Implement corrected numeric/status/display lifecycle consistently; keep unmapped/invalid diagnostics and unresolved counts | Later authorized Phase 2 |
| Later work | Whether actual functional limitations justify grouping/hybrids/alternatives or other modelling improvement | Phase 3 decision; not baseline |

## Recording future decisions

For each approval, record the decision, approving owner, rationale, evidence links, affected scope/phase, and the actual approval date only when known; never invent one. A tool suggestion is not an approval. Move a proposal to approved status only after explicit owner approval; preserve superseded decisions with their replacement reference. No training or evaluation results exist in this documentation deliverable.
