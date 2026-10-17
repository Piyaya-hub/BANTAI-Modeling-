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

Source: the owner's Phase 1 planning, publication, documentation-correction and subsequent Phase 2A instructions. D-018/D-020 retain their prior task wording; D-022/D-023 below supersede the role assignment and documentation-only authorization. No prior decision row is deleted. No additional approval date or performance evidence is inferred. These are owner requirements; technical defaults below remain proposals.

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

## Subsequent owner decisions — current Phase 2A authorization

| ID | Confirmed requirement | Replaces or clarifies |
| --- | --- | --- |
| D-022 | Codex now owns implementation, replacing Antigravity; the owner states Antigravity made no changes. | Supersedes the role assignment in D-018 and the original Antigravity handoff. No independent Antigravity work is asserted or reverted. |
| D-023 | Start Phase 2A: local Windows environment setup/readiness, corresponding exactly to implementation-plan Section 4 and Section 8 stages 2–3. | Supersedes D-020's documentation-only current-task boundary. Allows isolated dependency setup, official metadata checks and package/CPU/GPU/native-viewer probes; no detector implementation, model weights/loading or inference. [PHASES.md](PHASES.md) defines completion/stop criteria. |

## Subsequent target evidence — supplied Windows preflight

The owner supplied Command Prompt output reporting Windows 10 build 19045.6466, Git 2.45.1.windows.1, GTX 1660 Ti, NVIDIA driver 472.19 and 6144 MiB total VRAM (5235–5236 MiB free across successful queries). This supersedes the RTX 2080 Super assumption in D-012/D-015 for the current target, while preserving those rows as history. `py` is not recognized in that shell; Python itself is not yet proven absent. These are supplied observations, not a Codex-executed local check, CUDA compatibility result or accepted readiness decision. See [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md) for next checks.

These decisions are sourced from the owner's latest instruction; no additional approval date or local readiness evidence is invented. Initially missing Windows preflight/results were execution prerequisites, not repeated authorization requirements. Subsequent supplied evidence completed Phase 2A checks as recorded below and in [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md).

D-001/D-006 and D-009–D-011 remain current: modelling-only trash still images, direct prompts first, mixed categories/multiple candidate regions, and no automatic grouping in the baseline. Category codes are `individual_litter`, `group_of_individual_litter`, and `garbage_pile`. Predictions do not prove membership or physical-object/independent-region counts; unresolved counts remain null. Application modules, APIs, databases, road anomalies, video/cameras, incident consolidation, and hotspots remain outside scope.

## Subsequent supplied preparation progress

After the initial Store-alias failure, owner-supplied WinGet output reports Python 3.11.9 x64 installed from python.org with verified installer hash. A new terminal then reports Python 3.11.9 (AMD64), the real Python311 executable and 64 bits. The owner selected a local Documents/Python/Project Image-object Detection folder; clone output succeeded, followed by `.venv` reporting Python 3.11.9. No ML package, device execution, checkpoint or viewer readiness follows from these observations. Earlier proposed pins remain subject to actual package/device checks.

## Phase 2A technical completion evidence — no new scope approval

Supplied results now complete the preparation checks: dependency consistency/direct imports; CPU and CUDA tensors (2.0); native TkAgg rendering and explicit normal return; checkpoint API/model-card/configuration and same-revision processor metadata; official NVIDIA Table 2 driver minimum; and full 36-package freeze plus local environment record. Revision is `a2bb814dd30d776dcf7e30523b00659f4f141c71`, with Apache-2.0 metadata, non-gated access and safetensors availability. Windows driver 472.19 exceeds the documented CUDA 11.8 minor-compatibility minimum 452.39, and the actual tensor test passed. The toolkit's separate 522.06 listing is not that minimum.

These are technical observations and Codex's preparation assessment, not a new owner decision or automatic approval to implement/run a detector. All prior D-001–D-023 rows remain intact. No model-memory or detection-quality claim is made. [PHASE_2A_ENVIRONMENT_SNAPSHOT.md](PHASE_2A_ENVIRONMENT_SNAPSHOT.md) preserves the exact supplied metadata, processor settings, dependency freeze and evidence provenance; [PHASES.md](PHASES.md) records the stop gate.

## Technical proposals and pending verification

- P-001: The [architecture](ARCHITECTURE.md) and [plan](PHASE_1_IMPLEMENTATION_PLAN.md) retain the proposed detector workflow. Checkpoint/processor revision, configuration, license/access metadata, exact dependencies/source and basic Windows package/device/viewer readiness are now verified to the limited 2A scope in the [snapshot](PHASE_2A_ENVIRONMENT_SNAPSHOT.md). Prior unverified wording is historical. Model loading, memory fit, prompt/category quality and inference remain untested; 2B software fixtures do not establish them.
- P-002: The current [test plan](TEST_PLAN.md) applies D-016 functional checks. Earlier P-002's mandatory annotated/held-out protocol is superseded; formal annotated evaluation is optional, separately scoped future work.
- P-003: Follow [PHASES.md](PHASES.md). Earlier Phase 0-only task state was superseded by D-020; current Phase 2C authorization is D-025 below; D-024 remains the historical 2B scope.
- P-004: The proposed baseline is now observed in supplied target results: Python 3.11.9 x64, torch 2.7.1+cu118, torchvision 0.22.1+cu118, Transformers 4.51.3, Pillow 11.2.1 and Matplotlib 3.10.3. Installation/imports and `pip check` passed; CPU/GPU tensors each returned 2.0 with capability (7,5). Full 36-package freeze/source capture, native TkAgg rendering/normal return and pinned checkpoint/processor metadata have subsequently been verified in supplied outputs and preserved in the [snapshot](PHASE_2A_ENVIRONMENT_SNAPSHOT.md); no model compatibility/performance is inferred. Earlier source-only proposal status is retained in Git history.
- P-005: Starting prompts, `threshold=0.35`, `text_threshold=0.25`, exact aliases, visual grouping boundaries, identical-box/same-phrase collapse, and IoU >=0.8 review flags are unvalidated defaults. No general NMS or spatial grouping algorithm is selected. Routine refinements inside an authorized phase are recorded rather than treated as new owner approval gates.

| Area | Remaining evidence or operational choice | Relevant stage |
| --- | --- | --- |
| Checkpoint | Source/configuration/license/safetensors and immutable model/processor revision recorded in snapshot; retain those pins for later loading | Phase 2A complete; preserve recorded evidence |
| Windows runtime | Python/driver/GPU memory, installed versions/full freeze/CUDA source, successful CPU/GPU execution and native viewer recorded in snapshot | Phase 2A complete; preserve recorded evidence |
| Functional configuration | Prompts/thresholds, preprocessing, visual category boundaries, aliases, duplicate flags, source/output paths and reproducibility metadata | Preparation and routine implementation in authorized phases |
| Outputs | Implement corrected numeric/status/display lifecycle consistently; keep unmapped/invalid diagnostics and unresolved counts | Later authorized detector phase, outside 2A |
| Later work | Whether actual functional limitations justify grouping/hybrids/alternatives or other modelling improvement | Phase 3 decision; not baseline |

## Recording future decisions

For each approval, record the decision, approving owner, rationale, evidence links, affected scope/phase, and the actual approval date only when known; never invent one. A tool suggestion is not an approval. Move a proposal to approved status only after explicit owner approval; preserve superseded decisions with their replacement reference. No training or evaluation results exist in this documentation deliverable.

## Phase 2B owner authorization and implementation choices — 2026-10-17

| ID | Owner instruction | Effect |
| --- | --- | --- |
| D-024 | Implement Phase 2B — Input and Output Foundation: configuration, image loading, result-contract utilities and persistence only; synthetic predictions in tests only; preserve verified dependencies; commit/publish PR if possible, no merge. | Supersedes D-023's 2A-only stop boundary for this bounded foundation. No model loading/download/inference, viewer implementation, grouping, training, web/application modules or Phase 2C. Prior decisions/evidence remain historical records. |

Routine implementation choices: separate `bantai.preparation.v1` and `bantai.preparation-run.v1` schemas prevent decoding from being mislabeled functional success. The functional contract adds an explicit `inference.completed` guard. UUID run directories are reserved exclusively; source filenames use a SHA-256 of full source identity plus source-byte digest. External inputs use an absolute image ID because no repository-relative path exists; project inputs use relative IDs. Initial JSON uses atomic hard-link publication without overwrite; final display updates use same-directory `os.replace`. Filesystems lacking hard links fail visibly. Single writer per functional result is required; power-loss directory durability is not claimed. These are software choices, not measured model capability or new owner approval of prompt hypotheses.

The complete 36-package requirements snapshot is transcribed from supplied 2A evidence with the CUDA source retained. No package reinstall/fresh-install verification occurred. Actual 2B cloud software checks and target-version limitations are recorded separately in [PHASE_2B_CHECKS.md](PHASE_2B_CHECKS.md). Historical 2A readiness/snapshot stop wording applies to that earlier task and is superseded by D-024 for the foundation only.

## Phase 2C owner instruction — 2026-10-17

| ID | Owner instruction | Effect |
| --- | --- | --- |
| D-025 | “proceed now with phase 2C” | Supersedes D-024's stop-before-2C boundary. Implement the baseline previously described as the next phase: pinned loading/inference, exact mapping, conservative duplicates/review flags and native in-memory display. Preserve dependencies, originals, contract/evidence boundaries and no automatic merge. No grouping, training, web/application modules or later optimization. |

Codex interprets this phase instruction using the published 2B next-phase description and implementation-plan baseline; no successful model behavior or blanket validation of prompts/thresholds is inferred. Routine choices: preparation remains the default; inference is explicit, with optional cached-only execution. Raw query auditing precedes upstream threshold loss; missing decoded phrases remain empty rather than invented. Duplicate policy now uses identical raw boxes plus normalized phrases and highest-score retention; near duplicates remain. Functional results require an observed pinned Transformers version for completed inference; unavailable metadata is null for pre-model errors. Actual driver is not inferred from old 2A observations. Headless canvas and mock tests remain software evidence; target/model/native-photo execution is unrun. See [PHASE_2C_CHECKS.md](PHASE_2C_CHECKS.md).

## Terminal viewer owner instruction - 2026-10-17

D-026: The owner explicitly requested replacing TkAgg with ANSI/Unicode rendering in the VS Code terminal, results beneath each image, all raw query/non-padding token sigmoid scores, summary/full modes, complete JSON diagnostics, unchanged filtering and pinned model, tests/docs, and a dedicated branch commit without merging. This supersedes the prior native-window display requirement only. No grouping, training or web modules are authorized. PASS/FAIL denotes retention or strict threshold decisions, never measured correctness. See [terminal contract](TERMINAL_VIEWER.md).

## Terminal presentation correction owner instruction - 2026-10-17

D-027: Continue `feat/terminal-viewer-diagnostics`/PR #9; diagnose missing inline output, require IMAGE-first actual terminal rendering and aligned decimal-score tables, compact configurable summary selection, bounded full output, explicit special-token inclusion, separate threshold/policy states, and honest unsupported-terminal outcomes. Preserve raw precision, inference, thresholds, pinned checkpoint, originals and atomic display lifecycle. No new branch or merge. See [presentation contract and evidence](TERMINAL_VIEWER.md).

## Passing-only console and replay owner instruction - 2026-10-18

D-028: The owner supersedes below-threshold/full-dump console selection with passing-only retained query rows and passing ordinary-token rows, preserving all diagnostics in JSON. Authorizes independent synthetic renderer checks and source-verified display-only replay before further inference. Requires honest pending visual status when the actual Windows/VS Code surface cannot be inspected; keep the same branch/PR and no merge. Model revision, thresholds, originals and inference/display persistence remain protected. See [current terminal contract](TERMINAL_VIEWER.md).
