# Decision register

## Controlled corrected-target restart — 2026-10-20

D-048: The owner authorizes exactly one fresh two-epoch Stage 4 experiment from
the original pinned pretrained weights, using corrected valid-image box targets.
Mandatory coordinate/software/source/data/rights/runtime/baseline preflight must
pass before optimizer updates. Preserve all other historical settings: 221/47
train/validation images, batch 1, 256/384 resize, 384-square padding, FP32,
AdamW 1e-5/1e-4, clip 1, seed 42, frozen vision/text backbones, eight-class prompt,
native Stage 3D adapter/loss, 900 queries and corrected COCO protocol v2.
Use the three-step real-image pilot, discard its weights/state, then perform
442 full-run updates. No augmentation, scheduler, accumulation, extra epochs,
test access, dataset/configuration changes, historical resume, dependency changes,
uploads, publication, checkpoint promotion or automatic Git operations.
Compare pretrained, both corrected-protocol historical results, and both new
epochs; retain losses, AP/recall, fixed-threshold diagnostics, score distributions,
memory/timing, checkpoint integrity and preservation evidence in
[the corrected-retrain report](TACO_STAGE_4_CORRECTED_RETRAIN.md).
Stop after epoch 2/final validation and await owner review.

## Explicit opt-in validation viewer — 2026-10-20

D-047: The owner authorizes restoring the existing four-panel TACO viewer through
`taco_visual_review.py --gui`, preserving D-046's terminal-only default.
Optional `--image-id` selects a validation image directly. Reuse verified saved
predictions, ground truth, zoom/pan, labels, scores, raw-phrase tables and diagnostic
controls. No inference/training, protocol/configuration changes, annotated-image
exports, commits, pushes or merges are authorized by this task.

## Terminal-only epoch scores — 2026-10-20

D-046: The owner rejects the epoch diagnostic display and requests terminal output
for each epoch showing updated scores. This supersedes the Step 2 GUI workflow,
without changing the existing functional-inference viewer. Close the diagnostic
window and use verified saved epoch metrics/score distributions in the terminal.
No new training authorization, inference, threshold change or resume exception is
implied. Prior GUI implementation/evidence remains historical and is not launched.

## Stage 4 Step 2 local visual diagnostics — 2026-10-20

D-045: The owner authorizes a standalone local-only four-view validation replay,
systematic representative selection, annotation-relative IoU/classification/miss/
duplicate/confidence diagnostics and initial visual review. Reuse verified saved
records and the approved evaluation-only provenance pathway. Keep raw phrases
separate from assigned classes; diagnostic thresholds do not alter official metrics.
Preserve all user work, inference, strict resume, checkpoints, taxonomy, splits,
originals and environments. No training, optimizer restoration/updates, test access,
downloads, dependency changes, exports/publication, promotion or automatic commits.

[Actual evidence](TACO_STAGE_4_VISUAL_REVIEW.md) distinguishes automated annotation
associations and native runtime checks from pending human photo-level judgments.
Recommendations for a future corrected-target restart/resolution/score investigation
remain proposals requiring later authorization. No further experiment is started.

## Stage 4 Step 1C evaluation-only provenance — 2026-10-20

D-044: The owner explicitly authorizes a separate historical checkpoint evaluation
path that independently verifies original manifests/artifacts/provenance, exact
reviewed current source differences, model/processor/tokenizer/adapter and corpus/
rights compatibility. Preserve strict training-resume verification; never restore
optimizer/RNG/training position, update weights, access test data, publish private
artifacts, promote a checkpoint or automatically commit/push/merge. Qualify using
synthetic fixtures before reevaluating unchanged epoch-1/2 validation models.

[Completed evidence](TACO_STAGE_4_REEVALUATION.md) records corrected metrics, preserved
historical evidence, unchanged strict resume rejection and zero guarded forbidden
operations. Step 2 visual comparison is recommended, not started or implicitly
approved. Corrected evaluation does not repair historical training targets.

## Stage 4 Step 1B coordinate correction — 2026-10-20

D-043: The owner authorizes correction of valid-image target normalization and
native prediction decoding, independent regression tests and validation-only
comparison of pretrained/epoch-1/epoch-2 models. Preserve image processing, adapter,
architecture, loss, scoring policy, taxonomy, splits, checkpoints, owner edits and
environments. No optimizer updates, test access, publication, promotion, automatic
merge or Step 2 visual analysis. Stop checkpoint comparison if integrity/source
binding blocks it; do not falsify provenance or disable checks.

[Actual Step 1B evidence](TACO_STAGE_4_COORDINATES.md): corrections/tests pass;
pretrained reevaluation completed; epoch comparisons blocked by unchanged
exact-source/protocol checks. An evaluation-only provenance contract for historical
weights is a proposal requiring approval, not an implemented exception or permission
to retrain. Historical run artifacts remain unchanged.

## Stage 4 owner academic-use declaration — 2026-10-19

D-042: The owner confirms non-commercial academic purpose, local training/evaluation, no publication/redistribution/cloud upload of TACO photos or annotations, and authorizes proceeding on a genuinely documented lawful-use basis if supportable. This is project-owner workflow authorization, not rightsholder permission, institutional approval or definitive legal clearance. The [case-specific Section 185 record](TACO_STAGE_4_ACADEMIC_USE.md) documents a defensible limited-use claim with remaining uncertainty, bound to the exact 221 training and 47 validation images/annotations. The v2 gate records that claim with clearance flags false and third-party authority explicitly not claimed. No new phase, model/loss/precision change, test use, cloud spending, publication or automatic merge is authorized. Existing Stage 4 pilot/training/evaluation proceeds within that scope.

## Current Stage 4 authorization — 2026-10-19

| ID | Approved decision | Consequence |
| --- | --- | --- |
| D-041 | Owner authorizes real-image fine-tuning and initial validation directly on Stage 3D's verified adapter as one continuous implementation task, conditional on verified permission or an appropriately approved lawful-use basis for source photos and annotations/database. | Implement loader, full resumable checkpoints, progress, real-memory pilot and bounded 2–3 epochs with fixed pretrained/validation comparison; preserve eight labels/splits, inference/environment and original weights. If rights remain unresolved, finish runnable software and identify exact approval before processing actual photos. No test-photo use, cloud spending/upload/publication, promotion or PR merge. Earlier synthetic-only boundaries are superseded within this scope. |

Implementation/evidence: [Stage 4](TACO_STAGE_4.md). Software and generated-image qualification are recorded separately from real-photo gates. Missing rights evidence currently blocks actual photo processing/training; no permission, real-data outcome or benchmark is invented. The fixed initial protocol/settings in the implementation are bounded engineering choices, not measured optimal defaults or a taxonomy/architecture decision.

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

## Width-driven previews and full-mode restoration - 2026-10-18

D-029: The owner requests larger configurable-width/aspect-preserving terminal previews with vertical scrolling, fewer pauses, fixed table alignment, token rows grouped by query, passing-only summary and working full below-threshold diagnostics. This supersedes D-028's restriction on explicit full-mode console output only. Preserve all inference, checkpoint/threshold, source/JSON and terminal-only boundaries. Verify using saved-result replay without rerunning inference; same branch and no merge. See [current terminal contract](TERMINAL_VIEWER.md).

## Consistent mapping and presentation owner instruction - 2026-10-18

D-030: The owner requests consistent detection summaries containing image/query IDs, raw decoded phrase, mapped category and mapping reason, score, threshold status and rounded boxes. Keep unmapped predictions distinct from failed detections; never force incomplete phrases into categories. Improve wrapping and terminal preview quality within the actual grid, explain resolution limits without claiming recovered detail, retain complete raw precision/token scores in JSON and summary/full modes. Use saved results where possible; preserve model inference, prompts, thresholds, originals and existing functionality. Continue the same feature branch/PR without merging. See [current presentation contract](TERMINAL_VIEWER.md).

## Native viewer restoration owner instruction - 2026-10-18

D-031: The owner explicitly replaces terminal image rendering with an existing-dependency Matplotlib/TkAgg window, superseding terminal-only/no-pop-up instructions. Display the oriented original with valid retained boxes, matching query IDs, phrases/categories, percentage scores and original pixel coordinates; retain zoom/pan, readable labels and passing-only terminal tables. Keep in-memory rendering, no image copies/exports, source-verified replay and atomic separate inference/display statuses. Test alignment, empty images, closure and failures; report actual Windows observations separately. Preserve user configuration edits, model inference, prompts, thresholds and weights. Continue current branch/open PR without merging. Subsequent owner feedback prioritizes output over additional navigation checks and explicitly requests only mapped-category text on image labels; keep IDs, scores, phrases and coordinates in the same-window details panel and terminal tables. See [native viewer contract and evidence](NATIVE_VIEWER.md).

## One scrollable results tab owner instruction - 2026-10-18

D-032: The owner supersedes terminal-only and separate image-window layouts with one scrollable GUI tab repeating Image #N, Detection Summary, Text-Token Details and Configuration in that order. Only numbered headings precede images. Use raw decoded phrase labels immediately above valid passing boxes, bounded readable wrapping, original pixel ticks, zoom/pan and matching IDs where space permits; keep scores in the summary. Passing-only tables must be selectable/copyable and show actual saved configuration. Preserve complete JSON, inference/model/mapping behavior, originals and atomic separate display outcomes, and support source-verified multi-image replay without inference. Verify synthetic and actual Windows observations separately. Continue the existing branch, update an open PR if present, and do not merge. See [current scrollable GUI contract](SCROLLABLE_VIEWER.md).

The owner confirmed the corrected raw phrase labels were complete and subsequently requested smaller label fonts to reduce occupied image space. Use 8.5-point labels with an 8-point fallback, retaining full readable tables and the disclosed zoom/detail fallback. This changes presentation only.

## Phase 2D authorization - 2026-10-18

D-033: The owner explicitly authorizes Phase 2D - Display and Summaries, completing and validating the existing shared Tk/TkAgg viewer. Preserve local prompt/threshold/font edits, environment, model/mapping, originals and raw JSON. Use latest main and a focused branch, carry relevant unmerged work, prefer saved-image replay, distinguish automated from visual evidence, publish a PR without merging. Phase 2E, fine-tuning, grouping, new data collection and web modules remain excluded. Supersedes the earlier same-feature-branch instruction. See [2D acceptance evidence](PHASE_2D_CHECKS.md).

## Phase 2E authorization - 2026-10-18

D-034: The owner authorizes Phase 2E - Functional Checking and Handoff, superseding the stop-before-2E boundary. Use existing owner photos and pinned cached inference in the preserved Windows environment, compare current settings with a small controlled prompt/threshold set, inspect boxes and raw phrases, verify display/persistence and publish evidence without merging. Keep human confirmation separate; missing observations/scenarios remain pending. Stop before Phase 3 decisions or optimization. No training, grouping, model/dependency changes, collection or application modules. See [results and recommendations](PHASE_2E_RESULTS.md).

During this work the owner reported that current labels needed to be smaller and some were absent for small/nested boxes. This authorizes the focused presentation correction: 7.5/7-point defaults, placement beyond narrow box width while bounded by axes and clear of box borders, query IDs/connectors. Corrected human acceptance remains pending. Existing 8.5/8-point decisions above describe history. Experimental prompt/threshold findings and proposed Phase 3 directions are evidence/proposals, not approved defaults or phase transitions.

## TACO harmonization and annotation visualization authorization

D-035 (2026-10-18): The owner authorizes preparation of the existing external TACO dataset with the eight-label object taxonomy in [config/taco_taxonomy.v1.json](../config/taco_taxonomy.v1.json): cigarette, plastic_bottle, plastic_bag, plastic_wrapper, plastic_film, drink_can, plastic_cup, styrofoam_piece. Match the eleven explicitly approved source names exactly and preserve their IDs and all original metadata. Implement read-only auditing and local annotation visualization with class sampling and explicit excluded/Unlabeled litter handling. No source modification, additional downloads, final splits, implicit negative examples, dependency/model changes or fine-tuning. Keep the existing inference pipeline and viewer unchanged. See [evidence](TACO_PREPARATION.md).

Unresolved preparation decisions: excluded/unlabeled/crowd handling for later training, stored-versus-EXIF coordinate policy, duplicate-ID resolution, negative-origin box repairs, final splits and training export format. Conservative audit holds and explicit EXIF previews are review behavior, not owner approval of a training transformation. The eight object labels do not replace the three existing inference project categories.

## Fine-Tuning Preparation Stage 2 authorization

D-036 (2026-10-18): The owner reports representative annotations visually reviewed and authorizes non-destructive strict-clean conversion and seeded image-level train/validation/test splitting of existing external TACO. Use the eight-class taxonomy/order, default proposed 70/15/15 ratios and seed 42, preserve original metadata/source identity, quarantine whole missing/unaligned/invalid/mixed/unlabeled images, and report losses/bias. Do not clip boxes, repair duplicated source annotation IDs, assume trainer ignore flags, copy photos, download data, change dependencies/inference or load/fine-tune Grounding DINO. This supersedes D-035's export/split stop only. No merge. See [Stage 2 evidence](TACO_STAGE_2.md).

Implementation choices within this scope: hold non-identity/ambiguous EXIF cases in addition to dimension mismatches; coassign exact/perceptual candidate photograph groups; propagate dirty exact-peer quarantine and hold conflicting exact-peer annotations; preserve full source records alongside unique export annotation IDs. These choices do not approve source repairs or confirm that all near duplicates were found. Licensing, trainer compatibility, small evaluation classes and strict-clean sampling bias remain pending readiness decisions. The existing three final inference categories remain unchanged.

## Stage 3A assessment authorization

D-037 (2026-10-18): The owner authorizes review of completed Stage 2 and a technical assessment of exact pinned Grounding DINO Tiny training implementation, excluded/unlabeled handling, existing-dataset recovery, GPU/cloud feasibility, licensing and a small smoke/evaluation proposal. Preserve strict-clean exports, eight-label taxonomy, original sources and inference environment/pipeline. No fine-tuning, taxonomy modification or replacement training export is authorized. See [assessment, primary sources and pending gates](TACO_STAGE_3A.md).

Proposals, not approvals: use Transformers 4.51.3's native loss with a separate PyTorch trainer/environment; qualify missing SciPy/evaluation dependencies separately; retain strict-clean as the first smoke reference; review coordinates before expansion; investigate auxiliary supervision of known excluded categories while keeping unresolved unlabeled photos quarantined; prefer approximately 24 GB GPU headroom. Rights clearance, runtime memory/backward/save-reload qualification, provider/budget, auxiliary taxonomy/ignore semantics and future versioned expansion/evaluation remain owner decisions. No training readiness or model quality is established by this assessment.

## Stage 3B synthetic training authorization

D-038 (2026-10-18): The owner authorizes a separate reproducible training environment, required dependencies including SciPy, and bounded synthetic-only PyTorch forward/backward/optimizer checks using the exact pinned Grounding DINO Tiny and Transformers 4.51.3 native loss. Verify all-eight prompt mapping, COCO preprocessing/batching, actual GPU peaks and model/processor saving/reloading. Preserve functioning inference `.venv`, pipeline/settings, source photos/annotations, taxonomy and strict-clean splits. No TACO photo training, cloud uploads/spending, replacement exports or real-data fine-tuning. See [actual implementation and evidence](TACO_STAGE_3B.md).

Recorded implementation choices for this software qualification only: FP32 batch 2, frozen vision/BERT, 192/256 input edges, 900 queries and unchanged native loss. The earlier 128/192 attempt failed spatial top-k capacity and remains preserved; there is no automatic model/device fallback. The successful small synthetic run is not an approved real-data hyperparameter configuration or evidence that full training fits 6 GB. Native encoder regression lacks gradients and encoder classification dominates the loss; investigating/correcting that behavior, real-data readiness, rights, resume/evaluation design, taxonomy/ignore policies and expansion remain future decisions. No library version/weight replacement or hidden loss patch is approved by this result.

## Stage 3C loss investigation authorization

D-039 (2026-10-18): The owner authorizes tracing pinned native encoder/decoder losses and controlled synthetic target-count/empty experiments in `.venv-training`, with masks, matching, normalization, gradient/update and memory diagnostics. Preserve environments, inference, original checkpoint, taxonomy and splits. Do not train on real TACO, patch Transformers, silently change coefficients/architecture or spend on cloud resources. Explain any proposed correction with supporting evidence and request approval before implementing a substantive training-method change. See [actual results and next-stage proposal](TACO_STAGE_3C.md).

Investigation findings, not an adopted correction within Stage 3C: native encoder supervision uses a detached box copy and recomputed learned-query/pre-encoder text scores. Selected proposal tensors with the same criterion show expected image-conditioned scores and encoder-head gradient paths; Stage 3C hypothesis terms are diagnostic only and never optimized. Native source, weights, loss and Stage 3B implementation remain unchanged. Its proposed Stage 3D required separate approval, recorded below. Rights, evaluator/resume design, resolution memory, excluded annotations and dataset coverage remain independent prerequisites.

## Stage 3D adapter authorization and outcome

D-040 (2026-10-19): The owner explicitly approves the separate training-only encoder-output adapter and synthetic qualification using Stage 3C findings. Supply selected image-conditioned encoder proposal scores and undetached supervised boxes to the existing native loss; preserve detached decoder reference initialization, pretrained architecture/base weights, query count, text mapping and coefficients. Verify actual gradients/updates, empty paths, decoder equivalence, checkpoint/adapter reconstruction, memory and regression evidence. No installed-library modification/monkey patch, inference change, real TACO training, data/taxonomy/split modification, cloud spending or automatic promotion. Stop/report if invariants cannot be preserved, and stop after Stage 3D synthetic qualification.

Actual Stage 3D outcome is a bounded synthetic PASS, detailed in [evidence](TACO_STAGE_3D.md). Every corrected case has two optimizer steps, finite values, intended encoder updates and exact decoder equivalence; empty regression is correctly zero. Experimental model/processor weights and explicit adapter context reload exactly. The checkpoint is weights/context only with `resume_supported=false`; no inference promotion or real-data readiness is approved by this outcome. Real-data trainer/resume/evaluator design, rights, coverage/leakage and intended-resolution memory remain future prerequisites.
