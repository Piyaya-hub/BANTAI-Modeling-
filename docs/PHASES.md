# Development phases

Stage 4 D-048's **single corrected-target two-epoch restart is complete**:
442 full-run updates after three discarded pilot steps. Corrected v2 AP
0.203932 at epoch 1 and 0.084500 at epoch 2, versus 0.111576 pretrained.
[Results, preflight, limits and evidence](TACO_STAGE_4_CORRECTED_RETRAIN.md).
Stop for owner review; no further training, test evaluation or promotion.

Latest owner amendment D-047 restores the existing epoch comparison viewer through
explicit `--gui`, preserving D-046's **terminal-only per-epoch scores** by default.
Both paths reuse verified saved results without inference or training; `--image-id`
selects a validation image in GUI mode. Human visual acceptance remains pending.

Current Stage 4 substep: **Step 2 local diagnostic capability and automated analysis
implemented** (D-045); [native evidence and pending human checks](TACO_STAGE_4_VISUAL_REVIEW.md).
Sixteen systematic examples cover eight classes; all 47 validation images remain
available. Saved-record replay preserves provenance, official metrics and strict
resume. Human object/phrase/readability/error confirmation remains pending. Stop
after diagnostic review; no new fine-tuning, test access, promotion or publication.
The Step 1C and earlier current-state statements below are historical records.

Current Stage 4 substep: **Step 1C historical checkpoint reevaluation complete**
(D-044). [Evaluation-only provenance and corrected results](TACO_STAGE_4_REEVALUATION.md)
preserve strict resume while comparing the same 47 validation images. Corrected
pretrained / epoch-1 / epoch-2 AP: 0.111576 / 0.126627 / 0.066215; recall regresses
and high-score suppression persists. No new training, test access, promotion or
visual analysis. Step 2 visual comparison is the proposed next action. The initial
Stage 4 execution findings below are historical and used faulty geometry.

Current stage: **Stage 4 ? Real-Image Fine-Tuning and Initial Evaluation** (D-041/D-042). [The first local real-image experiment](TACO_STAGE_4_RUN.md) completed two epochs, actual source/coordinate/memory checks, complete checkpoints and fixed pretrained-versus-final validation. [The documented academic-use claim](TACO_STAGE_4_ACADEMIC_USE.md) records owner workflow authorization separately from third-party permission or definitive clearance. Runtime gates passed; **detection AP deteriorated despite lower loss**, so no promotion/quality acceptance is claimed. Test photos, taxonomy/splits, inference and base weights remain preserved. No new preparation phase, extended training, cloud spending, publication or merge.

Completed preparation stage: **Stage 3D - Training-Only Encoder-Output Adapter and Synthetic Qualification** (D-040). Acceptance required selected image-conditioned encoder scores/undetached supervised boxes with the original criterion, unchanged decoder reference detachment/architecture/text/coefficients, actual nonempty/empty updates, decoder equivalence, checkpoint/context replay, measured memory and a clear PASS/FAIL. [Stage 3D evidence](TACO_STAGE_3D.md) records a bounded synthetic **PASS**. Installed libraries, inference, original weights, taxonomy and splits remained preserved. Resume/full-resolution/real-data readiness was not established by that stage. Its stop-after-synthetic boundary is now superseded only within D-041's conditional scope; no automatic promotion, cloud spending or merge.

Completed **Stage 3C - Training Loss and Gradient Investigation** (D-039). Evidence is [TACO_STAGE_3C.md](TACO_STAGE_3C.md): pinned-source tracing, controlled counts/empty cases, matching/mask/normalization and update diagnostics, and causal factor isolation. Diagnostic hypotheses were not optimized in Stage 3C; D-040 separately authorized the implemented training-only correction.

Completed **Stage 3B - Training Environment and Synthetic Smoke Test** (D-038). Acceptance required an independent pinned environment, actual synthetic native-loss forward/backward/optimizer execution, all-eight prompt/COCO geometry/batching checks, measured GPU peaks, exact model/processor save-reload and source/inference preservation. [Actual Stage 3B evidence](TACO_STAGE_3B.md) records a bounded lifecycle pass and a failed undersized-input attempt. That pass is preserved; it is not a real-data readiness decision.

Completed **Stage 3A - Grounding DINO Tiny Training Compatibility and Dataset Expansion Assessment** (D-037), assessment only. Evidence is [TACO_STAGE_3A.md](TACO_STAGE_3A.md): exact-version loading/loss/text/COCO/persistence review, explicit excluded/unlabeled semantics, conditional recovery counts, analytical hardware assessment, rights gaps, proposed smoke/evaluation gates and owner decisions. Its source-only/unrun statements are historical; Stage 3B execution does not clear rights or expansion decisions.

Completed preparation stage: **Fine-Tuning Preparation Stage 2 - Dataset Conversion and Splitting** (D-036). Acceptance requires a versioned strict whole-image policy, complete quarantine/annotation traceability, eight-class COCO with original xywh geometry/external image references, seeded duplicate-aware multi-label image splits, loss/coverage reports, source preservation and independent reconstruction validation. See [implementation and actual results](TACO_STAGE_2.md). Structural/source validation passed; licensing and low evaluation coverage remain pending. Its strict-clean export is preserved as the reference; training qualification remains unrun.

Historical Stage 1 preparation authorization: **TACO Dataset Harmonization and Annotation Visualization** (D-035). Audit and visualize the existing external COCO source under the eight approved object labels, preserving excluded annotations and all metadata. Acceptance evidence is [TACO_PREPARATION.md](TACO_PREPARATION.md): versioned taxonomy, reproducible JSON/CSV audit, native sampled annotation viewing, source hashes and focused checks. This is separate from the inference pipeline and does not authorize final dataset splits, training exports, downloads, fine-tuning or broader Phase 3 optimization. Coordinate/exclusion/repair decisions remain pending. The earlier inference-phase records below are preserved history.

Latest owner viewer amendment requires one scrollable tab with repeated Image / Detection Summary / Text-Token Details / Configuration sections, raw phrase labels and passing-only GUI tables. Acceptance criterion 5 uses shared-window normal closure after all results persist, preserving separate display failures. This supersedes earlier per-image/terminal layouts with Phase 2E functional checking now authorized (D-034). See [scrollable viewer contract](SCROLLABLE_VIEWER.md).

Phase 0 is complete; the Phase 1 planning deliverable is published through merged PR #3. Publication does not establish environment preparation, successful detection, or owner acceptance of every proposal. Codex now owns implementation, replacing Antigravity (owner reports no Antigravity changes). Phase 2A is complete on supplied evidence; the owner now authorizes Phase 2E functional checking and handoff; routine defaults within an authorized phase do not introduce extra approval checkpoints.

| Phase | Work and current status | Acceptance evidence |
| --- | --- | --- |
| 0 — Project foundation | Complete: scope, architecture, decisions, agent rules, and phase gates published through PRs #1–#2. | Foundation documentation exists and owner completion is recorded. No inference or performance result is implied. |
| 1 — Model evaluation preparation | Planning deliverable published (PR #3); bounded Phase 2A technical preparation now complete on supplied target evidence. Detector/model execution remains unverified. Its environment-preparation actions are now authorized as Phase 2A below; no planning history is renumbered or detector authorization inferred. | Record exact checkpoint/configuration/conditions/safetensors/revision, same-revision processor, local Python/driver/memory, proposed package pins and resolved dependencies/CUDA source, GPU or diagnosed CPU fallback, native viewer, preprocessing, prompts/thresholds, visual-review rules, and corrected JSON/display contract. No photos or annotations are required to complete preparation; missing prerequisites remain explicit. |
| 2 — Pretrained inference and functional checking | Phase 2A preparation and Phase 2B foundation are published. Phase 2C baseline has recorded Windows CUDA/photo evidence. Phase 2E has five controlled Windows CUDA runs on three photos and documented qualitative limitations. Corrected display human acceptance and missing scene coverage remain pending; no Phase 3 decision is made. | Software checks cover inputs/errors, orientation/pixel coordinates, invalid-number nulls, mapping/unmapped results, conservative duplicate flags, strict JSON, display lifecycle, original preservation, and no automatic image exports. Actual photos demonstrate displayed boxes/labels/scores, readable summaries, JSON consistency, multiple candidate regions/mixed categories where available, documented timings and qualitative limitations. Distinguish passed/failed/skipped/unrun scenarios; no precision/recall/mAP, benchmark splits, annotations, or numeric accuracy targets are required. |
| 3 — Evidence-based optimization decision | Later owner review of functional observations and baseline limitations. | Record whether to retain prompting, adjust prompts/thresholds, investigate separately scoped grouping/hybrids/alternatives or model improvement, or stop. Automatic spatial grouping, DBSCAN, training, and fine-tuning remain outside the baseline; scope expansion requires explicit authorization. |
| 4 — Conditional model improvement and integration readiness | Only separately approved modelling improvements, reusable inference components, packaging/output schema and integration documentation. | Each approved deliverable has a defined validation plan and recorded evidence/limitations. Do not expand into frontend/backend application modules, APIs, databases, road anomalies, video/cameras, incidents, or hotspots. No completion is claimed now. |

## Phase 2A — Local Windows environment setup and readiness verification

The earlier 2A instruction authorized this preparation-only subset of the published plan: **Section 4 and Section 8 stages 2–3**. The earlier Phase 1 planning deliverable and broad Phase 2 detector scope remain historical context; this definition does not authorize the remainder of Phase 2. Codex owns execution; Antigravity is replaced and, per the owner, made no changes.

Required preparation evidence:

1. Inspect the actual Windows checkout and existing environments without discarding user work. Record Windows/Python architecture/version, NVIDIA driver, GPU identity and total/free memory. A cloud Linux runtime is not target evidence.
2. Inspect checkpoint metadata only: exact source/configuration/conditions/license/safetensors availability and immutable revision. No model weights, processor/model instantiation, photographs or datasets are needed or downloaded in this phase.
3. Verify the proposed Python/package/CUDA wheel choice against local preflight and official metadata; prepare an isolated environment without replacing global packages or an unrelated `.venv`. Preserve TLS/checksum verification. No separately installed CUDA toolkit/custom detector extension is assumed necessary.
4. Record exact resolved dependencies plus the CUDA/CPU installation source; pass `pip check` and imports. Verify CPU tensor operations and either actual GPU tensor execution with memory/build details or a diagnosed CPU fallback. A fallback is recorded as CPU readiness, not successful GPU verification.
5. Verify native Tk and Matplotlib display using a synthetic plot only, record successful render/normal return, and close it without saving an image. No photos, detector, model inference or annotated exports.
6. Record passed/failed/unrun outcomes and remaining prerequisites in [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md). Completion requires actual target evidence, not published commands or a successful cloud probe. Routine choices/probes within 2A need no new phase approval; missing access/evidence remains a prerequisite.

Historical 2A stop boundary (superseded for input/output foundation by the 2B instruction below): do not create detector/input-processing application code, load/download pretrained weights, run inference/photo checks, implement clustering, train/fine-tune, or build an application. The remaining detector/inference work requires later authorization. Do not merge the preparation branch or declare readiness without evidence.

## Phase 2B — Input and Output Foundation

Authorized by the earlier 2B instruction (D-024), superseding only the 2A-only stop boundary. This is implementation-plan stage 4 plus Section 7 result/persistence utilities. Configuration, image loading, result-contract validation and persistence are the entire scope. No model loading, weight downloads, inference or viewer implementation. Existing verified dependencies are preserved; requirements are transcribed from the 2A snapshot without reinstalling or claiming a tested fresh install.

Acceptance evidence:

1. Validated configurable repository-relative paths, pinned checkpoint/processor revision, proposed prompts/thresholds and device/display settings. Input/results/cache paths cannot overlap, including resolved symlinks.
2. Sorted nonrecursive JPEG/PNG discovery, actionable missing/empty-folder errors, unsupported skips and distinct decode/read errors. Originals are read only; EXIF orientation and RGB conversion occur in memory; oriented dimensions and source SHA-256 are recorded.
3. Separate `bantai.preparation.v1` records for the preparation-only CLI. `prepared` explicitly means decoded with inference unrun; there are no fabricated predictions or functional success/no-detection claims.
4. Reusable `bantai.functional.v1` utilities validate raw/clipped pixel boxes and scores, null invalid numbers with diagnostics, exclude invalid candidates from valid summaries, preserve unmapped candidates, separate processing/display outcomes and leave physical-object/independent-region counts unresolved. Explicit completion plus execution provenance is required for functional success.
5. UUID run directories, source-path/content digests, atomic strict-JSON initial writes without overwrite, and explicit display-only atomic updates. Failed final updates preserve earlier pending results. No viewer is implemented or invoked.
6. Synthetic software tests cover configuration/path errors, EXIF pixel coordinates, source preservation, numeric handling, status consistency, collision prevention, atomic-write failures and simulated display lifecycle. Fixtures are software evidence only. Actual commands/platform limits are recorded in [PHASE_2B_CHECKS.md](PHASE_2B_CHECKS.md).

## Phase 2C — Pretrained baseline inference and native display

The owner instructed Codex to proceed with Phase 2C (D-025), superseding the 2B stop boundary. This integrates implementation-plan stages 5–6 and the authorized baseline functional-check path, without automatic grouping, training or application modules.

Acceptance criteria:

1. Retain preparation-only default behavior; `--mode infer` explicitly enables pinned Grounding DINO Tiny. Load config, processor and safetensors at the same verified immutable revision into the configured cache, disable remote code/custom kernels, verify observed direct dependency versions, use float32/eval/inference mode, and record actual environment/preprocessing metadata.
2. Prefer CUDA for `auto`, recording visible CPU fallback when unavailable. Only CUDA out-of-memory in auto mode permits one CPU retry; explicit CUDA or unrelated failures are errors. Record model-load time separately, warm up once per actual device, synchronize CUDA timings and exclude decode/warm-up/display from the single-image measured interval. No latency or model-memory guarantee.
3. Use the pinned grounded postprocessor with `threshold`, `text_threshold`, `(height,width)` oriented original dimensions and `text_labels`. Audit raw outputs before thresholding can hide invalid values. Preserve invalid candidates and any phrases actually decoded; never invent a missing phrase/score.
4. Map exact normalized aliases only; retain unmapped/mixed candidates and raw phrases. Collapse display/summary duplicates only for identical raw boxes and normalized phrases, retaining the highest score with links. Keep near duplicates and cross-category conflicts. IoU/containment flags are review hints, never clusters, memberships or physical-object counts.
5. Persist functional JSON before display; display only valid retained candidates in native TkAgg, including successful no-detection images. Show readable labels and individual model-score percentages in the oriented pixel frame. Save shown/error only after the viewer returns or fails; preserve pending on interrupted/failed final persistence. No image export/temp-image file.
6. Check loader/API/device/postprocessing/persistence/viewer software behavior with explicitly synthetic fixtures and real headless canvas drawing where available. Record real model/target/photo/desktop results separately. A mocked loader or Agg canvas is not pretrained inference or native Windows rendering evidence.

Implementation and software checks are recorded in [PHASE_2C_CHECKS.md](PHASE_2C_CHECKS.md). Requirements and supplied 2A evidence remain unchanged; no reinstall/fresh-install claim.

## Phase 2D - Display and Summaries

Authorized 2026-10-18 (D-033). Complete and validate the existing viewer, preserving the model, prompts, thresholds, mapping, weights and dependencies. Acceptance evidence is recorded in [PHASE_2D_CHECKS.md](PHASE_2D_CHECKS.md).

1. One shared tab repeats Image #N, Detection Summary, Text-Token Details, Configuration for every image, including empty results.
2. Full EXIF-oriented source coordinates, aspect ratio and aligned valid retained passing boxes; honest raw phrases above borders, configurable small readable fonts, bounded wrapping and full-table/zoom fallback for boxes too small to label.
3. Passing-only selectable/copyable detections and ordinary tokens with matching query IDs, decimal score/threshold units and actual saved inference configuration. Processing failure, successful empty results and display outcomes remain distinct.
4. Original-preserving single/multi-image replay verifies all sources before display, without inference or temporary/annotated images. Results persist before display; atomic separate outcomes never erase predictions. Closure releases image/canvas resources.
5. Focused fixtures cover labels, edges, empty results, identity failures and persistence/GUI failures. Actual Windows replay verifies native scrolling, tables and closure. Human readability and interactive zoom/pan checks must be recorded separately; unavailable checks remain pending, never inferred from mocks or exit codes.

Historical 2D implementation/runtime checks are recorded above. The owner later reported oversized/missing labels on small/nested boxes; the 2E correction and automated regression evidence are in [2E results](PHASE_2E_RESULTS.md), with corrected human acceptance still pending. The former stop-before-2E boundary is superseded below; fine-tuning and automatic merging remain unauthorized.

## Phase 2E - Functional Checking and Handoff

Authorized 2026-10-18 (D-034): complete saved-result display checks, inspect only existing owner photos, record baseline settings, run pinned cached inference and compare a small controlled set of prompts/thresholds without selecting new defaults. No annotations or image collection required.

Acceptance evidence:

1. Inventory available isolated/group/pile/mixed/small/occluded/no-trash scenes, marking missing coverage unrun. Preserve originals and prior results with hashes.
2. Separate per-run config and UUIDs, unchanged images/model/dependencies, single-factor prompt or threshold comparisons, actual device and precisely scoped timings.
3. Inspect boxes and honest raw phrases together for misses, false detections, wrong labels, unmapped/incomplete phrases, duplicates and group/pile confusion; preserve unknown memberships and physical/region counts.
4. Verify GUI/tables/strict JSON consistency, source identity, in-memory rendering, closure and atomic separate outcomes; use synthetic fixtures for controlled failures/no detections/continuation and distinguish them from real model evidence.
5. Publish [PHASE_2E_RESULTS.md](PHASE_2E_RESULTS.md), commands, limitations and recommendations. Human acceptance is recorded only when explicitly confirmed; unavailable observations remain pending.

Current outcome: five successful CUDA runs and all 15 image/result overlays inspected; automated native replay passed. Owner-reported small/nested-box label defect corrected; final human acceptance pending. No clear pile/no-trash scenes were available. Qualitative limitations preclude declaring category/default-setting acceptance. Evidence and handoff are complete for review; stop before Phase 3 decisions or optimization implementation. No training, grouping, model/dependency changes, application modules or merge.

## Historical Phase 2C gate

Direct Windows follow-up on the padding fix now records successful cached CUDA inference on one photo and normal native display, with 53 software tests passing and one privilege-related skip. See [actual Windows evidence](PHASE_2C_WINDOWS_RUN.md). The owner reports correct detections and small piles needing review; broader scene coverage and phase acceptance remain unresolved. The paragraph below describes the earlier pre-correction gate.

Phase 2B is merged through PR #6, with current main `5db174b` at branch creation. Phase 2C baseline implementation is available for review; **valid target detections, model-memory/device acceptance and native photo-overlay rendering remain unverified**. Supplied follow-up reached the first model attempt after a completed weight download, but an implementation bug rejected expected text-padding logits; the focused correction is described in [PHASE_2C_CHECKS.md](PHASE_2C_CHECKS.md). The cloud itself still lacks torch/Transformers and target desktop/photos. An actual local-files-only CLI probe records `runtime/missing_dependency`, not no detections. This is an execution prerequisite, not a request for renewed phase authorization.

Use the existing verified Windows environment for the new tests and authorized functional run, recording actual JSON/native outcomes and qualitative limitations. Team photos are manually supplied and unannotated; missing scenarios remain unrun. Do not infer acceptance, merge or model quality from software checks. No automatic grouping, training, alternative-model selection, web/application modules or Phase 3 optimization is authorized.
