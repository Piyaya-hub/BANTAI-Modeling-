# Development phases

Phase 0 is complete; the Phase 1 planning deliverable is published through merged PR #3. Publication does not establish environment preparation, successful detection, or owner acceptance of every proposal. Codex now owns implementation, replacing Antigravity (owner reports no Antigravity changes). Phase 2A is complete on supplied evidence; the owner now authorizes Phase 2B input/output foundation only; routine defaults within an authorized phase do not introduce extra approval checkpoints.

| Phase | Work and current status | Acceptance evidence |
| --- | --- | --- |
| 0 — Project foundation | Complete: scope, architecture, decisions, agent rules, and phase gates published through PRs #1–#2. | Foundation documentation exists and owner completion is recorded. No inference or performance result is implied. |
| 1 — Model evaluation preparation | Planning deliverable published (PR #3); bounded Phase 2A technical preparation now complete on supplied target evidence. Detector/model execution remains unverified. Its environment-preparation actions are now authorized as Phase 2A below; no planning history is renumbered or detector authorization inferred. | Record exact checkpoint/configuration/conditions/safetensors/revision, same-revision processor, local Python/driver/memory, proposed package pins and resolved dependencies/CUDA source, GPU or diagnosed CPU fallback, native viewer, preprocessing, prompts/thresholds, visual-review rules, and corrected JSON/display contract. No photos or annotations are required to complete preparation; missing prerequisites remain explicit. |
| 2 — Pretrained inference and functional checking | Phase 2A preparation is complete; Phase 2B foundation is implemented for review. Model/inference/viewer work remains outside the current authorization. | Software checks cover inputs/errors, orientation/pixel coordinates, invalid-number nulls, mapping/unmapped results, conservative duplicate flags, strict JSON, display lifecycle, original preservation, and no automatic image exports. Actual photos demonstrate displayed boxes/labels/scores, readable summaries, JSON consistency, multiple candidate regions/mixed categories where available, documented timings and qualitative limitations. Distinguish passed/failed/skipped/unrun scenarios; no precision/recall/mAP, benchmark splits, annotations, or numeric accuracy targets are required. |
| 3 — Evidence-based optimization decision | Later owner review of functional observations and baseline limitations. | Record whether to retain prompting, adjust prompts/thresholds, investigate separately scoped grouping/hybrids/alternatives or model improvement, or stop. Automatic spatial grouping, DBSCAN, training, and fine-tuning remain outside the baseline; scope expansion requires explicit authorization. |
| 4 — Conditional model improvement and integration readiness | Only separately approved modelling improvements, reusable inference components, packaging/output schema and integration documentation. | Each approved deliverable has a defined validation plan and recorded evidence/limitations. Do not expand into frontend/backend application modules, APIs, databases, road anomalies, video/cameras, incidents, or hotspots. No completion is claimed now. |

## Phase 2A — Local Windows environment setup and readiness verification

The owner's current instruction explicitly authorizes this preparation-only subset of the published plan: **Section 4 and Section 8 stages 2–3**. The earlier Phase 1 planning deliverable and broad Phase 2 detector scope remain historical context; this definition does not authorize the remainder of Phase 2. Codex owns execution; Antigravity is replaced and, per the owner, made no changes.

Required preparation evidence:

1. Inspect the actual Windows checkout and existing environments without discarding user work. Record Windows/Python architecture/version, NVIDIA driver, GPU identity and total/free memory. A cloud Linux runtime is not target evidence.
2. Inspect checkpoint metadata only: exact source/configuration/conditions/license/safetensors availability and immutable revision. No model weights, processor/model instantiation, photographs or datasets are needed or downloaded in this phase.
3. Verify the proposed Python/package/CUDA wheel choice against local preflight and official metadata; prepare an isolated environment without replacing global packages or an unrelated `.venv`. Preserve TLS/checksum verification. No separately installed CUDA toolkit/custom detector extension is assumed necessary.
4. Record exact resolved dependencies plus the CUDA/CPU installation source; pass `pip check` and imports. Verify CPU tensor operations and either actual GPU tensor execution with memory/build details or a diagnosed CPU fallback. A fallback is recorded as CPU readiness, not successful GPU verification.
5. Verify native Tk and Matplotlib display using a synthetic plot only, record successful render/normal return, and close it without saving an image. No photos, detector, model inference or annotated exports.
6. Record passed/failed/unrun outcomes and remaining prerequisites in [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md). Completion requires actual target evidence, not published commands or a successful cloud probe. Routine choices/probes within 2A need no new phase approval; missing access/evidence remains a prerequisite.

Historical 2A stop boundary (superseded for input/output foundation by the 2B instruction below): do not create detector/input-processing application code, load/download pretrained weights, run inference/photo checks, implement clustering, train/fine-tune, or build an application. The remaining detector/inference work requires later authorization. Do not merge the preparation branch or declare readiness without evidence.

## Phase 2B — Input and Output Foundation

Authorized by the owner's current instruction (D-024), superseding only the 2A-only stop boundary. This is implementation-plan stage 4 plus Section 7 result/persistence utilities. Configuration, image loading, result-contract validation and persistence are the entire scope. No model loading, weight downloads, inference or viewer implementation. Existing verified dependencies are preserved; requirements are transcribed from the 2A snapshot without reinstalling or claiming a tested fresh install.

Acceptance evidence:

1. Validated configurable repository-relative paths, pinned checkpoint/processor revision, proposed prompts/thresholds and device/display settings. Input/results/cache paths cannot overlap, including resolved symlinks.
2. Sorted nonrecursive JPEG/PNG discovery, actionable missing/empty-folder errors, unsupported skips and distinct decode/read errors. Originals are read only; EXIF orientation and RGB conversion occur in memory; oriented dimensions and source SHA-256 are recorded.
3. Separate `bantai.preparation.v1` records for the preparation-only CLI. `prepared` explicitly means decoded with inference unrun; there are no fabricated predictions or functional success/no-detection claims.
4. Reusable `bantai.functional.v1` utilities validate raw/clipped pixel boxes and scores, null invalid numbers with diagnostics, exclude invalid candidates from valid summaries, preserve unmapped candidates, separate processing/display outcomes and leave physical-object/independent-region counts unresolved. Explicit completion plus execution provenance is required for functional success.
5. UUID run directories, source-path/content digests, atomic strict-JSON initial writes without overwrite, and explicit display-only atomic updates. Failed final updates preserve earlier pending results. No viewer is implemented or invoked.
6. Synthetic software tests cover configuration/path errors, EXIF pixel coordinates, source preservation, numeric handling, status consistency, collision prevention, atomic-write failures and simulated display lifecycle. Fixtures are software evidence only. Actual commands/platform limits are recorded in [PHASE_2B_CHECKS.md](PHASE_2B_CHECKS.md).

## Present gate

Phase 2A remains technically complete on owner-supplied Windows evidence in [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md) and the [snapshot](PHASE_2A_ENVIRONMENT_SNAPSHOT.md). Phase 2B foundation is implemented and checked for review; no owner acceptance or merge is inferred.

Stop before Phase 2C. A future authorized 2C may integrate pinned model/processor loading, actual inference, mapping/duplicate/overlap policy and native in-memory viewing under the existing plan; its exact scope still requires the owner's instruction. Model-memory fit, prompt/category quality and real-photo behavior remain unrun. No automatic grouping, training, web/application modules or automatic image exports. Photos/GPU access are unnecessary for 2B; team photos remain manually supplied and unannotated for later functional checks.
