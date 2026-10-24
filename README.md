# BANTAI Modeling

**Stage 4 investigations are complete through D-058; D-059 authorizes their focused publication to PR #18.**
[D-057 scale-matched training](docs/TACO_STAGE_4_576_TRAINING.md) completed at
384/576 in FP32. [D-058 trajectory audit](docs/TACO_STAGE_4_576_TRAJECTORY.md)
found progressive cigarette gains alongside larger-object class regressions.
All 50 corrected checkpoint bundles across five runs remain private and experimental.
No checkpoint is promoted, PR merge is not authorized, and further experiments require owner authorization.

## Stage 4 benchmark progression

All rows use the same 47 validation images / 75 GT boxes and corrected COCO v2
protocol. Seven-class AP excludes `plastic_bag`; cigarette counts are nonexclusive
same-class GT coverage over all 900 queries at score >=0.05 and IoU >=0.50,
not official COCO recall or calibrated correctness.

| State | Input scale | AP | AR100 | Seven-class AP | Cigarette coverage |
| --- | --- | --- | --- | --- | --- |
| Pretrained | 256/384 | 0.111576 | 0.363077 | 0.093769 | 1/24 |
| Pretrained (D-056) | 384/576 | 0.147847 | 0.426109 | 0.095418 | 3/24 |
| D-048 seed 42 E1 | 256/384 | 0.203932 | 0.396766 | 0.136242 | 2/24 |
| D-055 inference-only scaling | 256/384 trained → 384/576 evaluated | 0.247003 | 0.474763 | 0.152161 | 5/24 |
| D-057 scale-matched E1, step 221 | 384/576 | 0.249891 | 0.488141 | 0.181629 | 8/24 |
| D-058 mid-Epoch-2 audit, step 350 | 384/576 | 0.262337 | 0.428890 | 0.218249 | 14/24 |
| D-057 scale-matched E2, step 442 | 384/576 | 0.170508 | 0.370118 | 0.190859 | 15/24 |

Step 350 has the highest sampled eight-class AP, while E1 retains higher AR100.
Inspecting intermediate checkpoints on this small validation set is a diagnostic
audit with selection bias, not a checkpoint-selection rule or proof of generalization.
See the [wrap-up handoff](docs/HANDOFF.md) for the five-run inventory and owner-only next actions.
The [D-042 Section 185 academic-use record](docs/TACO_STAGE_4_ACADEMIC_USE.md)
is owner workflow authorization, not external permission, institutional approval or definitive legal clearance.
Detailed reports retain links to private local audit artifacts; these artifacts and
optional `work/` helper tests are not distributed. Tests requiring absent private
helpers skip explicitly. The baseline inference configuration remains unchanged.

## Historical Stage 4 investigation notices

Earlier status and stop notices below describe their respective completed substeps;
D-059 is the current documentation/publication authorization.

**D-056 replication completed.** [Three-model resolution report](docs/TACO_STAGE_4_RESOLUTION_REPLICATION.md)
finds shared small-bin/cigarette coverage gains but checkpoint-dependent raw AP.
Both new baselines reproduced exactly;188 inference-only forwards completed.
No training/promotion or core/GUI changes; stop for owner review.


**D-055 retry completed.** [Resolution comparison](docs/TACO_STAGE_4_RESOLUTION_EXPERIMENT.md)
improved fixed-checkpoint AP0.203932→0.247003, with partial small-object recovery
and three regressions. All gates passed; no training, promotion or core/GUI changes.
Stop for owner review. The initial stopped-attempt notice below is historical.


**D-055 stopped before model execution.** [Resolution attempt report](docs/TACO_STAGE_4_RESOLUTION_EXPERIMENT.md)
records a software-gate failure caused by the isolated runner's guard scoping.
No new inference or resolution result exists. Corrected guards await owner
review and authorization for a fresh attempt.


**Small-object/classification audit completed.** [Report](docs/TACO_STAGE_4_SMALL_OBJECT_CLASSIFICATION_AUDIT.md)
verifies actual preprocessing for all47 validation images, measures75 resized
GT boxes and compares nine saved model states. Small-size association and
well-localized class errors coexist; a fixed-weight scale comparison remains
a proposal. No new model inference or training; stop for owner review.


**Per-object saved detection review is ready.** [Report and GUI command](docs/TACO_STAGE_4_PER_OBJECT_REVIEW.md)
add opt-in GT selection/zoom and full overlap diagnostics for image777 and the
existing20 review images. Default viewer behavior is preserved. No new training
or inference; human annotation/material confirmation remains pending.


**Stage 4 cross-seed analysis and review preparation are complete.**
[The report](docs/TACO_STAGE_4_CROSS_SEED_ANALYSIS.md) compares all nine saved
states, per-class stability and failure patterns, with 20 human-review examples
and commands for selecting verified saved runs in the four-panel viewer.
No new experiment ran; human confirmation and owner review remain pending.


BANTAI is an academic intelligent-agent project for street-issue detection and hotspot prioritization. Its broader proposal includes roadside garbage, potholes, road-surface damage, incident management, and frequency-based hotspot prioritization.

This repository's initial development scope is **trash detection on still images only**. The approved starting model is **GroundingDINO-T (Swin-T / Tiny configuration)**, evaluated through pretrained text-guided inference. Hugging Face Transformers is the selected inference library. Phase 2A has recorded the exact checkpoint/processor revision, settings, installed dependencies and basic Windows runtime readiness. Prompt/category quality and thresholds remain unvalidated; publication of the plan does not establish their acceptance or readiness. Training, alternative models, and integration require later decisions based on evaluation evidence.

This repository is dedicated exclusively to AI modelling and computer vision: model evaluation/selection, preprocessing, detection, necessary postprocessing, accuracy/performance evaluation, approved optimization or potential fine-tuning, and reusable detection outputs for future integration. Frontend, backend application modules, central databases, incident management, reporting, and hotspot ranking are outside its implementation responsibility.

The three required output categories are **Garbage Pile**, **Individual Litter**, and **Group of Individual Litter**. Images may contain multiple independent regions and mixed categories. These are intended project outputs, not guaranteed pretrained classes. Direct prompting must be evaluated before considering spatial grouping; DBSCAN and other grouping methods are conditional alternatives requiring evidence and explicit authorization. See [scope](docs/SCOPE.md) for definitions and counting requirements.

Execution is intended for the owner’s Windows computer, preferring GPU with CPU fallback. Owner-supplied Command Prompt output now reports Windows 10 build 19045.6466, a GTX 1660 Ti with 6144 MiB VRAM, driver 472.19, and about 5235–5236 MiB free at the time of the query. This replaces the earlier RTX 2080 Super assumption. Git works. After the initial Store-alias/launcher issue, supplied output confirms Python 3.11.9 x64 installation, a successful local clone and `.venv` reporting Python 3.11.9. Supplied checks now show CPU/CUDA tensor results of 2.0, CUDA available, capability (7,5), matching direct library imports and no broken requirements. Native TkAgg rendering/normal return, checkpoint/processor metadata, official driver-reference comparison and exact dependency/source capture are also confirmed. Phase 2A technical preparation checks are complete; subsequent Phase 2C Windows CUDA execution is recorded separately below, with accuracy unmeasured; see the [preflight record](docs/PHASE_2A_READINESS.md). Checkpoint metadata reports Apache-2.0, non-gated access and safetensors at immutable revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`; future model and processor loading must use that same revision.

## Current state

**Paired seed-123 replication completed (D-051).** Exactly two two-epoch runs
completed: 1e-5 AP 0.140478→0.152880; 5e-6 AP 0.137919→0.113018.
This reverses the seed-42 epoch pattern and weakens a general lower-LR remedy
claim. [Cross-seed comparison and preservation evidence](docs/TACO_STAGE_4_SEED_REPLICATION.md).
No further experiment or promotion; await owner review.

**Controlled LR 5e-6 experiment completed (D-050).** One two-epoch run reached AP
0.137099 / 0.145952, reducing the earlier epoch-2 AP regression but falling short
of the 1e-5 epoch-1 AP 0.203932. Correct-class candidate coverage still declines.
[Five-state comparison, curves and preservation evidence](docs/TACO_STAGE_4_LR_EXPERIMENT.md).
No further training or checkpoint promotion; await owner review.

**Corrected-run performance investigation completed (D-049).**
[Diagnostics and proposed next controlled experiment](docs/TACO_STAGE_4_CORRECTED_DIAGNOSTICS.md)
show that epoch 2 retains good spatial candidates but loses correct-class
coverage, especially for bags/styrofoam. Score changes are mixed; the optimization
cause remains unresolved. Saved-result analysis only; await owner review.

**Controlled corrected-target restart completed (D-048).** One fresh two-epoch
run from the unchanged pretrained snapshot improved corrected-protocol AP to
0.203932 at epoch 1, then regressed to 0.084500 at epoch 2 (pretrained: 0.111576).
Exactly 442 updates plus three discarded pilot steps; no promotion or further
training. [Full comparison, loss curves, preflight and preservation evidence](docs/TACO_STAGE_4_CORRECTED_RETRAIN.md).
Await owner review.

**Latest owner preference: terminal-only epoch scores.** Run
`.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py` for the verified
saved scores of pretrained, epoch 1 and epoch 2. The epoch diagnostic GUI workflow
is also available explicitly with `--gui --image-id 42`; no new inference or
training is performed. Without `--gui`, the terminal-only default is unchanged.

**Stage 4 Step 2 adds local four-view diagnostic replay and initial annotation
analysis.** [Commands, representative sampling, findings and pending human
confirmation](docs/TACO_STAGE_4_VISUAL_REVIEW.md) compare the three saved model
states without inference. Native Windows replay/software checks passed; photo-level
judgments remain pending. No new training, test access or promotion occurred.
The Step 1C/earlier stage boundaries below describe their historical scope.

**Stage 4 Step 1C completed historical checkpoint reevaluation.** The separate
[evaluation-only provenance path and results](docs/TACO_STAGE_4_REEVALUATION.md)
preserve strict training resume. Corrected pretrained / epoch-1 / epoch-2 AP is
0.111576 / 0.126627 / 0.066215, with lower recall after training and persistent
score suppression. No training, promotion or Step 2 visual analysis occurred.
The Step 1B evaluation blockage below is historical and now resolved for evaluation.

**Stage 4 Step 1B corrects target/prediction coordinate frames.** [Correction and
validation evidence](docs/TACO_STAGE_4_COORDINATES.md) records pretrained AP
0.111576 on the same validation set; prior metrics below used incorrect geometry.
Saved epoch reevaluation is blocked by its exact-source contract, preserved intact.
No further training or Step 2 visual analysis occurred.

**Stage 4 completed the first local two-epoch real-image fine-tuning run**, using the verified Stage 3D adapter and the [documented owner-authorized academic-use claim](docs/TACO_STAGE_4_ACADEMIC_USE.md). [Measured results and checkpoint location](docs/TACO_STAGE_4_RUN.md): 442 full-run updates, memory about 2.38 GiB allocated, lower loss but **worse validation detection AP**. The experimental checkpoint remains private and unpromoted; test images, splits, taxonomy, inference and pretrained weights remain preserved. [Trainer/resume commands and initial history](docs/TACO_STAGE_4.md) remain available.

**Stage 3D implements the approved training-only encoder-output adapter.** [Synthetic qualification and complete evidence](docs/TACO_STAGE_3D.md) record a bounded **PASS**: all eight corrected optimizer steps, encoder classification/box-head updates, expected empty regression, exact decoder equivalence and model/processor/adapter replay. Architecture, text mapping, native loss coefficients, installed libraries and inference remain unchanged. Peak corrected-case allocator memory was 2.732 GiB allocated / 2.910 GiB reserved; this is not full-resolution or real-data readiness.

The completed [Stage 3C investigation](docs/TACO_STAGE_3C.md) supplies the causal evidence for the approved tensor-routing change. Its proposal/stop-before-adapter statements are historical. No TACO photo training, split/taxonomy changes, cloud use or checkpoint promotion occurred.

Completed **Stage 3B** provides the [isolated environment, smoke command and evidence](docs/TACO_STAGE_3B.md): two CUDA optimizer steps and model/processor save-reload with the exact checkpoint, Transformers 4.51.3 and SciPy in `.venv-training`. Peak allocated memory was 2.69 GiB for a small FP32 batch with frozen vision/BERT. The inference `.venv` is unchanged. No TACO photo training, taxonomy change, replacement export or cloud use occurred.

The earlier [Stage 3A assessment](docs/TACO_STAGE_3A.md) remains the source-backed expansion/licensing proposal. Its unrun training statements describe that historical stage; Stage 3B supplies bounded synthetic execution evidence only.

Completed **Fine-Tuning Preparation Stage 2** provides strict-clean eight-class COCO conversion and seeded, duplicate-aware image splits. The preserved subset contains 315 images/538 annotations: 221 training, 47 validation and 47 test images. Source photographs remain external; mixed/excluded/ambiguous images are quarantined whole. See [commands, class coverage, traceability and independent validation](docs/TACO_STAGE_2.md). Licensing and limited evaluation coverage remain unresolved; no fine-tuning is performed.

The earlier [TACO harmonization/audit and annotation viewer](docs/TACO_PREPARATION.md) remain unchanged. Their 1,248 readable/252 missing images and 3,938 annotations on readable images are the source pool, before Stage 2 whole-image exclusions.

Phase 2C includes pinned pretrained inference and native in-memory Matplotlib/TkAgg viewing. The padding fix has direct Windows CUDA evidence; see [Windows run](docs/PHASE_2C_WINDOWS_RUN.md). Phase 2E now authorizes functional checking and handoff. Five controlled cached CUDA runs on three existing photos are recorded in [Phase 2E results](docs/PHASE_2E_RESULTS.md); no default settings were selected. The shared Tk/TkAgg viewer retains raw phrase labels and passing-only GUI tables, with smaller labels and improved small/nested-box placement following owner feedback. See [Phase 2D evidence and remaining visual checks](docs/PHASE_2D_CHECKS.md). See [scrollable results tab and checks](docs/SCROLLABLE_VIEWER.md). Preparation remains the default and inference requires `--mode infer`.

The owner supplies unannotated team photographs, manually downloaded/extracted from Google Drive into a configurable project folder (proposed `data/input/`). The planned detection activity is functional checking and visual inspection, with no required annotation work, benchmark splits, precision/recall/mAP, or numeric accuracy targets.

The planned outputs are an in-memory display of original photographs with predicted boxes, raw decoded phrase labels, and decimal model scores in summaries; readable summaries; per-image JSON; and run/environment metadata. Preserve originals and do not automatically save annotated copies. Model-score percentages are not accuracy, calibrated correctness probabilities, or image-area coverage. Inference and display outcomes are recorded separately under the [proposed output contract](docs/PHASE_1_IMPLEMENTATION_PLAN.md#7-proposed-json-contract). Actual Windows CUDA execution is recorded; broader detection quality remains unverified.

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

## Preparation-only usage

Use the existing verified Windows `.venv`; no reinstall is needed:

```bat
.venv\Scripts\python.exe scripts\detect_images.py --help
.venv\Scripts\python.exe scripts\detect_images.py --input data/input --results results
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`--config`, `--input` and `--results` accept absolute paths or paths relative to the repository root, regardless of shell working directory. Configuration is [config/detection.json](config/detection.json). Missing/empty inputs return 2; image read/decode errors return 1 after continuing the batch; unsupported-only input returns 2 with skips recorded; completed preparation returns 0. Configuration/persistence failures return 2. Generated output is JSON only under an ignored run directory. `prepared` means decoded/oriented/RGB-converted in memory, with inference explicitly unrun. No predictions are generated. Default display intent is enabled, but preparation records always say `not_attempted`.

Reusable helpers live in [scripts/](scripts/): `configuration.py`, `inputs.py`, `results.py` and `persistence.py`. Section 7 functional-result construction requires explicit inference completion/provenance. New baseline modules are `inference.py`, `postprocessing.py`, `pipeline.py` and `viewer.py`. Synthetic predictions appear only in tests and are not observed model evidence.

## Authorized inference usage

In the unchanged verified Windows environment, first run the new software checks, then invoke inference on manually supplied unannotated images:

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\detect_images.py --mode infer --input data/input --results results
```

Inference mode may download pinned files on first use; `--local-files-only` requires the existing cache. Preparation remains the default. All processed images appear together in one scrollable Tk/TkAgg tab, repeating Image #N, Detection Summary, Text-Token Details and Configuration. Original pixel ticks, raw decoded phrase labels above boxes and zoom/pan are preserved. Passing-only tables are selectable/copyable; configuration shows the actual saved prompt, thresholds, model/revision and device. Full precision and raw token diagnostics remain in JSON; no temporary images or annotated exports are saved.

Test `.venv\Scripts\python.exe scripts\detect_images.py --mode display-check`. Replay without inference using `--mode replay --result <per-image.json> --image <matching-original>`; repeat both arguments to show several images in one tab. Every source is verified before display. See [exact run/multiple-image replay commands and evidence](docs/SCROLLABLE_VIEWER.md). `--no-display` retains text output; `--no-display --terminal-mode full` exposes complete raw diagnostics. Legacy preview-width/pagination options do not control the GUI.

Inference predictions are saved before the shared tab opens, and display stays pending until normal closure. GUI failures preserve predictions and receive separate display status; interrupted or failed final updates can leave pending. Unmapped is distinct from failed inference or threshold rejection; phrases are never completed or forced into categories.

Exit codes: 0 for completed requested processing/display; 1 for per-image processing or enabled-display errors after batch continuation; 2 for configuration/input-folder/persistence errors or unsupported-only inputs; 130 for interrupted inference runs. `no_detections` only follows successful model/postprocessing with no above-threshold or invalid candidates. Candidate counts are not physical object/independent-region counts, which remain unknown. Scores are not accuracy.

## Dependency reproduction reference

[requirements.txt](requirements.txt) contains direct supporting pins; [requirements-cuda.txt](requirements-cuda.txt) selects the original cu118 source for torch/torchvision. [requirements-windows.lock.txt](requirements-windows.lock.txt) transcribes all 36 observed packages, including bootstrap tools, with the cu118 supplementary index. It is a version snapshot without artifact hashes. No package reinstall or fresh installation was tested in 2B/2C. For a future clean Windows Python 3.11 x64 environment only, the complete snapshot can be applied with `python -m pip install -r requirements-windows.lock.txt`; the original two-source route uses `requirements-cuda.txt` followed by `requirements.txt`. Revalidate affected readiness on a rebuilt environment. Preserve the existing target environment.

## Next step

Review [the completed first real-image experiment](docs/TACO_STAGE_4_RUN.md) locally before any further training or promotion: detection AP deteriorated despite lower loss. No additional preparation phase or optimization is started automatically. Earlier preparation and inference follow-ups below are historical context.

Review [TACO audit counts, commands and pending decisions](docs/TACO_PREPARATION.md), particularly excluded/unlabeled objects in mixed images, EXIF coordinate alignment, duplicate IDs and negative boxes. Human annotation-viewer acceptance and a handling policy are needed before any later training-data export. The inference functional-check history follows below.

Review [Phase 2E comparisons and limitations](docs/PHASE_2E_RESULTS.md) and [handoff](docs/HANDOFF.md). Corrected small/nested-box label placement has automated Windows replay and agent overlay evidence; final human confirmation remains pending. Clear piles and no-trash scenes are unavailable. Stop before Phase 3 decisions or optimization; no merge, grouping or training is authorized.
