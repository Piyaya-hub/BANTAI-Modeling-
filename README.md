# BANTAI Modeling

## Current Stage 5 state through D-075

Stage 5 adds paper to the unchanged nine-specific taxonomy and keeps the 65-image
validation / 65-image test memberships frozen. Best measured Stage5 combined
**nine-specific validation AP is 0.231226**, from `soup_tier1_hr_blend2575` at 384/576.
D075 persists that exact state as a private **inference/replay-only** bundle, with
`resume_supported=false`; this is not production promotion or test generalization.
No Stage5 weights are wired into `scripts/detect_images.py`, and no merge is authorized.

| Decision | Measured progression / outcome | Official nine-specific AP (combined 65) | Diagnostic 5-material / 4-material AP |
| --- | --- | --- | --- |
| D-070 | 307-image, 256/384, two epochs | baseline 0.057617 -> E2 0.135663 | - |
| D-071 | 384/576 saved blend2575; HR E1 | 0.171262; 0.202020 | - |
| D-072 | HR E1 hierarchy and four-view GUI | same saved HR E1, not rerun | 0.314188 / 0.398843 |
| D-073 | 1,500-record expansion audit; 1,370 fresh photos + 130 frozen references | metadata only | - |
| D-074 | Tier 1: 336 training images; 384/576 E1; 20/80 soup | 0.218215;**0.231226** | soup 0.344601 / 0.402577 |
| D-075 | Exact soup persisted on CPU; docs/guarded checks/publication | no new forwards or metrics | no new evaluation |

All AP values are validation measurements. Material AP changes category agreement and
macro averaging; it must not replace official nine-specific AP. Separate original
47-image eight-prompt legacy results remain in the detailed reports. The historical
Stage4 eight-class champion D061 AP 0.265563 / coverage 38 of 75 is a different benchmark,
not interchangeable with the Stage5 combined 65-image /109-GT result.

Detailed evidence: [D070 training](docs/TACO_STAGE_5_TRAINING.md),
[D072 hierarchy and GUI](docs/TACO_STAGE_5_MATERIAL_HIERARCHY.md),
[D073 expansion audit](docs/TACO_STAGE_5_DATA_EXPANSION_AUDIT.md), and
[D074 Tier1 results/replay](docs/TACO_STAGE_5_TIER1_TRAINING.md).
Private runs, exports, rights records and weights are intentionally not published.
The durable soup is `runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/checkpoints/soup-tier1-hr-blend2575`.

Saved-only local replay auto-discovers the D074 run when complete:

```powershell
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_stage5_visual_review.py --gui --image-id 777 --score-threshold 0.10 --box-top-k 15
```

An explicit `--tier1-run-dir` overrides discovery. Without private runs/exports,
the viewer reports what is unavailable; it does not download weights or run inference.


**Historical Stage 4 wrap-up: complete through D-067; D-068 authorized synchronization and commit/push to PR #18 on `feat/stage4-real-image-training`.**
The best measured **official eight-class validation state** is D-061
`blend_350_442_2575`: COCO v2 **AP 0.265563**, without NMS or taxonomy remapping.
Its separate eight-class candidate coverage is **38/75** GTs, 15/24 cigarettes
and 5/7 Image-777 cigarettes at score >=0.05 and IoU >=0.50, over all 900 queries
with non-exclusive matching. The 38/75 count is not an official COCO recall metric.
This experimental in-memory weight blend is **not promoted and not wired into
`scripts/detect_images.py`**. All 64 saved checkpoint bundles remain private;
D-067 records their verification, and D-068 does not open or modify them.
No GPU/model execution, further experiment, merge or promotion is authorized by D-068.

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
| D-061 blend_350_442_2575 (best measured official AP) | 384/576 | 0.265563 | 0.403085 | 0.231365 | 15/24 |

D-061 has the highest measured official eight-class AP; E1 retains higher AR100.
Inspection/averaging on this small validation set retains selection bias and does
not establish generalization or checkpoint promotion. Diagnostic post-NMS AP and
mapped 4/3-category AP describe different conditions/tasks and must not replace
the official eight-class results above. [Full wrap-up and next actions](docs/HANDOFF.md).
The [D-042 Section 185 academic-use record](docs/TACO_STAGE_4_ACADEMIC_USE.md)
is owner workflow authorization, not external permission, institutional approval or definitive legal clearance.
Detailed reports retain links to private local audit artifacts; these artifacts and
optional `work/` helper tests are not distributed. Tests requiring absent private
helpers skip explicitly. The baseline inference configuration remains unchanged.

## D-060 through D-067 outcomes

Negative results below mean the intended quality objective was not achieved;
they are retained evidence, not successful optimization or an invitation to retry.

| Decision | Recorded outcome |
| --- | --- |
| [D-060 saved replay](docs/TACO_SAVED_REVIEW_384_576.md) | Saved 384/576 endpoint/trajectory review, both GUI modes, all 47 validation images; historical 256/384 choices preserved. |
| [D-061 weight integration](docs/TACO_STAGE_4_576_WEIGHT_INTEGRATION.md) | Four averages evaluated; 25/75 step 350/442 blend gives best official AP 0.265563 and raw eight-class coverage38/75. No promotion. |
| [D-062 cosine continuation](docs/TACO_STAGE_4_576_LR_DECAY.md) | **Negative:** endpoint AP 0.221592 and cigarette coverage 10/24, below the best blend and its15/24. |
| [D-063 NMS audit](docs/TACO_STAGE_4_NMS_AUDIT.md) | Saved-only class-aware/agnostic diagnostic suppression; official no-NMS metrics unchanged. |
| [D-064 pure-film damping](docs/TACO_STAGE_4_576_CLASS_BALANCED.md) | **Negative/partial recovery:** AP 0.258631/0.242594/0.257303 at350/442/blend; cigarette coverage 13/14/13 of24. Did not preserve the desired combined gains. |
| [D-065 material hierarchy](docs/TACO_STAGE_4_MATERIAL_HIERARCHY.md) | Reversible specific/material/polymer views and containment NMS; mapped metrics are diagnostic. The superseded backbone-unfreeze attempt stopped after 10 updates with zero new bundles; it is not a successful result. |
| [D-066 cigarette LR boost](docs/TACO_STAGE_4_576_CIGARETTE_BOOST.md) | **Negative:** endpoint/blend AP 0.250675/0.257129, cigarettes 14/24 and Image 777 4/7, below the prior blend's 15/24 and 5/7. |
| [D-067 score objective](docs/TACO_STAGE_4_576_SCORE_CALIBRATION.md) | **Negative:** endpoint/blend AP 0.245944/0.251234, Image 777 4/7; Image 833 cigarette FP at 0.05 increased 12 ->21/20. Strict F1 at 0.10 fell 0.162921 ->0.116129/0.121622. |

For saved visual review, the recommended **display-only** starting settings are
**threshold 0.10 or higher** and **`agnostic IoU 0.50 + containment 0.80`**.
Select these explicitly in the toolbar; historical raw-query/.05 defaults remain.
They reduce clutter and can hide distinct/low-score objects, including cigarettes;
they do not change official metrics or establish a production operating threshold.
The [saved replay guide](docs/TACO_SAVED_REVIEW_384_576.md) gives launch commands
and the optional `--box-top-k 15` setting. Human visual acceptance remains pending.

Run CPU software verification without model execution or private-artifact writes:

```powershell
& .\.venv-training\Scripts\python.exe -B .\scripts\verify_stage4_guarded.py
```

Logs and generated fixtures go to the printed system-temporary directory.

## Historical Stage 4 investigation notices

Earlier status and stop notices below describe their respective completed substeps;
D-068 is the current documentation/publication authorization; older notices retain historical scope.

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
