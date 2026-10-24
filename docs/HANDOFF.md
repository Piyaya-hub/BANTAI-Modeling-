# BANTAI Stage 4 handoff

## Final Stage 4 wrap-up and publication identity (D-059) — 2026-10-24

D-043–D-058 investigations are complete. D-059 authorizes this focused
documentation/scripts/tests publication on `feat/stage4-real-image-training`
to update PR #18, without merging it or promoting any checkpoint. README and
PHASES now include the seven-row corrected-protocol benchmark progression and
links to [D-057 training](TACO_STAGE_4_576_TRAINING.md) and
[D-058 trajectory results](TACO_STAGE_4_576_TRAJECTORY.md).

Publication base SHA: `c79d513e8990701f85db0cfe5a0acf4408de1d56`.
The wrap-up SHA is the single commit containing this entry, resolved with
`git log -1 --format=%H -- docs/HANDOFF.md`; its exact SHA and verified remote
branch status are reported to the owner after push. A commit cannot contain
its own SHA in its committed text. PR #18 remains open for owner review/merge.

### Preserved corrected checkpoint inventory

Each run retains eight periodic step bundles plus E1/E2 bundles:10 per run,
50 total. D-059 performs no checkpoint/model execution or modifications.

| Experiment | Training scale | Seed / LR | Private training run | Bundles |
| --- | --- | --- | --- | --- |
| D-048 corrected reference | 256/384 | 42 /1e-5 | `taco-312e568703a54f98a7c8f4b87b29cd39` | 10 |
| D-050 lower LR | 256/384 | 42 /5e-6 | `taco-dd541c3972834a64a16771df44a7e891` | 10 |
| D-051 paired replication | 256/384 | 123 /1e-5 | `taco-4bf521d7529e4fa4b168b96786020844` | 10 |
| D-051 paired replication | 256/384 | 123 /5e-6 | `taco-e7e4008e98a241e19047e14645f06ec1` | 10 |
| D-057 scale-matched | 384/576 | 42 /1e-5 | `taco-896e2f44146d4d54aee518e6087908ca` | 10 |

All remain experimental. Step350 has the highest sampled eight-class AP0.262337,
but lower AR100 than E1 and fewer correctly covered large objects; it is not
selected or promoted. Cigarette gains do not establish overall generalization.
The owner Section185 academic-use record (D-042) is not external permission,
institutional approval or definitive legal clearance.

### Verification and owner-only next actions

The D-059 guarded fast unit/CPU suite ran234 tests:219 passed,15 documented
skips,zero failures/errors. All three trajectory fixtures and the local scale,
seed/LR and saved-review fixtures passed. Protected file metadata (59,827 files)
matched before/after verification;50 corrected bundles remain across five runs.
All212 relative links across the14 authorized Markdown files resolve locally
(18 link to retained private artifacts). Staging/link/whitespace evidence is
recorded in this publication's commit message and final owner handoff. Live
optimizer-step and model-execution fixtures are excluded; no GPU/model runs.
Private helper-dependent tests skip explicitly on checkouts lacking `work/`.
Detailed report links to local audit artifacts remain private and are not
published with the scripts/tests/reports. `runs/`, `work/`, `exports/`, `data/`,
`BANTAI-Datasets/`, environments and rights records remain untouched and unstaged.
`ARGON_RESEARCH_HANDOFF.md` and `BANTAI_CONCEPT_PROPOSAL.md` remain local,
unstaged planning references.

Only the owner may elect optional human visual review, decide PR #18 review/merge,
or authorize a future stage. No further experiment, model selection, promotion,
test evaluation or merge follows automatically from this wrap-up.

## D-058 mid-Epoch-2 384/576 trajectory audit — 2026-10-24

Owner authorized inference-only evaluation of D-057 step250/300/350/400 on47
validation images,188 forwards total, reusing saved step0/221/442 endpoints.
Observed status: **completed**. 188 validation forwards completed; all three gates passed. Cigarette same-class trajectory [8, 10, 11, 14, 14, 15]; large-bin same-class trajectory [21, 18, 16, 15, 15, 14]. No checkpoint selection or promotion.
[Trajectory report, flip ledger and limitations](TACO_STAGE_4_576_TRAJECTORY.md).
Audit `scale576-trajectory-c84b9d9344e64578ba1f2fb85f353d0c`. Validation checkpoint inspection is diagnostic and
introduces selection bias; it is not a generalization claim or selection rule.
No training/backward, train-photo/test access, core/GUI/dependency changes,
exports, checkpoint promotion or Git publication. Stop for owner review.

## D-057 controlled 384/576 training — 2026-10-23

Owner authorized one3-step discarded pilot plus442 full updates from original
SNAPSHOT at seed42/LR1e-5/FP32/frozen backbones,221 train/47 validation. Only
the two trainer CLI bound values change512→576; defaults remain256/384.
Observed outcome: **completed**; 3 discarded pilot +442 full updates completed.
All five gates passed;231 software tests,zero failures/errors (three skips).
Trainer `taco-896e2f44146d4d54aee518e6087908ca` completed both221-image epochs.
Epoch1 AP0.249891/AR1000.488141 vs D-0550.247003/0.474763; Epoch2 AP0.170508/
AR1000.370118. Cigarette same-class coverage at0.05 is5→8→15/24; image777
0→1→5/7. Epoch2 improves small-bin coverage but loses ten previously correct
GTs and large-bin coverage, with bag AP collapsing on only two GTs.
Ten bundles verified; final optimizer step442 across558 parameters and exact
seed42 image order. Full allocated/reserved peaks3.55/3.93 GiB. All1766 protected
hashes matched after execution. Seven-state metrics CSV,1500-row per-object
CSV/JSON and full training/gradient/transition summaries are in the audit.
[Full gates, results and limits](TACO_STAGE_4_576_TRAINING.md).
Audit `scale576-experiment-0b2a846e659a429cbe8a65c1ca0444e1`. No retry, promotion, test access, dependency/GUI changes,
annotated exports or Git publication. Stop for owner review.

## D-056 multi-checkpoint resolution replication completed — 2026-10-23

[Full three-model report](TACO_STAGE_4_RESOLUTION_REPLICATION.md).
Run `resolution-replication-adf0cda0193041e38be1380b29b3e50b` passed all four gates and exactly188 real forwards:
47 low +47 high for Pretrained and123/1e-5/E2. Both low baselines exactly match
saved full predictions/raw queries/metrics/loss. D-05542/E1 was read-only.

Pretrained AP0.111576→0.147847;123/E2 AP0.152880→0.140163. D-055 reference remains
0.203932→0.247003. Seven-class AP excluding bag rises in all three. At .05,
cigarette same coverage1→3,2→5,6→10/24;123/E2 reaches19/24 at floor0, not an
operating-threshold endorsement. Image777 GT279/280 gain same-class at .05,
GT284 retains it and GT278 becomes wrong-only. Common gain GT497; common
new-wrong GT67/278; no regression shared by all three. Partial scale effect,
not uniform AP superiority or a complete classification/tiny-object repair.

229 software tests:215 passed,14 documented skips; all four resolution fixtures
passed. All1521 fresh protected hashes matched after model execution;40 checkpoint
bundles verified, both D-055 attempts unchanged. Model hashes unchanged; exactly
188 validation-photo reads, zero backward/updates/train-photo/test access.
225-row paired75-GT×3-model CSV/JSON and full summary archived locally. No
core/GUI/config/dependency changes, annotated exports, publication or promotion.
Stop for owner review; next-step proposals are unexecuted.

## D-055 single retry completed — 2026-10-23

[Full measured report](TACO_STAGE_4_RESOLUTION_EXPERIMENT.md).
Run `resolution-experiment-9cebf39b0982405d8dfdd954ae2dbed1` passed all five gates and exactly47 high-scale
validation forwards, after47 exact baseline forwards and one image777 pilot.
AP0.203932→0.247003; AR1000.396766→0.474763. At .05/.50, same-class coverage
28→33/75 (eight gains,three losses); any coverage49→56. The original4–<8px bin
improves any/same4/1→8/4, but <4px remains0/14. Cigarette any/same2/2→8/5;
image777 gains localization but no qualifying same-class result at .05.
Regressions GT60/88/250 are all originally larger objects. No automatic promotion.

Software228 run,214 passed,14 documented skips; all three resolution tests passed.
Exact full baseline predictions/raw queries/metrics/loss and unchanged parameter
hash verified. All1497 fresh protected hashes matched after execution;40 checkpoint
bundles retained. GPU inference fit does not prove high-scale training fit.
Private artifacts include both-scale geometry, baseline replay, pilot, high
evaluation,75-row paired CSV/JSON and fixed-bin/transition summary. Original
failed attempt is unchanged; retry count is consumed. Zero updates/backward,
training-photo/test access, annotated exports, dependency/core/GUI changes or Git
publication. Proposed human class/annotation review and one-checkpoint replication
remain unexecuted. Stop for owner review; earlier stopped notice is historical.

## D-055 stopped at software gate — 2026-10-23

[Resolution attempt report](TACO_STAGE_4_RESOLUTION_EXPERIMENT.md).
Attempt `resolution-experiment-629229adf1c6480d930aac79250d18f8` ran no real model inference, backward or optimizer steps.
The guarded228-test suite reported1 failure,89 errors and13 skips; all three new
resolution fixtures passed, but the overall gate failed. The harness erroneously
blocked synthetic fixture image saves and omitted an indirect gradient fixture
from exclusions. Corrected work-only guards are prepared but were not rerun.
All later gates, metrics, memory pilot and47-image comparison remain unattempted.
Read-only post-stop verification found579 prior-anchor files unchanged
and verified40 checkpoint bundles;221 training photos were not read or rehashed.
Core/GUI/config source remains unchanged. Stop per owner gate rule; a fresh
attempt requires explicit owner authorization. No commits/pushes/merges occurred.


## Small-object/classification audit completed — 2026-10-22

D-054: [Full audit and proposed next experiment](TACO_STAGE_4_SMALL_OBJECT_CLASSIFICATION_AUDIT.md)
verified actual CPU preprocessing/masks for all47 validation images and75 boxes,
then analyzed all nine saved states. Image777 content is350×256, not384-square.
Small resized boxes strongly associate with poor candidate coverage;20 pretrained
and21–27 trained annotations/state have any-class IoU>=.50 without correct-class
coverage at .05. Cans/wrappers have persistent class-related failures; cigarette
failures mostly lack qualifying any-class coverage. Causation remains unproven.

Nine focused checks passed;40 checkpoint bundles and800 protected files verified.
Original GUI, manifests, datasets, weights and current owner source changes are
unchanged. Initial owner image777 observations are recorded separately from
automated results. The single proposed experiment is fixed-weight inference-only
256/384 versus384/576 on the preserved42/1e-5/E1 checkpoint, with GPU fit unmeasured.
It has not run or been authorized. No new training/inference or Git publication.
Stop for owner review; no additional experiment is approved.


## Per-object saved diagnostics completed — 2026-10-22

D-053: [Per-object report and exact GUI command](TACO_STAGE_4_PER_OBJECT_REVIEW.md)
covers all eight image777 GT annotations and46 GT annotations in the unchanged
20-image review. Nine saved states are compared at all900/top100 ×0/.05/.25;
complete JSON/CSV evidence is separate from the original review manifest.
The explicit --per-object saved GUI supports GT selection/navigation, highlights,
zoom, complete overlap tables and a scrollable GT/model summary. Default GUI and
terminal routing are unchanged. For seed123/1e-5/E2, GT284 is hidden by the five-box
cap, while other missing GT have score or candidate-coverage limitations.
All29 focused tests and native checks of four object-mode choices plus standard
mode passed; the 20-image subset rendered. Human identity/annotation judgement
remains pending. No training, inference, dataset/metrics changes or Git operations.
Stop for human review; no experiment is authorized.


## Cross-seed analysis and visual-review preparation completed — 2026-10-22

D-052 is analysis only. [Full cross-seed report](TACO_STAGE_4_CROSS_SEED_ANALYSIS.md)
recomputes all nine saved COCO metric records, checks all 40 checkpoint bundles,
and compares class AP/AR, same/any candidate coverage, fixed-score diagnostics,
incorrect-class assignments and descriptive two-seed ranges. Cigarette/film
AP and AR improve over pretrained in all eight states; wrapper/can/Styrofoam
recall is lower throughout. LR/epoch AP rankings reverse by seed.

Twenty validation images and specific human questions are prepared in the local
review manifest. The report gives the exact PowerShell command and four run
keys. Explicit `--saved-review`/`--run-choice` reuses the four-panel viewer;
historical default replay and strict training verification remain unchanged.
All four saved choices and the actual terminal CLI passed; native replay rendered
all 20 examples and closed normally with zero exports. All 22 focused tests passed.
Human photo/material/annotation acceptance is pending. Recommended next action
is human data-quality review; a third paired seed remains a proposal, not
authorization. No new training/inference, test evaluation, promotion or Git
publication occurred. Stop for owner review.


## Paired seed-123 replication completed — 2026-10-22

D-051's A (1e-5/123) and B (5e-6/123) both completed and passed integrity audits:
884 full updates plus six discarded pilot updates, no extra runs. A AP improves
0.140478→0.152880; B AP regresses 0.137919→0.113018, reversing the seed-42
epoch pattern. Both rates change best AP epoch across seeds. This weakens a
general lower-LR prevention claim; two seeds do not establish superiority.

Configurable `--seed` defaults to 42 and retains the LR interface. Five seed
regression tests passed; guarded discovery passed 211 tests with three documented
skips. Original data, weights, driver/runtime, native adapter, rights gate and
exact baseline passed. Both full starts have identical RNG/parameter hashes and
442-image ordering; pilot-release checks passed. Twenty checkpoint bundles and
756 protected files passed, including unchanged A artifacts after B.
Study: `runs/training/seed123-pair-10acabc298064547bfab6d119a109359/`.
A: `runs/training/taco-4bf521d7529e4fa4b168b96786020844/` (AP/AR100 E2
0.152880/0.274295). B: `runs/training/taco-e7e4008e98a241e19047e14645f06ec1/`
(AP/AR100 E2 0.113018/0.291915). Both use seed 123.
[Replication report](TACO_STAGE_4_SEED_REPLICATION.md) includes the full cross-seed
deltas, class/coverage/score/loss tables, runtime caveats and evidence.
The original 42/1e-5/E1 remains strongest measured AP/AR100 and is not promoted.
Only these two runs were executed; a third paired seed 456 is a proposal only.
Stop for owner review, with no promotion, test access or Git operations.

## Controlled LR-only experiment completed — 2026-10-22

D-050's one fresh two-epoch LR 5e-6 comparison completed, exit 0, with exactly
442 full-run updates after three discarded pilot updates. New E1 AP/AR100:
0.137099/0.330432; E2: 0.145952/0.302977. Lower LR avoids the previous aggregate
AP collapse and beats the 1e-5 E2 result, but does not preserve the stronger 1e-5
E1 result (0.203932/0.396766). Same-class all900 coverage still falls 32→30.
No checkpoint is promoted. The next seed-only replication is a proposal only.

Guarded preflight passed: 206 tests, three documented skips, all 268 source/
geometry checks, exact pretrained baseline agreement and unchanged runtime/data/
rights. Four new LR tests passed. Actual optimizer groups and both saved epoch
optimizer states confirm 5e-6, with states at 221/442. Ten immutable bundles and
all 612 protected file hashes passed; seeded training coverage is exact.
Run: `runs/training/taco-dd541c3972834a64a16771df44a7e891/`.
Evidence: `runs/training/lr5e6-7d3cd8dad9064aa9949fb2da4bf421c0/`.
[LR report](TACO_STAGE_4_LR_EXPERIMENT.md) contains all five model states, per-class
AP/AP50/recall/coverage, score sweeps, losses, numeric curves and preservation.
The same run continued after a long pre-update wall-clock gap; lifecycle timing
is not an active-compute benchmark. No optimizer state or weights were resumed
from an earlier fine-tuned checkpoint. Historical
source-pinned TACO replay policies remain unchanged and reject the new trainer
fingerprint; normal BANTAI inference is unchanged. Training has stopped after
epoch 2/final validation. Await owner review, without promotion, test access,
further optimization or Git operations.

## Corrected-run diagnostics completed — 2026-10-21

D-049's [diagnostic report](TACO_STAGE_4_CORRECTED_DIAGNOSTICS.md) explains the
observed E1→E2 regression using saved validation results, without new inference
or training. Any-class all900 coverage at IoU >=0.50 rises 55→57 annotations;
same-class coverage falls 36→30. Bags/styrofoam lose useful class assignments
despite retained well-aligned boxes; cups also lose AP. Score shifts are mixed.
Overfitting, forgetting and excessive learning rate are not established causes.

Forty-five existing tests and three local diagnostic fixtures passed. Native
corrected-run replay checked 20 representative images, overlays, raw phrases,
zoom/pan and navigation; human photo-level confirmation remains pending.
Evidence: `runs/training/corrected-diagnostics-629e30fb299f442698de14650b279c6c/`.
The continuation rechecked 603 prior protected files: 602 still match; the owner's
production detection configuration changed before continuation and is preserved
with its exact current hash recorded separately. Training/model/data/checkpoint
identities remain unchanged. No existing source file was edited for this task.

The report ranks a fresh two-epoch LR-only comparison (1e-5→5e-6) first as a
proposal. No next experiment is implemented or authorized. Stop for owner review;
no test access, checkpoint promotion, uploads, commits, pushes or merges.

## Controlled corrected-target restart completed — 2026-10-20

D-048's single two-epoch run completed, exit 0: three discarded pilot steps,
442 full-run updates, no historical resume or further training. See
[complete corrected-retrain results and curves](TACO_STAGE_4_CORRECTED_RETRAIN.md).
Corrected v2 AP / AR100: pretrained 0.111576 / 0.363077; new epoch 1
0.203932 / 0.396766; new epoch 2 0.084500 / 0.211229. Epoch 2 regresses despite
lower corrected-target validation loss. Both new checkpoints remain experimental.

Preflight: 202 tests passed, three explicit skips, including all eight coordinate
tests passing; exact pretrained baseline replay and all 268 source/geometry
checks passed before any update. Ten new checkpoint bundles and 471 protected
file hashes passed; optimizer states are at step 442. Settings, rights
uncertainties, dataset, owner detection edits, source code and historical evidence
are preserved. No test access, dependency changes, uploads, promotion or Git
operations. New run: `runs/training/taco-312e568703a54f98a7c8f4b87b29cd39/`;
experiment: `runs/training/corrected-retrain-72650648148549ccb8aa8a176f9c5b0b/`.
Next action is owner review. A further single-variable experiment is a proposal,
not authorization; no automatic continuation.

## Explicit GUI opt-in — 2026-10-20

D-047 restores the existing four-panel viewer through `--gui` while retaining
D-046's terminal-only default. From PowerShell at the repository root:
`.\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --gui --image-id 42`.
The all-validation selector includes 42, 113, 200, 31, 1096, 252 and 1014.
Saved-result verification remains mandatory. Fresh GUI analysis JSON records
display status separately; original artifacts and all modelling settings remain
unchanged. No commits, pushes or merges. Human photo-level review remains pending.

Verification: 18 software tests passed; the actual default command printed the
verified metrics and exited 0. Native TkAgg replay checked all 16 representatives
(including all seven requested IDs), shared zoom/pan, tables/highlighting,
diagnostic controls and all-validation navigation with zero guarded model,
training or image-export calls. Evidence:
`runs/training/step2-review-be0b248ac2554160a06eac179cb6cc9b/native-runtime.json`.
The explicit `--gui --image-id 42` entry check passed and persisted `closed` in
`runs/training/step2-review-2901e7965fe9474584bf20a760f6e005/analysis.json`.

## Owner correction: terminal epoch output — 2026-10-20

D-046 supersedes the Step 2 epoch comparison GUI request: the owner wants only
terminal output per epoch with the updated scores. The diagnostic window was
closed normally. `taco_visual_review.py` now verifies saved provenance and prints
pretrained/epoch-1/epoch-2 AP, AP50, AP75, AR100, corrected-target validation loss,
score median/P99/maximum and counts >=0.25. Default execution opens no GUI, reads
no photographs, performs no inference/training and creates no new report files.
Run `.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py`.
Prior GUI evidence is historical; it is no longer the owner-requested workflow.

## Step 2 diagnostic replay — 2026-10-20

D-045 authorizes controlled local-only saved-result visual diagnostics and initial
error analysis. [Workflow, commands, selection, measured findings and pending human
checks](TACO_STAGE_4_VISUAL_REVIEW.md). Four synchronized TkAgg views, source hashes,
category/query/score labels, separate raw phrases, copyable tables, all-900 access
and full-validation navigation are implemented without inference/model loading.

Sixteen deterministic representatives cover all eight classes and contrasting
outcomes. Same-class IoU-0.50 matches at diagnostic .05 are 28/17/14 of 75; at .25,
epoch models hide 17/14 otherwise matching retained candidates. Wrapper matches
fall 9->1->1, can matches 2->1->0, styrofoam matches 4->2->0. These are annotation
comparisons, not confirmed visual errors/physical counts. Unmatched predictions
are not automatically false positives. Human feedback/annotation completeness and
readability remain pending; the separate interactive owner window may remain open.

Final analysis: `runs/training/step2-review-cce4f3dcac4c4c788cc8cab5957d396e/`.
Actual native replay checked all 16 representatives, synchronized coordinates,
tables, zoom/pan propagation, empty views and clean closure without image exports.
Fourteen new tests pass; safe full suite 201 tests, 198 passed/three skipped. No
training, test access, source/data/checkpoint changes, commit or promotion. Proposed
future training is a controlled corrected-target restart after human review, not an
approved run or automatic continuation. Stop after this diagnostic deliverable.

## Step 1C historical reevaluation — 2026-10-20

D-044 authorizes a separate evaluation-only provenance pathway and unchanged
validation-split reevaluation. [Design, commands, qualification and full results](TACO_STAGE_4_REEVALUATION.md).
Exact historical artifact/source anchors, reviewed Step 1B old/new source pairs,
current code/native sources, corpus/rights and model/processor/tokenizer/adapter
compatibility pass. Strict training-resume verification remains unchanged and
still rejects the corrected context for both checkpoints.

Corrected pretrained / epoch-1 / epoch-2 AP: **0.111576 / 0.126627 / 0.066215**;
AR100: **0.363077 / 0.275363 / 0.235395**. Epoch-1 higher aggregate AP coexists with
lower recall; epoch 2 regresses. All score/category outputs match historical ones
exactly; neither trained checkpoint has a score >=0.25. Small-object AP stays zero.
Recalculated valid-target losses differ from historical padded-target losses.

Successful private record: `runs/training/step1c-evaluation-3d24a67e1fe041e5a8acf56a26406db4/`.
94 validation photo reads, zero updates/test access/state restoration; guarded
runtime calls all zero. Nineteen new tests pass; full safe suite 187 tests,
184 passed/three skipped. Existing sources, artifacts, owner edits and packages
are preserved. No commit, push, merge, promotion or visual analysis occurred.

Next proposed action is Step 2 local visual comparison, requiring its own scope
authorization. The Step 1B historical reevaluation blocker below is now resolved
only through the separate evaluation contract; training resume is not relaxed.

## Step 1B coordinate correction — 2026-10-20

D-043 authorizes only the confirmed target/decoder corrections, independent tests
and validation-only comparison. [Implementation, evidence and limitations](TACO_STAGE_4_COORDINATES.md):
native valid-image targets and original-size decoding now pass eight independent
regressions; original functions fail them. Broader suite: 168 tests, 165 passed,
three skipped (one privilege limitation, two prohibited toy optimizer fixtures).

Corrected pretrained validation on the same 47 images/75 annotations: AP
0.111576016, AP50 0.158252880, AP75 0.135975042, AR100 0.363077235. All 42,300 query
scores/categories exactly match historical baseline; geometry alone changed the
measurement. New private evidence is under
`runs/training/step1b-evaluation-c8b127d8bf364c7dbb735795cd81c06d/`.

Epoch-1/2 artifact hashes pass, but their exact-source/protocol contracts reject
the corrected implementation. Those reevaluations are stopped without bypass or
manifest rewriting. Next owner decision: explicit evaluation-only provenance
handling for historical weights, preserving strict training-resume checks.
Earlier reported AP/loss below are historical measurements with incorrect geometry.
No new optimizer updates, test access, Step 2 analysis, promotion or PR merge.
Owner detection edits, original weights/data and environments remain preserved.

## Completed first real-image run — 2026-10-19

D-042 records the owner's actual non-commercial/local academic declaration and authorizes the existing workflow on a defensible, documented Section 185 claim. [The evidence record](TACO_STAGE_4_ACADEMIC_USE.md) expressly claims no rightsholder permission, institutional approval or definitive clearance; the v2 gate keeps clearance flags false and binds the assessment/corpus. A strictly verified source-JSON formatting change was accepted only because default serialization reconstructs the exact original recorded hash; source/export files were not rewritten.

Actual run `taco-1698bcb46b85435786dd2a199910910f`: **completed two epochs / 442 full-run updates**, plus three discarded pilot steps. Full-run mean losses 14.304753 / 12.103961; validation loss 19.348786 baseline -> 12.133389 final. **AP [0,1] deteriorated from 0.000298934 to 0.000057591**. Fixed score 0.25 diagnostics miss all 75 validation annotations after fine-tuning. Runtime success is not detection-quality acceptance. [Full measured results, commands, class limits, failures and checkpoint location](TACO_STAGE_4_RUN.md).

Real pilot peaks: 2.384 GiB allocated / 2.535 reserved; full run: 2.382 / 2.539 GiB. Lifecycle 449.386 seconds. Native reload of the final `checkpoints/epoch-02` and replay on all 47 validation photos matched every prediction/raw query, metric and mean loss exactly, with zero extra updates. All ten checkpoint bundles/hash lists and final optimizer/RNG state passed audit. Full suite: 160 run, 159 passed/one existing skip. No prior CUDA synthetic investigation was repeated.

**Next action:** review poor detections locally before proposing any longer training or changing resolution/prompts/objective; no further run or optimization was started automatically. Human image/annotation/error review remains pending because no photos were uploaded for agent inspection. Test photos remain unused, model/split/taxonomy/inference/environment preserved, experimental weights private and unpromoted, PR #18 unmerged. Training source is commit `c8c7102`; preserve the ignored rights record and bound assessment bytes for within-budget checkpoint resume. The first experiment has no remaining runtime blocker; it has a failed detection-quality outcome.

## Initial Stage 4 implementation handoff (historical)

Current 2026-10-19: D-041 authorizes real-image fine-tuning and initial validation as one continuous task, conditional on documented photo and annotation/database rights. Branch `feat/stage4-real-image-training` builds on Stage 3D `22bcdbc` (open PR #17, carrying #14–#16; fetched main `077657c`). [Stage 4 implementation, rights scope, commands, resume behavior and evidence](TACO_STAGE_4.md) are available. The loader/trainer/pilot/progress/COCO evaluator and complete optimizer/RNG/sampler checkpoints are implemented. The verified Stage 3D adapter remains unchanged; no native miswired loss is used.

Implementation commit `601bb81` is pushed in [PR #18 targeting main](https://github.com/Piyaya-hub/BANTAI-Modeling-/pull/18), open and unmerged. It carries the unmerged Stage 3A–3D lineage; publishing this code does not publish dataset/model artifacts or clear the rights gate.

Full suite: 158 run, 157 passed/one existing privilege skip. Final generated-image CUDA continuation `76903099b57f453da9514e6a06fce128` passes: exact pre-update weights/optimizer/RNG/batch restoration, loss equality, post-update parameter maximum difference `4.10e-8` (not bitwise identical), encoder updates, frozen preservation and decoder equivalence. A same-model repeated-update control differs by `8.38e-8`; an isolated strict-determinism probe identifies unsupported CUDA grid-sampler backward. Earlier tolerance, shared-alias diagnostic and combined-probe OOM failures remain documented. No model/resolution/loss fallback occurred. Source photos were not opened. The actual real-data CLI stopped at `blocked_rights` with zero photo reads/optimizer steps. No real pilot, actual-photo memory result, TACO-trained checkpoint or real validation metric is claimed; human review remains pending.

**Next action:** owner supplies documented permission or approved institutional/legal basis, authority/date/evidence/scope/obligations covering the exact 221 training and 47 validation photos plus annotation/database rights. Metadata contains 161 unspecified/null, 30 generic CC and 77 ODbL-notice images in that scope. Then execute the documented command directly: actual-photo coordinate preflight, unchanged baseline, measured memory pilot and bounded run; stop/report technical failures without silently changing settings. Test photos remain untouched. No additional standalone assessment stage, dataset changes, cloud spending, publication, inference promotion or merge.

Only `pycocotools==2.0.10` was added inside `.venv-training`, with a pinned Windows wheel hash; existing package versions and inference `.venv` are preserved. Owner detection edits are excluded from commits. Generated images/checkpoints/reports and any actual private rights evidence remain ignored.

## Historical Stage 3D handoff

Current 2026-10-19: D-040 approves the training-only encoder-output adapter and synthetic qualification. Branch `feat/stage3d-training-adapter` starts at Stage 3C `6ce0907` (PR #16, carrying prior #15/#14; main at branch creation `077657c`). Owner detection edits remain local and excluded. See [implementation, commands, complete PASS/FAIL evidence and limits](TACO_STAGE_3D.md).

Final run `adapter-a06cfea197c545fc82c6df91c0b39583` is **PASS** for FP32 batch 2, frozen vision/BERT, 192/256 synthetic inputs: two corrected steps each for 11/2/1/0-target cases. Encoder proposal projection/normalization and box head receive nonempty gradients, Adam state and updates; empty regression is zero. All trainable parameters have gradient tensors. Decoder outputs remain exactly equal with/without the adapter at identical weights, before/after updates. Offline model/processor/adapter-context replay and repeated eval losses match exactly. Resume is explicitly unsupported.

Peak case allocator memory is 2.732 GiB allocated / 2.910 GiB reserved; separate reload verification peaks at 2.998 GiB reserved while the original model remains resident. Full suite: 144 tests, 143 passed/one existing privilege skip; eleven new adapter tests passed. Installed Transformers, both environments, original snapshot, inference, taxonomy, strict-clean manifest and earlier synthetic/investigation code remain unchanged. Experimental weights/reports stay ignored, with no promotion. Stop here: real-data training, full-resolution/AMP fit, rights, data coverage/leakage acceptance, resumable trainer and evaluator qualification remain unrun/unresolved. No merge is authorized.

## Historical Stage 3C handoff

Current 2026-10-19: D-039 authorizes native loss investigation only; experiments completed late 2026-10-18 local time. Branch `investigate/stage3c-training-loss` starts from Stage 3B `b7a350d` (open PR #15, carrying Stage 3A #14; fetched main remains `077657c`). Local owner detection edits remain excluded. See [complete causal investigation, commands and approval proposal](TACO_STAGE_3C.md).

Controlled native CUDA cases with 11/2/1/0 targets completed, reset to the same pretrained state before each single native update. Encoder CE was 30378/167259/334564/334580; independent focal accounting and captured inputs identify learned-query/pre-encoder text scores rather than image-conditioned encoder proposal logits. Native boxes are detached; proposal-head probes have no gradients, Adam state or updates. Diagnostic selected encoder tensors recover box-head paths and give baseline CE 2.566 without changing the criterion. A separate factor audit proves graph restoration leaves loss values unchanged, whereas score routing changes matching. No hypothesis objective was optimized and no Transformers patch or library change occurred.

Peak diagnostic allocator memory was 2.535 GiB allocated / 2.840 GiB reserved. Tests: 133 run, 132 passed, one existing Windows privilege skip; seven new fixtures passed. Both environments, protected source/settings/splits, Stage 3B trainer and cached weights remain preserved. Next: owner review/approval of a proposed Stage 3D external training-only encoder-tensor adapter and synthetic qualification. Stop before that implementation and before real-data fine-tuning. Original-resolution/full-parameter/AMP training, rights/evaluation/resume and dataset coverage remain unresolved. No merge is authorized.

## Historical Stage 3B handoff

Current 2026-10-18: D-038 authorizes isolated training setup and synthetic-only smoke checks. Branch `feat/stage3b-synthetic-training` carries Stage 3A `0ed97d9` from open PR #14, based on main `077657c`. The owner's uncommitted detection settings are preserved. See [complete environment, commands, actual runs and limitations](TACO_STAGE_3B.md).

Installed `.venv-training` independently with the recorded Windows versions and SciPy 1.15.3; inference `.venv` is not modified. Twelve new tests pass. The full isolated suite runs 126 tests: 125 pass, one existing Windows privilege skip. The first CUDA attempt failed before loss because 128/192 images had too few encoder positions for 900 queries; a guard now rejects that condition. Explicit 192/256, FP32, batch 2, frozen vision/BERT passed two native forward/backward/AdamW updates and offline save/reload. Final run `ec713b1fbc47407b8d729d47fa13fbf2` records peak allocated/reserved memory 2.694/2.852 GiB and zero reload logit/box differences. All base snapshot/frozen parameter hashes remain unchanged.

Qualification remains limited: native encoder box predictions/losses have no gradients and ten otherwise-trainable encoder tensors receive none; encoder classification dominates the total loss (~60,766 initially). No loss/version patch was made. Next action is review of these findings and authorization for further trainer qualification before real-data fine-tuning. No TACO photo training, real-data metrics, changed exports/taxonomy, cloud upload/spend or inference changes occurred. Full-resolution/all-parameter/AMP fit, resume, rights and rare-class/leakage readiness remain unrun or unresolved. Do not merge automatically.

## Historical Stage 3A handoff

Current 2026-10-18: D-037 authorizes technical feasibility assessment only. Branch `docs/taco-stage3a-assessment` was created from completed Stage 2 `18b1094` and fast-forwarded to latest fetched main `077657c` after PR #13 was found merged externally; their file trees are identical. The owner's local `config/detection.json` edits remain excluded. See [recommendation, exact-version evidence, recovery bounds, rights gaps and proposed smoke protocol](TACO_STAGE_3A.md).

Recommended route is the pinned HF checkpoint with Transformers 4.51.3's existing loss in a separate future PyTorch trainer/environment. SciPy and pycocotools are absent locally. Token-span and synthetic COCO processor probes passed, but no model was constructed or trained. Training VRAM/backward/AMP/save-reload remain unrun. Approximately 24 GB GPU capacity is a planning recommendation, not a measured requirement. Excluded/crowd flags do not provide spatial loss ignoring.

The validated 315-image/538-object strict-clean export remains unchanged. Conditional review bounds: coordinates alone could add 44 images/55 mapped objects; known excluded-category cases could add 280/597. These are not approved additions. Of 315 retained photos, 185 have null license metadata, 35 only `CC`, and 95 an ODbL/OpenLitterMap notice; individual rights and annotation/database terms remain unresolved.

Next action is owner review/approval of rights handling, isolated trainer/smoke scope, hardware/budget and any expansion policy. No taxonomy change, replacement export, dependency installation or fine-tuning occurred. Stage 2's 13 focused tests passed again; independent validation and final documentation checks are recorded in the assessment. No merge is authorized.

## Historical Stage 2 handoff

Current 2026-10-18: D-036 authorizes strict-clean TACO conversion and splitting on existing `feat/taco-harmonization`, after the owner's representative annotation review. The local detection config is preserved and excluded from commits. New preparation/policy/validation code is independent of the unchanged detector, taxonomy, annotation viewer and dependencies.

Actual local export: `exports/taco/strict-clean-stage2-v1/` (ignored), 315 images / 538 annotations. Seed 42 gives train 221/397, validation 47/75, test 47/66 (images/annotations). All eight classes are present, but plastic bags have only two validation/two test images; cups and styrofoam also have small evaluation sets. Strict-clean selection sacrifices substantial mixed-scene coverage and can bias evaluation. COCO/source gates and independent reconstruction both passed; `training_ready=false` remains explicit.

Source images/annotations are unchanged and external. Full original metadata, image-level quarantine decisions, unique exported IDs and original IDs/category/record indexes are retained in JSON/CSV ledgers. All 152 dimension mismatches plus nine additional ambiguous orientation cases are held. Exact-byte/pixel duplicates were absent; two near-duplicate candidate pairs were investigated, with the eligible pair coassigned to train and the other quarantined. No ignore strategy, source repair, copy/download, model load or fine-tuning occurred.

See [files, commands, exact exclusion waterfall, class counts, checks and limitations](TACO_STAGE_2.md). Next action is owner review of bias/coverage and licensing/trainer compatibility before authorizing trainer work. PR #13 is the existing open review target; no merge is authorized.

Final checks: 114 tests, 113 passed/one existing privilege skip. Independent real-data reconstruction returned passed with no errors; all final artifacts retain matching checksums. Source hashes, the inference/audit/viewer code, taxonomy and dependencies are preserved. No photo copies are present in the export.

## Historical Stage 1 handoff


Current 2026-10-18: D-035 authorizes TACO harmonization/audit and annotation visualization, without fine-tuning, downloads or final splits. Branch `feat/taco-harmonization` starts at main `5351d57`. New independent scripts/config/tests leave all detector/model/threshold/viewer code and existing requirements untouched. The owner's changing detection config remains local and uncommitted.

Actual external source: 1,248 readable, 252 missing, zero unreadable images; 3,938 annotations on readable images. Conservative validation holds 520 otherwise-valid annotations for stored/declared dimension mismatch and nine invalid annotations on readable images: 3,409 all-class usable annotations, 1,653 mapped objects across 735 images. All 152 dimension mismatches match an EXIF-oriented preview; coordinate policy needs review. There are eleven invalid annotations in the full source (two duplicated IDs affecting four records, seven negative-origin boxes), 49 excluded source categories and explicit Unlabeled litter. 413 mapped images also have usable excluded annotations; no implicit backgrounds are generated.

Report is local/ignored at `reports/taco/review-v1/`; source photos and annotations remain external and unchanged. Native Windows checks exercised all-class sampling over ten actual photos and an EXIF preview over three, with selection, boxes, zoom/Home and closure. Agent inspected three in-memory annotation previews; human acceptance remains pending. Software suite: 100 passed/one existing privilege skip. See [file purposes, commands, counts, checks and limitations](TACO_PREPARATION.md).

Next: owner review of annotation previews and decisions on exclusions/unlabeled regions, coordinates and source anomalies before authorizing a training-data export/split stage. No training or grouping is authorized, and no merge is performed.

## Historical Phase 2E handoff


Current 2026-10-18: D-034 authorizes functional checking and handoff, stopping before Phase 3 decisions/optimization. Branch `feat/phase-2e-functional-checking` starts at latest fetched main `a3ec215` (PR #11 merged externally). Study baseline prompt was `plastic bag. drink pouch. cigarette butt` at 0.20/0.10. A subsequent local edit adds `a` before `cigarette butt`; it remains preserved and untested, alongside the untracked nested folder and unchanged environment. No defaults were selected.

Five cached CUDA experiments on the same three originals completed without model/processing failures: baseline, concrete prompt, original category prompt, box-only threshold and text-only threshold. All 15 raw-phrase/box overlays were inspected by the agent, with actual native replay verifying tables, zoom/pan callbacks, scrolling and closure. Findings include missed small litter, mixed/empty phrases, near duplicates, road-stripe false detection and pile/group confusion. No clear pile or clean no-trash scene is available. See [run IDs, timings, per-image comparisons, commands and limitations](PHASE_2E_RESULTS.md).

The owner reported oversized/missing labels on small/nested boxes. Corrected defaults are 7.5/7 points; labels can extend within the axes and move above enclosing borders, with query IDs/connectors. Final human acceptance is pending, not inferred from automated checks. A separately reproduced Windows atomic-replacement access-denial was fixed with bounded retries; permanent failures preserve old JSON. Final software suite: 91 passed/one privilege skip. No detection logic, mapping, dependencies or model revision changed, and no inference rerun was needed for either fix.

Next action: owner review of corrected viewer and Phase 2E evidence. Missing scenarios and category quality remain acceptance gaps. Phase 3 recommendations are proposals only; no optimization/grouping/training implementation or merge is authorized.

## Historical Phase 2D handoff


Current 2026-10-18: owner authorizes Display and Summaries (D-033), not Phase 2E or fine-tuning. Branch `feat/phase-2d-display-summaries` starts at latest fetched main `76c33f9` (merged PR #10); its tree already contained all previous viewer work. Local owner prompt `plastic bag. drink pouch`, thresholds 0.20/0.10 and untracked `BANTAI-Modeling-/` are preserved, excluded from this commit. No environment changes or inference rerun.

Existing: shared tab, raw phrase wrapping, original pixel axes, passing-only selectable tables, verified replay, persist-before-display and atomic separate status. Added: validated configurable label font defaults 8.5/8 points, display-setting persistence, explicit processing/error status and decimal-unit explanation in the GUI. Native replay passed on a detection photo and empty photo, including copy, scrolling and closure. Windows tests: 89 passed, one privilege skip. Final human small-label readability/interactive zoom-pan acceptance remains pending. See [commands, evidence and limitations](PHASE_2D_CHECKS.md). Next authorized action is review and outstanding visual checks; no merge or phase transition.

## Historical Phase 2C handoffs


Latest scrollable-tab correction (2026-10-18): continue the fetched `000b3b0` feature branch, preserving owner prompt/threshold edits and the untracked folder. One local Tk/TkAgg tab now repeats Image #N, Detection Summary, Text-Token Details and Configuration across the batch. Raw decoded phrases replace category labels; original pixel ticks, bounded readable labels above boxes and zoom/pan remain. Complete selectable GUI tables are passing-only and actual saved configuration is shown. Multi-image replay repeats `--result`/`--image`, verifies every source before pending persistence, and opens one shared window; inference persists the whole batch before display. Software checks: 88 passed/one privilege skip. The owner confirmed synthetic layout/scrolling/output; actual two-photo replay closed normally but exposed a missing q2 label, corrected with actual-glyph/whole-word wrapping and optional-ID omission. The owner confirmed the corrected photo labels were complete, then requested smaller labels. Final 8.5/8-point labels are software-verified; their separate visual confirmation remains pending. Actual Tk checks verified scrolling, selectable/copyable tables and clean closure, including a final smaller-label run. Four saved text replays preserved all images/JSON without model imports, inference or sockets. No open PR exists on the current branch (PR #9 was already merged externally); no merge is performed. See [current GUI commands and detailed evidence](SCROLLABLE_VIEWER.md). Earlier native side-panel/terminal layouts below are historical.

Latest native-viewer restoration (2026-10-18): same feature branch from fetched `b915bd3` (PR #9 already merged externally; no open branch PR found). Owner supersedes terminal-only/no-pop-up restrictions. Native in-memory TkAgg displays the full oriented image with valid passing boxes and category-only labels; a scrollable same-window panel contains matching IDs, raw phrases/categories, percentage scores and rounded pixel coordinates; passing-only terminal tables remain. Native windows retain zoom/pan without further navigation verification being requested. Replay identity and atomic status separation remain; no inference is rerun. Owner confirmed synthetic image/labels/navigation visibility. A resulting toolbar cleanup error was fixed; subsequent saved-photo replay `736be71a422949e3b04af39558c2f1af` closed and exited 0. The owner requested a label adjustment after the real-photo preview and clarified category-only on-image text. Final category-only/panel visual confirmation remains pending. Software checks: 77 passed/one privilege skip, including overlay alignment/empty results/closure/errors. Four saved-record summary/full replays preserved all JSON/photos without model imports or sockets. Owner prompt/threshold edits and untracked folder stay uncommitted and preserved; only backend migration is committed from configuration. See [current native usage and detailed evidence](NATIVE_VIEWER.md). Earlier terminal-only status below is historical. No merge.

Latest consistent-presentation correction (2026-10-18): fetched `6b0a14a`, same feature branch/PR #9. Each detection now uses identical wrapped Field/Value records with image/query IDs, quoted raw phrase, stored category/reason, decimal score, threshold status, rounded box and separate policy state. Unmapped retained/PASS predictions remain valid detections with no assigned category; incomplete phrases are never remapped. Preview descriptions expose source/sample dimensions and terminal resolution limits; Lanczos sampling never upscales originals. Existing Windows software checks: 81 passed/one privilege skip. Replayed all three saved records in summary/full with complete 900-query/13,500 ordinary-token coverage per full replay, no model imports, sockets or inference, and unchanged original hashes/JSON bytes. Windows PTY second-photo replay `052ed9155c7d4ab5bdf7bcf175dcf780` completed; owner VS Code visual inspection remains pending because its visual surface is inaccessible. See [current presentation and replay instructions](TERMINAL_VIEWER.md). No model/configuration/threshold/core inference changes or merge.

Latest width/grouping correction (2026-10-18): same branch/PR #9 from fetched `8b1be7e`. Preview width is configurable and height follows aspect ratio into scrollback. Summary/token tables stream without Enter pauses; tokens are grouped by query with an unwrapped status header. Summary remains passing-only, while explicit full mode restores every raw query/ordinary-token score, including below threshold. Saved-result replays on all three photos verified summary groups 2/3/2 and full coverage 900 queries x 15 ordinary tokens each, without model imports/inference or source/result mutation. Real Windows pseudo-terminal replay `d21364a916a4478baaf624499f324867` completed without input; owner VS Code visual inspection remains pending. Tests: 78 passed/one privilege skip. See [current usage and evidence](TERMINAL_VIEWER.md); older passing-only-for-full and height-limited/paging policies below are historical.

Latest passing-only/replay correction (2026-10-18): continuing the same PR #9 branch from fetched `b2c50c2`. Console tables now exclude all below-threshold/invalid/duplicate/special rows while preserving complete raw diagnostics. Added synthetic display-check and strict source-verified, atomically recorded replay with no model imports/execution and no mutation of the original result. A real Windows pseudo-terminal accepted native VT despite TERM=dumb, completed the synthetic check and first-photo replay (2 query/7 token rows). Visual verification in the owner's VS Code terminal remains pending because tools expose only text/control streams, not that terminal's visual surface. Latest software checks: 76 passed/one privilege skip. No inference rerun was needed. See [current commands, replay IDs and blocker](TERMINAL_VIEWER.md); earlier selection/full-console statements below are historical.

Latest presentation correction: continuing the existing PR #9 branch after fetching its remote head. Local user commit `4f2758e` had the identical tree to remote `97dbb8e`; it is retained by local tag `preserve-terminal-viewer-4f2758e`, and the branch was aligned without a merge or force push. The terminal viewer now checks actual stream capabilities, prints IMAGE first, keeps IDs at terminal-cell size, pages separate decimal-score tables, discloses summary selection/omissions and allows explicit full-mode special tokens. Noninteractive output is a display error, not shown. See [presentation contract and current evidence](TERMINAL_VIEWER.md). Actual cached CUDA run `a2ef6836b41b4e18a793642b1ad461a4` preserved all three photos and exactly matched prior detections/diagnostics; zero sockets. This tool's noninteractive output correctly returned 1 with three unsupported-terminal display errors while inference succeeded. VS Code visual verification remains pending.

Latest work: dedicated `feat/terminal-viewer-diagnostics` branch replaces TkAgg with in-memory ANSI/Unicode previews, results below images, and summary/full diagnostics. Normal predictions and existing result fields are preserved; complete aligned scores are added to JSON. Direct Windows checks: 57 tests passed/one symlink skip; offline CUDA full mode processed three photos with zero socket connections, 900 queries x 17 tokens per image, and 45,900 token lines. All displays finalized shown and originals were unchanged. VS Code visual readability remains uninspected. See [terminal contract and execution evidence](TERMINAL_VIEWER.md). Earlier native-viewer requirements and execution records below remain history.

Direct local follow-up (2026-10-17): fetched PR #8's branch at `65e085b` and used the unchanged Windows `.venv`. **53 software tests passed, one privilege-related symlink fixture skipped.** Cached-only inference on the one `data/input` photo completed on the GTX 1660 Ti with no fallback and two valid retained candidates (garbage pile 41.9%, individual litter 40.9% model scores); the original hash is unchanged. Native viewing returned normally; display is `shown`, run is `completed`, exit 0 with no errors. The owner reports correct detections and small real garbage piles; whether those small piles were missed or had undersized boxes remains unresolved. See [direct Windows evidence](PHASE_2C_WINDOWS_RUN.md) for run ID, timings, provenance and limitations. Earlier unrun statements below describe historical execution states; this single run does not establish category quality or Phase 2C acceptance.

Latest follow-up: PR #7 is merged. Owner-supplied Windows tests ran 50 cases with 49 passes/one privilege-related symlink skip. The supplied first model attempt downloaded safetensors and reached query extraction, but a validator bug rejected the model's intentional negative-infinity text padding on all 900 queries. The focused `fix/grounding-dino-padding-logits` branch corrects the audit using attention-mask/tail positions, while retaining true nonfinite/box/score checks. **54 cloud software tests pass**; corrected target inference/native overlay outcomes remain pending. Reuse cached model files and the existing environment; preserve the failed run. See the supplied-evidence/correction section in [PHASE_2C_CHECKS.md](PHASE_2C_CHECKS.md). Earlier implementation-time unrun statements below are historical and narrowed by these supplied results.

## Current result and execution limits

The owner instructed Codex to proceed with Phase 2C (D-025), superseding the previous stop-before-2C boundary. PR #6 is merged; this focused branch starts from current main `5db174b`. The baseline now implements pinned model/processor loading, float32 eval/inference mode, device handling/one auto-mode OOM CPU retry, exact mapping, conservative duplicate/review flags, readable summaries and native in-memory TkAgg viewing with the existing atomic display lifecycle. The CLI still defaults to preparation; inference requires `--mode infer`. Requirements and the existing verified target environment are preserved.

New modules: `scripts/inference.py`, `postprocessing.py`, `pipeline.py`, `viewer.py`; integration changes to configuration/CLI/result validation; `tests/test_inference.py` and a small foundation-fixture update; current-state docs and D-025. Model/processor/config use the verified revision; raw invalid queries are retained before threshold loss; no automatic grouping/membership/count inference or image exports.

Actual checks: **50 software tests passed** using existing cloud Python 3.13.5/Pillow 11.1.0/Matplotlib 3.10.1+dfsg1. CLI help, compilation and whitespace checks passed. A real cached-only CLI probe on a synthetic image returned exit 1 and persisted `runtime/missing_dependency` because torch is absent, with incomplete inference and unattempted display. This is not a no-detection/model result. Headless Agg drawing passed; native Tk calls are simulated. No package install/reinstall or weight download occurred. Full evidence, commands and assumptions: [PHASE_2C_CHECKS.md](PHASE_2C_CHECKS.md).

The implementation is ready for code review and target execution, **not verified model/target acceptance**. Actual model loading/inference, GTX 1660 Ti memory fit, native photo overlays and unannotated team-photo observations remain unrun. Run the new software tests and baseline in the existing Windows `.venv` using [README.md](../README.md); record actual JSON/native outcomes and limitations. Missing target/runtime/photos are execution prerequisites, not a request for repeated phase confirmation. Commit/publication outcomes are reported in the completion message; no merge or later optimization/grouping is authorized.

## Historical Phase 2B handoff — prior stop boundary superseded by D-025

## Current result and next boundary

The owner explicitly authorized Phase 2B, superseding the previous 2A-only stop boundary. Configuration, image preparation, result-contract utilities and atomic strict-JSON persistence are implemented for review on `feat/phase-2b-io-foundation` from current published main `dde98cf` (PR #5). No model loading, weight downloads, inference, viewer implementation or Phase 2C work occurred. No package was installed or reinstalled. The historical 2A handoff below remains preparation evidence, not the current authorization boundary.

Files: `config/detection.json`; `scripts/{configuration,inputs,results,persistence,detect_images}.py`; `tests/test_foundation.py`; three requirements files transcribed from the 2A snapshot; current-state docs and additive `/work/` ignore rule. CLI output is explicitly preparation-only, using separate schemas and `prepared` rather than functional inference success/no detections. Originals are read only and transformed solely in memory. Only JSON is persisted.

Actual checks: 22 synthetic software tests passed via `/usr/bin/python3 -m unittest discover -s tests -v`; CLI help, compilation and diff whitespace checks passed. Cloud Python 3.13.5/Pillow 11.1.0 were already installed; the verified Windows Python 3.11.9/Pillow 11.2.1 target was not directly accessible. New target software tests remain unrun. No fresh-install or GPU/model/viewer result is inferred. Full command/results, failure-injection coverage, persistence assumptions and limitations are in [PHASE_2B_CHECKS.md](PHASE_2B_CHECKS.md).

P-001 now distinguishes verified 2A metadata/basic runtime from untested model behavior; D-024 records the owner's bounded authorization, preserving D-001–D-023. Mapping/duplicate selection/overlap interpretation, model-memory fit, real-photo behavior and native overlays remain later work. The foundation is ready for review and eventual authorized 2C integration; run the new software checks in the unchanged target environment before claiming target validation. Commit/publication outcomes are reported in the completion message; no merge is authorized or claimed.

## Historical Phase 2A handoff — prior stop boundary superseded by D-024

## Result and ownership

**Phase 2A technical preparation checks are complete on owner-supplied Windows evidence. Stop here: detector implementation and inference have not begun and remain outside the authorization.** Codex owns implementation, replacing Antigravity; the owner reports Antigravity made no changes. Phase 0 is complete, Phase 1 planning is published, and this preparation corresponds exactly to plan Section 4 and Section 8 stages 2–3.

Cloud Codex reviewed actual target command outputs, a native-plot screenshot and explicit normal-return confirmation; it did not directly execute Windows commands. The local repository is in the owner's selected Documents/Python/Project Image-object Detection folder with a fresh `.venv`. The final evidence is preserved in [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md) and [PHASE_2A_ENVIRONMENT_SNAPSHOT.md](PHASE_2A_ENVIRONMENT_SNAPSHOT.md).

## Verified preparation

- Windows 10 build 19045.6466; Git 2.45.1.windows.1; Python 3.11.9 x64 installed from python.org via WinGet with installer-hash verification; local clone and venv confirmed.
- GTX 1660 Ti, 6144 MiB VRAM, driver 472.19, capability (7,5), CUDA build 11.8; CPU and CUDA tensors both returned 2.0. Earlier RTX 2080 Super assumption is superseded, not erased from history.
- torch 2.7.1+cu118, torchvision 0.22.1+cu118, Transformers 4.51.3, Pillow 11.2.1 and Matplotlib 3.10.3 installed/imported. `pip check` reported no broken requirements. All 36 frozen packages and cu118 installation source are recorded.
- Native TkAgg synthetic diagonal plot rendered; the owner explicitly confirmed `TkAgg viewer closed normally`. No image export was invoked.
- Official metadata/config/card/processor reads succeeded for `IDEA-Research/grounding-dino-tiny` at revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. Reported Apache-2.0 metadata, non-gated access, safetensors file, Tiny-consistent Swin configuration and same-revision processor settings are preserved.
- NVIDIA CUDA 11.8 Table 2 lists Windows minor-compatibility minimum >=452.39; driver 472.19 exceeds it and the actual CUDA tensor passed. The separate toolkit driver listing 522.06 is not the same minimum. Earlier cloud 403 limitations were resolved for these checks through successful target requests, without claiming cloud policy changed.
- The owner saved `.venv/phase2a-requirements.txt` and `.venv/phase2a-environment.json`; their supplied dependency/config values are preserved in the committed snapshot. `.gitignore` and historical decision rows remain intact.

## Remaining boundaries and next action

No current preparation blocker remains in the supplied evidence. No further installation or repetition of passed checks is needed for the unchanged environment. The owner can review the preparation branch and issue the next phase instruction; this technical completion does not accept a later phase on their behalf and does not authorize a merge.

Pretrained weights were not downloaded/loaded. No detector/application code, inference, team-photo processing, clustering, training/fine-tuning or annotated-image export occurred. Model-memory fit, custom category capability, detection quality and future model-specific driver behavior are intentionally untested. Later failures must be diagnosed from actual evidence; no accuracy or performance result is inferred from preparation.

Future authorized model and processor loading must use the same recorded revision. The baseline remains still-image trash detection, three intended categories, direct prompts first and multiple candidate regions; team photos are unannotated and manually supplied. Existing numeric/display contracts and no-automatic-image-export rules remain. Prompt/threshold/alias and grouping-boundary defaults are unvalidated proposals.

## Git publication and history

Latest fetched main for preparation is `7a776cf`, merging PR #4; PR #3 published the plan at `b366f76`. Work is on `docs/phase-2a-environment-preparation`, never direct main. Final commit/push/PR outcomes are reported in the completion message; no merge is claimed. Earlier PR API attempts returned `Forbidden`; do not claim a PR exists without confirmation.

D-001–D-023 preserve prior owner decisions, with D-022/D-023 assigning Codex and limiting this work to preparation. Previous pending/blocked readiness wording is historical and resolved by the supplied evidence above. No new implementation approval or owner acceptance date is invented.
