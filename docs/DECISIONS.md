# Decision register

## D-071 Stage 5 High-Resolution Adaptation and Model Soup - 2026-10-28

Owner requested accuracy maximization for 9-class waste detection incorporating paper (Option A).
1. Scale & Soup Analysis: Evaluating Stage 5 Run 1 at target 384/576 canvas scale lifted Combined AP from 0.135663 to 0.153029. Testing Model Soups revealed blend_2575 (25% Epoch 1 + 75% Epoch 2) as champion, achieving Combined AP 0.171262 and AP50 0.210596 while recovering styrofoam representation (0.148515) and maintaining paper AP (0.374468 overall, 0.473670 annex).
2. High-Resolution Fine-Tuning: Executed bounded 1-epoch (307 updates) training run stage5-paper-558bcc18d6cf4556a966d8cfc320c968 directly at 384/576 scale, LR 5e-6, warm-started from verified blend_2575 weights. Peak VRAM was ~3,306 MiB on GTX 1660 Ti.
3. Breakthrough Performance:
   - Combined 65 AP reached 0.202020 (AP50: 0.246182, AR100: 0.335380).
   - Legacy 47 AP reached 0.226047 (AP50: 0.263987, approaching Stage 4's 0.265).
   - Paper AP reached 0.421508 overall, and 0.586443 on the 18-image Annex.
   - Significant per-class gains: drink_can (0.6564), plastic_bottle (0.2525), cigarette (0.1115), plastic_cup (0.3198).
4. Model Soup: A 20/80 blend of blend_2575 and High-Res Epoch 1 achieved 0.194258 Combined AP and retained styrofoam piece AP at 0.148515.
5. Interactive Review & Post-Processing: taco_stage5_visual_review.py updated with 4-panel comparison (Ground Truth, Baseline, Stage 5 Run 1, Stage 5 High-Res) and interactive Overlap NMS filter (IoU 0.50 + IoMin 0.80).
6. Software integrity: 286 tests passed, 1 skipped, zero failures. No test split access or unauthorized checkpoint promotion.

## D-070 execution: two-epoch Stage 5 fine-tuning - 2026-10-28

The owner-requested command completed on feat/stage5-paper-category at its
256/384 default, LR 1e-5, two epochs from the bitwise-verified D-061 champion blend.
Exactly 614 training updates and 336 validation forwards completed; no unreported
pilot updates, test.json/test-photo access, commit/push/merge or promotion. Native
FP32/TF32-off loss/architecture and the six core Stage 4 SOURCES remain unchanged.

At the same 256/384 scale: combined nine-class AP 0.057617 ->0.113571 ->0.135663
(baseline/E1/E2); genuine original-eight-prompt legacy 47 AP 0.078141 ->0.131296
->0.144434. Paper AP on the 18-image annex is 0 ->0.116952 ->0.507431. These are
not direct comparisons with Stage 4 D-061 AP 0.265563 measured at 384/576.
The actual bound export is 307 train /595 boxes and 65 validation /109 boxes,
including 75 frozen-core and 34 annex boxes (20 paper). Earlier notes state 121
combined GTs; the report records the actual 109 without altering datasets or the
hashed rights record/assessment.

Launch blockers in the isolated Stage 5 trainer/evaluator were fixed before
execution: legacy path, COCO info, checkpoint-pointer replacement, full
model/processor/adapter/optimizer/RNG saving, removal of extra pilot updates,
native gradient auditing/norm 1 clipping and RNG-isolated validation. A separate
47-image eight-prompt pass now supplies the true legacy result; filtered
nine-prompt metrics remain separately labeled. Guarded suite: 286 run, 270 passed,
16 documented skips, zero failures/errors. Eight complete new checkpoint bundles,
614-step sampler trace, frozen hashes, two source bundles, all runtime/core source
hashes, dataset/rights hashes and 372 authorized photo hashes pass postchecks.
Existing 64 bundles remain unchanged; nothing is promoted or wired into inference.
[Results, actual counts, scope distinctions and checkpoint evidence](TACO_STAGE_5_TRAINING.md).

## D-070 Stage 5 paper-category authorization and Option A split adoption - 2026-10-26

Owner authorizes Stage 5 implementation incorporating paper (id: 8) and selects
Option A (Frozen Core + Annex). All 1,500 TACO images are verified present on local
disk (0 missing). The 14 TACO source categories are approved for paper: Normal paper,
Magazine paper, Wrapping paper, Tissues, Paper bag, Paper cup, Corrugated carton,
Other carton, Drink carton, Meal carton, Egg carton, Pizza box, Paper straw, and Toilet tube.

Option A preserves the original 221 training, 47 validation, and 47 test image
memberships byte-for-byte, appending the 122 newly eligible images (165 paper boxes,
106 original-class boxes) as follows:
- Training: 221 frozen original + 86 appended = 307 total images (120 paper boxes)
- Validation Annex: 47 frozen original + 18 appended = 65 total images (20 paper boxes)
- Held-out Test Annex: 47 frozen original + 18 appended = 65 total images (25 paper boxes)

Evaluation protocol preserves separate reporting for:
1. Legacy benchmark view on original 47 validation images / 75 GTs.
2. Paper annex view on 18 new validation images / 20 paper boxes.
3. Combined 9-class view on 65 validation images / 121 GTs.

Scope authorized: create config/taco_taxonomy.v2.json, document Section 185
academic-use basis in docs/TACO_STAGE_5_ACADEMIC_USE.md, build isolated export
exports/taco/strict-clean-stage5-v1, implement isolated 9-class training/evaluation
modules on feat/stage5-paper-category, and run 9-class fine-tuning warm-started
from champion D-061 weights. Existing 64 checkpoint bundles, frozen Stage 4 SOURCES,
and test-split privacy constraints remain strictly invariant.

## D-069 Stage 5 paper-category metadata audit - 2026-10-26

Owner authorized a read-only metadata audit and proposal after the D-068 safe point.
Stage 4 commit `9c3b6777724a799569c7e485815171ef4f3a878a` is already pushed to
PR #18's branch; `feat/stage5-paper-category` was created/pushed from that exact
commit, and all later audit work is on it. Phase 0 re-verification: 278 tests,
262 passed, 16 documented skips, zero failures/errors; public links and diff checks
passed. No redundant Stage 4 commit, merge, checkpoint promotion or new model run.

The proposed paper class combines the fourteen requested source names under a
new ID 8, preserving IDs 0..7. Raw support is 497 boxes /347 images; unchanged
strict-clean rules with this proposed mapping retain 165 boxes /122 images.
Zero overlap with the existing 315 eligible images; proposed total 437 images /
809 annotations, including 106 original-class boxes on the new images. Across
raw paper images, 72 contain Unlabeled litter and 160 another still-excluded class;
43 overlap (union 189). Whole-image quarantine remains unchanged. Cached image
metadata/fingerprints reproduce existing eligibility and annotation ledgers exactly;
these are conditional counts, not fresh photo validation.

The concrete split proposal locks all original 221/47/47 memberships and appends
86/18/18 new images, with 120/20/25 paper boxes (combined views 307/65/65).
The original 47/47 holdout files remain unchanged; annex/combined nine-class
metrics would be separate from the legacy 47-image eight-class benchmark. If total
holdout sizes must stay exactly 47/47, an alternative puts all 122 new images into
training (343/47/47), but leaves zero paper evaluation support. Rare-subtype
limitations, waterfall, counts, exact metadata anchors and proposed file-impact
list are in the [paper audit](TACO_STAGE_5_PAPER_AUDIT.md).

No photo bytes/decodes, test.json access, torch/Transformers import, GPU, training,
inference or export occurred. Nineteen source/metadata/core-source hashes remained
unchanged; all 64 checkpoints, old datasets/taxonomy and rights records are untouched.
No taxonomy v2, prompt/adapter/trainer/evaluator or new rights record was created.
**Stop for owner approval** of mapping and split plan; fresh qualification/export,
nine-class software, rights scope and model execution are not authorized by this
completed audit. Audit/report changes remain local and uncommitted for review.

## D-068 final Stage 4 synchronization and publication - 2026-10-25

Owner authorizes synchronizing D-060 through D-067 and committing/pushing only the
focused docs, scripts and tests to `origin/feat/stage4-real-image-training` for
PR #18. No merge, promotion, GPU, training or inference. The 64 checkpoint bundles,
private runs/work/exports/data, rights records, official taxonomy and six core
training SOURCES are excluded from edits/staging; unrelated research/proposal docs
remain unstaged. The earlier stopped backbone implementation is retained as
experimental software/history, not as a completed or qualified training result.

**Best measured official eight-class validation state: D-061
`blend_350_442_2575`, AP 0.265563, no NMS or remapping.** Its separate raw
eight-class candidate coverage is 38/75 at score >=0.05, IoU >=0.50, all 900 queries,
with non-exclusive matching; this count is not COCO recall. It is not promoted and
not wired into `scripts/detect_images.py`. D-062 cosine decay, D-064 pure-film
damping, D-066 cigarette LR boosting and D-067 auxiliary score calibration are
retained as negative results for their combined objectives, including partial
gains and regressions. D-060 replay, D-063 NMS and D-065 material/polymer views are
diagnostic capabilities, not changes to the official benchmark. [Synchronized
outcomes and limits](../README.md#d-060-through-d-067-outcomes).

Recommended saved-review **display-only** settings: threshold **0.10 or higher**,
plus **`agnostic IoU 0.50 + containment 0.80`**. Select explicitly; historical
raw/.05 defaults remain. Suppression can hide distinct or low-score true objects;
post-NMS or mapped-category numbers must not replace official eight-class AP and
raw candidate coverage. No production operating threshold is established.

D-068 verification: 278 tests, 262 passed, 16 documented skips, zero failures or
errors, using `scripts/verify_stage4_guarded.py`. Fifteen training-execution fixtures
are excluded; one Windows symlink fixture lacks privileges. Detector loading/
forward, CUDA initialization, backward and optimizer steps are blocked. Generated
fixtures/logs go to the system temporary directory; protected-directory writes are
guarded and none occurred. D-067's recorded bundle/native-GUI evidence is retained
without rerunning model/photo checks or re-opening checkpoints. Relative public
Markdown links and `git diff --check` are checked before commit; private artifact
links remain intentionally local. Human visual acceptance and the PR merge
decision remain with the owner; no follow-up experiment is implied.

## D-067 positive-confidence / absent-class auxiliary objective - 2026-10-25

Owner authorized exactly92 updates at steps351..442 from verified D-057 step350 with exact model/AdamW/RNG restore, unchanged native adapter/loss and frozen backbones/text. The isolated objective selects each GT query by detached argmax(IoU+mean ordinary-subword probability), sums weighted positive BCE (cigarette1.5, other.75, film0), and sums absent-class p>.03 selected-query mean penalties weighted10. LR8e-6 on67 batches,1e-6 on25 pure-film batches. Native checked_loss passes before augmentation, finite gradients/encoder updates and clip norm1 are audited. FP32/TF32-off384/576 and original sampler suffix remain fixed. Two complete new bundles and94 validation forwards completed; all62 prior plus2 new bundles verified (64 total), six core files unchanged. No reference GPU reruns, test access, commit, push, merge or promotion.

Endpoint official AP0.245944; prespecified25/75 original350/new442 blend AP0.251234, versus previous integrated0.265563. Cigarette coverage is15/24 and15/24; Image777 4/7 and4/7. At score.10, strict eight-class F1 is0.116129/0.121622; Image833 cigarette FP counts are0/0. Full official/material/polymer metrics, strict matching at.05/.08/.10/.12/.15/.20, Image777 GT278..284 scores/ranks/IoUs and Image833 FP audit are in the [D-067 report](TACO_STAGE_4_576_SCORE_CALIBRATION.md). This is one diagnostic validation experiment; higher confidence is not calibrated correctness or a promoted checkpoint.

The new saved choice is `42 / 1e-5 @ 384/576 (score-calibrated vs integrated)`, with prior D061 integration, cal_442 and cal_blend_2575. Both GUI modes retain taxonomy/NMS controls and support explicit top15; all47 validation IDs remain accessible. All12 prior choices are preserved. Guarded suite: 278 tests, 262 passed, 16 documented skips, zero failures/errors. Two native Tk cases pass all15 taxonomy/NMS combinations, navigation/GT selection/zoom/pan at777/top15 with no model/GPU/state/export execution. Human visual acceptance remains pending. Next authorized action is saved visual review; no further experiment or promotion follows automatically.

The combined confidence/suppression objective is not achieved: Image833 cigarette FP at.05 increases from12 in the prior integration to21/20, and maximum cigarette scores rise to0.099080/0.090432 from0.078764. Only GT280 on Image777 has a qualifying cigarette query above.10 in either new state; none exceed.15. No promotion is supported by this result.

## D-066 late-Epoch-2 cigarette boost - 2026-10-25

Owner authorized exactly92 updates at steps351..442 from verified D-057 step350 (epoch1/next_batch129), restoring exact model/AdamW/RNG. Original sampler suffix contains11 cigarette images (LR1.5e-5),25 pure-film images (LR1e-7),56 other images (LR5e-6), with no resampling or loss/architecture/core-source changes. Two47-image validations completed94 forwards: endpoint and prespecified50/50 new-endpoint/original-constant442 blend; no reference GPU reruns or test access. Complete new step400 and step442 bundles were saved; all60 prior plus2 new verified (62 total). No commit, push, merge or promotion.

Negative result for cigarette improvement: endpoint AP0.250675 and blend0.257129, both below prior integrated0.265563. Both new states cover14/24 cigarettes and4/7 on Image777, versus prior integration15/24 and5/7. Raw standard first15 coverage is2/7 for both, versus prior3/7; topK15 does not guarantee all cigarettes are drawn. Four-material diagnostic AP is0.307638/0.306513, versus prior0.309859; these mapped tasks are separate from official eight-class detection. Peak allocated VRAM3.550GiB. Full official/material/polymer metrics, all24-cigarette and GT278..284 score/rank/IoU ledgers, locked bins and launch commands are in the [D-066 report](TACO_STAGE_4_576_CIGARETTE_BOOST.md).

The new local choice is `42 / 1e-5 @ 384/576 (cigarette-boost vs integrated)`; panels are prior integration, new endpoint and new50/50 blend. Both GUI modes retain D-065 taxonomy/NMS controls and accept `--box-top-k 15`. Existing11 manifest choices are unchanged. Guarded suite: 273 tests, 257 passed, 16 documented skips, zero failures/errors. Two native Tk cases passed all15 taxonomy/NMS combinations at Image777, top15, all47 IDs and navigation/zoom/pan; no replay model/GPU/weight-state/export operations. Human visual acceptance remains pending.

Integrity exception retained transparently: the controller initially stopped after completing all updates/forwards because an older saved-GUI analysis.json changed concurrently from pending to closed and added photo-read count16. Reconstructing its pending form matches the exact pre-run hash, proving all other fields identical. All model/data/checkpoint files and six core bytes are unchanged. The stopped record, before/after hashes and reconstruction proof are retained; protected_after.unchanged remains false, with the verified display-only exception recorded separately. No historical file was rewritten and no training/inference was repeated. Next authorized action is owner saved visual review; no further experiment or promotion follows automatically.

## D-065 diagnostic material/hazard hierarchy - 2026-10-25

The owner's latest attached authorization replaces the earlier top-backbone-unfreezing request with saved-prediction-only Architecture A: specific object classes -> deterministic 4-material or 3-polymer-grouped category mapping -> optional diagnostic NMS -> matching/replay. No model loading/execution, CUDA initialization, backward or optimizer updates in this hierarchy task. Official eight-class taxonomy, six core SOURCES and all 60 checkpoint bundles are byte-for-byte unchanged; all seven saved official metric dictionaries reproduce exactly on CPU. No test.json/test-photo access, commit, push, merge or promotion.

Both Tk viewers now have reversible Taxonomy view and five NMS choices, retaining official eight-class/raw defaults. Mapped labels retain specific subtype and hazard flag; class-aware suppression and GT agreement use mapped IDs. Greedy containment suppresses at IoU >= .50 OR IoMin >= .80. Audit: seven states x three taxonomies x three suppression settings = 63 diagnostic COCO cases, plus strict one-to-one matching at .05/.07/.10/.12, non-exclusive coverage, Image 777/GT259/cigarettes, the fixed ten D-058 regressions and raw 4784 versus Stage-2 538 category support. Paper has zero Stage-2 support; Food waste/Organic has eight raw, zero Stage-2 annotations. Neither receives invented validated metrics.

Integrated blend: raw eight-class AP 0.265563 and coverage 38/75; raw four-material AP 0.309859 and 55/75. Four-material class-aware containment yields AP 0.313529 and 49/75, with strict TP/FP/FN 49/597/26 at .05. It suppresses nested q4 under q0 and gives GT259 material agreement, while reducing cigarette coverage from 15/24 to 14/24 (Image 777 remains 5/7). Material-task AP is not an improvement to official eight-class detection. Four-material grouping covers nine fixed plastic regressions; polymer grouping additionally accepts the Styrofoam confusion. [Complete report, artifacts, caveats and PowerShell commands](TACO_STAGE_4_MATERIAL_HIERARCHY.md).

Verification: 267 guarded tests, 251 passed, 16 documented skips, zero failures/errors. Four native Windows Tk cases pass all 15 taxonomy/NMS combinations in both modes at Image 777, all 47 validation IDs and navigation/zoom/pan; two focused final cases verify per-annotation hazard columns. All 60 bundles verified before/after; prior protected files and replay manifest unchanged. Human visual acceptance remains pending. Next authorized action is saved visual review; no training or promotion follows automatically.

Historical interruption is retained separately: the earlier vision-unfreeze attempt stopped after ten completed optimizer updates and eleven training forwards, zero validation forwards and zero new checkpoint bundles, on a zero stage-3 probe gradient. Its stopped private record is `runs/training/taco-vision-unfreeze-9452eb4dc47044a9a76cb232fe921b45/record.json`; it was not resumed and contributes no hierarchy results.

## D-064 pure-plastic-film-damped Epoch-2 continuation - 2026-10-25

Owner authorized exactly 221 updates from the verified D-057 E1 step 221 model/AdamW/RNG state. LR 1e-6 applies only to 61 pure-film batches; LR 1e-5 to 160 other batches, including all 30 cigarette images, with original ordering/loss/architecture. Three 47-image validations (CB350, CB442, in-memory 25/75 blend) completed 141 forwards; historical references were not rerun. All 55 prior and 5 new bundles verified (60 total); six core files and protected prior artifacts preserved. No test access, commit, push, merge or promotion.

CB Step 350 AP **0.258631**, Step 442 **0.242594**, and CB 25/75 blend **0.257303**. Cigarette coverage is 13/24, 14/24, 13/24; Image 777 coverage 4/7 for all three. The fixed ten-regression retention is 4/10, 2/10, 2/10; the nine-large subset is 3/9, 2/9, 2/9. This is partial recovery, not preservation of all desired gains. The requested fixed ten-loss set contains nine >=16-pixel objects and one 8--<16 object; both denominators are reported. [Full comparison, ledger and GUI commands](TACO_STAGE_4_576_CLASS_BALANCED.md). Next authorized action: owner inspects saved results; no further experiment or promotion follows automatically.

D-064 verification: 255 guarded tests, 239 passed, 16 documented skips, zero failures/errors. Four native Tk cases passed, each checking all three NMS options; all 47 image IDs selectable, correct labels, navigation and zoom/pan. No saved JSON changes or model/GPU execution during replay. Human visual acceptance remains pending.

## D-063 diagnostic NMS in saved replay - 2026-10-25

Owner authorized an explicit saved-GUI-only NMS selector and saved-prediction audit. Default none preserves historical behavior; optional class-aware/class-agnostic modes greedily suppress passing queries at inclusive IoU>=0.50. Filtering is image-wide, after source/score selection and before standard top K, or before per-GT matching in object mode. No model/GPU execution, core SOURCES edits, official protocol/metric updates, checkpoint changes, commit, push or merge.

All five saved 384/576 states were audited at 0.05/0.07 for remaining boxes, first 10/15 GT/cigarette coverage, and separately labeled post-NMS diagnostic COCO metrics. Agnostic NMS can trade correct-class coverage for fewer boxes. For the blend at 0.05/K 10, raw/class-aware/agnostic coverage is 30/31/27 of 75 GTs; cigarettes 10/11/11 of 24. Original official metric dictionaries reproduce exactly on CPU and remain unchanged. [Full audit, scope distinctions and exact GUI commands](TACO_STAGE_4_NMS_AUDIT.md).

All 55 checkpoint bundles and 1703 protected files verified unchanged; the six core bytes and no-NMS official protocol are identical. Guarded suite: 235 passed, 16 documented skips, zero failures/errors. Native runtime evidence is recorded separately; human visual confirmation remains pending.

Native verification: 24 Tk cases passed at Image 777, covering both GUI modes, all NMS choices, both score floors and top K 10/15. Defaults, selector callbacks, tables/boxes, raw restoration, navigation and zoom/pan passed. No model/GPU/weight/export operations or saved JSON mutations occurred. Human visual confirmation remains pending.

## D-062 controlled Epoch-2 cosine continuation - 2026-10-25

Owner authorized continuation from D-057 E1 step 221 using the exact restored model/AdamW/RNG state, 221 updates only and two validation evaluations (94 forwards). LR decays per step from 5e-6 to 1e-6. The six core SOURCES files stay byte-for-byte unchanged; the isolated continuation module reuses their verified functions. Five complete new bundles were saved; all 50 original plus5 new bundles verified, and protected prior artifacts preserved. No test access, reference GPU rerun, commit, push, merge or promotion.

Cosine step 350 AP **0.217172**; step 442 AP **0.221592**, above constant-LR E2 0.170508 but below E1 0.249891 and D-061 blend 0.265563. Endpoint cigarette coverage is 10/24 overall and 3/7 on Image 777; large-bin coverage 14/36. Passing queries at 0.05: 1133 across 47 images and 28 on Image 777, versus constant E2 1118/29. This is a negative/mixed diagnostic result for the combined preservation-and-clutter hypothesis, with unchanged validation selection bias. [Complete evidence and exact GUI commands](TACO_STAGE_4_576_LR_DECAY.md). Next authorized action: owner reviews saved results; no additional training or checkpoint promotion is implied.

## D-061 384/576 weight integration - 2026-10-25

Owner authorized exactly four inference-only in-memory FP32 parameter averages of D-057 step221/350/442, evaluated on the unchanged47 validation images, maximum188 forwards. Existing D-056/D-057/D-058 references are reused. Zero backward, optimizer updates, training-photo reads and test access; no checkpoint promotion, commit, push or merge.

Highest integrated eight-class AP: `blend_350_442_2575` **0.265563**, versus step350 0.262337 and E2 0.170508. All900/.05 same-class coverage: 38/75 objects, 15/24 cigarettes, and 5/7 Image777 cigarettes. This selects a diagnostic replay view only and retains validation selection bias. All50 saved checkpoint bundles verified before/after; frozen persisted tensors are bitwise identical; only558 canonical trainable tensors enter averaging. The first47-image result was retained after a cleanup guard stop; remaining states ran in isolated processes, with no repeated inference and CUDA context release between states. [Complete evidence and GUI commands](TACO_STAGE_4_576_WEIGHT_INTEGRATION.md).

## D-060 Saved 384/576 visual GUI replay ? 2026-10-24

Owner authorized saved-JSON replay for D-056/D-057/D-058 in standard and per-object modes, retaining all four 256/384 choices. No GPU, training or detector inference is authorized by this task. The local augmented manifest adds endpoint and step-350 choices; checkpoint JSON context/position and anchored evaluations bind their identity. No checkpoint promotion or merge. [Implementation, verification and exact commands](TACO_SAVED_REVIEW_384_576.md).

## D-059 Stage 4 wrap-up and focused publication — 2026-10-24

The owner authorizes closing the Stage 4 post-validation investigation series
D-043–D-058, synchronizing README/phase status through D-057/D-058, running
guarded fast unit/CPU checks, and publishing the explicitly listed scripts,
tests, reports and two SVG assets in one focused D-045–D-059 commit on
`feat/stage4-real-image-training` to update PR #18. No PR merge or checkpoint
promotion is authorized. [Final wrap-up state and inventory](HANDOFF.md).

This authorization supersedes earlier per-investigation publication stops only
within that allowlist. Preserve all runs, work, exports, data, environments,
rights records and50 corrected checkpoint bundles. No GPU/model execution,
training/inference, test-split/photo access or checkpoint modifications.
Local ARGON handoff and concept-proposal planning documents remain uncommitted.
Private helper-dependent tests explicitly skip when their unpublished work files
are absent; no private audit artifacts are added to Git.

The D-042 Section185 owner academic-use record remains distinct from external
permission, institutional approval and definitive legal clearance. Optional human
visual review, PR review/merge and future stages remain owner decisions, not
automatic follow-up actions. Higher sampled validation AP does not select a model.

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
Epoch1 AP0.249891/AR1000.488141 narrowly exceeds D-0550.247003/0.474763;
Epoch2 regresses to0.170508/0.370118 despite cigarette same-class coverage
improving5→8→15/24 at0.05 and seven-class AP excluding bag rising.
Ten new bundles verified; final optimizer step442 across558 parameters.
All1766 protected hashes matched after execution; no checkpoint is promoted.
[Full gates, results and limits](TACO_STAGE_4_576_TRAINING.md).
Audit `scale576-experiment-0b2a846e659a429cbe8a65c1ca0444e1`. No retry, promotion, test access, dependency/GUI changes,
annotated exports or Git publication. Stop for owner review.

## Multi-checkpoint resolution replication — 2026-10-23

D-056: The owner authorizes inference-only256/384 vs384/576 for unchanged
Pretrained and seed123/LR1e-5/E2 on47 validation images/75 GT, at most188 forwards.
Reuse D-055 seed42/E1 outputs read-only, without GPU repetition. Require guarded
software, fresh protected/source hashes, both exact low baselines before any
high pass, then both high evaluations. Fixed weights/FP32/900 queries/prompt/
taxonomy/COCO v2 and locked14/10/15/36 bins. No updates/backward, train-photo/test
access, core/GUI/dependency changes, exports, promotion or Git publication.
[Completed results and qualification](TACO_STAGE_4_RESOLUTION_REPLICATION.md)
show shared small-bin coverage gains, checkpoint-dependent raw AP and positive
bag-excluded AP deltas in all three states. No further experiment is authorized;
stop for owner review.

## D-055 single retry authorized and completed — 2026-10-23

The owner explicitly authorizes one retry from Gate1, preserving the original
stopped attempt `resolution-experiment-629229adf1c6480d930aac79250d18f8`.
The isolated pointer records that retry consumed and zero retries remaining.
All five gates and the47-image384/576 pass completed on the fixed42/1e-5/E1
checkpoint. [Measured results and limits](TACO_STAGE_4_RESOLUTION_EXPERIMENT.md)
record improved aggregate AP, partial small-object recovery, persistent <4px
failures and three larger-object regressions. No updates/backward, training-photo
reads, test access, core/GUI edits, promotions or Git publication. All next steps
remain proposals; stop for owner review. The initial failure record below is
historical and its no-retry statement is superseded by this explicit grant.

## Controlled inference-only resolution comparison — 2026-10-23

D-055: The owner authorizes one fixed seed42/LR1e-5/E1 validation comparison:
256/384 with384-square pad versus384/576 with576-square pad. Require software,
artifact/source, CPU geometry, exact CUDA baseline and one-image CUDA pilot gates,
in that order. Stop immediately at any failed gate; no fallback or parameter
changes. Use an isolated work harness and focused resolution tests; core training,
evaluation and GUI files remain untouched. No updates/backward, train-photo/test
access, promotion, exports, dependencies or Git publication. All proposals remain
unimplemented. [Actual stopped attempt and cause](TACO_STAGE_4_RESOLUTION_EXPERIMENT.md):
the software gate failed due to harness guard scoping; no real inference ran.
Prepared fixes are unqualified and no retry is authorized by this stopped record.


## Small-object and classification audit — 2026-10-22

D-054: The owner authorizes a read-only full-validation audit of actual256/384
preprocessing and384-square padding, all75 resized GT boxes and nine saved model
states. Separate size-associated localization, class assignment, score/rank
effects and annotation uncertainty; preserve the original20-image manifest and
existing GUI. Record current owner-supplied image777 observations separately.
Rank resolution/classification/data-quality directions as proposals only;
[report evidence and one next experiment](TACO_STAGE_4_SMALL_OBJECT_CLASSIFICATION_AUDIT.md).
No new training or model inference without separate authorization, optimizer
updates, model/data/protocol/GUI/dependency changes, test access, promotion,
exports, uploads or Git publication. Stop for owner review after verification.


## Per-object saved detection investigation — 2026-10-22

D-053: The owner authorizes diagnostic-only per-GT analysis for image777 and the
existing20-image review using verified saved predictions. Preserve the four-panel
GUI, terminal-only defaults and explicit saved-run selection. Add an opt-in
annotation selector, GT highlight/zoom and complete overlap tables; compare all
four seed/LR runs with unchanged pretrained baseline and same-run E1/E2.
Preserve the original manifest and add separate machine evidence. No training,
inference, optimizer/model/data/protocol/dependency changes, test use, promotion,
exports or automatic Git operations. Verify software and native Windows GUI;
[report findings](TACO_STAGE_4_PER_OBJECT_REVIEW.md), separate automated diagnostics
from human material/annotation confirmation, then stop for human review. No
training recommendation is authorized solely from image777.


## Cross-seed stability analysis — 2026-10-22

D-052: The owner authorizes saved-result analysis and visual-verification preparation
across pretrained and both epochs of four seed/LR runs (42/123, 1e-5/5e-6).
Preserve corrected COCO v2, original artifacts, owner changes and existing viewer
defaults. Prepare 12–20 validation examples and a local review manifest; permit
only the minimal explicit saved-run viewer extension. Separate automated
diagnostics from pending human confirmation. Compare the four possible research
directions as proposals only. No new training, inference, test evaluation, model
or dataset changes, dependency upgrades, promotion, exports, publication or Git
publication. Stop after [the report and review preparation](TACO_STAGE_4_CROSS_SEED_ANALYSIS.md)
for owner review; no additional experiment is authorized.

## Paired seed-123 replication — 2026-10-22

D-051: The owner authorizes exactly two fresh two-epoch runs at seed 123:
A with LR 1e-5 and B with LR 5e-6. Preserve all other corrected experiment
settings, original pretrained initialization, 221/47 corpus, eight-class prompt,
Stage 3D adapter, valid-image targets, native loss, frozen backbones, FP32,
batch 1, 256/384 resize, 384-square padding, weight decay 1e-4 and clip 1.
Add minimal configurable `--seed` support with default 42, propagating it to
Python/NumPy/CPU/CUDA RNGs, restart and epoch shuffling; record/test its exact
behavior without changing strict resume verification or old manifests.
Each run independently discards its three-step pilot, restarts from the pinned
pretrained model and performs 442 full updates (884 full + six pilot total).
Stop on critical preflight/integrity failure, and after both final validations.
No test access, model/data/inference changes, extra experiments, dependencies,
uploads, promotion or automatic Git operations. Compare both seeds/rates and
recommend further work only as proposals in [the replication report](TACO_STAGE_4_SEED_REPLICATION.md).

## Controlled learning-rate experiment — 2026-10-21

D-050: The owner authorizes exactly one fresh two-epoch experiment at learning
rate 5e-6, changing only learning rate from the corrected 1e-5 reference run.
Retain original pretrained weights, corrected targets, Stage 3D adapter, native
loss, 221/47 corpus/splits, eight-class prompt/taxonomy, seed 42, batch 1, frozen
backbones, FP32, 256/384 resize, 384-square padding, AdamW weight decay 1e-4,
clip 1, no augmentation/scheduler and corrected COCO v2 evaluation. Use three
discarded pilot steps, then exactly 442 full-run updates, stopping for safety
failures. Add minimal configurable learning-rate support if necessary, retain
default 1e-5, bind the chosen rate in settings/checkpoint context and test its
actual delivery to AdamW. Preflight must pass before updates. No historical
resume, test access, data/prompt/model/inference changes, dependency changes,
promotion, uploads, automatic tuning or Git operations. Compare all five
pretrained/reference/new states in [the LR experiment report](TACO_STAGE_4_LR_EXPERIMENT.md).
Stop after epoch 2/final validation for owner review.

## Corrected-run diagnostic investigation — 2026-10-20/21

D-049: The owner authorizes diagnostic-only investigation of corrected epoch 1
versus epoch 2 using verified saved results and the existing local visual viewer.
Verify comparability, score distributions/ranking, class assignment, localization,
losses/gradients and representative validation cases. Preserve source files,
configuration, data, checkpoints, strict resume and corrected COCO settings.
No training, optimizer updates, test access, promotion, dependency changes,
uploads or automatic Git operations. Report confirmed observations separately
from hypotheses and rank single-variable experiments as proposals only. See
[corrected diagnostics and next experiment proposal](TACO_STAGE_4_CORRECTED_DIAGNOSTICS.md).
Stop for owner review; no further experiment is approved by this decision.

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
