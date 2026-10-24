# Stage 4 Step 2 — Local diagnostic replay and error analysis

**Superseding owner correction (D-046): terminal output only for epoch scores.**
The epoch diagnostic window was closed normally. Run
`.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py` to print
verified saved epoch metrics, corrected-target losses and model-score summaries.
Default execution no longer launches a GUI or creates analysis files. The GUI
workflow below is now available again through explicit `--gui` opt-in.
`--analyze-only` remains an explicit JSON-analysis option without a window.

**2026-10-20 explicit GUI restoration:** Run from the repository in PowerShell:

```powershell
.\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --gui --image-id 42
```

`--image-id` requires `--gui` and accepts validation IDs only. The existing image
selector includes 42, 113, 200, 31, 1096, 252 and 1014; direct selection starts in
the all-validation scope. Omitting the ID retains the representative subset.
`--gui` and `--analyze-only` are mutually exclusive. GUI mode creates a fresh
analysis JSON with separate pending/closed/error display status; it never exports
annotated images. Existing predictions, configuration and metrics remain untouched.

Restoration verification: 18 software tests passed and the actual terminal command
exited 0. Native Windows checks passed all 16 representative images, shared
zoom/pan, table copying/highlighting, diagnostic filters and full-validation
navigation (`step2-review-be0b248ac2554160a06eac179cb6cc9b/native-runtime.json`
under `runs/training`). The new explicit CLI route opened image 42 with four
panels and closed normally (`step2-review-2901e7965fe9474584bf20a760f6e005/analysis.json`).
These are automated runtime checks, not human photo-level acceptance.

2026-10-20, owner authorization D-045. Existing branch
`feat/stage4-real-image-training`, HEAD `087e361`; all prior uncommitted work and
owner detection edits are preserved. Software, annotation-relative analysis and
native replay checks are complete. **Human photo-level confirmation is pending.**
No photographs/screenshots are transmitted for agent interpretation. Automated
matching and native rendering are not substitutes for visual judgments.

## Workflow and files

- `scripts/taco_visual_review.py`: independently anchored replay of the corrected
  Step 1B pretrained and Step 1C epoch results; analysis and fresh UUID records.
- `scripts/taco_error_analysis.py`: saved query/coordinate/ranking validation,
  diagnostic annotation matching, candidate coverage and deterministic selection.
- `scripts/taco_comparison_gui.py`: standalone four-panel TkAgg viewer with shared
  zoom/pan, original pixel coordinates/aspect ratio and source-verified RGB images.
- `config/taco_visual_review.v1.json`: fixed saved-record/result hashes. The
  approved evaluation compatibility policy and strict resume verifier are unchanged.
- `tests/test_taco_visual_review.py`: fourteen independent software tests.
- This report, README, phase status, handoff and decisions: current evidence/limits.

The viewer shows dashed ground truth, pretrained, epoch 1 and epoch 2 on the same
image. Prediction labels contain evaluation category, query ID and decimal score;
**raw decoded phrases are separate table columns**, including empty/incomplete
phrases. Small configurable five-point labels reuse the existing border-aware
placement; complete copyable tables remain available when labels cannot fit.
Selecting a table row highlights that query, including candidates outside box top K.
Original source identity/size/orientation is checked whenever an image is opened.

Default diagnostic threshold is **0.05**, with five boxes per model. Tables contain
all candidates passing the view filter, not only those five boxes. Switch to zero
for weak candidates, 0.25 for the previous diagnostic operating point, or all 900
queries to inspect candidates outside the official top 100. These are **view-only
settings**, not changes to COCO evaluation, inference configuration or validated
defaults. Save/export is omitted from the toolbar; no automatic image copies or
temporary annotated images exist. Notes require an explicit local button click
and remain separate from automated evidence. Startup/callback errors preserve saved
results; callback failures close a stale view and are recorded as display errors.

Run in the existing environment, with no installation or inference:

```powershell
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --gui --evaluation-record runs/training/step1c-evaluation-3d24a67e1fe041e5a8acf56a26406db4/record.json --baseline-record runs/training/step1b-evaluation-c8b127d8bf364c7dbb735795cd81c06d/record.json --historical-run runs/training/taco-1698bcb46b85435786dd2a199910910f --rights work/stage4-real/rights.json
```

Replace `--gui` with `--analyze-only` to produce JSON without a desktop window. The selector offers
the representative subset and all 47 validation images. No train/test mode exists.
Close the window normally to release images/canvas resources and finalize status.

## Provenance and selection

The approved evaluation-only checkpoint verifier is reused for both original
bundles. Current source compatibility, original artifact hashes, corpus and rights
must still pass; no optimizer/RNG/training position is restored. New replay anchors
bind the Step 1C record, Step 1B record and pretrained prediction artifact. Epoch
result hashes and saved provenance must equal fresh verification. Every image has
900 unique ranked queries; official predictions must equal the first 100 valid
queries exactly, including coordinates, scores and category IDs. Results are copied
for diagnostics, never rewritten. No new model load or inference is needed.

The initial **16-image** subset covers all eight classes using each class's smallest
normalized bounding box, then the most annotated scene, contrasting score-floor-zero
outcomes, threshold-hidden cases and maximal wrong-class/localization/duplicate/
unmatched query counts. Ties break by numeric image ID. Remaining room may include
median-size examples. Selection is deterministic, deliberately difficult and
**not a random/prevalence sample**. All 47 images remain accessible.

Representative numeric IDs/reasons and complete private diagnostics are in
`runs/training/step2-review-cce4f3dcac4c4c788cc8cab5957d396e/analysis.json`.
The 16 IDs are 155, 1250, 776, 407, 144, 1182, 1076, 906, 777, 42, 113, 200, 31,
1096, 252 and 1014. This reports identifiers/selection evidence, not photo content.

## Measured annotation-relative findings

Matching here is deterministic, score-ranked, same-class greedy IoU >=0.50 on
the saved top 100. It is **not COCO's multi-IoU AP algorithm**. Raw-900 coverage
is a separate, non-exclusive diagnostic and can include extremely weak candidates.
No error label establishes visual truth, annotation completeness or physical counts.

At diagnostic threshold 0.05 (the matched counts also hold at score floor zero):

| Annotation outcome | Pretrained | Epoch 1 | Epoch 2 |
| --- | ---: | ---: | ---: |
| Matched same class, top 100, IoU >=0.50 | 28 | 17 | 14 |
| Same-class IoU >=0.50 candidate only outside top 100 | 4 | 7 | 5 |
| IoU >=0.50 candidate exists only with another assigned class, all 900 | 21 | 30 | 32 |
| Best any-class candidate IoU in [0.10,0.50), all 900 | 10 | 14 | 18 |
| No candidate reaches IoU 0.10, all 900 | 12 | 7 | 6 |
| Total annotations | 75 | 75 | 75 |

The all-900 rows describe availability, not confident detections or one-to-one
object assignments. The implementation reports assignment conflicts separately;
none occur in this corpus. At threshold 0.25, pretrained matches 18 annotations
and hides another 10 compatible retained candidates. Epochs 1/2 match zero and
hide 17/14 compatible retained candidates. Thus lowering the view threshold exposes
some useful boxes but cannot recover the other 58/61 annotations from top-100
same-class IoU >=0.50 candidates. No official metrics/settings are changed.

At threshold 0.05, prediction-relative outcomes are:

| Query outcome | Pretrained | Epoch 1 | Epoch 2 |
| --- | ---: | ---: | ---: |
| Matched | 28 | 17 | 14 |
| Duplicate around an already matched same-class annotation | 45 | 14 | 10 |
| Wrong assigned class with IoU >=0.50 | 342 | 125 | 136 |
| IoU in [0.10,0.50) | 987 | 603 | 985 |
| Unmatched prediction (IoU <0.10) | 2689 | 1095 | 1981 |

These counts include repeated queries. Unmatched predictions are **not confirmed
false positives**: they may indicate unannotated litter, other objects or background.
Fewer errors above a threshold can result from score suppression, not better quality.

### Eight-class support and patterns

| Class | GT objects | Matched pretrained | Matched epoch 1 | Matched epoch 2 |
| --- | ---: | ---: | ---: | ---: |
| cigarette | 24 | 1 | 1 | 1 |
| plastic_bottle | 8 | 4 | 7 | 6 |
| plastic_bag | 2 | 2 | 2 | 2 |
| plastic_wrapper | 11 | 9 | 1 | 1 |
| plastic_film | 13 | 2 | 0 | 1 |
| drink_can | 7 | 2 | 1 | 0 |
| plastic_cup | 5 | 4 | 3 | 3 |
| styrofoam_piece | 5 | 4 | 2 | 0 |

Candidate overlaps frequently assign cans as bottles (33/24/30 wrong-class queries)
and wrappers as bottles (35/16/16). Styrofoam-to-film overlaps are also present
(16/16/20); epoch 2 has ten wrapper-to-film wrong-class queries. These are repeated
annotation-relative query associations, not human-confirmed material confusion.
Only two bags support bag results; no strong class-level generalization is justified.

Corrected COCO AP remains 0.111576 / 0.126627 / 0.066215, with AR100
0.363077 / 0.275363 / 0.235395. Epoch 1's higher aggregate AP coexists with fewer
IoU-0.50 matches and lower recall: class averaging, ranking and other IoU thresholds
matter. There is no contradiction and no measured improvement from view thresholds.

Scores are unchanged saved values. Maxima are 0.596656 / 0.184526 / 0.123536;
epoch models still have zero queries >=0.25. Confidence-suppression cause remains
unproven. Saved raw phrases may be empty/incomplete because their token threshold
is independent of evaluation class-score calculation; no category is forced from
those phrases. Full per-class logits are unavailable in saved records, limiting
alternative-class/margin analysis without fresh inference.

Using the pinned native resize-size helper on annotation dimensions, two cigarette
boxes have a short side below two pixels at 256/384 resize (minimum 1.631 pixels).
This supports investigating resolution limitations later, not a claim that resolution
alone causes misses. Small-object COCO AP/AR100 remain zero for all three states.

### Examples requiring human confirmation

| Image ID | Automated reason for review | Human confirmation |
| --- | --- | --- |
| 42 | Pretrained matches one annotation; epochs match none; wrong-class overlaps | Pending |
| 113 | Pretrained matches none; both epoch models match one | Pending |
| 200 | Pretrained/epoch 1 match one; epoch 2 matches none | Pending |
| 31 | Wrong-class overlaps across all three models | Pending |
| 1096 | Many low-IoU overlaps around a matched annotation | Pending |
| 252 | Six pretrained duplicate queries; epoch-1 candidate outside top 100 | Pending |
| 1014 | Many unmatched queries; annotation completeness/appearance unresolved | Pending |

No agent claim about actual materials, occlusion, litter completeness or visual
false positives is made. The local-only constraint precludes sending photo renders
for remote interpretation. An interactive owner window was opened; explicit notes
and confirmed feedback must remain separate from automated conclusions.

## Verification and preservation

Fourteen new tests cover independent IoU values, threshold versus rank suppression,
classification/localization/duplicates/conflicts, empty annotations, saved query and
pixel-coordinate identity, ranking/record tampering, test rejection, reproducible
class coverage, literal rendered extents/aspect, no exports and callback error closure.
**201 tests discovered: 198 passed, zero failures/errors, three skipped** (existing
Windows symlink privilege and two forbidden toy optimizer tests). Actual commands:

```powershell
.venv-training/Scripts/python.exe -B -m unittest discover -s tests -p test_taco_visual_review.py -v
.venv-training/Scripts/python.exe -B work/step2/check_software.py
.venv-training/Scripts/python.exe -B work/step2/check_native_viewer.py
.venv-training/Scripts/python.exe -B work/step2/check_native_viewer.py --callback-only
```

The ignored harnesses are local evidence, not required distributed entry points.
Actual Windows TkAgg replay checked all 16 representatives and full-set navigation,
four original extents/box coordinates, table correspondence/copy, selected-query
highlight, shared limit propagation, wheel handler, all-900 access, threshold-empty
images and normal closure. It performed 21 hash-verified photo reads including
repeats. Nine guards recorded zero model/training/state/export calls. Evidence:
`runs/training/step2-review-12ea62e5f84c44069d0023510ec0f24e/native-runtime.json`.
The final callback-error hook additionally passed a native **generated-image-only**
failure/closure check. Manual label readability and physical mouse gestures remain
pending; software/native checks alone do not establish visual success.

The interactive owner replay is a separate UUID record; closure stays pending
while its window is open. Saving explicit notes does not automatically mark all
visual checks passed. No temporary/annotated source-photo copies are created.

Preservation snapshot covers **1087 existing files**, including prior source work,
all historical runs/checkpoints, original snapshot, train/validation originals,
source annotations, exports excluding test JSON, taxonomy, rights and environment
reference files. Final audit passed: only four intended current-state Markdown files
differ; no unexpected differences, all 38 package versions unchanged and all 122
checked relative links resolve. Owner configuration and strict resume verifier
hashes match. Evidence is `preservation.json` beside the final analysis. No strict resume,
inference, model, prompt, threshold, split, installed library or dependency is changed.
No automatic commit, push, merge, publication or promotion.

## Recommendations — proposals only

1. Complete owner review of selected boxes, raw phrases, apparent materials and
   unmatched regions before treating annotation-relative associations as visual errors.
2. For a future separately authorized experiment, restart from the unchanged pinned
   pretrained model with **corrected targets**, preserving other settings initially.
   That isolates the confirmed supervision defect; historical bad-target training
   is not repaired by reevaluation and is not an approved resume starting point.
3. Track class-level recall/IoU, wrapper/can/styrofoam failures and score tails as
   well as AP. Do not choose settings merely to raise detection counts or select
   epoch 1 solely from a small aggregate AP gain.
4. Only after review, consider a separately memory-qualified resolution experiment
   for tiny objects and a diagnostic investigation of score suppression/alternative
   class logits. Neither is implemented or proven causal here.
5. Retain test isolation and the eight-label/split contracts. No new dataset,
   annotation repair, taxonomy change or training experiment is started.
