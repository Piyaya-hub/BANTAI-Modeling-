# D-060: Saved 384/576 visual replay

Owner authorization, 2026-10-24: enable existing saved validation predictions for D-056/D-057/D-058 in both standard and per-object GUI modes. No training, detector inference, GPU work, checkpoint promotion or merge.

The private local manifest is `runs/training/d060-saved-review/review-manifest.json`. It retains the four existing 256/384 entries unchanged and adds:

- `42 / 1e-5 @ 384/576`: D-056 pretrained 384/576, D-057 epoch 1 step 221, D-057 epoch 2 step 442.
- `42 / 1e-5 @ 384/576 (step 350)`: the same pretrained and epoch 1 states, plus D-058 step 350, explicitly labeled mid-Epoch 2. Highest sampled AP is diagnostic evidence with selection bias, not checkpoint promotion.

New entries verify SHA-256 anchors for prediction JSON, report, baseline, checkpoint manifests and supporting D-056/D-057 records. They verify corpus/rights/protocol/seed/LR, checkpoint context/position and reported metrics. Step 350 additionally binds its trajectory audit contract, position and metrics. Only checkpoint JSON metadata is read; no weights or optimizer state are loaded or hashed for these new entries. Existing entries retain their original loader checks. All query identities, ranks, coordinates and official prediction membership pass `saved_rows` validation.

## PowerShell launch commands

Run from the repository root. Each command starts at Image 777 with **all validation** selected. Use Previous/Next or the image selector to browse all 47 images. Close each window before launching the next command.

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
$review = "runs/training/d060-saved-review/review-manifest.json"
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576" --gui --image-id 777
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576" --gui --per-object --image-id 777
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576 (step 350)" --gui --image-id 777
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576 (step 350)" --gui --per-object --image-id 777
```

The historical manifest also remains usable. The new manifest supports its four original run-choice strings identically. Both manifests are local artifacts, not distributed data. Saved configuration includes explicit panel labels for the new entries; historical configuration is unchanged. Every launch creates a separate `saved-review-*` analysis record with atomic display status; originals remain unchanged. Standard mode retains its existing top-five drawing defaults; per-object mode exposes eligible overlaps from all 900 stored queries.

## Verification and limits

Fixture tests cover resolution-specific baseline loading, endpoint and mid-epoch positions, context/metric tampering, avoidance of checkpoint-weight reads, and both new choices dispatched to both GUI modes at Image 777. The full discoverable CPU unit suite runs under detector-forward/model-loading, CUDA initialization, backward and optimizer guards. Training-execution fixtures are explicitly skipped to honor D-060. Result: **236 tests, 220 passed, 16 explicitly skipped, zero failures/errors**. Skips comprise 15 training-execution fixtures and one Windows symlink-privilege fixture. Logs: `work/d060-saved-review/unit-tests.txt` and `unit-results.json`.

Local saved-artifact loading checks cover all six choices, 47 validation images and 900 queries per state/image, with no photo reads or torch/Transformers imports. For the four historical entries this supplementary audit substitutes checkpoint JSON metadata verification to avoid reading weights; existing unit tests cover their unchanged checkpoint-verifier dispatch. `work/d060-saved-review/artifact-loading.json` records results and unchanged hashes for audited source artifacts.

An additional guarded suite after fixing the per-run baseline fallback passed:
**236 tests,221 passed,15 skips,zero failures/errors**. This run permits mocked
trainer CLI propagation fixtures while excluding live backward/optimizer/module
execution. Its logs are `work/d060-saved-review/unit-tests-final.txt` and
`unit-results-final.json`. The baseline-override fixture also works without a
legacy shared-baseline key. Neither guarded run performs model inference.

Native Windows Tk smoke checks passed all four combinations of the two new
choices and standard/per-object modes at Image777. Each exposed all47 validation
IDs, correct panel titles, image navigation, shared wheel zoom and toolbar
zoom/pan; per-object mode also passed GT navigation. Windows closed normally.
Real model loading/forward, CUDA initialization, backward, optimizer updates,
checkpoint-state reads and image exports were blocked. Source JSON hashes
matched before/after. Evidence: `work/d060-saved-review/native-results.json`.
Automated GUI checks do not constitute owner photo/material/annotation acceptance;
human visual judgments remain pending. No commit, push or merge was performed.

## D-061 integrated saved replay - 2026-10-25

The same local manifest now adds `42 / 1e-5 @ 384/576 (350 vs 442 vs integrated)`.
Alongside GT, its panels are Step350, E2 step442 and the highest-AP integrated state.
Highest integrated eight-class AP: `blend_350_442_2575` **0.265563**, versus step350 0.262337 and E2 0.170508. All900/.05 same-class coverage: 38/75 objects, 15/24 cigarettes, and 5/7 Image777 cigarettes.
The first six choices are preserved verbatim. Replay verifies the completed D-061 audit, corpus/rights/protocol, reference hashes, exact averaging recipe and integrated prediction hash/metrics. It reads no weights and performs no inference. Saved configuration labels every panel explicitly; the internal legacy slot named `pretrained` now contains Step350 for this choice, and its displayed title is Step350.

Run from the repository root. Both commands start Image777 with all47 validation images selectable. Close one window before opening the next.

```powershell
$review = "runs/training/d060-saved-review/review-manifest.json"
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576 (350 vs 442 vs integrated)" --gui --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576 (350 vs 442 vs integrated)" --gui --per-object --image-id 777
```

[Full comparison and safeguards](TACO_STAGE_4_576_WEIGHT_INTEGRATION.md). The integrated view is diagnostic; no weights are saved/promoted. Human visual acceptance remains pending.

D-061 runtime verification: native standard/per-object Tk checks passed at Image777 with all47 IDs selectable, correct labels, navigation and zoom/pan; no model/GPU calls or saved JSON changes. Guarded suite:241 tests,225 passed,16 documented skips,zero failures/errors. Human visual acceptance remains pending. The selected25/75 blend partially recovers larger-object performance (17/36 coverage versus E1 21/36).

## D-062 cosine Epoch-2 continuation - 2026-10-25

Two new saved-only choices are registered. The main choice shows constant-LR step 350, cosine step 350 and cosine step 442 beside GT; the companion uses the D-061 25/75 integration as reference. All seven previous choices are unchanged. Replay verifies the completed continuation audit, exact schedule/restore and update counters, core-file preservation, reference/evaluation anchors and new checkpoint JSON context/position. No model or optimizer weights are read and no new inference/training occurs during replay.

Cosine step 350 AP **0.217172**; step 442 AP **0.221592**, above constant-LR E2 0.170508 but below E1 0.249891 and D-061 blend 0.265563. Endpoint cigarette coverage is 10/24 overall and 3/7 on Image 777; large-bin coverage 14/36. Passing queries at 0.05: 1133 across 47 images and 28 on Image 777, versus constant E2 1118/29.

Run from the repository root. Each command opens Image 777 with all 47 validation images selectable; close each window before opening the next.

```powershell
$review = "runs/training/d060-saved-review/review-manifest.json"
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5->cosine @ 384/576 (lr-decay)" --gui --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5->cosine @ 384/576 (lr-decay)" --gui --per-object --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5->cosine @ 384/576 (lr-decay vs integrated)" --gui --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5->cosine @ 384/576 (lr-decay vs integrated)" --gui --per-object --image-id 777
```

[Full D-062 comparison and safeguards](TACO_STAGE_4_576_LR_DECAY.md). Human visual acceptance remains pending.

D-062 verification: the guarded suite ran 245 tests (229 passed, 16 documented skips, zero failures/errors). Four native Tk smoke cases passed (two choices, standard/per-object), with Image 777, all 47 selectable IDs, correct labels, navigation and zoom/pan. Model/GPU calls, weight reads and image exports were blocked; saved JSON hashes stayed unchanged. Human visual acceptance remains pending.

## D-063 diagnostic NMS selector - 2026-10-25

Both saved GUI modes now offer **Diagnostic NMS**: **none (raw queries)** (default), **class-aware IoU 0.50**, and **class-agnostic IoU 0.50**. It affects only current in-memory tables/boxes and per-object matching; saved scores/predictions and official COCO v2 remain unchanged. Select a mode to apply immediately. Standard mode draws the first K score-ranked survivors; per-object mode retains GT-prioritized drawing. NMS can remove useful competing classes or distinct overlapping objects. [Measured effects and launch commands](TACO_STAGE_4_NMS_AUDIT.md).

## D-064 class-balanced saved replay - 2026-10-25

The main choice shows CB350, CB442 and CB 25/75 blend beside GT; the companion compares the D-061 blend with CB442 and the CB blend. All nine prior choices remain unchanged. Both modes preserve the D-063 Diagnostic NMS selector (default raw), source verification, full saved queries and separate display status. Saved replay reads checkpoint JSON context/positions and evaluation hashes, never model/optimizer weights or new inference. The verified LR rule and source/restoration contract appear in saved configuration.

CB Step 350 AP **0.258631**, Step 442 **0.242594**, and CB 25/75 blend **0.257303**. Cigarette coverage is 13/24, 14/24, 13/24; Image 777 coverage 4/7 for all three. The fixed ten-regression retention is 4/10, 2/10, 2/10; the nine-large subset is 3/9, 2/9, 2/9. This is partial recovery, not preservation of all desired gains.

Run from the repository root, closing each window before starting the next. Image 777 starts in all-validation scope with 47 IDs selectable.

```powershell
$review = "runs/training/d060-saved-review/review-manifest.json"
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576 (class-balanced)" --gui --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576 (class-balanced)" --gui --per-object --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576 (class-balanced vs integrated)" --gui --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576 (class-balanced vs integrated)" --gui --per-object --image-id 777
```

[Full D-064 comparison and safeguards](TACO_STAGE_4_576_CLASS_BALANCED.md). Human visual acceptance remains pending.

D-064 verification: 255 guarded tests, 239 passed, 16 documented skips, zero failures/errors. Four native Tk cases passed, each checking all three NMS options; all 47 image IDs selectable, correct labels, navigation and zoom/pan. No saved JSON changes or model/GPU execution during replay. Human visual acceptance remains pending.

## D-065 diagnostic taxonomy and containment controls

Every existing saved choice now supports **Taxonomy view**: official eight classes (default), four material/hazard categories, or three polymer-grouped material/hazard categories. Mapped labels/tables preserve specific subtype, raw phrase and hazard flag. Class-aware NMS and per-object green/magenta agreement use the active category. New class-aware and agnostic containment modes suppress at IoU >= .50 OR IoMin >= .80; raw remains the default. These are reversible display/audit controls, with unchanged saved JSON, checkpoints and official eight-class metrics. Containment can reduce correct-object/cigarette coverage; [measured audit, support table and exact launch commands](TACO_STAGE_4_MATERIAL_HIERARCHY.md).

Use the existing `42 / 1e-5 @ 384/576 (350 vs 442 vs integrated)` choice in either GUI mode at Image 777. Select the desired taxonomy and diagnostic NMS in the toolbar; select all900/.05 for the audit coverage pool. For CB350/CB442/CB25-75, use `42 / 1e-5 @ 384/576 (class-balanced)`. All 47 images remain accessible. No manifest changes are needed. Guarded verification: 251 passed, 16 documented skips, zero failures/errors; four full native cases plus two final hazard-table checks passed. Human visual acceptance remains pending.

## D-066 late-Epoch-2 cigarette-boost replay

New choice: `42 / 1e-5 @ 384/576 (cigarette-boost vs integrated)`, with prior D061 integration, new cigarette-boost step442, and its50/50 blend with original constant442. Both standard and per-object modes support D-065 official/material/polymer taxonomy and all five NMS modes. Add `--box-top-k 15` to either Image777 GUI command; it sets the initial cap without changing scores, matching or historical defaults. All47 validation images remain accessible. Raw first15 covers only2/7 cigarettes in each new state, so the cap is not a guarantee of drawing all cigarette GTs. [Complete report, query ledgers, verification exception and exact PowerShell commands](TACO_STAGE_4_576_CIGARETTE_BOOST.md). All11 prior choices are preserved; no checkpoint was promoted.

## D-067 score-calibrated saved replay

New choice: `42 / 1e-5 @ 384/576 (score-calibrated vs integrated)`, with prior D061 integration, cal_442, and0.25*original350+0.75*cal_442. Both GUI modes retain official/material/polymer taxonomy and all five diagnostic NMS modes. Use `--image-id 777 --box-top-k 15` in either GUI command; all47 images remain accessible, including Image833 for the absent-cigarette FP audit. [Full report, strict thresholds, score/rank/IoU and FP ledgers, verification and exact PowerShell commands](TACO_STAGE_4_576_SCORE_CALIBRATION.md). All12 previous choices remain unchanged. No checkpoint promotion.
