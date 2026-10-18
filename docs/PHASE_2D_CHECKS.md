# Phase 2D — Display and Summaries checks

Historical Phase 2D evidence. Phase 2E was subsequently authorized, and the owner reported small/nested-box label issues; see [current correction and results](PHASE_2E_RESULTS.md). Font defaults and prior acceptance gaps below describe the 2D version.

Date: 2026-10-18. Base: fetched main `76c33f9`, containing PR #10 and the complete prior scrollable viewer. Focused branch: `feat/phase-2d-display-summaries`. No outstanding viewer work needed cherry-picking. Owner prompt/threshold edits remain local and unchanged; the untracked nested folder is untouched. No packages, model weights, revision, mapping or detection logic changed. No inference was run.

## Changes and evidence

The existing implementation already provided one repeated four-section tab, raw phrase wrapping, original oriented pixel axes, passing-only selectable tables, verified single/multi-image replay and atomic separate display outcomes. This phase adds validated configurable fonts (8.5/8-point defaults retained), persists those display settings, shows processing status/errors explicitly and explains decimal score/threshold units. Extremely narrow or obstructed labels still use the disclosed complete-table/zoom fallback; phrases are never completed or remapped.

| Check | Actual outcome |
| --- | --- |
| Existing Windows `.venv` unit suite | 90 tests: 89 passed, one Windows symlink-privilege skip; 10.461 seconds on first run. |
| Agg label/coordinate fixtures | Passed original pixel arrays, fractional boxes, narrow/top-edge/long/incomplete phrases, border bounds, empty image, redraw and configurable font validation. Software evidence only. |
| Lifecycle and failure fixtures | Passed pending persistence before shared display, callback/init failure, prediction preservation, atomic-write failures, replay mismatch rejection and closure. No real desktop failure was induced. |
| Actual Windows saved two-image Tk replay | Passed one tab and two complete blocks, detection plus empty image, vertical wheel, Shift-wheel/horizontal scrollbar at 850x650, read-only selection/copy with prior clipboard restored, and normal closure. |
| Source preservation/replay isolation | Original photo and functional JSON bytes unchanged; zero socket connections and no torch, Transformers, inference or pipeline imports during native replay. |
| Human final visual acceptance | Pending: final smaller-label readability, dense/edge cases and interactive zoom/pan on the final version. Automated Tk success is not visual proof. Earlier owner feedback is preserved in the scrollable-viewer history. |

The native run command was `.venv/Scripts/python.exe work/verify_tab_runtime.py`; the identical reproducible harness is now tracked as `tests/windows_replay_check.py`. It uses the two existing owner photo/result paths listed below, auto-closes and does not require model access. First-run replay attempts: `49cbae0e303d4bdab7ba76af42f236f5` and `80205ffd805c463f95eadc5c45cf7f55`, under ignored `results/<id>/replay.json`. Both completed as shown, with visual verification pending. These are actual native runtime observations, not a detection-quality evaluation.

Final rerun of the tracked harness passed: attempts `831d17d8e6fb4d8caa0d9faf8958ae33` / `61ac52184c5a4d8bb9f95ad5c5e2fae4`. Selection passed; copy was skipped on this rerun to preserve an unavailable/non-text clipboard (copy passed in the first run). Final suite: 90 tests in 11.025 seconds, 89 passed/one skip. Relative Markdown validation resolved 128 links and parsed fenced JSON examples; `git diff --check` passed. Protected inference/postprocessing/input/result/persistence files, requirements and ignore rules match the base. Pipeline changes only persist the two new display settings.

## Reproduce in the existing Windows environment

Run from the repository root, without installation:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe tests/windows_replay_check.py
.venv/Scripts/python.exe scripts/detect_images.py --mode display-check
```

The last command opens three synthetic images for human checks: incomplete/long phrases, narrow edge boxes and an empty result. Close the window normally. Synthetic detections do not measure model quality.

Actual two-photo replay, without inference:

```powershell
.venv/Scripts/python.exe scripts/detect_images.py --mode replay --result results/a2ef6836b41b4e18a793642b1ad461a4/da4b761153fda0359330cde3950d3b4cc1a2247db5bbe6618aec59c092e4afc0.json --image "data/input/test#2.jpg" --result results/ea5da8d9bdc840bb924d5b20c2f1733c/9d75f19ceb224a47917ee88b751c4b74928536296000016824365d6cde7787e6.json --image "data/input/test#3.jpg"
```

For single-image replay, omit the second result/image pair. These private local files are deliberately untracked; another checkout must supply its own matching saved result and original. Every source identity/hash/oriented dimension/frame is checked before any attempt is saved. Replay reads the saved prompt and thresholds, not the current owner experiment settings. New replay attempts record current presentation settings separately.

Inspect each Image #N and the three following sections, zoom and pan using the toolbar, scroll through the whole batch, select/copy tables, and close. Check raw phrases against saved JSON, including incomplete phrases, and verify that empty images remain visible. Inspect replay status separately from original inference status. A viewer failure preserves predictions; interrupted or failed atomic finalization can leave pending records. No annotated or temporary image is written.

Configure optional `display.label_font_size` and `display.label_min_font_size` in the existing config; defaults are 8.5 and 8 points, both constrained to [8,16] with minimum <= preferred. Detection and token scores and thresholds remain decimal [0,1], with strict comparisons. Full raw precision and below-threshold diagnostics stay in JSON.

Phase 2D implementation and automated checks are ready for review; final human visual acceptance remains pending. No Phase 2E, grouping, training, fine-tuning, collection or web work is included. See [phase acceptance criteria](PHASES.md) and [viewer contract](SCROLLABLE_VIEWER.md).
