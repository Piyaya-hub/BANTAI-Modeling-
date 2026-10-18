# One scrollable results tab

The owner's latest instruction replaces both terminal image previews and separate per-image/side-panel windows. One local Tk window contains one **Detection Results** tab. Each image repeats this exact vertical sequence: **Image #1** and its image, **Detection Summary**, **Text-Token Details**, **Configuration**, then Image #2 and all four sections again. Only the numbered heading appears above each image. No browser or application/web module is involved.

## Commands

Use the existing Windows `.venv`; no package reinstall or weight download is needed for replay. Test three software-only images (including incomplete/long phrases, a narrow edge box and empty detections):

```powershell
.venv\Scripts\python.exe scripts\detect_images.py --mode display-check
```

Replay a saved result and matching original without loading or running the model:

```powershell
.venv\Scripts\python.exe scripts\detect_images.py --mode replay --result results/a2ef6836b41b4e18a793642b1ad461a4/da4b761153fda0359330cde3950d3b4cc1a2247db5bbe6618aec59c092e4afc0.json --image "data/input/test#2.jpg"
```

Repeat `--result` and `--image` to show multiple saved results together. The first result is paired with the first image, and so on; pair counts must match. This example includes a saved empty result:

```powershell
.venv\Scripts\python.exe scripts\detect_images.py --mode replay --result results/a2ef6836b41b4e18a793642b1ad461a4/da4b761153fda0359330cde3950d3b4cc1a2247db5bbe6618aec59c092e4afc0.json --image "data/input/test#2.jpg" --result results/ea5da8d9bdc840bb924d5b20c2f1733c/9d75f19ceb224a47917ee88b751c4b74928536296000016824365d6cde7787e6.json --image "data/input/test#3.jpg"
```

Explicit cached-model inference still uses the current local prompt/thresholds without changing them:

```powershell
.venv\Scripts\python.exe scripts\detect_images.py --mode infer --input data/input --local-files-only
```

Preparation remains the default. `--no-display` keeps the text-only path; `--no-display --terminal-mode full` retains all saved raw query/token diagnostics. GUI tables are always passing-only. Legacy preview-width/pagination options do not control the GUI. The configuration backend remains `TkAgg`, with `display.layout=scrollable_tab`; older configurations receive this default.

## Navigation, images and labels

Use the vertical scrollbar or mouse wheel to navigate the whole tab; Shift+wheel or the horizontal scrollbar moves sideways on narrow windows. Tables have their own horizontal scrollbars for long rows. They use read-only selectable text: drag to select, Ctrl+A to select a table, Ctrl+C to copy. Text is never silently truncated. Per-image Home/Back/Forward/Pan/Zoom controls remain below each image; Save/export controls are omitted. Close the one window to complete display for the batch.

The complete EXIF-oriented RGB original is passed to Matplotlib in memory. Its image extent is `(0, width, height, 0)`, with equal aspect ratio and original pixel-coordinate ticks on all four sides. A small display-only margin above the image accommodates top-edge labels; values outside the original dimensions have no coordinate labels. Window fitting, zoom and pan do not rescale stored boxes or change image bytes. Screen resolution limits the initial fit view; zoom inspects original samples without inventing detail.

Only valid retained predictions passing the saved strict box-threshold decisions get boxes. Labels show the **raw decoded phrase**, optionally prefixed by its matching query ID when space permits, never the mapped category or prompt substitution. Scores stay in the summary. Incomplete phrases such as `a group` remain incomplete. Empty decoded strings are represented explicitly as `""`.

Label layout is recalculated against actual screen dimensions on every draw, including zoom/pan. Following the owner's Phase 2E feedback about oversized/missing labels on small and nested boxes, defaults are 7.5 points with a 7-point fallback. Labels show query ID and the honest raw phrase, wrapping by actual glyph widths. Small boxes may borrow horizontal space within the axes; labels move upward to avoid enclosing borders and preceding labels. Thin connectors associate displaced labels with their boxes. Labels remain above their box and their backgrounds do not obscure retained box borders. Original coordinates and predictions are unchanged.

All labels were placed at the checked initial fit for the owner-reported run and all 15 Phase 2E photo/result combinations. This is not a guarantee for arbitrarily dense or zoomed scenes: when no readable space exists, the full phrase/query stays in the selectable summary, and zoom may reveal it. Offscreen box tops do not leave stray labels. Final owner acceptance of the corrected size/placement is pending; earlier human feedback reported a failure. See [Phase 2E evidence](PHASE_2E_RESULTS.md).

## Tables and actual configuration

Detection Summary contains query ID, exact quoted decoded phrase, stored mapped category/reason, decimal model score, rounded original-pixel xyxy box and threshold status. It excludes invalid, duplicate and below-threshold candidates. `unmapped` remains distinct from inference failure or threshold rejection; no category mapping is recalculated. Scores are not accuracy or calibrated probabilities. Only presentation values are rounded.

Text-Token Details contains passing ordinary tokens for those retained queries, including original tokenizer positions and token IDs. Special/padding, invalid and below-text-threshold rows are excluded. Complete raw scores, precision, invalid diagnostics and alignment remain unchanged in JSON.

Configuration shows the saved image identity, actual inference prompt, box/text thresholds, model ID and immutable revision, device actually used, and coordinate frame. Replay always uses the saved inference values, even when the current configuration differs. Missing historical information says `not recorded`; it is never inferred. Empty results still include the original image, all four sections, explicit empty detection/token messages and configuration.

## Persistence and source verification

Inference persists each result before it can enter the GUI, queues in-memory image copies, releases the model, then opens the one batch tab. Successful processing status/predictions remain independent of display status. All queued records remain pending until normal window closure. GUI initialization, rendering or callback failure records a separate display error for queued results; it never erases detections. Interruption leaves pending. Each final JSON update is atomic; a failed final update leaves that record pending, while previously finalized records remain finalized. This is not a batch-wide transaction.

Replay verifies **every pair** before creating any pending attempt: result stability, source path identity, SHA-256, oriented dimensions and coordinate frame. It imports no model/inference modules and never rewrites the original functional JSON/photo. Every source gets its own `results/<uuid>/replay.json`, with image number, saved provenance, layout, pending/final status and `inference_executed=false`. One shared window displays them all. Rendering creates no automatic annotated copies or temporary image files. Dependencies, weights, prompts, thresholds and category mapping are unchanged.

## Current configuration and verification

`display.label_font_size` and `display.label_min_font_size` are optional finite point sizes in [6,16], with minimum <= preferred. Defaults are now 7.5/7 points after the Phase 2E owner feedback. Add these fields to the existing display object to customize; no prompt or threshold change is needed. Current display settings are recorded in new inference/replay outcomes and shown alongside saved inference configuration. Historical results use the defaults unless replay display settings override them. Scores and thresholds use decimal [0,1] units throughout.

See [Phase 2E results](PHASE_2E_RESULTS.md) for current evidence and pending human acceptance; [Phase 2D checks](PHASE_2D_CHECKS.md) preserve the earlier results. The observations below are preserved history; PR #10 has since merged them into main.

## Historical verification - 2026-10-18

Work continues on `feat/terminal-viewer-diagnostics` from fetched `000b3b0`. Owner prompt/threshold edits and the untracked `BANTAI-Modeling-/` folder are preserved. No model execution or package change is needed for this presentation work.

Software fixtures: the existing Windows `.venv` ran **89 tests: 88 passed, one Windows symlink privilege skip**, latest run 9.413 seconds. Agg fixtures cover full-source arrays, fractional boxes/original-pixel alignment, raw/incomplete/long phrases, narrow/top-edge boxes, box-width/border bounds, readable font limits, redraw after zoom, and empty images. Mock Tk checks cover one-tab repeated order, wheel directions, initialization/callback failures and closure; table text retains complete selectable content. Batch tests verify every prediction persists pending before one window opens and viewer failures preserve all predictions. Multi-replay tests verify all sources before persistence, numbering and per-source outcomes.

Direct Windows observation: the owner confirmed the three-image synthetic layout, scrolling and output were readable and closed normally; the process exited 0. An actual two-image saved replay (one with predictions and one empty) also exited 0 with separate attempts `631f9fb73c914b0781d43d35db8e8034` and `a43156ccf6b048dda495a87c3a6697a3`. The owner reported a missing label. Inspection found that q2's width estimate created excessive line wrapping and a border collision. Actual-glyph wrapping, whole-word preference and optional-ID omission corrected this; the photo-shaped software regression now displays all three complete phrases without covering borders. The corrected replay (attempts `1c762d9578f148909d83b1a2e9144fe3` / `e876a3150705405fae50b838b20cbabc`) exited 0, and the owner confirmed the phrase labels were complete. The owner then requested smaller labels; the final 8.5/8-point adjustment passed software checks but has not had separate visual confirmation. No detection-quality acceptance is inferred.

Replayed four saved records (three older photo records and the newer empty record) in text-only summary/full modes with no inference, model imports or socket connections. Summary query counts were 2/3/2/0; each full replay covered 900 queries. All saved JSON bytes and original photo bytes stayed unchanged. Configuration in saved replay remains authoritative over local owner edits. No prompts, thresholds, model weights, mapping rules, packages or ignore rules changed.

Earlier per-image viewer evidence is historical in [NATIVE_VIEWER.md](NATIVE_VIEWER.md); earlier model evidence remains in [PHASE_2C_WINDOWS_RUN.md](PHASE_2C_WINDOWS_RUN.md). A successful event loop or normal closure is not visual proof or a detection-quality judgment. Truly tiny or obstructed labels still use the disclosed full-table/zoom fallback; missing on-image text never means its prediction was removed.

An actual native Tk runtime check on the two saved records verified one tab, exact repeated section order, read-only table selection/copy (restoring the previous text clipboard), vertical wheel and Shift+wheel/horizontal scrollbar navigation at 850x650, and normal closure. The final smaller-label runtime check also passed (attempts `a1d75cd78b594be381a1f7c1e1e3e3b2` / `78bffd30658442ceb79f8b71b21d8ab0`); copying was skipped in that last check to preserve an unavailable/non-text clipboard. No model modules, inference, socket connections or source/result mutations occurred. These automated Tk observations are software/runtime evidence, separate from the owner’s visual feedback.

Compilation, CLI help and whitespace checks passed. Documentation validation resolved 119 relative Markdown links and parsed five strict JSON examples. The inference/postprocessing/input/result/persistence core, dependency pins and ignore rules match the starting head. The current branch had no open PR at inspection; PR #9 was already merged externally. This work does not merge or reopen it.
