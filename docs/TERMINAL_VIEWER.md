# Scrollable terminal previews, grouped scores and replay

The current viewer renders a larger width-driven image and streams its tables together without Enter prompts by default. **Summary remains passing-only. Explicit full mode restores all saved raw queries and ordinary token scores, including below-threshold values.** This supersedes the earlier policy that forced both modes to passing-only. Original photos/results, model inference, thresholds and checkpoint remain unchanged.

## Retest saved results without inference

Run directly in the Windows VS Code integrated terminal, not Output/Debug Console or a redirected pipeline. Test the renderer independently if necessary:

```bat
.venv\Scripts\python.exe scripts\detect_images.py --mode display-check --preview-width 100
```

Replay the existing first-photo result with a larger preview and passing-only tables:

```bat
.venv\Scripts\python.exe scripts\detect_images.py --mode replay --result results/a2ef6836b41b4e18a793642b1ad461a4/274440b3fafe83d696f38e5f18e3630ada5109802f9eaf5fc0a20faa3ead9e9f.json --image "data/input/test#1.jpg" --preview-width 100 --terminal-mode summary
```

To inspect below-threshold scores, use the same replay with full mode:

```bat
.venv\Scripts\python.exe scripts\detect_images.py --mode replay --result results/a2ef6836b41b4e18a793642b1ad461a4/274440b3fafe83d696f38e5f18e3630ada5109802f9eaf5fc0a20faa3ead9e9f.json --image "data/input/test#1.jpg" --preview-width 100 --terminal-mode full --page-rows 50
```

Add `--include-special-tokens` in full mode to include active special-token diagnostics. Padding remains excluded. Full mode can be very long; use VS Code scrollback, inspect JSON, or explicitly add `--paginate` for pauses between larger full-mode chunks. `--no-paginate` disables a configured pause setting. Summary never pauses between its small tables. `--no-display` skips the preview and prints the selected table mode without image/visual success claims.

## Preview size and terminal capabilities

`display.preview_width` / `--preview-width` sets desired preview width in cells, 20..400, default 120. Actual width is limited to terminal columns minus one, avoiding automatic line wrap. **Image height follows the original aspect ratio; terminal height does not shrink the image.** Portrait images intentionally extend into vertical scrollback. The half-block renderer uses two image pixels per text row and assumes roughly 2:1 terminal cell height/width. Query IDs are normal-size terminal characters over the retained boxes, never tiny downscaled bitmap text.

Require interactive UTF-8 stdout, ANSI truecolor/Unicode support and at least 50 terminal columns. Terminal height only needs to be positive. Interactive stdin is required only for explicitly enabled full-mode pagination. Native Windows VT processing is enabled when available; a working native VT console or known terminal takes precedence over inherited TERM=dumb. Unsupported/redirection/Unicode/size errors remain honest display failures, with no image placeholder or literal-escape rendering claim. No browser, GUI, saved image or annotated export is used.

Configuration additions/defaults: `preview_width=120`, `paginate=false`; existing `terminal_mode=summary`, `page_rows=10`, `include_special_tokens=false` remain. `page_rows` controls repeated table headers/chunks, independently of Enter prompts. The legacy `summary_below_threshold` option stays ignored: summary is strictly passing-only. Both full and summary preserve every raw score at original precision in JSON.

## Output layout and grouping

Order stays IMAGE filename, actual ANSI/Unicode preview, DETECTION SUMMARY, TEXT-TOKEN SCORE DETAILS, CONFIGURATION. Only valid retained box-threshold-passing detections are drawn in either mode. Empty summary results still display the original without boxes and say no queries pass.

The detection table uses Query / Detected Phrase / Score / Box Status. Token tables are **grouped under `QUERY qN` headings**, with Token / Score / Text Threshold Status columns. This removes the repeated query column and reserves enough width for the complete status heading on supported terminals. Long cells wrap deliberately within their columns; narrow text-only output uses stacked fields. Query headings repeat on continued chunks, preserving grouping. Original token positions, IDs and tokenizer subwords remain visible.

Summary selects valid retained passing queries and only passing ordinary tokens. Full selects all saved raw queries, including below-threshold, duplicate and invalid diagnostics; group headings distinguish retained/duplicate/invalid/diagnostic-only records. Full displays PASS / BELOW THRESHOLD / INVALID using the original dtype-aware comparison masks; numeric problems remain null with diagnostics. Below-threshold phrases remain `not decoded` rather than invented. An older record lacking raw diagnostics reports that limitation without reconstructing missing scores. Active special tokens are explicit opt-in for full only.

All scores and thresholds are decimal [0,1] with four presentation decimals; raw precision remains untouched in JSON. PASS never means correctness, accuracy or calibrated correctness probability. Physical-object and independent-region counts remain unresolved. Full rows are diagnostic visibility and never create accepted boxes or change normal detection filtering.

## Replay identity and atomic lifecycle

Replay imports no torch, Transformers, detector or inference pipeline. It verifies the supplied functional JSON and original source path identity, SHA-256, EXIF-oriented dimensions and coordinate frame. It uses saved predictions/thresholds even if current configuration differs, and never rewrites the original JSON/photo. Each attempt gets a separate atomic `results/<uuid>/replay.json`: pending before rendering, final shown/error afterward; interruption or failed final persistence leaves pending. Deliberate disablement remains disabled. Mode, preview width and pagination settings are recorded in the attempt. Model inference/persistence schemas and complete raw diagnostics are preserved.

A final shown means output completed, not that a human inspected VS Code. Visual verification remains pending unless actually observed. The available tools expose a distinct Windows pseudo-terminal as text/control output and cannot inspect the owner's running VS Code terminal surface.

## Actual follow-up checks - 2026-10-18

- Fetched and inspected existing branch head `8b1be7e`; no tracked user changes existed. The untracked `BANTAI-Modeling-/` folder was preserved. Work stays on the same feature branch and PR #9; no merge.
- Existing Windows `.venv`: **79 tests ran: 78 passed, one Windows symlink privilege skip**, latest run 4.068 seconds. Tests cover width-driven portrait previews exceeding terminal height, no default Enter reads, opt-in pager failure, passing-only summary, full below-threshold/invalid/duplicate/special coverage, query grouping, and aligned unwrapped status headings at 49/79/119 columns. Original replay/source identity and atomic failure checks still pass.
- Replayed all three saved results in summary and full text modes, without model modules, sockets or inference. Summary query groups were 2/3/2, with 7/6/4 passing token rows. **Each full replay covered all 900 queries and 13,500 ordinary-token rows**, including below-threshold scores. Text-only replay attempts correctly recorded disabled display; this does not establish visual success.
- At 79 columns, in-memory previews were 53/53/31 rows high for the three photos, preserving aspect ratio and exceeding terminal height where appropriate. Source JSON bytes and all original hashes remained unchanged.
- A real Windows pseudo-terminal replay of the first photo with `--preview-width 70` exited 0 **without Enter input** and printed grouped summary/token tables with the complete Text Threshold Status header. Attempt: `results/d21364a916a4478baaf624499f324867/replay.json`, shown with visual verification pending. This is observed protocol/output execution, not owner VS Code visual inspection.

Compilation, CLI help and whitespace checks passed. Documentation checks resolved 108 relative Markdown links and parsed five JSON examples. The inference/input/result/persistence core, model settings, dependency pins and ignore rules match the fetched starting head.

No inference rerun, package change, checkpoint/threshold change, photo copy or annotated export was performed. Owner VS Code visual readability remains pending; the exact saved-result commands above allow inspection without loading the model.

## Historical verification before width-driven previews and restored full mode

- Latest source head was fetched and inspected at `b2c50c2` before work; tracked files were clean and the untracked `BANTAI-Modeling-/` folder was preserved. Work stays on `feat/terminal-viewer-diagnostics`/PR #9, without merge.
- **77 software tests: 76 passed, one Windows symlink privilege skip**, 4.016 seconds. Fixtures cover passing-only query/token output even with legacy full/special-token flags, invalid/duplicate exclusions, strict mask/equality boundaries, no-pass original-image rendering, raw precision, narrow tables, renderer errors and replay identity/persistence failures. Software fixtures are not owner visual evidence.
- Independently ran synthetic `--mode display-check` in a real Windows pseudo-terminal before any model activity: stdout/stdin TTY true, UTF-8, 80x24, TERM=dumb, no terminal-program/truecolor environment advertisement. Native Windows VT enabling succeeded; the command emitted the colored fixture/IDs and exited 0. This establishes an old false rejection path and successful native output execution, **not visually inspected VS Code rendering**.
- Replayed the first photo in that real pseudo-terminal without model execution. The output contained two passing retained query rows (`q0=0.4191`, `q1=0.4087`) and seven passing ordinary token rows, followed by unchanged thresholds `0.3500`/`0.2500`. Paging completed and exit was 0. New replay: `results/47259f21189e43468f4bdfbdb7f115e5/replay.json`, display shown, **visual_verification=pending**.
- Also tested the actual redirected/noninteractive tool stream. Replay emitted no ANSI, returned 1 and recorded unsupported-terminal display error at `results/393b7fa34dd44c15a7dc05b76e8f9db6/replay.json`. The isolated process imported no torch/Transformers/detector/pipeline module and attempted zero socket connections.
- Both attempts verified and preserved the saved source-result JSON and original photo hashes. The original record still contains all 900 queries x 17 non-padding tokens. No new inference, installation, threshold change, weight deletion, photo copy or annotated export occurred.

Static checks resolved 106 relative Markdown links and parsed five JSON examples; compilation, CLI help and whitespace checks passed. Inference/input/result/persistence source, checkpoint/prompt/threshold settings, requirements and ignore rules match the fetched starting head.

**Exact blocker:** tools expose a redirected stream and a separate Windows pseudo-terminal as text/control output, without visual access to the owner's running VS Code terminal surface. Neither is a visual inspection of the photo, colors or label legibility in that target. Owner visual verification remains pending; run the synthetic check and replay commands above inside VS Code and inspect their actual appearance before repeating inference. No visual success is claimed from writes or mocked tests.

## Historical passing/below-threshold presentation checks (superseded console policy)

Results from direct Windows software/model checks are recorded here separately from historical rendering claims below.

- Existing `.venv` software suite: **66 tests ran, 65 passed, one Windows symlink privilege skip** (error 1314). Latest run completed in 3.680 seconds. Regression checks cover IMAGE/table/configuration order, decimal units and strict boundaries, summary selection and omission counts, full/special-token coverage, width bounds at 32/50/79/120 columns, interactive pauses and cell-size ID placement, unsupported stream/color/Unicode/size diagnostics, renderer/write errors, closed input and unchanged persistence behavior. Compilation and CLI help passed. These are software fixtures, not owner visual evidence.

- Direct stream probe: noninteractive stdout, cp1252 before CLI UTF-8 configuration, TERM=dumb, fallback 80x24. No owner VS Code terminal/desktop is exposed to this tool; visual verification remains **pending**. The exact commands for that check are the interactive summary/full commands above.
- Actual Windows cached CUDA inference on the three current input photos, new ignored run `results/a2ef6836b41b4e18a793642b1ad461a4/`: inference succeeded for all three, 2/3/2 retained candidates and complete 900 x 17 query/token diagnostics each. **Every detection and raw diagnostic record exactly matched the prior CUDA run.** No socket connection was attempted; all original hashes were unchanged.
- The actual tool output was intentionally a noninteractive pipe. The corrected CLI returned **1**, run `completed_with_errors`, with zero processing errors and three separate `display.error.code=unsupported_terminal` outcomes. No ANSI escape was emitted; IMAGE headings, readable summary/token tables and configuration still appeared. This is a passed check of honest unsupported-output handling, not visual display success. Prior writes-only `shown` claims are not reused as current evidence.
- Full-mode table generation was checked against each actual saved model record with explicit special-token inclusion. Each complete raw record remains intact in JSON; full tables are divided into bounded chunks. Interactive layout, Enter paging, normal-size ID overlays, renderer failures and closed input are exercised by software fixtures, not claimed as observed VS Code rendering.


Documentation validation resolved 102 relative Markdown links and parsed five fenced JSON examples. Compilation, CLI help and whitespace checks passed; inference/result validation source, requirements and ignore-rule contents remain unchanged.

## Historical checks before the presentation correction

Software fixtures use synthetic predictions/logits, including real torch sigmoid operations on those synthetic values. They cover padding and special-token alignment, scores below thresholds, strict equality boundaries, invalid active values/nulls, unchanged filtered candidates, complete JSON with display disabled, legacy configuration, ANSI output before results, unchanged image pixels, forbidden image save/show calls, and output errors. Existing persistence/device/result checks remain.

Actual Windows/model execution results are recorded below separately from these fixtures. No fresh installation or package change is performed. Built-in `GroundingDinoProcessor` loading replaces generic processor auto-discovery: inspection and an actual run exposed optional Hub probes in the pinned AutoProcessor discovery path despite `local_files_only`. The known built-in processor receives the same revision/cache/local-only flags and preserves model preprocessing without upgrading Transformers.

### Direct Windows execution - 2026-10-17

- Existing Windows `.venv`: `.venv/Scripts/python.exe -m unittest discover -s tests -v` ran **58 tests in 3.617 seconds: 57 passed, one skipped** (Windows symlink privilege error 1314). Compilation, CLI help and whitespace checks passed. Two former Matplotlib/Tk viewer tests were replaced with terminal/diagnostic coverage; no GUI tests are presented as current rendering evidence.
- Actual full-mode cached CUDA run: `results/e814a57dc1204261b8600323b03fb35c/`, exit 0, run `completed`, all three displays `shown`, zero processing/display/persistence errors. Invoked the CLI entry function with `--mode infer --input data/input --results results --local-files-only --terminal-mode full` under an in-memory stdout/stderr capture and a socket-connect prohibition. **Zero socket connections**, no stderr. No image/log copy was saved by this verifier.
- All three manually supplied photos had **900 queries and 17 non-padding tokens each**. Full output contained **45,900 token-score lines**, plus all query rows. Strict JSON saved all query/token arrays in each per-image record (3,651,566 total JSON bytes including metadata). Actual device `cuda`; retained detections were 2, 3 and 2 for `test#1.jpg`, `test#2.jpg` and `test#3.jpg`. The first photo's entire `detections` array exactly matches the prior recorded CUDA run. These counts are candidate predictions, not physical-object counts.
- Original SHA-256 values for all three photos were checked before/after and remain unchanged. The run directory contains JSON only. The existing untracked `BANTAI-Modeling-/` folder, prior runs, cached weights, requirements and ignore rules were preserved.
- Captured output contains ANSI truecolor sequences and Unicode half blocks, with results below images. This proves output generation/write completion, **not visually inspected VS Code appearance**. Summary mode was additionally rendered from all three saved model results into an in-memory stream: each preview preceded its table and omitted full token rows. Output-failure behavior is covered by software fixtures. Detection quality on the additional photos has not been assessed; earlier owner feedback applies only to the first photo.

Documentation checks resolved 100 relative Markdown links and parsed five fenced JSON examples. Requirements and ignore-rule contents match the starting commit after normalizing Windows line endings.

Next authorized action: visually inspect terminal readability in VS Code and review baseline predictions on owner-supplied photos. No model/prompt/threshold optimization or phase acceptance is claimed.
