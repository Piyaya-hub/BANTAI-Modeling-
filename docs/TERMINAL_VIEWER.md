# Passing-only terminal viewer and display-only replay

The owner's latest instruction limits console tables to **valid retained queries passing the box threshold** and their **ordinary tokens passing the text threshold**. Below-threshold, invalid, duplicate, padding and special-token rows remain in complete diagnostics JSON, never in normal console tables. This supersedes the earlier summary-selection/full-dump console policy. Inference, prompts, thresholds and the pinned checkpoint are unchanged.

## Check the renderer before inference

Run directly in the owner's Windows **VS Code integrated terminal**, not the Output panel, Debug Console or a redirected pipeline:

```bat
.venv\Scripts\python.exe scripts\detect_images.py --mode display-check
```

This runs no model and reads no photo. It prints output-stream facts and draws an in-memory colored fixture with two boxes and normal-size `test0`/`test1` character labels. No image, browser, external window or annotated file is created. Inspect whether the colored image and both labeled boxes actually appear. Exit 0 alone establishes output completion, not visual success.

Requirements: interactive UTF-8 stdout, ANSI truecolor/Unicode support, at least 50 columns x 12 rows, and interactive stdin for paged photo display. Windows VT processing is explicitly enabled on native console handles. A recognized terminal or successfully enabled native Windows VT is used as capability evidence; inherited `TERM=dumb` alone no longer rejects a working Windows console. Unknown terminals, redirection, unsupported Unicode and insufficient dimensions receive a clear diagnostic rather than literal escape/image-placeholder output. The half-block renderer assumes roughly 2:1 terminal-cell height/width. Narrow or short terminals reduce photo detail; enlarge the terminal for inspection.

## Replay existing results without model execution

This exact command uses an existing local result and its matching original:

```bat
.venv\Scripts\python.exe scripts\detect_images.py --mode replay --result results/a2ef6836b41b4e18a793642b1ad461a4/274440b3fafe83d696f38e5f18e3630ada5109802f9eaf5fc0a20faa3ead9e9f.json --image "data/input/test#1.jpg"
```

For another record, supply its per-image functional JSON and unchanged original. Replay validates strict JSON/result structure and source identity: resolved source path identity, SHA-256 bytes, EXIF-oriented dimensions and coordinate frame must match. A copy at another path or modified original is rejected before display or replay output reservation. Saved thresholds and predictions are used unchanged, even if current configuration thresholds differ. Replay does not import torch, Transformers, the detector or inference pipeline, contact the Hub, load weights or discover the configured photo folder.

Replay creates a new ignored `results/<uuid>/replay.json` using `bantai.display-replay.v1`. It records the original result path/hash, saved run ID/model/threshold provenance, verified source and `inference_executed=false`. It atomically saves pending before display and final shown/error afterward. Interruption or a failed final write leaves pending; `--no-display` creates a deliberately disabled identity-verification attempt. **The original result JSON and its display status are never rewritten**, including when replaying an already final shown/error result. Complete raw diagnostics stay intact in that original JSON.

## Normal inference and layout

Once the renderer is visually confirmed, normal inference remains explicit:

```bat
.venv\Scripts\python.exe scripts\detect_images.py --mode infer --input data/input --local-files-only --page-rows 10
```

Each image block follows IMAGE filename, the actual in-memory annotated half-block image, DETECTION SUMMARY, TEXT-TOKEN SCORE DETAILS, then CONFIGURATION. Only valid retained passing boxes are drawn, with query IDs matching both tables and JSON. If no queries pass, the original image still appears with no boxes and an explicit empty-result message. If no ordinary tokens pass, the token table says so. Old result JSON without token diagnostics is disclosed without inventing scores.

Tables align within terminal width and use decimal scores/thresholds in [0,1], rounded to four places **only for presentation**. Stored dtype-aware comparison masks control filtering; equality does not pass. For older records without raw query masks, the viewer reproduces the pinned float32 comparison using Python struct conversion, without a torch import. Above-threshold valid unmapped predictions retain their actual phrase, without invented category names. Token text preserves subwords, original positions and IDs. Raw precision, invalid-value diagnostics and all excluded queries/tokens remain in JSON. PASS is a threshold decision, not visual correctness, accuracy or calibrated correctness probability. Physical-object and independent-region counts remain unresolved.

Press Enter after inspecting the image and at page boundaries. Ctrl+C stops with the persisted pending attempt unfinished. `--page-rows` bounds table data lines, further limited by terminal height. `--terminal-mode summary/full`, `--summary-below-threshold` and `--include-special-tokens` are accepted as legacy syntax but cannot enable excluded console rows; the latter two normalize to zero/false. Both console mode aliases now follow `passing_only`. Complete diagnostics can be inspected in the original result JSON, not an unbounded console dump.

The inference display record adds `console_policy=passing_only` and `visual_verification=pending`. Its existing persist-before-display and atomic shown/error lifecycle remains. Unsupported output produces display error with successful inference preserved. Inference stays preparation-only unless `--mode infer` is selected. `--no-display` explicitly skips rendering and still prints passing-only tables.

## Actual verification and remaining blocker - 2026-10-18

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
