# Terminal viewer and complete query diagnostics

The owner requires actual inline ANSI truecolor/Unicode images with retained bounding boxes and full-size query IDs, followed by readable tables. No GUI, browser or saved annotated-image copy is used. The pinned model, thresholds, numeric validation, normal filtering and complete JSON diagnostics are unchanged.

## Usage and terminal requirements

Run directly in the **VS Code integrated terminal**, with interactive stdin/stdout, UTF-8 and truecolor support. Do not redirect or pipe output when an image is required. Use a terminal at least 50 columns x 12 rows; a wider/taller terminal makes photos easier to inspect. The Unicode half-block renderer assumes a roughly 2:1 terminal cell height/width. Actual appearance depends on font/cell geometry.

```bat
.venv\Scripts\python.exe scripts\detect_images.py --mode infer --input data/input --local-files-only --terminal-mode summary --summary-below-threshold 1
.venv\Scripts\python.exe scripts\detect_images.py --mode infer --input data/input --local-files-only --terminal-mode full --page-rows 10
.venv\Scripts\python.exe scripts\detect_images.py --mode infer --input data/input --local-files-only --terminal-mode full --include-special-tokens --page-rows 10
```

After each image, press Enter to begin reading its tables. Continue with Enter at subsequent page boundaries. Ctrl+C stops the run and leaves any already persisted pending display unfinished. The preview is sized to terminal width/height without stretching. Query IDs are rendered as normal terminal characters over the preview rather than raster text squeezed into a tiny resized image; colors match the retained boxes. The image is not replaced by a placeholder.

`display.backend` remains `terminal`. Configuration defaults are `terminal_mode=summary`, `summary_below_threshold=1`, `page_rows=10`, `include_special_tokens=false`. CLI flags override them; below-threshold selection accepts 0..900 and page rows 1..100. Page size is also limited to actual terminal height; headers repeat and long cells wrap within their columns. Historical `TkAgg` configuration values normalize to `terminal` without launching a window. Preparation remains the safe default. `--no-display` explicitly requests text and JSON only, without interactive paging, with `display.status=disabled` after successful inference.

## Per-image layout and selection

The output order is:

```text
IMAGE: sample_01.jpg
(actual ANSI/Unicode photo, retained boxes and query IDs)

DETECTION SUMMARY
Query | Detected Phrase | Score | Box Status

TEXT-TOKEN SCORE DETAILS
Query | Token | Score | Text Threshold Status

CONFIGURATION
Box threshold: 0.3500
Text threshold: 0.2500
Model: Grounding DINO Tiny
```

The outline above documents the layout; the runtime emits actual colored half blocks, not the parenthesized description. Category/xyxy geometry and retained/duplicate/invalid policy states are reported separately within the summary section. Query IDs match the image, both tables and JSON.

- Summary shows valid retained predictions and up to N highest-scoring valid below-threshold queries, ties broken by original query index. Default N=1; set `--summary-below-threshold 0` for accepted predictions only. Token rows correspond to that selected query set. Query/token omission counts are disclosed, including hidden special-token rows. Summary may still span pages; image inspection happens before any tables scroll it away.
- Full shows every raw query and token in the same two table formats, in bounded pages. Active special tokens are hidden from the ordinary token table. Explicitly pass `--include-special-tokens` with full mode to include them. Token text retains tokenizer subwords, original positions and token IDs; special rows are marked. Padding is always excluded.
- Scores and thresholds use consistent **decimal [0,1] units**, e.g. `0.8600`. Only presentation is rounded to four decimals; JSON retains raw precision. The stored dtype-aware strict comparison controls `PASS` or `BELOW THRESHOLD`; equality is below threshold even when nearby values round to identical strings. Nonfinite/invalid values show `null` and `INVALID`, with their diagnostics separate. PASS is a threshold decision, never correctness. Duplicate/excluded states are separate policy information, never replacements for threshold status.
- Below-threshold queries display `not decoded`. No diagnostic phrase synthesis or accepted category mapping is added. Their rows are not accepted detections and do not create boxes. Physical-object and independent-region counts remain unresolved.

## Complete diagnostics and persistence

`bantai.functional.v1` fields retain their meanings. Additive `query_diagnostics` (`bantai.query-diagnostics.v1`) always saves every raw query, even in summary mode or with display disabled. Ordered `tokens` preserve original positions, IDs, tokenizer strings and special markers. Attention-mask zero positions, pad IDs and fixed-width padding tail are excluded. Active special tokens remain in JSON regardless of console settings.

Every query retains its original ID/index, maximum pre-filter sigmoid score, dtype-aware threshold decision, raw pixel box, actual upstream-decoded phrase/category or null, numeric diagnostics, aligned token sigmoid scores/decisions, and invalid-token positions. Normal pinned postprocessing still determines accepted detections. Invalid numbers remain null with diagnostics, and expected negative-infinity padding remains excluded. No NaN/Infinity or invented confidence is emitted. Old result records without raw diagnostics remain readable; unavailable scores are disclosed rather than fabricated.

Inference results and all diagnostics are atomically persisted before display. A capability check rejects noninteractive stdout, unknown/dumb color support, unrepresentable Unicode and insufficient dimensions with a clear message. It emits no ANSI image on unsupported output. Display is recorded as `error/unsupported_terminal`; successful inference remains intact, with a nonzero CLI outcome. An interactive output with noninteractive stdin is also rejected because paging cannot complete. Rendering/write failures remain separate display errors. Interruptions/final persistence failures retain pending as before.

A final `shown` means capability checks, output writes, paging and flush completed. It does **not** establish human visibility or visual correctness. The tool session is not the owner's VS Code terminal; its visual verification remains pending until the owner inspects the command above.

## Diagnosis of the missing image

Direct inspection of the actual tool stream found `isatty=False`, `TERM=dumb`, initial encoding `cp1252` (the CLI previously changed this to UTF-8), and fallback dimensions 80x24. The previous implementation emitted ANSI even into captured/noninteractive streams and treated successful writes as shown. Such output is not necessarily rendered by the tool/UI. Existing captured full-mode output contained 45,900 token lines across three photos, enough to scroll previews out of ordinary terminal scrollback. Neither observation proves what the owner's VS Code displayed; both are concrete failure paths addressed here.

Corrections add capability checks, IMAGE-before-render ordering, character-cell ID overlays, width-bounded aligned tables, compact disclosed selection, and inspection pauses/bounded pages. No threshold was lowered and no renderer is presented as visually verified from writes alone.

## Presentation correction checks - 2026-10-18

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
