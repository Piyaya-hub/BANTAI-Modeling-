# Terminal viewer and complete query diagnostics

The owner's 2026-10-17 instruction replaces the TkAgg pop-up with an ANSI truecolor/Unicode terminal preview. The existing Phase 2C inference architecture, model revision, package pins, normal filtering and result lifecycle remain in use. No image files, external windows, grouping, training or web modules are introduced.

## Usage

Run in the VS Code integrated terminal using the existing environment:

```bat
.venv\Scripts\python.exe scripts\detect_images.py --mode infer --input data/input --local-files-only --terminal-mode summary
.venv\Scripts\python.exe scripts\detect_images.py --mode infer --input data/input --local-files-only --terminal-mode full
```

`display.backend` is `terminal`; `display.terminal_mode` defaults to `summary`. Historical `TkAgg` configuration values are accepted and normalized to `terminal`, without launching a window. The CLI flag overrides the configured terminal mode. Preparation remains the default; it never loads the model. `--no-display` disables the preview while still printing the selected results/diagnostic mode and saving complete diagnostics.

The preview uses RGB upper-half-block Unicode characters with ANSI foreground/background colors, sized to the terminal. Retained valid boxes and query IDs are drawn only on a small in-memory preview. IDs link to the phrase/category, oriented-pixel xyxy box, model-score percentage and PASS/FAIL table below each image. Clipped display boxes and differing raw boxes are distinguished. Originals are never modified. Terminal cell resolution limits small-object visibility; this preview cannot replace detailed pixel-level examination. Color appearance depends on terminal/font support. Full mode can exceed terminal scrollback; JSON remains complete.

Detection-table PASS means a retained valid candidate. FAIL means excluded invalid/duplicate output. In the full raw-query section, PASS/FAIL means only the pinned strict `score > threshold` comparison; each token uses strict `score > text_threshold`. Numeric validity is reported separately. These statuses do not measure visual correctness. Percentages are model scores, not accuracy or calibrated correctness probabilities.

## Additive JSON contract

Existing `bantai.functional.v1` fields and filtered `detections` retain their meaning. Successful extraction adds `query_diagnostics` with its own `bantai.query-diagnostics.v1` version. Older result records without that field remain valid.

- `tokens`: ordered input positions, token IDs, tokenizer token strings and explicit special-token markers. Attention-masked positions, pad token IDs and the model's fixed-width padding tail are excluded. Active special tokens such as CLS/SEP are retained and identified; tokens are never relabeled as project categories.
- `queries`: every raw model query in its original order, with ID/index, maximum sigmoid model score, strict query-threshold decision, raw pixel xyxy box, numeric diagnostics, aligned `token_scores` and `token_threshold_pass` arrays, and invalid token positions. Every token array aligns by index to the shared `tokens` array, preserving original positions even across excluded padding slots.
- Scores come from `sigmoid(outputs.logits)` before threshold selection. The normal pinned grounded postprocessor still controls filtering and decoded phrases. Queries below its query threshold have null phrase/category; no label is invented from token scores. Queries above threshold retain the actual decoded phrase, including empty/unmapped output.
- Nonfinite active-token scores become null with indexed invalid-token metadata. Expected negative-infinity padding is excluded. Existing raw-query audits still reject unexpected nonfinite values, and existing detection numeric diagnostics remain intact. No NaN/Infinity is written to JSON.
- Complete diagnostics are saved in both summary/full modes and with `--no-display`. Summary prints filtered detections plus raw dimensions; full prints every query and every non-padding token score, including failures below either threshold. Rejected raw queries remain diagnostics and do not enter detection counts or overlays.

The display record adds backend/mode metadata. Initial inference plus diagnostics are atomically saved before display. `shown` now means terminal rendering/writes and stdout flush returned successfully; it does not prove human visibility or VS Code visual readability. Write/render failures keep inference saved and finalize display error. Interruption or failed final persistence can leave pending as before. No viewer waits for window closure.

## Verification

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
