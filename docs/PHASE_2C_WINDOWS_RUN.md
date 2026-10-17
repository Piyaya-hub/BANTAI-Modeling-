# Phase 2C direct Windows execution — 2026-10-17

## Provenance and preserved state

Codex directly executed this follow-up on the owner's Windows checkout, using the existing `.venv`. Fetched and switched to `fix/grounding-dino-padding-logits`, verified at `65e085b` (PR #8). Initial tracked files were clean; the untracked `BANTAI-Modeling-/` folder was left untouched. No packages were installed, requirements changed, cached weights deleted, prior results overwritten, or merge performed.

## Software fixtures

`.venv/Scripts/python.exe -m unittest discover -s tests -v` ran 54 tests in 4.802 seconds: **53 passed, one skipped**, exit 0. The symlink fixture skipped for Windows privilege error 1314. Matplotlib/Pyparsing deprecation warnings did not fail the canvas test. These synthetic fixtures are software evidence, separate from the following actual pretrained-model run.

## Actual cached model and photo run

Command: `.venv/Scripts/python.exe scripts/detect_images.py --mode infer --input data/input --results results --local-files-only`.

Local artifact directory: `results/575f58c86a5b498eabd7d85d1dd001b3/` (ignored). Per-image record: `274440b3fafe83d696f38e5f18e3630ada5109802f9eaf5fc0a20faa3ead9e9f.json`; accompanying `run.json`, `environment.json`, and `preprocessing.json` record execution details. No photographs or generated JSON are committed.

- Input: `data/input/test#1.jpg`, oriented dimensions 3000 × 4000. SHA-256 before and after inference: `8e4bc6101d2e21d45feac245ff01cda71aa0502155a97402df05c4c250949a4f`.
- Direct environment: Windows 10 build 19045, Python 3.11.9 x64, torch 2.7.1+cu118, torchvision 0.22.1+cu118, Transformers 4.51.3, Pillow 11.2.1, Matplotlib 3.10.3.
- Pinned model/processor revision: `a2bb814dd30d776dcf7e30523b00659f4f141c71`. Local-only loading succeeded. Actual device `cuda`, float32, custom kernels disabled; no fallback. Two completed forwards including warm-up.
- Separate direct `nvidia-smi` observation while the run was active: GTX 1660 Ti, driver 472.19, 6144 MiB total, 3809 MiB used, 2335 MiB free. This is a point-in-time whole-device observation, not peak model memory. The application environment manifest itself leaves driver null because it does not query it.
- Defaults unchanged: combined prompt `a garbage pile. an individual discarded item. a group of individual litter.`, threshold 0.35, text threshold 0.25.
- Processing `ok`, inference completed, two retained candidates, no invalid candidates or numeric diagnostics: q0 `a garbage pile` → `garbage_pile`, score 0.419058620929718; q1 `an individual discarded item` → `individual_litter`, score 0.4087294042110443. No group candidate was returned; this does not establish absence of a group in the photo. Physical-object and independent-region counts remain null.
- Model load 2166.734 ms; warm-up 3456.512 ms; one measured inference interval 755.641 ms. The measured interval includes processing/transfer/forward/grounded postprocessing/extraction and excludes decoding, model load, warm-up, duplicate/overlap review and display. These are single-run observations, not a benchmark.

The previous all-900-query padding rejection did not recur. The corrected audit permitted this actual cached CUDA run to produce valid retained predictions. This verifies execution for this input, not category correctness or general compatibility across inputs.

## Display and remaining evidence

The native viewer returned normally and the command exited 0. Final per-image display is `shown` with no error; run status is `completed` with zero processing, display or persistence errors. All four artifacts parse as strict JSON; the run directory contains JSON only. This is direct native lifecycle evidence, not a screenshot-based assessment by Codex.

Owner qualitative feedback after viewing: "the detection is correct, there are real garbage piles that are too small." This supports the owner's assessment of the displayed detections on this photo and flags small piles for further review. It does not specify whether the small piles were missed or whether their boxes were too small; that distinction remains unresolved.

Only one photo was processed. Beyond the owner feedback above, missed objects, false positives, pile/group ambiguity, mixed-scene coverage and prompt comparisons remain unverified. Scores are model scores, not accuracy or calibrated correctness probabilities. Phase 2C acceptance is not claimed. Next authorized action is review the small-pile observation and check additional owner-supplied photos within the existing baseline. No grouping, training, web modules or Phase 3 optimization is authorized.
