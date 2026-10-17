# Historical per-image native viewer

The latest owner instruction supersedes separate windows/side panels with [one scrollable results tab](SCROLLABLE_VIEWER.md). Commands and evidence below describe the earlier layout; use the linked current contract.

# Native image viewer and saved-result replay

The owner's latest instruction restores a native Matplotlib/TkAgg window, superseding terminal-only/no-pop-up requirements. Use the existing Windows `.venv`; no dependency reinstall is needed. Model inference, checkpoint revision, prompts, thresholds, source photos and saved raw diagnostics are unchanged.

## Run and replay

Test the native viewer independently with software fixture shapes, boxes and labels:

```bat
.venv\Scripts\python.exe scripts\detect_images.py --mode display-check
```

Replay an existing result and its matching original without loading the model or running inference:

```bat
.venv\Scripts\python.exe scripts\detect_images.py --mode replay --result results/a2ef6836b41b4e18a793642b1ad461a4/da4b761153fda0359330cde3950d3b4cc1a2247db5bbe6618aec59c092e4afc0.json --image "data/input/test#2.jpg"
```

Run explicitly requested cached-model inference on the input folder:

```bat
.venv\Scripts\python.exe scripts\detect_images.py --mode infer --input data/input --local-files-only
```

Each window blocks until closed, then the next photo opens. Close using the window's X. The terminal prints passing-only detection/token tables before opening the window, so they remain available during inspection. No Enter prompts or ANSI image escapes are used. A desktop session with working Tk is required; stdout can be redirected. Tk/display initialization, rendering, callback and closure failures are reported separately from inference.

For text-only inspection, add `--no-display`. Default summary remains passing-only; explicit `--no-display --terminal-mode full` prints all saved raw query and ordinary-token diagnostics, including below threshold. `--include-special-tokens` includes active special tokens only in this full diagnostic mode. Padding is excluded. Complete precision, numeric diagnostics and token alignment remain in JSON in every mode. With a native window enabled, console tables remain passing-only even if full mode is configured. Legacy `--preview-width` and `--paginate` are accepted for configuration compatibility and have no effect on native windows.

## Image and overlays

The loader verifies and applies EXIF orientation in memory. The viewer passes the entire oriented RGB image to Matplotlib, with image extent `(0, width, height, 0)` and equal aspect ratio. Boxes remain in the JSON's original oriented pixel-edge coordinates; resizing the window, zooming and panning change only the view. Initial window dimensions fit the screen. Screen resolution limits the initial fit view; zoom reveals additional original samples, without claiming recovered or invented detail.

Only valid retained box-threshold-passing predictions get overlays, including unmapped predictions. Invalid, duplicate and below-threshold candidates are excluded. On-image labels show only the mapped category, or `unmapped` when none is assigned, following the latest owner correction. A scrollable panel in the same window shows matching query IDs, decoded phrases/categories, model-score percentages and rounded original-pixel xyxy coordinates. This keeps complete details away from overlapping boxes. Scores are not accuracy or calibrated probabilities. Category mapping is never recalculated or forced for incomplete phrases. Box lines use two display points and labels ten points with contrasting black backgrounds, staying readable as the image zoom changes. Right/bottom-edge anchors move labels inward; dense overlaps can still require zooming.

Use toolbar **Zoom** (drag a rectangle), **Pan** (drag the image), **Home** (reset) and **Back/Forward**. The toolbar omits Save/export and subplot windows. No temporary image files, annotated copies, automatic exports or browser are used. Empty successful results still show the original image with no boxes and an explicit empty terminal summary.

Terminal records retain consistent image/query IDs, quoted raw phrase, stored category/reason, decimal model score, threshold decision, rounded xyxy coordinates and separate policy state. Unmapped is distinct from failed inference and from a failed threshold decision. Full precision is retained in JSON; presentation rounding does not change comparisons.

## Replay identity and persistence

Replay verifies source path identity, SHA-256, EXIF-oriented dimensions and coordinate frame against the saved functional result. It uses saved predictions and thresholds regardless of current configuration, and never rewrites that result or photo. It imports no torch, Transformers or inference pipeline. Every attempt writes a separate atomic `results/<uuid>/replay.json` with backend `TkAgg`, pending before display and shown after normal window closure. Callback/render failures produce display error; interruption or failed final persistence leaves pending. Deliberate disablement stays disabled. The inference pipeline likewise saves predictions before opening the window; viewer failure never erases them.

`shown` records normal display completion, not a detection-quality judgment or proof of human inspection. Visual observations and software lifecycle evidence are recorded separately below.

## Verification - 2026-10-18

Current work starts at fetched `b915bd3` on the existing feature branch. GitHub inspection found PR #9 already merged externally and no open PR on this branch; this work does not reopen or merge it. Owner prompt/threshold edits in `config/detection.json` and the untracked `BANTAI-Modeling-/` folder are preserved separately from this change. No inference, package reinstall, cached-weight deletion or source modification is performed.

Software fixtures: existing Windows `.venv` ran 78 tests (77 passed, one Windows symlink privilege skip), latest run 4.359 seconds. Agg checks verify the complete source array, exact pixel-edge extent, fractional box coordinates, exclusion policies, percentage/phrase/category/coordinate labels, contrasting backgrounds and empty images. Mock Tk checks cover normal closure, callback/init failure, cleanup ordering and native CLI dispatch. Existing replay identity/failure/atomic persistence tests remain. These fixtures do not establish real-photo detection quality.

Direct Windows execution: the first synthetic CLI attempt failed on a stale exception import before opening a window. After correction, the owner confirmed the synthetic image/labels/navigation were visible and closed normally; the process exposed a toolbar cleanup error afterward. The cleanup order was corrected and regression-tested. A subsequent actual saved-photo replay (`test#2.jpg`, attempt `results/736be71a422949e3b04af39558c2f1af/replay.json`) returned exit 0 after window closure, with backend TkAgg, display shown and inference unexecuted. This is native lifecycle evidence. A second replay (`1e56831db06a49fabc6821f1eff449f1`) also returned 0, but the owner reported that its multi-line labels needed adjustment. The final layout follows the owner’s clarification: category-only on-image labels, with scores, IDs, phrases and coordinates in a separate scrollable details panel. Final category-only layout readability remains visually unconfirmed; no detection-quality acceptance is inferred. Replay metadata retains `visual_verification=pending`; owner observations are recorded here separately.

Replayed all three earlier saved photo records plus the owner's newer empty-result record (`ea5da8d9bdc840bb924d5b20c2f1733c`, third photo) in text-only summary/full modes. Summary query counts were 2/3/2/0; every full replay covered all 900 saved queries. Source verification passed, source image and result bytes were unchanged, no model modules were imported and no socket connections occurred. Empty-result rendering is covered by Agg software checks; native empty-photo visual inspection remains unrun.

Earlier terminal checks remain historical in [TERMINAL_VIEWER.md](TERMINAL_VIEWER.md), and earlier model execution in [PHASE_2C_WINDOWS_RUN.md](PHASE_2C_WINDOWS_RUN.md). Exact commands above allow further output inspection without rerunning inference.

Compilation, CLI help and whitespace checks passed. Documentation validation resolved 114 relative Markdown links and parsed five JSON examples. Pinned dependencies, inference/postprocessing/input/result/persistence code and ignore rules are unchanged.
