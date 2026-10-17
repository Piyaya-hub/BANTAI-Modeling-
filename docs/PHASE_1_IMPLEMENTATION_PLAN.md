# BANTAI Phase 1 implementation plan — Codex ownership

Current update: Phase 2A and Phase 2B are published (PRs #5–#6). The owner instructed Codex to proceed with **Phase 2C**, superseding the previous stop-before-2C boundary. The pinned inference/mapping/native-display baseline is implemented for review. See [PHASES.md](PHASES.md) and [PHASE_2C_CHECKS.md](PHASE_2C_CHECKS.md). Historical task boundaries and fictional examples below remain history, not model evidence. No automatic grouping, training, application modules or merge is authorized.

Originally prepared 2026-10-17; published through merged PR #3. Status: published planning deliverable with detector proposals still unvalidated; current Phase 2A preparation authorization does not accept Phase 1 or authorize detector implementation. The owner now assigns implementation to Codex and reports Antigravity made no changes. Phase 2A is authorized for Section 4 and Section 8 stages 2–3 only: local Windows setup/readiness. Detector implementation, model-weight downloads/loading and inference remain unauthorized. Phase 2A technical preparation has since completed on owner-supplied Windows outputs/screenshots/confirmation, recorded in [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md) and the [exact snapshot](PHASE_2A_ENVIRONMENT_SNAPSHOT.md). Publication/authorization alone was not its evidence; detector execution remains unauthorized.

## 1. Repository state and requirement precedence

Repository: `Piyaya-hub/BANTAI-Modeling-`. Current Phase 2A inspection fetched `origin/main` at `7a776cf`, the merge of PR #4 containing the contract corrections. PR #3 previously published this plan at `b366f76`. A clean checkout was used to create `docs/phase-2a-environment-preparation`; no direct main edits or merge are authorized. Earlier inspection details remain in Git history and the decision register.

Read `AGENTS.md`, `README.md`, `.gitignore`, and all tracked `docs/` Markdown files. The tracked tree contains foundation/planning/preparation documentation and the ignore file, with the verified dependency snapshot preserved in documentation; no detector, executable dependency manifest/lock, inference configuration, photos/results/model cache is tracked. Ignored remote Python state is not evidence about the owner's Windows computer.

Phase 0 is complete and Phase 1 planning is published. The latest owner requirements select Transformers and **functional visual checks on manually supplied unannotated team photographs**. Older mandatory annotations, held-out benchmarks, formal accuracy metrics, Phase 0-only task state, and pending PR #2/#3 language are superseded; history is retained in [DECISIONS.md](DECISIONS.md). The related scope, architecture, phase gates, test plan, README and agent/handoff guidance are corrected alongside this plan.

Publication is not proof of local environment preparation, successful detection, or owner acceptance of every technical proposal. Phase 2A setup/readiness is now authorized under the agreed boundary; detector implementation and inference remain future work requiring later phase authorization. Routine defaults inside an authorized phase do not require extra approval checkpoints. No annotation work, benchmark split, precision/recall/mAP, or numeric accuracy targets are required for this activity.

## 2. Confirmed scope and evidence

Confirmed by owner: pretrained Grounding DINO Tiny, Hugging Face Transformers, Windows execution, GPU preferred with CPU fallback; the latest owner-supplied preflight identifies GTX 1660 Ti/6144 MiB and driver 472.19, superseding the earlier RTX 2080 Super report; no training/fine-tuning. Detect multiple independent still-image trash regions through direct prompts first. No potholes, road damage, video, cameras, dashboards, APIs, databases, incident consolidation, hotspot ranking, Drive integration, or automatic grouping.

### Official source findings

The findings below preserve official-source checks from the original planning task. Subsequent package/wheel metadata rechecks and target-side Windows readiness evidence are recorded in [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md); see the snapshot for the limited local operations actually verified. No official example or published result is a BANTAI experiment.

| Finding | Evidence and verification limit |
| --- | --- |
| `IDEA-Research/grounding-dino-tiny` is the model ID used in the official Transformers Grounding DINO example, with `AutoProcessor` and `AutoModelForZeroShotObjectDetection`. | [Pinned official documentation source](https://github.com/huggingface/transformers/blob/v4.51.3/docs/source/en/model_doc/grounding-dino.md) was read. It links the original IDEA-Research implementation. |
| Text-guided/open-set detection is suitable for trying descriptive trash prompts, not proof of custom category or grouping capability. | [Original project](https://github.com/IDEA-Research/GroundingDINO) and the official documentation source were read. Original project lists the Swin-T configuration; its [code license](https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE) is Apache-2.0. |
| Period-separated descriptions are supported; the grounded postprocessor is required to recover phrases. | Official documentation and [processor source at v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/grounding_dino/processing_grounding_dino.py) were read. |
| In 4.51.3, use `threshold`, `text_threshold`, `target_sizes=[(height,width)]`, and returned `text_labels`. `box_threshold` is a deprecated alias. | Processor implementation inspected directly. Its example documentation still uses older names; prefer the pinned implementation. Scores are the maximum sigmoid text-token score for each query, not a calibrated accuracy estimate. |
| Transformers 4.51.3 supports Python >=3.9 and declares torch >=2.0 in its torch extra; PyTorch 2.7.1 supports Python >=3.9. | Official [Transformers package metadata](https://pypi.org/pypi/transformers/4.51.3/json), [PyTorch package metadata](https://pypi.org/pypi/torch/2.7.1/json), and [Transformers setup source](https://github.com/huggingface/transformers/blob/v4.51.3/setup.py) were read. These establish declared compatibility, not a tested runtime combination. |
| A PyTorch 2.7.1+cu118 CPython 3.11 Windows x64 wheel exists. Torchvision 0.22.1 requires torch==2.7.1. | [Official CUDA 11.8 wheel index](https://download.pytorch.org/whl/cu118/torch/) and [torchvision metadata](https://pypi.org/pypi/torchvision/0.22.1/json) were read. No wheel was downloaded. |
| Pillow 11.2.1 and Matplotlib 3.10.3 have Windows CPython 3.11 wheels; Python requirements are >=3.9 and >=3.10 respectively. | Official project-published [Pillow metadata](https://pypi.org/pypi/pillow/11.2.1/json) and [Matplotlib metadata](https://pypi.org/pypi/matplotlib/3.10.3/json) were read. |
| The Transformers model exposes `disable_custom_kernels`; the baseline should use its PyTorch path without building the original GroundingDINO extension. | [Configuration source](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/grounding_dino/configuration_grounding_dino.py) and [model source](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/grounding_dino/modeling_grounding_dino.py) were read. Set `disable_custom_kernels=True` as a proposed compatibility default and record it. |

### Outstanding official-source verification

The proxy returned `403 Forbidden` for Hugging Face, pytorch.org, developer.nvidia.com, and docs.nvidia.com. Their pages were not inspected. These were access limitations during the original planning session, not evidence that the user's computer lacks access.

Before downloading weights later, Codex must open the exact [model card](https://huggingface.co/IDEA-Research/grounding-dino-tiny), configuration and files; confirm the Swin-T/Tiny configuration, publisher/conversion provenance, model-specific license/usage conditions and any access terms, availability of safetensors, and immutable Hub commit. Pin processor and weights to the same verified revision. Do not infer checkpoint licensing solely from the original code license. Do not use `trust_remote_code=True` or download an alternative checkpoint silently.

Check the detected Windows driver against NVIDIA's [CUDA 11.8 release notes](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html) and [compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). Confirm the actual GTX 1660 Ti architecture against [NVIDIA's GPU table](https://developer.nvidia.com/cuda-legacy-gpus). Turing / compute capability 7.5 is the expected hardware fact, not a locally observed result. Supplied preflight reports driver 472.19 and 6144 MiB VRAM; driver suitability for the proposed wheel and model-memory fit remain unverified. No numerical minimum driver requirement is asserted as verified here.

CUDA 11.8 is a conservative proposed wheel choice for this GPU, not an assertion about the installed driver. Use prebuilt PyTorch wheels and the Transformers PyTorch path; do not plan a separate CUDA toolkit, compiler, original GroundingDINO source build, or custom extension merely to enable GPU inference. The NVIDIA display driver is still required. Verify wheel/runtime operation locally. The `CUDA Version` in `nvidia-smi` describes driver capability, not proof that the Python wheel uses that CUDA version.

## 3. Minimal proposed repository layout

The following implementation files/folders remain proposals; none is created by this correction:

```text
AGENTS.md, README.md, docs/       existing foundation/planning; current guidance aligned
requirements.txt                 proposed direct dependency pins
requirements-windows.lock.txt    exact resolved environment after local verification
config/detection.json            prompts, thresholds, paths, device and display options
scripts/detect_images.py         local CLI entry point and small helper functions
tests/
  test_result_contract.py        mapping/coordinates/status/duplicate policy checks
  test_input_handling.py         invalid files, empty folder, source preservation
data/input/                      manually extracted user photos; ignored
results/                         JSON and run manifest only by default; ignored
.cache/huggingface/               model/processor cache; ignored
```

Keep one small script initially; split modules only if complexity warrants it. Use standard-library `argparse`, `json`, `pathlib`, `time`, `hashlib`, and `importlib.metadata` plus the packages below. No OpenCV, pandas, scikit-learn, DBSCAN, dataset library, web framework, or Drive SDK is required.

Resolve configured paths relative to the repository root, not the shell's working directory. Defaults: `data/input`, `results`, `.cache/huggingface`. Allow explicit overrides. Input originals are read-only to the tool. Process JPEG/PNG initially, one image at a time, sorted and nonrecursive by default. Preserve unsupported files and report skips; decode failures are errors. Missing/empty folders produce actionable messages, not a model-quality failure. No photos are required to finish this plan or implement non-model checks.

This correction appends missing ignore rules `/data/input/`, `/results/`, `/.cache/huggingface/` while preserving existing entries; no directories are created and nothing is untracked. Inspection found no tracked files in those paths. Check `git status` before committing; `.gitignore` does not untrack photos already committed. Never stage real photos, caches, local absolute paths, or generated results. Optional exported annotated images must also go under ignored results.

## 4. Windows environment verification and proposed setup

Phase 2A authorizes these setup/readiness steps on the actual Windows computer once preflight is available. Commands below are not claimed executed until results are recorded in [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md). This session currently has only cloud Linux access. Start at the actual repository root; do not reuse the cloud `.venv`.

### Read-only preflight

```powershell
Get-Location
py -0p
py -3.11 --version
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
```

If a project environment already exists, inspect it without modifying global packages:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip show torch torchvision transformers pillow matplotlib
```

Record Windows version/architecture, Python architecture, driver, GPU identity, total/free memory, and installed packages. If Python 3.11 x64 is absent, recommend installing a supported 3.11 x64 interpreter alongside existing Python; do not replace system Python. If `.venv` is unrelated, preserve it and use a separately named project environment.

### Recorded Phase 2A version baseline and reproduction

Originally proposed and now installed/checked on the supplied Windows target: Python 3.11.9 x64, `torch==2.7.1` with CUDA 11.8, `torchvision==0.22.1`, `transformers==4.51.3`, `Pillow==11.2.1`, and `matplotlib==3.10.3`. The limited Phase 2A import/dependency/CPU/GPU/viewer checks passed on that target. This is not a latest-version or model-execution claim; see the exact snapshot for evidence and reproduction limits. Let Transformers resolve its required tokenizers, safetensors, Hub client, and other dependencies within its declared ranges, then freeze the exact verified environment. Review current package advisories before locking; if a required fix changes this baseline, recheck the pinned API and record the reason. No torchaudio or accelerate is required for single-device `.to(device)` inference.

Setup is authorized by Phase 2A; run these proposed commands after checking actual Windows preflight and preserving any existing environment. No separate approval is needed for routine authorized setup, but missing target facts/access cannot be assumed:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install transformers==4.51.3 Pillow==11.2.1 matplotlib==3.10.3
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip freeze
.\.venv\Scripts\python.exe -m tkinter
```

The final command should open a small Tk window; close it. It checks the proposed native display backend. Use the venv executable directly; no PowerShell execution-policy change or activation is required. Capture exact versions into a reviewed lock with the CUDA index documented; a bare freeze alone does not identify the correct CUDA wheel source. Preserve TLS/checksum checks.

Post-install local probes, still not detector code:

```powershell
.\.venv\Scripts\python.exe -c "import sys,struct,torch,torchvision,transformers; print(sys.version,struct.calcsize('P')*8); print(torch.__version__,torchvision.__version__,transformers.__version__); print('wheel CUDA:',torch.version.cuda); print('available:',torch.cuda.is_available()); print('compiled architectures:',torch.cuda.get_arch_list()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('capability:',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
.\.venv\Scripts\python.exe -c "import torch; d='cuda' if torch.cuda.is_available() else 'cpu'; x=torch.ones(1,device=d); print(d,(x+x).item())"
```

Check reported capability against compiled architectures and the actual tensor operation. These probes test packages/device execution, not detection. GPU availability alone is insufficient evidence of compatibility.

Device setting: `auto` selects CUDA if available; otherwise CPU with a visible reason. Explicit `cuda` fails clearly if unusable. Explicit `cpu` always uses CPU. In `auto`, a CUDA out-of-memory failure may trigger one clearly recorded CPU retry after releasing GPU state; preserve failure and fallback provenance. Do not catch unrelated failures as successful fallback. Process batch size one in float32 initially; defer mixed precision and image resizing changes until justified. No memory/latency promise is made. A CUDA wheel can also run CPU inference; if installing CPU-only on a machine without a suitable driver, use the same torch/torchvision versions from the official `/whl/cpu` index in a clean environment instead.

## 5. Detection, display, and JSON workflow proposal

After implementation authorization:

1. Validate configuration and input paths; discover supported images without modifying originals. Set the project cache directory before initializing Hub access.
2. Load the verified processor and pretrained model once at the pinned Hub revision. Use safetensors if available, model evaluation mode, inference-only execution, and the recorded device/custom-kernel setting. Do not train or download data automatically.
3. Decode one image; normalize EXIF orientation in memory, convert to RGB, and record dimensions of the orientation-corrected original display. Coordinates refer to this displayed original pixel frame, before model resizing; document this EXIF convention explicitly. Never save the transformed source.
4. Submit the selected period-separated text prompt and image through the processor. Use the pinned grounded postprocessor with `threshold`, `text_threshold`, and original display `(height,width)`.
5. Retain candidate raw phrases, finite raw scores/coordinates, and proposed mappings. Check numeric validity before invalid values could silently disappear through threshold comparisons. Replace each nonfinite raw coordinate or invalid score with null plus diagnostics; any invalid coordinate makes the entire clipped/display box null. Invalid candidates are not displayed or counted as valid detections. Apply the precise numeric rules in section 7; never serialize NaN/Infinity.
6. Apply the conservative duplicate/overlap proposal below and preserve candidate decisions. Do not build clusters, derive group boxes, select general NMS, or infer membership/counts.
7. Print readable category/score summaries for valid retained predictions, including valid unmapped predictions; list invalid candidates separately with reasons and no invented percentages. Category tallies count valid retained candidate predictions, not physical objects or independent regions. Unresolved physical-object/independent-region counts remain null.
8. Atomically write per-image JSON and appropriate run/environment metadata before opening any viewer. Use run IDs plus source relative path/content digest to avoid basename collisions or overwriting earlier runs. Record processing `status` independently from `display.status`. Successful processing with viewing planned starts with display `pending`; deliberate disablement records `disabled`; failed/skipped processing records `not_attempted`.
9. With viewing enabled by default, attempt native Matplotlib TkAgg in-memory display of the oriented original, valid retained boxes, category/unmapped labels, candidate IDs and individual model-score percentages. Atomically update JSON to display `shown` after rendering and viewer return, or display `error` with its error details if viewing fails. A process stopping before that update leaves `pending` as unfinished, even if a window may have appeared. Display failure preserves inference detections/status; no unverified display success is claimed. Exclude viewer waiting from inference timing, close each figure and release resources. No browser/server or temp-image file is needed.
10. Continue a batch after an image error and return nonzero for processing errors, enabled-display failures, or result persistence errors; keep their distinct causes in metadata/console output. A final JSON update failure leaves the prior result intact (possibly pending) and must be reported. Annotated export remains an optional future `save_annotated=false` action, disabled by default and confined to an explicit ignored path. No source overwrites or automatic annotated copies. Default image-derived outputs are JSON/metadata, never extra image files.

A model score of 0.72 displays as `72.0% model score`. It is not accuracy, a calibrated probability of correctness, or percentage of image area. Preserve valid unrounded 0–1 scores in JSON; invalid scores are null with diagnostics and never displayed as percentages.

## 6. Prompt, mapping, and visual-review proposals

### Initial hypotheses

Use a single lower-case, period-separated string, following the pinned Transformers syntax:

- Combined candidate: `a garbage pile. an individual discarded item. a group of individual litter.`
- Object-specific alternative: `a discarded plastic bottle. a discarded wrapper. a discarded aluminum can. a discarded bag. a garbage pile. a group of scattered litter.`
- Compare with separate passes using each description individually on the same photos. Separate passes are distinct experiments; do not pool their predictions and double-count them.

Start with `threshold=0.35`, `text_threshold=0.25`; these are unvalidated hypotheses. Both must be configurable 0–1 values; record exact prompt text, order, and thresholds. `threshold` filters query scores; `text_threshold` affects which tokens form the decoded phrase. No prompt or threshold guarantees that an object is discarded, that members form a group, or that a group differs from a pile.

### Transparent phrase mapping

Use a versioned explicit alias table. Normalize case, trim whitespace and terminal punctuation, collapse spaces, and remove only a leading `a`/`an`. Map exact normalized phrases only:

| Example normalized phrase | Proposed category |
| --- | --- |
| `garbage pile` | `garbage_pile` |
| `individual discarded item`, `discarded plastic bottle`, `discarded wrapper`, `discarded aluminum can`, `discarded bag` | `individual_litter` |
| `group of individual litter`, `group of scattered litter` | `group_of_individual_litter` |

Keep every raw phrase. Partial phrases, `bottle` without an approved alias, merged descriptions, or ambiguous multi-category phrases map to `null` and remain visible as `unmapped`. Never force a prediction into a category by substring guessing. A category mapping is a lexical hypothesis, not confirmed visual identity or group membership.

### Annotation-free visual review rules

These are reviewer guidance, not annotation work, automatic grouping logic, or measured ground truth:

- `individual_litter`: one separately identifiable apparently discarded item; it is not confirmed independent merely because its prompt mapped to that category. If it appears inside a plausible group/pile region, mark membership unresolved.
- `group_of_individual_litter`: at least two visually distinct, apparently discarded items form a local arrangement, with visible separation rather than an accumulated heap. Two is a proposed review minimum, not an implemented distance algorithm; proximity and boundary judgments remain qualitative. A direct group prediction should have one box enclosing that arrangement.
- `garbage_pile`: accumulated/stacked/overlapping waste appears as a heap rather than merely several nearby items. Do not relabel every nearby collection as a pile, or assume every pile has individually countable members.
- Ambiguous ownership/discard status, occlusion, scale, group boundaries or pile-vs-group appearance: mark `uncertain` with a short review note rather than forcing a category. No annotation files or boxes are required from the team; these rules do not become a dataset labelling prerequisite.

Preserve individual model boxes and directly predicted group/pile boxes as separate candidate records with `level=object` or `level=region`. A group box does not establish member count. Do not synthesize a group box when the model returns only items: record missing group localization as a limitation.

Duplicate proposal: retain all raw candidates; by default collapse only numerically identical boxes with the same normalized phrase into one displayed/summary record, retaining the highest score and duplicate links. Flag substantial same-category overlap (starting review heuristic IoU >=0.8) without deleting candidates automatically. Cross-category overlap and object-within-region containment are review flags, not proof of duplication or membership. Keep conflicting categories visible and mark resolution pending. Identical-box/same-phrase collapse does not remove near duplicates. IoU flags are review heuristics only; no general non-maximum suppression algorithm or automatic clustering is selected.

The UI must distinguish object candidate count from region candidate count. Do not sum a group box and its members as independent regions. With unresolved membership, keep `independent_region_count` and `physical_object_count` null in JSON and report them as unknown. The five-object/two-region example from Phase 0 remains desired behavior, not a baseline guarantee or output to fabricate.

If later authorized spatial grouping creates a region, use `origin=derived_spatial_group`, explicit member IDs and method/parameters, and `model_score=null` for the derived region. Member scores remain on member model predictions. Do not average or invent a new model confidence. No such module is part of this implementation baseline.

### Functional comparisons

Use whatever team photos are available after manual Drive download/extraction. No collection program, upload integration, labels, required benchmark split, or required annotations. Run candidate prompt configurations separately on the same unchanged images, assign run IDs, and save configuration plus qualitative observations to compare changes.

Review missed visible trash, plausible false detections, wrong categories, unmapped phrases, duplicate boxes, missed group regions, group/pile confusion, multiple separate groups, isolated items distant from groups, piles near loose items, mixed-category images, small/distant/occluded items, and no relevant trash. Where no appropriate team image exists, record the scenario as unrun instead of claiming coverage.

Record cold model-loading time separately from first inference, then run one untimed warm-up per selected device/configuration before timed repeated observations. Record image dimensions, device/GPU/driver/packages, preprocessing, warm-up/repetition count, precision, and whether timings include decode, preprocessing, transfer, model and postprocessing. Synchronize CUDA before/after the timed GPU interval; exclude human display waiting. Use optional CUDA peak allocated memory observations on GPU and explain their scope. These are timing/resource observations, not a benchmark or accuracy evidence.

## 7. Proposed JSON contract

Schema version `bantai.functional.v1`, one object per source image/run. UTF-8 JSON; no NaN/Infinity. Required fields:

| Field | Proposed contract |
| --- | --- |
| `schema_version`, `run_id` | Strings identifying contract and experiment |
| `source` | `image_id` as project-relative path, `sha256` for source identity, `width`, `height`, `coordinate_frame` (EXIF-oriented original display); dimensions may be null on decode failure |
| `model` | `id`, immutable `revision`, `library`, `library_version`; successful runs must record the resolved checkpoint revision |
| `inference` | Requested/actual device, dtype, package/driver details or manifest reference, processor/preprocessing settings, custom-kernel setting, exact prompt, `threshold`, `text_threshold`, mapping version, duplicate policy, fallback details |
| `status` | Processing/inference outcome only: `ok`, `no_detections`, `error`, or `skipped_unsupported`; independent of display outcome |
| `display` | `enabled` boolean, `status` (`pending`, `shown`, `disabled`, `error`, `not_attempted`), and separate `error` object or null; lifecycle below |
| `score_definition` | Model score range `[0,1]`; UI percentage is score ×100, not accuracy |
| `box_format` | `xyxy`, floating original-display pixel edges, origin top-left, x right/y down; `raw_box_xyxy` preserves finite components and replaces each nonfinite component with null; `box_xyxy` is wholly null for any invalid coordinate or invalid geometry, otherwise a valid clipped box obeying `0 <= x0 < x1 <= width`, `0 <= y0 < y1 <= height` |
| `detections` | Above-threshold candidates plus invalid candidates encountered during extraction: ID, origin, raw phrase, category or null, mapping reason, level (`object`, `region`, `unknown`), `model_score` (valid 0–1 or null with diagnostics; future derived groups have null), raw/clipped box, state (`retained`, `duplicate`, `invalid`), duplicate reference, and `numeric_diagnostics` list |
| `review` | Overlap/containment flags, unresolved membership, optional qualitative notes; not ground truth. Counts of physical objects/independent regions default null |
| `timing` | Scope plus observed milliseconds or null; no fabricated measurements |
| `error` | Processing/inference failure only: null or stage/code/message. Display errors go in `display.error`; neither contains secrets |

`no_detections` means inference/postprocessing succeeded with zero above-threshold candidates and no invalid prediction diagnostics, not that the photograph contains no trash. `ok` requires at least one valid candidate, mapped or unmapped; mixed invalid/valid candidates retain diagnostics without erasing the valid ones. If every extracted candidate is invalid, use processing `status=error`, `error.code=invalid_predictions`, preserve the diagnostic candidate records, and use display `not_attempted`. Other pre-inference/decode failures have empty detections and useful error information. Unsupported files use `skipped_unsupported`, not inference failure or no detections.

### Invalid numeric values and candidate validity

- Numeric diagnostics use `field`, `reason`, and optionally `index` for a coordinate. Before strict serialization, sanitize all numeric fields, including timings/settings, so no NaN/Infinity is emitted. Use standards-compliant JSON serialization with `allow_nan=False` or an equivalent rejecting mode after sanitation. Invalid required numeric settings are configuration errors; do not execute inference with null thresholds. Missing/invalid metadata numbers are null with a diagnostic, never fabricated measurements.
- `raw_box_xyxy` is a four-element list. Preserve each finite raw coordinate; replace each nonfinite component with null and an indexed reason. The **entire** `box_xyxy` becomes null if any raw component is invalid. Do not partially draw an invalid box or guess a replacement coordinate.
- A fully finite box with reversed/degenerate geometry, or one that is empty after clipping, is invalid: preserve its finite raw list, set the whole `box_xyxy` null, and record the reason. A valid box partly outside image bounds preserves raw values and records a correctly clipped display box.
- `model_score` must be finite in [0,1]. A nonfinite or out-of-range model score becomes null with a reason; if an out-of-range value was finite, preserve it in the diagnostic's `finite_raw_value`. Other valid finite numeric values are retained. Invalid model scores do not represent low-confidence predictions and are never formatted as percentages. Future derived-group null scores are a separate provenance rule, not a model score to fabricate.
- Any invalid score or box makes candidate `state=invalid`, excludes it from displayed boxes and valid-detection summaries, and preserves its raw phrase/mapping/remaining finite raw values for inspection. A valid finite score on an invalid box may remain in diagnostic JSON, but the candidate is not a displayed valid detection. Software fixtures for invalid values test handling only, not real model performance.

### Display states and atomic output lifecycle

Default `display.enabled=true`. Display has its own status/error, separate from top-level processing status/error:

| `display.status` | Meaning |
| --- | --- |
| `pending` | Successful processing persisted and display planned, but no final outcome recorded. If interrupted before the final atomic update, this is an unfinished attempt, never inferred success. |
| `shown` | The enabled native viewer rendered the image/overlays and returned normally; the final outcome was atomically saved. Merely starting a process/window is insufficient. |
| `disabled` | Viewer deliberately disabled by configuration on a successfully processed image; records intent, not verification that viewing works. |
| `error` | Viewer attempt failed; `display.error` supplies stage/code/message and inference status/detections remain unchanged. |
| `not_attempted` | Processing failed or was skipped, so no viewer attempt was planned, regardless of enabled setting. |

Lifecycle: (1) atomically write inference JSON **before** viewer startup, with pending/disabled/not_attempted as applicable; (2) attempt enabled display for successful processing, including a successful no-detection image; (3) atomically update the existing result to shown/error plus `display.error` after the attempt. Write temporary JSON beside the result, flush/close it, and replace atomically; this does not create a temporary annotated image. Initial persistence failure prevents viewer startup and is reported; a failed final update preserves the earlier JSON and is an observable persistence error, not a fabricated final display state. Display waiting never belongs in inference timing. Display failure does not erase successful detections or turn `ok`/`no_detections` into inference `error`.

### Fictional detection example

Every prediction, score, dimension, revision, hash and ID below is illustrative, not an executed experiment. The repeated hex strings are placeholders, not verified checkpoint/source hashes. Implementation must write real metadata and complete configured settings. These examples depict fictional final `display.status=shown` records; the initial saved record would be pending. These fictional JSON examples were not produced by a detector or viewer run. The separately recorded real Phase 2A synthetic-plot check did not perform inference.

```json
{
  "schema_version": "bantai.functional.v1",
  "run_id": "fictional-run-A",
  "source": {
    "image_id": "data/input/example.jpg",
    "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "width": 1280,
    "height": 720,
    "coordinate_frame": "exif_oriented_original"
  },
  "model": {
    "id": "IDEA-Research/grounding-dino-tiny",
    "revision": "1111111111111111111111111111111111111111",
    "library": "transformers",
    "library_version": "4.51.3"
  },
  "inference": {
    "requested_device": "auto",
    "actual_device": "cpu",
    "dtype": "float32",
    "environment_manifest": "fictional-run-A/environment.json",
    "disable_custom_kernels": true,
    "prompt": "a garbage pile. an individual discarded item. a group of individual litter.",
    "threshold": 0.35,
    "text_threshold": 0.25,
    "mapping_version": "aliases-v1",
    "duplicate_policy": "exact-box-same-phrase-only",
    "fallback": null,
    "preprocessing_manifest": "fictional-run/preprocessing.json"
  },
  "status": "ok",
  "score_definition": {
    "range": [
      0,
      1
    ],
    "meaning": "model score; not accuracy"
  },
  "box_format": "xyxy_original_display_pixel_edges",
  "detections": [
    {
      "id": "d1",
      "origin": "model_prediction",
      "raw_phrase": "individual discarded item",
      "category": "individual_litter",
      "mapping_reason": "exact_alias",
      "level": "object",
      "model_score": 0.72,
      "raw_box_xyxy": [
        80.5,
        400.0,
        130.0,
        510.0
      ],
      "box_xyxy": [
        80.5,
        400.0,
        130.0,
        510.0
      ],
      "state": "retained",
      "duplicate_of": null,
      "numeric_diagnostics": []
    },
    {
      "id": "d2",
      "origin": "model_prediction",
      "raw_phrase": "group of individual litter",
      "category": "group_of_individual_litter",
      "mapping_reason": "exact_alias",
      "level": "region",
      "model_score": 0.61,
      "raw_box_xyxy": [
        450.0,
        330.0,
        780.0,
        600.0
      ],
      "box_xyxy": [
        450.0,
        330.0,
        780.0,
        600.0
      ],
      "state": "retained",
      "duplicate_of": null,
      "numeric_diagnostics": []
    },
    {
      "id": "d3",
      "origin": "model_prediction",
      "raw_phrase": "scattered",
      "category": null,
      "mapping_reason": "no_exact_alias",
      "level": "unknown",
      "model_score": 0.4,
      "raw_box_xyxy": [
        900.0,
        300.0,
        960.0,
        420.0
      ],
      "box_xyxy": [
        900.0,
        300.0,
        960.0,
        420.0
      ],
      "state": "retained",
      "duplicate_of": null,
      "numeric_diagnostics": []
    }
  ],
  "review": {
    "flags": [],
    "memberships": [],
    "physical_object_count": null,
    "independent_region_count": null,
    "notes": "Fictional predictions; category and membership are unverified."
  },
  "timing": {
    "scope": "not_run",
    "inference_ms": null
  },
  "error": null,
  "display": {
    "enabled": true,
    "status": "shown",
    "error": null
  }
}
```

### Fictional no-detection example

Also fictional; it demonstrates successful inference with zero above-threshold candidates, not proven absence of trash.

```json
{
  "schema_version": "bantai.functional.v1",
  "run_id": "fictional-run-B",
  "source": {
    "image_id": "data/input/example-empty.jpg",
    "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "width": 1280,
    "height": 720,
    "coordinate_frame": "exif_oriented_original"
  },
  "model": {
    "id": "IDEA-Research/grounding-dino-tiny",
    "revision": "1111111111111111111111111111111111111111",
    "library": "transformers",
    "library_version": "4.51.3"
  },
  "inference": {
    "requested_device": "auto",
    "actual_device": "cpu",
    "dtype": "float32",
    "environment_manifest": "fictional-run-B/environment.json",
    "disable_custom_kernels": true,
    "prompt": "a garbage pile. an individual discarded item. a group of individual litter.",
    "threshold": 0.35,
    "text_threshold": 0.25,
    "mapping_version": "aliases-v1",
    "duplicate_policy": "exact-box-same-phrase-only",
    "fallback": null,
    "preprocessing_manifest": "fictional-run/preprocessing.json"
  },
  "status": "no_detections",
  "score_definition": {
    "range": [
      0,
      1
    ],
    "meaning": "model score; not accuracy"
  },
  "box_format": "xyxy_original_display_pixel_edges",
  "detections": [],
  "review": {
    "flags": [],
    "memberships": [],
    "physical_object_count": null,
    "independent_region_count": null,
    "notes": "No detections above configured thresholds; not proof of no trash."
  },
  "timing": {
    "scope": "not_run",
    "inference_ms": null
  },
  "error": null,
  "display": {
    "enabled": true,
    "status": "shown",
    "error": null
  }
}
```

### Fictional invalid-number diagnostic example

This is a partial candidate record illustrating sanitation, not a standalone per-image result and not a model observation. A nonfinite first raw coordinate and model score have been replaced with null. The other finite raw coordinates remain; the entire display box is null. If this were the only candidate, processing would be `error` with code `invalid_predictions`, and display would be `not_attempted`.

```json
{
  "id": "fictional-invalid-candidate",
  "origin": "model_prediction",
  "raw_phrase": "garbage pile",
  "category": "garbage_pile",
  "mapping_reason": "exact_alias",
  "level": "region",
  "model_score": null,
  "raw_box_xyxy": [null, 40.0, 200.0, 160.0],
  "box_xyxy": null,
  "state": "invalid",
  "duplicate_of": null,
  "numeric_diagnostics": [
    {"field": "raw_box_xyxy", "index": 0, "reason": "nonfinite_coordinate"},
    {"field": "model_score", "reason": "nonfinite_score"}
  ]
}
```

### Fictional display-error lifecycle example

This is a partial final-outcome excerpt, not a standalone result. It shows successful inference preserved after a failed viewer attempt; earlier JSON was saved with display pending. Placeholder error text describes a fictional failure, not one observed here.

```json
{
  "status": "ok",
  "error": null,
  "display": {
    "enabled": true,
    "status": "error",
    "error": {
      "stage": "display",
      "code": "viewer_unavailable",
      "message": "Fictional native viewer failure; inference detections remain saved"
    }
  }
}
```

For a decode error, use processing `status=error`, null dimensions where unavailable, empty detections, processing error stage/code/message, and `display.status=not_attempted` with `display.error=null`. Never invent confidence/timing for failed stages or relabel failure as no detections. A processing error is distinct from the display-only failure above.

## 8. Ordered implementation checklist and acceptance gates

Codex owns these stages. Section 4 and stages 2–3 are now explicitly authorized as **Phase 2A**, with the boundary/acceptance evidence in [PHASES.md](PHASES.md#phase-2a--local-windows-environment-setup-and-readiness-verification). Stage 4 and Section 7 foundation utilities are now authorized as Phase 2B; stages 5–6 and baseline functional checking are now authorized as Phase 2C; the prior 2A/2B-only boundaries are historical. Phase 1 acceptance or Windows readiness is not inferred from the authorization. Routine defaults within 2A need no additional phase approval; report actual evidence and stop after preparation.

| Stage | Action | Concrete acceptance evidence |
| --- | --- | --- |
| 1 | Start a focused branch from current `main`; read instructions; confirm the corrected current guidance and preserve decision history. | Existing changes preserved; reviewed docs state Transformers + unannotated functional checking; no benchmark/annotation requirement introduced; no unrelated or direct-main changes. |
| 2 | Authorized Phase 2A: run Windows preflight; verify official checkpoint metadata/conditions and driver compatibility; finalize environment proposal. | Local Python/driver/memory/package outputs recorded; exact Hub revision/config/license verified; CUDA choice justified or CPU fallback selected; no remote-machine assumptions. Missing facts remain explicit. |
| 3 | Authorized Phase 2A: create isolated environment and lock compatible packages; verify native viewer and CPU/GPU tensor operations. | `pip check` passes; versions/device/build recorded; CPU path works; GPU either works or has a diagnosed fallback; Tk window opens/closes. No model detection claim from these checks. Owner reviews preparation evidence before the inference phase. |
| 4 | Authorized Phase 2B: add minimal configuration/input/results contract handling. | Paths configurable and project-relative; empty/missing folders actionable; unsupported/decode errors distinct; originals unchanged; JSON written atomically; no source photos/cache/results tracked. Non-model checks exercise actual boundary behavior. |
| 5 | Add pinned pretrained Transformers inference, exact mapping and conservative duplicate flags. | Processor/model use same verified revision; no training; grounded postprocessor uses verified API and original pixel sizes; raw phrases and scores preserved; unmapped outputs survive; no automatic clustering or fabricated group counts. |
| 6 | Add native in-memory display, readable summaries and per-image JSON. | Visible boxes/labels/score percentages match JSON; no overall accuracy; windows release memory; default output creates no annotated images; no image overwrites; `ok`, `no_detections`, `error` distinguished independently of display states. Strict JSON sanitizes invalid values to null with diagnostics, invalid candidates are excluded from displayed/valid summaries, and initial inference JSON plus final display outcome are persisted atomically. Fixtures cover orientation/coordinates, numeric nulls, mapping/unmapped phrases, duplicate limits, source preservation, persistence interruption/failure and all display outcomes without claiming model performance. |
| 7 | When team photos become available, run the functional checks and prompt comparisons. | Actual output/visual observations saved; multiple regions, no-detection path and invalid-input behavior checked; coverage gaps marked unrun; object-vs-region ambiguity and baseline limitations reported; timings have device/warm-up scope. No precision/recall/mAP/accuracy claim or automatic Phase 1 acceptance. |
| 8 | Prepare review handoff. | Focused diff, relevant non-model checks and actual functional results reported separately; exclusions respected; unresolved model limitations documented; owner decides acceptance/next phase. Grouping requires a new evidence-based decision and authorization. |

## 9. Remaining blockers and questions

No additional user preference is needed to complete this plan. Proposed defaults resolve routine file layout, versions, folders, display, prompt mapping, and initial thresholds.

Phase 2A setup/readiness is complete: the owner supplied local preflight/package/device/viewer outputs, checkpoint/card/config/processor metadata and the official NVIDIA driver-table excerpt. Earlier cloud source-access gaps were resolved through target-side requests; the exact evidence is in the snapshot. Before later authorized model execution, Codex must retain those pins and evidence rather than inventing model results. Those preparation checks no longer block the next phase; they do not authorize it or measure detection quality. CPU fallback can unblock GPU-specific issues, but it cannot establish GPU acceptance.

Phase 2A technical preparation checks are complete based on supplied target evidence; no setup blocker remains. Only later detector/inference phases require further phase authorization. Category boundaries, exact aliases, prompts/thresholds and overlap defaults remain technical proposals; the installed package versions have only the limited preparation verification in the snapshot; apply the corrected numeric/display requirements consistently. Publication alone does not accept every proposal. Team photos may arrive later; their absence blocks actual functional checking only, not planning or non-model implementation. There are no required annotations, benchmark splits, or unresolved dataset-collection questions.
