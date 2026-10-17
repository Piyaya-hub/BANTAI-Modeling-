# BANTAI Phase 1 implementation handoff for Antigravity

Prepared 2026-10-17. Status: proposal for review, not implementation authorization or Phase 1 acceptance. Codex performed repository inspection and source research only. No dependencies, model weights, datasets, or detector code were installed, downloaded, or created.

## 1. Inspected repository and requirement precedence

Repository: `Piyaya-hub/BANTAI-Modeling-`. Fetched `origin/main` is `8baf785` (merge of PR #2); PR #1 was previously merged at `0a9c738`. At initial inspection, the checkout was the prior documentation branch, `docs/finalize-phase-0-detection-scope`; its tracked contents matched latest `main`, and its working tree was clean. The owner subsequently requested publication of this plan to GitHub. It is now recorded on `docs/phase-1-implementation-plan` for review; this does not authorize implementation.

Inspected `AGENTS.md`, `README.md`, `.gitignore`, and `docs/SCOPE.md`, `docs/ARCHITECTURE.md`, `docs/PHASES.md`, `docs/DECISIONS.md`, `docs/TEST_PLAN.md`, `docs/HANDOFF.md`. These provide eight foundation Markdown files and a Python-oriented ignore file. There is no application source, dependency manifest/lock, detector configuration, or image collection in the tracked tree. Existing ignored remote Python state supplies no evidence about the user's Windows computer.

Phase 0 is complete per the owner and merged history. Preserve its modelling-only, still-image, three-category, multi-region, prompt-first decisions. Conflicts with the latest request:

- `docs/TEST_PLAN.md` requires annotations, held-out evaluation, precision/recall/AP and success thresholds; `docs/PHASES.md` Phases 1–2 and `docs/ARCHITECTURE.md` step 10 depend on annotated references. **For the present activity, replace those requirements with a functional detection check on unannotated team photographs.** No required annotations, benchmark split, dataset collection, or accuracy measurement.
- `README.md`, `AGENTS.md`, `docs/PHASES.md`, and `docs/HANDOFF.md` still describe Phase 0-only work or Phase 1 as not authorized. The latest instruction authorizes this planning task only. It does not authorize installation, inference implementation, experiments, or acceptance on the owner's behalf.
- `docs/HANDOFF.md` says the final Phase 0 revision is pending merge; fetched history now confirms PR #2 merged. Update that historical status during a later authorized documentation change.
- `docs/DECISIONS.md` leaves the inference implementation unresolved; the owner now specifies Hugging Face Transformers. Keep earlier decisions as history and record the new requirement rather than silently rewriting it.

This plan identifies these changes without rewriting the accepted scope, architecture, phases, decisions, or test plan. README navigation and the handoff status are updated to point to this planning deliverable. Antigravity should align the affected documents on a new task branch from then-current `main` when implementation is authorized. Preserve `.gitignore` rules; append only the necessary local-data/output/cache exclusions.

## 2. Confirmed scope and evidence

Confirmed by owner: pretrained Grounding DINO Tiny, Hugging Face Transformers, Windows with owner-reported NVIDIA RTX 2080 Super, GPU preferred with CPU fallback; no training/fine-tuning. Detect multiple independent still-image trash regions through direct prompts first. No potholes, road damage, video, cameras, dashboards, APIs, databases, incident consolidation, hotspot ranking, Drive integration, or automatic grouping.

### Official source findings

Source checks below were made during this planning task; no official example or published result is a BANTAI experiment.

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

The proxy returned `403 Forbidden` for Hugging Face, pytorch.org, developer.nvidia.com, and docs.nvidia.com. Their pages were not inspected. These are access limitations in this cloud session, not evidence that the user's computer lacks access.

Before downloading weights later, Antigravity must open the exact [model card](https://huggingface.co/IDEA-Research/grounding-dino-tiny), configuration and files; confirm the Swin-T/Tiny configuration, publisher/conversion provenance, model-specific license/usage conditions and any access terms, availability of safetensors, and immutable Hub commit. Pin processor and weights to the same verified revision. Do not infer checkpoint licensing solely from the original code license. Do not use `trust_remote_code=True` or download an alternative checkpoint silently.

Check the detected Windows driver against NVIDIA's [CUDA 11.8 release notes](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html) and [compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). Confirm RTX 2080 Super architecture against [NVIDIA's GPU table](https://developer.nvidia.com/cuda-legacy-gpus). Turing / compute capability 7.5 is the expected hardware fact, not a locally observed result. Driver suitability and installed memory remain unknown. No numerical minimum driver requirement is asserted as verified here.

CUDA 11.8 is a conservative proposed wheel choice for this GPU, not an assertion about the installed driver. Use prebuilt PyTorch wheels and the Transformers PyTorch path; do not plan a separate CUDA toolkit, compiler, original GroundingDINO source build, or custom extension merely to enable GPU inference. The NVIDIA display driver is still required. Verify wheel/runtime operation locally. The `CUDA Version` in `nvidia-smi` describes driver capability, not proof that the Python wheel uses that CUDA version.

## 3. Minimal proposed repository layout

The following files/folders are proposals; none was created in the repository:

```text
AGENTS.md, README.md, docs/       existing foundation; align conflicting guidance
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

Later append ignore rules `/data/input/`, `/results/`, `/.cache/huggingface/` while preserving existing entries. Check `git status` before committing; `.gitignore` does not untrack photos already committed. Never stage real photos, caches, local absolute paths, or generated results. Optional exported annotated images must also go under ignored results.

## 4. Windows environment verification and proposed setup

These PowerShell commands are **instructions for later execution on the target computer**, not commands run by Codex. Start at the actual repository root; do not reuse the cloud `.venv`.

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

### Proposed version baseline after authorization

Use Python 3.11 x64, `torch==2.7.1` with CUDA 11.8, `torchvision==0.22.1`, `transformers==4.51.3`, `Pillow==11.2.1`, and `matplotlib==3.10.3`. This is a reproducible source-checked baseline, not a claim to be latest or locally tested. Let Transformers resolve its required tokenizers, safetensors, Hub client, and other dependencies within its declared ranges, then freeze the exact verified environment. Review current package advisories before locking; if a required fix changes this baseline, recheck the pinned API and record the reason. No torchaudio or accelerate is required for single-device `.to(device)` inference.

Proposed commands, to run only when implementation/setup is authorized and preflight is satisfactory:

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
5. Retain every above-threshold candidate's raw phrase, original model score, raw pixel box and proposed category mapping. Validate finite scores and boxes; clip displayed boxes to image bounds and record both raw and clipped boxes. Degenerate/nonfinite boxes are flagged, retained diagnostically, and excluded from valid summaries.
6. Apply the conservative duplicate/overlap proposal below; preserve raw candidates and decisions. Do not build clusters or group boxes from member detections.
7. Print a readable per-image list: candidate ID, mapped category or `unmapped`, raw phrase, score as a percentage, and overlap/duplicate warnings. Category tallies count retained candidate predictions, not verified physical objects or independent regions. Do not invent an overall image accuracy or definite region count when memberships are unresolved.
8. Save one JSON result per image/run using a relative path plus content digest to avoid basename collisions; avoid overwriting prior runs. Write atomically. Continue a batch after an individual image error; return a nonzero exit status if errors occurred.
9. Display the oriented original with boxes, category labels, IDs, and model-score percentages via Matplotlib's native TkAgg window. The overlay and image stay in memory; close the figure and release image resources before the next image. No browser, server, or annotated-image temporary file is needed. If the display cannot open, report that fact and retain JSON/console output; do not claim display acceptance.
10. Annotated image export is an optional future `save_annotated=false` setting, disabled by default and written only to an explicitly chosen ignored output path. The default run creates JSON only. No source overwrites, implicit exports, or persistent viewer temp files.

A model score of 0.72 displays as `72.0% model score`. It is not accuracy, a calibrated probability of correctness, or percentage of image area. Preserve the unrounded 0–1 score in JSON.

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
- Ambiguous ownership/discard status, occlusion, scale, group boundaries or pile-vs-group appearance: mark `uncertain` with a short review note rather than forcing a category. No annotation files or boxes are required from the team.

Preserve individual model boxes and directly predicted group/pile boxes as separate candidate records with `level=object` or `level=region`. A group box does not establish member count. Do not synthesize a group box when the model returns only items: record missing group localization as a limitation.

Duplicate proposal: retain all raw candidates; by default collapse only numerically identical boxes with the same normalized phrase into one displayed/summary record, retaining the highest score and duplicate links. Flag substantial same-category overlap (starting review heuristic IoU >=0.8) without deleting candidates automatically. Cross-category overlap and object-within-region containment are review flags, not proof of duplication or membership. Keep conflicting categories visible and mark resolution pending. This avoids silently deleting legitimate overlapping objects or imposing a clustering algorithm.

The UI must distinguish object candidate count from region candidate count. Do not sum a group box and its members as independent regions. With unresolved membership, report independent-region/physical-object counts as unknown. The five-object/two-region example from Phase 0 remains desired behavior, not a baseline guarantee or output to fabricate.

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
| `inference` | Requested/actual device, dtype, package/driver details or manifest reference, custom-kernel setting, exact prompt, `threshold`, `text_threshold`, mapping version, duplicate policy, fallback details |
| `status` | `ok`, `no_detections`, `error`, or `skipped_unsupported` |
| `score_definition` | Model score range `[0,1]`; UI percentage is score ×100, not accuracy |
| `box_format` | `xyxy`, floating original-display pixel edges, origin top-left, x right/y down; raw coordinates retained; clipped valid box obeys `0 <= x0 < x1 <= width`, `0 <= y0 < y1 <= height` |
| `detections` | All above-threshold candidates: ID, origin, raw phrase, mapped category or null, mapping reason, level (`object`, `region`, `unknown`), score (0–1 for model; null for future derived), raw/clipped box, retained/duplicate/invalid state and references |
| `review` | Overlap/containment flags, unresolved membership, optional qualitative notes; not ground truth. Counts of physical objects/independent regions default null |
| `timing` | Scope plus observed milliseconds or null; no fabricated measurements |
| `error` | Null unless failure; otherwise stage/code/message, no secret or credential values |

`no_detections` means inference and postprocessing succeeded with zero candidates above configured thresholds, not that the image contains no trash. Nonempty unmapped results have `status=ok`. On `error`, detections is empty for that failed image and error is populated; never relabel failure as `no_detections`. Unsupported files are a separate skip. A display failure is recorded separately from successful inference and means the display requirement remains unmet.

### Fictional detection example

Every prediction, score, dimension, revision, hash and ID below is illustrative, not an executed experiment. The repeated hex strings are placeholders, not verified checkpoint/source hashes. Implementation must write real metadata and complete configured settings.

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
    "fallback": null
  },
  "status": "ok",
  "score_definition": {"range": [0, 1], "meaning": "model score; not accuracy"},
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
      "raw_box_xyxy": [80.5, 400.0, 130.0, 510.0],
      "box_xyxy": [80.5, 400.0, 130.0, 510.0],
      "state": "retained",
      "duplicate_of": null
    },
    {
      "id": "d2",
      "origin": "model_prediction",
      "raw_phrase": "group of individual litter",
      "category": "group_of_individual_litter",
      "mapping_reason": "exact_alias",
      "level": "region",
      "model_score": 0.61,
      "raw_box_xyxy": [450.0, 330.0, 780.0, 600.0],
      "box_xyxy": [450.0, 330.0, 780.0, 600.0],
      "state": "retained",
      "duplicate_of": null
    },
    {
      "id": "d3",
      "origin": "model_prediction",
      "raw_phrase": "scattered",
      "category": null,
      "mapping_reason": "no_exact_alias",
      "level": "unknown",
      "model_score": 0.40,
      "raw_box_xyxy": [900.0, 300.0, 960.0, 420.0],
      "box_xyxy": [900.0, 300.0, 960.0, 420.0],
      "state": "retained",
      "duplicate_of": null
    }
  ],
  "review": {
    "flags": [],
    "memberships": [],
    "physical_object_count": null,
    "independent_region_count": null,
    "notes": "Fictional predictions; category and membership are unverified."
  },
  "timing": {"scope": "not_run", "inference_ms": null},
  "error": null
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
    "fallback": null
  },
  "status": "no_detections",
  "score_definition": {"range": [0, 1], "meaning": "model score; not accuracy"},
  "box_format": "xyxy_original_display_pixel_edges",
  "detections": [],
  "review": {
    "flags": [],
    "memberships": [],
    "physical_object_count": null,
    "independent_region_count": null,
    "notes": "No detections above configured thresholds; not proof of no trash."
  },
  "timing": {"scope": "not_run", "inference_ms": null},
  "error": null
}
```

For a decode error, use `status=error`, null dimensions where unavailable, empty detections, and an error such as `{ "stage": "decode", "code": "unreadable_image", "message": "Image could not be decoded" }`. Do not invent confidence or timing for failed stages.

## 8. Ordered implementation checklist and acceptance gates

All stages below are future work for Antigravity after plan review and explicit implementation authorization; Phase 1 is not declared accepted here. The foundation's Phase 2 inference gate still requires owner authorization. Stages 4–7 describe that future implementation/check activity, not permission to execute it during preparation.

| Stage | Action | Concrete acceptance evidence |
| --- | --- | --- |
| 1 | Start a focused branch from current `main`; read instructions; align conflicting documentation with this request. | Existing changes preserved; reviewed docs state Transformers + unannotated functional checking; no benchmark/annotation requirement introduced; no unrelated or direct-main changes. |
| 2 | Run Windows preflight; verify official checkpoint metadata/conditions and driver compatibility; finalize environment proposal. | Local Python/driver/memory/package outputs recorded; exact Hub revision/config/license verified; CUDA choice justified or CPU fallback selected; no remote-machine assumptions. Missing facts remain explicit. |
| 3 | After setup authorization, create isolated environment and lock compatible packages; verify native viewer and CPU/GPU tensor operations. | `pip check` passes; versions/device/build recorded; CPU path works; GPU either works or has a diagnosed fallback; Tk window opens/closes. No model detection claim from these checks. Owner reviews preparation evidence before the inference phase. |
| 4 | After inference implementation authorization, add minimal configuration/input/results contract handling. | Paths configurable and project-relative; empty/missing folders actionable; unsupported/decode errors distinct; originals unchanged; JSON written atomically; no source photos/cache/results tracked. Non-model checks exercise actual boundary behavior. |
| 5 | Add pinned pretrained Transformers inference, exact mapping and conservative duplicate flags. | Processor/model use same verified revision; no training; grounded postprocessor uses verified API and original pixel sizes; raw phrases and scores preserved; unmapped outputs survive; no automatic clustering or fabricated group counts. |
| 6 | Add native in-memory display, readable summaries and per-image JSON. | Visible boxes/labels/score percentages match JSON; no overall accuracy; windows release memory; default output creates no annotated images; no image overwrites; `ok`, `no_detections`, `error` distinguished. Result-contract checks cover coordinates, mapping, duplicates and errors using fixtures without claiming model performance. |
| 7 | When team photos become available, run the functional checks and prompt comparisons. | Actual output/visual observations saved; multiple regions, no-detection path and invalid-input behavior checked; coverage gaps marked unrun; object-vs-region ambiguity and baseline limitations reported; timings have device/warm-up scope. No precision/recall/mAP/accuracy claim or automatic Phase 1 acceptance. |
| 8 | Prepare review handoff. | Focused diff, relevant non-model checks and actual functional results reported separately; exclusions respected; unresolved model limitations documented; owner decides acceptance/next phase. Grouping requires a new evidence-based decision and authorization. |

## 9. Remaining blockers and questions

No additional user preference is needed to complete this plan. Proposed defaults resolve routine file layout, versions, folders, display, prompt mapping, and initial thresholds.

Before setup/model execution: Antigravity must obtain local Windows preflight results and verify checkpoint card/revision/conditions plus driver compatibility through the official sources blocked in this session. These are concrete prerequisite checks, not requests for the user to pick technical versions or supply secret values. CPU fallback can unblock GPU-specific issues, but it cannot establish GPU acceptance.

Owner review and explicit implementation/phase authorization remain necessary before the future actions above. The proposed category boundaries, mapping, overlap policy, and JSON contract await that review. Team photos may arrive later; their absence blocks actual functional checking only, not planning or non-model implementation. There are no required annotations, benchmark splits, or unresolved dataset-collection questions.
