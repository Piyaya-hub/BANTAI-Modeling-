# BANTAI Modeling

BANTAI is an academic intelligent-agent project for street-issue detection and hotspot prioritization. Its broader proposal includes roadside garbage, potholes, road-surface damage, incident management, and frequency-based hotspot prioritization.

This repository's initial development scope is **trash detection on still images only**. The approved starting model is **GroundingDINO-T (Swin-T / Tiny configuration)**, evaluated through pretrained text-guided inference. Hugging Face Transformers is the selected inference library. Phase 2A has recorded the exact checkpoint/processor revision, settings, installed dependencies and basic Windows runtime readiness. Prompt/category quality and thresholds remain unvalidated; publication of the plan does not establish their acceptance or readiness. Training, alternative models, and integration require later decisions based on evaluation evidence.

This repository is dedicated exclusively to AI modelling and computer vision: model evaluation/selection, preprocessing, detection, necessary postprocessing, accuracy/performance evaluation, approved optimization or potential fine-tuning, and reusable detection outputs for future integration. Frontend, backend application modules, central databases, incident management, reporting, and hotspot ranking are outside its implementation responsibility.

The three required output categories are **Garbage Pile**, **Individual Litter**, and **Group of Individual Litter**. Images may contain multiple independent regions and mixed categories. These are intended project outputs, not guaranteed pretrained classes. Direct prompting must be evaluated before considering spatial grouping; DBSCAN and other grouping methods are conditional alternatives requiring evidence and explicit authorization. See [scope](docs/SCOPE.md) for definitions and counting requirements.

Execution is intended for the owner’s Windows computer, preferring GPU with CPU fallback. Owner-supplied Command Prompt output now reports Windows 10 build 19045.6466, a GTX 1660 Ti with 6144 MiB VRAM, driver 472.19, and about 5235–5236 MiB free at the time of the query. This replaces the earlier RTX 2080 Super assumption. Git works. After the initial Store-alias/launcher issue, supplied output confirms Python 3.11.9 x64 installation, a successful local clone and `.venv` reporting Python 3.11.9. Supplied checks now show CPU/CUDA tensor results of 2.0, CUDA available, capability (7,5), matching direct library imports and no broken requirements. Native TkAgg rendering/normal return, checkpoint/processor metadata, official driver-reference comparison and exact dependency/source capture are also confirmed. Phase 2A technical preparation checks are complete; subsequent Phase 2C Windows CUDA execution is recorded separately below, with accuracy unmeasured; see the [preflight record](docs/PHASE_2A_READINESS.md). Checkpoint metadata reports Apache-2.0, non-gated access and safetensors at immutable revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`; future model and processor loading must use that same revision.

## Current state

Phase 2C includes pinned pretrained inference and in-memory ANSI/Unicode terminal viewing. The padding fix has direct Windows CUDA evidence; see [Windows run](docs/PHASE_2C_WINDOWS_RUN.md). The owner subsequently authorized replacing TkAgg windows with terminal previews and complete raw query/token diagnostics. See [terminal viewer contract and checks](docs/TERMINAL_VIEWER.md). Preparation remains the default and inference requires `--mode infer`.

The owner supplies unannotated team photographs, manually downloaded/extracted from Google Drive into a configurable project folder (proposed `data/input/`). The planned detection activity is functional checking and visual inspection, with no required annotation work, benchmark splits, precision/recall/mAP, or numeric accuracy targets.

The planned outputs are an in-memory display of original photographs with predicted boxes, category labels, and individual model-score percentages; readable summaries; per-image JSON; and run/environment metadata. Preserve originals and do not automatically save annotated copies. Model-score percentages are not accuracy, calibrated correctness probabilities, or image-area coverage. Inference and display outcomes are recorded separately under the [proposed output contract](docs/PHASE_1_IMPLEMENTATION_PLAN.md#7-proposed-json-contract). Actual Windows CUDA execution is recorded; broader detection quality remains unverified.

## Documentation

- [Agent and collaboration rules](AGENTS.md)
- [Scope and exclusions](docs/SCOPE.md)
- [Proposed model inference architecture](docs/ARCHITECTURE.md)
- [Phases and acceptance criteria](docs/PHASES.md)
- [Approved decisions, proposals, and unresolved choices](docs/DECISIONS.md)
- [Validation and evaluation plan](docs/TEST_PLAN.md)
- [Current state and handoff](docs/HANDOFF.md)
- [Phase 1 implementation plan](docs/PHASE_1_IMPLEMENTATION_PLAN.md)
- [Phase 2A Windows preparation and readiness record](docs/PHASE_2A_READINESS.md)
- [Verified Phase 2A environment snapshot](docs/PHASE_2A_ENVIRONMENT_SNAPSHOT.md)

## Preparation-only usage

Use the existing verified Windows `.venv`; no reinstall is needed:

```bat
.venv\Scripts\python.exe scripts\detect_images.py --help
.venv\Scripts\python.exe scripts\detect_images.py --input data/input --results results
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`--config`, `--input` and `--results` accept absolute paths or paths relative to the repository root, regardless of shell working directory. Configuration is [config/detection.json](config/detection.json). Missing/empty inputs return 2; image read/decode errors return 1 after continuing the batch; unsupported-only input returns 2 with skips recorded; completed preparation returns 0. Configuration/persistence failures return 2. Generated output is JSON only under an ignored run directory. `prepared` means decoded/oriented/RGB-converted in memory, with inference explicitly unrun. No predictions are generated. Default display intent is enabled, but preparation records always say `not_attempted`.

Reusable helpers live in [scripts/](scripts/): `configuration.py`, `inputs.py`, `results.py` and `persistence.py`. Section 7 functional-result construction requires explicit inference completion/provenance. New baseline modules are `inference.py`, `postprocessing.py`, `pipeline.py` and `viewer.py`. Synthetic predictions appear only in tests and are not observed model evidence.

## Authorized inference usage

In the unchanged verified Windows environment, first run the new software checks, then invoke inference on manually supplied unannotated images:

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\detect_images.py --mode infer --input data/input --results results
```

Inference mode may download the pinned processor/config/safetensors into `.cache/huggingface` on first use. `--local-files-only` refuses Hub access and requires all pinned files already cached. `--device cpu` or `--device cuda` overrides the configured request; auto prefers CUDA and records unavailable-device or one OOM CPU fallback. Explicit CUDA failures are errors. Console output is passing-only: retained queries passing the box threshold and ordinary tokens passing the text threshold. Complete excluded/raw diagnostics stay in JSON. Both legacy terminal-mode aliases follow this policy. Use the VS Code integrated terminal and press Enter after images/pages; redirected output records a display error. First test `.venv\Scripts\python.exe scripts\detect_images.py --mode display-check`, then retest a saved result using `--mode replay --result <per-image.json> --image <matching-original>`, without model execution. See [exact commands, source verification and visual limitations](docs/TERMINAL_VIEWER.md). `--no-display` deliberately skips images and still prints passing-only tables. Scores and thresholds use decimal [0,1] units.

Inference results, `run.json`, `environment.json` and `preprocessing.json` are JSON-only outputs under a reserved run directory. Source images are unchanged. Default terminal viewing prints a truecolor half-block preview and results below it, then atomically updates display outcome after stdout flush. Closing/interruption before the final update can leave pending; failed display preserves successful inference. Failures to initialize the runtime/load the model are processing errors with no detections fabricated. Direct package versions must match the 2A baseline; missing/mismatched packages fail before any download.

Exit codes: 0 for completed requested processing/display; 1 for per-image processing or enabled-display errors after batch continuation; 2 for configuration/input-folder/persistence errors or unsupported-only inputs; 130 for interrupted inference runs. `no_detections` only follows successful model/postprocessing with no above-threshold or invalid candidates. Candidate counts are not physical object/independent-region counts, which remain unknown. Scores are not accuracy.

## Dependency reproduction reference

[requirements.txt](requirements.txt) contains direct supporting pins; [requirements-cuda.txt](requirements-cuda.txt) selects the original cu118 source for torch/torchvision. [requirements-windows.lock.txt](requirements-windows.lock.txt) transcribes all 36 observed packages, including bootstrap tools, with the cu118 supplementary index. It is a version snapshot without artifact hashes. No package reinstall or fresh installation was tested in 2B/2C. For a future clean Windows Python 3.11 x64 environment only, the complete snapshot can be applied with `python -m pip install -r requirements-windows.lock.txt`; the original two-source route uses `requirements-cuda.txt` followed by `requirements.txt`. Revalidate affected readiness on a rebuilt environment. Preserve the existing target environment.

## Next step

Review [terminal viewer checks](docs/TERMINAL_VIEWER.md) and [handoff](docs/HANDOFF.md). Cached CUDA full-mode execution succeeded on three local photos; visually inspect the terminal preview and qualitative detection behavior in VS Code. Keep additional scenarios and category-quality judgments separate from software checks. No automatic merge, grouping or later optimization is authorized.
