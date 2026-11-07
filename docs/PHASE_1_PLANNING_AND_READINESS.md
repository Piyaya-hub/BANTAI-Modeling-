# BANTAI Phase 1: Planning, Environment Readiness & Data Preparation

> **Scope**: Executive concept proposal, Phase 1 implementation planning, initial hardware verification (`GTX 1660 Ti`, Python `3.11.9`), and TACO dataset/split preparation (`D-001` through `D-020`).  
> **Status**: Completed & Published (PR #5, PR #6).

---

## Table of Contents
1. [Executive Overview & Scope Boundaries](#1-executive-overview--scope-boundaries)
2. [Concept Proposal & System Context](#2-concept-proposal--system-context)
3. [Phase 1 Implementation Plan](#3-phase-1-implementation-plan)
4. [Environment Snapshot & Hardware Readiness](#4-environment-snapshot--hardware-readiness)
5. [TACO Dataset Analysis & Initial Preparation](#5-taco-dataset-analysis--initial-preparation)
6. [Phase 1 Milestone Register (D-001 through D-020)](#6-phase-1-milestone-register-d-001-through-d-020)

---

## 1. Executive Overview & Scope Boundaries

The BANTAI project is an academic intelligent-agent research initiative for street-issue detection and hotspot prioritization. The initial development scope of this repository is **trash detection on still images and extracted video frames only**. Roadside garbage, potholes, road-surface damage, incident management, central databases, reporting, and frequency-based hotspot ranking belong to the broader municipal proposal; AI modelling and computer vision are this repository's exclusive implementation responsibility.

### Taxonomy Guardrails Established in Phase 1
- **Active Supervised Taxonomy**: Strictly locked to the 9 active classes (`0: cigarette`, `1: plastic_bottle`, `2: plastic_bag`, `3: plastic_wrapper`, `4: plastic_film`, `5: drink_can`, `6: plastic_cup`, `7: styrofoam_piece`, `8: paper`).
- **Canonical BANTAI Categories**: Mapped to the 5 municipal categories: `Plastic`, `Paper`, `Organic`, `Metal`, and `Potentially Hazardous`.
- **Zero Unsupervised Classes**: Zero support for unaudited classes (`battery`, `aerosol`, `broken_glass`, `syringe`, `food_waste`).
- **Spatial Grouping Regions**: Defined the three core spatial concepts:
  * `individual_litter` -> `"Individual Litter"` ($k = 1$)
  * `group_of_individual_litter` -> `"Group of Individual Litter"` ($2 \le k \le 7$)
  * `garbage_pile` -> `"Garbage Pile"` ($k \ge 8$)

---

## 2. Concept Proposal & System Context

# BANTAI Concept Proposal

**Institution:** Technological Institute of the Philippines — Quezon City  
**Department:** College of Computer Studies, Computer Science Department  
**Course:** CS 403 – Intelligent Agents, First Semester S.Y. 2026–2027  
**Faculty In-Charge:** Mr. Jess N. Garcia  

---

## (1) Proponents and Roles

| Name | Specified Role |
| :--- | :--- |
| **Aguila, Sophia Mae** | Integration & Quality Assurance / Integration Developer |
| **Gumabon, Joshua Philip** | Front-End / UI/UX Developer |
| **Lalamoro, Justine Mark** | Model Trainer / Developer |
| **Santiago, Samuel Rei** | Project Manager / Scripting & Analytics Developer |
| **Soriano, Jhun Deniel** | Full-Stack / Lead Web Developer |

---

## (2) Project Title
**BANTAI: An Intelligent Agent for Street Issue Detection and Hotspot Prioritization**

---

## (3) Background of the Study

Rapid urbanization, population growth, and expanding economic activity bring significant everyday challenges for local government units, particularly in keeping neighborhoods clean and maintaining safe road conditions (Reddy et al., 2025). In busy municipal areas, staying on top of waste management and identifying road degradation requires substantial workforce resources and continuous manual oversight. Even with local environmental ordinances and community sanitation guidelines in place, local personnel frequently run into practical bottlenecks when trying to identify, track, and address localized issues in a timely manner (Gamao & Caelian, 2023).

For many communities, the current monitoring framework relies heavily on manual field patrols, reactive resident complaints, and routine spot checks. To begin with, relying solely on manual inspection makes covering wide road networks difficult, as staff must physically survey areas to spot roadside garbage piles, potholes, and surface damage—a process that is time-consuming, labor-intensive, and prone to human oversight (Maske, 2025). Additionally, incident reports often become fragmented and repetitive, as residents and field staff frequently report the exact same issue through separate channels, which quickly clutters local databases with duplicate entries when no central mechanism exists to merge them.

Beyond reporting inefficiencies, local officials face ongoing challenges when allocating maintenance teams and equipment, as they lack an objective, data-driven way to prioritize issues based on how often or where they are being reported. Furthermore, existing operations rarely connect modern automated detection tools with a central web interface, leaving leaders without clear visual summaries or neighborhood trend data to help plan proactive cleanups and road repairs (Reddy et al., 2025). To help address these practical limitations, this study proposes an intelligent agent-based system designed to automatically detect anomalies through pretrained computer vision models, combine duplicate reports into consolidated records, and dynamically highlight high-priority hotspot locations based on report frequency across target communities.

---

## (4) Project Objectives

The researchers aim to develop an intelligent agent-based system that identifies selected environmental and road infrastructure anomalies and prioritizes reported incidents and hotspot locations within a barangay. Through the creation of a prototype, the system aims to support barangay personnel and officials in planning cleanup, inspection, and maintenance activities through computer vision-assisted detection and frequency-based decision support.

Specifically, the study aims:
1. To implement pretrained computer vision models capable of identifying selected environmental and road infrastructure anomalies, particularly garbage piles, potholes, and road surface damage, through user-submitted photographs and prerecorded video footage simulating CCTV-based anomaly monitoring.
2. To develop a web-based incident reporting and management system that records identified anomalies, their locations, timestamps, supporting images, detection results, and incident statuses, while providing a mechanism for consolidating multiple observations of the same incident to minimize duplicate records.
3. To design a frequency-based intelligent agent that automatically aggregates validated incident reports and observation records to dynamically prioritize individual incidents and identify hotspot locations based on reporting frequency, using report recency as a secondary criterion when rankings are equal.
4. To develop an accessible and user-friendly web dashboard that displays identified incidents, prioritized hotspot locations, incident statuses, and descriptive trends based on anomaly categories, locations, and reporting frequency to support barangay cleanup, inspection, and maintenance decisions.
5. To evaluate the developed system in terms of computer vision detection performance for each supported anomaly category, incident consolidation accuracy, hotspot prioritization consistency, processing time, usability, and basic accessibility.

---

## (5) Expected Output

The project is expected to produce a functional prototype of an intelligent agent-based environmental and road infrastructure anomaly detection and decision-support system. The system will integrate pretrained computer vision models, a web-based incident reporting and management platform, and a frequency-based intelligent agent to identify, organize, and prioritize reported anomalies. The computer vision component will utilize pretrained object detection models, such as RF-DETR Nano and Grounding DINO Tiny, to identify selected anomalies, particularly potholes, road surface damage, and roadside garbage accumulation. Anomaly identification will be demonstrated through two methods: user-submitted photographs and prerecorded video footage simulating CCTV-based monitoring, without requiring continuous live camera integration.

The system will provide a centralized database that stores incident information, including identified anomaly categories, reported locations, timestamps, supporting images, detection results, and incident statuses. Submitted reports and video-based observations will undergo validation and incident consolidation to minimize duplicate records. A frequency-based intelligent agent will automatically evaluate validated incident records and generate dynamic priority rankings based primarily on the number of valid supporting reports, using report recency as a secondary criterion for resolving equal rankings. The agent will also aggregate reporting frequency according to predefined streets or monitoring areas to identify and rank reported-anomaly hotspots. These rankings will be updated as new validated reports are received or existing incident statuses change.

An accessible and user-friendly web dashboard will serve as the primary interface for barangay personnel and officials. It will display identified anomalies, incident details, priority rankings, hotspot locations, and incident statuses, alongside descriptive summaries and visualizations of anomaly categories, reporting frequency, and location-based trends. Through these features, the system is expected to provide organized and data-driven information that supports the planning of cleanup, inspection, and maintenance activities. Priority rankings will represent the frequency of validated reports rather than the physical severity or actual danger of an incident, allowing barangay personnel to retain responsibility for determining appropriate actions.

The completed prototype will undergo evaluation to determine its detection performance across supported anomaly categories, effectiveness of incident consolidation, consistency of frequency-based prioritization and hotspot identification, and processing time. The web interface will also be assessed for usability and basic accessibility features, including readable content, sufficient visual contrast, properly labeled controls, and keyboard navigation. The resulting system will demonstrate the feasibility of combining pretrained computer vision models with a frequency-based intelligent agent to provide practical decision support for localized environmental and road infrastructure monitoring.

---

## References

- Gamao, R. A., & Caelian, M. V. (2023). Implementation and challenges of solid waste management in communities of a component city in the Philippines. *Technium Social Sciences Journal*, 44, 28–40. https://doi.org/10.47577/tssj.v44i1.8897
- Maske, Y. (2025). Urban civic issues image dataset: Potholes and garbage (QR4Change) (Version 2) [Data set]. *Mendeley Data*. https://doi.org/10.17632/zndzygc3p3.2
- Reddy, P. S., Battula, B., Pasha, M. F., Reddy, G. B. S., Bimbasri, P., & Virinchi, E. (2025). YOLO-based system for efficient garbage and pothole detection and reporting. *2025 Third International Conference on Networks, Multimedia and Information Technology (NMITCON)*, IEEE. https://doi.org/10.1109/NMITCON65824.2025.11188066


---

## 3. Phase 1 Implementation Plan

# BANTAI Phase 1 implementation plan — Codex ownership

Current update: Phase 2A and Phase 2B are published (PRs #5–#6). The owner instructed Codex to proceed with **Phase 2C**, superseding the previous stop-before-2C boundary. The pinned inference/mapping/native-display baseline is implemented for review. See [PHASES.md](PHASES.md) and [PHASE_2C_CHECKS.md](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md). Historical task boundaries and fictional examples below remain history, not model evidence. No automatic grouping, training, application modules or merge is authorized.

Originally prepared 2026-10-17; published through merged PR #3. Status: published planning deliverable with detector proposals still unvalidated; current Phase 2A preparation authorization does not accept Phase 1 or authorize detector implementation. The owner now assigns implementation to Codex and reports Antigravity made no changes. Phase 2A is authorized for Section 4 and Section 8 stages 2–3 only: local Windows setup/readiness. Detector implementation, model-weight downloads/loading and inference remain unauthorized. Phase 2A technical preparation has since completed on owner-supplied Windows outputs/screenshots/confirmation, recorded in [PHASE_2A_READINESS.md](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md) and the [exact snapshot](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md). Publication/authorization alone was not its evidence; detector execution remains unauthorized.

## 1. Repository state and requirement precedence

Repository: `Piyaya-hub/BANTAI-Modeling-`. Current Phase 2A inspection fetched `origin/main` at `7a776cf`, the merge of PR #4 containing the contract corrections. PR #3 previously published this plan at `b366f76`. A clean checkout was used to create `docs/phase-2a-environment-preparation`; no direct main edits or merge are authorized. Earlier inspection details remain in Git history and the decision register.

Read `AGENTS.md`, `README.md`, `.gitignore`, and all tracked `docs/` Markdown files. The tracked tree contains foundation/planning/preparation documentation and the ignore file, with the verified dependency snapshot preserved in documentation; no detector, executable dependency manifest/lock, inference configuration, photos/results/model cache is tracked. Ignored remote Python state is not evidence about the owner's Windows computer.

Phase 0 is complete and Phase 1 planning is published. The latest owner requirements select Transformers and **functional visual checks on manually supplied unannotated team photographs**. Older mandatory annotations, held-out benchmarks, formal accuracy metrics, Phase 0-only task state, and pending PR #2/#3 language are superseded; history is retained in [DECISIONS.md](DECISIONS.md). The related scope, architecture, phase gates, test plan, README and agent/handoff guidance are corrected alongside this plan.

Publication is not proof of local environment preparation, successful detection, or owner acceptance of every technical proposal. Phase 2A setup/readiness is now authorized under the agreed boundary; detector implementation and inference remain future work requiring later phase authorization. Routine defaults inside an authorized phase do not require extra approval checkpoints. No annotation work, benchmark split, precision/recall/mAP, or numeric accuracy targets are required for this activity.

## 2. Confirmed scope and evidence

Confirmed by owner: pretrained Grounding DINO Tiny, Hugging Face Transformers, Windows execution, GPU preferred with CPU fallback; the latest owner-supplied preflight identifies GTX 1660 Ti/6144 MiB and driver 472.19, superseding the earlier RTX 2080 Super report; no training/fine-tuning. Detect multiple independent still-image trash regions through direct prompts first. No potholes, road damage, video, cameras, dashboards, APIs, databases, incident consolidation, hotspot ranking, Drive integration, or automatic grouping.

### Official source findings

The findings below preserve official-source checks from the original planning task. Subsequent package/wheel metadata rechecks and target-side Windows readiness evidence are recorded in [PHASE_2A_READINESS.md](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md); see the snapshot for the limited local operations actually verified. No official example or published result is a BANTAI experiment.

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

Phase 2A authorizes these setup/readiness steps on the actual Windows computer once preflight is available. Commands below are not claimed executed until results are recorded in [PHASE_2A_READINESS.md](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md). This session currently has only cloud Linux access. Start at the actual repository root; do not reuse the cloud `.venv`.

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


---

## 4. Environment Snapshot & Hardware Readiness

# Phase 2A verified environment snapshot

## Status and provenance

Phase 2A technical preparation checks are complete on the Windows target based on the owner's supplied command outputs, screenshot and explicit normal-viewer-return confirmation. Codex guided the commands from a separate Linux cloud checkout and recorded the evidence; it did not directly execute Windows commands. This records preparation readiness, not owner acceptance of later implementation, model-memory fit, successful detection or accuracy.

The owner selected a local `Documents/Python/Project Image-object Detection` repository folder under their user profile. Its fresh `.venv` uses Python 3.11.9 x64. No prior user environment was replaced. The initial RTX 2080 Super report was superseded by actual supplied GTX 1660 Ti output; historical decision rows remain unchanged.

## Recorded target and checkpoint metadata

The following values are actual supplied observations, not fictional detection examples. Free VRAM is a past query observation, not guaranteed future availability. The model/config/API/card/preprocessor requests downloaded documentation/configuration only, not model weights. Both future model and processor loads must use the recorded revision.

```json
{
  "evidence_source": "Owner-supplied Windows command outputs, native-plot screenshot and explicit normal-return confirmation; not cloud-executed Windows checks",
  "phase": "2A environment preparation only",
  "os_version": "10.0.19045.6466",
  "git_version": "2.45.1.windows.1",
  "python_version": "3.11.9",
  "python_bits": 64,
  "environment_relative_path": ".venv",
  "device": {
    "name": "NVIDIA GeForce GTX 1660 Ti",
    "driver": "472.19",
    "total_memory_mib": 6144,
    "observed_free_memory_mib": [
      5235,
      5236
    ],
    "compute_capability": [
      7,
      5
    ],
    "cuda_build": "11.8",
    "cuda_available": true,
    "cpu_tensor_result": 2.0,
    "gpu_tensor_result": 2.0
  },
  "dependency_check": "No broken requirements found.",
  "native_viewer": {
    "backend": "TkAgg",
    "synthetic_plot_rendered": true,
    "normal_return_confirmed": true,
    "annotated_image_export": false
  },
  "model": {
    "id": "IDEA-Research/grounding-dino-tiny",
    "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    "license_metadata": "apache-2.0",
    "gated": false,
    "safetensors_files": [
      "model.safetensors"
    ],
    "observed_model_type": "grounding-dino",
    "observed_backbone": {
      "depths": [
        2,
        2,
        6,
        2
      ],
      "model_type": "swin",
      "num_heads": [
        3,
        6,
        12,
        24
      ],
      "out_features": [
        "stage2",
        "stage3",
        "stage4"
      ],
      "out_indices": [
        2,
        3,
        4
      ]
    }
  },
  "processor": {
    "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    "config": {
      "do_normalize": true,
      "do_pad": true,
      "do_rescale": true,
      "do_resize": true,
      "format": "coco_detection",
      "image_mean": [
        0.485,
        0.456,
        0.406
      ],
      "image_processor_type": "GroundingDinoImageProcessor",
      "image_std": [
        0.229,
        0.224,
        0.225
      ],
      "processor_class": "GroundingDinoProcessor",
      "resample": 2,
      "rescale_factor": 0.00392156862745098,
      "size": {
        "longest_edge": 1333,
        "shortest_edge": 800
      }
    }
  },
  "cuda_install_index": "https://download.pytorch.org/whl/cu118",
  "supporting_install_command": "python -m pip install transformers==4.51.3 Pillow==11.2.1 matplotlib==3.10.3",
  "local_artifacts": [
    ".venv/phase2a-requirements.txt",
    ".venv/phase2a-environment.json"
  ],
  "model_weights_downloaded_in_phase2a": false,
  "model_loaded_in_phase2a": false,
  "inference_run_in_phase2a": false
}
```

The processor configuration's `coco_detection` format is an upstream processor setting, not a requirement to annotate team photos. The baseline remains unannotated functional checking in a later authorized phase.

## Official sources and driver interpretation

- [Pinned model card](https://huggingface.co/IDEA-Research/grounding-dino-tiny/blob/a2bb814dd30d776dcf7e30523b00659f4f141c71/README.md): the supplied excerpt describes zero-shot detection and lower-case period-separated prompts; it does not validate BANTAI's categories.
- [Pinned model configuration](https://huggingface.co/IDEA-Research/grounding-dino-tiny/resolve/a2bb814dd30d776dcf7e30523b00659f4f141c71/config.json) and [processor configuration](https://huggingface.co/IDEA-Research/grounding-dino-tiny/resolve/a2bb814dd30d776dcf7e30523b00659f4f141c71/preprocessor_config.json): target HTTP responses were successful; metadata values are recorded above.
- [NVIDIA CUDA 11.8 release notes](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html): supplied target output from Table 2 lists Windows x86_64 driver >=452.39 for CUDA 11.8.x minor-version compatibility. Reported driver 472.19 exceeds that minimum, and actual CPU/CUDA tensor checks passed. The separately listed Windows driver 522.06 is associated with the toolkit distribution; it is not the same table's minor-compatibility minimum.
- [Official cu118 wheel index](https://download.pytorch.org/whl/cu118/torch/): cloud metadata inspection confirmed the Windows CPython 3.11 torch 2.7.1+cu118 wheel; target installation/import/device checks supplied the runtime evidence.

The earlier cloud proxy 403 responses remain historical access observations. Successful target HTTP requests resolved the checkpoint and driver-reference verification gaps; they do not imply that cloud network policy changed. Minor compatibility and a basic tensor test do not guarantee every future model operation or memory requirement. No driver/toolkit update was performed or required for the checks completed here; diagnose any future authorized model-specific failure using actual evidence.

The API reported Apache-2.0 license metadata and non-gated access. Preserve upstream license/notice conditions in later authorized distribution; the supplied model-card excerpt showed no additional application restriction. No weights were obtained during preparation.

## Exact dependency snapshot

The owner ran `pip freeze --all` in the target `.venv`. The complete supplied 36-package output is retained below. `pip check` passed and direct imports reported torch 2.7.1+cu118, torchvision 0.22.1+cu118, Transformers 4.51.3, Pillow 11.2.1 and Matplotlib 3.10.3. Pillow's initial transitive version 12.3.0 was explicitly replaced by the planned 11.2.1 before final verification.

```text
certifi==2026.7.22
charset-normalizer==3.5.2
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
filelock==3.32.3
fonttools==4.67.0
fsspec==2026.7.0
huggingface_hub==0.36.2
idna==3.20
Jinja2==3.1.6
kiwisolver==1.5.1
MarkupSafe==3.0.3
matplotlib==3.10.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.6
packaging==26.3
pillow==11.2.1
pip==24.0
pyparsing==3.3.3
python-dateutil==2.9.0.post0
PyYAML==6.0.3
regex==2026.9.29
requests==2.34.2
safetensors==0.8.0
setuptools==65.5.0
six==1.17.0
sympy==1.14.0
tokenizers==0.21.4
torch==2.7.1+cu118
torchvision==0.22.1+cu118
tqdm==4.70.1
transformers==4.51.3
typing_extensions==4.16.0
urllib3==2.8.0

```

CUDA torch/torchvision were installed with `--index-url https://download.pytorch.org/whl/cu118`; supporting packages used the command recorded above. The target saved `.venv/phase2a-requirements.txt` and `.venv/phase2a-environment.json`, confirmed by command output. The JSON contains Python/checkpoint/processor settings and the CUDA installation source; the full freeze includes bootstrap tools. These local files remain ignored under `.venv`. This committed document preserves their supplied package/configuration values without committing an environment, photographs, caches or weights.

This is a point-in-time Windows environment snapshot, not an artifact-hash lock or a claim that a fresh reinstall was tested. Keep the CUDA install source alongside versions when reproducing; a bare local-version freeze is insufficient to select the cu118 wheel source. Do not rerun passed readiness checks on the unchanged environment without a new failure or concern. If rebuilding or changing packages, revalidate the affected package/device/viewer operations. Python 3.11.9 came from the official python.org x64 installer through WinGet, whose supplied output confirmed installer-hash verification; no TLS/checksum checks were disabled.

## Completed and deliberately unrun work

Passed: interpreter/venv setup, exact direct package imports, dependency consistency, CPU/CUDA tensor execution, official checkpoint/configuration/license/safetensors metadata, same-revision processor settings, native TkAgg rendering/normal return, exact freeze and install-source capture. No current Phase 2A preparation blocker remains in the supplied evidence.

Not run by design: pretrained weight download/loading, detector implementation, inference, team-photo processing, grouping, training/fine-tuning, model-memory-fit tests and detection-quality evaluation. Preparation completion does not authorize them, merge the branch, accept every proposed prompt/threshold/alias rule or approve a later phase. Stop here pending the owner's next phase instruction. See [PHASES.md](PHASES.md), [PHASE_2A_READINESS.md](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md), and [HANDOFF.md](HANDOFF.md).


---

## 5. TACO Dataset Analysis & Initial Preparation

# TACO harmonization and annotation visualization

Stage 1 history: the owner subsequently reported representative annotation review and authorized strict-clean conversion/splitting in D-036. See [Stage 2 commands and evidence](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md). The earlier no-export/no-splits boundary below is superseded for that implementation; fine-tuning remains unauthorized. These audit/viewer utilities remain unchanged.

The owner authorized this preparation stage on 2026-10-18 (D-035): audit the existing external COCO dataset and visualize its annotations under the eight approved object labels. It creates review artifacts, not training examples or splits. The current Grounding DINO model, detection configuration, inference logic, three inference categories and results viewer are independent and unchanged.

## Files and requirements

| File | Purpose |
| --- | --- |
| [config/taco_taxonomy.v1.json](../config/taco_taxonomy.v1.json) | Version 1.0.0 mapping of 11 exact TACO category names to eight BANTAI labels. Original IDs/names are retained; label IDs 0-7 identify this preparation taxonomy, not a finalized trainer encoding. |
| [scripts/taco_dataset.py](../scripts/taco_dataset.py) | Read-only COCO validation, image hashes/readability/dimension checks, taxonomy lookup, report writing/reloading. |
| [scripts/audit_taco.py](../scripts/audit_taco.py) | Audit CLI with JSON/CSV review output and new-directory protection. |
| [scripts/view_taco.py](../scripts/view_taco.py) | Separate local Tk/TkAgg annotation viewer with deterministic per-class sampling, class selector, Next/Previous and zoom/pan. |
| [tests/test_taco.py](../tests/test_taco.py) | Synthetic mapping, COCO, malformed-box, missing/corrupt image, identity, EXIF and drawing checks. |
| [tests/windows_taco_check.py](../tests/windows_taco_check.py) | Optional native checks against a real local audit; auto-closes and checks source hashes. |
| [.gitignore](../.gitignore) | Adds `/reports/taco/`, `/data/taco/`, `/BANTAI-Datasets/` for reports and accidental in-repository dataset copies. The actual sibling dataset already lies outside Git. |

Use the existing verified Windows Python 3.11 `.venv`. The audit uses Python's standard library and already-installed Pillow; the viewer additionally uses already-installed Matplotlib and Python's Tk. No pycocotools, pandas, dataset downloader, model import, GPU, new dependency or package reinstall is needed. Existing dependency reproduction references remain in [README.md](../README.md); an installation is not performed by these tools.

Default dataset: `../BANTAI-Datasets/TACO/`; default source: its `annotations.json`. Relative paths resolve from the repository root even when invoked from another shell directory. `--annotations` overrides a path relative to the repository root; it is not relative to the dataset folder. All source access is read only.

## Audit commands and artifacts

```powershell
.venv/Scripts/python.exe scripts/audit_taco.py
.venv/Scripts/python.exe scripts/audit_taco.py --dataset-root ../BANTAI-Datasets/TACO --output reports/taco/my-review
```

The default creates a new UUID folder under ignored `reports/taco/`. Explicit output directories must be new and cannot overlap the external dataset. Existing reports are never overwritten. Exit 0 means no detected image/annotation/structural issues; **exit 1 means the audit completed and wrote reports with issues**, including known missing images; exit 2 means an input/configuration/write failure. A partial report after write failure must not be treated as a completed audit.

Each report folder contains:

| Artifact | Contents |
| --- | --- |
| `audit.json` | Concise counts, per-label object/unique-image counts, all original class counts and exclusion handling, structural issues, source/taxonomy hashes. |
| `class_counts.csv` | Eight proposed classes: totals, annotations attached to readable images, conservative usable objects, unique usable images and blocked/invalid counts. |
| `original_classes.csv` | All source categories, source IDs/names, proposed label or explicit exclusion state, and usable counts. |
| `image_issues.csv` | Missing, unsafe, unreadable and dimension-mismatched image records. |
| `annotation_issues.csv` | Unusable annotation indexes/IDs, validation and image-availability/alignment issues. |
| `images.jsonl` | Every original COCO image record plus path, image hash, status and dimensions/EXIF diagnostics. |
| `annotation_review.jsonl` | Every original annotation, unmodified metadata/segmentation/area/iscrowd/custom fields, original category identity, proposed label and explicit handling/validity state. Source list index disambiguates duplicate annotation IDs. |
| `taxonomy_snapshot.json` | Exact taxonomy values used for the audit. |
| `source_metadata.json` | Other original COCO top-level metadata, all original categories, licenses and scene annotations. |

These are review ledgers, not a filtered COCO training export. Excluded annotations remain in the ledgers and the viewer; absence of a mapped label never makes an object an implicit negative/background example. Image counts per class overlap and must not be summed into a dataset image total. Object counts mean annotated instances, not a claim that all real objects are annotated.

Validation checks unique integer IDs, category/image references, safe local image paths, successful full image decoding, declared versus actual dimensions, and finite COCO `[x,y,width,height]` boxes with positive sizes, nonnegative origins and endpoints within declared dimensions. Boxes are never silently clipped. Duplicate IDs are held as ambiguous rather than selecting one record. A missing image blocks usability but does not by itself make its annotation geometrically invalid. Nonstandard JSON constants and duplicate JSON keys fail source loading. Segmentation, area and crowd metadata are preserved; this stage does not validate mask geometry or establish crowd/loss policy.

## Viewer commands

Reuse a completed audit to avoid decoding the entire dataset on every launch:

```powershell
.venv/Scripts/python.exe scripts/view_taco.py --report reports/taco/review-v1 --sample-per-class 1
.venv/Scripts/python.exe scripts/view_taco.py --report reports/taco/review-v1 --classes cigarette plastic_bottle plastic_bag --sample-per-class 3 --seed 42
.venv/Scripts/python.exe scripts/view_taco.py --report reports/taco/review-v1 --classes excluded unlabeled --sample-per-class 3
.venv/Scripts/python.exe scripts/view_taco.py --report reports/taco/review-v1 --image-id 274
.venv/Scripts/python.exe scripts/view_taco.py --report reports/taco/review-v1 --classes plastic_film --sample-per-class 2 --list
```

`--report` uses its recorded dataset/taxonomy sources, verifies their hashes and checks each displayed image's hash before decoding. Do not combine it with source overrides. Without `--report`, the viewer performs a fresh in-memory audit using the same source arguments as the audit CLI. It does not write a report or any image. Repeat `--image-id` to inspect several specific readable/aligned images, including images without annotations. Sampling is deterministic per selected class and seed, with shared images deduplicated; it provides review examples, not statistically representative sampling or a train/test split.

Class selection controls which images are sampled; **all annotations on each selected image remain visible**. Labels show annotation ID, original TACO name and proposed BANTAI label/explicit exclusion state. Green solid boxes indicate selected mapped classes, blue other mapped classes, orange dashed excluded classes, purple dotted `Unlabeled litter`, red invalid annotation metadata when its geometry is drawable. Invalid boxes stay listed in the selectable, horizontally/vertically scrollable details table and are not drawn or clipped. Color is supplemented by line style and text. Crowded labels can overlap; use zoom/pan and the complete table. The toolbar omits Save; no annotated copies or temporary image files are generated. Close the window to release the figure/canvas and image resources.

## Coordinate orientation requires review

The default uses the **stored raster without EXIF transpose**, matching only records whose declared dimensions agree with that raster. This deliberately avoids importing the inference pipeline's automatic orientation rules into COCO annotation handling. Audit counts called `usable` use this conservative policy and require a valid annotation, readable image and no image-dimension issue.

The downloaded set contains 152 dimension mismatches. All 152 declared dimensions match an EXIF-oriented preview. To inspect a specific example without changing any source bytes or annotation coordinates:

```powershell
.venv/Scripts/python.exe scripts/view_taco.py --report reports/taco/review-v1 --image-id 228 --frame exif
```

EXIF mode applies Pillow's orientation only in memory and requires the resulting dimensions to match COCO metadata. Annotation coordinates are not transformed; this is a review of the hypothesis that the annotations already describe that orientation. A dimension match, including a 180-degree rotation with unchanged dimensions, does not prove every box is aligned. The report counts remain unchanged by this viewer option. Approve and validate a coordinate policy before converting data for training.

## Actual local audit — 2026-10-18

Base main `5351d57`, branch `feat/taco-harmonization`. Command:

```powershell
.venv/Scripts/python.exe scripts/audit_taco.py --output reports/taco/review-v1
```

The command exited **1**, correctly indicating a completed audit with issues. Original annotation SHA-256: `ac1ec605c9e1fcda767970de3457c69b4a43b62c02aca8c4c80f51d1b5ae8449`. Taxonomy version: 1.0.0. Exact original metadata and image hashes are in the ignored report.

- 1,500 source image records; **1,248 readable, 252 missing, zero unreadable/corrupted**. This reproduces the supplied image-availability observations.
- 4,784 source annotations across 60 source classes. **3,938 annotations are attached to readable images**, reproducing the supplied total at the availability level.
- A stricter check holds 520 otherwise-valid annotations on dimension-mismatched readable images and nine invalid annotations on readable images, leaving **3,409 usable annotations across all classes** under the stored-raster policy. Nothing is discarded from the review ledgers.
- Eleven invalid annotations in the entire source: both occurrences of duplicated IDs 309 and 4040 (four records), plus seven negative-origin boxes. Two of these eleven are attached to missing images. No unknown category or image references were observed in this source. All 11 required original category names are present.
- **1,653 usable mapped annotations in 735 unique images**; 1,756 usable excluded annotations. There are 49 original categories outside the mapping. **413 mapped-class images also contain usable excluded annotations**, making implicit-background handling unsafe without a defined policy.
- `Unlabeled litter` remains explicit: 517 source annotations, of which 362 are conservatively usable in 170 images.

| Label | Source annotation total | Attached to readable images | Conservative usable objects | Unique usable images |
| --- | --- | --- | --- | --- |
| `cigarette` | 667 | 539 | 478 | 155 |
| `plastic_bottle` | 335 | 275 | 246 | 185 |
| `plastic_bag` | 92 | 74 | 62 | 49 |
| `plastic_wrapper` | 299 | 261 | 192 | 151 |
| `plastic_film` | 451 | 381 | 344 | 231 |
| `drink_can` | 229 | 186 | 164 | 110 |
| `plastic_cup` | 104 | 93 | 84 | 71 |
| `styrofoam_piece` | 112 | 91 | 83 | 59 |

Negative origins are small but still explicit: annotation IDs 371, 387, 707, 859, 1621, 1777 and 2186. Whether clipping is appropriate requires a documented later decision. Duplicate IDs are preserved with source indexes rather than silently renumbered.

## Verification and remaining decisions

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -q
.venv/Scripts/python.exe tests/windows_taco_check.py --report reports/taco/review-v1
.venv/Scripts/python.exe tests/windows_taco_check.py --report reports/taco/review-v1 --frame exif --image-id 228
```

Actual existing Windows `.venv` results: **101 tests, 100 passed, one existing Windows symlink-privilege skip**, 11.552 seconds on the first full run and 6.112 seconds on the subsequent check. Nine new synthetic tests cover all eleven mappings, preserved COCO metadata, missing/corrupt/unsafe files, malformed boxes/references, duplicate IDs, exact xywh drawing, excluded/unlabeled styles, deterministic selection, report reuse, source identity and explicit EXIF preview. These fixtures do not establish annotation quality. CLI help, compilation and whitespace checks passed; 159 relative Markdown links and five strict JSON examples validated. Protected inference scripts and dependency files match the starting main; the owner's detection configuration remains uncommitted.

Native Tk check passed over ten downloaded source images, exercising one-per-class sampling, Next navigation, class selection, excluded/unlabeled boxes, selectable details, exact rectangle geometry, zoom/Home, figure cleanup and normal closure. Explicit EXIF-mode native check also passed over three source images. No model modules or network connections occurred. Source hashes remained unchanged. An independent final hash pass verified **all 1,248 audited source images unchanged**, and the original annotation hash matched.

Agent in-memory visual inspection: image 18 shows a can with `Drink can -> drink_can`; image 274 shows `Plastic film -> plastic_film` alongside an explicit `Unlabeled litter` box; image 228's EXIF preview places the excluded `Plastic glooves` box on the visible glove. The original spelling is preserved. These limited visual observations are separate from human acceptance, which remains pending. They do not certify the full dataset's box/category quality.

Before the next stage, the owner must decide how to handle excluded and unlabeled annotations in mixed images, approve an orientation/coordinate policy, resolve duplicate IDs and negative boxes, and assess sampled annotation quality/class imbalance. Later split design, source/license review for intended use, model loss/prompt encoding and actual fine-tuning need their own preparation/authorization. No final train/validation/test split, source correction, training export, download or fine-tuning occurred here.


---

## 6. Phase 1 Milestone Register (D-001 through D-020)

- **D-001**: Model direction established: Grounding DINO-T (Swin-T configuration) through Hugging Face Transformers.
- **D-002**: Direct text-guided zero-shot evaluation approved before considering grouping algorithms.
- **D-003**: Hardware target identified: NVIDIA GeForce GTX 1660 Ti, 6144 MiB VRAM, driver 472.19 on Windows 10/11 x64.
- **D-004**: Virtual environment `.venv` configured with Python 3.11.9 x64.
- **D-005**: Strict separation of concerns: Model A (trash detection) separated from web backend and Model B.
- **D-006**: Bounding box geometry formalized in standard COCO format `[x_min, y_min, width, height]`.
- **D-007**: Coordinate frame fixed to stored raster dimensions without EXIF transpose.
- **D-008**: Numeric contract: invalid numeric values (`NaN`, `Inf`) serialized as null with diagnostics.
- **D-009**: Processing status decoupled from display status (`display.status` vs `inference.status`).
- **D-010**: In-memory visualization chosen; automatic export of annotated images disabled to protect originals.
- **D-011**: Zero-test-split access rule established: `test.json` strictly held out.
- **D-012**: TACO dataset analysis: 1,500 images categorized, background non-trash images audited.
- **D-013**: Supervised class candidates mapped to canonical BANTAI categories.
- **D-014**: Bounded Tk/TkAgg viewer architecture proposed for visual verification.
- **D-015**: Exact phrase mapping contract formalized; unmapped phrases retain null category ID.
- **D-016**: Multi-image replay specification designed with deterministic ordering.
- **D-017**: Model score percentage defined strictly as confidence indicator, not ground-truth accuracy.
- **D-018**: Duplicate detection flags established without destructive NMS.
- **D-019**: Initial Hungarian matcher loss routing investigated.
- **D-020**: Phase 1 handoff completed; Phase 2 authorized.
