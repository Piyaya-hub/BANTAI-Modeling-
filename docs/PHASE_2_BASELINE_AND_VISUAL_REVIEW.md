# BANTAI Phase 2: Pinned Baseline Inference & Visual Review System

> **Scope**: Pinned Grounding DINO Swin-T baseline inference, native Tk/TkAgg in-memory scrollable review viewer, prompt/threshold comparison, and functional visual checking of owner-supplied street images (`D-021` through `D-034`).  
> **Status**: Completed & Verified.

---

## Table of Contents
1. [Executive Overview & Baseline Strategy](#1-executive-overview--baseline-strategy)
2. [Phase 2A: Environment & Pinned Dependency Verification](#2-phase-2a-environment--pinned-dependency-verification)
3. [Phase 2B: Foundations & Contract Checks (PR #6 Integration)](#3-phase-2b-foundations--contract-checks-pr-6-integration)
4. [Phase 2C: Pinned Baseline Inference & Windows Execution](#4-phase-2c-pinned-baseline-inference--windows-execution)
5. [Phase 2D: Multi-Image Replay & Native Viewer Lifecycle](#5-phase-2d-multi-image-replay--native-viewer-lifecycle)
6. [Phase 2E: Visual Review & Functional Handoff](#6-phase-2e-visual-review--functional-handoff)
7. [Scrollable GUI Viewer Specification](#7-scrollable-gui-viewer-specification)
8. [Phase 2 Milestone Register (D-021 through D-034)](#8-phase-2-milestone-register-d-021-through-d-034)

---

## 1. Executive Overview & Baseline Strategy

Phase 2 established the first runnable end-to-end computer vision pipeline using the pinned pretrained `IDEA-Research/grounding-dino-tiny` model without fine-tuning. The primary objectives were:
- Verifying local Windows runtime and GPU compatibility (GTX 1660 Ti, 6144 MiB VRAM).
- Implementing exact phrase mapping and numeric output contracts.
- Developing a non-destructive in-memory native viewer (`TkAgg`) to review model inferences alongside detection confidence tables.
- Conducting qualitative visual checks across owner-supplied unannotated test photographs.

---

## 2. Phase 2A: Environment & Pinned Dependency Verification

# Phase 2A — Windows environment preparation and readiness record

## Authorization and actual status

The owner authorizes Codex to perform **local Windows environment setup/readiness only**, corresponding to [PHASE_1_IMPLEMENTATION_PLAN.md](PHASE_1_PLANNING_AND_READINESS.md) Section 4 and Section 8 stages 2–3. Codex replaces Antigravity; the owner states Antigravity made no changes. See the [phase definition](PHASES.md#phase-2a--local-windows-environment-setup-and-readiness-verification) and D-022/D-023 in [DECISIONS.md](DECISIONS.md).

**Phase 2A technical preparation checks are complete based on supplied Windows evidence.** Python/venv, dependencies/imports, CPU/CUDA execution, native rendering/normal return, checkpoint/processor metadata, official driver reference, and exact environment capture are verified to the limited preparation scope. No detector or inference result is implied. The owner retains acceptance/next-phase authority; no further implementation is started.

Codex can execute only in a separate Linux cloud checkout. Target results below came from owner-run Windows commands, a supplied screenshot and explicit viewer confirmation, not direct cloud Windows execution. No cloud ML installation was used as a substitute. Local `.venv` evidence files were saved, and the supplied values are committed in the [environment snapshot](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md). No weights, photographs or inference were needed.

## Evidence ledger

| Check | Actual outcome in this task | Assessment / limitation |
| --- | --- | --- |
| Repository baseline | Fetched latest `main`: `7a776cf`, merge of PR #4; started from a clean checkout | Preserve local Windows checkout/user changes when inspecting it |
| Windows/Git/Python | Supplied CMD output: Windows 10 version 10.0.19045.6466; Git 2.45.1.windows.1; WinGet installed Python 3.11.9 x64 with verified hash; new-terminal Python probe returned 3.11.9/AMD64/64 bits and real Python311 executable | Use the verified interpreter and target venv; do not confuse earlier Store alias with the installed interpreter |
| GPU/driver/memory | Supplied successful `nvidia-smi` query: NVIDIA GeForce GTX 1660 Ti; driver 472.19; 6144 MiB total, 5235–5236 MiB free at query time | Actual supplied CUDA tensor test passed on this driver/build; model-memory fit remains untested |
| Proposed package metadata | Rechecked official PyPI: Transformers 4.51.3 requires Python >=3.9, torch extra >=2.0; torchvision 0.22.1 requires torch 2.7.1 | Imports and `pip check` passed; full 36-package freeze captured in local files and committed snapshot |
| Proposed CUDA wheel | Official cu118 index lists torch 2.7.1 CPython 3.11 Windows x64 wheel | Supplied target reports build 11.8, CUDA available, capability (7,5), successful GPU tensor |
| Checkpoint/configuration/license/revision/safetensors | Supplied official target HTTP output: IDEA-Research/grounding-dino-tiny; revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`; Apache-2.0 metadata; gated False; `model.safetensors` listed; model type grounding-dino; Swin depths [2,2,6,2] | Pinned processor config read at the same revision; no weights downloaded or model loaded |
| NVIDIA driver reference | Target successfully read official CUDA 11.8 notes: Table 2 minimum Windows driver >=452.39; toolkit distribution separately lists driver 522.06 | Reported 472.19 meets minor-compatibility minimum; successful CUDA tensor supports basic runtime readiness, not all model operations |
| Local checkout/environment | Supplied clone succeeded at the owner-selected Documents/Python/Project Image-object Detection folder on this preparation branch; `.venv` creation/verification returned Python 3.11.9 | Full freeze and environment JSON saved inside ignored `.venv`; complete supplied dependency/config values preserved in snapshot |
| CPU/GPU probes | Supplied results: torch 2.7.1+cu118; CUDA build 11.8; CPU result 2.0; CUDA available True; GTX 1660 Ti; capability (7,5); GPU result 2.0 | Passed basic CPU/CUDA execution only; no model-memory/performance inference |
| Native Tk/Matplotlib viewer | Supplied screenshot shows expected native synthetic diagonal-line plot with the Phase 2A title | Owner explicitly confirmed `TkAgg viewer closed normally`; rendering and normal return passed, no image export invoked |

Official references rechecked for metadata only: [Transformers metadata](https://pypi.org/pypi/transformers/4.51.3/json), [torchvision metadata](https://pypi.org/pypi/torchvision/0.22.1/json), [CUDA 11.8 wheel listing](https://download.pytorch.org/whl/cu118/torch/). Metadata verification does not establish Windows/GPU compatibility or acceptance of all proposed pins. Earlier source findings and limits remain in the plan; they are not upgraded to local evidence.

## Pinned checkpoint and supplied model-card evidence

The target successfully queried the official Hugging Face API and `config.json` at revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. It reports model `IDEA-Research/grounding-dino-tiny`, license metadata `apache-2.0`, `gated=False`, and a `model.safetensors` file. The configuration reports `model_type=grounding-dino` and a Swin backbone with depths [2,2,6,2], attention heads [3,6,12,24], and stage2/stage3/stage4 outputs, consistent with the selected Tiny direction.

The supplied model-card excerpt describes zero-shot detection without labelled data, links the original Grounding DINO paper and shows lower-case period-separated prompts. This documentation example was printed, not executed. It does not establish custom trash-category support or accuracy. License metadata identifies Apache-2.0; preserve upstream notices/conditions in any later authorized distribution. The shown excerpt adds no special application restriction, and no private/gated access was reported.

Later authorized loading must pin both model and processor to this same immutable revision; weight loading remains outside Phase 2A. Processor metadata was subsequently read at the same revision and saved with the full exact freeze and installation source; see the snapshot. The prior cloud Hugging Face 403 is no longer a target checkpoint-access blocker: target-side HTTP metadata succeeded.

## Supplied Command Prompt evidence and completed interpreter setup

The supplied transcript uses **Command Prompt (CMD)**, not PowerShell. Its successful GPU query identifies GTX 1660 Ti, replacing the earlier RTX 2080 Super assumption. The first failed GPU command accidentally joined `--format=csv` to `git --version`; the later correctly separated queries succeeded. This is a paste error, not a GPU failure. Git availability is confirmed by supplied output; `py` availability failed, but Python installation is still unknown. A subsequent combined line (`where python, python --version, ...`) ran only `where` lookups, not the intended version commands. It exposed a `WindowsApps` Python path but did not verify an interpreter; that path may be a Store alias. The follow-up `python --version` reported no Python interpreter and a Microsoft Store shortcut. WinGet then successfully installed Python 3.11.9 x64 from python.org and verified its installer hash. A new terminal confirmed a real Python311 executable, version 3.11.9 and 64 bits; a successful clone followed, then a new local `.venv` returned Python 3.11.9. These initial discovery checks are now historical, not blockers. No personal filesystem path from the transcript is needed in Git.

Initial discovery commands (historical; not needed again after the successful interpreter probe):

```bat
where python
python --version
where python3
winget --version
```

If `python --version` launches Microsoft Store, close it and record that an app-execution alias was encountered; do not treat it as an installed interpreter. If Python is found, inspect its version/architecture before creating a venv. `winget` availability informs the installation route if Python must be installed; no installation command is issued until discovery is resolved. This historical version-only uncertainty was subsequently narrowed by a successful supplied CUDA tensor probe on driver 472.19/cu118; official driver-reference access from the cloud remains blocked, and no model compatibility is inferred. No CUDA toolkit or model download is needed for discovery.

## 1. Windows preflight and local repository copy — completed; reproduction reference

A **checkout** means the local folder containing a cloned GitHub repository. The cloud copy used in this chat is not on the owner's PC. A local clone is now confirmed by supplied output. The owner selected `%USERPROFILE%\Documents\Python\Project Image-object Detection`, and `.venv` in that folder reports Python 3.11.9. The cloud repository is a separate checkout. Do not clone again or recreate the environment.

The initial Git/GPU checks already have supplied results. Continue with the CMD interpreter-discovery block above. For the fuller setup blocks below, open PowerShell explicitly (the current CMD prompt does not understand PowerShell variable or `Get-*` syntax). Use `py` commands only if the launcher is installed; otherwise substitute the verified Python executable. These hardware/tool checks do not require a repository folder. If Git is absent, install Git for Windows from its official distribution before cloning; diagnose missing Python/GPU tools separately. Do not request a token merely because a command has not been run.

For a future fresh machine only, if there is no local copy and Git works, create one without overwriting a directory. This has already succeeded on the current target; do not rerun it there. After this preparation branch is pushed, it can be cloned directly:

```powershell
$BantaiCheckout = Join-Path $env:USERPROFILE 'Documents\Python\Project Image-object Detection'
if (Test-Path $BantaiCheckout) { throw 'A folder already exists here; inspect it instead of overwriting or duplicating it.' }
git clone --branch docs/phase-2a-environment-preparation https://github.com/Piyaya-hub/BANTAI-Modeling-.git $BantaiCheckout
if ($LASTEXITCODE -ne 0) { throw 'Clone failed; preserve the output and diagnose Git/network/authentication access.' }
Set-Location $BantaiCheckout
```

A clone downloads repository documentation, not pretrained weights or datasets. If a local copy already exists, locate and inspect it instead; do not reset/switch over user changes blindly. This cloud chat has no direct access to that folder. User-run outputs or a Codex session connected to the Windows PC are needed for actual local setup evidence.

### Read-only preflight

These are PowerShell commands for the owner’s Windows computer. Do not dump environment variables or credential files. They do not download weights or run inference.

```powershell
Get-Location
git status --short
git branch --show-current
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture
py -0p
py -3.11 --version
py -3.11 -c "import sys,struct; print(sys.executable); print('Python:',sys.version); print('bits:',struct.calcsize('P')*8)"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
Test-Path .\.venv
```

If `.venv` exists, inspect without overwriting it:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip show torch torchvision transformers pillow matplotlib
```

If Python 3.11 x64 or `nvidia-smi` is absent, record the actual failure and diagnose it. Do not replace global Python or a driver blindly. A missing GPU tool does not prove the GPU is absent; CPU preparation can proceed with its GPU limitation recorded. Local outputs may be supplied back to Codex in this chat or verified by Codex operating in the Windows checkout. Cloud tools cannot perform these local actions.

## 2. Metadata and wheel choice — completed; reproduction reference

Inspect the [official model card](https://huggingface.co/IDEA-Research/grounding-dino-tiny), [model metadata API](https://huggingface.co/api/models/IDEA-Research/grounding-dino-tiny), and repository configuration/file listing. Confirm Tiny/Swin-T configuration, source/provenance, license and conditions, safetensors availability and exact immutable commit. Record these facts in this ledger or a referenced non-secret local record. Do not follow weight-file links, call model/processor `from_pretrained`, instantiate a detector, or download weights. Later model/processor loading must use the same recorded revision.

Check the real driver against [CUDA 11.8 release notes](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html) and [NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). The proposed cu118 wheel is a candidate until the target facts and device probes support it; do not treat `nvidia-smi`'s CUDA version as the wheel runtime. Do not install a separate CUDA toolkit or compile custom kernels merely to use prebuilt PyTorch. Do not download a detector to diagnose package readiness.

Keep the existing proposed baseline unless a diagnosed compatibility issue or applicable package advisory requires a documented change: Python 3.11 x64, torch 2.7.1/cu118, torchvision 0.22.1, Transformers 4.51.3, Pillow 11.2.1, Matplotlib 3.10.3. These direct versions were subsequently installed/imported and exercised to the limited Phase 2A scope on the supplied target; fresh environments still need their own affected checks. Recheck affected APIs if any version changes; no silent upgrade or altered checksum/TLS verification.

## Completed package/device checks and next steps

Owner-supplied installation/import results:

| Library | Observed target version |
| --- | --- |
| Python | 3.11.9 x64 |
| torch | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| Transformers | 4.51.3 |
| Pillow | 11.2.1 |
| Matplotlib | 3.10.3 |

The CUDA torch/torchvision install command used `https://download.pytorch.org/whl/cu118`. The supporting-package command pinned Transformers/Pillow/Matplotlib as above; Pillow 12.3.0 (initially resolved by torchvision) was replaced by the planned 11.2.1, and the final import check confirms 11.2.1. Installation logs and imports are supplied by the owner, not directly executed by cloud Codex. `pip check` returned `No broken requirements found.`

The CPU tensor produced 2.0. CUDA was available, identified GTX 1660 Ti/capability (7,5), and its GPU tensor also produced 2.0 without a reported error. No driver update is justified by an imagined failed check: this tested package/device route works on the reported driver 472.19. Model-memory fit, checkpoint loading and inference are deliberately untested. Do not repeat passing probes without a change/failure/concern.

Completed: the owner confirmed normal viewer return, saved the exact freeze/environment manifest and supplied pinned processor settings. Target NVIDIA documentation output verified the driver-table distinction. No preparation check remains pending; do not reinstall or repeat passing checks on the unchanged target. No detector implementation or inference is authorized.

## 3. Authorized isolated setup — reproduction only

The current target already has a fresh verified `.venv`; preserve and use it. The alternative PowerShell environment-creation example below is for future reproduction only, not the next action on this machine. Reuse any pre-existing environment only after inspection establishes it is suitable. Otherwise preserve it and create a new environment under existing ignored `venv/`, using a path that does not already exist. Example below deliberately refuses to overwrite that path. It creates environment/dependency files only, no detector code or photo directories.

```powershell
$BantaiEnvironment = Join-Path (Get-Location) 'venv\phase2a'
if (Test-Path $BantaiEnvironment) { throw 'Environment already exists; inspect it before reuse or choose a fresh ignored venv path.' }
py -3.11 -m venv $BantaiEnvironment
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
$BantaiPython = Join-Path $BantaiEnvironment 'Scripts\python.exe'
& $BantaiPython -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed; inspect the error before retrying.' }
& $BantaiPython -m pip install transformers==4.51.3 Pillow==11.2.1 matplotlib==3.10.3
if ($LASTEXITCODE -ne 0) { throw 'Supporting package installation failed; inspect the error.' }
& $BantaiPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
```

Use the CUDA index only after driver/build selection is justified. If CPU-only installation is the diagnosed choice, use the same proposed torch/torchvision versions from `https://download.pytorch.org/whl/cpu` in a fresh environment and record that source explicitly. CPU fallback is allowed without requesting phase approval again, but must not be called successful GPU verification. A CUDA wheel can run CPU operations too.

Record a non-secret local preparation manifest beside the venv, including selected interpreter, exact commands/package versions, CUDA/CPU index, device/driver, verification outcomes, date and any fallback diagnosis. `pip freeze` plus `pip check` provides dependency evidence but does not by itself preserve the CUDA installation source. Do not commit an unexecuted freeze as a verified lock. No raw credentials or broad process-environment dump belongs in evidence.

## 4. Package, CPU and GPU readiness — completed; reproduction reference

If reusing an existing inspected environment, set `$BantaiPython` to its `Scripts\python.exe`. Probe commands below do not load a model or an image.

```powershell
& $BantaiPython -m pip check
& $BantaiPython -m pip freeze
& $BantaiPython -c "import sys,struct,torch,torchvision,transformers,PIL,matplotlib; print(sys.version); print('bits:',struct.calcsize('P')*8); print('versions:',torch.__version__,torchvision.__version__,transformers.__version__,PIL.__version__,matplotlib.__version__); print('wheel CUDA:',torch.version.cuda); x=torch.ones(1,device='cpu'); assert (x+x).item()==2; print('CPU tensor: passed'); print('CUDA available:',torch.cuda.is_available()); print('compiled architectures:',torch.cuda.get_arch_list())"
```

Record command exits; an import failure or failed CPU operation is a readiness failure to diagnose. With a usable GPU, run the explicit GPU probe so CPU fallback cannot conceal GPU failure:

```powershell
& $BantaiPython -c "import torch; assert torch.cuda.is_available(),'CUDA unavailable'; print('GPU:',torch.cuda.get_device_name(0)); print('capability:',torch.cuda.get_device_capability(0)); print('free/total bytes:',torch.cuda.mem_get_info()); x=torch.ones(1,device='cuda'); assert (x+x).item()==2; torch.cuda.synchronize(); print('GPU tensor: passed')"
```

A printed CUDA-available flag alone is not a passed GPU check. Record the actual tensor outcome, compiled architectures/capability and memory observation. On failure diagnose driver/wheel/support/memory before retrying; alternatively record a justified CPU-only route and leave GPU as failed/unrun, not passed. These basic tensor probes do not establish model-memory fit, speed or detection correctness.

## 5. Native viewer readiness — completed; reproduction reference

On the interactive Windows desktop, run each command, visibly confirm rendering, close the window normally, and record exit status. No team image, overlay export, temporary image file or model is involved.

```powershell
& $BantaiPython -m tkinter
& $BantaiPython -c "import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt; f,a=plt.subplots(); a.plot([0,1],[0,1]); a.set_title('BANTAI Phase 2A viewer readiness — no model'); f.canvas.draw(); print('backend:',matplotlib.get_backend()); plt.show(block=True); plt.close(f); print('viewer returned normally')"
```

Tk alone is not proof the chosen Matplotlib backend renders; both checks matter. A process/GUI start or a remote headless backend is insufficient. If the actual native viewer cannot render, record a blocker rather than claiming viewer readiness. No display waiting is model timing, and no model timing exists in this phase.

## Completion and stop boundary

The ledger and [environment snapshot](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md) now contain the supplied target outputs, selected `.venv`, exact dependencies/source, pinned metadata and native viewer confirmation. CPU and GPU probes both passed, so no CPU-only fallback is claimed. No required preparation blocker remains; detector/model-memory/quality work remains deliberately unrun and outside 2A. This is a technical preparation result, not owner acceptance of a later phase.

Stop after these checks. Do not implement the detector or its input/output pipeline, download/load pretrained weights, process photos, run inference, select clustering, train/fine-tune or start application modules. Detector implementation/inference needs later authorization. Existing JSON numeric/display contract and fictional examples are preserved, not executed here.


---

## 3. Phase 2B: Foundations & Contract Checks (PR #6 Integration)

# Phase 2B — software checks and evidence

Implemented on 2026-10-17, branch `feat/phase-2b-io-foundation`, from published main `dde98cf0642672eb96422891e101c21c6b67bbc2` (merge of PR #5). Initial tracked checkout was clean; existing branches, ignored environment and user files were preserved. No worktree or direct-main changes. Shell Git fetch failed because the configured proxy could not be reached. The connected GitHub tools supplied current-main metadata and file contents; blob/tree/commit object hashes were verified locally before switching to the exact base. The base is locally shallow because its new parent objects were unavailable through shell Git. This does not change the published base or proposed diff.

## Actual verification

Commands executed in the cloud checkout:

| Command/check | Actual result | Scope |
| --- | --- | --- |
| `/usr/bin/python3 -m unittest discover -s tests -v` | 22 tests passed, zero failures/errors | Synthetic software checks only |
| `/usr/bin/python3 scripts/detect_images.py --help` | Exit 0; explicitly says preparation only, no inference or viewer | CLI entry point |
| `/usr/bin/python3 -m compileall -q scripts tests` | Exit 0 | Syntax/import compilation only |
| `git diff --check` | Exit 0 | Whitespace check |
| Documentation/snapshot consistency script | 84 local Markdown links valid; 5 fenced JSON examples parse strictly; all 36 requirements pins equal the supplied snapshot; original ignore rules preserved | Read-only checks, no dependency resolution/install |

The test runner is existing cloud Python **3.13.5**, Pillow **11.1.0**. The cloud `.venv` lacks Pillow; the initial import probe reported `ModuleNotFoundError`. No package was installed or reinstalled. The existing system Pillow was used for software checks. These are not executions on the verified Windows Python 3.11.9/Pillow 11.2.1 environment and do not replace its supplied 2A evidence. Running the new software tests on that unchanged target remains unrun; no fresh-install, Windows filesystem or GPU result is claimed by 2B.

Fixtures exercise:

- Repository-root path resolution from another working directory; absolute overrides; missing/invalid/duplicate-key/nonstandard JSON configuration; threshold type/range/nonfinite checks; pinned model/processor and device/display validation; path overlaps including symlinks.
- Sorted nonrecursive discovery; missing/empty/not-directory inputs; unsupported skips; corrupt images, wrong file content, missing/unreadable sources; continuation after per-file failures.
- Pixel-exact synthetic PNGs for all eight EXIF orientations; oriented dimensions and box clipping; RGB conversion from grayscale, RGBA and CMYK JPEG; byte SHA-256 equality before/after preparation; no generated image copies.
- NaN/Infinity/non-numeric coordinates and scores; reversed/degenerate/off-image boxes; finite raw values and out-of-range score diagnostics; valid clipping; invalid candidates excluded from valid summaries; unmapped phrases retained; unresolved physical/independent-region counts.
- `ok`, `no_detections`, all-invalid `error`, pre-inference errors and unsupported outcomes; inconsistent records rejected; explicit inference-completion/provenance guards. Every prediction is synthetic and appears only in tests. No model supplies predictions.
- Pending/disabled/not-attempted and simulated shown/display-error outcomes; interrupted pending record retained; final update failure leaves earlier bytes unchanged. These call persistence helpers only, with no implemented viewer.
- Distinct UUID run reservations; forced UUID collision refusal; full path/content source identities; no basename collisions or initial overwrites; failed initial/final publication; simulated write/flush/fsync/replace errors; strict serialization failures and temporary-file cleanup.
- Preparation CLI mixed batch and unsupported-only/missing/empty cases; JSON-only outputs, explicit inference-unrun flag, and no functional success/no-detections after decoding.

All temporary files, source fixtures and JSON artifacts live under ignored `work/` and are cleaned by the tests. No team photos, model cache, weights, image copies or generated results are tracked. Only JSON is persisted by the CLI, under the configured ignored results path by default.

## Implementation and limits

`config/detection.json` preserves the verified model/processor revision, with prompts/thresholds explicitly proposed and unvalidated. Display defaults to enabled, but preparation never invokes it. Paths are resolved before output; input/results/cache paths cannot overlap. The result helpers support Section 7 candidate sanitation, consistency checks, strict persistence and display-only finalization; they do not implement phrase mapping, duplicate selection, overlap heuristics, detector extraction or a viewer.

Functional results add `inference.completed` and require actual device plus environment/preprocessing manifest references for completed inference; the later integration must provide observed provenance and verify its installed library. No software guard can independently prove a caller actually ran a model. The preparation CLI uses separate schemas, `bantai.preparation.v1` and `bantai.preparation-run.v1`; its `prepared` state cannot be mistaken for functional inference success. Unsupported source bytes are not decoded or hashed; unreadable sources may also have null hashes. Successfully read supported sources, including decode failures, retain the actual SHA-256.

Run directories are reserved exclusively with UUIDs. Source output names digest full source path identity plus byte digest; hash-based identity follows SHA-256 collision resistance, while exclusive publication also refuses an existing destination. Project inputs retain project-relative IDs; explicitly external inputs use resolved absolute IDs. Initial publication requires hard-link support and fails observably otherwise; explicit updates use same-directory `os.replace` after flush/fsync/close. One writer per result is required. Atomic visibility and previous-file preservation are tested; directory power-loss durability, network filesystems and Windows behavior were not tested here. A failed run leaves its existing manifest `in_progress`, identifying unfinished work, and returns an observable error; it is not silently marked complete.

Requirements files transcribe all 36 supplied pins from [PHASE_2A_ENVIRONMENT_SNAPSHOT.md](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md), including bootstrap tools, and retain `https://download.pytorch.org/whl/cu118`. They are version snapshots without artifact hashes, not evidence of a tested fresh installation. No dependency changes or new packages were introduced.

## Readiness for 2C

The bounded foundation is ready for review and for later integration when the owner authorizes 2C. Before treating the integrated pipeline as target-validated, run these new software tests in the existing Windows environment. Preserve the recorded model/processor revision and 2A package/source evidence. Model loading/memory fit, real inference, custom-category quality, mapping/duplicates/overlap behavior and native photo overlays remain unrun. No team photos or GPU access were required for these software checks. No automatic grouping, training, web/application modules, merge or 2C implementation occurred.


---

## 4. Phase 2C: Pinned Baseline Inference & Windows Execution

# Phase 2C — baseline implementation and evidence

Terminal-viewer follow-up: [contract and actual checks](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md). This replaces the runtime GUI path while retaining the historical native-viewer evidence below.

Latest direct target evidence: [Windows cached-model follow-up](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md) records 53 passing software tests/one skip and actual successful CUDA inference after the padding fix. The cloud and supplied evidence below remain historical and separate.

Implemented for review on 2026-10-17 after the owner's “proceed now with phase 2C” instruction. Branch `feat/phase-2c-pretrained-inference` starts from published main `5db174bd402456582836c83fdf7c8a4446517d28`, the merge of PR #6. The initial checkout was clean. Existing branches/user files/environments and all requirements pins were preserved. Shell Git fetch still failed because the configured proxy could not be reached; the connected GitHub tools supplied the exact main commit, whose object hash and existing 2B tree were verified before branch creation. No merge is performed by this task.

## Implemented behavior

- Preparation remains the CLI default. `--mode infer` enables the pinned model/processor baseline, with `--device`, `--no-display` and `--local-files-only`. Missing/empty inputs fail before loading; unsupported/undecodable files never initialize a detector. Runtime initialization failure is recorded once and reused as an error for later decodable files rather than repeatedly attempting downloads.
- Direct dependency versions are observed and checked against the existing 2A baseline before imports/downloads. Config, processor and safetensors loading use `a2bb814dd30d776dcf7e30523b00659f4f141c71`, configured cache, `trust_remote_code=False`, float32, eval/inference mode and disabled custom kernels. Resolved config/model commit metadata and Tiny-consistent Swin type/depths/heads are checked. Installed package versions and actual processor settings are persisted; incomplete pre-model errors have null library-version metadata, not invented runtime observations.
- `auto` prefers CUDA, records CPU fallback if unavailable, and permits one CPU retry only after CUDA OOM. Explicit CUDA and unrelated failures do not silently fall back. Actual device and fallback history are saved and printed. Model-load time is separate; one warm-up occurs per actual device, and CUDA synchronization surrounds the measured processor/transfer/forward/grounded-postprocess/candidate-extraction interval. Decode, model load, warm-up, duplicate/overlap review and human viewing are excluded. A single observed interval is not a benchmark.
- The grounded adapter calls the pinned `threshold`/`text_threshold`/`target_sizes=[(height,width)]` API and reads `text_labels`. Raw logits/boxes are audited so upstream threshold comparisons cannot hide invalid queries. Invalid raw candidates below threshold remain diagnostic records; an unavailable decoded phrase stays empty. Infinite logits are invalid even if sigmoid would turn them into finite scores. The query selection uses torch's comparison mask to avoid Python-float versus float32 threshold-boundary differences.
- Exact versioned aliases preserve unmapped/combined/partial phrases. Only identical raw boxes plus normalized phrases are collapsed for display/summary, retaining the highest score and duplicate links. Near duplicates remain. Same-category IoU, cross-category overlap and object-within-region containment produce review flags only. No clusters, derived groups, memberships or physical-object/independent-region counts are invented.
- Functional JSON is atomically persisted before native viewing. Only valid retained candidates are drawn, including valid unmapped phrases, with per-candidate score percentages and oriented original pixel edges. Successful no-detection images can also be viewed. Normal render/return leads to a final atomic shown update; failure records display error without changing inference status/detections. Interrupted/final-write-failed outcomes remain pending. No source writes, annotated copies, image exports, temporary image files or web server.
- JSON-only run/environment/preprocessing metadata includes configuration, observed dependencies/device, fallback history, processor settings, measured timing scope and forward attempt/completion counts including warm-up. Driver is explicitly unqueried for this run; the historical 2A driver is not substituted as a new observation. Incomplete runs remain visible rather than being fabricated as complete success.

## Actual commands and outcomes

| Command/check | Actual result | Evidence limit |
| --- | --- | --- |
| `/usr/bin/python3 -m unittest discover -s tests -v` | **50 tests passed**, zero failures/errors/skips | Synthetic software tests; 22 foundation plus 28 baseline tests |
| `/usr/bin/python3 -m compileall -q scripts tests` | Exit 0 | Syntax compilation only |
| `/usr/bin/python3 scripts/detect_images.py --help` | Exit 0; preparation default and explicit infer/device/display/cache-only controls documented | CLI help, no model |
| `/usr/bin/python3 scripts/detect_images.py --mode infer --input work/phase2c-smoke/input --results work/phase2c-smoke/results --no-display --local-files-only` | Exit **1**; one synthetic decoded source persisted as `status=error`, `error.code=missing_dependency`, `inference.completed=false`, `display.status=not_attempted`, null observed Transformers version | Actual missing-runtime behavior, **not inference or no detections** |
| `git diff --check` | Exit 0 | Whitespace check |
| Documentation/dependency/artifact consistency script | 87 local Markdown links resolve; 5 JSON examples parse strictly; all 36 snapshot pins match; all three requirements files and ignore rules are byte-identical to main; runtime-error JSON and unchanged source hash verified | Read-only checks, no dependency resolution or installation |

The CLI probe's only input was a newly created synthetic 6×4 PNG, not a team photograph. Its source bytes/hash remain unchanged, and output consists of JSON only under ignored `work/`. Test fixtures are also created/cleaned under ignored `work/`; no fixture image, result, cache, environment or model weight is committed.

Actual cloud software environment: Python **3.13.5**, Pillow **11.1.0**, Matplotlib **3.10.1+dfsg1**, with **no torch or Transformers** in the system interpreter or cloud `.venv`. No packages were installed/reinstalled, and the verified Windows requirements were not changed. The real Agg canvas test completed despite a Fontconfig “No writable cache directories” warning; it drew the expected synthetic overlays/labels and did not save an image. This is not evidence of native Tk rendering or Windows operation. Mocked loader/device/processor/viewer fixtures explicitly use synthetic values and make no model-memory, CUDA or detection-quality claim.

Rechecked the official [Transformers v4.51.3 grounded processor source](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/grounding_dino/processing_grounding_dino.py) through the connected GitHub tool. It uses sigmoid token maxima, `scores > threshold`, `(height,width)` target sizes and `text_labels`; nonfinite-query auditing therefore precedes the upstream filter. This is source inspection, not executing that library here. Checkpoint/license/config/processor metadata and Windows readiness remain the supplied evidence in [PHASE_2A_ENVIRONMENT_SNAPSHOT.md](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md), not newly observed target/model results.

## Software coverage and pending evidence

Tests cover exact aliases and unmapped phrases; normalized-phrase/highest-score duplicate links; near duplicates and mixed overlaps without grouping; invalid-candidate exclusion; dependency/version failures; pinned loader flags/revision refusal; float32 threshold-mask boundary; raw invalid queries hidden by thresholding; postprocessor shape errors; inference-only context and oriented dimensions; auto CPU selection/one CUDA-OOM retry/explicit CUDA failure; warm-up/timing scope; persist-before-view; display disabled/error/shown and interruption; initial/final write failures; decode/unsupported/load/prediction errors with batch continuation; all-invalid processing errors; real headless rectangle/label/pixel-frame drawing without exports. The foundation suite continues to cover all EXIF orientations, source preservation, atomic failures and collision prevention.

**Unrun:** actual pinned processor/weight loading, actual CPU/GPU inference, model-memory fit, model-specific compatibility on GTX 1660 Ti, native Tk photo-overlay rendering/normal return, team-photo category/quality checks, qualitative prompt comparisons and actual model timings. These remain required before claiming target functional validation or Phase 2C acceptance. Missing cloud dependencies and unavailable Windows desktop/photos are execution prerequisites, not renewed authorization questions. No inference failure is inferred for the target from the cloud runtime error.

Use the unchanged verified Windows `.venv` to run the new software suite and authorized inference command in [README.md](../README.md). First inference can fetch pinned files; cached-only mode requires a complete existing cache. Preserve originals and upstream notices/conditions. Save actual JSON/metadata under ignored results and record qualitative outcomes/gaps separately. No new training, grouping, alternative-model/optimization, application/web modules or automatic merge is authorized.

## Supplied Windows follow-up and padding-logit correction

Subsequent owner-supplied CMD output confirms a clean branch switch to Phase 2C and a target software run: 50 tests executed in 5.459 seconds, `OK (skipped=1)` — 49 passed, one symlink fixture skipped for Windows privilege error 1314. Matplotlib/Pyparsing deprecation warnings did not fail the drawing check. No administrator access, dependency upgrade or reinstall is needed for those warnings. These are supplied Windows results, not cloud execution.

The next supplied transcript shows configuration/processor/tokenizer downloads and a completed 689 MB `model.safetensors` download. The photo attempt reached extracted queries q0–q899, but the original validator rejected all 900 with `nonfinite_model_logit`, yielding `error/invalid_predictions`, zero retained candidates and unresolved counts. The run is preserved under its original ignored results directory; it is not successful detection or no detections. The supplied terminal text does not establish actual device/timings, valid categories or native photo rendering.

Root cause identified in the pinned [model source](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/grounding_dino/modeling_grounding_dino.py), `GroundingDinoContrastiveEmbedding.forward`: the implementation deliberately fills masked text positions and the fixed-width tail to `max_text_len` with negative infinity. The original raw-logit audit incorrectly treated these expected sentinels as numeric failures on every query. This is an implementation bug; the transcript is not evidence of broken weights, GPU incompatibility or absent trash.

The correction derives active positions from the input attention mask, defaulting to all input tokens active only when that mask is absent, matching the pinned model. It permits negative infinity **only** at known masked/tail padding positions. NaN/positive infinity anywhere, negative infinity on active tokens, invalid scores/boxes and malformed/all-inactive masks remain errors. No raw infinity is written to JSON. The audit policy is recorded in preprocessing metadata. No model, dependency, prompt or threshold pin changes.

Regression verification in the existing cloud software environment: `/usr/bin/python3 -m unittest discover -s tests -v` now passes **54 tests** (zero failures/errors/skips). New synthetic fixtures cover padded and attention-masked negative infinity, absent attention masks, valid/below-threshold candidates, active-token NaN/±infinity, unexpected NaN/+infinity in masked/tail positions, and invalid text masks. Compilation, strict JSON/Markdown checks and whitespace checks also pass. These reproduce the padding contract in software; a corrected target model run remains pending. The cached files should be reused with `--local-files-only`; preserve the original failed run and let the next run reserve a new directory.


### Windows Execution Details
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


---

## 5. Phase 2D: Multi-Image Replay & Native Viewer Lifecycle

# Phase 2D — Display and Summaries checks

Historical Phase 2D evidence. Phase 2E was subsequently authorized, and the owner reported small/nested-box label issues; see [current correction and results](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md). Font defaults and prior acceptance gaps below describe the 2D version.

Date: 2026-10-18. Base: fetched main `76c33f9`, containing PR #10 and the complete prior scrollable viewer. Focused branch: `feat/phase-2d-display-summaries`. No outstanding viewer work needed cherry-picking. Owner prompt/threshold edits remain local and unchanged; the untracked nested folder is untouched. No packages, model weights, revision, mapping or detection logic changed. No inference was run.

## Changes and evidence

The existing implementation already provided one repeated four-section tab, raw phrase wrapping, original oriented pixel axes, passing-only selectable tables, verified single/multi-image replay and atomic separate display outcomes. This phase adds validated configurable fonts (8.5/8-point defaults retained), persists those display settings, shows processing status/errors explicitly and explains decimal score/threshold units. Extremely narrow or obstructed labels still use the disclosed complete-table/zoom fallback; phrases are never completed or remapped.

| Check | Actual outcome |
| --- | --- |
| Existing Windows `.venv` unit suite | 90 tests: 89 passed, one Windows symlink-privilege skip; 10.461 seconds on first run. |
| Agg label/coordinate fixtures | Passed original pixel arrays, fractional boxes, narrow/top-edge/long/incomplete phrases, border bounds, empty image, redraw and configurable font validation. Software evidence only. |
| Lifecycle and failure fixtures | Passed pending persistence before shared display, callback/init failure, prediction preservation, atomic-write failures, replay mismatch rejection and closure. No real desktop failure was induced. |
| Actual Windows saved two-image Tk replay | Passed one tab and two complete blocks, detection plus empty image, vertical wheel, Shift-wheel/horizontal scrollbar at 850x650, read-only selection/copy with prior clipboard restored, and normal closure. |
| Source preservation/replay isolation | Original photo and functional JSON bytes unchanged; zero socket connections and no torch, Transformers, inference or pipeline imports during native replay. |
| Human final visual acceptance | Pending: final smaller-label readability, dense/edge cases and interactive zoom/pan on the final version. Automated Tk success is not visual proof. Earlier owner feedback is preserved in the scrollable-viewer history. |

The native run command was `.venv/Scripts/python.exe work/verify_tab_runtime.py`; the identical reproducible harness is now tracked as `tests/windows_replay_check.py`. It uses the two existing owner photo/result paths listed below, auto-closes and does not require model access. First-run replay attempts: `49cbae0e303d4bdab7ba76af42f236f5` and `80205ffd805c463f95eadc5c45cf7f55`, under ignored `results/<id>/replay.json`. Both completed as shown, with visual verification pending. These are actual native runtime observations, not a detection-quality evaluation.

Final rerun of the tracked harness passed: attempts `831d17d8e6fb4d8caa0d9faf8958ae33` / `61ac52184c5a4d8bb9f95ad5c5e2fae4`. Selection passed; copy was skipped on this rerun to preserve an unavailable/non-text clipboard (copy passed in the first run). Final suite: 90 tests in 11.025 seconds, 89 passed/one skip. Relative Markdown validation resolved 128 links and parsed fenced JSON examples; `git diff --check` passed. Protected inference/postprocessing/input/result/persistence files, requirements and ignore rules match the base. Pipeline changes only persist the two new display settings.

## Reproduce in the existing Windows environment

Run from the repository root, without installation:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe tests/windows_replay_check.py
.venv/Scripts/python.exe scripts/detect_images.py --mode display-check
```

The last command opens three synthetic images for human checks: incomplete/long phrases, narrow edge boxes and an empty result. Close the window normally. Synthetic detections do not measure model quality.

Actual two-photo replay, without inference:

```powershell
.venv/Scripts/python.exe scripts/detect_images.py --mode replay --result results/a2ef6836b41b4e18a793642b1ad461a4/da4b761153fda0359330cde3950d3b4cc1a2247db5bbe6618aec59c092e4afc0.json --image "data/input/test#2.jpg" --result results/ea5da8d9bdc840bb924d5b20c2f1733c/9d75f19ceb224a47917ee88b751c4b74928536296000016824365d6cde7787e6.json --image "data/input/test#3.jpg"
```

For single-image replay, omit the second result/image pair. These private local files are deliberately untracked; another checkout must supply its own matching saved result and original. Every source identity/hash/oriented dimension/frame is checked before any attempt is saved. Replay reads the saved prompt and thresholds, not the current owner experiment settings. New replay attempts record current presentation settings separately.

Inspect each Image #N and the three following sections, zoom and pan using the toolbar, scroll through the whole batch, select/copy tables, and close. Check raw phrases against saved JSON, including incomplete phrases, and verify that empty images remain visible. Inspect replay status separately from original inference status. A viewer failure preserves predictions; interrupted or failed atomic finalization can leave pending records. No annotated or temporary image is written.

Configure optional `display.label_font_size` and `display.label_min_font_size` in the existing config; defaults are 8.5 and 8 points, both constrained to [8,16] with minimum <= preferred. Detection and token scores and thresholds remain decimal [0,1], with strict comparisons. Full raw precision and below-threshold diagnostics stay in JSON.

Phase 2D implementation and automated checks are ready for review; final human visual acceptance remains pending. No Phase 2E, grouping, training, fine-tuning, collection or web work is included. See [phase acceptance criteria](PHASES.md) and [viewer contract](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md).


---

## 6. Phase 2E: Visual Review & Functional Handoff

# Phase 2E — Functional Checking and Handoff

2026-10-18; authorized by D-034. Branch `feat/phase-2e-functional-checking` starts at fetched main `a3ec215` (PR #11 already merged). Local owner configuration and the untracked nested folder are preserved. This is a small qualitative baseline check, not training, a benchmark, measured accuracy, or a Phase 3 settings decision.

## Evidence and coverage

Codex inspected all three existing originals and all 15 experimental image/result combinations using source-verified, in-memory overlays. These are agent visual observations, subject to image ambiguity; they are not owner confirmation or ground-truth annotations. Windows Tk replay separately checked interaction callbacks and tables. No photos, generated result JSON, or annotated images are committed. No images were downloaded or collected.

| Original | Available scenario and qualitative content | Oriented dimensions |
| --- | --- | --- |
| `test#1.jpg` | Dominant isolated crumpled transparent bag, a small rectangular wrapper near the curb, and tiny scattered debris. Some tiny fragments cannot be distinguished confidently from stones/leaves. | 3000 x 4000 |
| `test#2.jpg` | Adjacent drink pouch and transparent bag, a small white paper/card, cigarette butt and separate pink wrapper. Supports adjacent/group-like items, mixed object types, nested predicted boxes and some overlapping/partly obscured material. It is not an unambiguous substantial garbage pile. | 3000 x 4000 |
| `test#3.jpg` | Several separated wrappers/paper-like pieces along a curb and a white piece near the wall; leaves and road texture are confounders. A cropped sack at the right edge may be in use rather than discarded. Supports small/distant and dispersed litter. | 5136 x 3984 |

Missing/unrun: clear substantial piles, unambiguous pile-plus-group mixed scenes, clean no-trash photographs, severe occlusion/night/weather variation, and agreed boundaries for multiple independent groups. A saved empty detector result is **not** a no-trash scene. Counts and group membership remain unresolved.

Original SHA-256 values, unchanged before/after all experiments and replay:

```text
test#1.jpg  8e4bc6101d2e21d45feac245ff01cda71aa0502155a97402df05c4c250949a4f
test#2.jpg  4a9ed5c4a7fff1bd4240d84f36968c55bd57ef33e702626b14c43f0755de2661
test#3.jpg  9685fe59d2e9c71c2cab07de392c6e7bc6843ffedb1646589c5e1819f3601f61
```

## Experiment plan and actual runs

The starting **owner experiment baseline**, not an accepted default, was `plastic bag. drink pouch. cigarette butt`, box threshold **0.20**, text threshold **0.10**, auto device and float32. This differs from the older two-description owner edit recorded during 2D. The checked-in default prompt/thresholds are unchanged; the local owner edit remains uncommitted.

Study artifacts: `results/phase2e-f601e7a2e3a6400db7b395c422559b58/`. `plan.json`, per-experiment config/logs, and `comparison.json` preserve the protocol, actual commands, manifests, complete environment snapshots and compact result comparisons. Each experiment has a separate UUID run under its named subdirectory. All 216 preexisting configuration/photo/JSON files captured at study start retained their hashes through the completed experiment checks. During final verification a subsequent local edit changed the prompt to `plastic bag. drink pouch. a cigarette butt`; this newer wording was preserved, not overwritten or tested here. The 215 captured photo/prior-JSON files remain unchanged. Results below apply to the saved baseline without the article, not the newer local wording.

| Experiment | Single factor changed from A | Box/text | Run ID | Retained candidates: photos 1 / 2 / 3 |
| --- | --- | --- | --- | --- |
| A-baseline | None | .20 / .10 | `65526f0148764151a93cb2610c1c8292` | 2 / 4 / 5 |
| B-concrete | Prompt: `crumpled plastic bag. empty drink pouch. cigarette butt. candy wrapper. discarded paper.` | .20 / .10 | `e81596b9c1974fbe845e9b78e92c03f7` | 13 / 12 / 12 |
| C-categories | Prompt: `a garbage pile. an individual discarded item. a group of individual litter.` | .20 / .10 | `2313ca0c538b407d88f47854fa751c12` | 9 / 16 / 9 |
| D-box-only | Box threshold only; A prompt retained | .35 / .10 | `40526fdf1a954c4a8cf37bce256476fb` | 2 / 2 / 3 |
| E-text-only | Text threshold only; A prompt retained | .20 / .25 | `473a8242810b4578910ec82a6f7d7bee` | 2 / 4 / 5 |

B changes the prompt as a whole; individual wording effects cannot be attributed separately. D and E each compare to A, not to one another. They deliberately examine the tradeoff in box retention versus decoded phrases; more boxes were never treated as better detection. No winning configuration was selected or silently written back.

Every run exited 0, completed all three images on **CUDA**, and recorded zero processing/display/persistence errors. Inference used `--no-display` to keep the experiments independent of viewer waiting, followed by separate native saved-result replay. Thus original inference records correctly say display `disabled`, while replay attempts say `shown`; the original records were not rewritten. Each of the 15 records retains 900 raw queries (13,500 total), including below-threshold results, in strict JSON. Scores/thresholds are decimal [0,1], never accuracy or calibrated probabilities.

A/D/E raw query and token scores match exactly, as do boxes that pass in both compared runs. For queries rejected only by D's raised box threshold, diagnostic pixel boxes differ by at most 0.000120 pixels: the existing extraction code uses upstream float32 pixel boxes for passing queries and Python arithmetic for rejected queries. An initial exact-all-boxes comparison exposed this distinction; it was inspected and documented, not changed or treated as a model-quality difference.

Pinned cached model/processor: `IDEA-Research/grounding-dino-tiny`, revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, safetensors, custom kernels disabled. Local-only loading succeeded, no fallback or new weights. Observed Windows 10 build 19045, Python 3.11.9 x64, torch 2.7.1+cu118, torchvision 0.22.1+cu118, Transformers 4.51.3, Pillow 11.2.1, Matplotlib 3.10.3. Separate `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader` reported GTX 1660 Ti, driver 472.19, 6144 MiB. Full installed-package snapshots were captured as observed, not installed or changed by this work; historical 2A transitive versions are not assumed to be today's versions.

| Run | Load ms | Warm-up ms (first photo only) | Inference ms: photos 1 / 2 / 3 |
| --- | --- | --- | --- |
| A | 1282.1 | 3696.3 | 893.8 / 868.3 / 1035.8 |
| B | 1276.3 | 3593.4 | 935.0 / 823.9 / 960.9 |
| C | 1322.6 | 3581.1 | 990.8 / 812.0 / 973.1 |
| D | 1349.0 | 3556.3 | 906.7 / 812.0 / 989.1 |
| E | 1362.9 | 3603.7 | 897.2 / 788.9 / 967.3 |

One measured repetition per image, four completed forwards per run including warm-up. CUDA timing is synchronized. The interval includes processor/transfer/forward/grounded postprocessing/candidate extraction; excludes decode, model loading, warm-up, duplicate/overlap review and display. These observations are not a performance benchmark or peak-memory measurement.

## Per-image qualitative comparison

Query IDs below are local to their run/image. “Missed” means visibly present but without a suitable retained box in this inspection, not a measured recall score. Material identity and whether an object was discarded can remain uncertain.

| Photo | A baseline | B concrete descriptions | C category descriptions | D box / E text changes |
| --- | --- | --- | --- | --- |
| 1 | q0 encloses the bag plausibly (.8629), q1 the small curb wrapper (.3779). Both decode `plastic bag drink pouch`, an ambiguous combined phrase inappropriate for confidently naming the small rectangle. Tiny debris remains largely unboxed. | q0 `crumpled plastic bag` is a better phrase match. Many near-duplicate bag and curb-wrapper boxes inflate 13 candidates. q4 `discarded paper` spans the painted road stripe: a clear background false detection. q15 `cigarette butt` encloses a tiny brown sliver; identity is uncertain at this scale. | Broad `a garbage pile a group` boxes span curb/background, despite no clear pile. Multiple bag/wrapper boxes overlap. Raw mixed phrases remain unmapped. | D retains the same two boxes. E retains their geometry; q1 becomes `drink pouch`, still not a reliable description of the small wrapper. Neither change resolves missed tiny debris. |
| 2 | q0 pouch, q1 bag, q2 enclosing both, q3 pink wrapper. All decode `plastic bag drink pouch`, so boxes alone overstate usefulness. The cigarette butt and white card have no separate suitable box. The enclosing q2 is not evidence of a mapped group or membership. | q0 `cigarette butt` plausibly covers the butt; q10 adds the white card but with a mixed phrase. q1 identifies the bag. q3 incorrectly calls the drink pouch `crumpled plastic bag`. Pink-wrapper phrases mix pouch/wrapper terms. Near-duplicate/nested boxes remain, including q13 across the adjacent objects. | q0 `a group` is incomplete/unmapped; q8 `a garbage pile` (.3404) maps lexically but spans adjacent objects that do not clearly form a pile. q5 spans almost the entire image, an overbroad pile/group prediction. Repeated individual boxes cover the pouch and pink wrapper; separate bag classification remains mixed. | D removes enclosing q2 and the real pink-wrapper q3, retaining only pouch/bag. E retains all four boxes but q3's decoded phrase becomes the explicit empty string `""`; no phrase is invented to fill it. |
| 3 | Five boxes: foreground blue packet, cropped right-edge sack, red/white wrapper, white curb piece and beige crumpled object. Identical `plastic bag drink pouch` phrases poorly distinguish them. The white litter near the wall is missed. Whether the sack is trash is unresolved. | Adds q12 on the wall-side white piece (`discarded paper`, .2016), close to the box cutoff; q10 `candy wrapper` is plausible on the red/white item. Repeated foreground packet/red-wrapper boxes and combined phrases remain. Sack false-positive status is still uncertain. | `a group` boxes span portions of the littered gutter, but stay incomplete/unmapped. Large pile/group boxes cover much of the walkway/wall: poor region specificity. q11 `discarded item` encloses a slab-like piece near the wall; whether this is litter or part of the damaged surface is uncertain. No exact mapped group is returned. | D loses the white and beige curb objects, retaining three boxes including the uncertain sack. E keeps five boxes; white piece becomes `plastic bag` and beige object's phrase is empty. Higher text threshold does not repair semantic correctness. |

All A/B/D/E candidates remain unmapped under the unchanged exact-alias policy. C has five mapped candidates across its three images (four individual-litter candidates and one pile candidate), none mapped to `group_of_individual_litter`. Mapping success is not visual correctness. Near duplicates remain intentionally because only identical raw-box/normalized-phrase duplicates are collapsed; no grouping, NMS, forced aliases or memberships were added. Broad group-like boxes exist in C, but reliable category/boundary behavior remains unmet. Physical-object and independent-region counts remain null.

## Display correction and acceptance status

Human feedback during this phase: the owner had checked the current viewer and reported that labels still needed to be smaller and some were absent on small boxes and boxes inside other boxes. This is a **failed prior visual check**, not blanket confirmation of zoom/pan, tables or closure.

Confirmed reproduction on saved owner run `4cd1bcf5e9904400a4e6dd4f3403ef9f`, at an 11 x 7.5-inch, 100-dpi constrained canvas: main omitted photo 1 q1; photo 2 q0/q2/q3; photo 3 q0/q2/q3/q4. The corrected renderer placed all these labels. No predictions were recomputed for this comparison.

The previous box-width-only layout and border-collision rejection hid these phrases. Labels now borrow horizontal space within axes and search upward to clear enclosing borders/earlier labels. Query IDs remain present and displaced labels use thin connector lines. Default fonts are now **7.5 points with 7-point fallback**, following the owner's smaller-font instruction; explicit values remain configurable in [6,16]. Raw phrases, stored boxes/scores and the owner's prompt/threshold file are untouched. Placement is recalculated on zoom/pan. The complete table/zoom fallback still applies if a crowded view has no available label space; long connector lines and dense scenes still need human review.

| Check | Outcome and evidence type |
| --- | --- |
| All five new three-photo batches replayed in native Windows Tk | Passed: one tab, complete repeated sections, all passing table cells match saved JSON, no labels omitted at checked initial fit, zoom/pan mouse callbacks change axes limits, original extents/boxes stay fixed, Home/scrollbars/wheel work and window closes normally. Automated runtime evidence, not human approval. |
| Empty-result image replay | Passed on historical saved empty photo alongside detections. Empty original still renders. This does not prove absence of trash. |
| Select/copy | Passed on initial two-image replay, restoring previous text clipboard. Selection passed on all experiment batches; copying skipped there to preserve unavailable/non-text clipboard. |
| In-memory agent overlay inspection | All 15 combinations reviewed; labels and boxes visible at inspected fit. Images were encoded only in memory for inspection, with no annotated file or temporary image. This is separate from a desktop screenshot/human check. |
| Final human label/readability acceptance | **Pending** after correction. Earlier reported small/nested-box failure is recorded above. No new owner confirmation of the corrected version is inferred. |
| Synthetic errors/no-detection/continuation/identity checks | Final suite: 92 tests in 6.841 seconds, 91 passed, one Windows symlink-privilege skip. Includes invalid/all-invalid candidates, decode/model/display errors, batch continuation, mismatch rejection, strict JSON and pending/final lifecycle. Fixtures are not real-photo model evidence. |
| Original/result protection | All 215 captured photo/prior-JSON files unchanged; the config passed experiment-time preservation checks and was subsequently edited locally as described above. Native replay imports no model modules and permits zero socket connections. Per-inference run folders contain only JSON; all original inference/replay schemas validate. Figures are cleared and image copies released on closure. |
| Real-photo errors | None in the five measured runs. No no-detection outcome occurred in these 15 images. Controlled failures/empty results are covered separately, not claimed as observed model outcomes. |

Native logs are in `<study>/<experiment>-native.log`, with each separate replay UUID. Original inference status remains independent of replay status; owner visual verification remains pending. See the [viewer contract](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md) and [2D history](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md).

After the persistence fix, a final native four-image replay checked all three records from the owner-reported `4cd1bc...` run plus the historical empty result. All initial-fit labels were placed (2/4/5/0 boxes), matching tables and zoom/pan/scroll/closure passed, and originals were unchanged. Replay attempts: `02ecfa838b394a23b08d92028284c740`, `12d76ba87acd4de9b2df37791a0a2dcd`, `6899913b378d40e49ff2714458f79953`, `e81ffa33d291498fbde6ad41012067b2`. Selection passed; clipboard copy was skipped to preserve unavailable/non-text content.

The strengthened Home-restoration assertion also passed on the historical detection/empty pair (`07066f40054741e2bc5028bfee7b03ab`, `f5552c3b221442a295c63994e937d236`). Final compilation and `git diff --check` passed; documentation validation resolved 140 relative Markdown links and five strict JSON examples. Study JSON and functional result schemas validated. Protected inference/postprocessing/input/result/pipeline files, dependency pins and ignore rules match main; the only persistence change is the bounded replacement retry below.

## Observed Windows persistence failure and bounded fix

After the initial 90-test suite passed, later 91-test runs failed on rapid atomic `run.json` replacement with Windows error 5 (access denied), including outside the sandbox. This was reproduced in a minimal synthetic pipeline; three controlled probes each succeeded after a 50-ms retry. The process holding/denying access was not identified; antivirus or filesystem indexing is not asserted as fact.

Explicit atomic replacement now retries Windows access/sharing errors 5/32/33 at 50, 100, 200 and 400 ms, then fails normally (at most 0.75 seconds of waiting). It never deletes/truncates the old destination or weakens no-overwrite initial publication. A new fixture proves transient recovery, bounded persistent failure, unchanged previous predictions/JSON and temporary-file cleanup. Final command `.venv/Scripts/python.exe -m unittest discover -s tests -q` passed the 92-test suite as recorded above. No dependency or inference logic changed. The five model runs had already completed successfully; this fix changes persistence resilience, not their detections or timing observations, and did not trigger model reruns.

## Local commands and next boundary

Use the existing environment from the repository root. Do not reinstall. The experiment helper snapshots the **current** owner config; rerunning it after owner edits is a new experiment, not an exact reproduction of the recorded baseline. Recorded per-experiment configs in the study preserve this run's settings.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe tests/phase2e_experiment.py
.venv/Scripts/python.exe tests/windows_replay_check.py
```

The experiment helper invokes the existing CLI five times with `--mode infer --config <snapshot> --local-files-only --no-display`, saves new UUIDs, compares hashes and validates results. To repeat just A using its recorded settings, use:

```powershell
.venv/Scripts/python.exe scripts/detect_images.py --mode infer --config results/phase2e-f601e7a2e3a6400db7b395c422559b58/A-baseline.json --local-files-only --no-display
```

For human inspection without inference (omit the second pair for single-image replay):

```powershell
.venv/Scripts/python.exe scripts/detect_images.py --mode replay --result results/4cd1bcf5e9904400a4e6dd4f3403ef9f/da4b761153fda0359330cde3950d3b4cc1a2247db5bbe6618aec59c092e4afc0.json --image "data/input/test#2.jpg" --result results/ea5da8d9bdc840bb924d5b20c2f1733c/9d75f19ceb224a47917ee88b751c4b74928536296000016824365d6cde7787e6.json --image "data/input/test#3.jpg"
```

To replay experiment records, replace each result with `<study>/<experiment>/<run-id>/<record>.json`, pairing with the unchanged source named inside it. `tests/windows_replay_check.py` also accepts any list of result paths and resolves their recorded sources, auto-checks callbacks and closes. The default command checks a historical detection-plus-empty pair. Private originals/results must exist locally; do not fabricate them in another checkout.

**Recommendations for owner Phase 3 review, not decisions:** retain these settings as unvalidated experiments; do not promote B just for higher counts or D/E for cleaner output. Review small-item misses, incorrect mixed phrases, near duplicates, uncertain sack/slab detections, and pile/group boundary confusion before choosing a direction. Consider separately authorized single-description prompting to isolate phrase interactions and owner review of available scenes; missing pile/no-trash coverage remains a limitation. Any grouping or model improvement needs a later explicit scope decision. This work stops at evidence and handoff: no Phase 3 optimization, grouping, model changes, training/fine-tuning or application modules.


---

## 7. Scrollable GUI Viewer Specification

# One scrollable results tab

The owner's latest instruction replaces both terminal image previews and separate per-image/side-panel windows. One local Tk window contains one **Detection Results** tab. Each image repeats this exact vertical sequence: **Image #1** and its image, **Detection Summary**, **Text-Token Details**, **Configuration**, then Image #2 and all four sections again. Only the numbered heading appears above each image. No browser or application/web module is involved.

## Commands

Use the existing Windows `.venv`; no package reinstall or weight download is needed for replay. Test three software-only images (including incomplete/long phrases, a narrow edge box and empty detections):

```powershell
.venv\Scripts\python.exe scripts\detect_images.py --mode display-check
```

Replay a saved result and matching original without loading or running the model:

```powershell
.venv\Scripts\python.exe scripts\detect_images.py --mode replay --result results/a2ef6836b41b4e18a793642b1ad461a4/da4b761153fda0359330cde3950d3b4cc1a2247db5bbe6618aec59c092e4afc0.json --image "data/input/test#2.jpg"
```

Repeat `--result` and `--image` to show multiple saved results together. The first result is paired with the first image, and so on; pair counts must match. This example includes a saved empty result:

```powershell
.venv\Scripts\python.exe scripts\detect_images.py --mode replay --result results/a2ef6836b41b4e18a793642b1ad461a4/da4b761153fda0359330cde3950d3b4cc1a2247db5bbe6618aec59c092e4afc0.json --image "data/input/test#2.jpg" --result results/ea5da8d9bdc840bb924d5b20c2f1733c/9d75f19ceb224a47917ee88b751c4b74928536296000016824365d6cde7787e6.json --image "data/input/test#3.jpg"
```

Explicit cached-model inference still uses the current local prompt/thresholds without changing them:

```powershell
.venv\Scripts\python.exe scripts\detect_images.py --mode infer --input data/input --local-files-only
```

Preparation remains the default. `--no-display` keeps the text-only path; `--no-display --terminal-mode full` retains all saved raw query/token diagnostics. GUI tables are always passing-only. Legacy preview-width/pagination options do not control the GUI. The configuration backend remains `TkAgg`, with `display.layout=scrollable_tab`; older configurations receive this default.

## Navigation, images and labels

Use the vertical scrollbar or mouse wheel to navigate the whole tab; Shift+wheel or the horizontal scrollbar moves sideways on narrow windows. Tables have their own horizontal scrollbars for long rows. They use read-only selectable text: drag to select, Ctrl+A to select a table, Ctrl+C to copy. Text is never silently truncated. Per-image Home/Back/Forward/Pan/Zoom controls remain below each image; Save/export controls are omitted. Close the one window to complete display for the batch.

The complete EXIF-oriented RGB original is passed to Matplotlib in memory. Its image extent is `(0, width, height, 0)`, with equal aspect ratio and original pixel-coordinate ticks on all four sides. A small display-only margin above the image accommodates top-edge labels; values outside the original dimensions have no coordinate labels. Window fitting, zoom and pan do not rescale stored boxes or change image bytes. Screen resolution limits the initial fit view; zoom inspects original samples without inventing detail.

Only valid retained predictions passing the saved strict box-threshold decisions get boxes. Labels show the **raw decoded phrase**, optionally prefixed by its matching query ID when space permits, never the mapped category or prompt substitution. Scores stay in the summary. Incomplete phrases such as `a group` remain incomplete. Empty decoded strings are represented explicitly as `""`.

Label layout is recalculated against actual screen dimensions on every draw, including zoom/pan. Following the owner's Phase 2E feedback about oversized/missing labels on small and nested boxes, defaults are 7.5 points with a 7-point fallback. Labels show query ID and the honest raw phrase, wrapping by actual glyph widths. Small boxes may borrow horizontal space within the axes; labels move upward to avoid enclosing borders and preceding labels. Thin connectors associate displaced labels with their boxes. Labels remain above their box and their backgrounds do not obscure retained box borders. Original coordinates and predictions are unchanged.

All labels were placed at the checked initial fit for the owner-reported run and all 15 Phase 2E photo/result combinations. This is not a guarantee for arbitrarily dense or zoomed scenes: when no readable space exists, the full phrase/query stays in the selectable summary, and zoom may reveal it. Offscreen box tops do not leave stray labels. Final owner acceptance of the corrected size/placement is pending; earlier human feedback reported a failure. See [Phase 2E evidence](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md).

## Tables and actual configuration

Detection Summary contains query ID, exact quoted decoded phrase, stored mapped category/reason, decimal model score, rounded original-pixel xyxy box and threshold status. It excludes invalid, duplicate and below-threshold candidates. `unmapped` remains distinct from inference failure or threshold rejection; no category mapping is recalculated. Scores are not accuracy or calibrated probabilities. Only presentation values are rounded.

Text-Token Details contains passing ordinary tokens for those retained queries, including original tokenizer positions and token IDs. Special/padding, invalid and below-text-threshold rows are excluded. Complete raw scores, precision, invalid diagnostics and alignment remain unchanged in JSON.

Configuration shows the saved image identity, actual inference prompt, box/text thresholds, model ID and immutable revision, device actually used, and coordinate frame. Replay always uses the saved inference values, even when the current configuration differs. Missing historical information says `not recorded`; it is never inferred. Empty results still include the original image, all four sections, explicit empty detection/token messages and configuration.

## Persistence and source verification

Inference persists each result before it can enter the GUI, queues in-memory image copies, releases the model, then opens the one batch tab. Successful processing status/predictions remain independent of display status. All queued records remain pending until normal window closure. GUI initialization, rendering or callback failure records a separate display error for queued results; it never erases detections. Interruption leaves pending. Each final JSON update is atomic; a failed final update leaves that record pending, while previously finalized records remain finalized. This is not a batch-wide transaction.

Replay verifies **every pair** before creating any pending attempt: result stability, source path identity, SHA-256, oriented dimensions and coordinate frame. It imports no model/inference modules and never rewrites the original functional JSON/photo. Every source gets its own `results/<uuid>/replay.json`, with image number, saved provenance, layout, pending/final status and `inference_executed=false`. One shared window displays them all. Rendering creates no automatic annotated copies or temporary image files. Dependencies, weights, prompts, thresholds and category mapping are unchanged.

## Current configuration and verification

`display.label_font_size` and `display.label_min_font_size` are optional finite point sizes in [6,16], with minimum <= preferred. Defaults are now 7.5/7 points after the Phase 2E owner feedback. Add these fields to the existing display object to customize; no prompt or threshold change is needed. Current display settings are recorded in new inference/replay outcomes and shown alongside saved inference configuration. Historical results use the defaults unless replay display settings override them. Scores and thresholds use decimal [0,1] units throughout.

See [Phase 2E results](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md) for current evidence and pending human acceptance; [Phase 2D checks](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md) preserve the earlier results. The observations below are preserved history; PR #10 has since merged them into main.

## Historical verification - 2026-10-18

Work continues on `feat/terminal-viewer-diagnostics` from fetched `000b3b0`. Owner prompt/threshold edits and the untracked `BANTAI-Modeling-/` folder are preserved. No model execution or package change is needed for this presentation work.

Software fixtures: the existing Windows `.venv` ran **89 tests: 88 passed, one Windows symlink privilege skip**, latest run 9.413 seconds. Agg fixtures cover full-source arrays, fractional boxes/original-pixel alignment, raw/incomplete/long phrases, narrow/top-edge boxes, box-width/border bounds, readable font limits, redraw after zoom, and empty images. Mock Tk checks cover one-tab repeated order, wheel directions, initialization/callback failures and closure; table text retains complete selectable content. Batch tests verify every prediction persists pending before one window opens and viewer failures preserve all predictions. Multi-replay tests verify all sources before persistence, numbering and per-source outcomes.

Direct Windows observation: the owner confirmed the three-image synthetic layout, scrolling and output were readable and closed normally; the process exited 0. An actual two-image saved replay (one with predictions and one empty) also exited 0 with separate attempts `631f9fb73c914b0781d43d35db8e8034` and `a43156ccf6b048dda495a87c3a6697a3`. The owner reported a missing label. Inspection found that q2's width estimate created excessive line wrapping and a border collision. Actual-glyph wrapping, whole-word preference and optional-ID omission corrected this; the photo-shaped software regression now displays all three complete phrases without covering borders. The corrected replay (attempts `1c762d9578f148909d83b1a2e9144fe3` / `e876a3150705405fae50b838b20cbabc`) exited 0, and the owner confirmed the phrase labels were complete. The owner then requested smaller labels; the final 8.5/8-point adjustment passed software checks but has not had separate visual confirmation. No detection-quality acceptance is inferred.

Replayed four saved records (three older photo records and the newer empty record) in text-only summary/full modes with no inference, model imports or socket connections. Summary query counts were 2/3/2/0; each full replay covered 900 queries. All saved JSON bytes and original photo bytes stayed unchanged. Configuration in saved replay remains authoritative over local owner edits. No prompts, thresholds, model weights, mapping rules, packages or ignore rules changed.

Earlier per-image viewer evidence is historical in [NATIVE_VIEWER.md](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md); earlier model evidence remains in [PHASE_2C_WINDOWS_RUN.md](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md). A successful event loop or normal closure is not visual proof or a detection-quality judgment. Truly tiny or obstructed labels still use the disclosed full-table/zoom fallback; missing on-image text never means its prediction was removed.

An actual native Tk runtime check on the two saved records verified one tab, exact repeated section order, read-only table selection/copy (restoring the previous text clipboard), vertical wheel and Shift+wheel/horizontal scrollbar navigation at 850x650, and normal closure. The final smaller-label runtime check also passed (attempts `a1d75cd78b594be381a1f7c1e1e3e3b2` / `78bffd30658442ceb79f8b71b21d8ab0`); copying was skipped in that last check to preserve an unavailable/non-text clipboard. No model modules, inference, socket connections or source/result mutations occurred. These automated Tk observations are software/runtime evidence, separate from the owner’s visual feedback.

Compilation, CLI help and whitespace checks passed. Documentation validation resolved 119 relative Markdown links and parsed five strict JSON examples. The inference/postprocessing/input/result/persistence core, dependency pins and ignore rules match the starting head. The current branch had no open PR at inspection; PR #9 was already merged externally. This work does not merge or reopen it.


---

## 8. Phase 2 Milestone Register (D-021 through D-034)

- **D-021**: Pinned model checkpoints selected: `IDEA-Research/grounding-dino-tiny` with exact Hugging Face safetensors revision.
- **D-022**: Local Windows environment verified: Python 3.11.9, CUDA 12.1 tensor creation passing on GTX 1660 Ti.
- **D-023**: Native TkAgg backend approved for in-memory visual review.
- **D-024**: Foundation merged in PR #6 with zero test regressions.
- **D-025**: Exact phrase mapping engine implemented (`scripts/taco_stage5_material_hierarchy.py`).
- **D-026**: Candidate validity rules enforced: non-finite values mapped to null with warning records.
- **D-027**: Atomic display lifecycle implemented (`pending` -> `shown` / `error`).
- **D-028**: Command-line interface created for deterministic execution and multi-image batch runs.
- **D-029**: Scrollable viewer layout approved: repeats Image #N, Detection Summary, Token Details, and Configuration.
- **D-030**: Passing-only selectable tables implemented with zoom/pan capabilities.
- **D-031**: Bounded labels above bounding boxes with pixel coordinate ticks verified.
- **D-032**: Owner unannotated images reviewed across prompt variations (`trash`, `garbage_pile`, `individual_litter`).
- **D-033**: Score threshold sensitivity evaluated (`0.15` vs `0.25` vs `0.35`).
- **D-034**: Phase 2E functional checking and handoff completed successfully; stop before Phase 3 model fine-tuning.
