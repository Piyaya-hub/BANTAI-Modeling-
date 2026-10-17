# Phase 2A — Windows environment preparation and readiness record

## Authorization and actual status

The owner authorizes Codex to perform **local Windows environment setup/readiness only**, corresponding to [PHASE_1_IMPLEMENTATION_PLAN.md](PHASE_1_IMPLEMENTATION_PLAN.md) Section 4 and Section 8 stages 2–3. Codex replaces Antigravity; the owner states Antigravity made no changes. See the [phase definition](PHASES.md#phase-2a--local-windows-environment-setup-and-readiness-verification) and D-022/D-023 in [DECISIONS.md](DECISIONS.md).

**Phase 2A technical preparation checks are complete based on supplied Windows evidence.** Python/venv, dependencies/imports, CPU/CUDA execution, native rendering/normal return, checkpoint/processor metadata, official driver reference, and exact environment capture are verified to the limited preparation scope. No detector or inference result is implied. The owner retains acceptance/next-phase authority; no further implementation is started.

Codex can execute only in a separate Linux cloud checkout. Target results below came from owner-run Windows commands, a supplied screenshot and explicit viewer confirmation, not direct cloud Windows execution. No cloud ML installation was used as a substitute. Local `.venv` evidence files were saved, and the supplied values are committed in the [environment snapshot](PHASE_2A_ENVIRONMENT_SNAPSHOT.md). No weights, photographs or inference were needed.

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

The ledger and [environment snapshot](PHASE_2A_ENVIRONMENT_SNAPSHOT.md) now contain the supplied target outputs, selected `.venv`, exact dependencies/source, pinned metadata and native viewer confirmation. CPU and GPU probes both passed, so no CPU-only fallback is claimed. No required preparation blocker remains; detector/model-memory/quality work remains deliberately unrun and outside 2A. This is a technical preparation result, not owner acceptance of a later phase.

Stop after these checks. Do not implement the detector or its input/output pipeline, download/load pretrained weights, process photos, run inference, select clustering, train/fine-tune or start application modules. Detector implementation/inference needs later authorization. Existing JSON numeric/display contract and fictional examples are preserved, not executed here.
