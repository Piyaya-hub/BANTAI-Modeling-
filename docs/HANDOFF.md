# BANTAI Phase 2C baseline handoff

## Current result and execution limits

The owner instructed Codex to proceed with Phase 2C (D-025), superseding the previous stop-before-2C boundary. PR #6 is merged; this focused branch starts from current main `5db174b`. The baseline now implements pinned model/processor loading, float32 eval/inference mode, device handling/one auto-mode OOM CPU retry, exact mapping, conservative duplicate/review flags, readable summaries and native in-memory TkAgg viewing with the existing atomic display lifecycle. The CLI still defaults to preparation; inference requires `--mode infer`. Requirements and the existing verified target environment are preserved.

New modules: `scripts/inference.py`, `postprocessing.py`, `pipeline.py`, `viewer.py`; integration changes to configuration/CLI/result validation; `tests/test_inference.py` and a small foundation-fixture update; current-state docs and D-025. Model/processor/config use the verified revision; raw invalid queries are retained before threshold loss; no automatic grouping/membership/count inference or image exports.

Actual checks: **50 software tests passed** using existing cloud Python 3.13.5/Pillow 11.1.0/Matplotlib 3.10.1+dfsg1. CLI help, compilation and whitespace checks passed. A real cached-only CLI probe on a synthetic image returned exit 1 and persisted `runtime/missing_dependency` because torch is absent, with incomplete inference and unattempted display. This is not a no-detection/model result. Headless Agg drawing passed; native Tk calls are simulated. No package install/reinstall or weight download occurred. Full evidence, commands and assumptions: [PHASE_2C_CHECKS.md](PHASE_2C_CHECKS.md).

The implementation is ready for code review and target execution, **not verified model/target acceptance**. Actual model loading/inference, GTX 1660 Ti memory fit, native photo overlays and unannotated team-photo observations remain unrun. Run the new software tests and baseline in the existing Windows `.venv` using [README.md](../README.md); record actual JSON/native outcomes and limitations. Missing target/runtime/photos are execution prerequisites, not a request for repeated phase confirmation. Commit/publication outcomes are reported in the completion message; no merge or later optimization/grouping is authorized.

## Historical Phase 2B handoff — prior stop boundary superseded by D-025

## Current result and next boundary

The owner explicitly authorized Phase 2B, superseding the previous 2A-only stop boundary. Configuration, image preparation, result-contract utilities and atomic strict-JSON persistence are implemented for review on `feat/phase-2b-io-foundation` from current published main `dde98cf` (PR #5). No model loading, weight downloads, inference, viewer implementation or Phase 2C work occurred. No package was installed or reinstalled. The historical 2A handoff below remains preparation evidence, not the current authorization boundary.

Files: `config/detection.json`; `scripts/{configuration,inputs,results,persistence,detect_images}.py`; `tests/test_foundation.py`; three requirements files transcribed from the 2A snapshot; current-state docs and additive `/work/` ignore rule. CLI output is explicitly preparation-only, using separate schemas and `prepared` rather than functional inference success/no detections. Originals are read only and transformed solely in memory. Only JSON is persisted.

Actual checks: 22 synthetic software tests passed via `/usr/bin/python3 -m unittest discover -s tests -v`; CLI help, compilation and diff whitespace checks passed. Cloud Python 3.13.5/Pillow 11.1.0 were already installed; the verified Windows Python 3.11.9/Pillow 11.2.1 target was not directly accessible. New target software tests remain unrun. No fresh-install or GPU/model/viewer result is inferred. Full command/results, failure-injection coverage, persistence assumptions and limitations are in [PHASE_2B_CHECKS.md](PHASE_2B_CHECKS.md).

P-001 now distinguishes verified 2A metadata/basic runtime from untested model behavior; D-024 records the owner's bounded authorization, preserving D-001–D-023. Mapping/duplicate selection/overlap interpretation, model-memory fit, real-photo behavior and native overlays remain later work. The foundation is ready for review and eventual authorized 2C integration; run the new software checks in the unchanged target environment before claiming target validation. Commit/publication outcomes are reported in the completion message; no merge is authorized or claimed.

## Historical Phase 2A handoff — prior stop boundary superseded by D-024

## Result and ownership

**Phase 2A technical preparation checks are complete on owner-supplied Windows evidence. Stop here: detector implementation and inference have not begun and remain outside the authorization.** Codex owns implementation, replacing Antigravity; the owner reports Antigravity made no changes. Phase 0 is complete, Phase 1 planning is published, and this preparation corresponds exactly to plan Section 4 and Section 8 stages 2–3.

Cloud Codex reviewed actual target command outputs, a native-plot screenshot and explicit normal-return confirmation; it did not directly execute Windows commands. The local repository is in the owner's selected Documents/Python/Project Image-object Detection folder with a fresh `.venv`. The final evidence is preserved in [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md) and [PHASE_2A_ENVIRONMENT_SNAPSHOT.md](PHASE_2A_ENVIRONMENT_SNAPSHOT.md).

## Verified preparation

- Windows 10 build 19045.6466; Git 2.45.1.windows.1; Python 3.11.9 x64 installed from python.org via WinGet with installer-hash verification; local clone and venv confirmed.
- GTX 1660 Ti, 6144 MiB VRAM, driver 472.19, capability (7,5), CUDA build 11.8; CPU and CUDA tensors both returned 2.0. Earlier RTX 2080 Super assumption is superseded, not erased from history.
- torch 2.7.1+cu118, torchvision 0.22.1+cu118, Transformers 4.51.3, Pillow 11.2.1 and Matplotlib 3.10.3 installed/imported. `pip check` reported no broken requirements. All 36 frozen packages and cu118 installation source are recorded.
- Native TkAgg synthetic diagonal plot rendered; the owner explicitly confirmed `TkAgg viewer closed normally`. No image export was invoked.
- Official metadata/config/card/processor reads succeeded for `IDEA-Research/grounding-dino-tiny` at revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. Reported Apache-2.0 metadata, non-gated access, safetensors file, Tiny-consistent Swin configuration and same-revision processor settings are preserved.
- NVIDIA CUDA 11.8 Table 2 lists Windows minor-compatibility minimum >=452.39; driver 472.19 exceeds it and the actual CUDA tensor passed. The separate toolkit driver listing 522.06 is not the same minimum. Earlier cloud 403 limitations were resolved for these checks through successful target requests, without claiming cloud policy changed.
- The owner saved `.venv/phase2a-requirements.txt` and `.venv/phase2a-environment.json`; their supplied dependency/config values are preserved in the committed snapshot. `.gitignore` and historical decision rows remain intact.

## Remaining boundaries and next action

No current preparation blocker remains in the supplied evidence. No further installation or repetition of passed checks is needed for the unchanged environment. The owner can review the preparation branch and issue the next phase instruction; this technical completion does not accept a later phase on their behalf and does not authorize a merge.

Pretrained weights were not downloaded/loaded. No detector/application code, inference, team-photo processing, clustering, training/fine-tuning or annotated-image export occurred. Model-memory fit, custom category capability, detection quality and future model-specific driver behavior are intentionally untested. Later failures must be diagnosed from actual evidence; no accuracy or performance result is inferred from preparation.

Future authorized model and processor loading must use the same recorded revision. The baseline remains still-image trash detection, three intended categories, direct prompts first and multiple candidate regions; team photos are unannotated and manually supplied. Existing numeric/display contracts and no-automatic-image-export rules remain. Prompt/threshold/alias and grouping-boundary defaults are unvalidated proposals.

## Git publication and history

Latest fetched main for preparation is `7a776cf`, merging PR #4; PR #3 published the plan at `b366f76`. Work is on `docs/phase-2a-environment-preparation`, never direct main. Final commit/push/PR outcomes are reported in the completion message; no merge is claimed. Earlier PR API attempts returned `Forbidden`; do not claim a PR exists without confirmation.

D-001–D-023 preserve prior owner decisions, with D-022/D-023 assigning Codex and limiting this work to preparation. Previous pending/blocked readiness wording is historical and resolved by the supplied evidence above. No new implementation approval or owner acceptance date is invented.
