# BANTAI preparation handoff

## Current authorization, ownership and Git state

Codex now owns implementation, replacing Antigravity. The owner states Antigravity made no changes; no unverified implementation is attributed to it. Phase 0 is complete and Phase 1 planning is published. Latest fetched `main` is `7a776cf`, merging PR #4's documentation/output-contract corrections; PR #3 published the plan at `b366f76`. The clean checkout was used to create `docs/phase-2a-environment-preparation`.

The owner authorizes **Phase 2A: local Windows environment setup and readiness verification**, corresponding to [plan](PHASE_1_IMPLEMENTATION_PLAN.md) Section 4 and Section 8 stages 2–3. [PHASES.md](PHASES.md) records the exact scope, acceptance evidence and stop boundary. No detector implementation, model-weight downloads/loading, inference, grouping, training or application work is authorized. Routine 2A preparation does not need repeated phase approval.

## Work and evidence

Collaboration/current-state guidance is aligned with Codex ownership. [PHASE_2A_READINESS.md](PHASE_2A_READINESS.md) records evidence, local PowerShell commands, proposed package choices and outstanding prerequisites. Existing `.gitignore` rules and historical decision rows are preserved.

The attached execution environment reports Linux, has no PowerShell executable or connection to the owner's Windows computer, and is not Windows readiness evidence. Actual Windows preflight outputs have been requested; the owner asked what a local checkout meant, so a Windows clone and existing `.venv` are unconfirmed. The readiness runbook explains the local copy and starts with hardware/tool checks that do not require it. Local setup/probes cannot be executed here. No target dependencies were installed, no viewer/GPU checks were claimed, and no model weights were downloaded or inference run. Available package metadata/wheel listing was rechecked; checkpoint metadata and NVIDIA driver documentation still returned proxy `403` errors. Their target-side verification remains outstanding.

## Next action and completion condition

Obtain Windows preflight from the actual checkout, then complete authorized isolated setup and readiness probes using the recorded commands. Record Windows/Python/driver/memory, checkpoint metadata conditions/revision/safetensors, exact dependencies/CUDA source, package checks, CPU/GPU tensor outcomes or a diagnosed CPU fallback, and native Tk/Matplotlib display without photographs or saved images. Preserve any existing `.venv`; a new environment is selected only after inspecting it.

Phase 2A is not ready/complete while required target evidence or metadata verification is missing. This is a target-access prerequisite, not a repeated authorization request. Stop after preparation; later detector implementation/inference needs separate phase authorization. No photographs or annotations are needed for 2A. Commit/push/PR outcomes are recorded in the completion report; no merge is claimed.

## Preserved project boundaries and history

The baseline remains still-image trash detection through pretrained Grounding DINO Tiny/Transformers, direct prompting first, three intended categories and multiple candidate regions. Team photos are manually supplied and unannotated. Later outputs remain in-memory overlays/model-score percentages, summaries, strict per-image JSON plus metadata, with no source overwrites or automatic annotated copies. Corrected invalid-number/display lifecycle rules remain in the plan. No benchmark/annotation requirement is reinstated.

Earlier assignments to Antigravity and documentation-only task instructions are preserved in [DECISIONS.md](DECISIONS.md) as superseded history (D-022/D-023 are current), not current blockers. Prior published plans and merged documentation are not proof of environment compatibility or successful detection.
