# BANTAI documentation and implementation handoff

## Current state and Git evidence

Phase 0 is complete. Fetching `main` for this correction confirmed `b366f769f38158675d02fda628c73971d30ed9bb`, the merge of PR #3 containing the Phase 1 planning deliverable. It was still the latest remote `main` at inspection, matching the previously reviewed commit. Earlier PRs #1 and #2 are also merged in its history. Planning publication is not verification of the owner’s hardware/environment, successful detection, or blanket acceptance of technical proposals.

The current focused branch is `docs/phase-1-contract-corrections`, created from that `main`. Codex owns documentation correction and requested ignore-rule additions only. Antigravity handles later authorized implementation. This task does not implement inference, install dependencies, download weights, run experiments, or merge. This correction’s commit/push/PR outcomes are reported in the task completion message; no merge is claimed.

## Corrections and current requirements

README, agent rules, scope, architecture, phases, decisions, test plan, and the [Phase 1 plan](PHASE_1_IMPLEMENTATION_PLAN.md) now distinguish completed Phase 0/published planning from unverified preparation. Current inputs are unannotated team photographs manually downloaded/extracted from Google Drive. Functional visual checks replace the older annotation/benchmark requirements; no annotation work, held-out splits, precision/recall/mAP, or numerical accuracy targets are needed.

The modelling-only baseline remains pretrained Grounding DINO Tiny through Hugging Face Transformers, still images, `individual_litter`, `group_of_individual_litter`, and `garbage_pile`, with multiple candidate regions and mixed categories. Direct prompting comes first. No automatic spatial grouping, DBSCAN, training, fine-tuning, or application modules belong to this baseline.

Outputs are in-memory boxes/labels/individual model-score percentages, readable summaries, per-image JSON, and run/environment metadata. Preserve sources; no automatic annotated copies. The [contract](PHASE_1_IMPLEMENTATION_PLAN.md#7-proposed-json-contract) nulls invalid numeric components with diagnostics, excludes invalid candidates from displayed boxes/valid summaries, and separates inference outcome from `display.status`. Persist inference JSON before opening the viewer, then atomically update display outcome; interrupted pending display remains unfinished.

Exact aliases can leave phrases unmapped. Identical-box/same-phrase collapse does not remove near duplicates; IoU flags are not clustering, membership, or general NMS. Physical-object and independent-region counts remain null when unresolved.

All existing `.gitignore` rules are preserved; project-local input/results/Hub-cache exclusions are appended. Inspection found no tracked files under `data/input/`, `results/`, or `.cache/huggingface/`; nothing was untracked or moved, and no such directories were created for this task.

## Remaining prerequisites and next intended action

Review this documentation correction. At the agreed phase boundary, authorize Antigravity’s next preparation/implementation phase; do not infer authorization or Phase 1 acceptance from this publication. Routine technical defaults inside an authorized phase do not need extra approval checkpoints.

On the owner’s Windows computer, verify Python, NVIDIA driver, actual RTX 2080 Super identity and memory availability, CUDA wheel/GPU execution or diagnosed CPU fallback, and the native viewer. Verify the exact checkpoint/configuration/conditions/safetensors/immutable revision, load model and processor from that same revision, and record resolved dependency versions plus CUDA installation source and preprocessing/settings. Proposed versions, prompts, thresholds, alias mapping and duplicate policy are not locally tested or automatically accepted.

Prior official-source access limitations remain disclosed in the plan; this correction adds no new compatibility findings. Team photos may arrive later: their absence blocks actual visual checking, not documentation or environment planning. No model has been implemented or evaluated in this work and no performance results exist here.

## Historical context

PR #1 introduced the initial foundation; PR #2 finalized scope; PR #3 published the technical plan. Earlier Phase 0-only authorization, pending PR #2/#3 wording, and mandatory annotation/benchmark prerequisites describe superseded requests. They are not current blockers. [DECISIONS.md](DECISIONS.md) retains the earlier decision IDs and explicit replacement references; Git history retains prior document wording.
