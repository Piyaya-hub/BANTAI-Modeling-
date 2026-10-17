# Phase 0 handoff

## Latest foundation

PR #1 merged the initial foundation into `main`; fetched Git history records merge commit `0a9c738`. These final revisions begin from that latest `main` on `docs/finalize-phase-0-detection-scope` and require a new pull request. The GitHub PR API returned `Forbidden` when checking PR #1; the merge was confirmed through repository history. These revisions are not claimed merged or accepted; owner review remains pending.

The approved repository responsibility is AI modelling and computer vision only. The initial scope is still-image trash detection using pretrained GroundingDINO-T (Swin-T / Tiny configuration), with direct text-guided detection evaluated first. Required outputs are Garbage Pile, Individual Litter, and Group of Individual Litter, with mixed categories and multiple independent regions per image. Preserve physical objects separately from region observations; do not double-count a group and its members as independent trash regions.

The owner reports an NVIDIA RTX 2080 Super development GPU; runtime availability and compatibility are unverified. Direct model support for the custom categories and group-level boxes remains experimental. DBSCAN and other spatial grouping methods are conditional alternatives requiring baseline evidence and explicit authorization. Application modules, central databases, incident management, reporting, and hotspot ranking remain outside this repository.

## Evidence and limits

The revision covers the eight foundation Markdown files. Validation checks their layout, relative references, consistent scope/categories/model/strategy/phase gates, unsupported-claim avoidance, documentation-only diff, and preserved `.gitignore`. The completion report records actual validation and commit/push/new-PR outcomes; a pushed branch alone is not a pull request or approval.

No inference code, ML dependency installation, datasets, checkpoints, grouping implementation, frontend, or backend is introduced. No model has been implemented, trained, or evaluated in this work; no successful detection, acceptable accuracy, or benchmark results are claimed. Phase 1 has not begun.

## Remaining prerequisites and next action

Review these Phase 0 revisions against [PHASES.md](PHASES.md). Do not merge as part of this task. Explicit owner authorization is required before Phase 1.

During a later authorized Phase 1, resolve exact implementation/checkpoint source and revision/configuration, dependency and local GPU compatibility, data rights/handling and annotations, operational category/grouping/overlap rules, prompt vocabulary/formatting and separate/combined comparisons, thresholds, object-level and region-level metrics, success criteria, held-out protocol, output schema/counting/provenance, and reproducibility requirements. See [DECISIONS.md](DECISIONS.md), [ARCHITECTURE.md](ARCHITECTURE.md), and [TEST_PLAN.md](TEST_PLAN.md). No unresolved choice is approved merely by appearing in these documents.
