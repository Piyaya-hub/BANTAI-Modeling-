# Validation and evaluation plan

## Phase 0 documentation checks

Verify that `AGENTS.md`, `README.md`, and the six required `docs/` files exist. Check relative Markdown links, consistent trash-only initial scope, explicit unresolved decisions, approval boundaries, phase acceptance criteria, and absence of unsupported model or benchmark claims. Inspect the Git diff for documentation-only changes and confirm `.gitignore` is unchanged. Do not run or invent model tests during this phase.

## Proposed future evaluation protocol

This section is a proposal, not executed testing. Resolve and approve its concrete choices in Phase 1 before Phase 2 evaluation.

- Establish a permitted, annotated still-image collection with a written trash definition and rules for ambiguous instances. Include negative images and varied relevant conditions; do not claim coverage without examining the collection.
- Separate exploratory prompt/threshold selection from held-out evaluation. Record dataset identities and any overlap or leakage limitations.
- Define bounding-box coordinate conventions, annotation matching rules, overlap threshold, score filtering, and handling of duplicate predictions before measuring results.
- Choose suitable detection metrics, such as precision, recall, and an agreed average-precision formulation where annotations support it. Define owner-approved numeric success criteria before evaluation; no target values are assumed here.
- Record per-image predictions, false positives, missed objects, and representative failure cases. Measure latency and resource use only under a documented device and timing procedure.
- Retain enough non-sensitive metadata and instructions to repeat the experiment, including the model revision and configuration.

## Future functional checks

After implementation is authorized, exercise valid images, images without trash, unreadable or unsupported inputs, and varied dimensions. Verify that boxes map to original image bounds and that labels and scores follow the chosen model's documented semantics. Distinguish a successful run with no detections from a failed run. Confirm the selected checkpoint is used for inference without training updates.

## Reporting rules

Keep passed, failed, skipped, and unrun checks separate. Record actual commands, environment, artifacts, and outcomes. A successful import, process start, or zero-test run does not validate detection quality. Report measured results even when they fail the agreed success criteria; do not adjust criteria retrospectively to manufacture success.

Current status: no model evaluation or application tests have been run as part of Phase 0. See [PHASES.md](PHASES.md) for gates and [HANDOFF.md](HANDOFF.md) for the repository handoff.
