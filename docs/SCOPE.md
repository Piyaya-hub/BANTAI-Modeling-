# Scope

## Broader academic proposal

The broader BANTAI proposal covers roadside garbage, potholes, road-surface damage, incident management, and frequency-based hotspot prioritization. These capabilities describe the research direction rather than the current repository's implementation commitment.

## Repository responsibility

This repository is exclusively for AI modelling and computer vision: pretrained model evaluation/selection, preprocessing, object detection, necessary postprocessing, accuracy/performance evaluation, approved optimization and potential fine-tuning, and reusable detection outputs for future integration. Frontend, backend application modules, reporting, incident management, central databases, and hotspot ranking belong outside this repository, including in later phases. Road-hazard detection, video processing, and additional model development need separate future approval.

## Approved initial direction

Current D-037 authorizes [Stage 3A technical feasibility assessment](TACO_STAGE_3A.md) only: exact-checkpoint training compatibility, dataset expansion options, hardware, licensing and a proposed smoke/evaluation procedure. It does not authorize training, taxonomy changes, replacement exports, installations or inference changes.

D-036 authorized the completed [TACO Stage 2 strict-clean conversion and splitting](TACO_STAGE_2.md) after representative annotation review. This superseded the Stage 1 export/split stop boundary below only. Its whole-image quarantine policy and validated reference export remain unchanged.

Additional owner-authorized preparation now covers auditing/visualizing the existing external TACO dataset under the eight object labels in [the versioned taxonomy](../config/taco_taxonomy.v1.json). This leaves inference categories/behavior intact and creates no training examples, final splits, downloads or fine-tuning. [TACO evidence and pending handling decisions](TACO_PREPARATION.md) define this bounded exception to the earlier stop-before-optimization boundary.

- Detect trash in still images.
- Functionally check pretrained GroundingDINO-T (Swin-T / Tiny configuration) through Hugging Face Transformers and direct text prompts. Automatic grouping, training, fine-tuning, and alternative models are outside this baseline.
- Establish documentation and verifiable phase gates before model work.

Phase 0 is complete and Phase 1 planning is published. Phase 2A technical environment preparation is complete on supplied evidence, recorded in the [snapshot](PHASE_2A_ENVIRONMENT_SNAPSHOT.md); qualitative detection quality remains unresolved. Codex owns implementation. Phase 2C has recorded Windows CUDA execution; current authorization is Phase 2E functional checking and handoff under [PHASES.md](PHASES.md). See [2E results](PHASE_2E_RESULTS.md); stop before Phase 3 decisions or optimization. No automatic grouping, training or application modules.

## Initial exclusions

No pothole or road-surface damage detection, video or real-time processing, incident-management workflow, geographic aggregation, hotspot ranking, frontend, backend, or deployment is included in the initial evaluation. No training, fine-tuning, or alternative-model implementation is approved.

## Required categories and multiple regions

| Category | Intended meaning | Example |
| --- | --- | --- |
| `garbage_pile` (Garbage Pile) | Concentrated accumulation of discarded waste, typically overlapping, stacked, or densely collected | A substantial roadside heap of mixed garbage |
| `individual_litter` (Individual Litter) | A single separately identifiable discarded object not assigned to a group | An isolated discarded bottle, wrapper, aluminum can, or bag |
| `group_of_individual_litter` (Group of Individual Litter) | Multiple separately distinguishable discarded objects that are spatially associated without necessarily forming a dense pile | Several bottles, wrappers, and cans scattered close together |

These are required project outputs, not guaranteed pretrained model classes. Category boundaries, a proposed minimum of two visually distinct items for a group, ambiguity, membership, and overlap rules are visual-review guidance and operational proposals, not required annotation work. Mark uncertain cases rather than force a classification. Nearby litter is not automatically a pile, and a pile is not automatically a group of individual litter.

One image can contain multiple independent regions and any mixture of all three categories. Preserve object-level detections separately from group-level regions. A group region and its members must not be double-counted as independent trash regions. Physical object counts must not be inferred from a group box when members cannot be resolved.

Illustrative requirement only: one isolated bottle plus four distinguishable litter objects in one separate group, with no pile, should yield 1 Individual Litter region, 1 Group of Individual Litter region, 0 Garbage Pile regions, 5 distinct physical litter objects, and 2 independent trash regions, assuming all objects and memberships are correctly identified. This is not a model prediction or benchmark.

Direct prompts must be evaluated first. They may produce individual object boxes without a group box, or confuse a group with a pile. Whether GroundingDINO-T can meet these requirements directly remains experimental. Spatial grouping is a conditional later option, not a required Phase 1 implementation.

## Current inputs, outputs, and unresolved boundaries

The owner supplies team photographs with no annotations or ground-truth boxes, downloads/extracts them manually from Google Drive, and places them in a configurable project-relative folder (proposed `data/input/`). Preserve originals. No dataset collection, Drive integration, required annotation work, benchmark splits, precision/recall/mAP, or numeric accuracy targets belong to this functional check.

Plan an in-memory display of photographs with predicted boxes, labels, and individual model-score percentages, a readable summary, per-image JSON, and run/environment metadata. No automatic annotated-image copies or source overwrites. Scores are not measured accuracy, calibrated correctness probabilities, or image-area percentages. Inference and display statuses are distinct; invalid numeric values become null with diagnostics, and invalid candidates are excluded from displayed boxes/valid summaries under the [contract](PHASE_1_IMPLEMENTATION_PLAN.md#7-proposed-json-contract).

Execution is on the owner’s Windows computer. Supplied preflight now reports GTX 1660 Ti, 6144 MiB VRAM and driver 472.19, superseding the earlier RTX 2080 Super report. Supplied output now confirms Python 3.11.9 x64 and a local `.venv`; supplied dependency/import and basic CPU/CUDA tensor checks passed; native viewing/normal return and exact metadata capture also passed, completing technical environment preparation; model-memory fit and detector behavior remain unverified; observed free VRAM is a time-specific measurement, not a capacity guarantee. Checkpoint/processor metadata and revision are verified in the 2A snapshot; prompts, thresholds and conservative mapping/duplicate policies remain unvalidated proposals. Do not assume all roadside objects are discarded trash or that a detection is an incident. Confirm the supplied photos may be processed without turning this task into data collection or required annotation.

A still-image detection alone does not establish location, repeated observations, incident identity, or hotspot frequency. Any future hotspot work belongs to the broader system and will need an approved data and aggregation policy outside this repository.

## Change control

The project owner must explicitly approve scope expansion and phase transitions. Record decisions and their basis in [DECISIONS.md](DECISIONS.md), and update [PHASES.md](PHASES.md) and [TEST_PLAN.md](TEST_PLAN.md) together when needed.
