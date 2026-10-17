# Scope

## Broader academic proposal

The broader BANTAI proposal covers roadside garbage, potholes, road-surface damage, incident management, and frequency-based hotspot prioritization. These capabilities describe the research direction rather than the current repository's implementation commitment.

## Repository responsibility

This repository is exclusively for AI modelling and computer vision: pretrained model evaluation/selection, preprocessing, object detection, necessary postprocessing, accuracy/performance evaluation, approved optimization and potential fine-tuning, and reusable detection outputs for future integration. Frontend, backend application modules, reporting, incident management, central databases, and hotspot ranking belong outside this repository, including in later phases. Road-hazard detection, video processing, and additional model development need separate future approval.

## Approved initial direction

- Detect trash in still images.
- Evaluate pretrained GroundingDINO-T (Swin-T / Tiny configuration) through direct text prompts before considering spatial grouping, training, alternative models, or integration.
- Establish documentation and verifiable phase gates before model work.

Phase 0 authorizes documentation only. The initial direction does not authorize immediate execution of later phases.

## Initial exclusions

No pothole or road-surface damage detection, video or real-time processing, incident-management workflow, geographic aggregation, hotspot ranking, frontend, backend, or deployment is included in the initial evaluation. No training, fine-tuning, or alternative-model implementation is approved.

## Required categories and multiple regions

| Category | Intended meaning | Example |
| --- | --- | --- |
| Garbage Pile | Concentrated accumulation of discarded waste, typically overlapping, stacked, or densely collected | A substantial roadside heap of mixed garbage |
| Individual Litter | A single separately identifiable discarded object not assigned to a group | An isolated discarded bottle, wrapper, aluminum can, or bag |
| Group of Individual Litter | Multiple separately distinguishable discarded objects that are spatially associated without necessarily forming a dense pile | Several bottles, wrappers, and cans scattered close together |

These are required project outputs, not guaranteed pretrained model classes. Operational definitions, annotation policy, minimum grouping criteria, ambiguity, membership, and overlap rules must be finalized before evaluation. Nearby litter is not automatically a pile, and a pile is not automatically a group of individual litter.

One image can contain multiple independent regions and any mixture of all three categories. Preserve object-level detections separately from group-level regions. A group region and its members must not be double-counted as independent trash regions. Physical object counts must not be inferred from a group box when members cannot be resolved.

Illustrative requirement only: one isolated bottle plus four distinguishable litter objects in one separate group, with no pile, should yield 1 Individual Litter region, 1 Group of Individual Litter region, 0 Garbage Pile regions, 5 distinct physical litter objects, and 2 independent trash regions, assuming all objects and memberships are correctly identified. This is not a model prediction or benchmark.

Direct prompts must be evaluated first. They may produce individual object boxes without a group box, or confuse a group with a pile. Whether GroundingDINO-T can meet these requirements directly remains experimental. Spatial grouping is a conditional later option, not a required Phase 1 implementation.

## Unresolved boundaries

The operational definition of trash, treatment of ambiguous objects, image sources and usage rights, annotation policy, dataset composition, and evaluation success thresholds require decisions before the relevant experiments. Do not assume that all roadside objects are trash or that a detection is an incident.

A still-image detection alone does not establish location, repeated observations, incident identity, or hotspot frequency. Any future hotspot work belongs to the broader system and will need an approved data and aggregation policy outside this repository.

## Change control

The project owner must explicitly approve scope expansion and phase transitions. Record decisions and their basis in [DECISIONS.md](DECISIONS.md), and update [PHASES.md](PHASES.md) and [TEST_PLAN.md](TEST_PLAN.md) together when needed.
