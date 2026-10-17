# Scope

## Broader academic proposal

The broader BANTAI proposal covers roadside garbage, potholes, road-surface damage, incident management, and frequency-based hotspot prioritization. These capabilities describe the research direction rather than the current repository's implementation commitment.

## Approved initial direction

- Detect trash in still images.
- Evaluate a pretrained Grounding DINO-family model before considering training, alternative models, or integration.
- Establish documentation and verifiable phase gates before model work.

Phase 0 authorizes documentation only. The initial direction does not authorize immediate execution of later phases.

## Initial exclusions

No pothole or road-surface damage detection, video or real-time processing, incident-management workflow, geographic aggregation, hotspot ranking, frontend, backend, or deployment is included in the initial evaluation. No training, fine-tuning, or alternative-model implementation is approved.

## Unresolved boundaries

The operational definition of trash, treatment of ambiguous objects, image sources and usage rights, annotation policy, dataset composition, and evaluation success thresholds require decisions before the relevant experiments. Do not assume that all roadside objects are trash or that a detection is an incident.

A still-image detection alone does not establish location, repeated observations, incident identity, or hotspot frequency. Any future hotspot work will need an approved data and aggregation policy.

## Change control

The project owner must explicitly approve scope expansion and phase transitions. Record decisions and their basis in [DECISIONS.md](DECISIONS.md), and update [PHASES.md](PHASES.md) and [TEST_PLAN.md](TEST_PLAN.md) together when needed.
