# Model inference architecture

Status: proposed design for later authorized evaluation; no inference pipeline is implemented.

## Intended inference path

1. Load a supported, authorized still image; retain its identifier and original dimensions.
2. Validate decoding and preprocess according to the selected implementation; retain the transform back to original coordinates.
3. Prepare documented text prompts for Garbage Pile, Individual Litter, and Group of Individual Litter.
4. Run pretrained GroundingDINO-T (Swin-T / Tiny configuration) inference without training updates, using the verified implementation, checkpoint, runtime, and device.
5. Obtain candidate bounding boxes, phrase associations, and model scores.
6. Apply recorded thresholds and appropriate prediction postprocessing; score semantics and filtering choices require verification. Do not assume calibrated probabilities.
7. Preserve object-level and candidate region-level detections separately, mapped to original coordinates.
8. Inspect overlaps, possible duplicate assignments, and candidate object-to-region memberships. Do not double-count a group and its members as independent regions. Membership and overlap resolution policies remain unresolved.
9. Generate reproducible structured detection results with prediction provenance and failure status. Exact schema and counting rules require Phase 1 decisions.
10. Compare predictions with appropriately annotated reference images under [TEST_PLAN.md](TEST_PLAN.md).

## Text-prompt-first experiment and conditional grouping

Direct text-guided detection is the approved first strategy. Illustrative prompt candidates include “garbage pile,” “discarded plastic bottle,” “scattered litter,” “group of discarded objects,” and “roadside garbage accumulation.” They are not final prompts. Phase 1 must investigate vocabulary, formatting, separate versus combined descriptions, thresholds, and the evaluation protocol.

A model phrase association is a prediction, not proof of an approved category or grouping definition. Preserve raw predictions separately from any category mapping, inferred classification, membership, or derived region, and record the rule/provenance of each derived result. Support mixed-category images and multiple independent regions as specified in [SCOPE.md](SCOPE.md).

GroundingDINO-T may return member boxes without one group box and may confuse scattered groups with piles. Measure localization and category distinction rather than assuming custom grouping understanding.

Only after baseline evaluation and explicit owner authorization may an additional spatial grouping module be introduced. DBSCAN, scale-normalized proximity grouping, custom bounding-box grouping, and hybrid approaches are candidates, not selected dependencies or mandatory Phase 1 components. Select an approach using evidence rather than theoretical suitability alone.

## Reproducibility record

A future experiment should capture model/checkpoint identity and revision, code and dependency versions, device, image identifiers, preprocessing, prompts, thresholds, postprocessing, random settings where relevant, timing methodology, and evaluation protocol. Do not store secret values or restricted images in Git.

## Boundaries and failure behavior

Image validation, inference, and evaluation are separate responsibilities within a future local experiment; they do not imply separate deployed services. Invalid input should produce a clear recorded failure rather than fabricated detections. An empty detection list is a valid possible output and must remain distinguishable from an execution failure.

There is no frontend, backend, incident database, geolocation service, or hotspot computation in this initial design. Exact implementation, checkpoint source/revision and configuration, licensing, dependency versions, hardware suitability, resource needs, output schema, and error-handling details remain unresolved. The owner reports an NVIDIA RTX 2080 Super development GPU; actual local availability and compatibility are not yet verified. Select them only during the appropriate authorized phase.
