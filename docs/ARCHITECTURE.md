# Model inference architecture

Status: proposed detector workflow for later phase authorization under Codex ownership. Phase 0 is complete and Phase 1 planning is published; Phase 2A environment/package/device/viewer readiness is verified to its preparation scope in the [snapshot](PHASE_2A_ENVIRONMENT_SNAPSHOT.md), while model execution, memory fit and successful detection remain unverified. Phase 2A currently authorizes local Windows environment setup/readiness only; the detector workflow below is not authorized for execution.

## Intended functional-check path

1. Discover supported still images in a configurable project-relative folder (proposed `data/input/`). The owner manually downloads/extracts unannotated team photos from Google Drive; no integration, collection, or annotation workflow is required.
2. Decode and normalize EXIF orientation in memory, convert to RGB, record original-display dimensions and source identity, and preprocess with the verified processor. Preserve source bytes and the coordinate transform back from model input.
3. Prepare documented text prompts for `individual_litter`, `group_of_individual_litter`, and `garbage_pile`; these are project categories, not guaranteed native model classes.
4. Run pretrained GroundingDINO-T (Swin-T / Tiny) through Hugging Face Transformers in inference-only mode. Model and processor must use the same verified immutable checkpoint revision. Prefer verified GPU execution on the owner’s Windows computer (latest supplied preflight: GTX 1660 Ti, 6 GiB VRAM; earlier RTX 2080 Super report superseded), with documented CPU fallback.
5. Obtain candidate boxes, raw phrases, and model scores. Apply recorded `threshold` and `text_threshold` using the selected library version, and map coordinates to EXIF-oriented original-display pixels.
6. Preserve raw phrases and finite raw values. Serialize each invalid numeric coordinate/score as null with a reason; any invalid coordinate makes the entire display box null. Flag invalid scores, boxes, or geometry; exclude invalid candidates from displayed boxes and valid-detection summaries. Follow the detailed [JSON contract](PHASE_1_IMPLEMENTATION_PLAN.md#7-proposed-json-contract), including strict serialization without NaN/Infinity.
7. Apply only the proposed explicit alias mapping and conservative identical-box/same-phrase duplicate policy. Retain unmapped candidates and diagnostics. Keep object-level and candidate region-level predictions distinct; inspect overlaps without inferring membership or independent-region counts. No automatic grouping or general non-maximum suppression algorithm is selected.
8. Atomically write per-image JSON and appropriate run/environment metadata before opening the viewer. Top-level `status` records inference/processing outcome: `ok`, `no_detections`, `error`, or `skipped_unsupported`. The initial result includes display `pending` for successful processing with viewing enabled, `disabled` for deliberate disablement after successful processing, or `not_attempted` for failed/skipped processing.
9. With display enabled by default and pending already saved, open a lightweight native viewer with in-memory boxes, labels, and individual model-score percentages. Attempt display and atomically update JSON to `shown` or display `error`. Deliberate disablement records `disabled`; failed/skipped processing records `not_attempted`. A stopped process leaving `pending` has an unfinished display attempt, not proven display success. Display failure never erases detections or changes successful inference to failure. Exclude viewer waiting from inference timing.
10. Produce a readable category/score summary and visually inspect detections under [TEST_PLAN.md](TEST_PLAN.md). Preserve originals; do not automatically save annotated image copies. No annotations, held-out benchmark splits, formal accuracy metrics, or numeric accuracy targets are required.

## Text-prompt-first strategy and conditional grouping

Illustrative prompts include “garbage pile,” “discarded plastic bottle,” and “group of scattered litter.” The plan proposes separate versus combined descriptions and configurable thresholds; neither wording nor values are validated settings. Exact aliases can leave partial or combined phrases unmapped; keep them for inspection.

Model phrase associations and lexical mappings are predictions, not confirmed category identity or membership. A direct group prediction may enclose several items, but member boxes need not be returned. Preserve both levels when present; never synthesize a group box or physical-object count from missing members. Keep physical-object and independent-region counts null when unresolved, including mixed-category scenes. See [SCOPE.md](SCOPE.md).

Identical-box/same-phrase collapse does not remove near-duplicate boxes. IoU and containment flags are review heuristics, not clustering, membership evidence, or an approved general suppression algorithm. Overlapping groups and members cannot simply be summed as independent regions.

GroundingDINO-T may miss group regions or confuse scattered groups and piles. Report those limitations. DBSCAN, proximity grouping, custom bounding-box grouping, and hybrids remain conditional future options requiring baseline evidence and separate authorization. If later authorized, derived classifications/regions must record origin, members, and method separately from model predictions; a derived group has no invented model confidence.

## Reproducibility and outcomes

Verify exact checkpoint/configuration/conditions/safetensors/revision, local Windows Python/driver/memory, CUDA wheel/GPU execution, and native viewer. Record exact resolved dependencies plus CUDA installation source, processor/preprocessing settings, EXIF convention, prompts, thresholds, device, dtype, custom-kernel setting, mapping/duplicate policy, fallback details, and timing scope. Package pins and operational defaults in the plan remain proposals pending local verification; publication is not compatibility evidence.

A model score ×100 is a display percentage, not accuracy, calibrated correctness probability, or image-area coverage. `no_detections` means successful inference with zero above-threshold candidates, not proof of no trash. Invalid-candidate-only output is diagnosed under the contract, not mislabeled as no detections. Display state and its error are recorded separately from processing state and its error. Source, result, metadata, and display failures must remain observable.

Formal annotated evaluation may be considered only as separately scoped future work. No frontend/backend application modules, API, database, road anomalies, video, cameras, incident consolidation, or hotspot computation belong to this workflow.
