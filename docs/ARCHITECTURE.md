# Model inference architecture

Status: proposed design for later authorized evaluation; no inference pipeline is implemented.

## Intended inference path

1. Load an authorized still image and preserve an image identifier and original dimensions.
2. Validate decoding and input constraints, then apply preprocessing required by the selected model implementation. Preserve the transform needed to map predictions back to original image coordinates.
3. Provide a documented trash-related text prompt to a pretrained Grounding DINO-family model. The exact model and checkpoint remain unresolved; record their provenance, license, and revision when selected.
4. Run inference with the selected runtime and device. Training and parameter updates are outside this path.
5. Apply explicitly recorded score and text matching thresholds and any supported postprocessing. Their definitions and values require investigation; do not assume scores are calibrated probabilities.
6. Produce candidate bounding boxes mapped to the original image, associated text labels, model-provided scores, and a reproducibility record. The exact storage schema is unresolved.
7. Inspect detections against annotations and summarize measured performance and failure cases according to [TEST_PLAN.md](TEST_PLAN.md).

Conceptual flow: still image + prompt → model-specific preprocessing → pretrained model inference → documented filtering → original-coordinate detections → evaluation evidence.

## Reproducibility record

A future experiment should capture model/checkpoint identity and revision, code and dependency versions, device, image identifiers, preprocessing, prompts, thresholds, postprocessing, random settings where relevant, timing methodology, and evaluation protocol. Do not store secret values or restricted images in Git.

## Boundaries and failure behavior

Image validation, inference, and evaluation are separate responsibilities within a future local experiment; they do not imply separate deployed services. Invalid input should produce a clear recorded failure rather than fabricated detections. An empty detection list is a valid possible output and must remain distinguishable from an execution failure.

There is no frontend, backend, incident database, geolocation service, or hotspot computation in this initial design. Model licensing, hardware suitability, resource needs, output schema, and error-handling details remain unresolved. Select them only during the appropriate authorized phase.
