# Stage 4 Step 1B — Coordinate correction, 2026-10-20

The owner authorizes correction of the two Step 1 coordinate defects, independent
regression checks and validation-only reevaluation. No optimizer updates, test
split access, promotion, Step 2 visual analysis or additional fine-tuning occurs.
Starting branch: `feat/stage4-real-image-training`, HEAD `087e361`; PR #18 lineage
is retained. The pre-existing owner edit to `config/detection.json` is preserved.

## Root causes and corrections

Transformers 4.51.3 normalizes processed annotations to the padded canvas, while
Grounding DINO proposals/predictions use the valid image frame. Native source
evidence is `GroundingDinoModel.generate_encoder_output_proposals` (division by
valid width/height), decoder reference scaling by valid ratios, and the native
processor's direct original-size prediction scaling.

`training_smoke.valid_image_loss_targets` verifies the processor's padded targets
and original image/category identities, then multiplies boxes by
`[padded_w/valid_w, padded_h/valid_h, padded_w/valid_w, padded_h/valid_h]`.
It independently checks the result against original COCO coordinates normalized
by original width/height. Both real-data `collate` and synthetic `prepare_batch`
use this helper. Other label metadata, source annotations and pixel inputs remain
unchanged. The existing adapter supplies these labels to both native loss branches.

`taco_evaluation.decode_predictions` now scales native normalized corners directly
by original width/height, then clips/rejects invalid boxes. The former additional
padding expansion is removed. Protocol `bantai.taco-evaluation.v2` explicitly names
the box frame; prompt, class scores, argmax, top 100, NMS-off and confidence rules
are unchanged. Functional inference, adapter code, architecture and coefficients
are unchanged.

For original 640x320, COCO `[160,80,160,80]`, resized 384x192 and padded 384x384,
loss targets are now `[.375,.375,.25,.25]`, replacing the incorrect
`[.375,.1875,.25,.125]`. The independently specified native prediction decodes
to the original box; the old decoder produced `[160,160,160,160]`.

## Software verification

New `tests/test_taco_coordinates.py` has eight processor/native-loss tests:
landscape, portrait and padded square; independently specified coordinates;
all eight class IDs; multiple/small/bottom/right-edge objects; legitimate clipping
and invalid-width rejection; pixel footprint/mask alignment; empty/identity checks;
native proposal/reference semantics; zero regression loss in both native branches;
and perfect AP50=1 versus wrong-frame AP50=0. Predictions are not copied from
processed targets. Native postprocessing is a separate oracle.

Actual commands in the existing isolated environment:

```powershell
.venv-training/Scripts/python.exe -B -m unittest discover -s tests -p test_taco_coordinates.py -v
.venv-training/Scripts/python.exe -B work/step1b/check_regressions.py
.venv-training/Scripts/python.exe -B work/step1b/check_software.py
.venv-training/Scripts/python.exe -B work/step1b/evaluate_baseline.py
```

The `work/step1b` scripts/logs are ignored local execution evidence, not distributed
project entry points. The regression harness loads original functions from
`087e361` into temporary in-memory modules, without replacing source/library files.
All eight corrected tests pass. Original targets and original decoder each produce
five failure events including subtests, independently; neither variant errors.
The initial clipping fixture used an out-of-domain center of 1.2 and was correctly
rejected; it was corrected to valid-domain boundary/zero-width cases without
relaxing prediction validation.

Broader discovery: **168 tests, 165 passed, zero failures/errors, three skipped**.
One skip is the existing Windows symlink privilege limitation. Two toy optimizer
tests are explicitly skipped for this task's no-update boundary. The local harness
guards AdamW/SGD steps and redirects trainer output to a temporary directory.
Do not run unfiltered optimizer tests under this boundary. No dependency changes.

## Validation evidence and historical comparison

New private record:
`runs/training/step1b-evaluation-c8b127d8bf364c7dbb735795cd81c06d/record.json`.
`baseline-corrected.json` retains complete local diagnostics; the original Stage 4
run and its JSON/checkpoints are unchanged. This is a new evaluation, not a rewrite
of historical evidence. Source fingerprints, v2 protocol, dataset/rights hashes,
runtime and original checkpoint hashes are bound in the record.

The pinned pretrained baseline ran on GTX 1660 Ti, FP32, batch 1, 256/384 resize,
384-square padding, identical eight-class prompt and the same 47 validation images /
75 annotations. Source hashes/orientation and rights gates passed. Exactly 47 photo
reads; zero optimizer updates; no test loader. No cloud/upload/publication.

Metrics are fractions, not percentages. Historical values used incorrect geometry.

| Model / evaluation | AP | AP50 | AP75 | AR100 |
| --- | ---: | ---: | ---: | ---: |
| Pretrained, historical v1 | 0.000298934 | 0.001329652 | 0 | 0.007884615 |
| Pretrained, corrected v2 | 0.111576016 | 0.158252880 | 0.135975042 | 0.363077235 |
| Epoch 1, historical v1 | 0.000006713 | 0.000016781 | 0 | 0.003846154 |
| Epoch 1, corrected v2 | blocked | blocked | blocked | blocked |
| Epoch 2, historical v1 | 0.000057591 | 0.000138055 | 0.000052946 | 0.015576923 |
| Epoch 2, corrected v2 | blocked | blocked | blocked | blocked |

| Class | Validation objects | Corrected pretrained AP | Corrected pretrained AR100 |
| --- | ---: | ---: | ---: |
| cigarette | 24 | 0.014851485 | 0.012500000 |
| plastic_bottle | 8 | 0.038241717 | 0.400000000 |
| plastic_bag | 2 | 0.236228623 | 0.550000000 |
| plastic_wrapper | 11 | 0.094467223 | 0.490909091 |
| plastic_film | 13 | 0.001574350 | 0.076923077 |
| drink_can | 7 | 0.108415842 | 0.214285714 |
| plastic_cup | 5 | 0.340655617 | 0.600000000 |
| styrofoam_piece | 5 | 0.058173267 | 0.560000000 |

Corrected baseline small-object AP/AR100 are both zero. Medium AP is 0.005940594.
Corrected baseline validation loss is 10.553226 versus historical 19.348786 at
identical pretrained weights: target semantics changed, so this is not learning.
Allocator peak: 951,008,768 allocated / 1,019,215,872 reserved bytes. Evaluation
lifecycle was 73.316 seconds including model loading, hashing and publication;
this is not isolated throughput and excludes initial environment preflight.

All 42,300 baseline query scores and category IDs match historical values exactly;
maximum score difference is zero. Invalid clipped boxes drop from 14,703 to zero.
The same 47 IDs and 4,700 retained predictions are present. Geometry, rather than
score/prompt changes, explains the corrected baseline measurement difference.

### Checkpoint comparison blocker

Both epoch bundles pass all ten historical artifact hashes. However,
`verify_checkpoint` rejects a context containing the corrected three source
fingerprints and v2 protocol: **Incomplete checkpoint or changed
dataset/rights/settings/runtime/source**. The historical replay additionally
requires exact original `source_identity`. No optimizer/checkpoint restore or
model loading was attempted for those bundles under the changed context.

Per the owner's explicit stop-if-source-binding-blocks rule, this part is stopped.
The checkpoint verifier, adapter manifest and historical evidence are unchanged;
passing the old context as if it described corrected execution is not a solution.
A separately approved evaluation-only provenance contract that verifies historical
weights while recording the new evaluator is required before the epoch comparison.
This must not relax training-resume integrity or rewrite historical manifests.

### Score suppression and limitations

Historical all-query score distributions (same 42,300 queries per model):

| Model | Median | P99 | Maximum | Queries >=0.25 |
| --- | ---: | ---: | ---: | ---: |
| Pretrained (also identical corrected scores) | 0.019574 | 0.187558 | 0.596656 | 216 |
| Epoch 1, historical | 0.029931 | 0.069502 | 0.184526 | 0 |
| Epoch 2, historical | 0.031275 | 0.067896 | 0.123536 | 0 |

The high-score tail contracted, while the median increased. Suppression above the
diagnostic threshold is real; its cause is not isolated. Corrected epoch geometry
and class AP remain unmeasured. Earlier weights were trained against incorrect
targets; an evaluation fix cannot undo that. Tiny validation classes, positive-only
strict-clean selection and low resolution limit conclusions. Successful geometry
tests and higher baseline AP do not establish adequate detection accuracy.

## Preservation and next action

The preservation snapshot covers 490 existing files, including all historical run
artifacts, original pretrained snapshot, train/validation source photographs,
export metadata excluding test JSON, taxonomy, owner settings, private rights
record and project source/docs. Package inventory records 38 training packages.
Final checks passed: zero unexpected file differences and all 38 package versions
unchanged, allowing only the explicitly changed source/docs files. The original
annotation/source and dataset context still match the historical contract. All 97
relative links checked in changed Markdown files resolve. No test JSON/photo is opened.

Changed project scope: three training/evaluation modules, the new coordinate tests,
this report, README current-state notice, handoff and decision register. Local probe
scripts/logs and generated evaluation JSON remain ignored. No commit, push, merge
or checkpoint promotion is performed in Step 1B.

**Next proposed action:** qualify an explicit evaluation-only provenance path for
the two historical checkpoints, then finish the fixed comparison before Step 2.
No further training or visual analysis is started by this task.
