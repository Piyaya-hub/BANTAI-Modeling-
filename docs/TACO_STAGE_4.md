# Stage 4 — Real-image fine-tuning and initial evaluation

## Current outcome

Owner authorization D-041 supersedes Stage 3D's stop-before-real-data boundary, subject to the owner's explicit image-use condition. This is one continuous implementation stage, not a new assessment phase. Branch `feat/stage4-real-image-training` builds directly on Stage 3D `22bcdbc` (open PR #17), carrying the unmerged Stage 3A–3D work above fetched main `077657c`.

The runnable loader, corrected-objective trainer, pilot, progress, complete resumable checkpoints and fixed validation evaluator are implemented. **Real-photo processing, the pilot, 2–3 epochs and real validation metrics remain BLOCKED by missing documented photo/annotation-use authorization. No fine-tuned TACO checkpoint exists.** Metadata inspection does not clear rights. No test photographs or test split JSON are loaded by these commands. Existing source data, eight labels, split exports, inference settings/environment, pretrained cache and Stage 3D adapter are preserved. The owner's local `config/detection.json` edits remain excluded from commits.

## Files and behavior

| File | Purpose |
| --- | --- |
| [taco_training_data.py](../scripts/taco_training_data.py) | Verify original annotation/export/taxonomy hashes, bind train/validation IDs to an explicit rights record, load external stored-raster RGB photos read-only, reject changed/missing/unsafe sources and invalid/crowd/ignored targets, and verify preprocessing coordinates/counts/classes. |
| [train_taco.py](../scripts/train_taco.py) | Rights gate, all-photo coordinate preflight, unchanged-pretrained validation, bounded real-image memory pilot, discard pilot weights/optimizer, then 2–3 epochs from the unchanged base; validation/checkpoints/progress and strict status reports. |
| [training_checkpoint.py](../scripts/training_checkpoint.py) | Native HF model/processor plus adapter context, AdamW state, Python/NumPy/CPU/CUDA RNG, deterministic epoch order and next-batch/global-step position. Immutable artifacts, complete hash manifest last, latest pointer only after completion. |
| [taco_evaluation.py](../scripts/taco_evaluation.py) | Fixed ordinary-subword class scoring, inverse geometry, COCO bbox metrics, raw query diagnostics and annotations-relative errors. Separate from the existing inference postprocessor. |
| [qualify_taco_trainer.py](../scripts/qualify_taco_trainer.py) | Actual pinned-model CUDA update/save/reload/continued-update qualification using generated images only. |
| [test_taco_training.py](../tests/test_taco_training.py) | Rights/no-read gates, source/annotation failures, geometry/span decoding, empty predictions, metrics, RNG/optimizer continuation and toy-update epoch/resume orchestration. |
| [taco_training_rights.example.json](../config/taco_training_rights.example.json) | Deliberately unapproved template; no inferred permissions. Actual evidence stays in ignored local paths. |
| [requirements-training-evaluation.txt](../requirements-training-evaluation.txt) | Only additional training-environment package: PyPI Windows CPython 3.11 x64 `pycocotools==2.0.10`, exact wheel SHA-256. No base package reinstall. |

COCO area is retained as segmentation/object area; it is not replaced with bounding-box area. Boxes remain `[x,y,width,height]` on the original stored raster, with no EXIF transpose, annotation clipping or silent object removal. All 268 train/validation photographs must decode, match recorded file hashes/dimensions/orientation, and preserve all target identities and coordinates through resizing/padding **before any optimizer update**. This geometric check does not substitute for human annotation correctness review.

## Narrow rights check — blocking gate

Metadata-only reinspection of the unchanged export on 2026-10-19 found:

| Split | Images / objects | Missing license | Unspecified `CC` | OpenLitterMap ODbL notice |
| --- | ---: | ---: | ---: | ---: |
| Train | 221 / 397 | 133 | 22 | 66 |
| Validation | 47 / 75 | 28 | 8 | 11 |
| Required scope | **268 / 472** | **161** | **30** | **77** |

The preserved source records contain Flickr URLs, but neither an image-by-image permission ledger nor an approved institutional/legal use basis was supplied. No Flickr images were downloaded or opened during this check. The [TACO README](https://github.com/pedropro/TACO/blob/master/README.md) identifies Flickr-hosted photographs; its [MIT software license](https://github.com/pedropro/TACO/blob/master/LICENSE) does not establish blanket photo permission. The [original Zenodo record](https://zenodo.org/records/3587843) did not expose usable photo/annotation authorization in its rendered Rights section. `CC` lacks variant/version and attribution details. The [ODbL text](https://opendatacommons.org/licenses/odbl/1-0/) distinguishes database rights from individual contents, so its presence alone does not establish photo rights. These are unresolved evidence gaps, not a legal conclusion that the intended use is prohibited.

**Required owner input:** documented permission or an appropriately approved lawful-use basis covering every retained training and validation photo **and** annotation/database use for this local academic experiment. Identify the actual approving rightsholder/institutional/legal authority, approval date, basis/evidence reference, permitted scope and obligations (credits/notices, restrictions, retention). Publication, cloud transfer and redistribution are outside this request. Generic instruction to proceed, repository MIT, an unspecified CC string, or filling an example with invented facts does not satisfy this gate. An approved lawful-use basis may be recorded instead of individual licenses when the authorized review actually covers this scope.

The rights record binds to the exact manifest, original annotation digest, split digests and IDs. It requires explicit photo and annotation/database clearance and the scope `local_academic_train_validation_no_publication`. This is an auditable owner declaration, not software verification of legal validity or automatic implementation of every obligation. Resolve operational obligations before marking it approved; the trainer never uploads/publishes or exports annotated copies. No real-image bytes or model loads occur if the record is absent/unapproved/mismatched. Test split permission is not requested because it remains untouched.

## Fixed initial experiment

- Pinned `IDEA-Research/grounding-dino-tiny`, revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, Transformers 4.51.3, local cache only, custom kernels disabled; `.venv-training` only.
- Every supervised loss goes through the unchanged verified `EncoderOutputAdapter`; plain model forwards are never passed labels. Native loss formula/weights, 900 queries, detached decoder initialization, architecture and eight ordered prompt spans stay unchanged.
- FP32, frozen vision/BERT parameters; native training/eval module modes, no augmentation, AMP, scheduler or accumulation. AdamW LR `1e-5`, weight decay `1e-4`, `foreach=false`, global clip norm 1, seed 42, workers 0.
- Default batch 1; aspect-preserving shortest edge 256 / longest edge 384, then bottom/right square padding to 384. Fixed padding protects the 900-proposal requirement even for narrow photos. It changes neither aspect ratio nor target alignment. Lower resolution limits tiny-object evaluation; it is a bounded initial experiment, not an optimized default. Explicit batch 2 is permitted by the adapter but not qualified at this real-photo resolution.
- Pretrained baseline first, then default three pilot updates on the most object-dense training photos (tie by image ID). Require finite losses/gradients/updated weights and nonzero encoder proposal/box updates; record memory on each step, reset allocator peaks for pilot/full run. Abort/report OOM or instability without changing settings. Pilot weights/optimizer are discarded before the full run.
- Default two epochs: 442 full-run updates at batch 1, plus three discarded pilot updates. Three epochs require explicit `--epochs 3` from the outset. Same fixed validation protocol runs on all 47 validation photos before training and after each epoch. Test images are never used. No checkpoint is automatically promoted to inference.

GPU progress prints epoch/batch/total, training loss and allocator peak; validation has per-image progress and epoch AP/loss summaries. Step timing includes synchronization/gradient audits; report lifecycle time includes hashing, preflight, model loads, pilot, baseline, training, validation and checkpoints. This is not a steady-state throughput benchmark. CUDA allocator measurements exclude desktop/driver allocations; actual real-photo peak/time remain unrun.

### Evaluation protocol

Mean sigmoid score over each class's ordinary subwords, one argmax class per query, no NMS, score floor 0, highest-scoring 100 **valid** boxes globally per image. Invert the saved padded/resized frame to original xywh and clip predictions only to image bounds; do not clip ground truth. Preserve all 900 raw query records with query IDs, raw incomplete/unmapped phrases (ordinary tokens above 0.10), scores and box validity separately. Raw phrase threshold does not filter AP predictions. This evaluation-only rule does not change inference prompts/thresholds or raw phrase mapping.

[COCO evaluator conventions](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py): bbox AP@[.50:.95], AP50/AP75, area metrics and AR1/10/100; per-class AP/AR/support. Source COCO object/segmentation areas remain the ground-truth area convention. Undefined metrics are JSON null. `pycocotools` loading handles zero detections explicitly. Saved annotation-relative diagnostics use fixed score 0.25 / IoU 0.5 for unmatched predictions, wrong classes, duplicates and missed annotations. Those thresholds are protocol choices, not validation-selected settings or calibrated correctness probabilities. Neither annotations-relative errors nor synthetic metric fixtures establish visible-photo truth; human raw-phrase/box/error review remains pending.

Validation bag support is only two objects/two images; cup five objects/four images and styrofoam five objects/five images. Results will be noisy, selection-biased toward strict-clean scenes and will not establish pile/group capabilities or annotation completeness. Excluded/unlabeled whole images remain quarantined, never silently used as training negatives. No export, taxonomy, split or excluded-object policy is changed.

## Commands

Use the preserved Windows environment. The new evaluator wheel was downloaded/installed once, with no dependencies changed:

```powershell
.venv-training/Scripts/python.exe -m pip download --no-deps --only-binary=:all: --dest work/stage4-wheels pycocotools==2.0.10
.venv-training/Scripts/python.exe -m pip install --no-deps --no-index --find-links work/stage4-wheels --require-hashes -r requirements-training-evaluation.txt
.venv-training/Scripts/python.exe -m pip check
.venv-training/Scripts/python.exe -m unittest discover -s tests -v
.venv-training/Scripts/python.exe scripts/qualify_taco_trainer.py --determinism-probe
.venv-training/Scripts/python.exe scripts/qualify_taco_trainer.py
```

The wheel hash is `03c3aacec2a6aa5171016303a539d07a7b22a34557456eadf0eb40853bdd813e`; compare the [PyPI release metadata](https://pypi.org/pypi/pycocotools/2.0.10/json). Existing package pins remain in [the original training lock](../requirements-training-windows.lock.txt). The additional hash pin is Windows/CPython 3.11 specific. Never apply it to inference `.venv`.

Prepare the local approval record without opening photographs (the parent directory must exist):

```powershell
New-Item -ItemType Directory -Force work/stage4 | Out-Null
.venv-training/Scripts/python.exe scripts/train_taco.py --write-context work/stage4/dataset_context.json
@'
import json
from pathlib import Path
record = json.loads(Path('config/taco_training_rights.example.json').read_text())
record['dataset'] = json.loads(Path('work/stage4/dataset_context.json').read_text())
Path('work/stage4/rights.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
'@ | .venv-training/Scripts/python.exe -
```

The generated record remains **unapproved**. Supply actual reviewed evidence/obligations before setting clearance fields and `approved=true`. Then, only after the rights gate is satisfied:

```powershell
.venv-training/Scripts/python.exe scripts/train_taco.py --rights work/stage4/rights.json --epochs 2 --batch-size 1 --shortest-edge 256 --longest-edge 384 --pilot-steps 3 --checkpoint-every 50 --device cuda
```

Each invocation creates a fresh ignored `runs/training/taco-<uuid>/`. Reports contain actual contract/settings/source/runtime/rights provenance; `baseline.json`, `pilot.json`, epoch validation JSON, `steps.jsonl`, immutable checkpoints and `latest_checkpoint.json` preserve evidence. Failed/incomplete checkpoint folders are never published as latest; previous complete checkpoints/predictions remain. No annotated images or image copies are generated. Safe interrupt sets an interrupted report; unsaved steps since the last checkpoint must be replayed.

Resume a **complete interrupted-run** periodic checkpoint, with identical options/environment/source/dataset/rights file bytes, using its concrete path:

```powershell
.venv-training/Scripts/python.exe scripts/train_taco.py --rights work/stage4/rights.json --epochs 2 --batch-size 1 --shortest-edge 256 --longest-edge 384 --pilot-steps 3 --checkpoint-every 50 --device cuda --resume runs/training/taco-<original-id>/checkpoints/step-000050
```

Resume verifies every checkpoint artifact before native HF loading, reconstructs the adapter and checks its context, restores optimizer/RNG/sampler position, rechecks actual-photo coordinates, and creates a new run linked to the source checkpoint. Baseline/pilot evidence is carried inside hashed checkpoint artifacts and reused; pilot is not repeated. Only remaining updates run. Epoch loss means in the new report cover observed remaining batches, not invented losses for already completed batches. A fully completed checkpoint, changed epoch budget/options/rights/code/runtime or sampler inconsistency is rejected; cross-version continuation and extending completed runs are not qualified. No scheduler/scaler exists to restore. Stage 3D adapter manifest's `resume_supported=false` still describes that older weights-only format; the new outer `bantai.training-checkpoint.v1` explicitly supports complete trainer continuation.

## Executed evidence and acceptance

- Metadata-only context generation succeeded; 221/397 train and 47/75 validation counts and original/split/taxonomy hashes matched. No test split read.
- Actual default trainer invocation without `--rights` returned **2**, report `taco-0cf6636982b14ac59c27e464e114e7c3/report.json`: `blocked_rights`, zero source-photo reads and zero optimizer steps. Subsequent unit-fixture report IDs are generated-image orchestration evidence, not TACO runs.
- Repeating that command against final source produced `taco-bfbbf57e8eb640978f561601b8334acf`, again exit **2**, zero photo reads/updates and `blocked_rights`.
- Focused tests initially exposed ordering assumptions in two test assertions (ranking ties and score order); assertions were corrected to compare identities/coordinates and error sets. No dataset/model correction followed from these fixture failures.
- Full suite: **158 run, 157 passed, one existing Windows privilege skip**. Fourteen new tests cover explicit rights/source/geometry/metric/checkpoint paths. Toy-update orchestration completes two generated-image epochs and resumes from a mid-epoch periodic checkpoint, yielding exactly matching CPU toy weights; this is software evidence only.
- Pinned CUDA synthetic trainer qualification: final measured run and preservation summary are recorded below. This does not prove real-photo memory fit, convergence or real validation accuracy.

Final actual Windows CUDA run: `stage4-synthetic-76903099b57f453da9514e6a06fce128`, **passed**, GTX 1660 Ti / 6,442,450,944 bytes total, observed driver **472.19**. It used generated eight-object imagery, FP32 batch 1 and 256/384 resize with 384-square padding. Four actual optimizer updates were executed: first step, uninterrupted second step, a repeated second step in the same model, and restored second step after HF reconstruction. First total loss was **16.3935146**; all continuation losses and complete raw components/cardinality diagnostics stay in the ignored report. First-step decoder CE/L1/GIoU were 2.501658 / 0.196366 / 0.833594; encoder CE/L1/GIoU were 3.259469 / 0.131314 / 0.782835. Encoder projection/box-head gradient norms were 3.96639 / 6.70193 and maximum updates about `1.00136e-5`; all trainable gradients/updated weights were finite and frozen weights remained exact.

Saved model parameters, optimizer state, complete RNG state and processor batch tensors restored **exactly before the update**. Resumed loss/components matched the uninterrupted branch; final parameter maximum difference was **4.097819e-8**, **not bitwise identical**. The same-model repeated-update control differed by **8.381903e-8**, showing that reconstruction is not required for small backward differences. CPU toy continuation was exact. Direct and adapted decoder predictions at identical weights were exactly equal. Adapter context and position restored successfully; unchanged base snapshot hashes passed. Synthetic COCO scoring ran end to end but its numbers are fixture outcomes, not trash-detection performance.

The uninterrupted, same-model repeated and reconstructed second-step total losses were each **15.7687292**. This synthetic loss reduction is a wiring observation, not evidence of real-photo learning or generalization.

Measured lifecycle **53.7992 seconds** includes loading/hashing, generated preprocessing, four updates with gradient audits, native checkpoint save/reload, the same-model control, decoder comparison and synthetic evaluation. No separate steady-state throughput benchmark was performed. Peak allocator use across that lifecycle: **2,847,563,264 bytes allocated (2.652 GiB)** / **2,971,664,384 reserved (2.768 GiB)**. Peaks were not reset between reload and initial steps. These values do not qualify actual-photo memory fit or batch two.

### Recorded failures and numerical qualification limits

Preliminary generated runs `745fdc0e804b4538b7b086f211c807d4` and `03525680a23a4831a3da340cfc62be44` passed the initial check. Repeat `876922e55d504d4db2307694903bb1d3` **failed** the provisional absolute `1e-7` post-update bound: maximum difference `1.005828e-7`, despite exact resumed loss and finite updates. This failure is retained; selecting an earlier passing run would not establish repeatability.

A controlled strict-determinism probe identified **`grid_sampler_2d_backward_cuda` has no deterministic implementation**, consistent with [PyTorch's CUDA backward documentation](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.grid_sample.html). Final independent probe `b4b1193673ac478180c809b487db69dc` completed with zero optimizer updates. Its exact reproducible command is above. The final continuation check therefore separately requires exact pre-update weight/optimizer/RNG/batch restoration, loss difference at most `1e-5`, and FP32 post-update closeness (`atol=2e-7`, `rtol=1e-6`), recording bitwise equality separately. The absolute term is roughly two FP32 epsilons, with a magnitude-scaled relative term; the same-model control and all observed differences are reported rather than claiming deterministic CUDA training. No loss, coefficients, model precision or training method was changed to pass this check.

Diagnostic attempt `4da0e721f7ea41d692c2fe0c68117dcc` also **failed** when a raw safetensors state load omitted tied decoder aliases. Only that new control used this load path; the real trainer already used native HF reconstruction successfully. The control now uses safetensors' shared-tensor-aware `load_model`, preserving ties and strict checks. Attempt `07bff54b988e415b9ee7a1f452a77893` **failed with CUDA OOM** on its resumed forward after combining an intentionally failing backward diagnostic and continuation in one process (4.24 GiB allocator use at the error). The probe is now isolated in a separate process; the final unchanged-resolution/model continuation above passed. These diagnostic failures are preserved and do not establish a real-photo VRAM result. No automatic resolution/batch/architecture fallback or installed-library patch occurred.

Preservation probe (`work/stage4/integrity.json`) passed for seven protected file hashes, including owner detection edits, original adapter and strict-clean manifest; all existing training package versions and the complete inference package inventory matched their pre-implementation snapshot. The only addition was `pycocotools==2.0.10` inside `.venv-training`; `pip check` passed. All 235 local Markdown links resolved and `git diff --check` passed. Generated reports/checkpoints, wheel and private approval records stay ignored. No photos were processed outside generated fixtures; no annotated images were exported or dataset/model artifacts published.

| Acceptance | State |
| --- | --- |
| Corrected objective/encoder updates and unchanged decoder inference | PASS on generated-image pinned CUDA check; Stage 3D preserved |
| Loader span/coordinate batching and inverse geometry | PASS on generated photographs; actual TACO preflight UNRUN |
| Native checkpoint, optimizer/RNG/sampler continuation | PASS on CPU fixtures and pinned CUDA generated images within recorded tolerance |
| COCO bbox/class/empty-result evaluator | PASS on controlled fixtures; real metrics UNRUN |
| Original data/splits/taxonomy/inference/adapter preservation | Checked separately; no image exports or publication |
| Rights condition covering 268 photos/472 annotations | BLOCKED; actual evidence/approved basis required |
| Real-image pilot, actual memory, 2–3 epochs and baseline comparison | UNRUN until rights condition passes |
| Human source/annotation/raw phrase/error review | PENDING; no observations invented |

This implementation can proceed directly to the requested run once the explicit rights condition is resolved; no new standalone preparation stage or training-method decision is proposed. Stop for OOM, nonfinite values, missing gradients/updates or invariant failures rather than changing architecture, objective, coefficients, precision or data silently.
