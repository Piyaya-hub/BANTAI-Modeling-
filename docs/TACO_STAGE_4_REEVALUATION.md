# Stage 4 Step 1C — Historical checkpoint reevaluation

Completed locally on 2026-10-20 under the owner's explicit evaluation-only
authorization (D-044). Branch `feat/stage4-real-image-training`, HEAD `087e361`;
existing Step 1B work and owner detection edits are preserved, uncommitted.
No training, test access, visual error analysis, publication or promotion occurs.

## Separate provenance responsibilities

`scripts/evaluation_provenance.py` accepts only an anchored historical report and
epoch manifest. It invokes the **unchanged** `verify_checkpoint` with the genuine
historical context to verify every original artifact, including training state
as bytes only. It rejects missing/unmanifested artifacts and unsafe file paths.
This is historical artifact verification, not a changed-source resume exception.

`config/taco_evaluation_compatibility.v1.json` is a reviewed local evaluation policy,
not a fabricated historical manifest or a digital signature. It binds exact old
and new source hashes and reasons for only the three Step 1B coordinate changes.
All six historical source fingerprints were independently reconstructed from Git
`087e361`. Five supporting files, including both new execution/verifier modules,
are also pinned, as are native processor sources and the corrected baseline record.
A filename-only allowlist is insufficient; any unexpected content change rejects.

The verifier independently checks dataset identity and the unchanged rights record
against the historical corpus/declaration. It records the actual rights-file SHA
separately from the legacy gate fingerprint; it does not relabel either as legal
clearance or change the owner's private record. Source, policy, historical context,
artifact, corpus and rights identities remain explicit in new private evidence.

`scripts/reevaluate_taco.py` validates the pinned architecture, resolved attention
backend, FP32 parameters, processor configuration, tokenizer vocabulary/backend/
special tokens and exact historical adapter manifest. Transient tokenizer call
padding/truncation state is excluded from backend comparison because the fixed
collator supplies the reviewed padding/no-truncation behavior. Both epoch bundles
pass provenance checks before either is loaded. Model weights load through native
safetensors only, all parameters are frozen and evaluation runs in inference mode.
All parameter values and artifact/source identities are rechecked afterward.

No optimizer is created; training state is never deserialized, and optimizer/RNG/
training position is never restored. The CLI accepts validation only, with no
resume/train/test mode. It retains 256/384 resize, 384-square padding, batch 1,
FP32, the existing eight-class prompt, scoring and Step 1B v2 COCO protocol.

## Qualification, failure and commands

Nineteen new synthetic tests in `tests/test_evaluation_provenance.py` cover valid
artifacts, modified/missing/unmanifested files, anchored manifest/report identity,
unexpected source changes, exact reviewed differences, historical provenance,
strict-resume rejection, rights/corpus rejection, validation-only access,
no state deserialization/restoration/updates, frozen inference execution and
configuration/backend semantics. All fixtures are generated in cleaned temporary
directories; their creator declaration is not real TACO permission.

The first 15 synthetic tests passed before accessing real bundles. Added tests
also verify different-but-valid dataset identity and forged source rejection.
Full safe discovery after final code: **187 tests, 184 passed, zero failures/errors,
three skipped**. Skips are the existing Windows symlink privilege case and two
toy optimizer tests excluded by this task's no-update boundary. AdamW/SGD updates
are guarded during discovery; the unfiltered suite is inappropriate here.

An initial actual attempt stopped before photo loading on
`_attn_implementation_autoset`. Installed Transformers 4.51.3
`configuration_utils.py:304` initializes this bookkeeping flag false;
`modeling_utils.py:2142` sets it true after backend selection. The final comparison
excludes only that flag plus path/version metadata, while explicitly comparing the
resolved backend. Regression tests ensure a changed backend/query count remains
visible. No library/config/checkpoint was patched. Failed attempt evidence remains
immutable under `step1c-evaluation-9dac1175305647fbbba9987030204372/`.

Actual commands:

```powershell
.venv-training/Scripts/python.exe -B -m unittest discover -s tests -p test_evaluation_provenance.py -v
.venv-training/Scripts/python.exe -B work/step1c/check_software.py
.venv-training/Scripts/python.exe -B work/step1c/run_guarded.py
```

The ignored `work/step1c` harnesses preserve local logs and instrument the versioned
CLI. Reproducible standalone command (creates a fresh UUID record each time):

```powershell
.venv-training/Scripts/python.exe -B scripts/reevaluate_taco.py --historical-run runs/training/taco-1698bcb46b85435786dd2a199910910f --rights work/stage4-real/rights.json --baseline-record runs/training/step1b-evaluation-c8b127d8bf364c7dbb735795cd81c06d/record.json
```

This intentionally requires the exact approved local corpus, private rights record,
historical bundles and reviewed source policy. Do not regenerate hashes merely to
accept unrelated changes; review and qualify a new policy instead. No dependency
installation is necessary. The existing training resume command is not an
evaluation entry point and still rejects corrected source fingerprints.

## Corrected comparison

New completed record:
`runs/training/step1c-evaluation-3d24a67e1fe041e5a8acf56a26406db4/record.json`.
Both `epoch-01-corrected.json` and `epoch-02-corrected.json` retain local predictions,
provenance and compatibility evidence. `operation-guards.json` records zero calls
to all nine guarded deserialization/optimizer/training/restoration functions.
`verification.json` independently confirms strict-resume rejection and exact
native COCO metric recomputation for both epochs.

Same 47 validation images / 75 annotations; 47 source-verified photo reads per
epoch, **94 total**, no train/test-photo reads. The corrected pretrained baseline
is reused from its hash-anchored Step 1B record, not rerun. Each epoch retains
42,300 queries and 4,700 eligible ranked predictions. All scores and category IDs
match historical outputs exactly; maximum score difference is zero. Incorrect
clipping disappears: invalid boxes fall from 11,170/8,722 to zero for epochs 1/2.

All metrics are fractions. AP is COCO AP@[.50:.95]; no confidence threshold change.

| Model state | Corrected AP | AP50 | AP75 | AR100 |
| --- | ---: | ---: | ---: | ---: |
| Pretrained | 0.111576016 | 0.158252880 | 0.135975042 | 0.363077235 |
| Epoch 1 | 0.126627255 | 0.176342758 | 0.149016053 | 0.275362554 |
| Epoch 2 | 0.066214526 | 0.106098144 | 0.073386262 | 0.235394814 |

Historical faulty-decoder AP values were 0.000298934 / 0.000006713 / 0.000057591.
They remain historical evidence, not valid localization-quality comparisons.
Epoch 1 now has higher aggregate AP than pretrained, but lower recall; epoch 2
has lower AP and recall. This supersedes conclusions based on faulty coordinates.

| Class (validation objects) | Pretrained AP | Epoch-1 AP | Epoch-2 AP |
| --- | ---: | ---: | ---: |
| cigarette (24) | 0.014851485 | 0.019801980 | 0.024752475 |
| plastic_bottle (8) | 0.038241717 | 0.105851102 | 0.044347139 |
| plastic_bag (2) | 0.236228623 | 0.517161716 | 0.283663366 |
| plastic_wrapper (11) | 0.094467223 | 0.000071230 | 0.001015486 |
| plastic_film (13) | 0.001574350 | 0 | 0.000307007 |
| drink_can (7) | 0.108415842 | 0.011881188 | 0 |
| plastic_cup (5) | 0.340655617 | 0.308745875 | 0.175630736 |
| styrofoam_piece (5) | 0.058173267 | 0.049504950 | 0 |

### Losses and scores

| Model | Historical padded-target validation loss | Recalculated valid-image-target validation loss |
| --- | ---: | ---: |
| Pretrained | 19.348786 | 10.553226 |
| Epoch 1 | 13.320138 | 9.312350 |
| Epoch 2 | 12.133389 | 10.211304 |

Columns use different target semantics and are **not directly equivalent**.
Recalculation makes no optimizer update and cannot undo historical training with
incorrect target normalization. Lower historical loss did not establish quality.

| Model | Median score | P90 | P99 | Maximum | Queries >=0.25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained | 0.019574 | 0.053525 | 0.187558 | 0.596656 | 216 |
| Epoch 1 | 0.029931 | 0.042793 | 0.069502 | 0.184526 | 0 |
| Epoch 2 | 0.031275 | 0.048297 | 0.067896 | 0.123536 | 0 |

Score distributions are unchanged by corrected evaluation. High-score suppression
persists despite increased medians; neither fine-tuned checkpoint passes the fixed
0.25 diagnostic operating threshold. AP ranks lower-scoring detections, so nonzero
AP is consistent with zero detections at that threshold. Scores are not calibrated
correctness probabilities; no new thresholds/defaults are selected.

GTX 1660 Ti, 6 GiB VRAM, CUDA 11.8, existing Transformers 4.51.3 environment.
Each epoch's allocator peak was 951,008,768 allocated / 1,019,215,872 reserved bytes
(about 0.886 / 0.949 GiB). Successful combined lifecycle: 140.172 seconds including
verification, loading, evaluations and publication; not isolated throughput.

## Preservation, remaining concerns and next action

Before documentation changes, all **83 existing project files and 992 protected
artifacts** matched initial hashes, including Step 1B edits, both original epoch
bundles and other historical runs, source photos/annotations, exports excluding
test JSON, original pretrained snapshot, taxonomy and private rights evidence.
All 38 training package versions remain unchanged. Only the explicitly updated
current-state Markdown files are permitted to differ in final checks. No existing
training, inference, adapter or checkpoint-verifier source is modified by Step 1C.

New files: `evaluation_provenance.py` (separate verifier), `reevaluate_taco.py`
(validation CLI), the versioned compatibility policy, nineteen regression tests
and this report. README, phase status, handoff and decisions reference this result.
All photos, predictions, private authorization and checkpoints remain ignored/local.
No automatic commit, push, merge, training continuation or promotion.

Small-object AP/AR100 remain zero for all three states. Wrapper/can/styrofoam
performance regresses substantially; bag AP depends on only two validation objects.
The 75-object, positive-only strict-clean validation set is too small and selected
to establish generalization or statistically reliable improvement. Error causes,
visible-but-unannotated litter, phrase/box alignment and false positives require
photo-level review; none is invented from metrics alone.

**Recommendation:** proceed to separately authorized Step 2 local visual comparison
of all three unchanged states on the same validation photos, retaining score/phrase
and class-specific failures. Provenance/evaluation no longer blocks that review.
Do not infer repaired training, acceptable accuracy or checkpoint promotion from
these metrics. Step 2 and further fine-tuning are not begun here.
