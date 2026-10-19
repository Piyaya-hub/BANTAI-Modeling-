# Stage 4 — First local real-image run

The existing Stage 4 pipeline **completed two real-image epochs** under D-042, using the [documented owner-authorized academic-use claim](TACO_STAGE_4_ACADEMIC_USE.md). This is execution evidence within Stage 4, not another preparation phase. Runtime/data/checkpoint gates passed. **Validation detection AP deteriorated despite lower loss: this checkpoint is experimental and is not approved for promotion.**

## Corpus, rights and preservation

Run: `taco-1698bcb46b85435786dd2a199910910f`, local Windows GTX 1660 Ti (6144 MiB), driver 472.19, PyTorch CUDA 11.8, Transformers 4.51.3 in the existing `.venv-training`. Model is the exact pinned Grounding DINO Tiny revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, with the unchanged Stage 3D encoder-output adapter and native loss formula/coefficients. No installed library or inference pipeline is patched.

Train: 221 photographs / 397 objects. Validation: the same 47 photographs / 75 objects for baseline and both epochs. All 268 passed actual source-file hash, stored-raster size/orientation and processor box/count/span checks before optimizer updates. No test split JSON or test photograph is loaded by the trainer. Originals, class order, splits and inference configuration remain unchanged.

The owner's declaration and a balanced Section 185 record support a limited local academic exception claim. `work/stage4-real/rights.json` binds the exact corpus and assessment: workflow authorization true, photo/database clearance false, third-party permission/institutional approval/definitive clearance explicitly not claimed. The actual record and all corpus/model artifacts stay ignored and local. No source photographs, raw annotations, raw prediction files or checkpoints are published or sent to cloud services.

Initial preflight detected **external formatting drift** in `annotations.json`: current SHA-256 `6dce30703959153e3207a7174c3326585d9593438f7af855689a0e2000ac50b9`, recorded original `ac1ec605c9e1fcda767970de3457c69b4a43b62c02aca8c4c80f51d1b5ae8449`. Parsing strictly and using default `json.dumps` reconstructs the exact originally hashed byte stream, proving retained content and order match; newline normalization alone did not. The source/export files were not rewritten. The loader accepts only that cryptographically verified reconstruction, records both hashes, and still rejects changed annotation values. All original split hashes remain matched. This was a pre-update integrity stop, not an ignored model failure.

## Exact execution

```powershell
.venv-training/Scripts/python.exe -m unittest discover -s tests -p test_taco_training.py -v
.venv-training/Scripts/python.exe scripts/train_taco.py --rights work/stage4-real/rights.json --epochs 2 --batch-size 1 --shortest-edge 256 --longest-edge 384 --pilot-steps 3 --checkpoint-every 50 --device cuda
.venv-training/Scripts/python.exe -m unittest discover -s tests -v
```

The private rights record is an actual owner-declaration record, not a reusable grant of permission for another user/corpus. Gate schema and factor/source requirements are in [the loader](../scripts/taco_training_data.py); do not turn legacy permission flags true to imitate an exception claim.

Configuration: FP32, frozen vision/BERT backbones, batch 1, aspect-preserving 256/384 resize and bottom/right padding to 384 square; seed 42; AdamW LR `1e-5`, weight decay `1e-4`, `foreach=false`, clip norm 1; no augmentation, scheduler, AMP, accumulation or workers. Pilot and baseline use only authorized train/validation data. The three pilot steps passed; pilot weights and optimizer were discarded, then the full experiment restarted from the unchanged base. No settings were changed to improve detection counts.

Evaluation follows [the previously fixed Stage 4 protocol](TACO_STAGE_4.md): mean ordinary-subword sigmoid scores, single argmax class/query, no NMS, score floor zero and top 100 valid predictions per image; same inverse geometry, original COCO annotations and COCO bbox metrics. Model-score and AP percentages are not interchangeable. Raw phrases and all query diagnostics remain local. Annotation-relative error counts are automated comparisons, not human judgments of every visible object.

## Verification and limits

Focused tests: **16 passed**. Full suite: **160 run, 159 passed, one existing Windows privilege skip**. New tests cover accurate v2 exception semantics/evidence binding, legacy gate preservation, exact source-format reconstruction and rejection of substantive change. Earlier CUDA synthetic investigations were not repeated. Full unit tests ran concurrently with part of epoch 1; lifecycle timing is not an isolated throughput benchmark.

Checkpoints are immutable native HF model/processor bundles plus adapter context, optimizer/RNG/epoch/batch state and experiment evidence. The latest pointer is published only after the complete artifact manifest. The existing [resume contract and exact-byte requirements](TACO_STAGE_4.md) apply; preserve the local rights record and bound assessment bytes. This run is not automatically promoted into inference, and test evaluation remains unauthorized/unrun.

Low resolution, strict-clean positive-scene selection and very small validation classes limit conclusions. Human source/box/raw-phrase error review remains pending on the local machine; no photographs are uploaded to obtain agent visual review. Completing optimizer steps or reducing loss does not establish improved detection performance.

## Completed results

Process exit **0**, report status **completed**, **442 full-run updates** (221 per epoch), plus **three discarded pilot updates**. The report counts 854 authorized photo reads: 268 preflight + 47 baseline + 3 pilot + 442 training + 94 epoch validation. A subsequent saved-checkpoint replay read the same 47 validation photos, made no optimizer updates and used no test photos. Training source corresponds to code commit `c8c7102`; the recorded source fingerprints match the run.

| Loss measurement | Baseline | Epoch 1 | Epoch 2 |
| --- | ---: | ---: | ---: |
| Mean training total, one image per batch | — | 14.304753 | 12.103961 |
| Mean validation total per image | 19.348786 | 13.320138 | 12.133389 |

Pilot totals were 10.433875, 11.014774 and 11.890447 on three different dense training images; they are not a same-image convergence curve. Full-run loss ranges were 7.515423–37.338787 and 7.866044–29.871866 for epochs 1/2. All losses/gradients/updated trainable weights were finite, supervised encoder heads updated, and frozen parameter hashes remained unchanged.

Mean unweighted component losses across each epoch:

| Epoch | Decoder CE | Decoder L1 | Decoder GIoU | Encoder CE | Encoder L1 | Encoder GIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.615250 | 0.122103 | 0.641084 | 2.633336 | 0.124029 | 0.647378 |
| 2 | 2.146493 | 0.108298 | 0.590290 | 2.190691 | 0.105200 | 0.590762 |

The original 2/5/2 class/L1/GIoU coefficients apply to each branch; cardinality diagnostics are retained but are not optimized components. No native uncorrected model-with-labels objective was invoked.

### Same-protocol validation comparison

All values below are **fractions on [0,1]**, not percentages or model-score thresholds. AP is COCO bbox AP@[.50:.95]. Baseline and both checkpoints use the same 47 images/75 annotations and fixed prompt, resize/padding, class decoding, top-100 and evaluator settings.

| Metric | Unchanged pretrained | Epoch 1 | Final epoch 2 |
| --- | ---: | ---: | ---: |
| AP | 0.000298934 | 0.000006713 | 0.000057591 |
| AP50 | 0.001329652 | 0.000016781 | 0.000138055 |
| AP75 | 0 | 0 | 0.000052946 |
| AR100 | 0.007884615 | 0.003846154 | 0.015576923 |

Additional area metrics remain in local JSON. Both baseline and final AP for small and medium objects were zero. Final AR100 increased slightly while precision/AP remained extremely low. No settings were selected to make these numbers look better.

| Class | Validation annotations / images | Baseline AP | Final AP |
| --- | ---: | ---: | ---: |
| cigarette | 24 / 7 | 0 | 0 |
| plastic_bottle | 8 / 8 | 0 | 0 |
| plastic_bag | 2 / 2 | 0 | 0 |
| plastic_wrapper | 11 / 11 | 0 | 0 |
| plastic_film | 13 / 9 | 0.000792079 | 0.000371488 |
| drink_can | 7 / 7 | 0 | 0 |
| plastic_cup | 5 / 4 | 0 | 0.000089236 |
| styrofoam_piece | 5 / 5 | 0.001599391 | 0 |

At the fixed diagnostic operating score **0.25**, IoU **0.5**, the baseline matched 2 annotations, missed 73, and had 189 unmatched predictions and 3 wrong-class predictions. Epochs 1/2 produced no predictions above that score and missed all 75 annotations. AP still evaluates lower-score predictions, so small nonzero AP is compatible with this result. These are annotation-relative software counts, not human confirmation of all visible litter or true physical object counts. Low score suppression/localization/class errors require local review before any longer experiment; the cause is not established here. No prompt, thresholds, taxonomy, dataset or objective was changed in response.

### Actual resource and timing measurements

- Real pilot: peak **2,559,891,968 allocated / 2,722,103,296 reserved bytes** = **2.384 / 2.535 GiB**.
- Full training/epoch-validation lifecycle after reset: **2,558,043,648 / 2,726,297,600 bytes** = **2.382 / 2.539 GiB**. Baseline/pilot counters are separate; these are PyTorch allocator peaks, not total device usage. Batch 1 at the specified resolution fitted; higher resolutions, batch 2, AMP and unfrozen training are unrun.
- Entire trainer: **449.386 seconds (7.490 minutes)** including environment/hash checks, actual-photo preflight, model loads, baseline, pilot, 442 updates, two validations and native checkpoint writes/hashes. Sum of individual update timings including gradient audits: **110.295 / 121.320 seconds** for epochs 1/2; these exclude preprocessing, checkpoint IO and validation.
- Baseline / epoch-1 / epoch-2 evaluation scope: **23.302 / 16.856 / 18.703 seconds**, including data reads, preprocessing, adapted validation loss, decoding and COCO metrics. No isolated warm-up/throughput claim.

### Checkpoint and replay

Final local bundle:

```text
runs/training/taco-1698bcb46b85435786dd2a199910910f/checkpoints/epoch-02/
```

`latest_checkpoint.json` points there with `epoch=2`, `next_batch=0`, `global_step=442`. Periodic steps 50/100/150/200/250/300/350/400 and epoch-1 checkpoints remain local. The final bundle contains model safetensors, processor, adapter manifest, optimizer/RNG/position state, complete experiment evidence and hashes. No experimental weights are loaded by existing inference.

Read-only native HF reload and replay on **all 47 validation images** passed after training. Metrics, all stored predictions, all raw query records and mean adapted loss matched the pre-save evaluation **exactly**. It verified all ten final checkpoint artifact hashes, adapter context and current code identity, performed zero updates and saved `checkpoint_replay.json`; lifecycle **42.603 seconds**, measured separately. The local probe command was `.venv-training/Scripts/python.exe work/stage4-real/replay_checkpoint.py`; it uses the versioned `load_model`, `verify_checkpoint`, rights-gated loader and `evaluate` helpers, with no test loader or optimizer. The earlier checkpoint continuation qualification remains applicable; no new synthetic investigation was run.

Private evidence files under the run directory: `report.json`, `baseline.json`, `pilot.json`, `steps.jsonl`, `validation-epoch-1.json`, `validation-epoch-2.json`, `checkpoint_replay.json`, `preservation_checks.json`, checkpoint manifests and latest pointer. Their exact metrics/diagnostics are authoritative over rounded documentation values. They are ignored, not published with this report.

Final preservation audit passed for nine protected inference/settings/adapter/taxonomy/export files, original snapshot and inference package inventory. All ten checkpoint bundles passed every recorded artifact hash. The final optimizer has **558 parameter states, each at step 442**, with Python, NumPy, CPU Torch and CUDA RNG records present. No package was installed or changed in this execution task. The owner's pre-existing local detection edit, including unrelated trailing whitespace, remains untouched and excluded from commits.

Final checks also confirmed steps 1–442 without gaps, exact once-per-epoch coverage of all 221 training IDs, identical 47 validation IDs and 42,300 raw query records in each evaluation, strict run JSON, and all **251 local Markdown links** resolving. Scoped whitespace checks exclude the owner's unrelated detection edit. No photographs or raw annotations were required to leave the computer for these checks.

**Remaining outcome:** no runtime blocker to this completed experiment. Detection quality has not passed acceptance; no checkpoint promotion, test evaluation, extended run or optimization implementation was performed. Review results locally before authorizing changes; no additional preparation phase is created.
