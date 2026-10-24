# Stage 3B: isolated environment and synthetic training smoke test

Follow-up [Stage 3C investigation](TACO_STAGE_3C.md) now traces the encoder tensor/gradient findings and proposes a correction requiring approval. The Stage 3B lifecycle measurements below remain valid historical evidence, not real-data readiness.

Executed on the owner's Windows computer, 2026-10-18, under D-038. **The bounded synthetic FP32 forward/backward/optimizer and save/reload smoke passed.** This qualifies only the recorded configuration. Native encoder-loss limitations below remain a real-data readiness blocker. No TACO photograph was read by the smoke command or used for training; the strict-clean dataset, inference `.venv`, pipeline, local inference settings, approved taxonomy and base checkpoint are preserved. No cloud service or spending was used.

## Implementation and boundaries

| Added/modified file | Purpose |
| --- | --- |
| [requirements-training-windows.lock.txt](../requirements-training-windows.lock.txt) | Existing Windows package pins plus SciPy 1.15.3, installed only in `.venv-training`. Existing inference requirements remain unchanged. |
| [requirements-training-windows.hashes.txt](../requirements-training-windows.hashes.txt) | SHA-256 locks for all 37 Windows/Python 3.11 artifacts, including pip/setuptools bootstrap wheels. |
| [scripts/smoke_train_grounding_dino.py](../scripts/smoke_train_grounding_dino.py) | CLI entry point; synthetic input only, bounded 1–5 steps, explicit device, no fallback. |
| [scripts/training_smoke.py](../scripts/training_smoke.py) | Environment guard, synthetic rectangles/COCO targets, prompt-span/geometry assertions, native PyTorch loop, diagnostics, source integrity and model/processor round trip. |
| [tests/test_training_smoke.py](../tests/test_training_smoke.py) | Twelve software tests; no pretrained model construction or real photos. Native loss fixtures require isolated SciPy. |
| [.gitignore](../.gitignore) | Adds only `.venv-training/` and `runs/training/`; environments, checkpoint weights, reports and logs remain local. |
| README, AGENTS and phase/scope/decision/test/handoff documents | Record current authorization, evidence and the stop before real-data fine-tuning. |

The CLI accepts no dataset or photo path. It generates one 320×240 landscape image with all eight classes and one 240×320 portrait image with three annotations. Rectangles are arbitrary colored shapes, **not synthetic evidence of recognizing trash**. Fixtures include a 4×4 edge box, nested boxes, opposite aspect ratios and unequal object counts. Tests separately cover an empty target set. The actual model smoke batch contains 8+3 objects, not an empty-target example.

The exact HF snapshot is read offline from `.cache/huggingface/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71/`. No alternate model, new weight download, replacement head or query-count change is used. Model and processor load with local-files-only settings, remote code disabled and custom kernels disabled. Loading information must have no missing/unexpected/mismatched keys/errors. Every cached snapshot file is hashed before and after the run.

All eight approved labels retain IDs 0–7. The separate training prompt is:

```text
cigarette. plastic bottle. plastic bag. plastic wrapper. plastic film. drink can. plastic cup. styrofoam piece.
```

The tokenizer produces 30 tokens including special tokens; native label maps match exactly eight ordered ordinary-text segments for each image. Source targets use COCO pixel xywh. Before processing, malformed/duplicate/ignored/out-of-bounds targets fail instead of being silently clipped. Processor output is checked against independently calculated normalized cxcywh coordinates in the **padded** image frame, preserving IDs/counts and integer tensor dtypes. Reloaded processor inputs and target geometry must match.

## Environment and reproduction

`include-system-site-packages=false` was verified in `.venv-training/pyvenv.cfg`; it inherits no inference/system packages. Training refuses to execute from inference `.venv` or a different prefix. It verifies every recorded version and runs `pip check` before touching the model. Existing inference package metadata and protected file hashes were captured separately for preservation checking.

| Component | Actual isolated environment |
| --- | --- |
| OS / Python | Windows 10, CPython 3.11.9 x64 |
| torch / torchvision | 2.7.1+cu118 / 0.22.1+cu118 |
| Transformers | 4.51.3 |
| SciPy / NumPy | 1.15.3 / 2.4.6 |
| Pillow / safetensors | 11.2.1 / 0.8.0 |
| GPU | NVIDIA GeForce GTX 1660 Ti, 6,442,450,944 bytes (6144 MiB), capability 7.5 |
| Driver / framework CUDA | NVIDIA 472.19 / torch CUDA 11.8; `nvidia-smi` separately advertises driver CUDA 11.4 |
| Dependency consistency | `pip check`: No broken requirements found |

The complete environment is in each ignored `report.json`; exact pins and wheel hashes are tracked. SciPy is the only added runtime package relative to the existing lock. No pycocotools, Trainer, Accelerate, cloud SDK or new model dependency is needed for this stage. Package installation reused cached wheels except where pip fetched missing package artifacts. This was a real isolated installation, not sharing/copying the inference site-packages.

From the repository root, use the existing Python 3.11.9 installation to create a **new** environment if it is absent; do not clear/overwrite an existing environment:

```powershell
.venv/Scripts/python.exe -m venv .venv-training
.venv-training/Scripts/python.exe -m pip install --require-hashes -r requirements-training-windows.hashes.txt
.venv-training/Scripts/python.exe -m pip check
.venv-training/Scripts/python.exe -m unittest discover -s tests -q
.venv-training/Scripts/python.exe scripts/smoke_train_grounding_dino.py --device cuda --steps 2 --batch-size 2 --shortest-edge 192 --longest-edge 256 --trainable fusion-decoder --precision fp32
```

Initial actual installation used `-r requirements-training-windows.lock.txt --report work/stage3b-install.json`. Hashes were then captured from pip's actual wheel records and Python's bundled pip/setuptools wheels. The hash-lock dry run passed. The download command below subsequently succeeded for **all 37 hash-enforced artifacts**, retained in the ignored wheelhouse. A second fresh environment installation from that wheelhouse was not performed. Python's [venv documentation](https://docs.python.org/3.11/library/venv.html) explains the isolated environment mechanism. These hash locks target Windows CPython 3.11 x64 CUDA 11.8 artifacts, not a portable Linux lock.

Optional local artifact preservation/verification, with no package installation:

```powershell
.venv-training/Scripts/python.exe -m pip download --require-hashes -r requirements-training-windows.hashes.txt --dest work/training-wheelhouse
```

The offline snapshot must already exist. The smoke command never downloads it. Each invocation creates a new UUID directory under `runs/training/`, prints the report path, and exits 0 for a complete smoke pass or 1 for a recorded failure. Invalid CLI arguments exit 2. It atomically updates strict JSON after each stage and step, preserves failed runs/tracebacks, and never overwrites an earlier checkpoint. A killed process may leave `status=running`; that is not a pass. Checkpoints are about 658 MiB each; no automatic deletion occurs.

## Actual runs and failures

| Run ID | Explicit configuration | Outcome |
| --- | --- | --- |
| `35ffac8ded62405aa6982a1fe9fabcd1` | CUDA, FP32, batch 2, 128/192 edges, frozen vision/BERT | **Failed before loss/optimizer:** `RuntimeError: selected index k out of range`. 170×170 padded features provide only 650 encoder positions for 900 queries. Not OOM; no optimizer step occurred. |
| `f4aa8b29e0234ce88e17f575e4ebfb13` | CUDA, FP32, batch 2, 192/256 edges, frozen vision/BERT, two steps | **Passed** native forward/backward/AdamW plus model/processor save/reload. |
| `ec713b1fbc47407b8d729d47fa13fbf2` | Same explicit settings; additional numeric/gradient diagnostics | **Passed** with final encoder-capacity/numeric checks and native encoder-gradient observations. This is the final qualification evidence. |

The first failure exposed a minimum spatial-capacity condition in the [pinned model's two-stage top-k selection](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/grounding_dino/modeling_grounding_dino.py). The utility now guards feature-position capacity and defaults to 192/256. Both landscape and portrait resize to 256×192 / 192×256, then pad to 256×256: 1,360 feature positions. Model architecture, 900 queries and 256 text-token capacity remain unchanged. The changed size was a new explicit run, not an automatic retry or hidden model modification.

## Final run measurements

Actual parameters: **172,249,090 total**, **35,836,936 trainable**. Vision/BERT parameters are frozen and their dropout behavior is held in evaluation mode while the remaining model trains. This is explicitly a partial-parameter training smoke. `--trainable all` exists for a separately requested synthetic experiment but was **not run**. FP16 is also implemented as an explicit option but **unrun**; no AMP or full-model fit is established.

Training: seed 42, AdamW LR `1e-5`, weight decay `1e-4`, `foreach=False`, gradient clipping norm 1.0, no scheduler/augmentation, FP32, no accumulation. Cached `two_stage=true`, `auxiliary_loss=false`, unchanged Hungarian matching costs 1/5/2 and native classification/L1/GIoU weights 2/5/2. No box/text threshold is applied to training targets or loss. This uses the [native Transformers 4.51.3 loss](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/loss/loss_grounding_dino.py), not a replacement criterion.

| Final run measurement | Step 1 | Step 2 |
| --- | ---: | ---: |
| Total native loss | 60765.60546875 | 60675.078125 |
| Decoder classification | 1.97500396 | 2.09115338 |
| Decoder L1 / GIoU | 0.08536888 / 0.59629178 | 0.10611068 / 0.64145774 |
| Encoder classification | 30378.0 | 30332.83984375 |
| Encoder L1 / GIoU | 0.32441053 / 1.20731497 | 0.24570787 / 1.08596432 |
| Gradient norm before clipping | 28920.767578125 | 28882.359375 |
| Tensors with gradients | 548 | 548 |
| Forward/backward/update including audits, seconds | 3.1655261 | 0.4754773 |

Losses/components and received gradients are finite, the gradient norm is nonzero, AdamW state is populated and the `model.level_embed` probe changes. Every frozen parameter retains its exact hash. The small change in total loss is wiring/debugging evidence only, not object recognition, convergence or useful learning.

Peak CUDA memory from before load/transfer through preprocessing and both training steps, measured with `torch.cuda.max_memory_*`: **2,892,561,920 allocated bytes (2.694 GiB)** and **3,061,841,920 reserved bytes (2.852 GiB)**. Reload/eval peaks after resetting counters: 921,864,704 allocated / 1,038,090,240 reserved bytes. The first failed run peaked at 1,168,617,984 allocated / 1,195,376,640 reserved bytes. These are PyTorch allocator measurements, not whole-device `nvidia-smi` peaks; other Windows display processes already used approximately 831 MiB before this work.

CUDA synchronization brackets the measured training interval. No separate warm-up occurred: step 1 includes lazy CUDA/kernel/optimizer initialization and should not be compared as steady-state throughput. Timings include native loss/gradient checks and clipping; exclude model load, synthetic generation, checkpoint serialization and reload. Load/transfer and total lifecycle timing are in the JSON. No OOM occurred at 192/256; original 800/1333 resolution, large batches, full parameter training and AMP remain unrun.

## Native-loss limitation requiring follow-up

The run passed the bounded lifecycle, **but native encoder supervision is not fully functional in the inspected path**:

- `encoder_pred_boxes.requires_grad=false`; encoder L1 and GIoU loss tensors also have no gradient. Ten declared trainable tensors receive no gradients: `model.enc_output` and its normalization weights/biases, plus all six `model.encoder_output_bbox_embed` layer weights/biases.
- Source review shows encoder proposal coordinates are detached before being returned for loss; top-k selection is discrete. The runtime flags corroborate that the recorded encoder box-loss terms cannot train the encoder box head in this path. Decoder regression and classification do have gradients.
- The encoder classification loss contributes over 99.9% of the recorded weighted total. Finite arithmetic does not validate its scale or suitability for fine-tuning. The cardinality diagnostic (894.5) also is not an accuracy metric.

No library patch, alternative loss, version upgrade, architecture change or omission of encoder loss was made to obtain a pass. **Recommendation: investigate/qualify the pinned encoder training path before real-data training approval.** A future correction or alternate training route must be explicit, source-reviewed and tested; it must not change the working inference environment. Stage 3A's proposed native route is now supported by a limited execution pass, not a complete training-readiness claim. Small variations between the two successful runs' decoder components also mean seeds are not a guarantee of bitwise CUDA reproducibility.

## Saving, reloading and preservation

Each successful run saves model safetensors and the processor together under its own `checkpoint/`, with `smoke_context.json` recording prompt, taxonomy, settings, base revision and completed steps. The processor retains its original default size; the recorded 192/256 override is reapplied for smoke replay. Full parameter hashes match after offline reload; loading diagnostics are empty. Reloaded pixel/text tensors, target geometry and padding match. Evaluation logits on active text positions and boxes match with **maximum differences 0.0 / 0.0** (declared tolerance `atol=rtol=1e-5`); expected masked `-inf` sentinels are checked separately. NaN/positive infinity cannot be hidden as padding.

All cached pretrained snapshot file hashes match before/after each attempt. The synthetic checkpoint is never promoted into inference configuration. This minimal qualification utility does **not** save optimizer/scheduler/scaler/RNG state for resume, and explicitly records `resume_supported=false`; it is not a production or resumable real-data trainer.

Preservation checks **passed**: inference environment complete package metadata and all seven hashes of detection settings, taxonomy, inference/viewer/preparation code, original inference lock and strict-clean preparation manifest match the before-install snapshot. Final implementation/checkpoint hashes match the successful report; all three run reports parse as strict JSON. Ignored-path checks cover the environment, weights and wheelhouse; 198 local Markdown file links resolve and diff whitespace checks pass. After process exit, `nvidia-smi` reports 792 MiB device use (the desktop remains active), with no claim of a whole-device peak measurement. Source photographs/annotations and split exports were not inputs to this stage's trainer and are not written anywhere by it. Existing Stage 2 source-integrity evidence remains in [Stage 2](TACO_STAGE_2.md) and [Stage 3A](TACO_STAGE_3A.md).

## Automated verification and remaining gates

Actual isolated software suite: **126 tests run, 125 passed, one existing Windows privilege/symlink skip**; 12 new smoke tests passed. Tests cover approved class order, all-eight synthetic coverage, malformed boxes/classes/metadata, empty targets, batch sizes 1/2, padding/coordinate math, source-target preservation, integer dtypes, text spans, native Hungarian/focal/L1/GIoU loss with independently calculated focal value and masked-token gradients, frozen hashing, loading mismatch rejection, numeric sentinels and minimum encoder capacity. CLI rejects real-photo inputs, automatic device fallback and more than five steps. These tests do not substitute for the actual CUDA/model lifecycle evidence above.

Pending before real-data fine-tuning: investigate encoder loss/gradient semantics and large loss scale; approve real-data trainer/resume/evaluation design; clear image/database rights; accept rare-class coverage and leakage protocol; measure the intended real-resolution configuration; authorize any new taxonomy/ignore/export policy separately. No detection quality metrics, real-data epochs, TACO training, new splits, cloud access or spend were executed. Stop here until separately authorized.
