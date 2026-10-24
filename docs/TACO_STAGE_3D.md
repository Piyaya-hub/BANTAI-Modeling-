# Stage 3D: training-only encoder-output adapter and synthetic qualification

**Assessment: PASS for the recorded corrected synthetic training objective.** Executed 2026-10-19 (Asia/Manila) on the owner's GTX 1660 Ti, under D-040. The approved adapter restores encoder supervision using selected image-conditioned proposal scores and undetached box values, while retaining the native decoder's detached initialization. Actual gradients, optimizer updates, finite losses, exact decoder equivalence and checkpoint replay passed. This does **not** establish real-data training readiness, detection quality, convergence or full-resolution memory fit.

No real photograph was used or loaded by the qualification command. Original TACO files, approved taxonomy, validated splits, inference pipeline/settings/environment, installed Transformers and pretrained snapshot remain unchanged. No cloud resources, dependency changes, architecture changes or automatic checkpoint promotion occurred. Stage 3D stops here.

## Files and implementation

| File | Purpose |
| --- | --- |
| [scripts/training_adapter.py](../scripts/training_adapter.py) | Explicit stateless training-only adapter, selected-proposal gathering, input/configuration guards and reproducibility manifest. |
| [scripts/qualify_training_adapter.py](../scripts/qualify_training_adapter.py) | Bounded synthetic CUDA/CPU qualification, actual corrected optimizer steps, gradient/update checks, equivalence and checkpoint round trip. No real-data input option. |
| [tests/test_training_adapter.py](../tests/test_training_adapter.py) | Eleven regression tests with a small standalone tensor model and cached tokenizer; no library patch or pretrained model construction in fixtures. |
| This report; AGENTS/README/phase/scope/test/decision/handoff updates | Current authorization, actual results, acceptance scope and remaining gates. |

The adapter is an ordinary Python object, not an alternative model, registered neural layer or subclass of Transformers. It does not mutate weights/configuration or install hooks/monkey patches. The original Stage 3B trainer and Stage 3C diagnostics remain unchanged and available for historical reproduction.

The forward sequence is:

1. Validate the pinned configuration, original coefficient values and fixed eight-class prompt. Validate normalized padded-frame targets and the qualified batch size (1 or 2).
2. Run the **unchanged model forward without `labels`**, preserving every decoder operation. This avoids invoking the miswired native loss inside the model; it does not skip encoder supervision.
3. Select exactly the native 900 indices from `enc_outputs_class.max(-1)`. Gather those logits and corresponding `enc_outputs_coord_logits`, applying sigmoid to the gathered coordinates **without detaching the supervised copy**.
4. Assert bitwise equality between selected box values, the native detached `encoder_pred_boxes` and native decoder `init_reference_points`; assert both native reference tensors remain detached.
5. Call the existing `GroundingDinoForObjectDetectionLoss` with unchanged decoder outputs, the gathered encoder outputs, native span maps/masks, original labels and original configuration. No custom focal, matching, normalization or weighting implementation replaces it.

The [pinned native model](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/grounding_dino/modeling_grounding_dino.py) and [native criterion](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/loss/loss_grounding_dino.py) remain installed unchanged. [Stage 3C](TACO_STAGE_3C.md) provides the source/causal evidence for the routing correction. The adapter leaves the original `ModelOutput` intact under `result.outputs` and returns the corrected `result.loss`/`result.loss_dict` separately.

Config guards reject changes to 900 queries, 256 text-token capacity, six decoder layers, two-stage mode, learned query initialization, head sharing and native matching/loss settings. `auxiliary_loss=false` is preserved; no decoder auxiliary terms are silently enabled or disabled. Matching costs remain 1/5/2; native classification/L1/GIoU weights remain 2/5/2, focal alpha .25. The eight labels/order and 30-token prompt are unchanged. A missing regression gradient, changed decoder detach boundary, different proposal values or unsupported text/configuration fails visibly; there is no alternate loss/device/model fallback.

## Environment and exact execution

The existing `.venv-training` was reused without installations: Python 3.11.9 x64, torch 2.7.1+cu118, torchvision 0.22.1+cu118, Transformers 4.51.3, SciPy 1.15.3, NumPy 2.4.6, Pillow 11.2.1 and safetensors 0.8.0. GPU: GTX 1660 Ti, 6144 MiB, capability 7.5; driver 472.19. The [Stage 3B version/hash locks](../requirements-training-windows.hashes.txt) remain the environment specification. The runner refuses execution from inference `.venv` and checks all pinned versions plus `pip check`.

The cached model/processor are loaded offline from the exact snapshot `IDEA-Research/grounding-dino-tiny` at `a2bb814dd30d776dcf7e30523b00659f4f141c71`. All loading mismatch/error lists are empty. Safetensors, remote-code-disabled and custom-kernel-disabled settings are retained. There is no new weight download or checkpoint-selector option.

From the repository root:

```powershell
.venv-training/Scripts/python.exe -m unittest discover -s tests -p test_training_adapter.py -q
.venv-training/Scripts/python.exe -m unittest discover -s tests -q
.venv-training/Scripts/python.exe scripts/qualify_training_adapter.py --device cuda --steps 2 --cases baseline_11 two_targets one_target empty
```

The command creates a fresh ignored UUID directory under `runs/training/adapter-<uuid>/`, prints its report path and exits 0 only after every invariant/qualification check passes. Failures yield `assessment=FAIL`, status/error/traceback and exit 1; malformed arguments exit 2. A killed process can leave `PENDING/running`, never a pass. Steps are bounded to 1–3 per case; the baseline must come first. The command accepts no photograph/dataset paths, automatic device mode, promotion or resume option.

## Actual synthetic runs

| Run suffix | Outcome |
| --- | --- |
| `aa095021049740d6acc5dcac96a28331` | Initial corrected objective: all four cases, eight total updates, decoder/repeat equivalence and checkpoint replay passed. |
| `a06cfea197c545fc82c6df91c0b39583` | **Final PASS** after improving adapter source-fingerprint portability across LF/CRLF checkouts and recording runtime versions in its saved manifest. All cases/equivalence/save-reload were requalified against the final implementation. |

No failed invariant was bypassed and neither attempt had NaN/Inf, OOM or a model/library error. The second run was warranted by a reproducibility metadata change, not a new objective or a search for lower loss. Both prior run directories remain preserved.

Controls: two in-memory 320×240/240×320 colored-rectangle images, same eight-class prompt, FP32, batch 2, seed 42, frozen vision/BERT, 192/256 resize limits and 256×256 batch padding. All model state is reset to the same pretrained state and RNG before each count case. There are **172,249,090 total / 35,836,936 trainable parameters**, unchanged from Stage 3B; no classifier replacement or query-count reduction occurs. Two AdamW steps per case use LR `1e-5`, weight decay `1e-4`, `foreach=False`, clip norm 1, no scheduler/accumulation/augmentation. The original 900-query geometry-capacity guard and pixel/token/target checks remain in use.

The reduced-count/empty controls omit labels on the synthetic rectangles deliberately to test software paths; they do not approve partially labeled real photographs or establish a real no-trash scenario. Known excluded/unlabeled TACO annotations remain under the existing whole-image quarantine policy.

### Final losses and gradient observations

| Case: target counts per image | Total corrected loss, step 1 | Total corrected loss, step 2 | Encoder CE, step 1 / 2 | Encoder L1, step 1 / 2 | Encoder GIoU, step 1 / 2 |
| --- | ---: | ---: | --- | --- | --- |
| Baseline 8 / 3 | 12.700611 | 14.114100 | 2.566194 / 3.136733 | .104934 / .114393 | .737060 / .662438 |
| Two targets 1 / 1 | 12.731905 | 13.204998 | 2.895101 / 2.929698 | .060598 / .076215 | .383864 / .307913 |
| One target 1 / 0 | 14.745838 | 14.014247 | 2.105351 / 2.444302 | .041174 / .029678 | .488563 / .317324 |
| Empty 0 / 0 | 8.863054 | 6.758975 | 1.116389 / 1.164306 | 0 / 0 | 0 / 0 |

Every total/component loss and received gradient was finite. Two-step training losses need not decrease monotonically: dropout, proposal selection and matching can change; these are wiring/numerical checks, not a convergence or recognition experiment. The first baseline value matches Stage 3C's independently computed hypothetical objective. The native pre-correction baseline total (60765.605469) is historical, not a newly run competing objective in this stage.

Baseline first-step component audit:

| Component | Actual value |
| --- | ---: |
| Decoder classification / L1 / GIoU | 1.975004 / .085369 / .596292 |
| Encoder classification / L1 / GIoU | 2.566194 / .104934 / .737060 |
| Global gradient norm before clipping | 56.696583 |
| Encoder CE path to `enc_output.weight`, L2 | 1.348285 |
| Encoder CE path to `enc_output_norm.weight`, L2 | 2.750754 |
| Encoder regression path to final encoder box weight, L2 | 4.341739 |
| Maximum update in each of these three probes | approximately `1.00136e-5` |

The contrastive scorer has no learned classifier parameter of its own: classification supervision now reaches the proposal projection/normalization and fused features. Nonempty cases require positive classification and box-probe gradients, actual nonzero updates and Adam state. All pass. Every declared trainable parameter receives a gradient tensor in all eight steps; none is listed as missing. A present tensor can still be zero, as appropriate for empty regression.

For the fully empty case, encoder box loss/box-path gradients and observed encoder box-weight updates are exactly zero; classification still has finite gradients and updates. A zero regression update here is expected, not a failed qualification. All frozen parameter hashes remain unchanged. Full gradient-path/probe/component values and clipped norms are retained in the ignored JSON report.

### Decoder equivalence and adapter reproducibility

For **all four cases before and after updates**, direct decoder inference and the adapter's decoder output are exactly equal at identical weights: maximum active-logit difference **0.0**, maximum box difference **0.0**. Initial references retain native detachment and exact box values during training calls. Architecture/state-key/shape/dtype hashes are unchanged. Weight hashes also match before/after pure adapter equivalence calls.

Repeated eval-mode adapter calls at fixed weights produce exactly the same decoder outputs and corrected scalar loss. This establishes that the adapter adds no stochastic operation; it does not guarantee bitwise reproducibility of CUDA optimizer trajectories. Tiny between-run variations in later losses are preserved rather than relabeled as exact training replay. No settings were selected based on those variations.

### Checkpoint and replay

After the baseline's two steps, a new local `checkpoint/` stores ordinary HF safetensors/configuration and the processor, plus `training_adapter.json`. The manifest records adapter version `bantai.grounding-dino.encoder-output.v1`, base revision, ordered labels/prompt/token IDs, configuration/loss contract, runtime versions and native/adapter source fingerprints. Adapter source hashes normalize only UTF-8 line endings so the same code survives Git's LF/CRLF conversion; substantive source changes still fail verification.

Offline reload passes empty loading diagnostics, exact model parameter/architecture hashes, identical processor input/target tensors, matching adapter context and exact decoder/total/component-loss equality. A deliberately changed manifest version is rejected. The processor retains the original cached default resolution; replay explicitly reapplies the saved 192/256 smoke override. The trained snapshot remains local and is not loaded by the existing inference pipeline.

**Weights alone do not activate this adapter.** Explicitly reconstruct `EncoderOutputAdapter(restored_model, restored_processor.tokenizer)` and call `verify_manifest` against `training_adapter.json` before adapted training. Calling the plain model with labels would still invoke its unchanged native objective; no global library behavior has been replaced.

`training_context.json` and the report explicitly say **`resume_supported=false`**. Optimizer/scheduler/scaler/RNG state, sampler position and exact continuation state are not saved or restored. This qualifies model/processor/adapter reconstruction, not interrupted-training resume. Production trainer/resume semantics need a separately approved implementation and test before a real-data run.

### Actual GPU memory and timing scope

Peak allocator use across case preprocessing, equivalence calls, diagnostic gradients and the two corrected steps: **2,933,199,360 bytes allocated (2.732 GiB)** / **3,124,756,480 reserved (2.910 GiB)**. Counter maxima can occur in different cases. Checkpoint replay, measured separately while the original model is also resident, peaks at **1,660,589,056 allocated / 3,219,128,320 reserved bytes** (about 1.547 / 2.998 GiB). These are PyTorch allocator peaks, not whole-device `nvidia-smi` peaks.

Final lifecycle time is **39.743 seconds**, including version checks, imports, hashing, preprocessing, equivalence, diagnostic traversals, optimizer steps, saving/reloading and final integrity checks. Step timings include loss/gradient/frozen-hash audits and synchronization, not ordinary training throughput. Equivalence forwards precede updates, and no separate training warm-up/steady-state benchmark was performed. This small frozen-backbone configuration fits locally; original 800/1333 resolution, fully unfrozen training, AMP, larger batches and long-run stability remain unrun.

## PASS/FAIL acceptance and remaining requirements

| Gate | Result |
| --- | --- |
| Selected image-conditioned encoder scores and undetached regression values | PASS |
| Existing native matcher/formula/normalization/masks/coefficients retained | PASS |
| Decoder detach boundary, query count, architecture and decoder inference unchanged | PASS |
| Finite nonempty/partial-empty/fully-empty objective and eight real optimizer steps | PASS |
| Nonempty encoder classification/box gradients, Adam state and actual updates | PASS |
| Expected zero empty regression; frozen parameters unchanged | PASS |
| Repeated adapter results and model/processor/manifest offline reconstruction | PASS |
| No installed-library/inference/base-snapshot/data-policy modification | PASS; preservation checks recorded below |
| Real-data training, convergence, metrics, full-resolution/AMP fit and resume | UNRUN / NOT QUALIFIED |

Regression suite: **144 tests run, 143 passed, one existing Windows privilege skip**; all **11 new adapter tests passed**. Fixtures verify native criterion dispatch/equality, selected-value gradient flow, unchanged parameters/decoder/input tensors, config/text/detach failures, finite empty paths, normalized-target validation, manifest tampering, CLI limits and source-fingerprint portability. They do not replace the actual pinned CUDA evidence.

Final preservation checks passed for both complete package-version inventories, nine protected inference/settings/taxonomy/policy/manifest/earlier-trainer files and three installed Transformers sources. Base-snapshot integrity flags, strict JSON, final implementation hashes and saved checkpoint artifact hashes all pass. The environment/weights/reports remain ignored; only the owner's existing detection settings remain as a separate local edit. All 219 local Markdown links resolve, CLI help and diff whitespace checks pass. After the qualification process exited, device usage was 876 MiB with the desktop active; this is a closure observation, not a whole-device peak or a memory-leak benchmark.

Remaining prerequisites before separately authorizing real-data fine-tuning: complete image/database rights review; accept strict-clean selection bias, rare-class support and leakage/held-out policy; preserve excluded/unlabeled handling explicitly; approve a real-data loader/trainer, complete resumable checkpoints and evaluator/class-span decoding; verify loader coordinates and preprocessing against actual annotations; measure intended real-resolution memory and establish a bounded experiment/evaluation protocol. Synthetic PASS alone does not clear any of these gates. No next stage is started automatically.
