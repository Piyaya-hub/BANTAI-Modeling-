# Stage 3C: native training loss and gradient investigation

Historical investigation: D-040 subsequently approved [Stage 3D's separate training-only adapter and synthetic qualification](TACO_STAGE_3D.md). The proposal/approval boundary below describes Stage 3C; Stage 3D neither patches this native library nor authorizes real-data fine-tuning.

Executed on the owner's Windows GTX 1660 Ti on 2026-10-18 under D-039; report finalized 2026-10-19 (Asia/Manila). **Do not proceed to real-image fine-tuning with the current native objective unchanged.** Evidence identifies encoder-supervision tensor wiring as the problem, rather than malformed eight-class targets, a failed Hungarian matcher or numerical overflow. The Stage 3B lifecycle pass remains valid, but did not establish that the encoder objective trains the intended branches.

Recommended next stage: owner-approved **Stage 3D, a training-only encoder-output adapter and synthetic qualification**. Preserve decoder initialization detachment, query count, architecture, loss formula/coefficients, checkpoint and inference environment. No such adapter, Transformers patch, alternate optimizer objective or real-data trainer was installed by Stage 3C. The hypothesis calculations below are diagnostic recomputations only.

## Questions answered

1. **Why no encoder box gradients?** HF 4.51.3 gathers top-k proposal coordinates, detaches them for decoder reference points, then reuses those detached coordinates as encoder box predictions supplied to the loss. The separately parameterized encoder box head therefore has no supervised path. Detaching decoder initialization is expected; using only that copy for a nonzero encoder regression objective is problematic for intended end-to-end proposal-head training.
2. **Why the large encoder classification loss?** With pinned `embedding_init_target=true`, the reported encoder logits are recomputed from learned decoder query embeddings and pre-encoder text features. Those are different representations from the image-conditioned encoder proposal scores used for top-k selection. Runtime reproduction confirms the reported logits contain no image-specific information for a shared prompt, have many large positive unmatched-token scores, and drive a large focal negative term.
3. **Is the current approach suitable?** Exact checkpoint loading, preprocessing, native decoder loss, optimizer execution and persistence work in the tested synthetic scope. The unchanged native encoder objective is not a suitable qualified end-to-end training baseline for BANTAI yet.
4. **Is a correction necessary?** Evidence supports correcting which encoder tensors reach the existing criterion, retaining an undetached coordinate copy for supervision and using selected proposal logits. Lowering weights, dropping encoder terms or changing normalization to hide the magnitude is not justified by these results.
5. **Safest next step?** Approve the narrowly scoped adapter and repeat synthetic update/inference-equivalence/save-reload checks before any real-photo approval. Rights, coverage, evaluation and real-resolution memory gates remain independent.

## Source trace and interpretation

The installed files were hashed before/after execution; none changed. Relevant pinned source:

| Source | Relevant path |
| --- | --- |
| [HF 4.51.3 model](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/grounding_dino/modeling_grounding_dino.py) | `GroundingDinoModel.forward`, installed lines 2327–2358: proposal scoring, gathering, detachment, learned-query initialization and reported encoder outputs. Detection forward lines 2669–2683 dispatches those outputs to the criterion. |
| [HF 4.51.3 Grounding DINO criterion](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/loss/loss_grounding_dino.py) | Matcher, span-positive target construction, focal loss, and separate encoder/decoder supervision. |
| [HF 4.51.3 shared detection loss](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/loss/loss_for_object_detection.py) | Box L1/GIoU, object-count normalization with minimum 1, and diagnostic cardinality. |

The pinned encoder has independent box-head parameters (`two_stage_bbox_embed_share=false`). Selection indices are discrete, so decoder supervision does not differentiate through top-k ranking. Its reference-coordinate initialization is also detached. Both are normal boundaries; neither requires that the separate supervised encoder coordinates be detached. Frozen backbones, no-match empty regression losses, logging-only cardinality and unused/tied aliases can legitimately have absent or zero gradients. The conclusion here instead uses nonempty targets, an independent trainable encoder head, explicit nonzero regression weights, source routing and actual lack of updates together.

For comparison, [IDEA's transformer](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/models/GroundingDINO/transformer.py) retains `refpoint_embed_undetach` and separate encoder features while detaching decoder initialization. Its [public model file](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/models/GroundingDINO/groundingdino.py) comments out intermediate training outputs, so it is not evidence of an executable full released training recipe. [MMDetection's Grounding DINO pre-decoder](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/grounding_dino.py) explicitly retains gathered top-k scores/coordinates for training and a detached reference copy for the decoder. These mutable upstream references were inspected on the assessment date, not installed or claimed equivalent to the exact HF checkpoint. No maintainer acknowledgement of this particular version's behavior was obtained; the assessment rests on source and controlled observations.

## Experimental controls and files

| File | Purpose |
| --- | --- |
| [scripts/investigate_training_loss.py](../scripts/investigate_training_loss.py) | Bounded synthetic diagnostic CLI; no dataset/photo path, alternate checkpoint, install or patch operation. |
| [scripts/loss_investigation.py](../scripts/loss_investigation.py) | Reuses existing synthetic preprocessing; audits three tensor branches, component gradients, independent focal arithmetic and native optimizer updates. |
| [tests/test_loss_investigation.py](../tests/test_loss_investigation.py) | Seven regression/causal fixtures for count controls, masks/normalization, expected detach boundary, diagnostic autograd isolation, empty targets and large unmatched positive logits. |
| This report plus AGENTS/README/phase/scope/decision/test/handoff updates | Actual state, evidence, proposal and required approval boundary. |

The original [Stage 3B trainer](../scripts/training_smoke.py), dependencies, eight-class taxonomy, source photos/annotations, dataset splits, inference code and both environments remain unchanged. Only `.venv-training` executes the investigation. Current versions are Python 3.11.9, torch 2.7.1+cu118, Transformers 4.51.3, SciPy 1.15.3; no package installation was needed.

All cases reuse the same two in-memory rectangle images, ordered eight-class prompt, FP32, batch 2, 192/256 edge limits, 900 queries, frozen vision/BERT, seed 42 and native coefficient values. Before each case, every state tensor is reset to the same pretrained value and the RNG is reset. Cases change only target lists: 8+3, 1+1, 1+0, and 0+0. Reduced-count/empty fixtures deliberately omit drawn rectangles as a software control; they are **not** a proposed partially labeled real-data policy or reviewed negative scenes.

Each case does one AdamW step using **only the original model-returned native total**, LR `1e-5`, weight decay `1e-4`, clip norm 1. Diagnostic `autograd.grad` calls do not populate optimizer `.grad`; this is asserted before the native backward. Hypothesis terms never enter an optimizer objective or saved checkpoint. A read-only classifier pre-hook records its two input calls and is removed immediately; it does not replace any output.

### Actual commands and runs

```powershell
.venv-training/Scripts/python.exe scripts/investigate_training_loss.py --device cuda --cases baseline_11 two_targets one_target empty
.venv-training/Scripts/python.exe scripts/investigate_training_loss.py --device cuda --cases baseline_11
.venv-training/Scripts/python.exe -m unittest discover -s tests -p test_loss_investigation.py -q
.venv-training/Scripts/python.exe -m unittest discover -s tests -q
```

Each invocation creates `runs/training/investigation-<uuid>/report.json`, ignored by Git, with actual component values, token/match diagnostics, eight representative parameter probes, gradients, optimizer changes, memory, environment and source hashes. It returns 0 for completed diagnostics, 1 for a recorded failure, 2 for invalid arguments; `completed` is not a training-readiness flag. It never automatically falls back from CUDA or changes settings.

| Local run suffix | Outcome |
| --- | --- |
| `bc67da89b13e4679938f78260899d46a` | Inspection-tool failure before experiments: a tied decoder box parameter was sought under its alias rather than its canonical name (`StopIteration`). Fixed probe lookup, not model/library code. Retained as a failed attempt. |
| `7624daeb430746e5b2d424da3998295b` | Completed all four target-count cases, all component/gradient audits and four independent native updates. |
| `ac08f80e30274c76b95279d0eecfe9f9` | Completed baseline factor-isolation audit: box detach boundary alone versus classification inputs alone, plus one independent native update. Final diagnostic implementation hashes match this run. |

## Loss accounting and target-count observations

| Target count (image 1 / image 2) | Normalizer | Native decoder CE | Native encoder CE | Native total | Selected-encoder CE, hypothesis only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 / 3 | 11 | 1.975004 | 30378.000000 | 60765.605469 | 2.566194 |
| 1 / 1 | 2 | 2.261436 | 167259.453125 | 334528.281250 | 2.895101 |
| 1 / 0 | 1 | 4.062028 | 334564.125000 | 669141.187500 | 2.105351 |
| 0 / 0 | 1 | 3.315138 | 334579.812500 | 669166.250000 | 1.116389 |

All losses and received gradients are finite; no OOM or CUDA failure occurred. Empty-target matching, native classification backward and the optimizer step succeeded. Its box losses are exactly zero. In the connected hypothesis branch, empty box gradients are zero on reachable paths, as expected; they are not evidence of a broken encoder.

The target/matcher/normalization audit found:

- Span maps preserve approved IDs/order and all ordinary subword tokens. Native matching uses focal class cost averaged across each class's token span, plus L1/GIoU costs 1/5/2. Matching is recomputed per branch; its indices are recorded. This single-process run has no distributed normalization or reduction.
- Two images × 900 queries × 30 active prompt tokens give **54,000 active query/token cells**; the baseline has 30 positive cells after matching. Beyond-prompt `-inf` padding is masked out. Periods/CLS/SEP are active negative targets under the native attention-mask rule. Independent elementwise focal reconstruction agrees within FP32 tolerance with native CE.
- Baseline native encoder focal loss is essentially all negatives: **30378.0** negative versus approximately **3.2e-19** positive. Special/delimiter positions contribute 19412.4955 (63.9%); ordinary positions still contribute 10965.5043. Removing delimiters would not resolve the underlying mismatch and was not tried as a training change.
- Focal loss sums active cells and divides by `max(number_of_boxes,1)`. Baseline unnormalized encoder focal sum is **334158**. With fewer targets most unmatched contributions remain, while the divisor falls; thus sparse/empty native losses grow sharply. This normalizer is an expected detection convention, not independently diagnosed as a defect. Encoder/decoder weights remain 2/5/2; cardinality is excluded from the optimized total.

## Why classification is different

For the baseline, hook inputs and outputs establish exact learned-query-dot-text reproduction of reported encoder logits. They are identical between the two different images and across target-count cases before updates. Selected actual encoder logits differ between images. Mean text-feature norm is **128.3993** for the pre-encoder inputs used by reported encoder CE versus **4.9987** in actual proposal scoring. Their respective query/visual-feature norms are 1.6260 versus 8.4878. These are measured feature-space differences, not a proposal to apply arbitrary normalization.

| Active logit statistic | Reported native encoder | Selected actual encoder | Decoder |
| --- | ---: | ---: | ---: |
| Minimum | -162.84694 | -9.76664 | -8.55079 |
| Mean | 5.31687 | -6.22954 | -5.40714 |
| Maximum | 74.89127 | -0.36482 | -0.02891 |
| Fraction positive | 0.63163 | 0 | 0 |

Widespread large positive scores on unmatched cells saturate the focal factor and produce a large negative-target BCE term. The selected image-conditioned scores, using the same labels/matcher/mask/criterion, do not exhibit this scale. Synthetic shapes are not meaningful trash examples, so low logits or low hypothesis loss do not establish useful recognition. The causal finding is the different tensor path, not an expectation that all pretrained losses must be small.

## Gradient and update observations

The following baseline norms use individual components **before weighting/clipping**, through representative named tensors; “absent” means no graph path (`None`), distinct from a zero gradient. Full names and additional values are in JSON.

| Probe | Native encoder CE L2 | Decoder CE L2 | Selected-encoder CE L2, audit only | Selected-encoder box L2, audit only |
| --- | ---: | ---: | ---: | ---: |
| `query_position_embeddings.weight` | 8408.1885 | 0.81291 | absent | absent |
| `text_projection.weight` | 11695.7031 | 0.27194 | 0.45574 | 0.28476 |
| `enc_output.weight` | absent | absent | 1.34828 | 0.71611 |
| `enc_output_norm.weight` | absent | absent | 2.75075 | 0.31877 |
| Encoder box head, last weight | absent | absent | absent | 4.34174 |
| Encoder layer 0, text attention query | absent | 0.07971 | 0.06765 | 0.04647 |
| Decoder final norm | absent | 0.92751 | absent | absent |

Native encoder L1/GIoU have `requires_grad=false` even for nonempty targets. The three proposal-related probes have no native total gradient, no Adam state and **exactly zero update**; Stage 3B recorded ten affected encoder tensors overall. Learned query/text projection and decoder/fusion probes do receive native gradients and nonzero updates (~`1e-5` maximum absolute). Thus the native loop is not wholly inert. It is training a different classification path while leaving the independent proposal head unchanged.

Native total gradient norms are 28920.77, 159133.91, 318290.44 and 318295.81 for the four count cases. Global clipping is dominated by encoder query/text gradients; nevertheless AdamW still produces nonzero decoder updates. Do not equate a smaller clipped gradient with proven lack of learning or predict real-photo accuracy from these probes. All frozen parameter hashes remain unchanged.

### Factor isolation and smallest proposed correction

The baseline diagnostic recomputations keep the original decoder outputs and criterion:

| Encoder inputs, diagnostic only | CE | L1 | GIoU | Encoder box head receives a path? |
| --- | ---: | ---: | ---: | --- |
| Native scores + detached coordinates | 30378.0000 | 0.324411 | 1.207315 | No |
| Native scores + same coordinates before detach | 30378.0000 | 0.324411 | 1.207315 | Yes, box-probe L2 3.02666 |
| Selected proposal scores + detached coordinates | 2.566194 | 0.104934 | 0.737060 | No |
| Selected proposal scores + coordinates before detach | 2.566194 | 0.104934 | 0.737060 | Yes, box-probe L2 4.34174 |

Coordinates before/after detachment are bitwise equal. Changing classification inputs changes Hungarian assignment, explaining the changed regression **values** despite identical coordinates. Merely preserving the regression graph changes gradients without changing any loss value. The hypothetical combined weighted encoder+unchanged-decoder total is **12.700610** on this baseline; it was never used for an optimizer step.

Proposed correction, **not an approved training implementation**: select the same 900 indices from existing `enc_outputs_class`; gather those scores and `enc_outputs_coord_logits.sigmoid()` **before** detachment for encoder supervision. Keep the detached coordinate copy for decoder initialization. Feed these tensors into the same native matcher/criterion with unchanged label maps, masks and weights. This can live in an explicit training-only wrapper/adapter without editing installed Transformers, changing architecture/weights or touching inference. Avoid an unreviewed global monkey patch. A newer library or MMDetection is a fallback requiring its own exact-checkpoint/conversion qualification, not an automatic replacement.

## Feasibility, verification and acceptance limits

Both completed runs record unchanged pretrained snapshot and installed library hashes. Target geometry/spans are checked with the unchanged Stage 3B helper. No candidate objective is optimized and no new checkpoint is saved. Stage 3B's verified native model/processor round trip remains historical evidence; a future adapter must qualify its own saved context, load consistency and resume semantics.

GPU allocator peaks across the four cases: **2,721,721,344 bytes allocated (2.535 GiB)**, **3,049,259,008 reserved (2.840 GiB)**. These include read-only diagnostic gradient traversals plus one native backward/update per reset case; no warm-up or original-resolution test occurred. Case wall times including audits are 5.740, 1.525, 1.509 and 1.489 seconds. They are not ordinary training throughput or whole-device peak memory. The small frozen-backbone configuration fits; full-resolution 800/1333, all-parameter/AMP training, large batches and long-run stability remain unrun.

Actual tests: **133 run, 132 passed, one existing Windows privilege skip**; all seven new targeted tests passed. Software fixtures support the causal distinctions but do not replace the actual pinned CUDA observations. FP32 reduction order can produce tiny reconstruction differences, and deterministic seeds do not guarantee bitwise CUDA training reproducibility.

Final preservation checks passed for both environments' complete package metadata, eight protected setting/source/manifest hashes, three installed library source hashes, strict report JSON and final diagnostic implementation hashes. The unchanged Stage 3B trainer is among those protected files. All 208 local Markdown file links resolve, CLI help and diff whitespace checks pass. After process exit, whole-device usage was 803 MiB with the desktop still active; this is a closure observation, not a peak measurement. Generated investigation reports remain ignored, and the owner's local detection edit remains outside the commit.

Before any real-image stage, require: approved adapter or alternative; nonempty encoder-head updates and empty-case stability under that actual objective; unchanged decoder outputs at identical weights; unchanged model/processor save-reload and explicit resume support; measured intended-resolution memory; pinned evaluator/class-span decoding with held-out policy; image/database rights clearance; rare-class coverage, leakage and excluded/unlabeled handling decisions. The eight labels and v1 splits remain intact and no real-data readiness is claimed.

**Approval requested for the next stage only:** implement and qualify the proposed external training-only tensor adapter using synthetic data, without real-photo fine-tuning, package upgrades or architecture changes. Stage 3C stops before that substantive method change.
