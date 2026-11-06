# Stage 4: pure-plastic-film-damped Epoch-2 continuation (D-064)

Owner-authorized controlled local experiment, 2026-10-25. Exact source: D-057 Epoch 1 step 221. Completed 221 updates in the original Epoch 2 order, two 47-image checkpoint evaluations and one 47-image in-memory blend evaluation (141 validation forwards total). No reference GPU reruns, test access, resampling, loss reweighting, commit, push, merge or checkpoint promotion. All six core training SOURCES files remain byte-for-byte unchanged.

Training metadata confirms 61/221 images (27.6%) have category set exactly {4} (plastic_film), 160 have at least one of the other seven classes, and 30 contain cigarette. Every pure-film batch uses LR 1e-6; all other batches, including all 30 cigarette images and mixed film/non-film images, use LR 1e-5. This is image-composition-dependent LR damping; it is not a balanced sampler or modified native loss. AdamW moments carry across batches, so damping scales the current complete update, including momentum/weight decay. The hypothesized causal role of film dominance is tested, not assumed proved.

Same pinned architecture/Stage 3D adapter, exact E1 model/AdamW/Python/NumPy/CPU/CUDA RNG restore, FP32, TF32 disabled, clip norm 1.0, epoch_batches(221,1,42,1), shortest 384/longest 576 and 576-square padding. Both checkpoint validations snapshot/restore RNG to preserve the training sequence. Official COCO v2 remains no-NMS; D-063 NMS is only a selectable saved-view diagnostic. The D-042 academic-use basis and uncertainty remain unchanged.

Private run: [taco-class-balanced-c3aadd78678548edaf6f8f409ad44044](../runs/training/taco-class-balanced-c3aadd78678548edaf6f8f409ad44044/record.json), [full comparison](../runs/training/taco-class-balanced-c3aadd78678548edaf6f8f409ad44044/class-balanced-summary.json), [ten-GT regression ledger](../runs/training/taco-class-balanced-c3aadd78678548edaf6f8f409ad44044/regression-ledger.json). Full evaluation JSONs, steps.jsonl (LR/categories/IDs/loss components), metrics/class/coverage/per-object CSVs and complete per-object JSON remain local.

## Measured result

CB Step 350 AP **0.258631**, Step 442 **0.242594**, and CB 25/75 blend **0.257303**. Cigarette coverage is 13/24, 14/24, 13/24; Image 777 coverage 4/7 for all three. The fixed ten-regression retention is 4/10, 2/10, 2/10; the nine-large subset is 3/9, 2/9, 2/9. This is partial recovery, not preservation of all desired gains.

At the midpoint, damping retains 4 of the historical ten losses versus 1 under constant LR, while cigarette coverage is 13 versus 14. At the endpoint it retains 2 versus 0, with AP 0.242594 versus 0.170508, but cigarette coverage 14 remains below constant E2's 15. Compared with global cosine decay, endpoint cigarette coverage improves from 10 to 14 and AP 0.221592 to 0.242594. Every one of the 30 cigarette batches received full LR; this does not guarantee retention of every validation cigarette.

The CB blend remains below D-061 blend AP 0.265563 and cigarette coverage 15/24, although it retains 2 of the ten historical regressions versus 1 and has higher AR100 (0.450380 versus 0.403085). Bag AP is 0.528524 versus 0.504950; wrapper AP is 0.105842 versus 0.154653. Overall coverage is 36/75 versus 38/75. These are competing gains/regressions on the same small validation set, not a universally superior checkpoint or proof that film-only batches caused the historical drift. No new default, further training or checkpoint promotion is selected.

## LR assignment totals

| Training image group | Batches | LR applied to every optimizer group |
| --- | --- | --- |
| Pure plastic_film only, category set {4} | 61 | 1e-6 |
| Any other class, including mixed film/non-film | 160 | 1e-5 |
| Cigarette subset of the 160 full-LR batches | 30 | 1e-5 |

All 221 log rows were checked against unchanged train.json. All five checkpoint steps happen to follow a full-LR batch; their last-batch LR column does not describe the entire epoch.

## Checkpoints and exact LR assignment

| Checkpoint | Position | Last-batch LR | Last-batch category IDs |
| --- | --- | --- | --- |
| step-000250 | {'epoch': 1, 'next_batch': 29, 'global_step': 250} | 1e-05 | [0] |
| step-000300 | {'epoch': 1, 'next_batch': 79, 'global_step': 300} | 1e-05 | [3, 7] |
| step-000350 | {'epoch': 1, 'next_batch': 129, 'global_step': 350} | 1e-05 | [5, 6] |
| step-000400 | {'epoch': 1, 'next_batch': 179, 'global_step': 400} | 1e-05 | [1] |
| epoch-02 | {'epoch': 2, 'next_batch': 0, 'global_step': 442} | 1e-05 | [5] |

The new contract binds the LR rule, exact 61/160/30 image-ID sets, source checkpoint/contract anchors, sampler, checkpoint/evaluation steps and validation RNG isolation. Saved optimizer counters equal 250/300/350/400/442 across all parameter states; all 558 trainable names and complete RNG state are retained. The blend is exactly 0.25*cb_step350+0.75*cb_step442 in CPU FP32 over 558 canonical trainable tensors; frozen persisted entries are bitwise identical and tied aliases preserved. No blended checkpoint is saved.

## Overall validation metrics

| State | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | seven_class_AP_excluding_plastic_bag | validation_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrained | 0.147847 | 0.179470 | 0.165594 | 0.000000 | 0.007261 | 0.172653 | 0.426109 | 0.095418 | 9.720814 |
| epoch-01 | 0.249891 | 0.286853 | 0.264765 | 0.060891 | 0.033495 | 0.303687 | 0.488141 | 0.181629 | 5.323725 |
| constant-350 | 0.262337 | 0.288625 | 0.278353 | 0.065773 | 0.033448 | 0.315487 | 0.428890 | 0.218249 | 4.880779 |
| constant-442 | 0.170508 | 0.196086 | 0.179972 | 0.075112 | 0.048109 | 0.206941 | 0.370118 | 0.190859 | 4.613316 |
| integrated | 0.265563 | 0.292911 | 0.280572 | 0.071254 | 0.045737 | 0.316106 | 0.403085 | 0.231365 | 4.659642 |
| cosine-442 | 0.221592 | 0.250303 | 0.241966 | 0.060891 | 0.042679 | 0.266250 | 0.407213 | 0.175050 | 5.151166 |
| cb_step350 | 0.258631 | 0.295110 | 0.279307 | 0.063749 | 0.042538 | 0.309946 | 0.463523 | 0.220556 | 5.055909 |
| cb_step442 | 0.242594 | 0.268843 | 0.255444 | 0.065016 | 0.050450 | 0.288197 | 0.377770 | 0.205115 | 4.763082 |
| cb_blend_2575 | 0.257303 | 0.291342 | 0.278200 | 0.062327 | 0.051883 | 0.306327 | 0.450380 | 0.218557 | 4.841995 |

Seven-class AP is the mean excluding plastic_bag. AP/AR are COCO fractions, not accuracy. Validation loss is the mean native loss per image. All six historical references reuse saved predictions/losses and reproduce their saved COCO metrics on CPU, without reference forwards.

## Eight-class AP / AR100

| Class | pretrained | epoch-01 | constant-350 | constant-442 | integrated | cosine-442 | cb_step350 | cb_step442 | cb_blend_2575 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cigarette | 0.014122 / 0.029167 | 0.051306 / 0.179167 | 0.076901 / 0.329167 | 0.082558 / 0.350000 | 0.081497 / 0.350000 | 0.067446 / 0.283333 | 0.066880 / 0.279167 | 0.086005 / 0.325000 | 0.084090 / 0.325000 |
| plastic_bottle | 0.053966 / 0.425000 | 0.189688 / 0.600000 | 0.236139 / 0.337500 | 0.180281 / 0.337500 | 0.193894 / 0.337500 | 0.166667 / 0.337500 | 0.194010 / 0.387500 | 0.175743 / 0.337500 | 0.193894 / 0.337500 |
| plastic_bag | 0.514851 / 0.700000 | 0.727723 / 0.950000 | 0.570957 / 1.000000 | 0.028053 / 0.500000 | 0.504950 / 0.500000 | 0.547383 / 0.950000 | 0.525157 / 1.000000 | 0.504950 / 0.500000 | 0.528524 / 1.000000 |
| plastic_wrapper | 0.074249 / 0.572727 | 0.041171 / 0.290909 | 0.029703 / 0.081818 | 0.128053 / 0.254545 | 0.154653 / 0.254545 | 0.022277 / 0.081818 | 0.054031 / 0.172727 | 0.108746 / 0.236364 | 0.105842 / 0.236364 |
| plastic_film | 0.002621 / 0.107692 | 0.028424 / 0.530769 | 0.025307 / 0.476923 | 0.033685 / 0.584615 | 0.029740 / 0.576923 | 0.024652 / 0.530769 | 0.024078 / 0.523077 | 0.028166 / 0.546154 | 0.026943 / 0.538462 |
| drink_can | 0.124851 / 0.214286 | 0.118812 / 0.114286 | 0.367162 / 0.385714 | 0.118812 / 0.114286 | 0.367162 / 0.385714 | 0.118812 / 0.114286 | 0.367162 / 0.385714 | 0.244554 / 0.257143 | 0.367162 / 0.385714 |
| plastic_cup | 0.335897 / 0.660000 | 0.404797 / 0.640000 | 0.386139 / 0.380000 | 0.386139 / 0.380000 | 0.386139 / 0.380000 | 0.419095 / 0.520000 | 0.431353 / 0.520000 | 0.386139 / 0.380000 | 0.345545 / 0.340000 |
| styrofoam_piece | 0.062220 / 0.700000 | 0.437207 / 0.600000 | 0.406391 / 0.440000 | 0.406483 / 0.440000 | 0.406466 / 0.440000 | 0.406401 / 0.440000 | 0.406377 / 0.440000 | 0.406451 / 0.440000 | 0.406428 / 0.440000 |

## Locked size-bin and cigarette coverage

Coverage is nonexclusive same-class GT coverage at score>=0.05, IoU>=0.50, NMS off. One query can cover multiple annotations; counts are not official COCO recall, accuracy or unique physical-object counts. Bins remain locked to the same 256/384 reference geometry used by D-058 (14/10/15/36 objects), not recomputed as 384/576 bins.

### all 900 @ 0.05

| State | Correct /75 | Cigarettes /24 | Image 777 /7 | extremely small <4 | very small 4–<8 | small 8–<16 | larger ≥16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pretrained | 29 | 3 | 0 | 0/14 | 2/10 | 4/15 | 23/36 |
| epoch-01 | 34 | 8 | 1 | 2/14 | 4/10 | 7/15 | 21/36 |
| constant-350 | 34 | 14 | 4 | 5/14 | 6/10 | 8/15 | 15/36 |
| constant-442 | 35 | 15 | 5 | 6/14 | 6/10 | 9/15 | 14/36 |
| integrated | 38 | 15 | 5 | 6/14 | 6/10 | 9/15 | 17/36 |
| cosine-442 | 30 | 10 | 3 | 2/14 | 5/10 | 9/15 | 14/36 |
| cb_step350 | 37 | 13 | 4 | 4/14 | 6/10 | 10/15 | 17/36 |
| cb_step442 | 36 | 14 | 4 | 5/14 | 6/10 | 9/15 | 16/36 |
| cb_blend_2575 | 36 | 13 | 4 | 4/14 | 6/10 | 9/15 | 17/36 |

### top100 @ 0.05

| State | Correct /75 | Cigarettes /24 | Image 777 /7 | extremely small <4 | very small 4–<8 | small 8–<16 | larger ≥16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pretrained | 29 | 3 | 0 | 0/14 | 2/10 | 4/15 | 23/36 |
| epoch-01 | 34 | 8 | 1 | 2/14 | 4/10 | 7/15 | 21/36 |
| constant-350 | 34 | 14 | 4 | 5/14 | 6/10 | 8/15 | 15/36 |
| constant-442 | 35 | 15 | 5 | 6/14 | 6/10 | 9/15 | 14/36 |
| integrated | 38 | 15 | 5 | 6/14 | 6/10 | 9/15 | 17/36 |
| cosine-442 | 30 | 10 | 3 | 2/14 | 5/10 | 9/15 | 14/36 |
| cb_step350 | 37 | 13 | 4 | 4/14 | 6/10 | 10/15 | 17/36 |
| cb_step442 | 36 | 14 | 4 | 5/14 | 6/10 | 9/15 | 16/36 |
| cb_blend_2575 | 36 | 13 | 4 | 4/14 | 6/10 | 9/15 | 17/36 |

## Fixed historical E1-to-E2 regression recovery

The requested ten-loss ledger is held fixed before this experiment: GT 27,104,177,250,258,303,404,414,423,443. Nine are in the locked >=16-pixel bin; GT 27 (plastic_bottle) is 8--<16. We report all ten requested regressions AND the nine-large subset rather than calling all ten large. Their saved E2 winning wrong classes include plastic_film on 8 of 10; that association does not establish the optimization cause. Correct below means a qualifying same-class query remains at all 900/0.05/IoU 0.50, not a human material judgment.

| State | Correct among ten /10 | Correct among large nine /9 |
| --- | --- | --- |
| pretrained | 7 | 7 |
| epoch-01 | 10 | 9 |
| constant-350 | 1 | 1 |
| constant-442 | 0 | 0 |
| integrated | 1 | 1 |
| cosine-442 | 2 | 2 |
| cb_step350 | 4 | 3 |
| cb_step442 | 2 | 2 |
| cb_blend_2575 | 2 | 2 |

| GT | Image | GT class | Locked bin | Constant E2 wrong class | CB350 correct | CB442 correct | CB blend correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 113 | plastic_bottle | small 8–<16 | plastic_film | True | False | False |
| 104 | 252 | plastic_wrapper | larger ≥16 | plastic_film | True | True | True |
| 177 | 464 | styrofoam_piece | larger ≥16 | plastic_wrapper | False | False | False |
| 250 | 762 | plastic_wrapper | larger ≥16 | plastic_film | False | False | False |
| 258 | 776 | plastic_bag | larger ≥16 | plastic_film | False | False | False |
| 303 | 833 | plastic_bag | larger ≥16 | plastic_wrapper | True | True | True |
| 404 | 1076 | plastic_cup | larger ≥16 | plastic_film | False | False | False |
| 414 | 1096 | plastic_bottle | larger ≥16 | plastic_film | False | False | False |
| 423 | 1126 | plastic_cup | larger ≥16 | plastic_film | True | False | False |
| 443 | 1156 | plastic_wrapper | larger ≥16 | plastic_film | False | False | False |

## Passing-query context (diagnostic only)

| State | Passing all 47 at.05 | Passing Image 777 at.05 |
| --- | --- | --- |
| pretrained | 4894 | 86 |
| epoch-01 | 1067 | 26 |
| constant-350 | 1224 | 34 |
| constant-442 | 1118 | 29 |
| integrated | 1152 | 31 |
| cosine-442 | 1133 | 28 |
| cb_step350 | 1246 | 34 |
| cb_step442 | 1213 | 34 |
| cb_blend_2575 | 1218 | 32 |

Lower candidate counts alone do not establish fewer false positives. Saved-view NMS may further alter displayed counts/coverage; it does not change these no-NMS official metrics.

## Safeguards and evidence

All 55 prior bundles verified before and after; all 5 new bundles verified afterward (60 total). Protected audit: 2005 prior files unchanged, including prior run artifacts, pinned snapshot, original train/validation photos, source/rights metadata, six core files and dependency/ignore rules. Photo-preservation hashing is separate from dataset decodes: 221 training and 141 validation decodes. No test.json or test photos were opened. The only later prior-artifact edit is the specifically requested D-060 manifest extension, archived before modification.

Exact restore digests verify E1 model weights, AdamW moments/groups/order and all RNG state; training_state.pt is loaded with weights_only=True. All 221 LR assignments and category sets match unchanged train.json, 61 damped and 160 full-LR, including 30/30 full-LR cigarette batches. Frozen backbone hashes match E1 after every checkpoint and validation; validation preserves all parameters and restores RNG exactly. Blend sources have identical frozen persisted state; only trainable tensors are averaged. A separate blend process guarantees GPU context release after training; no reference is rerun and no integration weights are promoted or persisted.

Pre-training guarded suite: 254 tests, zero failures/errors, 16 explicit skips. Final suite/native GUI evidence is archived separately. Generated software fixtures are distinct from real training/validation evidence; human visual acceptance remains pending. Repeated comparisons on this 47-image validation set retain selection bias and do not establish generalization or a checkpoint-selection rule.

## Saved GUI replay

The main choice shows CB350, CB442 and CB 25/75 blend beside GT. The companion puts the saved D-061 blend beside CB442 and the CB blend. All nine previous choices remain unchanged. Both GUI modes retain the D-063 Diagnostic NMS selector with raw queries as default. Each command starts Image 777 with all 47 validation images selectable; close one window before launching the next.

```powershell
$review = "runs/training/d060-saved-review/review-manifest.json"
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576 (class-balanced)" --gui --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576 (class-balanced)" --gui --per-object --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576 (class-balanced vs integrated)" --gui --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576 (class-balanced vs integrated)" --gui --per-object --image-id 777
```

Final guarded unit suite: 255 tests, 239 passed, 16 documented skips, zero failures/errors. Native GUI evidence is recorded separately; human visual acceptance remains pending.

Native Windows Tk checks passed four cases (two choices, standard/per-object), including all three D-063 NMS modes per case (12 control subchecks). Image 777, all 47 selectable IDs, explicit labels, image/GT navigation and zoom/pan passed. Saved JSON hashes stayed unchanged. Model/GPU calls, weight reads and image exports were blocked. During the earlier model run, dataset decodes were 221 training and 141 validation; native replay subsequently reads validation originals only, without model forwards. Human visual acceptance remains pending. See native-results.json in the private run.
