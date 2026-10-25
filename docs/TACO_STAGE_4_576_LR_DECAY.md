# Stage 4: controlled Epoch-2 cosine LR continuation (D-062)

Owner-authorized local run, 2026-10-25. Completed exactly **221 Epoch-2 optimizer updates** (steps 222..442), **221 training forwards** and **94 validation forwards** (47 at 350 and 47 at 442). There was no E1 replay, warm-up, pilot, reference GPU rerun, test access, commit, push, merge or checkpoint promotion. All six core training SOURCES files remain byte-for-byte unchanged.

The only optimization change is the per-step learning rate. The verified D-057 E1 checkpoint restores model weights, AdamW moments/groups and Python/NumPy/CPU/CUDA RNG state exactly. Same epoch_batches(221,1,42,1) order, batch size 1, eight-class prompt, native Stage3D loss/adapter, FP32, TF32 disabled, clip norm 1.0, shortest 384/longest 576 and 576-square padding. Midpoint validation snapshots/restores RNG state so evaluation cannot perturb the remaining stochastic training sequence. The D-042 academic-use basis and uncertainty remain unchanged.

Private run: [taco-lr-decay-702328b3a61c4e94a29a15543e1f2af2](../runs/training/taco-lr-decay-702328b3a61c4e94a29a15543e1f2af2/record.json), [full summary](../runs/training/taco-lr-decay-702328b3a61c4e94a29a15543e1f2af2/decay-summary.json), and [new checkpoint state verification](../runs/training/taco-lr-decay-702328b3a61c4e94a29a15543e1f2af2/checkpoint-state-verification.json). Full evaluations, steps.jsonl, metrics/per-class/coverage/per-object CSVs and full per-object JSON remain local.

## Measured outcome

Cosine step 350 AP **0.217172**; step 442 AP **0.221592**, above constant-LR E2 0.170508 but below E1 0.249891 and D-061 blend 0.265563. Endpoint cigarette coverage is 10/24 overall and 3/7 on Image 777; large-bin coverage 14/36. Passing queries at 0.05: 1133 across 47 images and 28 on Image 777, versus constant E2 1118/29.

The schedule avoids most of the constant-LR bag AP collapse (endpoint 0.547383 versus 0.028053), but does not preserve the complete desired behavior. Endpoint large-bin coverage remains 14/36, the same as constant E2 and below E1's 21/36. Drink-can AP remains 0.118812, below constant step 350/the blend's 0.367162. Cigarette coverage loses five GT instances versus E2/the blend (10 versus 15); total correct-class coverage is 30/75 versus 35/75 for E2 and 38/75 for the blend.

Clutter does not decrease uniformly: endpoint passing-query count rises from 1118 to 1133 across 47 images while Image 777 decreases from 29 to 28. The no-GT-IoU>=.50 proxy declines from 936 to 911 across 47 images but stays 21 on Image 777. These are candidate/localization diagnostics, not verified false-positive counts. This run does not demonstrate the combined preservation-and-background-reduction hypothesis, and no further experiment or promotion follows automatically.

## Schedule and checkpoints

`lr(k) = 1e-6 + 0.5 * (5e-6 - 1e-6) * (1 + cos(pi*k/220))`, integer k=0..220, global_step=222+k. Every parameter group receives the same LR before its optimizer step; restored moments are retained. Step 222 uses 5e-6; step 332 uses 3e-6; step 442 uses 1e-6.

| Checkpoint | Position | LR | Optimizer step counters |
| --- | --- | --- | --- |
| step-000250 | {'epoch': 1, 'next_batch': 29, 'global_step': 250} | 4.84224730623e-06 | [250.0] |
| step-000300 | {'epoch': 1, 'next_batch': 79, 'global_step': 300} | 3.88244220249e-06 | [300.0] |
| step-000350 | {'epoch': 1, 'next_batch': 129, 'global_step': 350} | 2.49156333161e-06 | [350.0] |
| step-000400 | {'epoch': 1, 'next_batch': 179, 'global_step': 400} | 1.34905620607e-06 | [400.0] |
| epoch-02 | {'epoch': 2, 'next_batch': 0, 'global_step': 442} | 1e-06 | [442.0] |

New contracts retain the six-file D-057 source identity and bind source manifest/contract, start position, exact cosine formula, sampler, checkpoint/evaluation steps and validation RNG isolation. These are separate continuation checkpoints; old contracts and bundles are unchanged.

## Overall validation comparison

| State | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | seven_class_AP_excluding_plastic_bag | validation_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrained | 0.147847 | 0.179470 | 0.165594 | 0.000000 | 0.007261 | 0.172653 | 0.426109 | 0.095418 | 9.720814 |
| epoch-01 | 0.249891 | 0.286853 | 0.264765 | 0.060891 | 0.033495 | 0.303687 | 0.488141 | 0.181629 | 5.323725 |
| constant-350 | 0.262337 | 0.288625 | 0.278353 | 0.065773 | 0.033448 | 0.315487 | 0.428890 | 0.218249 | 4.880779 |
| constant-442 | 0.170508 | 0.196086 | 0.179972 | 0.075112 | 0.048109 | 0.206941 | 0.370118 | 0.190859 | 4.613316 |
| integrated | 0.265563 | 0.292911 | 0.280572 | 0.071254 | 0.045737 | 0.316106 | 0.403085 | 0.231365 | 4.659642 |
| cosine-350 | 0.217172 | 0.245787 | 0.233905 | 0.060891 | 0.038860 | 0.262054 | 0.432952 | 0.170555 | 5.207454 |
| cosine-442 | 0.221592 | 0.250303 | 0.241966 | 0.060891 | 0.042679 | 0.266250 | 0.407213 | 0.175050 | 5.151166 |

Seven-class AP is the mean of seven per-class AP values excluding plastic_bag; official eight-class COCO metrics are unchanged. Loss is the mean native validation loss per image, not accuracy. All references reuse saved predictions/losses, with CPU-only COCO reproduction.

## Eight-class AP / AR100

| Class | pretrained | epoch-01 | constant-350 | constant-442 | integrated | cosine-350 | cosine-442 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cigarette | 0.014122 / 0.029167 | 0.051306 / 0.179167 | 0.076901 / 0.329167 | 0.082558 / 0.350000 | 0.081497 / 0.350000 | 0.057158 / 0.258333 | 0.067446 / 0.283333 |
| plastic_bottle | 0.053966 / 0.425000 | 0.189688 / 0.600000 | 0.236139 / 0.337500 | 0.180281 / 0.337500 | 0.193894 / 0.337500 | 0.151223 / 0.387500 | 0.166667 / 0.337500 |
| plastic_bag | 0.514851 / 0.700000 | 0.727723 / 0.950000 | 0.570957 / 1.000000 | 0.028053 / 0.500000 | 0.504950 / 0.500000 | 0.543494 / 1.000000 | 0.547383 / 0.950000 |
| plastic_wrapper | 0.074249 / 0.572727 | 0.041171 / 0.290909 | 0.029703 / 0.081818 | 0.128053 / 0.254545 | 0.154653 / 0.254545 | 0.019802 / 0.072727 | 0.022277 / 0.081818 |
| plastic_film | 0.002621 / 0.107692 | 0.028424 / 0.530769 | 0.025307 / 0.476923 | 0.033685 / 0.584615 | 0.029740 / 0.576923 | 0.025852 / 0.530769 | 0.024652 / 0.530769 |
| drink_can | 0.124851 / 0.214286 | 0.118812 / 0.114286 | 0.367162 / 0.385714 | 0.118812 / 0.114286 | 0.367162 / 0.385714 | 0.118812 / 0.114286 | 0.118812 / 0.114286 |
| plastic_cup | 0.335897 / 0.660000 | 0.404797 / 0.640000 | 0.386139 / 0.380000 | 0.386139 / 0.380000 | 0.386139 / 0.380000 | 0.414681 / 0.660000 | 0.419095 / 0.520000 |
| styrofoam_piece | 0.062220 / 0.700000 | 0.437207 / 0.600000 | 0.406391 / 0.440000 | 0.406483 / 0.440000 | 0.406466 / 0.440000 | 0.406356 / 0.440000 | 0.406401 / 0.440000 |

## Fixed-bin and cigarette coverage

Coverage is nonexclusive same-class GT coverage at score>=0.05 and IoU>=0.50, not official COCO recall, accuracy or physical-object counts. Size-bin membership remains locked to the same 256/384 reference geometry as D-058: 14/10/15/36 annotations. It is not recomputed or relabeled as 384/576 geometry.

### all900 @ 0.05

| State | Correct /75 | Cigarettes /24 | Image 777 /7 | extremely small <4 | very small 4–<8 | small 8–<16 | larger ≥16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pretrained | 29 | 3 | 0 | 0/14 | 2/10 | 4/15 | 23/36 |
| epoch-01 | 34 | 8 | 1 | 2/14 | 4/10 | 7/15 | 21/36 |
| constant-350 | 34 | 14 | 4 | 5/14 | 6/10 | 8/15 | 15/36 |
| constant-442 | 35 | 15 | 5 | 6/14 | 6/10 | 9/15 | 14/36 |
| integrated | 38 | 15 | 5 | 6/14 | 6/10 | 9/15 | 17/36 |
| cosine-350 | 32 | 10 | 3 | 2/14 | 5/10 | 9/15 | 16/36 |
| cosine-442 | 30 | 10 | 3 | 2/14 | 5/10 | 9/15 | 14/36 |

### top100 @ 0.05

| State | Correct /75 | Cigarettes /24 | Image 777 /7 | extremely small <4 | very small 4–<8 | small 8–<16 | larger ≥16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pretrained | 29 | 3 | 0 | 0/14 | 2/10 | 4/15 | 23/36 |
| epoch-01 | 34 | 8 | 1 | 2/14 | 4/10 | 7/15 | 21/36 |
| constant-350 | 34 | 14 | 4 | 5/14 | 6/10 | 8/15 | 15/36 |
| constant-442 | 35 | 15 | 5 | 6/14 | 6/10 | 9/15 | 14/36 |
| integrated | 38 | 15 | 5 | 6/14 | 6/10 | 9/15 | 17/36 |
| cosine-350 | 32 | 10 | 3 | 2/14 | 5/10 | 9/15 | 16/36 |
| cosine-442 | 30 | 10 | 3 | 2/14 | 5/10 | 9/15 | 14/36 |

## Passing prediction and clutter diagnostics

Each count is a valid saved query with winning-class score>=0.05. Extra candidates can be duplicates, wrong labels or unannotated/background regions; a lower count alone does not establish fewer false positives. The last two columns count queries whose maximum IoU with any GT is below 0.50, a diagnostic localization proxy rather than verified background truth.

### all900 @ 0.05

| State | Passing all 47 | Passing Image 777 | No GT IoU>=.50 all 47 | No GT IoU>=.50 Image 777 |
| --- | --- | --- | --- | --- |
| pretrained | 4894 | 86 | 4360 | 67 |
| epoch-01 | 1067 | 26 | 810 | 17 |
| constant-350 | 1224 | 34 | 1023 | 26 |
| constant-442 | 1118 | 29 | 936 | 21 |
| integrated | 1152 | 31 | 971 | 22 |
| cosine-350 | 1113 | 27 | 883 | 20 |
| cosine-442 | 1133 | 28 | 911 | 21 |

### top100 @ 0.05

| State | Passing all 47 | Passing Image 777 | No GT IoU>=.50 all 47 | No GT IoU>=.50 Image 777 |
| --- | --- | --- | --- | --- |
| pretrained | 4342 | 86 | 3828 | 67 |
| epoch-01 | 1067 | 26 | 810 | 17 |
| constant-350 | 1224 | 34 | 1023 | 26 |
| constant-442 | 1118 | 29 | 936 | 21 |
| integrated | 1152 | 31 | 971 | 22 |
| cosine-350 | 1113 | 27 | 883 | 20 |
| cosine-442 | 1133 | 28 | 911 | 21 |

## Execution safeguards and limits

All 50 original bundles verified before/after and all 5 new bundles verified after execution, 55 total. Protected inventory: 1888 unchanged files, including prior training artifacts, original train/validation photos, source metadata, pinned snapshot and six core files. Train/validation photo hashes are separate preservation reads; dataset decodes were exactly 221 train and 94 validation. No test metadata/photos were opened. The sole prior-artifact edit afterward is the explicitly requested D-060 manifest extension, with its previous contents archived.

Restore evidence: exact source position {'epoch': 1, 'next_batch': 0, 'global_step': 221}, expected E1 model hash, identical optimizer digest and identical Python/NumPy/CPU/CUDA RNG digest. training_state.pt was loaded only with weights_only=True. Every saved bundle's complete optimizer/RNG metadata was independently inspected on CPU: 558 trainable names, expected global position, scheduled group LR and AdamW step counters. Frozen backbone parameters were bitwise unchanged after each of five checkpoints and both validations. Evaluations preserve all parameters and restore RNG exactly. Training/backward/optimizer/validation counters enforce221/221/221/94; unplanned forwards, SGD, direct autograd, network and image exports are blocked. Core bytes before/after and full forward audit are in record.json.

Guarded software suite: 245 tests, zero failures/errors, 16 documented skips. Training-execution fixtures were excluded from the software gate; the D-062 orchestration test mocks train_step/evaluate and verifies all 221 assignments and positions without extra updates. This small fixed validation set and its repeated inspection retain selection bias; no result establishes generalization, deployment readiness or checkpoint promotion. Human visual acceptance remains pending.

## Saved GUI replay

The main choice shows constant-LR step 350, cosine step 350 and cosine step 442 beside GT. The companion substitutes the D-061 25/75 blend for the reference panel. Existing seven choices remain unchanged. Saved configuration includes the actual schedule, source checkpoint and per-panel LR. Both modes start at Image 777 with all 47 validation IDs selectable. Close one window before starting the next.

```powershell
$review = "runs/training/d060-saved-review/review-manifest.json"
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5->cosine @ 384/576 (lr-decay)" --gui --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5->cosine @ 384/576 (lr-decay)" --gui --per-object --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5->cosine @ 384/576 (lr-decay vs integrated)" --gui --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5->cosine @ 384/576 (lr-decay vs integrated)" --gui --per-object --image-id 777
```

D-062 verification: the guarded suite ran 245 tests (229 passed, 16 documented skips, zero failures/errors). Four native Tk smoke cases passed (two choices, standard/per-object), with Image 777, all 47 selectable IDs, correct labels, navigation and zoom/pan. Model/GPU calls, weight reads and image exports were blocked; saved JSON hashes stayed unchanged. Human visual acceptance remains pending.
