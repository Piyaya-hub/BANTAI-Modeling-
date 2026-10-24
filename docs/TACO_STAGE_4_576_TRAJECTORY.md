# Stage 4: 384/576 mid-Epoch-2 trajectory audit (D-058)

Owner-authorized single attempt, 2026-10-24. Observed status: **completed**.

This is a diagnostic trajectory audit of the same 47 validation images / 75 annotations, not an automatic checkpoint-selection rule or proof of generalization. Inspecting four additional checkpoints increases validation selection bias. No checkpoint is promoted and no test set is accessed. A change first observed at a sampled checkpoint occurred somewhere in the preceding interval; these observations cannot identify its exact unobserved optimizer update.

Fixed experiment: D-057 `taco-896e2f44146d4d54aee518e6087908ca`, seed42/LR1e-5, original GroundingDINO Tiny snapshot and Stage3D adapter, FP32, TF32 disabled, fixed eight-class prompt and COCO v2. Evaluation uses shortest384/longest576 and576-square padding. Only training update step varies. Existing academic-use basis and uncertainty remain unchanged.

Private audit: [scale576-trajectory-c84b9d9344e64578ba1f2fb85f353d0c](../runs/training/scale576-trajectory-c84b9d9344e64578ba1f2fb85f353d0c/record.json), [protected inventory](../runs/training/scale576-trajectory-c84b9d9344e64578ba1f2fb85f353d0c/protected-before.json), [guarded test log](../runs/training/scale576-trajectory-c84b9d9344e64578ba1f2fb85f353d0c/software-tests.txt).

## Gates and safeguards

| Gate | Result |
| --- | --- |
| 1 | passed |
| 2 | passed |
| 3 | passed |

Guarded suite: 234 tests, 0 failures, 0 errors, 16 documented skips. All three new ordering/position/locked-bin transition tests passed. Existing training-execution fixtures were excluded under the zero-backward guard; the skip list is archived. Generated temporary software image fixtures are permitted only during the suite; annotated source-image exports remain blocked.

Post-execution protected audit: `{"checked": 1688, "unchanged": true, "changed": []}`. All prior run files,50 corrected checkpoint bundles, source annotations, train/validation metadata, validation photos, rights, pinned snapshot, scripts/config/tests/docs and dependency locks are included. Training/test photographs are not opened or rehashed, preserving zero training-photo reads; no claim of a fresh hash of those excluded photos is made. Checkpoint `training_state.pt` files are hashed only, with `torch.load` blocked in Gates2/3.

All50 bundles verified, including exact positions for all ten D-057 checkpoints. Current six-file training `source_identity()` exactly matches the D-057 contract. Saved step0/221/442 COCO metrics reproduced exactly on CPU with zero endpoint GPU forwards. Four sequential CUDA models each ran47 forwards (188 total), with full frozen parameters, inference mode, distinct parameter hashes from the preceding checkpoint, unchanged hashes after evaluation, and model release before the next load. A CPU-only E1 model load establishes the first prior weight hash; it performs no forward. No optimizer or training-state deserialization is used.

## Seven-point overall trajectory

Metrics are fractions. Step0/221/442 reuse saved baseline/E1/E2 predictions and losses; only250/300/350/400 are newly evaluated.

| Step | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | macro_AP_excluding_bag | mean_loss_per_image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.147847 | 0.179470 | 0.165594 | 0.000000 | 0.007261 | 0.172653 | 0.299178 | 0.416109 | 0.426109 | 0.095418 | 9.720814 |
| 221 | 0.249891 | 0.286853 | 0.264765 | 0.060891 | 0.033495 | 0.303687 | 0.388094 | 0.465611 | 0.488141 | 0.181629 | 5.323725 |
| 250 | 0.241897 | 0.273259 | 0.261584 | 0.065347 | 0.037910 | 0.293615 | 0.361494 | 0.451349 | 0.475869 | 0.178858 | 5.186863 |
| 300 | 0.258596 | 0.291890 | 0.282016 | 0.058168 | 0.040147 | 0.310166 | 0.366991 | 0.436775 | 0.449235 | 0.218327 | 5.038743 |
| 350 | 0.262337 | 0.288625 | 0.278353 | 0.065773 | 0.033448 | 0.315487 | 0.356863 | 0.413225 | 0.428890 | 0.218249 | 4.880779 |
| 400 | 0.250190 | 0.273108 | 0.261154 | 0.064883 | 0.043208 | 0.298309 | 0.295187 | 0.361165 | 0.381277 | 0.213796 | 4.749100 |
| 442 | 0.170508 | 0.196086 | 0.179972 | 0.075112 | 0.048109 | 0.206941 | 0.281064 | 0.350527 | 0.370118 | 0.190859 | 4.613316 |

## All-eight-class AP and AR100 trajectories

Each cell is AP / AR100. Deltas identify the first observed sampled step, not the exact update within its interval.

| Class | 0 | 221 | 250 | 300 | 350 | 400 | 442 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cigarette | 0.014122 / 0.029167 | 0.051306 / 0.179167 | 0.056462 / 0.254167 | 0.064405 / 0.275000 | 0.076901 / 0.329167 | 0.077577 / 0.341667 | 0.082558 / 0.350000 |
| plastic_bottle | 0.053966 / 0.425000 | 0.189688 / 0.600000 | 0.202220 / 0.625000 | 0.186115 / 0.337500 | 0.236139 / 0.337500 | 0.180281 / 0.337500 | 0.180281 / 0.337500 |
| plastic_bag | 0.514851 / 0.700000 | 0.727723 / 0.950000 | 0.683168 / 0.950000 | 0.540476 / 1.000000 | 0.570957 / 1.000000 | 0.504950 / 0.500000 | 0.028053 / 0.500000 |
| plastic_wrapper | 0.074249 / 0.572727 | 0.041171 / 0.290909 | 0.013201 / 0.072727 | 0.029703 / 0.081818 | 0.029703 / 0.081818 | 0.029703 / 0.081818 | 0.128053 / 0.254545 |
| plastic_film | 0.002621 / 0.107692 | 0.028424 / 0.530769 | 0.027871 / 0.530769 | 0.025929 / 0.553846 | 0.025307 / 0.476923 | 0.025768 / 0.569231 | 0.033685 / 0.584615 |
| drink_can | 0.124851 / 0.214286 | 0.118812 / 0.114286 | 0.118812 / 0.114286 | 0.367162 / 0.385714 | 0.367162 / 0.385714 | 0.390594 / 0.400000 | 0.118812 / 0.114286 |
| plastic_cup | 0.335897 / 0.660000 | 0.404797 / 0.640000 | 0.415617 / 0.660000 | 0.448515 / 0.520000 | 0.386139 / 0.380000 | 0.386139 / 0.380000 | 0.386139 / 0.380000 |
| styrofoam_piece | 0.062220 / 0.700000 | 0.437207 / 0.600000 | 0.417822 / 0.600000 | 0.406459 / 0.440000 | 0.406391 / 0.440000 | 0.406509 / 0.440000 | 0.406483 / 0.440000 |

### Interval changes during Epoch2

Each cell is AP delta / AR100 delta from the prior sampled step.

| Class | 221→250 | 250→300 | 300→350 | 350→400 | 400→442 |
| --- | --- | --- | --- | --- | --- |
| cigarette | +0.005156 / +0.075000 | +0.007943 / +0.020833 | +0.012496 / +0.054167 | +0.000676 / +0.012500 | +0.004981 / +0.008333 |
| plastic_bottle | +0.012532 / +0.025000 | -0.016105 / -0.287500 | +0.050024 / +0.000000 | -0.055858 / +0.000000 | +0.000000 / +0.000000 |
| plastic_bag | -0.044554 / +0.000000 | -0.142692 / +0.050000 | +0.030481 / +0.000000 | -0.066007 / -0.500000 | -0.476898 / +0.000000 |
| plastic_wrapper | -0.027969 / -0.218182 | +0.016502 / +0.009091 | +0.000000 / +0.000000 | +0.000000 / +0.000000 | +0.098350 / +0.172727 |
| plastic_film | -0.000554 / +0.000000 | -0.001941 / +0.023077 | -0.000622 / -0.076923 | +0.000461 / +0.092308 | +0.007917 / +0.015385 |
| drink_can | +0.000000 / +0.000000 | +0.248350 / +0.271429 | +0.000000 / +0.000000 | +0.023432 / +0.014286 | -0.271782 / -0.285714 |
| plastic_cup | +0.010820 / +0.020000 | +0.032898 / -0.140000 | -0.062376 / -0.140000 | +0.000000 / +0.000000 | +0.000000 / +0.000000 |
| styrofoam_piece | -0.019385 / +0.000000 | -0.011362 / -0.160000 | -0.000069 / +0.000000 | +0.000118 / +0.000000 | -0.000026 / +0.000000 |

## Locked bins and cigarette trajectories

All75 GTs remain in original256/384 shortest-side bins14/10/15/36; cigarettes remain13/7/4. Cells are any-class / same-class coverage counts at IoU>=0.50. Inclusive score floors and stable official top100 rank are preserved. All900 is diagnostic, and floor0 is a candidate analysis rather than an operating-threshold recommendation. Coverage is nonexclusive per GT, so one query may cover multiple GTs; this is distinct from official COCO precision/recall.

### all900 @ 0.05

| Step | extremely small <4 any/same | very small 4–<8 any/same | small 8–<16 any/same | larger ≥16 any/same |
| --- | --- | --- | --- | --- |
| 0 | 1/0 | 6/2 | 14/4 | 36/23 |
| 221 | 2/2 | 7/4 | 13/7 | 34/21 |
| 250 | 2/2 | 8/5 | 13/9 | 34/18 |
| 300 | 3/3 | 8/5 | 13/9 | 34/16 |
| 350 | 5/5 | 9/6 | 13/8 | 34/15 |
| 400 | 5/5 | 9/6 | 13/8 | 34/15 |
| 442 | 6/6 | 9/6 | 13/9 | 34/14 |

| Step | All cigarettes N24 | Image777 N7 | Other images N17 | <4px N13 | 4–<8px N7 | 8–<16px N4 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 8/3 | 4/0 | 4/3 | 1/0 | 3/2 | 4/1 |
| 221 | 9/8 | 2/1 | 7/7 | 2/2 | 4/4 | 3/2 |
| 250 | 10/10 | 3/3 | 7/7 | 2/2 | 5/5 | 3/3 |
| 300 | 11/11 | 3/3 | 8/8 | 3/3 | 5/5 | 3/3 |
| 350 | 14/14 | 4/4 | 10/10 | 5/5 | 6/6 | 3/3 |
| 400 | 14/14 | 4/4 | 10/10 | 5/5 | 6/6 | 3/3 |
| 442 | 15/15 | 5/5 | 10/10 | 6/6 | 6/6 | 3/3 |

### all900 @ 0.0

| Step | extremely small <4 any/same | very small 4–<8 any/same | small 8–<16 any/same | larger ≥16 any/same |
| --- | --- | --- | --- | --- |
| 0 | 2/0 | 9/3 | 15/4 | 36/26 |
| 221 | 5/5 | 9/6 | 14/11 | 36/27 |
| 250 | 6/6 | 10/8 | 15/12 | 36/27 |
| 300 | 8/8 | 10/7 | 14/11 | 36/27 |
| 350 | 8/8 | 10/7 | 15/11 | 36/22 |
| 400 | 8/8 | 10/7 | 15/11 | 36/23 |
| 442 | 8/8 | 10/7 | 14/10 | 36/22 |

| Step | All cigarettes N24 | Image777 N7 | Other images N17 | <4px N13 | 4–<8px N7 | 8–<16px N4 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 12/4 | 5/0 | 7/4 | 2/0 | 6/3 | 4/1 |
| 221 | 15/14 | 5/4 | 10/10 | 5/5 | 6/5 | 4/4 |
| 250 | 17/17 | 6/6 | 11/11 | 6/6 | 7/7 | 4/4 |
| 300 | 19/19 | 7/7 | 12/12 | 8/8 | 7/7 | 4/4 |
| 350 | 19/19 | 7/7 | 12/12 | 8/8 | 7/7 | 4/4 |
| 400 | 19/19 | 7/7 | 12/12 | 8/8 | 7/7 | 4/4 |
| 442 | 19/19 | 7/7 | 12/12 | 8/8 | 7/7 | 4/4 |

### top100 @ 0.05

| Step | extremely small <4 any/same | very small 4–<8 any/same | small 8–<16 any/same | larger ≥16 any/same |
| --- | --- | --- | --- | --- |
| 0 | 1/0 | 6/2 | 14/4 | 36/23 |
| 221 | 2/2 | 7/4 | 13/7 | 34/21 |
| 250 | 2/2 | 8/5 | 13/9 | 34/18 |
| 300 | 3/3 | 8/5 | 13/9 | 34/16 |
| 350 | 5/5 | 9/6 | 13/8 | 34/15 |
| 400 | 5/5 | 9/6 | 13/8 | 34/15 |
| 442 | 6/6 | 9/6 | 13/9 | 34/14 |

| Step | All cigarettes N24 | Image777 N7 | Other images N17 | <4px N13 | 4–<8px N7 | 8–<16px N4 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 8/3 | 4/0 | 4/3 | 1/0 | 3/2 | 4/1 |
| 221 | 9/8 | 2/1 | 7/7 | 2/2 | 4/4 | 3/2 |
| 250 | 10/10 | 3/3 | 7/7 | 2/2 | 5/5 | 3/3 |
| 300 | 11/11 | 3/3 | 8/8 | 3/3 | 5/5 | 3/3 |
| 350 | 14/14 | 4/4 | 10/10 | 5/5 | 6/6 | 3/3 |
| 400 | 14/14 | 4/4 | 10/10 | 5/5 | 6/6 | 3/3 |
| 442 | 15/15 | 5/5 | 10/10 | 6/6 | 6/6 | 3/3 |

### top100 @ 0.0

| Step | extremely small <4 any/same | very small 4–<8 any/same | small 8–<16 any/same | larger ≥16 any/same |
| --- | --- | --- | --- | --- |
| 0 | 1/0 | 6/2 | 14/4 | 36/23 |
| 221 | 3/3 | 9/5 | 14/9 | 36/25 |
| 250 | 3/3 | 9/6 | 14/10 | 36/20 |
| 300 | 4/4 | 9/6 | 14/9 | 36/19 |
| 350 | 6/6 | 10/7 | 14/8 | 36/18 |
| 400 | 6/6 | 10/7 | 14/9 | 36/18 |
| 442 | 6/6 | 10/7 | 14/9 | 36/18 |

| Step | All cigarettes N24 | Image777 N7 | Other images N17 | <4px N13 | 4–<8px N7 | 8–<16px N4 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 8/3 | 4/0 | 4/3 | 1/0 | 3/2 | 4/1 |
| 221 | 13/10 | 5/2 | 8/8 | 3/3 | 6/5 | 4/2 |
| 250 | 13/12 | 5/4 | 8/8 | 3/3 | 6/6 | 4/3 |
| 300 | 14/13 | 5/4 | 9/9 | 4/4 | 6/6 | 4/3 |
| 350 | 17/16 | 6/5 | 11/11 | 6/6 | 7/7 | 4/3 |
| 400 | 17/16 | 6/5 | 11/11 | 6/6 | 7/7 | 4/3 |
| 442 | 17/16 | 6/5 | 11/11 | 6/6 | 7/7 | 4/3 |

## Disjoint GT states and successive transitions

Primary condition: all900 at0.05, IoU>=0.50. C=correct class, W=wrong class only, N=no qualifying localization. C takes priority if any same-class query qualifies, regardless of other classes. Each row sums to75.

| Step | correct_class | wrong_class_only | no_qualifying_localization |
| --- | --- | --- | --- |
| 0 | 29 | 28 | 18 |
| 221 | 34 | 22 | 19 |
| 250 | 34 | 23 | 18 |
| 300 | 33 | 25 | 17 |
| 350 | 34 | 27 | 14 |
| 400 | 34 | 27 | 14 |
| 442 | 35 | 27 | 13 |

### 0→221

| Prior / following | correct_class | wrong_class_only | no_qualifying_localization |
| --- | --- | --- | --- |
| correct_class | 21 | 8 | 0 |
| wrong_class_only | 10 | 14 | 4 |
| no_qualifying_localization | 3 | 0 | 15 |

| Locked bin | Gains of correct class | Losses of correct class |
| --- | --- | --- |
| extremely small <4 | 2 | 0 |
| very small 4–<8 | 2 | 0 |
| small 8–<16 | 4 | 1 |
| larger ≥16 | 5 | 7 |

### 221→250

| Prior / following | correct_class | wrong_class_only | no_qualifying_localization |
| --- | --- | --- | --- |
| correct_class | 31 | 3 | 0 |
| wrong_class_only | 2 | 20 | 0 |
| no_qualifying_localization | 1 | 0 | 18 |

| Locked bin | Gains of correct class | Losses of correct class |
| --- | --- | --- |
| extremely small <4 | 0 | 0 |
| very small 4–<8 | 1 | 0 |
| small 8–<16 | 2 | 0 |
| larger ≥16 | 0 | 3 |

### 250→300

| Prior / following | correct_class | wrong_class_only | no_qualifying_localization |
| --- | --- | --- | --- |
| correct_class | 29 | 5 | 0 |
| wrong_class_only | 3 | 20 | 0 |
| no_qualifying_localization | 1 | 0 | 17 |

| Locked bin | Gains of correct class | Losses of correct class |
| --- | --- | --- |
| extremely small <4 | 1 | 0 |
| very small 4–<8 | 0 | 0 |
| small 8–<16 | 1 | 1 |
| larger ≥16 | 2 | 4 |

### 300→350

| Prior / following | correct_class | wrong_class_only | no_qualifying_localization |
| --- | --- | --- | --- |
| correct_class | 31 | 2 | 0 |
| wrong_class_only | 0 | 25 | 0 |
| no_qualifying_localization | 3 | 0 | 14 |

| Locked bin | Gains of correct class | Losses of correct class |
| --- | --- | --- |
| extremely small <4 | 2 | 0 |
| very small 4–<8 | 1 | 0 |
| small 8–<16 | 0 | 1 |
| larger ≥16 | 0 | 1 |

### 350→400

| Prior / following | correct_class | wrong_class_only | no_qualifying_localization |
| --- | --- | --- | --- |
| correct_class | 34 | 0 | 0 |
| wrong_class_only | 0 | 27 | 0 |
| no_qualifying_localization | 0 | 0 | 14 |

| Locked bin | Gains of correct class | Losses of correct class |
| --- | --- | --- |
| extremely small <4 | 0 | 0 |
| very small 4–<8 | 0 | 0 |
| small 8–<16 | 0 | 0 |
| larger ≥16 | 0 | 0 |

### 400→442

| Prior / following | correct_class | wrong_class_only | no_qualifying_localization |
| --- | --- | --- | --- |
| correct_class | 31 | 3 | 0 |
| wrong_class_only | 3 | 24 | 0 |
| no_qualifying_localization | 1 | 0 | 13 |

| Locked bin | Gains of correct class | Losses of correct class |
| --- | --- | --- |
| extremely small <4 | 1 | 0 |
| very small 4–<8 | 0 | 0 |
| small 8–<16 | 1 | 0 |
| larger ≥16 | 2 | 3 |

## Endpoint-selected flip ledger

The11 gains and10 losses are defined by E1 vs E2 endpoints; intermediate reversals remain visible. W(label) gives the argmax class of the qualifying query with greatest IoU, then score, then lower query ID. It is a representative wrong query, not a unique object assignment. Full query IDs, scores, IoUs, raw phrases and alternative qualifying wrong classes are in the CSV/JSON artifacts.

### E1→E2 gains

| GT | Image | GT class | Locked bin | 221 | 250 | 300 | 350 | 400 | 442 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 155 | plastic_film | small 8–<16 | W(plastic_cup) | W(plastic_cup) | C | W(plastic_cup) | W(cigarette) | C |
| 67 | 157 | plastic_film | small 8–<16 | W(cigarette) | C | C | C | C | C |
| 162 | 419 | plastic_wrapper | larger ≥16 | W(cigarette) | W(plastic_film) | W(plastic_film) | W(plastic_film) | W(cigarette) | C |
| 253 | 768 | plastic_wrapper | larger ≥16 | W(cigarette) | W(plastic_bottle) | W(plastic_film) | W(cigarette) | W(cigarette) | C |
| 278 | 777 | cigarette | very small 4–<8 | N | N | N | C | C | C |
| 279 | 777 | cigarette | very small 4–<8 | N | C | C | C | C | C |
| 282 | 777 | cigarette | extremely small <4 | N | N | N | N | N | C |
| 284 | 777 | cigarette | small 8–<16 | W(styrofoam_piece) | C | C | C | C | C |
| 314 | 849 | cigarette | extremely small <4 | N | N | N | C | C | C |
| 315 | 849 | cigarette | extremely small <4 | N | N | N | C | C | C |
| 498 | 1161 | cigarette | extremely small <4 | N | N | C | C | C | C |

### E1→E2 losses

| GT | Image | GT class | Locked bin | 221 | 250 | 300 | 350 | 400 | 442 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 113 | plastic_bottle | small 8–<16 | C | C | W(plastic_cup) | W(plastic_cup) | W(plastic_film) | W(plastic_film) |
| 104 | 252 | plastic_wrapper | larger ≥16 | C | W(cigarette) | W(plastic_film) | W(plastic_film) | W(plastic_film) | W(plastic_film) |
| 177 | 464 | styrofoam_piece | larger ≥16 | C | C | W(plastic_wrapper) | W(plastic_wrapper) | W(plastic_wrapper) | W(plastic_wrapper) |
| 250 | 762 | plastic_wrapper | larger ≥16 | C | W(plastic_film) | W(cigarette) | W(plastic_film) | W(plastic_film) | W(plastic_film) |
| 258 | 776 | plastic_bag | larger ≥16 | C | C | W(plastic_film) | W(plastic_film) | W(plastic_film) | W(plastic_film) |
| 303 | 833 | plastic_bag | larger ≥16 | C | C | C | C | C | W(plastic_wrapper) |
| 404 | 1076 | plastic_cup | larger ≥16 | C | C | W(plastic_wrapper) | W(plastic_film) | W(plastic_film) | W(plastic_film) |
| 414 | 1096 | plastic_bottle | larger ≥16 | C | C | W(plastic_film) | W(plastic_film) | W(plastic_film) | W(plastic_film) |
| 423 | 1126 | plastic_cup | larger ≥16 | C | C | C | W(plastic_film) | W(plastic_film) | W(plastic_film) |
| 443 | 1156 | plastic_wrapper | larger ≥16 | C | W(cigarette) | W(plastic_film) | W(plastic_film) | W(plastic_film) | W(plastic_film) |

The ten endpoint losses occupy these verified bins: `{"small 8–<16": 1, "larger ≥16": 9}`. The request's description of all ten as large-object regressions is assessed against these locked memberships, not assumed.

## Training batch loss windows

The saved evaluation format includes total validation loss but no component-level validation losses. The table therefore reports saved training window means. Requested221–250 is interpreted as inclusive `global_step` indices (30 updates, including the final E1 update); the additional E2-only222–250 window contains the29 actual E2 updates before step250. Other intervals partition the remaining E2 updates. No training photos are read. CE/L1/GIoU are raw native loss terms before their fixed weighting; cardinality diagnostics remain in JSON.

| Global-step window | N | Mean total loss | loss_ce | loss_bbox | loss_giou | loss_ce_enc | loss_bbox_enc | loss_giou_enc | Mean gradient norm before clip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 221–250 | 30 | 5.671897 | 1.088900 | 0.026317 | 0.196912 | 1.175144 | 0.034309 | 0.223428 | 108.977682 |
| 251–300 | 50 | 5.990368 | 1.132857 | 0.019978 | 0.155503 | 1.361226 | 0.036648 | 0.204033 | 122.527592 |
| 301–350 | 50 | 5.883881 | 1.096326 | 0.020494 | 0.160084 | 1.321533 | 0.036187 | 0.222295 | 156.340874 |
| 351–400 | 50 | 5.452339 | 1.019899 | 0.018438 | 0.180192 | 1.148353 | 0.033235 | 0.248543 | 91.891898 |
| 401–442 | 42 | 5.914887 | 1.130179 | 0.019439 | 0.146529 | 1.339183 | 0.034473 | 0.206774 | 114.896708 |
| 222–250 | 29 | 5.725021 | 1.098678 | 0.025907 | 0.197429 | 1.188792 | 0.034627 | 0.226277 | 110.955609 |

## Interpretation and limits

At steps [221, 250, 300, 350, 400, 442], cigarette same-class counts are [8, 10, 11, 14, 14, 15]; large-bin same-class counts are [21, 18, 16, 15, 15, 14]. Sampled cigarette coverage is nondecreasing. Endpoint-selected first observed gains are distributed as {250: 3, 300: 2, 350: 3, 442: 3}; first observed losses as {250: 3, 300: 5, 350: 1, 442: 1}. These first observations can include subsequent recovery and relapse; the full ledger is authoritative.

The prespecified simultaneous-change criterion can only be assessed at checkpoint-interval resolution. Even if several changes first appear at the same saved step, their underlying optimizer updates could differ. Conversely, changes observed in different intervals establish temporal separation at this sampling resolution, not an actionable checkpoint-selection rule or causal proof that update count alone controls the tradeoff. Rare-class AP, query ranking, IoU and score-threshold crossings can all affect the summaries. No independent human material/annotation confirmation or test-set evidence is added.

The sampled evidence supports a gradual tradeoff rather than one simultaneous endpoint event. Of the ten endpoint losses, three first appear at250 (GT104/250/443), five at300 (GT27/177/258/404/414), one at350 (GT423), and one at442 (GT303). Nine belong to the >=16px bin; GT27 belongs to8–<16px. The seven endpoint cigarette gains first appear at250 (GT279/284),300 (GT498),350 (GT278/314/315), and442 (GT282). Thus gains and losses overlap in several intervals but are not concentrated in one sampled interval. This supports temporal separation at the available sampling resolution; the literal exact-update falsification criterion remains unresolved because intermediate optimizer updates were not evaluated.

Cigarette AP and AR100 rise at every sampled E2 checkpoint, and same-class coverage is nondecreasing. Film AP initially falls through350, then rises at400 and442; its AR100 improves at300, falls at350, then improves at400/442. Wrapper AP/AR100 deteriorate sharply at250; AP partially recovers at300, stays flat at350/400 and jumps at442. Wrapper AR100 remains below E1 even at442. Cup AP rises at250/300 before dropping at350, whereas its AR100 already drops at300. Styrofoam AP declines at250/300/350, briefly edges up at400 and slightly declines at442; its AR100 drops at300 and remains flat afterward. Bag AP falls at250/300, partly recovers at350, then falls at400 and collapses at442. These trajectories do not support a uniform class improvement.

Aggregate AP reaches its largest sampled value at350 (0.262337), but AR100 has already fallen to0.428890 from0.488141 at221 and large-bin same-class coverage to15/36 from21/36. Seven-class AP excluding bag reaches its largest sampled value at300 (0.218327), slightly above350 (0.218249). Observing these maxima does not select a checkpoint. The late400→442 AP drop (0.250190→0.170508) coincides with bag AP0.504950→0.028053 and can AP0.390594→0.118812, while wrapper and cigarette AP improve. Rare classes and rankings strongly affect the aggregate.

Bag GT258 becomes wrong-only at0.05 by300, represented by a plastic_film query, yet a lower-score bag query remains: at300 the official top100 includes a bag query at score0.019330 and IoU0.968041 (rank99). Hence bag AR100 can remain1.0 while the fixed0.05 diagnostic records a class loss. GT303 remains correct through400 and becomes wrong-only as plastic_wrapper at442, including at floor0. This late failure differs from losing a usable same-class candidate only at the chosen score floor. The ledger tracks annotation-level candidate outcomes rather than persistent query identities, so it does not prove a single query changed material class.

Endpoint-selected gains can also reverse temporarily: film GT64 is correct at300, wrong-only at350/400, and correct again at442. Other transient gains (GT16/150) are absent from the11 endpoint gains and are visible in the full75-GT matrices. Eight of the ten endpoint losses are represented by plastic_film at442, and two by plastic_wrapper; this is annotation-relative diagnostic evidence, not independent confirmation of material identity. Falling validation loss throughout E2 (5.323725→4.613316) does not track the late AP collapse.

The four GPU evaluations took approximately28.78/20.65/20.74/20.67s, with1,208,047,616 allocated and1,354,760,192 reserved peak bytes each. These observed evaluation timings include first-pass effects and are not a benchmark. No additional endpoint inference, resolution experiment, threshold change or training followed.

Artifacts: [summary JSON](../runs/training/scale576-trajectory-c84b9d9344e64578ba1f2fb85f353d0c/trajectory-summary.json), [metrics CSV](../runs/training/scale576-trajectory-c84b9d9344e64578ba1f2fb85f353d0c/trajectory-metrics.csv), [per-class CSV](../runs/training/scale576-trajectory-c84b9d9344e64578ba1f2fb85f353d0c/trajectory-per-class.csv), [per-object CSV](../runs/training/scale576-trajectory-c84b9d9344e64578ba1f2fb85f353d0c/trajectory-per-object.csv), [per-object JSON](../runs/training/scale576-trajectory-c84b9d9344e64578ba1f2fb85f353d0c/trajectory-per-object.json), [flip ledger CSV](../runs/training/scale576-trajectory-c84b9d9344e64578ba1f2fb85f353d0c/trajectory-flip-ledger.csv). Per-object tables contain2100 rows (7×75×4), preserving all requested source/threshold conditions.

## Stop state

Zero optimizer updates, backward passes, training-photo reads and test-split access. No core/GUI/config/dependency changes, annotated exports, commits, pushes, merges or checkpoint promotions. Stop for owner review; no further evaluation, training, retry or checkpoint-selection procedure is authorized. Final link/whitespace checks and protected-file rehash evidence are recorded in `record.json`.
