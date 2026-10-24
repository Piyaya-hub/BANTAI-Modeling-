# Stage 4: controlled 384/576 training (D-057)

Owner-authorized single attempt, 2026-10-23. Observed status: **completed**.

Only the trainer CLI upper bound changed from 512 to 576 in its conditional and error message. Defaults remain 256/384. Regression fixtures and this private experiment harness add no model, loss, architecture, preprocessing, GUI or evaluation changes.

Authorization: original pinned SNAPSHOT, seed42, LR1e-5, batch1, FP32, TF32 disabled, AdamW weight decay1e-4/foreachFalse, clip1, frozen vision/text backbones, native Stage3D adapter, corrected COCO v2, 221 train/47 validation. Input shortest384/longest576 with576-square padding. Exactly three discarded pilot updates and442 full updates permitted. No automatic retry/fallback. Rights remain the owner-recorded local academic Section185 basis, not definitive rights clearance.

Private audit: [scale576-experiment-0b2a846e659a429cbe8a65c1ca0444e1](../runs/training/scale576-experiment-0b2a846e659a429cbe8a65c1ca0444e1/record.json); [source diff](../runs/training/scale576-experiment-0b2a846e659a429cbe8a65c1ca0444e1/source-change.patch); [protected inventory](../runs/training/scale576-experiment-0b2a846e659a429cbe8a65c1ca0444e1/protected-before.json); [test log](../runs/training/scale576-experiment-0b2a846e659a429cbe8a65c1ca0444e1/software-tests.txt).

## Gates and preservation

| Gate | Observed result |
| --- | --- |
| 1 | passed |
| 2 | passed |
| 3 | passed |
| 4 | passed |
| 5 | passed |

Optimizer updates started/completed: 445/445. Software suite:231 tests,228 passed,zero failures/errors; two toy optimizer-update fixtures excluded and one Windows privilege skip. AdamW/SGD steps blocked in the suite; all eight independent coordinate tests and both new scale tests passed.

Protected post-run audit: `{"unchanged": true, "changed": [], "files": 1766}`.

Trainer run: [taco-896e2f44146d4d54aee518e6087908ca](../runs/training/taco-896e2f44146d4d54aee518e6087908ca/report.json). Exact fresh high-scale pretrained metrics/loss/predictions and42,300 raw queries matched D-056 before updates, after all268 authorized-photo coordinate checks. Pilot weights and optimizer were released; both loads matched the original pretrained parameter hash. Seed42 was reset before full Epoch1 batch0. Ten checkpoint bundles verified; final optimizer step442 across558 parameters; both epoch orders matched seed42. Frozen hashes and snapshot unchanged.

## Overall saved COCO v2 and validation loss

All seven explicitly enumerated states are included (the request calls these six). AP values are fractions, not percentages. Validation loss uses each state's recorded input scale.

| State | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | macro_AP_excluding_bag | mean_loss_per_image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrained_low | 0.111576 | 0.158253 | 0.135975 | 0.000000 | 0.005941 | 0.129427 | 0.294386 | 0.363077 | 0.363077 | 0.093769 | 10.553226 |
| pretrained_high | 0.147847 | 0.179470 | 0.165594 | 0.000000 | 0.007261 | 0.172653 | 0.299178 | 0.416109 | 0.426109 | 0.095418 | 9.720814 |
| D048_E1_low | 0.203932 | 0.254720 | 0.239987 | 0.019802 | 0.008101 | 0.240848 | 0.354968 | 0.388393 | 0.396766 | 0.136242 | 7.019476 |
| D048_E2_low | 0.084500 | 0.107561 | 0.095599 | 0.036337 | 0.011626 | 0.110052 | 0.160668 | 0.197728 | 0.211229 | 0.096572 | 5.926001 |
| D055_E1_high | 0.247003 | 0.295777 | 0.279521 | 0.029208 | 0.013996 | 0.295150 | 0.414916 | 0.462543 | 0.474763 | 0.152161 | 6.795396 |
| D057_E1_high | 0.249891 | 0.286853 | 0.264765 | 0.060891 | 0.033495 | 0.303687 | 0.388094 | 0.465611 | 0.488141 | 0.181629 | 5.323725 |
| D057_E2_high | 0.170508 | 0.196086 | 0.179972 | 0.075112 | 0.048109 | 0.206941 | 0.281064 | 0.350527 | 0.370118 | 0.190859 | 4.613316 |

## Training loss, gradients, memory and time

The six weighted loss terms are decoder/encoder CE, L1 (`loss_bbox`) and GIoU. Cardinality errors are additional native diagnostics, not weighted loss components.

### D048

Wall time including preflight/evaluation/checkpoints: 444.62s. Pilot allocated/reserved peaks: {'peak_allocated_bytes': 2559891968, 'peak_reserved_bytes': 2722103296}; full update allocated/reserved peaks: {'peak_allocated_bytes': 2558043648, 'peak_reserved_bytes': 2726297600}. Per-step reset peaks for D-057 are in `step-memory.json`; reference D-048 memory is its recorded cumulative peak.

Pilot update time: 1.70s; full update time: 231.74s. These include gradient audits but exclude photo loading, evaluation and checkpoint writes.

| Epoch | Quantity | Mean | Min | Max |
| --- | --- | --- | --- | --- |
| 1 | total loss | 8.569587 | 3.699923 | 31.023689 |
| 1 | loss_ce | 1.596428 | 0.544500 | 7.998923 |
| 1 | loss_bbox | 0.035419 | 0.004392 | 0.182184 |
| 1 | loss_giou | 0.236439 | 0.022307 | 1.033811 |
| 1 | cardinality_error | 898.203620 | 870.000000 | 899.000000 |
| 1 | loss_ce_enc | 1.764385 | 0.499364 | 8.011110 |
| 1 | loss_bbox_enc | 0.081487 | 0.008859 | 0.343700 |
| 1 | loss_giou_enc | 0.395276 | 0.075953 | 1.116167 |
| 1 | cardinality_error_enc | 898.203620 | 870.000000 | 899.000000 |
| 1 | gradient norm before clip | 147.016106 | 37.510464 | 1463.960449 |
| 1 | model.enc_output.weight | 7.164117 | 2.198709 | 57.568714 |
| 1 | model.encoder_output_bbox_embed.layers.2.weight | 27.447915 | 2.398377 | 153.864334 |
| 2 | total loss | 6.903368 | 3.207014 | 20.759693 |
| 2 | loss_ce | 1.220535 | 0.426467 | 4.348992 |
| 2 | loss_bbox | 0.029247 | 0.002133 | 0.163311 |
| 2 | loss_giou | 0.218400 | 0.008819 | 0.973127 |
| 2 | cardinality_error | 898.203620 | 870.000000 | 899.000000 |
| 2 | loss_ce_enc | 1.452690 | 0.557864 | 5.487482 |
| 2 | loss_bbox_enc | 0.064501 | 0.006094 | 0.293884 |
| 2 | loss_giou_enc | 0.325692 | 0.044010 | 1.034226 |
| 2 | cardinality_error_enc | 898.203620 | 870.000000 | 899.000000 |
| 2 | gradient norm before clip | 130.388697 | 33.288670 | 1619.742065 |
| 2 | model.enc_output.weight | 5.862709 | 1.964009 | 56.508945 |
| 2 | model.encoder_output_bbox_embed.layers.2.weight | 26.194640 | 3.792454 | 129.454956 |

### D057

Wall time including preflight/evaluation/checkpoints: 570.63s. Pilot allocated/reserved peaks: {'peak_allocated_bytes': 3812504064, 'peak_reserved_bytes': 4127195136}; full update allocated/reserved peaks: {'peak_allocated_bytes': 3815761920, 'peak_reserved_bytes': 4223664128}. Per-step reset peaks for D-057 are in `step-memory.json`; reference D-048 memory is its recorded cumulative peak.

Pilot update time: 2.32s; full update time: 326.72s. These include gradient audits but exclude photo loading, evaluation and checkpoint writes.

| Epoch | Quantity | Mean | Min | Max |
| --- | --- | --- | --- | --- |
| 1 | total loss | 7.221199 | 2.813112 | 27.022503 |
| 1 | loss_ce | 1.466496 | 0.391168 | 5.695916 |
| 1 | loss_bbox | 0.024775 | 0.004298 | 0.162392 |
| 1 | loss_giou | 0.186784 | 0.015933 | 0.907675 |
| 1 | cardinality_error | 898.203620 | 870.000000 | 899.000000 |
| 1 | loss_ce_enc | 1.544411 | 0.470225 | 7.584177 |
| 1 | loss_bbox_enc | 0.041213 | 0.006967 | 0.231202 |
| 1 | loss_giou_enc | 0.247938 | 0.042487 | 0.957622 |
| 1 | cardinality_error_enc | 898.203620 | 870.000000 | 899.000000 |
| 1 | gradient norm before clip | 138.493792 | 29.428591 | 1208.323364 |
| 1 | model.enc_output.weight | 6.600240 | 1.952748 | 75.997795 |
| 1 | model.encoder_output_bbox_embed.layers.2.weight | 25.783131 | 2.631623 | 195.506989 |
| 2 | total loss | 5.795386 | 2.257177 | 18.608345 |
| 2 | loss_ce | 1.094042 | 0.386736 | 4.368857 |
| 2 | loss_bbox | 0.020422 | 0.000712 | 0.164244 |
| 2 | loss_giou | 0.165921 | 0.001559 | 0.936534 |
| 2 | cardinality_error | 898.203620 | 870.000000 | 899.000000 |
| 2 | loss_ce_enc | 1.277268 | 0.375797 | 4.786035 |
| 2 | loss_bbox_enc | 0.035093 | 0.004343 | 0.176409 |
| 2 | loss_giou_enc | 0.221675 | 0.033287 | 0.952707 |
| 2 | cardinality_error_enc | 898.203620 | 870.000000 | 899.000000 |
| 2 | gradient norm before clip | 120.277795 | 28.358583 | 1741.689941 |
| 2 | model.enc_output.weight | 4.934398 | 1.835723 | 22.232281 |
| 2 | model.encoder_output_bbox_embed.layers.2.weight | 25.657309 | 2.724703 | 132.274277 |

## Locked size bins and cigarettes

75 independent GT objects remain in original256/384 bins14/10/15/36. Coverage uses any overlapping valid argmax query at IoU>=0.5, with inclusive score floors; same-class and any-class are separate. This is nonexclusive GT coverage, not COCO precision/recall. `top100` uses the original stable score rank, and `all900` is diagnostic. Floor0 shows low-score candidates and is not a proposed operating threshold.

### all900 @ 0.05

| State | extremely small <4 any/same | very small 4–<8 any/same | small 8–<16 any/same | larger ≥16 any/same |
| --- | --- | --- | --- | --- |
| D048_E1_low | 0/0 | 4/1 | 10/6 | 35/21 |
| D048_E2_low | 1/1 | 5/2 | 11/7 | 35/16 |
| D055_E1_high | 0/0 | 8/4 | 13/6 | 35/23 |
| D057_E1_high | 2/2 | 7/4 | 13/7 | 34/21 |
| D057_E2_high | 6/6 | 9/6 | 13/9 | 34/14 |

| State | Cigarette N24 any/same | Image777 N7 any/same | Other N17 any/same | Cigarette bins N13/7/4 any/same |
| --- | --- | --- | --- | --- |
| D048_E1_low | 2/2 | 0/0 | 2/2 | 0/0, 1/1, 1/1 |
| D048_E2_low | 5/5 | 1/1 | 4/4 | 1/1, 2/2, 2/2 |
| D055_E1_high | 8/5 | 3/0 | 5/5 | 0/0, 5/4, 3/1 |
| D057_E1_high | 9/8 | 2/1 | 7/7 | 2/2, 4/4, 3/2 |
| D057_E2_high | 15/15 | 5/5 | 10/10 | 6/6, 6/6, 3/3 |

### all900 @ 0.0

| State | extremely small <4 any/same | very small 4–<8 any/same | small 8–<16 any/same | larger ≥16 any/same |
| --- | --- | --- | --- | --- |
| D048_E1_low | 1/1 | 4/1 | 14/8 | 36/26 |
| D048_E2_low | 2/2 | 5/2 | 14/9 | 36/17 |
| D055_E1_high | 1/1 | 9/5 | 15/9 | 36/28 |
| D057_E1_high | 5/5 | 9/6 | 14/11 | 36/27 |
| D057_E2_high | 8/8 | 10/7 | 14/10 | 36/22 |

| State | Cigarette N24 any/same | Image777 N7 any/same | Other N17 any/same | Cigarette bins N13/7/4 any/same |
| --- | --- | --- | --- | --- |
| D048_E1_low | 5/5 | 2/2 | 3/3 | 1/1, 1/1, 3/3 |
| D048_E2_low | 8/8 | 3/3 | 5/5 | 2/2, 2/2, 4/4 |
| D055_E1_high | 11/7 | 5/1 | 6/6 | 1/1, 6/5, 4/1 |
| D057_E1_high | 15/14 | 5/4 | 10/10 | 5/5, 6/5, 4/4 |
| D057_E2_high | 19/19 | 7/7 | 12/12 | 8/8, 7/7, 4/4 |

### top100 @ 0.05

| State | extremely small <4 any/same | very small 4–<8 any/same | small 8–<16 any/same | larger ≥16 any/same |
| --- | --- | --- | --- | --- |
| D048_E1_low | 0/0 | 4/1 | 10/6 | 35/21 |
| D048_E2_low | 1/1 | 5/2 | 11/7 | 35/16 |
| D055_E1_high | 0/0 | 8/4 | 13/6 | 35/23 |
| D057_E1_high | 2/2 | 7/4 | 13/7 | 34/21 |
| D057_E2_high | 6/6 | 9/6 | 13/9 | 34/14 |

| State | Cigarette N24 any/same | Image777 N7 any/same | Other N17 any/same | Cigarette bins N13/7/4 any/same |
| --- | --- | --- | --- | --- |
| D048_E1_low | 2/2 | 0/0 | 2/2 | 0/0, 1/1, 1/1 |
| D048_E2_low | 5/5 | 1/1 | 4/4 | 1/1, 2/2, 2/2 |
| D055_E1_high | 8/5 | 3/0 | 5/5 | 0/0, 5/4, 3/1 |
| D057_E1_high | 9/8 | 2/1 | 7/7 | 2/2, 4/4, 3/2 |
| D057_E2_high | 15/15 | 5/5 | 10/10 | 6/6, 6/6, 3/3 |

### top100 @ 0.0

| State | extremely small <4 any/same | very small 4–<8 any/same | small 8–<16 any/same | larger ≥16 any/same |
| --- | --- | --- | --- | --- |
| D048_E1_low | 0/0 | 4/1 | 13/8 | 36/22 |
| D048_E2_low | 2/2 | 5/2 | 13/8 | 36/17 |
| D055_E1_high | 0/0 | 9/5 | 14/6 | 36/27 |
| D057_E1_high | 3/3 | 9/5 | 14/9 | 36/25 |
| D057_E2_high | 6/6 | 10/7 | 14/9 | 36/18 |

| State | Cigarette N24 any/same | Image777 N7 any/same | Other N17 any/same | Cigarette bins N13/7/4 any/same |
| --- | --- | --- | --- | --- |
| D048_E1_low | 4/4 | 2/2 | 2/2 | 0/0, 1/1, 3/3 |
| D048_E2_low | 7/7 | 2/2 | 5/5 | 2/2, 2/2, 3/3 |
| D055_E1_high | 10/6 | 5/1 | 5/5 | 0/0, 6/5, 4/1 |
| D057_E1_high | 13/10 | 5/2 | 8/8 | 3/3, 6/5, 4/2 |
| D057_E2_high | 17/16 | 6/5 | 11/11 | 6/6, 7/7, 4/3 |

## Per-class AP/AR100 and disjoint GT states

States below use all900 at0.05 and sum to75 independently for each checkpoint.

### D055_E1_high

Overall states: `{"wrong_class_only": 23, "correct_class": 33, "no_qualifying_localization": 19}`.

| Class | AP | AR100 | correct | wrong only | no qualifying localization |
| --- | --- | --- | --- | --- | --- |
| cigarette | 0.031416 | 0.075000 | 5 | 3 | 16 |
| plastic_bottle | 0.109898 | 0.575000 | 5 | 3 | 0 |
| plastic_bag | 0.910891 | 0.950000 | 2 | 0 | 0 |
| plastic_wrapper | 0.071429 | 0.409091 | 5 | 6 | 0 |
| plastic_film | 0.017526 | 0.446154 | 7 | 3 | 3 |
| drink_can | 0.177970 | 0.242857 | 2 | 5 | 0 |
| plastic_cup | 0.358376 | 0.620000 | 4 | 1 | 0 |
| styrofoam_piece | 0.298515 | 0.480000 | 3 | 2 | 0 |

### D057_E1_high

Overall states: `{"wrong_class_only": 22, "correct_class": 34, "no_qualifying_localization": 19}`.

| Class | AP | AR100 | correct | wrong only | no qualifying localization |
| --- | --- | --- | --- | --- | --- |
| cigarette | 0.051306 | 0.179167 | 8 | 1 | 15 |
| plastic_bottle | 0.189688 | 0.600000 | 5 | 3 | 0 |
| plastic_bag | 0.727723 | 0.950000 | 2 | 0 | 0 |
| plastic_wrapper | 0.041171 | 0.290909 | 4 | 7 | 0 |
| plastic_film | 0.028424 | 0.530769 | 7 | 2 | 4 |
| drink_can | 0.118812 | 0.114286 | 1 | 6 | 0 |
| plastic_cup | 0.404797 | 0.640000 | 4 | 1 | 0 |
| styrofoam_piece | 0.437207 | 0.600000 | 3 | 2 | 0 |

### D057_E2_high

Overall states: `{"wrong_class_only": 27, "no_qualifying_localization": 13, "correct_class": 35}`.

| Class | AP | AR100 | correct | wrong only | no qualifying localization |
| --- | --- | --- | --- | --- | --- |
| cigarette | 0.082558 | 0.350000 | 15 | 0 | 9 |
| plastic_bottle | 0.180281 | 0.337500 | 3 | 5 | 0 |
| plastic_bag | 0.028053 | 0.500000 | 0 | 2 | 0 |
| plastic_wrapper | 0.128053 | 0.254545 | 3 | 8 | 0 |
| plastic_film | 0.033685 | 0.584615 | 9 | 0 | 4 |
| drink_can | 0.118812 | 0.114286 | 1 | 6 | 0 |
| plastic_cup | 0.386139 | 0.380000 | 2 | 3 | 0 |
| styrofoam_piece | 0.406483 | 0.440000 | 2 | 3 | 0 |

## Paired transitions and interpretation

### D055_E1_high -> D057_E1_high

Rows are prior states; columns are following states.

| Prior / following | correct_class | wrong_class_only | no_qualifying_localization |
| --- | --- | --- | --- |
| correct_class | 29 | 4 | 0 |
| wrong_class_only | 3 | 18 | 2 |
| no_qualifying_localization | 2 | 0 | 17 |

Previously sub0.05 cigarette candidates gaining same-class coverage at0.05 (GT IDs): `[313]`. Cigarette same-class losses: `[]`. All four conditions' matrices/GT IDs are in comparison.json.

### D057_E1_high -> D057_E2_high

Rows are prior states; columns are following states.

| Prior / following | correct_class | wrong_class_only | no_qualifying_localization |
| --- | --- | --- | --- |
| correct_class | 24 | 10 | 0 |
| wrong_class_only | 5 | 17 | 0 |
| no_qualifying_localization | 6 | 0 | 13 |

Previously sub0.05 cigarette candidates gaining same-class coverage at0.05 (GT IDs): `[279, 284, 314, 498]`. Cigarette same-class losses: `[]`. All four conditions' matrices/GT IDs are in comparison.json.

Best new epoch by AP: D057_E1_high, AP0.249891 vs inference-only scaling0.247003; AR1000.488141 vs0.474763. Epoch2 AP delta vs new Epoch1: -0.079383. This is one seed and the same47-image validation set; no test or independent confirmation is claimed. The stated hypothesis requires comparison with the inference-only scale result, not merely with low-scale pretrained output. No checkpoint is promoted.

The hypothesis receives partial support. FP32 training fits the GPU. Relative to D-055, Epoch1 improves AP by only0.002888 and correct-class coverage33→34/75, with five gains and four class regressions. The original <4px group gains two correct-class objects;4–<8px same-class coverage stays4/10, while any-class coverage falls8→7. This is a modest aggregate gain with mixed paired outcomes, not uniform superiority.

Epoch2 improves cigarette same-class coverage8→15/24 (D-055:5/24), including image777 from1→5/7 (D-055:0/7), and other images7→10/17 (D-055:5/17). Floor0 candidate coverage is14→19/24, so four correct-class candidates still remain below0.05 at Epoch2. Four previously sub0.05 candidates gain qualifying coverage between new epochs: GT279/284/314/498. Original <4px cigarette coverage rises to6/13;4–<8px to6/7 and8–<16px to3/4. These are overlapping-candidate coverage counts, not independent detections or calibrated correctness.

Epoch2 is not an overall repair: total correct-class coverage34→35 conceals11 gains and10 losses. Large-bin same-class coverage falls21→14/36 while tiny/small-bin coverage improves. Seven-class AP excluding bag rises0.181629→0.190859 even as official eight-class AP falls0.249891→0.170508. Bag AP falls0.727723→0.028053 on only two GTs; its contribution to the aggregate AP delta is approximately−0.087459, outweighing positive net contributions from other classes. Rare-class sensitivity helps explain the AP reversal, but does not erase the paired regressions or falling AR100. Wrapper AP is worse than D-055 at Epoch1 (0.041171 vs0.071429), then improves to0.128053 at Epoch2 while its AR100 stays below D-055 (0.254545 vs0.409091). Scale-matched training does not consistently resolve wrapper confusion.

Mean training loss is lower than the low-scale reference in both epochs (7.221199/5.795386 vs8.569587/6.903368), with lower mean encoder/decoder CE, L1 and GIoU. Loss reduction does not establish better overall validation quality. Full-update peak allocated/reserved memory rises from about2.38/2.54 GiB to3.55/3.93 GiB; trainer wall time rises444.62→570.63s on this machine. Timings include audit/checkpoint overhead and are descriptive, not a controlled speed benchmark.

Artifacts: [comparison JSON](../runs/training/scale576-experiment-0b2a846e659a429cbe8a65c1ca0444e1/comparison.json), [metrics CSV](../runs/training/scale576-experiment-0b2a846e659a429cbe8a65c1ca0444e1/metrics-comparison.csv), [per-object CSV](../runs/training/scale576-experiment-0b2a846e659a429cbe8a65c1ca0444e1/per-object-comparison.csv), [per-object JSON](../runs/training/scale576-experiment-0b2a846e659a429cbe8a65c1ca0444e1/per-object-comparison.json). Per-object CSV includes all five trained states,75 GTs and all four conditions (1,500 rows), query IDs, scores, IoUs, classes, coverage and locked bins.

## Stop state

No test-split access, dependency changes, annotated exports, checkpoint promotion, commits, pushes or merges. Stop for owner review; no further run or retry authorized. Final documentation/link/diff verification is recorded separately in the experiment audit. Earlier stop notices in historical documents describe their respective completed tasks. D-057 supersedes their no-follow-up authorization for this single attempt only.
