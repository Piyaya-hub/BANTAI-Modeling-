# D-067: positive-confidence and absent-class auxiliary objective

Owner-authorized continuation from the verified D-057 step-000350 model/AdamW/RNG state: exactly92 forwards/backward passes/updates at steps351..442, preserving the suffix of epoch_batches(221,1,42,1), FP32, TF32 disabled, shortest384/longest576/square-pad576 and clip norm1.0. Both complete checkpoints, step400 and step442, were saved and verified. The endpoint cal_442 and prespecified cal_blend_2575 each received exactly47 validation forwards,94 total. The blend is0.25*original D-057 step350+0.75*cal_442, averaged in memory over558 canonical trainable tensors only. No reference GPU reruns, test.json or test-photo access, extra training, checkpoint promotion, commit, push or merge. The six core SOURCES and62 prior checkpoint bundles are unchanged;64 bundles verified in total.

This tests an auxiliary objective rather than replacing native focal loss. The native adapter and its loss/architecture remain unchanged; native checked_loss passes before adding auxiliary terms. The hypothesis linking focal weighting, query shifts and small-object scores is not assumed established. Model scores are not calibrated probabilities of correctness, despite the experimental name. The D-042 academic-use basis and uncertainty are unchanged.

Measured outcome: the combined objective is not achieved. Both new states retain15/24 cigarette coverage but lose Image777 coverage (4/7 versus old integration5/7); AP and strict F1 at.10 are lower. Image833 cigarette FP counts at.05 rise from12 to21/20, with maximum cigarette score0.078764 ->0.099080/0.090432. At.10, Image833 cigarette FP is zero in all three states, so that threshold does not establish a new suppression gain. On Image777, only GT280 has a qualifying cigarette query above.10 in either new state (0.134827/0.128243); none of the seven has one above.15. Other qualifying query scores remain low. This does not support promotion or further unrequested optimization.

## Exact objective and implementation

The isolated scripts are taco_score_calibration_epoch2.py and taco_score_calibration_loss.py. One adapter(model,batch) call supplies the unchanged native loss and900 predictions. The same pinned HF build_label_maps as the official decoder selects the eight ordinary subword spans, excluding delimiters/special tokens/masked tails. class_probs is mean(sigmoid(subword_logits)), shape[1,900,8]; class_logits is mean(subword_logits), same shape. sigmoid(mean(logits)) is not substituted for mean(sigmoid(logits)). Native predictions and GT boxes are both valid-image-normalized cxcywh, after the existing processor padding correction.

For each GT in processor order, choose argmax_q(IoU(pred_q,GT)+class_probs[q,GT_class]), with numeric query-ID-ascending ties. Selection and IoU are detached, so the auxiliary objective backpropagates through selected probabilities, not selection/box coordinates. Query reuse across GTs is allowed by this specified per-GT rule; it is not a one-to-one Hungarian assignment. Sum -w*log(p+1e-6) across GTs: cigarette1.5; classes1..7 weight.75 except plastic_film0. No minimum-IoU gate is added. Poorly localized/high-score queries can therefore be selected; the full selected IoU/query/score trace is retained.

For each class absent from the image, select queries with p strictly greater than.03 and compute -10*mean(log(1-p+1e-6)*p^2) over those selected queries, zero if none. Sum the per-class penalties. Present classes, including plastic_film when present, never receive this absent term; an absent plastic_film class does receive it. This selected-query mean makes the denominator explicit. Total loss=native+positive+absent. LR is8e-6 on67 remaining non-pure-film batches and1e-6 on25 pure-film batches; no resampling or per-class sampler. Original AdamW moments persist. Frozen backbones/text remain bitwise unchanged; finite gradients, encoder probes/nonzero updates, total-loss arithmetic and norm1 clipping are checked.

Exact source position is epoch1/next_batch129/global_step350. Source model hash matches independent D-058 step350 evidence, all558 original optimizer tensor names/order/moments/groups and Python/NumPy/CPU/CUDA RNG restore exactly. Validation snapshots/restores RNG and verifies unchanged weights. New optimizer counters are400/442; complete RNG and optimizer state are independently checked on CPU. Blend sources have bitwise-identical frozen entries and tied aliases are preserved; no blended checkpoint is saved. Endpoint and blend evaluation use separate GPU processes and no warmup forwards.

Private evidence: [taco-score-calibrated-246963f4958748b99627eb21318a97ad](../runs/training/taco-score-calibrated-246963f4958748b99627eb21318a97ad/record.json), [summary](../runs/training/taco-score-calibrated-246963f4958748b99627eb21318a97ad/score-calibration-summary.json), [strict assignments](../runs/training/taco-score-calibrated-246963f4958748b99627eb21318a97ad/strict-matching.json), [Image833 FP audit](../runs/training/taco-score-calibrated-246963f4958748b99627eb21318a97ad/image833-false-positives.json), [all24-cigarette ledger](../runs/training/taco-score-calibrated-246963f4958748b99627eb21318a97ad/cigarette-ledger.json). Full evaluation JSONs, class/coverage/threshold CSVs, per-GT diagnostics and92-step loss/selection/gradient trace remain local.

## Training objective trace

| Component | Mean over92 updates | Minimum | Maximum |
| --- | --- | --- | --- |
| native_loss | 5.712219 | 2.497808 | 16.254162 |
| loss_pos | 3.103495 | 0.000000 | 70.939301 |
| loss_absent | 0.055211 | 0.008220 | 0.181933 |
| loss | 8.870925 | 2.552066 | 79.181671 |
| gradient_norm_before_clip | 149.455305 | 46.958275 | 1199.681274 |

Nonzero-weight GT assignments: 128; 80 have selected IoU>=.50 at their training forward; mean IoU 0.583764. These are in-run training targets, not validation detections or independent accuracy. All class_probs/class_logits shapes are[1,900,8].

## Official eight-class metrics

Unchanged bantai.taco-evaluation.v2: single winning class/query, no NMS/no score floor, global100 valid predictions/image, COCO IoU.50:.05:.95. Validation loss is the native eight-class loss only, enabling comparison with previous states; auxiliary calibration terms are not silently added to official validation loss.

| State | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | seven_class_AP_excluding_plastic_bag | validation_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant-350 | 0.262337 | 0.288625 | 0.278353 | 0.065773 | 0.033448 | 0.315487 | 0.428890 | 0.218249 | 4.880779 |
| constant-442 | 0.170508 | 0.196086 | 0.179972 | 0.075112 | 0.048109 | 0.206941 | 0.370118 | 0.190859 | 4.613316 |
| integrated | 0.265563 | 0.292911 | 0.280572 | 0.071254 | 0.045737 | 0.316106 | 0.403085 | 0.231365 | 4.659642 |
| cal_442 | 0.245944 | 0.269195 | 0.254929 | 0.069400 | 0.041752 | 0.294034 | 0.378056 | 0.208943 | 4.666729 |
| cal_blend_2575 | 0.251234 | 0.276185 | 0.263083 | 0.065847 | 0.042927 | 0.301944 | 0.380836 | 0.214989 | 4.718684 |

cal_442 AP 0.245944, blend AP 0.251234; previous integrated AP 0.265563. Cigarette coverage at.05: 15/24 and 15/24; Image777 4/7 and 4/7, versus old integration15/24 and5/7. This is one bounded validation experiment, not a checkpoint selection or generalization claim.

## Per-class official AP and AR100

### constant-350

| Class | GT count | AP | AR100 |
| --- | --- | --- | --- |
| cigarette | 24 | 0.076901 | 0.329167 |
| plastic_bottle | 8 | 0.236139 | 0.337500 |
| plastic_bag | 2 | 0.570957 | 1.000000 |
| plastic_wrapper | 11 | 0.029703 | 0.081818 |
| plastic_film | 13 | 0.025307 | 0.476923 |
| drink_can | 7 | 0.367162 | 0.385714 |
| plastic_cup | 5 | 0.386139 | 0.380000 |
| styrofoam_piece | 5 | 0.406391 | 0.440000 |

### constant-442

| Class | GT count | AP | AR100 |
| --- | --- | --- | --- |
| cigarette | 24 | 0.082558 | 0.350000 |
| plastic_bottle | 8 | 0.180281 | 0.337500 |
| plastic_bag | 2 | 0.028053 | 0.500000 |
| plastic_wrapper | 11 | 0.128053 | 0.254545 |
| plastic_film | 13 | 0.033685 | 0.584615 |
| drink_can | 7 | 0.118812 | 0.114286 |
| plastic_cup | 5 | 0.386139 | 0.380000 |
| styrofoam_piece | 5 | 0.406483 | 0.440000 |

### integrated

| Class | GT count | AP | AR100 |
| --- | --- | --- | --- |
| cigarette | 24 | 0.081497 | 0.350000 |
| plastic_bottle | 8 | 0.193894 | 0.337500 |
| plastic_bag | 2 | 0.504950 | 0.500000 |
| plastic_wrapper | 11 | 0.154653 | 0.254545 |
| plastic_film | 13 | 0.029740 | 0.576923 |
| drink_can | 7 | 0.367162 | 0.385714 |
| plastic_cup | 5 | 0.386139 | 0.380000 |
| styrofoam_piece | 5 | 0.406466 | 0.440000 |

### cal_442

| Class | GT count | AP | AR100 |
| --- | --- | --- | --- |
| cigarette | 24 | 0.070026 | 0.366667 |
| plastic_bottle | 8 | 0.163861 | 0.337500 |
| plastic_bag | 2 | 0.504950 | 0.500000 |
| plastic_wrapper | 11 | 0.029703 | 0.081818 |
| plastic_film | 13 | 0.027785 | 0.538462 |
| drink_can | 7 | 0.378878 | 0.400000 |
| plastic_cup | 5 | 0.386139 | 0.380000 |
| styrofoam_piece | 5 | 0.406211 | 0.420000 |

### cal_blend_2575

| Class | GT count | AP | AR100 |
| --- | --- | --- | --- |
| cigarette | 24 | 0.070446 | 0.345833 |
| plastic_bottle | 8 | 0.204785 | 0.337500 |
| plastic_bag | 2 | 0.504950 | 0.500000 |
| plastic_wrapper | 11 | 0.029703 | 0.081818 |
| plastic_film | 13 | 0.028509 | 0.561538 |
| drink_can | 7 | 0.378878 | 0.400000 |
| plastic_cup | 5 | 0.386139 | 0.380000 |
| styrofoam_piece | 5 | 0.406462 | 0.440000 |

## Material/polymer diagnostic COCO metrics

Remap GT and the same saved global100 predictions/image, no NMS/score floor; actual4/3-category COCO lists and original GT areas retained. Eight-class calculations independently reproduce saved official metrics. Mapped AP is a different category task and is not improved subtype recognition.

| State | Taxonomy | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant-350 | 8-specific | 0.262337 | 0.288625 | 0.278353 | 0.065773 | 0.033448 | 0.315487 | 0.428890 |
| constant-350 | 4-material | 0.300493 | 0.361440 | 0.318400 | 0.065773 | 0.136507 | 0.419243 | 0.478464 |
| constant-350 | 3-material | 0.303410 | 0.385760 | 0.330505 | 0.065773 | 0.231992 | 0.440741 | 0.497385 |
| constant-442 | 8-specific | 0.170508 | 0.196086 | 0.179972 | 0.075112 | 0.048109 | 0.206941 | 0.370118 |
| constant-442 | 4-material | 0.245494 | 0.307580 | 0.251203 | 0.075112 | 0.139148 | 0.344764 | 0.419661 |
| constant-442 | 3-material | 0.225596 | 0.308955 | 0.235367 | 0.075112 | 0.239391 | 0.321230 | 0.419156 |
| integrated | 8-specific | 0.265563 | 0.292911 | 0.280572 | 0.071254 | 0.045737 | 0.316106 | 0.403085 |
| integrated | 4-material | 0.309859 | 0.376847 | 0.325922 | 0.071254 | 0.137069 | 0.431522 | 0.486877 |
| integrated | 3-material | 0.311858 | 0.402402 | 0.335028 | 0.071254 | 0.236609 | 0.451775 | 0.508874 |
| cal_442 | 8-specific | 0.245944 | 0.269195 | 0.254929 | 0.069400 | 0.041752 | 0.294034 | 0.378056 |
| cal_442 | 4-material | 0.309977 | 0.371092 | 0.321731 | 0.069400 | 0.129889 | 0.435362 | 0.489615 |
| cal_442 | 3-material | 0.314894 | 0.397029 | 0.333139 | 0.069400 | 0.225631 | 0.462017 | 0.519192 |
| cal_blend_2575 | 8-specific | 0.251234 | 0.276185 | 0.263083 | 0.065847 | 0.042927 | 0.301944 | 0.380836 |
| cal_blend_2575 | 4-material | 0.310492 | 0.371315 | 0.324194 | 0.065847 | 0.131100 | 0.435828 | 0.491971 |
| cal_blend_2575 | 3-material | 0.313668 | 0.395854 | 0.334722 | 0.065847 | 0.225015 | 0.460216 | 0.515278 |

## Candidate coverage and locked bins

Non-exclusive candidate coverage at score>=.05 and IoU>=.50; not COCO recall. Locked bins retain D-058 reference geometry256/384 and denominators14/10/15/36. Standard first15 coverage is separately calculated at the raw score-ranked official-top100 pool; per-object drawing uses GT-prioritized overlaps.

| State | Pool | GT /75 | Cig /24 | 777 /7 | 777 first15 /7 | <4 /14 | 4–<8 /10 | 8–<16 /15 | >=16 /36 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant-350 | all 900 (diagnostic) | 34 | 14 | 4 | 2 | 5 | 6 | 8 | 15 |
| constant-350 | official top100 | 34 | 14 | 4 | 2 | 5 | 6 | 8 | 15 |
| constant-442 | all 900 (diagnostic) | 35 | 15 | 5 | 3 | 6 | 6 | 9 | 14 |
| constant-442 | official top100 | 35 | 15 | 5 | 3 | 6 | 6 | 9 | 14 |
| integrated | all 900 (diagnostic) | 38 | 15 | 5 | 3 | 6 | 6 | 9 | 17 |
| integrated | official top100 | 38 | 15 | 5 | 3 | 6 | 6 | 9 | 17 |
| cal_442 | all 900 (diagnostic) | 36 | 15 | 4 | 2 | 5 | 7 | 8 | 16 |
| cal_442 | official top100 | 36 | 15 | 4 | 2 | 5 | 7 | 8 | 16 |
| cal_blend_2575 | all 900 (diagnostic) | 37 | 15 | 4 | 2 | 5 | 7 | 9 | 16 |
| cal_blend_2575 | official top100 | 37 | 15 | 4 | 2 | 5 | 7 | 9 | 16 |

## Strict one-to-one Precision / Recall / F1

All valid900 saved winning-class queries, no NMS/global100 cap; threshold-inclusive score filtering. Per image, greedily assign descending score/query-ID ascending to the maximum-IoU unmatched GT in the active category, IoU>=.50, GT-ID ascending ties. One prediction and GT can each supply at most one TP. FP/FN are annotation-relative, not independently confirmed background. Metrics are micro-aggregated over47 images. Zero denominators yield0. Remapping occurs before assignment. These differ from both non-exclusive coverage and COCO multi-IoU matching.

### constant-350

| Taxonomy | Threshold | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | 0.050000 | 34 | 1190 | 41 | 0.027778 | 0.453333 | 0.052348 |
| 8-specific | 0.080000 | 27 | 450 | 48 | 0.056604 | 0.360000 | 0.097826 |
| 8-specific | 0.100000 | 25 | 273 | 50 | 0.083893 | 0.333333 | 0.134048 |
| 8-specific | 0.120000 | 22 | 184 | 53 | 0.106796 | 0.293333 | 0.156584 |
| 8-specific | 0.150000 | 20 | 120 | 55 | 0.142857 | 0.266667 | 0.186047 |
| 8-specific | 0.200000 | 14 | 67 | 61 | 0.172840 | 0.186667 | 0.179487 |
| 4-material | 0.050000 | 54 | 1170 | 21 | 0.044118 | 0.720000 | 0.083141 |
| 4-material | 0.080000 | 48 | 429 | 27 | 0.100629 | 0.640000 | 0.173913 |
| 4-material | 0.100000 | 46 | 252 | 29 | 0.154362 | 0.613333 | 0.246649 |
| 4-material | 0.120000 | 44 | 162 | 31 | 0.213592 | 0.586667 | 0.313167 |
| 4-material | 0.150000 | 38 | 102 | 37 | 0.271429 | 0.506667 | 0.353488 |
| 4-material | 0.200000 | 33 | 48 | 42 | 0.407407 | 0.440000 | 0.423077 |
| 3-material | 0.050000 | 57 | 1167 | 18 | 0.046569 | 0.760000 | 0.087760 |
| 3-material | 0.080000 | 51 | 426 | 24 | 0.106918 | 0.680000 | 0.184783 |
| 3-material | 0.100000 | 49 | 249 | 26 | 0.164430 | 0.653333 | 0.262735 |
| 3-material | 0.120000 | 47 | 159 | 28 | 0.228155 | 0.626667 | 0.334520 |
| 3-material | 0.150000 | 42 | 98 | 33 | 0.300000 | 0.560000 | 0.390698 |
| 3-material | 0.200000 | 38 | 43 | 37 | 0.469136 | 0.506667 | 0.487179 |

### constant-442

| Taxonomy | Threshold | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | 0.050000 | 35 | 1083 | 40 | 0.031306 | 0.466667 | 0.058676 |
| 8-specific | 0.080000 | 29 | 407 | 46 | 0.066514 | 0.386667 | 0.113503 |
| 8-specific | 0.100000 | 27 | 246 | 48 | 0.098901 | 0.360000 | 0.155172 |
| 8-specific | 0.120000 | 23 | 167 | 52 | 0.121053 | 0.306667 | 0.173585 |
| 8-specific | 0.150000 | 19 | 116 | 56 | 0.140741 | 0.253333 | 0.180952 |
| 8-specific | 0.200000 | 14 | 63 | 61 | 0.181818 | 0.186667 | 0.184211 |
| 4-material | 0.050000 | 53 | 1065 | 22 | 0.047406 | 0.706667 | 0.088852 |
| 4-material | 0.080000 | 47 | 389 | 28 | 0.107798 | 0.626667 | 0.183953 |
| 4-material | 0.100000 | 45 | 228 | 30 | 0.164835 | 0.600000 | 0.258621 |
| 4-material | 0.120000 | 41 | 149 | 34 | 0.215789 | 0.546667 | 0.309434 |
| 4-material | 0.150000 | 37 | 98 | 38 | 0.274074 | 0.493333 | 0.352381 |
| 4-material | 0.200000 | 32 | 45 | 43 | 0.415584 | 0.426667 | 0.421053 |
| 3-material | 0.050000 | 56 | 1062 | 19 | 0.050089 | 0.746667 | 0.093881 |
| 3-material | 0.080000 | 50 | 386 | 25 | 0.114679 | 0.666667 | 0.195695 |
| 3-material | 0.100000 | 48 | 225 | 27 | 0.175824 | 0.640000 | 0.275862 |
| 3-material | 0.120000 | 44 | 146 | 31 | 0.231579 | 0.586667 | 0.332075 |
| 3-material | 0.150000 | 41 | 94 | 34 | 0.303704 | 0.546667 | 0.390476 |
| 3-material | 0.200000 | 36 | 41 | 39 | 0.467532 | 0.480000 | 0.473684 |

### integrated

| Taxonomy | Threshold | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | 0.050000 | 38 | 1114 | 37 | 0.032986 | 0.506667 | 0.061940 |
| 8-specific | 0.080000 | 30 | 411 | 45 | 0.068027 | 0.400000 | 0.116279 |
| 8-specific | 0.100000 | 29 | 252 | 46 | 0.103203 | 0.386667 | 0.162921 |
| 8-specific | 0.120000 | 25 | 164 | 50 | 0.132275 | 0.333333 | 0.189394 |
| 8-specific | 0.150000 | 22 | 113 | 53 | 0.162963 | 0.293333 | 0.209524 |
| 8-specific | 0.200000 | 17 | 59 | 58 | 0.223684 | 0.226667 | 0.225166 |
| 4-material | 0.050000 | 55 | 1097 | 20 | 0.047743 | 0.733333 | 0.089650 |
| 4-material | 0.080000 | 48 | 393 | 27 | 0.108844 | 0.640000 | 0.186047 |
| 4-material | 0.100000 | 47 | 234 | 28 | 0.167260 | 0.626667 | 0.264045 |
| 4-material | 0.120000 | 43 | 146 | 32 | 0.227513 | 0.573333 | 0.325758 |
| 4-material | 0.150000 | 39 | 96 | 36 | 0.288889 | 0.520000 | 0.371429 |
| 4-material | 0.200000 | 34 | 42 | 41 | 0.447368 | 0.453333 | 0.450331 |
| 3-material | 0.050000 | 58 | 1094 | 17 | 0.050347 | 0.773333 | 0.094540 |
| 3-material | 0.080000 | 51 | 390 | 24 | 0.115646 | 0.680000 | 0.197674 |
| 3-material | 0.100000 | 50 | 231 | 25 | 0.177936 | 0.666667 | 0.280899 |
| 3-material | 0.120000 | 46 | 143 | 29 | 0.243386 | 0.613333 | 0.348485 |
| 3-material | 0.150000 | 43 | 92 | 32 | 0.318519 | 0.573333 | 0.409524 |
| 3-material | 0.200000 | 38 | 38 | 37 | 0.500000 | 0.506667 | 0.503311 |

### cal_442

| Taxonomy | Threshold | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | 0.050000 | 36 | 1634 | 39 | 0.021557 | 0.480000 | 0.041261 |
| 8-specific | 0.080000 | 29 | 601 | 46 | 0.046032 | 0.386667 | 0.082270 |
| 8-specific | 0.100000 | 27 | 363 | 48 | 0.069231 | 0.360000 | 0.116129 |
| 8-specific | 0.120000 | 25 | 239 | 50 | 0.094697 | 0.333333 | 0.147493 |
| 8-specific | 0.150000 | 22 | 135 | 53 | 0.140127 | 0.293333 | 0.189655 |
| 8-specific | 0.200000 | 16 | 73 | 59 | 0.179775 | 0.213333 | 0.195122 |
| 4-material | 0.050000 | 55 | 1615 | 20 | 0.032934 | 0.733333 | 0.063037 |
| 4-material | 0.080000 | 49 | 581 | 26 | 0.077778 | 0.653333 | 0.139007 |
| 4-material | 0.100000 | 47 | 343 | 28 | 0.120513 | 0.626667 | 0.202151 |
| 4-material | 0.120000 | 46 | 218 | 29 | 0.174242 | 0.613333 | 0.271386 |
| 4-material | 0.150000 | 41 | 116 | 34 | 0.261146 | 0.546667 | 0.353448 |
| 4-material | 0.200000 | 35 | 54 | 40 | 0.393258 | 0.466667 | 0.426829 |
| 3-material | 0.050000 | 58 | 1612 | 17 | 0.034731 | 0.773333 | 0.066476 |
| 3-material | 0.080000 | 52 | 578 | 23 | 0.082540 | 0.693333 | 0.147518 |
| 3-material | 0.100000 | 50 | 340 | 25 | 0.128205 | 0.666667 | 0.215054 |
| 3-material | 0.120000 | 49 | 215 | 26 | 0.185606 | 0.653333 | 0.289086 |
| 3-material | 0.150000 | 45 | 112 | 30 | 0.286624 | 0.600000 | 0.387931 |
| 3-material | 0.200000 | 40 | 49 | 35 | 0.449438 | 0.533333 | 0.487805 |

### cal_blend_2575

| Taxonomy | Threshold | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | 0.050000 | 37 | 1498 | 38 | 0.024104 | 0.493333 | 0.045963 |
| 8-specific | 0.080000 | 30 | 553 | 45 | 0.051458 | 0.400000 | 0.091185 |
| 8-specific | 0.100000 | 27 | 342 | 48 | 0.073171 | 0.360000 | 0.121622 |
| 8-specific | 0.120000 | 24 | 224 | 51 | 0.096774 | 0.320000 | 0.148607 |
| 8-specific | 0.150000 | 21 | 133 | 54 | 0.136364 | 0.280000 | 0.183406 |
| 8-specific | 0.200000 | 15 | 71 | 60 | 0.174419 | 0.200000 | 0.186335 |
| 4-material | 0.050000 | 56 | 1479 | 19 | 0.036482 | 0.746667 | 0.069565 |
| 4-material | 0.080000 | 50 | 533 | 25 | 0.085763 | 0.666667 | 0.151976 |
| 4-material | 0.100000 | 47 | 322 | 28 | 0.127371 | 0.626667 | 0.211712 |
| 4-material | 0.120000 | 45 | 203 | 30 | 0.181452 | 0.600000 | 0.278638 |
| 4-material | 0.150000 | 40 | 114 | 35 | 0.259740 | 0.533333 | 0.349345 |
| 4-material | 0.200000 | 34 | 52 | 41 | 0.395349 | 0.453333 | 0.422360 |
| 3-material | 0.050000 | 59 | 1476 | 16 | 0.038436 | 0.786667 | 0.073292 |
| 3-material | 0.080000 | 53 | 530 | 22 | 0.090909 | 0.706667 | 0.161094 |
| 3-material | 0.100000 | 50 | 319 | 25 | 0.135501 | 0.666667 | 0.225225 |
| 3-material | 0.120000 | 48 | 200 | 27 | 0.193548 | 0.640000 | 0.297214 |
| 3-material | 0.150000 | 44 | 110 | 31 | 0.285714 | 0.586667 | 0.384279 |
| 3-material | 0.200000 | 39 | 47 | 36 | 0.453488 | 0.520000 | 0.484472 |

## Image777 cigarette scores, ranks and IoUs (GT278..284)

Best-IoU cigarette query is selected without a score floor, and may differ from the highest-score cigarette query with IoU>=.50. Both are shown to separate confidence from localization changes. A high-confidence query below IoU.50 is not counted correct. Raw first15 may omit qualifying low-ranked cigarettes; a cap15 does not guarantee all seven are drawn.

| State | GT | Best-IoU q | Score | Rank | IoU | Highest aligned q | Score | Rank | IoU | Covered @.05 | First15 covered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant-350 | 278 | 36 | 0.050408 | 34 | 0.870484 | 36 | 0.050408 | 34 | 0.870484 | True | False |
| constant-350 | 279 | 67 | 0.060135 | 22 | 0.744656 | 67 | 0.060135 | 22 | 0.744656 | True | False |
| constant-350 | 280 | 12 | 0.030205 | 85 | 0.871991 | 5 | 0.106792 | 6 | 0.853264 | True | True |
| constant-350 | 281 | 701 | 0.002279 | 888 | 0.561036 | 587 | 0.017031 | 229 | 0.518107 | False | False |
| constant-350 | 282 | 38 | 0.005241 | 722 | 0.537811 | 34 | 0.027806 | 98 | 0.530087 | False | False |
| constant-350 | 283 | 122 | 0.006848 | 604 | 0.689437 | 122 | 0.006848 | 604 | 0.689437 | False | False |
| constant-350 | 284 | 30 | 0.073511 | 13 | 0.785302 | 30 | 0.073511 | 13 | 0.785302 | True | True |
| constant-442 | 278 | 28 | 0.053679 | 25 | 0.881301 | 28 | 0.053679 | 25 | 0.881301 | True | False |
| constant-442 | 279 | 173 | 0.012061 | 309 | 0.794829 | 58 | 0.059589 | 17 | 0.767120 | True | False |
| constant-442 | 280 | 12 | 0.034520 | 62 | 0.851611 | 6 | 0.112753 | 3 | 0.844376 | True | True |
| constant-442 | 281 | 825 | 0.002462 | 865 | 0.876267 | 669 | 0.015506 | 231 | 0.537764 | False | False |
| constant-442 | 282 | 21 | 0.017191 | 208 | 0.585264 | 5 | 0.109173 | 4 | 0.520966 | True | True |
| constant-442 | 283 | 93 | 0.006655 | 584 | 0.769422 | 93 | 0.006655 | 584 | 0.769422 | False | False |
| constant-442 | 284 | 19 | 0.077522 | 9 | 0.776195 | 19 | 0.077522 | 9 | 0.776195 | True | True |
| integrated | 278 | 32 | 0.052704 | 29 | 0.877367 | 32 | 0.052704 | 29 | 0.877367 | True | False |
| integrated | 279 | 189 | 0.013614 | 273 | 0.806409 | 59 | 0.059266 | 20 | 0.763111 | True | False |
| integrated | 280 | 14 | 0.032971 | 69 | 0.858967 | 6 | 0.111749 | 3 | 0.845876 | True | True |
| integrated | 281 | 795 | 0.002210 | 883 | 0.793612 | 647 | 0.015248 | 250 | 0.529616 | False | False |
| integrated | 282 | 23 | 0.029291 | 89 | 0.576940 | 5 | 0.107911 | 5 | 0.502785 | True | True |
| integrated | 283 | 97 | 0.008332 | 474 | 0.730058 | 79 | 0.008667 | 449 | 0.536106 | False | False |
| integrated | 284 | 20 | 0.076084 | 9 | 0.780633 | 20 | 0.076084 | 9 | 0.780633 | True | True |
| cal_442 | 278 | 40 | 0.064624 | 27 | 0.865098 | 40 | 0.064624 | 27 | 0.865098 | True | False |
| cal_442 | 279 | 64 | 0.071423 | 23 | 0.760299 | 64 | 0.071423 | 23 | 0.760299 | True | False |
| cal_442 | 280 | 12 | 0.038413 | 80 | 0.865585 | 6 | 0.134827 | 3 | 0.851260 | True | True |
| cal_442 | 281 | 650 | 0.005260 | 772 | 0.703635 | 646 | 0.019350 | 246 | 0.550189 | False | False |
| cal_442 | 282 | 23 | 0.042709 | 66 | 0.568015 | 23 | 0.042709 | 66 | 0.568015 | False | False |
| cal_442 | 283 | 97 | 0.011969 | 401 | 0.708814 | 78 | 0.015930 | 295 | 0.516350 | False | False |
| cal_442 | 284 | 22 | 0.090584 | 9 | 0.801477 | 22 | 0.090584 | 9 | 0.801477 | True | True |
| cal_blend_2575 | 278 | 41 | 0.060901 | 28 | 0.863911 | 41 | 0.060901 | 28 | 0.863911 | True | False |
| cal_blend_2575 | 279 | 65 | 0.069370 | 21 | 0.755274 | 65 | 0.069370 | 21 | 0.755274 | True | False |
| cal_blend_2575 | 280 | 12 | 0.035395 | 84 | 0.868809 | 6 | 0.128243 | 3 | 0.851872 | True | True |
| cal_blend_2575 | 281 | 627 | 0.018316 | 246 | 0.545073 | 627 | 0.018316 | 246 | 0.545073 | False | False |
| cal_blend_2575 | 282 | 26 | 0.037498 | 76 | 0.560554 | 26 | 0.037498 | 76 | 0.560554 | False | False |
| cal_blend_2575 | 283 | 100 | 0.008617 | 558 | 0.725105 | 81 | 0.017327 | 259 | 0.508710 | False | False |
| cal_blend_2575 | 284 | 23 | 0.086532 | 10 | 0.791796 | 23 | 0.086532 | 10 | 0.791796 | True | True |

## Image833 false positives

Its validation annotations independently confirm no cigarette GT. Consequently every winning-class cigarette prediction there is an annotation-relative cigarette FP. Counts below include all valid900 queries above the score floor, no NMS/cap. Overall FP is strict unmatched count and includes duplicates, wrong class and insufficient localization. Taxonomy-specific full counts and unmatched query IDs are in JSON/CSV; cigarette counts are taxonomy invariant. Maximum cigarette score is over stored winning-class cigarette queries, not all nonwinning class probabilities. This is one negative image, not a general false-positive-rate estimate.

| State | Threshold | GT count | TP | FP | FN | Cigarette FP | Maximum cigarette score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constant-350 | 0.050000 | 1 | 1 | 27 | 0 | 9 | 0.070247 |
| constant-350 | 0.080000 | 1 | 1 | 7 | 0 | 0 | 0.070247 |
| constant-350 | 0.100000 | 1 | 1 | 4 | 0 | 0 | 0.070247 |
| constant-350 | 0.120000 | 1 | 1 | 2 | 0 | 0 | 0.070247 |
| constant-350 | 0.150000 | 1 | 1 | 2 | 0 | 0 | 0.070247 |
| constant-350 | 0.200000 | 1 | 1 | 1 | 0 | 0 | 0.070247 |
| constant-442 | 0.050000 | 1 | 0 | 26 | 1 | 12 | 0.080983 |
| constant-442 | 0.080000 | 1 | 0 | 6 | 1 | 1 | 0.080983 |
| constant-442 | 0.100000 | 1 | 0 | 4 | 1 | 0 | 0.080983 |
| constant-442 | 0.120000 | 1 | 0 | 3 | 1 | 0 | 0.080983 |
| constant-442 | 0.150000 | 1 | 0 | 2 | 1 | 0 | 0.080983 |
| constant-442 | 0.200000 | 1 | 0 | 2 | 1 | 0 | 0.080983 |
| integrated | 0.050000 | 1 | 1 | 26 | 0 | 12 | 0.078764 |
| integrated | 0.080000 | 1 | 1 | 5 | 0 | 0 | 0.078764 |
| integrated | 0.100000 | 1 | 1 | 3 | 0 | 0 | 0.078764 |
| integrated | 0.120000 | 1 | 1 | 2 | 0 | 0 | 0.078764 |
| integrated | 0.150000 | 1 | 1 | 1 | 0 | 0 | 0.078764 |
| integrated | 0.200000 | 1 | 1 | 1 | 0 | 0 | 0.078764 |
| cal_442 | 0.050000 | 1 | 1 | 38 | 0 | 21 | 0.099080 |
| cal_442 | 0.080000 | 1 | 1 | 8 | 0 | 2 | 0.099080 |
| cal_442 | 0.100000 | 1 | 1 | 4 | 0 | 0 | 0.099080 |
| cal_442 | 0.120000 | 1 | 1 | 3 | 0 | 0 | 0.099080 |
| cal_442 | 0.150000 | 1 | 1 | 2 | 0 | 0 | 0.099080 |
| cal_442 | 0.200000 | 1 | 1 | 1 | 0 | 0 | 0.099080 |
| cal_blend_2575 | 0.050000 | 1 | 1 | 37 | 0 | 20 | 0.090432 |
| cal_blend_2575 | 0.080000 | 1 | 1 | 8 | 0 | 1 | 0.090432 |
| cal_blend_2575 | 0.100000 | 1 | 1 | 4 | 0 | 0 | 0.090432 |
| cal_blend_2575 | 0.120000 | 1 | 1 | 3 | 0 | 0 | 0.090432 |
| cal_blend_2575 | 0.150000 | 1 | 1 | 2 | 0 | 0 | 0.090432 |
| cal_blend_2575 | 0.200000 | 1 | 1 | 1 | 0 | 0 | 0.090432 |

## Verification and boundaries

Guarded full suite: 278 tests, 262 passed, 16 documented skips, zero failures/errors. Skips are15 training-execution fixtures forbidden during software verification and one Windows symlink-privilege fixture. Native Windows Tk: 2 cases (standard/per-object), all15 taxonomy/NMS combinations at Image777/top15, all47 IDs, correct panels, navigation/GT selection/zoom/pan. Model loading/forward, CUDA, backward/updates, state deserialization and exports are blocked during replay; saved JSON unchanged. Human visual acceptance remains pending.

All62 prior plus2 new bundles verified with their contracts and file hashes, frozen hashes unchanged; six core bytes and official protocol unchanged. Peak allocated VRAM 3.550 GiB, reserved 4.023 GiB (whole training/endpoint process, not per-state); blend peak recorded separately. Prior protected file count 2273; unchanged=True. All12 previous replay choices are preserved when adding the new one.

## PowerShell saved replay

Run from the project root; close each window before launching the next. Box topK15 is explicit. Select **4 material/hazard categories (mapped)** or **3 material/hazard categories (polymer grouped)** in the toolbar; official eight-class/raw defaults remain. All five diagnostic NMS modes remain selectable and can alter displayed coverage; official scores are unchanged. The panels are old integration, cal_442 and cal_blend_2575. All47 validation images are accessible from Image777. Use Image833 in the selector to inspect the negative-image audit.

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
$review = 'runs/training/d060-saved-review/review-manifest.json'
$choice = '42 / 1e-5 @ 384/576 (score-calibrated vs integrated)'
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --saved-review $review --run-choice $choice --gui --image-id 777 --box-top-k 15
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --saved-review $review --run-choice $choice --gui --per-object --image-id 777 --box-top-k 15
```
