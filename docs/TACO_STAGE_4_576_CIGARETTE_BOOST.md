# D-066: late-Epoch-2 cigarette-boosted continuation

Owner-authorized controlled continuation from the verified D-057 step-000350 checkpoint, exact position epoch 1 / next_batch 129 / global_step 350. Completed exactly 92 training forwards, backward passes and AdamW updates at steps 351..442, plus two 47-image validations (94 forwards). Reference predictions were reused with zero GPU reruns. No test.json or test photos were opened. The six core SOURCES and all 60 prior checkpoint bundles remain byte-for-byte unchanged. Two complete new bundles at step400 and step442 bring the verified total to 62. No commit, push, merge or checkpoint promotion.

This experiment tests the combined tiered-LR hypothesis; it does not establish that film batches or negative images caused the historical score changes. Training remains the unchanged eight-class native loss. Material/polymer mappings are saved-result diagnostics, not training-label changes or visual material recognition. The D-042 academic-use basis and its uncertainties remain unchanged.

## Exact continuation and safeguards

The source model parameter hash matches the independently recorded D-058 step350 hash. Model weights, all558 trainable tensor names/order, complete AdamW moments/groups and Python/NumPy/CPU/CUDA RNG restore exactly before any update. Backbones and text remain frozen, with frozen-parameter hashes checked at both new saves and validation. Same pinned Stage3D adapter, original `epoch_batches(221,1,42,1)[129:]`, batch size1, original photos, FP32, TF32 disabled, shortest384/longest576/square-pad576, and native clip norm1.0. Finite gradients and nonzero encoder-probe updates are checked by the unchanged core step. Validation snapshots/restores RNG and verifies unchanged parameters.

Full epoch composition is30 cigarette images,61 pure-film images,130 other non-cigarette images (221 total). The ACTUAL remaining suffix has11 cigarette,25 pure-film and56 other batches (92 total). Cigarette takes priority in mixed images. Before every native optimizer step, all param groups receive LR1.5e-5 / 1e-7 / 5e-6 respectively. No resampling, loss reweighting, optimizer-state reset or backbone unfreezing. AdamW momentum persists across tiers, so pure-film damping reduces a full momentum/weight-decay update, not just an isolated class gradient.

The prespecified blend is exactly0.50 * new cigarette-boost step442 +0.50 * original D-057 constant-LR step442. Averaging is CPU FP32 over the558 canonical trainable tensors only; frozen persisted entries are bitwise identical across sources, tied aliases are preserved, and no blended checkpoint is saved. Endpoint and blend each run exactly47 forwards, with no warmup/reference reruns; separate processes release CUDA memory between them.

Private run: [taco-cigarette-boost-8f99ae980978433694ac99f8c37b2273](../runs/training/taco-cigarette-boost-8f99ae980978433694ac99f8c37b2273/record.json); [comparison summary](../runs/training/taco-cigarette-boost-8f99ae980978433694ac99f8c37b2273/cigarette-boost-summary.json), [all24-cigarette ledger](../runs/training/taco-cigarette-boost-8f99ae980978433694ac99f8c37b2273/cigarette-ledger.json), [per-object outcomes](../runs/training/taco-cigarette-boost-8f99ae980978433694ac99f8c37b2273/cigarette-boost-per-object.json). Full evaluation JSONs, original-frame predictions, LR/gradient/update trace, metrics/coverage/taxonomy CSVs, and complete checkpoint restore/verification evidence remain local.

| Checkpoint | Position | AdamW step counters | Last-batch LR | Trainable tensors |
| --- | --- | --- | --- | --- |
| 400 | {'epoch': 1, 'next_batch': 179, 'global_step': 400} | 400 | 0.000005 | 558 |
| 442 | {'epoch': 2, 'next_batch': 0, 'global_step': 442} | 442 | 0.000005 | 558 |

## Official eight-class results

All official metrics retain `bantai.taco-evaluation.v2`, single winning class/query, global top100 valid predictions/image and no NMS or score floor. Seven-class AP is the arithmetic mean excluding plastic_bag; it is supplementary. Validation loss retains the eight-class native loss for both evaluated states.

| State | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | seven_class_AP_excluding_plastic_bag | validation_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant-350 | 0.262337 | 0.288625 | 0.278353 | 0.065773 | 0.033448 | 0.315487 | 0.428890 | 0.218249 | 4.880779 |
| constant-442 | 0.170508 | 0.196086 | 0.179972 | 0.075112 | 0.048109 | 0.206941 | 0.370118 | 0.190859 | 4.613316 |
| integrated | 0.265563 | 0.292911 | 0.280572 | 0.071254 | 0.045737 | 0.316106 | 0.403085 | 0.231365 | 4.659642 |
| cig_boost_442 | 0.250675 | 0.276330 | 0.265333 | 0.064480 | 0.039871 | 0.300432 | 0.376967 | 0.214350 | 4.696627 |
| cig_boost_blend | 0.257129 | 0.283554 | 0.271279 | 0.057678 | 0.039979 | 0.308331 | 0.394606 | 0.221726 | 4.644986 |

New endpoint AP 0.250675; 50/50 blend AP 0.257129; previous integrated reference AP 0.265563. Endpoint and new blend cigarette coverage are 14/24 and 14/24, with Image777 4/7 and 4/7. These are measured validation comparisons with selection bias, not a generalization claim or automatic checkpoint choice.

This is a negative result for the proposed cigarette improvement: both new states have14/24 cigarette coverage and4/7 on Image777, below the old integrated reference15/24 and5/7. Neither exceeds its AP0.265563. Raw first15 coverage is only2/7 for both new states versus3/7 for the old integration. The requested top15 setting is implemented explicitly, but does not display all qualifying cigarettes under global score ordering. The per-object view and saved per-GT ledger expose the remaining query evidence. No checkpoint is selected or promoted.

## Per-class official AP / AR100

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

### cig_boost_442

| Class | GT count | AP | AR100 |
| --- | --- | --- | --- |
| cigarette | 24 | 0.077805 | 0.329167 |
| plastic_bottle | 8 | 0.204785 | 0.337500 |
| plastic_bag | 2 | 0.504950 | 0.500000 |
| plastic_wrapper | 11 | 0.029703 | 0.081818 |
| plastic_film | 13 | 0.028414 | 0.561538 |
| drink_can | 7 | 0.367162 | 0.385714 |
| plastic_cup | 5 | 0.386139 | 0.380000 |
| styrofoam_piece | 5 | 0.406442 | 0.440000 |

### cig_boost_blend

| Class | GT count | AP | AR100 |
| --- | --- | --- | --- |
| cigarette | 24 | 0.073669 | 0.350000 |
| plastic_bottle | 8 | 0.204785 | 0.337500 |
| plastic_bag | 2 | 0.504950 | 0.500000 |
| plastic_wrapper | 11 | 0.084653 | 0.163636 |
| plastic_film | 13 | 0.029176 | 0.600000 |
| drink_can | 7 | 0.367162 | 0.385714 |
| plastic_cup | 5 | 0.386139 | 0.380000 |
| styrofoam_piece | 5 | 0.406496 | 0.440000 |

## Candidate coverage and locked bins

Coverage uses any same-class query with score >=.05 and IoU >=.50, non-exclusive GT matching. It is not COCO recall or accuracy. Locked bins retain the exact D-058 256/384 reference geometry and denominators14/10/15/36; they are not recomputed at384/576. Standard first15 coverage uses score-ranked passing official-top100 candidates and raw NMS default. Per-object drawing prioritizes overlaps with the selected GT and is a different display ordering.

| State | Pool | GT /75 | Cig /24 | 777 /7 | 777 first15 /7 | 777 passing | <4 /14 | 4–<8 /10 | 8–<16 /15 | >=16 /36 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant-350 | all 900 (diagnostic) | 34 | 14 | 4 | 2 | 34 | 5 | 6 | 8 | 15 |
| constant-350 | official top100 | 34 | 14 | 4 | 2 | 34 | 5 | 6 | 8 | 15 |
| constant-442 | all 900 (diagnostic) | 35 | 15 | 5 | 3 | 29 | 6 | 6 | 9 | 14 |
| constant-442 | official top100 | 35 | 15 | 5 | 3 | 29 | 6 | 6 | 9 | 14 |
| integrated | all 900 (diagnostic) | 38 | 15 | 5 | 3 | 31 | 6 | 6 | 9 | 17 |
| integrated | official top100 | 38 | 15 | 5 | 3 | 31 | 6 | 6 | 9 | 17 |
| cig_boost_442 | all 900 (diagnostic) | 34 | 14 | 4 | 2 | 31 | 5 | 6 | 8 | 15 |
| cig_boost_442 | official top100 | 34 | 14 | 4 | 2 | 31 | 5 | 6 | 8 | 15 |
| cig_boost_blend | all 900 (diagnostic) | 36 | 14 | 4 | 2 | 30 | 5 | 6 | 9 | 16 |
| cig_boost_blend | official top100 | 36 | 14 | 4 | 2 | 30 | 5 | 6 | 9 | 16 |

## D-065 mapped material and polymer metrics

Raw no-NMS remapping of both GT and the same saved global-top100 predictions/image, no score floor, identical COCO settings/areas. These4/3-category tasks are different from official eight-class detection; higher mapped AP does not fix subtype predictions. Official metrics remain untouched. Existing GUI taxonomy and five diagnostic NMS options remain selectable and reversible; these mapped numbers do not apply diagnostic NMS.

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
| cig_boost_442 | 8-specific | 0.250675 | 0.276330 | 0.265333 | 0.064480 | 0.039871 | 0.300432 | 0.376967 |
| cig_boost_442 | 4-material | 0.307638 | 0.373321 | 0.325317 | 0.064480 | 0.141296 | 0.429061 | 0.482951 |
| cig_boost_442 | 3-material | 0.309792 | 0.397893 | 0.336111 | 0.064480 | 0.239462 | 0.449750 | 0.503445 |
| cig_boost_blend | 8-specific | 0.257129 | 0.283554 | 0.271279 | 0.057678 | 0.039979 | 0.308331 | 0.394606 |
| cig_boost_blend | 4-material | 0.306513 | 0.372033 | 0.320583 | 0.057678 | 0.132939 | 0.429185 | 0.490082 |
| cig_boost_blend | 3-material | 0.308341 | 0.396540 | 0.329793 | 0.057678 | 0.230519 | 0.450164 | 0.512662 |

## Image777: all seven cigarette GTs, scores, ranks and IoUs

The purpose statement calls all seven localized, but its quoted GT283 IoU0.468 is below the fixed IoU0.50 correctness boundary. No new relaxed criterion is introduced. The highest-IoU cigarette query can differ from the highest-score qualifying localization. The table reports both without a score floor, then coverage at.05 and whether a qualifying query appears in the standard first15 drawn boxes. A below-.05 query is stored, not drawn at the default floor. A box cap15 cannot guarantee all cigarettes are drawn when other queries outrank them; the table gives the actual outcome.

| State | GT | Best-IoU q | Score | Rank | IoU | Highest aligned q | Score | Rank | IoU | Covered @.05 | Drawn first15 |
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
| cig_boost_442 | 278 | 40 | 0.053515 | 28 | 0.878449 | 40 | 0.053515 | 28 | 0.878449 | True | False |
| cig_boost_442 | 279 | 63 | 0.061413 | 21 | 0.755807 | 63 | 0.061413 | 21 | 0.755807 | True | False |
| cig_boost_442 | 280 | 12 | 0.030826 | 89 | 0.872847 | 5 | 0.112413 | 4 | 0.851223 | True | True |
| cig_boost_442 | 281 | 620 | 0.016865 | 231 | 0.554442 | 620 | 0.016865 | 231 | 0.554442 | False | False |
| cig_boost_442 | 282 | 33 | 0.004874 | 760 | 0.556695 | 23 | 0.029292 | 102 | 0.544677 | False | False |
| cig_boost_442 | 283 | 103 | 0.007698 | 569 | 0.698741 | 103 | 0.007698 | 569 | 0.698741 | False | False |
| cig_boost_442 | 284 | 22 | 0.077281 | 10 | 0.783850 | 22 | 0.077281 | 10 | 0.783850 | True | True |
| cig_boost_blend | 278 | 35 | 0.054919 | 28 | 0.877958 | 35 | 0.054919 | 28 | 0.877958 | True | False |
| cig_boost_blend | 279 | 59 | 0.061301 | 19 | 0.762789 | 59 | 0.061301 | 19 | 0.762789 | True | False |
| cig_boost_blend | 280 | 13 | 0.033260 | 71 | 0.859382 | 6 | 0.113612 | 3 | 0.847737 | True | True |
| cig_boost_blend | 281 | 647 | 0.016130 | 237 | 0.550516 | 647 | 0.016130 | 237 | 0.550516 | False | False |
| cig_boost_blend | 282 | 23 | 0.030973 | 83 | 0.572402 | 23 | 0.030973 | 83 | 0.572402 | False | False |
| cig_boost_blend | 283 | 94 | 0.010481 | 395 | 0.718852 | 94 | 0.010481 | 395 | 0.718852 | False | False |
| cig_boost_blend | 284 | 21 | 0.077186 | 9 | 0.780347 | 21 | 0.077186 | 9 | 0.780347 | True | True |

## All24 validation cigarettes (including Image1161 GT496)

Full query identities, ranks, scores, IoUs, retained/top100 flags and source boxes are in the ledger JSON/CSV. The best-IoU query here has no score floor.

| State | Image | GT | Best-IoU q | Score | Rank | IoU | Covered @.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constant-350 | 144 | 55 | 221 | 0.015001 | 161 | 0.058140 | False |
| constant-350 | 144 | 56 | 534 | 0.005372 | 663 | 0.270737 | False |
| constant-350 | 144 | 57 | 253 | 0.026836 | 48 | 0.155108 | False |
| constant-350 | 155 | 65 | 675 | 0.008992 | 695 | 0.016657 | False |
| constant-350 | 165 | 72 | 74 | 0.002587 | 838 | 0.857397 | True |
| constant-350 | 691 | 260 | 49 | 0.034384 | 72 | 0.764249 | False |
| constant-350 | 691 | 261 | 76 | 0.015758 | 191 | 0.874380 | True |
| constant-350 | 777 | 278 | 36 | 0.050408 | 34 | 0.870484 | True |
| constant-350 | 777 | 279 | 67 | 0.060135 | 22 | 0.744656 | True |
| constant-350 | 777 | 280 | 12 | 0.030205 | 85 | 0.871991 | True |
| constant-350 | 777 | 281 | 701 | 0.002279 | 888 | 0.561036 | False |
| constant-350 | 777 | 282 | 38 | 0.005241 | 722 | 0.537811 | False |
| constant-350 | 777 | 283 | 122 | 0.006848 | 604 | 0.689437 | False |
| constant-350 | 777 | 284 | 30 | 0.073511 | 13 | 0.785302 | True |
| constant-350 | 849 | 313 | 101 | 0.014728 | 275 | 0.685091 | True |
| constant-350 | 849 | 314 | 52 | 0.050655 | 54 | 0.523323 | True |
| constant-350 | 849 | 315 | 76 | 0.014345 | 282 | 0.687854 | True |
| constant-350 | 849 | 316 | 4 | 0.090842 | 10 | 0.736738 | True |
| constant-350 | 1161 | 495 | 9 | 0.126352 | 8 | 0.112853 | False |
| constant-350 | 1161 | 496 | 17 | 0.008610 | 414 | 0.644968 | False |
| constant-350 | 1161 | 497 | 7 | 0.150664 | 5 | 0.854203 | True |
| constant-350 | 1161 | 498 | 166 | 0.016887 | 149 | 0.666614 | True |
| constant-350 | 1161 | 499 | 11 | 0.121840 | 10 | 0.685770 | True |
| constant-350 | 1161 | 500 | 3 | 0.218973 | 3 | 0.836320 | True |
| constant-442 | 144 | 55 | 145 | 0.018700 | 94 | 0.053932 | False |
| constant-442 | 144 | 56 | 675 | 0.008003 | 451 | 0.245893 | False |
| constant-442 | 144 | 57 | 223 | 0.023229 | 62 | 0.128860 | False |
| constant-442 | 155 | 65 | 890 | 0.014432 | 347 | 0.019297 | False |
| constant-442 | 165 | 72 | 2 | 0.230083 | 2 | 0.845342 | True |
| constant-442 | 691 | 260 | 42 | 0.042427 | 48 | 0.697205 | False |
| constant-442 | 691 | 261 | 10 | 0.051829 | 35 | 0.866694 | True |
| constant-442 | 777 | 278 | 28 | 0.053679 | 25 | 0.881301 | True |
| constant-442 | 777 | 279 | 173 | 0.012061 | 309 | 0.794829 | True |
| constant-442 | 777 | 280 | 12 | 0.034520 | 62 | 0.851611 | True |
| constant-442 | 777 | 281 | 825 | 0.002462 | 865 | 0.876267 | False |
| constant-442 | 777 | 282 | 21 | 0.017191 | 208 | 0.585264 | True |
| constant-442 | 777 | 283 | 93 | 0.006655 | 584 | 0.769422 | False |
| constant-442 | 777 | 284 | 19 | 0.077522 | 9 | 0.776195 | True |
| constant-442 | 849 | 313 | 93 | 0.016857 | 227 | 0.717695 | True |
| constant-442 | 849 | 314 | 44 | 0.051268 | 49 | 0.562878 | True |
| constant-442 | 849 | 315 | 64 | 0.017185 | 213 | 0.643536 | True |
| constant-442 | 849 | 316 | 6 | 0.096537 | 10 | 0.736828 | True |
| constant-442 | 1161 | 495 | 8 | 0.127751 | 10 | 0.122348 | False |
| constant-442 | 1161 | 496 | 16 | 0.007706 | 445 | 0.705402 | False |
| constant-442 | 1161 | 497 | 7 | 0.154329 | 6 | 0.802659 | True |
| constant-442 | 1161 | 498 | 10 | 0.129574 | 9 | 0.608203 | True |
| constant-442 | 1161 | 499 | 9 | 0.130212 | 8 | 0.692140 | True |
| constant-442 | 1161 | 500 | 3 | 0.221116 | 3 | 0.882274 | True |
| integrated | 144 | 55 | 162 | 0.017086 | 114 | 0.047860 | False |
| integrated | 144 | 56 | 621 | 0.008818 | 399 | 0.254589 | False |
| integrated | 144 | 57 | 229 | 0.021677 | 66 | 0.131897 | False |
| integrated | 155 | 65 | 563 | 0.014248 | 370 | 0.012719 | False |
| integrated | 165 | 72 | 2 | 0.228714 | 2 | 0.840826 | True |
| integrated | 691 | 260 | 44 | 0.040375 | 51 | 0.728688 | False |
| integrated | 691 | 261 | 9 | 0.053769 | 35 | 0.868913 | True |
| integrated | 777 | 278 | 32 | 0.052704 | 29 | 0.877367 | True |
| integrated | 777 | 279 | 189 | 0.013614 | 273 | 0.806409 | True |
| integrated | 777 | 280 | 14 | 0.032971 | 69 | 0.858967 | True |
| integrated | 777 | 281 | 795 | 0.002210 | 883 | 0.793612 | False |
| integrated | 777 | 282 | 23 | 0.029291 | 89 | 0.576940 | True |
| integrated | 777 | 283 | 97 | 0.008332 | 474 | 0.730058 | False |
| integrated | 777 | 284 | 20 | 0.076084 | 9 | 0.780633 | True |
| integrated | 849 | 313 | 96 | 0.016190 | 233 | 0.730609 | True |
| integrated | 849 | 314 | 43 | 0.051643 | 46 | 0.545725 | True |
| integrated | 849 | 315 | 65 | 0.016816 | 222 | 0.670322 | True |
| integrated | 849 | 316 | 6 | 0.094164 | 11 | 0.737827 | True |
| integrated | 1161 | 495 | 10 | 0.128324 | 10 | 0.120393 | False |
| integrated | 1161 | 496 | 16 | 0.008127 | 430 | 0.703405 | False |
| integrated | 1161 | 497 | 105 | 0.004190 | 732 | 0.844972 | True |
| integrated | 1161 | 498 | 12 | 0.128749 | 8 | 0.602466 | True |
| integrated | 1161 | 499 | 11 | 0.128568 | 9 | 0.686664 | True |
| integrated | 1161 | 500 | 3 | 0.220908 | 3 | 0.872145 | True |
| cig_boost_442 | 144 | 55 | 537 | 0.012232 | 248 | 0.088234 | False |
| cig_boost_442 | 144 | 56 | 581 | 0.010177 | 355 | 0.243104 | False |
| cig_boost_442 | 144 | 57 | 238 | 0.029965 | 39 | 0.188497 | False |
| cig_boost_442 | 155 | 65 | 515 | 0.015558 | 349 | 0.013311 | False |
| cig_boost_442 | 165 | 72 | 2 | 0.231929 | 2 | 0.826688 | True |
| cig_boost_442 | 691 | 260 | 44 | 0.034887 | 70 | 0.709875 | False |
| cig_boost_442 | 691 | 261 | 76 | 0.015320 | 194 | 0.836213 | True |
| cig_boost_442 | 777 | 278 | 40 | 0.053515 | 28 | 0.878449 | True |
| cig_boost_442 | 777 | 279 | 63 | 0.061413 | 21 | 0.755807 | True |
| cig_boost_442 | 777 | 280 | 12 | 0.030826 | 89 | 0.872847 | True |
| cig_boost_442 | 777 | 281 | 620 | 0.016865 | 231 | 0.554442 | False |
| cig_boost_442 | 777 | 282 | 33 | 0.004874 | 760 | 0.556695 | False |
| cig_boost_442 | 777 | 283 | 103 | 0.007698 | 569 | 0.698741 | False |
| cig_boost_442 | 777 | 284 | 22 | 0.077281 | 10 | 0.783850 | True |
| cig_boost_442 | 849 | 313 | 106 | 0.015224 | 268 | 0.717575 | True |
| cig_boost_442 | 849 | 314 | 51 | 0.052148 | 54 | 0.517797 | True |
| cig_boost_442 | 849 | 315 | 67 | 0.015992 | 250 | 0.689662 | True |
| cig_boost_442 | 849 | 316 | 5 | 0.096562 | 11 | 0.727628 | True |
| cig_boost_442 | 1161 | 495 | 11 | 0.133613 | 8 | 0.116484 | False |
| cig_boost_442 | 1161 | 496 | 17 | 0.010880 | 312 | 0.657179 | False |
| cig_boost_442 | 1161 | 497 | 8 | 0.159580 | 5 | 0.849500 | True |
| cig_boost_442 | 1161 | 498 | 149 | 0.017729 | 148 | 0.671160 | True |
| cig_boost_442 | 1161 | 499 | 10 | 0.131743 | 9 | 0.687793 | True |
| cig_boost_442 | 1161 | 500 | 3 | 0.225457 | 3 | 0.850109 | True |
| cig_boost_blend | 144 | 55 | 500 | 0.017259 | 112 | 0.065184 | False |
| cig_boost_blend | 144 | 56 | 611 | 0.009489 | 372 | 0.264778 | False |
| cig_boost_blend | 144 | 57 | 230 | 0.023605 | 57 | 0.118633 | False |
| cig_boost_blend | 155 | 65 | 550 | 0.014729 | 360 | 0.012867 | False |
| cig_boost_blend | 165 | 72 | 2 | 0.229906 | 2 | 0.835492 | True |
| cig_boost_blend | 691 | 260 | 44 | 0.032659 | 68 | 0.754644 | False |
| cig_boost_blend | 691 | 261 | 8 | 0.051468 | 34 | 0.868612 | True |
| cig_boost_blend | 777 | 278 | 35 | 0.054919 | 28 | 0.877958 | True |
| cig_boost_blend | 777 | 279 | 59 | 0.061301 | 19 | 0.762789 | True |
| cig_boost_blend | 777 | 280 | 13 | 0.033260 | 71 | 0.859382 | True |
| cig_boost_blend | 777 | 281 | 647 | 0.016130 | 237 | 0.550516 | False |
| cig_boost_blend | 777 | 282 | 23 | 0.030973 | 83 | 0.572402 | False |
| cig_boost_blend | 777 | 283 | 94 | 0.010481 | 395 | 0.718852 | False |
| cig_boost_blend | 777 | 284 | 21 | 0.077186 | 9 | 0.780347 | True |
| cig_boost_blend | 849 | 313 | 100 | 0.016302 | 238 | 0.735897 | True |
| cig_boost_blend | 849 | 314 | 45 | 0.051291 | 52 | 0.537745 | True |
| cig_boost_blend | 849 | 315 | 64 | 0.016940 | 228 | 0.641505 | True |
| cig_boost_blend | 849 | 316 | 5 | 0.096654 | 10 | 0.727483 | True |
| cig_boost_blend | 1161 | 495 | 10 | 0.131524 | 9 | 0.121112 | False |
| cig_boost_blend | 1161 | 496 | 16 | 0.009542 | 368 | 0.700033 | False |
| cig_boost_blend | 1161 | 497 | 7 | 0.158128 | 5 | 0.822716 | True |
| cig_boost_blend | 1161 | 498 | 12 | 0.131589 | 8 | 0.600989 | True |
| cig_boost_blend | 1161 | 499 | 11 | 0.130460 | 10 | 0.688302 | True |
| cig_boost_blend | 1161 | 500 | 3 | 0.223740 | 4 | 0.868076 | True |

## Memory and verification

Training plus endpoint-validation peak: allocated 3.550 GiB, reserved 4.023 GiB, target 6.00 GiB. This is a whole-process peak, not a per-checkpoint measurement. Blend worker peak is separately recorded in blend-record.json. Frozen-parameter hashes remain unchanged. All60 prior plus2 new bundles were verified with their contexts/file hashes; complete new optimizer counters400/442 and saved RNG were checked on CPU. Of 2207 prior protected files, 2206 rehashed unchanged; the sole exception is a separately written historical GUI display record, detailed below. Existing11 replay choices remain exactly preserved when the new choice is added.

The conservative controller initially stopped after all92 updates/94 validations because `saved-review-085e285ada21461ab52f75003ad1b471/analysis.json` changed concurrently. Reconstructing its pending form from the current closed form reproduces the exact pre-run SHA-256: the ONLY changes are display_status pending->closed and addition of source_verified_photo_reads=16. All other JSON fields are cryptographically identical; all model/data/checkpoint artifacts are unchanged. The stopped controller record and display-transition-reconciliation.json are retained, and protected_after.unchanged remains false to preserve the observation. The record is completed only after this separate display-lifecycle reconciliation; the historical file was never rewritten and no update or validation was repeated. The replay loader independently repeats the reconstruction check.

Guarded full discoverable suite: 273 tests; 257 passed, 16 documented skips, zero failures/errors. Skips are15 training-execution fixtures prohibited during software verification and one Windows symlink-privilege fixture. Native Windows Tk: 2 saved-choice cases, standard/per-object at Image777, initial box cap15, all47 image IDs, official/material/polymer selectors and all five diagnostic NMS options, navigation/GT selection/zoom/pan. Replay blocks model loading/forward, CUDA initialization, backward/optimizer updates, checkpoint-state reads and annotated exports; saved JSON hashes unchanged. Automated runtime checks do not constitute human visual acceptance, which remains pending.

## Exact PowerShell replay commands

Run from the project root and close each window before launching the next. These commands explicitly set Box top K15. Default taxonomy remains official eight classes, threshold.05 and raw queries. Select **4 material/hazard categories (mapped)** in the toolbar to inspect material agreement; **3 material/hazard categories (polymer grouped)** and all five diagnostic NMS modes are also available. Use **all900 (diagnostic)** for the coverage pool; NMS alters which candidates remain and can hide true objects. All47 validation images are available from the Image777 start. The three prediction panels are the old D061 integrated state, the new endpoint, and the new50/50 blend.

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
$review = 'runs/training/d060-saved-review/review-manifest.json'
$choice = '42 / 1e-5 @ 384/576 (cigarette-boost vs integrated)'
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --saved-review $review --run-choice $choice --gui --image-id 777 --box-top-k 15
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --saved-review $review --run-choice $choice --gui --per-object --image-id 777 --box-top-k 15
```
