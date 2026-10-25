# D-065: saved-prediction material/hazard hierarchy

Owner authorization: the latest attached D-065 replaces the earlier top-backbone-unfreezing request. This execution is a reversible diagnostic mapping of saved predictions, not training or evidence of new visual material recognition. No model was loaded or executed, CUDA initialized, backward pass or optimizer update performed in this hierarchy task. All 60 checkpoints, the six core training SOURCES and `config/taco_taxonomy.v1.json` remain byte-for-byte unchanged. Official eight-class `bantai.taco-evaluation.v2` remains no-NMS, and all seven saved official metric dictionaries reproduce exactly on CPU. No test.json or test photos were opened; no photos were decoded by the metric audit. Native replay reads validation photos only. No commit, push, merge or checkpoint promotion.

Historical interruption, separate from this zero-model task: the earlier D-065 vision-unfreeze attempt stopped after 10 optimizer updates / 11 training forwards, with no validation forwards or new checkpoint bundles, when its stage-3 query-weight gradient was zero. Its private stopped record is retained at `runs/training/taco-vision-unfreeze-9452eb4dc47044a9a76cb232fe921b45/record.json`; it was not resumed. It supplies none of the results below.

Measured integrated-blend result: raw eight-class AP 0.265563 / coverage 38 of 75; raw four-material AP 0.309859 / 55 of 75; raw three-material AP 0.311858 / 58 of 75. These are different category tasks. Four-material class-aware IoU yields AP 0.318226 / 54 of 75; adding containment yields AP 0.313529 / 49 of 75. At .05, four-material containment changes strict TP/FP/FN from raw 55/1097/20 to 49/597/26: precision rises from 0.047743 to 0.075851 while recall falls from 0.733333 to 0.653333. Cigarettes fall from 15/24 to 14/24, with Image 777 remaining 5/7. GT259 has no qualifying wrapper query in any of the seven eight-class raw states, but material agreement is recovered in the mapped raw views. The four-material blend recovers nine of the fixed ten regressions (the plastic objects); the polymer scheme also accepts the remaining Styrofoam/plastic confusion and covers all ten. This relabels class agreement; it does not repair subtype predictions.

## Diagnostic architecture and labels

Specific predicted class -> deterministic category mapping -> optional same-effective-category duplicate/containment suppression -> matching/display. Original query IDs, raw phrases, ranks, scores, boxes and specific IDs/names remain available. Each mapped row carries `specific_class`, `hazard_flag`, specific `material`, mapped name and label, e.g. `plastic_waste [plastic_bottle]`. Cigarette's hazard flag is a taxonomy rule, not a new pixel-level chemical/hazard measurement. In the polymer scheme, Styrofoam retains its specific `polystyrene_foam` material metadata while belonging to `plastic_polymer_waste`.

| Specific class | 4-material category | 3-material category | Material | Hazard |
| --- | --- | --- | --- | --- |
| cigarette | hazardous_cigarette | hazardous_cigarette | cellulose_acetate_filter | True |
| plastic_bottle | plastic_waste | plastic_polymer_waste | plastic | False |
| plastic_bag | plastic_waste | plastic_polymer_waste | plastic | False |
| plastic_wrapper | plastic_waste | plastic_polymer_waste | plastic | False |
| plastic_film | plastic_waste | plastic_polymer_waste | plastic | False |
| drink_can | metal_waste | metal_waste | metal | False |
| plastic_cup | plastic_waste | plastic_polymer_waste | plastic | False |
| styrofoam_piece | styrofoam_waste | plastic_polymer_waste | polystyrene_foam | False |

NMS has five selectable modes: historical raw, class-agnostic IoU .50, class-aware IoU .50, agnostic IoU .50 + containment .80, and class-aware IoU .50 + containment .80. Greedy order is score descending then numeric query ID ascending. Only already-kept queries suppress: IoU >= .50 or, for containment modes, intersection/minimum area >= .80. Empty-area boxes have IoMin zero. Class-aware modes compare active mapped IDs after remapping. Suppressed rows stay in saved JSON. Containment can hide a distinct object inside a larger box; agnostic containment can also hide cigarettes inside background/plastic boxes. Defaults remain official eight classes and raw queries.

The GUI **Taxonomy view** selector applies to both standard and per-object views. Mapped tables retain subtype and hazard flag. Standard `best_GT` chooses an annotation in the same active mapped category; historical official view retains its original best-overlap behavior. Per-object `best_same`, agreement, coverage and green/magenta colors use the active category; green means annotation-relative material agreement, not a correct original subtype. Source/threshold selection precedes NMS and standard first-K drawing; per-object drawing remains GT-prioritized. Switching back restores the historical view from unchanged raw rows.

## Evaluation method and comparability

Seven states, three taxonomies, three audit suppression settings = 63 metric cases. Audit settings are raw, class-aware IoU .50, and class-aware IoU .50 + containment .80 (both agnostic options are additionally checked in the GUI). AP uses all valid 900 saved winning-class queries, remapped before optional image-wide NMS, then a global top100 per image, with no score floor. COCO uses the same original-pixel GT areas, IoU .50:.05:.95 and maxDets 1/10/100. Mapped calculations use their actual 4/3 category lists in an isolated CPU helper; the official evaluator is untouched. Post-NMS top100 may include lower-ranked queries beyond the original official pool. These are diagnostic metrics; mapped AP changes the task and class averaging and is not evidence that eight-class detection improved.

Coverage is non-exclusive: any same-effective-category candidate at score >= .05 and IoU >= .50. Denominators remain 75 GTs, 24 cigarettes and 7 Image-777 cigarettes. Strict matching uses all retained-after-NMS valid queries, no top-K cap, inclusive score thresholds .05/.07/.10/.12, score-descending/query-ID-ascending greedy assignment to the highest-IoU unmatched same-category GT (GT-ID ascending ties). Each query and GT can contribute at most one TP. FP means unmatched relative to these annotations, not independently confirmed trash/background. P=TP/(TP+FP), R=TP/(TP+FN), F1=2PR/(P+R), zero when undefined. COCO AP and this IoU-.50 matching are distinct calculations.

Private evidence: [material-hierarchy-6574f73514444b9d8207615e453432b3](../runs/training/material-hierarchy-6574f73514444b9d8207615e453432b3/record.json), [full summary](../runs/training/material-hierarchy-6574f73514444b9d8207615e453432b3/hierarchy-summary.json), [metrics CSV](../runs/training/material-hierarchy-6574f73514444b9d8207615e453432b3/metrics-coverage.csv), [strict matching CSV](../runs/training/material-hierarchy-6574f73514444b9d8207615e453432b3/strict-matching.csv), [per-image one-to-one assignments](../runs/training/material-hierarchy-6574f73514444b9d8207615e453432b3/matching-ledger.json). Per-GT outcomes and suppression proofs are separate JSONs for every case.

## Recomputed metrics and candidate coverage

Every value below is diagnostic, including the raw eight-class rows independently reproduced from saved official predictions. `Reg` counts the fixed ten D-058 losses; it does not select a new regression set.

### step0

| Taxonomy | NMS | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | GT /75 | Cig /24 | 777 cig /7 | Reg /10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | raw | 0.147847 | 0.179470 | 0.165594 | 0.000000 | 0.007261 | 0.172653 | 0.426109 | 29 | 3 | 0 | 7 |
| 8-specific | aware IoU | 0.151613 | 0.186414 | 0.169392 | 0.024752 | 0.007261 | 0.177113 | 0.396893 | 28 | 3 | 0 | 7 |
| 8-specific | aware+containment | 0.145398 | 0.180046 | 0.161610 | 0.024752 | 0.007261 | 0.167982 | 0.324642 | 24 | 3 | 0 | 5 |
| 4-material | raw | 0.091754 | 0.140007 | 0.109086 | 0.000000 | 0.085931 | 0.129830 | 0.414709 | 46 | 3 | 0 | 10 |
| 4-material | aware IoU | 0.096211 | 0.151371 | 0.113697 | 0.024752 | 0.077008 | 0.135675 | 0.400687 | 45 | 3 | 0 | 10 |
| 4-material | aware+containment | 0.066513 | 0.112054 | 0.077241 | 0.024752 | 0.044004 | 0.089676 | 0.226712 | 24 | 3 | 0 | 5 |
| 3-material | raw | 0.100939 | 0.160597 | 0.118245 | 0.000000 | 0.125122 | 0.156312 | 0.328878 | 47 | 3 | 0 | 10 |
| 3-material | aware IoU | 0.104628 | 0.172220 | 0.122436 | 0.024752 | 0.117987 | 0.160285 | 0.306277 | 46 | 3 | 0 | 10 |
| 3-material | aware+containment | 0.066615 | 0.116640 | 0.081134 | 0.024752 | 0.056106 | 0.095634 | 0.169913 | 21 | 3 | 0 | 4 |

### step221

| Taxonomy | NMS | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | GT /75 | Cig /24 | 777 cig /7 | Reg /10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | raw | 0.249891 | 0.286853 | 0.264765 | 0.060891 | 0.033495 | 0.303687 | 0.488141 | 34 | 8 | 1 | 10 |
| 8-specific | aware IoU | 0.251114 | 0.289699 | 0.265512 | 0.060891 | 0.026804 | 0.305672 | 0.478446 | 34 | 8 | 1 | 10 |
| 8-specific | aware+containment | 0.247988 | 0.286621 | 0.262522 | 0.056436 | 0.021731 | 0.301347 | 0.447662 | 32 | 8 | 1 | 9 |
| 4-material | raw | 0.221300 | 0.271487 | 0.223513 | 0.060891 | 0.121296 | 0.322762 | 0.412466 | 47 | 8 | 1 | 10 |
| 4-material | aware IoU | 0.235707 | 0.296522 | 0.238776 | 0.060891 | 0.113547 | 0.343506 | 0.399485 | 46 | 8 | 1 | 10 |
| 4-material | aware+containment | 0.222599 | 0.280377 | 0.224576 | 0.056436 | 0.104880 | 0.322711 | 0.322241 | 40 | 8 | 1 | 9 |
| 3-material | raw | 0.181161 | 0.253119 | 0.182446 | 0.060891 | 0.204662 | 0.267501 | 0.356151 | 49 | 8 | 1 | 10 |
| 3-material | aware IoU | 0.201471 | 0.287372 | 0.202931 | 0.060891 | 0.193369 | 0.298942 | 0.335949 | 48 | 8 | 1 | 10 |
| 3-material | aware+containment | 0.194568 | 0.276856 | 0.195718 | 0.056436 | 0.135041 | 0.290638 | 0.304383 | 43 | 8 | 1 | 10 |

### step350

| Taxonomy | NMS | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | GT /75 | Cig /24 | 777 cig /7 | Reg /10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | raw | 0.262337 | 0.288625 | 0.278353 | 0.065773 | 0.033448 | 0.315487 | 0.428890 | 34 | 14 | 4 | 1 |
| 8-specific | aware IoU | 0.264278 | 0.291515 | 0.279465 | 0.064851 | 0.035750 | 0.317831 | 0.417520 | 34 | 14 | 4 | 1 |
| 8-specific | aware+containment | 0.263979 | 0.290119 | 0.279965 | 0.063366 | 0.031784 | 0.317512 | 0.404259 | 31 | 13 | 4 | 1 |
| 4-material | raw | 0.300493 | 0.361440 | 0.318400 | 0.065773 | 0.136507 | 0.419243 | 0.478464 | 54 | 14 | 4 | 9 |
| 4-material | aware IoU | 0.312422 | 0.383584 | 0.328950 | 0.064851 | 0.133654 | 0.436975 | 0.452518 | 53 | 14 | 4 | 9 |
| 4-material | aware+containment | 0.309328 | 0.379896 | 0.324913 | 0.063366 | 0.133610 | 0.431532 | 0.425355 | 49 | 13 | 4 | 9 |
| 3-material | raw | 0.303410 | 0.385760 | 0.330505 | 0.065773 | 0.231992 | 0.440741 | 0.497385 | 57 | 14 | 4 | 10 |
| 3-material | aware IoU | 0.320051 | 0.414440 | 0.346385 | 0.064851 | 0.227991 | 0.467365 | 0.478066 | 56 | 14 | 4 | 10 |
| 3-material | aware+containment | 0.314597 | 0.407098 | 0.338240 | 0.063366 | 0.173541 | 0.461626 | 0.444859 | 52 | 13 | 4 | 10 |

### step442

| Taxonomy | NMS | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | GT /75 | Cig /24 | 777 cig /7 | Reg /10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | raw | 0.170508 | 0.196086 | 0.179972 | 0.075112 | 0.048109 | 0.206941 | 0.370118 | 35 | 15 | 5 | 0 |
| 8-specific | aware IoU | 0.171547 | 0.198829 | 0.179815 | 0.075597 | 0.041596 | 0.208649 | 0.348091 | 35 | 15 | 5 | 0 |
| 8-specific | aware+containment | 0.170884 | 0.196494 | 0.180236 | 0.073515 | 0.033351 | 0.208031 | 0.335351 | 31 | 14 | 5 | 0 |
| 4-material | raw | 0.245494 | 0.307580 | 0.251203 | 0.075112 | 0.139148 | 0.344764 | 0.419661 | 53 | 15 | 5 | 9 |
| 4-material | aware IoU | 0.253183 | 0.321813 | 0.256585 | 0.075597 | 0.133164 | 0.356322 | 0.392273 | 52 | 15 | 5 | 9 |
| 4-material | aware+containment | 0.248451 | 0.314558 | 0.250185 | 0.073515 | 0.128626 | 0.349039 | 0.361664 | 47 | 14 | 5 | 9 |
| 3-material | raw | 0.225596 | 0.308955 | 0.235367 | 0.075112 | 0.239391 | 0.321230 | 0.419156 | 56 | 15 | 5 | 10 |
| 3-material | aware IoU | 0.236705 | 0.329312 | 0.243931 | 0.075597 | 0.230632 | 0.338241 | 0.397817 | 55 | 15 | 5 | 10 |
| 3-material | aware+containment | 0.232524 | 0.323531 | 0.237878 | 0.073515 | 0.224622 | 0.330440 | 0.365242 | 51 | 14 | 5 | 10 |

### blend_350_442_2575

| Taxonomy | NMS | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | GT /75 | Cig /24 | 777 cig /7 | Reg /10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | raw | 0.265563 | 0.292911 | 0.280572 | 0.071254 | 0.045737 | 0.316106 | 0.403085 | 38 | 15 | 5 | 1 |
| 8-specific | aware IoU | 0.271819 | 0.300309 | 0.285942 | 0.070770 | 0.042460 | 0.323961 | 0.447845 | 38 | 15 | 5 | 1 |
| 8-specific | aware+containment | 0.271327 | 0.298607 | 0.286629 | 0.069554 | 0.033153 | 0.323431 | 0.428374 | 34 | 14 | 5 | 1 |
| 4-material | raw | 0.309859 | 0.376847 | 0.325922 | 0.071254 | 0.137069 | 0.431522 | 0.486877 | 55 | 15 | 5 | 9 |
| 4-material | aware IoU | 0.318226 | 0.392823 | 0.332585 | 0.070770 | 0.133884 | 0.443791 | 0.461012 | 54 | 15 | 5 | 9 |
| 4-material | aware+containment | 0.313529 | 0.385907 | 0.327215 | 0.069554 | 0.128363 | 0.436743 | 0.427839 | 49 | 14 | 5 | 9 |
| 3-material | raw | 0.311858 | 0.402402 | 0.335028 | 0.071254 | 0.236609 | 0.451775 | 0.508874 | 58 | 15 | 5 | 10 |
| 3-material | aware IoU | 0.323889 | 0.424671 | 0.346181 | 0.070770 | 0.231720 | 0.470310 | 0.489177 | 57 | 15 | 5 | 10 |
| 3-material | aware+containment | 0.319689 | 0.419253 | 0.341115 | 0.069554 | 0.224227 | 0.462727 | 0.453571 | 53 | 14 | 5 | 10 |

### cb_step350

| Taxonomy | NMS | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | GT /75 | Cig /24 | 777 cig /7 | Reg /10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | raw | 0.258631 | 0.295110 | 0.279307 | 0.063749 | 0.042538 | 0.309946 | 0.463523 | 37 | 13 | 4 | 4 |
| 8-specific | aware IoU | 0.260191 | 0.297625 | 0.280524 | 0.062511 | 0.037135 | 0.312084 | 0.446384 | 37 | 13 | 4 | 4 |
| 8-specific | aware+containment | 0.260280 | 0.296880 | 0.281470 | 0.060891 | 0.028542 | 0.312384 | 0.437209 | 33 | 12 | 4 | 4 |
| 4-material | raw | 0.294172 | 0.358102 | 0.319390 | 0.063749 | 0.140196 | 0.413102 | 0.466605 | 53 | 13 | 4 | 9 |
| 4-material | aware IoU | 0.307233 | 0.381549 | 0.334047 | 0.062511 | 0.139153 | 0.431690 | 0.448191 | 52 | 13 | 4 | 9 |
| 4-material | aware+containment | 0.301152 | 0.372852 | 0.325981 | 0.060891 | 0.137867 | 0.422486 | 0.412695 | 47 | 12 | 4 | 9 |
| 3-material | raw | 0.292763 | 0.381693 | 0.326912 | 0.063749 | 0.238078 | 0.427197 | 0.481475 | 56 | 13 | 4 | 10 |
| 3-material | aware IoU | 0.311523 | 0.414566 | 0.345693 | 0.062511 | 0.236571 | 0.456376 | 0.471248 | 55 | 13 | 4 | 10 |
| 3-material | aware+containment | 0.302297 | 0.401798 | 0.331917 | 0.060891 | 0.177482 | 0.445245 | 0.429582 | 50 | 12 | 4 | 10 |

### cb_step442

| Taxonomy | NMS | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | GT /75 | Cig /24 | 777 cig /7 | Reg /10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | raw | 0.242594 | 0.268843 | 0.255444 | 0.065016 | 0.050450 | 0.288197 | 0.377770 | 36 | 14 | 4 | 2 |
| 8-specific | aware IoU | 0.246654 | 0.273638 | 0.258874 | 0.064252 | 0.047845 | 0.293420 | 0.422610 | 36 | 14 | 4 | 2 |
| 8-specific | aware+containment | 0.246426 | 0.272741 | 0.260069 | 0.062942 | 0.035371 | 0.293472 | 0.409869 | 32 | 13 | 4 | 2 |
| 4-material | raw | 0.270574 | 0.330970 | 0.283113 | 0.065016 | 0.138870 | 0.375982 | 0.447202 | 53 | 14 | 4 | 9 |
| 4-material | aware IoU | 0.281306 | 0.350703 | 0.292871 | 0.064252 | 0.136815 | 0.391475 | 0.420856 | 52 | 14 | 4 | 9 |
| 4-material | aware+containment | 0.277216 | 0.344108 | 0.290017 | 0.062942 | 0.131320 | 0.385789 | 0.389606 | 47 | 13 | 4 | 9 |
| 3-material | raw | 0.261300 | 0.343801 | 0.278689 | 0.065016 | 0.235582 | 0.371662 | 0.455411 | 56 | 14 | 4 | 10 |
| 3-material | aware IoU | 0.277730 | 0.374410 | 0.293326 | 0.064252 | 0.232534 | 0.397702 | 0.435462 | 55 | 14 | 4 | 10 |
| 3-material | aware+containment | 0.274145 | 0.369267 | 0.289673 | 0.062942 | 0.225197 | 0.391586 | 0.401371 | 51 | 13 | 4 | 10 |

## One-to-one matching at all four score floors

These TP counts can differ from non-exclusive coverage. All queries above the floor are counted; duplicate and low-quality candidates remain FP unless NMS suppresses them.

### step0

| Taxonomy | NMS | Score floor | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | raw | 0.05 | 29 | 4865 | 46 | 0.005926 | 0.386667 | 0.011672 |
| 8-specific | raw | 0.07 | 29 | 2983 | 46 | 0.009628 | 0.386667 | 0.018788 |
| 8-specific | raw | 0.1 | 27 | 1740 | 48 | 0.015280 | 0.360000 | 0.029316 |
| 8-specific | raw | 0.12 | 27 | 1271 | 48 | 0.020801 | 0.360000 | 0.039330 |
| 8-specific | aware IoU | 0.05 | 28 | 2122 | 47 | 0.013023 | 0.373333 | 0.025169 |
| 8-specific | aware IoU | 0.07 | 28 | 1371 | 47 | 0.020014 | 0.373333 | 0.037992 |
| 8-specific | aware IoU | 0.1 | 27 | 856 | 48 | 0.030578 | 0.360000 | 0.056367 |
| 8-specific | aware IoU | 0.12 | 27 | 669 | 48 | 0.038793 | 0.360000 | 0.070039 |
| 8-specific | aware+containment | 0.05 | 24 | 1002 | 51 | 0.023392 | 0.320000 | 0.043597 |
| 8-specific | aware+containment | 0.07 | 24 | 718 | 51 | 0.032345 | 0.320000 | 0.058752 |
| 8-specific | aware+containment | 0.1 | 23 | 487 | 52 | 0.045098 | 0.306667 | 0.078632 |
| 8-specific | aware+containment | 0.12 | 22 | 416 | 53 | 0.050228 | 0.293333 | 0.085770 |
| 4-material | raw | 0.05 | 46 | 4848 | 29 | 0.009399 | 0.613333 | 0.018515 |
| 4-material | raw | 0.07 | 44 | 2968 | 31 | 0.014608 | 0.586667 | 0.028507 |
| 4-material | raw | 0.1 | 43 | 1724 | 32 | 0.024335 | 0.573333 | 0.046688 |
| 4-material | raw | 0.12 | 43 | 1255 | 32 | 0.033128 | 0.573333 | 0.062637 |
| 4-material | aware IoU | 0.05 | 45 | 1709 | 30 | 0.025656 | 0.600000 | 0.049207 |
| 4-material | aware IoU | 0.07 | 43 | 1059 | 32 | 0.039020 | 0.573333 | 0.073067 |
| 4-material | aware IoU | 0.1 | 42 | 634 | 33 | 0.062130 | 0.560000 | 0.111851 |
| 4-material | aware IoU | 0.12 | 42 | 485 | 33 | 0.079696 | 0.560000 | 0.139535 |
| 4-material | aware+containment | 0.05 | 24 | 464 | 51 | 0.049180 | 0.320000 | 0.085258 |
| 4-material | aware+containment | 0.07 | 24 | 327 | 51 | 0.068376 | 0.320000 | 0.112676 |
| 4-material | aware+containment | 0.1 | 23 | 212 | 52 | 0.097872 | 0.306667 | 0.148387 |
| 4-material | aware+containment | 0.12 | 23 | 185 | 52 | 0.110577 | 0.306667 | 0.162544 |
| 3-material | raw | 0.05 | 47 | 4847 | 28 | 0.009604 | 0.626667 | 0.018917 |
| 3-material | raw | 0.07 | 47 | 2965 | 28 | 0.015604 | 0.626667 | 0.030450 |
| 3-material | raw | 0.1 | 45 | 1722 | 30 | 0.025467 | 0.600000 | 0.048860 |
| 3-material | raw | 0.12 | 44 | 1254 | 31 | 0.033898 | 0.586667 | 0.064093 |
| 3-material | aware IoU | 0.05 | 46 | 1549 | 29 | 0.028840 | 0.613333 | 0.055090 |
| 3-material | aware IoU | 0.07 | 46 | 949 | 29 | 0.046231 | 0.613333 | 0.085981 |
| 3-material | aware IoU | 0.1 | 44 | 551 | 31 | 0.073950 | 0.586667 | 0.131343 |
| 3-material | aware IoU | 0.12 | 43 | 414 | 32 | 0.094092 | 0.573333 | 0.161654 |
| 3-material | aware+containment | 0.05 | 21 | 317 | 54 | 0.062130 | 0.280000 | 0.101695 |
| 3-material | aware+containment | 0.07 | 21 | 212 | 54 | 0.090129 | 0.280000 | 0.136364 |
| 3-material | aware+containment | 0.1 | 20 | 137 | 55 | 0.127389 | 0.266667 | 0.172414 |
| 3-material | aware+containment | 0.12 | 20 | 118 | 55 | 0.144928 | 0.266667 | 0.187793 |

### step221

| Taxonomy | NMS | Score floor | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | raw | 0.05 | 34 | 1033 | 41 | 0.031865 | 0.453333 | 0.059545 |
| 8-specific | raw | 0.07 | 32 | 585 | 43 | 0.051864 | 0.426667 | 0.092486 |
| 8-specific | raw | 0.1 | 26 | 296 | 49 | 0.080745 | 0.346667 | 0.130982 |
| 8-specific | raw | 0.12 | 24 | 202 | 51 | 0.106195 | 0.320000 | 0.159468 |
| 8-specific | aware IoU | 0.05 | 34 | 757 | 41 | 0.042984 | 0.453333 | 0.078522 |
| 8-specific | aware IoU | 0.07 | 32 | 449 | 43 | 0.066528 | 0.426667 | 0.115108 |
| 8-specific | aware IoU | 0.1 | 26 | 242 | 49 | 0.097015 | 0.346667 | 0.151603 |
| 8-specific | aware IoU | 0.12 | 24 | 167 | 51 | 0.125654 | 0.320000 | 0.180451 |
| 8-specific | aware+containment | 0.05 | 32 | 555 | 43 | 0.054514 | 0.426667 | 0.096677 |
| 8-specific | aware+containment | 0.07 | 30 | 353 | 45 | 0.078329 | 0.400000 | 0.131004 |
| 8-specific | aware+containment | 0.1 | 25 | 200 | 50 | 0.111111 | 0.333333 | 0.166667 |
| 8-specific | aware+containment | 0.12 | 23 | 144 | 52 | 0.137725 | 0.306667 | 0.190083 |
| 4-material | raw | 0.05 | 47 | 1020 | 28 | 0.044049 | 0.626667 | 0.082312 |
| 4-material | raw | 0.07 | 46 | 571 | 29 | 0.074554 | 0.613333 | 0.132948 |
| 4-material | raw | 0.1 | 42 | 280 | 33 | 0.130435 | 0.560000 | 0.211587 |
| 4-material | raw | 0.12 | 38 | 188 | 37 | 0.168142 | 0.506667 | 0.252492 |
| 4-material | aware IoU | 0.05 | 46 | 615 | 29 | 0.069592 | 0.613333 | 0.125000 |
| 4-material | aware IoU | 0.07 | 45 | 334 | 30 | 0.118734 | 0.600000 | 0.198238 |
| 4-material | aware IoU | 0.1 | 42 | 152 | 33 | 0.216495 | 0.560000 | 0.312268 |
| 4-material | aware IoU | 0.12 | 38 | 97 | 37 | 0.281481 | 0.506667 | 0.361905 |
| 4-material | aware+containment | 0.05 | 40 | 355 | 35 | 0.101266 | 0.533333 | 0.170213 |
| 4-material | aware+containment | 0.07 | 39 | 200 | 36 | 0.163180 | 0.520000 | 0.248408 |
| 4-material | aware+containment | 0.1 | 36 | 95 | 39 | 0.274809 | 0.480000 | 0.349515 |
| 4-material | aware+containment | 0.12 | 34 | 67 | 41 | 0.336634 | 0.453333 | 0.386364 |
| 3-material | raw | 0.05 | 49 | 1018 | 26 | 0.045923 | 0.653333 | 0.085814 |
| 3-material | raw | 0.07 | 48 | 569 | 27 | 0.077796 | 0.640000 | 0.138728 |
| 3-material | raw | 0.1 | 44 | 278 | 31 | 0.136646 | 0.586667 | 0.221662 |
| 3-material | raw | 0.12 | 40 | 186 | 35 | 0.176991 | 0.533333 | 0.265781 |
| 3-material | aware IoU | 0.05 | 48 | 582 | 27 | 0.076190 | 0.640000 | 0.136170 |
| 3-material | aware IoU | 0.07 | 47 | 318 | 28 | 0.128767 | 0.626667 | 0.213636 |
| 3-material | aware IoU | 0.1 | 44 | 143 | 31 | 0.235294 | 0.586667 | 0.335878 |
| 3-material | aware IoU | 0.12 | 40 | 89 | 35 | 0.310078 | 0.533333 | 0.392157 |
| 3-material | aware+containment | 0.05 | 43 | 323 | 32 | 0.117486 | 0.573333 | 0.195011 |
| 3-material | aware+containment | 0.07 | 42 | 176 | 33 | 0.192661 | 0.560000 | 0.286689 |
| 3-material | aware+containment | 0.1 | 39 | 78 | 36 | 0.333333 | 0.520000 | 0.406250 |
| 3-material | aware+containment | 0.12 | 37 | 53 | 38 | 0.411111 | 0.493333 | 0.448485 |

### step350

| Taxonomy | NMS | Score floor | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | raw | 0.05 | 34 | 1190 | 41 | 0.027778 | 0.453333 | 0.052348 |
| 8-specific | raw | 0.07 | 30 | 606 | 45 | 0.047170 | 0.400000 | 0.084388 |
| 8-specific | raw | 0.1 | 25 | 273 | 50 | 0.083893 | 0.333333 | 0.134048 |
| 8-specific | raw | 0.12 | 22 | 184 | 53 | 0.106796 | 0.293333 | 0.156584 |
| 8-specific | aware IoU | 0.05 | 34 | 977 | 41 | 0.033630 | 0.453333 | 0.062615 |
| 8-specific | aware IoU | 0.07 | 30 | 505 | 45 | 0.056075 | 0.400000 | 0.098361 |
| 8-specific | aware IoU | 0.1 | 25 | 232 | 50 | 0.097276 | 0.333333 | 0.150602 |
| 8-specific | aware IoU | 0.12 | 22 | 158 | 53 | 0.122222 | 0.293333 | 0.172549 |
| 8-specific | aware+containment | 0.05 | 31 | 741 | 44 | 0.040155 | 0.413333 | 0.073200 |
| 8-specific | aware+containment | 0.07 | 28 | 418 | 47 | 0.062780 | 0.373333 | 0.107486 |
| 8-specific | aware+containment | 0.1 | 24 | 199 | 51 | 0.107623 | 0.320000 | 0.161074 |
| 8-specific | aware+containment | 0.12 | 21 | 143 | 54 | 0.128049 | 0.280000 | 0.175732 |
| 4-material | raw | 0.05 | 54 | 1170 | 21 | 0.044118 | 0.720000 | 0.083141 |
| 4-material | raw | 0.07 | 51 | 585 | 24 | 0.080189 | 0.680000 | 0.143460 |
| 4-material | raw | 0.1 | 46 | 252 | 29 | 0.154362 | 0.613333 | 0.246649 |
| 4-material | raw | 0.12 | 44 | 162 | 31 | 0.213592 | 0.586667 | 0.313167 |
| 4-material | aware IoU | 0.05 | 53 | 878 | 22 | 0.056928 | 0.706667 | 0.105368 |
| 4-material | aware IoU | 0.07 | 50 | 427 | 25 | 0.104822 | 0.666667 | 0.181159 |
| 4-material | aware IoU | 0.1 | 46 | 171 | 29 | 0.211982 | 0.613333 | 0.315068 |
| 4-material | aware IoU | 0.12 | 44 | 106 | 31 | 0.293333 | 0.586667 | 0.391111 |
| 4-material | aware+containment | 0.05 | 49 | 564 | 26 | 0.079935 | 0.653333 | 0.142442 |
| 4-material | aware+containment | 0.07 | 47 | 301 | 28 | 0.135057 | 0.626667 | 0.222222 |
| 4-material | aware+containment | 0.1 | 43 | 123 | 32 | 0.259036 | 0.573333 | 0.356846 |
| 4-material | aware+containment | 0.12 | 41 | 80 | 34 | 0.338843 | 0.546667 | 0.418367 |
| 3-material | raw | 0.05 | 57 | 1167 | 18 | 0.046569 | 0.760000 | 0.087760 |
| 3-material | raw | 0.07 | 54 | 582 | 21 | 0.084906 | 0.720000 | 0.151899 |
| 3-material | raw | 0.1 | 49 | 249 | 26 | 0.164430 | 0.653333 | 0.262735 |
| 3-material | raw | 0.12 | 47 | 159 | 28 | 0.228155 | 0.626667 | 0.334520 |
| 3-material | aware IoU | 0.05 | 56 | 862 | 19 | 0.061002 | 0.746667 | 0.112790 |
| 3-material | aware IoU | 0.07 | 53 | 417 | 22 | 0.112766 | 0.706667 | 0.194495 |
| 3-material | aware IoU | 0.1 | 49 | 164 | 26 | 0.230047 | 0.653333 | 0.340278 |
| 3-material | aware IoU | 0.12 | 47 | 99 | 28 | 0.321918 | 0.626667 | 0.425339 |
| 3-material | aware+containment | 0.05 | 52 | 541 | 23 | 0.087690 | 0.693333 | 0.155689 |
| 3-material | aware+containment | 0.07 | 50 | 278 | 25 | 0.152439 | 0.666667 | 0.248139 |
| 3-material | aware+containment | 0.1 | 46 | 111 | 29 | 0.292994 | 0.613333 | 0.396552 |
| 3-material | aware+containment | 0.12 | 44 | 69 | 31 | 0.389381 | 0.586667 | 0.468085 |

### step442

| Taxonomy | NMS | Score floor | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | raw | 0.05 | 35 | 1083 | 40 | 0.031306 | 0.466667 | 0.058676 |
| 8-specific | raw | 0.07 | 31 | 547 | 44 | 0.053633 | 0.413333 | 0.094946 |
| 8-specific | raw | 0.1 | 27 | 246 | 48 | 0.098901 | 0.360000 | 0.155172 |
| 8-specific | raw | 0.12 | 23 | 167 | 52 | 0.121053 | 0.306667 | 0.173585 |
| 8-specific | aware IoU | 0.05 | 35 | 900 | 40 | 0.037433 | 0.466667 | 0.069307 |
| 8-specific | aware IoU | 0.07 | 31 | 471 | 44 | 0.061753 | 0.413333 | 0.107452 |
| 8-specific | aware IoU | 0.1 | 27 | 216 | 48 | 0.111111 | 0.360000 | 0.169811 |
| 8-specific | aware IoU | 0.12 | 23 | 143 | 52 | 0.138554 | 0.306667 | 0.190871 |
| 8-specific | aware+containment | 0.05 | 31 | 724 | 44 | 0.041060 | 0.413333 | 0.074699 |
| 8-specific | aware+containment | 0.07 | 29 | 400 | 46 | 0.067599 | 0.386667 | 0.115079 |
| 8-specific | aware+containment | 0.1 | 26 | 193 | 49 | 0.118721 | 0.346667 | 0.176871 |
| 8-specific | aware+containment | 0.12 | 22 | 133 | 53 | 0.141935 | 0.293333 | 0.191304 |
| 4-material | raw | 0.05 | 53 | 1065 | 22 | 0.047406 | 0.706667 | 0.088852 |
| 4-material | raw | 0.07 | 49 | 529 | 26 | 0.084775 | 0.653333 | 0.150077 |
| 4-material | raw | 0.1 | 45 | 228 | 30 | 0.164835 | 0.600000 | 0.258621 |
| 4-material | raw | 0.12 | 41 | 149 | 34 | 0.215789 | 0.546667 | 0.309434 |
| 4-material | aware IoU | 0.05 | 52 | 806 | 23 | 0.060606 | 0.693333 | 0.111468 |
| 4-material | aware IoU | 0.07 | 48 | 390 | 27 | 0.109589 | 0.640000 | 0.187135 |
| 4-material | aware IoU | 0.1 | 45 | 157 | 30 | 0.222772 | 0.600000 | 0.324910 |
| 4-material | aware IoU | 0.12 | 41 | 94 | 34 | 0.303704 | 0.546667 | 0.390476 |
| 4-material | aware+containment | 0.05 | 47 | 592 | 28 | 0.073552 | 0.626667 | 0.131653 |
| 4-material | aware+containment | 0.07 | 45 | 296 | 30 | 0.131965 | 0.600000 | 0.216346 |
| 4-material | aware+containment | 0.1 | 42 | 128 | 33 | 0.247059 | 0.560000 | 0.342857 |
| 4-material | aware+containment | 0.12 | 39 | 78 | 36 | 0.333333 | 0.520000 | 0.406250 |
| 3-material | raw | 0.05 | 56 | 1062 | 19 | 0.050089 | 0.746667 | 0.093881 |
| 3-material | raw | 0.07 | 52 | 526 | 23 | 0.089965 | 0.693333 | 0.159265 |
| 3-material | raw | 0.1 | 48 | 225 | 27 | 0.175824 | 0.640000 | 0.275862 |
| 3-material | raw | 0.12 | 44 | 146 | 31 | 0.231579 | 0.586667 | 0.332075 |
| 3-material | aware IoU | 0.05 | 55 | 794 | 20 | 0.064782 | 0.733333 | 0.119048 |
| 3-material | aware IoU | 0.07 | 51 | 380 | 24 | 0.118329 | 0.680000 | 0.201581 |
| 3-material | aware IoU | 0.1 | 48 | 149 | 27 | 0.243655 | 0.640000 | 0.352941 |
| 3-material | aware IoU | 0.12 | 44 | 87 | 31 | 0.335878 | 0.586667 | 0.427184 |
| 3-material | aware+containment | 0.05 | 51 | 565 | 24 | 0.082792 | 0.680000 | 0.147612 |
| 3-material | aware+containment | 0.07 | 49 | 274 | 26 | 0.151703 | 0.653333 | 0.246231 |
| 3-material | aware+containment | 0.1 | 46 | 114 | 29 | 0.287500 | 0.613333 | 0.391489 |
| 3-material | aware+containment | 0.12 | 43 | 66 | 32 | 0.394495 | 0.573333 | 0.467391 |

### blend_350_442_2575

| Taxonomy | NMS | Score floor | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | raw | 0.05 | 38 | 1114 | 37 | 0.032986 | 0.506667 | 0.061940 |
| 8-specific | raw | 0.07 | 34 | 557 | 41 | 0.057530 | 0.453333 | 0.102102 |
| 8-specific | raw | 0.1 | 29 | 252 | 46 | 0.103203 | 0.386667 | 0.162921 |
| 8-specific | raw | 0.12 | 25 | 164 | 50 | 0.132275 | 0.333333 | 0.189394 |
| 8-specific | aware IoU | 0.05 | 38 | 927 | 37 | 0.039378 | 0.506667 | 0.073077 |
| 8-specific | aware IoU | 0.07 | 34 | 478 | 41 | 0.066406 | 0.453333 | 0.115843 |
| 8-specific | aware IoU | 0.1 | 29 | 218 | 46 | 0.117409 | 0.386667 | 0.180124 |
| 8-specific | aware IoU | 0.12 | 25 | 142 | 50 | 0.149701 | 0.333333 | 0.206612 |
| 8-specific | aware+containment | 0.05 | 34 | 735 | 41 | 0.044213 | 0.453333 | 0.080569 |
| 8-specific | aware+containment | 0.07 | 32 | 404 | 43 | 0.073394 | 0.426667 | 0.125245 |
| 8-specific | aware+containment | 0.1 | 28 | 191 | 47 | 0.127854 | 0.373333 | 0.190476 |
| 8-specific | aware+containment | 0.12 | 24 | 132 | 51 | 0.153846 | 0.320000 | 0.207792 |
| 4-material | raw | 0.05 | 55 | 1097 | 20 | 0.047743 | 0.733333 | 0.089650 |
| 4-material | raw | 0.07 | 51 | 540 | 24 | 0.086294 | 0.680000 | 0.153153 |
| 4-material | raw | 0.1 | 47 | 234 | 28 | 0.167260 | 0.626667 | 0.264045 |
| 4-material | raw | 0.12 | 43 | 146 | 32 | 0.227513 | 0.573333 | 0.325758 |
| 4-material | aware IoU | 0.05 | 54 | 833 | 21 | 0.060879 | 0.720000 | 0.112266 |
| 4-material | aware IoU | 0.07 | 50 | 402 | 25 | 0.110619 | 0.666667 | 0.189753 |
| 4-material | aware IoU | 0.1 | 47 | 162 | 28 | 0.224880 | 0.626667 | 0.330986 |
| 4-material | aware IoU | 0.12 | 43 | 94 | 32 | 0.313869 | 0.573333 | 0.405660 |
| 4-material | aware+containment | 0.05 | 49 | 597 | 26 | 0.075851 | 0.653333 | 0.135922 |
| 4-material | aware+containment | 0.07 | 47 | 300 | 28 | 0.135447 | 0.626667 | 0.222749 |
| 4-material | aware+containment | 0.1 | 44 | 128 | 31 | 0.255814 | 0.586667 | 0.356275 |
| 4-material | aware+containment | 0.12 | 41 | 78 | 34 | 0.344538 | 0.546667 | 0.422680 |
| 3-material | raw | 0.05 | 58 | 1094 | 17 | 0.050347 | 0.773333 | 0.094540 |
| 3-material | raw | 0.07 | 54 | 537 | 21 | 0.091371 | 0.720000 | 0.162162 |
| 3-material | raw | 0.1 | 50 | 231 | 25 | 0.177936 | 0.666667 | 0.280899 |
| 3-material | raw | 0.12 | 46 | 143 | 29 | 0.243386 | 0.613333 | 0.348485 |
| 3-material | aware IoU | 0.05 | 57 | 820 | 18 | 0.064994 | 0.760000 | 0.119748 |
| 3-material | aware IoU | 0.07 | 53 | 391 | 22 | 0.119369 | 0.706667 | 0.204239 |
| 3-material | aware IoU | 0.1 | 50 | 154 | 25 | 0.245098 | 0.666667 | 0.358423 |
| 3-material | aware IoU | 0.12 | 46 | 87 | 29 | 0.345865 | 0.613333 | 0.442308 |
| 3-material | aware+containment | 0.05 | 53 | 571 | 22 | 0.084936 | 0.706667 | 0.151645 |
| 3-material | aware+containment | 0.07 | 51 | 278 | 24 | 0.155015 | 0.680000 | 0.252475 |
| 3-material | aware+containment | 0.1 | 48 | 114 | 27 | 0.296296 | 0.640000 | 0.405063 |
| 3-material | aware+containment | 0.12 | 45 | 65 | 30 | 0.409091 | 0.600000 | 0.486486 |

### cb_step350

| Taxonomy | NMS | Score floor | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | raw | 0.05 | 37 | 1209 | 38 | 0.029695 | 0.493333 | 0.056018 |
| 8-specific | raw | 0.07 | 32 | 613 | 43 | 0.049612 | 0.426667 | 0.088889 |
| 8-specific | raw | 0.1 | 28 | 283 | 47 | 0.090032 | 0.373333 | 0.145078 |
| 8-specific | raw | 0.12 | 24 | 185 | 51 | 0.114833 | 0.320000 | 0.169014 |
| 8-specific | aware IoU | 0.05 | 37 | 961 | 38 | 0.037074 | 0.493333 | 0.068966 |
| 8-specific | aware IoU | 0.07 | 32 | 504 | 43 | 0.059701 | 0.426667 | 0.104746 |
| 8-specific | aware IoU | 0.1 | 28 | 239 | 47 | 0.104869 | 0.373333 | 0.163743 |
| 8-specific | aware IoU | 0.12 | 24 | 159 | 51 | 0.131148 | 0.320000 | 0.186047 |
| 8-specific | aware+containment | 0.05 | 33 | 716 | 42 | 0.044059 | 0.440000 | 0.080097 |
| 8-specific | aware+containment | 0.07 | 31 | 405 | 44 | 0.071101 | 0.413333 | 0.121331 |
| 8-specific | aware+containment | 0.1 | 27 | 203 | 48 | 0.117391 | 0.360000 | 0.177049 |
| 8-specific | aware+containment | 0.12 | 24 | 140 | 51 | 0.146341 | 0.320000 | 0.200837 |
| 4-material | raw | 0.05 | 53 | 1193 | 22 | 0.042536 | 0.706667 | 0.080242 |
| 4-material | raw | 0.07 | 49 | 596 | 26 | 0.075969 | 0.653333 | 0.136111 |
| 4-material | raw | 0.1 | 46 | 265 | 29 | 0.147910 | 0.613333 | 0.238342 |
| 4-material | raw | 0.12 | 41 | 168 | 34 | 0.196172 | 0.546667 | 0.288732 |
| 4-material | aware IoU | 0.05 | 52 | 840 | 23 | 0.058296 | 0.693333 | 0.107549 |
| 4-material | aware IoU | 0.07 | 49 | 410 | 26 | 0.106754 | 0.653333 | 0.183521 |
| 4-material | aware IoU | 0.1 | 46 | 169 | 29 | 0.213953 | 0.613333 | 0.317241 |
| 4-material | aware IoU | 0.12 | 41 | 104 | 34 | 0.282759 | 0.546667 | 0.372727 |
| 4-material | aware+containment | 0.05 | 47 | 527 | 28 | 0.081882 | 0.626667 | 0.144838 |
| 4-material | aware+containment | 0.07 | 45 | 275 | 30 | 0.140625 | 0.600000 | 0.227848 |
| 4-material | aware+containment | 0.1 | 42 | 118 | 33 | 0.262500 | 0.560000 | 0.357447 |
| 4-material | aware+containment | 0.12 | 38 | 75 | 37 | 0.336283 | 0.506667 | 0.404255 |
| 3-material | raw | 0.05 | 56 | 1190 | 19 | 0.044944 | 0.746667 | 0.084784 |
| 3-material | raw | 0.07 | 52 | 593 | 23 | 0.080620 | 0.693333 | 0.144444 |
| 3-material | raw | 0.1 | 49 | 262 | 26 | 0.157556 | 0.653333 | 0.253886 |
| 3-material | raw | 0.12 | 44 | 165 | 31 | 0.210526 | 0.586667 | 0.309859 |
| 3-material | aware IoU | 0.05 | 55 | 822 | 20 | 0.062714 | 0.733333 | 0.115546 |
| 3-material | aware IoU | 0.07 | 52 | 400 | 23 | 0.115044 | 0.693333 | 0.197343 |
| 3-material | aware IoU | 0.1 | 49 | 162 | 26 | 0.232227 | 0.653333 | 0.342657 |
| 3-material | aware IoU | 0.12 | 44 | 97 | 31 | 0.312057 | 0.586667 | 0.407407 |
| 3-material | aware+containment | 0.05 | 50 | 498 | 25 | 0.091241 | 0.666667 | 0.160514 |
| 3-material | aware+containment | 0.07 | 48 | 254 | 27 | 0.158940 | 0.640000 | 0.254642 |
| 3-material | aware+containment | 0.1 | 45 | 107 | 30 | 0.296053 | 0.600000 | 0.396476 |
| 3-material | aware+containment | 0.12 | 41 | 65 | 34 | 0.386792 | 0.546667 | 0.453039 |

### cb_step442

| Taxonomy | NMS | Score floor | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-specific | raw | 0.05 | 36 | 1177 | 39 | 0.029678 | 0.480000 | 0.055901 |
| 8-specific | raw | 0.07 | 31 | 597 | 44 | 0.049363 | 0.413333 | 0.088193 |
| 8-specific | raw | 0.1 | 27 | 275 | 48 | 0.089404 | 0.360000 | 0.143236 |
| 8-specific | raw | 0.12 | 25 | 177 | 50 | 0.123762 | 0.333333 | 0.180505 |
| 8-specific | aware IoU | 0.05 | 36 | 970 | 39 | 0.035785 | 0.480000 | 0.066605 |
| 8-specific | aware IoU | 0.07 | 31 | 509 | 44 | 0.057407 | 0.413333 | 0.100813 |
| 8-specific | aware IoU | 0.1 | 27 | 243 | 48 | 0.100000 | 0.360000 | 0.156522 |
| 8-specific | aware IoU | 0.12 | 25 | 159 | 50 | 0.135870 | 0.333333 | 0.193050 |
| 8-specific | aware+containment | 0.05 | 32 | 753 | 43 | 0.040764 | 0.426667 | 0.074419 |
| 8-specific | aware+containment | 0.07 | 30 | 422 | 45 | 0.066372 | 0.400000 | 0.113852 |
| 8-specific | aware+containment | 0.1 | 26 | 208 | 49 | 0.111111 | 0.346667 | 0.168285 |
| 8-specific | aware+containment | 0.12 | 24 | 144 | 51 | 0.142857 | 0.320000 | 0.197531 |
| 4-material | raw | 0.05 | 53 | 1160 | 22 | 0.043693 | 0.706667 | 0.082298 |
| 4-material | raw | 0.07 | 48 | 580 | 27 | 0.076433 | 0.640000 | 0.136558 |
| 4-material | raw | 0.1 | 45 | 257 | 30 | 0.149007 | 0.600000 | 0.238727 |
| 4-material | raw | 0.12 | 42 | 160 | 33 | 0.207921 | 0.560000 | 0.303249 |
| 4-material | aware IoU | 0.05 | 52 | 853 | 23 | 0.057459 | 0.693333 | 0.106122 |
| 4-material | aware IoU | 0.07 | 48 | 415 | 27 | 0.103672 | 0.640000 | 0.178439 |
| 4-material | aware IoU | 0.1 | 45 | 176 | 30 | 0.203620 | 0.600000 | 0.304054 |
| 4-material | aware IoU | 0.12 | 42 | 102 | 33 | 0.291667 | 0.560000 | 0.383562 |
| 4-material | aware+containment | 0.05 | 47 | 585 | 28 | 0.074367 | 0.626667 | 0.132956 |
| 4-material | aware+containment | 0.07 | 45 | 300 | 30 | 0.130435 | 0.600000 | 0.214286 |
| 4-material | aware+containment | 0.1 | 42 | 131 | 33 | 0.242775 | 0.560000 | 0.338710 |
| 4-material | aware+containment | 0.12 | 40 | 81 | 35 | 0.330579 | 0.533333 | 0.408163 |
| 3-material | raw | 0.05 | 56 | 1157 | 19 | 0.046167 | 0.746667 | 0.086957 |
| 3-material | raw | 0.07 | 51 | 577 | 24 | 0.081210 | 0.680000 | 0.145092 |
| 3-material | raw | 0.1 | 48 | 254 | 27 | 0.158940 | 0.640000 | 0.254642 |
| 3-material | raw | 0.12 | 45 | 157 | 30 | 0.222772 | 0.600000 | 0.324910 |
| 3-material | aware IoU | 0.05 | 55 | 841 | 20 | 0.061384 | 0.733333 | 0.113285 |
| 3-material | aware IoU | 0.07 | 51 | 405 | 24 | 0.111842 | 0.680000 | 0.192090 |
| 3-material | aware IoU | 0.1 | 48 | 169 | 27 | 0.221198 | 0.640000 | 0.328767 |
| 3-material | aware IoU | 0.12 | 45 | 95 | 30 | 0.321429 | 0.600000 | 0.418605 |
| 3-material | aware+containment | 0.05 | 51 | 562 | 24 | 0.083197 | 0.680000 | 0.148256 |
| 3-material | aware+containment | 0.07 | 49 | 279 | 26 | 0.149390 | 0.653333 | 0.243176 |
| 3-material | aware+containment | 0.1 | 46 | 119 | 29 | 0.278788 | 0.613333 | 0.383333 |
| 3-material | aware+containment | 0.12 | 44 | 70 | 31 | 0.385965 | 0.586667 | 0.465608 |

## Image 777: GT259 wrapper, cigarettes and nested q4

GT259 is evaluated at IoU >= .50 / score >= .05. Best overlap is not necessarily the highest-score query. `best_any` below uses largest IoU, then score/query-ID tie-breaks; `best_same` uses the same criterion restricted to the active category. Full outcomes retain every overlap. Cigarette coverage is in the metric tables; strict cigarette TP and GT259 assignment at all thresholds are in the summary/CSV.

| State | Taxonomy | NMS | GT259 covered | Best any overlap | Best same active category |
| --- | --- | --- | --- | --- | --- |
| step0 | 8-specific | raw | False | q33 plastic_bottle, s=0.142832, IoU=0.938934 | q51 plastic_wrapper, s=0.135903, IoU=0.444955 |
| step0 | 8-specific | aware IoU | False | q9 plastic_bag, s=0.279489, IoU=0.880026 | q51 plastic_wrapper, s=0.135903, IoU=0.444955 |
| step0 | 8-specific | aware+containment | False | q9 plastic_bag, s=0.279489, IoU=0.880026 | q23 plastic_wrapper, s=0.139582, IoU=0.335981 |
| step0 | 4-material | raw | True | q33 plastic_waste [plastic_bottle], s=0.142832, IoU=0.938934 | q33 plastic_waste [plastic_bottle], s=0.142832, IoU=0.938934 |
| step0 | 4-material | aware IoU | True | q1 plastic_waste [plastic_bottle], s=0.397668, IoU=0.878709 | q1 plastic_waste [plastic_bottle], s=0.397668, IoU=0.878709 |
| step0 | 4-material | aware+containment | True | q1 plastic_waste [plastic_bottle], s=0.397668, IoU=0.878709 | q1 plastic_waste [plastic_bottle], s=0.397668, IoU=0.878709 |
| step0 | 3-material | raw | True | q33 plastic_polymer_waste [plastic_bottle], s=0.142832, IoU=0.938934 | q33 plastic_polymer_waste [plastic_bottle], s=0.142832, IoU=0.938934 |
| step0 | 3-material | aware IoU | True | q1 plastic_polymer_waste [plastic_bottle], s=0.397668, IoU=0.878709 | q1 plastic_polymer_waste [plastic_bottle], s=0.397668, IoU=0.878709 |
| step0 | 3-material | aware+containment | True | q1 plastic_polymer_waste [plastic_bottle], s=0.397668, IoU=0.878709 | q1 plastic_polymer_waste [plastic_bottle], s=0.397668, IoU=0.878709 |
| step221 | 8-specific | raw | False | q1 plastic_bottle, s=0.251551, IoU=0.921028 | q9 plastic_wrapper, s=0.074958, IoU=0.282136 |
| step221 | 8-specific | aware IoU | False | q1 plastic_bottle, s=0.251551, IoU=0.921028 | q9 plastic_wrapper, s=0.074958, IoU=0.282136 |
| step221 | 8-specific | aware+containment | False | q1 plastic_bottle, s=0.251551, IoU=0.921028 | q9 plastic_wrapper, s=0.074958, IoU=0.282136 |
| step221 | 4-material | raw | True | q1 plastic_waste [plastic_bottle], s=0.251551, IoU=0.921028 | q1 plastic_waste [plastic_bottle], s=0.251551, IoU=0.921028 |
| step221 | 4-material | aware IoU | True | q1 plastic_waste [plastic_bottle], s=0.251551, IoU=0.921028 | q1 plastic_waste [plastic_bottle], s=0.251551, IoU=0.921028 |
| step221 | 4-material | aware+containment | True | q1 plastic_waste [plastic_bottle], s=0.251551, IoU=0.921028 | q1 plastic_waste [plastic_bottle], s=0.251551, IoU=0.921028 |
| step221 | 3-material | raw | True | q1 plastic_polymer_waste [plastic_bottle], s=0.251551, IoU=0.921028 | q1 plastic_polymer_waste [plastic_bottle], s=0.251551, IoU=0.921028 |
| step221 | 3-material | aware IoU | True | q1 plastic_polymer_waste [plastic_bottle], s=0.251551, IoU=0.921028 | q1 plastic_polymer_waste [plastic_bottle], s=0.251551, IoU=0.921028 |
| step221 | 3-material | aware+containment | True | q1 plastic_polymer_waste [plastic_bottle], s=0.251551, IoU=0.921028 | q1 plastic_polymer_waste [plastic_bottle], s=0.251551, IoU=0.921028 |
| step350 | 8-specific | raw | False | q0 plastic_bottle, s=0.280956, IoU=0.933830 | q11 plastic_wrapper, s=0.073877, IoU=0.260671 |
| step350 | 8-specific | aware IoU | False | q0 plastic_bottle, s=0.280956, IoU=0.933830 | q11 plastic_wrapper, s=0.073877, IoU=0.260671 |
| step350 | 8-specific | aware+containment | False | q0 plastic_bottle, s=0.280956, IoU=0.933830 | q11 plastic_wrapper, s=0.073877, IoU=0.260671 |
| step350 | 4-material | raw | True | q0 plastic_waste [plastic_bottle], s=0.280956, IoU=0.933830 | q0 plastic_waste [plastic_bottle], s=0.280956, IoU=0.933830 |
| step350 | 4-material | aware IoU | True | q0 plastic_waste [plastic_bottle], s=0.280956, IoU=0.933830 | q0 plastic_waste [plastic_bottle], s=0.280956, IoU=0.933830 |
| step350 | 4-material | aware+containment | True | q0 plastic_waste [plastic_bottle], s=0.280956, IoU=0.933830 | q0 plastic_waste [plastic_bottle], s=0.280956, IoU=0.933830 |
| step350 | 3-material | raw | True | q0 plastic_polymer_waste [plastic_bottle], s=0.280956, IoU=0.933830 | q0 plastic_polymer_waste [plastic_bottle], s=0.280956, IoU=0.933830 |
| step350 | 3-material | aware IoU | True | q0 plastic_polymer_waste [plastic_bottle], s=0.280956, IoU=0.933830 | q0 plastic_polymer_waste [plastic_bottle], s=0.280956, IoU=0.933830 |
| step350 | 3-material | aware+containment | True | q0 plastic_polymer_waste [plastic_bottle], s=0.280956, IoU=0.933830 | q0 plastic_polymer_waste [plastic_bottle], s=0.280956, IoU=0.933830 |
| step442 | 8-specific | raw | False | q0 plastic_bottle, s=0.282397, IoU=0.943702 | q13 plastic_wrapper, s=0.052556, IoU=0.259398 |
| step442 | 8-specific | aware IoU | False | q0 plastic_bottle, s=0.282397, IoU=0.943702 | q13 plastic_wrapper, s=0.052556, IoU=0.259398 |
| step442 | 8-specific | aware+containment | False | q0 plastic_bottle, s=0.282397, IoU=0.943702 | q13 plastic_wrapper, s=0.052556, IoU=0.259398 |
| step442 | 4-material | raw | True | q0 plastic_waste [plastic_bottle], s=0.282397, IoU=0.943702 | q0 plastic_waste [plastic_bottle], s=0.282397, IoU=0.943702 |
| step442 | 4-material | aware IoU | True | q0 plastic_waste [plastic_bottle], s=0.282397, IoU=0.943702 | q0 plastic_waste [plastic_bottle], s=0.282397, IoU=0.943702 |
| step442 | 4-material | aware+containment | True | q0 plastic_waste [plastic_bottle], s=0.282397, IoU=0.943702 | q0 plastic_waste [plastic_bottle], s=0.282397, IoU=0.943702 |
| step442 | 3-material | raw | True | q0 plastic_polymer_waste [plastic_bottle], s=0.282397, IoU=0.943702 | q0 plastic_polymer_waste [plastic_bottle], s=0.282397, IoU=0.943702 |
| step442 | 3-material | aware IoU | True | q0 plastic_polymer_waste [plastic_bottle], s=0.282397, IoU=0.943702 | q0 plastic_polymer_waste [plastic_bottle], s=0.282397, IoU=0.943702 |
| step442 | 3-material | aware+containment | True | q0 plastic_polymer_waste [plastic_bottle], s=0.282397, IoU=0.943702 | q0 plastic_polymer_waste [plastic_bottle], s=0.282397, IoU=0.943702 |
| blend_350_442_2575 | 8-specific | raw | False | q0 plastic_bottle, s=0.283106, IoU=0.941582 | q11 plastic_wrapper, s=0.056839, IoU=0.261201 |
| blend_350_442_2575 | 8-specific | aware IoU | False | q0 plastic_bottle, s=0.283106, IoU=0.941582 | q11 plastic_wrapper, s=0.056839, IoU=0.261201 |
| blend_350_442_2575 | 8-specific | aware+containment | False | q0 plastic_bottle, s=0.283106, IoU=0.941582 | q11 plastic_wrapper, s=0.056839, IoU=0.261201 |
| blend_350_442_2575 | 4-material | raw | True | q0 plastic_waste [plastic_bottle], s=0.283106, IoU=0.941582 | q0 plastic_waste [plastic_bottle], s=0.283106, IoU=0.941582 |
| blend_350_442_2575 | 4-material | aware IoU | True | q0 plastic_waste [plastic_bottle], s=0.283106, IoU=0.941582 | q0 plastic_waste [plastic_bottle], s=0.283106, IoU=0.941582 |
| blend_350_442_2575 | 4-material | aware+containment | True | q0 plastic_waste [plastic_bottle], s=0.283106, IoU=0.941582 | q0 plastic_waste [plastic_bottle], s=0.283106, IoU=0.941582 |
| blend_350_442_2575 | 3-material | raw | True | q0 plastic_polymer_waste [plastic_bottle], s=0.283106, IoU=0.941582 | q0 plastic_polymer_waste [plastic_bottle], s=0.283106, IoU=0.941582 |
| blend_350_442_2575 | 3-material | aware IoU | True | q0 plastic_polymer_waste [plastic_bottle], s=0.283106, IoU=0.941582 | q0 plastic_polymer_waste [plastic_bottle], s=0.283106, IoU=0.941582 |
| blend_350_442_2575 | 3-material | aware+containment | True | q0 plastic_polymer_waste [plastic_bottle], s=0.283106, IoU=0.941582 | q0 plastic_polymer_waste [plastic_bottle], s=0.283106, IoU=0.941582 |
| cb_step350 | 8-specific | raw | False | q0 plastic_bottle, s=0.287488, IoU=0.926461 | q10 plastic_wrapper, s=0.075737, IoU=0.267183 |
| cb_step350 | 8-specific | aware IoU | False | q0 plastic_bottle, s=0.287488, IoU=0.926461 | q10 plastic_wrapper, s=0.075737, IoU=0.267183 |
| cb_step350 | 8-specific | aware+containment | False | q0 plastic_bottle, s=0.287488, IoU=0.926461 | q10 plastic_wrapper, s=0.075737, IoU=0.267183 |
| cb_step350 | 4-material | raw | True | q0 plastic_waste [plastic_bottle], s=0.287488, IoU=0.926461 | q0 plastic_waste [plastic_bottle], s=0.287488, IoU=0.926461 |
| cb_step350 | 4-material | aware IoU | True | q0 plastic_waste [plastic_bottle], s=0.287488, IoU=0.926461 | q0 plastic_waste [plastic_bottle], s=0.287488, IoU=0.926461 |
| cb_step350 | 4-material | aware+containment | True | q0 plastic_waste [plastic_bottle], s=0.287488, IoU=0.926461 | q0 plastic_waste [plastic_bottle], s=0.287488, IoU=0.926461 |
| cb_step350 | 3-material | raw | True | q0 plastic_polymer_waste [plastic_bottle], s=0.287488, IoU=0.926461 | q0 plastic_polymer_waste [plastic_bottle], s=0.287488, IoU=0.926461 |
| cb_step350 | 3-material | aware IoU | True | q0 plastic_polymer_waste [plastic_bottle], s=0.287488, IoU=0.926461 | q0 plastic_polymer_waste [plastic_bottle], s=0.287488, IoU=0.926461 |
| cb_step350 | 3-material | aware+containment | True | q0 plastic_polymer_waste [plastic_bottle], s=0.287488, IoU=0.926461 | q0 plastic_polymer_waste [plastic_bottle], s=0.287488, IoU=0.926461 |
| cb_step442 | 8-specific | raw | False | q0 plastic_bottle, s=0.293842, IoU=0.933923 | q13 plastic_wrapper, s=0.058493, IoU=0.264527 |
| cb_step442 | 8-specific | aware IoU | False | q0 plastic_bottle, s=0.293842, IoU=0.933923 | q13 plastic_wrapper, s=0.058493, IoU=0.264527 |
| cb_step442 | 8-specific | aware+containment | False | q0 plastic_bottle, s=0.293842, IoU=0.933923 | q13 plastic_wrapper, s=0.058493, IoU=0.264527 |
| cb_step442 | 4-material | raw | True | q0 plastic_waste [plastic_bottle], s=0.293842, IoU=0.933923 | q0 plastic_waste [plastic_bottle], s=0.293842, IoU=0.933923 |
| cb_step442 | 4-material | aware IoU | True | q0 plastic_waste [plastic_bottle], s=0.293842, IoU=0.933923 | q0 plastic_waste [plastic_bottle], s=0.293842, IoU=0.933923 |
| cb_step442 | 4-material | aware+containment | True | q0 plastic_waste [plastic_bottle], s=0.293842, IoU=0.933923 | q0 plastic_waste [plastic_bottle], s=0.293842, IoU=0.933923 |
| cb_step442 | 3-material | raw | True | q0 plastic_polymer_waste [plastic_bottle], s=0.293842, IoU=0.933923 | q0 plastic_polymer_waste [plastic_bottle], s=0.293842, IoU=0.933923 |
| cb_step442 | 3-material | aware IoU | True | q0 plastic_polymer_waste [plastic_bottle], s=0.293842, IoU=0.933923 | q0 plastic_polymer_waste [plastic_bottle], s=0.293842, IoU=0.933923 |
| cb_step442 | 3-material | aware+containment | True | q0 plastic_polymer_waste [plastic_bottle], s=0.293842, IoU=0.933923 | q0 plastic_polymer_waste [plastic_bottle], s=0.293842, IoU=0.933923 |

The supplied rounded q4/q0 boxes have IoMin approximately 0.976 (the q4 top extends slightly above q0), rather than exactly 1.00; the containment .80 rule still applies. The following verifies the actual saved queries separately in every state. A query ID does not imply the same physical proposal across checkpoints. The listed suppression identity is the first higher-ranked kept query that qualifies, which may differ from q0.

| State | Taxonomy | NMS | q0 specific | q4 specific | q4 score | q4 kept | Suppressed by | IoU | IoMin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| step0 | 4-material | raw | styrofoam_piece | plastic_film | 0.351312 | True | n/a | n/a | n/a |
| step0 | 4-material | aware IoU | styrofoam_piece | plastic_film | 0.351312 | True | n/a | n/a | n/a |
| step0 | 4-material | aware+containment | styrofoam_piece | plastic_film | 0.351312 | False | 1 | 0.013103 | 1.000000 |
| step221 | 4-material | raw | plastic_bottle | plastic_cup | 0.103849 | True | n/a | n/a | n/a |
| step221 | 4-material | aware IoU | plastic_bottle | plastic_cup | 0.103849 | True | n/a | n/a | n/a |
| step221 | 4-material | aware+containment | plastic_bottle | plastic_cup | 0.103849 | False | 1 | 0.101985 | 0.984097 |
| step350 | 4-material | raw | plastic_bottle | plastic_cup | 0.113526 | True | n/a | n/a | n/a |
| step350 | 4-material | aware IoU | plastic_bottle | plastic_cup | 0.113526 | True | n/a | n/a | n/a |
| step350 | 4-material | aware+containment | plastic_bottle | plastic_cup | 0.113526 | False | 0 | 0.096729 | 0.992545 |
| step442 | 4-material | raw | plastic_bottle | plastic_cup | 0.093863 | True | n/a | n/a | n/a |
| step442 | 4-material | aware IoU | plastic_bottle | plastic_cup | 0.093863 | True | n/a | n/a | n/a |
| step442 | 4-material | aware+containment | plastic_bottle | plastic_cup | 0.093863 | False | 0 | 0.095074 | 0.975658 |
| blend_350_442_2575 | 4-material | raw | plastic_bottle | plastic_cup | 0.099266 | True | n/a | n/a | n/a |
| blend_350_442_2575 | 4-material | aware IoU | plastic_bottle | plastic_cup | 0.099266 | True | n/a | n/a | n/a |
| blend_350_442_2575 | 4-material | aware+containment | plastic_bottle | plastic_cup | 0.099266 | False | 0 | 0.095618 | 0.978411 |
| cb_step350 | 4-material | raw | plastic_bottle | plastic_cup | 0.110149 | True | n/a | n/a | n/a |
| cb_step350 | 4-material | aware IoU | plastic_bottle | plastic_cup | 0.110149 | True | n/a | n/a | n/a |
| cb_step350 | 4-material | aware+containment | plastic_bottle | plastic_cup | 0.110149 | False | 0 | 0.098306 | 0.989729 |
| cb_step442 | 4-material | raw | plastic_bottle | plastic_cup | 0.099592 | True | n/a | n/a | n/a |
| cb_step442 | 4-material | aware IoU | plastic_bottle | plastic_cup | 0.099592 | True | n/a | n/a | n/a |
| cb_step442 | 4-material | aware+containment | plastic_bottle | plastic_cup | 0.099592 | False | 0 | 0.097543 | 0.978349 |

## Fixed ten D-058 E1-to-E2 losses

The historical set contains nine locked large-bin objects and one medium-bin object, not ten large objects. True below means candidate coverage at .05 in the active category. Material agreement can recover subtype confusions without fixing the original eight-class prediction. Full query/IoU evidence is in the [regression ledger](../runs/training/material-hierarchy-6574f73514444b9d8207615e453432b3/regression-ledger.json).

### 8-specific / none (raw queries)

| GT | Image | Specific GT | step0 | step221 | step350 | step442 | blend_350_442_2575 | cb_step350 | cb_step442 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 113 | plastic_bottle | False | True | False | False | False | True | False |
| 104 | 252 | plastic_wrapper | True | True | False | False | False | True | True |
| 177 | 464 | styrofoam_piece | True | True | False | False | False | False | False |
| 250 | 762 | plastic_wrapper | False | True | False | False | False | False | False |
| 258 | 776 | plastic_bag | True | True | False | False | False | False | False |
| 303 | 833 | plastic_bag | True | True | True | False | True | True | True |
| 404 | 1076 | plastic_cup | True | True | False | False | False | False | False |
| 414 | 1096 | plastic_bottle | True | True | False | False | False | False | False |
| 423 | 1126 | plastic_cup | True | True | False | False | False | True | False |
| 443 | 1156 | plastic_wrapper | False | True | False | False | False | False | False |

### 8-specific / class-aware IoU 0.50

| GT | Image | Specific GT | step0 | step221 | step350 | step442 | blend_350_442_2575 | cb_step350 | cb_step442 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 113 | plastic_bottle | False | True | False | False | False | True | False |
| 104 | 252 | plastic_wrapper | True | True | False | False | False | True | True |
| 177 | 464 | styrofoam_piece | True | True | False | False | False | False | False |
| 250 | 762 | plastic_wrapper | False | True | False | False | False | False | False |
| 258 | 776 | plastic_bag | True | True | False | False | False | False | False |
| 303 | 833 | plastic_bag | True | True | True | False | True | True | True |
| 404 | 1076 | plastic_cup | True | True | False | False | False | False | False |
| 414 | 1096 | plastic_bottle | True | True | False | False | False | False | False |
| 423 | 1126 | plastic_cup | True | True | False | False | False | True | False |
| 443 | 1156 | plastic_wrapper | False | True | False | False | False | False | False |

### 8-specific / class-aware IoU 0.50 + containment 0.80

| GT | Image | Specific GT | step0 | step221 | step350 | step442 | blend_350_442_2575 | cb_step350 | cb_step442 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 113 | plastic_bottle | False | True | False | False | False | True | False |
| 104 | 252 | plastic_wrapper | True | True | False | False | False | True | True |
| 177 | 464 | styrofoam_piece | False | False | False | False | False | False | False |
| 250 | 762 | plastic_wrapper | False | True | False | False | False | False | False |
| 258 | 776 | plastic_bag | True | True | False | False | False | False | False |
| 303 | 833 | plastic_bag | True | True | True | False | True | True | True |
| 404 | 1076 | plastic_cup | True | True | False | False | False | False | False |
| 414 | 1096 | plastic_bottle | False | True | False | False | False | False | False |
| 423 | 1126 | plastic_cup | True | True | False | False | False | True | False |
| 443 | 1156 | plastic_wrapper | False | True | False | False | False | False | False |

### 4-material / none (raw queries)

| GT | Image | Specific GT | step0 | step221 | step350 | step442 | blend_350_442_2575 | cb_step350 | cb_step442 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 113 | plastic_bottle | True | True | True | True | True | True | True |
| 104 | 252 | plastic_wrapper | True | True | True | True | True | True | True |
| 177 | 464 | styrofoam_piece | True | True | False | False | False | False | False |
| 250 | 762 | plastic_wrapper | True | True | True | True | True | True | True |
| 258 | 776 | plastic_bag | True | True | True | True | True | True | True |
| 303 | 833 | plastic_bag | True | True | True | True | True | True | True |
| 404 | 1076 | plastic_cup | True | True | True | True | True | True | True |
| 414 | 1096 | plastic_bottle | True | True | True | True | True | True | True |
| 423 | 1126 | plastic_cup | True | True | True | True | True | True | True |
| 443 | 1156 | plastic_wrapper | True | True | True | True | True | True | True |

### 4-material / class-aware IoU 0.50

| GT | Image | Specific GT | step0 | step221 | step350 | step442 | blend_350_442_2575 | cb_step350 | cb_step442 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 113 | plastic_bottle | True | True | True | True | True | True | True |
| 104 | 252 | plastic_wrapper | True | True | True | True | True | True | True |
| 177 | 464 | styrofoam_piece | True | True | False | False | False | False | False |
| 250 | 762 | plastic_wrapper | True | True | True | True | True | True | True |
| 258 | 776 | plastic_bag | True | True | True | True | True | True | True |
| 303 | 833 | plastic_bag | True | True | True | True | True | True | True |
| 404 | 1076 | plastic_cup | True | True | True | True | True | True | True |
| 414 | 1096 | plastic_bottle | True | True | True | True | True | True | True |
| 423 | 1126 | plastic_cup | True | True | True | True | True | True | True |
| 443 | 1156 | plastic_wrapper | True | True | True | True | True | True | True |

### 4-material / class-aware IoU 0.50 + containment 0.80

| GT | Image | Specific GT | step0 | step221 | step350 | step442 | blend_350_442_2575 | cb_step350 | cb_step442 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 113 | plastic_bottle | True | True | True | True | True | True | True |
| 104 | 252 | plastic_wrapper | True | True | True | True | True | True | True |
| 177 | 464 | styrofoam_piece | False | False | False | False | False | False | False |
| 250 | 762 | plastic_wrapper | False | True | True | True | True | True | True |
| 258 | 776 | plastic_bag | False | True | True | True | True | True | True |
| 303 | 833 | plastic_bag | True | True | True | True | True | True | True |
| 404 | 1076 | plastic_cup | True | True | True | True | True | True | True |
| 414 | 1096 | plastic_bottle | False | True | True | True | True | True | True |
| 423 | 1126 | plastic_cup | False | True | True | True | True | True | True |
| 443 | 1156 | plastic_wrapper | True | True | True | True | True | True | True |

### 3-material / none (raw queries)

| GT | Image | Specific GT | step0 | step221 | step350 | step442 | blend_350_442_2575 | cb_step350 | cb_step442 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 113 | plastic_bottle | True | True | True | True | True | True | True |
| 104 | 252 | plastic_wrapper | True | True | True | True | True | True | True |
| 177 | 464 | styrofoam_piece | True | True | True | True | True | True | True |
| 250 | 762 | plastic_wrapper | True | True | True | True | True | True | True |
| 258 | 776 | plastic_bag | True | True | True | True | True | True | True |
| 303 | 833 | plastic_bag | True | True | True | True | True | True | True |
| 404 | 1076 | plastic_cup | True | True | True | True | True | True | True |
| 414 | 1096 | plastic_bottle | True | True | True | True | True | True | True |
| 423 | 1126 | plastic_cup | True | True | True | True | True | True | True |
| 443 | 1156 | plastic_wrapper | True | True | True | True | True | True | True |

### 3-material / class-aware IoU 0.50

| GT | Image | Specific GT | step0 | step221 | step350 | step442 | blend_350_442_2575 | cb_step350 | cb_step442 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 113 | plastic_bottle | True | True | True | True | True | True | True |
| 104 | 252 | plastic_wrapper | True | True | True | True | True | True | True |
| 177 | 464 | styrofoam_piece | True | True | True | True | True | True | True |
| 250 | 762 | plastic_wrapper | True | True | True | True | True | True | True |
| 258 | 776 | plastic_bag | True | True | True | True | True | True | True |
| 303 | 833 | plastic_bag | True | True | True | True | True | True | True |
| 404 | 1076 | plastic_cup | True | True | True | True | True | True | True |
| 414 | 1096 | plastic_bottle | True | True | True | True | True | True | True |
| 423 | 1126 | plastic_cup | True | True | True | True | True | True | True |
| 443 | 1156 | plastic_wrapper | True | True | True | True | True | True | True |

### 3-material / class-aware IoU 0.50 + containment 0.80

| GT | Image | Specific GT | step0 | step221 | step350 | step442 | blend_350_442_2575 | cb_step350 | cb_step442 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 113 | plastic_bottle | True | True | True | True | True | True | True |
| 104 | 252 | plastic_wrapper | False | True | True | True | True | True | True |
| 177 | 464 | styrofoam_piece | False | True | True | True | True | True | True |
| 250 | 762 | plastic_wrapper | False | True | True | True | True | True | True |
| 258 | 776 | plastic_bag | False | True | True | True | True | True | True |
| 303 | 833 | plastic_bag | True | True | True | True | True | True | True |
| 404 | 1076 | plastic_cup | True | True | True | True | True | True | True |
| 414 | 1096 | plastic_bottle | False | True | True | True | True | True | True |
| 423 | 1126 | plastic_cup | False | True | True | True | True | True | True |
| 443 | 1156 | plastic_wrapper | True | True | True | True | True | True | True |

## Raw TACO versus validated Stage 2 support

Counts use raw source annotations (including unavailable/excluded images) and the existing Stage-2 export-decision ledger, not test.json or test photos. Raw total 4,784; Stage 2 total 538. Whole-image strict-clean exclusion and the exact eight-class taxonomy select a subset; these totals are not the number of training or validation GTs.

| Raw category | Raw supercategory | Raw annotations | Stage 2 annotations |
| --- | --- | --- | --- |
| Aluminium foil | Aluminium foil | 62 | 0 |
| Battery | Battery | 2 | 0 |
| Aluminium blister pack | Blister pack | 6 | 0 |
| Carded blister pack | Blister pack | 1 | 0 |
| Other plastic bottle | Bottle | 50 | 5 |
| Clear plastic bottle | Bottle | 285 | 56 |
| Glass bottle | Bottle | 104 | 0 |
| Plastic bottle cap | Bottle cap | 209 | 0 |
| Metal bottle cap | Bottle cap | 80 | 0 |
| Broken glass | Broken glass | 138 | 0 |
| Food Can | Can | 34 | 0 |
| Aerosol | Can | 10 | 0 |
| Drink can | Can | 229 | 54 |
| Toilet tube | Carton | 5 | 0 |
| Other carton | Carton | 93 | 0 |
| Egg carton | Carton | 11 | 0 |
| Drink carton | Carton | 45 | 0 |
| Corrugated carton | Carton | 64 | 0 |
| Meal carton | Carton | 30 | 0 |
| Pizza box | Carton | 3 | 0 |
| Paper cup | Cup | 67 | 0 |
| Disposable plastic cup | Cup | 104 | 24 |
| Foam cup | Cup | 13 | 0 |
| Glass cup | Cup | 6 | 0 |
| Other plastic cup | Cup | 2 | 0 |
| Food waste | Food waste | 8 | 0 |
| Glass jar | Glass jar | 6 | 0 |
| Plastic lid | Lid | 77 | 0 |
| Metal lid | Lid | 10 | 0 |
| Other plastic | Other plastic | 273 | 0 |
| Magazine paper | Paper | 12 | 0 |
| Tissues | Paper | 42 | 0 |
| Wrapping paper | Paper | 12 | 0 |
| Normal paper | Paper | 82 | 0 |
| Paper bag | Paper bag | 27 | 0 |
| Plastified paper bag | Paper bag | 0 | 0 |
| Plastic film | Plastic bag & wrapper | 451 | 125 |
| Six pack rings | Plastic bag & wrapper | 5 | 0 |
| Garbage bag | Plastic bag & wrapper | 31 | 4 |
| Other plastic wrapper | Plastic bag & wrapper | 260 | 61 |
| Single-use carrier bag | Plastic bag & wrapper | 61 | 9 |
| Polypropylene bag | Plastic bag & wrapper | 3 | 0 |
| Crisp packet | Plastic bag & wrapper | 39 | 22 |
| Spread tub | Plastic container | 9 | 0 |
| Tupperware | Plastic container | 4 | 0 |
| Disposable food container | Plastic container | 38 | 0 |
| Foam food container | Plastic container | 15 | 0 |
| Other plastic container | Plastic container | 6 | 0 |
| Plastic glooves | Plastic glooves | 4 | 0 |
| Plastic utensils | Plastic utensils | 37 | 0 |
| Pop tab | Pop tab | 99 | 0 |
| Rope & strings | Rope & strings | 29 | 0 |
| Scrap metal | Scrap metal | 20 | 0 |
| Shoe | Shoe | 7 | 0 |
| Squeezable tube | Squeezable tube | 7 | 0 |
| Plastic straw | Straw | 157 | 0 |
| Paper straw | Straw | 4 | 0 |
| Styrofoam piece | Styrofoam piece | 112 | 37 |
| Unlabeled litter | Unlabeled litter | 517 | 0 |
| Cigarette | Cigarette | 667 | 141 |

Paper here means the four source classes under supercategory `Paper`: 148 raw annotations, zero Stage-2 annotations. Paper bag, plastified paper bag, paper cup, paper straw and carton classes are listed separately above and also have zero validated support. Organic corresponds to source `Food waste`: 8 raw annotations, zero Stage-2 annotations. Neither is an approved Stage-4 class or has validated GT support, so neither can receive a measured Stage-4 AP/recall or be introduced by mapping. Zero support does not establish that such litter is absent in deployment. No unsupported sixth/fifth material categories are invented.

## Verification, boundaries and replay

Guarded full discoverable suite: 267 tests, 251 passed, 16 documented skips, zero failures/errors. Skips: 15 training-execution fixtures forbidden in this task and one Windows symlink-privilege fixture. Native Windows Tk: 4 cases (two saved choices x standard/per-object), each checking all 15 taxonomy/NMS combinations, Image 777, all 47 IDs, title/provenance labels, navigation, zoom/pan and per-object GT navigation/green agreement. Model loading/forward, CUDA initialization, backward/optimizer updates, checkpoint deserialization and annotated exports are blocked in checks. All saved JSON hashes are unchanged. Two additional focused native cases verify final per-annotation hazard columns and values after that table addition. Runtime checks are not human visual acceptance, which remains pending.

Integrity: all 60 bundles verified before and after; 1831 prior protected files rehashed unchanged, including the official taxonomy and six core source files. Each bundle is checked against its existing training contract and file hashes; checkpoint tensors are never deserialized. The manifest retains all existing choices without modification. Academic-use rights and uncertainties remain as documented in D-042.

Run from the project root. Close the first window before running the second command. Select **4 material/hazard categories (mapped)** (or the polymer view), then **class-aware IoU 0.50 + containment 0.80** in the toolbar; use all900 and .05 for the audit's coverage pool. Default standard drawing remains top5; select top10/15 as desired. All 47 validation images are selectable from the Image-777 start.

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
$review = 'runs/training/d060-saved-review/review-manifest.json'
$choice = '42 / 1e-5 @ 384/576 (350 vs 442 vs integrated)'
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --saved-review $review --run-choice $choice --gui --image-id 777
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --saved-review $review --run-choice $choice --gui --per-object --image-id 777
```

For the class-balanced trajectory use the same two commands with `$choice = '42 / 1e-5 @ 384/576 (class-balanced)'`; its panels are CB350, CB442 and CB25/75. Existing pretrained/E1/E2 and step350 choices retain their original behavior. No new manifest entries or checkpoint promotions are needed because taxonomy/NMS controls operate on every existing saved choice.
