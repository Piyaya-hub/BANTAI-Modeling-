# Stage 4 — Cross-seed stability and classification analysis

D-052, analysis and visual-review preparation only, 2026-10-22. No new training, model loading, inference, test-split evaluation or checkpoint promotion. Human photo-level confirmation remains pending.

## Main findings

The learning-rate/epoch ranking reverses across seeds. Lowering LR does not consistently prevent second-epoch regression. Seed-42 / 1e-5 / E1 remains the highest observed AP/AR100; it is preserved without promotion. Two seeds are descriptive evidence, not a population estimate or statistically conclusive ranking. Class-aware AP mixes localization, assignment and ranking, so it cannot identify a causal training mechanism.

## Verified corpus, protocol and sources

Unchanged corrected `bantai.taco-evaluation.v2`: 47 validation images, 75 annotations, eight labels; 900 queries, mean subword sigmoid class scores, one argmax class/query, zero floor, top100 valid predictions/image, no NMS, COCO bbox IoU .50:.05:.95. Recomputed all nine COCO results exactly from saved predictions; `saved_rows` independently reconstructs stable rank and official top100. All four saved pretrained baselines agree exactly in metrics, raw queries and predictions. All checkpoint bundles of the four runs were hash-verified against their original contracts without restoring state. Prior protected inventory verified, excluding the explicitly extended viewer sources; no historical replay policy or training source changed.

Evidence: `runs\training\cross-seed-analysis-fd10923e40d24c40be5bab3ae0fa2c63`. Local hashes establish artifact consistency, not independent signed provenance. Existing owner-authorized academic-use record remains distinct from definitive rights clearance.


| Seed / LR | Saved run |
| --- | --- |
| 42 / 1e-5 | taco-312e568703a54f98a7c8f4b87b29cd39 |
| 42 / 5e-6 | taco-dd541c3972834a64a16771df44a7e891 |
| 123 / 1e-5 | taco-4bf521d7529e4fa4b168b96786020844 |
| 123 / 5e-6 | taco-e7e4008e98a241e19047e14645f06ec1 |



| State | AP | AP50 | AP75 | AR100 |
| --- | --- | --- | --- | --- |
| Pretrained | 0.111576 | 0.158253 | 0.135975 | 0.363077 |
| 42 / 1e-5 / E1 | 0.203932 | 0.254720 | 0.239987 | 0.396766 |
| 42 / 1e-5 / E2 | 0.084500 | 0.107561 | 0.095599 | 0.211229 |
| 42 / 5e-6 / E1 | 0.137099 | 0.182086 | 0.164359 | 0.330432 |
| 42 / 5e-6 / E2 | 0.145952 | 0.174521 | 0.158049 | 0.302977 |
| 123 / 1e-5 / E1 | 0.140478 | 0.169154 | 0.152226 | 0.276718 |
| 123 / 1e-5 / E2 | 0.152880 | 0.178946 | 0.164636 | 0.274295 |
| 123 / 5e-6 / E1 | 0.137919 | 0.176573 | 0.156206 | 0.341559 |
| 123 / 5e-6 / E2 | 0.113018 | 0.144608 | 0.125368 | 0.291915 |


## Two-seed descriptive aggregation

Each cell reports mean [minimum, maximum]; Δ is seed123 minus seed42. Range is the absolute Δ with two observations. Epochs stay separate; they are not independent seed replicates. No confidence intervals or significance tests are claimed.


| LR | Epoch | Metric | Mean | Min | Max | Range | Signed Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-5 | 1 | AP | 0.172205 | 0.140478 | 0.203932 | 0.063454 | -0.063454 |
| 1e-5 | 1 | AP50 | 0.211937 | 0.169154 | 0.254720 | 0.085566 | -0.085566 |
| 1e-5 | 1 | AP75 | 0.196107 | 0.152226 | 0.239987 | 0.087761 | -0.087761 |
| 1e-5 | 1 | AR100 | 0.336742 | 0.276718 | 0.396766 | 0.120048 | -0.120048 |
| 1e-5 | 2 | AP | 0.118690 | 0.084500 | 0.152880 | 0.068380 | 0.068380 |
| 1e-5 | 2 | AP50 | 0.143253 | 0.107561 | 0.178946 | 0.071385 | 0.071385 |
| 1e-5 | 2 | AP75 | 0.130117 | 0.095599 | 0.164636 | 0.069036 | 0.069036 |
| 1e-5 | 2 | AR100 | 0.242762 | 0.211229 | 0.274295 | 0.063066 | 0.063066 |
| 5e-6 | 1 | AP | 0.137509 | 0.137099 | 0.137919 | 0.000821 | 0.000821 |
| 5e-6 | 1 | AP50 | 0.179330 | 0.176573 | 0.182086 | 0.005513 | -0.005513 |
| 5e-6 | 1 | AP75 | 0.160282 | 0.156206 | 0.164359 | 0.008152 | -0.008152 |
| 5e-6 | 1 | AR100 | 0.335995 | 0.330432 | 0.341559 | 0.011127 | 0.011127 |
| 5e-6 | 2 | AP | 0.129485 | 0.113018 | 0.145952 | 0.032934 | -0.032934 |
| 5e-6 | 2 | AP50 | 0.159564 | 0.144608 | 0.174521 | 0.029914 | -0.029914 |
| 5e-6 | 2 | AP75 | 0.141709 | 0.125368 | 0.158049 | 0.032682 | -0.032682 |
| 5e-6 | 2 | AR100 | 0.297446 | 0.291915 | 0.302977 | 0.011062 | -0.011062 |


## Per-class stability

AP and recall below use full COCO IoU averaging; AP50 is saved separately in the previous replication analysis. N is validation annotations; image support is also shown. Rare classes can dominate macro-AP shifts. These are the same validation examples across states.

### cigarette — 24 annotations / 7 images


| State | AP | AR100 | Same / any all900 | Same / any top100 |
| --- | --- | --- | --- | --- |
| Pretrained | 0.014851 | 0.012500 | 3 / 5 | 1 / 1 |
| 42 / 1e-5 / E1 | 0.020136 | 0.050000 | 5 / 5 | 4 / 4 |
| 42 / 1e-5 / E2 | 0.025458 | 0.079167 | 8 / 8 | 7 / 7 |
| 42 / 5e-6 / E1 | 0.020116 | 0.033333 | 3 / 5 | 2 / 3 |
| 42 / 5e-6 / E2 | 0.025116 | 0.054167 | 5 / 7 | 4 / 4 |
| 123 / 1e-5 / E1 | 0.020507 | 0.054167 | 6 / 8 | 5 / 5 |
| 123 / 1e-5 / E2 | 0.034787 | 0.112500 | 9 / 9 | 8 / 8 |
| 123 / 5e-6 / E1 | 0.020092 | 0.029167 | 3 / 5 | 2 / 4 |
| 123 / 5e-6 / E2 | 0.020285 | 0.054167 | 6 / 8 | 4 / 4 |


### plastic_bottle — 8 annotations / 8 images


| State | AP | AR100 | Same / any all900 | Same / any top100 |
| --- | --- | --- | --- | --- |
| Pretrained | 0.038242 | 0.400000 | 4 / 8 | 4 / 8 |
| 42 / 1e-5 / E1 | 0.096181 | 0.462500 | 6 / 8 | 5 / 8 |
| 42 / 1e-5 / E2 | 0.103595 | 0.337500 | 3 / 8 | 3 / 8 |
| 42 / 5e-6 / E1 | 0.078044 | 0.462500 | 6 / 8 | 5 / 8 |
| 42 / 5e-6 / E2 | 0.143604 | 0.462500 | 5 / 8 | 5 / 8 |
| 123 / 1e-5 / E1 | 0.113039 | 0.400000 | 4 / 8 | 4 / 8 |
| 123 / 1e-5 / E2 | 0.055776 | 0.162500 | 4 / 8 | 2 / 8 |
| 123 / 5e-6 / E1 | 0.079703 | 0.587500 | 6 / 8 | 6 / 8 |
| 123 / 5e-6 / E2 | 0.144865 | 0.575000 | 6 / 8 | 6 / 8 |


### plastic_bag — 2 annotations / 2 images


| State | AP | AR100 | Same / any all900 | Same / any top100 |
| --- | --- | --- | --- | --- |
| Pretrained | 0.236229 | 0.550000 | 2 / 2 | 2 / 2 |
| 42 / 1e-5 / E1 | 0.677759 | 0.950000 | 2 / 2 | 2 / 2 |
| 42 / 1e-5 / E2 | 0.000000 | 0.000000 | 0 / 2 | 0 / 2 |
| 42 / 5e-6 / E1 | 0.454455 | 0.450000 | 2 / 2 | 1 / 2 |
| 42 / 5e-6 / E2 | 0.252475 | 0.500000 | 1 / 2 | 1 / 2 |
| 123 / 1e-5 / E1 | 0.504950 | 0.500000 | 1 / 2 | 1 / 2 |
| 123 / 1e-5 / E2 | 0.504950 | 0.500000 | 1 / 2 | 1 / 2 |
| 123 / 5e-6 / E1 | 0.504950 | 0.500000 | 1 / 2 | 1 / 2 |
| 123 / 5e-6 / E2 | 0.252475 | 0.500000 | 1 / 2 | 1 / 2 |


### plastic_wrapper — 11 annotations / 11 images


| State | AP | AR100 | Same / any all900 | Same / any top100 |
| --- | --- | --- | --- | --- |
| Pretrained | 0.094467 | 0.490909 | 11 / 11 | 9 / 11 |
| 42 / 1e-5 / E1 | 0.118247 | 0.272727 | 6 / 11 | 5 / 11 |
| 42 / 1e-5 / E2 | 0.135371 | 0.372727 | 6 / 11 | 6 / 11 |
| 42 / 5e-6 / E1 | 0.026708 | 0.254545 | 7 / 11 | 5 / 11 |
| 42 / 5e-6 / E2 | 0.053725 | 0.163636 | 3 / 11 | 3 / 11 |
| 123 / 1e-5 / E1 | 0.125683 | 0.327273 | 8 / 11 | 6 / 11 |
| 123 / 1e-5 / E2 | 0.160149 | 0.327273 | 5 / 11 | 5 / 11 |
| 123 / 5e-6 / E1 | 0.119790 | 0.372727 | 8 / 11 | 6 / 11 |
| 123 / 5e-6 / E2 | 0.127833 | 0.300000 | 5 / 11 | 5 / 11 |


### plastic_film — 13 annotations / 9 images


| State | AP | AR100 | Same / any all900 | Same / any top100 |
| --- | --- | --- | --- | --- |
| Pretrained | 0.001574 | 0.076923 | 2 / 10 | 2 / 9 |
| 42 / 1e-5 / E1 | 0.016310 | 0.384615 | 9 / 12 | 8 / 11 |
| 42 / 1e-5 / E2 | 0.026429 | 0.446154 | 10 / 11 | 10 / 11 |
| 42 / 5e-6 / E1 | 0.012116 | 0.323077 | 6 / 11 | 6 / 11 |
| 42 / 5e-6 / E2 | 0.020279 | 0.369231 | 9 / 11 | 8 / 11 |
| 123 / 1e-5 / E1 | 0.015385 | 0.392308 | 9 / 12 | 8 / 11 |
| 123 / 1e-5 / E2 | 0.032423 | 0.469231 | 10 / 12 | 9 / 11 |
| 123 / 5e-6 / E1 | 0.005826 | 0.323077 | 7 / 11 | 7 / 11 |
| 123 / 5e-6 / E2 | 0.010128 | 0.346154 | 8 / 11 | 7 / 11 |


### drink_can — 7 annotations / 7 images


| State | AP | AR100 | Same / any all900 | Same / any top100 |
| --- | --- | --- | --- | --- |
| Pretrained | 0.108416 | 0.214286 | 2 / 7 | 2 / 7 |
| 42 / 1e-5 / E1 | 0.118812 | 0.114286 | 1 / 7 | 1 / 7 |
| 42 / 1e-5 / E2 | 0.118812 | 0.114286 | 1 / 7 | 1 / 7 |
| 42 / 5e-6 / E1 | 0.000000 | 0.000000 | 0 / 7 | 0 / 7 |
| 42 / 5e-6 / E2 | 0.118812 | 0.114286 | 1 / 7 | 1 / 7 |
| 123 / 1e-5 / E1 | 0.000000 | 0.000000 | 0 / 7 | 0 / 7 |
| 123 / 1e-5 / E2 | 0.148515 | 0.142857 | 1 / 7 | 1 / 7 |
| 123 / 5e-6 / E1 | 0.000000 | 0.000000 | 0 / 7 | 0 / 7 |
| 123 / 5e-6 / E2 | 0.000000 | 0.000000 | 0 / 7 | 0 / 7 |


### plastic_cup — 5 annotations / 4 images


| State | AP | AR100 | Same / any all900 | Same / any top100 |
| --- | --- | --- | --- | --- |
| Pretrained | 0.340656 | 0.600000 | 4 / 5 | 4 / 5 |
| 42 / 1e-5 / E1 | 0.432673 | 0.560000 | 4 / 5 | 4 / 5 |
| 42 / 1e-5 / E2 | 0.266337 | 0.340000 | 2 / 5 | 2 / 5 |
| 42 / 5e-6 / E1 | 0.388111 | 0.580000 | 4 / 5 | 4 / 5 |
| 42 / 5e-6 / E2 | 0.345686 | 0.560000 | 4 / 5 | 4 / 5 |
| 123 / 1e-5 / E1 | 0.344260 | 0.540000 | 4 / 5 | 4 / 5 |
| 123 / 1e-5 / E2 | 0.286439 | 0.480000 | 3 / 5 | 3 / 5 |
| 123 / 5e-6 / E1 | 0.344199 | 0.600000 | 4 / 5 | 4 / 5 |
| 123 / 5e-6 / E2 | 0.348562 | 0.560000 | 4 / 5 | 4 / 5 |


### styrofoam_piece — 5 annotations / 5 images


| State | AP | AR100 | Same / any all900 | Same / any top100 |
| --- | --- | --- | --- | --- |
| Pretrained | 0.058173 | 0.560000 | 4 / 5 | 4 / 5 |
| 42 / 1e-5 / E1 | 0.151337 | 0.380000 | 3 / 5 | 2 / 5 |
| 42 / 1e-5 / E2 | 0.000000 | 0.000000 | 0 / 5 | 0 / 5 |
| 42 / 5e-6 / E1 | 0.117239 | 0.540000 | 4 / 5 | 4 / 5 |
| 42 / 5e-6 / E2 | 0.207921 | 0.200000 | 2 / 5 | 1 / 5 |
| 123 / 1e-5 / E1 | 0.000000 | 0.000000 | 0 / 5 | 0 / 5 |
| 123 / 1e-5 / E2 | 0.000000 | 0.000000 | 1 / 5 | 0 / 5 |
| 123 / 5e-6 / E1 | 0.028793 | 0.320000 | 3 / 5 | 3 / 5 |
| 123 / 5e-6 / E2 | 0.000000 | 0.000000 | 1 / 5 | 0 / 5 |



| Class | AP > base /8 | AP < base | AP = base | AR > base /8 | AR < base | E2 AP rises /4 | E2 AP falls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cigarette | 8 | 0 | 0 | 8 | 0 | 4 | 0 |
| plastic_bottle | 8 | 0 | 0 | 5 | 2 | 3 | 1 |
| plastic_bag | 7 | 1 | 0 | 1 | 7 | 0 | 3 |
| plastic_wrapper | 6 | 2 | 0 | 0 | 8 | 4 | 0 |
| plastic_film | 8 | 0 | 0 | 8 | 0 | 4 | 0 |
| drink_can | 4 | 4 | 0 | 0 | 8 | 2 | 0 |
| plastic_cup | 6 | 2 | 0 | 1 | 7 | 1 | 3 |
| styrofoam_piece | 3 | 5 | 0 | 0 | 8 | 1 | 2 |


Cigarette, wrapper and film AP increase from E1 to E2 in all four runs, but that does not imply uniform improvement over pretrained or improved recall. Wrapper/film material confusion requires human confirmation. Plastic bag AP is highly seed/epoch dependent with only two annotations; the seed-42 high-LR collapse does not reproduce at seed123. Styrofoam has five annotations and intermittent zero AP, so its apparent lower-LR rescue at seed42 is not stable across seeds. Cigarette has more annotations, but its tiny-object/score behavior still needs image review. No class-specific cause is inferred from support counts alone.

Cigarette, plastic film and bottle AP exceed pretrained in all eight trained states. Cigarette and film AR100 also exceed pretrained throughout. Wrapper, can and Styrofoam AR100 are lower throughout; bag AR100 is lower in seven states. Wrapper AP is higher in six states despite uniformly reduced recall. These are consistent within-set patterns, not evidence of calibrated confidence or broad deployment robustness.

## Localization, class assignment, scoring and duplicates

Candidate coverage is non-exclusive and has no score floor: a query may cover several annotations. Same/any use identical IoU .50 (and aggregate .75 below), all900 or official top100. Any-class minus same-class coverage identifies localized annotations without a correctly assigned-class candidate; it is not a count of visually confirmed material errors.


| State | Same/any900 @.50 | Same/any900 @.75 | Same/any100 @.50 |
| --- | --- | --- | --- |
| Pretrained | 32/53 | 21/37 | 28/48 |
| 42 / 1e-5 / E1 | 36/55 | 23/39 | 31/53 |
| 42 / 1e-5 / E2 | 30/57 | 17/41 | 29/56 |
| 42 / 5e-6 / E1 | 32/54 | 23/37 | 27/52 |
| 42 / 5e-6 / E2 | 30/56 | 17/41 | 27/53 |
| 123 / 1e-5 / E1 | 32/58 | 18/41 | 28/54 |
| 123 / 1e-5 / E2 | 34/59 | 19/43 | 29/57 |
| 123 / 5e-6 / E1 | 32/54 | 21/39 | 29/53 |
| 123 / 5e-6 / E2 | 31/57 | 16/38 | 27/53 |


The unchanged diagnostic matcher greedily matches score-ranked, class-compatible top100 candidates to unmatched GT at IoU >=.50. Annotation categories are exclusive: matched; assignment conflict (compatible candidate above threshold already consumed); threshold hidden; outside top100; wrong class only; poor localization (.10 <= best any IoU <.50); no substantial overlap (<.10). Query categories are separately counted; their counts are not object counts. Annotation incompleteness cannot be established by this matcher.

### Diagnostic score >= 0.0


| State | matched | assignment_conflict | threshold_hidden | outside_top100 | wrong_class_only | poor_localization_candidate | no_substantial_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 28 | 0 | 0 | 4 | 21 | 10 | 12 |
| 42 / 1e-5 / E1 | 31 | 0 | 0 | 5 | 19 | 9 | 11 |
| 42 / 1e-5 / E2 | 29 | 0 | 0 | 1 | 27 | 7 | 11 |
| 42 / 5e-6 / E1 | 27 | 0 | 0 | 5 | 22 | 11 | 10 |
| 42 / 5e-6 / E2 | 27 | 0 | 0 | 3 | 26 | 8 | 11 |
| 123 / 1e-5 / E1 | 28 | 0 | 0 | 4 | 26 | 8 | 9 |
| 123 / 1e-5 / E2 | 29 | 0 | 0 | 5 | 25 | 10 | 6 |
| 123 / 5e-6 / E1 | 29 | 0 | 0 | 3 | 22 | 11 | 10 |
| 123 / 5e-6 / E2 | 27 | 0 | 0 | 4 | 26 | 8 | 10 |



| State | matched | wrong_class | duplicate | localization_overlap_010_to_050 | unmatched_prediction |
| --- | --- | --- | --- | --- | --- |
| Pretrained | 28 | 358 | 51 | 1104 | 3159 |
| 42 / 1e-5 / E1 | 31 | 397 | 43 | 1033 | 3196 |
| 42 / 1e-5 / E2 | 29 | 300 | 29 | 804 | 3538 |
| 42 / 5e-6 / E1 | 27 | 332 | 32 | 1084 | 3225 |
| 42 / 5e-6 / E2 | 27 | 309 | 31 | 1024 | 3309 |
| 123 / 1e-5 / E1 | 28 | 302 | 22 | 987 | 3361 |
| 123 / 1e-5 / E2 | 29 | 229 | 24 | 594 | 3824 |
| 123 / 5e-6 / E1 | 29 | 298 | 25 | 1055 | 3293 |
| 123 / 5e-6 / E2 | 27 | 269 | 24 | 1030 | 3350 |


### Diagnostic score >= 0.05


| State | matched | assignment_conflict | threshold_hidden | outside_top100 | wrong_class_only | poor_localization_candidate | no_substantial_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 28 | 0 | 0 | 4 | 21 | 10 | 12 |
| 42 / 1e-5 / E1 | 28 | 0 | 3 | 5 | 19 | 9 | 11 |
| 42 / 1e-5 / E2 | 26 | 0 | 3 | 1 | 27 | 7 | 11 |
| 42 / 5e-6 / E1 | 25 | 0 | 2 | 5 | 22 | 11 | 10 |
| 42 / 5e-6 / E2 | 25 | 0 | 2 | 3 | 26 | 8 | 11 |
| 123 / 1e-5 / E1 | 25 | 0 | 3 | 4 | 26 | 8 | 9 |
| 123 / 1e-5 / E2 | 26 | 0 | 3 | 5 | 25 | 10 | 6 |
| 123 / 5e-6 / E1 | 25 | 0 | 4 | 3 | 22 | 11 | 10 |
| 123 / 5e-6 / E2 | 25 | 0 | 2 | 4 | 26 | 8 | 10 |



| State | matched | wrong_class | duplicate | localization_overlap_010_to_050 | unmatched_prediction |
| --- | --- | --- | --- | --- | --- |
| Pretrained | 28 | 342 | 45 | 987 | 2689 |
| 42 / 1e-5 / E1 | 28 | 296 | 27 | 402 | 719 |
| 42 / 1e-5 / E2 | 26 | 215 | 19 | 268 | 744 |
| 42 / 5e-6 / E1 | 25 | 264 | 29 | 505 | 920 |
| 42 / 5e-6 / E2 | 25 | 233 | 20 | 421 | 870 |
| 123 / 1e-5 / E1 | 25 | 226 | 18 | 390 | 819 |
| 123 / 1e-5 / E2 | 26 | 151 | 12 | 174 | 705 |
| 123 / 5e-6 / E1 | 25 | 246 | 20 | 512 | 1045 |
| 123 / 5e-6 / E2 | 25 | 198 | 20 | 393 | 771 |


### Diagnostic score >= 0.25


| State | matched | assignment_conflict | threshold_hidden | outside_top100 | wrong_class_only | poor_localization_candidate | no_substantial_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 18 | 0 | 10 | 4 | 21 | 10 | 12 |
| 42 / 1e-5 / E1 | 4 | 0 | 27 | 5 | 19 | 9 | 11 |
| 42 / 1e-5 / E2 | 9 | 0 | 20 | 1 | 27 | 7 | 11 |
| 42 / 5e-6 / E1 | 5 | 0 | 22 | 5 | 22 | 11 | 10 |
| 42 / 5e-6 / E2 | 5 | 0 | 22 | 3 | 26 | 8 | 11 |
| 123 / 1e-5 / E1 | 6 | 0 | 22 | 4 | 26 | 8 | 9 |
| 123 / 1e-5 / E2 | 11 | 0 | 18 | 5 | 25 | 10 | 6 |
| 123 / 5e-6 / E1 | 6 | 0 | 23 | 3 | 22 | 11 | 10 |
| 123 / 5e-6 / E2 | 8 | 0 | 19 | 4 | 26 | 8 | 10 |



| State | matched | wrong_class | duplicate | localization_overlap_010_to_050 | unmatched_prediction |
| --- | --- | --- | --- | --- | --- |
| Pretrained | 18 | 79 | 7 | 29 | 83 |
| 42 / 1e-5 / E1 | 4 | 12 | 0 | 0 | 0 |
| 42 / 1e-5 / E2 | 9 | 43 | 0 | 0 | 2 |
| 42 / 5e-6 / E1 | 5 | 10 | 0 | 1 | 5 |
| 42 / 5e-6 / E2 | 5 | 10 | 0 | 0 | 0 |
| 123 / 1e-5 / E1 | 6 | 18 | 0 | 1 | 1 |
| 123 / 1e-5 / E2 | 11 | 24 | 0 | 0 | 2 |
| 123 / 5e-6 / E1 | 6 | 12 | 0 | 1 | 6 |
| 123 / 5e-6 / E2 | 8 | 13 | 0 | 0 | 0 |


The .05 and .25 tables isolate score hiding without changing evaluation. Score-floor-zero still has rank and assignment effects; any900 absence indicates candidate localization failure relative to the saved GT, not proof an object is invisible. Duplicate counts describe same-class overlap with already matched annotations, not independent objects. Wrong-class query counts can be inflated by many near-identical candidates. Unmatched predictions may be background, localization errors or unannotated real objects; none is labelled a confirmed annotation defect.

## Seed-paired agreement and incorrect-class assignments


| LR | Epoch | Same annotation kind /75 | Both matched | Only42 matched | Only123 matched |
| --- | --- | --- | --- | --- | --- |
| 1e-5 | 1 | 55 | 20 | 8 | 5 |
| 1e-5 | 2 | 56 | 21 | 5 | 5 |
| 5e-6 | 1 | 65 | 22 | 3 | 3 |
| 5e-6 | 2 | 66 | 22 | 3 | 3 |


### Pretrained — wrong-class pairs at .05


| GT | Predicted | Queries | Distinct GT |
| --- | --- | --- | --- |
| plastic_wrapper | plastic_bottle | 35 | 10 |
| drink_can | plastic_bottle | 33 | 5 |
| drink_can | plastic_cup | 28 | 7 |
| plastic_wrapper | plastic_bag | 26 | 10 |
| plastic_bottle | plastic_wrapper | 19 | 7 |
| styrofoam_piece | plastic_film | 16 | 3 |
| plastic_wrapper | plastic_film | 13 | 6 |
| styrofoam_piece | plastic_bag | 12 | 3 |


### 42 / 1e-5 / E1 — wrong-class pairs at .05


| GT | Predicted | Queries | Distinct GT |
| --- | --- | --- | --- |
| plastic_wrapper | plastic_film | 46 | 11 |
| styrofoam_piece | plastic_film | 29 | 4 |
| drink_can | plastic_bottle | 23 | 6 |
| drink_can | plastic_film | 22 | 6 |
| plastic_wrapper | plastic_bottle | 20 | 9 |
| plastic_bottle | plastic_film | 17 | 5 |
| plastic_bottle | plastic_cup | 12 | 8 |
| styrofoam_piece | plastic_wrapper | 12 | 4 |


### 42 / 1e-5 / E2 — wrong-class pairs at .05


| GT | Predicted | Queries | Distinct GT |
| --- | --- | --- | --- |
| plastic_wrapper | plastic_film | 48 | 11 |
| plastic_bottle | plastic_film | 23 | 7 |
| styrofoam_piece | plastic_film | 23 | 4 |
| drink_can | plastic_film | 21 | 6 |
| plastic_cup | plastic_film | 12 | 5 |
| plastic_bag | plastic_film | 10 | 2 |
| drink_can | plastic_bottle | 8 | 5 |
| drink_can | cigarette | 6 | 4 |


### 42 / 5e-6 / E1 — wrong-class pairs at .05


| GT | Predicted | Queries | Distinct GT |
| --- | --- | --- | --- |
| plastic_wrapper | plastic_bottle | 27 | 8 |
| drink_can | plastic_bottle | 23 | 6 |
| styrofoam_piece | plastic_film | 20 | 4 |
| drink_can | plastic_cup | 19 | 7 |
| plastic_wrapper | plastic_film | 19 | 9 |
| plastic_bottle | plastic_film | 14 | 5 |
| plastic_wrapper | plastic_bag | 13 | 9 |
| drink_can | plastic_film | 12 | 5 |


### 42 / 5e-6 / E2 — wrong-class pairs at .05


| GT | Predicted | Queries | Distinct GT |
| --- | --- | --- | --- |
| plastic_wrapper | plastic_film | 40 | 11 |
| styrofoam_piece | plastic_film | 28 | 4 |
| drink_can | plastic_film | 22 | 6 |
| plastic_bottle | plastic_film | 16 | 4 |
| plastic_wrapper | plastic_bottle | 15 | 8 |
| drink_can | plastic_bottle | 14 | 6 |
| drink_can | plastic_cup | 13 | 7 |
| plastic_bottle | plastic_cup | 11 | 7 |


### 123 / 1e-5 / E1 — wrong-class pairs at .05


| GT | Predicted | Queries | Distinct GT |
| --- | --- | --- | --- |
| plastic_wrapper | plastic_film | 44 | 11 |
| drink_can | plastic_film | 23 | 6 |
| styrofoam_piece | plastic_film | 22 | 4 |
| plastic_bottle | plastic_film | 19 | 5 |
| drink_can | plastic_cup | 14 | 7 |
| plastic_wrapper | plastic_bottle | 13 | 7 |
| drink_can | plastic_bottle | 10 | 4 |
| plastic_cup | plastic_film | 9 | 4 |


### 123 / 1e-5 / E2 — wrong-class pairs at .05


| GT | Predicted | Queries | Distinct GT |
| --- | --- | --- | --- |
| plastic_wrapper | plastic_film | 31 | 11 |
| drink_can | plastic_film | 16 | 6 |
| plastic_bottle | plastic_film | 16 | 7 |
| styrofoam_piece | plastic_film | 13 | 4 |
| drink_can | cigarette | 9 | 7 |
| plastic_cup | plastic_film | 9 | 5 |
| drink_can | plastic_bottle | 7 | 4 |
| plastic_bottle | plastic_cup | 6 | 3 |


### 123 / 5e-6 / E1 — wrong-class pairs at .05


| GT | Predicted | Queries | Distinct GT |
| --- | --- | --- | --- |
| drink_can | plastic_bottle | 22 | 6 |
| plastic_wrapper | plastic_bottle | 21 | 8 |
| plastic_wrapper | plastic_film | 20 | 9 |
| styrofoam_piece | plastic_film | 18 | 4 |
| drink_can | plastic_film | 16 | 6 |
| plastic_bottle | plastic_film | 16 | 4 |
| drink_can | plastic_cup | 14 | 7 |
| plastic_wrapper | plastic_bag | 12 | 7 |


### 123 / 5e-6 / E2 — wrong-class pairs at .05


| GT | Predicted | Queries | Distinct GT |
| --- | --- | --- | --- |
| plastic_wrapper | plastic_film | 33 | 11 |
| styrofoam_piece | plastic_film | 23 | 4 |
| drink_can | plastic_film | 18 | 6 |
| drink_can | plastic_bottle | 15 | 6 |
| plastic_wrapper | plastic_bottle | 13 | 6 |
| drink_can | plastic_cup | 11 | 7 |
| plastic_bottle | plastic_film | 11 | 4 |
| plastic_bottle | plastic_cup | 8 | 5 |



| LR | Epoch | 42 GT/pred/annotation triples | 123 triples | Shared triples |
| --- | --- | --- | --- | --- |
| 1e-5 | 1 | 127 | 101 | 90 |
| 1e-5 | 2 | 95 | 82 | 65 |
| 5e-6 | 1 | 128 | 117 | 106 |
| 5e-6 | 2 | 105 | 95 | 81 |


Shared triples quantify annotation-specific consistency at the same LR/epoch. Disagreement is output variability, not evidence that either material label is visually correct. Full pair lists and annotation IDs are in private `analysis.json`.

At the same LR/epoch and score .05, annotation-kind agreement across seeds is
55/75 and 56/75 for 1e-5 E1/E2, versus 65/75 and 66/75 for 5e-6. Both seeds
match the same 20/21 annotations at 1e-5 E1/E2 and 22 at each 5e-6 epoch;
the remainder include stable failures and differing successes. Greater agreement
does not imply higher AP or prove that lower LR is generally more robust.
Wrong-class-only outcomes occur for the same 16/22 annotations at high-LR E1/E2,
and 19/23 at low-LR E1/E2. Wrapper→film, Styrofoam→film, bag→film and can→bottle
wrong-class overlapping queries occur in all eight trained states under these
conditions. This supports recurring annotation-relative assignments, while
their actual visual correctness remains unconfirmed.

### Most frequent wrong assignment for each GT class

Counts below use top100, score >=.05 and IoU >=.50. Each cell is predicted class: query count / distinct GT annotations; none means no such query, not absence of all class errors. Duplicate queries can dominate counts. All pairs remain in local JSON.

cigarette

| State | Predicted class | Queries | GT annotations |
| --- | --- | --- | --- |
| Pretrained | none | 0 | 0 |
| 42 / 1e-5 / E1 | plastic_film | 1 | 1 |
| 42 / 1e-5 / E2 | plastic_film | 1 | 1 |
| 42 / 5e-6 / E1 | none | 0 | 0 |
| 42 / 5e-6 / E2 | plastic_film | 1 | 1 |
| 123 / 1e-5 / E1 | plastic_film | 1 | 1 |
| 123 / 1e-5 / E2 | none | 0 | 0 |
| 123 / 5e-6 / E1 | none | 0 | 0 |
| 123 / 5e-6 / E2 | none | 0 | 0 |


plastic_bottle

| State | Predicted class | Queries | GT annotations |
| --- | --- | --- | --- |
| Pretrained | plastic_wrapper | 19 | 7 |
| 42 / 1e-5 / E1 | plastic_film | 17 | 5 |
| 42 / 1e-5 / E2 | plastic_film | 23 | 7 |
| 42 / 5e-6 / E1 | plastic_film | 14 | 5 |
| 42 / 5e-6 / E2 | plastic_film | 16 | 4 |
| 123 / 1e-5 / E1 | plastic_film | 19 | 5 |
| 123 / 1e-5 / E2 | plastic_film | 16 | 7 |
| 123 / 5e-6 / E1 | plastic_film | 16 | 4 |
| 123 / 5e-6 / E2 | plastic_film | 11 | 4 |


plastic_bag

| State | Predicted class | Queries | GT annotations |
| --- | --- | --- | --- |
| Pretrained | plastic_wrapper | 6 | 2 |
| 42 / 1e-5 / E1 | plastic_film | 10 | 2 |
| 42 / 1e-5 / E2 | plastic_film | 10 | 2 |
| 42 / 5e-6 / E1 | plastic_film | 5 | 2 |
| 42 / 5e-6 / E2 | plastic_film | 8 | 2 |
| 123 / 1e-5 / E1 | plastic_film | 7 | 2 |
| 123 / 1e-5 / E2 | plastic_film | 4 | 2 |
| 123 / 5e-6 / E1 | plastic_film | 7 | 2 |
| 123 / 5e-6 / E2 | plastic_film | 8 | 2 |


plastic_wrapper

| State | Predicted class | Queries | GT annotations |
| --- | --- | --- | --- |
| Pretrained | plastic_bottle | 35 | 10 |
| 42 / 1e-5 / E1 | plastic_film | 46 | 11 |
| 42 / 1e-5 / E2 | plastic_film | 48 | 11 |
| 42 / 5e-6 / E1 | plastic_bottle | 27 | 8 |
| 42 / 5e-6 / E2 | plastic_film | 40 | 11 |
| 123 / 1e-5 / E1 | plastic_film | 44 | 11 |
| 123 / 1e-5 / E2 | plastic_film | 31 | 11 |
| 123 / 5e-6 / E1 | plastic_bottle | 21 | 8 |
| 123 / 5e-6 / E2 | plastic_film | 33 | 11 |


plastic_film

| State | Predicted class | Queries | GT annotations |
| --- | --- | --- | --- |
| Pretrained | plastic_wrapper | 11 | 3 |
| 42 / 1e-5 / E1 | plastic_bottle | 9 | 6 |
| 42 / 1e-5 / E2 | cigarette | 4 | 2 |
| 42 / 5e-6 / E1 | plastic_bottle | 9 | 6 |
| 42 / 5e-6 / E2 | plastic_cup | 6 | 6 |
| 123 / 1e-5 / E1 | plastic_cup | 6 | 4 |
| 123 / 1e-5 / E2 | plastic_wrapper | 3 | 2 |
| 123 / 5e-6 / E1 | plastic_cup | 8 | 6 |
| 123 / 5e-6 / E2 | plastic_bottle | 6 | 4 |


drink_can

| State | Predicted class | Queries | GT annotations |
| --- | --- | --- | --- |
| Pretrained | plastic_bottle | 33 | 5 |
| 42 / 1e-5 / E1 | plastic_bottle | 23 | 6 |
| 42 / 1e-5 / E2 | plastic_film | 21 | 6 |
| 42 / 5e-6 / E1 | plastic_bottle | 23 | 6 |
| 42 / 5e-6 / E2 | plastic_film | 22 | 6 |
| 123 / 1e-5 / E1 | plastic_film | 23 | 6 |
| 123 / 1e-5 / E2 | plastic_film | 16 | 6 |
| 123 / 5e-6 / E1 | plastic_bottle | 22 | 6 |
| 123 / 5e-6 / E2 | plastic_film | 18 | 6 |


plastic_cup

| State | Predicted class | Queries | GT annotations |
| --- | --- | --- | --- |
| Pretrained | plastic_bottle | 9 | 3 |
| 42 / 1e-5 / E1 | plastic_film | 9 | 4 |
| 42 / 1e-5 / E2 | plastic_film | 12 | 5 |
| 42 / 5e-6 / E1 | plastic_film | 6 | 4 |
| 42 / 5e-6 / E2 | plastic_film | 9 | 4 |
| 123 / 1e-5 / E1 | plastic_film | 9 | 4 |
| 123 / 1e-5 / E2 | plastic_film | 9 | 5 |
| 123 / 5e-6 / E1 | plastic_film | 7 | 4 |
| 123 / 5e-6 / E2 | plastic_film | 7 | 3 |


styrofoam_piece

| State | Predicted class | Queries | GT annotations |
| --- | --- | --- | --- |
| Pretrained | plastic_film | 16 | 3 |
| 42 / 1e-5 / E1 | plastic_film | 29 | 4 |
| 42 / 1e-5 / E2 | plastic_film | 23 | 4 |
| 42 / 5e-6 / E1 | plastic_film | 20 | 4 |
| 42 / 5e-6 / E2 | plastic_film | 28 | 4 |
| 123 / 1e-5 / E1 | plastic_film | 22 | 4 |
| 123 / 1e-5 / E2 | plastic_film | 13 | 4 |
| 123 / 5e-6 / E1 | plastic_film | 18 | 4 |
| 123 / 5e-6 / E2 | plastic_film | 23 | 4 |


## Score distribution

All 42,300 raw queries/state; scores are sigmoid-derived and not accuracy or calibrated probabilities. Quantiles and >= counts are computed under the same conditions. Absolute score shrinkage alone does not explain AP changes under a zero-floor rank-based protocol.


| State | Median | P90 | P99 | Maximum | >=.05 | >=.10 | >=.25 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 0.019574 | 0.053525 | 0.187558 | 0.596656 | 4740 | 1524 | 216 |
| 42 / 1e-5 / E1 | 0.008768 | 0.024703 | 0.107758 | 0.343288 | 1472 | 483 | 16 |
| 42 / 1e-5 / E2 | 0.008959 | 0.024777 | 0.091586 | 0.376415 | 1272 | 358 | 54 |
| 42 / 5e-6 / E1 | 0.011358 | 0.029667 | 0.113798 | 0.341122 | 1743 | 532 | 21 |
| 42 / 5e-6 / E2 | 0.011610 | 0.028919 | 0.099546 | 0.354828 | 1569 | 418 | 15 |
| 123 / 1e-5 / E1 | 0.010990 | 0.027801 | 0.096743 | 0.355953 | 1478 | 400 | 26 |
| 123 / 1e-5 / E2 | 0.008852 | 0.023990 | 0.079393 | 0.396704 | 1068 | 271 | 37 |
| 123 / 5e-6 / E1 | 0.012747 | 0.032048 | 0.109008 | 0.344854 | 1848 | 516 | 25 |
| 123 / 5e-6 / E2 | 0.011343 | 0.027572 | 0.092980 | 0.359745 | 1407 | 366 | 21 |


## Human verification preparation

Twenty purposively selected validation images span the listed prompts; this is not a prevalence sample. No photo-level judgement has been made automatically. Selection uses saved matching/box-size metadata. Potential annotation issues are questions about unmatched boxes and material/extent ambiguity, not confirmed errors. No annotations are edited. All nine states are relevant for each entry; paired comparisons hold LR and epoch fixed. Review all GT objects and compare raw phrases with the assigned evaluation class.



| Image ID | GT categories | Selection reasons | Specific human question |
| --- | --- | --- | --- |
| 777 | cigarette, plastic_wrapper | COCO-small object; checkpoint-dependent detection; duplicate candidates; focus class: cigarette; focus class: plastic_wrapper; incorrect-class overlapping predictions; localized candidate without correct class; paired seed success/failure contrast; persistently missed at diagnostic 0.05; unmatched prediction: possible extra object or false positive; annotation check | GT259 (plastic_wrapper): confirm object presence, material and box extent; GT278 (cigarette): confirm object presence, material and box extent; GT279 (cigarette): confirm object presence, material and box extent; GT280 (cigarette): confirm object presence, material and box extent; GT281 (cigarette): confirm object presence, material and box extent; GT282 (cigarette): confirm object presence, material and box extent; GT283 (cigarette): confirm object presence, material and box extent; GT284 (cigarette): confirm object presence, material and box extent; Check GT278: seed42 poor_localization_candidate versus seed123 wrong_class_only at LR 1e-5, E1; At original-pixel zoom, is the small object actually distinguishable?; Do the overlapping same-class boxes describe one object or separate objects?; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 144 | cigarette, plastic_film | COCO-small object; checkpoint-dependent detection; consistent detection across all eight trained states; duplicate candidates; focus class: cigarette; focus class: plastic_film; incorrect-class overlapping predictions; localized candidate without correct class; persistently missed at diagnostic 0.05; unmatched prediction: possible extra object or false positive; annotation check | GT55 (cigarette): confirm object presence, material and box extent; GT56 (cigarette): confirm object presence, material and box extent; GT57 (cigarette): confirm object presence, material and box extent; GT58 (plastic_film): confirm object presence, material and box extent; GT59 (plastic_film): confirm object presence, material and box extent; GT60 (plastic_film): confirm object presence, material and box extent; GT61 (plastic_film): confirm object presence, material and box extent; Check GT58: seed42 no_substantial_overlap versus seed123 poor_localization_candidate at LR 1e-5, E1; At original-pixel zoom, is the small object actually distinguishable?; Do the overlapping same-class boxes describe one object or separate objects?; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 155 | cigarette, plastic_film | COCO-small object; checkpoint-dependent detection; focus class: cigarette; focus class: plastic_film; incorrect-class overlapping predictions; localized candidate without correct class; paired seed success/failure contrast; persistently missed at diagnostic 0.05; unmatched prediction: possible extra object or false positive; annotation check | GT64 (plastic_film): confirm object presence, material and box extent; GT65 (cigarette): confirm object presence, material and box extent; Check GT64: seed42 matched versus seed123 wrong_class_only at LR 1e-5, E2; At original-pixel zoom, is the small object actually distinguishable?; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 208 | styrofoam_piece | checkpoint-dependent detection; focus class: styrofoam_piece; incorrect-class overlapping predictions; localized candidate without correct class; paired seed success/failure contrast; unmatched prediction: possible extra object or false positive; annotation check | GT88 (styrofoam_piece): confirm object presence, material and box extent; Check GT88: seed42 matched versus seed123 wrong_class_only at LR 1e-5, E1; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 31 | drink_can | incorrect-class overlapping predictions; localized candidate without correct class; persistently missed at diagnostic 0.05; unmatched prediction: possible extra object or false positive; annotation check | GT9 (drink_can): confirm object presence, material and box extent; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 833 | plastic_bag | checkpoint-dependent detection; duplicate candidates; focus class: plastic_bag; incorrect-class overlapping predictions; localized candidate without correct class; paired seed success/failure contrast; unmatched prediction: possible extra object or false positive; annotation check | GT303 (plastic_bag): confirm object presence, material and box extent; Check GT303: seed42 wrong_class_only versus seed123 matched at LR 1e-5, E2; Do the overlapping same-class boxes describe one object or separate objects?; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 42 | drink_can | incorrect-class overlapping predictions; localized candidate without correct class; persistently missed at diagnostic 0.05; unmatched prediction: possible extra object or false positive; annotation check | GT16 (drink_can): confirm object presence, material and box extent; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 165 | cigarette | consistent detection across all eight trained states; duplicate candidates; focus class: cigarette; incorrect-class overlapping predictions; unmatched prediction: possible extra object or false positive; annotation check | GT72 (cigarette): confirm object presence, material and box extent; Do the overlapping same-class boxes describe one object or separate objects?; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 776 | plastic_bag | checkpoint-dependent detection; focus class: plastic_bag; incorrect-class overlapping predictions; localized candidate without correct class; paired seed success/failure contrast; unmatched prediction: possible extra object or false positive; annotation check | GT258 (plastic_bag): confirm object presence, material and box extent; Check GT258: seed42 matched versus seed123 wrong_class_only at LR 1e-5, E1; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 200 | plastic_wrapper | checkpoint-dependent detection; duplicate candidates; focus class: plastic_wrapper; incorrect-class overlapping predictions; localized candidate without correct class; paired seed success/failure contrast; unmatched prediction: possible extra object or false positive; annotation check | GT86 (plastic_wrapper): confirm object presence, material and box extent; Check GT86: seed42 wrong_class_only versus seed123 matched at LR 1e-5, E1; Do the overlapping same-class boxes describe one object or separate objects?; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 157 | plastic_film | focus class: plastic_film; localized candidate without correct class; persistently missed at diagnostic 0.05; unmatched prediction: possible extra object or false positive; annotation check | GT67 (plastic_film): confirm object presence, material and box extent; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 236 | styrofoam_piece | checkpoint-dependent detection; duplicate candidates; focus class: styrofoam_piece; incorrect-class overlapping predictions; localized candidate without correct class; paired seed success/failure contrast; unmatched prediction: possible extra object or false positive; annotation check | GT96 (styrofoam_piece): confirm object presence, material and box extent; Check GT96: seed42 matched versus seed123 wrong_class_only at LR 1e-5, E1; Do the overlapping same-class boxes describe one object or separate objects?; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 113 | plastic_bottle | checkpoint-dependent detection; incorrect-class overlapping predictions; localized candidate without correct class; paired seed success/failure contrast; unmatched prediction: possible extra object or false positive; annotation check | GT27 (plastic_bottle): confirm object presence, material and box extent; Check GT27: seed42 wrong_class_only versus seed123 matched at LR 1e-5, E2; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 1096 | plastic_bottle | checkpoint-dependent detection; duplicate candidates; incorrect-class overlapping predictions; paired seed success/failure contrast; unmatched prediction: possible extra object or false positive; annotation check | GT414 (plastic_bottle): confirm object presence, material and box extent; Check GT414: seed42 matched versus seed123 outside_top100 at LR 1e-5, E2; Do the overlapping same-class boxes describe one object or separate objects?; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 252 | plastic_wrapper | consistent detection across all eight trained states; duplicate candidates; focus class: plastic_wrapper; incorrect-class overlapping predictions; unmatched prediction: possible extra object or false positive; annotation check | GT104 (plastic_wrapper): confirm object presence, material and box extent; Do the overlapping same-class boxes describe one object or separate objects?; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 1014 | plastic_film | focus class: plastic_film; localized candidate without correct class; persistently missed at diagnostic 0.05; unmatched prediction: possible extra object or false positive; annotation check | GT367 (plastic_film): confirm object presence, material and box extent; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 849 | cigarette, plastic_bottle | COCO-small object; checkpoint-dependent detection; duplicate candidates; focus class: cigarette; incorrect-class overlapping predictions; localized candidate without correct class; paired seed success/failure contrast; persistently missed at diagnostic 0.05; unmatched prediction: possible extra object or false positive; annotation check | GT305 (plastic_bottle): confirm object presence, material and box extent; GT313 (cigarette): confirm object presence, material and box extent; GT314 (cigarette): confirm object presence, material and box extent; GT315 (cigarette): confirm object presence, material and box extent; GT316 (cigarette): confirm object presence, material and box extent; Check GT305: seed42 matched versus seed123 wrong_class_only at LR 1e-5, E1; At original-pixel zoom, is the small object actually distinguishable?; Do the overlapping same-class boxes describe one object or separate objects?; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 1161 | cigarette, plastic_bottle | COCO-small object; checkpoint-dependent detection; duplicate candidates; focus class: cigarette; incorrect-class overlapping predictions; localized candidate without correct class; paired seed success/failure contrast; persistently missed at diagnostic 0.05; unmatched prediction: possible extra object or false positive; annotation check | GT444 (plastic_bottle): confirm object presence, material and box extent; GT495 (cigarette): confirm object presence, material and box extent; GT496 (cigarette): confirm object presence, material and box extent; GT497 (cigarette): confirm object presence, material and box extent; GT498 (cigarette): confirm object presence, material and box extent; GT499 (cigarette): confirm object presence, material and box extent; GT500 (cigarette): confirm object presence, material and box extent; Check GT444: seed42 matched versus seed123 wrong_class_only at LR 1e-5, E1; At original-pixel zoom, is the small object actually distinguishable?; Do the overlapping same-class boxes describe one object or separate objects?; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 419 | plastic_wrapper | checkpoint-dependent detection; duplicate candidates; focus class: plastic_wrapper; incorrect-class overlapping predictions; localized candidate without correct class; paired seed success/failure contrast; unmatched prediction: possible extra object or false positive; annotation check | GT162 (plastic_wrapper): confirm object presence, material and box extent; Check GT162: seed42 matched versus seed123 wrong_class_only at LR 1e-5, E1; Do the overlapping same-class boxes describe one object or separate objects?; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |
| 691 | cigarette, drink_can | checkpoint-dependent detection; focus class: cigarette; incorrect-class overlapping predictions; localized candidate without correct class; paired seed success/failure contrast; persistently missed at diagnostic 0.05; unmatched prediction: possible extra object or false positive; annotation check | GT220 (drink_can): confirm object presence, material and box extent; GT260 (cigarette): confirm object presence, material and box extent; GT261 (cigarette): confirm object presence, material and box extent; Check GT220: seed42 matched versus seed123 wrong_class_only at LR 1e-5, E1; Are unmatched candidates real unannotated objects, localization errors or background? Do not edit labels? |


The full local manifest includes annotation IDs/boxes, exact model lists, per-annotation diagnostic kinds and the review question. Human findings remain pending; use explicit notes to distinguish them from automated classifications.

### Viewer use and limitations

The historical default replay is pinned to its old source/provenance policy and cannot select corrected runs. It is left intact, including any pre-existing source-identity rejection after trainer changes. The explicit `--saved-review` plus `--run-choice` path verifies local hashes, validation/rights identity, corrected protocol, original checkpoint contracts and saved query membership. No checkpoint parameters are loaded and no inference occurs. Four panels remain GT + pretrained + E1 + E2; switch run by relaunching with another key. The window title identifies the selected seed/LR and a selectable Saved configuration tab and `analysis.json` retain the actual complete saved configuration. Existing navigation, zoom/pan, diagnostic controls, raw-phrase tables and explicit human notes are reused. No annotated images are saved; display status remains separate.

Run from the repository root in PowerShell (quotes around the run key are required):

```powershell
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --gui --saved-review .\runs\training\cross-seed-analysis-fd10923e40d24c40be5bab3ae0fa2c63\review-manifest.json --run-choice "123 / 1e-5" --image-id 777
```

Other run keys: `42 / 1e-5`, `42 / 5e-6`, `123 / 5e-6`. Omit `--gui` for verification-only preparation. The representative subset contains these 20 IDs; all 47 remain accessible via All validation. Native automated runtime checks are separate from human visual acceptance.

## Confirmed observations and unresolved hypotheses

Confirmed from saved records: crossover in LR/epoch AP ranking; reduced upper raw scores versus pretrained; class-dependent candidate coverage and rare-class sensitivity; inconsistent bag/styrofoam preservation; systematic E1→E2 AP increases for cigarette/wrapper/film; neither losses nor a score maximum chooses the best detector. Consistency is measured only for this fixed two-seed validation set.

Unresolved: overfitting, catastrophic forgetting, causal effects of freezing/backbone optimization, material ambiguity, actual annotation omissions and the generalization of per-class gains. These require additional controlled evidence or human review. No cause is assigned to low AP, overlapping predictions or the recorded runtime gaps.

## Ranked next directions — proposals only


| Rank | Option | Unresolved question | Cost | Validation-selection risk |
| --- | --- | --- | --- | --- |
| 1 | D: data quality / class imbalance investigation | Are rare-class support, ambiguous materials or incomplete labels driving unstable macro metrics? Begin with this human manifest; no dataset edits authorized. | Low compute; human time | Lower if rubric fixed and annotations not tuned to model outputs; audit can still bias later decisions |
| 2 | A: one additional paired seed | Does the seed-dependent LR/epoch crossover recur? Prespecify seed456, both rates, same two epochs and endpoints. | Two complete runs /884 full updates plus pilots if approved | Moderate: no new knobs, but another look at the same validation set |
| 3 | C: classification-focused experiment | Does one justified change improve same-class coverage while preserving any-class localization? First establish actual material errors. | Design-dependent; needs paired controls | High if taxonomy/loss/prompt choices follow individual validation failures |
| 4 | B: shorter training / checkpoint selection | Do earlier points offer repeatable gains? Existing best epoch already flips by seed. | Extra saved/evaluated checkpoints or new bounded runs | High: dense epoch search can overfit 47-image validation |


Recommended next action is the prepared human data/material review (D), without changing labels. If an additional training experiment is subsequently authorized, recommend A: a prespecified third paired seed, with the current protocol and fixed endpoints. It answers the clearest unresolved replication question before changing classification objectives or searching checkpoint times. None of A–D is implemented here. Test images remain unused. Stop for owner review.

## Software and native replay verification

All four explicit saved-run choices loaded successfully with 47 validation images
and 42,300 raw queries per model, with no photo decoding during metadata-only
preparation. The actual terminal CLI also exited 0 for seed123 / 5e-6, producing
a separate local review record and opening no GUI. The unchanged default routing
and image-ID guards are covered by existing fixtures; the historical source-pinned
default was not requalified or claimed repaired by this extension.

The native automated Tk/TkAgg check used seed123 / 1e-5: rendered all 20 selected
images, navigated all47 scope, verified the four axes, top100/all900 filters,
thresholds 0 and .25, shared wheel zoom, saved-configuration tab and normal close.
It recorded 27 source-verified photo decodes including rerenders; original bytes
were unchanged. Image-save and figure-save calls were blocked during this check.
Display status persisted separately as closed, with zero annotated exports.
This is runtime evidence, not human confirmation of boxes or labels. The CLI
provides run selection by relaunch, not simultaneous cross-seed panels.

Focused tests: **22 passed, no skips/failures/errors** (18 existing diagnostics,
CLI/default/render/lifecycle checks plus four saved-review tests with schema,
workspace/hash, corpus, seed, checkpoint-position, metric, baseline and image-ID
tamper cases). The previous 214-test training qualification is historical; no
training/optimizer tests or model forwards were run in this task. Relative links,
Python syntax and Git diff whitespace were checked. Private runtime evidence is
`native-runtime.json`, `analysis.json`, `completion.json` and the saved review
records. All 40 checkpoint bundles were verified without loading their tensors.
