# Stage 4 — Corrected fine-tuning performance investigation

## 1. Executive summary

Diagnostic-only investigation, completed 2026-10-21, owner authorization D-049. No training,
optimizer updates, model loading, new forward passes or test-split access.

**Epoch 2's largest demonstrated failure is loss of correct-class candidates,
not a general loss of spatial box coverage.** All-900 any-class coverage at
IoU >=0.50 improves from 55 to 57 of 75 annotations, while same-class coverage
falls from 36 to 30. For both bag annotations and two previously detected
styrofoam annotations, well-aligned boxes remain, but no overlapping correctly
assigned-class candidate survives even among all 900 queries. Lowering a score
threshold cannot recover those missing class assignments.

AP drops from 0.203932 to 0.084500 and AR100 from 0.396766 to 0.211229.
Bag and styrofoam AP collapse to zero, and cup AP declines; other classes improve
or remain stable. Score changes are mixed: P99 decreases, but maximum score,
queries >=0.25 and matched-annotation count at 0.25 increase. Neither uniform
score suppression, a universal ranking collapse nor the old coordinate bug
explains the results.

This establishes output-level mechanisms, **not a proven optimization cause**.
Overfitting, catastrophic forgetting, an excessive learning rate and frozen
backbones remain hypotheses. A single-variable learning-rate comparison is the
recommended next proposal, subject to owner approval; it is not implemented.

## 2. Experimental comparability and provenance


The corrected run was discovered through
`work/corrected-retrain/experiment.json` → `training-result.json`, then checked
against the completed-run report and checkpoint audit:
`runs/training/taco-312e568703a54f98a7c8f4b87b29cd39`.
The historical directory is not used as the corrected model source.
Corrected epoch manifests identify global steps 221 and 442 and match their
previously recorded manifest hashes. The unchanged strict verifier checked all
checkpoint artifacts with their own original context; no state was restored.

Every historical/corrected contract field matches except the reviewed corrected
source fingerprints and already approved v2 evaluation protocol. No unintended
configuration difference was found. Both new evaluation metric/loss records match
the corresponding completed-run epoch summaries. The saved pretrained metrics,
loss, predictions and all raw queries exactly match the anchored corrected
baseline. COCO metrics were recomputed from saved predictions; no forward pass
was needed. New evaluation hashes are captured for this investigation's replay
and preservation audit; this is local integrity evidence, not a signed provenance
claim.

Fixed configuration: pretrained Grounding DINO Tiny revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71`, seed 42, batch 1, shortest/longest
256/384, fixed 384-square padding, FP32, AdamW LR 1e-5 / weight decay 1e-4,
foreach false, clip norm 1; frozen vision/text backbones; 221 training /
47 validation images; unchanged eight-class taxonomy/prompt; no augmentation,
scheduler, accumulation or workers. The three pilot updates were discarded
before loading the original pretrained snapshot for the full run. Sources,
original weights, dataset, rights record and native adapter/loss remain verified.
See [controlled-run evidence](TACO_STAGE_4_CORRECTED_RETRAIN.md) and
[coordinate correction](TACO_STAGE_4_COORDINATES.md).

All aggregate comparisons below use `bantai.taco-evaluation.v2`, mean ordinary
subword sigmoid class scores, single argmax class/query, no NMS, score floor zero,
top 100 valid predictions per image, COCO bbox IoU .50:.05:.95. All AP/recall values
are fractions on [0,1]. Diagnostic sweeps never modify these settings.


| Model | AP | AP50 | AP75 | AR100 |
| --- | --- | --- | --- | --- |
| Pretrained | 0.111576 | 0.158253 | 0.135975 | 0.363077 |
| Corrected E1 | 0.203932 | 0.254720 | 0.239987 | 0.396766 |
| Corrected E2 | 0.084500 | 0.107561 | 0.095599 | 0.211229 |

## 3. Confidence scores and candidate ranking

These sigmoid-derived model scores are not calibrated correctness probabilities
or accuracy. Each state has 42,300 raw queries (900 per image). Threshold counts
below use **>=**, including equality. Full raw/top-100 distributions and sweeps
are retained separately in private `diagnostics.json`.


| Model | Min | Median | P90 | P95 | P99 | Max |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 0.000902 | 0.019574 | 0.053525 | 0.081858 | 0.187558 | 0.596656 |
| Corrected E1 | 0.000888 | 0.008768 | 0.024703 | 0.038765 | 0.107758 | 0.343288 |
| Corrected E2 | 0.000949 | 0.008959 | 0.024777 | 0.037465 | 0.091586 | 0.376415 |

| Model | >=0.05 | >=0.10 | >=0.15 | >=0.20 | >=0.25 |
| --- | --- | --- | --- | --- | --- |
| Pretrained | 4740 | 1524 | 723 | 376 | 216 |
| Corrected E1 | 1472 | 483 | 206 | 86 | 16 |
| Corrected E2 | 1272 | 358 | 177 | 95 | 54 |

E2 suppresses part of the upper-middle score distribution: counts at 0.05,
0.10 and 0.15 and P95/P99 decline. Median/P90 rise slightly, and counts at 0.20/
0.25 and the maximum rise. Both trained states have lower median, P90, P95, P99 and maximum scores than
pretrained; their minimum-score changes are mixed. A global monotonic score rescaling would preserve
within-class AP ranking under this zero-floor protocol; absolute score shrinkage
alone does not explain the AP decline.

Per-class distributions below condition on each query's **winning assigned
class**, not its ground-truth class. Group membership changes across epochs,
so these are not matched-query probability trajectories. Non-winning class logits
were not saved; margins and alternate-class probabilities cannot be measured.


| Class | State | Queries | Median | P90 | P95 | P99 | Max | >=0.25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cigarette | Pretrained | 262 | 0.021084 | 0.055673 | 0.083591 | 0.152067 | 0.324497 | 1 |
| cigarette | Corrected E1 | 8865 | 0.007965 | 0.022476 | 0.032464 | 0.064710 | 0.239240 | 0 |
| cigarette | Corrected E2 | 22650 | 0.008920 | 0.023689 | 0.034109 | 0.066077 | 0.277122 | 1 |
| plastic_bottle | Pretrained | 4858 | 0.019380 | 0.062428 | 0.108329 | 0.277664 | 0.461679 | 56 |
| plastic_bottle | Corrected E1 | 2487 | 0.009432 | 0.035267 | 0.072063 | 0.206823 | 0.343288 | 7 |
| plastic_bottle | Corrected E2 | 1177 | 0.009576 | 0.039023 | 0.078254 | 0.248308 | 0.376415 | 12 |
| plastic_bag | Pretrained | 3514 | 0.022523 | 0.059508 | 0.092661 | 0.203323 | 0.412574 | 20 |
| plastic_bag | Corrected E1 | 1050 | 0.009622 | 0.027835 | 0.039985 | 0.158689 | 0.231075 | 0 |
| plastic_bag | Corrected E2 | 496 | 0.009241 | 0.022551 | 0.030831 | 0.065740 | 0.102404 | 0 |
| plastic_wrapper | Pretrained | 5814 | 0.018204 | 0.050583 | 0.078348 | 0.181197 | 0.416231 | 24 |
| plastic_wrapper | Corrected E1 | 3224 | 0.009047 | 0.024360 | 0.036471 | 0.094053 | 0.261348 | 4 |
| plastic_wrapper | Corrected E2 | 931 | 0.012726 | 0.040303 | 0.071428 | 0.236832 | 0.337949 | 9 |
| plastic_film | Pretrained | 2662 | 0.030332 | 0.097458 | 0.142371 | 0.273868 | 0.462033 | 39 |
| plastic_film | Corrected E1 | 10890 | 0.009198 | 0.030535 | 0.053353 | 0.140182 | 0.270841 | 3 |
| plastic_film | Corrected E2 | 10395 | 0.009117 | 0.027305 | 0.046146 | 0.145169 | 0.329006 | 29 |
| drink_can | Pretrained | 61 | 0.044719 | 0.149324 | 0.180304 | 0.210240 | 0.218516 | 0 |
| drink_can | Corrected E1 | 24 | 0.028594 | 0.146404 | 0.164710 | 0.215441 | 0.230461 | 0 |
| drink_can | Corrected E2 | 34 | 0.014632 | 0.084299 | 0.137256 | 0.266029 | 0.295823 | 1 |
| plastic_cup | Pretrained | 9490 | 0.019783 | 0.053311 | 0.081268 | 0.174150 | 0.430635 | 23 |
| plastic_cup | Corrected E1 | 3647 | 0.010780 | 0.035345 | 0.058014 | 0.132986 | 0.288202 | 2 |
| plastic_cup | Corrected E2 | 1416 | 0.010801 | 0.041478 | 0.064839 | 0.152669 | 0.289484 | 2 |
| styrofoam_piece | Pretrained | 15639 | 0.018243 | 0.044590 | 0.061787 | 0.138189 | 0.596656 | 53 |
| styrofoam_piece | Corrected E1 | 12113 | 0.008261 | 0.017961 | 0.025097 | 0.056126 | 0.227507 | 0 |
| styrofoam_piece | Corrected E2 | 5201 | 0.008177 | 0.016894 | 0.023761 | 0.056735 | 0.179966 | 0 |

At fixed score >=0.05, official top100, same-class greedy matching at IoU
>=0.50, the following groups show the score/IoU relationship. “Unmatched”
means no annotation overlap >=0.10 under this diagnostic, not human-confirmed
false positives. Full quantiles are in JSON. These are different sets of detections
across epochs, so their medians are not paired changes.


| State | Group | Queries | Median score | P99 score | Median IoU |
| --- | --- | --- | --- | --- | --- |
| Pretrained | matched | 28 | 0.297601 | 0.420369 | 0.815385 |
| Pretrained | duplicate | 45 | 0.121229 | 0.318151 | 0.677985 |
| Pretrained | wrong_class | 342 | 0.168583 | 0.414732 | 0.708160 |
| Pretrained | localization_overlap_010_to_050 | 987 | 0.091224 | 0.291777 | 0.259903 |
| Pretrained | unmatched_prediction | 2689 | 0.076354 | 0.346294 | 0.000000 |
| Corrected E1 | matched | 28 | 0.208072 | 0.287190 | 0.861255 |
| Corrected E1 | duplicate | 27 | 0.097995 | 0.203806 | 0.777833 |
| Corrected E1 | wrong_class | 296 | 0.133611 | 0.280071 | 0.784209 |
| Corrected E1 | localization_overlap_010_to_050 | 402 | 0.074539 | 0.191004 | 0.280760 |
| Corrected E1 | unmatched_prediction | 719 | 0.068098 | 0.201941 | 0.000000 |
| Corrected E2 | matched | 26 | 0.211218 | 0.333390 | 0.756292 |
| Corrected E2 | duplicate | 19 | 0.080809 | 0.183140 | 0.729866 |
| Corrected E2 | wrong_class | 215 | 0.136229 | 0.336697 | 0.802148 |
| Corrected E2 | localization_overlap_010_to_050 | 268 | 0.072702 | 0.199306 | 0.247749 |
| Corrected E2 | unmatched_prediction | 744 | 0.068405 | 0.209195 | 0.000000 |

Useful-candidate ranking uses the highest-scoring same-class candidate with IoU
>=0.50 for each annotation, not a claim that query IDs represent the same object
across checkpoints. Among 23 annotations with such a candidate
in both epochs, rank improves for 15, worsens for 3
and is unchanged for 5. All-900 same-class coverage loses 13
annotations and gains seven. Top100 same-class coverage loses 11 and gains nine.
Thus there are localized ranking failures, but disappearance of correctly assigned
useful candidates is stronger evidence than a blanket ranking-collapse claim.
Within-class dataset-wide ranking also affects AP and is not fully summarized by
these within-image ranks.


## 4. Per-class detection and class assignment

AP50 is recomputed from the same COCO precision tensor at IoU .50, all areas,
maxDets 100. AR100 uses the original multi-IoU definition. Class counts and
per-class matching counts below refer to annotations, not calibrated classification
accuracy.


| Class | Train objects/images | Val objects | Base AP | E1 AP | E2 AP |
| --- | --- | --- | --- | --- | --- |
| cigarette | 110/30 | 24 | 0.014851 | 0.020136 | 0.025458 |
| plastic_bottle | 44/35 | 8 | 0.038242 | 0.096181 | 0.103595 |
| plastic_bag | 9/9 | 2 | 0.236229 | 0.677759 | 0.000000 |
| plastic_wrapper | 58/53 | 11 | 0.094467 | 0.118247 | 0.135371 |
| plastic_film | 97/85 | 13 | 0.001574 | 0.016310 | 0.026429 |
| drink_can | 39/31 | 7 | 0.108416 | 0.118812 | 0.118812 |
| plastic_cup | 16/16 | 5 | 0.340656 | 0.432673 | 0.266337 |
| styrofoam_piece | 24/20 | 5 | 0.058173 | 0.151337 | 0.000000 |

| Class | Base AP50 | E1 AP50 | E2 AP50 | Base AR100 | E1 AR100 | E2 AR100 |
| --- | --- | --- | --- | --- | --- | --- |
| cigarette | 0.049505 | 0.050588 | 0.053022 | 0.012500 | 0.050000 | 0.079167 |
| plastic_bottle | 0.051105 | 0.136787 | 0.111259 | 0.400000 | 0.462500 | 0.337500 |
| plastic_bag | 0.297480 | 0.752475 | 0.000000 | 0.550000 | 0.950000 | 0.000000 |
| plastic_wrapper | 0.149874 | 0.144790 | 0.199483 | 0.490909 | 0.272727 | 0.372727 |
| plastic_film | 0.003556 | 0.026081 | 0.041275 | 0.076923 | 0.384615 | 0.446154 |
| drink_can | 0.143564 | 0.148515 | 0.148515 | 0.214286 | 0.114286 | 0.114286 |
| plastic_cup | 0.464008 | 0.616146 | 0.306931 | 0.600000 | 0.560000 | 0.340000 |
| styrofoam_piece | 0.106931 | 0.162376 | 0.000000 | 0.560000 | 0.380000 | 0.000000 |

| Class | Base matches .05 | E1 matches .05 | E2 matches .05 | Base matches .25 | E1 matches .25 | E2 matches .25 |
| --- | --- | --- | --- | --- | --- | --- |
| cigarette | 1 | 2 | 5 | 1 | 0 | 1 |
| plastic_bottle | 4 | 5 | 3 | 4 | 1 | 2 |
| plastic_bag | 2 | 2 | 0 | 1 | 0 | 0 |
| plastic_wrapper | 9 | 4 | 6 | 4 | 1 | 3 |
| plastic_film | 2 | 8 | 9 | 1 | 0 | 1 |
| drink_can | 2 | 1 | 1 | 0 | 0 | 1 |
| plastic_cup | 4 | 4 | 2 | 3 | 2 | 1 |
| styrofoam_piece | 4 | 2 | 0 | 4 | 0 | 0 |

All eight AP values improve from pretrained to E1. From E1 to E2, bag,
styrofoam and cup AP decline; cigarette, bottle, wrapper and film AP improve;
can AP is unchanged. Recall is mixed and is not implied by AP changes.
Bag AP's 0.677759→0 alone contributes about -0.084720 to the aggregate
-0.119432 AP change; styrofoam contributes -0.018917 and cup -0.020792.
Other classes offset part of that decline. Bag evidence rests on only two
validation annotations; macro-averaging gives it the same class weight as
cigarette's 24 annotations.

Recurring wrong-class overlaps at >=0.05 / IoU >=0.50 are shown below.
Counts are repeated queries, not independent mistakes or unique objects;
the best-overlapping annotation is the reference.


| GT → predicted | Base queries | E1 queries | E2 queries |
| --- | --- | --- | --- |
| plastic_wrapper → plastic_film | 13 | 46 | 48 |
| styrofoam_piece → plastic_film | 16 | 29 | 23 |
| plastic_bag → plastic_film | 1 | 10 | 10 |
| drink_can → plastic_bottle | 33 | 23 | 8 |
| drink_can → plastic_film | 6 | 22 | 21 |
| plastic_bottle → plastic_film | 10 | 17 | 23 |
| plastic_cup → plastic_film | 4 | 9 | 12 |

These are annotation-relative category confusions. Visual similarity of the
materials and annotation completeness still require human review; they are not
inferred from the category names alone.

## 5. Localization, duplicates and concrete regression cases

Candidate coverage counts each annotation with at least one qualifying candidate;
it is an oracle-availability measure, not one-to-one recall. All900 results include
very weak candidates outside official top100.


| Candidate scope | Base coverage .50/.75 | E1 coverage .50/.75 | E2 coverage .50/.75 | Base median best IoU | E1 median best IoU | E2 median best IoU |
| --- | --- | --- | --- | --- | --- | --- |
| best_any_all900 | 53/37 | 55/39 | 57/41 | 0.702275 | 0.770327 | 0.788216 |
| best_same_all900 | 32/21 | 36/23 | 30/17 | 0.233969 | 0.451579 | 0.324410 |
| best_any_top100 | 48/37 | 53/39 | 56/39 | 0.702275 | 0.770327 | 0.766352 |
| best_same_top100 | 28/21 | 31/21 | 29/16 | 0.053224 | 0.371666 | 0.063863 |

Paired per-annotation best any-class IoU improves for 50 annotations and
worsens for 25 (mean change +0.023826). Best same-class IoU improves for 41,
worsens for 27 and is unchanged for seven, but its mean change is -0.081919:
large class-specific losses outweigh many small gains. Matched-set median IoU
falls 0.861255→0.756292, while wrong-class-overlap median IoU rises
0.784209→0.802148. This combination supports class-assignment failures with
some localization degradation, not a universal inability to place boxes.

The following annotation-level cases isolate the distinction. Zero same-class
IoU means no correctly assigned candidate overlaps that annotation, not necessarily
that the class is never predicted anywhere in the image.


| Image | Annotation | Class | E1 best same IoU | E2 best same IoU | E2 best any IoU | E1 useful rank | E2 useful rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 208 | 88 | styrofoam_piece | 0.916965 | 0.000000 | 0.930837 | 2 | — |
| 236 | 96 | styrofoam_piece | 0.975801 | 0.000000 | 0.986003 | 4 | — |
| 263 | 106 | styrofoam_piece | 0.499241 | 0.000000 | 0.965105 | — | — |
| 464 | 177 | styrofoam_piece | 0.525028 | 0.347737 | 0.978424 | 335 | — |
| 776 | 258 | plastic_bag | 0.968015 | 0.000000 | 0.964086 | 2 | — |
| 833 | 303 | plastic_bag | 0.939137 | 0.000000 | 0.968691 | 3 | — |
| 906 | 324 | styrofoam_piece | 0.006491 | 0.002298 | 0.743943 | — | — |

| State | Invalid saved boxes | Boundary-touching boxes | Duplicates .05 | Wrong-class queries .05 | Low-overlap queries .05 | Unmatched queries .05 |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 0 | 7182 | 45 | 342 | 987 | 2689 |
| Corrected E1 | 0 | 6141 | 27 | 296 | 402 | 719 |
| Corrected E2 | 0 | 5374 | 19 | 215 | 268 | 744 |

Duplicate counts decrease at the fixed view threshold, so increased duplicate
count is not the measured primary regression. Counts depend on scores and do not
prove the underlying proposals ceased duplicating objects. All saved boxes are
valid and within original-image bounds. Boundary-touching boxes cannot be
classified as clipped versus naturally on-edge: the pre-clipping normalized
boxes were not saved. There is no evidence here that the previous coordinate
defect returned; current source identity and the independent coordinate tests pass.

## 6. Training and validation losses, gradients and numerical behavior

Both new epochs use corrected targets. Historical padded-target training losses
are not mixed into these comparisons. Classification means below are the native
CE/focal-loss components, not classification accuracy. Encoder and decoder
components retain their original coefficients (classification 2, L1 5, GIoU 2).


| Epoch | Train total | Val total | Decoder cls | Decoder L1 | Decoder GIoU | Encoder cls | Encoder L1 | Encoder GIoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8.569587 | 7.019476 | 1.596428 | 0.035419 | 0.236439 | 1.764385 | 0.081487 | 0.395276 |
| 2 | 6.903368 | 5.926001 | 1.220535 | 0.029247 | 0.218400 | 1.452690 | 0.064501 | 0.325692 |

| Epoch | Preclip norm min | Median | P99 | Max | Clipped steps |
| --- | --- | --- | --- | --- | --- |
| 1 | 37.510464 | 114.219307 | 867.433313 | 1463.960449 | 221 |
| 2 | 33.288670 | 84.110298 | 993.875891 | 1619.742065 | 221 |

Every training loss component decreases in mean. The preclip gradient norm is
large and variable, and all 442 steps invoke norm-1 clipping; the median falls
while the maximum increases. Saved losses, norms and encoder probe updates are
finite. The unchanged trainer audits all trainable gradients/updated weights,
positive encoder-head updates and frozen hashes; the completed run passed.
These observations do not establish numerical divergence or an excessive learning
rate. Two encoder probes are recorded, not a complete per-layer gradient history.

Lower total validation loss can coexist with worse AP because the weighted native
matching/token/box objective is different from ranked, macro-averaged, multi-IoU
detection AP. Improvements on common targets/background or box losses can coexist
with loss of correct class assignments on rare classes. That is a supported
explanation of the mismatch, not a measured decomposition of validation loss.
**Per-image validation component losses and logits were not saved**, so their
contributions cannot be recovered from total loss alone. No additional forward
pass was necessary for the conclusions above. Existing
[training/validation curves](assets/taco_stage4_corrected_retrain_curves.svg)
remain unchanged.

## 7. Representative local visual-review cases

The existing four-panel `ComparisonWindow` is reused without editing any existing
source. A local launcher verifies this investigation's recorded hashes and corrected
run manifests before passing the original validation image, ground truth, pretrained
rows and corrected epoch rows to it. The window title explicitly identifies the
corrected run; historical epoch predictions are never substituted. Existing
`taco_visual_review.py` terminal default and `--gui` behavior remain byte-identical.

Local PowerShell command, from the repository root:

```powershell
.\.venv-training\Scripts\python.exe -B .\work\corrected-diagnosticsiewer.py --gui --image-id 776
```

Without `--gui`, this launcher prints verified corrected-run AP/AR100 only.
All 47 validation IDs remain accessible. Diagnostic filters, zoom/pan, boxes,
scores, raw-phrase tables and explicit human notes reuse the existing GUI.
No annotated-image copies are saved. The launcher and evidence are local ignored
artifacts, not a published replacement entry point.

The deterministic sample covers class-size extremes, E1 improvements, E2
regressions, missing/wrong-class/duplicate/weak candidates, plus every bag/styrofoam
validation image. It is not a prevalence sample. Reasons below are automated;
**human photo-level confirmation remains pending for every case**. Native rendering
checks establish source/overlay identity and controls, not correctness of visible
material labels or completeness of annotations. No image/screenshot was uploaded
for agent interpretation.


| Validation ID | Automated selection reason |
| --- | --- |
| 155 | smallest normalized bbox for cigarette |
| 1250 | smallest normalized bbox for plastic_bottle |
| 776 | smallest normalized bbox for plastic_bag; bag/styrofoam epoch-2 AP collapse |
| 407 | smallest normalized bbox for plastic_wrapper |
| 144 | smallest normalized bbox for plastic_film; epoch1 improves over pretrained at score floor zero |
| 1182 | smallest normalized bbox for drink_can |
| 1076 | smallest normalized bbox for plastic_cup |
| 906 | smallest normalized bbox for styrofoam_piece; bag/styrofoam epoch-2 AP collapse |
| 777 | most annotated objects; most threshold-hidden annotations at .25 |
| 42 | pretrained beats epoch1 at score floor zero |
| 113 | epoch2 deteriorates versus epoch1 at score floor zero |
| 464 | largest wrong_class count across models at diagnostic .05; bag/styrofoam epoch-2 AP collapse |
| 466 | largest localization_overlap_010_to_050 count across models at diagnostic .05; largest duplicate count across models at diagnostic .05 |
| 1014 | largest unmatched_prediction count across models at diagnostic .05 |
| 1161 | median normalized bbox for cigarette |
| 1243 | median normalized bbox for plastic_bottle |
| 208 | bag/styrofoam epoch-2 AP collapse |
| 236 | bag/styrofoam epoch-2 AP collapse |
| 263 | bag/styrofoam epoch-2 AP collapse |
| 833 | bag/styrofoam epoch-2 AP collapse |

## 8. Confirmed observations, supported hypotheses and unresolved questions

**Confirmed:** experiment settings and sources match the intended corrected run;
any-class box availability improves while same-class availability declines;
rare-class AP collapses explain most aggregate AP loss; score shifts are mixed;
all training loss means decrease; no recorded numerical failure; diagnostics
and native replay use only authorized validation images/saved predictions.

**Supported hypotheses, not established causes:**

- Optimization trajectory sensitivity: one additional epoch at the fixed recipe
  changes assignments and loses rare-class coverage despite better losses.
  A learning-rate comparison can test sensitivity; “too high” is not yet measured.
- Class imbalance: training annotations range from 9 bags to 110 cigarettes;
  bag, styrofoam and cup have limited support. E2 winning-query composition shifts
  strongly toward cigarette (8,865→22,650), while styrofoam falls
  12,113→5,201 and bag 1,050→496. This is consistent with assignment drift,
  but counts of weak queries are not object prevalence or proof of imbalance
  causing the regression.
- Resolution limits: actual recorded resize geometry gives 14 of 75 annotation
  boxes a short side below four pixels, including two below two pixels
  (minimum 1.631031). Small-object difficulty is plausible, but unchanged resolution
  alone cannot explain the temporal E1→E2 class collapse.
- Frozen-backbone capacity may limit adaptation; no frozen/unfrozen comparison
  exists, so this remains weakly supported as a cause of this regression.

**Unresolved:** overfitting versus optimization/class-assignment drift, learning-rate
sensitivity, seed variance, per-class logit margins, validation component changes,
unclipped box geometry, and human material/annotation judgments. Lower validation
loss does not rule out AP overfitting, but it also does not prove it. Forgetting
of the model's broader pretrained capabilities was not evaluated; “catastrophic
forgetting” is not established.

## 9. Preservation and verification


The saved-only audit verified 603 protected file hashes
before and after analysis: original authorized photos, annotations/splits, snapshot,
source/configuration/test files, private rights assessment, historical results and
checkpoints, corrected run and earlier corrected-run evidence. No original file
was rewritten. Strict training-resume verification is unchanged.

On continuation on 2026-10-21, the checkout was at external commit `c79d513`.
A fresh comparison of all 603 prior protected files found 602 unchanged and one
pre-existing difference: `config/detection.json`. Its current bytes are preserved.
This production inference configuration is not the direct eight-class training
prompt or the saved evaluation protocol. The local diagnostic launcher now checks
an exact reviewed prior/current hash pair for this file, recorded in
`continuation-2026-10-21.json`; every other protected file must retain its original
hash. No generalized source exception, training-resume change, historical manifest
rewrite or original configuration edit was introduced. The initial replay attempt
correctly stopped on this difference before opening any photograph. The earlier
20-image native completion below predates this continuation; it is not human
confirmation.
The final terminal-only corrected replay passed on 2026-10-21, printing the
verified pretrained/E1/E2 AP and AR100 with exit code 0 and no image rendering.

Software checks: eight existing coordinate tests, 18 existing viewer/diagnostic
tests and 19 existing provenance tests passed; three new local diagnostic fixtures
passed (quantiles/threshold equality, class-versus-geometry/rank separation, literal
per-class AP50 oracle). Initial module-style provenance test invocation hit its
existing sibling-import requirement; rerunning with `unittest discover -s tests`
passed all 19 without source edits. No optimizer-bearing test was run.

Native corrected-viewer checks: passed 20 selected validation images, source overlays, raw-phrase tables, shared pan/wheel zoom, all900 mode and all47 navigation; normal closure.
Human visual confirmation remains pending. No new inference or training was used.

Private evidence directory: `runs/training/corrected-diagnostics-629e30fb299f442698de14650b279c6c`.
`diagnostics.json` contains all score/class distributions and losses;
`annotation-detail.json` contains every annotation's candidate IoUs/ranks and paired
changes; `matching-detail.json` preserves all six diagnostic score floors;
`protected-before.json` and `verification.json` record integrity. Native display
evidence is separate under `viewer-*/display.json`. Stored scores and raw phrases
are never replaced with inferred confidence. Exact JSON values are authoritative
over rounded tables. Rights uncertainties remain as documented in
[the unchanged limited-use assessment](TACO_STAGE_4_ACADEMIC_USE.md).

## 10. Recommended next controlled experiment — proposal only

| Rank | Single changed factor | Diagnostic value | Complexity / limit |
| --- | --- | --- | --- |
| 1 | Learning rate 1e-5 → 5e-6 | Tests whether reduced step size preserves E2 class coverage and detection metrics | Low code change; same two epochs, seed, optimizer, data and remaining settings; does not alone prove overfitting |
| 2 | Seed only, retain the original corrected recipe | Tests whether rare-class collapse and E1 advantage repeat | Low implementation cost; another full bounded run; one extra seed still gives limited uncertainty estimates |
| 3 | Shortest-edge resize only, with longest edge/padding unchanged | Tests small-object visibility sensitivity | Memory preflight required; affected images depend on aspect ratio; not the leading explanation for bag/styrofoam collapse |
| 4 | One explicitly designed sampling-balance policy only | Tests class-support sensitivity | More design complexity with mixed-class images; repeated-image exposure must be controlled and recorded |
| 5 | One backbone-freezing policy only | Tests representational limitations | Highest memory/implementation cost; little direct evidence that it explains this epoch-specific failure |

For proposal 1, start again from the original pinned pretrained snapshot, keep the
same 221/47 split, two epochs, batch 1, 256/384 resize/padding, seed 42, frozen
backbones, losses, weight decay, clipping and corrected v2 evaluation. Retain the
three-step discarded pilot. Compare both epochs with this completed run using AP,
AR100, each class's AP/recall, score-floor-zero same-class coverage and fixed
0.05/0.25 matches, especially bags/styrofoam/cups. A recovered E2 outcome would
support learning-rate sensitivity; failure would weaken that particular remedy,
not identify a unique cause. Do not judge success solely by loss, score maxima or
two rare bag annotations.

**Stop:** diagnostic analysis is complete. No proposed experiment has been
implemented, no checkpoint promoted, and no commit/push/merge performed.
Owner review and approval are required before any new training.
