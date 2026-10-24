# Stage 4 — Small-object and classification root-cause audit

D-054, 2026-10-22. Read-only model investigation on the full **47 validation images /75 annotations**, plus separately identified examples from the existing 20-image review. No new model inference, training, optimizer updates, dataset/protocol edits or GUI changes. Findings below are measurements of saved outputs, hypotheses and owner-supplied human observations, explicitly separated.

## 1. Executive summary

**Small resized size is strongly associated with poor saved candidate coverage, while class assignment independently limits many well-localized larger objects. Neither association proves a unique training root cause.** At score >=.05, the <4-pixel group has any-class coverage 0–2/14 across nine states, compared with35–36/36 in the >=16-pixel group. Cigarettes have shortest sides1.631–9.818 pixels, with13/24 below4,7/24 at4–<8,4/24 at8–<16 and none>=16. Their same-class coverage is0–2/13,0–2/7 and1–2/4 respectively, depending on state. The last group has only four examples and three belong to image777.

At all900 /score>=.05 /IoU>=.50, **20/75 pretrained annotations and21–27/75 annotations per trained state** have any-class localization but no qualifying correctly assigned class. These are annotation-relative output classifications, not human-verified material errors. Cigarettes instead predominantly lack any qualifying localization:18–23/24 per state, with zero wrong-class-only cases at this threshold in every state. Score filtering hides some otherwise useful low-score candidates, so these counts do not show an exclusively spatial or visibility failure.

The most informative next controlled experiment is **inference-only input-scale comparison on one fixed checkpoint**, at existing256/384 versus proposed384/576, with compatible square padding declared as part of the input-scale policy. Keep weights,47 images, labels, prompt,900 queries, FP32, decoding, diagnostic .05/.50 and COCO v2 fixed. It directly tests whether additional input detail improves tiny-object coverage, without seed/training changes. This is a proposal only; no new forward pass was executed. Higher resolution is not expected by itself to resolve all large-object class errors.

## 2. Evidence and verified resizing/padding

Read the six prior Stage4 reports: [corrected run](TACO_STAGE_4_CORRECTED_RETRAIN.md), [initial diagnostics](TACO_STAGE_4_CORRECTED_DIAGNOSTICS.md), [LR comparison](TACO_STAGE_4_LR_EXPERIMENT.md), [seed replication](TACO_STAGE_4_SEED_REPLICATION.md), [cross-seed](TACO_STAGE_4_CROSS_SEED_ANALYSIS.md) and [per-object](TACO_STAGE_4_PER_OBJECT_REVIEW.md). Work occurred in the existing local feature branch and working tree; no GitHub synchronization or publication is assumed.

The pinned local processor was loaded offline, without model weights. The actual `get_batch`→`collate` implementation processed each source-verified validation image once on CPU at256/384 with384-square padding. A neural-module execution guard, optimizer-step guard, image-save guard and network guard were active during processing. Every padding mask exactly equalled a top-left rectangular valid-content region; its sum equalled valid width×height. All 75 corrected targets agreed with an independent original-to-valid-pixel box transform within0.001 pixels. Maximum observed error: 0.000044319 pixels. All 47 class-index/token-span order checks passed. No image was treated as stretched384-square content.

The processor uses aspect-ratio resizing, caps the longest edge, rounds the capped shorter size and floors the calculated other dimension. Integer rounding can make x/y scale factors slightly different. Effective padding is on right and bottom only. For original (W,H), valid (Rw,Rh): sx=Rw/W, sy=Rh/H; [x,y,w,h] maps to [x·sx,y·sy,w·sx,h·sy]. Corrected loss targets divide centers/widths by original W and heights by original H; multiplying them by valid dimensions reproduces resized geometry. Padding384×384 is used for storage/attention masks, not usable content area.


| Actual valid W×H | Images | Right/bottom padding |
| --- | --- | --- |
| 175×384 | 6 | 209 / 0 |
| 256×272 | 1 | 128 / 112 |
| 256×341 | 26 | 128 / 43 |
| 341×256 | 9 | 43 / 128 |
| 350×256 | 1 | 34 / 128 |
| 384×175 | 2 | 0 / 209 |
| 384×216 | 1 | 0 / 168 |
| 384×256 | 1 | 0 / 128 |


Private reproducibility evidence: `runs\training\small-object-audit-c3e654439bc94e7b8d34f9080e216874`. `preprocessing.json` and `validation-image-geometry.csv` retain all 47 original/resized/padded sizes, pad extents, sx/sy, valid area and target errors. `validation-object-sizes.csv` contains all 75 original/resized xywh boxes, bbox area, saved segmentation area and valid-area fractions. `candidate-audit.json` and `validation-object-coverage.csv` retain all675 GT/state combinations under four source/floor conditions (2700 records): all900/top100 ×0/.05. Original review manifest is unchanged. Local hashes establish consistency, not signed provenance.

## 3. Object-size distribution by class

Fixed groups use shortest resized **bounding-box** side: <4;4–<8;8–<16;>=16 pixels. These are diagnostic bins, not COCO small/medium/large or physical object dimensions. Bbox area=w×h; it differs from saved polygon/segmentation area. Resized bbox area fraction uses valid Rw×Rh, excluding padding. Tiny width, occlusion, texture and material ambiguity cannot be inferred from area alone.


| Class | Val GT | Images | Train GT metadata | <4 | 4–<8 | 8–<16 | >=16 | Shortest side min/Q1/median/Q3/max px | Resized bbox area min/Q1/median/Q3/max px² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cigarette | 24 | 7 | 110 | 13 | 7 | 4 | 0 | 1.631 / 2.766 / 3.345 / 5.958 / 9.818 | 3.914 / 17.241 / 21.501 / 48.112 / 132.099 |
| plastic_bottle | 8 | 8 | 44 | 0 | 2 | 2 | 4 | 4.078 / 11.242 / 18.405 / 33.686 / 111.590 | 54.965 / 199.012 / 500.181 / 1374.371 / 22136.079 |
| plastic_bag | 2 | 2 | 9 | 0 | 0 | 0 | 2 | 39.282 / 51.762 / 64.242 / 76.722 / 89.203 | 2263.457 / 3757.151 / 5250.845 / 6744.539 / 8238.233 |
| plastic_wrapper | 11 | 11 | 58 | 0 | 0 | 1 | 10 | 12.460 / 23.042 / 36.960 / 55.090 / 113.231 | 301.587 / 1053.910 / 1709.197 / 3675.033 / 17486.641 |
| plastic_film | 13 | 9 | 97 | 1 | 0 | 5 | 7 | 2.592 / 12.096 / 18.928 / 31.163 / 206.277 | 13.926 / 211.989 / 577.942 / 1084.160 / 53905.119 |
| drink_can | 7 | 7 | 39 | 0 | 0 | 2 | 5 | 11.072 / 16.673 / 31.146 / 57.767 / 70.787 | 218.251 / 460.626 / 1658.709 / 5616.805 / 8116.834 |
| plastic_cup | 5 | 4 | 16 | 0 | 0 | 1 | 4 | 15.373 / 16.105 / 18.492 / 18.492 / 40.471 | 339.864 / 387.050 / 433.166 / 525.987 / 1657.409 |
| styrofoam_piece | 5 | 5 | 24 | 0 | 1 | 0 | 4 | 7.000 / 48.451 / 54.888 / 144.941 / 161.401 | 96.623 / 3234.044 / 3748.961 / 25348.453 / 30975.788 |


Small cells, especially bags(2 GT), cans/cups/Styrofoam and the four >=8-pixel cigarettes, cannot support broad causal claims. Train counts are metadata only; no training photos were processed in this audit. Class and size are confounded: the extremely-small group contains13 cigarettes and one film annotation. Cigarettes lack a >=16-pixel comparison group.

## 4. Complete validation coverage by resized size

Saved states are the unchanged pretrained baseline plus E1/E2 from each explicit42/123 ×1e-5/5e-6 run. Existing saved loader verifies artifact hashes, source corpus/rights, corrected protocol and epoch checkpoints against original contracts. All four pretrained raw-query sets are identical. Each official metric record was already independently recomputed in the prior cross-seed analysis; this audit does not alter AP or define a new official evaluation.

Coverage is per GT, non-exclusive, at IoU>=.50; candidates can cover more than one annotation. Best overlap maximizes positive IoU then score then lowest query ID. A positive-IoU candidate is not necessarily adequately localized. A missing candidate means absent under these saved/score/source conditions, never model detection impossibility. Best IoU is recorded as0 when no positive overlap exists; candidate class/score stay null. Image-level existence of a passing query is explicitly separate from overlap/IoU>=.50 coverage for a particular annotation. All900 have one stored argmax class/query; alternative class logits, unsaved proposals/decoder layers and masks cannot be recovered. Top100 is an official rank limit, not the limit on available diagnostics.

### Any / correct-class coverage at score >=0.05


| State | extremely small <4 (N=14) | very small 4–<8 (N=10) | small 8–<16 (N=15) | larger ≥16 (N=36) |
| --- | --- | --- | --- | --- |
| Pretrained | 0 / 0 | 3 / 0 | 9 / 4 | 36 / 24 |
| 42 / 1e-5 / E1 | 0 / 0 | 4 / 1 | 10 / 6 | 35 / 21 |
| 42 / 1e-5 / E2 | 1 / 1 | 5 / 2 | 11 / 7 | 35 / 16 |
| 42 / 5e-6 / E1 | 0 / 0 | 3 / 0 | 10 / 5 | 35 / 20 |
| 42 / 5e-6 / E2 | 0 / 0 | 4 / 1 | 11 / 6 | 35 / 18 |
| 123 / 1e-5 / E1 | 1 / 1 | 4 / 1 | 10 / 5 | 35 / 18 |
| 123 / 1e-5 / E2 | 2 / 2 | 5 / 2 | 11 / 7 | 35 / 15 |
| 123 / 5e-6 / E1 | 0 / 0 | 3 / 0 | 10 / 5 | 35 / 20 |
| 123 / 5e-6 / E2 | 0 / 0 | 4 / 1 | 10 / 5 | 35 / 19 |


### Any / correct-class coverage at score >=0.0


| State | extremely small <4 (N=14) | very small 4–<8 (N=10) | small 8–<16 (N=15) | larger ≥16 (N=36) |
| --- | --- | --- | --- | --- |
| Pretrained | 1 / 1 | 4 / 1 | 12 / 4 | 36 / 26 |
| 42 / 1e-5 / E1 | 1 / 1 | 4 / 1 | 14 / 8 | 36 / 26 |
| 42 / 1e-5 / E2 | 2 / 2 | 5 / 2 | 14 / 9 | 36 / 17 |
| 42 / 5e-6 / E1 | 1 / 1 | 4 / 1 | 13 / 5 | 36 / 25 |
| 42 / 5e-6 / E2 | 1 / 1 | 5 / 1 | 14 / 8 | 36 / 20 |
| 123 / 1e-5 / E1 | 2 / 2 | 5 / 1 | 15 / 7 | 36 / 22 |
| 123 / 1e-5 / E2 | 3 / 3 | 6 / 3 | 14 / 9 | 36 / 19 |
| 123 / 5e-6 / E1 | 1 / 1 | 4 / 1 | 13 / 7 | 36 / 23 |
| 123 / 5e-6 / E2 | 1 / 1 | 6 / 2 | 14 / 7 | 36 / 21 |


The smallest bins have much lower any-class coverage than larger bins in every state, including at floor zero. Intermediate bins are not guaranteed to form a strictly monotonic class-correct progression: large objects can still be assigned the wrong class. Repeated states are not independent observations; no pooled675-object significance claim is made.

### Cigarettes separately (24 annotations)


| State | <4 N13 any/same | 4–<8 N7 any/same | 8–<16 N4 any/same | Cigarette same at floor0 /24 |
| --- | --- | --- | --- | --- |
| Pretrained | 0/0 | 0/0 | 1/1 | 3 |
| 42 / 1e-5 / E1 | 0/0 | 1/1 | 1/1 | 5 |
| 42 / 1e-5 / E2 | 1/1 | 2/2 | 2/2 | 8 |
| 42 / 5e-6 / E1 | 0/0 | 0/0 | 1/1 | 3 |
| 42 / 5e-6 / E2 | 0/0 | 1/1 | 2/2 | 5 |
| 123 / 1e-5 / E1 | 1/1 | 1/1 | 1/1 | 6 |
| 123 / 1e-5 / E2 | 2/2 | 2/2 | 2/2 | 9 |
| 123 / 5e-6 / E1 | 0/0 | 0/0 | 1/1 | 3 |
| 123 / 5e-6 / E2 | 0/0 | 1/1 | 1/1 | 6 |


Within cigarettes, <4px correct coverage is0–15.4%,4–<8px0–28.6%,8–<16px25–50%. This is a strong descriptive association with missed candidates, consistent with a small-input-detail limitation, but the last denominator is four and annotations/scene/score/class factors co-vary. It is not proof that changing resolution will recover the objects. Individual cigarette values are listed below rather than extrapolating a size law.


| Image | GT | Resized bbox W×H | Shortest px | Group | Valid bbox-area fraction |
| --- | --- | --- | --- | --- | --- |
| 144 | 55 | 5.565×1.920 | 1.920 | extremely small <4 | 0.00015899 |
| 144 | 56 | 2.686×8.352 | 2.686 | extremely small <4 | 0.00033388 |
| 144 | 57 | 2.782×7.296 | 2.782 | extremely small <4 | 0.00030208 |
| 155 | 65 | 1.631×2.400 | 1.631 | extremely small <4 | 0.00005825 |
| 165 | 72 | 11.424×8.155 | 8.155 | small 8–<16 | 0.00138638 |
| 691 | 260 | 4.810×6.477 | 4.810 | very small 4–<8 | 0.00035693 |
| 691 | 261 | 6.275×3.343 | 3.343 | extremely small <4 | 0.00024029 |
| 777 | 278 | 6.909×12.364 | 6.909 | very small 4–<8 | 0.00095336 |
| 777 | 279 | 7.273×13.818 | 7.273 | very small 4–<8 | 0.00112161 |
| 777 | 280 | 8.000×9.455 | 8.000 | small 8–<16 | 0.00084416 |
| 777 | 281 | 9.818×13.455 | 9.818 | small 8–<16 | 0.00147432 |
| 777 | 282 | 2.909×6.545 | 2.909 | extremely small <4 | 0.00021251 |
| 777 | 283 | 2.545×6.182 | 2.545 | extremely small <4 | 0.00017562 |
| 777 | 284 | 13.455×8.727 | 8.727 | small 8–<16 | 0.00131051 |
| 849 | 313 | 2.719×6.686 | 2.719 | extremely small <4 | 0.00020825 |
| 849 | 314 | 2.928×6.059 | 2.928 | extremely small <4 | 0.00020325 |
| 849 | 315 | 4.183×3.343 | 3.343 | extremely small <4 | 0.00016019 |
| 849 | 316 | 5.020×4.388 | 4.388 | very small 4–<8 | 0.00025231 |
| 1161 | 495 | 2.301×2.925 | 2.301 | extremely small <4 | 0.00007709 |
| 1161 | 496 | 3.346×4.597 | 3.346 | extremely small <4 | 0.00017621 |
| 1161 | 497 | 6.275×4.806 | 4.806 | very small 4–<8 | 0.00034542 |
| 1161 | 498 | 5.647×3.343 | 3.343 | extremely small <4 | 0.00021626 |
| 1161 | 499 | 4.392×5.433 | 4.392 | very small 4–<8 | 0.00027333 |
| 1161 | 500 | 6.902×5.642 | 5.642 | very small 4–<8 | 0.00044604 |


## 5. Localization, class assignment and score/rank effects

Primary disjoint states: correct-class candidate; any-class IoU>=.50 but no same-class candidate; no any-class IoU>=.50, all at score>=.05. The last state is a failure of **qualifying** localization and can include useful boxes below score floor. Annotation uncertainty is an independent human-review attribute, not a fourth automatic label assigned to pixels.


| State | Correct /75 | Class-related /75 | No qualifying localization /75 | Any /75 | Any floor0 | Same floor0 | Same hidden by score | Any hidden by score | Same lost outside top100 at .05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 28 | 20 | 27 | 48 | 53 | 32 | 4 | 5 | 0 |
| 42 / 1e-5 / E1 | 28 | 21 | 26 | 49 | 55 | 36 | 8 | 6 | 0 |
| 42 / 1e-5 / E2 | 26 | 26 | 23 | 52 | 57 | 30 | 4 | 5 | 0 |
| 42 / 5e-6 / E1 | 25 | 23 | 27 | 48 | 54 | 32 | 7 | 6 | 0 |
| 42 / 5e-6 / E2 | 25 | 25 | 25 | 50 | 56 | 30 | 5 | 6 | 0 |
| 123 / 1e-5 / E1 | 25 | 25 | 25 | 50 | 58 | 32 | 7 | 8 | 0 |
| 123 / 1e-5 / E2 | 26 | 27 | 22 | 53 | 59 | 34 | 8 | 6 | 0 |
| 123 / 5e-6 / E1 | 25 | 23 | 27 | 48 | 54 | 32 | 7 | 6 | 0 |
| 123 / 5e-6 / E2 | 25 | 24 | 26 | 49 | 57 | 31 | 6 | 8 | 0 |


Across the eight trained states, 37 distinct validation annotations show the class-related failure in at least one state, and 11 show it in all eight. These are unions/intersections over the same75 GTs, not independent samples. Stable IDs: 9, 16, 19, 223, 242, 256, 324, 372, 388, 503, 527.

At .05, no correct-class IoU>=.50 coverage is lost by moving from all900 to official top100 in any state. Rank exclusion can still matter at floor0, while the default five-box drawing cap can hide qualifying queries as documented earlier. Of the 36 larger objects,14–20 per trained state have localized-but-no-correct-class coverage at .05; resolution alone is not an established remedy for these cases.

### Class-specific consistency


| Class | N | Trained wrong-class-only min–max | Trained no-any min–max | Wrong-only all8 GT IDs | No-any all8 GT IDs |
| --- | --- | --- | --- | --- | --- |
| cigarette | 24 | 0–0 | 18–23 | [] | [55, 56, 57, 65, 260, 278, 279, 280, 281, 282, 283, 314, 315, 495, 496, 497, 498, 499] |
| plastic_bottle | 8 | 2–6 | 0–0 | [223, 527] | [] |
| plastic_bag | 2 | 0–2 | 0–0 | [] | [] |
| plastic_wrapper | 11 | 5–8 | 0–0 | [242, 256] | [] |
| plastic_film | 13 | 0–4 | 4–4 | [] | [58, 61, 67, 367] |
| drink_can | 7 | 6–7 | 0–0 | [9, 16, 19, 372, 503] | [] |
| plastic_cup | 5 | 1–3 | 0–0 | [388] | [] |
| styrofoam_piece | 5 | 1–5 | 0–0 | [324] | [] |


Cigarette failures are consistently dominated by absent qualifying any-class coverage. Cans, wrappers, bottles and Styrofoam repeatedly have localized candidates without a qualifying correct class; bags also suffer but have only two annotations. Film combines class-related failures with four consistently absent any-class cases. This supports distinct output mechanisms, not a common confirmed implementation defect.

### Descriptive class-confusion summary

For each GT/model, count only the single best-overlap qualifying any-class candidate when no correct-class candidate qualifies. This yields at most one confusion entry/GT/state and excludes duplicate-query inflation. If the best-overlap class differs but another valid same-class candidate exists, it remains correctly covered and is excluded from the failure confusion table. Ground-truth labels remain provisional human material references.

#### cigarette


| State | GT N | Class-related failures | Best-any winning classes among these failures |
| --- | --- | --- | --- |
| Pretrained | 24 | 0 | none |
| 42 / 1e-5 / E1 | 24 | 0 | none |
| 42 / 1e-5 / E2 | 24 | 0 | none |
| 42 / 5e-6 / E1 | 24 | 0 | none |
| 42 / 5e-6 / E2 | 24 | 0 | none |
| 123 / 1e-5 / E1 | 24 | 0 | none |
| 123 / 1e-5 / E2 | 24 | 0 | none |
| 123 / 5e-6 / E1 | 24 | 0 | none |
| 123 / 5e-6 / E2 | 24 | 0 | none |


#### plastic_bottle


| State | GT N | Class-related failures | Best-any winning classes among these failures |
| --- | --- | --- | --- |
| Pretrained | 8 | 4 | plastic_cup:2, plastic_wrapper:2 |
| 42 / 1e-5 / E1 | 8 | 3 | plastic_cup:2, plastic_wrapper:1 |
| 42 / 1e-5 / E2 | 8 | 5 | plastic_cup:2, plastic_film:3 |
| 42 / 5e-6 / E1 | 8 | 3 | plastic_cup:2, plastic_wrapper:1 |
| 42 / 5e-6 / E2 | 8 | 3 | plastic_cup:2, plastic_wrapper:1 |
| 123 / 1e-5 / E1 | 8 | 4 | plastic_cup:1, plastic_film:2, plastic_wrapper:1 |
| 123 / 1e-5 / E2 | 8 | 6 | cigarette:1, plastic_cup:2, plastic_film:3 |
| 123 / 5e-6 / E1 | 8 | 2 | plastic_cup:1, plastic_wrapper:1 |
| 123 / 5e-6 / E2 | 8 | 2 | plastic_cup:1, plastic_wrapper:1 |


#### plastic_bag


| State | GT N | Class-related failures | Best-any winning classes among these failures |
| --- | --- | --- | --- |
| Pretrained | 2 | 0 | none |
| 42 / 1e-5 / E1 | 2 | 0 | none |
| 42 / 1e-5 / E2 | 2 | 2 | plastic_film:2 |
| 42 / 5e-6 / E1 | 2 | 1 | plastic_bottle:1 |
| 42 / 5e-6 / E2 | 2 | 1 | plastic_film:1 |
| 123 / 1e-5 / E1 | 2 | 1 | plastic_film:1 |
| 123 / 1e-5 / E2 | 2 | 1 | cigarette:1 |
| 123 / 5e-6 / E1 | 2 | 1 | plastic_cup:1 |
| 123 / 5e-6 / E2 | 2 | 1 | plastic_film:1 |


#### plastic_wrapper


| State | GT N | Class-related failures | Best-any winning classes among these failures |
| --- | --- | --- | --- |
| Pretrained | 11 | 2 | plastic_bag:2 |
| 42 / 1e-5 / E1 | 11 | 7 | plastic_cup:2, plastic_film:5 |
| 42 / 1e-5 / E2 | 11 | 5 | plastic_film:5 |
| 42 / 5e-6 / E1 | 11 | 7 | plastic_bag:1, plastic_bottle:4, plastic_film:1, styrofoam_piece:1 |
| 42 / 5e-6 / E2 | 11 | 8 | cigarette:2, plastic_bag:1, plastic_bottle:4, styrofoam_piece:1 |
| 123 / 1e-5 / E1 | 11 | 6 | plastic_bag:1, plastic_bottle:2, plastic_film:3 |
| 123 / 1e-5 / E2 | 11 | 6 | cigarette:1, plastic_bottle:2, plastic_film:3 |
| 123 / 5e-6 / E1 | 11 | 5 | plastic_bag:1, plastic_bottle:1, plastic_cup:1, plastic_film:2 |
| 123 / 5e-6 / E2 | 11 | 6 | plastic_bag:1, plastic_bottle:2, plastic_film:2, styrofoam_piece:1 |


#### plastic_film


| State | GT N | Class-related failures | Best-any winning classes among these failures |
| --- | --- | --- | --- |
| Pretrained | 13 | 7 | plastic_bag:1, plastic_cup:4, styrofoam_piece:2 |
| 42 / 1e-5 / E1 | 13 | 1 | plastic_cup:1 |
| 42 / 1e-5 / E2 | 13 | 0 | none |
| 42 / 5e-6 / E1 | 13 | 3 | cigarette:1, plastic_cup:2 |
| 42 / 5e-6 / E2 | 13 | 2 | plastic_cup:2 |
| 123 / 1e-5 / E1 | 13 | 1 | plastic_cup:1 |
| 123 / 1e-5 / E2 | 13 | 1 | plastic_cup:1 |
| 123 / 5e-6 / E1 | 13 | 4 | plastic_cup:4 |
| 123 / 5e-6 / E2 | 13 | 2 | plastic_cup:2 |


#### drink_can


| State | GT N | Class-related failures | Best-any winning classes among these failures |
| --- | --- | --- | --- |
| Pretrained | 7 | 5 | plastic_bag:2, plastic_cup:3 |
| 42 / 1e-5 / E1 | 7 | 6 | cigarette:1, plastic_bottle:3, plastic_cup:1, plastic_film:1 |
| 42 / 1e-5 / E2 | 7 | 6 | cigarette:1, plastic_bottle:2, plastic_cup:1, plastic_film:2 |
| 42 / 5e-6 / E1 | 7 | 7 | cigarette:1, plastic_bottle:1, plastic_cup:4, plastic_film:1 |
| 42 / 5e-6 / E2 | 7 | 6 | plastic_bottle:1, plastic_cup:3, plastic_film:2 |
| 123 / 1e-5 / E1 | 7 | 7 | cigarette:1, plastic_cup:2, plastic_film:3, plastic_wrapper:1 |
| 123 / 1e-5 / E2 | 7 | 6 | cigarette:4, plastic_bottle:1, plastic_film:1 |
| 123 / 5e-6 / E1 | 7 | 7 | plastic_bag:1, plastic_bottle:1, plastic_cup:3, plastic_film:1, plastic_wrapper:1 |
| 123 / 5e-6 / E2 | 7 | 7 | plastic_bottle:2, plastic_cup:1, plastic_film:3, plastic_wrapper:1 |


#### plastic_cup


| State | GT N | Class-related failures | Best-any winning classes among these failures |
| --- | --- | --- | --- |
| Pretrained | 5 | 1 | plastic_bag:1 |
| 42 / 1e-5 / E1 | 5 | 1 | cigarette:1 |
| 42 / 1e-5 / E2 | 5 | 3 | cigarette:1, plastic_bottle:1, plastic_film:1 |
| 42 / 5e-6 / E1 | 5 | 1 | plastic_film:1 |
| 42 / 5e-6 / E2 | 5 | 1 | plastic_film:1 |
| 123 / 1e-5 / E1 | 5 | 1 | plastic_film:1 |
| 123 / 1e-5 / E2 | 5 | 2 | plastic_film:2 |
| 123 / 5e-6 / E1 | 5 | 1 | plastic_film:1 |
| 123 / 5e-6 / E2 | 5 | 1 | plastic_film:1 |


#### styrofoam_piece


| State | GT N | Class-related failures | Best-any winning classes among these failures |
| --- | --- | --- | --- |
| Pretrained | 5 | 1 | plastic_cup:1 |
| 42 / 1e-5 / E1 | 5 | 3 | plastic_cup:1, plastic_wrapper:2 |
| 42 / 1e-5 / E2 | 5 | 5 | plastic_cup:1, plastic_film:2, plastic_wrapper:2 |
| 42 / 5e-6 / E1 | 5 | 1 | plastic_cup:1 |
| 42 / 5e-6 / E2 | 5 | 4 | plastic_cup:1, plastic_film:3 |
| 123 / 1e-5 / E1 | 5 | 5 | plastic_cup:1, plastic_film:3, plastic_wrapper:1 |
| 123 / 1e-5 / E2 | 5 | 5 | cigarette:1, plastic_cup:1, plastic_film:2, plastic_wrapper:1 |
| 123 / 5e-6 / E1 | 5 | 3 | plastic_cup:1, plastic_film:1, plastic_wrapper:1 |
| 123 / 5e-6 / E2 | 5 | 5 | plastic_bottle:1, plastic_film:3, plastic_wrapper:1 |


### Implementation defect versus model/annotation limitation

Source audit and focused fixtures found no class-index/span permutation in the qualified batch1 path: Stage2 category IDs and dataset labels follow the eight-label tuple; collate decodes every class span and checks order on every sample; training adapter and evaluation both use native `build_label_maps`; native matched classification targets select those maps by GT class ID; evaluation averages ordinary subword sigmoid scores then selects one argmax class/query. A synthetic all-eight target-binding test and existing all-class decoder/geometry oracle passed. No real logits or loss gradients were generated.

Native Hungarian matching combines class cost, L1 and GIoU under existing coefficients. Supervision is token-level focal loss for matched queries. Partially shared words such as plastic occur at distinct prompt positions; lexical overlap and annotation material ambiguity remain hypotheses, not a demonstrated mapping bug. Raw decoded phrase uses a separate token threshold and can be incomplete or include several words; it must not be substituted for assigned evaluation class. No library or model fixes were made. Saved winner-only outputs cannot reveal margins between alternative classes or reconstruct counterfactual matching, so a subtle implementation/optimization cause is not ruled out.

## 6. Cross-seed consistency


| LR | Epoch | Any both | Same both | Same only42 | Same only123 | Class failure both |
| --- | --- | --- | --- | --- | --- | --- |
| 1e-5 | 1 | 49 | 20 | 8 | 5 | 17 |
| 1e-5 | 2 | 52 | 21 | 5 | 5 | 22 |
| 5e-6 | 1 | 48 | 22 | 3 | 3 | 20 |
| 5e-6 | 2 | 49 | 22 | 3 | 3 | 22 |


The extreme-small-versus-large coverage gap repeats across both seeds/rates/epochs; class-specific errors repeat despite aggregate LR/epoch AP crossover. Two seeds cannot establish population variability or resolution causation. Aggregate AP rankings do not substitute for per-object identities or coverage.

## 7. Image777 reference case and representativeness

Owner-supplied human review in the current request reports: wrapper localization with a different assigned category; multiple cigarettes failing IoU>=.50; GT284 qualifying after E2 in both inspected1e-5 runs; several cigarette identities/boundaries uncertain. This is owner feedback, not new agent visual inspection or confirmation that every GT material is correct. No annotation IDs were specifically confirmed uncertain. The original human report is retained separately as `human-review-input.json`.


| GT | Category | Original bbox | Actual resized bbox W×H | Group | Valid-area fraction |
| --- | --- | --- | --- | --- | --- |
| 259 | plastic_wrapper | [975.0, 81.0, 116.0, 263.0] | 21.091×47.818 | larger ≥16 | 0.01125590 |
| 278 | cigarette | [1086.0, 984.0, 38.0, 68.0] | 6.909×12.364 | very small 4–<8 | 0.00095336 |
| 279 | cigarette | [1032.0, 936.0, 40.0, 76.0] | 7.273×13.818 | very small 4–<8 | 0.00112161 |
| 280 | cigarette | [1040.0, 772.0, 44.0, 52.0] | 8.000×9.455 | small 8–<16 | 0.00084416 |
| 281 | cigarette | [1112.0, 954.0, 54.0, 74.0] | 9.818×13.455 | small 8–<16 | 0.00147432 |
| 282 | cigarette | [1124.0, 786.0, 16.0, 36.0] | 2.909×6.545 | extremely small <4 | 0.00021251 |
| 283 | cigarette | [1016.0, 900.0, 14.0, 34.0] | 2.545×6.182 | extremely small <4 | 0.00017562 |
| 284 | cigarette | [1622.0, 294.0, 74.0, 48.0] | 13.455×8.727 | small 8–<16 | 0.00131051 |


Image777 is originally1925×1408 and its valid content is350×256 after resize, with34px right and128px bottom padding. Its seven cigarettes span shortest sides2.545–9.818px; wrapper is21.091×47.818px. It contains2/13 of all <4px cigarettes,2/7 at4–<8 and3/4 at8–<16. Thus image777 is biased toward the larger cigarette bins relative to the remaining17 cigarette annotations; it cannot stand in for all validation objects. The wrapper is in the larger group, so its persistent wrong class cannot be explained merely by being below4 or8 pixels. Resolution could still affect material cues, but its limitation is unproven.


| State | 777 cigarette same /7 | Other cigarette same /17 | 777 any /7 | Other any /17 |
| --- | --- | --- | --- | --- |
| Pretrained | 0 | 1 | 0 | 1 |
| 42 / 1e-5 / E1 | 0 | 2 | 0 | 2 |
| 42 / 1e-5 / E2 | 1 | 4 | 1 | 4 |
| 42 / 5e-6 / E1 | 0 | 1 | 0 | 1 |
| 42 / 5e-6 / E2 | 1 | 2 | 1 | 2 |
| 123 / 1e-5 / E1 | 0 | 3 | 0 | 3 |
| 123 / 1e-5 / E2 | 1 | 5 | 1 | 5 |
| 123 / 5e-6 / E1 | 0 | 1 | 0 | 1 |
| 123 / 5e-6 / E2 | 0 | 2 | 0 | 2 |


GT282/283 have no same-class IoU>=.50 at floor0 in any state. In seed123/1e-5/E2, GT284 passes .05/.50 but is hidden by the standard five-box display; GT278 and GT280 have same-class qualifying geometry only below .05. The full per-object report contains exact query/score/IoU identities for all eight GTs. The current audit checks full validation rather than treating that example as a confirmed resolution bottleneck.

## 8. Representative cases from the unchanged20-image review

Quantitative claims above use all 47/75. The following purposive examples only prepare balanced human review. Eligible examples are restricted to the existing 20 IDs. Visible ambiguity is not automatically inferred from box size or failure: the only supplied human uncertainty is the unspecified cigarette subset of image777. Other entries ask questions rather than declaring defects.


| Image /GT | Class | Resized W×H px | Selection evidence | Models | Human question |
| --- | --- | --- | --- | --- | --- |
| 144 / 55 | cigarette | 5.565×1.920 | Extremely small; no any-class IoU>=.50 even at floor0 in every trained state | 42 / 1e-5 / E1; 42 / 1e-5 / E2; 42 / 5e-6 / E1; 42 / 5e-6 / E2; 123 / 1e-5 / E1; 123 / 1e-5 / E2; 123 / 5e-6 / E1; 123 / 5e-6 / E2 | Is there a separate identifiable object with a reliable box/material label at original pixels? |
| 144 / 56 | cigarette | 2.686×8.352 | Extremely small; no any-class IoU>=.50 even at floor0 in every trained state | 42 / 1e-5 / E1; 42 / 1e-5 / E2; 42 / 5e-6 / E1; 42 / 5e-6 / E2; 123 / 1e-5 / E1; 123 / 1e-5 / E2; 123 / 5e-6 / E1; 123 / 5e-6 / E2 | Is there a separate identifiable object with a reliable box/material label at original pixels? |
| 144 / 57 | cigarette | 2.782×7.296 | Extremely small; no any-class IoU>=.50 even at floor0 in every trained state | 42 / 1e-5 / E1; 42 / 1e-5 / E2; 42 / 5e-6 / E1; 42 / 5e-6 / E2; 123 / 1e-5 / E1; 123 / 1e-5 / E2; 123 / 5e-6 / E1; 123 / 5e-6 / E2 | Is there a separate identifiable object with a reliable box/material label at original pixels? |
| 691 / 261 | cigarette | 6.275×3.343 | Small successful counterexample at .05/.50 | 123 / 1e-5 / E2 | Does the qualifying box describe the annotated object rather than adjacent debris? |
| 849 / 313 | cigarette | 2.719×6.686 | Small successful counterexample at .05/.50 | 42 / 1e-5 / E2; 123 / 1e-5 / E1; 123 / 1e-5 / E2 | Does the qualifying box describe the annotated object rather than adjacent debris? |
| 849 / 316 | cigarette | 5.020×4.388 | Small successful counterexample at .05/.50 | 42 / 1e-5 / E1; 42 / 1e-5 / E2; 42 / 5e-6 / E2; 123 / 1e-5 / E1; 123 / 1e-5 / E2; 123 / 5e-6 / E2 | Does the qualifying box describe the annotated object rather than adjacent debris? |
| 31 / 9 | drink_can | 31.146×53.255 | Larger object with localization but no correct-class candidate | 42 / 1e-5 / E1; 42 / 1e-5 / E2; 42 / 5e-6 / E1; 42 / 5e-6 / E2; 123 / 1e-5 / E1; 123 / 1e-5 / E2; 123 / 5e-6 / E1; 123 / 5e-6 / E2 | Confirm material identity; does the predicted class name another object or reflect real ambiguity? Size alone is unlikely to explain missing localization because it is already adequate. |
| 42 / 16 | drink_can | 70.787×114.665 | Larger object with localization but no correct-class candidate | 42 / 1e-5 / E1; 42 / 1e-5 / E2; 42 / 5e-6 / E1; 42 / 5e-6 / E2; 123 / 1e-5 / E1; 123 / 1e-5 / E2; 123 / 5e-6 / E1; 123 / 5e-6 / E2 | Confirm material identity; does the predicted class name another object or reflect real ambiguity? Size alone is unlikely to explain missing localization because it is already adequate. |
| 776 / 258 | plastic_bag | 57.621×39.282 | Larger object with localization but no correct-class candidate | 42 / 1e-5 / E2; 42 / 5e-6 / E1; 42 / 5e-6 / E2; 123 / 1e-5 / E1; 123 / 1e-5 / E2; 123 / 5e-6 / E1; 123 / 5e-6 / E2 | Confirm material identity; does the predicted class name another object or reflect real ambiguity? Size alone is unlikely to explain missing localization because it is already adequate. |
| 777 / 259 | plastic_wrapper | 21.091×47.818 | Larger object with localization but no correct-class candidate | 42 / 1e-5 / E1; 42 / 5e-6 / E1; 42 / 5e-6 / E2; 123 / 1e-5 / E1; 123 / 1e-5 / E2; 123 / 5e-6 / E1; 123 / 5e-6 / E2 | Confirm material identity; does the predicted class name another object or reflect real ambiguity? Size alone is unlikely to explain missing localization because it is already adequate. |
| 776 / 258 | plastic_bag | 57.621×39.282 | Focus material: recurring annotation-relative class failure | 42 / 1e-5 / E2; 42 / 5e-6 / E1; 42 / 5e-6 / E2; 123 / 1e-5 / E1; 123 / 1e-5 / E2; 123 / 5e-6 / E1; 123 / 5e-6 / E2 | Is the annotated material distinct from film/wrapper/bag/Styrofoam, or visually ambiguous? An aligned alternate class is not independent confirmation of material. |
| 155 / 64 | plastic_film | 15.063×8.928 | Focus material: recurring annotation-relative class failure | 42 / 1e-5 / E1; 42 / 5e-6 / E1; 42 / 5e-6 / E2; 123 / 1e-5 / E1; 123 / 1e-5 / E2; 123 / 5e-6 / E1; 123 / 5e-6 / E2 | Is the annotated material distinct from film/wrapper/bag/Styrofoam, or visually ambiguous? An aligned alternate class is not independent confirmation of material. |
| 208 / 88 | styrofoam_piece | 48.451×77.376 | Focus material: recurring annotation-relative class failure | 42 / 1e-5 / E2; 42 / 5e-6 / E2; 123 / 1e-5 / E1; 123 / 1e-5 / E2; 123 / 5e-6 / E1; 123 / 5e-6 / E2 | Is the annotated material distinct from film/wrapper/bag/Styrofoam, or visually ambiguous? An aligned alternate class is not independent confirmation of material. |
| 31 / 9 | drink_can | 31.146×53.255 | Well-aligned best-overlap class changes cigarette→plastic_bottle | 42 / 1e-5 / E1; 42 / 1e-5 / E2 | Which material/extent is plausible? These may be different queries, not proof the same physical proposal changed class. |
| 31 / 9 | drink_can | 31.146×53.255 | Well-aligned best-overlap class changes plastic_bottle→plastic_film | 42 / 5e-6 / E1; 42 / 5e-6 / E2 | Which material/extent is plausible? These may be different queries, not proof the same physical proposal changed class. |
| 31 / 9 | drink_can | 31.146×53.255 | Well-aligned best-overlap class changes plastic_film→plastic_bottle | 123 / 1e-5 / E1; 123 / 1e-5 / E2 | Which material/extent is plausible? These may be different queries, not proof the same physical proposal changed class. |
| 777 / 278 | cigarette | 6.909×12.364 | Owner-reported ambiguity at image level; exact uncertain GT not supplied | Pretrained; 42 / 1e-5 / E1; 42 / 1e-5 / E2; 42 / 5e-6 / E1; 42 / 5e-6 / E2; 123 / 1e-5 / E1; 123 / 1e-5 / E2; 123 / 5e-6 / E1; 123 / 5e-6 / E2 | Which of GT278–284 are visually uncertain, and why? Do not assume this specific GT is already confirmed ambiguous. |


Reuse the existing saved-review GUI with --per-object; no GUI changes or annotated exports were made for this audit. Example launch from the repository root:

```powershell
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --gui --saved-review .\runs\training\cross-seed-analysis-fd10923e40d24c40be5bab3ae0fa2c63\review-manifest.json --run-choice "123 / 1e-5" --image-id 777 --per-object
```

## 9. Confirmed findings, supported hypotheses and unknowns

**Confirmed:** actual aspect-ratio resize and padding masks;75 independent box transforms; the strong descriptive smallest-versus-larger coverage gap; recurring correct-localization/no-correct-class outputs; score/rank exclusions; exact eight-class span binding in batch1 fixtures; image777 larger wrapper and tiny cigarette sizes. All follow actual processor or saved-output checks.

**Supported but unproven:** limited resized detail contributes to tiny-object failures; learned material/class assignment and class ambiguity contribute to well-localized errors. Direct evidence does not identify frozen backbones, learning rate, data imbalance, native matching weights or token semantics as a unique cause. Class counts and rare examples explain why evidence is unstable, not why a particular query fails.

**Unknown:** true visual identity and correctness of uncertain boxes; extent of annotation omissions; class-score alternatives/margins; how much higher resolution helps; whether larger inputs fit the local GPU under unchanged processing; whether improvements generalize beyond this small validation set. No test split was opened and no new inference was authorized.

## 10. Ranked options and single next controlled experiment


| Rank | Option | Evidence strength /question | Cost /isolation limit |
| --- | --- | --- | --- |
| 1 | A: fixed-weight input-resolution comparison | Strong repeated size association, including no-floor any-class failures; tests whether input scale changes actual candidate recovery | One higher-scale47-image inference-only evaluation on a preselected checkpoint; GPU-fit uncertain. Direct paired comparison holds training/seed/weights fixed |
| 2 | B: classification-focused investigation | Strong20–27 object/state class-related failures, including larger objects; no demonstrated span-index defect | Static checks completed; deeper matching/logit tracing needs a prespecified separately authorized diagnostic forward. Several possible causes remain hard to isolate |
| 3 | C: systematic dataset-quality review | Owner-reported uncertainty and rare support are real concerns; no measured annotation-error prevalence | Human time; preserve labels and blind rubric where practical. Can resolve material uncertainty but does not itself isolate input-scale effect |


**Single proposed experiment:** preselect the preserved seed42/1e-5/E1 checkpoint, compare existing256/384 inputs with384/576 using identical weights and fixed complete validation protocol. No training or threshold/prompt changes. The existing collator ties square pad size to longest edge;576-square padding is a necessary declared companion to576 content capacity, not stretched image content. This experiment tests the combined input-scale policy, not an independently isolated padding effect. Do not select another resolution/checkpoint after inspecting outcomes.

Prespecify primary endpoints: any-class and same-class IoU>=.50 coverage for the original fixed size bins, at both score0 and .05; within-GT changes and image777-excluded cigarette results. Keep each GT assigned to its current-resolution bin to avoid moving denominators. Record new pixel sizes separately. Secondary: unchanged COCO AP/AP50/AP75/AR100 and per-class error identities; check regressions in larger/class-confused cases. A null outcome would weaken the usefulness of this moderate scale increase, not prove resolution irrelevant. Positive gains would still not establish new generalization or annotation truth.

GPU consideration: the recorded GTX1660Ti has6GiB. Prior384-square FP32 training peaks were2,558,043,648 allocated bytes (~2.382GiB) and2,726,297,600 reserved (~2.539GiB), excluding total process/driver use. A576-square canvas has2.25× pixels and nominal multi-scale encoder positions(3060→6885), while queries remain900. This is not a2.25× total-memory prediction: activations, attention structure, fixed model parameters and inference/training lifetimes differ. No larger-input memory measurement or fit guarantee exists. Propose a separately approved one-image inference-only fit check before the fixed full pass; stop on OOM, never silently reduce settings. Some1.6–2.9px boxes remain below4px after1.5× scaling, so this moderate experiment may not recover every tiny object. Do not recommend a training configuration as proven feasible.

No option has been implemented as an experiment. Human material review remains necessary alongside interpreting candidate agreement. Stop after this audit for owner review.

## Appendix A. All 47 actual image transforms


| Image | Original W×H | Valid W×H | Pad L/T/R/B | sx /sy | Valid/canvas fraction |
| --- | --- | --- | --- | --- | --- |
| 31 | 1537×2049 | 256×341 | [0, 0, 128, 43] | 0.16655823 / 0.16642265 | 0.592014 |
| 42 | 1537×2049 | 256×341 | [0, 0, 128, 43] | 0.16655823 / 0.16642265 | 0.592014 |
| 55 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 113 | 1824×4000 | 175×384 | [0, 0, 209, 0] | 0.09594298 / 0.09600000 | 0.455729 |
| 144 | 1824×4000 | 175×384 | [0, 0, 209, 0] | 0.09594298 / 0.09600000 | 0.455729 |
| 155 | 1824×4000 | 175×384 | [0, 0, 209, 0] | 0.09594298 / 0.09600000 | 0.455729 |
| 157 | 4000×1824 | 384×175 | [0, 0, 0, 209] | 0.09600000 / 0.09594298 | 0.455729 |
| 165 | 4000×1824 | 384×175 | [0, 0, 0, 209] | 0.09600000 / 0.09594298 | 0.455729 |
| 166 | 1824×4000 | 175×384 | [0, 0, 209, 0] | 0.09594298 / 0.09600000 | 0.455729 |
| 200 | 1824×4000 | 175×384 | [0, 0, 209, 0] | 0.09594298 / 0.09600000 | 0.455729 |
| 208 | 1824×4000 | 175×384 | [0, 0, 209, 0] | 0.09594298 / 0.09600000 | 0.455729 |
| 236 | 3264×2448 | 341×256 | [0, 0, 43, 128] | 0.10447304 / 0.10457516 | 0.592014 |
| 252 | 4032×3024 | 341×256 | [0, 0, 43, 128] | 0.08457341 / 0.08465608 | 0.592014 |
| 263 | 3024×4032 | 256×341 | [0, 0, 128, 43] | 0.08465608 / 0.08457341 | 0.592014 |
| 377 | 4160×3120 | 341×256 | [0, 0, 43, 128] | 0.08197115 / 0.08205128 | 0.592014 |
| 407 | 3120×4160 | 256×341 | [0, 0, 128, 43] | 0.08205128 / 0.08197115 | 0.592014 |
| 419 | 3120×4160 | 256×341 | [0, 0, 128, 43] | 0.08205128 / 0.08197115 | 0.592014 |
| 424 | 4208×2368 | 384×216 | [0, 0, 0, 168] | 0.09125475 / 0.09121622 | 0.562500 |
| 464 | 3120×4160 | 256×341 | [0, 0, 128, 43] | 0.08205128 / 0.08197115 | 0.592014 |
| 466 | 3120×4160 | 256×341 | [0, 0, 128, 43] | 0.08205128 / 0.08197115 | 0.592014 |
| 487 | 3120×4160 | 256×341 | [0, 0, 128, 43] | 0.08205128 / 0.08197115 | 0.592014 |
| 676 | 1536×2048 | 256×341 | [0, 0, 128, 43] | 0.16666667 / 0.16650391 | 0.592014 |
| 691 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 700 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 734 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 762 | 2442×2597 | 256×272 | [0, 0, 128, 112] | 0.10483210 / 0.10473623 | 0.472222 |
| 768 | 3264×2448 | 341×256 | [0, 0, 43, 128] | 0.10447304 / 0.10457516 | 0.592014 |
| 772 | 3264×2448 | 341×256 | [0, 0, 43, 128] | 0.10447304 / 0.10457516 | 0.592014 |
| 776 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 777 | 1925×1408 | 350×256 | [0, 0, 34, 128] | 0.18181818 / 0.18181818 | 0.607639 |
| 806 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 833 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 849 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 906 | 3264×2448 | 341×256 | [0, 0, 43, 128] | 0.10447304 / 0.10457516 | 0.592014 |
| 973 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 1009 | 3264×2448 | 341×256 | [0, 0, 43, 128] | 0.10447304 / 0.10457516 | 0.592014 |
| 1014 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 1030 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 1074 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 1076 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 1096 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 1126 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 1156 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 1161 | 2448×3264 | 256×341 | [0, 0, 128, 43] | 0.10457516 / 0.10447304 | 0.592014 |
| 1182 | 6000×4000 | 384×256 | [0, 0, 0, 128] | 0.06400000 / 0.06400000 | 0.666667 |
| 1243 | 3264×2448 | 341×256 | [0, 0, 43, 128] | 0.10447304 / 0.10457516 | 0.592014 |
| 1250 | 3264×2448 | 341×256 | [0, 0, 43, 128] | 0.10447304 / 0.10457516 | 0.592014 |


## Appendix B. All 75 object measurements


| Image /GT | Class | Original bbox W×H | Original bbox area | Resized bbox W×H | Resized area | Area/valid | Group |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 31/9 | drink_can | 187.000×320.000 | 59840.000 | 31.146×53.255 | 1658.709 | 0.01900097 | larger ≥16 |
| 42/16 | drink_can | 425.000×689.000 | 292825.000 | 70.787×114.665 | 8116.834 | 0.09298060 | larger ≥16 |
| 55/19 | drink_can | 492.000×711.000 | 349812.000 | 51.451×74.280 | 3821.796 | 0.04377974 | larger ≥16 |
| 113/27 | plastic_bottle | 191.000×138.000 | 26358.000 | 18.325×13.248 | 242.771 | 0.00361266 | small 8–<16 |
| 144/55 | cigarette | 58.000×20.000 | 1160.000 | 5.565×1.920 | 10.684 | 0.00015899 | extremely small <4 |
| 144/56 | cigarette | 28.000×87.000 | 2436.000 | 2.686×8.352 | 22.437 | 0.00033388 | extremely small <4 |
| 144/57 | cigarette | 29.000×76.000 | 2204.000 | 2.782×7.296 | 20.300 | 0.00030208 | extremely small <4 |
| 144/58 | plastic_film | 56.000×27.000 | 1512.000 | 5.373×2.592 | 13.926 | 0.00020724 | extremely small <4 |
| 144/59 | plastic_film | 119.000×142.000 | 16898.000 | 11.417×13.632 | 155.639 | 0.00231606 | small 8–<16 |
| 144/60 | plastic_film | 246.000×277.000 | 68142.000 | 23.602×26.592 | 627.624 | 0.00933964 | larger ≥16 |
| 144/61 | plastic_film | 146.000×162.000 | 23652.000 | 14.008×15.552 | 217.847 | 0.00324178 | small 8–<16 |
| 155/64 | plastic_film | 157.000×93.000 | 14601.000 | 15.063×8.928 | 134.483 | 0.00200123 | small 8–<16 |
| 155/65 | cigarette | 17.000×25.000 | 425.000 | 1.631×2.400 | 3.914 | 0.00005825 | extremely small <4 |
| 157/67 | plastic_film | 137.000×168.000 | 23016.000 | 13.152×16.118 | 211.989 | 0.00315461 | small 8–<16 |
| 165/72 | cigarette | 119.000×85.000 | 10115.000 | 11.424×8.155 | 93.164 | 0.00138638 | small 8–<16 |
| 166/73 | plastic_film | 498.000×126.000 | 62748.000 | 47.780×12.096 | 577.942 | 0.00860033 | small 8–<16 |
| 200/86 | plastic_wrapper | 482.000×385.000 | 185570.000 | 46.245×36.960 | 1709.197 | 0.02543448 | larger ≥16 |
| 208/88 | styrofoam_piece | 505.000×806.000 | 407030.000 | 48.451×77.376 | 3748.961 | 0.05578810 | larger ≥16 |
| 236/96 | styrofoam_piece | 1674.000×1386.000 | 2320164.000 | 174.888×144.941 | 25348.453 | 0.29037359 | larger ≥16 |
| 252/104 | plastic_wrapper | 640.000×709.000 | 453760.000 | 54.127×60.021 | 3248.765 | 0.03721550 | larger ≥16 |
| 263/106 | styrofoam_piece | 696.000×649.000 | 451704.000 | 58.921×54.888 | 3234.044 | 0.03704688 | larger ≥16 |
| 377/150 | drink_can | 1411.000×781.000 | 1101991.000 | 115.661×64.082 | 7411.813 | 0.08490439 | larger ≥16 |
| 407/159 | plastic_wrapper | 295.000×152.000 | 44840.000 | 24.205×12.460 | 301.587 | 0.00345476 | small 8–<16 |
| 419/162 | plastic_wrapper | 1380.000×1884.000 | 2599920.000 | 113.231×154.434 | 17486.641 | 0.20031435 | larger ≥16 |
| 424/163 | plastic_wrapper | 658.000×863.729 | 568333.945 | 60.046×78.786 | 4730.762 | 0.05703562 | larger ≥16 |
| 464/177 | styrofoam_piece | 2339.000×1969.000 | 4605491.000 | 191.918×161.401 | 30975.788 | 0.35483628 | larger ≥16 |
| 466/178 | plastic_film | 2514.000×3188.000 | 8014632.000 | 206.277×261.324 | 53905.119 | 0.61749815 | larger ≥16 |
| 487/183 | plastic_bottle | 1360.000×2420.000 | 3291200.000 | 111.590×198.370 | 22136.079 | 0.25357495 | larger ≥16 |
| 676/216 | plastic_film | 377.000×484.000 | 182468.023 | 62.833×80.588 | 5063.606 | 0.05800502 | larger ≥16 |
| 676/217 | plastic_film | 326.000×408.000 | 133008.000 | 54.333×67.934 | 3691.059 | 0.04228210 | larger ≥16 |
| 691/220 | drink_can | 193.000×275.000 | 53075.000 | 20.183×28.730 | 579.860 | 0.00664245 | larger ≥16 |
| 691/260 | cigarette | 46.000×62.000 | 2852.000 | 4.810×6.477 | 31.159 | 0.00035693 | very small 4–<8 |
| 691/261 | cigarette | 60.000×32.000 | 1920.000 | 6.275×3.343 | 20.977 | 0.00024029 | extremely small <4 |
| 700/223 | plastic_bottle | 124.000×50.000 | 6200.000 | 12.967×5.224 | 67.737 | 0.00077594 | very small 4–<8 |
| 734/242 | plastic_wrapper | 323.000×379.000 | 122417.000 | 33.778×39.595 | 1337.441 | 0.01532076 | larger ≥16 |
| 762/250 | plastic_wrapper | 450.000×799.000 | 359550.000 | 47.174×83.684 | 3947.758 | 0.05669460 | larger ≥16 |
| 768/253 | plastic_wrapper | 421.000×239.000 | 100619.000 | 43.983×24.993 | 1099.291 | 0.01259269 | larger ≥16 |
| 772/256 | plastic_wrapper | 581.000×536.000 | 311416.000 | 60.699×56.052 | 3402.309 | 0.03897439 | larger ≥16 |
| 776/258 | plastic_bag | 551.000×376.000 | 207176.000 | 57.621×39.282 | 2263.457 | 0.02592853 | larger ≥16 |
| 777/259 | plastic_wrapper | 116.000×263.000 | 30508.000 | 21.091×47.818 | 1008.529 | 0.01125590 | larger ≥16 |
| 777/278 | cigarette | 38.000×68.000 | 2584.000 | 6.909×12.364 | 85.421 | 0.00095336 | very small 4–<8 |
| 777/279 | cigarette | 40.000×76.000 | 3040.000 | 7.273×13.818 | 100.496 | 0.00112161 | very small 4–<8 |
| 777/280 | cigarette | 44.000×52.000 | 2288.000 | 8.000×9.455 | 75.636 | 0.00084416 | small 8–<16 |
| 777/281 | cigarette | 54.000×74.000 | 3996.000 | 9.818×13.455 | 132.099 | 0.00147432 | small 8–<16 |
| 777/282 | cigarette | 16.000×36.000 | 576.000 | 2.909×6.545 | 19.041 | 0.00021251 | extremely small <4 |
| 777/283 | cigarette | 14.000×34.000 | 476.000 | 2.545×6.182 | 15.736 | 0.00017562 | extremely small <4 |
| 777/284 | cigarette | 74.000×48.000 | 3552.000 | 13.455×8.727 | 117.421 | 0.00131051 | small 8–<16 |
| 806/294 | plastic_cup | 387.000×392.000 | 151704.000 | 40.471×40.953 | 1657.409 | 0.01898609 | larger ≥16 |
| 833/303 | plastic_bag | 853.000×884.000 | 754052.000 | 89.203×92.354 | 8238.233 | 0.09437126 | larger ≥16 |
| 849/305 | plastic_bottle | 132.000×252.000 | 33264.000 | 13.804×26.327 | 363.419 | 0.00416306 | small 8–<16 |
| 849/313 | cigarette | 26.000×64.000 | 1664.000 | 2.719×6.686 | 18.180 | 0.00020825 | extremely small <4 |
| 849/314 | cigarette | 28.000×58.000 | 1624.000 | 2.928×6.059 | 17.743 | 0.00020325 | extremely small <4 |
| 849/315 | cigarette | 40.000×32.000 | 1280.000 | 4.183×3.343 | 13.984 | 0.00016019 | extremely small <4 |
| 849/316 | cigarette | 48.000×42.000 | 2016.000 | 5.020×4.388 | 22.025 | 0.00025231 | very small 4–<8 |
| 906/324 | styrofoam_piece | 67.000×132.000 | 8844.000 | 7.000×13.804 | 96.623 | 0.00110685 | very small 4–<8 |
| 973/356 | plastic_film | 298.000×333.000 | 99234.000 | 31.163×34.790 | 1084.160 | 0.01241935 | larger ≥16 |
| 1009/365 | plastic_film | 271.000×227.000 | 61517.000 | 28.312×23.739 | 672.091 | 0.00769899 | larger ≥16 |
| 1014/367 | plastic_film | 181.000×183.000 | 33123.000 | 18.928×19.119 | 361.878 | 0.00414542 | larger ≥16 |
| 1030/372 | drink_can | 248.000×126.000 | 31248.000 | 25.935×13.164 | 341.393 | 0.00391076 | small 8–<16 |
| 1074/387 | plastic_cup | 147.000×241.000 | 35427.000 | 15.373×25.178 | 387.050 | 0.00443377 | small 8–<16 |
| 1074/388 | plastic_cup | 272.000×177.000 | 48144.000 | 28.444×18.492 | 525.987 | 0.00602533 | larger ≥16 |
| 1076/404 | plastic_cup | 154.000×202.000 | 31108.000 | 16.105×21.104 | 339.864 | 0.00389323 | larger ≥16 |
| 1096/414 | plastic_bottle | 547.000×530.000 | 289910.000 | 57.203×55.371 | 3167.349 | 0.03628287 | larger ≥16 |
| 1126/423 | plastic_cup | 224.000×177.000 | 39648.000 | 23.425×18.492 | 433.166 | 0.00496203 | larger ≥16 |
| 1156/443 | plastic_wrapper | 250.000×174.000 | 43500.000 | 26.144×18.178 | 475.250 | 0.00544412 | larger ≥16 |
| 1161/444 | plastic_bottle | 253.000×281.000 | 71093.000 | 26.458×29.357 | 776.711 | 0.00889744 | larger ≥16 |
| 1161/495 | cigarette | 22.000×28.000 | 616.000 | 2.301×2.925 | 6.730 | 0.00007709 | extremely small <4 |
| 1161/496 | cigarette | 32.000×44.000 | 1408.000 | 3.346×4.597 | 15.383 | 0.00017621 | extremely small <4 |
| 1161/497 | cigarette | 60.000×46.000 | 2760.000 | 6.275×4.806 | 30.154 | 0.00034542 | very small 4–<8 |
| 1161/498 | cigarette | 54.000×32.000 | 1728.000 | 5.647×3.343 | 18.879 | 0.00021626 | extremely small <4 |
| 1161/499 | cigarette | 42.000×52.000 | 2184.000 | 4.392×5.433 | 23.861 | 0.00027333 | very small 4–<8 |
| 1161/500 | cigarette | 66.000×54.000 | 3564.000 | 6.902×5.642 | 38.938 | 0.00044604 | very small 4–<8 |
| 1182/503 | drink_can | 308.000×173.000 | 53284.000 | 19.712×11.072 | 218.251 | 0.00222017 | small 8–<16 |
| 1243/525 | plastic_bottle | 265.000×220.000 | 58300.000 | 27.685×23.007 | 636.944 | 0.00729637 | larger ≥16 |
| 1250/527 | plastic_bottle | 129.000×39.000 | 5031.000 | 13.477×4.078 | 54.965 | 0.00062964 | very small 4–<8 |


## Verification, preservation and stop record

Nine focused checks passed with no skips, failures or errors: five independent
dimension/bin/box-transform tests, one all-eight matched target/span identity
test, and three existing native processor/decoder coordinate oracles. Tests
used generated logits and processor calls; neural-module execution and optimizer
steps were blocked. No model parameters were loaded. All 2700 audit records
passed coverage/IoU consistency checks, and every bbox valid-area fraction
equals its original-image area fraction within 1e-12 under the x/y transform.

All 40 checkpoint bundles of the four corrected runs were hash-verified against
their original contracts without restoring state. The 800 protected file hashes,
including current GUI/trainer source, prior review artifacts/manifests, original
weights, authorized train/validation bytes, configuration, rights/source evidence,
prior reports and inspected installed source files, matched before/after.
Exactly 47 authorized validation photos were decoded by the CPU processor; no
test metadata/photos were read. Existing GUI behavior was preserved by source
identity; it was reused through its documented command and was not rebuilt or
requalified as a new visual judgement. Current owner modifications remain intact.

The run folder retains full-resolution numerical records, CSVs, software log,
protected hashes, source copies and completion evidence. Relative Markdown
links, helper Python syntax and Git whitespace were checked. No new training,
model inference, optimizer updates, promotion, dataset/annotation/metric changes,
dependency changes, annotated exports, uploads or commits/pushes/merges occurred.
The proposed scale comparison requires separate owner authorization. Stop for
owner review; human uncertainty has not been converted into annotation edits.
