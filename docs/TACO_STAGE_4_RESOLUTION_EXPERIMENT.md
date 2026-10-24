# Stage 4 — Controlled inference-only input-resolution comparison (D-055)

## Executive summary

Completed the owner-authorized single retry on2026-10-23. All five gates passed, followed by exactly47 high-resolution validation forwards. There were47 exact baseline replay forwards and one image777 high-resolution pilot:95 real model forwards total, **zero backward passes or optimizer updates**, zero train-photo reads and zero test-split access. Original stopped attempt remains preserved; no third attempt was authorized or executed.

The fixed seed42/LR1e-5/E1 checkpoint changed AP **0.203932→0.247003** (Δ+0.043071) and AR100 **0.396766→0.474763** (Δ+0.077997). Exact per-object and class changes follow; this is inference-scale sensitivity of one frozen model on the same small validation set, not new training or proof of generalization.


| Original size bin | N | Any256/384→384/576 @.05 | Same256/384→384/576 @.05 |
| --- | --- | --- | --- |
| extremely small <4 | 14 | 0→0 | 0→0 |
| very small 4–<8 | 10 | 4→8 | 1→4 |
| small 8–<16 | 15 | 10→13 | 6→6 |
| larger ≥16 | 36 | 35→35 | 21→23 |


The hypothesis is assessed against all four prespecified source/floor conditions, paired gains/losses and larger-object regressions below. “Meaningful” recovery had no numeric acceptance margin in D-055; effect counts are reported without inventing a retrospective pass threshold. No production or checkpoint promotion follows.

## Authorization, isolated variable and provenance

Active retry: `runs\training\resolution-experiment-9cebf39b0982405d8dfdd954ae2dbed1`. Original failed attempt: `runs/training/resolution-experiment-629229adf1c6480d930aac79250d18f8`, preserved byte-for-byte in the protected inventory. The owner explicitly authorized one retry from Gate1; the pointer records it consumed, with zero retries remaining.

Fixed checkpoint: `runs/training/taco-312e568703a54f98a7c8f4b87b29cd39/checkpoints/epoch-01`. Input policy is256/384 with384-square padding versus384/576 with576-square padding; pad size is tied to longest edge by the unchanged collator. Aspect ratio is preserved; image content is not stretched square. Labels/prompt,900 queries, native adapter/loss/architecture, FP32, batch1, same47 validation images/75 annotations, zero-floor top100 COCO v2 decoder and no NMS remain unchanged. Class-correct diagnostics use the saved GT labels and require human material confirmation.

The runner/harness is isolated under `work/resolution_experiment/`; core training/evaluation, provenance and GUI/functional inference scripts were not modified. Existing owner changes, including unrelated documents, remain intact. No branch publication is assumed. Token/target semantics passed `validate_bundle` against the pinned processor/configuration and saved native adapter manifest. Every parameter was frozen before evaluation, `model.eval()` and `torch.inference_mode()` were enforced for each real forward, and all parameters remain FP32.

Current training source differs from the checkpoint only through the reviewed D-050/D-051 LR/seed CLI additions; the other five required implementation fingerprints match exactly. The original trainer archive and later qualified trainer archive were checked. Strict resume/compatibility policies were not weakened. `training_state.pt` was hashed only; model loading used safetensors and blocked `torch.load`.

## Five ordered preflight gates and exact baseline


| Gate | Measured result |
| --- | --- |
| 1 Guarded suite | 228 tests; 0 failures; 0 errors; 14 documented skips; all3 new resolution fixtures passed |
| 2 Artifacts/provenance | 1497 fresh protected hashes; all40 corrected checkpoint bundles verified against original contracts |
| 3 CPU geometry | 47 images/75 boxes at each scale;94 verified source-photo reads; top-left rectangular masks, valid target transforms and class spans passed |
| 4 CUDA baseline | Exact metrics, full predictions,42300 raw query rows and mean loss equal the saved E1 result; weights unchanged |
| 5 High-scale pilot | Image777,8 annotations,900 queries, finite active logits/loss and valid normalized boxes; passed without OOM |


The first failed attempt’s blanket image-save guard blocked synthetic fixtures; its indirect gradient fixture was also omitted from exclusions. The retry allows only generated temporary/in-memory software fixtures, while actual experiment image exports remain blocked. All gradient/optimizer-execution fixtures are excluded or guarded; the existing symlink privilege skip is reported separately in `software-results.json`. The full suite passed without a model-scale experiment before Gate2.

The256/384 replay exactly matches the historical saved `validation-epoch-1.json`: full-precision metric dictionaries, complete official prediction list, all42300 raw rows (including categories, scores, original-pixel boxes and phrases) and mean loss. Maximum numeric delta is0 for the compared outputs. Evaluation timing is newly measured and is not expected to match the old wall-clock record.

All95 forwards were counted and shape checked. The model parameter hash before, after baseline, after pilot and after high-scale evaluation was unchanged: `71efbabb97949b312cf3f7eae3d6eeae75c47b33bf31ab56e3b4cc4cd916104e`. Optimizer step, Tensor backward/autograd backward and gradient calls were blocked. Active-token logits are finite; padded text positions retain the required -inf sentinel rather than being misreported as invalid active logits.

## Two-scale geometry and tiny-cigarette threshold crossings

CPU preflight independently transformed original boxes using sx=validW/originalW and sy=validH/originalH, comparing against the actual corrected processor target within0.001 pixels. For every image, pixel_mask is exactly ones in the top-left valid rectangle and zeros in right/bottom padding; its sum is validW×validH. Encoder positions rise3060→6885 while queries remain900. Rounding can make the scale slightly different from exactly1.5 in either axis.


| Scale | Distinct valid sizes (W×H: image count) | Maximum transform error px |
| --- | --- | --- |
| 256x384 | 175×384: 6; 256×272: 1; 256×341: 26; 341×256: 9; 350×256: 1; 384×175: 2; 384×216: 1; 384×256: 1 | 4.431876271837609e-05 |
| 384x576 | 263×576: 6; 384×408: 1; 384×511: 2; 384×512: 24; 512×384: 9; 525×384: 1; 576×263: 2; 576×324: 1; 576×384: 1 | 4.55195761333016e-05 |


Full47-image geometry and75 paired box measurements are retained in `geometry-256x384-vs-384x576.json`, `validation-image-geometry-comparison.csv` and `per-object-comparison.csv`.

Of the13 cigarettes below4px initially, **4 remain below4px** and **9 cross to>=4px**. Remaining IDs: [55, 65, 283, 495]; crossing IDs: [56, 57, 261, 282, 313, 314, 315, 496, 498]. These are pixel-dimension crossings, not detection outcomes.


| Secondary high-scale measured bin | N (new measured dimensions) |
| --- | --- |
| extremely small <4 | 5 |
| very small 4–<8 | 15 |
| small 8–<16 | 9 |
| larger ≥16 | 46 |


All outcome tables below lock each GT to its original256/384 bin: N14/10/15/36. The secondary measured-size distribution does not change these denominators. Bins use shortest bbox side and are diagnostic categories, not COCO area definitions or physical sizes.

## Primary endpoint1 — locked-bin coverage

Coverage at IoU>=.50 is non-exclusive per GT; the same query can cover multiple GTs. Source all900 includes all valid saved queries; top100 uses official retained membership. Score floors0 and .05 are applied consistently. Stored outputs retain only one argmax class/query and decoded phrases; alternative class logits/masks/unsaved proposals are unavailable. None of these diagnostics replaces official COCO AP.

### all 900 (diagnostic) @ 0.05


| Original bin | N | Baseline any/same | High any/same | Δ any | Δ same |
| --- | --- | --- | --- | --- | --- |
| extremely small <4 | 14 | 0/0 | 0/0 | 0 | 0 |
| very small 4–<8 | 10 | 4/1 | 8/4 | 4 | 3 |
| small 8–<16 | 15 | 10/6 | 13/6 | 3 | 0 |
| larger ≥16 | 36 | 35/21 | 35/23 | 0 | 2 |


### all 900 (diagnostic) @ 0.0


| Original bin | N | Baseline any/same | High any/same | Δ any | Δ same |
| --- | --- | --- | --- | --- | --- |
| extremely small <4 | 14 | 1/1 | 1/1 | 0 | 0 |
| very small 4–<8 | 10 | 4/1 | 9/5 | 5 | 4 |
| small 8–<16 | 15 | 14/8 | 15/9 | 1 | 1 |
| larger ≥16 | 36 | 36/26 | 36/28 | 0 | 2 |


### official top100 @ 0.05


| Original bin | N | Baseline any/same | High any/same | Δ any | Δ same |
| --- | --- | --- | --- | --- | --- |
| extremely small <4 | 14 | 0/0 | 0/0 | 0 | 0 |
| very small 4–<8 | 10 | 4/1 | 8/4 | 4 | 3 |
| small 8–<16 | 15 | 10/6 | 13/6 | 3 | 0 |
| larger ≥16 | 36 | 35/21 | 35/23 | 0 | 2 |


### official top100 @ 0.0


| Original bin | N | Baseline any/same | High any/same | Δ any | Δ same |
| --- | --- | --- | --- | --- | --- |
| extremely small <4 | 14 | 0/0 | 0/0 | 0 | 0 |
| very small 4–<8 | 10 | 4/1 | 9/5 | 5 | 4 |
| small 8–<16 | 15 | 13/8 | 14/6 | 1 | -2 |
| larger ≥16 | 36 | 36/22 | 36/27 | 0 | 5 |


## Primary endpoint2 —24 cigarettes across7 images

### all 900 (diagnostic) @ 0.05


| Group | N | Baseline any/same | High any/same |
| --- | --- | --- | --- |
| extremely small <4 | 13 | 0/0 | 0/0 |
| very small 4–<8 | 7 | 1/1 | 5/4 |
| small 8–<16 | 4 | 1/1 | 3/1 |
| all | 24 | 2/2 | 8/5 |
| image777 | 7 | 0/0 | 3/0 |
| other_images | 17 | 2/2 | 5/5 |


### all 900 (diagnostic) @ 0.0


| Group | N | Baseline any/same | High any/same |
| --- | --- | --- | --- |
| extremely small <4 | 13 | 1/1 | 1/1 |
| very small 4–<8 | 7 | 1/1 | 6/5 |
| small 8–<16 | 4 | 3/3 | 4/1 |
| all | 24 | 5/5 | 11/7 |
| image777 | 7 | 2/2 | 5/1 |
| other_images | 17 | 3/3 | 6/6 |


### official top100 @ 0.05


| Group | N | Baseline any/same | High any/same |
| --- | --- | --- | --- |
| extremely small <4 | 13 | 0/0 | 0/0 |
| very small 4–<8 | 7 | 1/1 | 5/4 |
| small 8–<16 | 4 | 1/1 | 3/1 |
| all | 24 | 2/2 | 8/5 |
| image777 | 7 | 0/0 | 3/0 |
| other_images | 17 | 2/2 | 5/5 |


### official top100 @ 0.0


| Group | N | Baseline any/same | High any/same |
| --- | --- | --- | --- |
| extremely small <4 | 13 | 0/0 | 0/0 |
| very small 4–<8 | 7 | 1/1 | 6/5 |
| small 8–<16 | 4 | 3/3 | 4/1 |
| all | 24 | 4/4 | 10/6 |
| image777 | 7 | 2/2 | 5/1 |
| other_images | 17 | 2/2 | 5/5 |


### Individual24 cigarettes — all 900 (diagnostic) @ 0.05

Dimensions are W×H (shortest side) in valid resized pixels. Best overlap means highest positive IoU, with score and query/class identity shown; an overlap need not reach.50. No positive overlap is recorded as none.


| Image /GT | Locked bin | Baseline W×H (min) | High W×H (min) | Baseline best any | Baseline best same | High best any | High best same |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 144/55 | extremely small <4 | 5.565×1.920 (1.920) | 8.363×2.880 (2.880) | none | none | q21 plastic_film; s=0.100626; IoU=0.000161 | none |
| 144/56 | extremely small <4 | 2.686×8.352 (2.686) | 4.037×12.528 (4.037) | none | none | q21 plastic_film; s=0.100626; IoU=0.000338 | none |
| 144/57 | extremely small <4 | 2.782×7.296 (2.782) | 4.181×10.944 (4.181) | none | none | q27 plastic_bottle; s=0.056017; IoU=0.004120 | none |
| 155/65 | extremely small <4 | 1.631×2.400 (1.631) | 2.451×3.600 (2.451) | q70 plastic_film; s=0.053849; IoU=0.000344 | none | q793 plastic_film; s=0.057182; IoU=0.000398 | none |
| 165/72 | small 8–<16 | 11.424×8.155 (8.155) | 17.136×12.256 (12.256) | q6 cigarette; s=0.239240; IoU=0.671417 | q6 cigarette; s=0.239240; IoU=0.671417 | q5 cigarette; s=0.183005; IoU=0.681839 | q5 cigarette; s=0.183005; IoU=0.681839 |
| 691/260 | very small 4–<8 | 4.810×6.477 (4.810) | 7.216×9.725 (7.216) | q22 plastic_bottle; s=0.058734; IoU=0.158856 | none | q27 styrofoam_piece; s=0.069268; IoU=0.344412 | none |
| 691/261 | extremely small <4 | 6.275×3.343 (3.343) | 9.412×5.020 (5.020) | q16 styrofoam_piece; s=0.061177; IoU=0.349499 | none | q6 styrofoam_piece; s=0.156931; IoU=0.485312 | none |
| 777/278 | very small 4–<8 | 6.909×12.364 (6.909) | 10.364×18.545 (10.364) | q42 styrofoam_piece; s=0.052581; IoU=0.133090 | none | q43 plastic_cup; s=0.050787; IoU=0.648400 | none |
| 777/279 | very small 4–<8 | 7.273×13.818 (7.273) | 10.909×20.727 (10.909) | q42 styrofoam_piece; s=0.052581; IoU=0.150600 | none | q17 plastic_film; s=0.109799; IoU=0.002322 | none |
| 777/280 | small 8–<16 | 8.000×9.455 (8.000) | 12.000×14.182 (12.000) | q12 plastic_film; s=0.105463; IoU=0.001801 | none | q19 styrofoam_piece; s=0.070925; IoU=0.748087 | none |
| 777/281 | small 8–<16 | 9.818×13.455 (9.818) | 14.727×20.182 (14.727) | q42 styrofoam_piece; s=0.052581; IoU=0.204579 | none | q29 styrofoam_piece; s=0.067555; IoU=0.423056 | none |
| 777/282 | extremely small <4 | 2.909×6.545 (2.909) | 4.364×9.818 (4.364) | q12 plastic_film; s=0.105463; IoU=0.000453 | none | q21 plastic_cup; s=0.060413; IoU=0.298844 | none |
| 777/283 | extremely small <4 | 2.545×6.182 (2.545) | 3.818×9.273 (3.818) | q68 styrofoam_piece; s=0.075229; IoU=0.000669 | none | q9 styrofoam_piece; s=0.086437; IoU=0.001112 | none |
| 777/284 | small 8–<16 | 13.455×8.727 (8.727) | 20.182×13.091 (13.091) | q12 plastic_film; s=0.105463; IoU=0.002796 | none | q24 styrofoam_piece; s=0.078591; IoU=0.652363 | none |
| 849/313 | extremely small <4 | 2.719×6.686 (2.719) | 4.078×10.039 (4.078) | q64 cigarette; s=0.056847; IoU=0.397560 | q64 cigarette; s=0.056847; IoU=0.397560 | q19 cigarette; s=0.067537; IoU=0.473064 | q19 cigarette; s=0.067537; IoU=0.473064 |
| 849/314 | extremely small <4 | 2.928×6.059 (2.928) | 4.392×9.098 (4.392) | q5 cigarette; s=0.078806; IoU=0.382078 | q5 cigarette; s=0.078806; IoU=0.382078 | q20 cigarette; s=0.064631; IoU=0.304153 | q20 cigarette; s=0.064631; IoU=0.304153 |
| 849/315 | extremely small <4 | 4.183×3.343 (3.343) | 6.275×5.020 (5.020) | q74 plastic_film; s=0.055752; IoU=0.003725 | none | q29 plastic_film; s=0.059727; IoU=0.000608 | none |
| 849/316 | very small 4–<8 | 5.020×4.388 (4.388) | 7.529×6.588 (6.588) | q29 cigarette; s=0.052121; IoU=0.682032 | q29 cigarette; s=0.052121; IoU=0.682032 | q15 cigarette; s=0.056287; IoU=0.745540 | q15 cigarette; s=0.056287; IoU=0.745540 |
| 1161/495 | extremely small <4 | 2.301×2.925 (2.301) | 3.451×4.392 (3.451) | q25 cigarette; s=0.090755; IoU=0.052516 | q25 cigarette; s=0.090755; IoU=0.052516 | q13 cigarette; s=0.099039; IoU=0.071661 | q13 cigarette; s=0.099039; IoU=0.071661 |
| 1161/496 | extremely small <4 | 3.346×4.597 (3.346) | 5.020×6.902 (5.020) | q42 plastic_film; s=0.067096; IoU=0.005283 | none | q5 cigarette; s=0.156842; IoU=0.313246 | q5 cigarette; s=0.156842; IoU=0.313246 |
| 1161/497 | very small 4–<8 | 6.275×4.806 (4.806) | 9.412×7.216 (7.216) | q221 styrofoam_piece; s=0.064095; IoU=0.000428 | none | q22 cigarette; s=0.094184; IoU=0.603208 | q22 cigarette; s=0.094184; IoU=0.603208 |
| 1161/498 | extremely small <4 | 5.647×3.343 (3.343) | 8.471×5.020 (5.020) | q221 styrofoam_piece; s=0.064095; IoU=0.000268 | none | q86 styrofoam_piece; s=0.086817; IoU=0.000902 | none |
| 1161/499 | very small 4–<8 | 4.392×5.433 (4.392) | 6.588×8.157 (6.588) | q221 styrofoam_piece; s=0.064095; IoU=0.000339 | none | q31 cigarette; s=0.081359; IoU=0.515814 | q31 cigarette; s=0.081359; IoU=0.515814 |
| 1161/500 | very small 4–<8 | 6.902×5.642 (5.642) | 10.353×8.471 (8.471) | q20 cigarette; s=0.072006; IoU=0.321824 | q20 cigarette; s=0.072006; IoU=0.321824 | q2 cigarette; s=0.198196; IoU=0.633423 | q2 cigarette; s=0.198196; IoU=0.633423 |


### Individual24 cigarettes — all 900 (diagnostic) @ 0.0

Dimensions are W×H (shortest side) in valid resized pixels. Best overlap means highest positive IoU, with score and query/class identity shown; an overlap need not reach.50. No positive overlap is recorded as none.


| Image /GT | Locked bin | Baseline W×H (min) | High W×H (min) | Baseline best any | Baseline best same | High best any | High best same |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 144/55 | extremely small <4 | 5.565×1.920 (1.920) | 8.363×2.880 (2.880) | q257 plastic_film; s=0.010362; IoU=0.015574 | q732 cigarette; s=0.008580; IoU=0.014304 | q622 plastic_film; s=0.004674; IoU=0.037225 | none |
| 144/56 | extremely small <4 | 2.686×8.352 (2.686) | 4.037×12.528 (4.037) | q362 plastic_cup; s=0.007348; IoU=0.082531 | none | q523 plastic_film; s=0.006085; IoU=0.028881 | none |
| 144/57 | extremely small <4 | 2.782×7.296 (2.782) | 4.181×10.944 (4.181) | q530 styrofoam_piece; s=0.003003; IoU=0.029918 | none | q508 plastic_film; s=0.014522; IoU=0.067832 | q94 cigarette; s=0.026299; IoU=0.013755 |
| 155/65 | extremely small <4 | 1.631×2.400 (1.631) | 2.451×3.600 (2.451) | q649 styrofoam_piece; s=0.009470; IoU=0.005641 | q744 cigarette; s=0.006997; IoU=0.000471 | q597 styrofoam_piece; s=0.010656; IoU=0.014403 | q781 cigarette; s=0.006335; IoU=0.005928 |
| 165/72 | small 8–<16 | 11.424×8.155 (8.155) | 17.136×12.256 (12.256) | q6 cigarette; s=0.239240; IoU=0.671417 | q6 cigarette; s=0.239240; IoU=0.671417 | q121 plastic_wrapper; s=0.028072; IoU=0.723150 | q5 cigarette; s=0.183005; IoU=0.681839 |
| 691/260 | very small 4–<8 | 4.810×6.477 (4.810) | 7.216×9.725 (7.216) | q120 plastic_film; s=0.011763; IoU=0.171159 | q46 cigarette; s=0.045445; IoU=0.030932 | q40 plastic_film; s=0.009563; IoU=0.489961 | q83 cigarette; s=0.017818; IoU=0.084629 |
| 691/261 | extremely small <4 | 6.275×3.343 (3.343) | 9.412×5.020 (5.020) | q16 styrofoam_piece; s=0.061177; IoU=0.349499 | q140 cigarette; s=0.040420; IoU=0.056774 | q6 styrofoam_piece; s=0.156931; IoU=0.485312 | q133 cigarette; s=0.007133; IoU=0.067107 |
| 777/278 | very small 4–<8 | 6.909×12.364 (6.909) | 10.364×18.545 (10.364) | q348 plastic_cup; s=0.006958; IoU=0.464773 | q317 cigarette; s=0.006057; IoU=0.251738 | q437 plastic_cup; s=0.027298; IoU=0.777332 | q195 cigarette; s=0.002781; IoU=0.105820 |
| 777/279 | very small 4–<8 | 7.273×13.818 (7.273) | 10.909×20.727 (10.909) | q354 plastic_cup; s=0.019032; IoU=0.305236 | q719 cigarette; s=0.003218; IoU=0.207051 | q190 plastic_bottle; s=0.005508; IoU=0.770263 | q71 cigarette; s=0.032385; IoU=0.593985 |
| 777/280 | small 8–<16 | 8.000×9.455 (8.000) | 12.000×14.182 (12.000) | q105 cigarette; s=0.022276; IoU=0.509056 | q105 cigarette; s=0.022276; IoU=0.509056 | q339 plastic_cup; s=0.016329; IoU=0.827016 | q122 cigarette; s=0.021836; IoU=0.253230 |
| 777/281 | small 8–<16 | 9.818×13.455 (9.818) | 14.727×20.182 (14.727) | q111 plastic_film; s=0.033184; IoU=0.454610 | q317 cigarette; s=0.006057; IoU=0.260288 | q294 plastic_cup; s=0.028302; IoU=0.856932 | q195 cigarette; s=0.002781; IoU=0.253431 |
| 777/282 | extremely small <4 | 2.909×6.545 (2.909) | 4.364×9.818 (4.364) | q92 cigarette; s=0.023566; IoU=0.171872 | q92 cigarette; s=0.023566; IoU=0.171872 | q21 plastic_cup; s=0.060413; IoU=0.298844 | q579 cigarette; s=0.001317; IoU=0.086320 |
| 777/283 | extremely small <4 | 2.545×6.182 (2.545) | 3.818×9.273 (3.818) | q683 plastic_film; s=0.026276; IoU=0.014824 | q702 cigarette; s=0.008663; IoU=0.006609 | q285 plastic_cup; s=0.025675; IoU=0.281080 | q597 cigarette; s=0.003837; IoU=0.038080 |
| 777/284 | small 8–<16 | 13.455×8.727 (8.727) | 20.182×13.091 (13.091) | q71 cigarette; s=0.017263; IoU=0.649737 | q71 cigarette; s=0.017263; IoU=0.649737 | q121 plastic_film; s=0.035063; IoU=0.730136 | q413 cigarette; s=0.007823; IoU=0.165541 |
| 849/313 | extremely small <4 | 2.719×6.686 (2.719) | 4.078×10.039 (4.078) | q64 cigarette; s=0.056847; IoU=0.397560 | q64 cigarette; s=0.056847; IoU=0.397560 | q72 cigarette; s=0.015096; IoU=0.575818 | q72 cigarette; s=0.015096; IoU=0.575818 |
| 849/314 | extremely small <4 | 2.928×6.059 (2.928) | 4.392×9.098 (4.392) | q21 cigarette; s=0.020294; IoU=0.503131 | q21 cigarette; s=0.020294; IoU=0.503131 | q149 cigarette; s=0.008714; IoU=0.430235 | q149 cigarette; s=0.008714; IoU=0.430235 |
| 849/315 | extremely small <4 | 4.183×3.343 (3.343) | 6.275×5.020 (5.020) | q60 cigarette; s=0.049891; IoU=0.255212 | q60 cigarette; s=0.049891; IoU=0.255212 | q261 cigarette; s=0.013191; IoU=0.443463 | q261 cigarette; s=0.013191; IoU=0.443463 |
| 849/316 | very small 4–<8 | 5.020×4.388 (4.388) | 7.529×6.588 (6.588) | q29 cigarette; s=0.052121; IoU=0.682032 | q29 cigarette; s=0.052121; IoU=0.682032 | q15 cigarette; s=0.056287; IoU=0.745540 | q15 cigarette; s=0.056287; IoU=0.745540 |
| 1161/495 | extremely small <4 | 2.301×2.925 (2.301) | 3.451×4.392 (3.451) | q25 cigarette; s=0.090755; IoU=0.052516 | q25 cigarette; s=0.090755; IoU=0.052516 | q13 cigarette; s=0.099039; IoU=0.071661 | q13 cigarette; s=0.099039; IoU=0.071661 |
| 1161/496 | extremely small <4 | 3.346×4.597 (3.346) | 5.020×6.902 (5.020) | q133 cigarette; s=0.013220; IoU=0.040132 | q133 cigarette; s=0.013220; IoU=0.040132 | q34 plastic_wrapper; s=0.011463; IoU=0.347304 | q5 cigarette; s=0.156842; IoU=0.313246 |
| 1161/497 | very small 4–<8 | 6.275×4.806 (4.806) | 9.412×7.216 (7.216) | q595 cigarette; s=0.007774; IoU=0.015275 | q595 cigarette; s=0.007774; IoU=0.015275 | q71 cigarette; s=0.007536; IoU=0.674475 | q71 cigarette; s=0.007536; IoU=0.674475 |
| 1161/498 | extremely small <4 | 5.647×3.343 (3.343) | 8.471×5.020 (5.020) | q532 cigarette; s=0.008206; IoU=0.011902 | q532 cigarette; s=0.008206; IoU=0.011902 | q311 plastic_cup; s=0.012692; IoU=0.351656 | q73 cigarette; s=0.047674; IoU=0.308628 |
| 1161/499 | very small 4–<8 | 4.392×5.433 (4.392) | 6.588×8.157 (6.588) | q320 cigarette; s=0.006178; IoU=0.002832 | q320 cigarette; s=0.006178; IoU=0.002832 | q144 cigarette; s=0.005864; IoU=0.561668 | q144 cigarette; s=0.005864; IoU=0.561668 |
| 1161/500 | very small 4–<8 | 6.902×5.642 (5.642) | 10.353×8.471 (8.471) | q33 cigarette; s=0.023818; IoU=0.390861 | q33 cigarette; s=0.023818; IoU=0.390861 | q2 cigarette; s=0.198196; IoU=0.633423 | q2 cigarette; s=0.198196; IoU=0.633423 |


## Primary endpoint3 — paired75-object transitions

Disjoint annotation-relative states at all900 /score>=.05 /IoU>=.50. Correct means a qualifying same-class candidate; wrong-only means any-class coverage without same-class coverage; no-localization means no qualifying any-class candidate at this floor, not inability to detect at any setting. No uncertain material label is declared visually correct.


| Baseline→high | correct_class | wrong_class_only | no_qualifying_localization |
| --- | --- | --- | --- |
| correct_class | 25 | 2 | 1 |
| wrong_class_only | 5 | 16 | 0 |
| no_qualifying_localization | 3 | 5 | 18 |


Correct-class gains (8 GTs): [86, 106, 150, 177, 443, 497, 499, 500]. Correct-class losses/regressions (3 GTs): [60, 88, 250].


| GT | From | To |
| --- | --- | --- |
| 67 | no_qualifying_localization | wrong_class_only |
| 278 | no_qualifying_localization | wrong_class_only |
| 280 | no_qualifying_localization | wrong_class_only |
| 284 | no_qualifying_localization | wrong_class_only |
| 367 | no_qualifying_localization | wrong_class_only |


Full GT IDs for every matrix cell and all75 paired states are in `size-bin-and-transition-summary.json` and `per-object-comparison.csv`; regressions are retained, not hidden by aggregate gains.

## Secondary — official COCO v2 and loss

Metrics are fractions, not accuracy. Pretrained is the historical unchanged256/384 reference, not a newly executed high-scale pretrained model. All metric dictionaries for baseline replay and high evaluation recomputed exactly from saved predictions.


| Metric | Pretrained256/384 | Fixed E1 baseline | Fixed E1 high | Δ high−baseline |
| --- | --- | --- | --- | --- |
| AP | 0.111576 | 0.203932 | 0.247003 | 0.043071 |
| AP50 | 0.158253 | 0.254720 | 0.295777 | 0.041057 |
| AP75 | 0.135975 | 0.239987 | 0.279521 | 0.039534 |
| AP_small | 0.000000 | 0.019802 | 0.029208 | 0.009406 |
| AP_medium | 0.005941 | 0.008101 | 0.013996 | 0.005895 |
| AP_large | 0.129427 | 0.240848 | 0.295150 | 0.054303 |
| AR1 | 0.294386 | 0.354968 | 0.414916 | 0.059948 |
| AR10 | 0.363077 | 0.388393 | 0.462543 | 0.074151 |
| AR100 | 0.363077 | 0.396766 | 0.474763 | 0.077997 |
| AR_small | 0.000000 | 0.018182 | 0.027273 | 0.009091 |
| AR_medium | 0.005769 | 0.015385 | 0.023077 | 0.007692 |
| AR_large | 0.455504 | 0.498383 | 0.595516 | 0.097134 |
| Mean loss/image | 10.553226 | 7.019476 | 6.795396 | -0.224081 |



| Class | N | Baseline AP | High AP | Δ AP | Baseline AR100 | High AR100 | Δ AR100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cigarette | 24 | 0.020136 | 0.031416 | 0.011279 | 0.050000 | 0.075000 | 0.025000 |
| plastic_bottle | 8 | 0.096181 | 0.109898 | 0.013717 | 0.462500 | 0.575000 | 0.112500 |
| plastic_bag | 2 | 0.677759 | 0.910891 | 0.233132 | 0.950000 | 0.950000 | 0.000000 |
| plastic_wrapper | 11 | 0.118247 | 0.071429 | -0.046818 | 0.272727 | 0.409091 | 0.136364 |
| plastic_film | 13 | 0.016310 | 0.017526 | 0.001217 | 0.384615 | 0.446154 | 0.061538 |
| drink_can | 7 | 0.118812 | 0.177970 | 0.059158 | 0.114286 | 0.242857 | 0.128571 |
| plastic_cup | 5 | 0.432673 | 0.358376 | -0.074297 | 0.560000 | 0.620000 | 0.060000 |
| styrofoam_piece | 5 | 0.151337 | 0.298515 | 0.147178 | 0.380000 | 0.480000 | 0.100000 |



| Class | Baseline correct/wrong/no-localization | High correct/wrong/no-localization | Baseline wrong-only best-any classes | High wrong-only best-any classes |
| --- | --- | --- | --- | --- |
| cigarette | 2/0/22 | 5/3/16 | {} | {'plastic_cup': 1, 'styrofoam_piece': 2} |
| plastic_bottle | 5/3/0 | 5/3/0 | {'plastic_cup': 2, 'plastic_wrapper': 1} | {'plastic_film': 1, 'plastic_cup': 2} |
| plastic_bag | 2/0/0 | 2/0/0 | {} | {} |
| plastic_wrapper | 4/7/0 | 5/6/0 | {'plastic_film': 5, 'plastic_cup': 2} | {'plastic_cup': 2, 'plastic_film': 1, 'plastic_bag': 1, 'plastic_bottle': 2} |
| plastic_film | 8/1/4 | 7/3/3 | {'plastic_cup': 1} | {'plastic_cup': 2, 'styrofoam_piece': 1} |
| drink_can | 1/6/0 | 2/5/0 | {'cigarette': 1, 'plastic_film': 1, 'plastic_bottle': 3, 'plastic_cup': 1} | {'plastic_cup': 3, 'plastic_film': 1, 'cigarette': 1} |
| plastic_cup | 4/1/0 | 4/1/0 | {'cigarette': 1} | {'plastic_bag': 1} |
| styrofoam_piece | 2/3/0 | 3/2/0 | {'plastic_wrapper': 2, 'plastic_cup': 1} | {'plastic_wrapper': 1, 'plastic_bottle': 1} |


Confusion counts use one best-overlap winner per GT only when no same-class candidate qualifies, so duplicates are not independent classification samples. Complete qualifying best-any class summaries are also retained in JSON. Material identity and uncertain annotation extent still need human review.

## Raw score distributions, memory and runtime

Each evaluated scale has42300 raw query rows. Sigmoid-derived class scores are not calibrated correctness probabilities. Counts include equality at the specified floor.


| Scale | min | p25 | median | p75 | p90 | p99 | max | >=.25 | >=.05 | Invalid clipped boxes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256/384 | 0.000888 | 0.005900 | 0.008768 | 0.013850 | 0.024703 | 0.107758 | 0.343288 | 16 | 1472 | 0 |
| 384/576 | 0.000643 | 0.005281 | 0.008340 | 0.013817 | 0.026065 | 0.112652 | 0.356226 | 30 | 1682 | 0 |



| Scale | Wall seconds | ms/image | Peak allocated bytes /GiB | Peak reserved bytes /GiB |
| --- | --- | --- | --- | --- |
| 256/384 | 23.412374 | 498.135619 | 951008768 / 0.885696 | 1019215872 / 0.949219 |
| 384/576 | 21.020865 | 447.252455 | 1208047616 / 1.125082 | 1354760192 / 1.261719 |


CUDA was synchronized for wall measurements. The baseline pass occurred first, including cold-start overhead; the high pass followed the baseline and pilot with warm kernels/caches. Its lower single-pass wall time is not evidence of generally faster high-resolution inference. Evaluation time includes validation photo loading/preprocessing, forward/loss, decoding and CPU metrics; it excludes checkpoint loading and is not isolated GPU throughput. Peak memory is the PyTorch allocator, not total process/driver use. The same local GTX1660Ti has6GiB; successful high-scale inference does **not** establish high-scale training memory fit.

Pilot image777 peak allocated/reserved bytes: 1179385344 / 1350565888; finite loss 5.593037. No OOM or precision/batch/resolution fallback occurred.

## Interpretation and ranked proposals

**The moderate scale increase helps some localization and class-correct
coverage, but does not resolve the smallest-object or classification bottleneck.**
The original <4px group remains0/14 any/same at .05 and1/14 any/same at floor0.
Within the13 <4px cigarettes the floor0 count stays1, but GT313 gains coverage
while GT314 loses it; unchanged totals do not mean identical outputs. Nine tiny
cigarettes cross the4px pixel threshold, yet the original group has no net
coverage recovery. The 4–<8px group improves any/same4/1→8/4 at .05 and4/1→9/5
at floor0. These paired output changes support input-scale sensitivity for that
bin, not a general statement that more pixels fix all small objects.

For all24 cigarettes, any/same coverage rises2/2→8/5 at .05 and5/5→11/7 at
floor0. Image777 alone rises0/0→3/0 at .05: GT278/280/284 move from no qualifying
localization to wrong-class-only, so better localization did not yield qualifying
cigarette labels there. Other images improve2/2→5/5 at .05, showing correct-class
gains outside777. GT282/283 remain difficult. The larger group retains35/36 any
coverage and improves21→23 same-class coverage at .05, but all three correct-class
regressions are originally larger objects: GT60 (film,image144) becomes no
qualifying localization; GT88 (Styrofoam,image208) and GT250 (wrapper,image762)
become wrong-class-only. There is no net larger-bin coverage decline, but the
individual regressions matter.

Overall correct/wrong-only/no-localization changes28/21/26→33/23/19 at .05.
Thus seven additional annotations gain any-class localization in aggregate,
but the class-related failure count rises21→23. This includes newly localized
wrong-class candidates rather than a uniform deterioration in class assignment.
Coverage is non-exclusive and not an accuracy or calibrated-confidence measure.
Wrapper same-class coverage rises4→5/11 even though its AP falls0.118247→0.071429;
cup AP falls0.432673→0.358376. Ranked COCO quality and coverage are different.

AP increases by0.043071, but the macro gain is heavily influenced by rare classes:
bag(2 GT) contributes+0.029141 and Styrofoam(5 GT)+0.018397 to the aggregate delta,
offset by wrapper−0.005852 and cup−0.009287 and other class shifts. Cigarette AP
contributes only+0.001410. The overall improvement cannot be attributed solely
to tiny-cigarette recovery. Two rare classes contributing more than the net
gain limits generalization claims. Saved GT material/extent uncertainty remains
separate from automated class agreement.

**Prespecified hypothesis assessment:** recovery in the 4–<8px bin supports the
limited hypothesis for moderately small objects. No net recovery in the original
<4px bin, persistent class-only failures and three larger-object regressions
mean the1.5× inference increase does not resolve the broader bottleneck. There
was no numerical acceptance margin for “meaningful” in the authorization; no
post-hoc significance threshold or production acceptance is invented. These
are fixed-weight within-GT comparisons for one checkpoint/validation set, not
new training evidence or a multi-checkpoint replication.

Ranked next-step proposals only:

1. Review the saved paired class changes and regressions with a fixed human
   material/extent rubric, including newly localized image777 cigarettes and
   GT60/88/250. This has no model-compute cost and addresses the now visible
   localization-to-class gap; it does not authorize label edits.
2. If separately authorized, replicate the same fixed scale pair on one other
   preselected frozen checkpoint to test whether the effect survives checkpoint
   choice and rare-class sensitivity. Keep all endpoints/bins fixed, avoiding a
   resolution search over the same small validation set.
3. Only after owner review, consider a separately designed classification or
   training study. Do not mix scale, loss, prompts, labels and learning-rate
   changes. Successful inference provides no576-scale training memory evidence.

None of these follow-ups ran. No checkpoint, display default or production
configuration was promoted or changed.


PowerShell capture reported a shell exit status1 while redirecting tqdm progress from stderr as `NativeCommandError`. The runner itself reached completed/final-completed, every gate passed, the full artifact set is present, and its final integrity record is clean. This wrapper/progress diagnostic was not treated as numerical failure and did not trigger any repeat model pass.

## Verification, preservation and stop state

The private directory retains `record.json`, software logs/results, fresh before-hashes, source-review diff, both-scale geometry JSON/CSV, exact baseline replay JSON, pilot JSON, high evaluation JSON, all75 per-object CSV/JSON and fixed-bin/transition summary. No annotated photos were saved. The original failed attempt and prior runs are unchanged.

The experiment rehashed all **1497 protected files** after model execution, with no changed bytes. This covers all prior runs and40 corrected checkpoint bundles, core/GUI scripts, tests, configuration including detection.json, prior manifests/reports, snapshot, source annotations and train/validation split metadata plus validation photos. Training-photo bytes were deliberately never opened or hashed, and real test.json was blocked. There were189 authorized validation photo decodes:94 CPU geometry +47 baseline +1 pilot +47 high. `training_state.pt` was never deserialized, no model weights changed, no backward or optimizer operation executed.

Only the specifically required current documentation is updated after that immutable-input verification; its previous bytes are separately archived and final checks distinguish these authorized document updates from protected model/data/core files. Relative links, Python syntax and `git diff --check` are checked separately. No dependencies, uploads, commits, pushes, merges or promotions. Stop for owner review; no training or additional scale experiment is authorized.
