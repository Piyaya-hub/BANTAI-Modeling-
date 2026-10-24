# Stage 4 — Multi-checkpoint input-resolution replication (D-056)

## Executive summary

Completed the two preselected frozen model states on the exact47 validation images /75 annotations. All four gates passed. Exactly188 real model forwards ran:47 per model per scale for Pretrained and seed123/1e-5/E2. Both low-scale reproductions match saved metrics, predictions,42300 raw queries and loss exactly. D-055 seed42/1e-5/E1 is reused from hashed saved artifacts and was **not rerun on GPU**.


| Model | AP low→high | Δ AP | 7-class AP without bag low→high | Δ no-bag AP | AR100 low→high |
| --- | --- | --- | --- | --- | --- |
| Pretrained | 0.111576→0.147847 | 0.036271 | 0.093769→0.095418 | 0.001650 | 0.363077→0.426109 |
| 42 / 1e-5 / E1 | 0.203932→0.247003 | 0.043071 | 0.136242→0.152161 | 0.015919 | 0.396766→0.474763 |
| 123 / 1e-5 / E2 | 0.152880→0.140163 | -0.012717 | 0.102584→0.124118 | 0.021534 | 0.274295→0.324815 |


**Partial replication:** input-scale increases consistently improve4–<8px
any/same-class coverage and cigarette candidate coverage across all three fixed
states. Aggregate AP is not uniformly improved: it rises for Pretrained and the
D-055 reference but falls for seed123/E2. Removing the rare bag class makes the
seven-class AP delta positive for all three, so the shared localization benefit
is not solely a bag artifact, while aggregate AP sensitivity is checkpoint/class
dependent. Stronger learned cigarette classification at seed123/E2 converts more
newly localized boxes, but many correct-class candidates remain below .05.
This is not a promoted resolution default or evidence for changing training.


## Authorization, gates and fixed variables

Run: `runs\training\resolution-replication-adf0cda0193041e38be1380b29b3e50b`. D-056 date:2026-10-23. One attempt, zero automatic retries, no third model GPU pass or extra memory pilot. The D-055 failed and completed attempt directories remain unchanged. Models are the original pinned GroundingDINO-T snapshot and seed123/LR1e-5/E2. The third paired result is read-only seed42/LR1e-5/E1 from D-055.

Input scale/padding policy is the only within-model variable:256/384 with384-square pad versus384/576 with576-square pad. Valid content retains aspect ratio. Fixed prompt/taxonomy,900 queries, FP32, batch1, disabled TF32, CUBLAS workspace :4096:8, saved native adapter/architecture and corrected COCO v2 remain unchanged. Every real call was frozen, eval and inference_mode. Each model was loaded once; between passes the inactive model was on CPU, so only the model being evaluated occupied the GPU. CPU offload did not alter parameter hashes.


| Gate | Actual evidence |
| --- | --- |
| 1 Guarded software | 229 tests, zero failures/errors; 14 documented skips; all4 resolution tests passed |
| 2 Provenance | 1521 fresh protected-file hashes; all40 checkpoint bundles; D055 archived artifacts and dual pretrained anchors verified |
| 3 Both exact baselines | Both47-image low evaluations passed all equality fields before either high evaluation; weights unchanged |
| 4 Both high evaluations | Two47-image high passes,188 total forwards,188 validation-photo reads; all parameter hashes unchanged |


Software guards reuse D-055 scoped temporary/in-memory fixture saves and block actual image exports, autograd/backward and optimizer steps. Training/gradient-execution fixtures are skipped to preserve zero backward passes; one skip is the pre-existing Windows symlink privilege issue. The added test verifies multi-model locked-bin identities, transition counts, common GT sets and mismatched-GT rejection. Logs retain exact skip reasons. Synthetic software fixtures are not extra real model inference.

Pretrained low outputs exactly match both the seed123-run baseline and Step1B corrected baseline. E2 exactly matches its own saved validation-epoch-2.json. All fields use full precision; maximum compared numerical delta is0. Artifact verification never deserializes training_state.pt; torch.load was blocked for real model loading/evaluation and safetensors were used. No core training/evaluation/GUI files or inference configuration were edited.

## Overall COCO v2, loss and rare-bag control

AP/AR values are fractions, not accuracy or calibrated probabilities. All six metric records recompute exactly from saved predictions. Seven-class macro-AP is a descriptive mean excluding plastic_bag, not a changed official COCO protocol or class taxonomy. The bag class has only2 GT annotations.

### Pretrained


| Metric | 256/384 | 384/576 | Δ high−low |
| --- | --- | --- | --- |
| AP | 0.111576 | 0.147847 | 0.036271 |
| AP50 | 0.158253 | 0.179470 | 0.021217 |
| AP75 | 0.135975 | 0.165594 | 0.029619 |
| AP_small | 0.000000 | 0.000000 | 0.000000 |
| AP_medium | 0.005941 | 0.007261 | 0.001320 |
| AP_large | 0.129427 | 0.172653 | 0.043226 |
| AR1 | 0.294386 | 0.299178 | 0.004792 |
| AR10 | 0.363077 | 0.416109 | 0.053032 |
| AR100 | 0.363077 | 0.426109 | 0.063032 |
| AR_small | 0.000000 | 0.000000 | 0.000000 |
| AR_medium | 0.005769 | 0.013462 | 0.007692 |
| AR_large | 0.455504 | 0.532669 | 0.077165 |
| mean_loss_per_image | 10.553226 | 9.720814 | -0.832412 |
| macro_AP_excluding_bag | 0.093769 | 0.095418 | 0.001650 |


### 42 / 1e-5 / E1


| Metric | 256/384 | 384/576 | Δ high−low |
| --- | --- | --- | --- |
| AP | 0.203932 | 0.247003 | 0.043071 |
| AP50 | 0.254720 | 0.295777 | 0.041057 |
| AP75 | 0.239987 | 0.279521 | 0.039534 |
| AP_small | 0.019802 | 0.029208 | 0.009406 |
| AP_medium | 0.008101 | 0.013996 | 0.005895 |
| AP_large | 0.240848 | 0.295150 | 0.054303 |
| AR1 | 0.354968 | 0.414916 | 0.059948 |
| AR10 | 0.388393 | 0.462543 | 0.074151 |
| AR100 | 0.396766 | 0.474763 | 0.077997 |
| AR_small | 0.018182 | 0.027273 | 0.009091 |
| AR_medium | 0.015385 | 0.023077 | 0.007692 |
| AR_large | 0.498383 | 0.595516 | 0.097134 |
| mean_loss_per_image | 7.019476 | 6.795396 | -0.224081 |
| macro_AP_excluding_bag | 0.136242 | 0.152161 | 0.015919 |


### 123 / 1e-5 / E2


| Metric | 256/384 | 384/576 | Δ high−low |
| --- | --- | --- | --- |
| AP | 0.152880 | 0.140163 | -0.012717 |
| AP50 | 0.178946 | 0.170532 | -0.008414 |
| AP75 | 0.164636 | 0.153433 | -0.011202 |
| AP_small | 0.037947 | 0.052387 | 0.014439 |
| AP_medium | 0.013685 | 0.038059 | 0.024374 |
| AP_large | 0.180709 | 0.162094 | -0.018614 |
| AR1 | 0.216515 | 0.248278 | 0.031763 |
| AR10 | 0.268205 | 0.314598 | 0.046393 |
| AR100 | 0.274295 | 0.324815 | 0.050519 |
| AR_small | 0.054545 | 0.063636 | 0.009091 |
| AR_medium | 0.028846 | 0.157692 | 0.128846 |
| AR_large | 0.325257 | 0.376327 | 0.051070 |
| mean_loss_per_image | 5.526925 | 5.872892 | 0.345967 |
| macro_AP_excluding_bag | 0.102584 | 0.124118 | 0.021534 |


## Locked size-bin coverage

The same75 GTs stay in their D-055256/384 shortest-side bins N14/10/15/36. Geometry/pixel transforms are model-independent and reused from the verified D-055 processor geometry; no extra real forwards or photo decoding were used for binning. Diagnostic IoU>=.50 coverage is independent per GT and non-exclusive. One query may cover multiple annotations; candidate counts are not physical object counts.

All900 includes every valid saved query; top100 uses reconstructed official stable membership. Scores0 and .05 are inclusive. Saved outputs contain one winning class/query and phrase; alternative class logits, masks and unsaved proposals are not recoverable. These measurements remain separate from official ranked COCO AP.

### all 900 (diagnostic) @ 0.05


| Model | Original bin | N | Low any/same | High any/same | Δ any | Δ same |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained | extremely small <4 | 14 | 0/0 | 1/0 | 1 | 0 |
| Pretrained | very small 4–<8 | 10 | 3/0 | 6/2 | 3 | 2 |
| Pretrained | small 8–<16 | 15 | 9/4 | 14/4 | 5 | 0 |
| Pretrained | larger ≥16 | 36 | 36/24 | 36/23 | 0 | -1 |
| 42 / 1e-5 / E1 | extremely small <4 | 14 | 0/0 | 0/0 | 0 | 0 |
| 42 / 1e-5 / E1 | very small 4–<8 | 10 | 4/1 | 8/4 | 4 | 3 |
| 42 / 1e-5 / E1 | small 8–<16 | 15 | 10/6 | 13/6 | 3 | 0 |
| 42 / 1e-5 / E1 | larger ≥16 | 36 | 35/21 | 35/23 | 0 | 2 |
| 123 / 1e-5 / E2 | extremely small <4 | 14 | 2/2 | 2/2 | 0 | 0 |
| 123 / 1e-5 / E2 | very small 4–<8 | 10 | 5/2 | 9/5 | 4 | 3 |
| 123 / 1e-5 / E2 | small 8–<16 | 15 | 11/7 | 13/9 | 2 | 2 |
| 123 / 1e-5 / E2 | larger ≥16 | 36 | 35/15 | 35/17 | 0 | 2 |


### all 900 (diagnostic) @ 0.0


| Model | Original bin | N | Low any/same | High any/same | Δ any | Δ same |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained | extremely small <4 | 14 | 1/1 | 2/0 | 1 | -1 |
| Pretrained | very small 4–<8 | 10 | 4/1 | 9/3 | 5 | 2 |
| Pretrained | small 8–<16 | 15 | 12/4 | 15/4 | 3 | 0 |
| Pretrained | larger ≥16 | 36 | 36/26 | 36/26 | 0 | 0 |
| 42 / 1e-5 / E1 | extremely small <4 | 14 | 1/1 | 1/1 | 0 | 0 |
| 42 / 1e-5 / E1 | very small 4–<8 | 10 | 4/1 | 9/5 | 5 | 4 |
| 42 / 1e-5 / E1 | small 8–<16 | 15 | 14/8 | 15/9 | 1 | 1 |
| 42 / 1e-5 / E1 | larger ≥16 | 36 | 36/26 | 36/28 | 0 | 2 |
| 123 / 1e-5 / E2 | extremely small <4 | 14 | 3/3 | 8/8 | 5 | 5 |
| 123 / 1e-5 / E2 | very small 4–<8 | 10 | 6/3 | 10/7 | 4 | 4 |
| 123 / 1e-5 / E2 | small 8–<16 | 15 | 14/9 | 15/11 | 1 | 2 |
| 123 / 1e-5 / E2 | larger ≥16 | 36 | 36/19 | 36/20 | 0 | 1 |


### official top100 @ 0.05


| Model | Original bin | N | Low any/same | High any/same | Δ any | Δ same |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained | extremely small <4 | 14 | 0/0 | 1/0 | 1 | 0 |
| Pretrained | very small 4–<8 | 10 | 3/0 | 6/2 | 3 | 2 |
| Pretrained | small 8–<16 | 15 | 9/4 | 14/4 | 5 | 0 |
| Pretrained | larger ≥16 | 36 | 35/24 | 36/23 | 1 | -1 |
| 42 / 1e-5 / E1 | extremely small <4 | 14 | 0/0 | 0/0 | 0 | 0 |
| 42 / 1e-5 / E1 | very small 4–<8 | 10 | 4/1 | 8/4 | 4 | 3 |
| 42 / 1e-5 / E1 | small 8–<16 | 15 | 10/6 | 13/6 | 3 | 0 |
| 42 / 1e-5 / E1 | larger ≥16 | 36 | 35/21 | 35/23 | 0 | 2 |
| 123 / 1e-5 / E2 | extremely small <4 | 14 | 2/2 | 2/2 | 0 | 0 |
| 123 / 1e-5 / E2 | very small 4–<8 | 10 | 5/2 | 9/5 | 4 | 3 |
| 123 / 1e-5 / E2 | small 8–<16 | 15 | 11/7 | 13/9 | 2 | 2 |
| 123 / 1e-5 / E2 | larger ≥16 | 36 | 35/15 | 35/17 | 0 | 2 |


### official top100 @ 0.0


| Model | Original bin | N | Low any/same | High any/same | Δ any | Δ same |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained | extremely small <4 | 14 | 0/0 | 1/0 | 1 | 0 |
| Pretrained | very small 4–<8 | 10 | 3/0 | 6/2 | 3 | 2 |
| Pretrained | small 8–<16 | 15 | 10/4 | 14/4 | 4 | 0 |
| Pretrained | larger ≥16 | 36 | 35/24 | 36/23 | 1 | -1 |
| 42 / 1e-5 / E1 | extremely small <4 | 14 | 0/0 | 0/0 | 0 | 0 |
| 42 / 1e-5 / E1 | very small 4–<8 | 10 | 4/1 | 9/5 | 5 | 4 |
| 42 / 1e-5 / E1 | small 8–<16 | 15 | 13/8 | 14/6 | 1 | -2 |
| 42 / 1e-5 / E1 | larger ≥16 | 36 | 36/22 | 36/27 | 0 | 5 |
| 123 / 1e-5 / E2 | extremely small <4 | 14 | 3/3 | 3/3 | 0 | 0 |
| 123 / 1e-5 / E2 | very small 4–<8 | 10 | 5/2 | 9/5 | 4 | 3 |
| 123 / 1e-5 / E2 | small 8–<16 | 15 | 13/8 | 15/10 | 2 | 2 |
| 123 / 1e-5 / E2 | larger ≥16 | 36 | 36/16 | 36/18 | 0 | 2 |


## Cigarettes:24 GTs across7 images

Original cigarette bins are13 (<4px),7 (4–<8px),4 (8–<16px). Image777 has7; other images have17. Image1161 is a six-GT subset of those other images, not an additional independent group. Class agreement is annotation-relative; owner-reported material/extent uncertainties remain unresolved.

### all 900 (diagnostic) @ 0.05


| Model | Group | N | Low any/same | High any/same | Δ any | Δ same |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained | extremely small <4 | 13 | 0/0 | 1/0 | 1 | 0 |
| Pretrained | very small 4–<8 | 7 | 0/0 | 3/2 | 3 | 2 |
| Pretrained | small 8–<16 | 4 | 1/1 | 4/1 | 3 | 0 |
| Pretrained | all | 24 | 1/1 | 8/3 | 7 | 2 |
| Pretrained | image777 | 7 | 0/0 | 4/0 | 4 | 0 |
| Pretrained | other_images | 17 | 1/1 | 4/3 | 3 | 2 |
| Pretrained | image1161 | 6 | 0/0 | 2/2 | 2 | 2 |
| 42 / 1e-5 / E1 | extremely small <4 | 13 | 0/0 | 0/0 | 0 | 0 |
| 42 / 1e-5 / E1 | very small 4–<8 | 7 | 1/1 | 5/4 | 4 | 3 |
| 42 / 1e-5 / E1 | small 8–<16 | 4 | 1/1 | 3/1 | 2 | 0 |
| 42 / 1e-5 / E1 | all | 24 | 2/2 | 8/5 | 6 | 3 |
| 42 / 1e-5 / E1 | image777 | 7 | 0/0 | 3/0 | 3 | 0 |
| 42 / 1e-5 / E1 | other_images | 17 | 2/2 | 5/5 | 3 | 3 |
| 42 / 1e-5 / E1 | image1161 | 6 | 0/0 | 3/3 | 3 | 3 |
| 123 / 1e-5 / E2 | extremely small <4 | 13 | 2/2 | 2/2 | 0 | 0 |
| 123 / 1e-5 / E2 | very small 4–<8 | 7 | 2/2 | 6/5 | 4 | 3 |
| 123 / 1e-5 / E2 | small 8–<16 | 4 | 2/2 | 3/3 | 1 | 1 |
| 123 / 1e-5 / E2 | all | 24 | 6/6 | 11/10 | 5 | 4 |
| 123 / 1e-5 / E2 | image777 | 7 | 1/1 | 4/3 | 3 | 2 |
| 123 / 1e-5 / E2 | other_images | 17 | 5/5 | 7/7 | 2 | 2 |
| 123 / 1e-5 / E2 | image1161 | 6 | 1/1 | 3/3 | 2 | 2 |


### all 900 (diagnostic) @ 0.0


| Model | Group | N | Low any/same | High any/same | Δ any | Δ same |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained | extremely small <4 | 13 | 1/1 | 2/0 | 1 | -1 |
| Pretrained | very small 4–<8 | 7 | 1/1 | 6/3 | 5 | 2 |
| Pretrained | small 8–<16 | 4 | 3/1 | 4/1 | 1 | 0 |
| Pretrained | all | 24 | 5/3 | 12/4 | 7 | 1 |
| Pretrained | image777 | 7 | 2/0 | 5/0 | 3 | 0 |
| Pretrained | other_images | 17 | 3/3 | 7/4 | 4 | 1 |
| Pretrained | image1161 | 6 | 0/0 | 3/2 | 3 | 2 |
| 42 / 1e-5 / E1 | extremely small <4 | 13 | 1/1 | 1/1 | 0 | 0 |
| 42 / 1e-5 / E1 | very small 4–<8 | 7 | 1/1 | 6/5 | 5 | 4 |
| 42 / 1e-5 / E1 | small 8–<16 | 4 | 3/3 | 4/1 | 1 | -2 |
| 42 / 1e-5 / E1 | all | 24 | 5/5 | 11/7 | 6 | 2 |
| 42 / 1e-5 / E1 | image777 | 7 | 2/2 | 5/1 | 3 | -1 |
| 42 / 1e-5 / E1 | other_images | 17 | 3/3 | 6/6 | 3 | 3 |
| 42 / 1e-5 / E1 | image1161 | 6 | 0/0 | 3/3 | 3 | 3 |
| 123 / 1e-5 / E2 | extremely small <4 | 13 | 3/3 | 8/8 | 5 | 5 |
| 123 / 1e-5 / E2 | very small 4–<8 | 7 | 3/3 | 7/7 | 4 | 4 |
| 123 / 1e-5 / E2 | small 8–<16 | 4 | 3/3 | 4/4 | 1 | 1 |
| 123 / 1e-5 / E2 | all | 24 | 9/9 | 19/19 | 10 | 10 |
| 123 / 1e-5 / E2 | image777 | 7 | 3/3 | 7/7 | 4 | 4 |
| 123 / 1e-5 / E2 | other_images | 17 | 6/6 | 12/12 | 6 | 6 |
| 123 / 1e-5 / E2 | image1161 | 6 | 1/1 | 5/5 | 4 | 4 |


### official top100 @ 0.05


| Model | Group | N | Low any/same | High any/same | Δ any | Δ same |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained | extremely small <4 | 13 | 0/0 | 1/0 | 1 | 0 |
| Pretrained | very small 4–<8 | 7 | 0/0 | 3/2 | 3 | 2 |
| Pretrained | small 8–<16 | 4 | 1/1 | 4/1 | 3 | 0 |
| Pretrained | all | 24 | 1/1 | 8/3 | 7 | 2 |
| Pretrained | image777 | 7 | 0/0 | 4/0 | 4 | 0 |
| Pretrained | other_images | 17 | 1/1 | 4/3 | 3 | 2 |
| Pretrained | image1161 | 6 | 0/0 | 2/2 | 2 | 2 |
| 42 / 1e-5 / E1 | extremely small <4 | 13 | 0/0 | 0/0 | 0 | 0 |
| 42 / 1e-5 / E1 | very small 4–<8 | 7 | 1/1 | 5/4 | 4 | 3 |
| 42 / 1e-5 / E1 | small 8–<16 | 4 | 1/1 | 3/1 | 2 | 0 |
| 42 / 1e-5 / E1 | all | 24 | 2/2 | 8/5 | 6 | 3 |
| 42 / 1e-5 / E1 | image777 | 7 | 0/0 | 3/0 | 3 | 0 |
| 42 / 1e-5 / E1 | other_images | 17 | 2/2 | 5/5 | 3 | 3 |
| 42 / 1e-5 / E1 | image1161 | 6 | 0/0 | 3/3 | 3 | 3 |
| 123 / 1e-5 / E2 | extremely small <4 | 13 | 2/2 | 2/2 | 0 | 0 |
| 123 / 1e-5 / E2 | very small 4–<8 | 7 | 2/2 | 6/5 | 4 | 3 |
| 123 / 1e-5 / E2 | small 8–<16 | 4 | 2/2 | 3/3 | 1 | 1 |
| 123 / 1e-5 / E2 | all | 24 | 6/6 | 11/10 | 5 | 4 |
| 123 / 1e-5 / E2 | image777 | 7 | 1/1 | 4/3 | 3 | 2 |
| 123 / 1e-5 / E2 | other_images | 17 | 5/5 | 7/7 | 2 | 2 |
| 123 / 1e-5 / E2 | image1161 | 6 | 1/1 | 3/3 | 2 | 2 |


### official top100 @ 0.0


| Model | Group | N | Low any/same | High any/same | Δ any | Δ same |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained | extremely small <4 | 13 | 0/0 | 1/0 | 1 | 0 |
| Pretrained | very small 4–<8 | 7 | 0/0 | 3/2 | 3 | 2 |
| Pretrained | small 8–<16 | 4 | 1/1 | 4/1 | 3 | 0 |
| Pretrained | all | 24 | 1/1 | 8/3 | 7 | 2 |
| Pretrained | image777 | 7 | 0/0 | 4/0 | 4 | 0 |
| Pretrained | other_images | 17 | 1/1 | 4/3 | 3 | 2 |
| Pretrained | image1161 | 6 | 0/0 | 2/2 | 2 | 2 |
| 42 / 1e-5 / E1 | extremely small <4 | 13 | 0/0 | 0/0 | 0 | 0 |
| 42 / 1e-5 / E1 | very small 4–<8 | 7 | 1/1 | 6/5 | 5 | 4 |
| 42 / 1e-5 / E1 | small 8–<16 | 4 | 3/3 | 4/1 | 1 | -2 |
| 42 / 1e-5 / E1 | all | 24 | 4/4 | 10/6 | 6 | 2 |
| 42 / 1e-5 / E1 | image777 | 7 | 2/2 | 5/1 | 3 | -1 |
| 42 / 1e-5 / E1 | other_images | 17 | 2/2 | 5/5 | 3 | 3 |
| 42 / 1e-5 / E1 | image1161 | 6 | 0/0 | 3/3 | 3 | 3 |
| 123 / 1e-5 / E2 | extremely small <4 | 13 | 3/3 | 3/3 | 0 | 0 |
| 123 / 1e-5 / E2 | very small 4–<8 | 7 | 2/2 | 6/5 | 4 | 3 |
| 123 / 1e-5 / E2 | small 8–<16 | 4 | 3/3 | 4/3 | 1 | 0 |
| 123 / 1e-5 / E2 | all | 24 | 8/8 | 13/11 | 5 | 3 |
| 123 / 1e-5 / E2 | image777 | 7 | 2/2 | 5/3 | 3 | 1 |
| 123 / 1e-5 / E2 | other_images | 17 | 6/6 | 8/8 | 2 | 2 |
| 123 / 1e-5 / E2 | image1161 | 6 | 1/1 | 3/3 | 2 | 2 |


### Image777 exact cigarette transitions at all900 /.05

Candidate columns show the best positive overlap; adequate localization requires IoU>=.50. “Wrong-only” means an adequately localized candidate but no qualifying cigarette-class candidate; it is not a human material verdict.


| Model | GT | Original bin | Low state | High state | Low best any | High best any | High best same |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 278 | very small 4–<8 | no_qualifying_localization | wrong_class_only | q54 plastic_wrapper; s=0.066164; IoU=0.295527 | q71 plastic_cup; s=0.078311; IoU=0.668666 | none |
| Pretrained | 279 | very small 4–<8 | no_qualifying_localization | no_qualifying_localization | q408 plastic_wrapper; s=0.072368; IoU=0.141320 | q62 plastic_wrapper; s=0.074061; IoU=0.104589 | none |
| Pretrained | 280 | small 8–<16 | no_qualifying_localization | wrong_class_only | q29 plastic_film; s=0.132262; IoU=0.001877 | q39 plastic_cup; s=0.060244; IoU=0.724756 | none |
| Pretrained | 281 | small 8–<16 | no_qualifying_localization | wrong_class_only | q54 plastic_wrapper; s=0.066164; IoU=0.419855 | q154 plastic_cup; s=0.050723; IoU=0.737589 | none |
| Pretrained | 282 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q11 styrofoam_piece; s=0.104387; IoU=0.000612 | q36 plastic_cup; s=0.079814; IoU=0.282023 | none |
| Pretrained | 283 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q408 plastic_wrapper; s=0.072368; IoU=0.001262 | q62 plastic_wrapper; s=0.074061; IoU=0.016376 | none |
| Pretrained | 284 | small 8–<16 | no_qualifying_localization | wrong_class_only | q111 styrofoam_piece; s=0.060000; IoU=0.006689 | q56 plastic_wrapper; s=0.077154; IoU=0.627081 | none |
| 42 / 1e-5 / E1 | 278 | very small 4–<8 | no_qualifying_localization | wrong_class_only | q42 styrofoam_piece; s=0.052581; IoU=0.133090 | q43 plastic_cup; s=0.050787; IoU=0.648400 | none |
| 42 / 1e-5 / E1 | 279 | very small 4–<8 | no_qualifying_localization | no_qualifying_localization | q42 styrofoam_piece; s=0.052581; IoU=0.150600 | q17 plastic_film; s=0.109799; IoU=0.002322 | none |
| 42 / 1e-5 / E1 | 280 | small 8–<16 | no_qualifying_localization | wrong_class_only | q12 plastic_film; s=0.105463; IoU=0.001801 | q19 styrofoam_piece; s=0.070925; IoU=0.748087 | none |
| 42 / 1e-5 / E1 | 281 | small 8–<16 | no_qualifying_localization | no_qualifying_localization | q42 styrofoam_piece; s=0.052581; IoU=0.204579 | q29 styrofoam_piece; s=0.067555; IoU=0.423056 | none |
| 42 / 1e-5 / E1 | 282 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q12 plastic_film; s=0.105463; IoU=0.000453 | q21 plastic_cup; s=0.060413; IoU=0.298844 | none |
| 42 / 1e-5 / E1 | 283 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q68 styrofoam_piece; s=0.075229; IoU=0.000669 | q9 styrofoam_piece; s=0.086437; IoU=0.001112 | none |
| 42 / 1e-5 / E1 | 284 | small 8–<16 | no_qualifying_localization | wrong_class_only | q12 plastic_film; s=0.105463; IoU=0.002796 | q24 styrofoam_piece; s=0.078591; IoU=0.652363 | none |
| 123 / 1e-5 / E2 | 278 | very small 4–<8 | no_qualifying_localization | wrong_class_only | q45 plastic_film; s=0.073893; IoU=0.002016 | q24 plastic_cup; s=0.052701; IoU=0.788665 | none |
| 123 / 1e-5 / E2 | 279 | very small 4–<8 | no_qualifying_localization | correct_class | q45 plastic_film; s=0.073893; IoU=0.002372 | q42 cigarette; s=0.058579; IoU=0.719825 | q42 cigarette; s=0.058579; IoU=0.719825 |
| 123 / 1e-5 / E2 | 280 | small 8–<16 | no_qualifying_localization | correct_class | q45 plastic_film; s=0.073893; IoU=0.001785 | q11 cigarette; s=0.097451; IoU=0.830025 | q11 cigarette; s=0.097451; IoU=0.830025 |
| 123 / 1e-5 / E2 | 281 | small 8–<16 | no_qualifying_localization | no_qualifying_localization | q45 plastic_film; s=0.073893; IoU=0.003118 | q19 plastic_film; s=0.052865; IoU=0.454999 | none |
| 123 / 1e-5 / E2 | 282 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q48 cigarette; s=0.061120; IoU=0.313471 | q10 cigarette; s=0.094534; IoU=0.432292 | q10 cigarette; s=0.094534; IoU=0.432292 |
| 123 / 1e-5 / E2 | 283 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q45 plastic_film; s=0.073893; IoU=0.000371 | q111 cigarette; s=0.059620; IoU=0.445101 | q111 cigarette; s=0.059620; IoU=0.445101 |
| 123 / 1e-5 / E2 | 284 | small 8–<16 | correct_class | correct_class | q96 cigarette; s=0.053945; IoU=0.581194 | q22 cigarette; s=0.080203; IoU=0.756241 | q22 cigarette; s=0.080203; IoU=0.756241 |


### Image1161 exact cigarette transitions at all900 /.05

Candidate columns show the best positive overlap; adequate localization requires IoU>=.50. “Wrong-only” means an adequately localized candidate but no qualifying cigarette-class candidate; it is not a human material verdict.


| Model | GT | Original bin | Low state | High state | Low best any | High best any | High best same |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 495 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q31 plastic_cup; s=0.052680; IoU=0.015684 | q79 cigarette; s=0.068146; IoU=0.059207 | q79 cigarette; s=0.068146; IoU=0.059207 |
| Pretrained | 496 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q288 plastic_bottle; s=0.073199; IoU=0.007034 | q11 cigarette; s=0.163185; IoU=0.283291 | q11 cigarette; s=0.163185; IoU=0.283291 |
| Pretrained | 497 | very small 4–<8 | no_qualifying_localization | correct_class | q193 plastic_film; s=0.050620; IoU=0.000889 | q47 cigarette; s=0.086884; IoU=0.568909 | q47 cigarette; s=0.086884; IoU=0.568909 |
| Pretrained | 498 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q77 styrofoam_piece; s=0.056472; IoU=0.000820 | q211 styrofoam_piece; s=0.053198; IoU=0.003629 | none |
| Pretrained | 499 | very small 4–<8 | no_qualifying_localization | no_qualifying_localization | q122 styrofoam_piece; s=0.050246; IoU=0.000481 | q89 cigarette; s=0.062484; IoU=0.493487 | q89 cigarette; s=0.062484; IoU=0.493487 |
| Pretrained | 500 | very small 4–<8 | no_qualifying_localization | correct_class | q77 styrofoam_piece; s=0.056472; IoU=0.002061 | q7 cigarette; s=0.224784; IoU=0.592892 | q7 cigarette; s=0.224784; IoU=0.592892 |
| 42 / 1e-5 / E1 | 495 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q25 cigarette; s=0.090755; IoU=0.052516 | q13 cigarette; s=0.099039; IoU=0.071661 | q13 cigarette; s=0.099039; IoU=0.071661 |
| 42 / 1e-5 / E1 | 496 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q42 plastic_film; s=0.067096; IoU=0.005283 | q5 cigarette; s=0.156842; IoU=0.313246 | q5 cigarette; s=0.156842; IoU=0.313246 |
| 42 / 1e-5 / E1 | 497 | very small 4–<8 | no_qualifying_localization | correct_class | q221 styrofoam_piece; s=0.064095; IoU=0.000428 | q22 cigarette; s=0.094184; IoU=0.603208 | q22 cigarette; s=0.094184; IoU=0.603208 |
| 42 / 1e-5 / E1 | 498 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q221 styrofoam_piece; s=0.064095; IoU=0.000268 | q86 styrofoam_piece; s=0.086817; IoU=0.000902 | none |
| 42 / 1e-5 / E1 | 499 | very small 4–<8 | no_qualifying_localization | correct_class | q221 styrofoam_piece; s=0.064095; IoU=0.000339 | q31 cigarette; s=0.081359; IoU=0.515814 | q31 cigarette; s=0.081359; IoU=0.515814 |
| 42 / 1e-5 / E1 | 500 | very small 4–<8 | no_qualifying_localization | correct_class | q20 cigarette; s=0.072006; IoU=0.321824 | q2 cigarette; s=0.198196; IoU=0.633423 | q2 cigarette; s=0.198196; IoU=0.633423 |
| 123 / 1e-5 / E2 | 495 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q14 cigarette; s=0.118885; IoU=0.090097 | q9 cigarette; s=0.140423; IoU=0.111842 | q9 cigarette; s=0.140423; IoU=0.111842 |
| 123 / 1e-5 / E2 | 496 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q21 cigarette; s=0.082819; IoU=0.034666 | q4 cigarette; s=0.232351; IoU=0.456025 | q4 cigarette; s=0.232351; IoU=0.456025 |
| 123 / 1e-5 / E2 | 497 | very small 4–<8 | no_qualifying_localization | correct_class | q33 styrofoam_piece; s=0.067773; IoU=0.000352 | q8 cigarette; s=0.176926; IoU=0.826890 | q8 cigarette; s=0.176926; IoU=0.826890 |
| 123 / 1e-5 / E2 | 498 | extremely small <4 | no_qualifying_localization | no_qualifying_localization | q33 styrofoam_piece; s=0.067773; IoU=0.000220 | q28 cigarette; s=0.097238; IoU=0.495674 | q28 cigarette; s=0.097238; IoU=0.495674 |
| 123 / 1e-5 / E2 | 499 | very small 4–<8 | no_qualifying_localization | correct_class | q33 styrofoam_piece; s=0.067773; IoU=0.000278 | q14 cigarette; s=0.136898; IoU=0.686131 | q14 cigarette; s=0.136898; IoU=0.686131 |
| 123 / 1e-5 / E2 | 500 | very small 4–<8 | correct_class | correct_class | q4 cigarette; s=0.143191; IoU=0.578241 | q2 cigarette; s=0.244701; IoU=0.818214 | q2 cigarette; s=0.244701; IoU=0.818214 |


Full24-cigarette identities and best-any/same query classes, scores and IoUs at each source/floor are in the225-row paired GT/model CSV and JSON. A missing overlap is absent under these saved/source/score conditions, not a claim the model could never detect the object.

## Paired75-GT transition matrices and cross-checkpoint consistency

Disjoint primary states: correct_class (same-class IoU>=.50 at score>=.05), wrong_class_only (any coverage but no same-class coverage), no_qualifying_localization (no any-class IoU>=.50 at this floor). There are75 GTs per matrix, not225 independent annotations.

### Pretrained


| Low→high | correct_class | wrong_class_only | no_qualifying_localization |
| --- | --- | --- | --- |
| correct_class | 24 | 4 | 0 |
| wrong_class_only | 3 | 17 | 0 |
| no_qualifying_localization | 2 | 7 | 18 |


Gained correct-class IDs: [163, 220, 242, 497, 500]. Lost correct-class IDs: [16, 250, 259, 443]. Newly localized wrong-only IDs: [64, 67, 261, 278, 280, 281, 284].

### 42 / 1e-5 / E1


| Low→high | correct_class | wrong_class_only | no_qualifying_localization |
| --- | --- | --- | --- |
| correct_class | 25 | 2 | 1 |
| wrong_class_only | 5 | 16 | 0 |
| no_qualifying_localization | 3 | 5 | 18 |


Gained correct-class IDs: [86, 106, 150, 177, 443, 497, 499, 500]. Lost correct-class IDs: [60, 88, 250]. Newly localized wrong-only IDs: [67, 278, 280, 284, 367].

### 123 / 1e-5 / E2


| Low→high | correct_class | wrong_class_only | no_qualifying_localization |
| --- | --- | --- | --- |
| correct_class | 20 | 6 | 0 |
| wrong_class_only | 9 | 18 | 0 |
| no_qualifying_localization | 4 | 2 | 16 |


Gained correct-class IDs: [64, 96, 163, 220, 242, 259, 279, 280, 305, 388, 444, 497, 499]. Lost correct-class IDs: [27, 104, 150, 162, 253, 423]. Newly localized wrong-only IDs: [67, 278].

### Common effects across all three states


| Effect | GT IDs (intersection of all three model results) |
| --- | --- |
| gain | [497] |
| loss | [] |
| newly_localized_wrong | [67, 278] |


Full support lists in `GT_effect_support` show which model(s) exhibit each GT effect; empty intersections mean no GT shares that exact effect across all three, not absence of individual gains. GT identities matter even when totals coincide. Full matrix-cell IDs remain in JSON.

## Per-class AP, AR100, disjoint states and confusion

N is fixed validation annotation support. Class-failure confusion uses one best-any overlap per GT only where no correct-class candidate qualifies, so many duplicate queries cannot inflate independent classification sample counts.

### cigarette


| Model | N | AP low | AP high | Δ AP | AR low | AR high | Δ AR | Low correct/wrong/no-loc | High correct/wrong/no-loc | Low wrong winner classes | High wrong winner classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 24 | 0.014851 | 0.014122 | -0.000730 | 0.012500 | 0.029167 | 0.016667 | 1/0/23 | 3/5/16 | {} | {'styrofoam_piece': 1, 'plastic_cup': 3, 'plastic_wrapper': 1} |
| 42 / 1e-5 / E1 | 24 | 0.020136 | 0.031416 | 0.011279 | 0.050000 | 0.075000 | 0.025000 | 2/0/22 | 5/3/16 | {} | {'plastic_cup': 1, 'styrofoam_piece': 2} |
| 123 / 1e-5 / E2 | 24 | 0.034787 | 0.078591 | 0.043805 | 0.112500 | 0.237500 | 0.125000 | 6/0/18 | 10/1/13 | {} | {'plastic_cup': 1} |


### plastic_bottle


| Model | N | AP low | AP high | Δ AP | AR low | AR high | Δ AR | Low correct/wrong/no-loc | High correct/wrong/no-loc | Low wrong winner classes | High wrong winner classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 8 | 0.038242 | 0.053966 | 0.015724 | 0.400000 | 0.425000 | 0.025000 | 4/4/0 | 4/4/0 | {'plastic_wrapper': 2, 'plastic_cup': 2} | {'plastic_cup': 3, 'plastic_bag': 1} |
| 42 / 1e-5 / E1 | 8 | 0.096181 | 0.109898 | 0.013717 | 0.462500 | 0.575000 | 0.112500 | 5/3/0 | 5/3/0 | {'plastic_cup': 2, 'plastic_wrapper': 1} | {'plastic_film': 1, 'plastic_cup': 2} |
| 123 / 1e-5 / E2 | 8 | 0.055776 | 0.087008 | 0.031233 | 0.162500 | 0.337500 | 0.175000 | 2/6/0 | 3/5/0 | {'plastic_cup': 2, 'cigarette': 1, 'plastic_film': 3} | {'plastic_film': 4, 'plastic_cup': 1} |


### plastic_bag


| Model | N | AP low | AP high | Δ AP | AR low | AR high | Δ AR | Low correct/wrong/no-loc | High correct/wrong/no-loc | Low wrong winner classes | High wrong winner classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 2 | 0.236229 | 0.514851 | 0.278623 | 0.550000 | 0.700000 | 0.150000 | 2/0/0 | 2/0/0 | {} | {} |
| 42 / 1e-5 / E1 | 2 | 0.677759 | 0.910891 | 0.233132 | 0.950000 | 0.950000 | 0.000000 | 2/0/0 | 2/0/0 | {} | {} |
| 123 / 1e-5 / E2 | 2 | 0.504950 | 0.252475 | -0.252475 | 0.500000 | 0.500000 | 0.000000 | 1/1/0 | 1/1/0 | {'cigarette': 1} | {'cigarette': 1} |


### plastic_wrapper


| Model | N | AP low | AP high | Δ AP | AR low | AR high | Δ AR | Low correct/wrong/no-loc | High correct/wrong/no-loc | Low wrong winner classes | High wrong winner classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 11 | 0.094467 | 0.074249 | -0.020218 | 0.490909 | 0.572727 | 0.081818 | 9/2/0 | 8/3/0 | {'plastic_bag': 2} | {'plastic_bottle': 2, 'plastic_bag': 1} |
| 42 / 1e-5 / E1 | 11 | 0.118247 | 0.071429 | -0.046818 | 0.272727 | 0.409091 | 0.136364 | 4/7/0 | 5/6/0 | {'plastic_film': 5, 'plastic_cup': 2} | {'plastic_cup': 2, 'plastic_film': 1, 'plastic_bag': 1, 'plastic_bottle': 2} |
| 123 / 1e-5 / E2 | 11 | 0.160149 | 0.039152 | -0.120997 | 0.327273 | 0.200000 | -0.127273 | 5/6/0 | 5/6/0 | {'plastic_film': 3, 'plastic_bottle': 2, 'cigarette': 1} | {'plastic_film': 5, 'plastic_bottle': 1} |


### plastic_film


| Model | N | AP low | AP high | Δ AP | AR low | AR high | Δ AR | Low correct/wrong/no-loc | High correct/wrong/no-loc | Low wrong winner classes | High wrong winner classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 13 | 0.001574 | 0.002621 | 0.001047 | 0.076923 | 0.107692 | 0.030769 | 2/7/4 | 2/9/2 | {'plastic_cup': 4, 'styrofoam_piece': 2, 'plastic_bag': 1} | {'plastic_cup': 4, 'styrofoam_piece': 2, 'plastic_bottle': 2, 'plastic_bag': 1} |
| 42 / 1e-5 / E1 | 13 | 0.016310 | 0.017526 | 0.001217 | 0.384615 | 0.446154 | 0.061538 | 8/1/4 | 7/3/3 | {'plastic_cup': 1} | {'plastic_cup': 2, 'styrofoam_piece': 1} |
| 123 / 1e-5 / E2 | 13 | 0.032423 | 0.032725 | 0.000301 | 0.469231 | 0.569231 | 0.100000 | 8/1/4 | 9/1/3 | {'plastic_cup': 1} | {'cigarette': 1} |


### drink_can


| Model | N | AP low | AP high | Δ AP | AR low | AR high | Δ AR | Low correct/wrong/no-loc | High correct/wrong/no-loc | Low wrong winner classes | High wrong winner classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 7 | 0.108416 | 0.124851 | 0.016436 | 0.214286 | 0.214286 | 0.000000 | 2/5/0 | 2/5/0 | {'plastic_bag': 2, 'plastic_cup': 3} | {'plastic_bottle': 1, 'plastic_cup': 3, 'plastic_bag': 1} |
| 42 / 1e-5 / E1 | 7 | 0.118812 | 0.177970 | 0.059158 | 0.114286 | 0.242857 | 0.128571 | 1/6/0 | 2/5/0 | {'cigarette': 1, 'plastic_film': 1, 'plastic_bottle': 3, 'plastic_cup': 1} | {'plastic_cup': 3, 'plastic_film': 1, 'cigarette': 1} |
| 123 / 1e-5 / E2 | 7 | 0.148515 | 0.118812 | -0.029703 | 0.142857 | 0.114286 | -0.028571 | 1/6/0 | 1/6/0 | {'plastic_bottle': 1, 'plastic_film': 1, 'cigarette': 4} | {'plastic_film': 5, 'plastic_cup': 1} |


### plastic_cup


| Model | N | AP low | AP high | Δ AP | AR low | AR high | Δ AR | Low correct/wrong/no-loc | High correct/wrong/no-loc | Low wrong winner classes | High wrong winner classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 5 | 0.340656 | 0.335897 | -0.004758 | 0.600000 | 0.660000 | 0.060000 | 4/1/0 | 4/1/0 | {'plastic_bag': 1} | {'plastic_bottle': 1} |
| 42 / 1e-5 / E1 | 5 | 0.432673 | 0.358376 | -0.074297 | 0.560000 | 0.620000 | 0.060000 | 4/1/0 | 4/1/0 | {'cigarette': 1} | {'plastic_bag': 1} |
| 123 / 1e-5 / E2 | 5 | 0.286439 | 0.304620 | 0.018182 | 0.480000 | 0.440000 | -0.040000 | 3/2/0 | 3/2/0 | {'plastic_film': 2} | {'plastic_wrapper': 1, 'plastic_film': 1} |


### styrofoam_piece


| Model | N | AP low | AP high | Δ AP | AR low | AR high | Δ AR | Low correct/wrong/no-loc | High correct/wrong/no-loc | Low wrong winner classes | High wrong winner classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 5 | 0.058173 | 0.062220 | 0.004047 | 0.560000 | 0.700000 | 0.140000 | 4/1/0 | 4/1/0 | {'plastic_cup': 1} | {'plastic_bottle': 1} |
| 42 / 1e-5 / E1 | 5 | 0.151337 | 0.298515 | 0.147178 | 0.380000 | 0.480000 | 0.100000 | 2/3/0 | 3/2/0 | {'plastic_wrapper': 2, 'plastic_cup': 1} | {'plastic_wrapper': 1, 'plastic_bottle': 1} |
| 123 / 1e-5 / E2 | 5 | 0.000000 | 0.207921 | 0.207921 | 0.000000 | 0.200000 | 0.200000 | 0/5/0 | 1/4/0 | {'plastic_film': 2, 'cigarette': 1, 'plastic_wrapper': 1, 'plastic_cup': 1} | {'plastic_wrapper': 2, 'plastic_film': 1, 'plastic_bottle': 1} |


## Scores, wall time and GPU memory

All42300 queries/state. Scores are not calibrated probabilities or accuracy; counts include equality. Invalid/zero-area boxes remain excluded from valid diagnostics.


| Model | Scale | Min | P25 | Median | P75 | P90 | P99 | Max | >=.25 | >=.05 | Invalid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 256/384 | 0.000902 | 0.013238 | 0.019574 | 0.031051 | 0.053525 | 0.187558 | 0.596656 | 216 | 4740 | 0 |
| Pretrained | 384/576 | 0.000773 | 0.010786 | 0.017549 | 0.029248 | 0.055584 | 0.209972 | 0.564774 | 248 | 4894 | 0 |
| 42 / 1e-5 / E1 | 256/384 | 0.000888 | 0.005900 | 0.008768 | 0.013850 | 0.024703 | 0.107758 | 0.343288 | 16 | 1472 | 0 |
| 42 / 1e-5 / E1 | 384/576 | 0.000643 | 0.005281 | 0.008340 | 0.013817 | 0.026065 | 0.112652 | 0.356226 | 30 | 1682 | 0 |
| 123 / 1e-5 / E2 | 256/384 | 0.000954 | 0.005965 | 0.008852 | 0.013992 | 0.023990 | 0.079393 | 0.396704 | 37 | 1068 | 0 |
| 123 / 1e-5 / E2 | 384/576 | 0.000727 | 0.005268 | 0.008354 | 0.014043 | 0.024763 | 0.088076 | 0.418207 | 47 | 1238 | 0 |



| Model | Scale | Wall sec | ms/image | Peak allocated bytes /GiB | Peak reserved bytes /GiB | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 256/384 | 23.287342 | 495.475366 | 951008768 / 0.885696 | 1019215872 / 0.949219 | D056 new |
| Pretrained | 384/576 | 20.968061 | 446.128966 | 1208047616 / 1.125082 | 1354760192 / 1.261719 | D056 new |
| 42 / 1e-5 / E1 | 256/384 | 23.412374 | 498.135619 | 951008768 / 0.885696 | 1019215872 / 0.949219 | D055 saved |
| 42 / 1e-5 / E1 | 384/576 | 21.020865 | 447.252455 | 1208047616 / 1.125082 | 1354760192 / 1.261719 | D055 saved |
| 123 / 1e-5 / E2 | 256/384 | 15.653013 | 333.042838 | 951008768 / 0.885696 | 1019215872 / 0.949219 | D056 new |
| 123 / 1e-5 / E2 | 384/576 | 20.645762 | 439.271526 | 1208047616 / 1.125082 | 1354760192 / 1.261719 | D056 new |


Measured time includes preprocessing/photo loading, forward/loss, decoding and CPU metrics, with CUDA synchronization, excluding weight loading/device transfer. Only one model resides on GPU per new pass; other model weights remain on CPU. Fixed order and warmup/caches differ, and the D-055 reference ran separately, so single-pass wall times are not a general hardware-throughput ranking. Allocator peaks exclude total driver/process use. FP32 high-scale inference fit on the6GiB GTX1660Ti does not establish training fit.

## Interpretation of the prespecified replication hypothesis

The prespecified size/coverage hypothesis **partly replicates**. At all900 /
score>=.05, original4–<8px any/same coverage changes3/0→6/2 for Pretrained,
4/1→8/4 for42/E1 and5/2→9/5 for123/E2. The8–<16px bin gains any coverage in all
three(9→14,10→13,11→13), but same-class coverage stays4 and6 in Pretrained/42E1
and rises7→9 only in123/E2. Bag annotations are not in these small bins, so these
paired coverage improvements cannot be explained by a rare-bag AP artifact.
The <4px group still has little qualifying coverage; increased scale is not a
complete tiny-object remedy.

The overall metric result is checkpoint-dependent. Raw AP deltas are+0.036271,
+0.043071 and−0.012717 for Pretrained,42/E1 and123/E2. Bag-excluded seven-class
mean deltas are+0.001650,+0.015919 and+0.021534. Pretrained bag AP increases
0.236229→0.514851; its bag contribution to aggregate delta is+0.034828, nearly
the entire+0.036271 raw gain. Seed123 bag AP halves0.504950→0.252475, contributing
−0.031559 to aggregate AP and reversing its positive non-bag contribution.
This controls one rare class descriptively, but does not eliminate sensitivity
to other rare classes(Styrofoam has5 GT) or create a new official benchmark.

All three improve cigarette qualifying candidate coverage, but their class
conversion differs. At .05 any/same goes1/1→8/3 for Pretrained,2/2→8/5 for42/E1
and6/6→11/10 for123/E2. At floor0 it goes5/3→12/4,5/5→11/7 and9/9→19/19.
The source shorthand “highest saved cigarette recall9/24” refers here to
all900/floor0 candidate coverage, not official COCO recall averaged over IoUs.
Seed123/E2 official cigarette AR100 rises0.112500→0.237500 and AP
0.034787→0.078591; Pretrained cigarette AP slightly falls0.014851→0.014122
despite increased AR100 and candidate coverage. Scoring/ranking/localization
thresholds explain why these measures need not move together.

For image777 at .05, Pretrained gains4 any-class but0 same-class boxes/GTs;
42/E1 gains3 any and0 same;123/E2 changes1/1→4/3. Seed123 GT279 and GT280 gain
correct-class coverage, GT284 retains it, GT278 becomes wrong-class-only(cup
winner), and GT281/282/283 still have no qualifying localization. At floor0,
123/E2 reaches7/7 any/same coverage for this image, so four further correctly
assigned boxes exist but fail the fixed score condition. This is saved-candidate
evidence, not an acceptable operating threshold or human confirmation of all
seven annotated materials/bounds. Lowering scores also admits many unrelated
queries; no threshold change is authorized.

Image1161 supports gains outside777. At .05 same-class coverage changes0→2/6
for Pretrained,0→3/6 for42/E1 and1→3/6 for123/E2. In123/E2 GT497/499 gain while
GT500 remains correctly covered; GT495/496/498 stay unqualified at .05. At
floor0,123/E2 has5/6 same-class coverage. The other17 cigarette GTs outside777
also improve, as separately tabulated, so conclusions do not rest on one image.

Paired75-GT correct-class gains/losses are5/4 for Pretrained,8/3 for42/E1,
13/6 for123/E2. GT497(cigarette,image1161) is the only correct-class gain common
to all three. No correct-class regression is shared by all three; that does not
erase the individual losses. GT67(film) and GT278(cigarette) consistently shift
from no qualifying localization to wrong-class-only. Aggregate disjoint states
correct/wrong/no-localization go28/20/27→29/28/18,28/21/26→33/23/19,
and26/27/22→33/26/16. More localized objects can expose a class-assignment gap
even while coverage and recall improve.

**Hypothesis verdict:** consistent small-bin localization and cigarette
coverage gains support a shared inference-scale effect. Uniform AP superiority
and full same-class conversion do not replicate. The reversal in raw seed123 AP,
its positive bag-excluded AP and differing per-GT transitions show that rare-class
composition and learned classification/ranking remain material. This controlled
within-model policy comparison does not isolate pixels from padding/proposal
effects, prove material identity, estimate population significance, or establish
high-resolution training fit. Three fixed states on the same75 annotations are
not three independent validation datasets.

Next-step proposals only: first review the paired class/annotation cases and
regressions with a fixed human rubric, including GT67/278 and all individual
losses. If separately authorized, a narrowly specified class-score/ranking
diagnostic could inspect margins/GT correspondence that winner-only saved
outputs cannot resolve. Do not launch a resolution search or combine training,
prompt, label, loss and threshold changes based on this small validation set.
No new experiment, annotation edit or promotion follows this report.


## Preservation, artifacts and stop state

The experiment verified **1521 fresh protected hashes** after the188 model forwards, with zero changed bytes. This includes both D-055 attempt directories, all prior runs,40 corrected checkpoint bundles, snapshot, dataset/rights/split metadata, original source annotation file, authorized validation photos, current scripts/tests/config and prior documentation. No training-photo bytes were read or hashed; real test.json and non-validation photo paths were blocked. `config/detection.json` and all core/GUI files stayed unchanged.

Private outputs include `record.json`, software log/results, `protected-before.json`, both exact replay JSONs, both high-scale evaluation JSONs, `three-model-summary.json`, and `three-model-per-object-comparison.csv`/JSON. The CSV has225 GT/model rows (75 GT×3 models), each side-by-side across the two scales and all four source/floor conditions. No original run files or manifests were overwritten. Source copies and execution evidence are archived locally.

Only authorized current documentation is updated after the protected-input check; its previous bytes are archived and final verification distinguishes those maintenance edits. Relative Markdown links, helper Python syntax and Git whitespace are checked. No backward/optimizer steps, training-photo/test access, architecture/core/GUI changes, dependency changes, annotated exports, uploads, commits, pushes, merges or promotion. Stop for owner review; no additional model pass or follow-up experiment is authorized.
