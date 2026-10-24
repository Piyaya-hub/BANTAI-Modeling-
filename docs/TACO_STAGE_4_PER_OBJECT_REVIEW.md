# Stage 4 — Per-object detection error investigation

D-053, 2026-10-22. Saved diagnostics and opt-in visualization only. No model loading, inference, training, optimizer steps, test evaluation or annotation edits. Human review is pending.

## Main finding

The top-five display conceals some qualifying predictions, but it is not the
only reason cigarettes appear missing. For the previously launched seed123 /
1e-5 / E2, GT284 has a cigarette query q96 (score 0.053945, IoU 0.581194)
outside the default five. GT278 and GT280 have same-class IoU >=.50 candidates
only below .05 (scores 0.016279 and 0.045389). GT279/281/282/283 lack same-class
IoU >=.50 even at floor zero in that state. The wrapper GT259 has an aligned
bottle-labelled candidate but no same-class IoU >=.50 candidate in that state.
GT282 and GT283 are COCO-small cigarettes and lack same-class IoU >=.50 across
all nine saved states even at floor zero. This diagnoses these saved predictions;
it does not establish the objects' true material or detection impossibility.

Across the selected 20 images /46 annotations, correct-class coverage at .05
ranges from 10 to15 GT objects across trained states, while any-class coverage
ranges from19 to24. The default five cover only7 to12 same-class GT objects.
These are purposive-subset diagnostics, not new benchmark results.

## Scope, source identity and limits

The unchanged 20-image review manifest supplies image IDs and four explicit run choices. Each run was loaded through the existing hash/protocol/corpus/rights/checkpoint verifier; E1/E2 come from that same selected run. All four pretrained raw-query sets agree exactly. The original manifest and historical artifacts remain unchanged. No COCO protocol, filtering, taxonomy or metrics were changed.

Private evidence: `runs\training\per-object-review-ad37feb7b2234305bf35215410f98686`: `per-object-diagnostics.json`, `per-object-summary.csv`, `protected-before.json`, runtime checks and completion record. JSON is authoritative; the CSV contains one primary all900/.05 row per GT/state. Full overlap lists are retained in JSON for primary scope.

All 900 stored queries per image are available, including queries outside the official top100. Each retains only its winning class, score, clipped box and decoded phrase. Alternative per-class logits, masks, unsaved proposals or earlier decoder candidates cannot be recovered. Invalid/zero-area boxes are excluded. Primary diagnostics use all valid saved queries, score **>=.05**, IoU **>=.50**, independently for each annotation. Best overlap maximizes positive IoU, then score, then lowest query ID; a non-overlapping same-class query is reported as absent, not an overlapping detection. Scores are not accuracy.

Coverage is non-exclusive: one query may cover more than one GT. Existing class-aware greedy top100 matching is also retained separately; it does not replace official COCO evaluation. Empty coverage means no qualifying candidate in these saved outputs under these conditions, not that the model could never detect the object.

The former drawing cap shows only five passing official predictions. Tables already contained more rows, but image overlays hid them. The new explicit object mode ignores that global rank-five drawing subset for analysis, prioritizes best same-class and differing-class overlaps, and lists every eligible positive-IoU overlap. The standard five-box GUI behavior remains unchanged without `--per-object`.

## Image 777: ground-truth breakdown


| GT ID | Category | Original bbox [x,y,w,h] | COCO area | COCO-small (<1024) |
| --- | --- | --- | --- | --- |
| 259 | plastic_wrapper | [975.0, 81.0, 116.0, 263.0] | 21735.0 | False |
| 278 | cigarette | [1086.0, 984.0, 38.0, 68.0] | 1372.0 | False |
| 279 | cigarette | [1032.0, 936.0, 40.0, 76.0] | 1346.0 | False |
| 280 | cigarette | [1040.0, 772.0, 44.0, 52.0] | 1084.0 | False |
| 281 | cigarette | [1112.0, 954.0, 54.0, 74.0] | 1696.0 | False |
| 282 | cigarette | [1124.0, 786.0, 16.0, 36.0] | 498.0 | True |
| 283 | cigarette | [1016.0, 900.0, 14.0, 34.0] | 360.0 | True |
| 284 | cigarette | [1622.0, 294.0, 74.0, 48.0] | 1656.0 | False |


Eight annotations: seven cigarettes and one plastic wrapper. These are saved annotation labels, not a new human material verdict. The following tables include the pretrained state once and all eight trained states; each selected GUI run uses the same pretrained baseline and only its own two epochs.

### GT259 — plastic_wrapper — bbox [975.0, 81.0, 116.0, 263.0]

Best any / same columns: query ID / predicted class / model score / IoU. Primary scope is all900, score >=.05.


| State | Best any | Best same | Any ≥.50 | Same ≥.50 | Same visible in default five |
| --- | --- | --- | --- | --- | --- |
| Pretrained | q15 / plastic_cup / 0.266342 / 0.803671 | q24 / plastic_wrapper / 0.121183 / 0.662020 | True | True | False |
| 42 / 1e-5 / E1 | q23 / plastic_film / 0.080930 / 0.901502 | q19 / plastic_wrapper / 0.094135 / 0.324482 | True | False | False |
| 42 / 1e-5 / E2 | q9 / plastic_film / 0.158777 / 0.914682 | q4 / plastic_wrapper / 0.150021 / 0.688317 | True | True | True |
| 42 / 5e-6 / E1 | q4 / plastic_bottle / 0.226216 / 0.844769 | q12 / plastic_wrapper / 0.083175 / 0.434134 | True | False | False |
| 42 / 5e-6 / E2 | q20 / plastic_bottle / 0.156334 / 0.876231 | q7 / plastic_wrapper / 0.077773 / 0.432321 | True | False | False |
| 123 / 1e-5 / E1 | q17 / plastic_bottle / 0.154616 / 0.869622 | q5 / plastic_wrapper / 0.087775 / 0.402437 | True | False | False |
| 123 / 1e-5 / E2 | q13 / plastic_bottle / 0.199203 / 0.895646 | q12 / plastic_wrapper / 0.062064 / 0.272324 | True | False | False |
| 123 / 5e-6 / E1 | q10 / plastic_bottle / 0.145145 / 0.855022 | q6 / plastic_wrapper / 0.091142 / 0.452593 | True | False | False |
| 123 / 5e-6 / E2 | q14 / plastic_bottle / 0.194094 / 0.882120 | q7 / plastic_wrapper / 0.068792 / 0.419222 | True | False | False |


Without a score floor (all900):


| State | Best any (score / IoU retained) | Best same (score / IoU retained) |
| --- | --- | --- |
| Pretrained | q15 / plastic_cup / 0.266342 / 0.803671 | q24 / plastic_wrapper / 0.121183 / 0.662020 |
| 42 / 1e-5 / E1 | q23 / plastic_film / 0.080930 / 0.901502 | q19 / plastic_wrapper / 0.094135 / 0.324482 |
| 42 / 1e-5 / E2 | q9 / plastic_film / 0.158777 / 0.914682 | q4 / plastic_wrapper / 0.150021 / 0.688317 |
| 42 / 5e-6 / E1 | q4 / plastic_bottle / 0.226216 / 0.844769 | q635 / plastic_wrapper / 0.005899 / 0.499909 |
| 42 / 5e-6 / E2 | q16 / plastic_film / 0.044299 / 0.892766 | q7 / plastic_wrapper / 0.077773 / 0.432321 |
| 123 / 1e-5 / E1 | q10 / plastic_film / 0.036406 / 0.945715 | q5 / plastic_wrapper / 0.087775 / 0.402437 |
| 123 / 1e-5 / E2 | q13 / plastic_bottle / 0.199203 / 0.895646 | q12 / plastic_wrapper / 0.062064 / 0.272324 |
| 123 / 5e-6 / E1 | q9 / plastic_film / 0.020929 / 0.878749 | q6 / plastic_wrapper / 0.091142 / 0.452593 |
| 123 / 5e-6 / E2 | q11 / plastic_film / 0.014892 / 0.904635 | q7 / plastic_wrapper / 0.068792 / 0.419222 |


### GT278 — cigarette — bbox [1086.0, 984.0, 38.0, 68.0]

Best any / same columns: query ID / predicted class / model score / IoU. Primary scope is all900, score >=.05.


| State | Best any | Best same | Any ≥.50 | Same ≥.50 | Same visible in default five |
| --- | --- | --- | --- | --- | --- |
| Pretrained | q54 / plastic_wrapper / 0.066164 / 0.295527 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 1e-5 / E1 | q42 / styrofoam_piece / 0.052581 / 0.133090 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 1e-5 / E2 | q36 / plastic_film / 0.079824 / 0.002020 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 5e-6 / E1 | q279 / plastic_wrapper / 0.056411 / 0.118418 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 5e-6 / E2 | q19 / plastic_film / 0.110689 / 0.002020 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 1e-5 / E1 | q22 / plastic_film / 0.116221 / 0.002024 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 1e-5 / E2 | q45 / plastic_film / 0.073893 / 0.002016 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 5e-6 / E1 | q52 / plastic_wrapper / 0.051344 / 0.318407 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 5e-6 / E2 | q22 / plastic_film / 0.108030 / 0.002026 | none (no positive-IoU eligible overlap) | False | False | False |


Without a score floor (all900):


| State | Best any (score / IoU retained) | Best same (score / IoU retained) |
| --- | --- | --- |
| Pretrained | q476 / plastic_cup / 0.028392 / 0.407567 | none (no positive-IoU eligible overlap) |
| 42 / 1e-5 / E1 | q348 / plastic_cup / 0.006958 / 0.464773 | q317 / cigarette / 0.006057 / 0.251738 |
| 42 / 1e-5 / E2 | q182 / cigarette / 0.021708 / 0.461824 | q182 / cigarette / 0.021708 / 0.461824 |
| 42 / 5e-6 / E1 | q336 / plastic_cup / 0.017554 / 0.432138 | none (no positive-IoU eligible overlap) |
| 42 / 5e-6 / E2 | q227 / plastic_cup / 0.016745 / 0.592273 | q367 / cigarette / 0.021638 / 0.090859 |
| 123 / 1e-5 / E1 | q174 / plastic_cup / 0.008081 / 0.539639 | q451 / cigarette / 0.018785 / 0.129651 |
| 123 / 1e-5 / E2 | q220 / cigarette / 0.016279 / 0.559455 | q220 / cigarette / 0.016279 / 0.559455 |
| 123 / 5e-6 / E1 | q318 / plastic_cup / 0.021768 / 0.449553 | none (no positive-IoU eligible overlap) |
| 123 / 5e-6 / E2 | q237 / plastic_cup / 0.023035 / 0.510174 | q834 / cigarette / 0.010526 / 0.222127 |


### GT279 — cigarette — bbox [1032.0, 936.0, 40.0, 76.0]

Best any / same columns: query ID / predicted class / model score / IoU. Primary scope is all900, score >=.05.


| State | Best any | Best same | Any ≥.50 | Same ≥.50 | Same visible in default five |
| --- | --- | --- | --- | --- | --- |
| Pretrained | q408 / plastic_wrapper / 0.072368 / 0.141320 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 1e-5 / E1 | q42 / styrofoam_piece / 0.052581 / 0.150600 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 1e-5 / E2 | q36 / plastic_film / 0.079824 / 0.002377 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 5e-6 / E1 | q279 / plastic_wrapper / 0.056411 / 0.139315 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 5e-6 / E2 | q19 / plastic_film / 0.110689 / 0.002377 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 1e-5 / E1 | q22 / plastic_film / 0.116221 / 0.002382 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 1e-5 / E2 | q45 / plastic_film / 0.073893 / 0.002372 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 5e-6 / E1 | q447 / plastic_wrapper / 0.054883 / 0.132603 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 5e-6 / E2 | q22 / plastic_film / 0.108030 / 0.002384 | none (no positive-IoU eligible overlap) | False | False | False |


Without a score floor (all900):


| State | Best any (score / IoU retained) | Best same (score / IoU retained) |
| --- | --- | --- |
| Pretrained | q524 / plastic_cup / 0.006945 / 0.293177 | none (no positive-IoU eligible overlap) |
| 42 / 1e-5 / E1 | q354 / plastic_cup / 0.019032 / 0.305236 | q719 / cigarette / 0.003218 / 0.207051 |
| 42 / 1e-5 / E2 | q276 / cigarette / 0.026607 / 0.386358 | q276 / cigarette / 0.026607 / 0.386358 |
| 42 / 5e-6 / E1 | q315 / plastic_cup / 0.016271 / 0.367918 | none (no positive-IoU eligible overlap) |
| 42 / 5e-6 / E2 | q465 / cigarette / 0.008718 / 0.458887 | q465 / cigarette / 0.008718 / 0.458887 |
| 123 / 1e-5 / E1 | q414 / cigarette / 0.021854 / 0.399670 | q414 / cigarette / 0.021854 / 0.399670 |
| 123 / 1e-5 / E2 | q297 / cigarette / 0.007787 / 0.443507 | q297 / cigarette / 0.007787 / 0.443507 |
| 123 / 5e-6 / E1 | q271 / plastic_cup / 0.021157 / 0.394793 | none (no positive-IoU eligible overlap) |
| 123 / 5e-6 / E2 | q535 / plastic_cup / 0.004444 / 0.427111 | q834 / cigarette / 0.010526 / 0.209084 |


### GT280 — cigarette — bbox [1040.0, 772.0, 44.0, 52.0]

Best any / same columns: query ID / predicted class / model score / IoU. Primary scope is all900, score >=.05.


| State | Best any | Best same | Any ≥.50 | Same ≥.50 | Same visible in default five |
| --- | --- | --- | --- | --- | --- |
| Pretrained | q29 / plastic_film / 0.132262 / 0.001877 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 1e-5 / E1 | q12 / plastic_film / 0.105463 / 0.001801 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 1e-5 / E2 | q36 / plastic_film / 0.079824 / 0.001789 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 5e-6 / E1 | q11 / styrofoam_piece / 0.148935 / 0.001811 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 5e-6 / E2 | q19 / plastic_film / 0.110689 / 0.001789 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 1e-5 / E1 | q22 / plastic_film / 0.116221 / 0.001792 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 1e-5 / E2 | q45 / plastic_film / 0.073893 / 0.001785 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 5e-6 / E1 | q13 / plastic_film / 0.143909 / 0.001811 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 5e-6 / E2 | q22 / plastic_film / 0.108030 / 0.001794 | none (no positive-IoU eligible overlap) | False | False | False |


Without a score floor (all900):


| State | Best any (score / IoU retained) | Best same (score / IoU retained) |
| --- | --- | --- |
| Pretrained | q84 / plastic_cup / 0.030437 / 0.573036 | none (no positive-IoU eligible overlap) |
| 42 / 1e-5 / E1 | q105 / cigarette / 0.022276 / 0.509056 | q105 / cigarette / 0.022276 / 0.509056 |
| 42 / 1e-5 / E2 | q77 / cigarette / 0.042398 / 0.546447 | q77 / cigarette / 0.042398 / 0.546447 |
| 42 / 5e-6 / E1 | q77 / plastic_cup / 0.021041 / 0.526150 | q451 / cigarette / 0.007235 / 0.329056 |
| 42 / 5e-6 / E2 | q74 / cigarette / 0.029864 / 0.503689 | q74 / cigarette / 0.029864 / 0.503689 |
| 123 / 1e-5 / E1 | q71 / cigarette / 0.027535 / 0.522550 | q71 / cigarette / 0.027535 / 0.522550 |
| 123 / 1e-5 / E2 | q65 / cigarette / 0.045389 / 0.553866 | q65 / cigarette / 0.045389 / 0.553866 |
| 123 / 5e-6 / E1 | q67 / plastic_cup / 0.027199 / 0.558934 | q649 / cigarette / 0.011765 / 0.073261 |
| 123 / 5e-6 / E2 | q72 / cigarette / 0.028021 / 0.562805 | q72 / cigarette / 0.028021 / 0.562805 |


### GT281 — cigarette — bbox [1112.0, 954.0, 54.0, 74.0]

Best any / same columns: query ID / predicted class / model score / IoU. Primary scope is all900, score >=.05.


| State | Best any | Best same | Any ≥.50 | Same ≥.50 | Same visible in default five |
| --- | --- | --- | --- | --- | --- |
| Pretrained | q54 / plastic_wrapper / 0.066164 / 0.419855 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 1e-5 / E1 | q42 / styrofoam_piece / 0.052581 / 0.204579 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 1e-5 / E2 | q36 / plastic_film / 0.079824 / 0.003124 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 5e-6 / E1 | q279 / plastic_wrapper / 0.056411 / 0.183126 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 5e-6 / E2 | q19 / plastic_film / 0.110689 / 0.003124 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 1e-5 / E1 | q22 / plastic_film / 0.116221 / 0.003131 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 1e-5 / E2 | q45 / plastic_film / 0.073893 / 0.003118 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 5e-6 / E1 | q52 / plastic_wrapper / 0.051344 / 0.411616 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 5e-6 / E2 | q22 / plastic_film / 0.108030 / 0.003134 | none (no positive-IoU eligible overlap) | False | False | False |


Without a score floor (all900):


| State | Best any (score / IoU retained) | Best same (score / IoU retained) |
| --- | --- | --- |
| Pretrained | q284 / plastic_cup / 0.025651 / 0.473060 | none (no positive-IoU eligible overlap) |
| 42 / 1e-5 / E1 | q111 / plastic_film / 0.033184 / 0.454610 | q317 / cigarette / 0.006057 / 0.260288 |
| 42 / 1e-5 / E2 | q106 / cigarette / 0.019845 / 0.601338 | q106 / cigarette / 0.019845 / 0.601338 |
| 42 / 5e-6 / E1 | q172 / plastic_bag / 0.025055 / 0.413560 | none (no positive-IoU eligible overlap) |
| 42 / 5e-6 / E2 | q148 / plastic_cup / 0.018145 / 0.574308 | q457 / cigarette / 0.007654 / 0.324330 |
| 123 / 1e-5 / E1 | q123 / plastic_cup / 0.017688 / 0.604888 | q454 / cigarette / 0.005465 / 0.086467 |
| 123 / 1e-5 / E2 | q178 / cigarette / 0.009037 / 0.485980 | q178 / cigarette / 0.009037 / 0.485980 |
| 123 / 5e-6 / E1 | q165 / plastic_cup / 0.017194 / 0.481684 | none (no positive-IoU eligible overlap) |
| 123 / 5e-6 / E2 | q136 / plastic_cup / 0.012643 / 0.553114 | q539 / cigarette / 0.006421 / 0.066372 |


### GT282 — cigarette — bbox [1124.0, 786.0, 16.0, 36.0]

Best any / same columns: query ID / predicted class / model score / IoU. Primary scope is all900, score >=.05.


| State | Best any | Best same | Any ≥.50 | Same ≥.50 | Same visible in default five |
| --- | --- | --- | --- | --- | --- |
| Pretrained | q11 / styrofoam_piece / 0.104387 / 0.000612 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 1e-5 / E1 | q12 / plastic_film / 0.105463 / 0.000453 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 1e-5 / E2 | q74 / cigarette / 0.051683 / 0.285029 | q74 / cigarette / 0.051683 / 0.285029 | False | False | False |
| 42 / 5e-6 / E1 | q11 / styrofoam_piece / 0.148935 / 0.000456 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 5e-6 / E2 | q19 / plastic_film / 0.110689 / 0.000450 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 1e-5 / E1 | q22 / plastic_film / 0.116221 / 0.000451 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 1e-5 / E2 | q48 / cigarette / 0.061120 / 0.313471 | q48 / cigarette / 0.061120 / 0.313471 | False | False | False |
| 123 / 5e-6 / E1 | q13 / plastic_film / 0.143909 / 0.000456 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 5e-6 / E2 | q22 / plastic_film / 0.108030 / 0.000452 | none (no positive-IoU eligible overlap) | False | False | False |


Without a score floor (all900):


| State | Best any (score / IoU retained) | Best same (score / IoU retained) |
| --- | --- | --- |
| Pretrained | q149 / plastic_cup / 0.024442 / 0.130586 | none (no positive-IoU eligible overlap) |
| 42 / 1e-5 / E1 | q92 / cigarette / 0.023566 / 0.171872 | q92 / cigarette / 0.023566 / 0.171872 |
| 42 / 1e-5 / E2 | q74 / cigarette / 0.051683 / 0.285029 | q74 / cigarette / 0.051683 / 0.285029 |
| 42 / 5e-6 / E1 | q86 / plastic_cup / 0.020530 / 0.164145 | none (no positive-IoU eligible overlap) |
| 42 / 5e-6 / E2 | q72 / cigarette / 0.033634 / 0.221725 | q72 / cigarette / 0.033634 / 0.221725 |
| 123 / 1e-5 / E1 | q58 / cigarette / 0.032821 / 0.255178 | q58 / cigarette / 0.032821 / 0.255178 |
| 123 / 1e-5 / E2 | q48 / cigarette / 0.061120 / 0.313471 | q48 / cigarette / 0.061120 / 0.313471 |
| 123 / 5e-6 / E1 | q71 / plastic_cup / 0.027301 / 0.211536 | q649 / cigarette / 0.011765 / 0.024996 |
| 123 / 5e-6 / E2 | q67 / cigarette / 0.033304 / 0.240127 | q67 / cigarette / 0.033304 / 0.240127 |


### GT283 — cigarette — bbox [1016.0, 900.0, 14.0, 34.0]

Best any / same columns: query ID / predicted class / model score / IoU. Primary scope is all900, score >=.05.


| State | Best any | Best same | Any ≥.50 | Same ≥.50 | Same visible in default five |
| --- | --- | --- | --- | --- | --- |
| Pretrained | q408 / plastic_wrapper / 0.072368 / 0.001262 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 1e-5 / E1 | q68 / styrofoam_piece / 0.075229 / 0.000669 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 1e-5 / E2 | q36 / plastic_film / 0.079824 / 0.000372 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 5e-6 / E1 | q279 / plastic_wrapper / 0.056411 / 0.001377 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 5e-6 / E2 | q643 / styrofoam_piece / 0.071277 / 0.000643 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 1e-5 / E1 | q22 / plastic_film / 0.116221 / 0.000373 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 1e-5 / E2 | q45 / plastic_film / 0.073893 / 0.000371 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 5e-6 / E1 | q447 / plastic_wrapper / 0.054883 / 0.003500 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 5e-6 / E2 | q22 / plastic_film / 0.108030 / 0.000373 | none (no positive-IoU eligible overlap) | False | False | False |


Without a score floor (all900):


| State | Best any (score / IoU retained) | Best same (score / IoU retained) |
| --- | --- | --- |
| Pretrained | q739 / plastic_cup / 0.008016 / 0.086690 | none (no positive-IoU eligible overlap) |
| 42 / 1e-5 / E1 | q683 / plastic_film / 0.026276 / 0.014824 | q702 / cigarette / 0.008663 / 0.006609 |
| 42 / 1e-5 / E2 | q291 / cigarette / 0.031099 / 0.058207 | q291 / cigarette / 0.031099 / 0.058207 |
| 42 / 5e-6 / E1 | q740 / styrofoam_piece / 0.004299 / 0.024744 | none (no positive-IoU eligible overlap) |
| 42 / 5e-6 / E2 | q465 / cigarette / 0.008718 / 0.032598 | q465 / cigarette / 0.008718 / 0.032598 |
| 123 / 1e-5 / E1 | q380 / cigarette / 0.005107 / 0.046727 | q380 / cigarette / 0.005107 / 0.046727 |
| 123 / 1e-5 / E2 | q152 / cigarette / 0.049419 / 0.229137 | q152 / cigarette / 0.049419 / 0.229137 |
| 123 / 5e-6 / E1 | q806 / styrofoam_piece / 0.006070 / 0.016841 | none (no positive-IoU eligible overlap) |
| 123 / 5e-6 / E2 | q394 / plastic_film / 0.008965 / 0.014599 | q464 / cigarette / 0.003719 / 0.000697 |


### GT284 — cigarette — bbox [1622.0, 294.0, 74.0, 48.0]

Best any / same columns: query ID / predicted class / model score / IoU. Primary scope is all900, score >=.05.


| State | Best any | Best same | Any ≥.50 | Same ≥.50 | Same visible in default five |
| --- | --- | --- | --- | --- | --- |
| Pretrained | q111 / styrofoam_piece / 0.060000 / 0.006689 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 1e-5 / E1 | q12 / plastic_film / 0.105463 / 0.002796 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 1e-5 / E2 | q27 / cigarette / 0.057071 / 0.610385 | q27 / cigarette / 0.057071 / 0.610385 | True | True | False |
| 42 / 5e-6 / E1 | q11 / styrofoam_piece / 0.148935 / 0.002811 | none (no positive-IoU eligible overlap) | False | False | False |
| 42 / 5e-6 / E2 | q43 / cigarette / 0.051427 / 0.634536 | q43 / cigarette / 0.051427 / 0.634536 | True | True | False |
| 123 / 1e-5 / E1 | q22 / plastic_film / 0.116221 / 0.002783 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 1e-5 / E2 | q96 / cigarette / 0.053945 / 0.581194 | q96 / cigarette / 0.053945 / 0.581194 | True | True | False |
| 123 / 5e-6 / E1 | q13 / plastic_film / 0.143909 / 0.002811 | none (no positive-IoU eligible overlap) | False | False | False |
| 123 / 5e-6 / E2 | q22 / plastic_film / 0.108030 / 0.002786 | none (no positive-IoU eligible overlap) | False | False | False |


Without a score floor (all900):


| State | Best any (score / IoU retained) | Best same (score / IoU retained) |
| --- | --- | --- |
| Pretrained | q217 / plastic_cup / 0.017246 / 0.674472 | none (no positive-IoU eligible overlap) |
| 42 / 1e-5 / E1 | q71 / cigarette / 0.017263 / 0.649737 | q71 / cigarette / 0.017263 / 0.649737 |
| 42 / 1e-5 / E2 | q92 / cigarette / 0.025903 / 0.676684 | q92 / cigarette / 0.025903 / 0.676684 |
| 42 / 5e-6 / E1 | q57 / plastic_cup / 0.029252 / 0.618801 | q251 / cigarette / 0.010991 / 0.126278 |
| 42 / 5e-6 / E2 | q43 / cigarette / 0.051427 / 0.634536 | q43 / cigarette / 0.051427 / 0.634536 |
| 123 / 1e-5 / E1 | q98 / cigarette / 0.012633 / 0.842760 | q98 / cigarette / 0.012633 / 0.842760 |
| 123 / 1e-5 / E2 | q96 / cigarette / 0.053945 / 0.581194 | q96 / cigarette / 0.053945 / 0.581194 |
| 123 / 5e-6 / E1 | q115 / plastic_cup / 0.012504 / 0.752334 | q274 / cigarette / 0.013120 / 0.149907 |
| 123 / 5e-6 / E2 | q49 / cigarette / 0.048038 / 0.632850 | q49 / cigarette / 0.048038 / 0.632850 |


## Localization, class, threshold and display separation


| State | Same .05 /8 | Any .05 /8 | Same in five /8 | Same all900 floor0 | Localized without same .05 | Duplicate queries top100/.05 |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 1 | 1 | 0 | 1 | 0 | 1 |
| 42 / 1e-5 / E1 | 0 | 1 | 0 | 2 | 1 | 0 |
| 42 / 1e-5 / E2 | 2 | 2 | 1 | 4 | 0 | 0 |
| 42 / 5e-6 / E1 | 0 | 1 | 0 | 0 | 1 | 0 |
| 42 / 5e-6 / E2 | 1 | 2 | 0 | 2 | 1 | 0 |
| 123 / 1e-5 / E1 | 0 | 1 | 0 | 2 | 1 | 0 |
| 123 / 1e-5 / E2 | 1 | 2 | 0 | 3 | 1 | 1 |
| 123 / 5e-6 / E1 | 0 | 1 | 0 | 0 | 1 | 0 |
| 123 / 5e-6 / E2 | 0 | 1 | 0 | 2 | 1 | 0 |


A localized wrong-class-only outcome here means any coverage but no same-class coverage at .05. Use the no-floor result to separate score suppression from absence of an assigned-class candidate. A same-class candidate outside top100 is rank-excluded; one within top100 but outside the first five is display-hidden. JSON includes all six source/floor combinations (all900/top100 × 0/.05/.25), so these explanations can be checked independently. Multiple same-class qualifying candidates are overlap/duplication prompts; they are not extra independent detected objects.

## Existing 20-image batch: descriptive cross-run results


| State | GT objects | Same .05 | Any .05 | Same in five | Localized without same .05 | No same IoU≥.5 at floor0 | Duplicate top100 queries/.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 46 | 13 | 19 | 10 | 6 | 31 | 20 |
| 42 / 1e-5 / E1 | 46 | 15 | 20 | 12 | 5 | 27 | 5 |
| 42 / 1e-5 / E2 | 46 | 13 | 23 | 7 | 10 | 29 | 2 |
| 42 / 5e-6 / E1 | 46 | 11 | 19 | 9 | 8 | 32 | 9 |
| 42 / 5e-6 / E2 | 46 | 13 | 21 | 10 | 8 | 29 | 3 |
| 123 / 1e-5 / E1 | 46 | 10 | 21 | 7 | 11 | 32 | 1 |
| 123 / 1e-5 / E2 | 46 | 12 | 24 | 8 | 12 | 28 | 3 |
| 123 / 5e-6 / E1 | 46 | 10 | 19 | 9 | 9 | 32 | 5 |
| 123 / 5e-6 / E2 | 46 | 10 | 20 | 9 | 10 | 31 | 3 |


This purposive subset is not a prevalence or generalization sample. Counts measure coverage relative to annotations, independently per GT. It contains tiny boxes labelled cigarettes, but whether an object is a cigarette fragment or another material requires human observation; bbox size cannot establish material identity.


| Image | GT count | COCO-small cigarette GT IDs | Wrong-class-only GT/state cases | Missing same at floor0 cases | Same covered but hidden by five | Duplicate queries | Unmatched queries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 31 | 1 | [] | 8 | 8 | 0 | 0 | 85 |
| 42 | 1 | [] | 8 | 8 | 0 | 0 | 95 |
| 113 | 1 | [] | 1 | 1 | 0 | 0 | 69 |
| 144 | 7 | [55] | 4 | 41 | 6 | 13 | 132 |
| 155 | 2 | [65] | 7 | 15 | 1 | 0 | 264 |
| 157 | 1 | [] | 0 | 8 | 0 | 0 | 127 |
| 165 | 1 | [] | 0 | 0 | 0 | 5 | 192 |
| 200 | 1 | [] | 3 | 3 | 0 | 1 | 66 |
| 208 | 1 | [] | 6 | 6 | 0 | 0 | 37 |
| 236 | 1 | [] | 5 | 4 | 0 | 0 | 7 |
| 252 | 1 | [] | 0 | 0 | 2 | 2 | 31 |
| 419 | 1 | [] | 4 | 4 | 0 | 1 | 121 |
| 691 | 3 | [] | 5 | 20 | 1 | 0 | 101 |
| 776 | 1 | [] | 7 | 6 | 0 | 0 | 68 |
| 777 | 8 | [282, 283] | 7 | 49 | 3 | 1 | 99 |
| 833 | 1 | [] | 1 | 1 | 0 | 3 | 148 |
| 849 | 5 | [313, 314, 315, 316] | 3 | 16 | 9 | 0 | 231 |
| 1014 | 1 | [] | 0 | 2 | 0 | 0 | 220 |
| 1096 | 1 | [] | 1 | 0 | 0 | 2 | 75 |
| 1161 | 7 | [495, 496, 499] | 3 | 48 | 1 | 3 | 85 |


Batch case counts sum eight trained states, not distinct objects; query counts can include many near duplicates. Unmatched predictions flag possible omitted objects, background or box-extent errors. They do not establish incorrect/incomplete annotations. Full JSON preserves exact GT/query identities and raw phrases.

### Same-LR/epoch seed-paired coverage in the 20-image subset

All900, score >=.05, IoU >=.50; set membership is per GT, non-exclusive.
This compares actual annotation identities, not just equal aggregate counts.

| LR | Epoch | Same coverage both seeds | Only42 | Only123 | Wrong-class-only both seeds |
| --- | --- | --- | --- | --- | --- |
| 1e-5 | 1 | 8 | 7 | 2 | 4 |
| 1e-5 | 2 | 9 | 4 | 3 | 8 |
| 5e-6 | 1 | 9 | 2 | 1 | 7 |
| 5e-6 | 2 | 10 | 3 | 0 | 8 |


Counts differ by seed while some missing/class-assignment cases repeat.
The two seeds and selected examples cannot establish a causal training failure.

## Using the enhanced four-panel viewer

Explicit object mode starts with all900 at .05, while normal viewer defaults remain official top100 / five boxes / .05. Choose GT ID, Previous GT or Next GT; all eight image777 annotations are available. The scrollable Per-annotation results tab contains all GT × three model rows and supports Ctrl+C. Selecting a row chooses its GT. Candidate tabs show every positive-IoU overlap with raw phrase, assigned class, score, rank and IoU to the selected GT; agreement/difference is relative to its annotation.

The yellow dashed GT box appears in all panels. Green candidate borders agree with its annotated class; magenta differ. To reduce clutter no text labels are drawn in object mode; exact text is in copyable tables. The box cap draws up to max(2, Box top K), always prioritizing a same-class and differing-class candidate if present; it is explicitly disclosed and never limits the tables/analysis. Uncheck Zoom to selected GT for the full image, or use shared toolbar pan/zoom. The view source and threshold can be changed diagnostically without altering official predictions.

Run in PowerShell from the repository root:

```powershell
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --gui --saved-review .\runs\training\cross-seed-analysis-fd10923e40d24c40be5bab3ae0fa2c63\review-manifest.json --run-choice "123 / 1e-5" --image-id 777 --per-object
```

Other run keys: `42 / 1e-5`, `42 / 5e-6`, `123 / 5e-6`. Relaunch with the desired key; no mixing of epochs between runs. Remove `--per-object` for the unchanged standard viewer. Terminal-only default routing is unchanged; the new flag requires explicit saved-review GUI mode. No annotated image export is offered. Display lifecycle stays separate from successful saved verification.

## Pending human questions and stop boundary

For each cigarette-labelled GT, is there a visible separate object, what material/fragment is it, and is its extent correct? Do adjacent annotations represent separate items or ambiguous fragments? Does a selected wrong-class query describe that object or nearby texture? Are multiple boxes duplicates of one object? Are unmatched predictions extra real objects or background? Preserve uncertainty; do not edit annotations based only on this diagnostic. No new human verdict is claimed.

No training recommendation is made from image777 or this selected subset. Stop after diagnostic preparation and native verification for human review. No further experiment is authorized.

## Software, native verification and preservation

29 focused software tests passed, zero skips/errors/failures: the existing18
diagnostic/GUI/lifecycle fixtures and four saved-run fixtures plus seven new
object tests. New tests independently cover IoU-versus-score ranking, candidates
outside the first five/top100, inclusive .05/.50 boundaries, zero-floor behavior,
wrong classes, empty overlaps, duplicated/shared candidates, tie ordering,
nonmutation, zoom limits and explicit CLI guards. No optimizer tests were run.

The actual Windows Tk/TkAgg CLI was exercised with --per-object for all four
saved-run choices. Each displayed all eight image777 GT IDs, 24 GT/model summary
rows and every eligible overlap, with best queries checked against the saved
machine analysis. E1/E2 matches were checked per selected run. The 20-image
subset also rendered under seed123/1e-5. Threshold/source controls, full view,
zoom, table copying and normal closure passed. A fifth native invocation without
--per-object confirmed official top100 /five-box defaults and absence of the
new controls. Separate display JSON records closed normally. Image/figure save
calls and network connections were blocked during these native checks; zero
annotated exports occurred. Native runtime success is not human quality approval.

The original manifest, prior cross-seed evidence and protected data/source/config
hashes were checked before and after. All40 checkpoint bundles of the four runs
were verified against their original contracts without restoring tensors.
The base comparison GUI, existing owner trainer changes, evaluation protocol,
training pipeline, original weights and datasets remain unchanged. Only an
explicit flag routes to the derived object-inspection window. No checkpoints,
predictions, original images, metrics or prior review notes were overwritten.
Python syntax, relative documentation links and Git whitespace checks passed.
No dependencies, model forwards, optimizer updates, test split, promotion,
publication, commits, pushes or merges. Stop for human review.
