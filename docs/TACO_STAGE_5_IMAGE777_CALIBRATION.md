# D-077: Stage 5 Image 777 display calibration audit

## D-078: Nine-class score-vector and material-pooling audit

### Bounded extraction

D078 verified the D076 `soup_tier2_d074_champion_5050` inference bundle at
parameter SHA-256
`cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422`, then completed
the score-vector extraction with **65 CUDA forwards** over `combined_65` validation images
at 384/576, FP32, TF32 disabled. No train photos/split, test metadata/photos,
training updates, backward passes, or optimizer steps were used. The extracted
float32 array has shape **65 x 900 x 9**, containing each query's mean ordinary-
subword sigmoid probability for each of the nine prompted classes. All 58,500
queries reproduced the saved winning class exactly and each winner score had
maximum absolute error **0.0**. One earlier failed extraction attempt had already
completed a model forward before aborting in the loss matcher, so cumulative D078
detector forward invocations were **66**, exceeding the owner-authorized 65 cap by
one. The failed attempt saved no vectors; no more detector forwards were run after
the overrun was identified. The 65 saved rows below are from the completed pass.
Existing checkpoint, saved predictions, rights, export, taxonomy, and Stage4
core-source hashes remained unchanged.

Private vector artifact:
`runs/training/stage5-score-vectors-5619e219fa264287aa0564c4565a2472/`
(`query-scores.npz`, SHA-256
`d597b6072852538c7c09997664c635f7c88850f3bc66cda9ce00b56dc35f4060`). CPU-only
pooling results are in
`runs/training/stage5-score-vector-audit-6ba2b48e835847fab31a380b54db4133/`.

### Image 777 GT 259

The frozen Stage4 Image777 GT259 resolves by TACO source annotation 2390 to
Stage5 validation image 777 / annotation 408 (`plastic_wrapper`). Query q0 has
IoU **0.9253985905** and the complete score vector is:

| Class | q0 score |
| --- | ---: |
| cigarette | 0.1130877063 |
| plastic_bottle | 0.2749033570 |
| plastic_bag | 0.1952677816 |
| plastic_wrapper | 0.2554721832 |
| plastic_film | 0.2927520871 |
| drink_can | 0.2038989961 |
| plastic_cup | 0.1313864738 |
| styrofoam_piece | 0.1276722550 |
| paper | 0.3078179955 |

The complete vectors for q0 and every other query with IoU >=0.50 are below.
IoUs are listed above the columns; values are mean subword sigmoid probabilities.

| Query | q0 | q2 | q1 | q4 | q35 | q3 | q56 | q113 | q882 | q15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IoU | 0.9253985905 | 0.7271879079 | 0.9324330862 | 0.6814951569 | 0.8984831068 | 0.8733524111 | 0.6900122785 | 0.7163020939 | 0.5694414920 | 0.8884305966 |
| cigarette | 0.1130877063 | 0.0609950423 | 0.0496455729 | 0.0150869489 | 0.0074456516 | 0.0064447885 | 0.0062645925 | 0.0049912701 | 0.0034914250 | 0.0032494087 |
| plastic_bottle | 0.2749033570 | 0.0551909618 | 0.0201320294 | 0.0055255350 | 0.0041919281 | 0.0034638003 | 0.0017410831 | 0.0024074635 | 0.0019969523 | 0.0019053705 |
| plastic_bag | 0.1952677816 | 0.0512920320 | 0.0133888833 | 0.0045412495 | 0.0027617509 | 0.0022771116 | 0.0017155581 | 0.0016004387 | 0.0017266956 | 0.0013052432 |
| plastic_wrapper | 0.2554721832 | 0.0721060187 | 0.0182857253 | 0.0063237045 | 0.0043193996 | 0.0036056759 | 0.0019303709 | 0.0021153903 | 0.0021512136 | 0.0017453864 |
| plastic_film | 0.2927520871 | 0.0822635293 | 0.0383198708 | 0.0113229332 | 0.0068620648 | 0.0055805398 | 0.0057383077 | 0.0033354631 | 0.0047036419 | 0.0023957565 |
| drink_can | 0.2038989961 | 0.0495207310 | 0.0229158215 | 0.0056691044 | 0.0033120294 | 0.0027149660 | 0.0016336272 | 0.0019829553 | 0.0012520944 | 0.0013553286 |
| plastic_cup | 0.1313864738 | 0.0445739776 | 0.0205396060 | 0.0057653752 | 0.0031968823 | 0.0027272208 | 0.0016227255 | 0.0021438107 | 0.0013158787 | 0.0014736746 |
| styrofoam_piece | 0.1276722550 | 0.0515668243 | 0.0106359143 | 0.0039452706 | 0.0031206745 | 0.0026776558 | 0.0027502892 | 0.0014218170 | 0.0018137400 | 0.0013892625 |
| paper | 0.3078179955 | 0.0939621106 | 0.0253064018 | 0.0116103226 | 0.0066985404 | 0.0055609308 | 0.0029711935 | 0.0032060244 | 0.0028281719 | 0.0024250341 |

Ten queries have q0 or IoU >=0.50 with this GT: q0, q2, q1, q4, q35, q3, q56,
q113, q882, q15. q0's paper score is only 0.015066 below its highest plastic
subtype score (`plastic_film`), and 0.015065 below the maximum across plastic
subtypes; paper remains the raw five-/four-material winner. Noisy-OR pooling
raises q0's 5-material plastic score to **0.733114** and its four-material polymer
score to **0.767188**, flipping this single query to plastic. Capped top-two sum
also flips the winner (0.567655 in both views). Max-subtype is identical to raw
top-1 material mapping and leaves q0 as paper.

### Validation-wide result: pooling is not beneficial

All scores below are diagnostic re-evaluations from the one extracted vector set;
they do not alter the official nine-specific COCO dictionary. AP uses top 100 valid
queries/image and no NMS. Strict TP/FP/FN uses all valid queries, one-to-one correct-
material matching, IoU >=0.50, no NMS.

| Taxonomy / pooling | AP | AP50 | AR100 | TP/FP/FN @ .05 | @ .10 | @ .15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5-material raw top-1 / max subtype | 0.342725 | 0.419523 | 0.505848 | 80/1426/29 | 68/283/41 | 60/108/49 |
| 5-material noisy-OR | 0.105068 | 0.168661 | 0.197504 | 59/3654/50 | 56/1060/53 | 52/549/57 |
| 5-material capped top-two | 0.107140 | 0.167242 | 0.230587 | 62/1868/47 | 56/494/53 | 50/222/59 |
| 4-material raw top-1 / max subtype | 0.400957 | 0.494067 | 0.600996 | 85/1421/24 | 72/279/37 | 63/105/46 |
| 4-material noisy-OR | 0.138704 | 0.216085 | 0.249863 | 66/4527/43 | 63/1291/46 | 59/652/50 |
| 4-material capped top-two | 0.143208 | 0.216897 | 0.293846 | 68/1899/41 | 63/499/46 | 57/221/52 |

The image-specific plastic flip is outweighed by a large validation-wide loss in
AP and precision. D078 therefore adds **no score pooling/calibration GUI toggle**;
raw argmax remains the default and only material-view score behavior. Noisy-OR and
top-two are useful audit hypotheses, not adopted mappings. Official nine-specific
metrics and previously saved JSON remain byte-for-byte separate and unchanged.

### Tcl/Tk and native GUI verification

Native smoke previously failed before widget construction because Tcl could not
traverse the installed Python's `AppData\\Local` junction using ordinary paths.
The non-destructive helper `scripts/tk_runtime.py` derives the standard Tcl/Tk
directories from `sys.base_prefix`, repairs unset/stale values or Tcl-unusable ordinary base paths,
and uses an extended-length Windows path where required. Valid existing overrides
are preserved; no interpreter files are changed. The Image777 smoke passed all
**40** combinations (four taxonomy views x five NMS modes x two threshold profiles),
with zero TclError and the source validation-photo SHA-256 unchanged. Command:

```powershell
& .\.venv-training\Scripts\python.exe -B .\tests\windows_stage5_image777_check.py
```

Full D078 evidence is recorded in the decision register and private manifests;
the vector file and reports are local diagnostic artifacts and are not committed.

## Scope and saved-query limits

This is a CPU-only diagnostic of saved validation predictions. It reads the 65-image
Stage 5 validation annotations, the frozen 47-image validation annotations to resolve
historical review IDs, and eleven saved prediction files (65 x 900 winning-class queries
each). It does not open test metadata or photos, load weights, execute a detector, alter
official no-NMS COCO metrics, or write into existing run/checkpoint/export/rights
directories. All per-image controls are display operating points, not calibrated
correctness probabilities or official metrics.

The complete 660-case strict one-to-one audit, Image 777 query ledger, and input
SHA-256s are in the private local artifact
`runs/training/stage5-image777-calibration-306d313f59bd4634bd171484cc3270fd/summary.json`;
the same 660 rows are in `strict-matching.csv`. The audit spans eleven states, three
score-slider values (0.05/0.10/0.15), uniform and calibrated modes, five NMS modes,
and nine-specific/five-material matching. The JSON records each source digest and
confirms all inputs remained unchanged.

Important limitation: the saved files contain only query ID, winning category, one
scalar score and box. They do **not** contain the 9-class logit/probability vector.
So this audit can compare winning-class labels, thresholds and NMS, but cannot explain
the model's hidden paper-versus-plastic logits, pool plastic subtype probabilities,
or apply a defensible class-prior/logit calibration. Two CPU-averaged balanced soups
were separately evaluated using 65 combined-view CUDA forwards each, within D-077's
224-forward maximum. Their complete prediction files and proof are in
`runs/training/stage5-balanced-soup-eval-1453160757f14d6c89104989c9593a68/`.
The high-resolution category flip is observable; its latent score cause is not
identifiable from these exports.

The balanced states were `0.40 * Run1 blend2575 + 0.60 * Tier1 E1` and
`0.40 * Run1 blend2575 + 0.60 * Tier2 E1`, averaged in CPU memory and evaluated
on combined_65 at 384/576 FP32 with TF32 off. Exactly 65 forwards completed per
state. The first setup attempt failed at CuBLAS before a forward completed; that
attempt is counted, making 131 cumulative attempts against the 224 cap. There were
zero training updates, optimizer steps, or backward passes. Source checkpoint
bundles, exports, rights, taxonomies, and six Stage 4 core sources passed the
before/after checks recorded in the private run summary.

## GT 259 on Image 777

Historical review GT IDs resolve to the Stage 5 export annotations by shared source
TACO annotation provenance (GT 259 -> export annotation 408; GT 278..284 -> 427..433).
The leading IoU>=0.50 candidates by score are shown below as
`query:class / score / IoU / rank` in the raw 900-query order. The complete ledger
contains all 116 candidates at IoU>=0.50 across all eleven states.
For every listed row, `plastic_bottle`/`plastic_film` map to
`plastic_waste` / `plastic_polymer_waste` (5-material / 4-material), `paper` maps
to `paper_waste` / `paper_waste`, and `cigarette` maps to
`hazardous_cigarette` / `hazardous_cigarette`.

| Saved state | Leading matching queries |
|---|---|
| Run 1 baseline, 256/384 | `q3:plastic_film / .230102 / .947 / 1`; `q0:plastic_bottle / .167493 / .687 / 2`; `q4:plastic_film / .056552 / .586 / 30` |
| Run 1 Epoch 2, 256/384 | `q0:plastic_bottle / .296058 / .920 / 1`; `q3:paper / .125049 / .687 / 3`; `q2:plastic_film / .050989 / .608 / 46` |
| Run 1 blend2575, 384/576 | `q0:plastic_bottle / .295378 / .930 / 1`; `q3:paper / .094547 / .727 / 7`; `q1:cigarette / .054681 / .934 / 36` |
| High-Res E1, 384/576 | `q0:paper / .302496 / .929 / 1`; `q2:paper / .099316 / .730 / 6`; `q1:cigarette / .052791 / .934 / 30` |
| Tier 1 E1 | `q0:paper / .312148 / .926 / 1`; `q2:paper / .096062 / .729 / 6`; `q1:cigarette / .046941 / .933 / 39` |
| Tier 1 20/80 soup | `q0:paper / .303204 / .930 / 1`; `q2:paper / .093517 / .728 / 7`; `q1:cigarette / .048335 / .934 / 40` |
| Tier 2 E1 | `q0:paper / .313045 / .922 / 1`; `q2:paper / .094716 / .727 / 7`; `q1:cigarette / .050049 / .930 / 35` |
| Tier 2 20/80 soup | `q0:paper / .304582 / .926 / 1`; `q2:paper / .093118 / .728 / 7`; `q1:cigarette / .051723 / .932 / 34` |
| Tier 2 + D074 50/50 soup | `q0:paper / .307818 / .925 / 1`; `q2:paper / .093962 / .727 / 7`; `q1:cigarette / .049646 / .932 / 36` |
| Balanced 40/60 + Tier 1 E1 | `q0:paper / .293880 / .932 / 1`; `q2:paper / .092843 / .728 / 7`; `q1:cigarette / .051487 / .935 / 34` |
| Balanced 40/60 + Tier 2 E1 | `q0:paper / .295375 / .930 / 1`; `q2:paper / .092248 / .727 / 7`; `q1:cigarette / .052988 / .933 / 31` |

In five-material mapping, Run 1's leading `plastic_bottle`/`plastic_film` candidates
are `plastic_waste`; in four-material mapping they are `plastic_polymer_waste`.
The high-resolution leading `paper` query maps to `paper_waste` in both views. Its
box remains almost unchanged and well localized (IoU .922-.930); query q0's winning
class flips while the score rises from about .295 to .302-.313. This is a class-score
decision change, not evidence that the box moved. The saved scalar-only output cannot
show whether paper's logit rose or plastic's fell. Run1 blend2575 and High-Res E1 are
both evaluated at 384/576, so resolution is held fixed in that comparison; the
checkpoint state differs. That associates the winner flip with the model state, but
the saved winning-class score cannot isolate the underlying class-logit changes.

At a permissive 0.05 floor, Tier 2 E1 also has `q1:cigarette` at score .050049 and
IoU .930 on the same GT259 region. Class-aware NMS permits this different-class
overlap; cross-class IoU NMS lets the higher-scoring q0 survive. This demonstrates
why class-aware NMS can leave a cross-label duplicate. It does not correct q0's paper
classification, and the plastic-versus-paper flip itself persists under NMS.

## Cigarette query ledger on Image 777

For each cigarette GT, selection is the highest-scoring predicted cigarette with
IoU>=.50; if no such query exists, the best-IoU cigarette query is shown. Entries are
`query / score / raw rank / IoU`. This lets the score band be read without conflating
box overlap and confidence. An entry below .05 is not a passing detection at the
audit's conventional .05 coverage floor.

| GT | Baseline | Run 1 E2 | Run 1 blend | High-Res E1 |
|---|---|---|---|---|
| 278 | q273 / .0198 / 191 / .522 | q165 / .0209 / 245 / .508 | q433 / .0199 / 263 / .902 | q35 / .0715 / 12 / .892 |
| 279 | q847 / .0119 / 338 / .379 | q450 / .0253 / 182 / .536 | q211 / .0667 / 20 / .797 | q61 / .0699 / 16 / .831 |
| 280 | q71 / .0380 / 63 / .486 | q104 / .0452 / 64 / .532 | q10 / .1138 / 2 / .826 | q8 / .1270 / 2 / .849 |
| 281 | q183 / .0232 / 162 / .690 | q139 / .0225 / 214 / .696 | q581 / .0193 / 273 / .669 | q103 / .0178 / 276 / .811 |
| 282 | q39 / .0568 / 29 / .340 | q32 / .0774 / 15 / .425 | q46 / .0325 / 116 / .521 | q6 / .1201 / 3 / .579 |
| 283 | q269 / .0387 / 60 / .122 | q52 / .0544 / 39 / .523 | q73 / .0815 / 9 / .541 | q52 / .0861 / 8 / .675 |
| 284 | q59 / .0590 / 25 / .536 | q18 / .0872 / 9 / .517 | q22 / .0950 / 6 / .795 | q21 / .1042 / 5 / .811 |

| GT | Tier 1 E1 | Tier 1 20/80 soup | Tier 2 E1 | Tier 2 20/80 soup | Tier 2 + D074 50/50 |
|---|---|---|---|---|---|
| 278 | q32 / .0721 / 13 / .903 | q430 / .0157 / 314 / .844 | q31 / .0741 / 12 / .893 | q435 / .0157 / 316 / .840 | q32 / .0716 / 12 / .887 |
| 279 | q66 / .0676 / 17 / .828 | q68 / .0682 / 17 / .828 | q58 / .0703 / 15 / .833 | q61 / .0697 / 17 / .830 | q77 / .0688 / 17 / .842 |
| 280 | q7 / .1279 / 2 / .859 | q8 / .1252 / 2 / .852 | q7 / .1296 / 2 / .865 | q8 / .1264 / 2 / .857 | q8 / .1271 / 2 / .859 |
| 281 | q607 / .0142 / 331 / .629 | q611 / .0170 / 280 / .660 | q537 / .0152 / 307 / .860 | q519 / .0180 / 261 / .853 | q614 / .0157 / 305 / .690 |
| 282 | q6 / .1206 / 3 / .596 | q6 / .1191 / 3 / .571 | q5 / .1246 / 3 / .586 | q6 / .1223 / 3 / .563 | q6 / .1215 / 3 / .577 |
| 283 | q48 / .0860 / 8 / .707 | q52 / .0858 / 8 / .664 | q49 / .0874 / 8 / .689 | q85 / .0856 / 8 / .682 | q51 / .0863 / 8 / .684 |
| 284 | q19 / .1064 / 5 / .850 | q20 / .1045 / 5 / .824 | q19 / .1089 / 5 / .833 | q20 / .1067 / 5 / .818 | q19 / .1068 / 5 / .824 |

| GT | Balanced 40/60 + Tier 1 E1 | Balanced 40/60 + Tier 2 E1 |
|---|---|---|
| 278 | q432 / .0156 / 332 / .862 | q435 / .0180 / 268 / .879 |
| 279 | q67 / .0680 / 18 / .821 | q64 / .0686 / 17 / .818 |
| 280 | q9 / .1214 / 2 / .844 | q9 / .1226 / 2 / .847 |
| 281 | q605 / .0167 / 300 / .626 | q610 / .0169 / 290 / .693 |
| 282 | q8 / .1156 / 3 / .551 | q7 / .1178 / 3 / .543 |
| 283 | q57 / .0839 / 8 / .636 | q56 / .0853 / 8 / .639 |
| 284 | q20 / .1017 / 5 / .825 | q20 / .1038 / 5 / .832 |

The score-band counts across all seven GT cigarettes in each state are:

| State | <.05 | [.05,.10) | >=.10 |
|---|---:|---:|---:|
| Run 1 baseline | 5 | 2 | 0 |
| Run 1 Epoch 2 | 4 | 3 | 0 |
| Run 1 blend2575 | 3 | 3 | 1 |
| High-Res E1 | 1 | 3 | 3 |
| Tier 1 E1 | 1 | 3 | 3 |
| Tier 1 20/80 soup | 2 | 2 | 3 |
| Tier 2 E1 | 1 | 3 | 3 |
| Tier 2 20/80 soup | 2 | 2 | 3 |
| Tier 2 + D074 50/50 soup | 1 | 3 | 3 |
| Balanced 40/60 + Tier 1 | 2 | 2 | 3 |
| Balanced 40/60 + Tier 2 | 2 | 2 | 3 |

Across the original 384/576 reference/trained states, selected localized cigarette scores by band are:
High-Res E1 **1 below .05, 3 in [.05,.10), 3 >=.10**; Tier 1 E1 **1/3/3**;
Tier 1 soup **2/2/3**; Tier 2 E1 **1/3/3**; Tier 2 20/80 soup **2/2/3**;
Tier 2 + D074 50/50 **1/3/3**. Both balanced 40/60 states are **2/2/3**. Thus uniform .10 hides three localized cigarettes in the E1 checkpoints;
the calibrated .06 cigarette floor displays those in-band candidates while retaining the
three >=.10 candidates. Cigarettes scoring below .06 remain hidden.

## 65-image strict operating-point comparison

Rows below use the Tier 2 + D074 50/50 state and aggregate per-image, one-to-one,
correct-class matching at IoU>=.50. `Boxes` counts all post-threshold/post-NMS
queries across the 65 images. This is a thresholded diagnostic, not COCO AP/AR.
The full artifact contains the same measures for every saved state and setting.

| Taxonomy | Display threshold | NMS | TP | FP | FN | Boxes |
|---|---|---|---:|---:|---:|---:|
| 9-specific | Uniform .05 | raw | 61 | 1445 | 48 | 1506 |
| 9-specific | Uniform .05 | class-aware IoU .50 | 61 | 1302 | 48 | 1363 |
| 9-specific | Uniform .05 | class-aware IoU .50 + containment .80 | 58 | 1056 | 51 | 1114 |
| 9-specific | Uniform .05 | agnostic IoU .50 | 55 | 1217 | 54 | 1272 |
| 9-specific | Uniform .05 | agnostic IoU .50 + containment .80 | 52 | 791 | 57 | 843 |
| 9-specific | Uniform .10 | raw | 50 | 301 | 59 | 351 |
| 9-specific | Uniform .10 | class-aware IoU .50 | 50 | 281 | 59 | 331 |
| 9-specific | Uniform .10 | class-aware IoU .50 + containment .80 | 50 | 257 | 59 | 307 |
| 9-specific | Uniform .10 | agnostic IoU .50 | 48 | 268 | 61 | 316 |
| 9-specific | Uniform .10 | agnostic IoU .50 + containment .80 | 47 | 219 | 62 | 266 |
| 9-specific | Calibrated .06 cigarette / .15 macro (slider .10) | raw | 53 | 684 | 56 | 737 |
| 9-specific | Calibrated .06 cigarette / .15 macro (slider .10) | class-aware IoU .50 | 53 | 653 | 56 | 706 |
| 9-specific | Calibrated .06 cigarette / .15 macro (slider .10) | class-aware IoU .50 + containment .80 | 52 | 563 | 57 | 615 |
| 9-specific | Calibrated .06 cigarette / .15 macro (slider .10) | agnostic IoU .50 | 51 | 633 | 58 | 684 |
| 9-specific | Calibrated .06 cigarette / .15 macro (slider .10) | agnostic IoU .50 + containment .80 | 49 | 461 | 60 | 510 |
| 9-specific | Uniform .15 | raw | 43 | 125 | 66 | 168 |
| 9-specific | Uniform .15 | class-aware IoU .50 | 43 | 119 | 66 | 162 |
| 9-specific | Uniform .15 | class-aware IoU .50 + containment .80 | 42 | 113 | 67 | 155 |
| 9-specific | Uniform .15 | agnostic IoU .50 | 41 | 117 | 68 | 158 |
| 9-specific | Uniform .15 | agnostic IoU .50 + containment .80 | 39 | 103 | 70 | 142 |
| 5-material | Uniform .05 | raw | 80 | 1426 | 29 | 1506 |
| 5-material | Uniform .05 | class-aware IoU .50 | 80 | 1257 | 29 | 1337 |
| 5-material | Uniform .05 | class-aware IoU .50 + containment .80 | 75 | 1000 | 34 | 1075 |
| 5-material | Uniform .05 | agnostic IoU .50 | 74 | 1198 | 35 | 1272 |
| 5-material | Uniform .05 | agnostic IoU .50 + containment .80 | 70 | 773 | 39 | 843 |
| 5-material | Uniform .10 | raw | 68 | 283 | 41 | 351 |
| 5-material | Uniform .10 | class-aware IoU .50 | 68 | 258 | 41 | 326 |
| 5-material | Uniform .10 | class-aware IoU .50 + containment .80 | 67 | 228 | 42 | 295 |
| 5-material | Uniform .10 | agnostic IoU .50 | 67 | 249 | 42 | 316 |
| 5-material | Uniform .10 | agnostic IoU .50 + containment .80 | 65 | 201 | 44 | 266 |
| 5-material | Calibrated .06 cigarette / .15 macro (slider .10) | raw | 70 | 667 | 39 | 737 |
| 5-material | Calibrated .06 cigarette / .15 macro (slider .10) | class-aware IoU .50 | 70 | 634 | 39 | 704 |
| 5-material | Calibrated .06 cigarette / .15 macro (slider .10) | class-aware IoU .50 + containment .80 | 69 | 541 | 40 | 610 |
| 5-material | Calibrated .06 cigarette / .15 macro (slider .10) | agnostic IoU .50 | 69 | 615 | 40 | 684 |
| 5-material | Calibrated .06 cigarette / .15 macro (slider .10) | agnostic IoU .50 + containment .80 | 67 | 443 | 42 | 510 |
| 5-material | Uniform .15 | raw | 60 | 108 | 49 | 168 |
| 5-material | Uniform .15 | class-aware IoU .50 | 60 | 100 | 49 | 160 |
| 5-material | Uniform .15 | class-aware IoU .50 + containment .80 | 59 | 91 | 50 | 150 |
| 5-material | Uniform .15 | agnostic IoU .50 | 59 | 99 | 50 | 158 |
| 5-material | Uniform .15 | agnostic IoU .50 + containment .80 | 57 | 85 | 52 | 142 |

As expected, cross-class IoU NMS suppresses more boxes and often also suppresses
correct matches. Containment suppression lowers FP further but has additional TP
losses. At .10 uniform, moving from raw to cross-class IoU NMS changes 9-specific
TP/FP/FN from 50/301/59 to 48/268/61. The calibrated profile at slider .10 surfaces
more cigarette candidates, but increases aggregate FPs (raw 301 ->684; cross-class
IoU 268 ->633) and is not a net precision improvement. At .15 uniform, clutter falls
but recall falls further. None of these operating points is a universally best
threshold.

The new balanced soups do not restore GT259 to plastic: both retain `q0:paper`.
Their official, no-NMS combined nine-specific AP is **0.186977** for 40/60 + Tier1
and **0.215196** for 40/60 + Tier2, below Tier1 E1 / its 20/80 soup
(0.218215 / 0.231226) and Tier2 E1 / its D074 50/50 soup (0.232968 / 0.235994).
Parameter SHA-256s are `1f6ce5b88d8ea67f43c049ba0dced42b9937528343cf57912b18fe60b17cce07`
and `d6456851d3890ef6391d65872d09e3622457c2765171307c8c125148a9cebfe4`.
Both balanced states localize five of the seven Image 777 cigarettes with scores
>=.05; GT278 and GT281 remain below .05. Each has three >=.10 and two [.05,.10)
cigarettes. This does not improve official AP or retain all seven cigarettes at the
.05 score floor.

At uniform .10 on the balanced states, strict 9-specific TP/FP/FN are:

| State | raw | class-aware IoU .50 | agnostic IoU .50 | agnostic IoU .50 + containment |
|---|---:|---:|---:|---:|
| 40/60 + Tier1 | 46/326/63 | 46/309/63 | 43/289/66 | 42/233/67 |
| 40/60 + Tier2 | 48/328/61 | 48/311/61 | 45/293/64 | 44/235/65 |

## Display changes and recommendation

The viewer now offers five modes: `raw`, `class_aware_iou_0.50`,
`class_aware_iou_0.50_containment_0.80`, `agnostic_iou_0.50`, and
`agnostic_iou_0.50_containment_0.80`. Mapping still occurs before class-aware NMS;
agnostic NMS explicitly compares across predicted classes. A second selector switches
between the existing uniform score control and `Calibrated Profile (Cigarette 0.06 /
Macro 0.15)`. For calibrated display, cigarette uses floor .06 and every other
specific class uses `max(slider, .15)`. These are reversible display filters only.
The viewer starts at uniform threshold mode with agnostic IoU .50 selected.

For initial review, use **uniform .10**, **agnostic IoU .50**, and Top-K 15 when the
goal is one label per overlapping region. For inspecting faint cigarettes, switch to
the explicit calibrated profile and raw or agnostic IoU mode; inspect suppressed boxes
with raw mode if a true object seems missing. Do not treat the calibrated profile as
validated: it exposed more cigarettes but added substantial false positives across the
65-image set. Containment .80 is the most aggressive option and reduces true matches.
No pooling/prior view was added because the saved files lack score vectors. The two
balanced soups were tested as described above; neither fixed the GT259 class flip.

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_stage5_visual_review.py --gui --image-id 777 --score-threshold 0.10 --box-top-k 15
```

The viewer auto-discovers complete Tier1/Tier2 saved states. Select the desired
fourth-panel state and use `Threshold Mode` / `NMS` controls in the toolbar.

Unit tests cover score floors, cross-class/containment suppression, and GUI selector
integration. The requested native Tk check was attempted but **could not initialize**:
Python raises `TclError: Can't find a usable init.tcl` even though the Tcl/Tk files are
present at the standard Python 3.11 path. The smoke therefore did not reach GUI
assertions; it must be rerun after repairing the local Tcl/Tk runtime. This is not
reported as a native GUI pass.
