# D-076: Tier 2 training-only sibling expansion at 384/576

Owner-authorized execution on `feat/stage5-paper-category`, 2026-10-30. Stop after
verification and owner review: no commit, push, merge or checkpoint promotion.

## Versioned corpus and unchanged evaluation boundary

`exports/taco/strict-clean-stage5-tier2-v1` has **384 training images /744 boxes**.
This adds48 images and107 annotations relative to Tier1 (336/637). Training-only
sibling mapping follows D073 Tier2, without Other plastic and without accessory
filtering. Image1173 remains quarantined via its exact duplicate1305 with Unlabeled
litter. All existing exports, rights records, checkpoints, both taxonomy files and
the six core Stage4 SOURCES remain unchanged.

| ID / label | Tier1 boxes | Tier2 boxes | Training-only additional source categories |
| --- | --- | --- | --- |
| 0 cigarette | 139 | 147 | none |
| 1 plastic_bottle | 48 | 49 | none |
| 2 plastic_bag | 17 | 23 | Polypropylene bag |
| 3 plastic_wrapper | 72 | 74 | Six pack rings |
| 4 plastic_film | 125 | 134 | none |
| 5 drink_can | 47 | 64 | Food Can, Aerosol |
| 6 plastic_cup | 24 | 54 | Disposable food container, Spread tub, Other plastic container |
| 7 styrofoam_piece | 41 | 54 | Foam cup, Foam food container |
| 8 paper | 124 | 145 | Plastified paper bag |

Original class boxes are also unlocked on the newly eligible images; the gains are
not limited to newly mapped source categories. The unchanged taxonomy v2 file and
validation categories remain nine-specific. Additional mappings are recorded separately
in `training-sibling-mapping.json` and source-annotation provenance; validation/test
labels are never broadened. Counts do not establish semantic equivalence or complete
visual annotation.

The canonical `validation.json` is byte-identical to the original Stage5 validation
export (65 images /109 GTs); the new export also provides an identical `val.json`
alias to match the authorization wording. The source file is named `validation.json`,
not `val.json`. Frozen test JSON is reconstructed entirely from raw metadata and the
original eligibility ledger, including original annotation numbering/serialization,
and its in-memory hash matches the recorded original65-image test hash. Existing
and new test JSONs are not read; no held-out test photograph is opened.

All **449 authorized train/validation photos** are checked for SHA-256, dimensions
and EXIF orientation and bound in new `work/stage5-paper/rights_tier2.json`. Per-read
hash checking remains active in the Stage5 loader. The updated [academic-use assessment](TACO_STAGE_5_ACADEMIC_USE.md)
records this exact corpus and owner-authorized Section185 claim with uncertainty,
without claiming rightsholder permission, institutional approval or definitive legal
clearance. Prior exact assessment bytes are archived by their bound SHA-256; prior
rights records remain unchanged and verifiable.

## Exact paired continuation and bounded evaluations

Warm-start from verified Stage5 high-resolution E1 step307:
`stage5-paper-558bcc18d6cf4556a966d8cfc320c968/checkpoints/epoch-01`.
Parameter SHA-256 must equal
`d1310a00f540e4ea8770b32491fc47576b14cd1f4b81282756bab6ba8ddb5c1d`
before training. A fresh AdamW optimizer and seed42 phase are used, matching the D074
warm-start/settings; old optimizer/RNG states are not resumed. The dataset/exposure
count and training labels differ, so this compares corpus strategies rather than
isolating one causal factor. Shared images may appear in different shuffled order.

Exactly384 updates use `epoch_batches(384,1,42,0)`, batch1,384/576 resize with576-square
padding, FP32, TF32 disabled, AdamW LR2e-6 /WD1e-4 /foreach=False, clip norm1.0.
All558 original neck/encoder/decoder trainable tensors remain trainable; Swin-T and
BERT parameters/buffers remain frozen. Native checked loss and gradient/update probes
reuse unchanged Stage4 functions. Full resumable checkpoints at steps100/200/300 and
epoch-01/384 bind model, processor, adapter, optimizer, RNG, position and source context.

Exactly336 validation forwards cover three evaluations of112 each (65 nine-class-prompt
plus47 separate original-eight-prompt forwards), without rerunning D071/D074 references:

- Tier2 E1 endpoint.
- `soup_tier2_hr_blend2575`:0.20 * reconstructed Stage5 blend2575 +0.80 * Tier2 E1.
  The reference is reconstructed in nested CPU FP32 order (0.25 Run1 E1 +0.75 Run1 E2)
  and checked against its known parameter hash before the second average.
- `soup_tier2_d074_champion_5050`:0.50 * the verified durable D074 best soup +0.50 * Tier2 E1.
  The D074 source parameter hash must equal
  `fc0c497e1163509e96ec3a72c1f85729f18336aa1f688eee998c3c29031465eb`.

Only CPU trainable tensors enter averaging; frozen entries are compared bitwise.
Validation preserves RNG and checks unchanged parameters. GPU optimizer state is
released after training, and the model returns to CPU/cache cleanup between soups.
The highest official combined65 nine-specific AP among exactly these three candidates
is saved as a private hash-verified inference/replay bundle at `checkpoints/best-inference`.
Its manifest marks `resume_supported=false`; selecting this private replay bundle is
not production promotion. Source checkpoints are immutable.

## Commands (already executed; do not repeat training for replay)

```powershell
& .\.venv-training\Scripts\python.exe -B .\scripts\train_taco_stage5.py --warmstart stage5_highres_epoch1 --dataset exports/taco/strict-clean-stage5-tier2-v1 --rights work/stage5-paper/rights_tier2.json --epochs 1 --learning-rate 2e-6 --shortest 384 --longest 576 --seed 42 --device cuda --tier2-soups
```

## Measured results

The run completed **384 updates /384 training forwards and336 validation forwards** in 546.014 seconds for training, three evaluations, checkpoint I/O and best-bundle persistence (not active GPU time). Peak allocated VRAM was 3641.432 MiB /3.556 GiB. All reference rows reuse saved results, without GPU reruns.

### Official combined65 nine-specific (65 images /109 GTs)

| State (all384/576) | AP | AP50 | AP75 | AR100 |
| --- | --- | --- | --- | --- |
| D071 blend2575 | 0.171262 | 0.210596 | 0.168672 | 0.296896 |
| D071 HR E1 | 0.202020 | 0.246182 | 0.223509 | 0.335380 |
| D074 Tier1 E1 | 0.218215 | 0.264039 | 0.239418 | 0.356558 |
| D074 20/80 champion | 0.231226 | 0.278475 | 0.254363 | 0.369025 |
| D074 HR50/50 | 0.203191 | 0.246748 | 0.223349 | 0.337143 |
| D076 Tier2 E1 | 0.232968 | 0.280211 | 0.255207 | 0.373224 |
| D076 blend20/Tier2 80 | 0.225850 | 0.272254 | 0.248019 | 0.364579 |
| D076 D074champion50/Tier2 50 | 0.235994 | 0.282890 | 0.258859 | 0.379813 |

### Separate original-eight-prompt legacy47 (47 images /75 GTs)

| State (all384/576) | AP | AP50 | AP75 | AR100 |
| --- | --- | --- | --- | --- |
| D071 blend2575 | 0.173157 | 0.204869 | 0.185029 | 0.295804 |
| D071 HR E1 | 0.226047 | 0.263987 | 0.244547 | 0.351175 |
| D074 Tier1 E1 | 0.254373 | 0.293990 | 0.274078 | 0.391527 |
| D074 20/80 champion | 0.246970 | 0.285918 | 0.266473 | 0.368988 |
| D074 HR50/50 | 0.258472 | 0.299164 | 0.276689 | 0.391607 |
| D076 Tier2 E1 | 0.253277 | 0.297824 | 0.273198 | 0.388780 |
| D076 blend20/Tier2 80 | 0.249515 | 0.291653 | 0.266118 | 0.391664 |
| D076 D074champion50/Tier2 50 | 0.250362 | 0.291868 | 0.271069 | 0.389741 |

### Official annex18 nine-specific (18 images /34 GTs)

| State (all384/576) | AP | AP50 | AP75 | AR100 |
| --- | --- | --- | --- | --- |
| D071 blend2575 | 0.127095 | 0.229226 | 0.064007 | 0.246875 |
| D071 HR E1 | 0.154036 | 0.240087 | 0.202165 | 0.250000 |
| D074 Tier1 E1 | 0.153986 | 0.240379 | 0.202860 | 0.245000 |
| D074 20/80 champion | 0.155024 | 0.239741 | 0.203972 | 0.265625 |
| D074 HR50/50 | 0.153561 | 0.239119 | 0.202588 | 0.243750 |
| D076 Tier2 E1 | 0.155533 | 0.241560 | 0.204786 | 0.240000 |
| D076 blend20/Tier2 80 | 0.154093 | 0.241560 | 0.202325 | 0.244375 |
| D076 D074champion50/Tier2 50 | 0.153563 | 0.239039 | 0.202612 | 0.243750 |

### Combined nine-specific class AP /AR100

| Class | D071 blend2575 | D071 HR E1 | D074 Tier1 E1 | D074 20/80 champion | D074 HR50/50 | D076 Tier2 E1 | D076 blend20/Tier2 80 | D076 D074champion50/Tier2 50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cigarette | 0.065575 / 0.303846 | 0.111456 / 0.373077 | 0.113204 / 0.384615 | 0.105698 / 0.365385 | 0.112190 / 0.392308 | 0.112535 / 0.376923 | 0.111058 / 0.361538 | 0.111849 / 0.380769 |
| plastic_bottle | 0.153401 / 0.277778 | 0.252475 / 0.366667 | 0.331400 / 0.477778 | 0.274752 / 0.366667 | 0.252475 / 0.366667 | 0.331400 / 0.477778 | 0.252475 / 0.366667 | 0.331400 / 0.477778 |
| plastic_bag | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 |
| plastic_wrapper | 0.013201 / 0.076923 | 0.015842 / 0.076923 | 0.000000 / 0.000000 | 0.015842 / 0.076923 | 0.015842 / 0.076923 | 0.000000 / 0.000000 | 0.015842 / 0.076923 | 0.015842 / 0.076923 |
| plastic_film | 0.054518 / 0.564706 | 0.040660 / 0.552941 | 0.041203 / 0.523529 | 0.045461 / 0.570588 | 0.040677 / 0.535294 | 0.041773 / 0.517647 | 0.041778 / 0.529412 | 0.042013 / 0.541176 |
| drink_can | 0.411881 / 0.414286 | 0.656436 / 0.657143 | 0.670297 / 0.671429 | 0.656436 / 0.657143 | 0.670297 / 0.671429 | 0.656436 / 0.657143 | 0.656436 / 0.657143 | 0.656436 / 0.657143 |
| plastic_cup | 0.319802 / 0.316667 | 0.319802 / 0.316667 | 0.395545 / 0.466667 | 0.410693 / 0.466667 | 0.319802 / 0.316667 | 0.384724 / 0.466667 | 0.384724 / 0.466667 | 0.395545 / 0.466667 |
| styrofoam_piece | 0.148515 / 0.142857 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.148515 / 0.142857 | 0.000000 / 0.000000 | 0.148515 / 0.142857 | 0.148515 / 0.142857 | 0.148515 / 0.142857 |
| paper | 0.374468 / 0.575000 | 0.421508 / 0.675000 | 0.412287 / 0.685000 | 0.423633 / 0.675000 | 0.417440 / 0.675000 | 0.421328 / 0.720000 | 0.421824 / 0.680000 | 0.422343 / 0.675000 |

### Validation loss (native nine-prompt, mean per image)

| State | Mean validation loss |
| --- | --- |
| D071 blend2575 | 5.340110 |
| D071 HR E1 | 4.036080 |
| D074 Tier1 E1 | 3.970092 |
| D074 20/80 champion | 4.098259 |
| D074 HR50/50 | 4.008276 |
| D076 Tier2 E1 | 3.993757 |
| D076 blend20/Tier2 80 | 4.063366 |
| D076 D074champion50/Tier2 50 | 4.005876 |

Tier2 E1 improves combined AP over Tier1 E1 (0.218215 ->0.232968), including Styrofoam AP0 ->0.148515, but its separate legacy AP is slightly lower (0.254373 ->0.253277). The 20% Run1-blend soup is a negative result relative to Tier2 E1: AP0.225850. The 50/50 D074-champion/Tier2 soup is the highest-AP new candidate at **0.235994**, compared with D074 best0.231226 (absolute gain0.004768). Styrofoam AP remains0.148515 and wrapper AP0.015842; plastic_bag AP remains zero. Improvements are not uniform, and the small fixed validation set plus repeated validation selection does not establish statistical significance, test generalization or production readiness.

The highest-AP local Tier2 bundle is `checkpoints/best-inference`, state `soup_tier2_d074_champion_5050`. Its independently checked serialized parameter SHA-256 is **cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422**. It includes model safetensors, processor/tokenizer, adapter and hash manifest, with `resume_supported=false`; no optimizer state is implied. Four separate full resumable training bundles remain at100/200/300/384. Nothing is wired into detect_images.py or promoted. The existing D074 bundle remains unchanged.

## Saved GUI review

The fourth-panel selector now includes all three Tier2 states alongside all three Tier1 states and original HR E1. The newest completed local Tier2 run is discovered automatically; `--tier2-run-dir` provides an explicit override. Existing D074 discovery/default display is retained, so the owner chooses a Tier2 state in the selector. All four taxonomy views and raw/class-aware IoU/class-aware containment NMS remain available. Only verified validation photos are decoded.

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_stage5_visual_review.py --gui --tier2-run-dir .\runs\training\stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14 --image-id 777 --score-threshold 0.10 --box-top-k 15
```

## Qualification and private evidence

Guarded suite: **313 tests,297 passed,16 documented skips,zero failures/errors**. The five added fixtures check non-mutating training-only mapping, all prior/new rights compatibility and exact class counts, frozen validation alias/test hash, exact bounded settings, and Tier2 discovery/explicit override. Skips cover15 training-execution fixtures and one missing Windows symlink privilege.

CPU postchecks verify all four full checkpoint bundles (558 optimizer counters each, exact sampler positions/RNG and distinct saved weight hashes), the best replay manifest/hash and AP selection, old source bundles and D074 soup, all449 photo digests, previous rights records and exports, and runtime/core source hashes. That process did not initialize CUDA.

[Full run report](../runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14/report.json), [completion proof](../runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14/completion-verification.json), [guarded results](../runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14/verification/unit-results.json), [best replay manifest](../runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14/checkpoints/best-inference/soup_manifest.json), [epoch predictions](../runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14/validation-epoch-1.json), [20/80 soup predictions](../runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14/soup_tier2_hr_blend2575.json), and [50/50 champion-soup predictions](../runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14/soup_tier2_d074_champion_5050.json). No commit/push/merge/promotion was performed; stop for owner review.

## D072 CPU material hierarchy (81 cases)

Three new states x three image views x three taxonomies x three NMS modes. The diagnostic legacy47 uses the nine-class-prompt subset, distinct from the separate eight-prompt legacy evaluation above. Raw nine-specific combined AP/AP50/AP75/AR100 reproduce official saved metrics within1e-12. COCO uses post-NMS global top100 without a score floor; coverage uses all retained queries with score>=.05 and same-category IoU>=.50, nonexclusively. Mapped tasks change category agreement/macro averaging, not the specific model predictions or official protocol.

### D076 diagnostic combined_65

| State | Taxonomy | NMS | AP | AP50 | AP75 | AR100 | Covered /GT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tier2 E1 | 9-specific | raw | 0.232968 | 0.280211 | 0.255207 | 0.373224 | 60 / 109 |
| Tier2 E1 | 9-specific | class_aware_iou_0.50 | 0.232095 | 0.281439 | 0.253190 | 0.362176 | 60 / 109 |
| Tier2 E1 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.232582 | 0.282539 | 0.253184 | 0.362656 | 59 / 109 |
| Tier2 E1 | 5-material | raw | 0.338991 | 0.416715 | 0.363470 | 0.509181 | 79 / 109 |
| Tier2 E1 | 5-material | class_aware_iou_0.50 | 0.335336 | 0.418998 | 0.356644 | 0.492391 | 79 / 109 |
| Tier2 E1 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.335728 | 0.420072 | 0.355522 | 0.495416 | 77 / 109 |
| Tier2 E1 | 4-material | raw | 0.395853 | 0.490139 | 0.426614 | 0.605927 | 84 / 109 |
| Tier2 E1 | 4-material | class_aware_iou_0.50 | 0.391539 | 0.493309 | 0.418909 | 0.586662 | 84 / 109 |
| Tier2 E1 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.391649 | 0.494430 | 0.416380 | 0.588084 | 81 / 109 |
| Blend20/Tier2 80 | 9-specific | raw | 0.225850 | 0.272254 | 0.248019 | 0.364579 | 57 / 109 |
| Blend20/Tier2 80 | 9-specific | class_aware_iou_0.50 | 0.224913 | 0.273239 | 0.246377 | 0.351142 | 57 / 109 |
| Blend20/Tier2 80 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.224682 | 0.272166 | 0.245975 | 0.348679 | 55 / 109 |
| Blend20/Tier2 80 | 5-material | raw | 0.342659 | 0.420343 | 0.368732 | 0.502185 | 77 / 109 |
| Blend20/Tier2 80 | 5-material | class_aware_iou_0.50 | 0.338885 | 0.422536 | 0.362562 | 0.483402 | 77 / 109 |
| Blend20/Tier2 80 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.336774 | 0.418135 | 0.359908 | 0.477719 | 72 / 109 |
| Blend20/Tier2 80 | 4-material | raw | 0.400026 | 0.494862 | 0.432404 | 0.596545 | 82 / 109 |
| Blend20/Tier2 80 | 4-material | class_aware_iou_0.50 | 0.395560 | 0.497972 | 0.424835 | 0.574979 | 82 / 109 |
| Blend20/Tier2 80 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.392660 | 0.492408 | 0.420974 | 0.566600 | 76 / 109 |
| D074champion50/Tier2 50 | 9-specific | raw | 0.235994 | 0.282890 | 0.258859 | 0.379813 | 61 / 109 |
| D074champion50/Tier2 50 | 9-specific | class_aware_iou_0.50 | 0.234990 | 0.284232 | 0.257027 | 0.366932 | 61 / 109 |
| D074champion50/Tier2 50 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.235054 | 0.284233 | 0.256700 | 0.363588 | 58 / 109 |
| D074champion50/Tier2 50 | 5-material | raw | 0.342725 | 0.419523 | 0.369158 | 0.505848 | 80 / 109 |
| D074champion50/Tier2 50 | 5-material | class_aware_iou_0.50 | 0.338783 | 0.422298 | 0.362627 | 0.488064 | 80 / 109 |
| D074champion50/Tier2 50 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.337524 | 0.420218 | 0.360003 | 0.483967 | 75 / 109 |
| D074champion50/Tier2 50 | 4-material | raw | 0.400957 | 0.494067 | 0.433767 | 0.600996 | 85 / 109 |
| D074champion50/Tier2 50 | 4-material | class_aware_iou_0.50 | 0.396260 | 0.497901 | 0.425736 | 0.580680 | 85 / 109 |
| D074champion50/Tier2 50 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.394390 | 0.494682 | 0.421955 | 0.574155 | 79 / 109 |

### D076 diagnostic legacy_47

| State | Taxonomy | NMS | AP | AP50 | AP75 | AR100 | Covered /GT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tier2 E1 | 9-specific | raw | 0.227015 | 0.268597 | 0.243304 | 0.354547 | 40 / 75 |
| Tier2 E1 | 9-specific | class_aware_iou_0.50 | 0.226961 | 0.269953 | 0.243366 | 0.343449 | 40 / 75 |
| Tier2 E1 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.226767 | 0.270439 | 0.242276 | 0.337519 | 40 / 75 |
| Tier2 E1 | 5-material | raw | 0.342716 | 0.432198 | 0.366834 | 0.477827 | 55 / 75 |
| Tier2 E1 | 5-material | class_aware_iou_0.50 | 0.341492 | 0.434744 | 0.365496 | 0.464606 | 55 / 75 |
| Tier2 E1 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.340157 | 0.433879 | 0.361354 | 0.454029 | 54 / 75 |
| Tier2 E1 | 4-material | raw | 0.404569 | 0.525541 | 0.438584 | 0.580790 | 59 / 75 |
| Tier2 E1 | 4-material | class_aware_iou_0.50 | 0.403396 | 0.529697 | 0.436725 | 0.565007 | 59 / 75 |
| Tier2 E1 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.400710 | 0.526510 | 0.429991 | 0.546573 | 57 / 75 |
| Blend20/Tier2 80 | 9-specific | raw | 0.222014 | 0.261309 | 0.238481 | 0.348643 | 38 / 75 |
| Blend20/Tier2 80 | 9-specific | class_aware_iou_0.50 | 0.221681 | 0.262509 | 0.238592 | 0.333178 | 38 / 75 |
| Blend20/Tier2 80 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.220398 | 0.259154 | 0.237185 | 0.321920 | 36 / 75 |
| Blend20/Tier2 80 | 5-material | raw | 0.348624 | 0.436825 | 0.375699 | 0.479029 | 54 / 75 |
| Blend20/Tier2 80 | 5-material | class_aware_iou_0.50 | 0.346525 | 0.439453 | 0.373886 | 0.461561 | 54 / 75 |
| Blend20/Tier2 80 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.341897 | 0.430926 | 0.367656 | 0.440327 | 50 / 75 |
| Blend20/Tier2 80 | 4-material | raw | 0.411994 | 0.531134 | 0.448263 | 0.581421 | 58 / 75 |
| Blend20/Tier2 80 | 4-material | class_aware_iou_0.50 | 0.409765 | 0.535288 | 0.446518 | 0.560462 | 58 / 75 |
| Blend20/Tier2 80 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.402870 | 0.523491 | 0.437312 | 0.529275 | 53 / 75 |
| D074champion50/Tier2 50 | 9-specific | raw | 0.233309 | 0.274122 | 0.250291 | 0.367393 | 42 / 75 |
| D074champion50/Tier2 50 | 9-specific | class_aware_iou_0.50 | 0.232958 | 0.275382 | 0.250413 | 0.351928 | 42 / 75 |
| D074champion50/Tier2 50 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.232462 | 0.274703 | 0.249190 | 0.345037 | 40 / 75 |
| D074champion50/Tier2 50 | 5-material | raw | 0.348624 | 0.437775 | 0.374777 | 0.484638 | 57 / 75 |
| D074champion50/Tier2 50 | 5-material | class_aware_iou_0.50 | 0.346721 | 0.440504 | 0.373018 | 0.467811 | 57 / 75 |
| D074champion50/Tier2 50 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.343451 | 0.435819 | 0.366986 | 0.450183 | 53 / 75 |
| D074champion50/Tier2 50 | 4-material | raw | 0.412829 | 0.532275 | 0.448535 | 0.588997 | 61 / 75 |
| D074champion50/Tier2 50 | 4-material | class_aware_iou_0.50 | 0.410793 | 0.536579 | 0.446812 | 0.568795 | 61 / 75 |
| D074champion50/Tier2 50 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.405669 | 0.529261 | 0.437692 | 0.542027 | 56 / 75 |

### D076 diagnostic annex_18

| State | Taxonomy | NMS | AP | AP50 | AP75 | AR100 | Covered /GT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tier2 E1 | 9-specific | raw | 0.155533 | 0.241560 | 0.204786 | 0.240000 | 20 / 34 |
| Tier2 E1 | 9-specific | class_aware_iou_0.50 | 0.153901 | 0.243123 | 0.200524 | 0.233750 | 20 / 34 |
| Tier2 E1 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.155686 | 0.244592 | 0.202660 | 0.234375 | 19 / 34 |
| Tier2 E1 | 5-material | raw | 0.248093 | 0.326841 | 0.265644 | 0.387500 | 24 / 34 |
| Tier2 E1 | 5-material | class_aware_iou_0.50 | 0.237030 | 0.329117 | 0.245323 | 0.361250 | 24 / 34 |
| Tier2 E1 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.240016 | 0.331682 | 0.248842 | 0.370000 | 23 / 34 |
| Tier2 E1 | 4-material | raw | 0.328737 | 0.422813 | 0.350456 | 0.512222 | 25 / 34 |
| Tier2 E1 | 4-material | class_aware_iou_0.50 | 0.314788 | 0.425774 | 0.325392 | 0.481667 | 25 / 34 |
| Tier2 E1 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.318788 | 0.429730 | 0.330037 | 0.494444 | 24 / 34 |
| Blend20/Tier2 80 | 9-specific | raw | 0.154093 | 0.241560 | 0.202325 | 0.244375 | 19 / 34 |
| Blend20/Tier2 80 | 9-specific | class_aware_iou_0.50 | 0.152864 | 0.243008 | 0.198863 | 0.241250 | 19 / 34 |
| Blend20/Tier2 80 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.154849 | 0.246941 | 0.200094 | 0.247500 | 19 / 34 |
| Blend20/Tier2 80 | 5-material | raw | 0.245341 | 0.326495 | 0.263903 | 0.390000 | 23 / 34 |
| Blend20/Tier2 80 | 5-material | class_aware_iou_0.50 | 0.236394 | 0.328458 | 0.246101 | 0.368750 | 23 / 34 |
| Blend20/Tier2 80 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.237286 | 0.327454 | 0.247974 | 0.371250 | 22 / 34 |
| Blend20/Tier2 80 | 4-material | raw | 0.323794 | 0.422070 | 0.346234 | 0.515556 | 24 / 34 |
| Blend20/Tier2 80 | 4-material | class_aware_iou_0.50 | 0.312470 | 0.424549 | 0.324320 | 0.490556 | 24 / 34 |
| Blend20/Tier2 80 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.313885 | 0.424138 | 0.326818 | 0.496111 | 23 / 34 |
| D074champion50/Tier2 50 | 9-specific | raw | 0.153563 | 0.239039 | 0.202612 | 0.243750 | 19 / 34 |
| D074champion50/Tier2 50 | 9-specific | class_aware_iou_0.50 | 0.152301 | 0.241475 | 0.198896 | 0.241250 | 19 / 34 |
| D074champion50/Tier2 50 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.153286 | 0.242006 | 0.200233 | 0.228750 | 18 / 34 |
| D074champion50/Tier2 50 | 5-material | raw | 0.246709 | 0.322460 | 0.266148 | 0.383750 | 23 / 34 |
| D074champion50/Tier2 50 | 5-material | class_aware_iou_0.50 | 0.236979 | 0.326758 | 0.247819 | 0.361250 | 23 / 34 |
| D074champion50/Tier2 50 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.238295 | 0.327091 | 0.249772 | 0.358750 | 22 / 34 |
| D074champion50/Tier2 50 | 4-material | raw | 0.326301 | 0.417027 | 0.350016 | 0.505556 | 24 / 34 |
| D074champion50/Tier2 50 | 4-material | class_aware_iou_0.50 | 0.313958 | 0.422668 | 0.327401 | 0.479444 | 24 / 34 |
| D074champion50/Tier2 50 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.315875 | 0.423653 | 0.330004 | 0.479444 | 23 / 34 |

### Raw combined per-material AP /AR100

| State | Taxonomy | Material | AP | AR100 |
| --- | --- | --- | --- | --- |
| Tier2 E1 | 5-material | hazardous_cigarette | 0.112535 | 0.376923 |
| Tier2 E1 | 5-material | plastic_waste | 0.356140 | 0.648980 |
| Tier2 E1 | 5-material | metal_waste | 0.656436 | 0.657143 |
| Tier2 E1 | 5-material | styrofoam_waste | 0.148515 | 0.142857 |
| Tier2 E1 | 5-material | paper_waste | 0.421328 | 0.720000 |
| Tier2 E1 | 4-material | hazardous_cigarette | 0.112535 | 0.376923 |
| Tier2 E1 | 4-material | plastic_polymer_waste | 0.393112 | 0.669643 |
| Tier2 E1 | 4-material | metal_waste | 0.656436 | 0.657143 |
| Tier2 E1 | 4-material | paper_waste | 0.421328 | 0.720000 |
| Blend20/Tier2 80 | 5-material | hazardous_cigarette | 0.111058 | 0.361538 |
| Blend20/Tier2 80 | 5-material | plastic_waste | 0.375463 | 0.669388 |
| Blend20/Tier2 80 | 5-material | metal_waste | 0.656436 | 0.657143 |
| Blend20/Tier2 80 | 5-material | styrofoam_waste | 0.148515 | 0.142857 |
| Blend20/Tier2 80 | 5-material | paper_waste | 0.421824 | 0.680000 |
| Blend20/Tier2 80 | 4-material | hazardous_cigarette | 0.111058 | 0.361538 |
| Blend20/Tier2 80 | 4-material | plastic_polymer_waste | 0.410785 | 0.687500 |
| Blend20/Tier2 80 | 4-material | metal_waste | 0.656436 | 0.657143 |
| Blend20/Tier2 80 | 4-material | paper_waste | 0.421824 | 0.680000 |
| D074champion50/Tier2 50 | 5-material | hazardous_cigarette | 0.111849 | 0.380769 |
| D074champion50/Tier2 50 | 5-material | plastic_waste | 0.374480 | 0.673469 |
| D074champion50/Tier2 50 | 5-material | metal_waste | 0.656436 | 0.657143 |
| D074champion50/Tier2 50 | 5-material | styrofoam_waste | 0.148515 | 0.142857 |
| D074champion50/Tier2 50 | 5-material | paper_waste | 0.422343 | 0.675000 |
| D074champion50/Tier2 50 | 4-material | hazardous_cigarette | 0.111849 | 0.380769 |
| D074champion50/Tier2 50 | 4-material | plastic_polymer_waste | 0.413199 | 0.691071 |
| D074champion50/Tier2 50 | 4-material | metal_waste | 0.656436 | 0.657143 |
| D074champion50/Tier2 50 | 4-material | paper_waste | 0.422343 | 0.675000 |

[Full81-case JSON](../runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14/material-hierarchy/summary.json) and [CSV](../runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14/material-hierarchy/summary.csv) retain every per-class/material AP/AR100 and per-GT coverage result across all views/NMS settings. No photos or model execution were used in this diagnostic stage.

Native Tk check passed **84 combinations** on source-verified validation Image777: seven fourth-panel states x four taxonomy views x three NMS modes. [Native evidence](../runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14/verification/native-tk.json). The D074 default remains selected until the owner chooses a Tier2 state; no GUI default weights are promoted into inference.


The best new soup has raw5-material AP **0.342725** and raw4-material AP **0.400957**,
slightly below D074 best **0.344601 /0.402577** despite its higher nine-specific AP.
Coverage is61/109 nine-specific,80/109 five-material and85/109 four-material versus
D074's59/80/85. These tasks disagree on ranking and cannot be interchanged. The reported
small AP gain is validation-selected, with no statistical/test-generalization claim.

Final CPU comparison also verified36 raw9-specific AP/AP50/AP75/AR100 values across all three new states and all three image views within1e-12; diagnostic legacy47 matches `legacy_47_nine_class_conditioning`, not the separate eight-prompt result.
