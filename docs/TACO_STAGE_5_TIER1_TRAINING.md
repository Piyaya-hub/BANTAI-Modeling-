# D-074: Tier 1 strict-clean high-resolution continuation

Owner authorization received 2026-10-29; execution/qualification 2026-10-30 on
`feat/stage5-paper-category`. No commit, push, merge or checkpoint promotion.

## Versioned corpus and rights evidence

The new export `exports/taco/strict-clean-stage5-tier1-v1` contains **336 training
images / 637 annotations**, **65 validation images / 109 annotations**, and 65 frozen
test images. The 30 newly eligible downloaded training images are added and baseline
Image 1173 is excluded by exact-duplicate quarantine propagation from Image 1305,
which contains Unlabeled litter. The original Stage 2 and Stage 5 exports remain unchanged.
The nine-class taxonomy and six core Stage 4 training SOURCES remain unchanged.

Validation JSON is byte-identical to the original Stage 5 validation export. The new
test JSON is reconstructed from raw annotation metadata and the original preparation
ledger, including the original global annotation numbering and Windows serialization.
Its in-memory SHA-256 matches the original manifest's frozen test hash before writing.
Neither the original nor new test JSON is read, and no test photo is opened.

All 336 train and 65 validation photos are checked for SHA-256, raster dimensions and
EXIF orientation before execution. The new `work/stage5-paper/rights_tier1.json` binds
all 401 photo digests and the exact corpus/manifests. The Stage 5 loader verifies the
matching bound digest on each Tier 1 photo access. The owner-authorized Section 185
local academic-use claim remains distinct from rightsholder permission, institutional
approval and definitive legal clearance, which remain not claimed. See the updated
[academic-use assessment](TACO_STAGE_5_ACADEMIC_USE.md). Its previous exact bytes are
archived by content hash so the unchanged original rights record remains verifiable.
The archive fallback accepts only the same pre-bound SHA-256, not a replacement basis.

## Exact bounded execution contract

Warm-start from the fully verified Stage 5 high-resolution E1 checkpoint
`stage5-paper-558bcc18d6cf4556a966d8cfc320c968/checkpoints/epoch-01`.
Parameter SHA-256 is **d1310a00f540e4ea8770b32491fc47576b14cd1f4b81282756bab6ba8ddb5c1d**,
computed independently from verified safetensors and reproduced after loading.
Frozen parameters and buffers are checked bitwise against the pinned model.
This is a fresh AdamW optimizer and seed-42 phase, not restoration of the old optimizer.

One epoch uses `epoch_batches(336,1,42,0)` for **336 updates**, batch size 1, 384/576
shortest/longest edges and 576 square padding, FP32, TF32 disabled, AdamW LR 2e-6,
weight decay 1e-4, foreach=False, clip norm 1.0. All 558 original trainable tensors remain
trainable; Swin-T and BERT backbones remain frozen. Native checked loss, finite-gradient,
nonzero encoder-probe and optimizer-update audits reuse unchanged Stage 4 functions.
Complete resumable bundles are saved at steps 100, 200, 300 and epoch-01/336, with
processor, adapter, optimizer, RNG, sampler position, contract and commit manifests.

Exactly three 112-forward validations are authorized: epoch-end and two soups.
Each performs 65 nine-class-prompt forwards plus 47 separate original-eight-class-prompt
forwards. This totals **336 validation forwards**; no baseline reference is rerun.
Saved historical high-resolution E1 and blend metrics provide reference states.
Validation isolates/restores RNG and verifies parameter/frozen hashes.

CPU-only FP32 `average_trainable` produces these in-memory states:

- `soup_tier1_hr_blend2575`: 0.20 * existing Stage 5 blend_2575 + 0.80 * Tier 1 E1.
  Existing blend_2575 is first reconstructed as 0.25 * Run 1 E1 + 0.75 * Run 1 E2,
  verifying its known parameter hash before the second averaging operation.
- `soup_tier1_hr_5050`: 0.50 * existing high-resolution E1 + 0.50 * Tier 1 E1.

Only trainable CPU tensors enter averaging; frozen parameters/buffers never do.
Optimizer GPU states are released after training, and the model returns to CPU with
cache cleanup between soup evaluations. Source checkpoint bundles are verified before
reading and never written. Full evaluation JSONs retain all 900 queries per image.
The official v3 protocol is unchanged and uses no NMS. Material metrics are a separate
postprocessing diagnostic with 9-specific/5-material/4-material schemes and raw,
class-aware IoU .50, or class-aware IoU .50 + containment .80.

## Execution commands

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
& .\.venv-training\Scripts\python.exe -B .\scripts\train_taco_stage5.py --warmstart stage5_highres_epoch1 --dataset exports/taco/strict-clean-stage5-tier1-v1 --rights work/stage5-paper/rights_tier1.json --epochs 1 --learning-rate 2e-6 --shortest 384 --longest 576 --seed 42 --device cuda --tier1-soups
```

This command has already been executed for D-074; running it again creates another
training run and is not necessary for replay.

## Measured results

The bounded run completed with **336 training updates / 336 training forwards** and **336 validation forwards**. It recorded 501.915 seconds for training, three evaluations, checkpoint saving and CPU soup construction (not active GPU time), with peak allocated VRAM **3,641.432 MiB / 3.556 GiB**. References below reuse saved measurements; they were not rerun.

### Official nine-specific combined_65 (65 images / 109 GTs)

| State (all 384/576) | AP | AP50 | AP75 | AR100 |
| --- | --- | --- | --- | --- |
| Stage5 high-res E1 | 0.202020 | 0.246182 | 0.223509 | 0.335380 |
| Stage5 blend2575 | 0.171262 | 0.210596 | 0.168672 | 0.296896 |
| Tier1 E1 | 0.218215 | 0.264039 | 0.239418 | 0.356558 |
| Tier1 + blend2575 (20/80) | 0.231226 | 0.278475 | 0.254363 | 0.369025 |
| Tier1 + HR E1 (50/50) | 0.203191 | 0.246748 | 0.223349 | 0.337143 |

### Separate original-eight-prompt legacy_47 (47 images / 75 GTs)

| State (all 384/576) | AP | AP50 | AP75 | AR100 |
| --- | --- | --- | --- | --- |
| Stage5 high-res E1 | 0.226047 | 0.263987 | 0.244547 | 0.351175 |
| Stage5 blend2575 | 0.173157 | 0.204869 | 0.185029 | 0.295804 |
| Tier1 E1 | 0.254373 | 0.293990 | 0.274078 | 0.391527 |
| Tier1 + blend2575 (20/80) | 0.246970 | 0.285918 | 0.266473 | 0.368988 |
| Tier1 + HR E1 (50/50) | 0.258472 | 0.299164 | 0.276689 | 0.391607 |

### Official nine-specific annex_18 (18 images / 34 GTs)

| State (all 384/576) | AP | AP50 | AP75 | AR100 |
| --- | --- | --- | --- | --- |
| Stage5 high-res E1 | 0.154036 | 0.240087 | 0.202165 | 0.250000 |
| Stage5 blend2575 | 0.127095 | 0.229226 | 0.064007 | 0.246875 |
| Tier1 E1 | 0.153986 | 0.240379 | 0.202860 | 0.245000 |
| Tier1 + blend2575 (20/80) | 0.155024 | 0.239741 | 0.203972 | 0.265625 |
| Tier1 + HR E1 (50/50) | 0.153561 | 0.239119 | 0.202588 | 0.243750 |

### Combined nine-specific per-class AP and AR100

Cells are AP / AR100; these are official no-NMS nine-specific metrics.

| Class | HR E1 reference | Blend2575 reference | Tier1 E1 | Tier1 + blend2575 | Tier1 + HR E1 |
| --- | --- | --- | --- | --- | --- |
| cigarette | 0.111456 / 0.373077 | 0.065575 / 0.303846 | 0.113204 / 0.384615 | 0.105698 / 0.365385 | 0.112190 / 0.392308 |
| plastic_bottle | 0.252475 / 0.366667 | 0.153401 / 0.277778 | 0.331400 / 0.477778 | 0.274752 / 0.366667 | 0.252475 / 0.366667 |
| plastic_bag | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 |
| plastic_wrapper | 0.015842 / 0.076923 | 0.013201 / 0.076923 | 0.000000 / 0.000000 | 0.015842 / 0.076923 | 0.015842 / 0.076923 |
| plastic_film | 0.040660 / 0.552941 | 0.054518 / 0.564706 | 0.041203 / 0.523529 | 0.045461 / 0.570588 | 0.040677 / 0.535294 |
| drink_can | 0.656436 / 0.657143 | 0.411881 / 0.414286 | 0.670297 / 0.671429 | 0.656436 / 0.657143 | 0.670297 / 0.671429 |
| plastic_cup | 0.319802 / 0.316667 | 0.319802 / 0.316667 | 0.395545 / 0.466667 | 0.410693 / 0.466667 | 0.319802 / 0.316667 |
| styrofoam_piece | 0.000000 / 0.000000 | 0.148515 / 0.142857 | 0.000000 / 0.000000 | 0.148515 / 0.142857 | 0.000000 / 0.000000 |
| paper | 0.421508 / 0.675000 | 0.374468 / 0.575000 | 0.412287 / 0.685000 | 0.423633 / 0.675000 | 0.417440 / 0.675000 |

The 20/80 blend-reference soup has the highest measured Stage 5 combined nine-specific AP in this bounded comparison (**0.231226**), retaining the blend reference Styrofoam AP **0.148515**. Tier1 E1 improves combined AP over HR E1 (**0.202020 -> 0.218215**) but Styrofoam stays at zero, as in HR E1. The 50/50 HR soup reaches higher separate legacy eight-prompt AP (**0.258472**) while combined nine-specific AP is lower (**0.203191**). These scopes differ and must not be ranked as one metric. Plastic_bag AP remains zero; gains are not uniform, and no test generalization or new deployment readiness is established. The historical D061 eight-class 47-image AP 0.265563 remains a separate task; the new nine-class combined result does not replace it. No checkpoint is promoted or connected to detect_images.py.

## Saved review and qualification

The fourth-panel state selector offers original HR E1, Tier1 E1 and both soups, retaining baseline and Run 1 E2 beside GT. The details tab and plot title identify the selected state. All four taxonomy views and all three class-aware/raw NMS settings remain available.

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_stage5_visual_review.py --gui --tier1-run-dir .\runs\training\stage5-tier1-69d61fae160544468f155d34f0cb6337 --image-id 777 --score-threshold 0.10 --box-top-k 15
```

Guarded suite: **305 tests, 289 passed, 16 documented skips, zero failures/errors**. Six new Tier1 fixtures check legacy/new rights compatibility, metadata-only frozen reconstruction, strict execution bounds, frozen validation identity/test rejection, cached-NMS equivalence to direct calculation, and GUI state integration. Native Tk source-verified Image 777 passed four fourth-panel states x four taxonomy views x three NMS modes = **48 combinations**. This is automated rendering evidence, not owner visual approval.

Post-execution CPU checks verified the four complete resumable bundles, all 558 optimizer counters at their expected steps, complete RNG fields and positions, distinct saved weight hashes, exact one-pass sampler order, all 401 photo digests, unchanged core/Stage5 runtime sources, unchanged existing exports (without reading test JSONs), original rights record preservation, and three source bundles. The verification process did not initialize CUDA.

Private evidence: [run report](../runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/report.json), [completion verification](../runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/completion-verification.json), [unit results](../runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/verification/unit-results.json), [native Tk proof](../runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/verification/native-tk.json), [epoch predictions](../runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/validation-epoch-1.json), [20/80 soup predictions](../runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/soup_tier1_hr_blend2575.json), and [50/50 soup predictions](../runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/soup_tier1_hr_5050.json). Existing checkpoints, exports, taxonomy v2 and six core SOURCES remain untouched.

## D-072 material-hierarchy diagnostics

The CPU-only saved-prediction audit covers 81 cases: three new states x three image views x three taxonomies x three NMS settings. Its legacy_47 view is the **nine-class-prompt image subset**, distinct from the separate eight-prompt legacy values above. Raw nine-specific combined AP/AP50/AP75/AR100 must reproduce each saved official state. Per-GT coverage uses score >=.05 and IoU >=.50 over all retained queries; COCO uses no score floor and global top100 after NMS. Material mapping changes category agreement and macro averaging; diagnostic AP does not replace official nine-specific AP.

### combined_65: D-074 diagnostic cases

| State | Taxonomy | NMS | AP | AP50 | AP75 | AR100 | Covered / GT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tier1 E1 | 9-specific | raw | 0.218215 | 0.264039 | 0.239418 | 0.356558 | 59 / 109 |
| Tier1 E1 | 9-specific | class_aware_iou_0.50 | 0.217264 | 0.264994 | 0.237518 | 0.345834 | 59 / 109 |
| Tier1 E1 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.217089 | 0.264000 | 0.237321 | 0.340957 | 56 / 109 |
| Tier1 E1 | 5-material | raw | 0.313507 | 0.387041 | 0.335882 | 0.486984 | 80 / 109 |
| Tier1 E1 | 5-material | class_aware_iou_0.50 | 0.309798 | 0.389321 | 0.329464 | 0.468834 | 80 / 109 |
| Tier1 E1 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.308364 | 0.386807 | 0.327265 | 0.464281 | 75 / 109 |
| Tier1 E1 | 4-material | raw | 0.398819 | 0.488386 | 0.426662 | 0.612493 | 86 / 109 |
| Tier1 E1 | 4-material | class_aware_iou_0.50 | 0.394469 | 0.491730 | 0.419230 | 0.591463 | 86 / 109 |
| Tier1 E1 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.392576 | 0.488511 | 0.416069 | 0.584560 | 80 / 109 |
| Tier1 + blend2575 (20/80) | 9-specific | raw | 0.231226 | 0.278475 | 0.254363 | 0.369025 | 59 / 109 |
| Tier1 + blend2575 (20/80) | 9-specific | class_aware_iou_0.50 | 0.230236 | 0.279842 | 0.252463 | 0.354435 | 59 / 109 |
| Tier1 + blend2575 (20/80) | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.229642 | 0.278863 | 0.251522 | 0.347824 | 56 / 109 |
| Tier1 + blend2575 (20/80) | 5-material | raw | 0.344601 | 0.421697 | 0.371203 | 0.509710 | 80 / 109 |
| Tier1 + blend2575 (20/80) | 5-material | class_aware_iou_0.50 | 0.340762 | 0.424805 | 0.364575 | 0.488033 | 80 / 109 |
| Tier1 + blend2575 (20/80) | 5-material | class_aware_iou_0.50_containment_0.80 | 0.338391 | 0.420680 | 0.361433 | 0.480262 | 74 / 109 |
| Tier1 + blend2575 (20/80) | 4-material | raw | 0.402577 | 0.496785 | 0.435733 | 0.604739 | 85 / 109 |
| Tier1 + blend2575 (20/80) | 4-material | class_aware_iou_0.50 | 0.398207 | 0.501066 | 0.427805 | 0.579684 | 85 / 109 |
| Tier1 + blend2575 (20/80) | 4-material | class_aware_iou_0.50_containment_0.80 | 0.394889 | 0.495168 | 0.423384 | 0.569141 | 78 / 109 |
| Tier1 + HR E1 (50/50) | 9-specific | raw | 0.203191 | 0.246748 | 0.223349 | 0.337143 | 58 / 109 |
| Tier1 + HR E1 (50/50) | 9-specific | class_aware_iou_0.50 | 0.202181 | 0.248271 | 0.221357 | 0.323634 | 58 / 109 |
| Tier1 + HR E1 (50/50) | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.202320 | 0.248305 | 0.221139 | 0.320290 | 55 / 109 |
| Tier1 + HR E1 (50/50) | 5-material | raw | 0.316987 | 0.390511 | 0.340392 | 0.482441 | 79 / 109 |
| Tier1 + HR E1 (50/50) | 5-material | class_aware_iou_0.50 | 0.312986 | 0.393402 | 0.333589 | 0.462758 | 79 / 109 |
| Tier1 + HR E1 (50/50) | 5-material | class_aware_iou_0.50_containment_0.80 | 0.311777 | 0.391471 | 0.331033 | 0.458661 | 74 / 109 |
| Tier1 + HR E1 (50/50) | 4-material | raw | 0.402064 | 0.491462 | 0.431446 | 0.607452 | 85 / 109 |
| Tier1 + HR E1 (50/50) | 4-material | class_aware_iou_0.50 | 0.397325 | 0.495633 | 0.423107 | 0.584698 | 85 / 109 |
| Tier1 + HR E1 (50/50) | 4-material | class_aware_iou_0.50_containment_0.80 | 0.395628 | 0.492805 | 0.419463 | 0.578173 | 79 / 109 |

### legacy_47: D-074 diagnostic cases

| State | Taxonomy | NMS | AP | AP50 | AP75 | AR100 | Covered / GT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tier1 E1 | 9-specific | raw | 0.202910 | 0.242244 | 0.218219 | 0.330451 | 40 / 75 |
| Tier1 E1 | 9-specific | class_aware_iou_0.50 | 0.202844 | 0.243845 | 0.218292 | 0.319794 | 40 / 75 |
| Tier1 E1 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.201919 | 0.241498 | 0.217194 | 0.310539 | 38 / 75 |
| Tier1 E1 | 5-material | raw | 0.298608 | 0.383562 | 0.320582 | 0.439251 | 56 / 75 |
| Tier1 E1 | 5-material | class_aware_iou_0.50 | 0.297206 | 0.386606 | 0.319440 | 0.424989 | 56 / 75 |
| Tier1 E1 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.293290 | 0.380561 | 0.313883 | 0.406479 | 52 / 75 |
| Tier1 E1 | 4-material | raw | 0.410277 | 0.524102 | 0.439516 | 0.595148 | 61 / 75 |
| Tier1 E1 | 4-material | class_aware_iou_0.50 | 0.409104 | 0.528917 | 0.438014 | 0.577976 | 61 / 75 |
| Tier1 E1 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.403395 | 0.520315 | 0.429975 | 0.550325 | 56 / 75 |
| Tier1 + blend2575 (20/80) | 9-specific | raw | 0.230387 | 0.270724 | 0.246318 | 0.349164 | 40 / 75 |
| Tier1 + blend2575 (20/80) | 9-specific | class_aware_iou_0.50 | 0.230065 | 0.271973 | 0.246323 | 0.331615 | 40 / 75 |
| Tier1 + blend2575 (20/80) | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.228577 | 0.268431 | 0.244612 | 0.320878 | 37 / 75 |
| Tier1 + blend2575 (20/80) | 5-material | raw | 0.350763 | 0.441815 | 0.377576 | 0.480712 | 56 / 75 |
| Tier1 + blend2575 (20/80) | 5-material | class_aware_iou_0.50 | 0.348706 | 0.444562 | 0.375361 | 0.458436 | 56 / 75 |
| Tier1 + blend2575 (20/80) | 5-material | class_aware_iou_0.50_containment_0.80 | 0.344157 | 0.436292 | 0.369238 | 0.438244 | 51 / 75 |
| Tier1 + blend2575 (20/80) | 4-material | raw | 0.415601 | 0.536664 | 0.452733 | 0.583568 | 60 / 75 |
| Tier1 + blend2575 (20/80) | 4-material | class_aware_iou_0.50 | 0.413277 | 0.540915 | 0.450002 | 0.556295 | 60 / 75 |
| Tier1 + blend2575 (20/80) | 4-material | class_aware_iou_0.50_containment_0.80 | 0.406573 | 0.529657 | 0.441145 | 0.526497 | 54 / 75 |
| Tier1 + HR E1 (50/50) | 9-specific | raw | 0.186776 | 0.223039 | 0.200372 | 0.306654 | 39 / 75 |
| Tier1 + HR E1 (50/50) | 9-specific | class_aware_iou_0.50 | 0.186405 | 0.224597 | 0.200240 | 0.290589 | 39 / 75 |
| Tier1 + HR E1 (50/50) | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.185981 | 0.224188 | 0.199068 | 0.283698 | 37 / 75 |
| Tier1 + HR E1 (50/50) | 5-material | raw | 0.302595 | 0.387126 | 0.325825 | 0.441335 | 56 / 75 |
| Tier1 + HR E1 (50/50) | 5-material | class_aware_iou_0.50 | 0.300502 | 0.389919 | 0.323556 | 0.422024 | 56 / 75 |
| Tier1 + HR E1 (50/50) | 5-material | class_aware_iou_0.50_containment_0.80 | 0.297372 | 0.385486 | 0.317613 | 0.404396 | 52 / 75 |
| Tier1 + HR E1 (50/50) | 4-material | raw | 0.415308 | 0.529054 | 0.447145 | 0.597926 | 61 / 75 |
| Tier1 + HR E1 (50/50) | 4-material | class_aware_iou_0.50 | 0.413246 | 0.533658 | 0.444882 | 0.574315 | 61 / 75 |
| Tier1 + HR E1 (50/50) | 4-material | class_aware_iou_0.50_containment_0.80 | 0.408318 | 0.526985 | 0.435869 | 0.547547 | 56 / 75 |

### annex_18: D-074 diagnostic cases

| State | Taxonomy | NMS | AP | AP50 | AP75 | AR100 | Covered / GT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tier1 E1 | 9-specific | raw | 0.153986 | 0.240379 | 0.202860 | 0.245000 | 19 / 34 |
| Tier1 E1 | 9-specific | class_aware_iou_0.50 | 0.152355 | 0.241381 | 0.198896 | 0.241250 | 19 / 34 |
| Tier1 E1 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.153265 | 0.241735 | 0.200189 | 0.228750 | 18 / 34 |
| Tier1 E1 | 5-material | raw | 0.251819 | 0.329958 | 0.271021 | 0.411250 | 24 / 34 |
| Tier1 E1 | 5-material | class_aware_iou_0.50 | 0.240886 | 0.332119 | 0.251627 | 0.386250 | 24 / 34 |
| Tier1 E1 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.243877 | 0.335111 | 0.255484 | 0.383750 | 23 / 34 |
| Tier1 E1 | 4-material | raw | 0.332584 | 0.426103 | 0.356178 | 0.536667 | 25 / 34 |
| Tier1 E1 | 4-material | class_aware_iou_0.50 | 0.318814 | 0.428683 | 0.331908 | 0.507222 | 25 / 34 |
| Tier1 E1 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.322582 | 0.432622 | 0.336533 | 0.507222 | 24 / 34 |
| Tier1 + blend2575 (20/80) | 9-specific | raw | 0.155024 | 0.239741 | 0.203972 | 0.265625 | 19 / 34 |
| Tier1 + blend2575 (20/80) | 9-specific | class_aware_iou_0.50 | 0.154102 | 0.242343 | 0.200602 | 0.263125 | 19 / 34 |
| Tier1 + blend2575 (20/80) | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.154688 | 0.246462 | 0.200045 | 0.247500 | 19 / 34 |
| Tier1 + blend2575 (20/80) | 5-material | raw | 0.253116 | 0.329056 | 0.272676 | 0.433750 | 24 / 34 |
| Tier1 + blend2575 (20/80) | 5-material | class_aware_iou_0.50 | 0.243950 | 0.334534 | 0.254615 | 0.411250 | 24 / 34 |
| Tier1 + blend2575 (20/80) | 5-material | class_aware_iou_0.50_containment_0.80 | 0.244049 | 0.335780 | 0.255633 | 0.396250 | 23 / 34 |
| Tier1 + blend2575 (20/80) | 4-material | raw | 0.334035 | 0.424452 | 0.358172 | 0.563889 | 25 / 34 |
| Tier1 + blend2575 (20/80) | 4-material | class_aware_iou_0.50 | 0.322532 | 0.431268 | 0.335861 | 0.537778 | 25 / 34 |
| Tier1 + blend2575 (20/80) | 4-material | class_aware_iou_0.50_containment_0.80 | 0.322814 | 0.433475 | 0.336710 | 0.523889 | 24 / 34 |
| Tier1 + HR E1 (50/50) | 9-specific | raw | 0.153561 | 0.239119 | 0.202588 | 0.243750 | 19 / 34 |
| Tier1 + HR E1 (50/50) | 9-specific | class_aware_iou_0.50 | 0.152332 | 0.241626 | 0.198896 | 0.241250 | 19 / 34 |
| Tier1 + HR E1 (50/50) | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.153229 | 0.241900 | 0.200189 | 0.228750 | 18 / 34 |
| Tier1 + HR E1 (50/50) | 5-material | raw | 0.246711 | 0.322472 | 0.266148 | 0.383750 | 23 / 34 |
| Tier1 + HR E1 (50/50) | 5-material | class_aware_iou_0.50 | 0.236983 | 0.326876 | 0.247819 | 0.361250 | 23 / 34 |
| Tier1 + HR E1 (50/50) | 5-material | class_aware_iou_0.50_containment_0.80 | 0.238207 | 0.326986 | 0.249718 | 0.358750 | 22 / 34 |
| Tier1 + HR E1 (50/50) | 4-material | raw | 0.326294 | 0.417017 | 0.350016 | 0.505556 | 24 / 34 |
| Tier1 + HR E1 (50/50) | 4-material | class_aware_iou_0.50 | 0.313961 | 0.422786 | 0.327401 | 0.479444 | 24 / 34 |
| Tier1 + HR E1 (50/50) | 4-material | class_aware_iou_0.50_containment_0.80 | 0.315757 | 0.423514 | 0.329932 | 0.479444 | 23 / 34 |

### Raw combined per-material AP / AR100

| State | Taxonomy | Category | AP | AR100 |
| --- | --- | --- | --- | --- |
| Tier1 E1 | 5-material | hazardous_cigarette | 0.113204 | 0.384615 |
| Tier1 E1 | 5-material | plastic_waste | 0.371749 | 0.693878 |
| Tier1 E1 | 5-material | metal_waste | 0.670297 | 0.671429 |
| Tier1 E1 | 5-material | styrofoam_waste | 0.000000 | 0.000000 |
| Tier1 E1 | 5-material | paper_waste | 0.412287 | 0.685000 |
| Tier1 E1 | 4-material | hazardous_cigarette | 0.113204 | 0.384615 |
| Tier1 E1 | 4-material | plastic_polymer_waste | 0.399489 | 0.708929 |
| Tier1 E1 | 4-material | metal_waste | 0.670297 | 0.671429 |
| Tier1 E1 | 4-material | paper_waste | 0.412287 | 0.685000 |
| Tier1 + blend2575 (20/80) | 5-material | hazardous_cigarette | 0.105698 | 0.365385 |
| Tier1 + blend2575 (20/80) | 5-material | plastic_waste | 0.388723 | 0.708163 |
| Tier1 + blend2575 (20/80) | 5-material | metal_waste | 0.656436 | 0.657143 |
| Tier1 + blend2575 (20/80) | 5-material | styrofoam_waste | 0.148515 | 0.142857 |
| Tier1 + blend2575 (20/80) | 5-material | paper_waste | 0.423633 | 0.675000 |
| Tier1 + blend2575 (20/80) | 4-material | hazardous_cigarette | 0.105698 | 0.365385 |
| Tier1 + blend2575 (20/80) | 4-material | plastic_polymer_waste | 0.424543 | 0.721429 |
| Tier1 + blend2575 (20/80) | 4-material | metal_waste | 0.656436 | 0.657143 |
| Tier1 + blend2575 (20/80) | 4-material | paper_waste | 0.423633 | 0.675000 |
| Tier1 + HR E1 (50/50) | 5-material | hazardous_cigarette | 0.112190 | 0.392308 |
| Tier1 + HR E1 (50/50) | 5-material | plastic_waste | 0.385007 | 0.673469 |
| Tier1 + HR E1 (50/50) | 5-material | metal_waste | 0.670297 | 0.671429 |
| Tier1 + HR E1 (50/50) | 5-material | styrofoam_waste | 0.000000 | 0.000000 |
| Tier1 + HR E1 (50/50) | 5-material | paper_waste | 0.417440 | 0.675000 |
| Tier1 + HR E1 (50/50) | 4-material | hazardous_cigarette | 0.112190 | 0.392308 |
| Tier1 + HR E1 (50/50) | 4-material | plastic_polymer_waste | 0.408328 | 0.691071 |
| Tier1 + HR E1 (50/50) | 4-material | metal_waste | 0.670297 | 0.671429 |
| Tier1 + HR E1 (50/50) | 4-material | paper_waste | 0.417440 | 0.675000 |

All per-category AP/AR100 values for every NMS and view, GT denominators and per-GT coverage are retained in the full JSON. No photo/model execution occurs in this CPU-only diagnostic stage. All three raw nine-specific combined AP/AP50/AP75/AR100 sets reproduce the corresponding official saved metric dictionaries within 1e-12.

[D-074 full material cases](../runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/material-hierarchy/summary.json). The diagnostic legacy_47 rows use nine-class-prompt conditioning; they must not be substituted for the separate eight-prompt legacy evaluation in the official training report.


## D-075: persisted private replay bundle - 2026-10-30

The exact best measured soup is now saved at
`checkpoints/soup-tier1-hr-blend2575` inside this run, with parameter SHA-256
`fc0c497e1163509e96ec3a72c1f85729f18336aa1f688eee998c3c29031465eb`.
Its `soup_manifest.json` marks `resume_supported=false`; this is a weight-averaged
inference/replay bundle with no optimizer state, not a resumable training checkpoint.
D075 performed zero model forwards or GPU execution. Source/evaluation hashes remain
unchanged. The viewer now auto-discovers this complete local run and selects the
best soup by default; the explicit D074 command above continues to work.
