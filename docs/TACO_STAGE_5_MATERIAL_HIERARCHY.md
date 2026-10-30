# D-072: Stage 5 saved-prediction material and hazard hierarchy

## D-074 Tier 1 extension ? 2026-10-30

The one-epoch 336-update Tier1 experiment and two in-memory soups add **81 CPU-only
saved-prediction cases** below (three states x three image views x three taxonomies
x three NMS settings). Existing D072 inputs/results remain unchanged. Official
nine-specific combined AP is 0.218215 for Tier1 E1, 0.231226 for its 20/80 blend-reference
soup, and 0.203191 for the 50/50 HR-reference soup. The 20/80 soup's raw five-material
AP is 0.344601 / coverage 80 of 109, and raw four-material AP is 0.402577 / coverage 85 of 109;
its raw nine-specific coverage is 59 of 109. These category tasks/coverage counts remain
separate from official AP. No state is promoted. See the [complete D074 training,
qualification and replay report](TACO_STAGE_5_TIER1_TRAINING.md).


Owner authorization executed on `feat/stage5-paper-category`, 2026-10-29.
This is a reversible diagnostic relabeling of saved queries, not new material recognition,
training, weight updates, or checkpoint promotion. The official nine-class taxonomy and
saved official metric dictionaries remain unchanged. See the preceding
[D-065 contract](TACO_STAGE_4_MATERIAL_HIERARCHY.md).

Measured high-resolution E1 combined raw results: nine-specific AP **0.202020**, coverage
**57/109**; five-material AP **0.314188**, **78/109**; four-material AP **0.398843**,
**84/109**. Broader category agreement accepts subtype errors and changes macro averaging.
All four raw nine-class combined metric sets independently reproduce the saved official
AP/AP50/AP75/AR100 within 1e-12. No new champion is selected or promoted by this audit.

## Mapping and replay

| Specific class | Five-material | Four-material | Material | Hazard |
| --- | --- | --- | --- | --- |
| cigarette | hazardous_cigarette | hazardous_cigarette | cellulose_acetate_filter | True |
| plastic_bottle | plastic_waste | plastic_polymer_waste | plastic | False |
| plastic_bag | plastic_waste | plastic_polymer_waste | plastic | False |
| plastic_wrapper | plastic_waste | plastic_polymer_waste | plastic | False |
| plastic_film | plastic_waste | plastic_polymer_waste | plastic | False |
| drink_can | metal_waste | metal_waste | metal | False |
| plastic_cup | plastic_waste | plastic_polymer_waste | plastic | False |
| styrofoam_piece | styrofoam_waste | plastic_polymer_waste | polystyrene_foam | False |
| paper | paper_waste | paper_waste | paper_fiber | False |

The isolated [hierarchy module](../scripts/taco_stage5_material_hierarchy.py) copies each
record and preserves its original specific ID/name, numeric query ID, raw phrase, raw
score, and original-pixel box. Hazard is a taxonomy flag, not a measured chemical property.
No probability pooling, subtype correction, or geometry adjustment occurs.

The [four-panel viewer](../scripts/taco_stage5_visual_review.py) retains GT, baseline,
Run 1 Epoch 2, and high-resolution Epoch 1 panels. The Taxonomy View selector offers
five-material, four-material polymer, nine-specific, and hierarchical five-material
labels. Both GT and predictions are remapped before class-aware suppression.
The legend and tables reflect the active category; tables retain specific_class,
hazard_flag and actual query/annotation IDs. Threshold and top-K act on the active
saved-query list. NMS offers raw, class-aware IoU .50, and class-aware IoU .50 plus
IoMin .80. Nine-specific remains the default taxonomy; containment remains the
viewer default NMS. The viewer now uses all saved 900 queries, rather than only the
old exported top100, and refuses missing high-resolution data instead of substituting
low-resolution results. Validation source photos are hash-verified before decoding.
Explicit Image 777 selection switches from the paper filter to all 65 images.

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_stage5_visual_review.py --gui --image-id 777 --score-threshold 0.10 --box-top-k 15
```

Select the desired **Taxonomy View** and NMS mode in the toolbar. All 65 validation
images remain browsable with the All Images filter. The four-panel GUI displays
three saved model states plus GT; the blend state is additionally included in the
metric audit below.

## Evaluation method and boundaries

Four states x three image views x three taxonomies x three NMS settings produce
108 diagnostic cases. Each state contains 65 x 900 original nine-class-prompt queries.
`combined_65` has 109 GTs, `legacy_47` has 75 and `annex_18` has 34.
Here legacy_47 means the original image subset under **nine-class prompt conditioning**;
it is not the separately saved original-eight-class-prompt evaluation. Annex and
combined metrics include paper. Different category lists and prompt conditions change
the task and macro averaging, so mapped AP must not replace official specific-class AP.
The baseline and Run 1 E2 use 256/384; soup and high-resolution E1 use 384/576.
Across-scale differences also include the input-scale change.

For each image, valid positive-area queries are sorted by score descending and numeric
query ID ascending. Specific-to-material mapping precedes class-aware NMS. Only an
already-kept query of the same active category can suppress a later query at inclusive
IoU >= .50 or (containment mode) IoMin >= .80. A contained distinct object can therefore
be hidden; suppression is diagnostic, not proof of duplicate identity.
COCO uses the first 100 retained queries per image with no score floor, original GT
areas, IoU .50:.05:.95 and maxDets 1/10/100. Coverage uses all retained queries at
score >= .05 and IoU >= .50 with the same active class; matching is nonexclusive and
is not official COCO recall or one-to-one precision. Cases preserve per-GT outcomes.
Absent-class COCO metrics are null rather than invented zeros.

The metric audit decodes no photos and imports no model execution path. Native Tk
verification decodes only the source-verified validation photo for Image 777. No
`test.json`, test photos, checkpoint weights or optimizer state were opened. Input
prediction JSONs, validation annotations, taxonomy v2 and the six core Stage 4 sources
are hashed before/after the audit. Existing private artifacts are never written.
No CUDA/model execution, backward passes, optimizer steps, commits, pushes, merges,
or checkpoint promotions occur in D-072.

## Qualification

Guarded suite: 291 tests, **275 passed, 16 documented skips, zero failures/errors**.
Skips cover 15 training-execution fixtures and one unavailable Windows symlink privilege.
Fixtures cover all class mappings/provenance, paper and hazard flags, remap-before-NMS,
polymer vs Styrofoam separation, deterministic tied-score order, perfect/empty COCO
predictions, and selector integration. Native Tk Image 777 passed all four taxonomy
views x three NMS settings; this is automated rendering evidence, not owner visual approval.

## Diagnostic results

Private evidence: [full metrics and per-GT coverage](../runs/training/stage5-material-hierarchy-3925a7c594d143afb3d9b4d594ad1cb6/summary.json), [CSV](../runs/training/stage5-material-hierarchy-3925a7c594d143afb3d9b4d594ad1cb6/summary.csv), [qualification](../runs/training/stage5-material-hierarchy-3925a7c594d143afb3d9b4d594ad1cb6/qualification.json), [guarded test results](../runs/training/stage5-material-hierarchy-3925a7c594d143afb3d9b4d594ad1cb6/unit-results.json), and [mapped-record examples](../runs/training/stage5-material-hierarchy-3925a7c594d143afb3d9b4d594ad1cb6/mapping-examples.json). Input SHA-256 values are in the full summary. All before/after hashes matched.

### combined_65

| State | Taxonomy | NMS | AP | AP50 | AP75 | AR100 | Covered / GT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_256_384 | 9-specific | raw | 0.057617 | 0.070768 | 0.059660 | 0.147380 | 23 / 109 |
| baseline_256_384 | 9-specific | class_aware_iou_0.50 | 0.058208 | 0.071935 | 0.060179 | 0.143531 | 23 / 109 |
| baseline_256_384 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.057885 | 0.070893 | 0.060230 | 0.126026 | 21 / 109 |
| baseline_256_384 | 5-material | raw | 0.061765 | 0.090612 | 0.065845 | 0.152166 | 50 / 109 |
| baseline_256_384 | 5-material | class_aware_iou_0.50 | 0.065974 | 0.100318 | 0.070328 | 0.139922 | 50 / 109 |
| baseline_256_384 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.066326 | 0.098408 | 0.071518 | 0.130722 | 44 / 109 |
| baseline_256_384 | 4-material | raw | 0.093002 | 0.131685 | 0.098193 | 0.195948 | 57 / 109 |
| baseline_256_384 | 4-material | class_aware_iou_0.50 | 0.099631 | 0.145477 | 0.105114 | 0.182555 | 57 / 109 |
| baseline_256_384 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.098885 | 0.140650 | 0.106678 | 0.169780 | 50 / 109 |
| epoch2_256_384 | 9-specific | raw | 0.135663 | 0.173659 | 0.143699 | 0.247600 | 43 / 109 |
| epoch2_256_384 | 9-specific | class_aware_iou_0.50 | 0.135960 | 0.175714 | 0.145035 | 0.242892 | 43 / 109 |
| epoch2_256_384 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.135748 | 0.174526 | 0.145314 | 0.235752 | 41 / 109 |
| epoch2_256_384 | 5-material | raw | 0.225095 | 0.285884 | 0.246719 | 0.364185 | 66 / 109 |
| epoch2_256_384 | 5-material | class_aware_iou_0.50 | 0.223918 | 0.290209 | 0.246173 | 0.352253 | 66 / 109 |
| epoch2_256_384 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.223877 | 0.289493 | 0.247402 | 0.340290 | 63 / 109 |
| epoch2_256_384 | 4-material | raw | 0.248672 | 0.322415 | 0.276006 | 0.419581 | 70 / 109 |
| epoch2_256_384 | 4-material | class_aware_iou_0.50 | 0.247505 | 0.327915 | 0.276157 | 0.405941 | 70 / 109 |
| epoch2_256_384 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.245748 | 0.323667 | 0.274626 | 0.389457 | 66 / 109 |
| blend_2575_384_576 | 9-specific | raw | 0.171262 | 0.210596 | 0.168672 | 0.296896 | 50 / 109 |
| blend_2575_384_576 | 9-specific | class_aware_iou_0.50 | 0.171659 | 0.212047 | 0.168981 | 0.290838 | 50 / 109 |
| blend_2575_384_576 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.171119 | 0.209945 | 0.169154 | 0.283036 | 48 / 109 |
| blend_2575_384_576 | 5-material | raw | 0.275480 | 0.342716 | 0.276385 | 0.430871 | 74 / 109 |
| blend_2575_384_576 | 5-material | class_aware_iou_0.50 | 0.273341 | 0.344394 | 0.272398 | 0.416413 | 74 / 109 |
| blend_2575_384_576 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.272651 | 0.342506 | 0.271890 | 0.421546 | 70 / 109 |
| blend_2575_384_576 | 4-material | raw | 0.322000 | 0.404507 | 0.325088 | 0.510337 | 79 / 109 |
| blend_2575_384_576 | 4-material | class_aware_iou_0.50 | 0.319395 | 0.406796 | 0.320116 | 0.493922 | 79 / 109 |
| blend_2575_384_576 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.317993 | 0.403449 | 0.318796 | 0.498935 | 74 / 109 |
| highres_epoch1_384_576 | 9-specific | raw | 0.202020 | 0.246182 | 0.223509 | 0.335380 | 57 / 109 |
| highres_epoch1_384_576 | 9-specific | class_aware_iou_0.50 | 0.201053 | 0.247156 | 0.221842 | 0.321418 | 57 / 109 |
| highres_epoch1_384_576 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.200992 | 0.246526 | 0.221685 | 0.318049 | 54 / 109 |
| highres_epoch1_384_576 | 5-material | raw | 0.314188 | 0.389516 | 0.339887 | 0.476962 | 78 / 109 |
| highres_epoch1_384_576 | 5-material | class_aware_iou_0.50 | 0.310333 | 0.391891 | 0.333670 | 0.458002 | 78 / 109 |
| highres_epoch1_384_576 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.308275 | 0.387983 | 0.331175 | 0.453763 | 72 / 109 |
| highres_epoch1_384_576 | 4-material | raw | 0.398843 | 0.490698 | 0.431261 | 0.600412 | 84 / 109 |
| highres_epoch1_384_576 | 4-material | class_aware_iou_0.50 | 0.394632 | 0.494518 | 0.423993 | 0.583152 | 84 / 109 |
| highres_epoch1_384_576 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.391579 | 0.489303 | 0.420117 | 0.572370 | 77 / 109 |

### legacy_47

| State | Taxonomy | NMS | AP | AP50 | AP75 | AR100 | Covered / GT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_256_384 | 9-specific | raw | 0.076478 | 0.096261 | 0.076939 | 0.175702 | 21 / 75 |
| baseline_256_384 | 9-specific | class_aware_iou_0.50 | 0.077476 | 0.098219 | 0.077818 | 0.170294 | 21 / 75 |
| baseline_256_384 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.077093 | 0.096704 | 0.078167 | 0.152826 | 19 / 75 |
| baseline_256_384 | 5-material | raw | 0.084453 | 0.127632 | 0.085085 | 0.182612 | 40 / 75 |
| baseline_256_384 | 5-material | class_aware_iou_0.50 | 0.093643 | 0.146657 | 0.095165 | 0.172356 | 40 / 75 |
| baseline_256_384 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.095660 | 0.147491 | 0.098062 | 0.165625 | 37 / 75 |
| baseline_256_384 | 4-material | raw | 0.144843 | 0.208255 | 0.146722 | 0.251389 | 45 / 75 |
| baseline_256_384 | 4-material | class_aware_iou_0.50 | 0.158453 | 0.235330 | 0.161209 | 0.239268 | 45 / 75 |
| baseline_256_384 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.157331 | 0.228840 | 0.163791 | 0.226894 | 41 / 75 |
| epoch2_256_384 | 9-specific | raw | 0.137352 | 0.160919 | 0.142071 | 0.211009 | 25 / 75 |
| epoch2_256_384 | 9-specific | class_aware_iou_0.50 | 0.137527 | 0.161896 | 0.142244 | 0.206642 | 25 / 75 |
| epoch2_256_384 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.136950 | 0.160086 | 0.142465 | 0.199670 | 24 / 75 |
| epoch2_256_384 | 5-material | raw | 0.223805 | 0.281473 | 0.235965 | 0.313175 | 44 / 75 |
| epoch2_256_384 | 5-material | class_aware_iou_0.50 | 0.222078 | 0.284277 | 0.232574 | 0.303800 | 44 / 75 |
| epoch2_256_384 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.220842 | 0.281256 | 0.233357 | 0.287935 | 42 / 75 |
| epoch2_256_384 | 4-material | raw | 0.239736 | 0.316868 | 0.258166 | 0.353698 | 47 / 75 |
| epoch2_256_384 | 4-material | class_aware_iou_0.50 | 0.237870 | 0.320845 | 0.254709 | 0.342460 | 47 / 75 |
| epoch2_256_384 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.232699 | 0.310128 | 0.249872 | 0.318218 | 44 / 75 |
| blend_2575_384_576 | 9-specific | raw | 0.162184 | 0.189189 | 0.167004 | 0.275553 | 33 / 75 |
| blend_2575_384_576 | 9-specific | class_aware_iou_0.50 | 0.162628 | 0.191219 | 0.167446 | 0.267180 | 33 / 75 |
| blend_2575_384_576 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.160659 | 0.186199 | 0.166648 | 0.248350 | 31 / 75 |
| blend_2575_384_576 | 5-material | raw | 0.274630 | 0.345613 | 0.273422 | 0.407418 | 52 / 75 |
| blend_2575_384_576 | 5-material | class_aware_iou_0.50 | 0.272039 | 0.347652 | 0.268631 | 0.390671 | 52 / 75 |
| blend_2575_384_576 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.268642 | 0.341488 | 0.265389 | 0.373924 | 49 / 75 |
| blend_2575_384_576 | 4-material | raw | 0.323121 | 0.417445 | 0.324045 | 0.485065 | 56 / 75 |
| blend_2575_384_576 | 4-material | class_aware_iou_0.50 | 0.319590 | 0.420192 | 0.317204 | 0.464484 | 56 / 75 |
| blend_2575_384_576 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.313587 | 0.409920 | 0.310503 | 0.438600 | 52 / 75 |
| highres_epoch1_384_576 | 9-specific | raw | 0.184721 | 0.220906 | 0.200054 | 0.303226 | 38 / 75 |
| highres_epoch1_384_576 | 9-specific | class_aware_iou_0.50 | 0.184240 | 0.221361 | 0.200099 | 0.286279 | 38 / 75 |
| highres_epoch1_384_576 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.183311 | 0.219025 | 0.199054 | 0.277104 | 35 / 75 |
| highres_epoch1_384_576 | 5-material | raw | 0.297634 | 0.382983 | 0.324513 | 0.433196 | 55 / 75 |
| highres_epoch1_384_576 | 5-material | class_aware_iou_0.50 | 0.295508 | 0.384801 | 0.322728 | 0.414686 | 55 / 75 |
| highres_epoch1_384_576 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.291325 | 0.377777 | 0.316886 | 0.397619 | 50 / 75 |
| highres_epoch1_384_576 | 4-material | raw | 0.409589 | 0.524521 | 0.446416 | 0.586977 | 60 / 75 |
| highres_epoch1_384_576 | 4-material | class_aware_iou_0.50 | 0.407409 | 0.528096 | 0.444540 | 0.564628 | 60 / 75 |
| highres_epoch1_384_576 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.400906 | 0.517372 | 0.435622 | 0.538997 | 54 / 75 |

### annex_18

| State | Taxonomy | NMS | AP | AP50 | AP75 | AR100 | Covered / GT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_256_384 | 9-specific | raw | 0.029054 | 0.032356 | 0.032356 | 0.103125 | 2 / 34 |
| baseline_256_384 | 9-specific | class_aware_iou_0.50 | 0.029275 | 0.032672 | 0.032672 | 0.103125 | 2 / 34 |
| baseline_256_384 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.028713 | 0.031869 | 0.031869 | 0.081250 | 2 / 34 |
| baseline_256_384 | 5-material | raw | 0.081181 | 0.108324 | 0.103472 | 0.207500 | 10 / 34 |
| baseline_256_384 | 5-material | class_aware_iou_0.50 | 0.078976 | 0.110070 | 0.099886 | 0.172500 | 10 / 34 |
| baseline_256_384 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.074629 | 0.098740 | 0.096397 | 0.145000 | 7 / 34 |
| baseline_256_384 | 4-material | raw | 0.102950 | 0.135731 | 0.129974 | 0.280556 | 12 / 34 |
| baseline_256_384 | 4-material | class_aware_iou_0.50 | 0.102310 | 0.140649 | 0.128163 | 0.241667 | 12 / 34 |
| baseline_256_384 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.098791 | 0.129038 | 0.126450 | 0.211111 | 9 / 34 |
| epoch2_256_384 | 9-specific | raw | 0.145543 | 0.247010 | 0.200307 | 0.211250 | 18 / 34 |
| epoch2_256_384 | 9-specific | class_aware_iou_0.50 | 0.146538 | 0.250672 | 0.202955 | 0.209375 | 18 / 34 |
| epoch2_256_384 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.146736 | 0.249762 | 0.203462 | 0.200000 | 17 / 34 |
| epoch2_256_384 | 5-material | raw | 0.221691 | 0.317075 | 0.260763 | 0.315000 | 22 / 34 |
| epoch2_256_384 | 5-material | class_aware_iou_0.50 | 0.220505 | 0.325528 | 0.261156 | 0.293750 | 22 / 34 |
| epoch2_256_384 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.222295 | 0.327201 | 0.263627 | 0.286250 | 21 / 34 |
| epoch2_256_384 | 4-material | raw | 0.285363 | 0.401036 | 0.335817 | 0.410556 | 23 / 34 |
| epoch2_256_384 | 4-material | class_aware_iou_0.50 | 0.283906 | 0.411791 | 0.336487 | 0.386111 | 23 / 34 |
| epoch2_256_384 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.287325 | 0.415417 | 0.340762 | 0.377778 | 22 / 34 |
| blend_2575_384_576 | 9-specific | raw | 0.127095 | 0.229226 | 0.064007 | 0.246875 | 17 / 34 |
| blend_2575_384_576 | 9-specific | class_aware_iou_0.50 | 0.127877 | 0.230526 | 0.064636 | 0.246875 | 17 / 34 |
| blend_2575_384_576 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.130638 | 0.235987 | 0.066848 | 0.234375 | 17 / 34 |
| blend_2575_384_576 | 5-material | raw | 0.209200 | 0.301570 | 0.186752 | 0.413750 | 22 / 34 |
| blend_2575_384_576 | 5-material | class_aware_iou_0.50 | 0.206647 | 0.304317 | 0.182193 | 0.393750 | 22 / 34 |
| blend_2575_384_576 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.211265 | 0.309777 | 0.186956 | 0.396250 | 21 / 34 |
| blend_2575_384_576 | 4-material | raw | 0.282018 | 0.392042 | 0.260484 | 0.566667 | 23 / 34 |
| blend_2575_384_576 | 4-material | class_aware_iou_0.50 | 0.279456 | 0.396298 | 0.254682 | 0.544444 | 23 / 34 |
| blend_2575_384_576 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.287808 | 0.406545 | 0.263152 | 0.552778 | 22 / 34 |
| highres_epoch1_384_576 | 9-specific | raw | 0.154036 | 0.240087 | 0.202165 | 0.250000 | 19 / 34 |
| highres_epoch1_384_576 | 9-specific | class_aware_iou_0.50 | 0.153008 | 0.242613 | 0.198672 | 0.247500 | 19 / 34 |
| highres_epoch1_384_576 | 9-specific | class_aware_iou_0.50_containment_0.80 | 0.154779 | 0.246885 | 0.200011 | 0.241250 | 19 / 34 |
| highres_epoch1_384_576 | 5-material | raw | 0.247627 | 0.323134 | 0.266164 | 0.388750 | 23 / 34 |
| highres_epoch1_384_576 | 5-material | class_aware_iou_0.50 | 0.237665 | 0.327669 | 0.247436 | 0.366250 | 23 / 34 |
| highres_epoch1_384_576 | 5-material | class_aware_iou_0.50_containment_0.80 | 0.238026 | 0.326760 | 0.249445 | 0.358750 | 22 / 34 |
| highres_epoch1_384_576 | 4-material | raw | 0.327332 | 0.417727 | 0.349829 | 0.511111 | 24 / 34 |
| highres_epoch1_384_576 | 4-material | class_aware_iou_0.50 | 0.316464 | 0.425644 | 0.328392 | 0.512778 | 24 / 34 |
| highres_epoch1_384_576 | 4-material | class_aware_iou_0.50_containment_0.80 | 0.315467 | 0.423157 | 0.329530 | 0.479444 | 23 / 34 |

### Per-material AP for every state, view and NMS setting

#### 5-material

| State | View | NMS | hazardous_cigarette | plastic_waste | metal_waste | styrofoam_waste | paper_waste |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_256_384 | combined_65 | raw | 0.022451 | 0.286374 | 0.000000 | 0.000000 | 0.000000 |
| baseline_256_384 | combined_65 | class_aware_iou_0.50 | 0.022498 | 0.307374 | 0.000000 | 0.000000 | 0.000000 |
| baseline_256_384 | combined_65 | class_aware_iou_0.50_containment_0.80 | 0.022410 | 0.309222 | 0.000000 | 0.000000 | 0.000000 |
| baseline_256_384 | legacy_47 | raw | 0.029469 | 0.308344 | 0.000000 | 0.000000 | n/a |
| baseline_256_384 | legacy_47 | class_aware_iou_0.50 | 0.029547 | 0.345024 | 0.000000 | 0.000000 | n/a |
| baseline_256_384 | legacy_47 | class_aware_iou_0.50_containment_0.80 | 0.029409 | 0.353233 | 0.000000 | 0.000000 | n/a |
| baseline_256_384 | annex_18 | raw | 0.000000 | 0.324723 | n/a | 0.000000 | 0.000000 |
| baseline_256_384 | annex_18 | class_aware_iou_0.50 | 0.000000 | 0.315902 | n/a | 0.000000 | 0.000000 |
| baseline_256_384 | annex_18 | class_aware_iou_0.50_containment_0.80 | 0.000000 | 0.298516 | n/a | 0.000000 | 0.000000 |
| epoch2_256_384 | combined_65 | raw | 0.030489 | 0.349506 | 0.287129 | 0.148515 | 0.309834 |
| epoch2_256_384 | combined_65 | class_aware_iou_0.50 | 0.030488 | 0.342455 | 0.287129 | 0.148515 | 0.311007 |
| epoch2_256_384 | combined_65 | class_aware_iou_0.50_containment_0.80 | 0.030497 | 0.341295 | 0.287129 | 0.148515 | 0.311950 |
| epoch2_256_384 | legacy_47 | raw | 0.038306 | 0.361863 | 0.287129 | 0.207921 | n/a |
| epoch2_256_384 | legacy_47 | class_aware_iou_0.50 | 0.038289 | 0.354975 | 0.287129 | 0.207921 | n/a |
| epoch2_256_384 | legacy_47 | class_aware_iou_0.50_containment_0.80 | 0.038336 | 0.349984 | 0.287129 | 0.207921 | n/a |
| epoch2_256_384 | annex_18 | raw | 0.000000 | 0.379332 | n/a | 0.000000 | 0.507431 |
| epoch2_256_384 | annex_18 | class_aware_iou_0.50 | 0.000000 | 0.369015 | n/a | 0.000000 | 0.513006 |
| epoch2_256_384 | annex_18 | class_aware_iou_0.50_containment_0.80 | 0.000000 | 0.371925 | n/a | 0.000000 | 0.517256 |
| blend_2575_384_576 | combined_65 | raw | 0.065575 | 0.376963 | 0.411881 | 0.148515 | 0.374468 |
| blend_2575_384_576 | combined_65 | class_aware_iou_0.50 | 0.065484 | 0.366020 | 0.411881 | 0.148515 | 0.374803 |
| blend_2575_384_576 | combined_65 | class_aware_iou_0.50_containment_0.80 | 0.065727 | 0.356305 | 0.411881 | 0.148515 | 0.380826 |
| blend_2575_384_576 | legacy_47 | raw | 0.082293 | 0.396426 | 0.411881 | 0.207921 | n/a |
| blend_2575_384_576 | legacy_47 | class_aware_iou_0.50 | 0.082160 | 0.386195 | 0.411881 | 0.207921 | n/a |
| blend_2575_384_576 | legacy_47 | class_aware_iou_0.50_containment_0.80 | 0.082530 | 0.372237 | 0.411881 | 0.207921 | n/a |
| blend_2575_384_576 | annex_18 | raw | 0.002746 | 0.360382 | n/a | 0.000000 | 0.473670 |
| blend_2575_384_576 | annex_18 | class_aware_iou_0.50 | 0.002896 | 0.348490 | n/a | 0.000000 | 0.475203 |
| blend_2575_384_576 | annex_18 | class_aware_iou_0.50_containment_0.80 | 0.003180 | 0.343773 | n/a | 0.000000 | 0.498108 |
| highres_epoch1_384_576 | combined_65 | raw | 0.111456 | 0.381539 | 0.656436 | 0.000000 | 0.421508 |
| highres_epoch1_384_576 | combined_65 | class_aware_iou_0.50 | 0.111586 | 0.368879 | 0.656436 | 0.000000 | 0.414764 |
| highres_epoch1_384_576 | combined_65 | class_aware_iou_0.50_containment_0.80 | 0.112108 | 0.354461 | 0.656436 | 0.000000 | 0.418369 |
| highres_epoch1_384_576 | legacy_47 | raw | 0.133482 | 0.400619 | 0.656436 | 0.000000 | n/a |
| highres_epoch1_384_576 | legacy_47 | class_aware_iou_0.50 | 0.133657 | 0.391938 | 0.656436 | 0.000000 | n/a |
| highres_epoch1_384_576 | legacy_47 | class_aware_iou_0.50_containment_0.80 | 0.134376 | 0.374490 | 0.656436 | 0.000000 | n/a |
| highres_epoch1_384_576 | annex_18 | raw | 0.001611 | 0.402453 | n/a | 0.000000 | 0.586443 |
| highres_epoch1_384_576 | annex_18 | class_aware_iou_0.50 | 0.001694 | 0.374511 | n/a | 0.000000 | 0.574454 |
| highres_epoch1_384_576 | annex_18 | class_aware_iou_0.50_containment_0.80 | 0.002016 | 0.364862 | n/a | 0.000000 | 0.585224 |

#### 4-material

| State | View | NMS | hazardous_cigarette | plastic_polymer_waste | metal_waste | paper_waste |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_256_384 | combined_65 | raw | 0.022451 | 0.349558 | 0.000000 | 0.000000 |
| baseline_256_384 | combined_65 | class_aware_iou_0.50 | 0.022498 | 0.376027 | 0.000000 | 0.000000 |
| baseline_256_384 | combined_65 | class_aware_iou_0.50_containment_0.80 | 0.022410 | 0.373132 | 0.000000 | 0.000000 |
| baseline_256_384 | legacy_47 | raw | 0.029469 | 0.405061 | 0.000000 | n/a |
| baseline_256_384 | legacy_47 | class_aware_iou_0.50 | 0.029547 | 0.445813 | 0.000000 | n/a |
| baseline_256_384 | legacy_47 | class_aware_iou_0.50_containment_0.80 | 0.029409 | 0.442583 | 0.000000 | n/a |
| baseline_256_384 | annex_18 | raw | 0.000000 | 0.308851 | n/a | 0.000000 |
| baseline_256_384 | annex_18 | class_aware_iou_0.50 | 0.000000 | 0.306931 | n/a | 0.000000 |
| baseline_256_384 | annex_18 | class_aware_iou_0.50_containment_0.80 | 0.000000 | 0.296373 | n/a | 0.000000 |
| epoch2_256_384 | combined_65 | raw | 0.030489 | 0.367238 | 0.287129 | 0.309834 |
| epoch2_256_384 | combined_65 | class_aware_iou_0.50 | 0.030488 | 0.361402 | 0.287129 | 0.311000 |
| epoch2_256_384 | combined_65 | class_aware_iou_0.50_containment_0.80 | 0.030497 | 0.353415 | 0.287129 | 0.311950 |
| epoch2_256_384 | legacy_47 | raw | 0.038306 | 0.393772 | 0.287129 | n/a |
| epoch2_256_384 | legacy_47 | class_aware_iou_0.50 | 0.038289 | 0.388191 | 0.287129 | n/a |
| epoch2_256_384 | legacy_47 | class_aware_iou_0.50_containment_0.80 | 0.038336 | 0.372632 | 0.287129 | n/a |
| epoch2_256_384 | annex_18 | raw | 0.000000 | 0.348658 | n/a | 0.507431 |
| epoch2_256_384 | annex_18 | class_aware_iou_0.50 | 0.000000 | 0.338713 | n/a | 0.513006 |
| epoch2_256_384 | annex_18 | class_aware_iou_0.50_containment_0.80 | 0.000000 | 0.344718 | n/a | 0.517256 |
| blend_2575_384_576 | combined_65 | raw | 0.065575 | 0.436076 | 0.411881 | 0.374468 |
| blend_2575_384_576 | combined_65 | class_aware_iou_0.50 | 0.065484 | 0.425413 | 0.411881 | 0.374803 |
| blend_2575_384_576 | combined_65 | class_aware_iou_0.50_containment_0.80 | 0.065727 | 0.413512 | 0.411881 | 0.380852 |
| blend_2575_384_576 | legacy_47 | raw | 0.082293 | 0.475190 | 0.411881 | n/a |
| blend_2575_384_576 | legacy_47 | class_aware_iou_0.50 | 0.082160 | 0.464730 | 0.411881 | n/a |
| blend_2575_384_576 | legacy_47 | class_aware_iou_0.50_containment_0.80 | 0.082530 | 0.446351 | 0.411881 | n/a |
| blend_2575_384_576 | annex_18 | raw | 0.002746 | 0.369638 | n/a | 0.473670 |
| blend_2575_384_576 | annex_18 | class_aware_iou_0.50 | 0.002896 | 0.360270 | n/a | 0.475203 |
| blend_2575_384_576 | annex_18 | class_aware_iou_0.50_containment_0.80 | 0.003180 | 0.362136 | n/a | 0.498108 |
| highres_epoch1_384_576 | combined_65 | raw | 0.111456 | 0.405972 | 0.656436 | 0.421508 |
| highres_epoch1_384_576 | combined_65 | class_aware_iou_0.50 | 0.111586 | 0.395741 | 0.656436 | 0.414764 |
| highres_epoch1_384_576 | combined_65 | class_aware_iou_0.50_containment_0.80 | 0.112108 | 0.379414 | 0.656436 | 0.418358 |
| highres_epoch1_384_576 | legacy_47 | raw | 0.133482 | 0.438848 | 0.656436 | n/a |
| highres_epoch1_384_576 | legacy_47 | class_aware_iou_0.50 | 0.133657 | 0.432135 | 0.656436 | n/a |
| highres_epoch1_384_576 | legacy_47 | class_aware_iou_0.50_containment_0.80 | 0.134376 | 0.411907 | 0.656436 | n/a |
| highres_epoch1_384_576 | annex_18 | raw | 0.001611 | 0.393940 | n/a | 0.586443 |
| highres_epoch1_384_576 | annex_18 | class_aware_iou_0.50 | 0.001694 | 0.373244 | n/a | 0.574454 |
| highres_epoch1_384_576 | annex_18 | class_aware_iou_0.50_containment_0.80 | 0.002016 | 0.359307 | n/a | 0.585079 |

The full JSON also retains per-category AR100, object/image denominators, and nine-specific per-class metrics for all 108 cases. To repeat the CPU-only audit into another unique private folder:

```powershell
& .\.venv-training\Scripts\python.exe -B -m scripts.taco_stage5_material_hierarchy
```

## D-074 complete diagnostic tables

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
