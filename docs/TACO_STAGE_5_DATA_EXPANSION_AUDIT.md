# D-073: Stage 5 dataset expansion audit and proposal

Owner-authorized read-only audit on `feat/stage5-paper-category`, 2026-10-29.
No new export, training, inference, CUDA execution, checkpoint access, taxonomy change,
commit or push. Existing datasets and the six core Stage 4 sources remain unchanged.
The [isolated audit script](../scripts/taco_stage5_data_expansion_audit.py) writes only
into a unique private audit folder and blocks held-out photo, test.json and checkpoint
reads with a process-wide audit hook.

## Physical scope and frozen boundaries

The audit accounts for all 1,500 raw TACO image records. To respect the requirement that
the 65 validation and 65 test images remain untouchable, **1,370 non-held-out photos
are physically rescanned and fingerprinted; 130 held-out photos are not opened**.
Frozen fingerprints and eligibility are taken from the existing preparation ledger,
not from test.json. Their current on-disk content is intentionally not revalidated.
This is not a claim of 1,500 fresh photo decodes. The raw annotation document is read
as metadata, including the membership needed to exclude held-out images.

Fresh physical checks use the existing raster decoder, declared/actual dimensions,
EXIF orientation, annotation validation and no-repair policy. Whole-image strict-clean
rules are replayed through unchanged `taco_preparation.make_plan` for each proposal.
No source photographs are changed or copied. Frozen split IDs and their hashes are
recorded in summary.json, and their 65/65 membership is excluded from every training
candidate set. No new validation or test images are appended.

Exact-file, exact-pixel and near-duplicate checks use the existing dHash/aHash/aspect
policy (both Hamming distances <=4, relative aspect difference <=0.02). Components are
built across all readable records, including ineligible bridge images. Any component
touching either frozen split is excluded from proposed training. Near candidates are
conservatively treated as connected, not claimed to be visually confirmed identities.
This algorithm cannot guarantee detection of every crop, rotation or changed viewpoint.

## Cumulative proposal definitions

- **Tier 1:** unchanged nine-class mapping and strict whole-image quarantine, using the
  fresh physical evidence for non-held-out records instead of historical missing flags.
- **Tier 2:** Tier 1 plus the requested sibling categories, with **Other plastic evaluated
  separately** because it is not specifically a wrapper category. Main Tier 2 adds
  Polypropylene bag -> plastic_bag; Six pack rings -> plastic_wrapper; Food Can/Aerosol
  -> drink_can; Disposable food container/Spread tub/Other plastic container ->
  plastic_cup; Foam cup/Foam food container -> styrofoam_piece; Plastified paper bag
  -> paper. All original class IDs remain unchanged in the proposal.
- **Tier 3:** Tier 2 plus training-only omission of valid non-crowd/non-ignore accessory
  boxes: Plastic bottle cap, Metal bottle cap, Plastic lid, Metal lid, Plastic straw,
  Plastic utensils, Pop tab, Aluminium foil and Glass bottle. A retained mapped box
  is still required. Invalid annotations, crowd/ignore, Unlabeled litter, and every
  other excluded category still quarantine the image. Duplicate checks remain active.

These are metadata proposals, not edits to taxonomy v2 or an approval that the added
subtypes are equivalent. Containers are broader than cups; food cans/aerosols are
broader than drink cans; foam cups differ from loose Styrofoam pieces. Omitted accessory
boxes would become unlabeled background under the current native loss: this is **not
ignore-region training**, and may hurt material detection. The audit does not assess
annotation completeness from pixels or perform a semantic visual review.

## Measured results

All 252 historically missing images are now readable. All 1,370 physically scanned records are readable, but readability does not imply strict eligibility. The fresh strict waterfall still excludes 313 non-held-out images for raster/declared dimension mismatch, nine for EXIF ambiguity, and ten for invalid annotations before category/quarantine decisions.

| Proposal | Training images | Boxes | Gross newly eligible images | Net images vs 307 | Net boxes vs 595 | Previously missing among new images |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 307 | 595 | 0 | 0 | 0 | - |
| tier1 | 336 | 637 | 30 | +29 | +42 | 30 |
| tier2 | 384 | 744 | 78 | +77 | +149 | 32 |
| tier3 | 584 | 1260 | 278 | +277 | +665 | 50 |
| tier2_with_other_plastic | 461 | 912 | 155 | +154 | +317 | 43 |
| tier3_with_other_plastic | 682 | 1484 | 376 | +375 | +889 | 66 |

These are proposed candidate totals, not new exports. One original training image is excluded in every tier: **1173**, which is exact-file and exact-pixel identical to restored **1305**. Image 1305 has an Unlabeled litter annotation (GT4050). Under the existing coassign-and-propagate-quarantine policy, this quarantines 1173 too, removing its one plastic_wrapper box (source GT3636) from the proposal. Tier 1 adds 30 images / 43 boxes and removes one image / one box: net +29 / +42. The existing 307-image export remains untouched.

### Per-class box counts (net change from baseline in parentheses)

| ID / class | Baseline | Tier 1 | Tier 2 | Tier 3 |
| --- | --- | --- | --- | --- |
| 0: cigarette | 135 | 139 (+4) | 147 (+12) | 252 (+117) |
| 1: plastic_bottle | 46 | 48 (+2) | 49 (+3) | 156 (+110) |
| 2: plastic_bag | 16 | 17 (+1) | 23 (+7) | 40 (+24) |
| 3: plastic_wrapper | 67 | 72 (+5) | 74 (+7) | 97 (+30) |
| 4: plastic_film | 117 | 125 (+8) | 134 (+17) | 169 (+52) |
| 5: drink_can | 42 | 47 (+5) | 64 (+22) | 165 (+123) |
| 6: plastic_cup | 20 | 24 (+4) | 54 (+34) | 75 (+55) |
| 7: styrofoam_piece | 32 | 41 (+9) | 54 (+22) | 63 (+31) |
| 8: paper | 120 | 124 (+4) | 145 (+25) | 243 (+123) |

### Per-class image support (net change from baseline in parentheses)

Images may carry multiple classes, so columns cannot be summed to obtain dataset size.

| ID / class | Baseline | Tier 1 | Tier 2 | Tier 3 |
| --- | --- | --- | --- | --- |
| 0: cigarette | 40 | 42 (+2) | 47 (+7) | 89 (+49) |
| 1: plastic_bottle | 37 | 39 (+2) | 40 (+3) | 127 (+90) |
| 2: plastic_bag | 16 | 17 (+1) | 22 (+6) | 30 (+14) |
| 3: plastic_wrapper | 60 | 65 (+5) | 67 (+7) | 85 (+25) |
| 4: plastic_film | 102 | 108 (+6) | 116 (+14) | 137 (+35) |
| 5: drink_can | 33 | 37 (+4) | 47 (+14) | 99 (+66) |
| 6: plastic_cup | 19 | 23 (+4) | 52 (+33) | 69 (+50) |
| 7: styrofoam_piece | 26 | 32 (+6) | 45 (+19) | 52 (+26) |
| 8: paper | 86 | 90 (+4) | 101 (+15) | 165 (+79) |

### Mapping sensitivity

Adding Other plastic to plastic_wrapper on top of Tier 2 increases totals by 77 images / 168 boxes (384/744 -> 461/912). On top of Tier 3 it adds 98 images / 224 boxes (584/1260 -> 682/1484). These gains include original-class annotations unlocked on the same photographs, not just newly mapped subtype boxes. Main Tier 3 omits 290 accessory annotations on its selected photos; the Other plastic variant omits 323.

| Added source category | Target ID | Raw boxes / images | Retained in Tier 2 boxes / images | Retained in Tier 3 boxes / images |
| --- | --- | --- | --- | --- |
| Polypropylene bag | 2 | 3 / 3 | 1 / 1 | 1 / 1 |
| Six pack rings | 3 | 5 / 4 | 0 / 0 | 1 / 1 |
| Food Can | 5 | 34 / 20 | 10 / 4 | 27 / 13 |
| Aerosol | 5 | 10 / 10 | 5 / 5 | 8 / 8 |
| Disposable food container | 6 | 38 / 37 | 21 / 21 | 24 / 23 |
| Spread tub | 6 | 9 / 9 | 7 / 7 | 7 / 7 |
| Other plastic container | 6 | 6 / 6 | 0 / 0 | 0 / 0 |
| Foam cup | 7 | 13 / 11 | 5 / 5 | 6 / 6 |
| Foam food container | 7 | 15 / 13 | 8 / 8 | 8 / 8 |
| Plastified paper bag | 8 | 0 / 0 | 0 / 0 | 0 / 0 |

Retained source-category counts are components of the per-class totals, not additional boxes to add to them. Raw counts are annotation metadata across all 1,500 records; frozen images are always excluded from retained counts.

### Strict waterfall and duplicate evidence

| Primary disposition (all 1,500 records) | Tier 1 | Tier 2 | Tier 3 |
| --- | --- | --- | --- |
| frozen_heldout | 130 | 130 | 130 |
| missing_or_unreadable | 0 | 0 | 0 |
| invalid_image_metadata | 0 | 0 | 0 |
| dimension_mismatch | 313 | 313 | 313 |
| ambiguous_exif_orientation | 9 | 9 | 9 |
| invalid_annotation | 10 | 10 | 10 |
| crowd_or_ignore_annotation | 0 | 0 | 0 |
| unlabeled_annotation | 182 | 182 | 182 |
| excluded_annotation | 519 | 471 | 161 |
| no_mapped_annotations | 0 | 0 | 110 |
| exact_duplicate_conflict | 0 | 0 | 0 |
| exact_duplicate_quarantine | 1 | 1 | 1 |
| duplicate_component_touches_heldout | 0 | 0 | 0 |
| training_candidate | 336 | 384 | 584 |

Detected exact duplicate group: [1173, 1305]. Near-duplicate candidates: [1081, 1110] and [1131, 1157]. None of their connected components touches frozen validation/test IDs; no additional training candidate is removed by the held-out-component check. The exact group still propagates annotation quarantine as described above. Frozen singleton references and all duplicate edges are retained in the JSON. Detected candidate absence is not proof that every semantic near-duplicate has been found.

## Proposed bounded 384/576 continuation (requires owner approval)

Recommend **Tier 1 first**, with **one epoch**, batch size 1, LR **2e-6**, AdamW weight decay 1e-4, clip norm 1.0, FP32, TF32 disabled and unchanged frozen backbones. Use a separately verified private Stage 5 high-resolution E1 state as the proposed warm-start; no checkpoint is promoted by this recommendation. The lower LR and one-epoch bound limit drift after expansion; their benefit is a hypothesis, not a measured result. Keep the existing 65-image validation and 65-image test memberships fixed. Test images remain unused.

Tier 2 should be a separately approved ablation after subtype semantics review. Tier 3 should wait for explicit acceptance of accessory-as-background behavior or a separately authorized ignore-region design. Do not bundle these changes into a single unmeasurable experiment. Do **not** authorize two epochs initially: a second equal-length epoch would double training exposures and should require review of official nine-specific raw metrics, per-class AP and cigarette coverage. Mapped-taxonomy AP is diagnostic and must not substitute for specific-class retention. No repeated test-set selection. Any expanded export/rights corpus record would be a new, separately approved artifact, leaving current exports intact.

Runtime projections use the recorded GTX 1660 Ti high-resolution run: 307 updates, mean audited step 0.922972 seconds, recorded total 427.098791 seconds, with approximately 143.746296 seconds of aggregate validation/checkpoint/loop overhead. They are arithmetic planning estimates, not new timing measurements. Formula per epoch: `(0.922972 * N + 143.746296) / 60` minutes. The overhead allocation is approximate; model loading or an extra baseline evaluation can add time, and more crowded images or I/O stalls may slow updates. Budget 1.5-2 times the point estimate.

| Proposed set | Updates / 1 epoch | 1-epoch point estimate | 2-epoch point estimate | 1-epoch planning budget |
| --- | --- | --- | --- | --- |
| tier1 | 336 | 7.6 min | 15.1 min | 11.3-15.1 min |
| tier2 | 384 | 8.3 min | 16.6 min | 12.5-16.6 min |
| tier3 | 584 | 11.4 min | 22.8 min | 17.1-22.8 min |

## Qualification and evidence

The audit blocks all held-out photograph opens and test.json reads; no model packages are imported. All hashed input metadata, taxonomy v2 and six core sources match before/after. The saved baseline train/validation/eligibility hashes match the preparation manifest. All proposed training sets exclude the exact frozen 65/65 ID lists. The source audit script, complete IDs, per-class counts, per-image fingerprints, rule outcomes and duplicate groups are saved in the private summary.

[Structured summary](../runs/training/stage5-data-expansion-audit-5412a13c0258467ba33a1f475cc4b65c/summary.json). [Guarded test results](../runs/training/stage5-data-expansion-audit-5412a13c0258467ba33a1f475cc4b65c/unit-results.json) and [qualification evidence](../runs/training/stage5-data-expansion-audit-5412a13c0258467ba33a1f475cc4b65c/qualification.json) are saved beside it. **299 guarded tests: 283 passed, 16 documented skips, zero failures/errors.** Skips cover 15 training-execution fixtures and one unavailable Windows symlink privilege. The eight new software fixtures cover accessory-only relaxation, unlabeled/unsupported exclusions, invalid/crowd/ignore accessories, sibling/Other plastic mapping, empty-after-filtering rejection, transitive near-duplicate leakage, frozen/blocked ID exclusion, and exact-duplicate quarantine propagation.

No export, training, inference, checkpoint promotion, commit or push was performed. Stop here for owner review of the candidate definitions and schedule.
