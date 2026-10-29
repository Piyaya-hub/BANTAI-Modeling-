# D-069: Stage 5 paper-category metadata audit and proposal

Date: 2026-10-26. **Proposal only; stopped for owner approval.** No export, taxonomy v2 file, model loading, GPU initialization, training, inference, photo read/decode or test.json access occurred. Existing datasets, 64 checkpoint bundles, six core SOURCES, taxonomy v1 and rights records are unchanged. The audit is on `feat/stage5-paper-category`, created and pushed from Stage 4 safe-point commit `9c3b6777724a799569c7e485815171ef4f3a878a`. PR #18 / `feat/stage4-real-image-training` remains unmerged. Phase 0 had already been committed/pushed; re-verification passed 278 tests (262 passed, 16 documented skips, zero failures/errors), relative link targets and diff checks. No redundant safe-point commit was needed.

The fourteen proposed source categories contain **497 boxes on 347 distinct raw images**. Applying the existing strict-clean rules with only the proposed paper mapping added retains **165 paper boxes on 122 images**. None overlap the existing **315** eligible images. A prospective nine-class corpus would have **437 images and 809 total annotations**: the original 538, 165 new paper boxes, and 106 original-eight-class boxes on the newly eligible images. These are conditional metadata counts, not a newly validated dataset or model result.

D-061 `blend_350_442_2575` remains the best measured official eight-class Stage 4 validation state (AP 0.265563); its separate raw candidate coverage is 38/75. It is not promoted or wired into `scripts/detect_images.py`. This audit supplies no nine-class accuracy or AP.

## Evidence, method and limits

The read-only [audit utility](../scripts/taco_paper_audit.py) reads raw TACO `annotations.json`, the saved raw image/annotation audit and the strict-clean-stage2-v1 eligibility, annotation-decision, policy and duplicate ledgers. It checks the saved metadata hashes against the preparation manifest. Current raw JSON has previously documented serialization-only drift: default JSON serialization reconstructs the exact originally recorded source hash. The parsed image/annotation arrays match the recorded audit originals exactly.

The original [strict-clean planner](../scripts/taco_preparation.py) is reused **in memory**, replacing photo fingerprint computation with the already recorded fingerprints. First, it reproduces the complete existing image-eligibility and annotation-decision ledgers exactly, including original split assignments. Then only the fourteen source names are provisionally mapped to paper, and the same eligibility/duplicate rules are recomputed. Nothing is exported or written into a dataset directory. Counts use original source boxes without repair, clipping, dropping excluded annotations or inventing background.

Readability, raster dimensions, EXIF and file/pixel/perceptual fingerprints are **cached historical image evidence**. No photo bytes are opened, decoded or newly hashed here. Thus this is not fresh proof that the photos remain present or unchanged; a future owner-authorized export would require the normal fresh photo validation. Exact-duplicate groups are empty in the cached corpus; two global near-duplicate candidates are reused. The 122 newly eligible images are 122 singleton leakage groups, with no group forcing one into an existing split.

The audit guard rejects photo/checkpoint files and any file named test.json. It allows writes only to a new system-temporary evidence directory. torch and transformers are never imported. Nineteen input metadata/core-source hashes are identical before and after the audit. The frozen test membership is read from the eligibility ledger, not test.json; no held-out photos or predictions are inspected. This metadata-level split planning does not authorize later test evaluation or training on test images.

### Input anchors

| Metadata input | SHA-256 |
| --- | --- |
| Raw annotations, current bytes | 6dce30703959153e3207a7174c3326585d9593438f7af855689a0e2000ac50b9 |
| Raw annotations, exact original reconstruction | ac1ec605c9e1fcda767970de3457c69b4a43b62c02aca8c4c80f51d1b5ae8449 |
| preparation_manifest.json | a45a3131f7167c9ce9c8ca5e99eb547e2eedf7f67b73e269c3af4b877f5c9265 |
| image_eligibility.jsonl | b2839015178cbefaee3d878ae83344667511ba84c99237d515e0eba7a5dfdadb |
| annotation_decisions.jsonl | cf1066492f46b1f008db09dbab0398760aeba5d3a5c0035c7b6aa6ca219993d7 |
| images.jsonl | fb9d7d867a3aa7ae1d7f52310554c4b33a62330dc1c2990a74a9c02e372a3de2 |
| annotation_review.jsonl | 428a2e04c98e5ea843b77dfd53e562181e069a0cfec800b9ae746d830e5aba67 |
| Audit utility | 49d816c10267ed6bf04420acbc6eebc84509a2449c03712bf2df41fc4eb20a5c |

## Source-category support

Raw columns cover all source records, including unreadable/ineligible images. Retained columns apply the proposed nine-class mapping with the unchanged strict-clean policy. Image counts by category overlap and must not be summed; the union is 347 raw / 122 retained.

| Raw source category | ID | Raw boxes | Raw images | Retained boxes | Retained images |
| --- | --- | --- | --- | --- | --- |
| Normal paper | 33 | 82 | 68 | 20 | 18 |
| Magazine paper | 30 | 12 | 5 | 10 | 3 |
| Wrapping paper | 32 | 12 | 11 | 2 | 2 |
| Tissues | 31 | 42 | 36 | 9 | 7 |
| Paper bag | 34 | 27 | 23 | 8 | 7 |
| Paper cup | 20 | 67 | 62 | 13 | 12 |
| Corrugated carton | 17 | 64 | 41 | 43 | 26 |
| Other carton | 14 | 93 | 79 | 37 | 33 |
| Drink carton | 16 | 45 | 41 | 4 | 4 |
| Meal carton | 18 | 30 | 27 | 14 | 14 |
| Egg carton | 15 | 11 | 9 | 3 | 3 |
| Pizza box | 19 | 3 | 3 | 1 | 1 |
| Paper straw | 56 | 4 | 4 | 1 | 1 |
| Toilet tube | 13 | 5 | 3 | 0 | 0 |

## Proposed single paper class

Append `id: 8, name: paper` to the existing IDs 0..7, mapping exactly the fourteen names in the table above. IDs/names/mappings of the original eight classes remain unchanged. The actual `config/taco_taxonomy.v1.json` is not edited, and no v2 file is created in this phase.

The intended class is a broad **paper/cardboard object category**, including tissues, paper packaging, tubes and cartons. Paper cups and drink cartons can contain coatings/composite layers; this mapping follows source object labels and does not assert pure material composition, recyclability or visual certainty. `Plastified paper bag` is not included; neither are generic plastic, foam, unlabeled or unknown categories. Any such annotation still quarantines its entire image. The broader label definition and its coated-packaging boundary require owner approval.

Toilet tube has zero retained support; Pizza box and Paper straw have one retained box each. The proposed class cannot establish subtype-level performance for unsupported/rare source types. Retained support is biased toward cartons; no classification quality can be inferred from counts alone.

Under the **unchanged existing eight-class mapping**, paper support remains zero and no paper-bearing image enters the existing 315. The 165/122 result requires the proposed mapping; it is not a finding that paper was secretly part of Stage 4.

## Strict-clean waterfall

Starting paper cohort: 347 images / 497 paper boxes. Whole-image quarantine removes all paper boxes on a rejected image even if the paper boxes themselves are valid. The first applicable rule is counted once, using the exact original priority order; rows with zero removals are retained to show that no safeguard was dropped.

| Rule | Paper images removed | Paper boxes removed | Images remaining | Paper boxes remaining |
| --- | --- | --- | --- | --- |
| missing_or_unreadable | 52 | 64 | 295 | 433 |
| invalid_image_metadata | 0 | 0 | 295 | 433 |
| dimension_mismatch | 37 | 55 | 258 | 378 |
| ambiguous_exif_orientation | 1 | 2 | 257 | 376 |
| invalid_annotation | 3 | 9 | 254 | 367 |
| crowd_or_ignore_annotation | 0 | 0 | 254 | 367 |
| unlabeled_annotation | 43 | 66 | 211 | 301 |
| excluded_annotation | 89 | 136 | 122 | 165 |
| no_mapped_annotations | 0 | 0 | 122 | 165 |
| exact_duplicate_conflict | 0 | 0 | 122 | 165 |
| exact_duplicate_quarantine | 0 | 0 | 122 | 165 |

Before relaxing only the category mapping, the original eight-class planner reproduces all existing decisions exactly. In the same paper cohort, its excluded_annotation stage removes all remaining 211 images / 301 paper boxes, leaving zero. With the proposed paper mapping, that stage removes 89 images / 136 boxes and leaves 122 /165. Invalid annotation checks, crowd/ignore quarantine, unlabeled handling, EXIF/raster rules and duplicate rules stay unchanged.

### Overlapping excluded-annotation flags

Across all 347 raw paper-bearing images, **72** also contain `Unlabeled litter`; **160** contain at least one other category excluded even after the proposed mapping; **43** contain both, so **189** contain either. Those189 carry 291 paper boxes. These flags overlap and include images already lost to earlier physical/annotation rules. They must not be confused with or added to the sequential waterfall's 43 unlabeled and 89 excluded-image removals. No retained paper image contains an unlabeled or still-excluded annotation.

| Still-excluded source category on paper-bearing images | Boxes | Images |
| --- | --- | --- |
| Aluminium foil | 26 | 15 |
| Aluminium blister pack | 2 | 2 |
| Glass bottle | 23 | 16 |
| Plastic bottle cap | 43 | 30 |
| Metal bottle cap | 21 | 11 |
| Broken glass | 16 | 3 |
| Food Can | 9 | 7 |
| Aerosol | 3 | 3 |
| Foam cup | 4 | 3 |
| Glass cup | 4 | 2 |
| Food waste | 3 | 2 |
| Glass jar | 1 | 1 |
| Plastic lid | 33 | 27 |
| Metal lid | 5 | 3 |
| Other plastic | 53 | 31 |
| Polypropylene bag | 1 | 1 |
| Spread tub | 4 | 4 |
| Disposable food container | 8 | 7 |
| Foam food container | 2 | 2 |
| Other plastic container | 3 | 3 |
| Plastic glooves | 1 | 1 |
| Plastic utensils | 12 | 9 |
| Pop tab | 20 | 16 |
| Rope & strings | 4 | 4 |
| Scrap metal | 3 | 2 |
| Shoe | 3 | 3 |
| Plastic straw | 78 | 40 |
| Unlabeled litter | 203 | 72 |

## Frozen splits and append-only proposal

**Preserve the original 221 training, 47 validation and 47 test image memberships and their current files/annotations unchanged.** New IDs are disjoint from all 315 originals. Use cached exact/near-duplicate groups as hard coassignment constraints; a new group touching an old split would be anchored there, and any conflict would block the proposal. No such anchoring/conflict occurs for the 122 new singleton groups.

For a concrete proposal, apply the existing seeded rare-label group allocator (seed42,70/15/15) **only to the new groups**, keeping all original memberships locked. This yields 86 new training, 18 new validation and 18 new test images. This is a proposed paper annex, not a replacement of the two original 47-image holdouts. Future versioned manifests could reference both frozen legacy sets and the separately appended sets; current files would remain byte-for-byte unchanged.

| Split | Frozen original images | Appended paper images | Combined future view | Paper boxes |
| --- | --- | --- | --- | --- |
| train | 221 | 86 | 307 | 120 |
| validation | 47 | 18 | 65 | 20 |
| test | 47 | 18 | 65 | 25 |

Frozen original membership anchors were derived from image_eligibility.jsonl without opening test.json:

| Original split | Sorted image-ID list SHA-256 |
| --- | --- |
| train | d4306a68f5598d85347d4ef9ae77cb34f148e67c5a4fe9c64347e221d186ee81 |
| validation | 7ac7e42f4503cbc864f5a871df5f25e0b032f25e6a4e7e0bc752383fcdd68952 |
| test | d16b075903dd87bfc551054f7a270f3808b67e4013482c764b00c0d6a40efb49 |

### Paper subtype distribution in this candidate allocation

| Source type | Train boxes/images | New validation boxes/images | New test boxes/images |
| --- | --- | --- | --- |
| Normal paper | 12/10 | 4/4 | 4/4 |
| Magazine paper | 7/1 | 1/1 | 2/1 |
| Wrapping paper | 1/1 | 0/0 | 1/1 |
| Tissues | 6/5 | 3/2 | 0/0 |
| Paper bag | 8/7 | 0/0 | 0/0 |
| Paper cup | 12/11 | 1/1 | 0/0 |
| Corrugated carton | 30/18 | 3/3 | 10/5 |
| Other carton | 28/24 | 4/4 | 5/5 |
| Drink carton | 3/3 | 0/0 | 1/1 |
| Meal carton | 9/9 | 4/4 | 1/1 |
| Egg carton | 3/3 | 0/0 | 0/0 |
| Pizza box | 1/1 | 0/0 | 0/0 |
| Paper straw | 0/0 | 0/0 | 1/1 |
| Toilet tube | 0/0 | 0/0 | 0/0 |

This allocator balances the nine mapped classes, not all fourteen paper subtypes. Some rare source types are absent from individual holdouts; the only Paper straw instance goes to the proposed new test annex, while Toilet tube has no support anywhere. This is an explicit limitation for owner review, not a claim that the allocation is optimal or that every subtype can be learned/evaluated. No allocation was selected using model predictions or test performance.

### Validation comparability

- Keep a **legacy benchmark view** on the same 47 validation images /75 GTs with the original eight-class prompt, decoder, no-NMS COCO v2 settings and class set. Future nine-class software would need a versioned legacy-evaluation path; the existing eight-label adapter cannot simply consume a ninth label. A changed prompt or nine-way query argmax can alter all predictions, so merely dropping paper results afterward does not guarantee the same protocol.
- Report a **paper annex view** on 18 new validation images /20 paper boxes, plus every other retained annotation on those images. Never discard the other 106 original-class boxes across the 122 new images as background. The new test annex would have 18 images /25 paper boxes and remains unevaluated.
- If a combined 65-image nine-class validation view is used, it changes both image distribution and category averaging. Its aggregate AP is not directly comparable to Stage 4 AP 0.265563. Paper AP on the original 47 has zero annotated support and is undefined, not zero performance and not an extra denominator in an eight-class average. Keep legacy, annex and combined results separate.
- If "unchanged47/47" means the **total evaluation sets must remain exactly 47 each**, the conservative alternative is to append all 122 images to training: 343 train /47 validation /47 test, with 165/0/0 paper boxes. This preserves the old totals but provides no paper validation or test benchmark. Owner approval must choose the frozen-core-plus-annex interpretation or this train-only alternative; neither is executed now.

## Files a future nine-class implementation would need

The following is an impact proposal only. None of these taxonomy/model/data/rights files is edited in this phase. Prefer isolated v2/Stage 5 modules so the six frozen Stage 4 SOURCES and all saved replay choices remain reproducible.

| Area | Existing dependency / constraint | Proposed future file or action |
| --- | --- | --- |
| Taxonomy | config/taco_taxonomy.v1.json defines IDs 0..7 | Create config/taco_taxonomy.v2.json as a NEW file, append paper ID 8; preserve v1. The generic taxonomy schema may remain compatible with version2.0.0. |
| Prompt and labels | scripts/training_smoke.py owns LABELS/PROMPT and eight-label checks | Create scripts/taco_training_labels_v2.py (or training_smoke_v2.py); append paper to the eight-label prompt, preserve order and validate nine ordinary spans. Do not edit the frozen original. |
| Adapter | scripts/training_adapter.py imports fixed labels/prompt and binds token IDs/source hashes | Create scripts/training_adapter_v2.py with nine-label checks and a distinct adapter manifest/version; keep native architecture/loss contracts. |
| Loader and dataset context | scripts/taco_training_data.py binds taxonomy v1 and category array | Create scripts/taco_training_data_v2.py for a new versioned export and locked/annex membership; no old split-file edits. |
| Trainer | scripts/train_taco.py uses the original loader/adapter/context | Create scripts/train_taco_stage5.py or equivalent isolated runner. Explicitly distinguish warm-start weights from an exact optimizer/RNG resume across changed prompt/data contracts. No run authorized here. |
| Evaluator | scripts/taco_evaluation.py has range(8), fixed label maps and COCO v2 | Create scripts/taco_evaluation_v3.py with explicit nine categories and separate protocol plus a legacy-eight-class path; never rewrite official Stage 4 results. |
| Checkpoints | scripts/training_checkpoint.py verifies complete context equality | Generic save/verify may be reusable unchanged; new context must bind taxonomy/prompt/adapter/splits. Existing64 bundles must not be rewritten or mislabeled nine-class. |
| Preparation/export | scripts/prepare_taco.py and taco_preparation.py already use taxonomy metadata | Future fresh audit with v2 mapping, same strict rules and a NEW export directory/append-only manifest. Review/qualify any frozen-split override separately. No export now. |
| Rights/evidence | work/stage4-real/rights.json covers the exact original train/validation snapshot | Create a new private Stage 5 rights record and exact corpus/basis assessment, e.g. work/stage5-paper/rights.json plus docs/TACO_STAGE_5_ACADEMIC_USE.md. Existing declaration does not automatically cover the extra images; workflow approval is not rightsholder permission or definitive legal clearance. |
| GUI/diagnostics | taco_error_analysis.py, taco_material_hierarchy.py and saved GUI helpers assume eight classes | Add versioned nine-class replay/label/color/mapping handling while preserving all old choices. Decide paper treatment in material/polymer views explicitly; do not reuse their old 4/3-category numbers as nine-class benchmarks. |
| Qualification | Existing tests bind eight labels/protocol and Stage 4 sources | Add future nine-label span/loader/adapter/evaluator/rights/legacy-compatibility and split-lock tests before any separately authorized model run. |
| Baseline inference | scripts/detect_images.py, scripts/inference.py and config/detection.json | No change required or authorized by this audit; no training checkpoint is promoted or wired into baseline inference. |

## Reproduction, approval and stop

Run the audit utility from the repository root:

```powershell
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_paper_audit.py
```

It prints the conditional counts and a new system-temporary evidence directory, including full input SHA-256s and the proposed new-ID assignments. No proposed dataset, taxonomy, checkpoint or rights record is written. The saved baseline planner reproduction and all input-hash preservation assertions passed. Phase 0 guarded software verification had 262 passes and 16 documented skips, with no model/GPU execution. Public Markdown links and diff whitespace are checked after writing this report.

**STOP for owner approval of the fourteen-name paper definition and the split interpretation/allocation.** A future fresh photo audit/export, changed lawful-use/corpus record, nine-class implementation, checkpoint warm start and any model execution require their own explicit scope approval. No export or training follows automatically. This report and metadata-only utility are local changes on the Stage 5 branch, not a merged or promoted model artifact.
