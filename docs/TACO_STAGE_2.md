# Fine-Tuning Preparation — Stage 2: Dataset Conversion and Splitting

Authorized by the owner in D-036, after representative TACO annotation review. This stage implements a strict-clean eight-class COCO preparation on the existing `feat/taco-harmonization` branch. Source files, detection configuration, inference pipeline/viewer, model revision, dependencies and three final BANTAI inference categories remain separate. No model is loaded or fine-tuned.

## Files and commands

| File | Purpose |
| --- | --- |
| [config/taco_preparation_policy.v1.json](../config/taco_preparation_policy.v1.json) | Version 1.0.0 strict eligibility rules, coordinate/duplicate policy, near-duplicate thresholds, default seed and split ratios. |
| [scripts/prepare_taco.py](../scripts/prepare_taco.py) | Preparation CLI: fresh source audit, whole-image eligibility, duplicate investigation, conversion, splits and reports. |
| [scripts/taco_preparation.py](../scripts/taco_preparation.py) | Eligibility, complete source traceability, COCO conversion, statistics, output gates and independent reconstruction. |
| [scripts/taco_splits.py](../scripts/taco_splits.py) | File/pixel fingerprints, candidate near-duplicate search and deterministic multi-label group assignment. These are photograph leakage groups, not spatial clustering of objects. |
| [scripts/validate_taco_prepared.py](../scripts/validate_taco_prepared.py) | Read-only independent validator against current source metadata and photographs. |
| [tests/test_taco_preparation.py](../tests/test_taco_preparation.py) | Synthetic conversion, exclusions, duplicate/conflict handling, deterministic grouping/splits, checksum tampering, source reconstruction and preservation checks. |
| [.gitignore](../.gitignore) | Adds `/exports/taco/`; generated exports and reports remain untracked. |

Use the existing Windows `.venv`; only standard-library modules and already-installed Pillow are needed. No package installation, model cache access, GPU or pycocotools is required. Run from the repository root:

```powershell
.venv/Scripts/python.exe scripts/prepare_taco.py --audit reports/taco/local-review --output exports/taco/my-strict-clean --seed 42
.venv/Scripts/python.exe scripts/validate_taco_prepared.py --dataset exports/taco/my-strict-clean
.venv/Scripts/python.exe -m unittest discover -s tests -q
```

Choose a **new** output directory each time; prior exports are never overwritten. Omitting `--output` creates `exports/taco/<uuid>/`. `--ratios .70 .15 .15` explicitly selects train/validation/test fractions; all three must be positive and sum to one. Relative paths resolve from the repository root. The default audit is `reports/taco/local-review/`; source/taxonomy hashes must match it, and eligibility is freshly reconstructed from the source, not trusted from editable JSONL review ledgers.

Preparation exits 0 after its export gates pass; exits 2 for source/settings/output/gate failures. Standalone validation exits 0 for a pass, 1 for reconstructed content mismatches, and 2 for input/integrity/schema failures. A preparation directory without the final `preparation_manifest.json` with `status=validated` is incomplete and must not be used.

## Strict eligibility and provenance

An image is retained only if it is readable, dimensionally aligned in the stored raster, and every recorded annotation is valid and maps to the eight approved object labels. An empty image is quarantined rather than manufactured into a negative. Any excluded category, `Unlabeled litter`, bad category/reference/box, unresolved duplicated annotation ID, crowd/ignore region or invalid area/crowd metadata quarantines the **whole image**. No annotation is silently dropped from a retained photograph, no box is clipped, no source ID is repaired and no COCO ignore semantics are assumed to work in a future trainer.

All 152 dimension-mismatched images are excluded even when EXIF could align their dimensions. Nine further dimensionally matching images have EXIF orientation 0 (eight images) or 3 (one image). The policy also holds these ambiguous cases rather than assuming an orientation at export. This is stricter than Stage 1's dimension-only conservative count; it is not a source-file correction or an approved training transform.

Exact byte/pixel duplicates are assigned together. If an exact peer is ineligible, quarantine propagates to clean peers to avoid false background. If otherwise clean exact peers disagree in mapped class/box annotations, all are quarantined as ambiguous. Original records remain intact in the eligibility/annotation ledgers. Safe original image IDs are retained; exported annotation IDs are unique and dense across all splits, derived from eligible source record order. New annotation IDs do not repair or include source records held for duplicate IDs.

Category IDs and array ordering exactly follow [the eight-class taxonomy](../config/taco_taxonomy.v1.json), 0 through 7. These are preparation IDs; compatibility with trainer background/token conventions still needs verification. COCO boxes preserve `[x,y,width,height]` values at full source precision. Area, segmentation, crowd and custom annotation metadata are preserved. Mask geometry is not evaluated as part of box detection preparation.

COCO images contain `id`, `file_name`, `width`, `height` and a `source_taco` provenance object. Original image metadata, including source license strings, is retained there; strings are not mislabeled as numeric COCO license IDs. Each annotation's `source_taco` object retains original annotation ID, record index, complete annotation and original category identity. All original categories/licenses/scene metadata also remain in `source_metadata.json`. There is no claim that licenses have been cleared.

Images are **referenced externally**, never copied. For an exported record, resolve its `file_name` under `preparation_manifest.json:image_root` (also `COCO info.image_root`), currently `../BANTAI-Datasets/TACO/`. A generic COCO loader needs that image root supplied explicitly. Its handling of EXIF, category IDs, crowd regions, captions/tokens and extra provenance fields must be verified before using a Grounding DINO trainer.

## Split method and leakage limitations

Default fractions are 70/15/15 and seed 42. Splits are by image groups, never annotations. Exact identity uses SHA-256 of bytes plus SHA-256 of decoded RGBA pixels and dimensions, catching re-encodings with exactly equal pixels. Resizing/orienting/cropping source photos is not performed.

Near-duplicate investigation compares unrotated 64-bit dHash and average hash, requiring both Hamming distances <=4 and relative aspect difference <=0.02. Eligible candidate pairs are coassigned to one split, including transitive links; no images/annotations are merged or synthesized. False positives can constrain class balance, while crops/rotations/viewpoint changes/less-similar scenes can escape these heuristics. Same-location or same-object leakage outside the detected candidates remains unresolved. The pair list is a review aid, not proof that all leakage has been eliminated.

The splitter chooses rare remaining labels first, uses image-level multi-label quotas and image-capacity targets, and assigns whole leakage groups with seeded tie breaking. Largest-remainder image targets and small-class goals are recorded. Group size and conflicting multi-label requirements can prevent exact ratios or perfect stratification. `class_counts.csv` separately reports objects, unique image records and unique exact photograph fingerprints per class/split. Counts for multiple labels can overlap on the same image.

## Generated artifacts

| Artifact | Purpose |
| --- | --- |
| `train.json`, `validation.json`, `test.json` | Eight-class COCO splits with external image references and full source traceability. |
| `preparation_manifest.json` | Versioned schema/policy/taxonomy, source hashes, seed/ratios/algorithm, artifact checksums, validation status and pending readiness requirements. Written last. |
| `policy_snapshot.json`, `taxonomy_snapshot.json` | Exact preparation settings, independent of later configuration edits. |
| `image_eligibility.jsonl` / `.csv` | Every source image record, inclusion/quarantine reasons, original counts, fingerprints, split/group assignment. |
| `annotation_decisions.jsonl` | Every original annotation, source category, image-level exclusion reasons and exported ID/split or null. |
| `statistics.json`, `class_counts.csv` | Source/strict/split counts, imbalance, unique-image coverage, target versus actual allocation and limitations. |
| `eligibility_waterfall.csv` | Primary-rule image/annotation/mapped-annotation losses and remaining totals. |
| `duplicate_review.json` | Exact groups/conflicts, near-duplicate candidates and final split groups. |
| `source_metadata.json` | Original top-level metadata, licenses, scene annotations and category records. |

Primary exclusion counts form an ordered waterfall and sum to the full loss; `reason_counts_overlapping` preserves all applicable reasons and must not be summed as distinct removals. Annotation counts at each step include all recorded classes, with separate mapped-object losses. The ledgers preserve every quarantined annotation rather than making it background.

## Actual Windows result

Final preparation command:

```powershell
.venv/Scripts/python.exe scripts/prepare_taco.py --audit reports/taco/local-review --output exports/taco/strict-clean-stage2-v1
.venv/Scripts/python.exe scripts/validate_taco_prepared.py --dataset exports/taco/strict-clean-stage2-v1
```

Source annotations SHA-256: `ac1ec605c9e1fcda767970de3457c69b4a43b62c02aca8c4c80f51d1b5ae8449`; taxonomy version 1.0.0, policy version 1.0.0, seed 42. The local export is approximately 8 MB of JSON/CSV; no photographs or generated artifacts are committed. A development output `exports/taco/strict-clean-v1/` was retained without overwrite; the final schema/result is `strict-clean-stage2-v1/`.

The strict-clean subset contains **315 images and 538 annotations**, versus Stage 1's 735 images/1,653 mapped usable annotations. It quarantines 1,185 of the 1,500 source image records. Excluding cluttered/mixed scenes sacrifices 1,115 previously usable mapped annotations and may strongly bias future evaluation toward simpler, fully mapped scenes. This subset should not be treated as representative of deployment images.

| Primary rule | Images removed | All annotations removed | Mapped annotations removed |
| --- | --- | --- | --- |
| Missing/unreadable | 252 | 846 | 389 |
| Invalid image metadata | 0 | 0 | 0 |
| Dimension mismatch | 152 | 521 | 241 |
| Additional ambiguous EXIF orientation | 9 | 60 | 30 |
| Invalid/ambiguous annotation | 8 | 35 | 14 |
| Crowd/ignore annotation | 0 | 0 | 0 |
| Unlabeled litter present | 167 | 1,156 | 480 |
| Other excluded category present | 597 | 1,628 | 597 |
| Empty/no mapped annotation | 0 | 0 | 0 |
| Exact duplicate conflict / dirty exact peer | 0 | 0 | 0 |
| **Retained** | **315** | **538** | **538** |

The dimension-mismatch step removes 521 recorded annotations, including an invalid record; Stage 1's 520 figure counted otherwise-valid annotations. This is a counting-scope difference. Missing/dimension/EXIF rules take precedence over later annotation reasons in the waterfall.

| Split | Images / unique exact photographs | Annotations |
| --- | --- | --- |
| Train | 221 / 221 | 397 |
| Validation | 47 / 47 | 75 |
| Test | 47 / 47 | 66 |

Actual image fractions are approximately 70.16/14.92/14.92%; object fractions differ because photos have different object densities. All eight labels occur in all three splits, without fabricated examples.

| Label | Total objects | Train objects / images | Validation objects / images | Test objects / images |
| --- | --- | --- | --- | --- |
| cigarette | 141 | 110 / 30 | 24 / 7 | 7 / 6 |
| plastic_bottle | 61 | 44 / 35 | 8 / 8 | 9 / 7 |
| plastic_bag | 13 | 9 / 9 | 2 / 2 | 2 / 2 |
| plastic_wrapper | 83 | 58 / 53 | 11 / 11 | 14 / 11 |
| plastic_film | 125 | 97 / 85 | 13 / 9 | 15 / 12 |
| drink_can | 54 | 39 / 31 | 7 / 7 | 8 / 7 |
| plastic_cup | 24 | 16 / 16 | 5 / 4 | 3 / 3 |
| styrofoam_piece | 37 | 24 / 20 | 5 / 5 | 8 / 4 |

**Limited evaluation coverage:** plastic bags have just two validation and two test images; plastic cups have four/three, and styrofoam pieces have four test images. These are insufficient for reliable class-wise conclusions and require owner review before training/evaluation acceptance. Presence in every split does not establish adequacy. Automated zero/one-image alerts did not trigger; they are severe-shortfall checks, not adequacy guarantees. The largest-to-smallest nonzero class object ratio is approximately 10.85 (141 cigarettes versus 13 bags). No oversampling, augmentation or weighting decision is made.

No exact byte/pixel duplicates were found among 1,248 readable photos. Two near-duplicate pairs were flagged:

- Image IDs **1081/1110**, `batch_6/000100.JPG` and `batch_6/000069.JPG`: both eligible and assigned to train together. Agent inspection found nearly identical bottle/vegetation views.
- Image IDs **1131/1157**, `batch_6/000047.JPG` and `batch_6/000019.JPG`: both quarantined for excluded annotations. Agent inspection found closely related jug/vegetation views.

These are agent visual observations, not owner confirmation of provenance. Broader scene-level duplicate review remains unresolved.

## Validation evidence and next boundary

The final local preparation exited 0. Its gates checked source-consistent COCO content, taxonomy order, unique annotation/image IDs, geometry, eligibility, deterministic assignments, exact-photo/split-group leakage and unchanged source hashes. Standalone validation re-decodes current sources, recomputes fingerprints/quarantine/splits, checks all artifact hashes, and compares all three exports, eligibility ledgers, annotation traceability and statistics against independent reconstruction. A checksum alone cannot certify a semantically incorrect export.

Standalone reconstruction exited **0**, returned `passed=true`, `errors=[]`, `recomputed_from_current_sources=true`, and the same 221/47/47 image and 397/75/66 annotation counts. It retained `training_ready=false` and the four pending readiness requirements. All 1,248 captured readable-image hashes and source metadata matched through reconstruction.

Software checks include whole-image mixed/unlabeled/invalid quarantine, byte/pixel duplicate coassignment, dirty-peer propagation, conflicting duplicate annotations, near-candidate grouping, image-level multi-label stratification, seed/order determinism, small-class alerts, COCO geometry/references, source preservation, identical generated files on repeated fixture runs, and rejection of a wrong source category even when artifact checksums are updated. Existing inference checks continue unchanged. No target model or weights are imported by the preparation utilities.

Final existing Windows `.venv` command `.venv/Scripts/python.exe -m unittest discover -s tests -q` ran **114 tests in 6.678 seconds: 113 passed, one existing Windows symlink-privilege skip**. Thirteen new Stage 2 tests passed, including policy safeguards. CLI help, compilation and `git diff --check` passed; documentation validation resolved 174 relative links and five strict JSON examples. All final export artifact checksums passed. Inference/viewer/audit utilities, the approved taxonomy and dependency files match the starting branch; the owner's detection configuration remains uncommitted.

The owner reports representative annotations were visually reviewed before Stage 2. That does not certify full annotation completeness, all scene relationships or downstream trainer behavior. COCO box/schema/source validation is only one readiness gate; the manifest explicitly retains `training_ready=false` and pending licensing, trainer compatibility, near-duplicate review and class-coverage acceptance.

Next actions require review of strict-clean bias/low coverage and unresolved licensing/trainer input conventions. Alternative ignore-region strategies remain proposals until the actual trainer is verified to honor them. Orientation/box/ID repairs, broader dataset retention, split redesign and any Grounding DINO fine-tuning are outside this completed implementation. No dependency/model/inference/viewer changes, source-photo copies, downloads or merge are performed.
