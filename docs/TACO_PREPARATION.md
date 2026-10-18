# TACO harmonization and annotation visualization

Stage 1 history: the owner subsequently reported representative annotation review and authorized strict-clean conversion/splitting in D-036. See [Stage 2 commands and evidence](TACO_STAGE_2.md). The earlier no-export/no-splits boundary below is superseded for that implementation; fine-tuning remains unauthorized. These audit/viewer utilities remain unchanged.

The owner authorized this preparation stage on 2026-10-18 (D-035): audit the existing external COCO dataset and visualize its annotations under the eight approved object labels. It creates review artifacts, not training examples or splits. The current Grounding DINO model, detection configuration, inference logic, three inference categories and results viewer are independent and unchanged.

## Files and requirements

| File | Purpose |
| --- | --- |
| [config/taco_taxonomy.v1.json](../config/taco_taxonomy.v1.json) | Version 1.0.0 mapping of 11 exact TACO category names to eight BANTAI labels. Original IDs/names are retained; label IDs 0-7 identify this preparation taxonomy, not a finalized trainer encoding. |
| [scripts/taco_dataset.py](../scripts/taco_dataset.py) | Read-only COCO validation, image hashes/readability/dimension checks, taxonomy lookup, report writing/reloading. |
| [scripts/audit_taco.py](../scripts/audit_taco.py) | Audit CLI with JSON/CSV review output and new-directory protection. |
| [scripts/view_taco.py](../scripts/view_taco.py) | Separate local Tk/TkAgg annotation viewer with deterministic per-class sampling, class selector, Next/Previous and zoom/pan. |
| [tests/test_taco.py](../tests/test_taco.py) | Synthetic mapping, COCO, malformed-box, missing/corrupt image, identity, EXIF and drawing checks. |
| [tests/windows_taco_check.py](../tests/windows_taco_check.py) | Optional native checks against a real local audit; auto-closes and checks source hashes. |
| [.gitignore](../.gitignore) | Adds `/reports/taco/`, `/data/taco/`, `/BANTAI-Datasets/` for reports and accidental in-repository dataset copies. The actual sibling dataset already lies outside Git. |

Use the existing verified Windows Python 3.11 `.venv`. The audit uses Python's standard library and already-installed Pillow; the viewer additionally uses already-installed Matplotlib and Python's Tk. No pycocotools, pandas, dataset downloader, model import, GPU, new dependency or package reinstall is needed. Existing dependency reproduction references remain in [README.md](../README.md); an installation is not performed by these tools.

Default dataset: `../BANTAI-Datasets/TACO/`; default source: its `annotations.json`. Relative paths resolve from the repository root even when invoked from another shell directory. `--annotations` overrides a path relative to the repository root; it is not relative to the dataset folder. All source access is read only.

## Audit commands and artifacts

```powershell
.venv/Scripts/python.exe scripts/audit_taco.py
.venv/Scripts/python.exe scripts/audit_taco.py --dataset-root ../BANTAI-Datasets/TACO --output reports/taco/my-review
```

The default creates a new UUID folder under ignored `reports/taco/`. Explicit output directories must be new and cannot overlap the external dataset. Existing reports are never overwritten. Exit 0 means no detected image/annotation/structural issues; **exit 1 means the audit completed and wrote reports with issues**, including known missing images; exit 2 means an input/configuration/write failure. A partial report after write failure must not be treated as a completed audit.

Each report folder contains:

| Artifact | Contents |
| --- | --- |
| `audit.json` | Concise counts, per-label object/unique-image counts, all original class counts and exclusion handling, structural issues, source/taxonomy hashes. |
| `class_counts.csv` | Eight proposed classes: totals, annotations attached to readable images, conservative usable objects, unique usable images and blocked/invalid counts. |
| `original_classes.csv` | All source categories, source IDs/names, proposed label or explicit exclusion state, and usable counts. |
| `image_issues.csv` | Missing, unsafe, unreadable and dimension-mismatched image records. |
| `annotation_issues.csv` | Unusable annotation indexes/IDs, validation and image-availability/alignment issues. |
| `images.jsonl` | Every original COCO image record plus path, image hash, status and dimensions/EXIF diagnostics. |
| `annotation_review.jsonl` | Every original annotation, unmodified metadata/segmentation/area/iscrowd/custom fields, original category identity, proposed label and explicit handling/validity state. Source list index disambiguates duplicate annotation IDs. |
| `taxonomy_snapshot.json` | Exact taxonomy values used for the audit. |
| `source_metadata.json` | Other original COCO top-level metadata, all original categories, licenses and scene annotations. |

These are review ledgers, not a filtered COCO training export. Excluded annotations remain in the ledgers and the viewer; absence of a mapped label never makes an object an implicit negative/background example. Image counts per class overlap and must not be summed into a dataset image total. Object counts mean annotated instances, not a claim that all real objects are annotated.

Validation checks unique integer IDs, category/image references, safe local image paths, successful full image decoding, declared versus actual dimensions, and finite COCO `[x,y,width,height]` boxes with positive sizes, nonnegative origins and endpoints within declared dimensions. Boxes are never silently clipped. Duplicate IDs are held as ambiguous rather than selecting one record. A missing image blocks usability but does not by itself make its annotation geometrically invalid. Nonstandard JSON constants and duplicate JSON keys fail source loading. Segmentation, area and crowd metadata are preserved; this stage does not validate mask geometry or establish crowd/loss policy.

## Viewer commands

Reuse a completed audit to avoid decoding the entire dataset on every launch:

```powershell
.venv/Scripts/python.exe scripts/view_taco.py --report reports/taco/review-v1 --sample-per-class 1
.venv/Scripts/python.exe scripts/view_taco.py --report reports/taco/review-v1 --classes cigarette plastic_bottle plastic_bag --sample-per-class 3 --seed 42
.venv/Scripts/python.exe scripts/view_taco.py --report reports/taco/review-v1 --classes excluded unlabeled --sample-per-class 3
.venv/Scripts/python.exe scripts/view_taco.py --report reports/taco/review-v1 --image-id 274
.venv/Scripts/python.exe scripts/view_taco.py --report reports/taco/review-v1 --classes plastic_film --sample-per-class 2 --list
```

`--report` uses its recorded dataset/taxonomy sources, verifies their hashes and checks each displayed image's hash before decoding. Do not combine it with source overrides. Without `--report`, the viewer performs a fresh in-memory audit using the same source arguments as the audit CLI. It does not write a report or any image. Repeat `--image-id` to inspect several specific readable/aligned images, including images without annotations. Sampling is deterministic per selected class and seed, with shared images deduplicated; it provides review examples, not statistically representative sampling or a train/test split.

Class selection controls which images are sampled; **all annotations on each selected image remain visible**. Labels show annotation ID, original TACO name and proposed BANTAI label/explicit exclusion state. Green solid boxes indicate selected mapped classes, blue other mapped classes, orange dashed excluded classes, purple dotted `Unlabeled litter`, red invalid annotation metadata when its geometry is drawable. Invalid boxes stay listed in the selectable, horizontally/vertically scrollable details table and are not drawn or clipped. Color is supplemented by line style and text. Crowded labels can overlap; use zoom/pan and the complete table. The toolbar omits Save; no annotated copies or temporary image files are generated. Close the window to release the figure/canvas and image resources.

## Coordinate orientation requires review

The default uses the **stored raster without EXIF transpose**, matching only records whose declared dimensions agree with that raster. This deliberately avoids importing the inference pipeline's automatic orientation rules into COCO annotation handling. Audit counts called `usable` use this conservative policy and require a valid annotation, readable image and no image-dimension issue.

The downloaded set contains 152 dimension mismatches. All 152 declared dimensions match an EXIF-oriented preview. To inspect a specific example without changing any source bytes or annotation coordinates:

```powershell
.venv/Scripts/python.exe scripts/view_taco.py --report reports/taco/review-v1 --image-id 228 --frame exif
```

EXIF mode applies Pillow's orientation only in memory and requires the resulting dimensions to match COCO metadata. Annotation coordinates are not transformed; this is a review of the hypothesis that the annotations already describe that orientation. A dimension match, including a 180-degree rotation with unchanged dimensions, does not prove every box is aligned. The report counts remain unchanged by this viewer option. Approve and validate a coordinate policy before converting data for training.

## Actual local audit — 2026-10-18

Base main `5351d57`, branch `feat/taco-harmonization`. Command:

```powershell
.venv/Scripts/python.exe scripts/audit_taco.py --output reports/taco/review-v1
```

The command exited **1**, correctly indicating a completed audit with issues. Original annotation SHA-256: `ac1ec605c9e1fcda767970de3457c69b4a43b62c02aca8c4c80f51d1b5ae8449`. Taxonomy version: 1.0.0. Exact original metadata and image hashes are in the ignored report.

- 1,500 source image records; **1,248 readable, 252 missing, zero unreadable/corrupted**. This reproduces the supplied image-availability observations.
- 4,784 source annotations across 60 source classes. **3,938 annotations are attached to readable images**, reproducing the supplied total at the availability level.
- A stricter check holds 520 otherwise-valid annotations on dimension-mismatched readable images and nine invalid annotations on readable images, leaving **3,409 usable annotations across all classes** under the stored-raster policy. Nothing is discarded from the review ledgers.
- Eleven invalid annotations in the entire source: both occurrences of duplicated IDs 309 and 4040 (four records), plus seven negative-origin boxes. Two of these eleven are attached to missing images. No unknown category or image references were observed in this source. All 11 required original category names are present.
- **1,653 usable mapped annotations in 735 unique images**; 1,756 usable excluded annotations. There are 49 original categories outside the mapping. **413 mapped-class images also contain usable excluded annotations**, making implicit-background handling unsafe without a defined policy.
- `Unlabeled litter` remains explicit: 517 source annotations, of which 362 are conservatively usable in 170 images.

| Label | Source annotation total | Attached to readable images | Conservative usable objects | Unique usable images |
| --- | --- | --- | --- | --- |
| `cigarette` | 667 | 539 | 478 | 155 |
| `plastic_bottle` | 335 | 275 | 246 | 185 |
| `plastic_bag` | 92 | 74 | 62 | 49 |
| `plastic_wrapper` | 299 | 261 | 192 | 151 |
| `plastic_film` | 451 | 381 | 344 | 231 |
| `drink_can` | 229 | 186 | 164 | 110 |
| `plastic_cup` | 104 | 93 | 84 | 71 |
| `styrofoam_piece` | 112 | 91 | 83 | 59 |

Negative origins are small but still explicit: annotation IDs 371, 387, 707, 859, 1621, 1777 and 2186. Whether clipping is appropriate requires a documented later decision. Duplicate IDs are preserved with source indexes rather than silently renumbered.

## Verification and remaining decisions

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -q
.venv/Scripts/python.exe tests/windows_taco_check.py --report reports/taco/review-v1
.venv/Scripts/python.exe tests/windows_taco_check.py --report reports/taco/review-v1 --frame exif --image-id 228
```

Actual existing Windows `.venv` results: **101 tests, 100 passed, one existing Windows symlink-privilege skip**, 11.552 seconds on the first full run and 6.112 seconds on the subsequent check. Nine new synthetic tests cover all eleven mappings, preserved COCO metadata, missing/corrupt/unsafe files, malformed boxes/references, duplicate IDs, exact xywh drawing, excluded/unlabeled styles, deterministic selection, report reuse, source identity and explicit EXIF preview. These fixtures do not establish annotation quality. CLI help, compilation and whitespace checks passed; 159 relative Markdown links and five strict JSON examples validated. Protected inference scripts and dependency files match the starting main; the owner's detection configuration remains uncommitted.

Native Tk check passed over ten downloaded source images, exercising one-per-class sampling, Next navigation, class selection, excluded/unlabeled boxes, selectable details, exact rectangle geometry, zoom/Home, figure cleanup and normal closure. Explicit EXIF-mode native check also passed over three source images. No model modules or network connections occurred. Source hashes remained unchanged. An independent final hash pass verified **all 1,248 audited source images unchanged**, and the original annotation hash matched.

Agent in-memory visual inspection: image 18 shows a can with `Drink can -> drink_can`; image 274 shows `Plastic film -> plastic_film` alongside an explicit `Unlabeled litter` box; image 228's EXIF preview places the excluded `Plastic glooves` box on the visible glove. The original spelling is preserved. These limited visual observations are separate from human acceptance, which remains pending. They do not certify the full dataset's box/category quality.

Before the next stage, the owner must decide how to handle excluded and unlabeled annotations in mixed images, approve an orientation/coordinate policy, resolve duplicate IDs and negative boxes, and assess sampled annotation quality/class imbalance. Later split design, source/license review for intended use, model loss/prompt encoding and actual fine-tuning need their own preparation/authorization. No final train/validation/test split, source correction, training export, download or fine-tuning occurred here.
