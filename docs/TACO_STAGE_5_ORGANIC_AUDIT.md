# D-080: TACO Organic / Food waste eligibility audit

Owner-authorized, read-only CPU audit on `feat/stage5-paper-category`. No training,
detector inference, test JSON read, or held-out photo access occurred. The audit
read raw `annotations.json`, reused cached split membership and 1,500-image
physical/fingerprint metadata, then freshly decoded only the seven Food waste
images after confirming none belongs to either frozen split. All seven fresh file
hashes, dimensions, and EXIF values matched the cached physical ledger.

## Source category and raw support

Raw TACO contains one category matching the organic/biodegradable/food-waste name
search: **Food waste** (source category ID 25), with **8 boxes on 7 images**. The
name search also checked `organic`, `biodegradable`, `compost`, `food scrap`,
`fruit waste`, and `vegetable waste`; no other category matched. None of the 7
images intersects the frozen 65 validation or 65 test IDs. They have no existing
split assignment because the current taxonomy excludes Food waste; eligible
records below are proposed training additions only. The fixed validation and test
splits gain zero Organic boxes.

The new class is a hypothetical ID 9, `food_waste` -> BANTAI `Organic`; this audit
does not edit taxonomy v2 or create an export.

## Physical and per-image ledger

All rasters have valid declared/actual dimensions, are readable, and have EXIF
orientation 1. Every Food waste box passes the existing xywh bounds checks and
source-area / crowd validity checks. Duplicate components use the D-073 exact and
near-duplicate policy with cached fingerprints for the 1,500-record corpus and
fresh fingerprints for these seven images. All seven are singleton components;
none is an exact duplicate, near candidate, or component touching a frozen ID.
Resized box sizes are pre-square-pad dimensions at `shortest_edge=384`,
`longest_edge=576`; square padding does not change width/height.

| Image | File | Food waste annotation IDs | Other annotations on image | Frozen split | D-073 cached / fresh raster and EXIF | Duplicate component |
| ---: | --- | --- | --- | --- | --- | --- |
| 68 | `batch_1/000090.JPG` | 163 | Food Can | none | valid; 2448x3264; orientation 1; hash matches | [68] |
| 72 | `batch_1/000094.JPG` | 171 | Plastic film | none | valid; 2448x3264; orientation 1; hash matches | [72] |
| 81 | `batch_1/000102.JPG` | 203, 204 | Corrugated carton x3; Unlabeled litter | none | valid; 2448x3264; orientation 1; hash matches | [81] |
| 92 | `batch_1/000119.JPG` | 244 | none | none | valid; 2448x3264; orientation 1; hash matches | [92] |
| 704 | `batch_2/000022.JPG` | 2237 | Plastic bottle cap; Unlabeled litter; Normal paper | none | valid; 2448x3264; orientation 1; hash matches | [704] |
| 841 | `batch_3/IMG_4916.JPG` | 2611 | Disposable food container | none | valid; 2448x3264; orientation 1; hash matches | [841] |
| 924 | `batch_4/000054.JPG` | 2835 | Other plastic x3 | none | valid; 3264x2448; orientation 1; hash matches | [924] |

| Annotation | Image | Original bbox `[x,y,w,h]` px | Bbox area px² | Source polygon area px² | 384/576 resized bbox `w x h` px | Tier 1 | Tier 2 | Tier 3 |
| ---: | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 163 | 68 | [1118,1008,386,380] | 146,680 | 47,527.5 | 60.5 x 59.6 | blocked: Food Can | train candidate | train candidate |
| 171 | 72 | [665,936,1057,1023] | 1,081,311 | 704,098 | 165.8 x 160.5 | train candidate | train candidate | train candidate |
| 203 | 81 | [1177,1887,52,48] | 2,496 | 2,134.5 | 8.2 x 7.5 | blocked: Unlabeled litter | blocked: Unlabeled litter | blocked: Unlabeled litter |
| 204 | 81 | [1014,1875,50,46] | 2,300 | 1,567.5 | 7.8 x 7.2 | blocked: Unlabeled litter | blocked: Unlabeled litter | blocked: Unlabeled litter |
| 244 | 92 | [922,1748,492,775] | 381,300 | 254,732 | 77.2 x 121.6 | train candidate | train candidate | train candidate |
| 2237 | 704 | [1346,1183,52,54] | 2,808 | 2,242 | 8.2 x 8.5 | blocked: Unlabeled litter | blocked: Unlabeled litter | blocked: Unlabeled litter |
| 2611 | 841 | [1042,1305,144,212] | 30,528 | 21,034.5 | 22.6 x 33.3 | blocked: Disposable food container | train candidate | train candidate |
| 2835 | 924 | [1611,364,477,406] | 193,662 | 118,239.5 | 74.8 x 63.7 | blocked: Other plastic | blocked: Other plastic | blocked: Other plastic |

“Train candidate” means a new image in the training-only expansion proposal; it
does not imply an approved dataset, split, or taxonomy change. `bbox area` is
width x height; source segmentation area is included separately because TACO's
polygon area is not the same quantity.

## Strict-clean waterfall with hypothetical Organic class

The unchanged waterfall remains whole-image quarantine: invalid/missing raster,
dimension mismatch, ambiguous EXIF, invalid box, crowd/ignore, Unlabeled litter,
other excluded annotation, no mapped annotation, then duplicate conflict /
quarantine. Tier 1 adds only Food waste to the existing 0..8 mapping. Tier 2 also
adds the D-073 sibling mappings. Tier 3 additionally omits only the approved
accessory boxes; it does not relax Unlabeled litter or other excluded-category
quarantine.

| Tier | Eligible Food waste images / boxes | Added candidate image IDs | Blocked Food waste images / boxes | Primary block reason |
| --- | ---: | --- | ---: | --- |
| Tier 1 + Food waste -> Organic | **2 / 2** | 72, 92 | 5 / 6 | 2 images / 3 boxes: Unlabeled litter; 3 images / 3 boxes: Food Can, Disposable food container, or Other plastic |
| Tier 2 siblings + Food waste -> Organic | **4 / 4** | 68, 72, 92, 841 | 3 / 4 | 2 images / 3 boxes: Unlabeled litter; 1 image / 1 box: Other plastic |
| Tier 3 accessory omission + Food waste -> Organic | **4 / 4** | 68, 72, 92, 841 | 3 / 4 | Same exclusions as Tier 2; Image 704's Plastic bottle cap can be omitted, but its Unlabeled litter still quarantines the full image |

On Image 68, Tier 2 maps Food Can to `drink_can`; on Image 841 it maps
Disposable food container to `plastic_cup`. Tier 3 unlocks no additional Food
waste image because the only otherwise accessory-relaxed case (Image 704) also has
Unlabeled litter. Image 924's three Other plastic annotations remain excluded in
the primary Tier 2/Tier 3 definition; D-073 treated that sibling separately.
Where a row has both Unlabeled and excluded annotations, the table reports the
first applicable strict-waterfall rule; the full reason lists are in the JSON.

The eight boxes are 100% annotation-valid by the metadata/box checks, but this says
nothing about whether every visible food-waste object was annotated. The eligible
images contain respectively 2, 1, 1, and 1 total boxes in the Tier 2/Tier 3 sets;
they do not provide independent image-level validation support.

## Feasibility and recommendation

TACO alone does **not** support a defensible tenth supervised class here. Even the
largest requested tier yields only **4 images / 4 boxes** for training, while all
65 validation and 65 test images remain frozen and contain **zero** Food waste
annotations. Creating an Organic validation split from four examples would leave
at most three training images (and still provide a fragile, nonrepresentative
estimate); Tier 1 would leave only one training image if one of its two were held
out. A tenth class would therefore be severely under-supported and impossible to
evaluate on the fixed validation/test splits.

Recommendation: keep the supervised taxonomy and weights at IDs `0..8`. Treat
`"food waste."` as a separately routed zero-shot prompt hypothesis, evaluated only
after explicit authorization and with an independently permitted, representative
Organic validation set. If supervised Organic performance is required, obtain a
permissioned external supplement with enough diverse train and validation images
before adding class ID 9. Do not infer competence from the raw TACO count, and do
not alter the frozen splits to manufacture support.

The machine-readable ledger and source hashes are in the private
[audit summary](../runs/training/stage5-organic-audit-230f56adfa31496fb475d439ff588f20/summary.json).
The script freshly decoded only the seven non-held-out Food waste rasters; it read
no `test.json`, frozen validation/test photographs, checkpoint, or model package.

Qualification: **327 guarded tests, 311 passed, 16 documented skips, zero
failures/errors**. Guard evidence records zero GPU access, detector calls,
optimizer updates, and protected-directory writes. The four new CPU fixtures cover
class-9 mapping, Tier 2 sibling unlocks, persistent Unlabeled quarantine after
Tier 3 accessory omission, and 384/576 box scaling. The Markdown link checker
reported no missing local destinations; `git diff --check` passed. D-079 is
committed locally as `287ac5f`; D-080 changes remain uncommitted for owner review.
