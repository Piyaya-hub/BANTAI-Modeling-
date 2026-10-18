# Stage 3A: Grounding DINO Tiny compatibility and dataset expansion assessment

Historical assessment: [Stage 3B](TACO_STAGE_3B.md) now records an independently installed synthetic environment and actual bounded CUDA execution. The unrun statements below describe Stage 3A; Stage 3B does not authorize real-data training or clear dataset/license decisions.

Assessment date: 2026-10-18. D-037 authorizes investigation and recommendations only. Stage 2 code/export at `18b1094` was reviewed on Windows. The eight-label taxonomy, strict-clean export, source photographs/annotations, inference code/configuration, cached weights and environment are preserved. No model was constructed, no forward/backward pass or optimizer step was run, and no replacement export, installation or download occurred. Tokenizer/image-processor probes below are CPU software checks, not model or training evidence.

## Recommendation and readiness

**Prefer a small separate PyTorch training program using Transformers 4.51.3's existing `GroundingDinoForObjectDetection` loss and the exact HF checkpoint.** This avoids checkpoint conversion and can save directly in the existing HF format. First seek approval for an isolated training environment and a bounded strict-clean smoke test after rights clearance. Keep the eight classes and current splits as the initial reference. For expansion, prioritize reviewed coordinate cases, then consider auxiliary supervision for known excluded categories under a separately approved versioned policy. Continue quarantining unresolved `Unlabeled litter` images. Do not expand by deleting annotations or by changing the current policy in place.

This is **source-compatible in principle, not runtime-qualified for training**. SciPy is absent, backward/AMP/save-reload have not been exercised, available VRAM has not been measured for training, and photo licensing is unresolved. `training_ready=false` remains correct. Recommend access to a Linux GPU with approximately 24 GB VRAM for the first meaningful training experiment; 6 GB local capacity should be treated as a possible reduced smoke-test target, not an established full-training configuration.

## Stage 2 review

Reviewed `taco_dataset.py`, `taco_preparation.py`, `taco_splits.py`, the preparation/validation CLIs, their tests, policy/taxonomy snapshots, generated eligibility/annotation ledgers, manifests and class statistics. The implementation reconstructs eligibility from original sources, quarantines entire ambiguous/mixed images, preserves complete source records, keeps photographs external, groups candidate duplicate photographs before splitting, and independently reconstructs exported content. Those safeguards should remain.

| Evidence | Observed result |
| --- | --- |
| Source | 1,248 readable / 252 missing photographs; 3,938 annotations on readable images; original COCO SHA-256 `ac1ec605c9e1fcda767970de3457c69b4a43b62c02aca8c4c80f51d1b5ae8449` |
| Current export | `exports/taco/strict-clean-stage2-v1/`, 315 images / 538 annotations |
| Split | Train 221/397, validation 47/75, test 47/66 (images/objects), seed 42 |
| Focused checks rerun in Stage 3A | 13 Stage 2 tests passed |
| Independent real-source validation rerun | Passed; `errors=[]`, `recomputed_from_current_sources=true`, `training_ready=false`; 221/397, 47/75, 47/66 unchanged |
| Loss of coverage | Only 538 of 2,289 mapped source annotations retained; includes missing/unaligned/mixed-scene losses, not just unavailable images |
| Leakage limitation | No exact duplicates found previously; two perceptual candidate pairs investigated. Hash similarity is not exhaustive provenance/scene separation. |

No confirmed in-scope Stage 2 defect was found in this review. Structural validity does not prove complete annotation, rights clearance, training compatibility, balanced sampling or adequate test coverage. Strict-clean favors simpler scenes and lacks deliberately reviewed negative scenes. Plastic bags have only 9 training, 2 validation and 2 test images; cups 16/4/3; styrofoam 20/5/4. Classwise evaluation will be unstable. See [Stage 2 history and full counts](TACO_STAGE_2.md).

## Compatible implementation contract

### Exact loading and candidate environments

Use `IDEA-Research/grounding-dino-tiny`, revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, for **both model and processor**. Proposed trainer loading uses `GroundingDinoForObjectDetection.from_pretrained` with `revision`, `cache_dir`, `local_files_only=True`, `use_safetensors=True`, `trust_remote_code=False`, and `disable_custom_kernels=True`; processor uses the same identity/cache/revision and offline setting. Check loading information for unexpected/missing/mismatched keys; do not use `ignore_mismatched_sizes`, replace the text-conditioned head with an eight-output classifier, or switch to another Tiny checkpoint. Record weights/config/processor hashes and every training override. The [pinned model card](https://huggingface.co/IDEA-Research/grounding-dino-tiny/blob/a2bb814dd30d776dcf7e30523b00659f4f141c71/README.md) identifies the HF model family.

Candidate environment: Python 3.11.9, torch 2.7.1+cu118, torchvision 0.22.1+cu118, Transformers 4.51.3, with the rest of the verified environment captured separately. Local inspection also reports safetensors 0.8.0. Additions proposed only for the isolated environment: `scipy==1.15.3` for Hungarian assignment and `pycocotools==2.0.10` for later COCO evaluation. Their published [SciPy](https://pypi.org/project/scipy/1.15.3/) and [pycocotools](https://pypi.org/project/pycocotools/2.0.10/) releases are candidates, not an installed/tested lock. Verify resolution, wheel compatibility and `pip check`, then capture complete versions, wheel hashes, OS, CUDA/driver and trainer commit. A plain PyTorch loop needs no Trainer/Accelerate integration. Do not install anything into the working inference `.venv`.

Alternatives considered:

| Route | Assessment |
| --- | --- |
| Existing HF 4.51.3 loss + small PyTorch loop | Recommended: exact HF weights/processor and native save format; adapter, collation and training lifecycle still need implementation and smoke qualification. |
| Official IDEA GroundingDINO repository | Its [README](https://github.com/IDEA-Research/GroundingDINO#label-todo) still lists training code as unreleased. Inference demos are not a reproducible training implementation. |
| MMDetection Grounding DINO | [Maintainer training/custom-COCO recipes](https://github.com/open-mmlab/mmdetection/tree/main/configs/grounding_dino) exist. They use MMDetection-formatted checkpoints, extra dependencies and their own configuration. Exact HF-to-MMDetection weight conversion and round-trip equivalence are unverified; not the first choice for preserving this checkpoint/inference pipeline. Pin a specific repository commit if later selected; mutable `main` is research evidence only. |
| Newer Transformers / community forks / LoRA | Potential fallbacks, not qualified here. No assumed drop-in training or adapter coverage; require separate source/version/license and round-trip review. |

### Training supervision and losses

Installed [4.51.3 model source](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/grounding_dino/modeling_grounding_dino.py) accepts `labels`, builds text-span maps, and dispatches to its detection loss. `class_labels` are zero-based **prompt-segment indices**, not original TACO IDs. Keep all eight prompt segments in the same order for every image, including classes absent from that image. Assert the token-span/category correspondence; do not create a variable image-specific prompt without remapping targets. Keep the model's 900 queries and 256-token capacity initially.

The [4.51.3 criterion/matcher](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/loss/loss_grounding_dino.py) uses Hungarian assignment with class/L1/GIoU costs (cached config: 1/5/2), token sigmoid focal classification loss plus box L1/GIoU (weighted 2/5/2), and encoder supervision when `two_stage=true`. The cached config has `auxiliary_loss=false`; decoder auxiliary losses require an explicit recorded override, not an assumption. Cardinality diagnostics are not a substitute for detection metrics. The matcher requires SciPy's `linear_sum_assignment`; it is missing locally. No loss execution was attempted.

Proposed canonical text, deliberately separate from the owner's inference prompt:

```text
cigarette. plastic bottle. plastic bag. plastic wrapper. plastic film. drink can. plastic cup. styrofoam piece.
```

This is a trainer proposal, not an inference default or taxonomy modification. Names with spaces map to the approved underscore keys in ID order 0–7. A cached tokenizer plus `build_label_maps` probe produced **30 tokens including special tokens and exactly eight matching segments**. Inspect all subword spans, punctuation, truncation and masks; repeat these checks for batches/padding and any future prompt change. Keep raw phrase decoding for qualitative review, including incomplete/unmapped phrases.

### Annotation adapter and collation

Read existing split JSON and external `info.image_root`; resolve paths safely and verify identities. Decode stored raster with Pillow/RGB **without EXIF transpose**, matching the Stage 2 coordinate frame. Pass one COCO target per image: `{'image_id': id, 'annotations': [...]}` with the exported category IDs and pixel `[x,y,width,height]` boxes, positive area and `iscrowd=0`. Preserve provenance in separate metadata rather than sending nested source records as model tensors.

The [image processor implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/grounding_dino/image_processing_grounding_dino.py) converts xywh to corners, resizes, normalizes to center-x/center-y/width/height and pads. Feed original xywh once; never pre-normalize or convert twice. Its crowd filtering is **not** loss-region ignoring. It also clips boxes, so the loader must reject invalid/out-of-bounds input before processing rather than rely on this behavior. A controlled 100×100 synthetic image with `[10,20,30,40]` produced `[.25,.40,.30,.40]` and preserved class ID 7.

Collate images and text with processor padding/pixel masks, retaining `labels` as a list of variable-length target dictionaries. Verify normalized boxes relative to the padded model frame, original dimensions, category IDs, object counts and round-trip geometry for portrait/landscape, edge and small boxes. Move tensors, including label tensors, to the selected device; do not cast integer IDs/masks to FP16. Initial preprocessing reference is cached shortest edge 800 / longest edge 1333, ImageNet normalization. Reduced smoke resolution is a separate recorded experiment.

### Saving and reload

After future authorization: save to a new ignored run directory using `model.save_pretrained(..., safe_serialization=True)` and `processor.save_pretrained(...)`. Save ordered taxonomy/prompt spans, dataset/policy/source hashes, split IDs, base revision, optimizer/scheduler/scaler/RNG states, step, precision, frozen modules and environment separately for resumption. Never overwrite the base cache or inference configuration. Reload model **and processor from the saved directory offline**, require clean loading diagnostics, and compare fixed-image eval outputs before/after reload within a declared numeric tolerance. Test optimizer resume separately; weights alone do not resume training. These APIs are documented by [HF model persistence](https://huggingface.co/docs/transformers/v4.51.3/main_classes/model#transformers.PreTrainedModel.save_pretrained) and [processor persistence](https://huggingface.co/docs/transformers/v4.51.3/main_classes/processors#transformers.ProcessorMixin.save_pretrained). Neither round trip was executed here.

## Excluded and unlabeled annotations

**Do not drop excluded boxes and call the rest clean.** The inspected classification loss constructs zero token targets for unmatched queries and has no spatial `ignore`/crowd-region masking. Simply carrying `ignore`, `iscrowd`, segmentation or provenance fields through COCO does not stop negative supervision. Discarding an object that belongs to one of the eight labels is especially harmful; unknown litter could be such an object. Known non-target categories *can* be legitimate negatives in a deliberately approved closed-set task, but that policy has not been approved and must not be introduced silently.

| Strategy | Suitability |
| --- | --- |
| Current whole-image strict quarantine | Safest supported reference; small and biased but no recorded excluded objects silently removed. Retain as v1. |
| Eight targets plus auxiliary known source categories | Preferred expansion candidate to investigate: supervise known excluded objects with meaningful positive labels; evaluate eight target classes separately. Requires explicit taxonomy/prompt/export approval. Rare/ambiguous source categories still need review. |
| All 60 source categories including `Unlabeled litter` | Not recommended automatically. Unknown is an annotation state, not a coherent semantic class; it may overlap the eight labels. A broad taxonomy does not solve incomplete labels or rights issues. |
| Custom region-ignore loss | Research alternative only. Must exclude suitable unmatched-query/token negatives in decoder **and encoder/auxiliary** paths without suppressing valid overlapping target positives; test matching and overlap behavior. Native COCO flags do not implement this. Greater software/validation risk than the first smoke test warrants. |
| Reviewed crops or manual annotation completion | Possible later approval: include only regions whose target/non-target status is reviewed, transform every retained box, preserve crop/source provenance and coassign all views. Crops that cut through unknown/overlapping objects are not automatically clean. No crops or relabeling generated here. |

Eight approved labels plus 48 known excluded source names (excluding `Unlabeled litter`) would give 56 prompt segments. An exploratory cached-tokenizer count is **209 tokens**, below 256 for the exact lowercased source names. This only establishes text length; no taxonomy is adopted and no semantic/loss/performance adequacy is claimed. Synonyms or added descriptions can exceed the limit. Positive auxiliary supervision may reduce silent omission, but does not prove that source annotations cover every visible object.

## Dataset recovery assessment

Counts below come from the current `image_eligibility.jsonl`: count quarantined images with mapped annotations whose **entire reason set** is contained in the proposed review set. They are conditional upper bounds on already-recorded target objects, not usable new exports, not counts of newly annotated objects, and not additive. Other validity, licensing, leakage and human review gates remain mandatory.

| Review set | Potential additional images | Recorded mapped objects in those images |
| --- | ---: | ---: |
| Coordinate/dimension and EXIF reasons only | 44 | 55 |
| Known excluded-category reason only | 280 | 597 |
| Excluded plus unlabeled reasons | 406 | 1,077 |
| Coordinates plus all exclusion reasons | 512 | 1,346 |

The broader combined ceiling is 827 images / 1,884 mapped objects **including** the existing 315/538, before review and license filtering. This is not the 1,248-image availability count. Images without target objects and cases with invalid annotations remain outside these particular bounds.

Recommended order, subject to owner decisions:

1. Resolve rights and review the 44 otherwise-clean coordinate cases first. All 152 dimension mismatches previously matched EXIF-oriented dimensions, but dimension equality does not prove the annotation frame. Check actual corners/segmentations across orientation cases. If approved later, create a new non-destructive loader/transform policy with forward/inverse geometry tests; do not rotate original files or patch v1 exports.
2. Review the 280 known-excluded mixed images for auxiliary classes or explicitly approved non-target semantics. Prioritize bags/cups/styrofoam and mixed/small/occluded scenes, not aggregate object count alone. No blanket `other trash` merge is justified without checking confusion with targets.
3. For the additional unlabeled-containing cases, visually resolve every unknown region by annotation/review or keep the whole photo quarantined. Predicted pseudo-labels are not verified ground truth. Do not assign group membership or turn unknown litter into a generic negative.
4. Investigate eight invalid-annotation images separately. Slight negative origins and duplicate IDs require explicit correction/provenance decisions; automatic clipping/renumbering is not approved. The 252 missing photographs stay unrun; no download is proposed in this stage.
5. Preserve current validation/test identities as held-out anchors during any later expansion. Keep exact/near/sequence/source peers on the same side; train expansion must not use their alternate views, crops or relabels. If scene/photographer leakage or severe sampling bias requires a new evaluation protocol, version it explicitly and retain v1 results. Do not silently regenerate all splits.

## Hardware feasibility

Cached safetensors header inspection found 990 tensors with 172,277,902 stored elements (978 FP32 tensors and 12 INT64 tensors). This includes buffers/tied-storage details, so it is not a measured trainable-parameter count. Treating roughly 172.3 million elements as trainable gives about **2.57 GiB** at 16 bytes/element for FP32 parameters, gradients and two Adam moments. This arithmetic excludes activations, attention workspaces, temporary optimizer buffers, CUDA context, image/text batches and allocator fragmentation. It is not a peak-memory estimate or evidence that 6 GB fits.

| Resource | Proposed use and limitation |
| --- | --- |
| Existing Windows GTX 1660 Ti, 6 GB | Optional batch-1 smoke at a recorded 512/768 edge cap, frozen vision backbone/BERT, FP16 autocast + scaler if qualified, limited workers. Try full cached 800/1333 resolution only as a separate memory probe. OOM is an honest failed gate, not permission to silently alter the architecture. |
| Linux 16 GB GPU | Plausible frozen-backbone/batch-1 candidate; full training at original resolution remains unmeasured. |
| Linux approximately 24 GB GPU | Recommended planning target for headroom and controlled comparisons. No guaranteed batch size/fit, provider availability, price or duration. Owner approves provider/budget and any photo transfer first. |
| CPU / free Colab | CPU for loader/probes, not recommended for full training throughput. Colab GPUs, quotas and runtime continuity are [not guaranteed](https://research.google.com/colaboratory/faq.html); do not base reproducibility on a particular free allocation. |

Preserve the Windows inference environment/driver. Use a separate locked environment on either target, with custom kernels disabled initially. Freeze module parameters explicitly and keep frozen dropout behavior recorded; do not wrap the whole training forward in `inference_mode`. Test supported gradient checkpointing before relying on savings. Gradient accumulation changes effective batch size, but does not remove the single-image activation peak. FP16 can reduce some allocations; do not assume BF16 support or an automatic speedup on this GPU. [PyTorch AMP](https://docs.pytorch.org/docs/2.7/amp.html) describes autocast/scaling; [CUDA notes](https://docs.pytorch.org/docs/2.7/notes/cuda.html) explain memory accounting and nondeterministic operations. CUDA `grid_sample` backward may prevent bitwise reproducibility even with fixed seeds.

After authorization, report actual GPU/driver, precision, trainable/frozen counts, input/padded dimensions, batch size, optimizer implementation, peak allocated/reserved CUDA memory and `nvidia-smi` process/device use. Warm up, reset peaks and synchronize timing. Separate model load, decode/processing, forward/backward/step and evaluation; accumulation and evaluation can have different peaks. Measure several batches including largest/densest images. No training timings or memory-fit results exist yet.

## Licensing and provenance

Direct metadata inspection finds the source `licenses` array empty. Per-image values are `null` (715), `CC` (319), and `ODBL (c) OpenLitterMap & Contributors` (466). In the strict-clean subset:

| Split | Missing (`null`) | Unspecified `CC` | ODbL/OpenLitterMap notice |
| --- | ---: | ---: | ---: |
| Train | 133 | 22 | 66 |
| Validation | 28 | 8 | 11 |
| Test | 24 | 5 | 18 |
| Total | 185 | 35 | 95 |

The [TACO repository license](https://github.com/pedropro/TACO/blob/master/LICENSE) is MIT for software/documentation; do not infer blanket permission for individual Flickr photographs from it. The [maintainer README](https://github.com/pedropro/TACO) identifies Flickr-hosted images, COCO annotations and requests citation of Proença/Simões. The inspected [Zenodo v1.0 record](https://zenodo.org/records/3587843) did not expose a usable license designation in its rendered Rights section; the dataset site returned HTTP 502. Therefore annotation/database rights and individual photo permissions still need confirmation for this exact local snapshot.

`CC` alone lacks variant/version, creator and license URI. Even a specific [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) requires attribution and change notices and does not clear all privacy/personality rights. NC/ND/SA variants must not be assumed equivalent. The [ODbL text](https://opendatacommons.org/licenses/odbl/1-0/) distinguishes database rights from individual content rights and contains notice/share-alike provisions. An ODbL notice alone does not identify the photograph's copyright license. Do not assert that trained weights automatically are, or are not, derivative databases/works; that remains a rights-review question.

Before approving training/photo transfer/publication, establish an image-level rights ledger: original URL/photo ID, creator/credit, precise image license/version/URI, database/annotation terms, evidence date, permitted internal training/cloud transfer/redistribution, modifications and unresolved status. Preserve original strings/records; unknowns remain unresolved. Seek maintainer/rightsholder or institutional review as appropriate. Export/annotation redistribution, thesis figures and released weights need their own decision. Retain TACO and Grounding DINO citations, the base model's Apache-2.0 notices and any required third-party notices; [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) does not grant rights to training photographs. No images, derived exports or weights are published in this assessment.

## Proposed smoke test and evaluation (not executed)

1. **Preflight gate:** rights clearance for selected photos, approved isolated environment, exact source/export/model hashes, clean loading diagnostics and matching model/processor. Independently revalidate v1. Assert label-span order, no token truncation, geometry/count preservation and batch padding. Exercise a synthetic batch with empty targets, all classes, different box counts and edge boxes; these are software fixtures, not real negative-scene evidence. Add a controlled ignored-region fixture only if such a loss is later authorized.
2. **Tiny run:** choose 8–16 existing **training** images covering eight classes, small objects and different aspect ratios; record exact IDs. Start batch 1, BERT/vision backbone frozen, fixed seed 42, AdamW proposed LR `1e-5`, weight decay `1e-4`, 20 optimizer steps, no augmentation and no scheduler. Use unchanged loss coefficients and query count. Reduced resolution and FP16 are explicit separate configurations; prefer FP32 on sufficient VRAM for the first numerical reference. These settings test wiring only and are not selected training defaults.
3. **Numerical/lifecycle gates:** require finite total/component losses and intended nonzero gradients, optimizer state/update, unchanged frozen tensors, no invalid boxes/targets, recorded peaks/timings, and complete run statuses. Abort/report NaN/OOM rather than silently switch settings. Save/reload in a new directory; compare fixed eval outputs and test resume. A short loss reduction/overfit check is useful debugging evidence, never proof of generalization. No required accuracy target or improvement is invented.
4. **Baseline comparison:** after smoke approval, compare untouched pretrained weights against the smoke checkpoint on the same validation IDs, resolution, ordered text, decoder and score rule. Use a separately specified class-span scoring rule (proposed mean of ordinary subword sigmoid scores, matching the matcher's token averaging) to return class IDs and xywh boxes for COCO evaluation. Define multi-class duplicates and top-100 per-image handling before running, test conversion with fixtures, preserve raw phrases/scores independently, and document that this evaluator differs from raw-phrase threshold display. Do not silently change the existing detector/postprocessor. Do not use experimental inference 0.20/0.10 thresholds as training filters or validated evaluation settings.
5. **Evaluation:** use a future pinned COCO bbox evaluator for AP@[.50:.95], AP50/AP75, per-class support and recall; the [official evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) defines IoU/area/max-detection conventions. Use low, explicitly recorded score filtering for AP and a separately validation-selected operating threshold for qualitative displays. Report undefined metrics as unavailable, not zero or fabricated accuracy. Inspect misses, wrong phrases/classes, false positives, duplicates, small/occluded objects and mixed scenes. Bootstrap by photograph/scene only if justified; two test bag images cannot support a reliable classwise claim.
6. **Held-out policy:** validation alone selects prompts, settings and checkpoints. Keep the 47-image test split untouched until the protocol is fixed; evaluate once for a reported final comparison after later authorization. The current team photos remain qualitative external checks unless independently annotated and approved. Unknown/excluded regions are not evaluation negatives by default. Eight object labels do not establish pile/group detection or membership; those three existing project outputs remain a separate unresolved task.

## Actual commands, outcomes and reproducibility

Executed from the repository root in the existing Windows environment:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_taco_preparation.py -q
.venv/Scripts/python.exe scripts/validate_taco_prepared.py --dataset exports/taco/strict-clean-stage2-v1
```

Focused tests: **13 passed**, 1.053 seconds. Independent real-source reconstruction: **passed**, `errors=[]`, `recomputed_from_current_sources=true`, `training_ready=false`; all split counts and recorded artifact/source hashes remain consistent. Pending requirements remain license review, trainer compatibility, near-duplicate visual review and class coverage acceptance. No inference tests were rerun because no implementation changed. Stage 2's historical full-suite result remains 113 passed / one existing privilege skip, not a newly executed result.

Both self-contained documentation probes below were re-executed successfully; 182 local Markdown file links resolved and `git diff --check` passed. Stage 2 PR #13 was found merged externally into fetched main `077657c`; the assessment branch was fast-forwarded to that identical file tree. GitHub CLI is unavailable locally, so publication uses the connected GitHub tools. Local owner detection edits and ignored reports/scratch stay outside the commit.

Installed implementation SHA-256 values captured for source identity:

| File relative to `transformers/` | SHA-256 |
| --- | --- |
| `models/grounding_dino/modeling_grounding_dino.py` | `c6529257808bf67ad47b9bd94a6c9e4468d74fda2c66c69d06cba81b1c0d346d` |
| `loss/loss_grounding_dino.py` | `a94d09a3c6bd4f23ef50ea45ce0cc9739a1df865ef0046bb75a496c1901a385c` |
| `models/grounding_dino/image_processing_grounding_dino.py` | `25c2c59f9b776f8c8711f2ba55ab85427634054e386cff6d481af49f857f15ad` |

Read-only source/metadata/tokenizer/processor probes were executed with an ignored local scratch script `work/stage3a_review.py`; the successful summary is ignored at `reports/taco/stage3a-assessment.json`. Its first attempt failed on an incorrect report-field name (`original_image` instead of `image`); correcting that inspection script did not alter dataset code or evidence. Reproduce the essential compatibility probes with this self-contained PowerShell command (no model construction or network):

```powershell
@'
import importlib.metadata as metadata
import json, math, struct
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoTokenizer, GroundingDinoImageProcessor
from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
for name in ('torch', 'torchvision', 'transformers', 'scipy', 'safetensors', 'pycocotools'):
    try: print(name, metadata.version(name))
    except metadata.PackageNotFoundError: print(name, 'NOT INSTALLED')
p = Path('.cache/huggingface/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71')
with (p/'model.safetensors').open('rb') as f:
    header = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
print('stored elements', sum(math.prod(v['shape']) for k,v in header.items() if k != '__metadata__'))
t = AutoTokenizer.from_pretrained(p, local_files_only=True)
prompt = 'cigarette. plastic bottle. plastic bag. plastic wrapper. plastic film. drink can. plastic cup. styrofoam piece.'
ids = t(prompt, return_tensors='pt')['input_ids']
maps = build_label_maps(torch.zeros(1,1,256), ids)[0]
spans = [t.decode(ids[0][row[:ids.shape[1]].bool()]) for row in maps]
assert spans == [s.strip() for s in prompt.split('.') if s.strip()]
print('tokens', ids.shape[1], 'segments', spans)
processor = GroundingDinoImageProcessor.from_pretrained(p, local_files_only=True)
target = {'image_id':1, 'annotations':[{'id':1,'category_id':7,'bbox':[10,20,30,40],'area':1200,'iscrowd':0}]}
with Image.new('RGB',(100,100)) as im:
    labels = processor(images=im, annotations=target, size={'shortest_edge':100,'longest_edge':100}, return_tensors='pt')['labels'][0]
assert labels['class_labels'].tolist() == [7]
assert torch.allclose(labels['boxes'], torch.tensor([[.25,.4,.3,.4]]))
print('synthetic xywh conversion passed')
'@ | .venv/Scripts/python.exe -
```

Reproduce the expansion bounds and rights metadata counts without changing an export:

```powershell
@'
import json
from collections import Counter
from pathlib import Path
p = Path('exports/taco/strict-clean-stage2-v1')
rows = [json.loads(s) for s in (p/'image_eligibility.jsonl').read_text().splitlines()]
for relax in ({'dimension_mismatch','ambiguous_exif_orientation'}, {'excluded_annotation'}, {'excluded_annotation','unlabeled_annotation'}, {'dimension_mismatch','ambiguous_exif_orientation','excluded_annotation','unlabeled_annotation'}):
    subset = [r for r in rows if r['reasons'] and not (set(r['reasons'])-relax) and r['mapped_annotation_count']]
    print(sorted(relax), len(subset), sum(r['mapped_annotation_count'] for r in subset))
for split in ('train','validation','test'):
    data = json.loads((p/f'{split}.json').read_text())
    print(split, Counter(i['source_taco']['image'].get('license') for i in data['images']))
'@ | .venv/Scripts/python.exe -
```

## Decisions still requiring owner approval

- A separate trainer/environment, candidate dependency lock and bounded smoke execution; no training approval is inferred from this assessment.
- Rights clearance and any cloud upload/provider/budget, including whether unresolved-license photos can be used for the intended academic purpose.
- Coordinate/annotation correction policy and reviewed examples before any new export.
- Keep strict eight-class training only, or approve a separate auxiliary taxonomy/ignore strategy; resolve unlabeled regions explicitly. The approved taxonomy file is unchanged.
- A future versioned expansion/evaluation protocol, rare-class coverage goals and leakage review, including whether broader human annotation is worth the effort.

Stage 3A completes a technical recommendation, not training qualification. Backward/AMP/VRAM/save-reload, actual smoke learning, detection metrics, expanded-photo review and rights clearance remain **unrun or unresolved**. No current inference behavior or fine-tuning outcome is claimed.
