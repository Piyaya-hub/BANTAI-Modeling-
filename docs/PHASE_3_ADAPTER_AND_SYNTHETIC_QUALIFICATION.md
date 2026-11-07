# BANTAI Phase 3: Stage 3D Adapter & Synthetic Qualification

> **Scope**: Architecture investigation, loss routing analysis, isolated training environment setup (`.venv-training`), and Stage 3D adapter implementation with synthetic qualification (`D-035` through `D-040`).  
> **Status**: Completed & Qualified on Synthetic Data.

---

## Table of Contents
1. [Executive Overview & Objectives](#1-executive-overview--objectives)
2. [Stage 3A: Training Boundary Assessment & Adapter Feasibility](#2-stage-3a-training-boundary-assessment--adapter-feasibility)
3. [Stage 3B: Isolated Training Environment Setup (.venv-training)](#3-stage-3b-isolated-training-environment-setup-venv-training)
4. [Stage 3C: Loss Investigation & Hungarian Matcher Diagnostics](#4-stage-3c-loss-investigation--hungarian-matcher-diagnostics)
5. [Stage 3D: Architecture Adapter & Synthetic Qualification](#5-stage-3d-architecture-adapter--synthetic-qualification)
6. [Phase 3 Milestone Register (D-035 through D-040)](#6-phase-3-milestone-register-d-035-through-d-040)

---

## 1. Executive Overview & Objectives

Phase 3 established the architectural foundation for fine-tuning Grounding DINO Swin-T on real waste images. Because Grounding DINO is natively an open-vocabulary grounding model with complex text-vision cross-attention, standard object detection fine-tuning scripts cannot directly train it without an adapter.

Key Phase 3 Invariants:
1. **Isolated Training Environment**: Isolated dependencies in `.venv-training` to prevent version contamination of the inference environment.
2. **Native Loss Architecture**: Implemented custom loss adapter without modifying or monkey-patching installed Hugging Face Transformers files.
3. **Synthetic Qualification**: Rigorous qualification on generated synthetic images before any real photographic data was accessed.

---

## 2. Stage 3A: Training Boundary Assessment & Adapter Feasibility

# Stage 3A: Grounding DINO Tiny compatibility and dataset expansion assessment

Historical assessment: [Stage 3B](PHASE_3_ADAPTER_AND_SYNTHETIC_QUALIFICATION.md) now records an independently installed synthetic environment and actual bounded CUDA execution. The unrun statements below describe Stage 3A; Stage 3B does not authorize real-data training or clear dataset/license decisions.

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

No confirmed in-scope Stage 2 defect was found in this review. Structural validity does not prove complete annotation, rights clearance, training compatibility, balanced sampling or adequate test coverage. Strict-clean favors simpler scenes and lacks deliberately reviewed negative scenes. Plastic bags have only 9 training, 2 validation and 2 test images; cups 16/4/3; styrofoam 20/5/4. Classwise evaluation will be unstable. See [Stage 2 history and full counts](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md).

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


---

## 3. Stage 3B: Isolated Training Environment Setup (.venv-training)

# Stage 3B: isolated environment and synthetic training smoke test

Follow-up [Stage 3C investigation](PHASE_3_ADAPTER_AND_SYNTHETIC_QUALIFICATION.md) now traces the encoder tensor/gradient findings and proposes a correction requiring approval. The Stage 3B lifecycle measurements below remain valid historical evidence, not real-data readiness.

Executed on the owner's Windows computer, 2026-10-18, under D-038. **The bounded synthetic FP32 forward/backward/optimizer and save/reload smoke passed.** This qualifies only the recorded configuration. Native encoder-loss limitations below remain a real-data readiness blocker. No TACO photograph was read by the smoke command or used for training; the strict-clean dataset, inference `.venv`, pipeline, local inference settings, approved taxonomy and base checkpoint are preserved. No cloud service or spending was used.

## Implementation and boundaries

| Added/modified file | Purpose |
| --- | --- |
| [requirements-training-windows.lock.txt](../requirements-training-windows.lock.txt) | Existing Windows package pins plus SciPy 1.15.3, installed only in `.venv-training`. Existing inference requirements remain unchanged. |
| [requirements-training-windows.hashes.txt](../requirements-training-windows.hashes.txt) | SHA-256 locks for all 37 Windows/Python 3.11 artifacts, including pip/setuptools bootstrap wheels. |
| [scripts/smoke_train_grounding_dino.py](../scripts/smoke_train_grounding_dino.py) | CLI entry point; synthetic input only, bounded 1–5 steps, explicit device, no fallback. |
| [scripts/training_smoke.py](../scripts/training_smoke.py) | Environment guard, synthetic rectangles/COCO targets, prompt-span/geometry assertions, native PyTorch loop, diagnostics, source integrity and model/processor round trip. |
| [tests/test_training_smoke.py](../tests/test_training_smoke.py) | Twelve software tests; no pretrained model construction or real photos. Native loss fixtures require isolated SciPy. |
| [.gitignore](../.gitignore) | Adds only `.venv-training/` and `runs/training/`; environments, checkpoint weights, reports and logs remain local. |
| README, AGENTS and phase/scope/decision/test/handoff documents | Record current authorization, evidence and the stop before real-data fine-tuning. |

The CLI accepts no dataset or photo path. It generates one 320×240 landscape image with all eight classes and one 240×320 portrait image with three annotations. Rectangles are arbitrary colored shapes, **not synthetic evidence of recognizing trash**. Fixtures include a 4×4 edge box, nested boxes, opposite aspect ratios and unequal object counts. Tests separately cover an empty target set. The actual model smoke batch contains 8+3 objects, not an empty-target example.

The exact HF snapshot is read offline from `.cache/huggingface/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71/`. No alternate model, new weight download, replacement head or query-count change is used. Model and processor load with local-files-only settings, remote code disabled and custom kernels disabled. Loading information must have no missing/unexpected/mismatched keys/errors. Every cached snapshot file is hashed before and after the run.

All eight approved labels retain IDs 0–7. The separate training prompt is:

```text
cigarette. plastic bottle. plastic bag. plastic wrapper. plastic film. drink can. plastic cup. styrofoam piece.
```

The tokenizer produces 30 tokens including special tokens; native label maps match exactly eight ordered ordinary-text segments for each image. Source targets use COCO pixel xywh. Before processing, malformed/duplicate/ignored/out-of-bounds targets fail instead of being silently clipped. Processor output is checked against independently calculated normalized cxcywh coordinates in the **padded** image frame, preserving IDs/counts and integer tensor dtypes. Reloaded processor inputs and target geometry must match.

## Environment and reproduction

`include-system-site-packages=false` was verified in `.venv-training/pyvenv.cfg`; it inherits no inference/system packages. Training refuses to execute from inference `.venv` or a different prefix. It verifies every recorded version and runs `pip check` before touching the model. Existing inference package metadata and protected file hashes were captured separately for preservation checking.

| Component | Actual isolated environment |
| --- | --- |
| OS / Python | Windows 10, CPython 3.11.9 x64 |
| torch / torchvision | 2.7.1+cu118 / 0.22.1+cu118 |
| Transformers | 4.51.3 |
| SciPy / NumPy | 1.15.3 / 2.4.6 |
| Pillow / safetensors | 11.2.1 / 0.8.0 |
| GPU | NVIDIA GeForce GTX 1660 Ti, 6,442,450,944 bytes (6144 MiB), capability 7.5 |
| Driver / framework CUDA | NVIDIA 472.19 / torch CUDA 11.8; `nvidia-smi` separately advertises driver CUDA 11.4 |
| Dependency consistency | `pip check`: No broken requirements found |

The complete environment is in each ignored `report.json`; exact pins and wheel hashes are tracked. SciPy is the only added runtime package relative to the existing lock. No pycocotools, Trainer, Accelerate, cloud SDK or new model dependency is needed for this stage. Package installation reused cached wheels except where pip fetched missing package artifacts. This was a real isolated installation, not sharing/copying the inference site-packages.

From the repository root, use the existing Python 3.11.9 installation to create a **new** environment if it is absent; do not clear/overwrite an existing environment:

```powershell
.venv/Scripts/python.exe -m venv .venv-training
.venv-training/Scripts/python.exe -m pip install --require-hashes -r requirements-training-windows.hashes.txt
.venv-training/Scripts/python.exe -m pip check
.venv-training/Scripts/python.exe -m unittest discover -s tests -q
.venv-training/Scripts/python.exe scripts/smoke_train_grounding_dino.py --device cuda --steps 2 --batch-size 2 --shortest-edge 192 --longest-edge 256 --trainable fusion-decoder --precision fp32
```

Initial actual installation used `-r requirements-training-windows.lock.txt --report work/stage3b-install.json`. Hashes were then captured from pip's actual wheel records and Python's bundled pip/setuptools wheels. The hash-lock dry run passed. The download command below subsequently succeeded for **all 37 hash-enforced artifacts**, retained in the ignored wheelhouse. A second fresh environment installation from that wheelhouse was not performed. Python's [venv documentation](https://docs.python.org/3.11/library/venv.html) explains the isolated environment mechanism. These hash locks target Windows CPython 3.11 x64 CUDA 11.8 artifacts, not a portable Linux lock.

Optional local artifact preservation/verification, with no package installation:

```powershell
.venv-training/Scripts/python.exe -m pip download --require-hashes -r requirements-training-windows.hashes.txt --dest work/training-wheelhouse
```

The offline snapshot must already exist. The smoke command never downloads it. Each invocation creates a new UUID directory under `runs/training/`, prints the report path, and exits 0 for a complete smoke pass or 1 for a recorded failure. Invalid CLI arguments exit 2. It atomically updates strict JSON after each stage and step, preserves failed runs/tracebacks, and never overwrites an earlier checkpoint. A killed process may leave `status=running`; that is not a pass. Checkpoints are about 658 MiB each; no automatic deletion occurs.

## Actual runs and failures

| Run ID | Explicit configuration | Outcome |
| --- | --- | --- |
| `35ffac8ded62405aa6982a1fe9fabcd1` | CUDA, FP32, batch 2, 128/192 edges, frozen vision/BERT | **Failed before loss/optimizer:** `RuntimeError: selected index k out of range`. 170×170 padded features provide only 650 encoder positions for 900 queries. Not OOM; no optimizer step occurred. |
| `f4aa8b29e0234ce88e17f575e4ebfb13` | CUDA, FP32, batch 2, 192/256 edges, frozen vision/BERT, two steps | **Passed** native forward/backward/AdamW plus model/processor save/reload. |
| `ec713b1fbc47407b8d729d47fa13fbf2` | Same explicit settings; additional numeric/gradient diagnostics | **Passed** with final encoder-capacity/numeric checks and native encoder-gradient observations. This is the final qualification evidence. |

The first failure exposed a minimum spatial-capacity condition in the [pinned model's two-stage top-k selection](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/grounding_dino/modeling_grounding_dino.py). The utility now guards feature-position capacity and defaults to 192/256. Both landscape and portrait resize to 256×192 / 192×256, then pad to 256×256: 1,360 feature positions. Model architecture, 900 queries and 256 text-token capacity remain unchanged. The changed size was a new explicit run, not an automatic retry or hidden model modification.

## Final run measurements

Actual parameters: **172,249,090 total**, **35,836,936 trainable**. Vision/BERT parameters are frozen and their dropout behavior is held in evaluation mode while the remaining model trains. This is explicitly a partial-parameter training smoke. `--trainable all` exists for a separately requested synthetic experiment but was **not run**. FP16 is also implemented as an explicit option but **unrun**; no AMP or full-model fit is established.

Training: seed 42, AdamW LR `1e-5`, weight decay `1e-4`, `foreach=False`, gradient clipping norm 1.0, no scheduler/augmentation, FP32, no accumulation. Cached `two_stage=true`, `auxiliary_loss=false`, unchanged Hungarian matching costs 1/5/2 and native classification/L1/GIoU weights 2/5/2. No box/text threshold is applied to training targets or loss. This uses the [native Transformers 4.51.3 loss](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/loss/loss_grounding_dino.py), not a replacement criterion.

| Final run measurement | Step 1 | Step 2 |
| --- | ---: | ---: |
| Total native loss | 60765.60546875 | 60675.078125 |
| Decoder classification | 1.97500396 | 2.09115338 |
| Decoder L1 / GIoU | 0.08536888 / 0.59629178 | 0.10611068 / 0.64145774 |
| Encoder classification | 30378.0 | 30332.83984375 |
| Encoder L1 / GIoU | 0.32441053 / 1.20731497 | 0.24570787 / 1.08596432 |
| Gradient norm before clipping | 28920.767578125 | 28882.359375 |
| Tensors with gradients | 548 | 548 |
| Forward/backward/update including audits, seconds | 3.1655261 | 0.4754773 |

Losses/components and received gradients are finite, the gradient norm is nonzero, AdamW state is populated and the `model.level_embed` probe changes. Every frozen parameter retains its exact hash. The small change in total loss is wiring/debugging evidence only, not object recognition, convergence or useful learning.

Peak CUDA memory from before load/transfer through preprocessing and both training steps, measured with `torch.cuda.max_memory_*`: **2,892,561,920 allocated bytes (2.694 GiB)** and **3,061,841,920 reserved bytes (2.852 GiB)**. Reload/eval peaks after resetting counters: 921,864,704 allocated / 1,038,090,240 reserved bytes. The first failed run peaked at 1,168,617,984 allocated / 1,195,376,640 reserved bytes. These are PyTorch allocator measurements, not whole-device `nvidia-smi` peaks; other Windows display processes already used approximately 831 MiB before this work.

CUDA synchronization brackets the measured training interval. No separate warm-up occurred: step 1 includes lazy CUDA/kernel/optimizer initialization and should not be compared as steady-state throughput. Timings include native loss/gradient checks and clipping; exclude model load, synthetic generation, checkpoint serialization and reload. Load/transfer and total lifecycle timing are in the JSON. No OOM occurred at 192/256; original 800/1333 resolution, large batches, full parameter training and AMP remain unrun.

## Native-loss limitation requiring follow-up

The run passed the bounded lifecycle, **but native encoder supervision is not fully functional in the inspected path**:

- `encoder_pred_boxes.requires_grad=false`; encoder L1 and GIoU loss tensors also have no gradient. Ten declared trainable tensors receive no gradients: `model.enc_output` and its normalization weights/biases, plus all six `model.encoder_output_bbox_embed` layer weights/biases.
- Source review shows encoder proposal coordinates are detached before being returned for loss; top-k selection is discrete. The runtime flags corroborate that the recorded encoder box-loss terms cannot train the encoder box head in this path. Decoder regression and classification do have gradients.
- The encoder classification loss contributes over 99.9% of the recorded weighted total. Finite arithmetic does not validate its scale or suitability for fine-tuning. The cardinality diagnostic (894.5) also is not an accuracy metric.

No library patch, alternative loss, version upgrade, architecture change or omission of encoder loss was made to obtain a pass. **Recommendation: investigate/qualify the pinned encoder training path before real-data training approval.** A future correction or alternate training route must be explicit, source-reviewed and tested; it must not change the working inference environment. Stage 3A's proposed native route is now supported by a limited execution pass, not a complete training-readiness claim. Small variations between the two successful runs' decoder components also mean seeds are not a guarantee of bitwise CUDA reproducibility.

## Saving, reloading and preservation

Each successful run saves model safetensors and the processor together under its own `checkpoint/`, with `smoke_context.json` recording prompt, taxonomy, settings, base revision and completed steps. The processor retains its original default size; the recorded 192/256 override is reapplied for smoke replay. Full parameter hashes match after offline reload; loading diagnostics are empty. Reloaded pixel/text tensors, target geometry and padding match. Evaluation logits on active text positions and boxes match with **maximum differences 0.0 / 0.0** (declared tolerance `atol=rtol=1e-5`); expected masked `-inf` sentinels are checked separately. NaN/positive infinity cannot be hidden as padding.

All cached pretrained snapshot file hashes match before/after each attempt. The synthetic checkpoint is never promoted into inference configuration. This minimal qualification utility does **not** save optimizer/scheduler/scaler/RNG state for resume, and explicitly records `resume_supported=false`; it is not a production or resumable real-data trainer.

Preservation checks **passed**: inference environment complete package metadata and all seven hashes of detection settings, taxonomy, inference/viewer/preparation code, original inference lock and strict-clean preparation manifest match the before-install snapshot. Final implementation/checkpoint hashes match the successful report; all three run reports parse as strict JSON. Ignored-path checks cover the environment, weights and wheelhouse; 198 local Markdown file links resolve and diff whitespace checks pass. After process exit, `nvidia-smi` reports 792 MiB device use (the desktop remains active), with no claim of a whole-device peak measurement. Source photographs/annotations and split exports were not inputs to this stage's trainer and are not written anywhere by it. Existing Stage 2 source-integrity evidence remains in [Stage 2](PHASE_2_BASELINE_AND_VISUAL_REVIEW.md) and [Stage 3A](PHASE_3_ADAPTER_AND_SYNTHETIC_QUALIFICATION.md).

## Automated verification and remaining gates

Actual isolated software suite: **126 tests run, 125 passed, one existing Windows privilege/symlink skip**; 12 new smoke tests passed. Tests cover approved class order, all-eight synthetic coverage, malformed boxes/classes/metadata, empty targets, batch sizes 1/2, padding/coordinate math, source-target preservation, integer dtypes, text spans, native Hungarian/focal/L1/GIoU loss with independently calculated focal value and masked-token gradients, frozen hashing, loading mismatch rejection, numeric sentinels and minimum encoder capacity. CLI rejects real-photo inputs, automatic device fallback and more than five steps. These tests do not substitute for the actual CUDA/model lifecycle evidence above.

Pending before real-data fine-tuning: investigate encoder loss/gradient semantics and large loss scale; approve real-data trainer/resume/evaluation design; clear image/database rights; accept rare-class coverage and leakage protocol; measure the intended real-resolution configuration; authorize any new taxonomy/ignore/export policy separately. No detection quality metrics, real-data epochs, TACO training, new splits, cloud access or spend were executed. Stop here until separately authorized.


---

## 4. Stage 3C: Loss Investigation & Hungarian Matcher Diagnostics

# Stage 3C: native training loss and gradient investigation

Historical investigation: D-040 subsequently approved [Stage 3D's separate training-only adapter and synthetic qualification](PHASE_3_ADAPTER_AND_SYNTHETIC_QUALIFICATION.md). The proposal/approval boundary below describes Stage 3C; Stage 3D neither patches this native library nor authorizes real-data fine-tuning.

Executed on the owner's Windows GTX 1660 Ti on 2026-10-18 under D-039; report finalized 2026-10-19 (Asia/Manila). **Do not proceed to real-image fine-tuning with the current native objective unchanged.** Evidence identifies encoder-supervision tensor wiring as the problem, rather than malformed eight-class targets, a failed Hungarian matcher or numerical overflow. The Stage 3B lifecycle pass remains valid, but did not establish that the encoder objective trains the intended branches.

Recommended next stage: owner-approved **Stage 3D, a training-only encoder-output adapter and synthetic qualification**. Preserve decoder initialization detachment, query count, architecture, loss formula/coefficients, checkpoint and inference environment. No such adapter, Transformers patch, alternate optimizer objective or real-data trainer was installed by Stage 3C. The hypothesis calculations below are diagnostic recomputations only.

## Questions answered

1. **Why no encoder box gradients?** HF 4.51.3 gathers top-k proposal coordinates, detaches them for decoder reference points, then reuses those detached coordinates as encoder box predictions supplied to the loss. The separately parameterized encoder box head therefore has no supervised path. Detaching decoder initialization is expected; using only that copy for a nonzero encoder regression objective is problematic for intended end-to-end proposal-head training.
2. **Why the large encoder classification loss?** With pinned `embedding_init_target=true`, the reported encoder logits are recomputed from learned decoder query embeddings and pre-encoder text features. Those are different representations from the image-conditioned encoder proposal scores used for top-k selection. Runtime reproduction confirms the reported logits contain no image-specific information for a shared prompt, have many large positive unmatched-token scores, and drive a large focal negative term.
3. **Is the current approach suitable?** Exact checkpoint loading, preprocessing, native decoder loss, optimizer execution and persistence work in the tested synthetic scope. The unchanged native encoder objective is not a suitable qualified end-to-end training baseline for BANTAI yet.
4. **Is a correction necessary?** Evidence supports correcting which encoder tensors reach the existing criterion, retaining an undetached coordinate copy for supervision and using selected proposal logits. Lowering weights, dropping encoder terms or changing normalization to hide the magnitude is not justified by these results.
5. **Safest next step?** Approve the narrowly scoped adapter and repeat synthetic update/inference-equivalence/save-reload checks before any real-photo approval. Rights, coverage, evaluation and real-resolution memory gates remain independent.

## Source trace and interpretation

The installed files were hashed before/after execution; none changed. Relevant pinned source:

| Source | Relevant path |
| --- | --- |
| [HF 4.51.3 model](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/grounding_dino/modeling_grounding_dino.py) | `GroundingDinoModel.forward`, installed lines 2327–2358: proposal scoring, gathering, detachment, learned-query initialization and reported encoder outputs. Detection forward lines 2669–2683 dispatches those outputs to the criterion. |
| [HF 4.51.3 Grounding DINO criterion](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/loss/loss_grounding_dino.py) | Matcher, span-positive target construction, focal loss, and separate encoder/decoder supervision. |
| [HF 4.51.3 shared detection loss](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/loss/loss_for_object_detection.py) | Box L1/GIoU, object-count normalization with minimum 1, and diagnostic cardinality. |

The pinned encoder has independent box-head parameters (`two_stage_bbox_embed_share=false`). Selection indices are discrete, so decoder supervision does not differentiate through top-k ranking. Its reference-coordinate initialization is also detached. Both are normal boundaries; neither requires that the separate supervised encoder coordinates be detached. Frozen backbones, no-match empty regression losses, logging-only cardinality and unused/tied aliases can legitimately have absent or zero gradients. The conclusion here instead uses nonempty targets, an independent trainable encoder head, explicit nonzero regression weights, source routing and actual lack of updates together.

For comparison, [IDEA's transformer](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/models/GroundingDINO/transformer.py) retains `refpoint_embed_undetach` and separate encoder features while detaching decoder initialization. Its [public model file](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/models/GroundingDINO/groundingdino.py) comments out intermediate training outputs, so it is not evidence of an executable full released training recipe. [MMDetection's Grounding DINO pre-decoder](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/grounding_dino.py) explicitly retains gathered top-k scores/coordinates for training and a detached reference copy for the decoder. These mutable upstream references were inspected on the assessment date, not installed or claimed equivalent to the exact HF checkpoint. No maintainer acknowledgement of this particular version's behavior was obtained; the assessment rests on source and controlled observations.

## Experimental controls and files

| File | Purpose |
| --- | --- |
| [scripts/investigate_training_loss.py](../scripts/investigate_training_loss.py) | Bounded synthetic diagnostic CLI; no dataset/photo path, alternate checkpoint, install or patch operation. |
| [scripts/loss_investigation.py](../scripts/loss_investigation.py) | Reuses existing synthetic preprocessing; audits three tensor branches, component gradients, independent focal arithmetic and native optimizer updates. |
| [tests/test_loss_investigation.py](../tests/test_loss_investigation.py) | Seven regression/causal fixtures for count controls, masks/normalization, expected detach boundary, diagnostic autograd isolation, empty targets and large unmatched positive logits. |
| This report plus AGENTS/README/phase/scope/decision/test/handoff updates | Actual state, evidence, proposal and required approval boundary. |

The original [Stage 3B trainer](../scripts/training_smoke.py), dependencies, eight-class taxonomy, source photos/annotations, dataset splits, inference code and both environments remain unchanged. Only `.venv-training` executes the investigation. Current versions are Python 3.11.9, torch 2.7.1+cu118, Transformers 4.51.3, SciPy 1.15.3; no package installation was needed.

All cases reuse the same two in-memory rectangle images, ordered eight-class prompt, FP32, batch 2, 192/256 edge limits, 900 queries, frozen vision/BERT, seed 42 and native coefficient values. Before each case, every state tensor is reset to the same pretrained value and the RNG is reset. Cases change only target lists: 8+3, 1+1, 1+0, and 0+0. Reduced-count/empty fixtures deliberately omit drawn rectangles as a software control; they are **not** a proposed partially labeled real-data policy or reviewed negative scenes.

Each case does one AdamW step using **only the original model-returned native total**, LR `1e-5`, weight decay `1e-4`, clip norm 1. Diagnostic `autograd.grad` calls do not populate optimizer `.grad`; this is asserted before the native backward. Hypothesis terms never enter an optimizer objective or saved checkpoint. A read-only classifier pre-hook records its two input calls and is removed immediately; it does not replace any output.

### Actual commands and runs

```powershell
.venv-training/Scripts/python.exe scripts/investigate_training_loss.py --device cuda --cases baseline_11 two_targets one_target empty
.venv-training/Scripts/python.exe scripts/investigate_training_loss.py --device cuda --cases baseline_11
.venv-training/Scripts/python.exe -m unittest discover -s tests -p test_loss_investigation.py -q
.venv-training/Scripts/python.exe -m unittest discover -s tests -q
```

Each invocation creates `runs/training/investigation-<uuid>/report.json`, ignored by Git, with actual component values, token/match diagnostics, eight representative parameter probes, gradients, optimizer changes, memory, environment and source hashes. It returns 0 for completed diagnostics, 1 for a recorded failure, 2 for invalid arguments; `completed` is not a training-readiness flag. It never automatically falls back from CUDA or changes settings.

| Local run suffix | Outcome |
| --- | --- |
| `bc67da89b13e4679938f78260899d46a` | Inspection-tool failure before experiments: a tied decoder box parameter was sought under its alias rather than its canonical name (`StopIteration`). Fixed probe lookup, not model/library code. Retained as a failed attempt. |
| `7624daeb430746e5b2d424da3998295b` | Completed all four target-count cases, all component/gradient audits and four independent native updates. |
| `ac08f80e30274c76b95279d0eecfe9f9` | Completed baseline factor-isolation audit: box detach boundary alone versus classification inputs alone, plus one independent native update. Final diagnostic implementation hashes match this run. |

## Loss accounting and target-count observations

| Target count (image 1 / image 2) | Normalizer | Native decoder CE | Native encoder CE | Native total | Selected-encoder CE, hypothesis only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 / 3 | 11 | 1.975004 | 30378.000000 | 60765.605469 | 2.566194 |
| 1 / 1 | 2 | 2.261436 | 167259.453125 | 334528.281250 | 2.895101 |
| 1 / 0 | 1 | 4.062028 | 334564.125000 | 669141.187500 | 2.105351 |
| 0 / 0 | 1 | 3.315138 | 334579.812500 | 669166.250000 | 1.116389 |

All losses and received gradients are finite; no OOM or CUDA failure occurred. Empty-target matching, native classification backward and the optimizer step succeeded. Its box losses are exactly zero. In the connected hypothesis branch, empty box gradients are zero on reachable paths, as expected; they are not evidence of a broken encoder.

The target/matcher/normalization audit found:

- Span maps preserve approved IDs/order and all ordinary subword tokens. Native matching uses focal class cost averaged across each class's token span, plus L1/GIoU costs 1/5/2. Matching is recomputed per branch; its indices are recorded. This single-process run has no distributed normalization or reduction.
- Two images × 900 queries × 30 active prompt tokens give **54,000 active query/token cells**; the baseline has 30 positive cells after matching. Beyond-prompt `-inf` padding is masked out. Periods/CLS/SEP are active negative targets under the native attention-mask rule. Independent elementwise focal reconstruction agrees within FP32 tolerance with native CE.
- Baseline native encoder focal loss is essentially all negatives: **30378.0** negative versus approximately **3.2e-19** positive. Special/delimiter positions contribute 19412.4955 (63.9%); ordinary positions still contribute 10965.5043. Removing delimiters would not resolve the underlying mismatch and was not tried as a training change.
- Focal loss sums active cells and divides by `max(number_of_boxes,1)`. Baseline unnormalized encoder focal sum is **334158**. With fewer targets most unmatched contributions remain, while the divisor falls; thus sparse/empty native losses grow sharply. This normalizer is an expected detection convention, not independently diagnosed as a defect. Encoder/decoder weights remain 2/5/2; cardinality is excluded from the optimized total.

## Why classification is different

For the baseline, hook inputs and outputs establish exact learned-query-dot-text reproduction of reported encoder logits. They are identical between the two different images and across target-count cases before updates. Selected actual encoder logits differ between images. Mean text-feature norm is **128.3993** for the pre-encoder inputs used by reported encoder CE versus **4.9987** in actual proposal scoring. Their respective query/visual-feature norms are 1.6260 versus 8.4878. These are measured feature-space differences, not a proposal to apply arbitrary normalization.

| Active logit statistic | Reported native encoder | Selected actual encoder | Decoder |
| --- | ---: | ---: | ---: |
| Minimum | -162.84694 | -9.76664 | -8.55079 |
| Mean | 5.31687 | -6.22954 | -5.40714 |
| Maximum | 74.89127 | -0.36482 | -0.02891 |
| Fraction positive | 0.63163 | 0 | 0 |

Widespread large positive scores on unmatched cells saturate the focal factor and produce a large negative-target BCE term. The selected image-conditioned scores, using the same labels/matcher/mask/criterion, do not exhibit this scale. Synthetic shapes are not meaningful trash examples, so low logits or low hypothesis loss do not establish useful recognition. The causal finding is the different tensor path, not an expectation that all pretrained losses must be small.

## Gradient and update observations

The following baseline norms use individual components **before weighting/clipping**, through representative named tensors; “absent” means no graph path (`None`), distinct from a zero gradient. Full names and additional values are in JSON.

| Probe | Native encoder CE L2 | Decoder CE L2 | Selected-encoder CE L2, audit only | Selected-encoder box L2, audit only |
| --- | ---: | ---: | ---: | ---: |
| `query_position_embeddings.weight` | 8408.1885 | 0.81291 | absent | absent |
| `text_projection.weight` | 11695.7031 | 0.27194 | 0.45574 | 0.28476 |
| `enc_output.weight` | absent | absent | 1.34828 | 0.71611 |
| `enc_output_norm.weight` | absent | absent | 2.75075 | 0.31877 |
| Encoder box head, last weight | absent | absent | absent | 4.34174 |
| Encoder layer 0, text attention query | absent | 0.07971 | 0.06765 | 0.04647 |
| Decoder final norm | absent | 0.92751 | absent | absent |

Native encoder L1/GIoU have `requires_grad=false` even for nonempty targets. The three proposal-related probes have no native total gradient, no Adam state and **exactly zero update**; Stage 3B recorded ten affected encoder tensors overall. Learned query/text projection and decoder/fusion probes do receive native gradients and nonzero updates (~`1e-5` maximum absolute). Thus the native loop is not wholly inert. It is training a different classification path while leaving the independent proposal head unchanged.

Native total gradient norms are 28920.77, 159133.91, 318290.44 and 318295.81 for the four count cases. Global clipping is dominated by encoder query/text gradients; nevertheless AdamW still produces nonzero decoder updates. Do not equate a smaller clipped gradient with proven lack of learning or predict real-photo accuracy from these probes. All frozen parameter hashes remain unchanged.

### Factor isolation and smallest proposed correction

The baseline diagnostic recomputations keep the original decoder outputs and criterion:

| Encoder inputs, diagnostic only | CE | L1 | GIoU | Encoder box head receives a path? |
| --- | ---: | ---: | ---: | --- |
| Native scores + detached coordinates | 30378.0000 | 0.324411 | 1.207315 | No |
| Native scores + same coordinates before detach | 30378.0000 | 0.324411 | 1.207315 | Yes, box-probe L2 3.02666 |
| Selected proposal scores + detached coordinates | 2.566194 | 0.104934 | 0.737060 | No |
| Selected proposal scores + coordinates before detach | 2.566194 | 0.104934 | 0.737060 | Yes, box-probe L2 4.34174 |

Coordinates before/after detachment are bitwise equal. Changing classification inputs changes Hungarian assignment, explaining the changed regression **values** despite identical coordinates. Merely preserving the regression graph changes gradients without changing any loss value. The hypothetical combined weighted encoder+unchanged-decoder total is **12.700610** on this baseline; it was never used for an optimizer step.

Proposed correction, **not an approved training implementation**: select the same 900 indices from existing `enc_outputs_class`; gather those scores and `enc_outputs_coord_logits.sigmoid()` **before** detachment for encoder supervision. Keep the detached coordinate copy for decoder initialization. Feed these tensors into the same native matcher/criterion with unchanged label maps, masks and weights. This can live in an explicit training-only wrapper/adapter without editing installed Transformers, changing architecture/weights or touching inference. Avoid an unreviewed global monkey patch. A newer library or MMDetection is a fallback requiring its own exact-checkpoint/conversion qualification, not an automatic replacement.

## Feasibility, verification and acceptance limits

Both completed runs record unchanged pretrained snapshot and installed library hashes. Target geometry/spans are checked with the unchanged Stage 3B helper. No candidate objective is optimized and no new checkpoint is saved. Stage 3B's verified native model/processor round trip remains historical evidence; a future adapter must qualify its own saved context, load consistency and resume semantics.

GPU allocator peaks across the four cases: **2,721,721,344 bytes allocated (2.535 GiB)**, **3,049,259,008 reserved (2.840 GiB)**. These include read-only diagnostic gradient traversals plus one native backward/update per reset case; no warm-up or original-resolution test occurred. Case wall times including audits are 5.740, 1.525, 1.509 and 1.489 seconds. They are not ordinary training throughput or whole-device peak memory. The small frozen-backbone configuration fits; full-resolution 800/1333, all-parameter/AMP training, large batches and long-run stability remain unrun.

Actual tests: **133 run, 132 passed, one existing Windows privilege skip**; all seven new targeted tests passed. Software fixtures support the causal distinctions but do not replace the actual pinned CUDA observations. FP32 reduction order can produce tiny reconstruction differences, and deterministic seeds do not guarantee bitwise CUDA training reproducibility.

Final preservation checks passed for both environments' complete package metadata, eight protected setting/source/manifest hashes, three installed library source hashes, strict report JSON and final diagnostic implementation hashes. The unchanged Stage 3B trainer is among those protected files. All 208 local Markdown file links resolve, CLI help and diff whitespace checks pass. After process exit, whole-device usage was 803 MiB with the desktop still active; this is a closure observation, not a peak measurement. Generated investigation reports remain ignored, and the owner's local detection edit remains outside the commit.

Before any real-image stage, require: approved adapter or alternative; nonempty encoder-head updates and empty-case stability under that actual objective; unchanged decoder outputs at identical weights; unchanged model/processor save-reload and explicit resume support; measured intended-resolution memory; pinned evaluator/class-span decoding with held-out policy; image/database rights clearance; rare-class coverage, leakage and excluded/unlabeled handling decisions. The eight labels and v1 splits remain intact and no real-data readiness is claimed.

**Approval requested for the next stage only:** implement and qualify the proposed external training-only tensor adapter using synthetic data, without real-photo fine-tuning, package upgrades or architecture changes. Stage 3C stops before that substantive method change.


---

## 5. Stage 3D: Architecture Adapter & Synthetic Qualification

# Stage 3D: training-only encoder-output adapter and synthetic qualification

**Assessment: PASS for the recorded corrected synthetic training objective.** Executed 2026-10-19 (Asia/Manila) on the owner's GTX 1660 Ti, under D-040. The approved adapter restores encoder supervision using selected image-conditioned proposal scores and undetached box values, while retaining the native decoder's detached initialization. Actual gradients, optimizer updates, finite losses, exact decoder equivalence and checkpoint replay passed. This does **not** establish real-data training readiness, detection quality, convergence or full-resolution memory fit.

No real photograph was used or loaded by the qualification command. Original TACO files, approved taxonomy, validated splits, inference pipeline/settings/environment, installed Transformers and pretrained snapshot remain unchanged. No cloud resources, dependency changes, architecture changes or automatic checkpoint promotion occurred. Stage 3D stops here.

## Files and implementation

| File | Purpose |
| --- | --- |
| [scripts/training_adapter.py](../scripts/training_adapter.py) | Explicit stateless training-only adapter, selected-proposal gathering, input/configuration guards and reproducibility manifest. |
| [scripts/qualify_training_adapter.py](../scripts/qualify_training_adapter.py) | Bounded synthetic CUDA/CPU qualification, actual corrected optimizer steps, gradient/update checks, equivalence and checkpoint round trip. No real-data input option. |
| [tests/test_training_adapter.py](../tests/test_training_adapter.py) | Eleven regression tests with a small standalone tensor model and cached tokenizer; no library patch or pretrained model construction in fixtures. |
| This report; AGENTS/README/phase/scope/test/decision/handoff updates | Current authorization, actual results, acceptance scope and remaining gates. |

The adapter is an ordinary Python object, not an alternative model, registered neural layer or subclass of Transformers. It does not mutate weights/configuration or install hooks/monkey patches. The original Stage 3B trainer and Stage 3C diagnostics remain unchanged and available for historical reproduction.

The forward sequence is:

1. Validate the pinned configuration, original coefficient values and fixed eight-class prompt. Validate normalized padded-frame targets and the qualified batch size (1 or 2).
2. Run the **unchanged model forward without `labels`**, preserving every decoder operation. This avoids invoking the miswired native loss inside the model; it does not skip encoder supervision.
3. Select exactly the native 900 indices from `enc_outputs_class.max(-1)`. Gather those logits and corresponding `enc_outputs_coord_logits`, applying sigmoid to the gathered coordinates **without detaching the supervised copy**.
4. Assert bitwise equality between selected box values, the native detached `encoder_pred_boxes` and native decoder `init_reference_points`; assert both native reference tensors remain detached.
5. Call the existing `GroundingDinoForObjectDetectionLoss` with unchanged decoder outputs, the gathered encoder outputs, native span maps/masks, original labels and original configuration. No custom focal, matching, normalization or weighting implementation replaces it.

The [pinned native model](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/grounding_dino/modeling_grounding_dino.py) and [native criterion](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/loss/loss_grounding_dino.py) remain installed unchanged. [Stage 3C](PHASE_3_ADAPTER_AND_SYNTHETIC_QUALIFICATION.md) provides the source/causal evidence for the routing correction. The adapter leaves the original `ModelOutput` intact under `result.outputs` and returns the corrected `result.loss`/`result.loss_dict` separately.

Config guards reject changes to 900 queries, 256 text-token capacity, six decoder layers, two-stage mode, learned query initialization, head sharing and native matching/loss settings. `auxiliary_loss=false` is preserved; no decoder auxiliary terms are silently enabled or disabled. Matching costs remain 1/5/2; native classification/L1/GIoU weights remain 2/5/2, focal alpha .25. The eight labels/order and 30-token prompt are unchanged. A missing regression gradient, changed decoder detach boundary, different proposal values or unsupported text/configuration fails visibly; there is no alternate loss/device/model fallback.

## Environment and exact execution

The existing `.venv-training` was reused without installations: Python 3.11.9 x64, torch 2.7.1+cu118, torchvision 0.22.1+cu118, Transformers 4.51.3, SciPy 1.15.3, NumPy 2.4.6, Pillow 11.2.1 and safetensors 0.8.0. GPU: GTX 1660 Ti, 6144 MiB, capability 7.5; driver 472.19. The [Stage 3B version/hash locks](../requirements-training-windows.hashes.txt) remain the environment specification. The runner refuses execution from inference `.venv` and checks all pinned versions plus `pip check`.

The cached model/processor are loaded offline from the exact snapshot `IDEA-Research/grounding-dino-tiny` at `a2bb814dd30d776dcf7e30523b00659f4f141c71`. All loading mismatch/error lists are empty. Safetensors, remote-code-disabled and custom-kernel-disabled settings are retained. There is no new weight download or checkpoint-selector option.

From the repository root:

```powershell
.venv-training/Scripts/python.exe -m unittest discover -s tests -p test_training_adapter.py -q
.venv-training/Scripts/python.exe -m unittest discover -s tests -q
.venv-training/Scripts/python.exe scripts/qualify_training_adapter.py --device cuda --steps 2 --cases baseline_11 two_targets one_target empty
```

The command creates a fresh ignored UUID directory under `runs/training/adapter-<uuid>/`, prints its report path and exits 0 only after every invariant/qualification check passes. Failures yield `assessment=FAIL`, status/error/traceback and exit 1; malformed arguments exit 2. A killed process can leave `PENDING/running`, never a pass. Steps are bounded to 1–3 per case; the baseline must come first. The command accepts no photograph/dataset paths, automatic device mode, promotion or resume option.

## Actual synthetic runs

| Run suffix | Outcome |
| --- | --- |
| `aa095021049740d6acc5dcac96a28331` | Initial corrected objective: all four cases, eight total updates, decoder/repeat equivalence and checkpoint replay passed. |
| `a06cfea197c545fc82c6df91c0b39583` | **Final PASS** after improving adapter source-fingerprint portability across LF/CRLF checkouts and recording runtime versions in its saved manifest. All cases/equivalence/save-reload were requalified against the final implementation. |

No failed invariant was bypassed and neither attempt had NaN/Inf, OOM or a model/library error. The second run was warranted by a reproducibility metadata change, not a new objective or a search for lower loss. Both prior run directories remain preserved.

Controls: two in-memory 320×240/240×320 colored-rectangle images, same eight-class prompt, FP32, batch 2, seed 42, frozen vision/BERT, 192/256 resize limits and 256×256 batch padding. All model state is reset to the same pretrained state and RNG before each count case. There are **172,249,090 total / 35,836,936 trainable parameters**, unchanged from Stage 3B; no classifier replacement or query-count reduction occurs. Two AdamW steps per case use LR `1e-5`, weight decay `1e-4`, `foreach=False`, clip norm 1, no scheduler/accumulation/augmentation. The original 900-query geometry-capacity guard and pixel/token/target checks remain in use.

The reduced-count/empty controls omit labels on the synthetic rectangles deliberately to test software paths; they do not approve partially labeled real photographs or establish a real no-trash scenario. Known excluded/unlabeled TACO annotations remain under the existing whole-image quarantine policy.

### Final losses and gradient observations

| Case: target counts per image | Total corrected loss, step 1 | Total corrected loss, step 2 | Encoder CE, step 1 / 2 | Encoder L1, step 1 / 2 | Encoder GIoU, step 1 / 2 |
| --- | ---: | ---: | --- | --- | --- |
| Baseline 8 / 3 | 12.700611 | 14.114100 | 2.566194 / 3.136733 | .104934 / .114393 | .737060 / .662438 |
| Two targets 1 / 1 | 12.731905 | 13.204998 | 2.895101 / 2.929698 | .060598 / .076215 | .383864 / .307913 |
| One target 1 / 0 | 14.745838 | 14.014247 | 2.105351 / 2.444302 | .041174 / .029678 | .488563 / .317324 |
| Empty 0 / 0 | 8.863054 | 6.758975 | 1.116389 / 1.164306 | 0 / 0 | 0 / 0 |

Every total/component loss and received gradient was finite. Two-step training losses need not decrease monotonically: dropout, proposal selection and matching can change; these are wiring/numerical checks, not a convergence or recognition experiment. The first baseline value matches Stage 3C's independently computed hypothetical objective. The native pre-correction baseline total (60765.605469) is historical, not a newly run competing objective in this stage.

Baseline first-step component audit:

| Component | Actual value |
| --- | ---: |
| Decoder classification / L1 / GIoU | 1.975004 / .085369 / .596292 |
| Encoder classification / L1 / GIoU | 2.566194 / .104934 / .737060 |
| Global gradient norm before clipping | 56.696583 |
| Encoder CE path to `enc_output.weight`, L2 | 1.348285 |
| Encoder CE path to `enc_output_norm.weight`, L2 | 2.750754 |
| Encoder regression path to final encoder box weight, L2 | 4.341739 |
| Maximum update in each of these three probes | approximately `1.00136e-5` |

The contrastive scorer has no learned classifier parameter of its own: classification supervision now reaches the proposal projection/normalization and fused features. Nonempty cases require positive classification and box-probe gradients, actual nonzero updates and Adam state. All pass. Every declared trainable parameter receives a gradient tensor in all eight steps; none is listed as missing. A present tensor can still be zero, as appropriate for empty regression.

For the fully empty case, encoder box loss/box-path gradients and observed encoder box-weight updates are exactly zero; classification still has finite gradients and updates. A zero regression update here is expected, not a failed qualification. All frozen parameter hashes remain unchanged. Full gradient-path/probe/component values and clipped norms are retained in the ignored JSON report.

### Decoder equivalence and adapter reproducibility

For **all four cases before and after updates**, direct decoder inference and the adapter's decoder output are exactly equal at identical weights: maximum active-logit difference **0.0**, maximum box difference **0.0**. Initial references retain native detachment and exact box values during training calls. Architecture/state-key/shape/dtype hashes are unchanged. Weight hashes also match before/after pure adapter equivalence calls.

Repeated eval-mode adapter calls at fixed weights produce exactly the same decoder outputs and corrected scalar loss. This establishes that the adapter adds no stochastic operation; it does not guarantee bitwise reproducibility of CUDA optimizer trajectories. Tiny between-run variations in later losses are preserved rather than relabeled as exact training replay. No settings were selected based on those variations.

### Checkpoint and replay

After the baseline's two steps, a new local `checkpoint/` stores ordinary HF safetensors/configuration and the processor, plus `training_adapter.json`. The manifest records adapter version `bantai.grounding-dino.encoder-output.v1`, base revision, ordered labels/prompt/token IDs, configuration/loss contract, runtime versions and native/adapter source fingerprints. Adapter source hashes normalize only UTF-8 line endings so the same code survives Git's LF/CRLF conversion; substantive source changes still fail verification.

Offline reload passes empty loading diagnostics, exact model parameter/architecture hashes, identical processor input/target tensors, matching adapter context and exact decoder/total/component-loss equality. A deliberately changed manifest version is rejected. The processor retains the original cached default resolution; replay explicitly reapplies the saved 192/256 smoke override. The trained snapshot remains local and is not loaded by the existing inference pipeline.

**Weights alone do not activate this adapter.** Explicitly reconstruct `EncoderOutputAdapter(restored_model, restored_processor.tokenizer)` and call `verify_manifest` against `training_adapter.json` before adapted training. Calling the plain model with labels would still invoke its unchanged native objective; no global library behavior has been replaced.

`training_context.json` and the report explicitly say **`resume_supported=false`**. Optimizer/scheduler/scaler/RNG state, sampler position and exact continuation state are not saved or restored. This qualifies model/processor/adapter reconstruction, not interrupted-training resume. Production trainer/resume semantics need a separately approved implementation and test before a real-data run.

### Actual GPU memory and timing scope

Peak allocator use across case preprocessing, equivalence calls, diagnostic gradients and the two corrected steps: **2,933,199,360 bytes allocated (2.732 GiB)** / **3,124,756,480 reserved (2.910 GiB)**. Counter maxima can occur in different cases. Checkpoint replay, measured separately while the original model is also resident, peaks at **1,660,589,056 allocated / 3,219,128,320 reserved bytes** (about 1.547 / 2.998 GiB). These are PyTorch allocator peaks, not whole-device `nvidia-smi` peaks.

Final lifecycle time is **39.743 seconds**, including version checks, imports, hashing, preprocessing, equivalence, diagnostic traversals, optimizer steps, saving/reloading and final integrity checks. Step timings include loss/gradient/frozen-hash audits and synchronization, not ordinary training throughput. Equivalence forwards precede updates, and no separate training warm-up/steady-state benchmark was performed. This small frozen-backbone configuration fits locally; original 800/1333 resolution, fully unfrozen training, AMP, larger batches and long-run stability remain unrun.

## PASS/FAIL acceptance and remaining requirements

| Gate | Result |
| --- | --- |
| Selected image-conditioned encoder scores and undetached regression values | PASS |
| Existing native matcher/formula/normalization/masks/coefficients retained | PASS |
| Decoder detach boundary, query count, architecture and decoder inference unchanged | PASS |
| Finite nonempty/partial-empty/fully-empty objective and eight real optimizer steps | PASS |
| Nonempty encoder classification/box gradients, Adam state and actual updates | PASS |
| Expected zero empty regression; frozen parameters unchanged | PASS |
| Repeated adapter results and model/processor/manifest offline reconstruction | PASS |
| No installed-library/inference/base-snapshot/data-policy modification | PASS; preservation checks recorded below |
| Real-data training, convergence, metrics, full-resolution/AMP fit and resume | UNRUN / NOT QUALIFIED |

Regression suite: **144 tests run, 143 passed, one existing Windows privilege skip**; all **11 new adapter tests passed**. Fixtures verify native criterion dispatch/equality, selected-value gradient flow, unchanged parameters/decoder/input tensors, config/text/detach failures, finite empty paths, normalized-target validation, manifest tampering, CLI limits and source-fingerprint portability. They do not replace the actual pinned CUDA evidence.

Final preservation checks passed for both complete package-version inventories, nine protected inference/settings/taxonomy/policy/manifest/earlier-trainer files and three installed Transformers sources. Base-snapshot integrity flags, strict JSON, final implementation hashes and saved checkpoint artifact hashes all pass. The environment/weights/reports remain ignored; only the owner's existing detection settings remain as a separate local edit. All 219 local Markdown links resolve, CLI help and diff whitespace checks pass. After the qualification process exited, device usage was 876 MiB with the desktop active; this is a closure observation, not a whole-device peak or a memory-leak benchmark.

Remaining prerequisites before separately authorizing real-data fine-tuning: complete image/database rights review; accept strict-clean selection bias, rare-class support and leakage/held-out policy; preserve excluded/unlabeled handling explicitly; approve a real-data loader/trainer, complete resumable checkpoints and evaluator/class-span decoding; verify loader coordinates and preprocessing against actual annotations; measure intended real-resolution memory and establish a bounded experiment/evaluation protocol. Synthetic PASS alone does not clear any of these gates. No next stage is started automatically.


---

## 6. Phase 3 Milestone Register (D-035 through D-040)

- **D-035**: Training feasibility confirmed: Grounding DINO Swin-T fine-tuning is achievable with frozen backbone and trained head.
- **D-036**: Dedicated training virtual environment `.venv-training` created with PyTorch 2.x and Transformers 4.40.1.
- **D-037**: DetrHungarianMatcher empty targets diagnosis completed; verified handling of partial-empty and fully-empty bounding box batches.
- **D-038**: Custom Hungarian matcher and loss routing adapter designed (`scripts/training_adapter.py`).
- **D-039**: Detached decoder initialization contract formalized; query/text embeddings properly detached to prevent uncontrolled gradient leakage.
- **D-040**: Stage 3D adapter and synthetic qualification authorized and completed with 144 unit tests passing; zero installed library patches verified.
