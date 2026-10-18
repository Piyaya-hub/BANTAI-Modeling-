# Phase 2E — Functional Checking and Handoff

2026-10-18; authorized by D-034. Branch `feat/phase-2e-functional-checking` starts at fetched main `a3ec215` (PR #11 already merged). Local owner configuration and the untracked nested folder are preserved. This is a small qualitative baseline check, not training, a benchmark, measured accuracy, or a Phase 3 settings decision.

## Evidence and coverage

Codex inspected all three existing originals and all 15 experimental image/result combinations using source-verified, in-memory overlays. These are agent visual observations, subject to image ambiguity; they are not owner confirmation or ground-truth annotations. Windows Tk replay separately checked interaction callbacks and tables. No photos, generated result JSON, or annotated images are committed. No images were downloaded or collected.

| Original | Available scenario and qualitative content | Oriented dimensions |
| --- | --- | --- |
| `test#1.jpg` | Dominant isolated crumpled transparent bag, a small rectangular wrapper near the curb, and tiny scattered debris. Some tiny fragments cannot be distinguished confidently from stones/leaves. | 3000 x 4000 |
| `test#2.jpg` | Adjacent drink pouch and transparent bag, a small white paper/card, cigarette butt and separate pink wrapper. Supports adjacent/group-like items, mixed object types, nested predicted boxes and some overlapping/partly obscured material. It is not an unambiguous substantial garbage pile. | 3000 x 4000 |
| `test#3.jpg` | Several separated wrappers/paper-like pieces along a curb and a white piece near the wall; leaves and road texture are confounders. A cropped sack at the right edge may be in use rather than discarded. Supports small/distant and dispersed litter. | 5136 x 3984 |

Missing/unrun: clear substantial piles, unambiguous pile-plus-group mixed scenes, clean no-trash photographs, severe occlusion/night/weather variation, and agreed boundaries for multiple independent groups. A saved empty detector result is **not** a no-trash scene. Counts and group membership remain unresolved.

Original SHA-256 values, unchanged before/after all experiments and replay:

```text
test#1.jpg  8e4bc6101d2e21d45feac245ff01cda71aa0502155a97402df05c4c250949a4f
test#2.jpg  4a9ed5c4a7fff1bd4240d84f36968c55bd57ef33e702626b14c43f0755de2661
test#3.jpg  9685fe59d2e9c71c2cab07de392c6e7bc6843ffedb1646589c5e1819f3601f61
```

## Experiment plan and actual runs

The starting **owner experiment baseline**, not an accepted default, was `plastic bag. drink pouch. cigarette butt`, box threshold **0.20**, text threshold **0.10**, auto device and float32. This differs from the older two-description owner edit recorded during 2D. The checked-in default prompt/thresholds are unchanged; the local owner edit remains uncommitted.

Study artifacts: `results/phase2e-f601e7a2e3a6400db7b395c422559b58/`. `plan.json`, per-experiment config/logs, and `comparison.json` preserve the protocol, actual commands, manifests, complete environment snapshots and compact result comparisons. Each experiment has a separate UUID run under its named subdirectory. All 216 preexisting configuration/photo/JSON files captured at study start retained their hashes through the completed experiment checks. During final verification a subsequent local edit changed the prompt to `plastic bag. drink pouch. a cigarette butt`; this newer wording was preserved, not overwritten or tested here. The 215 captured photo/prior-JSON files remain unchanged. Results below apply to the saved baseline without the article, not the newer local wording.

| Experiment | Single factor changed from A | Box/text | Run ID | Retained candidates: photos 1 / 2 / 3 |
| --- | --- | --- | --- | --- |
| A-baseline | None | .20 / .10 | `65526f0148764151a93cb2610c1c8292` | 2 / 4 / 5 |
| B-concrete | Prompt: `crumpled plastic bag. empty drink pouch. cigarette butt. candy wrapper. discarded paper.` | .20 / .10 | `e81596b9c1974fbe845e9b78e92c03f7` | 13 / 12 / 12 |
| C-categories | Prompt: `a garbage pile. an individual discarded item. a group of individual litter.` | .20 / .10 | `2313ca0c538b407d88f47854fa751c12` | 9 / 16 / 9 |
| D-box-only | Box threshold only; A prompt retained | .35 / .10 | `40526fdf1a954c4a8cf37bce256476fb` | 2 / 2 / 3 |
| E-text-only | Text threshold only; A prompt retained | .20 / .25 | `473a8242810b4578910ec82a6f7d7bee` | 2 / 4 / 5 |

B changes the prompt as a whole; individual wording effects cannot be attributed separately. D and E each compare to A, not to one another. They deliberately examine the tradeoff in box retention versus decoded phrases; more boxes were never treated as better detection. No winning configuration was selected or silently written back.

Every run exited 0, completed all three images on **CUDA**, and recorded zero processing/display/persistence errors. Inference used `--no-display` to keep the experiments independent of viewer waiting, followed by separate native saved-result replay. Thus original inference records correctly say display `disabled`, while replay attempts say `shown`; the original records were not rewritten. Each of the 15 records retains 900 raw queries (13,500 total), including below-threshold results, in strict JSON. Scores/thresholds are decimal [0,1], never accuracy or calibrated probabilities.

A/D/E raw query and token scores match exactly, as do boxes that pass in both compared runs. For queries rejected only by D's raised box threshold, diagnostic pixel boxes differ by at most 0.000120 pixels: the existing extraction code uses upstream float32 pixel boxes for passing queries and Python arithmetic for rejected queries. An initial exact-all-boxes comparison exposed this distinction; it was inspected and documented, not changed or treated as a model-quality difference.

Pinned cached model/processor: `IDEA-Research/grounding-dino-tiny`, revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, safetensors, custom kernels disabled. Local-only loading succeeded, no fallback or new weights. Observed Windows 10 build 19045, Python 3.11.9 x64, torch 2.7.1+cu118, torchvision 0.22.1+cu118, Transformers 4.51.3, Pillow 11.2.1, Matplotlib 3.10.3. Separate `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader` reported GTX 1660 Ti, driver 472.19, 6144 MiB. Full installed-package snapshots were captured as observed, not installed or changed by this work; historical 2A transitive versions are not assumed to be today's versions.

| Run | Load ms | Warm-up ms (first photo only) | Inference ms: photos 1 / 2 / 3 |
| --- | --- | --- | --- |
| A | 1282.1 | 3696.3 | 893.8 / 868.3 / 1035.8 |
| B | 1276.3 | 3593.4 | 935.0 / 823.9 / 960.9 |
| C | 1322.6 | 3581.1 | 990.8 / 812.0 / 973.1 |
| D | 1349.0 | 3556.3 | 906.7 / 812.0 / 989.1 |
| E | 1362.9 | 3603.7 | 897.2 / 788.9 / 967.3 |

One measured repetition per image, four completed forwards per run including warm-up. CUDA timing is synchronized. The interval includes processor/transfer/forward/grounded postprocessing/candidate extraction; excludes decode, model loading, warm-up, duplicate/overlap review and display. These observations are not a performance benchmark or peak-memory measurement.

## Per-image qualitative comparison

Query IDs below are local to their run/image. “Missed” means visibly present but without a suitable retained box in this inspection, not a measured recall score. Material identity and whether an object was discarded can remain uncertain.

| Photo | A baseline | B concrete descriptions | C category descriptions | D box / E text changes |
| --- | --- | --- | --- | --- |
| 1 | q0 encloses the bag plausibly (.8629), q1 the small curb wrapper (.3779). Both decode `plastic bag drink pouch`, an ambiguous combined phrase inappropriate for confidently naming the small rectangle. Tiny debris remains largely unboxed. | q0 `crumpled plastic bag` is a better phrase match. Many near-duplicate bag and curb-wrapper boxes inflate 13 candidates. q4 `discarded paper` spans the painted road stripe: a clear background false detection. q15 `cigarette butt` encloses a tiny brown sliver; identity is uncertain at this scale. | Broad `a garbage pile a group` boxes span curb/background, despite no clear pile. Multiple bag/wrapper boxes overlap. Raw mixed phrases remain unmapped. | D retains the same two boxes. E retains their geometry; q1 becomes `drink pouch`, still not a reliable description of the small wrapper. Neither change resolves missed tiny debris. |
| 2 | q0 pouch, q1 bag, q2 enclosing both, q3 pink wrapper. All decode `plastic bag drink pouch`, so boxes alone overstate usefulness. The cigarette butt and white card have no separate suitable box. The enclosing q2 is not evidence of a mapped group or membership. | q0 `cigarette butt` plausibly covers the butt; q10 adds the white card but with a mixed phrase. q1 identifies the bag. q3 incorrectly calls the drink pouch `crumpled plastic bag`. Pink-wrapper phrases mix pouch/wrapper terms. Near-duplicate/nested boxes remain, including q13 across the adjacent objects. | q0 `a group` is incomplete/unmapped; q8 `a garbage pile` (.3404) maps lexically but spans adjacent objects that do not clearly form a pile. q5 spans almost the entire image, an overbroad pile/group prediction. Repeated individual boxes cover the pouch and pink wrapper; separate bag classification remains mixed. | D removes enclosing q2 and the real pink-wrapper q3, retaining only pouch/bag. E retains all four boxes but q3's decoded phrase becomes the explicit empty string `""`; no phrase is invented to fill it. |
| 3 | Five boxes: foreground blue packet, cropped right-edge sack, red/white wrapper, white curb piece and beige crumpled object. Identical `plastic bag drink pouch` phrases poorly distinguish them. The white litter near the wall is missed. Whether the sack is trash is unresolved. | Adds q12 on the wall-side white piece (`discarded paper`, .2016), close to the box cutoff; q10 `candy wrapper` is plausible on the red/white item. Repeated foreground packet/red-wrapper boxes and combined phrases remain. Sack false-positive status is still uncertain. | `a group` boxes span portions of the littered gutter, but stay incomplete/unmapped. Large pile/group boxes cover much of the walkway/wall: poor region specificity. q11 `discarded item` encloses a slab-like piece near the wall; whether this is litter or part of the damaged surface is uncertain. No exact mapped group is returned. | D loses the white and beige curb objects, retaining three boxes including the uncertain sack. E keeps five boxes; white piece becomes `plastic bag` and beige object's phrase is empty. Higher text threshold does not repair semantic correctness. |

All A/B/D/E candidates remain unmapped under the unchanged exact-alias policy. C has five mapped candidates across its three images (four individual-litter candidates and one pile candidate), none mapped to `group_of_individual_litter`. Mapping success is not visual correctness. Near duplicates remain intentionally because only identical raw-box/normalized-phrase duplicates are collapsed; no grouping, NMS, forced aliases or memberships were added. Broad group-like boxes exist in C, but reliable category/boundary behavior remains unmet. Physical-object and independent-region counts remain null.

## Display correction and acceptance status

Human feedback during this phase: the owner had checked the current viewer and reported that labels still needed to be smaller and some were absent on small boxes and boxes inside other boxes. This is a **failed prior visual check**, not blanket confirmation of zoom/pan, tables or closure.

Confirmed reproduction on saved owner run `4cd1bcf5e9904400a4e6dd4f3403ef9f`, at an 11 x 7.5-inch, 100-dpi constrained canvas: main omitted photo 1 q1; photo 2 q0/q2/q3; photo 3 q0/q2/q3/q4. The corrected renderer placed all these labels. No predictions were recomputed for this comparison.

The previous box-width-only layout and border-collision rejection hid these phrases. Labels now borrow horizontal space within axes and search upward to clear enclosing borders/earlier labels. Query IDs remain present and displaced labels use thin connector lines. Default fonts are now **7.5 points with 7-point fallback**, following the owner's smaller-font instruction; explicit values remain configurable in [6,16]. Raw phrases, stored boxes/scores and the owner's prompt/threshold file are untouched. Placement is recalculated on zoom/pan. The complete table/zoom fallback still applies if a crowded view has no available label space; long connector lines and dense scenes still need human review.

| Check | Outcome and evidence type |
| --- | --- |
| All five new three-photo batches replayed in native Windows Tk | Passed: one tab, complete repeated sections, all passing table cells match saved JSON, no labels omitted at checked initial fit, zoom/pan mouse callbacks change axes limits, original extents/boxes stay fixed, Home/scrollbars/wheel work and window closes normally. Automated runtime evidence, not human approval. |
| Empty-result image replay | Passed on historical saved empty photo alongside detections. Empty original still renders. This does not prove absence of trash. |
| Select/copy | Passed on initial two-image replay, restoring previous text clipboard. Selection passed on all experiment batches; copying skipped there to preserve unavailable/non-text clipboard. |
| In-memory agent overlay inspection | All 15 combinations reviewed; labels and boxes visible at inspected fit. Images were encoded only in memory for inspection, with no annotated file or temporary image. This is separate from a desktop screenshot/human check. |
| Final human label/readability acceptance | **Pending** after correction. Earlier reported small/nested-box failure is recorded above. No new owner confirmation of the corrected version is inferred. |
| Synthetic errors/no-detection/continuation/identity checks | Final suite: 92 tests in 6.841 seconds, 91 passed, one Windows symlink-privilege skip. Includes invalid/all-invalid candidates, decode/model/display errors, batch continuation, mismatch rejection, strict JSON and pending/final lifecycle. Fixtures are not real-photo model evidence. |
| Original/result protection | All 215 captured photo/prior-JSON files unchanged; the config passed experiment-time preservation checks and was subsequently edited locally as described above. Native replay imports no model modules and permits zero socket connections. Per-inference run folders contain only JSON; all original inference/replay schemas validate. Figures are cleared and image copies released on closure. |
| Real-photo errors | None in the five measured runs. No no-detection outcome occurred in these 15 images. Controlled failures/empty results are covered separately, not claimed as observed model outcomes. |

Native logs are in `<study>/<experiment>-native.log`, with each separate replay UUID. Original inference status remains independent of replay status; owner visual verification remains pending. See the [viewer contract](SCROLLABLE_VIEWER.md) and [2D history](PHASE_2D_CHECKS.md).

After the persistence fix, a final native four-image replay checked all three records from the owner-reported `4cd1bc...` run plus the historical empty result. All initial-fit labels were placed (2/4/5/0 boxes), matching tables and zoom/pan/scroll/closure passed, and originals were unchanged. Replay attempts: `02ecfa838b394a23b08d92028284c740`, `12d76ba87acd4de9b2df37791a0a2dcd`, `6899913b378d40e49ff2714458f79953`, `e81ffa33d291498fbde6ad41012067b2`. Selection passed; clipboard copy was skipped to preserve unavailable/non-text content.

The strengthened Home-restoration assertion also passed on the historical detection/empty pair (`07066f40054741e2bc5028bfee7b03ab`, `f5552c3b221442a295c63994e937d236`). Final compilation and `git diff --check` passed; documentation validation resolved 140 relative Markdown links and five strict JSON examples. Study JSON and functional result schemas validated. Protected inference/postprocessing/input/result/pipeline files, dependency pins and ignore rules match main; the only persistence change is the bounded replacement retry below.

## Observed Windows persistence failure and bounded fix

After the initial 90-test suite passed, later 91-test runs failed on rapid atomic `run.json` replacement with Windows error 5 (access denied), including outside the sandbox. This was reproduced in a minimal synthetic pipeline; three controlled probes each succeeded after a 50-ms retry. The process holding/denying access was not identified; antivirus or filesystem indexing is not asserted as fact.

Explicit atomic replacement now retries Windows access/sharing errors 5/32/33 at 50, 100, 200 and 400 ms, then fails normally (at most 0.75 seconds of waiting). It never deletes/truncates the old destination or weakens no-overwrite initial publication. A new fixture proves transient recovery, bounded persistent failure, unchanged previous predictions/JSON and temporary-file cleanup. Final command `.venv/Scripts/python.exe -m unittest discover -s tests -q` passed the 92-test suite as recorded above. No dependency or inference logic changed. The five model runs had already completed successfully; this fix changes persistence resilience, not their detections or timing observations, and did not trigger model reruns.

## Local commands and next boundary

Use the existing environment from the repository root. Do not reinstall. The experiment helper snapshots the **current** owner config; rerunning it after owner edits is a new experiment, not an exact reproduction of the recorded baseline. Recorded per-experiment configs in the study preserve this run's settings.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe tests/phase2e_experiment.py
.venv/Scripts/python.exe tests/windows_replay_check.py
```

The experiment helper invokes the existing CLI five times with `--mode infer --config <snapshot> --local-files-only --no-display`, saves new UUIDs, compares hashes and validates results. To repeat just A using its recorded settings, use:

```powershell
.venv/Scripts/python.exe scripts/detect_images.py --mode infer --config results/phase2e-f601e7a2e3a6400db7b395c422559b58/A-baseline.json --local-files-only --no-display
```

For human inspection without inference (omit the second pair for single-image replay):

```powershell
.venv/Scripts/python.exe scripts/detect_images.py --mode replay --result results/4cd1bcf5e9904400a4e6dd4f3403ef9f/da4b761153fda0359330cde3950d3b4cc1a2247db5bbe6618aec59c092e4afc0.json --image "data/input/test#2.jpg" --result results/ea5da8d9bdc840bb924d5b20c2f1733c/9d75f19ceb224a47917ee88b751c4b74928536296000016824365d6cde7787e6.json --image "data/input/test#3.jpg"
```

To replay experiment records, replace each result with `<study>/<experiment>/<run-id>/<record>.json`, pairing with the unchanged source named inside it. `tests/windows_replay_check.py` also accepts any list of result paths and resolves their recorded sources, auto-checks callbacks and closes. The default command checks a historical detection-plus-empty pair. Private originals/results must exist locally; do not fabricate them in another checkout.

**Recommendations for owner Phase 3 review, not decisions:** retain these settings as unvalidated experiments; do not promote B just for higher counts or D/E for cleaner output. Review small-item misses, incorrect mixed phrases, near duplicates, uncertain sack/slab detections, and pile/group boundary confusion before choosing a direction. Consider separately authorized single-description prompting to isolate phrase interactions and owner review of available scenes; missing pile/no-trash coverage remains a limitation. Any grouping or model improvement needs a later explicit scope decision. This work stops at evidence and handoff: no Phase 3 optimization, grouping, model changes, training/fine-tuning or application modules.
