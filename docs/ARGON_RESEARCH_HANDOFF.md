# Paste into Argon — BANTAI planning and research handoff

You are the **technical planner, research reviewer, experiment designer and decision-support partner** for **BANTAI's trash-only Grounding DINO Tiny modeling component**. Codex is the implementation agent on the user's Windows repository. The owner reviews and authorizes every experiment or code change.

**Immediately read `BANTAI_MASTER_CONTEXT.md` and `CONVERSATION_CHRONOLOGY.md`,** plus the original proposal excerpt when product context matters. Do not treat this compact prompt as a substitute for those files. Treat conversation-derived statuses as historic evidence to be verified by Codex in the actual current worktree.

## Essential current state

- Original larger concept: “BANTAI: An Intelligent Agent for Street Issue Detection and Hotspot Prioritization,” including potholes/road damage, incident deduplication, frequency-based hotspots and a web UI. **Current modeling repo scope: trash-only still-image Grounding DINO Tiny; no app, video, incident or hotspot code.**
- Repository: `Piyaya-hub/BANTAI-Modeling-`; local Windows directory historically `C:\Users\Justine\Documents\Python\Project Image-object Detection`.
- Pinned model: `IDEA-Research/grounding-dino-tiny` revision `a2bb814dd30d776dcf7e30523b00659f4f141c71` via Transformers. TACO experimental classes: cigarette, plastic_bottle, plastic_bag, plastic_wrapper, plastic_film, drink_can, plastic_cup, styrofoam_piece. Do **not** conflate these with app-level Individual Litter, Group of Individual Litter, Garbage Pile.
- TACO: 221 training images/397 objects; 47 validation/75 objects. Test split untouched. The owner's separately sourced unannotated photos were for functional checking, **not TACO training**.
- A critical padded-vs-valid-image target coordinate bug and bad evaluation rescaling were discovered and fixed. Corrected COCO v2 pretrained AP `0.111576`, AR100 `0.363077`, AP50 `0.158253`. Historical faulty-target weights should not be resumed under corrected source.
- Corrected training, seed42 LR1e-5 Epoch1 yielded strongest validation AP `0.203932`, AR100 `0.396766`; Epoch2 AP `0.084500`. Seed42 LR5e-6 Epoch1/2 AP `0.137099/0.145952`. Seed123 LR1e-5 Epoch1/2 AP `0.140478/0.152880`, LR5e-6 `0.137919/0.113018`. Seed effects reverse neat learning-rate narratives. No fine-tuned checkpoint promoted.
- Cross-seed saved visual comparison is available. Image777 has GT259 wrapper + seven GT278–GT284 cigarette annotations. Wrapper often localized but class incorrect; six cigarettes mostly unmatched at IoU>=0.5. GT284 obtains a qualifying cigarette match at Epoch2 for both examined 1e-5 seeds. Selected images may have ambiguous material GT; **do not rewrite validation GT just because detections fail**.
- Latest small-object/classification audit of 47 images/75 GT/nine model states: **13/24 cigarettes have shortest resized GT dimension <4 px at the current 256/384 resize**; qualifying coverage 0–2/13 across states; 18–23/24 cigarettes lack qualifying localization. Localization-with-wrong-class: pretrained 20/75, fine-tuned 21–27/75 at score>=.05/IoU>=.5. Class-related misses particularly cans 6–7/7 and wrappers 5–8/11. These are diagnostic categories, not official AP. Image777 valid resized dimensions 350x256, padded384x384.
- **Next proposal, NOT executed at the handoff date:** controlled **inference-only** resolution comparison **256/384 vs 384/576** using the **same seed42 LR1e-5 Epoch1** checkpoint. No training, no change to model weights, taxonomy, prompt, thresholds, evaluator or dataset. Need exact baseline reproduction, memory pilot for higher res, per-class AP and same/any-class coverage and careful report; halt if GPU fit or geometry preflight fails. Full protocol is in `NEXT_EXPERIMENT_RESOLUTION.md`.

## Your operating behavior

1. **Reason before implementation.** Begin with a short diagnosis of what is known, what is suspected, what evidence would falsify the hypothesis and the single variable being isolated.
2. **Honor phase boundaries.** Avoid switching to app backend/frontend, video, live CCTV, potholes or hotspot AI while modeling is the agreed focus. Do not silently introduce clustering, augmentation, higher-res training, model-family switch, new dataset, unfreezing, or automatic threshold tuning.
3. **Ask Codex for the actual evidence needed.** Specify exact source files/records to inspect; use immutable run IDs, hashes, tests, GPU info, official metrics, supplementary diagnostic metrics and stop conditions. Do not treat Codex prose as proof if underlying context is ambiguous.
4. **Prefer controlled experiments.** Keep test split untouched; keep original images/checkpoints protected; acknowledge the risk of fitting many design decisions to 47 repeated validation photos.
5. **Separate error modes.** No qualifying candidate vs poor localization vs good localization with wrong label vs confidence/ranking vs dataset ambiguity. Contrast category AP, IoU and score distributions; model scores are not probabilities of correctness.
6. **Do not conflate the two pipelines.** Normal model functional-inference GUI/JSON and formal TACO training/evaluation use different policies and goals. Do not apply TACO tweaks to production/default detection without an explicit reason.
7. **Protect provenance.** Old source-pinned checkpoints cannot be freely resumed after code changes. Evaluation-only replays have a governed path; training resume strict verification remains intact.
8. **Use the owner–planner–implementer loop.** Propose and explain a plan; supply a detailed copyable Codex task when requested; wait for owner go-ahead for expensive operations; assess returned results; do not make retrospective experiments sound completed when they were only proposed.
9. **Presentation:** user appreciates direct, practical, beginner-friendly explanations paired with exact engineering detail, not hollow praise; avoid overclaiming causal explanations or declaring small AP gains decisive. Prompts should be methodical, acceptance-driven and intentionally scoped.

## Communication contract with Codex

For each approved task, provide:
- Goal and explicit scientific hypothesis.
- Trusted saved model/run identity and why chosen.
- One isolated variable; fixed comparison settings.
- Preflight gates, test requirements and stop criteria.
- Exact expected outputs and local report location.
- Data/privacy/rights/test-split/source-integrity safeguards.
- Explicit prohibition on automatic commits, pushes, merges, checkpoint promotion, or extra experiments.

Demand a finish report with actual commands executed, counts/hashes/tests, metrics, artifacts, comparative tables, limitations, and whether any requirements were unmet. If implementation diverges from planning, request clarification rather than allowing hidden parameter changes.

## Your first task on this transfer

**Do not invent a completed resolution experiment.** Assess `NEXT_EXPERIMENT_RESOLUTION.md` for rigor and feasibility, especially whether (a) current `256/384` baseline is reproduced from the exact retained checkpoint, (b) `384/576` preprocessing is compatible with model positional/mask handling and GPU, (c) a single evaluation-only variable is isolated, (d) comparison remains apples-to-apples across original GT and diagnostic size bins, and (e) evidence is enough to justify *possibly* later training at higher resolution. When the user approves, turn it into a precisely scoped Codex execution request.

If any task or file is inaccessible, explicitly say what was not available; **never assert you directly inspected a Windows-local path that you cannot access**.
# IA Detection Model Project — chronological conversation extraction
**Status:** A topic-and-decision extraction from visible messages, prior-conversation summaries, user-reported Codex outcomes, and attached project text files. **Not an exact verbatim export of all project chats.** References to Codex outcomes mean user-reported local outcomes, not fresh repo verification performed by this handoff.

## Project chat titles/topics recovered

1. **AI Project Planning** (2026-10-16): First substantial AI-assisted project; user wanted ChatGPT for planning/questions, Antigravity for execution and Codex where helpful, with industry practices and shared GitHub repo collaboration.
2. **Plan Detection Phases** (2026-10-16): Phase-by-phase construction of Grounding DINO Tiny trash-only still-image detector; asked to distinguish phases and establish an adjacent support session (named “2.0 Support session”).
3. **AI Project Phase 1 Planning** (2026-10-17): User would copy team-provided Google Drive photos manually; wanted visible single-image boxes without duplicating originals, JSON/scores and detection summary, and a clear implementation architecture before coding.
4. **Project Phase 2 Implementation** (2026-10-17 onward): Iterative prompts for Codex to build GUI and in-memory visualization, keep coordinates beside image, put small decoded labels over boxes, scrollable image/summary/token/config tabs; then move from Phase 2A/B into 2D and 2E, validate pushed changes. Later asked whether fine-tuning a single category (e.g. plastic wrapper) until confidence increases was a good strategy.
5. **Stage 4 training and evaluation investigations** (2026-10-18–23): A succession of TACO training, correction, provenance, visual verification, seed/LR replication and object-size audits. The currently visible thread includes the detailed numerical results and proposed next high-resolution inference-only experiment.

## Oct 16 — problem decomposition and governance

- **Original product proposition:** “BANTAI: An Intelligent Agent for Street Issue Detection and Hotspot Prioritization”; broad system for municipal/barrangay garbage, potholes, road damage, consolidated incident reports, validation, reporting frequency rankings, prerecorded video and web dashboard.
- **Modeling repo boundary selected:** focus **only on trash from still images** with pretrained Grounding DINO Tiny. No premature pothole/damage model, video stream, full-web UI, incident DB, hotspot agents or severity predictions.
- **Phased approach:** Phase 0 foundation, Phase 1 planning/evaluation criteria, Phase 2 pretrained functional inference, Phase 3 optimization decision, Phase 4 conditional improvement/readiness.
- **Multi-region requirement:** one photo can contain separate instances, a cluster and a pile; annotate/identify separate regions. Text-prompt detection first. Consider spatial clustering/DBSCAN only if later evidence justifies grouping; don't assume it proves a garbage pile.
- **AI collaboration:** ChatGPT plans and validates, Codex/Antigravity implement through GitHub. Prevent AI tools from silently making conflicting updates. User wants to understand implementation rather than copy code without review.

## Oct 17 — planning and first working inference

- **Phase 0 completed/merged historically**; main planning documents included architecture, handoff, decisions and testing. Phase 1 selected HF `IDEA-Research/grounding-dino-tiny` with pinned model/environment later verified by executed run.
- **Data:** Google Drive ZIP photos manually downloaded/extracted to local project. These had **no boxes/classes**; designed for qualitative functional tests, not supervised training. Data images must remain unchanged.
- **Expected functional output:** JSON results with scores/classes, one in-memory image showing boxes, percentage-like summaries clearly labeled as score/count summaries (not accuracy), raw phrase/token evidence, not duplicate annotated image copies on disk.
- **Pretrained functional experiments:** thresholds changed (`0.20/0.10`, `0.35/0.25`, `0.60/0.50`) for box/text; raw overlapping compound labels such as `plastic bag drink pouch` and aliases mapping to `unmapped` discussed. Separate prompt `a cigarette butt.` was proposed to isolate recognition from competition among descriptions. In one logged 3-image `plastic bag` run at 0.35/0.25, no detections; this was a functional outcome, not AP.
- **Execution style:** ask Codex to update particular files after each review, validate before moving on; preserve deterministic model version, image hashes, prompts, thresholds and output contract.

## Oct 17–18 — Phase 2A–2E interface and evaluation handoff

- Phase **2A** built/checked environment readiness and pinned model access.
- Phase **2B** implemented or prepared local image processing/config, deterministic run/result structure and JSON.
- Phase **2C** implemented inference, prompt/threshold filtering, phrase mapping and validation.
- Phase **2D** implemented a scrollable Tk/TkAgg presentation with **Image**, **Detection Summary**, **Text-Token Details**, **Configuration** areas. User explicitly requested raw decoded phrase above box, font small enough not to obscure box, coordinates beside the image/axes, zoom/pan, and threshold-only display. Historic PR #11 reported merged at `a3ec215`.
- Phase **2E** aimed to inspect real owner-supplied photos with the pretrained model; compare prompt combinations, permissive/moderate/strict thresholds, duplicate/missed boxes, human-visible errors and GUI/JSON consistency. Do not conflate Phase2E with later formal TACO evaluation and training.

## Oct 18–19 — initial Stage 4 TACO training

- TACO subset established for formal 8-class localization/classification with 221 training photos/397 objects and 47 validation photos/75 objects. Untouched test split excluded.
- Grounding DINO Tiny fine-tuning adapter; frozen language and vision backbones, trainable selected detection parameters, AdamW 1e-5, 2 epochs, no augmentation or LR schedule, image resize 256/384 and square 384 pad, FP32, batch1, seed42. A three-step memory pilot was discarded before full run; 442 full optimizer updates.
- Original run ID `taco-1698bcb46b85435786dd2a199910910f`; around 7.49 minutes; historical peak allocated VRAM 2.382 GiB. Loss fell, pre-correction AP apparently catastrophic; no checkpoint promotion, Stage4 PR #18 reported unmerged.
- Assistant and Codex investigated whether the problem was confidence suppression, epoch count, dataset, prompt confusion, or an implementation defect. Critical to note: training loss reduction cannot be equated to AP improvement.

## Oct 19–20 — discovery and correction of coordinate defects

- **Step1A audit:** training target boxes normalized to 384 padded canvas instead of each valid image; native Grounding DINO normalized proposals valid-image frame. Evaluation also applied extra padding-ratio scaling to predicted boxes, shifting/enlarging them; self-referential tests erroneously passed. Scores/token mapping and COCO metric algorithm were separately verified.
- **Step1B fixes:** corrected `scripts/training_smoke.py`, `scripts/taco_training_data.py`, `scripts/taco_evaluation.py`, added `tests/test_taco_coordinates.py`; evaluation protocol v2. Corrected pretrained baseline AP **0.111576**, AP50 **0.158253**, AP75 **0.135975**, AR100 **0.363077**. Small-object AP 0.00. Original metrics not comparable to v2. Exact baseline reproduction was repeatedly enforced in subsequent preflights.
- **Step1C:** source-fingerprint safety blocked historical resume. Added **evaluation-only** verified compatibility to load historical checkpoints against corrected evaluator without changing strict resume verifier. Historic epoch1 AP=.126627, AR100=.275363; epoch2 AP=.066215, AR100=.235395. Saved checkpoints still trained under wrong targets.
- Score suppression: pretrained 216 raw queries >=.25 (max score .596656), historical epochs both 0 raw queries >=.25 (max .184526/.123536). Cause not established by correction alone.

## Oct 20 — visual comparison and corrected training restart

- Created four-panel saved-prediction validation viewer: GT, pretrained, historic E1, historic E2. Image IDs chosen for diagnostic reasons: 42,113,200,31,1096,252,1014. At score .05/IoU>=.5, pretrained matched 28/75 GT, historic E1 17/75, historic E2 14/75 (diagnostic, nonexclusive criteria and protocol details matter).
- Owner D-046 mandated **terminal-only default**. An explicit `--gui` restored existing viewer without changing that default; subsequently saved-run choice and `--per-object` mode added. Validated Windows launch, no image exports.
- Corrected two-epoch restart **from pretrained**, not historical weights, same settings except fixed valid-image target geometry. Seed42 LR1e-5: E1 AP=.203932/AR100=.396766, E2 AP=.084500/AR100=.211229. The first corrected epoch became strongest observed validation checkpoint; second regressed while validation loss declined. Retained all checkpoints, no promotion.

## Oct 21 — diagnosis of epoch regression and isolated LR change

- Fine-tuned E1→E2 `1e-5` seed42: any-class IoU coverage 55→57 GT (increased) while correct-class coverage 36→30 (decreased), consistent with wrong-category matches rather than total absence of localization. Bags and styrofoam AP collapsed on some well-aligned examples. A blanket confidence-suppression, overfitting or excessive-LR root-cause claim was not established.
- Codex added `--learning-rate` default1e-5 and verified passing tests/optimizer param groups/source context; strict source fingerprints preserved. Verified exact baseline before optimizer updates.
- Seed42 LR5e-6 isolated experiment: E1 AP=.137099/AR100=.330432, E2 AP=.145952/AR100=.302977, preserved 442 updates +3 pilot. Correct-class coverage still 32→30. A single seed made lower LR appear steadier; planner called for replication instead of declaring victory.

## Oct 22 — two-seed replication, cross-seed review, and GT object drill-down

- Added `--seed` default42 with RNG/sampler/provenance tests. New seed123 trained LR1e-5 and LR5e-6 separately: 884 combined full updates + six discarded pilot, all data and checkpoints protected.
- AP table (protocol v2):

| Seed | LR | E1 AP | E2 AP |
|---:|---:|---:|---:|
|42|1e-5|.203932|.084500|
|42|5e-6|.137099|.145952|
|123|1e-5|.140478|.152880|
|123|5e-6|.137919|.113018|

- **Interpretation:** seed123 reverses apparent “lower LR prevents second-epoch AP collapse” pattern. Two seeds not enough to claim LR superiority.
- Cross-seed analysis selected 20 validation images with explicit human-review questions. Cigarettes and film improved in several aggregate class metrics; wrapper recall remains comparatively lower, bag/styrofoam variable with little GT support. Four run choices available in viewer.
- Per-object viewer added annotation select/navigation/auto-zoom, all eligible overlaps, score/class/IoU tables, and saved-run selection. Visualizer default remained terminal-only; runs/read-only data protected.
- **Image777 human review:** GT259 plastic wrapper plus seven cigarette GT278–GT284. Wrapper near curb appears real and well-localized by several candidates, often wrong label; pretrained has correctly labeled qualifying box, seed42 E2 recovers it, seed123 E2 does not. Six GT278–GT283 cigarette annotations usually have no qualifying candidate IoU>=.5 across viewed seeds; GT284 qualifies at Epoch2 under both seed42 and seed123 `1e-5` runs. Some pale objects/boxes merit independent human material-identity review; don't rewrite GT merely because model fails. Image-777 non-exclusive correct-class coverage: pretrained 1/8; seed42 E1 0/8, E2 2/8; seed123 E1 0/8, E2 1/8. These are *per-image diagnostics*, not AP. Review screenshots do not prove whole-dataset results by themselves.
- Full-set small-object/class audit: 47 images /75 GT /9 states. 13/24 cigarettes shorter than4 pixels in valid resized image; only0–2 qualifying among these13 per state. No qualifying localization for18–23/24 cigarettes across states; 20/75 pretrained and21–27/75 fine-tuned annotations have any-class localization but wrong-class-only at .05/.5. Cans and wrappers consistently have class-assignment difficulties. Image777 is valid resized 350×256 padded 384×384, not a native 384×384 stretch.

## Oct 23 — transfer request and latest pending experiment

- The user is transferring **planning/research to Argon** and **implementation to Codex** and asked to extract this Project-folder conversation history. This transfer packet was created to preserve scope, experiment provenance, explicit owner approvals, methodology and next task.
- **Next unexecuted proposal:** keep seed42 LR1e-5 corrected E1 frozen, compare *inference-only* `256/384` resize/pad384 with `384/576` resize/pad576 on the exact 47-image validation set. Verify standard AP .203932 and AR100 .396766, memory/coordinate preflight, fixed scores/IoU/evaluator, small-object category coverage, resource cost and report. No new training and no test access. See `NEXT_EXPERIMENT_RESOLUTION.md`.

## Changes in plans that must not be lost

- **Initially no training** → after pretrained functional checks, formal TACO fine-tuning became a separately authorized Stage 4 research track. Do not reframe original exploratory photos as annotated training data.
- **Original historical Stage4 AP tiny** → corrected evaluation protocol completely changed the valid baseline; earlier numbers are obsolete for comparisons.
- **Score suppression hypothesis** → corrected run revealed correct-class vs any-class candidate disparity; broader cause remains unresolved.
- **“Lower LR stable” hypothesis** → seed123 refuted a simple across-seed conclusion.
- **“Only labels are wrong” hypothesis** → full size audit shows severe localization gaps in very small cigarettes as well.
- **“Higher resolution would fix it”** remains **untested** as of this transfer; next step is **inference-only** compatibility and quality comparison, not training at higher resolution.
- **D-046 terminal-only default** → optional GUI explicit `--gui`, diagnostic `--per-object`; do not remove defaults or automatically save extra annotated images.

## Sources and limits

Sources incorporated: direct user/Codex status messages and screenshots in currently available project thread; retrieved prior conversation summaries of named project sessions; attached `Concept-Proposal.txt`, `Trash Detection Support.txt`, `Pasted markdown.md`. Some older chats were not available as full raw text; exact dialogue and all source-file commit histories cannot be reconstructed verbatim. Review original chats or Codex local reports if exact wording/sequence matters. Most results were user-relayed from Codex, not fresh tests run by the document author.
# BANTAI — canonical project context extracted from IA Detection Model conversations
**Handoff snapshot:** 2026-10-23 (Philippines, UTC+08)  
**Status:** Detailed reconstructed conversation context; verify current code/artifact state locally before work.

## 1. Project identity and scope separation

**Project:** BANTAI: An Intelligent Agent for Street Issue Detection and Hotspot Prioritization.  
**Course context:** CS 403 — Intelligent Agents, TIP Quezon City, first semester AY 2026–2027.  
**Repository:** `https://github.com/Piyaya-hub/BANTAI-Modeling-`  
**User's Windows working directory (historically reported):** `C:\Users\Justine\Documents\Python\Project Image-object Detection`.

The uploaded **Concept Proposal** describes a *larger system*: barangay incident reports, garbage piles, potholes, road-surface issues, user-uploaded photographs and prerecorded simulated-CCTV footage, incident deduplication, frequency-based hotspot ranking with recency tie-breakers, and accessible web dashboards. Those are the overall product goals, **not the current repo's implemented scope**. In the current repo, focus on **still-image, trash-only Grounding DINO Tiny modeling and experimental evaluation**, with no live-video integration, app UI/API, database/incident management, pothole detector, automatic hotspot ranking or production deployment unless explicitly and separately authorized.

The source proposal envisages RF-DETR Nano and Grounding DINO Tiny as possible detectors for multiple anomaly types. The modeling repository's adopted detector is **Grounding DINO Tiny**. Distinguish three **high-level output concepts** — `Individual Litter`, `Group of Individual Litter`, `Garbage Pile` — from the **eight TACO fine-tuning labels**: `cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`. High-level pile/group status cannot be assumed to follow directly from one eight-class detection; any clustering/grouping requires evidence and authorization. DBSCAN was discussed only as a conditional possibility, not approved as a reliable garbage-pile classifier.

Original project team roles in attached proposal: Sophia Mae Aguila (integration/QA), Joshua Philip Gumabon (UI/UX), Justine Mark Lalamoro (model trainer/developer), Samuel Rei Santiago (PM/scripting and analytics), Jhun Deniel Soriano (full-stack/lead web). Avoid conflating other members' application scopes with this model repo.

## 2. Working method and owner constraints

The user plans and reviews with an AI assistant and has Codex execute in the Windows repository; earlier Antigravity also executed phases. Now **Argon owns planning and diagnostic review**, **Codex owns implementation and local checks**, and the user authorizes each step. Prompts should be ready-to-paste, staged, precise, testable, with explicit change and stop boundaries. Explain technical reasons plainly; never treat code-generation output as independently verified until checks are supplied. User aims to learn what architecture and methods do, not rely on blind “vibe coding.”

Core engineering expectations established repeatedly:
- Read and follow `AGENTS.md`; inspect actual git status and local changes; do not assume remote GitHub includes uncommitted work.
- Keep original photos unmodified. Show bounding boxes in memory rather than automatically producing duplicates. JSON detection results are useful; preserve provenance and timestamps.
- The owner's original Google Drive/manual-image folder contained **unannotated exploratory photos for functional checks**, not formal supervised training data. The separately prepared **TACO dataset** is the fine-tuning/evaluation dataset.
- Do not put raw TACO photographs, annotations, checkpoints, rights documents, or private predictions into a cloud service/public PR unless specifically authorized. Local academic-use record does not by itself establish comprehensive downstream legal clearance.
- Train/validation/test boundaries are strict. **Never open the untouched test split** as part of model selection or exploratory tuning.
- Changes to source code may invalidate strict source-pinned training-resume contracts. Do not bypass fingerprint checks or rewrite historic manifests; a separately audited **evaluation-only compatibility/provenance path** exists.
- Do not quietly install/update dependencies; change prompts/taxonomy/resolution/precision/architecture; promote a checkpoint; commit/push/merge; train, or launch extra experiments beyond approved bounds.
- Scores are not accuracy percentages. Keep box threshold, text-token threshold, raw decoded phrase, final category, IoU, AP and recall conceptually separate.

## 3. Model, runtime and datasets

**Pinned model:** Hugging Face Transformers `IDEA-Research/grounding-dino-tiny`, revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. Archived early runs reference Transformers `4.51.3`, FP32 on CUDA, Windows. Do not assume current interpreter/GPU package versions from this archive; Codex must inspect. There have been inconsistent GPU identifiers mentioned across conversations; use current local CUDA introspection, not prior model names.

**Normal pretrained functional-inference path (separate from TACO evaluations):** `.venv\Scripts\python.exe scripts\detect_images.py --mode infer --input data/input --results results --local-files-only`. The original script opens an in-memory GUI and emits JSON. Early exploratory prompts included `plastic bag`, `a cigarette butt`, and multiple dot-separated phrases. Example thresholds: permissive `box=0.20, text=0.10`, moderate `0.35/0.25`, strict `0.60/0.50`. Lower thresholds may show more low-quality candidates and muddled labels (e.g., “plastic bag drink pouch”). `aliases-v1` exact phrase-to-category mapping can yield `unmapped`; this is distinct from a model recognition failure. The experimental TACO eight-class protocol is **not the same** as normal functional inference's phrase mapping.

**TACO research dataset:** 221 training images (397 annotations), 47 validation images (75 annotations); collectively 268 images in train + validation. At least one untouched test partition exists, but its count is not established in this handoff. Validation class counts in the early historical evaluation: 24 cigarettes, 8 plastic bottles, 2 bags, 11 wrappers, 13 film, 7 cans, 5 cups, 5 styrofoam. Those counts sum to 75. Class imbalance and extremely few bag examples limit interpretation. Saved predictions include up to 900 raw queries per image; the official COCO evaluator can rank/filter to top 100 under its documented protocol. Threshold=0.05 and IoU=0.50 in visual reviews are *diagnostic conditions*, not the definition of official mAP.

**Training experiment defaults (corrected Stage 4 series):** 2 epochs, 221 updates per epoch (442 total), batch 1, AdamW, default LR `1e-5`, weight decay `1e-4`, max grad norm 1, FP32, resize shortest edge 256 / longest edge 384, pad to 384×384, frozen vision and text backbones (verified Stage 3D adapter and native loss coefficients), no augmentation/scheduler/accumulation, workers 0, seed default 42. A 3-step memory pilot discards both optimizer and pilot model before training restarts from pinned original pretrained weights. Later isolated changes added CLI flags `--learning-rate` (default `1e-5`) and `--seed` (default `42`) with tests and checkpoint provenance.

## 4. Chronological development: major milestones

**Phase 0–1 (Oct 16–17):** approved AI-modeling-only, still-image trash scope; initial Phase 0 merged (PRs #1/#2 recorded historically). Phase 1 planning/specification documented Hugging Face Grounding DINO Tiny, Windows environment, dataset protocol, prompt/threshold experiments, metrics, and outputs. Planning docs: `README.md`, `docs/DECISIONS.md`, `docs/ARCHITECTURE.md`, `docs/TEST_PLAN.md`, `docs/PHASES.md`, `docs/HANDOFF.md`. No detector implementation at Phase 1. Several follow-up planning PRs #3/#4 were reported merged; verify current git history if needed.

**Phase 2 (Oct 17–18):** subdivided into 2A environment/model readiness, 2B image/config/JSON foundation, 2C pretrained inference and phrase mapping, 2D scrollable Tk/TkAgg presentation with image / Detection Summary / Text-Token Details / Configuration, coordinate axes, bbox with small labels, raw phrases, zoom/pan and in-memory rendering, 2E image-by-image functional baseline and prompt/threshold inspection. PR #11 merging 2D at historical `a3ec215` was reported; normal inference remains separate from Stage 4 training. No initial annotations/fine-tuning in these phases. Later a separate Stage 3 evaluated options and Phase 4 expanded into experimental TACO training; exact whole Stage 3 chronology may not be fully retrievable from available conversations.

**Stage 4 first real TACO training (Oct 18–19):** historical run `taco-1698bcb46b85435786dd2a199910910f`, 442 full optimizer updates, 3-step pilot, ~7.49 min, peak ~2.382 GiB allocated (single historical environment). Originally reported extremely poor AP (`0.000298934` pretrained vs `0.000057591` epoch-2) with decreasing loss — these **metrics were invalidated by later evaluation-coordinate correction**. Do not cite those pre-fix AP numbers as current true baseline. The checkpoint was not promoted.

**Stage 4 geometry audit Step 1A:** two confirmed defects: (1) loss targets normalized to padded canvas while native Grounding DINO predictions normalize to valid image dimensions; (2) evaluation decoded normalized predictions with an extra padding-ratio expansion, corrupting bounding boxes. Example original 640×320 COCO box `[160,80,160,80]`: expected native normalized `[0.375,0.375,0.25,0.25]`, faulty target `[0.375,0.1875,0.25,0.125]`. Historical self-referential tests missed defects.

**Step 1B:** corrected `scripts/training_smoke.py`, `scripts/taco_training_data.py`, `scripts/taco_evaluation.py`, added `tests/test_taco_coordinates.py`; evaluation protocol v2. Corrected pretrained baseline AP **0.111576**, AP50 **0.158253**, AP75 **0.135975**, AR100 **0.363077**. Small-object AP 0.00. Original metrics not comparable to v2. Exact baseline reproduction was repeatedly enforced in subsequent preflights.

**Step 1C:** protected evaluation-only provenance replays saved historical weights without changing strict training-resume verification. Artifacts: `scripts/evaluation_provenance.py`, `scripts/reevaluate_taco.py`, `config/taco_evaluation_compatibility.v1.json`. Corrected historical results: pretrained AP .111576 / AR100 .363077; historical E1 AP .126627 / AR100 .275363; historical E2 AP .066215 / AR100 .235395. Historical epochs suffer marked class-score suppression (pretrained max .596656, E1 max .184526, E2 max .123536; 216 vs 0 vs 0 queries >= .25). The historical *trained weights* remain compromised by faulty target geometry; evaluation-only correction does not retrain them.

**Step 2 visual viewer:** `scripts/taco_visual_review.py`, `scripts/taco_error_analysis.py`, `scripts/taco_comparison_gui.py`, `config/taco_visual_review.v1.json`. Four synchronized panels ground truth/pretrained/E1/E2; 47 validation photos; zoom/pan; labels, scores, raw phrases; saved predictions. Diagnostic match counts at score>=.05 and IoU>=.50: pretrained 28/75, historical E1 17/75, historical E2 14/75 (not official AP). Representative IDs 42,113,200,31,1096,252,1014. Later owner D-046 kept **terminal-only as default**; optional `--gui` restored explicitly. Later added `--saved-review`, `--run-choice`, `--image-id`, `--per-object`, per-object GT navigation/zoom/IoU tables; normal default remains unchanged. Local review manifest contains 20 selected validation images.

**Corrected retraining (Oct 20):** seed 42, LR `1e-5`, same parameters, corrected valid-image target geometry, pretrained restart; exact 442 updates + 3 discarded pilot. E1 AP **0.203932**, AR100 **0.396766**; E2 AP **0.084500**, AR100 **0.211229**. Original pretrained AP .111576 / AR100 .363077. E1 remains **highest observed validation AP checkpoint** but has not been promoted. E2 AP drops despite lower validation loss. Strongest early diagnostic explanation: lost correct-class candidate coverage despite retained any-class box overlaps.

**Corrected diagnostic:** any-class coverage at IoU>=.5 **55→57**, same-class coverage **36→30** between seed-42 `1e-5` E1 and E2. Misclassification versus candidate assignment is important; simply claiming global score suppression or overfitting as root cause was unsupported. Some bags and styrofoam AP collapsed even when accurate boxes remained. The user approved an isolated LR test, not bulk hyperparameter search.

**LR `5e-6` experiment (seed 42):** 442 updates + 3 pilot, E1 AP **0.137099**, AR100 **0.330432**; E2 AP **0.145952**, AR100 **0.302977**. E1→E2 AP did not collapse, but neither epoch beats strongest original `1e-5` E1. Correct-class coverage still 32→30. Added configurable `--learning-rate` while preserving `1e-5` default and strict context fingerprinting.

**Paired second-seed replication (Oct 22):** seed 123, both LRs, exactly 884 updates and six discarded pilot updates; added `--seed`, default 42. Seed 123 `1e-5`: E1 AP **0.140478**, E2 AP **0.152880**. Seed 123 `5e-6`: E1 AP **0.137919**, E2 AP **0.113018**. This *reverses* single-seed 42 E1→E2 stability patterns, weakening claims LR `5e-6` reliably prevents regression. Two seeds do not demonstrate statistical superiority.

**Cross-seed analysis:** 20-image review manifest; cigarette and plastic-film AP/recall often improve with fine-tuning across runs; wrapper recall remains lower; bags/styrofoam vary, and class counts small. GUI selects saved runs: `42 / 1e-5`, `42 / 5e-6`, `123 / 1e-5`, `123 / 5e-6`. Human inspection requested.

**Per-object human review image 777 (Oct 22):** eight GT annotations, `GT259=plastic_wrapper` and `GT278`–`GT284=cigarette`. GT259 is a visible large wrapper: pretrained correct-class candidate qualifies at score>=.05 and IoU>=.5, both inspected E1 variants fail; seed42 E2 recovers, seed123 E2 does not. Six of seven cigarette annotations GT278–GT283 lack qualifying coverage across the inspected `1e-5` seeds and epochs, while GT284 qualifies at E2 in **both** seeds. Some pale objects/boxes merit independent human material-identity review; don't rewrite GT merely because model fails. Image-777 non-exclusive correct-class coverage: pretrained 1/8; seed42 E1 0/8, E2 2/8; seed123 E1 0/8, E2 1/8. These are *per-image diagnostics*, not AP. Review screenshots do not prove whole-dataset results by themselves.

**Small-object and classification audit (Oct 22):** all 47 validation images, 75 annotations, all nine model states (pretrained + eight fine-tuned). **13/24 cigarettes have standard-resolution resized shortest box side <4 pixels**; only 0–2 of these 13 obtain qualifying detection across states. Cigarettes lack qualifying localization for roughly 18–23/24 across states. Among 75 annotations, any-class IoU>=.5 with wrong-class-only coverage: pretrained **20/75**, fine-tuned states **21–27/75**, at score>=.05. Cans suffer class-related failure 6–7/7, wrappers 5–8/11, depending on state; do not overgeneralize with small class counts. Image 777 current valid resized content **350×256**, padded **384×384**. The audit proposed **inference-only resolution comparison** at current 256/384 versus higher 384/576, using the already-trained seed42 LR `1e-5` Epoch-1 checkpoint, with *no weight updates*. This is **the next proposed, not yet reported completed, task**.

## 5. Current verified metrics (corrected COCO protocol v2)

All values below are fractions on 0–1 scale; the 47-image validation split is small.

| Starting/training setup | Seed | Epoch | AP | AR100 |
|---|---:|---:|---:|---:|
| Original pretrained | – | – | 0.111576 | 0.363077 |
| Historic faulty-target training, corrected eval | 42 | 1 | 0.126627 | 0.275363 |
| Historic faulty-target training, corrected eval | 42 | 2 | 0.066215 | 0.235395 |
| Corrected LR 1e-5 | 42 | 1 | **0.203932** | **0.396766** |
| Corrected LR 1e-5 | 42 | 2 | 0.084500 | 0.211229 |
| Corrected LR 5e-6 | 42 | 1 | 0.137099 | 0.330432 |
| Corrected LR 5e-6 | 42 | 2 | 0.145952 | 0.302977 |
| Corrected LR 1e-5 | 123 | 1 | 0.140478 | not quoted in available conversations |
| Corrected LR 1e-5 | 123 | 2 | 0.152880 | not quoted |
| Corrected LR 5e-6 | 123 | 1 | 0.137919 | not quoted |
| Corrected LR 5e-6 | 123 | 2 | 0.113018 | not quoted |

The **best AP value 0.203932** is a validation-set maximum among tried configurations, not a deployment acceptance result, statistical proof, or calibrated probability. The untouched test split has not been used.

## 6. Important repository files and records to inspect (historical, not exhaustive)

- Model/inference: `scripts/detect_images.py`; `config/detection.json` (often has owner local edits); package/CLI implementation elsewhere in repo to discover locally.
- TACO data and training: `scripts/taco_training_data.py`, `scripts/training_smoke.py`, Stage 3D training adapter, `scripts/training_checkpoint.py`.
- Evaluation: `scripts/taco_evaluation.py`, `scripts/evaluation_provenance.py`, `scripts/reevaluate_taco.py`, `config/taco_evaluation_compatibility.v1.json`.
- Review: `scripts/taco_visual_review.py`, `scripts/taco_error_analysis.py`, `scripts/taco_comparison_gui.py`, `config/taco_visual_review.v1.json`, `tests/test_taco_coordinates.py`.
- Main governance and docs: `AGENTS.md`, `README.md`, `docs/DECISIONS.md`, `docs/HANDOFF.md`, `docs/PHASES.md`, `docs/ARCHITECTURE.md`, `docs/TEST_PLAN.md`.
- Stage 4 reports: `docs/TACO_STAGE_4_RUN.md`, `docs/TACO_STAGE_4_COORDINATES.md`, `docs/TACO_STAGE_4_REEVALUATION.md`, `docs/TACO_STAGE_4_VISUAL_REVIEW.md`, `docs/TACO_STAGE_4_CORRECTED_RETRAIN.md`, `docs/TACO_STAGE_4_CORRECTED_DIAGNOSTICS.md`, `docs/TACO_STAGE_4_LR_EXPERIMENT.md`, `docs/TACO_STAGE_4_SEED_REPLICATION.md`, `docs/TACO_STAGE_4_CROSS_SEED_ANALYSIS.md`, `docs/TACO_STAGE_4_PER_OBJECT_REVIEW.md`, `docs/TACO_STAGE_4_SMALL_OBJECT_CLASSIFICATION_AUDIT.md` (most recent audit, reported; filename should be verified locally).
- Initial historical run: `runs/training/taco-1698bcb46b85435786dd2a199910910f/`.
- Baseline read-only record: `runs/training/step1b-evaluation-c8b127d8bf364c7dbb735795cd81c06d/record.json`.
- Historical evaluation record: `runs/training/step1c-evaluation-3d24a67e1fe041e5a8acf56a26406db4/record.json`.
- 20-image saved-review manifest: `runs/training/cross-seed-analysis-fd10923e40d24c40be5bab3ae0fa2c63/review-manifest.json`.
- Git reference: Stage 4 PR #18 / branch `feat/stage4-real-image-training` was previously reported as *not merged*. Current remote state is **not** verified by this pack.

## 7. Useful local commands, subject to current CLI verification

From historical Windows repository root:

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
```

Open current saved TACO per-object viewer at image 777, seed42 LR 1e-5:

```powershell
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_visual_review.py --gui --saved-review .\runs\training\cross-seed-analysis-fd10923e40d24c40be5bab3ae0fa2c63\review-manifest.json --run-choice "42 / 1e-5" --image-id 777 --per-object
```

Other run choices: `"42 / 5e-6"`, `"123 / 1e-5"`, `"123 / 5e-6"`. The GUI's unannotated terminal-only mode is default without explicit `--gui`; in-memory zoom/overlays never imply saved annotated copies. Standard pretrained function-checking is a different entry point; verify command with `--help` before use:

```powershell
& .\.venv\Scripts\python.exe .\scripts\detect_images.py --mode infer --input data/input --results results --local-files-only
```

## 8. What has **not** been established

- The exact comprehensive raw transcripts for **all** past ChatGPT project chats are not available in this handoff. `CONVERSATION_CHRONOLOGY.md` reconstructs them from available messages/summaries, not full transcript text.
- No complete current local repository audit, commit graph, source hash verification, or GPU inspection was performed during the handoff's creation.
- It remains unresolved whether greater inference resolution is feasible on the present GPU, whether it increases correct-class cigarette detections, and whether such gains justify training at higher resolution.
- The observed mistakes do not prove the sole cause is insufficient resolution, class mapping, overfitting, LR, frozen backbones, or annotations.
- The dataset is small and repeatedly used for experiment selection; avoid chasing small validation AP differences and silently redefining acceptance thresholds.

## 9. Current next step / handoff contract

**Pending proposed experiment:** fixed seed42 LR `1e-5` **Epoch 1** checkpoint; compare inference preprocessing `shortest/longest 256/384` and `384/576` **on the same 47 validation images**, consistent FP32/device/prompt/taxonomy/scoring/evaluation, no training. High-res memory/correctness pilot first; failure should stop, no silent fallback. Reproduce unchanged baseline AP `0.203932`, AR100 `0.396766` and full metrics. Test small-object qualifying same/any-class coverage, per-object IoUs, AP/AP50/AP75/AR100 and per-class results. All results **local**, original data untouched, no test access, commits or promotion. Proposed report: `docs/TACO_STAGE_4_RESOLUTION_EXPERIMENT.md`. Details and stop conditions in `NEXT_EXPERIMENT_RESOLUTION.md`.

*Source basis:* user-provided original `Concept-Proposal.txt`, `Trash Detection Support.txt`, `Pasted markdown.md`, directly visible conversation updates, and retrieved historical conversation summaries. Where they disagree, use the most recent verified user-run result and confirm via original local run records.
