# Stage 4: diagnostic saved-query NMS audit (D-063)

Owner-authorized saved-data diagnostic, 2026-10-25. No model loading/execution, inference, GPU initialization, backward pass, optimizer update or audit photo decoding. All six core training SOURCES files and all 55 checkpoint bundles are unchanged. No commit, push or merge.

The saved validation GUI now offers an explicit **Diagnostic NMS** selector in both standard and per-object modes. **none (raw queries)** remains the default. Class-aware and class-agnostic modes use greedy score-descending/query-ID-ascending suppression at inclusive IoU>=0.50. A suppressed query never suppresses another query. Scores, boxes, predicted classes, original ranks, stored JSON, source photos and official metrics are never rewritten. This is not object grouping or a physical-object counting rule.

The original protocol is retained byte-for-byte: `bantai.taco-evaluation.v2` represents no NMS as `nms: false` (there is no `nms_iou` key). The owner's `nms_iou: None` no-NMS requirement is preserved without editing that representation. Every recomputed metric below is explicitly **diagnostic**, not a replacement for the official saved scores or an approved inference setting.

Private audit: [nms-audit-bd560d56085f4a22854f506a40ba40e6](../runs/training/nms-audit-bd560d56085f4a22854f506a40ba40e6/record.json), [complete JSON results](../runs/training/nms-audit-bd560d56085f4a22854f506a40ba40e6/nms-summary.json), [counts/coverage/metrics CSV](../runs/training/nms-audit-bd560d56085f4a22854f506a40ba40e6/nms-counts-coverage-metrics.csv), and [suppression/query-ID ledger](../runs/training/nms-audit-bd560d56085f4a22854f506a40ba40e6/nms-suppression-ledger.json). The ledger identifies each suppressed query, its kept suppressor and IoU, and retains Image 777 kept-query details.

## Observed effect

NMS reduces visible clutter but does not universally preserve correct-class coverage. At score 0.05, the 25/75 blend has 1152 passing queries across 47 images and 31 on Image 777. Class-aware NMS retains 965/28; class-agnostic NMS retains 853/27. At K 10, class-aware coverage increases from 30 to 31 of 75 GTs and 10 to 11 of 24 cigarettes; class-agnostic coverage drops to 27/75 even while reaching 11/24 cigarettes. Image 777 cigarette coverage remains 3/7 for this blend at both K 10 and K 15 under every mode.

For constant E2 step 442 at 0.05/K 15, agnostic NMS reveals one more Image 777 cigarette (3 to4 of 7), but total GT coverage drops from 30 to 26 of 75. For E1 step 221 at 0.05/K 15, GT coverage drops from 32 to 21 with agnostic NMS. Higher-scored competing classes or different overlapping objects can suppress a useful box. These measured losses are why suppression stays explicit and reversible; no NMS mode is automatically selected or promoted.

On the unchanged original-top100 pool at score floor 0.00, blend diagnostic AP/AR100 is 0.265563/0.403085 without NMS, 0.266194/0.385345 with class-aware NMS, and 0.262950/0.345922 with class-agnostic NMS. The small AP increase under class-aware suppression accompanies lower AR100; it is not an across-the-board improvement and does not change any official score.

## Audit method and GUI behavior

The exact five saved 384/576 states are pretrained step 0 (D-056), E1 step 221 and E2 step 442 (D-057), step 350 (D-058), and blend_350_442_2575 (D-061). Their hashes, corpus/rights identity, 47 validation images, 75 GTs, 24 cigarettes and 7 Image 777 cigarettes are verified. All 900 query identities/ranking/coordinates and official top100 membership are validated by saved_rows. The five original official COCO metric dictionaries reproduce exactly on CPU.

Standard display order: valid/source filtering, inclusive score floor, image-wide NMS, then the first K kept queries in descending score order. All 900 is the primary audit pool; the current GUI default, **official top100**, is audited separately. Source filtering happens BEFORE NMS. Original query IDs and ranks remain intact; GUI tables show kept queries, including those beyond the drawing cap.

Coverage means any same-class query among the first K boxes PER IMAGE with IoU>=0.50 to a GT. Coverage is nonexclusive: one query may cover several GTs. It is not official COCO recall, accuracy, a calibrated probability, or a physical-object count. The first-K tables below match **standard mode**. Per-object mode applies the same global NMS before GT matching, then retains its existing GT-prioritized overlap drawing and tables; that ordering is not the global first K. Its raw all 900/no-score-floor diagnostic remains explicitly based on the unsuppressed saved rows.

NMS operates only in memory. Switching back to none restores the raw view; it does not recover newly inferred information. The control uses a separate toolbar row so existing controls remain accessible. Selecting it applies the view while preserving zoom. Explicit human notes include nms_mode alongside threshold, source and top K. Existing run choices and saved configuration remain unchanged.

## Remaining boxes and first-K coverage

Each coverage cell is **GTs/75; cigarettes/24; Image 777 cigarettes/7** (numerators shown). Remaining counts include all kept passing queries before K. Both score boundaries are inclusive.

### all 900 (diagnostic), score >= 0.05

| State | NMS | Remaining all 47 | Remaining Image 777 | First10 GT; cig;777 | First15 GT; cig;777 |
| --- | --- | --- | --- | --- | --- |
| step 0 | none | 4894 | 86 | 21; 1; 0 | 23; 1; 0 |
| step 0 | class-aware | 2150 | 40 | 24; 1; 0 | 25; 2; 0 |
| step 0 | class-agnostic | 1576 | 29 | 14; 2; 0 | 15; 3; 0 |
| step 221 | none | 1067 | 26 | 27; 4; 0 | 32; 7; 1 |
| step 221 | class-aware | 791 | 18 | 28; 5; 1 | 32; 7; 1 |
| step 221 | class-agnostic | 609 | 15 | 19; 6; 1 | 21; 8; 1 |
| step 350 | none | 1224 | 34 | 26; 8; 1 | 30; 10; 2 |
| step 350 | class-aware | 1011 | 29 | 27; 8; 1 | 30; 10; 2 |
| step 350 | class-agnostic | 892 | 28 | 24; 10; 2 | 24; 10; 2 |
| step 442 | none | 1118 | 29 | 29; 11; 3 | 30; 11; 3 |
| step 442 | class-aware | 935 | 27 | 29; 11; 3 | 30; 11; 3 |
| step 442 | class-agnostic | 828 | 25 | 25; 11; 3 | 26; 12; 4 |
| blend_350_442_2575 | none | 1152 | 31 | 30; 10; 3 | 33; 11; 3 |
| blend_350_442_2575 | class-aware | 965 | 28 | 31; 11; 3 | 33; 11; 3 |
| blend_350_442_2575 | class-agnostic | 853 | 27 | 27; 11; 3 | 27; 11; 3 |

### all 900 (diagnostic), score >= 0.07

| State | NMS | Remaining all 47 | Remaining Image 777 | First10 GT; cig;777 | First15 GT; cig;777 |
| --- | --- | --- | --- | --- | --- |
| step 0 | none | 3012 | 56 | 21; 1; 0 | 23; 1; 0 |
| step 0 | class-aware | 1399 | 29 | 24; 1; 0 | 25; 2; 0 |
| step 0 | class-agnostic | 981 | 19 | 14; 2; 0 | 15; 3; 0 |
| step 221 | none | 617 | 17 | 27; 4; 0 | 31; 7; 1 |
| step 221 | class-aware | 481 | 12 | 28; 5; 1 | 31; 7; 1 |
| step 221 | class-agnostic | 350 | 10 | 19; 6; 1 | 20; 7; 1 |
| step 350 | none | 636 | 14 | 26; 8; 1 | 29; 10; 2 |
| step 350 | class-aware | 535 | 12 | 27; 8; 1 | 29; 10; 2 |
| step 350 | class-agnostic | 456 | 11 | 24; 10; 2 | 24; 10; 2 |
| step 442 | none | 578 | 11 | 29; 11; 3 | 30; 11; 3 |
| step 442 | class-aware | 502 | 10 | 29; 11; 3 | 30; 11; 3 |
| step 442 | class-agnostic | 423 | 9 | 25; 11; 3 | 25; 11; 3 |
| blend_350_442_2575 | none | 591 | 11 | 30; 10; 3 | 33; 11; 3 |
| blend_350_442_2575 | class-aware | 512 | 10 | 31; 11; 3 | 33; 11; 3 |
| blend_350_442_2575 | class-agnostic | 435 | 9 | 27; 11; 3 | 27; 11; 3 |

### official top100, score >= 0.05

| State | NMS | Remaining all 47 | Remaining Image 777 | First10 GT; cig;777 | First15 GT; cig;777 |
| --- | --- | --- | --- | --- | --- |
| step 0 | none | 4342 | 86 | 21; 1; 0 | 23; 1; 0 |
| step 0 | class-aware | 1929 | 40 | 24; 1; 0 | 25; 2; 0 |
| step 0 | class-agnostic | 1403 | 29 | 14; 2; 0 | 15; 3; 0 |
| step 221 | none | 1067 | 26 | 27; 4; 0 | 32; 7; 1 |
| step 221 | class-aware | 791 | 18 | 28; 5; 1 | 32; 7; 1 |
| step 221 | class-agnostic | 609 | 15 | 19; 6; 1 | 21; 8; 1 |
| step 350 | none | 1224 | 34 | 26; 8; 1 | 30; 10; 2 |
| step 350 | class-aware | 1011 | 29 | 27; 8; 1 | 30; 10; 2 |
| step 350 | class-agnostic | 892 | 28 | 24; 10; 2 | 24; 10; 2 |
| step 442 | none | 1118 | 29 | 29; 11; 3 | 30; 11; 3 |
| step 442 | class-aware | 935 | 27 | 29; 11; 3 | 30; 11; 3 |
| step 442 | class-agnostic | 828 | 25 | 25; 11; 3 | 26; 12; 4 |
| blend_350_442_2575 | none | 1152 | 31 | 30; 10; 3 | 33; 11; 3 |
| blend_350_442_2575 | class-aware | 965 | 28 | 31; 11; 3 | 33; 11; 3 |
| blend_350_442_2575 | class-agnostic | 853 | 27 | 27; 11; 3 | 27; 11; 3 |

### official top100, score >= 0.07

| State | NMS | Remaining all 47 | Remaining Image 777 | First10 GT; cig;777 | First15 GT; cig;777 |
| --- | --- | --- | --- | --- | --- |
| step 0 | none | 3012 | 56 | 21; 1; 0 | 23; 1; 0 |
| step 0 | class-aware | 1399 | 29 | 24; 1; 0 | 25; 2; 0 |
| step 0 | class-agnostic | 981 | 19 | 14; 2; 0 | 15; 3; 0 |
| step 221 | none | 617 | 17 | 27; 4; 0 | 31; 7; 1 |
| step 221 | class-aware | 481 | 12 | 28; 5; 1 | 31; 7; 1 |
| step 221 | class-agnostic | 350 | 10 | 19; 6; 1 | 20; 7; 1 |
| step 350 | none | 636 | 14 | 26; 8; 1 | 29; 10; 2 |
| step 350 | class-aware | 535 | 12 | 27; 8; 1 | 29; 10; 2 |
| step 350 | class-agnostic | 456 | 11 | 24; 10; 2 | 24; 10; 2 |
| step 442 | none | 578 | 11 | 29; 11; 3 | 30; 11; 3 |
| step 442 | class-aware | 502 | 10 | 29; 11; 3 | 30; 11; 3 |
| step 442 | class-agnostic | 423 | 9 | 25; 11; 3 | 25; 11; 3 |
| blend_350_442_2575 | none | 591 | 11 | 30; 10; 3 | 33; 11; 3 |
| blend_350_442_2575 | class-aware | 512 | 10 | 31; 11; 3 | 33; 11; 3 |
| blend_350_442_2575 | class-agnostic | 435 | 9 | 27; 11; 3 | 27; 11; 3 |

## Official saved scores (unchanged)

| State | Official AP | Official AR100 |
| --- | --- | --- |
| step 0 | 0.147847 | 0.426109 |
| step 221 | 0.249891 | 0.488141 |
| step 350 | 0.262337 | 0.428890 |
| step 442 | 0.170508 | 0.370118 |
| blend_350_442_2575 | 0.265563 | 0.403085 |

## Post-NMS diagnostic COCO metrics: NMS-only comparison

This table starts with the **same original official top100 prediction pool**, with score floor 0.00, and only applies suppression. No lower-ranked queries are backfilled. None reproduces the complete original official metrics exactly, providing a control. AP/AR100 in this table are still labeled diagnostic; they do not overwrite the official protocol/results.

| State | NMS | Diagnostic AP | Diagnostic AR100 |
| --- | --- | --- | --- |
| step 0 | none | 0.147847 | 0.426109 |
| step 0 | class-aware | 0.151533 | 0.394289 |
| step 0 | class-agnostic | 0.135089 | 0.220265 |
| step 221 | none | 0.249891 | 0.488141 |
| step 221 | class-aware | 0.251077 | 0.475561 |
| step 221 | class-agnostic | 0.204602 | 0.279466 |
| step 350 | none | 0.262337 | 0.428890 |
| step 350 | class-aware | 0.264208 | 0.414635 |
| step 350 | class-agnostic | 0.251956 | 0.316559 |
| step 442 | none | 0.170508 | 0.370118 |
| step 442 | class-aware | 0.171810 | 0.348091 |
| step 442 | class-agnostic | 0.164478 | 0.252899 |
| blend_350_442_2575 | none | 0.265563 | 0.403085 |
| blend_350_442_2575 | class-aware | 0.266194 | 0.385345 |
| blend_350_442_2575 | class-agnostic | 0.262950 | 0.345922 |

## Post-filter diagnostic COCO metrics: score floor plus NMS

These diagnostic calculations use the unchanged CPU COCO calculator and globally cap its input to 100 kept predictions per image AFTER score filtering/NMS. Thus the all 900 pool can backfill original ranks beyond 100 after suppression; official-top100 cannot. Comparisons with official scores also change the score floor. The none rows here are **not** presented as the original official metrics. No validation loss is recomputed because this task executes no model.

### all 900 (diagnostic), score >= 0.05 — diagnostic only

| State | NMS | Diagnostic AP | Diagnostic AR100 |
| --- | --- | --- | --- |
| step 0 | none | 0.147847 | 0.426109 |
| step 0 | class-aware | 0.151533 | 0.394289 |
| step 0 | class-agnostic | 0.135106 | 0.222765 |
| step 221 | none | 0.249103 | 0.449367 |
| step 221 | class-aware | 0.250097 | 0.444559 |
| step 221 | class-agnostic | 0.204455 | 0.271093 |
| step 350 | none | 0.253864 | 0.350934 |
| step 350 | class-aware | 0.254552 | 0.346126 |
| step 350 | class-agnostic | 0.251862 | 0.310549 |
| step 442 | none | 0.166609 | 0.287514 |
| step 442 | class-aware | 0.167325 | 0.277739 |
| step 442 | class-agnostic | 0.164244 | 0.245046 |
| blend_350_442_2575 | none | 0.265186 | 0.382901 |
| blend_350_442_2575 | class-aware | 0.265865 | 0.375048 |
| blend_350_442_2575 | class-agnostic | 0.262710 | 0.335625 |

### all 900 (diagnostic), score >= 0.07 — diagnostic only

| State | NMS | Diagnostic AP | Diagnostic AR100 |
| --- | --- | --- | --- |
| step 0 | none | 0.147821 | 0.416109 |
| step 0 | class-aware | 0.151533 | 0.394289 |
| step 0 | class-agnostic | 0.135089 | 0.220265 |
| step 221 | none | 0.248757 | 0.441434 |
| step 221 | class-aware | 0.249672 | 0.436626 |
| step 221 | class-agnostic | 0.204213 | 0.267968 |
| step 350 | none | 0.253414 | 0.334026 |
| step 350 | class-aware | 0.254103 | 0.333065 |
| step 350 | class-agnostic | 0.251665 | 0.303257 |
| step 442 | none | 0.166239 | 0.273251 |
| step 442 | class-aware | 0.167049 | 0.268444 |
| step 442 | class-agnostic | 0.163982 | 0.235751 |
| blend_350_442_2575 | none | 0.264816 | 0.368198 |
| blend_350_442_2575 | class-aware | 0.265558 | 0.365313 |
| blend_350_442_2575 | class-agnostic | 0.262441 | 0.325890 |

### official top100, score >= 0.05 — diagnostic only

| State | NMS | Diagnostic AP | Diagnostic AR100 |
| --- | --- | --- | --- |
| step 0 | none | 0.147847 | 0.426109 |
| step 0 | class-aware | 0.151533 | 0.394289 |
| step 0 | class-agnostic | 0.135089 | 0.220265 |
| step 221 | none | 0.249103 | 0.449367 |
| step 221 | class-aware | 0.250097 | 0.444559 |
| step 221 | class-agnostic | 0.204455 | 0.271093 |
| step 350 | none | 0.253864 | 0.350934 |
| step 350 | class-aware | 0.254552 | 0.346126 |
| step 350 | class-agnostic | 0.251862 | 0.310549 |
| step 442 | none | 0.166609 | 0.287514 |
| step 442 | class-aware | 0.167325 | 0.277739 |
| step 442 | class-agnostic | 0.164244 | 0.245046 |
| blend_350_442_2575 | none | 0.265186 | 0.382901 |
| blend_350_442_2575 | class-aware | 0.265865 | 0.375048 |
| blend_350_442_2575 | class-agnostic | 0.262710 | 0.335625 |

### official top100, score >= 0.07 — diagnostic only

| State | NMS | Diagnostic AP | Diagnostic AR100 |
| --- | --- | --- | --- |
| step 0 | none | 0.147821 | 0.416109 |
| step 0 | class-aware | 0.151533 | 0.394289 |
| step 0 | class-agnostic | 0.135089 | 0.220265 |
| step 221 | none | 0.248757 | 0.441434 |
| step 221 | class-aware | 0.249672 | 0.436626 |
| step 221 | class-agnostic | 0.204213 | 0.267968 |
| step 350 | none | 0.253414 | 0.334026 |
| step 350 | class-aware | 0.254103 | 0.333065 |
| step 350 | class-agnostic | 0.251665 | 0.303257 |
| step 442 | none | 0.166239 | 0.273251 |
| step 442 | class-aware | 0.167049 | 0.268444 |
| step 442 | class-agnostic | 0.163982 | 0.235751 |
| blend_350_442_2575 | none | 0.264816 | 0.368198 |
| blend_350_442_2575 | class-aware | 0.265558 | 0.365313 |
| blend_350_442_2575 | class-agnostic | 0.262441 | 0.325890 |

## Integrity and verification

All 55 checkpoint bundles were verified before and after. Fresh per-phase SHA-256 inventories cover 1703 protected files, including every prior training artifact, original manifests/results, six core files, rights/source metadata, AGENTS.md, ignore rules and dependency locks; none changed. Source photographs and test.json were not opened by the audit. Bundle file hashes are compared against each saved manifest using that phase's fresh hash inventory, avoiding duplicate reads of large files. The core official protocol dictionary is also checked unchanged in memory. No model/optimizer checkpoint is deserialized.

Guarded suite: 251 tests, 235 passed, 16 documented skips, zero failures/errors. Training-execution fixtures and the existing Windows symlink-privilege fixture are explicitly skipped. New tests cover raw order/input immutability, score/query-ID ties, inclusive IoU 0.50, class boundaries, nontransitive greedy behavior, source/score/invalid filtering, suppression before top K, real Agg rendering in both GUI classes, reversible raw selection and per-object raw diagnostic provenance.

Native Windows Tk smoke checks passed 24 cases: standard/per-object modes, all three NMS choices, thresholds 0.05/0.07 and top K 10/15. The actual readonly selector events dispatched Apply, tables and drawn boxes matched filtered data, all 47 image IDs remained selectable, and image/GT navigation and zoom/pan worked. Returning to raw mode restored the view. Saved JSON hashes and in-memory source rows were unchanged; model/GPU calls, weight reads and exports were blocked. See native-results.json in the private audit. These are runtime checks, separate from Agg fixtures and human photo judgments.

The initial audit invocation encountered Python's static nesting limit in the guard wrapper before any code/data execution. Guards were moved into ExitStack; the single completed data audit above performed no model/GPU execution. Human visual confirmation remains pending. This is a diagnostic audit of a repeatedly inspected 47-image validation set, with selection bias. Fewer boxes do not establish better physical-object detection, and suppression may remove real overlapping instances.

## PowerShell launch

Run from the repository root and close one window before launching the next. Both open Image 777 with all 47 validation IDs selectable. The example compares step 350, step 442 and the blend. The selector is available for every historical/new saved run choice.

```powershell
$review = "runs/training/d060-saved-review/review-manifest.json"
$choice = "42 / 1e-5 @ 384/576 (350 vs 442 vs integrated)"
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice $choice --gui --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice $choice --gui --per-object --image-id 777
```

In the GUI set source to **all 900 (diagnostic)**, **View threshold** to 0.05 or 0.07, and **Box top K** to 10 or 15, then click **Apply view settings**. In **Diagnostic NMS**, select **class-aware IoU 0.50** or **class-agnostic IoU 0.50**; selection applies immediately. **none (raw queries)** remains the default. First-K coverage in this report uses standard mode; per-object mode remains GT-prioritized.
