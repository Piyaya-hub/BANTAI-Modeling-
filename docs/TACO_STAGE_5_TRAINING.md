# Stage 5: two-epoch nine-class fine-tuning (D-070 execution)

Completed locally on the owner-authorized Stage 5 branch; execution recorded 2026-10-28. The requested command ran with its existing **256/384 default**, FP32, TF32 disabled, AdamW LR1e-5/weight decay1e-4/foreachFalse and norm1 clipping. It trained exactly **two epochs /614 updates** on 307 images, with no extra pilot updates. Baseline and two epoch-end validations each used 65 nine-class forwards plus 47 original-eight-prompt forwards, **336 validation forwards total**. No test.json or test photos were opened; no checkpoint was promoted or wired into baseline inference. No commit/push/merge was performed for this run.

Warm start reproduces D-061 `blend_350_442_2575` bitwise (parameter SHA-256 490bd4b1082bf0a515dc10b0b3c7d64a180b03cd2fbd2c1ee3e93d4374a8c69e). It averages 0.25*D-057 step350+0.75*step442 over 558 canonical trainable tensors; frozen entries match the pinned snapshot. This is a fresh optimizer/seeded nine-class run, not restoration of Stage 4 optimizer/RNG state. The six core Stage 4 SOURCES, old datasets, taxonomy v1 and previous 64 checkpoints remain unchanged. Both source checkpoint bundles were independently verified again after execution. Eight complete new bundles were verified with full processor/adapter/model/optimizer/RNG/context files.

## Paired results at 256/384

D-061's historical AP 0.265563 was measured at 384/576, so it is not a scale-matched reference for the results below. All three rows below use the same 256/384 settings. No NMS, remapping or score floor is applied to COCO metrics. Scores are not calibrated correctness probabilities; the annex has only 20 annotated paper boxes, and this is validation evidence, not a test/generalization claim.

| State | Combined 65 /9-class AP | Legacy47 /true8-prompt AP | Annex18 AP | Paper AP on annex |
| --- | --- | --- | --- | --- |
| Warm-start baseline | 0.057617 | 0.078141 | 0.029054 | 0.000000 |
| Epoch 1 | 0.113571 | 0.131296 | 0.018931 | 0.116952 |
| Epoch 2 | 0.135663 | 0.144434 | 0.145543 | 0.507431 |

The final epoch improves the paired warm-start baseline on all three reported views and learns paper (annex paper AP0 ->0.507431). This does not promote a checkpoint or establish that it exceeds the Stage 4 champion at the champion's 384/576 evaluation condition.

## Actual dataset and evaluation scopes

The bound export/rights record contains 307 training images /595 boxes and 65 validation images /109 boxes. The original 221 training images /397 annotations and 47 validation images /75 annotations are preserved. The appended validation annex has 18 images /34 boxes: 20 paper and 14 other-class boxes. The existing D-070 notes state 121 combined GTs; actual export metadata gives 109. The data and hashed rights assessment/record were left unchanged, and this discrepancy is reported explicitly. All 372 authorized train/validation photos match their recorded file SHA-256s before/after processing. Original 47 test IDs are disjoint from these 372 IDs, verified using only the old eligibility ledger.

Combined 65 and annex 18 use the nine-class prompt/decoder and COCO v3. A genuine separate 47-image pass uses the original eight-class prompt/adapter/decoder and COCO v2 at 256/384. The saved `legacy_47_nine_class_conditioning` field additionally records eight-category scoring after nine-class conditioning; it is not mislabeled as an original-protocol result. All model-forward counts are measured by a root-model hook. Validation snapshots/restores RNG and verifies unchanged parameters; the new Stage 5 training sampler is epoch_batches(307,1,42,zero_based_epoch).

### Combined 65: nine classes

| State | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Warm-start baseline | 0.057617 | 0.070768 | 0.059660 | 0.019500 | 0.010764 | 0.064884 | 0.147380 |
| Epoch 1 | 0.113571 | 0.142033 | 0.120170 | 0.036801 | 0.014066 | 0.135625 | 0.213169 |
| Epoch 2 | 0.135663 | 0.173659 | 0.143699 | 0.023841 | 0.024421 | 0.161676 | 0.247600 |

| Class | GT count | Baseline AP | Epoch1 AP | Epoch2 AP | Epoch2 AR100 |
| --- | --- | --- | --- | --- | --- |
| cigarette | 26 | 0.022451 | 0.030596 | 0.030489 | 0.111538 |
| plastic_bottle | 9 | 0.064029 | 0.075743 | 0.115248 | 0.211111 |
| plastic_bag | 4 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| plastic_wrapper | 13 | 0.119500 | 0.111400 | 0.051485 | 0.153846 |
| plastic_film | 17 | 0.025280 | 0.036661 | 0.037018 | 0.400000 |
| drink_can | 7 | 0.000000 | 0.287129 | 0.287129 | 0.285714 |
| plastic_cup | 6 | 0.287297 | 0.241254 | 0.241254 | 0.283333 |
| styrofoam_piece | 7 | 0.000000 | 0.148515 | 0.148515 | 0.142857 |
| paper | 20 | 0.000000 | 0.090839 | 0.309834 | 0.640000 |

### Frozen47: original eight-prompt pass

| State | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Warm-start baseline | 0.078141 | 0.098126 | 0.078563 | 0.020835 | 0.010712 | 0.089818 | 0.170559 |
| Epoch 1 | 0.131296 | 0.160343 | 0.144322 | 0.040359 | 0.015097 | 0.164043 | 0.226823 |
| Epoch 2 | 0.144434 | 0.176464 | 0.150136 | 0.025000 | 0.025414 | 0.181276 | 0.222835 |

| Class | GT count | Baseline AP | Epoch1 AP | Epoch2 AP | Epoch2 AR100 |
| --- | --- | --- | --- | --- | --- |
| cigarette | 24 | 0.027532 | 0.038722 | 0.037832 | 0.091667 |
| plastic_bottle | 8 | 0.070792 | 0.060924 | 0.096535 | 0.162500 |
| plastic_bag | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| plastic_wrapper | 11 | 0.152475 | 0.255587 | 0.170297 | 0.218182 |
| plastic_film | 13 | 0.026644 | 0.045548 | 0.063017 | 0.484615 |
| drink_can | 7 | 0.000000 | 0.148515 | 0.287129 | 0.285714 |
| plastic_cup | 5 | 0.347681 | 0.292739 | 0.292739 | 0.340000 |
| styrofoam_piece | 5 | 0.000000 | 0.208331 | 0.207921 | 0.200000 |

### Annex18: nine classes

| State | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Warm-start baseline | 0.029054 | 0.032356 | 0.032356 | 0.000000 | 0.000000 | 0.033217 | 0.103125 |
| Epoch 1 | 0.018931 | 0.054068 | 0.015838 | 0.000000 | 0.000000 | 0.021659 | 0.080000 |
| Epoch 2 | 0.145543 | 0.247010 | 0.200307 | 0.000000 | 0.000000 | 0.166370 | 0.211250 |

| Class | GT count | Baseline AP | Epoch1 AP | Epoch2 AP | Epoch2 AR100 |
| --- | --- | --- | --- | --- | --- |
| cigarette | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| plastic_bottle | 1 | 0.000000 | 0.000000 | 0.600000 | 0.600000 |
| plastic_bag | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| plastic_wrapper | 2 | 0.100990 | 0.000000 | 0.016832 | 0.100000 |
| plastic_film | 4 | 0.131441 | 0.034499 | 0.040084 | 0.350000 |
| drink_can | 0 | n/a | n/a | n/a | n/a |
| plastic_cup | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| styrofoam_piece | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| paper | 20 | 0.000000 | 0.116952 | 0.507431 | 0.640000 |

## Launch corrections and qualification

The newly supplied isolated trainer/evaluator required concrete launch fixes before model execution: the legacy JSON path was relative to the photo root, COCO sub-data lacked info, the second latest-checkpoint write would fail, saves omitted optimizer/RNG/processor/adapter state, and two pilot optimizer updates were not counted as epoch updates. These were corrected only in Stage 5 modules. Training now reuses the unchanged audited core step, checks finite gradients and nonzero encoder updates, clips at 1.0, isolates validation RNG, and saves/verifies complete resumable bundles. The legacy comparison now performs a separate eight-prompt forward pass rather than claiming equivalence after filtering nine-class predictions.

Software qualification: 286 tests, 270 passed, 16 documented skips, zero failures/errors. Fifteen live training fixtures are excluded during guarded software verification; one Windows symlink fixture lacks privileges. New regression fixtures cover COCO result loading, repeated checkpoint pointer updates/resume positions, and true legacy-evaluation routing. This software-only suite performed no detector/GPU/backward/optimizer execution. The subsequent real run completed 614 forwards/backward updates and 336 validation forwards with all native loss/gradient/frozen-parameter checks passing. No additional model execution was used for post-run verification.

## Saved artifacts and integrity

Private run: [stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4](../runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/report.json). [Completion checks](../runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/completion-verification.json), [execution/qualification record](../runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/operator-verification.json), baseline.json, validation-epoch-1.json, validation-epoch-2.json, steps.jsonl and contract.json retain full predictions, configuration and evidence. Source snapshots and guarded-test logs are archived inside the private run. No annotated image exports or source-photo modifications occurred.

| Bundle | Position | Optimizer counter | Trainable tensors |
| --- | --- | --- | --- |
| epoch-01 | {'epoch': 1, 'next_batch': 0, 'global_step': 307} | 307 | 558 |
| epoch-02 | {'epoch': 2, 'next_batch': 0, 'global_step': 614} | 614 | 558 |
| step-000100 | {'epoch': 0, 'next_batch': 100, 'global_step': 100} | 100 | 558 |
| step-000200 | {'epoch': 0, 'next_batch': 200, 'global_step': 200} | 200 | 558 |
| step-000300 | {'epoch': 0, 'next_batch': 300, 'global_step': 300} | 300 | 558 |
| step-000400 | {'epoch': 1, 'next_batch': 93, 'global_step': 400} | 400 | 558 |
| step-000500 | {'epoch': 1, 'next_batch': 193, 'global_step': 500} | 500 | 558 |
| step-000600 | {'epoch': 1, 'next_batch': 293, 'global_step': 600} | 600 | 558 |

All eight bundles have distinct model hashes, complete 558-tensor AdamW state with counters matching saved global steps, all RNG components, and identical frozen-parameter hashes. The source model bundles, six core source hashes, Stage 5 runtime-source hashes, train/validation file hashes, rights record and 372 photo hashes pass post-run checks. Existing artifacts are read-only to the trainer outside its unique new run directory. CUDA memory was released when the process exited. Console peak allocated memory was approximately 2440 MiB; this rounded console observation is not an exact independently sampled peak. The trainer's recorded wall interval includes per-epoch validation/checkpoint I/O and process delays; it is not active GPU time.

## Command executed

```powershell
& .\.venv-training\Scripts\python.exe .\scripts\train_taco_stage5.py --warmstart blend --epochs 2 --learning-rate 1e-5 --evaluate-baseline
```

Source and result changes remain local on feat/stage5-paper-category. No further epoch, new resolution experiment, test evaluation, deployment or checkpoint promotion follows automatically.

## High-Resolution Fine-Tuning at 384/576 Canvas Scale

To maximize detection accuracy and evaluate Swin-T multi-scale representations at target inference resolution, an authorized high-resolution fine-tuning run (\stage5-paper-558bcc18d6cf4556a966d8cfc320c968\) was executed on 2026-10-28:
- **Warm-start**: Stage 5 \lend_2575\ (25% Epoch 1 + 75% Epoch 2 from the initial run, verified parameter SHA-256 a8b832754cfbf16fa56e4742299c61a8f60cdba6cba9d6c9758ec85ad7b3e98\).
- **Input Geometry**: \shortest=384, longest=576\ (matching D-061 evaluation condition). Peak VRAM measured ~3306 MiB on GTX 1660 Ti.
- **Budget**: Exactly 1 bounded epoch (307 updates) at AdamW LR e-6\ with weight decay 1e-4 and FP32.
- **Validation**: 112 validation forwards at baseline and epoch end.

### High-Resolution Performance Progression

| Model State | Canvas Scale | Combined 65 AP | Combined 65 AP50 | Combined 65 AR100 | Legacy 47 AP | Paper AP (Annex) |
|---|---|---:|---:|---:|---:|---:|
| Run 1 Warm-Start Baseline | 256/384 | 0.057617 | 0.070768 | 0.147380 | 0.078141 | 0.000000 |
| Run 1 Epoch 2 Final | 256/384 | 0.135663 | 0.173659 | 0.247600 | 0.144434 | 0.507431 |
| Run 1 Epoch 2 @ Scale | 384/576 | 0.153029 | 0.199155 | 0.287338 | 0.173781 | 0.526727 |
| Model Soup \lend_2575\ | 384/576 | 0.171262 | 0.210596 | 0.296896 | 0.173157 | 0.473670 |
| **High-Res Epoch 1** | **384/576** | **0.202020** | **0.246182** | **0.335380** | **0.226047** | **0.586443** |
| High-Res Soup (20/80) | 384/576 | 0.194258 | 0.237781 | 0.328400 | 0.182191 | 0.551020 |

### Per-Class AP Comparison at High Resolution (Combined 65 images)

| Category | GT Objects | Run 1 Ep2 (256/384) | \lend_2575\ (384/576) | **High-Res Ep1 (384/576)** | High-Res Ep1 AR100 |
|---|---:|---:|---:|---:|---:|
| \cigarette\ | 26 | 0.030489 | 0.065575 | **0.111456** | 0.373077 |
| \plastic_bottle\ | 9 | 0.115248 | 0.153401 | **0.252475** | 0.366667 |
| \plastic_bag\ | 4 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| \plastic_wrapper\ | 13 | 0.051485 | 0.013201 | 0.015842 | 0.076923 |
| \plastic_film\ | 17 | 0.037018 | 0.054518 | 0.040660 | 0.552941 |
| \drink_can\ | 7 | 0.287129 | 0.411881 | **0.656436** | 0.657143 |
| \plastic_cup\ | 6 | 0.241254 | 0.319802 | **0.319802** | 0.316667 |
| \styrofoam_piece\ | 7 | 0.148515 | **0.148515** | 0.000000 | 0.000000 |
| \paper\ (all images) | 20 | 0.309834 | 0.374468 | **0.421508** | 0.675000 |

### Artifact Integrity
- Run Directory: uns/training/stage5-paper-558bcc18d6cf4556a966d8cfc320c968- Generated Checkpoints: \step-000100\, \step-000200\, \step-000300\, \epoch-01\.
- Post-processing: Interactive review tool (\scripts/taco_stage5_visual_review.py\) supports side-by-side 4-panel visual comparison across Ground Truth, Baseline, Stage 5 Run 1, and Stage 5 High-Res, with an interactive Overlap NMS filter toggle.
- Regression tests: 286 passed, 1 skipped, zero failures. No test split access or unapproved promotion.
