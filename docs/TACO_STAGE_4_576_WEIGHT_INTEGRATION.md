# Stage 4: 384/576 checkpoint weight integration (D-061)

Owner-authorized inference-only evaluation, 2026-10-25. All four integrated states completed exactly 47 validation forwards each (188 total). Step 0, 221, 350 and 442 reference predictions and losses were reused without GPU reruns. No backward, optimizer updates, training-photo reads, test-split access, checkpoint writes, commit, push, merge or promotion.

This tests parameter-space averages of the same D-057 trajectory, not prediction ensembles or additional training. All evaluations use the unchanged Stage 3D adapter and native loss, eight-class prompt, 384/576 resize, 576-square padding, FP32, TF32 disabled and `bantai.taco-evaluation.v2`. The D-042 academic-use declaration and its uncertainties remain unchanged.

Private run: [scale576-integration-c70249cde2f2422cb89b22232fd2b525](../runs/training/scale576-integration-c70249cde2f2422cb89b22232fd2b525/record.json). Full evaluation JSONs, [summary](../runs/training/scale576-integration-c70249cde2f2422cb89b22232fd2b525/integration-summary.json), metrics/per-class/coverage CSVs, full per-object JSON and per-object CSV remain local.

## Exact parameter recipes

| State | Step221 weight | Step350 weight | Step442 weight |
| --- | --- | --- | --- |
| blend_350_442_5050 | 0 | 0.500000 | 0.500000 |
| blend_350_442_2575 | 0 | 0.250000 | 0.750000 |
| blend_221_442_5050 | 0.500000 | 0 | 0.500000 |
| soup_221_350_442 | 0.333333 | 0.333333 | 0.333333 |

## Overall metrics

| State | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | seven_class_AP_excluding_plastic_bag | validation_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| step0 | 0.147847 | 0.179470 | 0.165594 | 0.000000 | 0.007261 | 0.172653 | 0.426109 | 0.095418 | 9.720814 |
| step221 | 0.249891 | 0.286853 | 0.264765 | 0.060891 | 0.033495 | 0.303687 | 0.488141 | 0.181629 | 5.323725 |
| step350 | 0.262337 | 0.288625 | 0.278353 | 0.065773 | 0.033448 | 0.315487 | 0.428890 | 0.218249 | 4.880779 |
| step442 | 0.170508 | 0.196086 | 0.179972 | 0.075112 | 0.048109 | 0.206941 | 0.370118 | 0.190859 | 4.613316 |
| blend_350_442_5050 | 0.262599 | 0.288991 | 0.278057 | 0.066166 | 0.043174 | 0.314063 | 0.453701 | 0.222320 | 4.701619 |
| blend_350_442_2575 | 0.265563 | 0.292911 | 0.280572 | 0.071254 | 0.045737 | 0.316106 | 0.403085 | 0.231365 | 4.659642 |
| blend_221_442_5050 | 0.228873 | 0.258320 | 0.248536 | 0.056559 | 0.039870 | 0.275609 | 0.429970 | 0.184133 | 4.960877 |
| soup_221_350_442 | 0.256997 | 0.289875 | 0.280266 | 0.056683 | 0.041400 | 0.307597 | 0.447672 | 0.215210 | 4.912308 |

Seven-class AP is the arithmetic mean of the seven per-class AP values excluding plastic_bag. It does not change the official eight-class evaluation. Validation loss is the saved/evaluated mean native loss per image, not accuracy.

## All eight classes

Each cell is AP / AR100.

| Class | step0 | step221 | step350 | step442 | blend_350_442_5050 | blend_350_442_2575 | blend_221_442_5050 | soup_221_350_442 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cigarette | 0.014122 / 0.029167 | 0.051306 / 0.179167 | 0.076901 / 0.329167 | 0.082558 / 0.350000 | 0.078552 / 0.345833 | 0.081497 / 0.350000 | 0.063203 / 0.300000 | 0.066598 / 0.312500 |
| plastic_bottle | 0.053966 / 0.425000 | 0.189688 / 0.600000 | 0.236139 / 0.337500 | 0.180281 / 0.337500 | 0.204785 / 0.337500 | 0.193894 / 0.337500 | 0.172112 / 0.337500 | 0.180281 / 0.337500 |
| plastic_bag | 0.514851 / 0.700000 | 0.727723 / 0.950000 | 0.570957 / 1.000000 | 0.028053 / 0.500000 | 0.544554 / 1.000000 | 0.504950 / 0.500000 | 0.542052 / 1.000000 | 0.549505 / 0.950000 |
| plastic_wrapper | 0.074249 / 0.572727 | 0.041171 / 0.290909 | 0.029703 / 0.081818 | 0.128053 / 0.254545 | 0.084653 / 0.163636 | 0.154653 / 0.254545 | 0.070388 / 0.181818 | 0.029703 / 0.081818 |
| plastic_film | 0.002621 / 0.107692 | 0.028424 / 0.530769 | 0.025307 / 0.476923 | 0.033685 / 0.584615 | 0.028511 / 0.576923 | 0.029740 / 0.576923 | 0.026646 / 0.546154 | 0.024972 / 0.553846 |
| drink_can | 0.124851 / 0.214286 | 0.118812 / 0.114286 | 0.367162 / 0.385714 | 0.118812 / 0.114286 | 0.367162 / 0.385714 | 0.367162 / 0.385714 | 0.118812 / 0.114286 | 0.367162 / 0.385714 |
| plastic_cup | 0.335897 / 0.660000 | 0.404797 / 0.640000 | 0.386139 / 0.380000 | 0.386139 / 0.380000 | 0.386139 / 0.380000 | 0.386139 / 0.380000 | 0.431353 / 0.520000 | 0.431353 / 0.520000 |
| styrofoam_piece | 0.062220 / 0.700000 | 0.437207 / 0.600000 | 0.406391 / 0.440000 | 0.406483 / 0.440000 | 0.406438 / 0.440000 | 0.406466 / 0.440000 | 0.406420 / 0.440000 | 0.406403 / 0.440000 |

## Object and cigarette coverage

All counts are nonexclusive same-class GT coverage at IoU >=0.50 and score >=0.05. all900 uses all stored queries; top100 uses official retained queries. These are diagnostic counts, not official COCO recall, physical-object counts or calibrated probabilities.

### all900 @ 0.05

| State | Correct /75 | Cigarettes /24 | Image 777 cigarettes /7 | extremely small <4 | very small 4–<8 | small 8–<16 | larger ≥16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| step0 | 29 | 3 | 0 | 0/14 | 2/10 | 4/15 | 23/36 |
| step221 | 34 | 8 | 1 | 2/14 | 4/10 | 7/15 | 21/36 |
| step350 | 34 | 14 | 4 | 5/14 | 6/10 | 8/15 | 15/36 |
| step442 | 35 | 15 | 5 | 6/14 | 6/10 | 9/15 | 14/36 |
| blend_350_442_5050 | 34 | 13 | 4 | 4/14 | 6/10 | 8/15 | 16/36 |
| blend_350_442_2575 | 38 | 15 | 5 | 6/14 | 6/10 | 9/15 | 17/36 |
| blend_221_442_5050 | 31 | 11 | 3 | 3/14 | 5/10 | 7/15 | 16/36 |
| soup_221_350_442 | 32 | 11 | 3 | 3/14 | 5/10 | 8/15 | 16/36 |
### top100 @ 0.05

| State | Correct /75 | Cigarettes /24 | Image 777 cigarettes /7 | extremely small <4 | very small 4–<8 | small 8–<16 | larger ≥16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| step0 | 29 | 3 | 0 | 0/14 | 2/10 | 4/15 | 23/36 |
| step221 | 34 | 8 | 1 | 2/14 | 4/10 | 7/15 | 21/36 |
| step350 | 34 | 14 | 4 | 5/14 | 6/10 | 8/15 | 15/36 |
| step442 | 35 | 15 | 5 | 6/14 | 6/10 | 9/15 | 14/36 |
| blend_350_442_5050 | 34 | 13 | 4 | 4/14 | 6/10 | 8/15 | 16/36 |
| blend_350_442_2575 | 38 | 15 | 5 | 6/14 | 6/10 | 9/15 | 17/36 |
| blend_221_442_5050 | 31 | 11 | 3 | 3/14 | 5/10 | 7/15 | 16/36 |
| soup_221_350_442 | 32 | 11 | 3 | 3/14 | 5/10 | 8/15 | 16/36 |

Size-bin membership is locked to the same 256/384 reference geometry used by D-058 (14/10/15/36 objects), using its existing resized minimum-side bins. Bins are not recomputed per state or relabeled as 384/576 geometry.

## Result and replay selection

The highest eight-class AP among the four integrations is **blend_350_442_2575**, AP **0.265563**. This selects a saved diagnostic view only. Comparing averages on the same 47-image validation set adds selection bias; it establishes neither generalization nor a deployment/checkpoint-selection rule. No integrated weights are persisted or promoted.

The recovery is partial. The 25/75 blend retains E2 cigarette coverage (15/24 overall, 5/7 on Image 777), increases wrapper AP from 0.128053 to 0.154653, and restores drink-can AP to Step 350's 0.367162. Large-bin coverage rises from E2's 14/36 to 17/36, below E1's 21/36. Its bag AP 0.504950 remains below Step 350's 0.570957 and E1's0.727723; bag AR100 remains 0.50. Overall AR100 0.403085 also remains below Step 350 0.428890 and E1 0.488141. The AP advantage over Step350 is only 0.003226 on this validation set. The 50/50 350/442 blend has higher AR100 0.453701 than the selected 25/75 blend, demonstrating that highest AP does not dominate every metric. At the 0.05 floor, all900 and top100 diagnostic coverage counts happen to agree for every state.

## Integrity and execution evidence

All **50** bundles were verified before and **50** after execution. Protected audit: 1608 files, unchanged. Frozen parameters and persisted buffers are bitwise identical across step 221/350/442. Exactly 558 canonical trainable tensors enter CPU FP32 weighted addition; saved tied aliases resolve to the same parameter identity, preserving all ties. All other tensors are copied unchanged. Source checkpoint files and optimizer states are never overwritten; training_state.pt is hashed only with torch.load blocked.

| Integrated state | Changed trainable tensors vs221 | Forwards | Peak allocated bytes | CUDA bytes before process exit | Process exited |
| --- | --- | --- | --- | --- | --- |
| blend_350_442_5050 | 557 | 47 | 1208047616 | 33554432 | True |
| blend_350_442_2575 | 557 | 47 | 1208047616 | 33554432 | True |
| blend_221_442_5050 | 557 | 47 | 1208047616 | 33554432 | True |
| soup_221_350_442 | 558 | 47 | 1208047616 | 33554432 | True |

Every forward is guarded for inference mode, frozen parameters, FP32, tensor shape and the 47/state and 188/total budgets. Model hashes were unchanged after evaluation. After the first completed 47-image evaluation, the model object was released but the allocator still reported 32 MiB; the cleanup guard stopped before another state. That process then exited, releasing its CUDA context. The first evaluation was retained unchanged. Each remaining state ran once in a separate process, starting with zero allocated CUDA bytes, and that process exited before the next state. This guarantees context release between evaluations without repeating any forward. The cleanup stop and worker evidence remain in the run directory. The first 47 photo-read count is derived from the completed evaluator loop and query identities; remaining workers directly recorded 47 reads each. CPU-only COCO reproduction verifies the four reference metrics. Validation-photo preservation hashing is separate from decode/read counts. Training/test photos and test.json were not opened. Network and image exports were blocked.

Pre-execution guarded suite: 241 tests, zero failures/errors, 16 documented skips for training-execution fixtures and Windows symlink privilege. Four new CPU weight tests passed, including tied aliases, nonpersistent buffers and signed-zero bitwise guards; a saved-integration replay test checks source/recipe/metric/audit tampering. Two preflight stops occurred before any forward: an overly broad file guard rejected allowed annotation metadata, and the key-coverage guard initially included two runtime-only buffers absent from persisted state. Both guards were corrected and both zero-forward stop records retained. No inference was repeated. Bundle verification uses the freshly read per-phase SHA-256 inventory, avoiding duplicate reads of each large file within a phase.

The sole authorized prior-artifact edit is the requested D-060 manifest extension, applied after the unchanged execution audit; its previous contents are archived in this run. All old run-choice entries are preserved. Native GUI interaction and human photo judgments are separate from successful JSON loading/tests; human acceptance remains pending.

Native Windows Tk smoke checks passed in both standard and per-object modes: Image 777, all 47 IDs selectable, correct panel labels, image navigation, GT navigation and zoom/pan. Model construction/forward, CUDA initialization, checkpoint-state reads and exports were blocked; audited saved JSON hashes remained unchanged. This is automated runtime evidence; human visual acceptance remains pending. See `native-results.json` in the private run.

## PowerShell saved replay

From the repository root; each command starts Image 777 in all-validation scope. Previous/Next and the selector browse all 47 images. The three prediction panels show Step 350, Epoch 2 step 442, and the selected integration alongside GT. Close one window before opening the next.

```powershell
$review = "runs/training/d060-saved-review/review-manifest.json"
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576 (350 vs 442 vs integrated)" --gui --image-id 777
.venv-training/Scripts/python.exe -B scripts/taco_visual_review.py --saved-review $review --run-choice "42 / 1e-5 @ 384/576 (350 vs 442 vs integrated)" --gui --per-object --image-id 777
```
