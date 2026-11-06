# Stage 5 Organic Hybrid Showcase ? D-082

## Evaluation contract and verification

The corrected hybrid prompt was preflighted on CPU with the pinned processor and `build_label_maps`: the nine-class baseline produced exactly 9 ordinary phrase spans and the hybrid prompt produced exactly 10 in the expected `0..9` order. The D-076 `best-inference` bundle files and champion parameter digest matched `cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422` before inference.

Exactly **8 CUDA forwards** were completed: 4 images ? 2 prompts. Inputs used 384/576 resize, 576 square pad, FP32, TF32 disabled. No backward/update, test JSON, or held-out photo access occurred. Existing official validation metrics were not changed; this is a four-image prompt diagnostic only. The raw query score is the mean sigmoid probability over each class phrase span, not calibrated correctness probability.

Prompt 9: `cigarette. plastic bottle. plastic bag. plastic wrapper. plastic film. drink can. plastic cup. styrofoam piece. paper.`

Prompt 10: `cigarette. plastic bottle. plastic bag. plastic wrapper. plastic film. drink can. plastic cup. styrofoam piece. paper. food waste.`

### Per-GT query summary

?Best any? is the geometrically highest-IoU valid query regardless of class, with its top-1 predicted class. ?Top Organic? is the highest `food_waste` phrase score across 900 queries. ?Best localized Organic? is the highest `food_waste` score among boxes with IoU ? 0.50. Rank is within the 900 queries sorted by the indicated class score. BANTAI category labels use the existing five-category mapping.

| Image | GT | Source GT / BANTAI label | Top Organic (q, score, IoU, rank) | Best localized Organic (q, score, IoU, rank) | 9-prompt best any (q, class / BANTAI, score, IoU) | 10-prompt best any (q, class / BANTAI, score, IoU) |
|---:|---:|---|---|---|---|---|
| 68 | 162 | Food Can / Metal | q4, 0.180122, 0.1483, 1 | q10, 0.145884, 0.5390, 4 | q7, paper / Paper, 0.012074, 0.9895 | q0, drink_can / Metal, 0.447053, 0.9829 |
| 68 | 163 | Food waste / Organic | q4, 0.180122, 0.8147, 1 | q4, 0.180122, 0.8147, 1 | q8, plastic_film / Plastic, 0.006234, 0.8901 | q9, food_waste / Organic, 0.138023, 0.8911 |
| 72 | 170 | Plastic film / Plastic | q0, 0.216378, 0.9867, 1 | q0, 0.216378, 0.9867, 1 | q6, paper / Paper, 0.010716, 0.9878 | q0, plastic_film / Plastic, 0.322920, 0.9867 |
| 72 | 171 | Food waste / Organic | q0, 0.216378, 0.4182, 1 | q1, 0.176177, 0.8997, 2 | q8, plastic_film / Plastic, 0.107981, 0.9203 | q8, paper / Paper, 0.012367, 0.8998 |
| 92 | 244 | Food waste / Organic | q0, 0.232682, 0.9686, 1 | q0, 0.232682, 0.9686, 1 | q0, plastic_wrapper / Plastic, 0.339414, 0.9689 | q0, plastic_wrapper / Plastic, 0.327066, 0.9686 |
| 841 | 2610 | Disposable food container / Plastic | q3, 0.182473, 0.0176, 1 | q0, 0.150312, 0.9695, 2 | q10, paper / Paper, 0.005646, 0.9846 | q2, paper / Paper, 0.017637, 0.9763 |
| 841 | 2611 | Food waste / Organic | q3, 0.182473, 0.9220, 1 | q3, 0.182473, 0.9220, 1 | q3, drink_can / Metal, 0.136264, 0.9302 | q3, food_waste / Organic, 0.182473, 0.9220 |

## 10-class vectors for every query overlapping an Organic GT at IoU ? 0.50

The table includes every valid 10-prompt query with IoU ? 0.50 against each Organic box, sorted by IoU. Values are phrase-mean sigmoid scores in the fixed class order shown in the header. These full rows also appear in `evaluation.json`; `gui-showcase.json` preserves all 900 query vectors per image for saved replay.

### Image 68 ? GT 163 (10 queries)

| Query | IoU | cigarette | plastic_bottle | plastic_bag | plastic_wrapper | plastic_film | drink_can | plastic_cup | styrofoam_piece | paper | food_waste |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q9 | 0.8911 | 0.072544 | 0.033826 | 0.057445 | 0.097596 | 0.058888 | 0.027603 | 0.043727 | 0.015577 | 0.044345 | 0.138023 |
| q8 | 0.8853 | 0.004544 | 0.003786 | 0.003745 | 0.003957 | 0.005786 | 0.002219 | 0.003409 | 0.002419 | 0.005402 | 0.006588 |
| q4 | 0.8147 | 0.101512 | 0.044868 | 0.073246 | 0.137098 | 0.087703 | 0.045748 | 0.061476 | 0.021011 | 0.065444 | 0.180122 |
| q139 | 0.7289 | 0.004837 | 0.001781 | 0.001966 | 0.002220 | 0.002745 | 0.000685 | 0.001301 | 0.000992 | 0.003308 | 0.004441 |
| q252 | 0.6709 | 0.002754 | 0.001361 | 0.001329 | 0.001302 | 0.001771 | 0.000731 | 0.001119 | 0.001255 | 0.001996 | 0.002081 |
| q3 | 0.6612 | 0.090540 | 0.045194 | 0.078839 | 0.148985 | 0.103552 | 0.053227 | 0.072623 | 0.024562 | 0.079903 | 0.178520 |
| q13 | 0.6525 | 0.002575 | 0.002282 | 0.002067 | 0.001933 | 0.002921 | 0.001328 | 0.002013 | 0.001550 | 0.002801 | 0.003206 |
| q47 | 0.6357 | 0.002254 | 0.001464 | 0.001404 | 0.001293 | 0.001814 | 0.000931 | 0.001239 | 0.001289 | 0.001817 | 0.002159 |
| q18 | 0.5655 | 0.003385 | 0.002942 | 0.003050 | 0.003236 | 0.004910 | 0.001822 | 0.002833 | 0.002294 | 0.004995 | 0.005526 |
| q17 | 0.5533 | 0.087855 | 0.035170 | 0.062978 | 0.135876 | 0.098932 | 0.044295 | 0.053462 | 0.021354 | 0.077147 | 0.168679 |

### Image 72 ? GT 171 (31 queries)

| Query | IoU | cigarette | plastic_bottle | plastic_bag | plastic_wrapper | plastic_film | drink_can | plastic_cup | styrofoam_piece | paper | food_waste |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q8 | 0.8998 | 0.007990 | 0.004515 | 0.006999 | 0.005732 | 0.009575 | 0.005290 | 0.005505 | 0.002889 | 0.012367 | 0.008751 |
| q1 | 0.8997 | 0.043643 | 0.054399 | 0.081365 | 0.089227 | 0.136908 | 0.053467 | 0.080359 | 0.024142 | 0.097603 | 0.176177 |
| q17 | 0.8579 | 0.002143 | 0.001370 | 0.001990 | 0.001903 | 0.002414 | 0.000964 | 0.001221 | 0.001024 | 0.003760 | 0.002289 |
| q13 | 0.7981 | 0.002423 | 0.001564 | 0.002078 | 0.001990 | 0.002613 | 0.001159 | 0.001526 | 0.000943 | 0.004172 | 0.002900 |
| q29 | 0.7807 | 0.020327 | 0.023961 | 0.047890 | 0.045718 | 0.071672 | 0.020844 | 0.034954 | 0.010964 | 0.054951 | 0.082169 |
| q70 | 0.7124 | 0.004256 | 0.002639 | 0.005650 | 0.006244 | 0.007940 | 0.002545 | 0.004073 | 0.002830 | 0.008217 | 0.014691 |
| q11 | 0.6842 | 0.025341 | 0.022729 | 0.032442 | 0.033849 | 0.051738 | 0.024055 | 0.037360 | 0.011298 | 0.038476 | 0.085658 |
| q473 | 0.6598 | 0.001532 | 0.000549 | 0.001295 | 0.001017 | 0.001286 | 0.000375 | 0.000693 | 0.000539 | 0.001581 | 0.001637 |
| q72 | 0.6539 | 0.000973 | 0.000405 | 0.000896 | 0.000782 | 0.000937 | 0.000279 | 0.000474 | 0.000377 | 0.001280 | 0.001348 |
| q110 | 0.6106 | 0.005851 | 0.003820 | 0.005183 | 0.004297 | 0.006671 | 0.003596 | 0.004471 | 0.001435 | 0.005270 | 0.013606 |
| q79 | 0.6072 | 0.002508 | 0.000890 | 0.001294 | 0.001072 | 0.001512 | 0.000759 | 0.000946 | 0.000591 | 0.002149 | 0.002354 |
| q314 | 0.5997 | 0.002700 | 0.001721 | 0.004057 | 0.005174 | 0.005096 | 0.001140 | 0.002029 | 0.002556 | 0.007910 | 0.006301 |
| q555 | 0.5992 | 0.001438 | 0.000592 | 0.001212 | 0.000984 | 0.001227 | 0.000403 | 0.000642 | 0.000490 | 0.001728 | 0.001475 |
| q21 | 0.5954 | 0.001017 | 0.000589 | 0.001110 | 0.000955 | 0.001295 | 0.000387 | 0.000604 | 0.000475 | 0.001805 | 0.001324 |
| q384 | 0.5890 | 0.005150 | 0.002946 | 0.004710 | 0.004800 | 0.006791 | 0.002873 | 0.004135 | 0.002215 | 0.006375 | 0.014208 |
| q81 | 0.5850 | 0.001505 | 0.000586 | 0.001457 | 0.001264 | 0.001476 | 0.000402 | 0.000757 | 0.000544 | 0.001866 | 0.002332 |
| q27 | 0.5811 | 0.001143 | 0.000820 | 0.001497 | 0.001489 | 0.001661 | 0.000482 | 0.000691 | 0.000581 | 0.002428 | 0.001694 |
| q6 | 0.5787 | 0.028029 | 0.031874 | 0.086044 | 0.130927 | 0.125629 | 0.023559 | 0.050297 | 0.027510 | 0.140509 | 0.107939 |
| q91 | 0.5727 | 0.002763 | 0.001385 | 0.002538 | 0.002560 | 0.003153 | 0.001329 | 0.002117 | 0.001163 | 0.003835 | 0.006943 |
| q259 | 0.5726 | 0.002004 | 0.000778 | 0.001308 | 0.001193 | 0.001600 | 0.000668 | 0.000858 | 0.000739 | 0.002194 | 0.002297 |
| q30 | 0.5676 | 0.011109 | 0.005201 | 0.010326 | 0.009198 | 0.015147 | 0.005132 | 0.006756 | 0.003601 | 0.018404 | 0.013666 |
| q75 | 0.5675 | 0.001140 | 0.000556 | 0.001075 | 0.000956 | 0.001121 | 0.000368 | 0.000627 | 0.000470 | 0.001676 | 0.001440 |
| q89 | 0.5621 | 0.010414 | 0.013693 | 0.030388 | 0.029004 | 0.038650 | 0.010670 | 0.017730 | 0.009982 | 0.030335 | 0.043920 |
| q125 | 0.5605 | 0.003630 | 0.001798 | 0.003658 | 0.004872 | 0.004710 | 0.001370 | 0.002513 | 0.002558 | 0.006387 | 0.007145 |
| q210 | 0.5589 | 0.001003 | 0.000721 | 0.001475 | 0.001315 | 0.001425 | 0.000465 | 0.000729 | 0.000763 | 0.002315 | 0.001592 |
| q593 | 0.5524 | 0.002020 | 0.000686 | 0.001181 | 0.001191 | 0.001539 | 0.000574 | 0.000631 | 0.000903 | 0.002608 | 0.001909 |
| q271 | 0.5328 | 0.002072 | 0.000590 | 0.001334 | 0.001235 | 0.001596 | 0.000578 | 0.000884 | 0.000830 | 0.001919 | 0.003441 |
| q808 | 0.5194 | 0.004073 | 0.003435 | 0.008236 | 0.011768 | 0.009119 | 0.002754 | 0.004653 | 0.005587 | 0.015997 | 0.014705 |
| q165 | 0.5178 | 0.007104 | 0.005272 | 0.010991 | 0.012961 | 0.014033 | 0.004550 | 0.007492 | 0.005526 | 0.014899 | 0.017015 |
| q187 | 0.5120 | 0.001438 | 0.000730 | 0.001250 | 0.001078 | 0.001294 | 0.000501 | 0.000757 | 0.000554 | 0.002006 | 0.001611 |
| q119 | 0.5023 | 0.010649 | 0.004906 | 0.006825 | 0.007422 | 0.010651 | 0.005581 | 0.007474 | 0.003264 | 0.010033 | 0.024309 |

### Image 92 ? GT 244 (14 queries)

| Query | IoU | cigarette | plastic_bottle | plastic_bag | plastic_wrapper | plastic_film | drink_can | plastic_cup | styrofoam_piece | paper | food_waste |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q0 | 0.9686 | 0.073347 | 0.166398 | 0.171464 | 0.327066 | 0.325154 | 0.109815 | 0.110729 | 0.173834 | 0.229766 | 0.232682 |
| q1 | 0.9662 | 0.050407 | 0.028166 | 0.026052 | 0.024754 | 0.059351 | 0.030695 | 0.025273 | 0.020174 | 0.030628 | 0.054089 |
| q10 | 0.9615 | 0.002751 | 0.002403 | 0.002348 | 0.002319 | 0.003088 | 0.002133 | 0.001760 | 0.002304 | 0.002638 | 0.003798 |
| q3 | 0.9595 | 0.012846 | 0.009180 | 0.007403 | 0.006186 | 0.014276 | 0.009074 | 0.006535 | 0.006410 | 0.010282 | 0.015523 |
| q7 | 0.9540 | 0.002433 | 0.001619 | 0.001546 | 0.001487 | 0.002065 | 0.001360 | 0.001165 | 0.001616 | 0.001840 | 0.002469 |
| q8 | 0.9415 | 0.002882 | 0.002243 | 0.002180 | 0.002114 | 0.002826 | 0.002057 | 0.001783 | 0.002113 | 0.002437 | 0.003611 |
| q29 | 0.9330 | 0.002358 | 0.001422 | 0.001395 | 0.001336 | 0.001840 | 0.001159 | 0.001000 | 0.001502 | 0.001797 | 0.002123 |
| q38 | 0.9017 | 0.002094 | 0.001167 | 0.001180 | 0.001107 | 0.001531 | 0.000928 | 0.000788 | 0.001325 | 0.001580 | 0.001762 |
| q47 | 0.8583 | 0.002281 | 0.001097 | 0.000979 | 0.000888 | 0.001368 | 0.000872 | 0.000794 | 0.001061 | 0.001302 | 0.001666 |
| q16 | 0.7736 | 0.032156 | 0.024563 | 0.019368 | 0.028607 | 0.061289 | 0.015747 | 0.012904 | 0.020665 | 0.026452 | 0.045334 |
| q13 | 0.7640 | 0.002255 | 0.001243 | 0.001145 | 0.001071 | 0.001902 | 0.001090 | 0.000875 | 0.001239 | 0.001470 | 0.002176 |
| q5 | 0.7360 | 0.003116 | 0.001070 | 0.000850 | 0.000731 | 0.001602 | 0.000887 | 0.000720 | 0.001019 | 0.001221 | 0.001827 |
| q124 | 0.6080 | 0.007982 | 0.003294 | 0.003008 | 0.003943 | 0.008345 | 0.002231 | 0.001678 | 0.002895 | 0.003876 | 0.007923 |
| q378 | 0.5104 | 0.001762 | 0.000512 | 0.000459 | 0.000368 | 0.001111 | 0.000487 | 0.000328 | 0.000715 | 0.000668 | 0.000993 |

### Image 841 ? GT 2611 (7 queries)

| Query | IoU | cigarette | plastic_bottle | plastic_bag | plastic_wrapper | plastic_film | drink_can | plastic_cup | styrofoam_piece | paper | food_waste |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q3 | 0.9220 | 0.108177 | 0.134011 | 0.080990 | 0.130861 | 0.174605 | 0.150863 | 0.136575 | 0.050158 | 0.094798 | 0.182473 |
| q7 | 0.9150 | 0.026019 | 0.024996 | 0.020204 | 0.017642 | 0.030359 | 0.020810 | 0.021156 | 0.012053 | 0.023812 | 0.027843 |
| q20 | 0.8820 | 0.009135 | 0.005394 | 0.003769 | 0.003592 | 0.005696 | 0.004426 | 0.004887 | 0.003981 | 0.008609 | 0.006688 |
| q10 | 0.8597 | 0.015337 | 0.006536 | 0.004202 | 0.003801 | 0.006015 | 0.005338 | 0.006324 | 0.005451 | 0.011102 | 0.007776 |
| q30 | 0.7424 | 0.009068 | 0.004638 | 0.003255 | 0.003006 | 0.004668 | 0.003875 | 0.004435 | 0.003523 | 0.007634 | 0.005698 |
| q27 | 0.5366 | 0.003463 | 0.002063 | 0.002054 | 0.001566 | 0.002556 | 0.001641 | 0.001883 | 0.001829 | 0.003255 | 0.002354 |
| q65 | 0.5327 | 0.004300 | 0.002825 | 0.002076 | 0.001920 | 0.003305 | 0.002199 | 0.002446 | 0.002124 | 0.004238 | 0.003143 |

## Reading the four Organic cases

- Image 68 / GT 163: the 10-prompt pass has a localized `food_waste` query q9 (score 0.138023, IoU 0.8911, top-1 rank 6); the top Organic score is q4 (0.180122, IoU 0.8147, rank 1).
- Image 72 / GT 171: q1 is the best localized Organic prediction (0.176177, IoU 0.8997, rank 2), while the highest Organic score is q0 (0.216378, IoU 0.4182, rank 1), whose box falls below the 0.50 localization cutoff.
- Image 92 / GT 244: the best Organic query is q0 (0.232682, IoU 0.9686, rank 1), but its top-1 class remains `plastic_wrapper` (BANTAI Plastic).
- Image 841 / GT 2611: q3 predicts `food_waste` / Organic at 0.182473, IoU 0.9220, rank 1.

These four examples show that the added phrase can align strongly on three GT boxes, while Image 92 remains Plastic by top-1 class and Image 72 has a higher-scoring poorly localized Organic query. This is a small diagnostic sample and does not establish generalization or justify adding a supervised tenth class. The D-080 audit found only four clean Tier 2 Food waste images/boxes and no fixed validation/test Organic support.

## Saved artifact and GUI replay

Private run: `runs/training/stage5-organic-hybrid-505b80d9f7cf430a96fcea4c7471d7f4` (`evaluation.json`, `gui-showcase.json`, `manifest.json`). The replay manifest verifies the exact checkpoint parameter digest, both artifact hashes, four bound photo hashes, and 8-forward count. `scripts/taco_stage5_visual_review.py` auto-discovers this completed artifact and adds the optional filter while leaving the 65-image view as the default. Native Tk replay passed 65 showcase renders, verified 9-/10-prompt rows for all four images, and reopened frozen-validation Image 777 with zero TclError.

PowerShell command to open Image 68 in the Organic Showcase filter (then browse 72, 92, and 841 from the Filter/Image controls):

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_stage5_visual_review.py --gui --image-id 68 --score-threshold 0.10 --box-top-k 15
```

PowerShell command to open the unchanged frozen validation view at Image 777:

```powershell
Set-Location 'C:\Users\Justine\Documents\Python\Project Image-object Detection'
& .\.venv-training\Scripts\python.exe -B .\scripts\taco_stage5_visual_review.py --gui --image-id 777 --score-threshold 0.10 --box-top-k 15
```
