# Stage 5 High-Resolution (448/672) 1-Epoch Fine-Tuning & Multi-Scale Evaluation (D-086)

Owner authorization **D-086** executes a strictly bounded high-resolution (`448/672`) fine-tuning run, multi-scale weight soup evaluation, and CPU multi-scale query fusion audit on branch `feat/stage5-paper-category`.

---

## 1. Executive Summary & Core Results

- **Authorization**: `D-086`
- **Run Directory**: `runs/training/stage5-d086-highres-20261031-a`
- **Active Supervised Taxonomy**: Strictly frozen 9 active classes (`0..8`: `cigarette`, `plastic_bottle`, `plastic_bag`, `plastic_wrapper`, `plastic_film`, `drink_can`, `plastic_cup`, `styrofoam_piece`, `paper`). Zero unaudited classes added.
- **Dataset**: `exports/taco/strict-clean-stage5-d083-core-v1` (`379` train images / `731` annotations; `65` validation images / `109` annotations).
- **Rights Digest**: Verified `work/stage5-paper/rights_d083_core.json`.
- **Warm-Start Checkpoint**: Verified D-083 champion bundle `runs/training/stage5-d083-core-20261031-a/checkpoints/best-inference` (`soup_d083_d074_5050`, parameter SHA-256 `82c0a5bf3a104db5f5ff35e1cbfc1bd239595ec9946e22cf085986985d95fddc`).
- **Frozen Backbone Verification**: Verified Swin-T and BERT parameter SHA-256 `161c9814bffe1657ae0debd7e21b7ed1c41f29f3ba43fffe879ab31ba36a0b67` before, during, and after training.
- **Strict Guarding**: Zero test-split reads (`test.json` unopened; 0 held-out test photos opened). Integrity snapshots verified before and after the run.
- **Hardware Adaptation**: High-resolution training at `448/672` square pad `(1, 3, 672, 672)` on the NVIDIA GeForce GTX 1660 Ti (6144 MiB VRAM) was stabilized under a 5800 MiB safety gate by offloading AdamW optimizer state tracking (`exp_avg`, `exp_avg_sq`) to CPU while keeping all forward, adapter, and backward passes on CUDA. Peak VRAM was **`4135.3 MiB`** (well below the 5800 MiB ceiling).
- **Execution Budget**: Exactly 379 training updates (1 epoch at batch size 1) + exactly 448 validation forwards (4 states $\times$ 112 forwards: 65 combined + 47 legacy). Total elapsed time: `1648.45` seconds (~27.5 minutes). Mean training loss: `4.675446`.
- **No-Regression Promotion Rule**: The best single-pass candidate within D-086 is `soup_d086_d083_2575_std384` with combined-65 AP = **`0.235485`**. Because this does not strictly exceed the verified D-083 champion baseline AP (**`0.236129`**), the repository default inference bundle remains unchanged at the D-083 champion.
- **Multi-Scale Query Fusion Breakthrough**: Fusing standard-resolution (`384/576`) queries with high-resolution (`448/672`) queries via class-aware IoU 0.55 NMS yields a **new benchmark high AP of `0.244213`** (`+0.008084` over D-083 champion) on 9-Specific detection and **`0.422337`** (`+0.021089` over D-083 champion) on BANTAI 5-Category aggregation.

---

## 2. D-086 Candidate State Evaluations

Each state was evaluated across the 65 combined validation images (9-Specific) and the 47 frozen legacy validation images (8-Class original prompt) using deterministic RNG:

| State | Evaluation Resolution | Combined-65 AP | Combined-65 AP50 | Combined-65 AR100 | Legacy-47 AP | Legacy-47 AP50 | Legacy-47 AR100 | Mean Val Loss |
|---|---|---|---|---|---|---|---|---|
| `D086-HR448 E1` | `448/672` | `0.223966` | `0.259868` | `0.363111` | `0.259092` | `0.300713` | `0.369474` | `3.640584` |
| `soup_d086_d083_5050_hr448` | `448/672` | `0.221398` | `0.257867` | `0.359390` | `0.258351` | `0.299539` | `0.367391` | `3.699244` |
| `soup_d086_d083_5050_std384` | `384/576` | `0.233953` | `0.279831` | `0.374361` | `0.254587` | `0.295231` | `0.392048` | `3.956932` |
| **`soup_d086_d083_2575_std384`** | `384/576` | **`0.235485`** | **`0.280866`** | **`0.374587`** | **`0.252930`** | **`0.294963`** | **`0.390182`** | `3.977407` |

*Reference D-083 Champion (`soup_d083_d074_5050` @ 384/576)*: Combined-65 AP = `0.236129`, AP50 = `0.282686`, AR100 = `0.378128`; Legacy-47 AP = `0.252820`.

### Parameter Checkpoint Digests
- `D086-HR448 E1`: `c101467e18436f0157cb110c370f8be4b482945f82e4872978a0033cb8973f1c`
- `soup_d086_d083_5050`: `522e12757ad67413c6a103dc328b0c585f3d2db66ae46151ffb36359b595719d`
- `soup_d086_d083_2575`: `202ee007132838e381c43a7d9ed4b82d7dae728e0325e464c6c562a0b9ab92e7`

---

## 3. Per-Class 9-Specific AP & Resolution Dynamics

Comparing per-class AP highlights the distinct complementary strengths of high-resolution vs standard-resolution inference:

| Class | D-083 Champion (`384/576`) | `D086-HR448 E1` (`448/672`) | `soup 50:50` (`448/672`) | `soup 50:50` (`384/576`) | `soup 25:75` (`384/576`) | High-Res Δ vs Champion |
|---|---|---|---|---|---|---|
| `cigarette` | `0.111154` | **`0.116874`** | `0.116277` | `0.108441` | `0.112458` | **`+0.005720`** |
| `plastic_bottle` | `0.331400` | **`0.490174`** | `0.468120` | `0.331400` | `0.331400` | **`+0.158774`** |
| `plastic_bag` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| `plastic_wrapper` | `0.015842` | **`0.019650`** | `0.020169` | `0.000000` | `0.000000` | **`+0.003808`** |
| `plastic_film` | `0.045093` | **`0.068783`** | `0.067870` | `0.045210` | `0.045544` | **`+0.023690`** |
| `drink_can` | `0.656436` | `0.555941` | `0.555941` | `0.670297` | **`0.670297`** | `-0.100495` |
| `plastic_cup` | `0.395545` | `0.319802` | `0.319802` | `0.384724` | `0.395545` | `-0.075743` |
| `styrofoam_piece` | `0.148515` | `0.074257` | `0.074257` | `0.148515` | `0.148515` | `-0.074257` |
| `paper` | `0.421176` | `0.370209` | `0.370149` | `0.416993` | `0.415607` | `-0.050967` |

### Scale Bucket Analysis
High-resolution training and inference fundamentally reshape detection performance by object scale:
- **`AP_small`**: Rose from `0.062222` (D-083 Champion) to **`0.072489`** in `D086-HR448 E1` and **`0.073580`** in `soup_d086_d083_5050_hr448` (`AR_small` jumped `0.120833` $\to$ **`0.175000`**, a `+44.8%` relative gain in small object recall).
- **`AP_medium`**: Jumped from `0.062677` (D-083 Champion) to **`0.194330`** in `D086-HR448 E1` (`+0.131653`, over 3$\times$ improvement), with `AR_medium` jumping from `0.250000` to **`0.328571`**.
- **`AP_large`**: Decreased slightly from `0.266914` to `0.245077` at `448/672`, as large contextual objects experience receptive field contraction at higher zoom.

---

## 4. Multi-Scale Query Fusion Audit (`multiscale-fusion-audit.json`)

To leverage both the superior small/medium object accuracy of high-resolution (`448/672`) inference and the large-object context of standard-resolution (`384/576`) inference, a CPU-only multi-scale query fusion audit was performed by pooling the 900 raw queries from D-083 Champion @ 384/576 with the 900 raw queries from D086-HR448 E1 @ 448/672, followed by class-aware IoU 0.55 NMS (keeping top 100 queries per image):

| Metric | D-083 Champion (`384/576`) | `D086-HR448 E1` (`448/672`) | Multi-Scale Query Fusion | Fusion Gain vs Champion |
|---|---|---|---|---|
| **9-Specific AP** | `0.236129` | `0.223966` | **`0.244213`** | **`+0.008084`** |
| **9-Specific AP50** | `0.282686` | `0.259868` | **`0.291926`** | **`+0.009240`** |
| **9-Specific AP75** | `0.254133` | `0.238478` | **`0.266855`** | **`+0.012722`** |
| **9-Specific AR100** | `0.378128` | `0.363111` | **`0.411469`** | **`+0.033341`** |
| **BANTAI 5-Category AP** | `0.401248` | `0.369499` | **`0.422337`** | **`+0.021089`** |
| **BANTAI 5-Category AP50** | `0.490408` | `0.437606` | **`0.517015`** | **`+0.026607`** |
| **BANTAI 5-Category AP75** | `0.421714` | `0.395123` | **`0.448412`** | **`+0.026698`** |
| **BANTAI 5-Category AR100** | `0.606422` | `0.585385` | **`0.637740`** | **`+0.031318`** |

### Per-Class Fused Performance
- `cigarette`: AP = **`0.122242`** (`+0.011088` vs Champion)
- `plastic_bottle`: AP = **`0.419554`** (`+0.088154` vs Champion)
- `plastic_bag`: AP = `0.000000`
- `plastic_wrapper`: AP = **`0.041867`** (`+0.026025` vs Champion)
- `plastic_film`: AP = **`0.068454`** (`+0.023361` vs Champion)
- `drink_can`: AP = **`0.680693`** (`+0.024257` vs Champion)
- `plastic_cup`: AP = `0.357673`
- `styrofoam_piece`: AP = `0.074257`
- `paper`: AP = **`0.433174`** (`+0.011998` vs Champion)

### BANTAI 5-Category Breakdown (Fused)
- **`Metal`**: AP = **`0.680693`**, AR100 = `0.685714`
- **`Plastic`**: AP = **`0.453241`**, AR100 = `0.751786`
- **`Paper`**: AP = **`0.433174`**, AR100 = `0.725000`
- **`Potentially Hazardous`**: AP = **`0.122242`**, AR100 = `0.388462`
- **`Organic`**: Unannotated on validation split (`None`).

---

## 5. D-079 Score-Preserving Consensus Audit

Applying the D-079 consensus profile to the best D-086 single-pass state (`soup_d086_d083_2575_std384`):
- All bounding box coordinates and query scores strictly preserved (`all_score_bbox_query_values_preserved: true`).
- Changed queries: `2`.
- **Baseline 9-Specific AP**: `0.235485` $\to$ **Consensus 9-Specific AP**: **`0.238178`** (`+0.002693`).
- **Baseline BANTAI 5-Category AP**: `0.401248` $\to$ **Consensus BANTAI 5-Category AP**: **`0.410301`** (`+0.009053`).

---

## 6. Verification and Reproducibility

- Unit tests: `tests/test_stage5_d086_highres.py` passed (`0.614s`).
- Guarded suite: `.venv-training\Scripts\python.exe -B scripts\verify_stage4_guarded.py` verified zero writes to protected artifacts and zero test splits opened.
