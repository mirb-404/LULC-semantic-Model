# Run log

One entry per Kaggle run. Seed 42 unless noted. Test = 20 held-out sites, 302 patches, all 12 months.

## Baseline (before 2026-10-24)

- Notebook: first version (U-Net ResNet-34, three-month input, 60 epochs, early stopping).
- Dataset: v1.
- Test OA 0.843, mIoU 0.559. Train mIoU 0.79 vs val 0.47 (overfitting). Weak: trees vs cropland, boreal forest, snow.

## Run 1: 2026-10-25 morning (Kaggle sync `5da7e0f`)

- Notebook: 38-cell version (variants three_month, three_month_pca; CE + Dice; ReduceLROnPlateau; no EMA). DEV: 15 epochs, val months 0/4/8.
- Dataset: `optimalbhandari/sentine-1-all`, 1958 / 327 / 302 patches (dataset v3 patches).

| Variant | best val mIoU | test OA | test mIoU |
|---|---|---|---|
| three_month | 0.541 | 0.817 | 0.559 |
| three_month_pca | 0.644 | 0.835 | 0.546 |

Test IoU (three_month / three_month_pca): Built-up 0.935 / 0.947, Cropland 0.518 / 0.472, Trees 0.431 / 0.562, Shrub & grass 0.227 / 0.325, Bare 0.722 / 0.799, Water 0.679 / 0.535, Wetland 0.427 / 0.310, Snow 0.536 / 0.414.

## Run 2: 2026-10-25 afternoon (Kaggle sync `d714799`, local `Weights/Test_2`)

- Notebook and settings: identical to run 1.
- Dataset: `optimalbhandari/sentinel-1-2-0`, same patch counts as run 1. This notebook does not read `terrain/`.

| Variant | best val mIoU (epoch) | test OA | test mIoU |
|---|---|---|---|
| three_month | 0.530 (14) | 0.823 | 0.538 |
| three_month_pca | 0.634 (14) | 0.813 | 0.485 |

Test IoU (three_month / three_month_pca): Built-up 0.942 / 0.941, Cropland 0.556 / 0.361, Trees 0.478 / 0.614, Shrub & grass 0.152 / 0.273, Bare 0.739 / 0.651, Water 0.443 / 0.468, Wetland 0.611 / 0.155, Snow 0.383 / 0.418.

Per region (three_month): Africa 0.289, Asia 0.506, Europe 0.451, N. America 0.406, Oceania 0.399, S. America 0.371 mIoU.
Per focus group (three_month): bare 1.000, cropland 0.316, shrub_grass 0.032, snow_ice 0.149, trees 0.182, urban 0.335, wetland 0.216 mIoU.

## What runs 1 and 2 show together

- Runs 1 and 2 share code, data and seed. Their gap is run-to-run noise: test mIoU 0.02 (three_month) to 0.06 (three_month_pca), single-class IoU up to 0.24.
- PCA effect, stable across both runs: val mIoU +0.10; test Trees +0.13 to +0.14, Shrub & grass +0.10 to +0.12.
- PCA effect, unstable: test mIoU −0.01 to −0.05, Wetland −0.12 to −0.46, Cropland −0.05 to −0.20.
- Interpretation: the yearly PCA vector helps classes with a stable yearly signature (trees, shrub). It hurts seasonal classes on unseen test sites, probably because it encodes site-specific patterns (train-val gap 0.127 vs 0.054). The `three_month_pca_stats` variant (VV min/max, water months) targets exactly those classes.
- Method consequence: rank variants by val mIoU over all 12 months, use the full 60-epoch budget, use 2-3 seeds for the final comparison, and treat test differences below about 0.05 mIoU as noise.

## Run 4: 2026-10-25 evening (commit `5a6d075` "Test run 4", `Weights/Test_4`)

- Notebook: improved 44-cell version (EMA, warmup + cosine, CE + 0.5 Lovász, seasonal stats, terrain, GPU-side input and eval, val on all 12 months every 2nd epoch). DEV: 15 epochs, variants three_month_pca, three_month_pca_stats, three_month_pca_stats_terrain.
- Hardware: Datalab RTX A6000, torch 2.9.1+cu128. Small `/dev/shm`, so `NUM_WORKERS = 0`.
- Dataset: `Dataset/Dataset` with `terrain/` loaded. 1958 / 327 / 302 patches from 134 / 22 / 20 sites (same as runs 1 and 2). 14 static channels precomputed in 131 s (5.1 GB).
- `EMA_DECAY = 0.99` in the notebook (section 8 planned 0.998).

| Variant | channels | best val mIoU (epoch) | test OA | test mIoU | train-val gap |
|---|---|---|---|---|---|
| three_month_pca | 15 | 0.613 (15) | 0.878 | 0.664 | 0.224 |
| three_month_pca_stats | 20 | 0.679 (15) | 0.839 | 0.566 | 0.166 |
| three_month_pca_stats_terrain | 23 | 0.654 (15) | 0.832 | 0.558 | 0.216 |

Test IoU (pca / pca_stats / pca_stats_terrain): Built-up 0.943 / 0.944 / 0.947, Cropland 0.399 / 0.461 / 0.568, Trees 0.660 / 0.620 / 0.606, Shrub & grass 0.482 / 0.421 / 0.305, Bare 0.725 / 0.614 / 0.357, Water 0.631 / 0.672 / 0.490, Wetland 0.780 / 0.100 / 0.570, Snow 0.689 / 0.695 / 0.617.

`BEST` (by val mIoU): three_month_pca_stats. Its model card and weights are in `Weights/Test_4/` (`Test_4.zip`).

Per region (three_month_pca_stats): Africa 0.406, Asia 0.595, Europe 0.482, N. America 0.361, Oceania 0.429, S. America 0.376 mIoU.
Per focus group (three_month_pca_stats): bare 1.000, cropland 0.364, shrub_grass 0.121, snow_ice 0.333, trees 0.261, urban 0.413, wetland 0.060 mIoU.

Reading:
- Val ranks pca_stats > pca_stats_terrain > pca. Test ranks pca > pca_stats > pca_stats_terrain. The rankings disagree, and the test differences (up to 0.106 mIoU) come mostly from Wetland (0.10 to 0.78) and Bare (0.36 to 0.73). Each of these rests on about 16 test patches from one site. This is noise, not a variant effect.
- Seasonal stats, on val: +0.066 mIoU over pca. Terrain on top of stats: −0.025 val. One DEV run, so neither is conclusive.
- The new training setup lifts test mIoU for three_month_pca from 0.485-0.546 (runs 1-2) to 0.664, and test Shrub & grass from 0.27-0.33 to 0.48. With the run-to-run noise from runs 1-2 this is suggestive, not proven.
- Best epoch is 15 of 15 for all variants, and val mIoU still rose at the last epoch. 15 epochs is too short.
- Train-val gap grew to 0.17-0.22 (runs 1-2: 0.05-0.13). Probably because train mIoU is now measured on the faster-learning setup; watch it in the 60-epoch run.
- Epoch times vary from 9 s to 164 s with no pattern (validated epochs are not always slower). Likely shared hardware or the main-process data loading. Not a code issue.
- The `lr_scheduler.step()` before `optimizer.step()` warning comes from the AMP scaler skipping the first step. Harmless.

Next: the user runs with `DEV = False` (60 epochs).

## Run 5: 2026-10-26 (commit `f170b8d` "Test run 5 full", `Weights/Test_5_Full`, incomplete)

- Notebook: improved version before the speed changes. FINAL: 60 epochs, all 5 variants, VAL_EVERY 1. Datalab RTX A6000, `NUM_WORKERS = 0`.
- Datalab closed during `three_month_pca` (epoch 3). No test results.

| Variant | best val mIoU (epoch) | train mIoU at best | gap |
|---|---|---|---|
| single_month | 0.548 (45) | 0.760 | 0.21 |
| three_month | 0.599 (50) | 0.825 | 0.23 |

Reading:
- three_month beats single_month by 0.05 val mIoU: the month window helps.
- Overfitting: `three_month` val loss lowest at epoch 27 (0.598) and rising to 0.648 by epoch 60 while train loss fell; val mIoU flat after about epoch 30. 60 epochs is more than needed.
- 13-30 s per epoch.

Next: Test_6 with regularisation (weight decay 0.05, label smoothing 0.1, dropout 0.2, month dropout 0.15, 40 epochs) and the new `three_month_stats` variant. See `reports/Notebook method and pipeline.md`, section 9.

## Run 6: 2026-10-26 (`Weights/Test_6`, notebook commits `b4b5175` + `5414256`)

- FINAL, 40 epochs, PATIENCE 10, VAL_EVERY 2. Datalab RTX A6000, bf16, batch 64, all data on the GPU (train 7.6 GB, val + test 2.2 GB), no workers.
- New regularisation: weight decay 0.05, label smoothing 0.1, dropout 0.2, month dropout 0.15, VV/VH shift separate. New variant `three_month_stats`.
- 196 epochs in 27 min (about 8 s per epoch; Test_5 had 13-30 s).

| Variant | best val mIoU (epoch) | test OA | test mIoU | gap |
|---|---|---|---|---|
| three_month | 0.570 (34) | 0.843 | 0.585 | 0.184 |
| three_month_stats | 0.657 (36) | 0.914 | 0.742 | 0.208 |
| three_month_pca | 0.666 (34) | 0.874 | 0.667 | 0.207 |
| three_month_pca_stats | 0.671 (26, early stop 36) | 0.865 | 0.617 | 0.212 |
| three_month_pca_stats_terrain | 0.694 (34) | 0.890 | 0.701 | 0.208 |

Test IoU (three_month / stats / pca / pca_stats / terrain):
- Built-up 0.943 / 0.951 / 0.948 / 0.950 / 0.958
- Cropland 0.612 / 0.862 / 0.521 / 0.649 / 0.500
- Trees 0.352 / 0.755 / 0.630 / 0.652 / 0.659
- Shrub 0.490 / 0.648 / 0.408 / 0.473 / 0.417
- Bare 0.748 / 0.829 / 0.728 / 0.660 / 0.817
- Water 0.514 / 0.657 / 0.634 / 0.558 / 0.868
- Wetland 0.516 / 0.702 / 0.651 / 0.352 / 0.651
- Snow 0.503 / 0.529 / 0.813 / 0.643 / 0.740

`BEST` (S1-only, by val) = three_month_pca_stats. Its scores:
- Per region (mIoU): Africa 0.420, Asia 0.629, Europe 0.501, N. America 0.378, Oceania 0.473, S. America 0.384.
- Per site type (OA / mIoU): bare 1.000/1.000, cropland 0.840/0.430, shrub_grass 0.672/0.138, snow_ice 0.829/0.353, trees 0.905/0.250, urban 0.917/0.432, wetland 0.372/0.117.

Reading:
- Val loss minimum now at epoch 34-36 of 40 and flat at the end (Test_5: minimum at epoch 27, then rising). Exception: pca_stats (minimum at 26, rising). Gap only slightly smaller (0.18-0.21 vs 0.21-0.23).
- three_month_stats is best on test by 0.075-0.125 over the PCA variants (above the 0.05 noise), but on val stats / pca / pca_stats are within 0.014 (noise). Val and test disagree.
- three_month val 0.570 vs 0.599 in Test_5: possibly mild underfitting from the stronger regularisation, or noise.
- Terrain: best val, Water IoU 0.868 (HAND). Excluded as deliverable (S1-only brief).

Next: 3 seeds each of three_month_stats and three_month_pca_stats, choose by mean val; tie → three_month_stats (prior arguments). `VAL_EVERY = 1`.
