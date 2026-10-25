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
