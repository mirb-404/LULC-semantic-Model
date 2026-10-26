# Test run 6: results and findings

Date: 2026-10-26. Run folder: `Weights/Test_6/` (`Test_6.zip`). Notebook: the version with the GPU speed changes (commit `b4b5175`) and the overfitting changes (commit `5414256`). Method: `reports/Notebook method and pipeline.md`. Strategy and PCA background: `reports/literature survey_reiterate.md`.

## 1. Setup

| Item | Value |
|---|---|
| Mode | FINAL, `MAX_EPOCHS = 40`, `PATIENCE = 10`, `VAL_EVERY = 2` |
| Variants | `three_month`, `three_month_stats` (new), `three_month_pca`, `three_month_pca_stats`, `three_month_pca_stats_terrain` |
| Hardware | Datalab RTX A6000 (51 GB, compute capability 8.6), 32 CPU cores, `/dev/shm` 0.07 GB |
| Precision / batch | bfloat16, batch 64 |
| Data placement | train 7.6 GB and val + test 2.2 GB, all on the GPU; no DataLoader workers |
| Regularisation | weight decay 0.05 (not on biases / BatchNorm), label smoothing 0.1, Dropout2d 0.2, month dropout 0.15, VV/VH shifted separately |
| Data | 1958 / 327 / 302 patches from 134 / 22 / 20 sites (unchanged) |

## 2. Speed

- **196 epochs in 27 minutes**, about 8 s per epoch: 6-7 s for a training-only epoch, about 10 s with validation.
- Test_5_Full on the same GPU with the old data loading took 13-30 s per epoch.
- Holding all data on the GPU fixed the `/dev/shm` problem without any change to the Datalab container.

## 3. Results

| Variant | Channels | best val mIoU (epoch) | test OA | test mIoU | test mIoU vs baseline | train-val gap |
|---|---|---|---|---|---|---|
| three_month | 9 | 0.570 (34) | 0.843 | 0.585 | +0.026 | 0.184 |
| three_month_stats | 14 | 0.657 (36) | **0.914** | **0.742** | **+0.182** | 0.208 |
| three_month_pca | 15 | 0.666 (34) | 0.874 | 0.667 | +0.107 | 0.207 |
| three_month_pca_stats | 20 | 0.671 (26) | 0.865 | 0.617 | +0.058 | 0.212 |
| three_month_pca_stats_terrain (ablation) | 23 | **0.694** (34) | 0.890 | 0.701 | +0.142 | 0.208 |

The baseline is test OA 0.843 and mIoU 0.559 (commit `0811060`, old dataset). `three_month_pca_stats` stopped early at epoch 36. All the others ran the full 40 epochs.

### Test IoU per class

| Class | three_month | three_month_stats | three_month_pca | three_month_pca_stats | ..._terrain |
|---|---|---|---|---|---|
| Built-up | 0.943 | 0.951 | 0.948 | 0.950 | 0.958 |
| Cropland | 0.612 | **0.862** | 0.521 | 0.649 | 0.500 |
| Trees | 0.352 | **0.755** | 0.630 | 0.652 | 0.659 |
| Shrub & grass | 0.490 | **0.648** | 0.408 | 0.473 | 0.417 |
| Bare / sparse | 0.748 | **0.829** | 0.728 | 0.660 | 0.817 |
| Water | 0.514 | 0.657 | 0.634 | 0.558 | **0.868** |
| Wetland & mangrove | 0.516 | **0.702** | 0.651 | 0.352 | 0.651 |
| Snow & ice | 0.503 | 0.529 | **0.813** | 0.643 | 0.740 |

### Selected model (`BEST` = three_month_pca_stats) by region and site type

| Region | Patches | OA | mIoU |
|---|---|---|---|
| Africa | 49 | 0.820 | 0.420 |
| Asia | 109 | 0.924 | 0.629 |
| Europe | 32 | 0.963 | 0.501 |
| N. America | 64 | 0.689 | 0.378 |
| Oceania | 16 | 0.966 | 0.473 |
| S. America | 32 | 0.947 | 0.384 |

| Site type | Patches | OA | mIoU |
|---|---|---|---|
| bare | 16 | 1.000 | 1.000 |
| cropland | 17 | 0.840 | 0.430 |
| shrub_grass | 16 | 0.672 | 0.138 |
| snow_ice | 13 | 0.829 | 0.353 |
| trees | 16 | 0.905 | 0.250 |
| urban | 208 | 0.917 | 0.432 |
| wetland | 16 | 0.372 | 0.117 |

These per-group mIoU values come from few pixels per class inside each group. Read the OA column first.

## 4. Overfitting

| Variant | val loss minimum (epoch) | val loss at the end | train mIoU at the end |
|---|---|---|---|
| three_month | 0.692 (34) | 0.743 (40) | 0.758 |
| three_month_stats | 0.570 (34) | 0.573 (40) | 0.858 |
| three_month_pca | 0.566 (36) | 0.568 (40) | 0.870 |
| three_month_pca_stats | 0.583 (26) | 0.597 (36) | 0.888 |
| three_month_pca_stats_terrain | 0.539 (34) | 0.542 (40) | 0.901 |

**Better:**
- In Test_5_Full (60 epochs, weak regularisation), `three_month` val loss bottomed out at epoch 27 and then rose from 0.598 to 0.648. Now val loss reaches its minimum near the end (epoch 34-36 of 40) and stays flat for 4 of 5 variants.
- Train mIoU of `three_month` at the end fell from 0.83 to 0.76, so the regularisation acts.

**Not solved:**
- The train-val gap shrank only a little: 0.21-0.23 in Test_5, 0.18-0.21 now (baseline 0.31).
- `three_month_pca_stats`, the variant with the most channels and with PCA, still overfits: best epoch 26, then val loss rose and early stopping ended it at 36.
- `three_month` val mIoU (0.570) is lower than in Test_5 (0.599 at epoch 50 of 60). The stronger regularisation plus the shorter schedule may underfit the plain 9-channel model a little. The difference is within run-to-run noise.

## 5. Main finding: the PCA ablation

**The test results support the literature argument against PCA:**
- `three_month_stats` has no PCA, only the five order-independent statistics. It has the best test mIoU (0.742) and OA (0.914) of all variants, terrain included.
- It is best in 5 of 8 classes: Cropland +0.21 over `pca_stats`, Trees +0.10, Shrub +0.18, Bare +0.17, Wetland +0.35.
- The 0.125 lead over `three_month_pca_stats` is far above the 0.05 test noise measured in runs 1-2.
- This matches the survey's hypothesis. PCA loadings are tied to calendar months, so the same seasonal pattern looks different at sites in other hemispheres or climates. The statistics (VV min/max, VH std, VH spread, water months) do not depend on month order. On unseen test sites, adding PCA to the statistics lowered test mIoU.

**But validation does not show it:**

| | stats | pca | pca_stats |
|---|---|---|---|
| best val mIoU | 0.657 | 0.666 | 0.671 |

- The three yearly-feature variants are within 0.014 on val, which is noise for 22 val sites.
- The notebook correctly chooses by val, never by test, and picked `three_month_pca_stats`, the weakest of the three on test.
- The choice between them is therefore not decided by the data we are allowed to use for selection.

**Why val and test can disagree:**
- 22 val and 20 test sites; rare classes rest on 1-2 sites each.
- Test is 54 % built-up, val 35 %.
- Val and test contain different biomes and regions. One wetland or cropland site can move a class IoU by 0.3 or more: Wetland IoU ranges from 0.35 to 0.70 here.

**Honesty for the report:** the test scores have now been seen. Any decision taken after this point should be justified on val (more seeds) or on prior arguments (literature, simplicity), and the report must say that the test results were known.

## 6. Other findings

- **Temporal yearly context is the biggest lever.** Every yearly-feature variant beats plain `three_month` on val by 0.09-0.12. The literature says the same: temporal information matters most for SAR.
- **Terrain helps, but is excluded.** It has the best val (0.694) and a much better Water IoU (0.868 vs 0.56-0.66), thanks to HAND (height above drainage). The brief requires Sentinel-1 as the only input, so it stays a reported ablation.
- **Weak spots of the selected model:**
  - North America (mIoU 0.378): Anchorage and the boreal and snow sites.
  - shrub_grass sites (0.138).
  - wetland sites (OA 0.372).
  - Snow test IoU rests on 13 patches.
- **Built-up bias check:** Built-up IoU is 0.94-0.96 for every variant, and the test set is 54 % built-up, which inflates OA. Check the Built-up column of `confusion_matrix.png` for other classes predicted as built-up.

## 7. Recommendations

1. **Decide stats vs pca_stats on validation with more seeds.** Train `three_month_stats` and `three_month_pca_stats` with 3 seeds each, about 5-6 min per run at 8 s per epoch, about 35 min in total. Choose by mean val mIoU and report mean ± std.
2. **If they still tie, deliver `three_month_stats`.** The prior arguments favour it:
   - order-independent features, as in ESA WorldCover;
   - fewer channels (14 vs 20);
   - the prototype needs no PCA weights;
   - the survey predicted the PCA weakness before this run.
3. **Set `VAL_EVERY = 1`.** Validation now costs only about 3-4 s, and the best epoch becomes exact.
4. **Keep the regularisation.** Optionally restore 60 epochs for the plain `three_month` only if it is reported as a baseline. It is not a deliverable.
5. **For the report:**
   - say that test is 54 % built-up and that OA is inflated by it;
   - give per-class IoU and the noise level (about 0.05 test mIoU);
   - present terrain as an ablation;
   - state that the test scores were seen before the final model choice.
