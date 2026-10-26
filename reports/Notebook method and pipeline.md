# Notebook method and pipeline (`Prototype.ipynb`)

Date: 2026-10-26. State: commit `b4b5175` plus the overfitting changes for **Test_6** (not run yet). This document describes what the notebook does, step by step, and why. It draws on `reports/literature survey_reiterate.md` (strategy check and the PCA question) and `research_notes/run_log.md` (all runs).

## 1. Problem and constraints

- **Task:** semantic segmentation of land use / land cover (LULC) into 8 classes: Built-up, Cropland, Trees, Shrub & grass, Bare / sparse, Water, Wetland & mangrove, Snow & ice.
- **Input:** Sentinel-1 only (exam brief: "uses Sentinel-1 data as its only input"). Terrain is trained only as an ablation and can never be the delivered model.
- **Output used by the prototype:** 12 monthly maps for any location, class percentages per month, month-to-month change and total change in percentage points.
- **Hardware:** Kaggle T4 (16 GB, 4 CPU cores) or Datalab RTX A6000 (48 GB, small `/dev/shm`). The notebook adapts to both.

## 2. Data (notebook sections 2-3)

| Item | Value |
|---|---|
| Sites | 176 worldwide (cities plus rural and biome-gap sites), year 2021 |
| Tile | 1024 x 1024 px at 10 m (10.24 km) |
| Radar | 12 monthly composites x VV, VH, stored as uint8: `dB = code x 0.2 - 35`, code 0 = no data |
| Labels | pixels where ESA WorldCover and Google Dynamic World agree; disagreement = ignore (-1) |
| Split | by site: 134 train / 22 val / 20 test sites; 1958 / 327 / 302 patches of 256 x 256 px |
| Class shares (train) | Built-up 30.8 %, Trees 20.3 %, Cropland 11.3 %, Shrub 9.5 %, Bare 8.1 %, Water 8.0 %, Snow 6.5 %, Wetland 5.4 % |

- **Train:** the full tiles stay in memory, so patches can be cut at random positions.
- **Val and test:** fixed patches, so scores are comparable between runs.
- **Split by site:** a test place is never seen in training, which is stricter than a random patch split.

## 3. Hardware check (section 1, new cell)

This cell detects the GPU, its memory and generation, the CPU cores and the `/dev/shm` size, then sets:

- **Mixed precision:** bfloat16 on Ampere or newer (A6000), float16 with `GradScaler` on older GPUs (T4).
- **Batch size:** 64 with 24 GB or more of GPU memory, else 32.
- **DataLoader workers:** 0 if `/dev/shm` < 1 GB (otherwise the workers die with "Bus error"), else up to 8.

Section 6 then measures the data and keeps it **on the GPU** when it fits next to the model. Batches are then cut on the GPU and no workers are needed: on the A6000 this applies to all data, on the T4 to val and test only.

## 4. Input features (sections 4, 5, 5b)

| Channels | Count | What | Why |
|---|---|---|---|
| monthly | 3 per month | VV, VH, VV-VH in dB, standardised with train statistics | VV: buildings, rough surfaces; VH: vegetation volume; VV-VH: their ratio |
| month window | 1 or 3 months | target month plus the 2 before it, wrapping around the year | temporal context is the biggest lever for SAR (survey) |
| PCA | 6 | projection of the pixel's 36 yearly values (12 months x 3) | compact yearly summary, 92 % of the variance |
| seasonal statistics | 5 | VV min, VV max, VH std, VH spread (2nd highest - 2nd lowest), share of months with VV < -18 dB | order-independent yearly summary, as in ESA WorldCover (percentiles) |
| terrain (ablation only) | 3 | elevation, slope (Copernicus GLO-30), log HAND (MERIT Hydro) | not Sentinel-1, so never in the delivered model |

All static channels (PCA, statistics, terrain) are computed once per tile and stored as float16.

**PCA vs statistics (from the survey).** PCA works in the dB domain, where speckle is roughly additive, so the method is valid. Its loadings are tied to calendar months, however. A crop cycle in the southern hemisphere is about 6 months out of phase with one in the north, so the same land cover gets different PCA values in the two hemispheres. This fits what we measured:

- the train-val gap was 0.127 with PCA and 0.054 without (runs 1-2);
- the test gains for wetland and cropland were unstable.

The statistics do not depend on month order and follow ESA WorldCover. The new `three_month_stats` variant tests whether PCA adds anything once the statistics exist.

## 5. Variants (the ablation chain)

| Variant | Channels | Purpose |
|---|---|---|
| `single_month` | 3 | one month only. Done: Test_5_Full, val mIoU 0.548 |
| `three_month` | 9 | temporal context (3 months) |
| `three_month_stats` | 14 | + order-independent yearly statistics (new) |
| `three_month_pca` | 15 | + PCA yearly summary |
| `three_month_pca_stats` | 20 | + both |
| `three_month_pca_stats_terrain` | 23 | + terrain (ablation, not deliverable) |

The final run trains the five three-month variants. **`BEST` is chosen by val mIoU among the Sentinel-1-only variants.** Test is only reported, never used to choose.

## 6. Augmentation and class balance (section 6)

**Augmentation:**
- random crop position (±128 px around the indexed patch);
- flips and 90° rotations;
- a random brightness shift of ±1 dB, drawn separately for VV and VH, plus 0.5 dB noise;
- **month dropout** (new): each input month is blanked with a 15 % chance.

All of it runs on the GPU for the whole batch.

**Class balance**, as the lecture notes ask ("address data imbalance"):
- class weights `w = sqrt(median_freq / freq)` in the loss (Built-up 0.57 ... Wetland 1.35);
- a patch sampler that draws patches with rare classes more often.

## 7. Model (section 7)

- **Architecture:** a U-Net with an ImageNet-pretrained ResNet-34 encoder (24.5 M parameters).
- **First layer:** its RGB filters are averaged and copied to every input channel, scaled by `3 / channels`.
- **Decoder:** 5 up-sampling steps with skip connections, then Dropout2d (0.2) and a 1x1 convolution giving 8 class scores per pixel.

Every piece is in the lectures: U-Net (01), ResNet, skip connections, transfer learning (07), batchnorm and dropout (05).

## 8. Loss, optimiser and schedule (sections 8-9)

| Part | Setting |
|---|---|
| loss | weighted sparse cross-entropy (label smoothing 0.1) + 0.5 x Lovász-softmax (on 25 % of pixels) |
| logged loss | plain weighted cross-entropy, for train and val alike |
| optimiser | AdamW, lr 6e-4, weight decay 0.05 on conv weights (not on biases or BatchNorm) |
| schedule | 1 warmup epoch, then cosine to 0 at `MAX_EPOCHS` (40 final, 15 DEV) |
| epoch | 2500 patches from the class-aware sampler (39 steps at batch 64) |
| EMA | weight average with decay 0.99; val, the checkpoint and test use the EMA model |
| validation | all 12 months of the 22 val sites, every 2nd epoch and at the last epoch |
| early stopping | patience 10 epochs on EMA val mIoU; the best checkpoint is saved at every improvement |

Lovász-softmax optimises a smooth version of IoU, the reported metric. EMA averages out the noisy updates caused by noisy labels.

## 9. Overfitting: diagnosis and measures (new for Test_6)

What Test_5_Full showed (60 epochs, before the changes):

| Variant | best val mIoU (epoch) | train mIoU at best | gap | val loss minimum |
|---|---|---|---|---|
| `single_month` | 0.548 (45) | 0.760 | 0.21 | epoch 45 |
| `three_month` | 0.599 (50) | 0.825 | 0.23 | 0.598 at epoch 27, then rising to 0.648 |

- Val mIoU flattened after about epoch 30 while train kept improving. This is the lecture-05 picture of overfitting.
- With only 134 training sites, the model learns site-specific detail.

Measures, each aimed at one cause:

| Measure | Change | Reason |
|---|---|---|
| L2 regularisation | AdamW weight decay 1e-4 → 0.05 | 1e-4 had almost no effect; 0.05 is in the usual AdamW range |
| label smoothing | 0.1 in the training loss | the labels are imperfect (each source map is ~74-77 % correct), so the model should not become fully certain |
| dropout | 0.1 → 0.2 | less co-adaptation in the decoder |
| month dropout, VV/VH shifted separately | new augmentation | more input variety, robustness to missing months |
| shorter schedule | 60 → 40 epochs, patience 15 → 10 | the learning rate reaches its low values while val is still at its best |
| feature choice | `three_month_stats` added | PCA showed a larger train-val gap, and the statistics are order-independent |

What to check in Test_6:
- a train-val gap clearly below 0.2;
- val loss no longer rising in the second half;
- val mIoU at least as high as in Test_5 (0.599 for `three_month`).

**Outcome (Test_6):**
- Val loss now reaches its minimum at epoch 34-36 of 40 and stays flat. Only `three_month_pca_stats` still rises after epoch 26.
- The gap shrank only to 0.18-0.21.
- `three_month` val mIoU was 0.570, slightly below Test_5. That may be mild underfitting, or noise.
- See `reports/Test run 6 results.md`.

## 10. Evaluation and outputs (sections 10-12)

**Test (20 unseen sites, all 12 months):**
- OA, mIoU, IoU and F1 per class;
- a confusion matrix;
- scores per region and per site type;
- mIoU per month, which shows the seasonal effect;
- example predictions.

**Reading the numbers:**
- They count only pixels where both label maps agree (about 18 % of pixels are ignored), so they are not comparable with published S1 benchmarks.
- Runs 1 and 2 used the same code and data, yet their test results differed by up to 0.06 mIoU. Smaller differences are noise.
- Wetland and snow test scores rest on 1-2 sites each.

**Files in `Weights/<RUN_NAME>/`:**
- `<variant>_best.pt` and `<variant>_history.csv`;
- `test_metrics.json`;
- `model_card.json`, with everything the prototype needs to rebuild the inputs: channels, month window, normalisation, PCA weights, statistics and terrain definitions, dB decoding, classes and colours;
- the plots;
- a zip of the whole folder.

## 11. Run history (short)

| Run | Setup | Key result |
|---|---|---|
| baseline | old dataset, 3 months, 60 epochs | test OA 0.843, mIoU 0.559; gap 0.31 |
| runs 1-2 | old notebook, DEV | identical code differs by up to 0.06 test mIoU, so that is the noise level; PCA val +0.10 |
| Test_4 | improved notebook, DEV, A6000 | val: pca_stats 0.679 > terrain 0.654 > pca 0.613; test order reversed (noise) |
| Test_5_Full | 60 epochs, 5 variants | Datalab closed during `three_month_pca`; finished: single_month 0.548, three_month 0.599 val mIoU; gap 0.21-0.23 |
| Test_6 | overfitting measures + `three_month_stats`, 40 epochs, all data on the GPU | 27 min total (~8 s per epoch). Val: terrain 0.694 > pca_stats 0.671 > pca 0.666 > stats 0.657 > three_month 0.570. Test mIoU: **stats 0.742** > terrain 0.701 > pca 0.667 > pca_stats 0.617 > three_month 0.585. Val loss no longer rises late; gap 0.18-0.21. Details: `reports/Test run 6 results.md` |
