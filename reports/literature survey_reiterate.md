# Literature survey (reiterated): is our strategy right for the problem?

Date: 2026-10-26. Scope: check the training strategy against the course lectures (`Documentation/01-07`) and the exam brief, and answer one question: is PCA the right tool for Sentinel-1 radar time series?

## Short answer

**We are heading in the right direction.** The core design (U-Net with a pretrained ResNet-34, weighted sparse cross-entropy, augmentation, early stopping, split by site) matches both the lectures and the remote-sensing literature. Two things need correcting, and one needs a cheap extra experiment:

1. **Terrain must not be the delivered model.** The brief says the model "uses Sentinel-1 data as its only input". `three_month_pca_stats_terrain` uses a DEM and HAND, so it can only be reported as an ablation. The notebook currently picks `BEST` from all variants, including terrain.
2. **PCA is acceptable but not the best-justified temporal feature for SAR.** It is not wrong, but it has a specific weakness for our global dataset (see below). The seasonal statistics are the better-founded choice, because they are what operational SAR land-cover products use.
3. **Missing ablation:** we never trained `three_month_stats` (stats without PCA). Without it we cannot say whether PCA adds anything once the statistics exist.

## 1. Strategy vs the lectures

| Technique in the notebook | Lecture | Status |
|---|---|---|
| Sparse categorical cross-entropy | 02-04, `lecture_notes.txt` | used (weighted) |
| Class imbalance handling | `lecture_notes.txt` ("address data imbalance") | class weights + class-aware patch sampler |
| Adam optimiser | 05 (Adam, "a good place to start") | AdamW = Adam + decoupled L2 |
| L2 regularisation | 05 | weight decay 1e-4 |
| Dropout | 05, 07 (AlexNet) | Dropout2d 0.1 before the head |
| Batch normalisation | 05 | in ResNet-34 and decoder |
| Early stopping on validation | 05, `lecture_notes.txt` | patience 15 on val mIoU |
| Train / val / test split | 05 | by site (stricter than random: no spatial leakage) |
| Data augmentation | 05, 07 | flips, 90° rotations, crop jitter, dB shift + noise |
| Learning-rate decay | 04 | warmup + cosine |
| CNN, ReLU, pooling, 1x1 conv head | 06, 07 | ResNet-34 encoder, 1x1 conv classifier |
| ResNet, skip connections, transfer learning | 07 | ImageNet ResNet-34, U-Net skips |
| U-Net for semantic segmentation | 01 (named with FCN, SegNet, DeepLab) | used |

Not in the lectures, so the report must justify them in one sentence each:

- **Lovász-softmax:** a smooth surrogate of IoU, the metric we report (Berman et al., CVPR 2018).
- **EMA weights:** averages out the noisy updates from noisy labels. It answers lecture 05's point that "error measurements are rarely smooth" during early stopping.
- **PCA and seasonal statistics:** feature engineering from the S1 time series (section 2).

Verdict: nothing in the training algorithm is wrong for the problem. It is the course toolbox applied correctly, plus three justified additions.

## 2. Is PCA right for radar data?

### What we do

For each pixel we take 36 values: 12 months x (VV, VH, VV-VH) in dB, standardised. We project them on the first 6 principal components, fitted on train pixels (92 % of the variance). These 6 maps are extra input channels.

### What speaks for it

- **The dB (log) domain is the right place for linear methods on SAR.** Speckle is multiplicative in linear power and becomes roughly additive in dB, so a linear projection like PCA is reasonable there. Our PCA runs on dB values, which is correct.
- **PCA is used on Sentinel-1 time series in the literature.** Functional PCA extracted seasonal trends from S1 for wetland mapping, and the first three scores closely matched an existing land-cover map ([Statistics Canada](https://www.statcan.gc.ca/en/data-science/network/extracting-temporal-trends)). PCA is also a common feature-reduction step for S1/S2 classifiers ([arXiv 2312.10798](https://arxiv.org/pdf/2312.10798)).
- **Our own runs:** in runs 1 and 2, PCA gave **+0.10 val mIoU** and **+0.13 Trees, +0.10 Shrub test IoU** (`research_notes/run_log.md`).

### What speaks against it

1. **PCA keeps variance, not class information.** The first component is the yearly average brightness, which is the strongest signal but already visible in the 3 monthly channels. Subtle but useful seasonal patterns (a short flood, a harvest) have little variance and may land in the discarded components.
2. **PCA is tied to calendar months. This is the main problem for a global dataset.** Each loading belongs to a fixed month ("VH in July"). A crop cycle in Argentina or Australia runs about 6 months out of phase with one in Germany, so the same land cover gets different PCA values in the two hemispheres. The model can only learn this per site, which fits what we measured:
   - the train-val mIoU gap was 0.127 with PCA vs 0.054 without (runs 1-2);
   - test gains were unstable (Wetland −0.12 to −0.46, Cropland −0.05 to −0.20).
3. **It duplicates what the network can learn itself.** The first convolution is also a learned linear combination of its input channels. Fed with all 12 months, it could learn a projection chosen for the classes instead of for variance. PCA is a fixed, unsupervised version of that layer.
4. **Missing months are filled with the mean (0 after standardisation) before the projection.** PCA mixes this fill value into all 6 components, while a statistic like VV min simply ignores the gap.

### What operational SAR land-cover products use instead

- **ESA WorldCover**, one of our two label sources, summarises the S1 time series as **temporal percentiles (p10, p50, p90) and the interquartile range**, after a multitemporal speckle filter. It feeds these into the classifier together with a DEM ([CEOS WorldCover summary](https://ceos.org/gst/worldcover.html)).
- **Seasonality metrics** separated the main land-cover classes well in S1 time series (Dubois et al., ISPRS Annals 2020, [DLR elib](https://elib.dlr.de/140347)).
- **Harmonic (sine/cosine) fits** of S1 backscatter are used operationally by TU Wien and the GFM flood service. They suppress speckle and keep the seasonal shape ([TU Wien dataset](https://researchdata.tuwien.at/records/x8p2j-1tj74)). Harmonic coefficients matched or beat large monthly-composite feature sets in optical studies ([USDA](https://research.fs.usda.gov/treesearch/55701)).
- **Learned temporal encoders** (U-TAE, L-TAE) give the largest documented gains. See `research_notes/Sentinel 1 LULC segmentation improvements/architectures_and_pretrained_encoders.md`. They are out of scope before the deadline (9 Nov 2026).

Percentiles, min/max, std and the share of water months **do not depend on month order**. They are the same for a field in Germany and in Argentina, so they avoid problem 2. Our `stats` channels (VV min, VV max, VH std, VH spread from the 2nd-highest and 2nd-lowest months, water months) follow the WorldCover idea. **The statistics are the better-founded feature. PCA is a defensible extra, not a mistake.**

Test run 4 is consistent with this but not conclusive. `three_month_pca_stats` had the best val mIoU (0.679 vs 0.613 for PCA alone), but its test results swing with one or two sites per class.

## 3. Recommendations

| # | Action | Effort | Why |
|---|---|---|---|
| 1 | Choose `BEST` only from S1-only variants; report terrain as an ablation | 1 line | the brief requires S1 as the only input |
| 2 | Add the variant `three_month_stats` (stats, no PCA) | 1 line in `ALL_VARIANTS` / `RUN` | the decisive ablation: does PCA add anything once the stats exist? |
| 3 | Keep PCA in the report as a tested idea, and explain the calendar-phase weakness | text | shows critical thinking; the brief asks for a "critical discussion" |
| 4 | Optional: replace "VH spread" with true p10/p50/p90 + IQR of VV and VH | small | matches WorldCover exactly; easy to cite |
| 5 | Not now: U-TAE or the 12-month input | large | best upside, but too late for the deadline |

Apply 1 and 2 only after Test_5_full is pushed and pulled, to avoid merge conflicts in `Prototype.ipynb`.

## 4. What to write in the exam report

- **Design choice:** temporal summary features from S1 only, in the dB domain, after the WorldCover approach (percentile-style statistics).
- **PCA:** tested as a compact yearly summary. It helped classes with a stable yearly signature (trees, shrub). It is tied to calendar months, which is a weakness for sites in both hemispheres; this is why order-independent statistics were added.
- **Terrain:** a separate ablation showing what non-S1 context would add. It is not part of the delivered S1-only model.
- **Noise:** single-run test differences below about 0.05 mIoU are noise (runs 1 and 2 used the same code and data).

## Sources

- Statistics Canada, functional PCA on Sentinel-1 seasonal trends: https://www.statcan.gc.ca/en/data-science/network/extracting-temporal-trends
- S1/S2 Random Forest ensembles with PCA features: https://arxiv.org/pdf/2312.10798
- ESA WorldCover algorithm summary (S1 percentiles, IQR, speckle filter, DEM): https://ceos.org/gst/worldcover.html
- Dubois et al. 2020, Sentinel-1 seasonality for land-cover classes: https://elib.dlr.de/140347
- TU Wien Sentinel-1 harmonic parameters: https://researchdata.tuwien.at/records/x8p2j-1tj74
- Harmonic regression vs monthly composites (Landsat): https://research.fs.usda.gov/treesearch/55701
- Our runs: `research_notes/run_log.md`; earlier survey: `reports/Sentinel 1 LULC segmentation improvements.md`
