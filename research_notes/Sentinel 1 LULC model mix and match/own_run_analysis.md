# Own run analysis: where the Sentinel-1 LULC model fails and what the evidence says about model changes

Date: 2026-11-03. Scope: local project data only (no web). Nothing was trained or executed except read-only numpy/pandas analysis of run files.

Sources used (all local, paths relative to `E:\Github Projects\Deep_Learning_Exam\`):
- [PR-Berlin] `prototype/runs/20261102_101102_52.5200_13.4050_2021-01/` (arrays.npz, run.json, CSVs)
- [PR-Thane1] `prototype/runs/20261102_101149_19.2183_72.9781_2021-01/`
- [PR-Thane2] `prototype/runs/20261102_101452_19.2145_72.9841_2021-01/`
- [I2-Full] `Weights/Iter2_Test_1_Full.zip` (test_metrics.json, model_card.json, timing.json, variants/month_year_history.csv, confusion_matrix.png)
- [I1-T2/T4] `Weights/Weights.zip` and `Weights/Weights/Test_2`, `Test_4` (history CSVs, Test_4 test_metrics.json)
- [BF] `reports/2026-10-31 Prototype test 1 Black Forest.md`, sections 10-11
- [PRX] `reports/2026-11-02 Prototype runs and model experiments.md`
- [T6] `reports/Test run 6 results.md`
- [DEV] `reports/DEV runs 2026-10-25.md`
- [RL] `research_notes/run_log.md`
- [SC] `SESSION_CONTEXT.md`
- [CALC] my own computation on the files above; scripts in the session scratchpad (`p.py`, `q.py`, `m.py`). Method notes are given where a number is computed rather than read.

Method notes for [CALC]:
- "Annual mode" = most frequent class over the 12 monthly maps in `arrays.npz['lulc']` (these maps are already 3-month softmax-smoothed by the prototype). Compared against `reference` (Esri 2021 mapped to our 7 classes). All 512 x 512 pixels valid in all three runs.
- Radar: `radar_codes[2:14]` = Jan-Dec 2021, channel 0 = VV, 1 = VH, dB = code*0.2-35, code 0 = nodata. Per-pixel annual features: VV/VH mean, VV-VH mean, VH std, VV/VH min/max, share of months with VV < -18 dB ("wetmon").
- Separability: (1) rank AUC of each feature between misclassified pixels and correctly classified pixels of the true class and of the predicted class (AUC 0.5 = indistinguishable, near 0 or 1 = separable); (2) nearest-centroid test in standardised 9-feature space, centroids from correctly classified pixels of each class. Both are crude (univariate / linear, centroids biased toward what the model already gets right) and should be read as indications, not proofs.
- Boundary distance: city-block distance (in 10 m pixels) to the nearest pixel where the Esri reference class changes, capped at 12.
- Important caveat: the model is trained on ESA WorldCover (+ Dynamic World agreement weighting), the prototype check uses Esri 2021. Disagreement with Esri therefore mixes model error with WorldCover-vs-Esri definition differences. All three prototype runs are urban (Berlin, Thane); none is a rural test [PRX].

---

## 1. Where does the model fail in the logged prototype runs?

### Takeaway
Against Esri, the errors are mostly at class boundaries (Thane: 47-48 % of all errors lie within 20 m of an Esri boundary, which is 11-12 % of the area; interior error rate only 2-3 %), and most large confusions involve pixels whose annual radar statistics look like the class the model predicted, not like the reference class. The main confusions are Esri Grassland/rangeland → Man-made/Trees/Crop, Esri Crop → Trees/Man-made, Esri Water → Trees/Man-made at creek edges, and Esri built area → Trees in Berlin parks/streets. Only Crop ↔ Grassland shows a sign that the radar holds information the model does not fully use. Monthly flicker is very low (0.05-0.49 % of pixels per month step) and concentrated in Grassland and Crop pixels and at boundaries.

### Cited Findings

**Overall agreement and baseline**
- Annual-mode agreement with Esri: Berlin 96.6 %, Thane1 89.7 %, Thane2 88.2 %; majority-class baseline 97.1 %, 75.7 %, 75.5 % — [CALC] on [PR-Berlin], [PR-Thane1], [PR-Thane2]; same values as run.json `reference_agreement_pct` and [PRX] section 1.
- Monthly agreement is flat over the year: Berlin 0.966-0.967, Thane1 0.895-0.898, Thane2 0.880-0.883 — [CALC].
- Class shares change at most 0.06 pp (Berlin) and 0.58 pp (Thane1) between the highest and lowest month — [CALC].
- Scene counts: Berlin 18-24 per month, Thane 2-4 per month (Sentinel-1A only, one orbit) — run.json, [PRX].

**Confusion matrices (rows = Esri reference, columns = model annual mode, pixel counts)** — [CALC]

Berlin (classes present: Man, Trees, Grass, Water):

| Esri \ model | Man | Trees | Grass | Water | Recall | Precision | IoU |
|---|---|---|---|---|---|---|---|
| Man-made | 246732 | 6655 | 939 | 168 | 0.970 | 0.996 | 0.966 |
| Trees | 32 | 5467 | 7 | 0 | 0.993 | 0.448 | 0.446 |
| Grassland | 125 | 89 | 524 | 0 | 0.710 | 0.356 | 0.311 |
| Water | 806 | 0 | 0 | 600 | 0.427 | 0.781 | 0.381 |

Thane1 (19.2183, 72.9781):

| Esri \ model | Man | Crop | Trees | Grass | Water | Recall | Precision | IoU |
|---|---|---|---|---|---|---|---|---|
| Man-made | 194351 | 923 | 2524 | 268 | 368 | 0.979 | 0.940 | 0.922 |
| Crop fields | 1533 | 2449 | 2088 | 798 | 4 | 0.356 | 0.402 | 0.233 |
| Trees | 6132 | 325 | 28303 | 182 | 76 | 0.808 | 0.756 | 0.641 |
| Grassland | 3629 | 2396 | 3500 | 1748 | 6 | 0.155 | 0.575 | 0.139 |
| Water | 1145 | 0 | 1019 | 44 | 8333 | 0.791 | 0.948 | 0.758 |

Thane2 (19.2145, 72.9841):

| Esri \ model | Man | Crop | Trees | Grass | Water | Recall | Precision | IoU |
|---|---|---|---|---|---|---|---|---|
| Man-made | 193376 | 1091 | 2810 | 301 | 316 | 0.977 | 0.947 | 0.927 |
| Crop fields | 1720 | 5467 | 4441 | 1166 | 0 | 0.427 | 0.526 | 0.309 |
| Trees | 4161 | 228 | 15959 | 59 | 18 | 0.781 | 0.546 | 0.474 |
| Grassland | 3374 | 3603 | 3867 | 1167 | 0 | 0.097 | 0.421 | 0.086 |
| Water | 1467 | 0 | 2143 | 78 | 15332 | 0.806 | 0.979 | 0.792 |

- Largest error cells: Berlin Esri Man-made → Trees (6655 px, 75 % of all Berlin errors); Thane Esri Grassland is spread almost evenly over Man-made / Crop / Trees (each 3374-3867 px) and only 10-16 % is found; Esri Crop → Trees (2088-4441 px) is larger than Crop → Grassland (798-1166 px); Esri Trees → Man-made 4161-6132 px — [CALC].

**Boundaries vs interiors** — [CALC]

| Run | Share of area within 2 px (20 m) of an Esri boundary | Error rate there | Share of all errors there | Error rate > 11 px from a boundary | Share of errors > 11 px | OA if the 2-px band is excluded |
|---|---|---|---|---|---|---|
| Berlin | 2.6 % | 39.7 % | 30.4 % | 1.5 % | 39.9 % | 97.6 % |
| Thane1 | 10.7 % | 46.1 % | 48.0 % | 2.2 % | 14.4 % | 94.0 % |
| Thane2 | 11.6 % | 47.2 % | 46.7 % | 2.9 % | 16.0 % | 92.9 % |

- Error rate falls monotonically with distance in all runs (Thane1: 46 % at 0-2 px, 26 % at 3-5 px, 13 % at 6-11 px, 2 % beyond) — [CALC].
- Per class, share of the class's errors within 2 px of a boundary: Water 78-83 %, Trees 55-61 %, Man-made 23-50 %, Crop 40-51 %, Grassland 34 % (Thane; Berlin 99 %). Grassland and Crop have the largest share of errors deep inside Esri patches (> 10 px: Grassland 26 %, Crop 15-25 %) — [CALC].
- Berlin is the exception: 40 % of its errors lie > 11 px inside built area, almost all Esri Man-made → Trees (park and street-tree canopy that Esri counts as built area) — [CALC]; [PRX] notes the parks are matched (99 % of Esri tree pixels found).

**Are the misclassified pixels radar-separable? (annual radar features of correct vs wrong pixels)** — [CALC], Thane values unless stated, Thane1 / Thane2 where both are given

| Case | n (Thane1/2) | VV mean | VH mean | VH std | AUC (VH mean) wrong vs correct-true-class | AUC (VH mean) wrong vs correct-predicted-class | Nearest correct-class centroid of the wrong pixels |
|---|---|---|---|---|---|---|---|
| Crop, correct | 2449 / 5467 | -9.6 / -9.3 | -17.3 / -16.6 | 3.6 / 3.3 | - | - | - |
| Grassland, correct | 1748 / 1167 | -7.9 / -8.7 | -14.7 / -15.2 | 2.2 / 2.2 | - | - | - |
| Trees, correct | 28303 / 15959 | -6.2 / -6.6 | -12.6 / -12.7 | 1.3 / 1.3 | - | - | - |
| Water, correct | 8333 / 15332 | -16.5 / -17.4 | -21.7 / -22.2 | 1.6 / 1.6 | - | - | - |
| Man-made, correct | ~194k | -3.2 | -11.6 | 1.5 | - | - | - |
| Crop → Trees | 2088 / 4441 | -7.3 / -7.4 | -13.1 / -13.2 | 1.4 / 1.4 | 0.98 / 0.96 | 0.38 / 0.35 | Trees 85 % / 82 % |
| Crop → Man-made | 1533 / 1720 | -6.7 | -13.2 | 1.6 | 0.93 / 0.90 | 0.33 / 0.34 | Trees 52-56 %, Grass 24-28 %, Man 13-14 %, Crop 6-7 % |
| Crop → Grassland | 798 / 1166 | -10.1 / -9.6 | -16.8 / -16.0 | 2.9 / 2.7 | 0.58 / 0.62 | 0.22 / 0.39 | **Crop 57 % / 48 %**, Grass 36-38 % |
| Grassland → Crop | 2396 / 3603 | -7.8 / -7.6 | -15.4 / -15.0 | 2.6 / 2.4 | 0.38 / 0.53 | 0.84 / 0.78 | Grass 63 % / 54 %, Crop 27-28 % |
| Grassland → Trees | 3500 / 3867 | -7.0 / -7.1 | -13.4 | 1.6 / 1.5 | 0.76 / 0.82 | 0.31 / 0.32 | Trees 62 % / 69 % |
| Grassland → Man-made | 3629 / 3374 | -5.7 | -13.4 | 1.6 | 0.73 / 0.80 | 0.29 | Trees 44-45 %, Grass 28-32 %, Man 22-25 % |
| Water → Trees | 1019 / 2143 | -7.7 / -7.4 | -13.7 / -13.4 | 1.4 | 0.97 | 0.29 / 0.36 | Trees 72-74 % |
| Water → Man-made | 1145 / 1467 | -7.7 | -15.0 / -15.5 | 1.6 | 0.92 | 0.25 / 0.20 | Grass 38-43 %, Trees 34-37 % |
| Berlin Man-made → Trees | 6655 | -7.6 | -13.7 | 0.65 | 0.11 (vs correct Man) | 0.41 (vs correct Trees) | Trees 93 % |
| Berlin Water → Man-made | 806 | -8.7 | -16.0 | 0.61 | 0.79 (vs correct Water) | 0.09 (vs correct Man) | Grass 44 %, Trees 25 % |

- Esri-Water pixels the model calls Trees or Man-made have VV about -7.5 dB, i.e. 9-10 dB brighter than correct water (-16.5 to -17.4 dB) and never dark enough to count as a "water month" (wetmon 0.00-0.01 vs 0.47-0.56 for correct water); 78-83 % of them lie within 20 m of an Esri boundary — [CALC]. In the radar these pixels are vegetation or structures, not open water.
- Esri-Crop pixels the model calls Trees have VH std 1.4 dB (like correct trees, 1.3) instead of 3.3-3.6 dB (correct crop); AUC on VH std vs correct crop 0.01-0.04 (almost perfectly separated from real crop signature) — [CALC]. They carry no crop-like seasonal signal.
- Crop → Grassland pixels are the opposite case: their VH std (2.7-2.9) is between crop (3.3-3.6) and grassland (2.2), AUC on VH std vs correct grassland 0.67-0.74, and the nearest centroid is Crop for 48-57 % of them — [CALC]. Part of this confusion is radar-separable with the annual statistics.
- Grassland → Crop pixels look radar-like Grassland (nearest centroid Grass 54-63 %) — [CALC]; combined with the previous point this means the Crop/Grass boundary is drawn in the wrong place in both directions for a subset of pixels, i.e. the decision boundary, not the signal, is the problem for that subset.

**Flicker (raw prototype output = smoothed monthly maps, month-to-month class change)** — [CALC]

| Run | Overall flicker % | Man-made | Crop | Trees | Grassland | Water | Boundary (≤1 px) vs interior (>5 px) |
|---|---|---|---|---|---|---|---|
| Berlin | 0.05 | 0.03 | - | 0.42 | 0.91 | 0.67 | 0.40 vs 0.04 |
| Thane1 | 0.49 | 0.21 | 3.70 | 1.13 | 5.47 | 0.46 | 1.71 vs 0.27 |
| Thane2 | 0.48 | 0.22 | 2.39 | 1.27 | 5.41 | 0.24 | 1.74 vs 0.24 |

- In Thane, 28-29 % of pixels whose annual mode is Grassland and 13-20 % of Crop pixels change class at least once in the year, vs 1.1 % for Man-made and 1-2 % for Water — [CALC].
- Thane has about 2.5x larger month-to-month VH change than Berlin (mean |ΔVH| 1.56-1.58 dB vs 0.61 dB), consistent with 2-4 vs 18-24 scenes per month (more speckle in the monthly mean, plus monsoon) — [CALC]; scene counts from run.json. Flicker is about 10x Berlin's.
- 55-89 Thane pixels have only 1 valid month; their error rate is 84-95 % — [CALC]. Negligible area (< 0.04 %).

### Inferences
- The model is not failing seasonally any more: monthly agreement and class shares are flat across the year in all three runs, so the 1st-iteration failure mode (winter water/wetland/crop) does not reappear here.
- Roughly half of the Thane disagreement with Esri is a 20 m boundary band. With Sentinel-1 GRD resolving about 20 x 22 m [PRX 4.1], a 1-2 pixel boundary band is within the sensor's resolution limit and within Esri/WorldCover registration and mixed-pixel effects. This is a resolution/label-geometry limit, not something a bigger encoder can remove; a boundary-tolerant loss/score or soft (probability) class shares [PRX 3.3] fit this evidence.
- Most of the big confusion blocks (Water→Trees, Crop→Trees, Grassland→Trees/Man-made, Berlin Man-made→Trees) are pixels whose annual radar statistics match the predicted class. For these, the model is following the radar; a different model cannot fix them unless it learns to override the radar with context. These are label-definition or sensor-physics cases (mangrove fringe vs creek water, Esri rangeland / vacant land, tree canopy over streets counted as built, tree-lined or plantation "crop").
- Crop ↔ Grassland is the one confusion where wrongly classified pixels are partly radar-separable by temporal variability. This is the strongest local evidence that a better temporal model (using the full monthly sequence rather than one month + 9 order-free statistics) could gain something.
- Flicker is concentrated in exactly the classes and places with weak separability (Grassland, Crop, boundaries) and in low-scene-count areas; it is a symptom of low confidence, not a separate failure.

### Gaps
- No rural prototype run exists yet (Black Forest clearing and rural mosaics still to run [PRX]); all prototype evidence above is urban/peri-urban. The `terrain` key described in the task is not present in any `arrays.npz` (keys: radar_codes, lulc, valid, reference), so no slope/shadow analysis was possible.
- Raw (unsmoothed) monthly maps and softmax probabilities are not saved in `arrays.npz`, so per-pixel confidence and raw flicker cannot be computed for the prototype runs.
- Esri ≠ training labels (WorldCover). I could not separate "model wrong vs WorldCover" from "WorldCover vs Esri definition" for these windows because WorldCover is not stored in the run files.
- No ground truth for the creek/mangrove edge; whether Esri-Water→Trees pixels are mangrove, mudflat or misregistration cannot be decided from these files.

---

## 2. What do the training/test runs show so far (variants, gains, noise, per-class/region weaknesses, learning curves)?

### Takeaway
The biggest measured gains came from data/input changes (yearly temporal context +0.09-0.12 val mIoU; longer schedule +0.04-0.055 rural test mIoU; rural sites + terrain flattening + labels-on-all-pixels), not from architecture, which has never been varied (U-Net ResNet-34 throughout). Identical runs differ by 0.015-0.02 mIoU (iteration 2) and 0.02-0.06 test mIoU (iteration 1), as large as most variant differences. The current model (`Iter2_Test_1_Full`, month_year) overfits mildly after epoch 22 but only reaches train mIoU 0.85-0.88, which points to a label-noise ceiling rather than insufficient capacity. Weakest: Water (rural IoU 0.49, precision 0.59), Snow & ice (0.45), Crop fields (0.66, 21 % → Grassland); weakest regions Oceania and Africa; weakest site types shrub_grass, wetland, snow_ice.

### Cited Findings

**Every run/variant compared so far**

1st iteration (8 classes, city-heavy test 54 % built-up, consensus labels; test mIoU not comparable with iteration 2):

| Run | Variant | Val mIoU (best epoch) | Test OA | Test mIoU | Train-val gap | Source |
|---|---|---|---|---|---|---|
| Baseline (pre 10-24) | three_month | - | 0.843 | 0.559 | 0.31 | [RL] |
| Baseline | single_month | - | 0.810 | 0.507 | - | [SC] |
| Run 1 DEV | three_month | 0.541 | 0.817 | 0.559 | - | [DEV], [RL] |
| Run 1 DEV | three_month_pca | 0.644 | 0.835 | 0.546 | - | [DEV] |
| Run 2 DEV (identical) | three_month | 0.530 (14) | 0.823 | 0.538 | 0.054 | [DEV], [I1-T2] |
| Run 2 DEV (identical) | three_month_pca | 0.634 (14) | 0.813 | 0.485 | 0.127 | [DEV], [I1-T2] |
| Run 4 DEV | three_month_pca | 0.613 (15) | 0.878 | 0.664 | 0.224 | [RL], [I1-T2/T4] |
| Run 4 DEV | three_month_pca_stats | 0.679 (15) | 0.839 | 0.566 | 0.166 | [RL] |
| Run 4 DEV | ..._stats_terrain | 0.654 (15) | 0.832 | 0.558 | 0.216 | [RL] |
| Run 5 (incomplete) | single_month | 0.548 (45) | - | - | 0.21 | [RL] |
| Run 5 (incomplete) | three_month | 0.599 (50) | - | - | 0.23 | [RL] |
| Test_6 FINAL | three_month | 0.570 (34) | 0.843 | 0.585 | 0.184 | [T6] |
| Test_6 FINAL | three_month_stats | 0.657 (36) | 0.914 | 0.742 | 0.208 | [T6] |
| Test_6 FINAL | three_month_pca | 0.666 (34) | 0.874 | 0.667 | 0.207 | [T6] |
| Test_6 FINAL | three_month_pca_stats (selected by val) | 0.671 (26) | 0.865 | 0.617 | 0.212 | [T6] |
| Test_6 FINAL | ..._pca_stats_terrain (ablation, not S1-only) | 0.694 (34) | 0.890 | 0.701 | 0.208 | [T6] |

2nd iteration (7 classes, rural objective, gamma0 terrain flattening, WorldCover on all pixels with disagreement weight 0.5; main score = rural test mIoU on agreement pixels):

| Run | Variant | Inputs | Val mIoU (epoch) | Test mIoU all | Rural mIoU | Rural smoothed | Rural all-pixels | Flicker raw → smoothed % | Source |
|---|---|---|---|---|---|---|---|---|---|
| Iter2_Test_1 DEV 15 ep | year_stats | 9 | 0.714 (15) | 0.729 | 0.646 | 0.646 | 0.565 | 0 | [BF 10] |
| Iter2_Test_1 DEV | month_year | 12 | 0.741 (12) | 0.714 | 0.639 | 0.641 | 0.534 | 3.0 → 1.2 | [BF 10] |
| Iter2_Test_1 DEV | three_month_year | 18 | 0.714 (14) | 0.722 | 0.639 | 0.640 | 0.559 | 2.5 → 1.5 | [BF 10] |
| Repeat DEV (identical) | year_stats | 9 | 0.717 | - | 0.643 | - | - | 0 | [BF 11.1] |
| Repeat DEV | month_year | 12 | 0.723 | 0.699 | 0.624 | 0.625 | 0.528 | 2.7 → 1.1 | [BF 11.1] |
| Repeat DEV | three_month_year | 18 | 0.715 | - | 0.643 | - | - | 2.5 | [BF 11.1] |
| Iter2_Test_1_Full (40 ep, current model) | month_year | 12 | 0.732 (22) | 0.750 | 0.679 | 0.680 | 0.597 | 2.4 → 0.9 | [BF 11.2], [I2-Full] |

- 3rd-iteration experiments (encoders E0-E5, training T1-T2, labels L1-L4) are designed and notebooks exist, but no results are in the repo — [PRX 3-4]; `Notebooks/3rd_iteration_*.ipynb` untracked per git status.

**What each change gained**
- Yearly temporal context: every yearly-feature variant beat plain three_month on val by 0.09-0.12 — [T6 6]. PCA alone: val +0.10 in both DEV runs — [DEV].
- Month window: three_month beats single_month by 0.05 val mIoU — [RL Run 5].
- Seasonal statistics instead of PCA: Test_6 three_month_stats best on test by 0.075-0.125 over PCA variants (above noise), but val could not separate them (0.657 / 0.666 / 0.671) and selected the worst on test — [T6 5].
- New training setup (EMA, warmup+cosine, CE + 0.5 Lovász, GPU eval): three_month_pca test mIoU 0.485-0.546 → 0.664 (suggestive) — [RL Run 4].
- Regularisation (wd 0.05, label smoothing 0.1, dropout 0.2, month dropout 0.15): gap 0.21-0.23 → 0.18-0.21; three_month val 0.599 → 0.570 (possible mild underfit or noise) — [T6 4].
- Terrain channels (DEM, slope, HAND): best val 0.694 and Water IoU 0.868 vs 0.56-0.66; excluded (S1-only brief) — [T6 6].
- 2nd iteration (rural sites, gamma0 flattening, labels on all pixels, 7 classes, whole-year statistics): train-val gap 0.18-0.21 → 0.12; VV-slope correlation 0.74-0.82 → -0.14-0.05; Black Forest winter failure gone — [BF 11.3].
- 40-epoch schedule vs 15-epoch DEV: rural test mIoU +0.040 to +0.055 (0.624/0.639 → 0.679); snow IoU 0.32 → 0.45, bare 0.83 → 0.89 — [BF 11.2].
- Temporal smoothing (mean softmax t-1, t, t+1): flicker 2.4 % → 0.9 % (rural 2.37 → 0.93 %), rural mIoU 0.679 → 0.680 — [I2-Full].

**Noise between identical runs**
- Iteration 1: test mIoU 0.021 (three_month) and 0.061 (three_month_pca); single-class IoU up to 0.24 (Water 0.679 → 0.443; Wetland 0.427 → 0.611) — [DEV].
- Iteration 2: val mIoU up to 0.018 (month_year 0.741 → 0.723), rural test 0.015 (0.639 → 0.624); year_stats and three_month_year moved ≤ 0.004 — [BF 11.1].
- Val within-run oscillation: ±7 points between epochs in an early DEV run — [SC].
- Rare-class test IoU rests on 1-2 sites (~16 patches) each — [DEV], [T6 5].

**Per-class weaknesses (current model, rural test sites, agreement pixels)** — [I2-Full], confusion_matrix.png

| Class | IoU | IoU all WorldCover pixels | Recall | Precision | Main confusion |
|---|---|---|---|---|---|
| Man-made | 0.75 | 0.50 | 0.89 | 0.82 | 4 % → Crop; 14 % of predicted Man-made is Grassland |
| Crop fields | 0.66 | 0.62 | 0.78 | 0.81 | 21 % → Grassland |
| Trees | 0.83 | 0.74 | 0.90 | 0.92 | 9 % → Grassland |
| Grassland | 0.68 | 0.60 | 0.83 | 0.79 | 8 % → Crop, 8 % → Trees |
| Bare ground | 0.89 | 0.85 | 0.92 | 0.97 | 5 % → Snow |
| Water | 0.49 | 0.27 | 0.74 | 0.59 | 13 % → Grassland; 23 % of predicted Water is Trees, 12 % Bare |
| Snow & ice | 0.45 | 0.59 | 0.64 | 0.60 | 24 % → Bare, 10 % → Grassland; 39 % of predicted Snow is Bare |

- All test sites (agreement): mIoU 0.750, OA 0.872; all WorldCover pixels mIoU 0.671 — [I2-Full].
- Monthly rural mIoU ranges 0.668-0.688 over the 12 months (no seasonal weak month) — [I2-Full].
- 1st iteration for comparison: snow best with the calendar-tied PCA variant (0.813 vs 0.529 without PCA); Water best only with HAND (0.868) — [T6].

**Per-region / site-type weaknesses**
- Iter2_Test_1_Full, all test patches: Asia 0.656, Europe 0.654, South America 0.584, North America 0.569 (highest flicker 4.7 → 1.8 %), Africa 0.492, Oceania 0.441 — [BF 11.2].
- By site type: mosaic 0.462, cropland 0.449, urban 0.485, trees 0.377, shrub_grass 0.233, snow_ice 0.222 (OA 0.545), wetland 0.223 (flicker 10.1 → 3.6 %) — [BF 11.2]. Per-type mIoU is depressed by few classes per group.
- Same pattern in iteration 1: N. America 0.378 (Anchorage, boreal, snow), shrub_grass 0.138, wetland OA 0.372 — [T6].
- Test pixel shares: Water 2.1 %, Snow 2.3 %; train Snow 2.8 %, Water 5.4 % — [BF 10.1].
- Label agreement (WorldCover vs Dynamic World) on training pixels: overall 73.6 %; Grassland 55.1 %, Bare 70.7 %, Crop 71.9 %, Trees 78.0 %, Snow 93.2 %, Man-made 95.0 %, Water 96.0 % — [BF 10.1].

**Learning curves**
- Iter2_Test_1_Full: val mIoU 0.557 (ep 4) → 0.704 (ep 8) → 0.732 (ep 22, best) → 0.711 (ep 32, early stop); val loss min 0.530 at ep 22, 0.568 at ep 32; train mIoU 0.854 at ep 22, 0.885 at ep 32 — [I2-Full] history CSV.
- Agreement-pixel val mIoU 0.732 vs all-pixel val mIoU 0.597 at the best epoch — [I2-Full] history CSV.
- DEV runs (15 epochs): best epoch 14-15 of 15 in iteration 1 (Test_2: 14/15; Test_4: 15/15 for all variants) → under-trained — [I1-T2/T4], [RL].
- Iter2 DEV month_year peaked at ep 12 (run 1) and ep 10 then fell to 0.709 (run 2) — [BF 11.1].
- Test_5 (weak regularisation, 60 ep): val loss min at ep 27 then rising 0.598 → 0.648 — [RL Run 5]. Test_6: minimum at ep 34-36 of 40 except pca_stats (26) — [T6 4].
- Training time: 4.7 min for 32 epochs on RTX A6000 — [I2-Full] timing.json.

### Inferences
- Learning-curve shape: fast rise to ~0.70 val by epoch 8, slow climb to 0.73, then mild overfit. Train mIoU stays at 0.85-0.88 even when overfitting, i.e. the network cannot fit its own training labels much better. With 26 % of training pixels being label disagreements (45 % for Grassland), this pattern is what label noise produces; it is not the signature of an under-capacity network (an under-capacity model would plateau with low train and val together and no val-loss rise).
- Because identical runs differ by up to ~0.02 val mIoU and val has only 21 sites, any architecture change expected to add 0.01-0.03 [PRX 2] is at or below the detection limit with 1-2 seeds. The val/test disagreement in Test_6 (val picked the worst test variant) shows the selection set itself is too small to resolve 0.01-level differences.
- The rural all-pixel scores fall most for Man-made (0.75 → 0.50) and Water (0.49 → 0.27). Those disagreement pixels sit mostly at edges (roads, river banks), consistent with the boundary concentration found in section 1.
- Snow's all-pixel IoU is higher than its agreement IoU (0.59 vs 0.45); a plausible reading is that the agreement subset keeps the hardest seasonal snow/rock pixels. Not verified.

### Gaps
- `Weights/Iter2_Test_1/` and `Iter2_Test_1_20261031_144820/` (DEV runs) and `Weights/Test_6/` are not present locally; their numbers come only from the reports. Only `Iter2_Test_1_Full.zip` and `Weights.zip` (Test_2, Test_4) are available as raw files.
- No per-site, per-class confusion for the rural test sites is stored as data (only the PNG), so I could not check whether Water false positives come from one or many sites.
- No multi-seed run of the final configuration exists; the ±0.02 noise estimate rests on one accidental repeat.
- No 3rd-iteration results exist yet.

---

## 3. Diagnosis by failure category and what "mix and match" the evidence supports or argues against

### Takeaway
(a) Label-definition issues and (b) radar-physics/resolution limits explain most of the remaining error; (c) model capacity/architecture is the least supported cause; (d) data coverage limits the rare classes and, above all, the ability to measure small improvements. The evidence supports label-handling changes, a stronger temporal model for Crop/Grass and Snow, boundary-tolerant training/evaluation, and seed ensembling; it argues against expecting much from bigger encoders, against further up-weighting rare classes, and against calendar-tied temporal features.

### Cited Findings

**(a) Label-definition issues**
- Grassland is the least agreed class in training labels (55 % WorldCover/Dynamic World agreement) — [BF 10.1]; it is the class with the lowest Esri recall in Thane (10-16 %) — [CALC] — and the largest rural confusion partner (Crop 21 % → Grassland) — [I2-Full].
- Esri "rangeland" covers open scrub, vacant urban land and mudflats that WorldCover often calls built-up, bare or trees — [PRX 1]. Wrongly classified Thane Esri-Grassland pixels are radar-brighter than correct Grassland (VV -5.7 vs -7.9 dB) and closest to the Trees centroid — [CALC].
- Berlin: 6655 Esri-built pixels predicted as Trees have tree-like radar (nearest centroid Trees 93 %, AUC vs correct Trees ≈ 0.41-0.45) — [CALC]; Esri counts street and courtyard canopy as built area (inference from [PRX 1] "parks matched" + radar).
- Wetland class removed in iteration 2 because its labels were 30-45 % correct — [BF 11.3]; WorldCover mangroves → Trees, herbaceous wetland → Grassland — [SC]. The Thane Esri-Water → Trees pixels at creek edges (radar VV -7.5 dB, never water-dark) fit the mangrove/mudflat definition boundary — [CALC].
- Rural mIoU drops from 0.679 (agreement pixels) to 0.597 (all WorldCover pixels) — [I2-Full]; train mIoU caps at 0.85-0.88 — [I2-Full].
- Roads: rural roads (3-8 m) missing from labels, sub-pixel and below S1 resolution; judged small effect — [PRX 4.1].

**(b) Radar-physics limits**
- Snow vs bare ground: 24 % of snow → bare, 39 % of predicted snow is bare — [I2-Full]; "dry snow and rock both scatter weakly" — [BF 10.6]. The calendar-aware PCA variant scored snow 0.813 vs 0.529 without — [T6]; order-free statistics lose the timing of snow cover — [BF 11.4].
- Water false positives: 23 % of predicted water is trees — [I2-Full]; attributed to radar shadow behind slopes or dark wet forest — [BF 10.6]; only HAND fixed it (0.868), not allowed — [T6].
- Crops vs grass outside the growing season look similar in C-band — [BF 10.6]; 20-21 % of crops → grassland persists from DEV to full run (0.660 → 0.656, no change) — [BF 11.2].
- Resolution: S1 GRD ~20 x 22 m resampled to 10 m — [PRX 4.1]; 47-48 % of Thane errors and 30 % of Berlin errors lie within 20 m of an Esri boundary — [CALC].
- Sparse acquisitions: 2-4 scenes/month in Thane vs 18-24 in Berlin; month-to-month VH change 1.56 vs 0.61 dB; flicker 0.48 vs 0.05 % — [CALC], run.json.

**(c) Model capacity / architecture**
- Only one architecture tried (U-Net ResNet-34, ~24.5 M parameters) — [PRX 2-3].
- Train mIoU 0.85-0.88, val peak 0.732 at epoch 22, then val loss rises — [I2-Full]: overfits training labels mildly, does not underfit them.
- 2218 training patches from 139 sites — [BF 10.1]; ResNet-50 "may overfit the 2218 patches" (project expectation) — [PRX 3.1].
- Crop → Grassland wrong pixels are partly radar-separable by VH temporal std (nearest centroid Crop 48-57 %) — [CALC]: the one case where a better use of the time series could help.
- Variant evidence in iteration 2: month_year vs year_stats vs three_month_year all within 0.007 rural test and within the 0.02 noise — [BF 10.3, 11.1]. More monthly input (18 vs 12 channels) did not help.
- Year_stats (no monthly input) was better on snow (0.52) and bare (0.88) in DEV; month_year better on man-made, crops, grassland — [BF 10.3].

**(d) Data coverage**
- Rare classes: Water 2.1 %, Snow 2.3 % of test pixels; a few false patches move IoU a lot — [BF 10.6].
- Rare-class IoU swings by up to 0.24 between identical runs because each rests on 1-2 sites — [DEV]; Wetland 0.10-0.78 across Run 4 variants from about 16 patches of one site — [RL Run 4].
- Weak regions Oceania 0.441, Africa 0.492; weak site types shrub_grass 0.233, wetland 0.223, snow_ice 0.222 — [BF 11.2]; same regions/types weak in iteration 1 — [T6].
- Val 21 sites (18 rural), test 17 (13 rural) — [BF 10.1]; val and test rankings disagreed in Run 4 and Test_6 — [RL], [T6 5].
- Dataset is 66 % urban relative to the rural objective (user memory note); iteration 2 cut Man-made to 9.1 % of train pixels — [BF 10.1].

### Inferences (what to mix and match)

**Encoder (E1-E5 in [PRX 3.1])**
- Argues against large gains: errors are boundary-band and radar-inseparable pixels; train mIoU capped by label noise; the 0.01-0.03 expected gain [PRX 2] is at the noise level. A deeper encoder (ResNet-50) is the least supported option given mild overfitting on 2218 patches.
- Mild support for testing smaller/more parameter-efficient encoders (EfficientNet-B0, DenseNet-121) and the no-pretraining control (E5): the overfit after epoch 22 suggests capacity is not the constraint, so a smaller model might match ResNet-34 at lower cost. The value is the documented negative/neutral result.

**Temporal model**
- Supported for Crop ↔ Grassland and Snow ↔ Bare: Crop/Grass wrong pixels carry crop-like VH variability that the current 1-month + 9-statistic input does not exploit; snow needs timing information (PCA's 0.813 snow) that order-free statistics discard.
- Constraint from evidence: calendar-tied features (PCA on calendar months) generalised badly to unseen sites/hemispheres [T6 5]. A temporal encoder over the 12-month sequence would need phase-robustness (e.g. random cyclic shift of the year as augmentation, or attention pooling that is not tied to absolute month). This is a design inference, not tested.
- Against simply adding more months as channels: three_month_year (18 ch) did not beat month_year (12 ch) [BF 10.3].

**Loss**
- Against further up-weighting Water/Snow: their problem is precision (0.59-0.60), i.e. false positives; higher weights would push recall up and precision down.
- Supported: boundary-tolerant loss/weighting (down-weight or ignore a 1-2 pixel band at label edges), since ~half of the error is in a 20 m band that the sensor cannot resolve. T2 (CE only, no Lovász) is a reasonable ablation but has no specific evidence for or against.

**Label handling (L1-L4 in [PRX 4])**
- Strongest support of all changes: Grassland 55 % agreement, rural agreement vs all-pixel gap 0.08, train-mIoU ceiling, and definition-driven Esri mismatches. Agreement-only training (L1), label smoothing (L3) and soft bootstrapping (L4) target exactly this. Evaluation should keep agreement-pixel scoring as the main score so label noise does not hide gains.
- A coarser class definition (merging Grassland into a broader "low vegetation" or reporting Crop+Grass together) would remove the largest confusion by construction; whether that fits the product is a product decision (the project kept them separate on purpose [SC]).

**Ensembling**
- Supported, modestly: run-to-run noise of ~0.02 means averaging 2-3 seeds of the same model would both stabilise the score and reduce flicker (smoothing already showed softmax averaging cuts flicker 2.4 → 0.9 % at no cost). Multi-encoder ensembling has no direct evidence yet. Cost: 2-3x weights and inference [PRX 3.4].
- Soft class percentages (probability averaging over area [PRX 3.3]) fit the boundary-band evidence and cost no training.

**Input features (S1-only)**
- Water false positives in shadow/dark forest are the main S1-only gap; terrain fixed them but is not allowed. Evidence supports adding S1-derived geometry information that the current input lacks: the layover/shadow mask is already computed and then discarded (masked to NaN, filled with 0 after standardisation [I2-Full model_card]); feeding it (or local incidence angle) as a channel is an S1-only option. Inference, not tested.
- Separate ascending/descending statistics: shadow falls on different slopes per orbit, so real water is dark in both and shadow in one. Inference, not tested; Thane has only one orbit [PRX 1], so this would not help everywhere.
- Scene-count-aware input (number of scenes per month) could let the model down-weight noisy months in sparse areas (Thane 2-4 scenes). Inference.

**Data**
- Supported: more training sites for the weakest groups (Oceania, Africa, shrub/grass, wetland edges, snow/ice) and, more importantly, more val/test sites, because current val (21 sites) cannot resolve differences below ~0.02-0.05 and has already selected the wrong variant once.
- Rural prototype checks are still missing; the urban Thane/Berlin runs cannot validate the rural objective.

### Gaps
- No experiment has yet isolated architecture effects (3rd-iteration notebooks not run), so category (c) is diagnosed only indirectly (learning curves, separability, variant ties).
- Separability test uses annual statistics recomputed from the prototype's monthly means; the model sees standardised statistics with slightly different definitions (2nd lowest/highest rather than min/max), and it sees spatial context, which a per-pixel test ignores. A pixel that looks inseparable per pixel could still be separable with context.
- No per-pixel softmax probabilities are stored, so calibration/uncertainty-based options (e.g. abstaining on low-confidence boundary pixels) cannot be evaluated from existing files.
- The Water false-positive cause (shadow vs wet forest vs label) is asserted in the reports but not checked against terrain data in any stored file.
