# Session Context

This is a handover note for the next working session: what has been done, where things stand, and what comes next. Last updated: 2026-10-24.

## 1. Project goal

This is the exam project for SRH Hochschule Heidelberg's Data Engineering 1 course. The full requirements are in `Documentation/requirements_info.txt` (slides 9–10 of `Documentation/01 - Introduction and Organization.pdf`).

- Train a deep learning model for **LULC semantic segmentation with Sentinel-1 as the only input**.
- Build a prototype:
  - **Input:** lat/lon and a start month.
  - **Processing:** 12 monthly LULC maps.
  - **Calculation:** per-class percentages per month, the month-to-month change, and the total change from first to last month (both in percentage points).
  - **Visualization:** the maps.
- Submit:
  - the dataset (a download link if it is too big);
  - notebooks with training curves and results;
  - the prototype source, with a README and a requirements file;
  - the model weights;
  - a screen recording of a run on an urban area.
- The report covers:
  - Dataset and preprocessing: calibration, dB conversion, speckle filtering, terrain correction, normalization, patching, label matching, augmentation, splits.
  - Model and design reasoning.
  - Results: OA, mIoU, per-class IoU/F1, training curves, variant comparison.
  - Prototype: architecture, compositing, change calculation, and a case study checked against an independent LULC map.
  - Critical discussion: unseen regions, SAR class confusion, seasonal effects, and how errors inflate change percentages.
- `Documentation/lecture_notes.txt` adds three points: address class imbalance, get high accuracy across all classes, and use sparse categorical cross-entropy and early stopping.

## 2. User preferences

- Replies in "caveman" style (terse), as set in the global CLAUDE.md.
- The user wants a non-generic dataset that **works worldwide**, and a **compact** one: at most 4–5 GB.
- **Never run notebooks or training code locally.** Edit `Prototype.ipynb` only; the user runs it on Kaggle (GPU T4). Static checks such as `ast.parse` are fine.
- **Do not install torch** in the local `.venv`.
- The user commits and pushes themselves. Do not commit unless asked. If asked, commit on a new branch off `main`.
- Remove comments and emojis from code before commits. Docstrings were kept.

## 3. Environment

- Windows 11. The repository is at `E:\Github Projects\Deep_Learning_Exam`.
- `.venv`: Python 3.13, created with `uv`; system Python 3.15 has no torch wheels. Installed packages: earthengine-api, rasterio, geopandas, pyogrio, numpy, tqdm, matplotlib. These match `requirements.txt` for the dataset pipeline, apart from geopandas and pyogrio, which are left over from the Urban Atlas version.
- Google Earth Engine:
  - The project ID is in `.env` as `Project_ID = ...`. `.env` is git-ignored.
  - Authenticated; credentials are in `~/.config/earthengine/credentials`.
- The local GPU (RTX 3050 Ti, 4 GB) is too small for training, so training runs on Kaggle.
- Kaggle:
  - The notebook `LULC_Deep_Learning` is synced to this repo as `Prototype.ipynb` through Kaggle's GitHub integration (commits named "Kaggle Notebook | LULC_Deep_Learning | Version N").
  - The dataset was uploaded as `optimalbhandari/sentinel-1`. On Kaggle it appears at `../input/datasets/optimalbhandari/sentinel-1/Dataset`.
- Disk: about 48 GB free on E:.

## 4. Dataset (done)

Full documentation: `Dataset/README.md`, which includes a work log, per-site table, limitations and design decisions.

- **Source:** Google Earth Engine.
  - Input: `COPERNICUS/S1_GRD`, IW mode, VV + VH, ascending and descending passes, year **2021**.
  - Label sources: `ESA/WorldCover/v200` and `GOOGLE/DYNAMICWORLD/V1` (the 2021 per-pixel mode).
  - **Esri 10 m LULC 2021** is held out as the independent sanity-check map.
- **Preprocessing:**
  - Earth Engine already applies the orbit file, border and thermal noise removal, calibration to sigma0, terrain correction (SRTM) and dB conversion.
  - The pipeline adds:
    - an incidence-angle mask (31–45°);
    - a monthly mean in linear power (temporal speckle reduction);
    - conversion back to dB;
    - a 3×3 focal median filter.
- **Storage:**
  - Backscatter is uint8: `dB = code*0.2 - 35`, code 0 = no data.
  - Each site is a 1024×1024 px tile (10.24 km) at 10 m, in a local UTM CRS.
  - `s1/<site>.tif` holds 24 bands (VV_01, VH_01, ..., VH_12).
  - `labels/<site>.tif` holds 3 bands: consensus, worldcover, dynamicworld.
- **Labels:** 8 classes, plus 0 = ignore.
  - Classes: Built-up, Cropland, Trees, Shrub & grass, Bare/sparse, Water, Wetland & mangrove, Snow & ice.
  - Consensus rule: a pixel is labelled where WorldCover and Dynamic World agree.
  - Wetland exception: WorldCover wetland is accepted when Dynamic World says trees, grass/shrub, water or wetland, because Dynamic World has no mangrove class.
  - The rule is `config.consensus_labels`. `download.py --relabel` recomputes the labels offline.
- **Sites:** 116 in total.
  - 77 urban sites across 6 regions.
  - 39 rural sites, each focused on one rare class: cropland 11, trees 6, bare 6, shrub/grass 4, wetland 6, snow/ice 6.
  - Gobi, the Greenland margin and Brisbane were replaced for missing Sentinel-1 IW coverage.
- **Splits (by site):** train 78 sites / 1214 patches, val 18 / 270, test 20 / 296.
  - Patches are 256×256 and stored as an index (`patches/<split>_patches.csv`: site, region, focus, row, col).
  - A patch is kept if at most 60% of it is ignored and every month has at least 99% valid pixels.
- **Class shares in train:** built-up 46.9%, trees 13.3%, water 9.6%, cropland 8.1%, bare 7.4%, snow 5.4%, wetland 5.3%, shrub/grass 3.9%.
- **Normalization stats (train):** VV −8.31 ± 6.05 dB, VH −15.66 ± 5.95 dB. The notebook recomputes 3-channel stats itself, including the VV−VH ratio (7.32 ± 2.87).
- **Size:** about 2.4 GB in total. 2.2 GB of S1 tiles, 12 MB of labels, 226 MB of previews, plus a small index and stats files.
- **Code in `data_pipeline/`:**
  - `config.py`: sites, classes, encoding, consensus rule.
  - `download.py`: GEE download, with retries and continue-on-failure.
  - `make_patches.py`: patch index and statistics.
  - `preview.py`: quick-look PNGs.

Approaches that were dropped:
- ASF/NASA raw GRD with SNAP processing (too heavy).
- Urban Atlas 2018 labels (Europe only).
- int16 values with 15 km tiles and copied `.npy` patches (about 20 GB).
- Lowering the resolution and sharpening later (it cannot recover detail and amplifies speckle).

## 5. Training notebook `Prototype.ipynb` (in progress)

32 cells, PyTorch, written for Kaggle.

- **Paths:** relative. Data is found by searching `../input`, `Dataset` and `../Dataset` for `patches/train_patches.csv`. Output goes to `outputs/`.
- **Loading:** all patches are loaded into RAM (2.8 GB, about 170 s on Kaggle).
- **Features:** VV, VH and VV−VH per month, standardized with train-only statistics; NaN becomes 0.
- **Training items:** each item is a patch with a random month. With `n_months > 1`, the previous months are added; the earliest month is repeated for January and February.
- **Augmentation:** flips and 90° rotations.
- **Imbalance handling:**
  - class weights `sqrt(median_freq/freq)`, normalized: built-up 0.41 up to shrub & grass 1.43;
  - a `WeightedRandomSampler` on the largest class weight among the classes covering at least 1% of a patch.
- **Model:** U-Net with an ImageNet-pretrained torchvision ResNet-34 encoder. The first convolution is adapted to `3*n_months` channels. About 24 M parameters.
- **Variants:** `VARIANTS = {"single_month": 1, "three_month": 3}`.
- **Loss:** weighted cross-entropy (`ignore_index=-1`; equivalent to sparse categorical cross-entropy) plus 0.5 × Dice.
- **Training:** AdamW (lr 3e-4, weight decay 1e-4), AMP, `ReduceLROnPlateau` on val mIoU, early stopping (`PATIENCE = 8`).
- **Settings in section 9:** `MAX_EPOCHS` (**currently 2, for the smoke test; set to 60 for the full run**), `BATCH = 16`, `NUM_WORKERS = 2`.
- **Evaluation:**
  - test OA, mIoU and per-class IoU/F1 for every patch × all 12 months;
  - variant comparison;
  - confusion matrix;
  - mIoU per region, per focus and per month (seasonal effect);
  - example predictions.
- **Exports to `outputs/`:** `<variant>_best.pt`, `<variant>_history.csv`, `test_metrics.json`, `model_card.json` (the preprocessing the prototype needs), and the plots.

Kaggle run status:
- Kaggle version 3 ran up to section 4 and failed with `ValueError: cannot reindex on an axis with duplicate labels` (a `pd.crosstab` over concatenated split tables with duplicate row labels).
- Two fixes were made locally:
  - `ignore_index=True` in that concat;
  - `int` month indexing in `evaluate(per_month=True)`.
- The git log shows "Fixed value error problem" and Kaggle Version 4 after that. **Whether the 2-epoch smoke test has passed end to end is unknown.** Check the latest outputs in `Prototype.ipynb`.
- Expected runtime on a T4: about 40–60 s per epoch, about 45–90 min for both variants.
- Accelerator: use GPU T4 or P100. A TPU would need torch_xla. T4 x2 uses only one GPU, since the notebook has no DataParallel.

## 6. Open items and next steps

1. **Check the smoke test result** in `Prototype.ipynb` outputs (cells after section 9). Fix any errors, then set `MAX_EPOCHS = 60` for the full run.
2. **Optional: Kaggle API workflow** to push and run from here without a browser.
   - The user puts `kaggle.json` at `C:\Users\Bhand\.kaggle\`.
   - Then: install `kaggle` in `.venv`, add a `kernel-metadata.json` (the notebook ID for LULC_Deep_Learning, dataset `optimalbhandari/sentinel-1`, GPU and internet on), and use `kaggle kernels push / status / output`.
   - This was proposed; the user has not answered yet.
3. **Optional: move consensus and patching logic into the notebook**, so that label or filter tuning needs no dataset re-upload. This was proposed and has not been answered either.
4. **Prototype:**
   - Input lat/lon and a start month; GEE monthly composites reuse `download.py`'s `monthly_composite`.
   - Tile inference with the best model and `model_card.json`.
   - 12 maps, class percentages, month-to-month and total change in percentage points.
   - A comparison with Esri LULC 2021.
   - A README, `requirements.txt` and a screen recording on an urban area.
5. **Report:** the material is in `Dataset/README.md` (dataset sections and work log) and in the notebook outputs.
6. **Dataset download link** for the submission: the Kaggle dataset `optimalbhandari/sentinel-1`, which needs to be made public or shared.
7. **Housekeeping:** `requirements.txt` still lacks the training packages (the notebook uses Kaggle's preinstalled torch and torchvision). geopandas and pyogrio are no longer needed by the pipeline.

## 7. Update 2026-10-24

- **Baseline full run** (commit `0811060`, 116 sites): `three_month` test OA 0.843, mIoU 0.559; `single_month` 0.810 / 0.507. Weak: trees→cropland (0.24), water→snow (0.19), boreal trees test site OA 0.28, January mIoU 0.38 (window repeated Jan), train/val mIoU gap 0.31.
- **Dataset v2:** 38 rural sites added (`config.EXTRA_RURAL_SITES`); Vasyugan → Danube delta (no IW month); finland_boreal, australia_spinifex, simpson_desert dropped (0 patches). Now 151 sites, 1670 / 278 / 296 patches, 3.1 GB. Test split unchanged. README updated (sections 1, 4, 6, 7, 8, 10, 12, 13).
- **User must re-upload** `Dataset/` as a new version of Kaggle dataset `optimalbhandari/sentinel-1`.
- **Notebook rewritten** (38 cells, friendlier markdown): `DEV` switch (3 months val, 15 epochs, 2 variants), wrap-around month window, random crop jitter ±128 px from full train tiles, dB shift/noise augmentation, PCA section (36-dim yearly vector → 6 PCs) and `three_month_pca` variant, fixed `SAMPLES_PER_EPOCH = 2500`, `NUM_WORKERS = 4`, model card with PCA weights. Not yet run on Kaggle.
- **Dataset v3 (same day):** 25 biome-gap sites (`config.BIOME_GAP_SITES`: tundra, savanna, Mediterranean, smallholder, Kampala, Port-au-Prince, steep terrain, dry forest). Seasonal snow label fix: 4th label band `dynamicworld_snowfree`, rule in `config.consensus_labels`; all labels re-downloaded (old labels backed up in the session scratchpad only). Now 176 sites, 1958 / 327 / 302 patches, 3.6 GB. README updated incl. full per-site table. Test sites unchanged, +6 test patches from the snow fix.

## 8. Improvement round (2026-10-24, after the first DEV run on dataset v3)

### 8.1 What the DEV run showed (15 epochs, val on 3 months)

| Variant | best val mIoU | test OA | test mIoU |
|---|---|---|---|
| baseline `three_month` (old data, 60 epochs) | ~0.47 | 0.843 | 0.559 |
| `three_month` (new data) | 0.546 | 0.814 | 0.535 |
| `three_month_pca` (new data) | **0.656** | **0.834** | 0.544 |

- **Overfitting is mostly fixed:** the train-val mIoU gap fell from 0.31 to 0.05 (`three_month`) and 0.12 (`three_month_pca`). The new sites and the augmentation work.
- **PCA helps:** trees IoU 0.45 → 0.60, shrub 0.26 → 0.34, water 0.49 → 0.54. **Wetland got worse:** 0.42 → 0.22, probably because seasonally flooded land now looks like water to the model.
- **Test is not better than the baseline yet.** It was a 15-epoch DEV run, and val mIoU jumps about ±7 points between epochs (0.656 → 0.583). Differences of 1–2 points between runs are therefore noise. Fixing this is step 1 below.
- **PCA was 5x slower** (165 s vs 33 s per epoch), because the 6 components were recomputed for every sample. **Fixed:** the PCA maps are now computed once per tile and stored (float16).

### 8.2 Research (4 parallel literature searches)

Full report: `reports/Sentinel 1 LULC segmentation improvements.md`; raw notes with sources: `research_notes/Sentinel 1 LULC segmentation improvements/`.

Key points to remember for the exam report:
- **Label ceiling.** The labels themselves are imperfect. ESA WorldCover is 76.7% correct overall, but only ~47-49% for shrubland and 30-45% for herbaceous wetland. Dynamic World agrees with experts on 73.8% of pixels. Our weakest classes (wetland, shrub) are also the weakest in the label products. Realistic target on our agreement test set: mIoU ~0.58-0.62.
- **Our numbers are not comparable to published S1-only scores** (e.g. SpecSAR-Former, S1-only on Dynamic World classes: mIoU 0.24), because our test set only contains pixels where two label maps agree (the easy pixels; ~18% ignored). Say this openly in the report.
- **Temporal information is the biggest lever for SAR** (U-TAE on PASTIS: 63.1 mIoU vs 28.3 for a single date). This is why the 3-month window and the PCA annual summary helped.
- **A plain U-Net is a strong baseline**; big foundation models give little for S1 only and are too heavy for a T4.

### 8.3 What we are changing now (and why)

| # | Change | Where | Why | Expected effect |
|---|---|---|---|---|
| 1 | **EMA weights** (a running average of the model weights, decay 0.998), used for validation, saving and testing | notebook | Averages out the epoch-to-epoch jumps, so the curves and model choice become reliable | +0.5 to +1.5 mIoU, smoother val curve |
| 2 | **Warmup + cosine learning-rate schedule** instead of ReduceLROnPlateau | notebook | A fixed schedule gives every variant the same training, so variants are fairly comparable; LR decays smoothly to 0 | part of the +0.5 to +1.5 above |
| 3 | **Lovász-softmax loss instead of Dice** (CE + 0.5 Lovász) | notebook | Lovász directly optimises IoU, the metric we report | +0.3 to +1.5 mIoU |
| 4 | **Seasonal statistics channels** from the 12 months we already have: VV min, VV max, VH std, VH spread (p90-p10), share of months that look like water (VV < -18 dB) | notebook | Cropland changes a lot over the year, trees do not; seasonally flooded wetland is water only in some months, a lake in all of them | targets cropland and wetland |
| 5 | **Terrain channels** (optional ablation): elevation and slope (Copernicus GLO-30 DEM), height above nearest drainage (HAND, MERIT Hydro) | new `Dataset/terrain/`, `data_pipeline/download.py`, notebook | Wetland and water lie low, flat and close to rivers; snow lies high. Terrain is static context, not a second sensor, so it is reported as "S1 + terrain" next to the pure S1 result | uncertain; mainly wetland, water, snow |

New variants: `three_month_pca_stats` (still Sentinel-1 only) and `three_month_pca_stats_terrain` (S1 + terrain). The comparison between them is the ablation for the report.

Not done (and why): large foundation models (too heavy, little gain on S1), GLCM texture (no evidence it helps a CNN), polarimetric entropy/alpha and InSAR coherence (need SLC data, not in GEE), focal loss (hard pixels here are often wrong labels). Later options: U-TAE over all 12 months, self-training on the ignored pixels, sliding-window inference with overlap in the prototype.

### 8.4 Status of this round (end of 2026-10-24)

Done:
- **Terrain layers downloaded** for all 176 sites: `Dataset/terrain/<site>.tif`, 3 int16 bands (`elevation_m`, `slope_decideg`, `hand_dm`), same grid as the S1 tiles, 0.50 GB. Dataset now **4.28 GB**. Code: `terrain_image()` in `data_pipeline/download.py` (Copernicus `COPERNICUS/DEM/GLO30_2024_1` + `MERIT/Hydro/v1_0_1` band `hnd`), `config.TERRAIN_DIR` / `TERRAIN_BANDS`. Rerunning `download.py` only fetches missing files. GLO-30 is a surface model: buildings and canopy add some height/slope in cities.
- **`Prototype.ipynb`** (41 cells, syntax-checked, not run yet):
  - new section **5b**: seasonal statistics + terrain channels, all static channels (6 PCA + 5 stats + 3 terrain = 14) precomputed once per tile as float16 (`STATIC_TILE`, `STATIC_X`, ~5 GB RAM); each variant picks its channels with `static_index(variant)`;
  - variants: `single_month`, `three_month`, `three_month_pca`, **`three_month_pca_stats`** (S1 only), **`three_month_pca_stats_terrain`** (S1 + terrain); DEV runs the last three; terrain variants are skipped automatically if the dataset has no `terrain/` folder;
  - loss = weighted CE + 0.5 x **Lovász-softmax** (Dice removed);
  - **EMA** model (`EMA_DECAY = 0.998`) used for val, checkpoint and test; **1 warmup epoch + cosine** LR schedule per step; `PATIENCE = 15`;
  - `model_card.json` now also stores the static channel list and the stats/terrain definitions and normalisation (the prototype must reproduce them).
- `Dataset/README.md` updated (quick facts, folder layout, sources, design log, work log entry 14).

Next for the user:
1. Upload `Dataset/` (incl. the new `terrain/` folder) as a new version of the Kaggle dataset.
2. Run `Prototype.ipynb` with `DEV = True` (~30-40 min for 3 variants).
3. Compare: does `three_month_pca_stats` beat `three_month_pca` (seasonal stats help)? Does `..._terrain` beat `..._stats` (terrain helps, mainly wetland/water/snow IoU)? Is the val curve smoother (EMA)?
4. If good: `DEV = False` for the final run (5 variants x 60 epochs, roughly 3-4 h on a T4).

Study questions to be able to answer in the exam:
- Why does EMA make validation more stable, and why is it not the same as a lower learning rate?
- Why is a fixed cosine schedule fairer for comparing variants than ReduceLROnPlateau?
- What does Lovász-softmax optimise, and why does that match mIoU better than cross-entropy?
- Why would "water months" separate a lake from a seasonally flooded wetland?
- Why is terrain reported as an ablation, and what does HAND measure?
- Why are our mIoU numbers not comparable to published S1-only benchmarks (agreement-filtered labels)?

## 9. Update 2026-10-25: two DEV runs, notebook mix-up, run folders

Details: `research_notes/run_log.md` (all runs) and `reports/DEV runs 2026-10-25.md` (write-up).

### 9.1 What happened

- Commits `5da7e0f` ("Used the wrong dataset") and `d714799` ("Test on adding new biomes") are Kaggle syncs. Both hold the **old 38-cell notebook**: variants `three_month` and `three_month_pca`, ReduceLROnPlateau, CE + Dice, no EMA, no stats, no terrain. The improved 41-cell notebook from section 8 (commit `865480f`) **has never run on Kaggle**, because Kaggle synced its older copy over it.
- The dataset was not the problem. Both runs loaded 1958 / 327 / 302 patches from 134 / 22 / 20 sites, the same as local `Dataset/patches/`:
  - run 1 (`5da7e0f`): `../input/datasets/optimalbhandari/sentine-1-all/Dataset`;
  - run 2 (`d714799`): `../input/datasets/optimalbhandari/sentinel-1-2-0/Dataset`.
  The old notebook never reads `terrain/`, so it does not matter whether that folder was in the Kaggle dataset.
- Code, data and seed were the same in both runs. **The difference between them is run-to-run noise.** This is the first direct noise measurement.

### 9.2 Results (DEV: 15 epochs, val on months 0/4/8, test on all 12 months)

| Run | Variant | best val mIoU | test OA | test mIoU |
|---|---|---|---|---|
| 1 (`5da7e0f`) | three_month | 0.541 | 0.817 | 0.559 |
| 1 | three_month_pca | 0.644 | 0.835 | 0.546 |
| 2 (`d714799`, `Weights/Test_2`) | three_month | 0.530 | 0.823 | 0.538 |
| 2 | three_month_pca | 0.634 | 0.813 | 0.485 |

Reading:
- Same code, same data: test mIoU moved by 0.021 (three_month) and 0.061 (three_month_pca). Single-class IoU moved by up to 0.24 (Water three_month 0.679 → 0.443). **Single-run differences below about 0.05 test mIoU are noise.** Rare-class test IoU comes from 1-2 sites each (≈16 patches).
- PCA is consistent in two ways across both runs: **val mIoU +0.10**, and **test IoU for Trees +0.13 to +0.14 and Shrub & grass +0.10 to +0.12**. Its test mIoU is lower in both runs (−0.013, −0.053), mainly through Cropland and Wetland. Train-val gap at the best epoch is 0.127 with PCA vs 0.054 without, which points to site-specific overfitting of the yearly PCA vector.
- The best epoch was 14 of 15 in every case. DEV runs are too short to rank variants.
- `BEST` is chosen by **test** mIoU ("best variant on test"). That is test-set leakage. It must be chosen by val mIoU.
- Weak classes are unchanged: test mIoU on the focus sites is shrub_grass 0.03, snow_ice 0.15, trees 0.18, wetland 0.22.

### 9.3 Notebook changes this session

`Prototype.ipynb` is again the improved notebook from `865480f` (now 44 cells, syntax-checked, not run). The old 38-cell version with the run-1/run-2 outputs stays in git at `d714799`. Changes on top of `865480f`:

- `find_data_root()` lists every attached dataset and prefers the one with `terrain/`.
- Validation uses all 12 months in DEV too (`VAL_MONTHS = list(range(12))`). It was 3 months, which made val too noisy.
- `BEST` is chosen by **best val mIoU**; test is only reported.
- Speed: the previous run was CPU-bound (32 s/epoch without PCA, 158 s with PCA built on the CPU, 4 Kaggle vCPUs).
  - Datasets now return raw uint8 codes, float16 static channels and labels. `gpu_input()` does the dB lookup, brightness shift + noise, VV-VH and standardisation for the whole batch on the GPU; `gpu_augment()` does flips and rotations. `model_input()` is gone; `month_features()` remains only for fitting the PCA.
  - Lovász runs on a random 25 % of pixels (`LOVASZ_SAMPLE`). The loss returns `(total, ce)`; logged train and val loss are both CE only, so they are comparable. Val loss no longer computes Lovász.
  - DEV validates every 2nd epoch (`VAL_EVERY`), plus the last epoch; patience counts epochs. Curves plot val only on validated epochs.
  - `cudnn.benchmark = True`, model and inputs in channels-last layout.
  - Val and test arrays live on the GPU (`EVAL_GPU`, about 2 GB); `eval_batches(split, variant, months, ids)` yields model-ready batches, `evaluate(model, batches)` consumes them. `EvalDataset` is gone. Reason: on the RTX A6000 (Datalab, batch 64) train epochs took 10 s but CPU-side validation 70-150 s with GPU at 0 %.
  - `/dev/shm` < 1 GB (Docker default 64 MB) sets `NUM_WORKERS = 0` (fixes "Bus error" in workers).
  - Timing: `RUN_START` in the imports cell, minutes per variant in the training cell, `timing.json` in the run folder.
- New glossary cell after the title, and a "How to read the training log" part in section 9 (loss values as probabilities, loss vs mIoU, typical patterns).
- Output folder is now `Weights/<RUN_NAME>/`, set by `RUN_NAME = "Test_3"` in cell 2. If that folder already has files, the run goes to `Weights/<RUN_NAME>_<timestamp>/`, so nothing is overwritten. On Kaggle the path is `/kaggle/working/Weights/Test_3/`.
- `<variant>_best.pt` is saved at every val improvement, not only at the end, so a crash keeps the best weights.
- New section 12 "Save the run" at the end: saves all final weights and histories again and zips the folder (`Weights/Test_3.zip`) for one download.
- Files per run folder match the user's local `Weights/Test_2/`: `<variant>_best.pt`, `<variant>_history.csv`, `model_card.json`, `test_metrics.json`, and the plots.
- Local `Weights/Test_2/three_month_best.pt` is dated 12:02, but the other files are dated 14:11. It is probably the **run 1** weights. Re-download it from the run-2 Kaggle output if it matters.
- `Weights/` is not in `.gitignore`. Each `.pt` is about 98 MB, close to GitHub's 100 MB file limit.
- Also new: `pyproject.toml` + `uv.lock` (Python 3.13, torch 2.11 cu128, same as Kaggle), for `uv sync` in other repos.

### 9.4 Next steps

1. Commit and push `Prototype.ipynb`, then **pull it into Kaggle before running**, or Kaggle syncs the old copy back again.
2. DEV run of that notebook on the dataset version with `terrain/`. Check that cell 3 prints the right path and that cell 7 does not print "terrain variants skipped".
3. Final run: 2-3 seeds per final variant, or at least report the noise level from 9.2 next to each comparison.

## 10. Update 2026-10-26: Test run 4 logged

Details: `research_notes/run_log.md`, section "Run 4".

- First run of the improved notebook (commit `5a6d075`, `Weights/Test_4/`), DEV, on the Datalab RTX A6000 with `terrain/` loaded.
- Results (best val mIoU / test OA / test mIoU): three_month_pca 0.613 / 0.878 / 0.664; three_month_pca_stats 0.679 / 0.839 / 0.566; three_month_pca_stats_terrain 0.654 / 0.832 / 0.558. `BEST` by val = three_month_pca_stats.
- Val and test rankings disagree. The test gaps come from Wetland (IoU 0.10-0.78) and Bare (0.36-0.73), each from about 16 patches of one site. Treat them as noise.
- Best epoch 15/15 for every variant, so 15 epochs is too short. Train-val gap 0.17-0.22.
- `EMA_DECAY` is 0.99 in the notebook, not the 0.998 planned in section 8.
- `/dev/shm` was small, so `NUM_WORKERS = 0`. Epoch times varied 9-164 s.

### 10.1 Next steps

1. The user sets `DEV = False` and runs the final 60-epoch run.
2. Not possible for now (user's call): more seeds, the `/dev/shm` fix. Report the noise level from runs 1-2 and 4 next to each comparison instead.

## 11. Update 2026-10-26: Test_5_Full, git cleanup, speed and overfitting changes

Full description of the notebook: `reports/Notebook method and pipeline.md`. Strategy check and PCA question: `reports/literature survey_reiterate.md`.

### 11.1 Git

- `.env` (GCP project ID) removed from all history; `.gitignore` fixed (`.env`, `Weights/`, `Documentation/*.pdf`). Force-pushed.
- The user wants **no "Co-Authored-By: Claude" lines** in commits or PRs.
- Lecture PDFs are no longer tracked (`01 - Introduction and Organization.pdf` is still in old history, commit `e05461e`).
- After every force push, the Datalab clone needs `git fetch && git reset --hard origin/main` (not pull/merge). Suggested there: `git config pull.ff only`.

### 11.2 Test_5_Full (60 epochs, all 5 variants, old data loading, A6000)

- Datalab closed during `three_month_pca` epoch 3; the run is dead. Finished: `single_month` best val mIoU 0.548 (epoch 45, early stop at 60), `three_month` 0.599 (epoch 50).
- **Overfitting:** train mIoU at the best epoch 0.76 / 0.825, gap 0.21-0.23. `three_month` val loss lowest at epoch 27 (0.598), then rising to 0.648 while train loss kept falling; val mIoU flat after about epoch 30.
- 13-30 s per epoch (about 20 min per variant).

### 11.3 Strategy check (lectures + literature)

- The training algorithm matches lectures 01-07: U-Net, ResNet-34, transfer learning, Adam(W), L2, dropout, batchnorm, early stopping, augmentation, sparse CE with class weights. Lovász, EMA, PCA and the statistics are extras that need a justification in the report.
- **The brief requires Sentinel-1 as the only input:** terrain is an ablation only.
- **PCA:** valid in dB, but its loadings are tied to calendar months (hemisphere phase shift → site-specific). The seasonal statistics are order-independent and follow ESA WorldCover (S1 percentiles); they are the better-founded feature.

### 11.4 Notebook changes (all marked `CHANGED (2026-10-26...)` in code, *(changed 2026-10-26...)* in markdown)

Speed (commit `b4b5175`):
- New hardware-check cell: bf16 on Ampere / fp16 + GradScaler on T4, TF32, `BATCH` 64 (≥ 24 GB) or 32, `NUM_WORKERS` from `/dev/shm`.
- Data kept on the GPU when it fits (`gpu_train_batches`, `eval_batches`); DataLoader only as fallback.

Overfitting and ablation (uncommitted, for Test_6):
- `WEIGHT_DECAY` 1e-4 → 0.05 (not on biases / BatchNorm); `LABEL_SMOOTHING` 0.1 in the optimised CE (logged CE stays plain); `DROPOUT` 0.1 → 0.2; `MONTH_DROP` 0.15; VV/VH brightness shift drawn separately; `MAX_EPOCHS` 60 → 40 (final); `PATIENCE` 15 → 10.
- New variant `three_month_stats`. `RUN` = three_month, three_month_stats, three_month_pca, three_month_pca_stats, three_month_pca_stats_terrain.
- `BEST` chosen by val mIoU among Sentinel-1-only variants.
- `RUN_NAME = "Test_6"`; `DEV` is still `False` in cell 3.
- Not run, only syntax- and name-checked.

### 11.5 Next steps

1. Commit and push, then on Datalab `git fetch && git reset --hard origin/main` before running.
2. Run Test_6. Check: train-val gap clearly below 0.2, val loss not rising late, `three_month` val mIoU ≥ 0.599, and whether `three_month_stats` matches or beats `three_month_pca_stats` (if so, drop PCA).
3. Then: prototype (section 6 item 4), report, dataset link, `requirements.txt`.
