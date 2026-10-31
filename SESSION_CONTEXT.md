# Session Context

This is a handover note for the next working session: what has been done, where things stand, and what comes next. Last updated: 2026-10-31.

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

## 12. Update 2026-10-26: Test run 6 (first complete FINAL run)

Details: `reports/Test run 6 results.md` (full write-up), `research_notes/run_log.md` ("Run 6"). The method is in `reports/Notebook method and pipeline.md`, the strategy and PCA background in `reports/literature survey_reiterate.md` (now with an "Update after Test_6" section).

### 12.1 Setup

- Notebook commits `b4b5175` (GPU speed) + `5414256` (overfitting, the user's commit). FINAL, 40 epochs, 5 three-month variants.
- Datalab RTX A6000, bf16, batch 64, all data on the GPU, no workers.
- **196 epochs in 27 min (about 8 s per epoch, Test_5: 13-30 s).** The GPU-resident data fixed the `/dev/shm` problem.

### 12.2 Results (best val mIoU / test OA / test mIoU)

| Variant | val | test OA | test mIoU |
|---|---|---|---|
| three_month | 0.570 | 0.843 | 0.585 |
| three_month_stats | 0.657 | **0.914** | **0.742** |
| three_month_pca | 0.666 | 0.874 | 0.667 |
| three_month_pca_stats (BEST by val) | 0.671 | 0.865 | 0.617 |
| three_month_pca_stats_terrain (ablation) | 0.694 | 0.890 | 0.701 |

All beat the baseline (test mIoU 0.559).

### 12.3 Findings

- **Overfitting improved, not solved.**
  - Val loss now reaches its minimum at epoch 34-36 of 40 and stays flat (Test_5: rising after epoch 27).
  - Gap 0.18-0.21 (was 0.21-0.23).
  - pca_stats still overfits (best epoch 26, early stop 36).
- **PCA ablation:** `three_month_stats` (no PCA) is best on test by 0.075-0.125 mIoU, best in Cropland 0.86, Trees 0.76, Shrub 0.65, Bare 0.83 and Wetland 0.70. This matches the survey's prediction (calendar-bound PCA vs order-independent statistics).
- **But val cannot separate stats / pca / pca_stats** (0.657 / 0.666 / 0.671, within noise). The val rule picked pca_stats, the weakest of the three on test. The test scores have now been seen; the report must say so.
- Every yearly-feature variant beats plain three_month by 0.09-0.12 val mIoU: yearly context is the biggest lever.
- Terrain: best val, Water IoU 0.868 (HAND), excluded (S1-only brief).
- three_month val 0.570 < Test_5's 0.599: possibly mild underfitting from the regularisation, or noise.
- Weak spots of the selected model: N. America mIoU 0.378, shrub_grass sites 0.138, wetland sites OA 0.372. Test is 54 % built-up (Built-up IoU 0.94-0.96), which inflates OA.

### 12.4 Next steps

1. Seed run: `three_month_stats` and `three_month_pca_stats`, 3 seeds each (about 35 min). Choose by mean val mIoU. If they tie, deliver `three_month_stats` (WorldCover-style, order-independent, 14 channels, no PCA weights in the prototype). Not set up in the notebook yet.
2. `VAL_EVERY = 1` (validation now about 3-4 s).
3. Log the chosen model in `model_card.json`, then build the prototype, write the report, add the dataset link and complete `requirements.txt`.

## 13. Update 2026-10-27: Streamlit prototype, first live test

Details: `reports/2026-10-27 Prototype 1.md` (test results, Datalab setup fixes, improvement research).

### 13.1 Prototype

- Folder `prototype/`: `app.py` (Streamlit), `pipeline.py` (features, inference, statistics), `gee.py` (Earth Engine), `model.py`, `launch.py`, `README.md`, `requirements.txt`, `.streamlit/config.toml`.
- Weights by upload (`.pt` + the same run's `model_card.json`) or from `Weights/<run>/` on the server. The variant is detected from the first layer's channel count.
- `launch.py` runs the app on Datalab: JupyterHub proxy link if available, else a Cloudflare quick tunnel with a random `?key=` access key. On Datalab the Cloudflare tunnel worked.
- The Earth Engine project ID is never sent to the browser (blank field = `EE_PROJECT` / `.env` on the server).

### 13.2 Datalab Earth Engine setup (what worked)

- `uv venv` in `prototype/`, no `sudo`. `uv add gcloud` was a mistake (unrelated PyPI package; touched `pyproject.toml` / `uv.lock`, revert them).
- `earthengine authenticate --auth_mode=notebook`, on the login page a project that can host the login client, and **without** "read-only scopes" (read-only tokens fail on refresh with `invalid_scope`).
- The Datalab-provided Cloud project has no Earth Engine API / registration and the user cannot register it. Use the user's own registered project via `export EE_PROJECT=...`.

### 13.3 Test results (Test_6 `three_month_stats`, 2021, 5.12 km)

| Site | Agreement with Esri 2021 | Note |
|---|---|---|
| Central Berlin (52.52, 13.405) | 96.2 % | Esri is 97.1 % built-up: majority baseline is higher, test is uninformative. Parks, Spree correct. Water recall 59.5 % (thin river) |
| Potsdam (52.42, 13.03) | 77.9 % | Shapes match (lakes, fields, forest). Model 39.4 % Trees vs Esri 22.8 %, Built-up 32.6 % vs 48.0 %: tree-rich suburbs |

- The Trees vs Built-up disagreement is mostly definitional: Esri counts vegetation mixed with buildings as built area, WorldCover (our label source) counts urban trees as trees.
- Report agreement per class and against the majority-class baseline; use mixed areas for the sanity check and the screen recording.

### 13.4 Next steps (prototype)

1. Fairer sanity check: majority baseline, per-class F1, kappa, harmonised Built-up/Trees agreement.
2. Temporal smoothing (3-month softmax average or HMM), show raw vs smoothed change.
3. Flip test-time augmentation (2-4 × inference time).
4. Confidence map.
5. Sanity check on 5-6 mixed sites for the report. Retraining ideas (VV/VH texture channels, peri-urban sites) are future work.

## 14. Update 2026-10-31: Black Forest failure → 2nd iteration (rural objective)

Full story with evidence: `reports/2026-10-31 Prototype test 1 Black Forest.md` (sections 1-5 the failure, section 6 the decisions incl. 6.9 one final model, section 7 the training philosophy: variant names, months, PCA, terrain; section 8 the status before the build; section 9 the Datalab build and the data Q&A).

### 14.1 What happened

- The user ran the prototype (Test_6 `three_month_stats`, the 1st-iteration best model, test mIoU 0.742) on a Black Forest clearing: forest with meadows, no water, no buildings.
- The 12 monthly maps were wrong in a **seasonal** way: water, wetland and cropland in Jan-May and Dec, almost none in Jun-Nov, plus red built-up spots and rounded blob shapes. The PCA variant was more stable (yearly information outweighs the month).
- The user set the **product goal: rural land worldwide**. Cities are already well mapped, so the tool adds little there. Pixel-perfect maps are not possible (Sentinel-1 real resolution about 20 × 22 m; labels 65-75 % correct); the goal is the right classes, clean shapes at 20-30 m and no seasonal flicker.
- The test point is near the 1st-iteration training tile `black_forest` (48.2 N, 8.2 E), so the model failed on a landscape it had seen.

### 14.2 Causes found

| Cause | Evidence |
|---|---|
| no radiometric terrain correction (sigma0): hillsides look like buildings (facing the sensor) or water (facing away) | live check, July 2021 single scenes: VV correlated 0.74-0.82 with the slope towards the sensor (nepal_midhills, andes_sacred_valley, black_forest) |
| consensus labels ignored disagreement pixels, which are mostly boundaries, so edges were never trained | blob shapes; the 3 × 3 median filter blurred edges further |
| 3-month window without enough yearly context: a dark winter month (snow, frozen or wet soil) flips forest to water | seasonal error pattern; PCA (yearly) variant stable |
| city-heavy data: 79 of 176 sites urban (45 %), built-up the largest class (30.8 % of train, 54.2 % of test pixels) | test mIoU dominated by built-up (IoU 0.94-0.96), says little about rural land |
| Wetland & mangrove: worst labels (WorldCover herbaceous wetland 30-45 % correct) | wetland was a main winter false alarm |

Correction: earlier in the session the urban share was quoted as "66 % of sites, built-up 46.9 %"; those numbers belong to the old 116-site dataset. Fixed in the report, the notebook and the README.

### 14.3 Decisions (all implemented, not yet run)

1. **Rural objective**: model built, chosen and tested for rural land; cities only as towns, city edges and for the brief's urban screen recording.
2. **Terrain flattening**: gamma0, Vollrath et al. 2020 volume model on GLO-30, layover / shadow masked; 3 × 3 median dropped. Verified live: correlation with slope 0.74-0.82 → -0.14-0.05, VV spatial std -30-40 %, 97-99.8 % valid pixels.
3. **Sites (178)**: 54 mixed rural mosaics (41 train / 7 val / 6 test; 5 with lakes or rivers), 92 single-class rural sites (5 training glacier sites dropped), 32 towns and city edges (20 kept cities + Kampala, Port-au-Prince + 10 new city edges with Man-made 30-51 %). Mosaic rule: largest WorldCover class ≤ 80 %, second ≥ 10 % (the first rule, "≥ 3 classes with 5 %", rejected the Black Forest: WorldCover maps pasture as grassland). All new sites checked live; 14 mosaics moved by 0.15°. Test includes `black_forest_villages` (47.95 N, 8.10 E). Freiburg rejected as a city edge (19 km from that test site).
4. **7 classes** (the user chose names and colours): 1 Man-made `#e31a1c`, 2 Crop fields `#ffd92f`, 3 Trees `#1a9850`, 4 Grassland `#a6d96a`, 5 Bare ground `#bf812d`, 6 Water `#2c7fb8`, 7 Snow & ice `#dfe9f0` (light blue-grey, not white). Wetland removed: WorldCover 95 mangroves → Trees, 90 herbaceous wetland → Grassland, Dynamic World flooded vegetation → Grassland. Strict agreement rule, `SNOW = 7`. "Built-up" is not only buildings (roads, paved surfaces), hence "Man-made" and not "bldg". Crop fields vs Grassland and Trees vs Grassland stay separate on purpose.
5. **Class balance**: realistic shares, mild correction (sqrt class weights, class-aware sampler, Lovász). Estimated train shares (7 classes, before the city-edge / water / glacier changes): Trees 32.8, Grassland 24.5, Crop fields 14.4, Bare 10.2, Snow 7.5, Man-made 6.2, Water 4.5 % (about 7:1). Equal shares rejected: the model would over-predict rare classes. No extra tree data (Trees already largest). Per-class precision added to watch over-prediction.
6. **Labels**: WorldCover on every pixel; agreement pixels loss weight 1, disagreement 0.5 (`DISAGREE_WEIGHT`); Lovász and main scores on agreement pixels, second score on all pixels.
7. **Input**: 9 order-independent seasonal statistics in every variant; variants `year_stats` (stats only), `month_year` (1 month + stats), `three_month_year` (3 months + stats); `MONTH_DROP` 0.3; no PCA, no terrain channels.
8. **Evaluation**: main score rural test mIoU; all 12 months per test patch, raw vs 3-month smoothed softmax; **flicker** (month-to-month class changes); row- and column-normalised confusion (recall, precision); 12-month maps of every mosaic test site. `BEST` = best val mIoU among monthly variants. mIoU is not comparable with the 1st iteration (7 classes, new test set).

### 14.4 Repository changes (not committed)

- `Prototype.ipynb` → `Notebooks/1st_iteration.ipynb` (git mv, unchanged). **The Kaggle GitHub sync pointed at `Prototype.ipynb`**: relink on Kaggle if it is still used. Run from `Notebooks/`, the 1st-iteration notebook writes to `Notebooks/Weights/` (ignored by git).
- New `Notebooks/2nd_iteration.ipynb` (41 cells, pyflakes-clean, not run). Generated from a script in the session scratchpad; edit the notebook directly from now on. Finds `Dataset_2nd_iteration/` by the `gamma0` tag in the S1 GeoTIFFs; writes `Weights/<RUN_NAME>/` at the repo root (`RUN_NAME = "Iter2_Test_1"`). Label code: 0-6 class (agreement), 7-13 class + 7 (WorldCover only), -1 no label.
  - **One final weight** (user request): each run folder holds `final_model.pt` (the model to use, copy of `BEST`) + `model_card.json`; variant checkpoints and histories go to `variants/`. `DEV = True` (default) trains all 3 variants for 15 epochs only to choose; `DEV = False` trains only `FINAL_VARIANT` (default `month_year`). The report's variant comparison comes from the DEV run and the 1st-iteration ablation (Test_6).
- `data_pipeline/` builds `Dataset_2nd_iteration/` (env `LULC_DATASET` overrides; the 1st-iteration scripts are at commit `40e1280`):
  - `config.py`: `MOSAIC_SITES`, `is_mosaic`, `URBAN_KEPT`, `CITY_EDGE_SITES`, `DROPPED_SITES`, 7 `CLASSES`, new WorldCover / Dynamic World mapping, strict `consensus_labels`, `TRAIN_LABEL_BAND = "worldcover"`.
  - `download.py`: flattening, no median, mosaic check, months without scenes skipped, `--workers`, `--terrain` (optional), auth hint, `skipped_sites.json`, GeoTIFF tag `preprocessing`.
  - `make_patches.py`: filters on WorldCover, `MIN_VALID` 0.95, summary column `agree_pct`. `preview.py`: WorldCover label.
  - New `data_pipeline/README.md`: changes table and Datalab steps.
- **Prototype updated for the 2nd iteration, 1st-iteration weights still work** (switch: `"iteration": 2` in `model_card.json`):
  - New `data_pipeline/sentinel1.py`: the Sentinel-1 processing (month collection, GLO-30 slope/aspect, flattening, monthly composite), imported by both `download.py` and `prototype/gee.py`, so training and prototype cannot drift apart. `download.py` refactor verified: identical Earth Engine graph, 0 code difference.
  - `gee.py`: `s1_stack(region, months, iteration)` (2: sentinel1.py; 1: sigma0 + 3×3 median), `ESRI_TO_CLASS` per iteration (2: rangeland and flooded vegetation → Grassland, snow → 7).
  - `pipeline.py`: 2nd-iteration variant from the card (`n_months`, `static_channels`, class count checked against the weights), 9 seasonal statistics, `predict_probs` + 3-month smoothing (first / last month: one neighbour; the notebook's test wrapped the year).
  - `app.py`: `final_model.pt` listed first, local tiles from `Dataset_2nd_iteration/` (WorldCover band as reference), class names from the card, light colours drawn grey in charts.
  - Proof: stub test (no torch locally): statistics identical to the notebook, input channels 9/12/18 for n_months 0/1/3, smoothing runs; live Earth Engine at `black_forest_villages` 2021: prototype stack vs `download.py` 0 code difference (iteration 2) and vs commit `40e1280` 0 difference (iteration 1). Not yet run end to end with real weights.
- `.gitignore`: `Dataset_2nd_iteration/`.

### 14.6 Dataset build on Datalab and data Q&A (2026-10-31)

Full write-up: report section 9.

- **Datalab environment:** the root `.venv` is broken (symlink to a Python from an older image; `sudo` does not help). Use `uv venv .venv-data && source .venv-data/bin/activate && uv pip install -r requirements.txt`. `.venv-data/` is git-ignored.
- **Earth Engine project:** `EE_PROJECT` was `datalab-512920` (Datalab's project; noncommercial access expired). Use the user's own registered project (the `Project_ID` in the local `.env`); the user added `export EE_PROJECT=...` to `~/.bashrc` on Datalab. The Earth Engine login itself stayed valid (`~/.config/earthengine/credentials`; `git reset` does not touch it). `download.py` `initialize()` now stops at once with `PROJECT_HINT` for expired / unregistered projects instead of retrying.
- **Progress bar:** `download.py` shows a tqdm bar (sites done / to go, elapsed, ETA) and prints how many sites are already downloaded; SKIPPED / FAILED lines via `tqdm.write`. Restart is safe (`.tmp.tif` + rename; finished files skipped; missing `_stats.json` → `--relabel`).
- **Result:** 177 of 178 sites; `java_rice_hills` skipped (June 2021: 0 Sentinel-1 IW scenes). 2-site test: agreement 88.0 % (black_forest_villages), 61.9 % (krakow_edge). `Dataset_2nd_iteration/` = 3.7 GB. `make_patches.py` and `preview.py` done on Datalab; class distribution not yet reviewed.
- **Exam compliance confirmed:** Sentinel-1 is the only model input; Earth Engine is only the data platform; WorldCover / Dynamic World are labels (not input); the GLO-30 DEM is preprocessing (terrain correction, listed in the brief); Esri is the independent sanity check. Google Earth screenshots are not evidence. In the report, describe the monthly mean of 15-22 scenes as the speckle filter (temporal multilooking).
- **User's understanding of the previews (explained):** panel 1 = radar January, panel 2 = radar July (both input), panel 3 = WorldCover translated into our 7 classes (training target, not a model result). "ignore" = no label (mostly open sea); disagreement pixels are no longer ignored. Patches = 256 × 256 px windows (input size + quality filter: ≤ 60 % unlabelled, ≥ 95 % radar every month), stored as an index.
- **Roads:** the user saw roads in the radar that WorldCover misses. Rural roads (3-8 m) are sub-pixel at 10 m and below the ~20 m radar resolution. Decision: accept and document (critical discussion); OpenStreetMap roads and a road class rejected.
- **Manual label editing:** rejected (scale, inconsistency, reproducibility). Only rule-based fixes for systematic errors after the first results; optional small hand-checked gold test set for the report.
- **Prototype:** updated for the 2nd iteration (14.4), shared `data_pipeline/sentinel1.py`, verified 0 difference against the dataset at `black_forest_villages`.
- **Commits:** the user commits and pushes. On Datalab, after each push: `git fetch && git reset --hard origin/main` (data and `.venv-data/` are untracked and survive).

### 14.5 Next steps

1. Done: dataset built on Datalab (14.6). Still to do: print and review the class distribution (`patches/class_distribution.json`), look at the previews of steep tiles and `black_forest_villages`.
2. Run `Notebooks/2nd_iteration.ipynb`, DEV first. Read rural mIoU, flicker raw vs smoothed, precision / recall per class, Man-made IoU on the 4 test cities (below about 0.85: add city-edge sites).
3. Prototype: done (14.4); run it with the first `final_model.pt` + `model_card.json`.
4. Repeat the Black Forest test with the new model.
