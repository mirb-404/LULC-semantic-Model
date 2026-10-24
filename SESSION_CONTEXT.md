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
