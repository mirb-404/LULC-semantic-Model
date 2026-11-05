# Dataset pipeline (2nd iteration)

Builds `Dataset_2nd_iteration/` from Google Earth Engine: Sentinel-1 monthly composites (2021), WorldCover / Dynamic World labels and the patch index used by `Notebooks/2nd_iteration.ipynb`.

The 1st iteration (`Dataset/`, `Notebooks/1st_iteration.ipynb`) was built by the version of these scripts at commit `40e1280`.

## What changed from the 1st iteration

| | 1st iteration | 2nd iteration | Why |
|---|---|---|---|
| Sites | 176, 79 urban (city centres, 45 %) | 178: 22 towns + 10 city edges, 92 single-class rural (glacier sites halved), 54 mixed rural **mosaics** (5 with lakes / rivers) | the product targets rural land; city edges keep towns working |
| Classes | 8 (Built-up, Cropland, Trees, Shrub & grass, Bare / sparse, Water, Wetland & mangrove, Snow & ice) | 7: Man-made, Crop fields, Trees, Grassland, Bare ground, Water, Snow & ice (mangrove → Trees, marsh → Grassland) | wetland labels are only 30-45 % correct and caused winter false alarms |
| Mosaic check | none | WorldCover largest class ≤ 80 %, second ≥ 10 %, checked before download | each tile teaches boundaries and transitions (field, meadow, forest edge, hamlet) |
| Radiometry | sigma0 (geometric terrain correction only) | **gamma0, terrain-flattened** (Vollrath et al. 2020 volume model), layover / shadow masked | slopes facing the sensor looked like buildings, slopes facing away like water |
| Speckle filter | 3 × 3 focal median | none (monthly mean only) | the median blurred field edges, roads and streams |
| Training label | consensus (pixels where both maps agree, wetland exception) | WorldCover everywhere; consensus band (strict agreement) marks agreement | the consensus mask removed most boundary pixels, so the model learned blobs |
| Patch filter | ≥ 99 % valid radar every month | ≥ 95 % | flattening masks a few layover / shadow pixels in steep tiles |
| Terrain layers | always downloaded | only with `--terrain` | the brief is Sentinel-1 only; the DEM is still used on the server for flattening |

Flattening check (July 2021, one scene per orbit, 30 m): the correlation between VV and the terrain slope towards the sensor fell from 0.74-0.82 to -0.14-0.05 in `nepal_midhills`, `andes_sacred_valley` and the 1st-iteration `black_forest` tile; the spatial standard deviation of VV fell by 30-40 %; 97-99.8 % of pixels stay valid.

## 4th iteration: later years and new sites (2026-11-05)

Added to the same `Dataset_2nd_iteration/` folder; the 2021 files and the val/test sites are unchanged.

- **17 new training sites** (`config.ITER4_SITES`, `iteration` 4 in the patch index): bare ground outside deserts (eroded laterite, salt pan, mines, degraded highlands), smallholder tropical fields (terraces, paddies, hill plots), bare rock, and 4 pure-class tiles (one class >= 60 %: salt pans Etosha and Uyuni, irrigated crops Gezira and central Thailand) that show each class alone. Each has a `need` rule (minimum WorldCover share of a class, checked before downloading); tiles that fail are listed in `skipped_sites.json` with their class shares, so the coordinates can be moved. All are at least `MIN_TEST_DISTANCE_KM` (30 km) from every val/test site; `download.py` prints a warning for any train site closer than that.
- **Radar of 2022, 2023 and 2024** (`config.EXTRA_YEARS`) for every training site except towns and glaciers (`EXTRA_YEARS_SKIP_FOCUS`: stable land cover, not the product goal): `s1_<year>/<site>.tif`, `labels_<year>/<site>.tif` (WorldCover 2021 as the label, Dynamic World of that year for the agreement band, so pixels that changed after 2021 get the lower disagreement weight). Up to `MAX_EMPTY_MONTHS_EXTRA` (2) months without scenes are accepted for training tiles of any year (Sentinel-1B was lost in December 2021, and parts of Africa have sparse coverage); val and test tiles must have every month.
- `make_patches.py` writes `patches/train_extra_patches.csv` (same windows as 2021, checked again on that year's radar) and an `iteration` column in every patch index. Normalisation stays that of the 2021 training patches.
- Size: about 11-12 GB in total (2021: 3.7 GB). Time: about 380 site-years; roughly 2.5-3.5 h with `--workers 8` (32 requests at a time; 2021 took about 50 s per site with 4 workers). If the log shows many `retry after error` lines with "Too many requests", restart with `--workers 6`.
- Notebook: `Notebooks/4th_iteration_multiyear.ipynb`.

**One command for everything** (download with automatic retries, patch index, `dataset_manifest.json`, and two zip archives next to the dataset folder: `Dataset_2nd_iteration_2021.zip` for the 2021 part used by `Iter2_Test_1_Full`, `Dataset_2nd_iteration.zip` with all years):
```bash
nohup python data_pipeline/build_dataset.py --workers 8 > build.log 2>&1 &
tail -f build.log
```
`--steps patches manifest zip` repeats only the offline steps; `--steps ... previews` adds the quick-look PNGs; `--years` and `--sites` limit the download.

The single scripts (after steps 1-3 below):
```bash
nohup python data_pipeline/download.py --workers 8 > download_iter4.log 2>&1 &   # 2021 new sites first, then 2022-2024
tail -f download_iter4.log
python data_pipeline/make_patches.py
```
`--years 2022` limits the run to one year; the run does 2021 first, then one year after the other, and can be restarted at any time.

## Run on Datalab (or any remote Linux machine)

Run the steps in this order, from the repository root.

1. **Environment** (no `sudo`):
   ```bash
   uv venv .venv-data && source .venv-data/bin/activate
   uv pip install -r requirements.txt
   ```
2. **Earth Engine login** (once per machine; there is no browser on the server):
   ```bash
   earthengine authenticate --auth_mode=notebook
   ```
   Open the printed link, choose a Google Cloud project that can host the login client, **do not tick "read-only scopes"**, and paste the code back.
3. **Project**: the Cloud project registered for Earth Engine (the Datalab-provided project is not registered):
   ```bash
   export EE_PROJECT=<your-project-id>
   ```
   Alternatively put `Project_ID = <id>` in `.env` at the repository root, or pass `--project`.
4. **Download** (about 2-4 GB; runs in the background, survives a closed browser tab):
   ```bash
   nohup python data_pipeline/download.py --workers 4 > download.log 2>&1 &
   tail -f download.log
   ```
   - Rerun the same command after an interruption: existing files are skipped.
   - A progress bar shows sites done, elapsed time and the estimated time left (`sites: 40%| 70/176 [1:02:11<1:34:30, 53.5s/site]`); already downloaded sites are not counted. `SKIPPED` sites (no mosaic, or a month without Sentinel-1 scenes) are listed in `Dataset_2nd_iteration/skipped_sites.json`; `FAILED` sites are retried on the next run.
   - More `--workers` is faster, but Earth Engine limits concurrent requests per project; 4-6 is safe.
5. **Patch index and statistics**:
   ```bash
   python data_pipeline/make_patches.py
   ```
6. **Previews** (optional quick-look PNGs, January / July / label):
   ```bash
   python data_pipeline/preview.py
   ```
   Check a few steep tiles (`nepal_midhills`, `andes_sacred_valley`, `black_forest_villages`): hillsides should no longer show a bright / dark pattern.

Other useful options:
- `--sites black_forest_villages harz_mosaic` downloads only these sites (for a quick test).
- `--relabel` recomputes the consensus band offline after a change to `config.consensus_labels`.
- `LULC_DATASET=Dataset python data_pipeline/make_patches.py` points the scripts at another dataset folder.

## Files

| File | Content |
|---|---|
| `config.py` | sites (`URBAN_SITES` filtered by `URBAN_KEPT`, `CITY_EDGE_SITES`, `RURAL_SITES` minus `DROPPED_SITES`, `MOSAIC_SITES`), 7 classes, uint8 encoding, consensus rule, mosaic rule, `TRAIN_LABEL_BAND` |
| `sentinel1.py` | Sentinel-1 processing (flattening, masks, monthly composite), shared with the prototype |
| `download.py` | Earth Engine download: monthly composites via `sentinel1.py`, labels, mosaic check, parallel sites, progress bar, resume |
| `make_patches.py` | patch index per split, normalisation stats, class distribution, per-site summary |
| `build_dataset.py` | runs everything in order: download (with retries), patches, optional previews, `dataset_manifest.json`, zip archives |
| `preview.py` | quick-look PNGs |
