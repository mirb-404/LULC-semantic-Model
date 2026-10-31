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
   - `[k/178 sites]` lines show progress. `SKIPPED` sites (no mosaic, or a month without Sentinel-1 scenes) are listed in `Dataset_2nd_iteration/skipped_sites.json`; `FAILED` sites are retried on the next run.
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
| `download.py` | Earth Engine download: terrain flattening, monthly composites, labels, mosaic check, parallel sites, resume |
| `make_patches.py` | patch index per split, normalisation stats, class distribution, per-site summary |
| `preview.py` | quick-look PNGs |
