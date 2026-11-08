# Sentinel-1 LULC prototype (Streamlit)

Pick a place (search by name, preset, coordinates or a click on the map) and a start month. The app downloads 12 monthly Sentinel-1 composites, runs the final U-Net (`Weights/Iter2_Test_1_Full`, loaded by default), and shows everything on one screen:

- a **live map** (Satellite or Streets). Satellite shows the newest photo published before the end of the analysis year (Esri World Imagery Wayback), so it follows the Year box; the caption gives the date the photo was taken. A switch under the map (off by default) shows a cloud-free Sentinel-2 photo of the selected month inside the box, so the month slider and **Play** step through the photos together with the maps; cloudy parts let the older photo show through. On the map, the segmentation of the selected month is drawn inside the bounding box of the analysed area; a layer menu switches between the model, the reference maps and none;
- the **Sentinel-1 radar input** of the same month beside the map, with the number of scenes;
- **one month slider with a play button** that drives the map and the radar together;
- **one chart beside the map** with a switch: area % per class, change vs last month, or change since the first month. Change is shown as paired bars per class (faded = earlier month, solid = selected month) labelled with the change in percentage points. The bars use the map colours, so they are also the legend; classes absent from the area are hidden. Below it, the **radar input** of the selected month;
- **all 12 months as thumbnails below the map** (model maps, Sentinel-2 photos or radar), the shown month framed in yellow;
- **Map options** (background Satellite/Streets, layer opacity) in a small menu; the monthly Sentinel-2 photo inside the box is a switch under the map, off by default. Zoom goes to level 20: beyond about zoom 17 the satellite photo is enlarged (Esri has no deeper rural tiles), and the 10 m map pixels are shown sharp, not blurred;
- an **analysis section**: summary cards (largest class, largest change, flicker, scenes per month, agreement with the Esri map vs the majority baseline), the class shares of all 12 months as stacked bars, the total and month-to-month change, the reference check, downloads (PNG, GeoTIFF, CSV, run summary) and notes on how to read the results.

Classes of the final model (2nd iteration): Man-made, Crop fields, Trees, Grassland, Bare ground, Water, Snow & ice. Other runs (3rd-iteration `best_model.pt`, 1st-iteration `<variant>_best.pt`) can be chosen under **Advanced** in the sidebar.

## Setup

Python 3.11-3.13.

```bash
cd prototype
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA build of PyTorch first (see pytorch.org), for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

The CPU also works. A 5 x 5 km area takes about 1-2 minutes for all 12 months.

### Google Earth Engine (required for any location)

1. A Google account with access to Earth Engine, and a Google Cloud project registered for Earth Engine (https://earthengine.google.com, then "Get started"). Non-commercial use is free.
2. Log in once on this machine:
   ```bash
   earthengine authenticate
   ```
3. Put the project ID in a `.env` file at the repository root (git-ignored, so the ID stays private): copy `.env.example` to `.env` and fill in `Project_ID = <your-project-id>`. Alternative: the environment variable `EE_PROJECT` (wins over `.env`). The app has no project field: the ID stays on the server.
4. Check that everything works:
   ```bash
   python check_auth.py                 # or: python check_auth.py --project <id>
   ```
   It checks the credentials file, the login and project, and read access to Sentinel-1 and the Esri map, and prints the fix for the first step that fails (exit code 0 = ready).

No other accounts or API keys are needed.

## Run

```bash
cd prototype
streamlit run app.py
```

Streamlit opens http://localhost:8501. Run it from the `prototype/` folder so that `.streamlit/config.toml` applies; it raises the upload limit to 500 MB, because checkpoints are about 98 MB.

### Run on Datalab or another remote machine

The app runs on the server; `launch.py` makes it reachable from your browser:

```bash
cd prototype
pip install -r requirements.txt
earthengine authenticate --auth_mode=notebook   # headless login: open the printed link, paste the code back
python launch.py
```

- `--tunnel auto` (default): a link through JupyterHub's `jupyter-server-proxy` if it is installed (protected by the hub login), otherwise a Cloudflare quick tunnel.
- `--tunnel cloudflare`: public `https://<random>.trycloudflare.com` link. No account is needed; `cloudflared` is downloaded once to `~/.cache/cloudflared`. The printed link contains a random access key (`?key=...`); the app refuses requests without it. Anyone with the full link can use the app and your Earth Engine quota, so do not share it.
- `--tunnel jupyter`, `--tunnel none`, `--port 8501` are also available.

From a notebook cell: `!cd prototype && python launch.py`. Interrupting the cell stops the app.

The Earth Engine project ID is never sent to the browser: the app reads `EE_PROJECT` or `Project_ID` from `.env` on the server and hides it in error messages.

## How to use

1. **Place:** search a town, region or address (OpenStreetMap Nominatim, press Enter; pick another match from the list if the first is wrong), choose a preset (rural test sites never used in training, Thane, Berlin), type latitude and longitude, or click the map. The search runs on the server and needs internet access there. The dashed yellow box shows the area of the next run.
2. **Year, start month, area** (2.56, 5.12 or 10.24 km square), then **Run**. A status box shows the steps (download, prediction per month, statistics) and the time taken.
3. **Explore:** move the month slider or press **Play**; the map overlay, the radar image and the bar chart follow. The model runs only once per Run; the slider never reruns it. Use the opacity slider to compare the segmentation with the satellite image, and the overlay switch to show the Esri reference map.
4. **Analysis** (below the map): summary cards, then the tabs *Year overview*, *Change*, *Reference check*, *Downloads and run log*, *How to read the results*.

**Advanced** (sidebar):
- **Model:** the final model (default), another run folder in `../Weights/` with `model_card.json`, or uploaded files (weights + the `model_card.json` of the same run).
  - **2nd/3rd iteration** (`"iteration": 2` in the card): the card defines the inputs, classes, encoder and smoothing; the app checks the input and class counts.
  - **1st iteration** (cards without `iteration`): the variant follows from the first layer's channel count: 3 `single_month`, 9 `three_month`, 14 `three_month_stats`, 15 `three_month_pca`, 20 `three_month_pca_stats`, 23 `three_month_pca_stats_terrain`.
- **Data source:** Google Earth Engine (any place), or a local dataset tile (`../Dataset_2nd_iteration/s1/` or `../Dataset/s1/`, year 2021, no Earth Engine needed; its reference is the training label, so **not** an independent check).
- **Patch overlap:** the area is predicted in 256 x 256 px windows. More overlap gives smoother maps but takes longer.

## How it works

```
lat/lon, start month
   │
   ▼  gee.py: Earth Engine, the same processing as the model's training data
Sentinel-1 GRD (IW, VV+VH) ─► per month, 2nd iteration: data_pipeline/sentinel1.py itself (gamma0 terrain
   │                          flattening, layover / shadow / incidence 31-45° masked, mean in linear power, dB)
   │                          1st iteration: incidence 31-45°, mean in linear power, dB, 3x3 median
   │                          uint8 codes (dB = code x 0.2 - 35); 14 months = 2 before the start + 12
   ▼  pipeline.py: features as in the model's training notebook
monthly channels: VV, VH, VV-VH of the target month and the n_months - 1 months before it, standardised
static channels from the 12 analysis months:
   2nd iteration: 9 seasonal statistics
   1st iteration (if the variant uses them): PCA (6), seasonal statistics (5), terrain (3)
   │
   ▼  model.py: U-Net (encoder from the model card, default ResNet-34), sliding window 256 px, overlapping softmax summed
   │  2nd iteration, if the card says so: probabilities averaged over months t-1, t, t+1
12 class maps (one per month)
   │
   ▼  pipeline.py
class % per month, month-to-month change (pp), total change first → last month (pp)
annual mode vs Esri 10 m Annual LULC (independent reference)
```

- **Class percentages:** the share of valid pixels per class in each month. Pixels without radar data in all 12 months are excluded.
- **Change:** `pct[month] - pct[previous month]` and `pct[last] - pct[first]`, in percentage points.
- **Sanity check:**
  - The most frequent class per pixel over the 12 months is compared with Esri's annual map for the year that covers most of the analysis months (Esri exists for 2017-2025; other years use the nearest one).
  - Esri classes are mapped onto ours: 2nd iteration rangeland and flooded vegetation → Grassland, snow/ice → Snow & ice; 1st iteration rangeland → Shrub & grass, flooded vegetation → Wetland; clouds → ignored.
  - Esri is not one of the training label sources (ESA WorldCover and Dynamic World), so it is an independent check.

## Files

| File | Content |
|---|---|
| `app.py` | Streamlit interface: place and time controls, live map + radar viewer with month slider, analysis, downloads |
| `mapview.py` | puts the UTM prediction grid on the web map (resampled to Web Mercator, so it lines up with the basemap), the area box, GeoTIFF export, the Esri Wayback release and photo date for the analysis period |
| `gee.py` | Earth Engine: grid, monthly Sentinel-1 composites (2nd iteration via `../data_pipeline/sentinel1.py`), terrain, Esri reference, chunked download |
| `pipeline.py` | model loading (variant from the model card or the channel count), input features, sliding-window inference, temporal smoothing, percentages and changes |
| `model.py` | the U-Net with an exchangeable encoder (ResNet-34, ResNet-50, VGG-16-BN, DenseNet-121, EfficientNet-B0), identical to the 3rd-iteration notebooks; cards without `encoder` mean ResNet-34 |
| `launch.py` | starts the app on a remote machine, with a JupyterHub proxy link or a Cloudflare tunnel |
| `check_auth.py` | checks the Earth Engine login, project and data access |
| `runlog.py` | saves every run to `runs/<time>_<place>/` (git-ignored): `run.json` (settings, model, months, scene counts, sanity-check score), `arrays.npz` (radar codes, class maps, valid mask, reference), the CSV tables and PNGs of the monthly maps, radar input, class shares, change and sanity check |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | upload limit |

## Limitations

- **Basemap tiles** (Esri World Imagery and its Wayback archive, OpenStreetMap) are loaded by the browser, so the viewer's computer needs internet access. The Wayback version is the last one published before the analysis period ended, but its photo can be years older (the caption shows the capture date). The monthly Sentinel-2 photos (one cloud-free median per analysis month, about 15 s extra per run) are for display only and are not a model input.
- **Seasonal effects look like change.** The training label is annual, so snow, wet soil, harvest or flooding change the radar signal and can flip a pixel's class for some months. The month-to-month change mixes real change with these effects.
- **Errors inflate the change.** If 5 % of pixels are misclassified differently in two months, the change table shows several pp of change even when nothing changed.
- **Unfamiliar regions:** the models were trained on sites of 2021 (1st iteration: 134 training sites, city-heavy; 2nd iteration: about 130 training sites, mostly rural). Performance drops on landscapes unlike the training data. On the 1st-iteration test sites the weakest results were boreal / snow areas, shrubland and wetland.
- **2nd-iteration smoothing:** with `temporal_smoothing`, a one-month change is smoothed away; changes that last 2 months or more stay. The first and last month are averaged with one neighbour only.
- **SAR confusion:** smooth surfaces (water, snow, sand, wet soil) can all be dark. Trees and cropland both scatter in the vegetation volume.
- **Missing months:** if Earth Engine has no scene for a month, that input is treated as missing (the app warns). The prediction then relies on the other months and the yearly channels.
- **Edges:** months before 2015 have no Sentinel-1 IW coverage for most places; Sentinel-1B failed in Dec 2021, so 2022-2024 have fewer scenes; Sentinel-1C adds scenes again from 2025.
