# Sentinel-1 LULC prototype (Streamlit)

Enter a latitude/longitude and a start month. The app downloads 12 monthly Sentinel-1 composites, runs the trained U-Net, and shows:

- **12 monthly land use / land cover maps** with the classes of the model in use. 2nd iteration (7 classes): Man-made, Crop fields, Trees, Grassland, Bare ground, Water, Snow & ice. 1st iteration (8 classes): Built-up, Cropland, Trees, Shrub & grass, Bare / sparse, Water, Wetland & mangrove, Snow & ice;
- the **share of each class per month** (% of the area);
- the **month-to-month change** and the **total change** from the first to the last month, in percentage points;
- a **sanity check** against the independent Esri 10 m Annual LULC map;
- the Sentinel-1 input as a false-colour image.

You choose the model yourself: `final_model.pt` from a 2nd-iteration run folder (or a 1st-iteration `<variant>_best.pt`) together with that run's `model_card.json`.

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
3. Enter the project ID in the sidebar, or leave it blank to use the environment variable `EE_PROJECT` or `Project_ID = ...` in the repository's `.env` file (git-ignored).

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

The Earth Engine project ID is never sent to the browser: leave the field blank to use `EE_PROJECT` or `Project_ID` from `.env` on the server.

## How to use

1. **Model** (sidebar):
   - *Upload files*: a weights file plus the `model_card.json` from the **same run folder**. 2nd iteration: `final_model.pt`; 1st iteration: for example `three_month_stats_best.pt`.
   - *Local Weights folder*: pick a run folder in `../Weights/` that contains `model_card.json`, then a `.pt` file (`final_model.pt` is listed first).

   **2nd iteration** (`"iteration": 2` in the model card): the card says everything: number of monthly inputs (`n_months` 0, 1 or 3), the 9 seasonal statistics, the 7 classes, and whether the monthly maps are smoothed over 3 months (`temporal_smoothing`). Weights and card must come from the same run; the app checks the input and class counts.

   **1st iteration** (cards without `iteration`): the variant is detected from the first layer's channel count:

   | Channels | Variant |
   |---|---|
   | 3 | `single_month` |
   | 9 | `three_month` |
   | 14 | `three_month_stats` |
   | 15 | `three_month_pca` |
   | 20 | `three_month_pca_stats` |
   | 23 | `three_month_pca_stats_terrain` |

   Every 1st-iteration variant of a run works with that run's model card. The card stores the normalisation and the PCA, statistics and terrain parameters.
2. **Location and time:**
   - *Google Earth Engine*: latitude, longitude, start year and month, and area (2.56, 5.12 or 10.24 km square).
   - *Local dataset tile (offline test)*: a tile from the dataset of the model's iteration (`../Dataset_2nd_iteration/s1/` or `../Dataset/s1/`, year 2021). No Earth Engine is needed. The reference shown is the training label, so it is **not** an independent check.
3. **Patch overlap:** the area is predicted in 256 x 256 px windows. More overlap gives smoother maps but takes longer.
4. Press **Run**.

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
   ▼  model.py: U-Net / ResNet-34, sliding window 256 px, overlapping softmax summed
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
  - The most frequent class per pixel over the 12 months is compared with Esri's annual map for the year that covers most of the analysis months (Esri exists for 2017-2024; other years use the nearest one).
  - Esri classes are mapped onto ours: 2nd iteration rangeland and flooded vegetation → Grassland, snow/ice → Snow & ice; 1st iteration rangeland → Shrub & grass, flooded vegetation → Wetland; clouds → ignored.
  - Esri is not one of the training label sources (ESA WorldCover and Dynamic World), so it is an independent check.

## Files

| File | Content |
|---|---|
| `app.py` | Streamlit interface, caching, plots, downloads |
| `gee.py` | Earth Engine: grid, monthly Sentinel-1 composites (2nd iteration via `../data_pipeline/sentinel1.py`), terrain, Esri reference, chunked download |
| `pipeline.py` | model loading (variant from the model card or the channel count), input features, sliding-window inference, temporal smoothing, percentages and changes |
| `model.py` | the U-Net / ResNet-34 architecture (identical to the notebook) |
| `launch.py` | starts the app on a remote machine, with a JupyterHub proxy link or a Cloudflare tunnel |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | upload limit |

## Limitations

- **Seasonal effects look like change.** The training label is annual, so snow, wet soil, harvest or flooding change the radar signal and can flip a pixel's class for some months. The month-to-month change mixes real change with these effects.
- **Errors inflate the change.** If 5 % of pixels are misclassified differently in two months, the change table shows several pp of change even when nothing changed.
- **Unfamiliar regions:** the models were trained on sites of 2021 (1st iteration: 134 training sites, city-heavy; 2nd iteration: about 130 training sites, mostly rural). Performance drops on landscapes unlike the training data. On the 1st-iteration test sites the weakest results were boreal / snow areas, shrubland and wetland.
- **2nd-iteration smoothing:** with `temporal_smoothing`, a one-month change is smoothed away; changes that last 2 months or more stay. The first and last month are averaged with one neighbour only.
- **SAR confusion:** smooth surfaces (water, snow, sand, wet soil) can all be dark. Trees and cropland both scatter in the vegetation volume.
- **Missing months:** if Earth Engine has no scene for a month, that input is treated as missing (the app warns). The prediction then relies on the other months and the yearly channels.
- **Edges:** months before 2015 have no Sentinel-1 IW coverage for most places; Sentinel-1B failed in Dec 2021, so 2022-2024 have fewer scenes.
