# Global Sentinel-1 LULC Dataset (2021)

This is a compact, global dataset for **land use / land cover (LULC) semantic segmentation from Sentinel-1 SAR only**. Each sample is a 10 m pixel grid with 12 monthly Sentinel-1 composites (VV and VH backscatter) and an 8-class label map. The labels are derived from the per-pixel consensus of two independent global land-cover products, ESA WorldCover 2021 and Google Dynamic World.

All data is retrieved from Google Earth Engine by the scripts in `data_pipeline/`. Only this README is tracked in git; the data files are produced locally (about 3.1 GB).

## 1. Quick facts

| Item | Value |
|---|---|
| Input | Sentinel-1 GRD, IW mode, VV + VH, 12 monthly composites (Jan–Dec 2021) |
| Labels | Consensus of ESA WorldCover v200 (2021) and Dynamic World V1 (2021 mode), 8 classes + ignore |
| Pixel size / CRS | 10 m, local UTM zone per site |
| Sites | 151 (77 urban, 74 rural), all continents except Antarctica |
| Site size | 10.24 × 10.24 km (1024 × 1024 px) |
| Patches | 256 × 256 px; 1670 train / 278 val / 296 test (2244 of 2416 possible) |
| Split | by site (no site appears in two splits), every region and class focus in each split |
| Sentinel-1 scenes used | 15,555 (median 61 per site for the year) |
| Size on disk | 2.8 GB S1 tiles, 19 MB labels, 289 MB previews, < 1 MB index/statistics |
| Independent check map | Esri 10 m LULC 2021, **not used** anywhere in this dataset (reserved for the prototype sanity check) |

## 2. Folder layout and file formats

```
Dataset/
  s1/
    <site>.tif                   24 bands uint8, 1024 x 1024, deflate-compressed GeoTIFF
                                 band order: VV_01, VH_01, VV_02, VH_02, ..., VV_12, VH_12
    <site>_scene_counts.json     number of S1 scenes averaged into each month's composite
  labels/
    <site>.tif                   3 bands uint8 on exactly the same grid as s1/<site>.tif
                                   band 1 "consensus"    training label (0 = ignore)
                                   band 2 "worldcover"   WorldCover mapped to the 8 classes
                                   band 3 "dynamicworld" Dynamic World 2021 mode mapped to the 8 classes
    <site>_stats.json            kept_pct (share of labelled pixels kept by the consensus) and
                                 pixel counts per class for each band
  patches/
    train_patches.csv            index, site, region, focus, row, col  (row/col = top-left pixel in the site tile)
    val_patches.csv
    test_patches.csv
    norm_stats.json              per-polarisation mean/std in dB, train split only
    class_distribution.json      pixel counts and shares per class and split
    site_summary.csv             per site: split, region, focus, patches kept, labels kept %, class shares %
  previews/
    <site>.png                   S1 false colour (January, July) next to the training labels
  download_log.txt               log of the last download run
```

### Backscatter encoding

To keep the dataset small, backscatter is stored as **uint8** instead of float:

```
code = clip(round((dB - DB_MIN) / DB_STEP), 1, 255)      DB_MIN = -35.0, DB_STEP = 0.2
dB   = code * 0.2 - 35.0                                  code 0 = no data
```

- The codes 1–255 cover −34.8 dB to +16.0 dB. Values outside this range are clipped: very calm water below, and strong urban corner reflectors above.
- The quantisation error is at most ±0.1 dB. The residual speckle in a monthly composite is about 0.5–1 dB, so the loss is negligible.
- Compared with int16, this halves the size: 19 MB per site instead of about 40 MB.

`config.decode_db()` converts codes back to float dB, with NaN for no data.

### Reading a patch

```python
import sys, csv
import numpy as np, rasterio
from rasterio.windows import Window
sys.path.insert(0, "data_pipeline")
import config

row = next(csv.DictReader(open("Dataset/patches/train_patches.csv")))
win = Window(int(row["col"]), int(row["row"]), config.PATCH, config.PATCH)
with rasterio.open(f"Dataset/s1/{row['site']}.tif") as src:
    x = config.decode_db(src.read(window=win)).reshape(12, 2, config.PATCH, config.PATCH)  # months, (VV, VH), H, W
with rasterio.open(f"Dataset/labels/{row['site']}.tif") as src:
    y = src.read(1, window=win)                                                          # H, W, 0 = ignore
```

Patches are stored as an **index** (site, row, col) into the tiles, not as copied arrays. This avoids storing every pixel twice; a full copy would have needed about 13 GB more. The tiles are internally tiled in 256 × 256 blocks, so reading one patch reads exactly one block per band.

## 3. Data sources

| Source | Earth Engine ID | Use | Licence |
|---|---|---|---|
| Sentinel-1 GRD (ESA Copernicus) | `COPERNICUS/S1_GRD` | model input | Copernicus free, full and open data policy |
| ESA WorldCover 10 m v200 (2021) | `ESA/WorldCover/v200` | label source 1 | CC BY 4.0 |
| Google Dynamic World V1 | `GOOGLE/DYNAMICWORLD/V1` | label source 2 | CC BY 4.0 |
| Esri 10 m Land Use/Land Cover | `projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS` | independent check (not in dataset) | CC BY 4.0 |

References:
- Zanaga, D. et al. (2022). *ESA WorldCover 10 m 2021 v200*.
- Brown, C. F. et al. (2022). *Dynamic World, Near real-time global 10 m land use land cover mapping*. Scientific Data 9, 251.

### Why the year 2021

- WorldCover v200 is a 2021 map, so the imagery and the second label source come from the same year. That avoids training on land-cover changes that happened between the image date and the label date.
- 2021 is the last full year with **both Sentinel-1A and Sentinel-1B**. Sentinel-1B failed on 23 Dec 2021. Two satellites mean a denser revisit and more scenes per monthly composite, which leaves less speckle.

## 4. Site selection

### Urban sites (77)

Each urban site is a 10.24 km square centred on a city centre. Together they cover 6 regions: Europe, North America, South America, Africa, Asia and Oceania. The cities differ in climate (tropical, arid, temperate, continental, polar), building style and terrain. Urban areas are required by the project: the prototype is demonstrated on an urban area.

### Rural sites (39)

A first pilot run on Berlin and Lagos showed that squares around city centres are about 85% built-up:

| Pilot (15 km tiles) | Built-up | Cropland | Trees | Shrub/grass | Bare | Water | Wetland | Snow |
|---|---|---|---|---|---|---|---|---|
| Berlin | 1.23 M | 21 k | 220 k | 43 k | 0.7 k | 31 k | 0 | 0 |
| Lagos | 1.30 M | 0.1 k | 38 k | 19 k | 1 k | 648 k | 14 k | 0 |

A model trained only on such tiles would learn to predict "built-up" almost everywhere. It would also never see cropland, bare ground, wetland or snow, but the prototype has to map these classes anywhere in the world. Therefore 39 rural sites were added, each centred on a landscape dominated by one under-represented class:

| Focus | Train | Val | Test |
|---|---|---|---|
| Cropland | Iowa corn belt, Punjab, Poltava (UA), Pampas, Nile delta, Mato Grosso soy, Mekong delta rice | Beauce (FR), Kansas centre pivots | North China Plain, Kano farmland |
| Trees | Amazon, Congo, Black Forest, Borneo | Ontario boreal | Siberian taiga |
| Bare / sparse | Algerian Sahara, Egyptian Western Desert, Badain Jaran, Namib | Atacama | Rub' al Khali |
| Shrub & grass | Mongolian steppe, Texas shrubland | Kazakh steppe | Serengeti |
| Wetland & mangrove | Sundarbans, Pantanal, Sudd, Inner Niger Delta | Okavango delta | Everglades |
| Snow & ice | Aletsch glacier, Columbia Icefield, Patagonian icefield, Vatnajökull | Svalbard glacier | Karakoram (Baltoro) |

### Extra rural sites (35, second round)

The first full training run (test mIoU 0.56) showed three weaknesses: trees were confused with cropland (the only test forest is boreal, but no boreal forest was in train), snow and shrub & grass had few training sites, and train mIoU (0.79) was far above val (0.47), so the model overfitted the training sites. 38 more rural sites were therefore added, chosen for **new biomes** rather than more of the same. All of them go to train, except one snow site for val (val had only 0.3% snow). The **test set is unchanged**, so scores before and after stay comparable.

| Focus | Train | Val |
|---|---|---|
| Trees | Swedish taiga, Alaska boreal, Québec boreal, Appalachians, Olympic rainforest, Sikhote-Alin, Gabon, Tasmania, Papua | |
| Snow & ice | Mont Blanc, Jostedalsbreen, Juneau Icefield, Tien Shan (Inylchek), Khumbu, Tasman glacier | Caucasus (Elbrus) |
| Shrub & grass | Nebraska Sandhills, Great Basin sagebrush, Patagonian steppe, Karoo, Tibetan grassland | |
| Cropland | Saskatchewan wheat, Central Valley (CA), Punjab (PK), Karawang rice, Heilongjiang maize, Australian wheatbelt, Hungarian plain | |
| Wetland & mangrove | Hudson Bay lowland, Danube delta, Louisiana marsh, Niger delta mangrove | |
| Bare / sparse | Taklamakan | |
| Water | Lake Balaton, Finnish lakeland | |

- **Replaced:** Vasyugan mire (57 N, 78 E) had a month without IW scenes and failed; it was replaced by the Danube delta.
- **Dropped:** Finnish boreal forest, Australian spinifex and the Simpson desert kept 0 patches, because WorldCover and Dynamic World agree on only 3–22% of their pixels. They were removed.
- Some sites came out differently from their focus: the Great Basin and Karoo are labelled mostly *bare*, not shrub. The label maps decide, and the focus is only the reason a site was chosen.

### Coverage check and replaced sites

Before downloading, the number of Sentinel-1 IW scenes per month was checked for every site. Three planned sites were replaced:

| Planned site | Problem | Replacement |
|---|---|---|
| Gobi desert (43 N, 105 E) | 0 IW scenes in all 12 months of 2021 | Badain Jaran desert (40 N, 102 E) |
| Greenland ice-sheet margin (67.15 N, 50 W) | 0 IW scenes (Greenland is acquired mainly in EW mode) | Columbia Icefield (52.2 N, 117.3 W) |
| Brisbane | one month of 2021 with 0 IW scenes | Adelaide |

Sites with thin coverage were kept, because every month still has at least one scene. Atlanta, Chicago, New York, Borneo, Ontario and Namib each have one month with a single scene. Single-scene months contain more speckle; see Limitations.

### Splits

- The data is split **by site**, so all patches of one site are in the same split. Patches from the same site are spatially autocorrelated, so a random patch-level split would leak information and overstate the test scores.
- Val and test contain sites from every region and every rural focus class. The test score therefore measures transfer to **unseen places**.

| Split | Sites | Patches | Urban patches | Rural patches |
|---|---|---|---|---|
| train | 112 | 1670 | 826 | 844 |
| val | 19 | 278 | 169 | 109 |
| test | 20 | 296 | 202 | 94 |

## 5. Sentinel-1 preprocessing

### Already applied by Earth Engine (`COPERNICUS/S1_GRD`)

Earth Engine processes every GRD scene with the ESA Sentinel-1 Toolbox:
1. Apply the orbit file (restituted or precise orbit metadata).
2. GRD border noise removal.
3. Thermal noise removal.
4. **Radiometric calibration** to backscatter coefficient sigma0.
5. **Terrain correction** (orthorectification) with SRTM 30 m, or ASTER GDEM above 60° latitude.
6. **Conversion to dB**: 10·log10(sigma0).

### Applied by this pipeline (`data_pipeline/download.py`)

1. **Scene selection:** instrument mode IW, only scenes with both VV and VH, and both ascending and descending passes, over the site rectangle in each calendar month of 2021.
2. **Edge masking:** pixels with an incidence angle outside 31°–45° are masked. This removes residual near/far-range border artefacts and the most extreme geometries.
3. **Monthly compositing in linear power:** each scene is converted from dB back to linear power, then all scenes of the month are averaged. Averaging is done in linear units because averaging dB values is biased low. The temporal mean is a multilook: with N scenes, the speckle variance drops by roughly a factor N.
4. **Back to dB, then spatial speckle filter:** a 3 × 3 focal median on the composite. It removes the remaining speckle outliers but keeps edges such as roads and shorelines. A larger or Lee-type filter was not used, to avoid blurring 10 m structures.
5. **Resampling and gridding:** composites are computed directly on a site-specific UTM grid. The grid is a square of 10.24 km centred on the site coordinate, with corners snapped to multiples of 10 m. This is the same grid that the labels use.
6. **Encoding:** uint8 as described in section 2.
7. **Download:** `ee.data.computePixels` in 512 × 512 px chunks, 4 in parallel, with retries for transient network or authentication errors. Each site is written to a temporary file first and renamed when complete, so an interrupted run leaves no partial tiles.

### Normalisation

`patches/norm_stats.json` holds the per-polarisation mean and standard deviation in dB over all valid pixels and all 12 months of the **train** patches:

| Band | Mean (dB) | Std (dB) |
|---|---|---|
| VV | −8.31 | 6.05 |
| VH | −15.66 | 5.95 |

The training code standardises inputs with these values: (dB − mean) / std. Only train statistics are used, so no information leaks from val or test.

## 6. Labels

### Class scheme

The 8 target classes are chosen to be **separable with C-band SAR**. Classes that the radar cannot distinguish reliably are merged.

| id | Class | WorldCover 2021 codes | Dynamic World labels | Why |
|---|---|---|---|---|
| 0 | ignore | (disagreement / no data) | – | excluded from the loss and the metrics |
| 1 | Built-up | 50 built-up | 6 built | strong double-bounce, high VV |
| 2 | Cropland | 40 cropland | 4 crops | strong seasonal backscatter change |
| 3 | Trees | 10 tree cover | 1 trees | high volume scattering (VH) all year |
| 4 | Shrub & grass | 20 shrubland, 30 grassland | 2 grass, 5 shrub_and_scrub | shrub and grass are nearly indistinguishable in C-band |
| 5 | Bare / sparse | 60 bare/sparse, 100 moss & lichen | 7 bare | low and stable backscatter; moss/lichen behave like sparse surfaces |
| 6 | Water | 80 permanent water | 0 water | specular reflection, very low backscatter |
| 7 | Wetland & mangrove | 90 herbaceous wetland, 95 mangroves | 3 flooded_vegetation | water under vegetation, double-bounce |
| 8 | Snow & ice | 70 snow and ice | 8 snow_and_ice | strong seasonal (wet / dry snow) signature |

### Consensus rule (`config.consensus_labels`)

Each global map alone has an overall accuracy of only roughly 70–80%, and the errors of the two maps are largely independent. Using only pixels where both agree gives much cleaner training labels than either map alone:

```
label = worldcover            where worldcover == dynamicworld
label = 7 (wetland)           where worldcover == 7 and dynamicworld in {trees, shrub & grass, water, wetland}
label = 0 (ignore)            everywhere else
```

**Wetland exception.** The strict rule removed almost all wetland. Dynamic World has no mangrove class: it labels mangroves as *trees* and most marshes as *grass/shrub* or *water*, while its *flooded_vegetation* class is very narrow. Examples from the first run:

| Site | WorldCover wetland px | Dynamic World wetland px | Labels kept, strict rule | Labels kept, with exception |
|---|---|---|---|---|
| Sundarbans | 745,239 | 43,044 | 33% | 100% |
| Sudd | 1,022,712 | 154,503 | 16% | 98.4% |
| Inner Niger Delta | 986,731 | 73,401 | 10% | 92.2% |

With the exception, wetland rose from 0.1% to 5.3% of the training labels. In wetland areas, WorldCover is therefore trusted whenever Dynamic World gives a compatible (vegetated or wet) class. All other classes use strict agreement.

**Dynamic World annual mode.** Dynamic World provides one label per Sentinel-2 scene. The per-pixel **mode** over all 2021 scenes gives one annual label, which can be compared with the annual WorldCover map. The mode suppresses cloud and shadow errors from single scenes.

### Matching labels to the imagery

- Both label products are requested with the same Earth Engine `computePixels` grid as the Sentinel-1 composite: same UTM CRS, same 10 m affine transform, same 1024 × 1024 size.
- They are resampled with **nearest neighbour**, the Earth Engine default, so class values are never averaged.
- Label pixels and image pixels therefore correspond one-to-one, without any later reprojection.
- Both label sources are already 10 m products, so no resolution change is involved.
- The raw mapped WorldCover and Dynamic World bands are kept in every label file. This allows the consensus rule to be changed offline later with `download.py --relabel`, without re-downloading.

On average **79.8%** of the pixels per site are kept as labels (min 24.1% at Kano farmland, max 100% at Sundarbans). Low values mark transitional landscapes where the two maps disagree, for example savanna versus cropland or sparse grass versus bare ground.

## 7. Patching

- Each 1024 × 1024 site tile is cut into a non-overlapping 4 × 4 grid of 256 × 256 patches, giving 16 candidates per site and 2416 in total.
- A patch is **kept** when both of these hold:
  - at most **60%** of its pixels are *ignore*: ignored pixels do not contribute to the loss, but patches that are mostly unlabelled carry little signal;
  - in **every month**, at least **99%** of its pixels have valid VV and VH.
- **2244 patches kept (92.9%).** Sites with few patches: Khumbu 1, Australian wheatbelt 1, Gabon 2, Great Basin 2, Swedish taiga 4, Kano farmland 1, Mongolian steppe 5, Svalbard 8, Anchorage 10, Khulna 11 (low label agreement).
- 256 px = 2.56 km gives enough spatial context for urban texture and field patterns, and fits a U-Net-style encoder (divisible by 32) on a 4 GB GPU.
- Each patch keeps **all 12 months**. Every month shares the same annual label, which lets the training loader:
  - sample a random month per patch: temporal augmentation that teaches the model to classify any single month, as the prototype requires;
  - stack several months if a multi-temporal model variant is tested.
- Augmentation (flips, 90° rotations, month sampling) is applied **at training time**. Nothing is augmented on disk.

## 8. Class distribution

Share of labelled pixels per split (from `patches/class_distribution.json`):

| Class | Train | Val | Test |
|---|---|---|---|
| Built-up | 34.2% | 40.2% | 54.7% |
| Cropland | 10.9% | 12.1% | 5.0% |
| Trees | 17.8% | 11.8% | 10.8% |
| Shrub & grass | 6.2% | 5.5% | 7.7% |
| Bare / sparse | 8.2% | 8.4% | 8.4% |
| Water | 9.0% | 14.8% | 4.8% |
| Wetland & mangrove | 6.2% | 4.4% | 6.3% |
| Snow & ice | 7.5% | 2.9% | 2.3% |
| *Ignored pixels (share of all pixels)* | *17.5%* | *18.8%* | *17.4%* |

Train pixel counts per class: built-up 30.9 M, cropland 9.8 M, trees 16.1 M, shrub & grass 5.6 M, bare 7.4 M, water 8.1 M, wetland 5.6 M, snow & ice 6.7 M.

Built-up is still the largest class, by design: the prototype targets urban areas. The imbalance is about 6:1 between the largest and smallest class (12:1 before the extra rural sites), so training should use **class-weighted or focal loss** and report **per-class IoU/F1**, not only overall accuracy.

## 9. Rebuilding the dataset

Requirements:
- Python 3.13 with `requirements.txt` installed (`.venv`).
- A Google Earth Engine account:
  - Register at https://code.earthengine.google.com/register (noncommercial).
  - Create or select a Google Cloud project and enable the Earth Engine API for it.
  - Put the project ID in `.env` as `Project_ID = <id>`.
  - Run `.venv\Scripts\earthengine authenticate` once.

Steps, from the repository root:

1. `.venv\Scripts\python data_pipeline\download.py`
   - Downloads S1 tiles and raw labels for all sites, then computes the consensus labels.
   - Existing files are skipped, so rerunning after an interruption only fetches what is missing. Failed sites are listed at the end.
   - Takes about 45 minutes on a normal connection.
2. `.venv\Scripts\python data_pipeline\make_patches.py`: patch index, normalisation and class statistics.
3. `.venv\Scripts\python data_pipeline\preview.py`: quick-look PNGs (optional).

Optional: `download.py --relabel` recomputes the consensus labels from the stored WorldCover and Dynamic World bands, offline. Run this after changing `config.consensus_labels`.

All settings are in `data_pipeline/config.py`: sites, splits, year, tile size, encoding, class mappings and the consensus rule.

## 10. Design decisions log

| Decision | Alternatives considered | Reason |
|---|---|---|
| Google Earth Engine as source | ASF DAAC raw GRD + SNAP | GEE already applies calibration, terrain correction and dB conversion, and computes composites server-side. The same retrieval code can serve the prototype for any lat/lon. |
| Global consensus labels | Copernicus Urban Atlas 2018 (Europe only); a single global map | The prototype must work worldwide. Urban Atlas only covers European cities. The consensus of two maps is cleaner than either map alone and gives a curated dataset rather than a copy of one product. |
| Esri LULC kept out | using it as a third vote | The project requires a sanity check against an independent map not used for training. |
| 8 merged classes | the full 11 WorldCover classes | Shrub vs grass and moss/lichen vs bare are not separable with C-band SAR. Merging raises achievable accuracy. |
| Added 39 rural sites | urban sites only | The urban pilot was about 85% built-up; rare classes would be missing entirely. |
| uint8 encoding, 10.24 km tiles, patch index | int16, 15 km tiles, copied patch arrays (~20 GB) | Keeps the dataset under 4 GB (2.4 GB) with no measurable loss: ±0.1 dB rounding vs ~0.5–1 dB speckle. |
| Full 10 m resolution kept | coarser pixels, then sharpening | Detail lost by downsampling cannot be recovered by sharpening; sharpening SAR also amplifies speckle. |
| Wetland exception in the consensus | strict agreement only | Dynamic World has no mangrove class, so strict agreement removed nearly all wetland pixels. |
| 35 extra rural sites in new biomes, test set unchanged | more patches from existing sites | The first model overfitted the training sites and failed on an unseen biome (boreal forest). New places add variety; a fixed test set keeps before/after scores comparable. |
| Patch ignore limit 60% | 50% | At 50%, Kano and Mongolia kept 0 patches and dropped out of the dataset. 60% keeps every site. |

## 11. Limitations

- **Label noise remains.** The consensus removes many errors, but where both maps make the same error (correlated errors), the error stays in the labels. Both maps are optical-based (Sentinel-2), so they can share errors, for example on cloud-prone tropical sites.
- **Consensus selects "easy" pixels.** Ambiguous pixels (mixed urban/green, field edges, savanna) are mostly ignored. Test scores on consensus labels are therefore **optimistic** compared with performance on all pixels. The prototype has to classify every pixel, including these hard ones.
- **Annual labels, monthly inputs.** One 2021 label is used for all 12 months. Seasonal states differ from the annual label: flooded fields, seasonal snow, harvested or bare cropland, seasonally dry rivers. This label noise is exactly the effect that can make **seasonal change look like land-use change** in the prototype.
- **Snow is seasonal in SAR.** Dry snow is nearly transparent to C-band, while wet snow is very dark. At Aletsch, January and July composites look very different for the same "snow & ice" label (see `previews/aletsch_glacier.png`).
- **Uneven revisit.** Scenes per month range from 1 (some Americas, tropical and boreal sites) to about 29 (Europe). Fewer scenes mean noisier composites. Since 2022, Sentinel-1B is gone, so prototype runs for 2022–2024 will see sparser data than the training year. Sentinel-1C has been operating since 2025.
- **Terrain effects.** Layover and radar shadow in steep mountains (Karakoram, Alps, Patagonia) are not masked. Earth Engine's terrain correction fixes the geometry, but not the radiometric terrain effects (no terrain flattening).
- **Mixed orbits.** Ascending and descending passes are averaged. That reduces speckle, but backscatter of oriented structures (buildings, row crops) depends on the look direction.
- **Geographic coverage is still limited.** 116 sites of 10 km cannot represent every landscape. Expect lower accuracy in regions unlike the training data, for example tundra, very dense informal settlements, or high-altitude plateaus.
- **Class imbalance.** Built-up dominates; shrub & grass, snow and wetland are 4–5% each in train. Val has very little snow (0.3%), so snow val scores are noisy.

## 12. Work log: how this dataset was built

The steps below are in the order they were done, including the approaches that were dropped.

1. **Requirements review.** The task, from `Documentation/requirements_info.txt` and the lecture slides:
   - Train an LULC segmentation model with Sentinel-1 as the only input.
   - Build a prototype that takes a lat/lon and a start month and produces 12 monthly maps, with class percentages and change.
   - The report must describe the data source, preprocessing (calibration, dB, speckle, terrain correction), label matching, splits and class distribution.
2. **Choosing a data source.**
   - NASA Earthdata / ASF DAAC (raw Sentinel-1 GRD) was considered first. Raw scenes need full local SNAP processing and are about 1 GB each, which suits neither a prototype that must fetch any location on demand nor a compact dataset.
   - **Google Earth Engine** was chosen instead. It already provides calibrated, terrain-corrected dB backscatter and builds composites server-side.
3. **First label choice: Copernicus Urban Atlas 2018.** It was chosen to avoid a "generic" global dataset. A first pipeline was written for 22 European cities: download S1 from GEE, rasterize the Urban Atlas polygons onto the S1 grid, cut patches.
4. **Switch to global labels.** The prototype has to work anywhere in the world, and Urban Atlas only covers Europe.
   - The labels were replaced by the **consensus of ESA WorldCover 2021 and Dynamic World 2021**. It is curated, and cleaner than either map alone.
   - **Esri LULC 2021** was held out as the independent map for the case-study sanity check.
   - The reference year moved to **2021**, the WorldCover year and the last full year of Sentinel-1A + 1B.
   - 77 urban sites across 6 regions were defined, with splits by site.
   - The pipeline was rewritten as one GEE download script (`download.py`). Labels are sampled on the same grid as the imagery, so no rasterization or reprojection step is needed.
5. **Environment.** Python 3.15 had no PyTorch wheels, so a **Python 3.13 venv** (`.venv`, created with `uv`) was set up with the dataset packages only. PyTorch is not installed yet. The GEE project ID is read from `.env` (`Project_ID`), and GEE was authenticated with `earthengine authenticate`.
6. **Pilot download (Berlin, Lagos).** The pipeline worked and the labels lined up with the imagery. The pilot also showed that city-centre tiles are about **85% built-up**, with almost no cropland, bare ground, wetland or snow.
7. **Added 39 rural sites**, each focused on one under-represented class: cropland, trees, bare, shrub & grass, wetland, snow & ice. A coverage check of Sentinel-1 IW scenes per month for every site led to three replacements:
   - Gobi → Badain Jaran (no IW scenes).
   - Greenland margin → Columbia Icefield (no IW scenes).
   - Brisbane → Adelaide (one month with no scenes).

   This gave **116 sites**.
8. **Size reduction.** The first full download format (int16, 15.36 km tiles, patches copied into `.npy` arrays) was heading for about **20 GB**. The target was at most 4–5 GB. The download was stopped and restarted from scratch with:
   - **uint8 backscatter**: 0.2 dB steps, ±0.1 dB error, far below speckle;
   - **10.24 km tiles**: 16 patches per site, all 116 sites kept;
   - **patches as an index** into the tiles instead of copied arrays.

   Lowering the spatial resolution and sharpening later was discussed and rejected. Lost detail cannot be recovered, and sharpening amplifies SAR speckle. The full 10 m resolution was kept and only the value precision was reduced. Result: **2.4 GB**.
9. **Download robustness.** The full run twice stopped on network timeouts to `oauth2.googleapis.com`. Retries were added for Earth Engine startup and for every request, and a failing site no longer aborts the run. The third run downloaded all 116 sites in about 45 minutes.
10. **Label quality fixes after the full download.**
    - **Wetland:** strict agreement kept only 0.1% wetland in train, because Dynamic World has no mangrove class. A wetland exception was added (`config.consensus_labels`), raising wetland to 5.3%. To allow this, the consensus is now computed locally from the stored WorldCover and Dynamic World bands, so the rule can be changed offline (`download.py --relabel`) without re-downloading.
    - **Patch filter:** with a 50% ignore limit, Kano farmland and the Mongolian steppe kept 0 patches. The limit was raised to 60%, which keeps every site. The result is 1780 patches (1214 / 270 / 296).
11. **Statistics, previews, documentation.** These were generated:
    - normalisation statistics (train only);
    - class distribution per split;
    - a per-site summary;
    - a preview PNG for every site.

    The previews were checked visually, for example Berlin, Sundarbans and Aletsch. The Aletsch preview shows the seasonal snow effect noted in the Limitations. This README documents the result.

12. **Extra rural sites after the first training run.** The first full run (U-Net, ResNet-34, three-month input) reached test OA 0.84 and mIoU 0.56, but confused trees with cropland, did poorly on the boreal test forest and the snow test site, and overfitted (train mIoU 0.79 vs val 0.47). 38 rural sites in new biomes were added (see section 4); Vasyugan was replaced by the Danube delta, and three sites with 0 kept patches were removed. Result: **151 sites, 2244 patches, 3.1 GB**. The test split is unchanged.

**Next steps (not part of the dataset):**
- A training notebook: data loader with month sampling and augmentation, a class-weighted loss, a U-Net-style model, and OA / mIoU / per-class IoU and F1 on the test split.
- The prototype, reusing `download.py`'s compositing for any lat/lon and start month.

## 13. Per-site table

- **Scenes/month:** min–max Sentinel-1 scenes in a monthly composite.
- **Labels kept %:** share of pixels kept by the consensus.
- **Patches:** kept patches out of 16.
- **Class columns:** % of the labelled pixels in the kept patches.

| Site | Lat | Lon | Region | Focus | Split | Scenes/month (min–max) | Labels kept % | Patches | Built | Crop | Trees | Shrub/grass | Bare | Water | Wetland | Snow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| london | 51.51 | -0.13 | europe | urban | train | 17–24 | 76.7 | 16 | 90.2 | 0.0 | 4.6 | 1.1 | 0.0 | 4.1 | 0.0 | 0.0 |
| paris | 48.86 | 2.35 | europe | urban | train | 12–16 | 87.1 | 16 | 96.4 | 0.0 | 1.8 | 0.1 | 0.0 | 1.7 | 0.0 | 0.0 |
| berlin | 52.52 | 13.40 | europe | urban | train | 22–29 | 70.9 | 16 | 85.3 | 0.0 | 11.2 | 1.5 | 0.1 | 2.0 | 0.0 | 0.0 |
| madrid | 40.42 | -3.70 | europe | urban | train | 11–17 | 76.7 | 16 | 85.8 | 0.0 | 9.3 | 4.6 | 0.0 | 0.2 | 0.0 | 0.0 |
| rome | 41.90 | 12.50 | europe | urban | train | 15–23 | 72.2 | 16 | 83.4 | 0.6 | 12.0 | 2.4 | 0.0 | 1.6 | 0.0 | 0.0 |
| warsaw | 52.23 | 21.01 | europe | urban | train | 18–21 | 69.1 | 16 | 74.2 | 0.1 | 20.6 | 0.8 | 0.0 | 4.2 | 0.1 | 0.0 |
| stockholm | 59.33 | 18.07 | europe | urban | train | 9–22 | 70.5 | 16 | 49.9 | 0.0 | 23.0 | 2.0 | 0.0 | 25.0 | 0.2 | 0.0 |
| amsterdam | 52.37 | 4.90 | europe | urban | train | 17–21 | 71.8 | 15 | 66.7 | 0.2 | 15.3 | 1.4 | 0.1 | 16.1 | 0.1 | 0.0 |
| lisbon | 38.72 | -9.14 | europe | urban | train | 13–17 | 84.9 | 16 | 44.3 | 0.2 | 7.2 | 0.9 | 0.1 | 47.4 | 0.0 | 0.0 |
| kyiv | 50.45 | 30.52 | europe | urban | train | 18–24 | 73.6 | 16 | 61.6 | 0.0 | 20.7 | 0.6 | 0.1 | 16.6 | 0.4 | 0.0 |
| innsbruck | 47.27 | 11.40 | europe | urban | train | 11–20 | 74.5 | 15 | 24.0 | 6.2 | 60.9 | 7.4 | 0.0 | 1.4 | 0.0 | 0.0 |
| reykjavik | 64.15 | -21.94 | europe | urban | train | 14–15 | 85.2 | 16 | 17.2 | 0.0 | 1.2 | 7.2 | 0.1 | 73.3 | 1.0 | 0.0 |
| budapest | 47.50 | 19.04 | europe | urban | val | 8–11 | 64.1 | 15 | 75.2 | 0.2 | 15.8 | 1.5 | 0.1 | 7.2 | 0.0 | 0.0 |
| helsinki | 60.17 | 24.94 | europe | urban | val | 9–25 | 85.2 | 16 | 25.5 | 0.1 | 11.8 | 0.7 | 0.1 | 60.3 | 1.5 | 0.0 |
| frankfurt | 50.11 | 8.68 | europe | urban | test | 17–20 | 69.4 | 16 | 57.2 | 1.5 | 37.2 | 1.4 | 0.0 | 2.8 | 0.0 | 0.0 |
| athens | 37.98 | 23.73 | europe | urban | test | 15–20 | 87.8 | 16 | 93.2 | 0.0 | 4.4 | 2.0 | 0.1 | 0.3 | 0.0 | 0.0 |
| new_york | 40.71 | -74.01 | n_america | urban | train | 1–3 | 88.5 | 16 | 62.5 | 0.0 | 1.8 | 0.6 | 0.0 | 34.9 | 0.1 | 0.0 |
| los_angeles | 34.05 | -118.24 | n_america | urban | train | 12–22 | 86.5 | 16 | 96.9 | 0.0 | 1.2 | 1.4 | 0.1 | 0.4 | 0.0 | 0.0 |
| houston | 29.76 | -95.37 | n_america | urban | train | 5–8 | 66.9 | 16 | 94.3 | 0.0 | 2.8 | 1.9 | 0.4 | 0.6 | 0.0 | 0.0 |
| toronto | 43.65 | -79.38 | n_america | urban | train | 3–6 | 79.0 | 16 | 57.7 | 0.0 | 6.4 | 0.9 | 0.2 | 34.7 | 0.1 | 0.0 |
| mexico_city | 19.43 | -99.13 | n_america | urban | train | 17–21 | 87.2 | 16 | 99.2 | 0.0 | 0.6 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| denver | 39.74 | -104.99 | n_america | urban | train | 8–12 | 71.5 | 16 | 96.4 | 0.1 | 0.2 | 1.4 | 0.4 | 1.5 | 0.0 | 0.0 |
| miami | 25.76 | -80.19 | n_america | urban | train | 2–3 | 88.8 | 16 | 41.2 | 0.0 | 1.4 | 0.4 | 0.0 | 54.3 | 2.7 | 0.0 |
| minneapolis | 44.98 | -93.27 | n_america | urban | train | 2–3 | 57.6 | 14 | 82.2 | 0.0 | 7.5 | 1.9 | 0.0 | 8.2 | 0.1 | 0.0 |
| new_orleans | 29.95 | -90.07 | n_america | urban | train | 2–3 | 73.7 | 16 | 80.5 | 0.0 | 2.3 | 3.4 | 0.3 | 13.4 | 0.1 | 0.0 |
| atlanta | 33.75 | -84.39 | n_america | urban | val | 1–3 | 56.3 | 15 | 71.9 | 0.0 | 25.2 | 2.1 | 0.1 | 0.7 | 0.0 | 0.0 |
| vancouver | 49.28 | -123.12 | n_america | urban | val | 6–11 | 72.8 | 14 | 47.5 | 0.0 | 6.2 | 0.6 | 0.2 | 45.6 | 0.0 | 0.0 |
| chicago | 41.88 | -87.63 | n_america | urban | test | 1–4 | 90.9 | 16 | 56.9 | 0.0 | 0.3 | 0.3 | 0.2 | 42.3 | 0.0 | 0.0 |
| phoenix | 33.45 | -112.07 | n_america | urban | test | 8–12 | 74.9 | 16 | 90.0 | 0.2 | 0.4 | 1.2 | 8.0 | 0.3 | 0.0 | 0.0 |
| anchorage | 61.22 | -149.90 | n_america | urban | test | 3–6 | 50.7 | 10 | 25.6 | 0.0 | 9.8 | 0.3 | 0.1 | 64.2 | 0.0 | 0.0 |
| sao_paulo | -23.55 | -46.63 | s_america | urban | train | 3–5 | 89.2 | 16 | 96.9 | 0.0 | 2.2 | 0.4 | 0.0 | 0.5 | 0.0 | 0.0 |
| buenos_aires | -34.60 | -58.38 | s_america | urban | train | 7–11 | 89.8 | 16 | 62.7 | 0.0 | 3.1 | 0.8 | 0.1 | 31.7 | 1.6 | 0.0 |
| santiago | -33.45 | -70.67 | s_america | urban | train | 8–13 | 87.0 | 16 | 96.6 | 0.1 | 1.3 | 1.1 | 0.8 | 0.1 | 0.0 | 0.0 |
| manaus | -3.12 | -60.02 | s_america | urban | train | 2–3 | 88.5 | 16 | 58.7 | 0.0 | 10.5 | 0.2 | 0.1 | 29.7 | 0.8 | 0.0 |
| brasilia | -15.79 | -47.88 | s_america | urban | train | 2–3 | 65.9 | 16 | 43.6 | 0.1 | 4.5 | 27.2 | 0.3 | 24.2 | 0.0 | 0.0 |
| quito | -0.18 | -78.47 | s_america | urban | train | 9–20 | 77.2 | 16 | 68.0 | 0.3 | 29.2 | 2.3 | 0.0 | 0.2 | 0.0 | 0.0 |
| corumba | -19.01 | -57.65 | s_america | urban | train | 6–9 | 87.1 | 16 | 15.5 | 0.0 | 34.3 | 2.8 | 0.0 | 5.1 | 42.2 | 0.0 |
| lima | -12.05 | -77.04 | s_america | urban | val | 5–9 | 86.0 | 16 | 91.8 | 0.0 | 0.1 | 0.1 | 8.0 | 0.1 | 0.0 | 0.0 |
| bogota | 4.71 | -74.07 | s_america | urban | test | 3–5 | 85.7 | 16 | 88.7 | 0.0 | 7.7 | 2.9 | 0.0 | 0.6 | 0.0 | 0.0 |
| curitiba | -25.43 | -49.27 | s_america | urban | test | 3–6 | 81.9 | 16 | 88.1 | 0.0 | 10.4 | 1.1 | 0.0 | 0.4 | 0.0 | 0.0 |
| cairo | 30.04 | 31.24 | africa | urban | train | 6–8 | 76.8 | 15 | 85.2 | 0.9 | 0.1 | 0.1 | 7.5 | 6.1 | 0.0 | 0.0 |
| lagos | 6.52 | 3.38 | africa | urban | train | 4–6 | 90.0 | 16 | 69.0 | 0.0 | 0.8 | 0.0 | 0.1 | 27.0 | 3.2 | 0.0 |
| johannesburg | -26.20 | 28.05 | africa | urban | train | 6–8 | 71.2 | 15 | 82.7 | 0.2 | 5.9 | 6.9 | 3.8 | 0.6 | 0.0 | 0.0 |
| addis_ababa | 9.03 | 38.74 | africa | urban | train | 4–6 | 77.3 | 16 | 91.1 | 0.8 | 6.7 | 1.3 | 0.1 | 0.0 | 0.0 | 0.0 |
| kinshasa | -4.44 | 15.27 | africa | urban | train | 4–6 | 59.3 | 14 | 69.4 | 0.1 | 19.9 | 10.5 | 0.0 | 0.0 | 0.0 | 0.0 |
| dakar | 14.72 | -17.47 | africa | urban | train | 1–3 | 66.7 | 13 | 49.6 | 0.0 | 0.5 | 1.1 | 0.9 | 47.9 | 0.1 | 0.0 |
| dar_es_salaam | -6.79 | 39.21 | africa | urban | train | 2–6 | 69.1 | 15 | 80.6 | 0.0 | 16.2 | 1.5 | 0.1 | 1.4 | 0.1 | 0.0 |
| khartoum | 15.50 | 32.56 | africa | urban | train | 2–3 | 72.7 | 16 | 65.2 | 16.4 | 0.0 | 0.1 | 18.0 | 0.3 | 0.0 | 0.0 |
| casablanca | 33.57 | -7.59 | africa | urban | val | 18–21 | 81.0 | 16 | 85.4 | 2.1 | 0.2 | 0.7 | 0.7 | 10.9 | 0.1 | 0.0 |
| lusaka | -15.39 | 28.32 | africa | urban | val | 2–3 | 65.5 | 16 | 89.4 | 0.4 | 5.2 | 4.8 | 0.0 | 0.3 | 0.0 | 0.0 |
| nairobi | -1.29 | 36.82 | africa | urban | test | 5–8 | 73.9 | 16 | 76.6 | 0.2 | 9.9 | 13.0 | 0.2 | 0.0 | 0.0 | 0.0 |
| accra | 5.60 | -0.19 | africa | urban | test | 5–5 | 82.6 | 16 | 91.8 | 0.0 | 6.0 | 1.6 | 0.2 | 0.4 | 0.0 | 0.0 |
| delhi | 28.61 | 77.21 | asia | urban | train | 3–6 | 67.2 | 16 | 72.0 | 1.4 | 24.1 | 0.8 | 0.1 | 1.7 | 0.0 | 0.0 |
| mumbai | 19.08 | 72.88 | asia | urban | train | 2–4 | 69.1 | 16 | 84.6 | 1.2 | 6.9 | 0.3 | 0.2 | 4.5 | 2.3 | 0.0 |
| dhaka | 23.81 | 90.41 | asia | urban | train | 4–6 | 63.1 | 13 | 83.1 | 2.2 | 4.4 | 2.6 | 2.1 | 5.0 | 0.6 | 0.0 |
| beijing | 39.90 | 116.41 | asia | urban | train | 4–8 | 81.2 | 16 | 96.0 | 0.0 | 2.0 | 0.0 | 0.0 | 1.9 | 0.0 | 0.0 |
| shanghai | 31.23 | 121.47 | asia | urban | train | 4–6 | 84.6 | 16 | 91.7 | 0.0 | 1.1 | 0.0 | 0.1 | 7.1 | 0.0 | 0.0 |
| tokyo | 35.68 | 139.65 | asia | urban | train | 4–8 | 91.4 | 16 | 98.4 | 0.0 | 1.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| jakarta | -6.21 | 106.85 | asia | urban | train | 2–5 | 88.9 | 16 | 97.4 | 0.0 | 2.0 | 0.3 | 0.0 | 0.4 | 0.0 | 0.0 |
| manila | 14.60 | 120.98 | asia | urban | train | 3–7 | 92.8 | 16 | 68.5 | 0.0 | 0.4 | 0.2 | 0.0 | 30.9 | 0.0 | 0.0 |
| ho_chi_minh | 10.82 | 106.63 | asia | urban | train | 7–8 | 83.3 | 16 | 96.3 | 0.4 | 0.5 | 2.1 | 0.1 | 0.6 | 0.0 | 0.0 |
| riyadh | 24.71 | 46.68 | asia | urban | train | 2–3 | 58.7 | 14 | 82.7 | 0.0 | 0.0 | 0.0 | 17.2 | 0.0 | 0.0 | 0.0 |
| dubai | 25.20 | 55.27 | asia | urban | train | 4–8 | 72.5 | 16 | 48.1 | 0.1 | 0.0 | 0.6 | 22.3 | 28.6 | 0.2 | 0.0 |
| almaty | 43.24 | 76.94 | asia | urban | train | 7–8 | 61.6 | 16 | 73.6 | 0.1 | 24.0 | 2.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| ulaanbaatar | 47.89 | 106.91 | asia | urban | train | 2–6 | 76.2 | 16 | 53.2 | 0.0 | 24.8 | 20.8 | 0.3 | 0.8 | 0.0 | 0.0 |
| khulna | 22.85 | 89.54 | asia | urban | train | 6–9 | 52.4 | 11 | 41.9 | 17.7 | 27.4 | 0.1 | 0.6 | 12.3 | 0.0 | 0.0 |
| seoul | 37.57 | 126.98 | asia | urban | val | 3–8 | 79.5 | 16 | 73.7 | 0.0 | 18.5 | 0.1 | 0.0 | 7.6 | 0.0 | 0.0 |
| karachi | 24.86 | 67.00 | asia | urban | val | 3–6 | 61.1 | 13 | 56.1 | 0.7 | 0.8 | 1.1 | 10.2 | 26.0 | 5.0 | 0.0 |
| kolkata | 22.57 | 88.36 | asia | urban | val | 4–6 | 81.2 | 16 | 86.5 | 0.6 | 3.4 | 0.2 | 0.0 | 9.1 | 0.1 | 0.0 |
| bangkok | 13.76 | 100.50 | asia | urban | test | 5–9 | 83.4 | 16 | 94.4 | 0.0 | 1.6 | 0.3 | 0.0 | 3.8 | 0.0 | 0.0 |
| tehran | 35.69 | 51.39 | asia | urban | test | 3–6 | 92.6 | 16 | 99.3 | 0.1 | 0.2 | 0.1 | 0.3 | 0.1 | 0.0 | 0.0 |
| chengdu | 30.57 | 104.07 | asia | urban | test | 7–10 | 59.5 | 16 | 80.7 | 6.7 | 7.4 | 0.7 | 1.0 | 3.5 | 0.0 | 0.0 |
| sydney | -33.87 | 151.21 | oceania | urban | train | 6–9 | 75.8 | 16 | 61.8 | 0.0 | 6.0 | 2.5 | 0.0 | 29.6 | 0.1 | 0.0 |
| melbourne | -37.81 | 144.96 | oceania | urban | train | 3–6 | 78.8 | 16 | 77.3 | 0.0 | 6.1 | 3.2 | 0.2 | 12.9 | 0.3 | 0.0 |
| auckland | -36.85 | 174.76 | oceania | urban | train | 9–11 | 76.3 | 16 | 46.8 | 0.0 | 6.0 | 3.1 | 0.0 | 42.7 | 1.4 | 0.0 |
| perth | -31.95 | 115.86 | oceania | urban | val | 3–5 | 82.8 | 16 | 70.5 | 0.0 | 5.8 | 4.8 | 0.1 | 18.8 | 0.1 | 0.0 |
| adelaide | -34.93 | 138.60 | oceania | urban | test | 4–6 | 74.9 | 16 | 91.4 | 0.1 | 4.7 | 3.7 | 0.0 | 0.1 | 0.0 | 0.0 |
| iowa_corn_belt | 42.30 | -93.90 | n_america | cropland | train | 6–8 | 89.8 | 16 | 0.8 | 71.5 | 23.1 | 3.4 | 0.0 | 1.2 | 0.0 | 0.0 |
| punjab_india | 30.60 | 75.30 | asia | cropland | train | 6–9 | 94.0 | 16 | 5.8 | 94.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| poltava_ukraine | 49.30 | 33.50 | europe | cropland | train | 17–21 | 92.3 | 16 | 0.5 | 92.8 | 3.6 | 0.7 | 0.0 | 1.1 | 1.4 | 0.0 |
| pampas_argentina | -35.00 | -61.50 | s_america | cropland | train | 4–5 | 75.8 | 16 | 0.0 | 42.0 | 0.6 | 57.1 | 0.0 | 0.2 | 0.1 | 0.0 |
| nile_delta | 30.95 | 30.85 | africa | cropland | train | 12–16 | 91.2 | 16 | 12.7 | 87.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| mato_grosso_soy | -13.10 | -56.10 | s_america | cropland | train | 2–3 | 93.4 | 16 | 0.1 | 82.8 | 16.5 | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| mekong_delta_rice | 10.30 | 105.50 | asia | cropland | train | 7–9 | 72.2 | 16 | 21.9 | 55.7 | 5.6 | 0.1 | 0.0 | 16.7 | 0.0 | 0.0 |
| beauce_france | 48.25 | 1.60 | europe | cropland | val | 22–26 | 95.4 | 16 | 2.5 | 92.8 | 4.6 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| kansas_pivots | 37.95 | -100.70 | n_america | cropland | val | 4–6 | 89.7 | 16 | 0.3 | 87.5 | 0.0 | 12.0 | 0.2 | 0.0 | 0.0 | 0.0 |
| north_china_plain | 37.00 | 115.50 | asia | cropland | test | 4–6 | 86.2 | 16 | 19.6 | 80.2 | 0.1 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 |
| kano_farmland | 12.20 | 8.30 | africa | cropland | test | 2–3 | 24.1 | 1 | 3.3 | 91.5 | 0.0 | 4.8 | 0.5 | 0.0 | 0.0 | 0.0 |
| amazon_forest | -5.00 | -63.00 | s_america | trees | train | 4–6 | 95.9 | 16 | 0.0 | 0.0 | 94.0 | 0.0 | 0.0 | 5.7 | 0.3 | 0.0 |
| congo_forest | 1.00 | 23.50 | africa | trees | train | 2–3 | 99.9 | 16 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| black_forest | 48.20 | 8.20 | europe | trees | train | 17–23 | 91.6 | 16 | 1.2 | 0.1 | 91.0 | 7.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| borneo_forest | 0.90 | 114.00 | asia | trees | train | 1–3 | 98.1 | 15 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ontario_boreal | 50.00 | -85.00 | n_america | trees | val | 1–5 | 90.7 | 16 | 0.0 | 0.0 | 98.8 | 0.5 | 0.0 | 0.1 | 0.5 | 0.0 |
| siberia_taiga | 60.00 | 100.00 | asia | trees | test | 2–3 | 91.3 | 16 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| sahara_algeria | 26.00 | 4.00 | africa | bare | train | 4–6 | 99.0 | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| western_desert | 27.00 | 29.00 | africa | bare | train | 3–6 | 100.0 | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| badain_jaran | 40.00 | 102.00 | asia | bare | train | 5–8 | 99.6 | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 | 0.0 |
| namib_desert | -24.75 | 15.30 | africa | bare | train | 1–3 | 94.7 | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| atacama_desert | -24.50 | -69.80 | s_america | bare | val | 14–15 | 98.8 | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| rub_al_khali | 20.50 | 50.50 | asia | bare | test | 2–3 | 100.0 | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| mongolia_steppe | 47.00 | 103.00 | asia | shrub_grass | train | 1–3 | 28.0 | 5 | 0.0 | 54.8 | 0.0 | 44.8 | 0.0 | 0.0 | 0.4 | 0.0 |
| texas_shrubland | 30.50 | -101.50 | n_america | shrub_grass | train | 8–12 | 86.9 | 16 | 0.0 | 0.0 | 9.4 | 90.0 | 0.6 | 0.0 | 0.0 | 0.0 |
| kazakh_steppe | 50.50 | 70.00 | asia | shrub_grass | val | 3–5 | 62.6 | 13 | 0.0 | 0.6 | 0.1 | 31.1 | 0.5 | 51.6 | 16.2 | 0.0 |
| serengeti | -2.33 | 34.83 | africa | shrub_grass | test | 2–6 | 95.2 | 16 | 0.0 | 0.0 | 0.7 | 99.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| sundarbans | 21.95 | 89.18 | asia | wetland | train | 4–6 | 100.0 | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 28.9 | 71.1 | 0.0 |
| pantanal | -17.50 | -56.80 | s_america | wetland | train | 2–3 | 91.2 | 16 | 0.0 | 0.0 | 72.6 | 0.4 | 0.0 | 0.9 | 26.1 | 0.0 |
| sudd_wetland | 7.50 | 30.50 | africa | wetland | train | 4–8 | 98.4 | 16 | 0.0 | 0.0 | 0.0 | 1.4 | 0.0 | 0.5 | 98.1 | 0.0 |
| inner_niger_delta | 15.00 | -4.50 | africa | wetland | train | 2–3 | 92.2 | 16 | 0.0 | 0.5 | 0.0 | 3.0 | 0.0 | 0.1 | 96.4 | 0.0 |
| okavango_delta | -19.30 | 22.90 | africa | wetland | val | 2–3 | 93.5 | 16 | 0.0 | 0.0 | 8.5 | 39.8 | 0.0 | 0.0 | 51.7 | 0.0 |
| everglades | 25.80 | -80.80 | n_america | wetland | test | 2–3 | 98.8 | 16 | 0.4 | 0.0 | 1.9 | 0.0 | 0.0 | 0.0 | 97.6 | 0.0 |
| aletsch_glacier | 46.50 | 8.03 | europe | snow_ice | train | 19–24 | 81.4 | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 1.2 | 0.0 | 0.0 | 98.8 |
| columbia_icefield | 52.20 | -117.30 | n_america | snow_ice | train | 10–16 | 77.8 | 16 | 0.0 | 0.0 | 4.2 | 0.7 | 20.8 | 1.0 | 0.0 | 73.3 |
| patagonia_icefield | -49.30 | -73.20 | s_america | snow_ice | train | 4–8 | 98.5 | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 | 0.0 | 99.6 |
| vatnajokull | 64.40 | -16.80 | europe | snow_ice | train | 19–23 | 100.0 | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| svalbard_glacier | 78.65 | 16.40 | europe | snow_ice | val | 3–6 | 50.0 | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 91.8 | 0.0 | 8.2 |
| karakoram_baltoro | 35.75 | 76.40 | asia | snow_ice | test | 4–6 | 64.2 | 13 | 0.0 | 0.0 | 0.0 | 0.9 | 36.1 | 0.2 | 0.0 | 62.8 |
| sweden_taiga | 64.00 | 17.00 | europe | trees | train | 11–27 | 35.2 | 4 | 0.0 | 0.0 | 98.5 | 0.1 | 0.0 | 1.3 | 0.0 | 0.0 |
| alaska_boreal | 65.50 | -148.50 | n_america | trees | train | 4–6 | 74.7 | 16 | 0.0 | 0.0 | 95.8 | 3.9 | 0.1 | 0.2 | 0.0 | 0.0 |
| quebec_boreal | 49.50 | -73.00 | n_america | trees | train | 3–6 | 59.5 | 13 | 0.0 | 0.0 | 93.5 | 2.6 | 0.0 | 3.8 | 0.1 | 0.0 |
| appalachian_forest | 38.60 | -79.80 | n_america | trees | train | 2–3 | 94.6 | 16 | 0.0 | 0.0 | 99.4 | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| olympic_rainforest | 47.80 | -123.80 | n_america | trees | train | 7–11 | 78.1 | 16 | 0.0 | 0.0 | 93.8 | 0.8 | 0.3 | 0.1 | 0.0 | 5.0 |
| sikhote_alin_forest | 46.00 | 135.50 | asia | trees | train | 2–6 | 85.7 | 15 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gabon_forest | -0.50 | 11.50 | africa | trees | train | 2–3 | 93.2 | 2 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| tasmania_forest | -42.50 | 146.50 | oceania | trees | train | 3–9 | 97.0 | 16 | 0.0 | 0.0 | 98.0 | 1.9 | 0.0 | 0.0 | 0.0 | 0.0 |
| papua_forest | -5.00 | 142.00 | oceania | trees | train | 2–6 | 98.8 | 16 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| mont_blanc_massif | 45.88 | 6.95 | europe | snow_ice | train | 16–22 | 68.4 | 13 | 0.0 | 0.0 | 9.0 | 6.0 | 8.3 | 0.1 | 0.0 | 76.6 |
| jostedalsbreen | 61.70 | 7.00 | europe | snow_ice | train | 34–40 | 88.0 | 15 | 0.0 | 0.0 | 0.1 | 0.4 | 0.3 | 0.0 | 0.0 | 99.2 |
| juneau_icefield | 58.60 | -134.30 | n_america | snow_ice | train | 3–6 | 99.7 | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| tien_shan_inylchek | 42.20 | 79.90 | asia | snow_ice | train | 4–6 | 62.9 | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 63.5 | 5.1 | 0.0 | 31.4 |
| khumbu_himalaya | 27.95 | 86.85 | asia | snow_ice | train | 5–9 | 63.4 | 1 | 0.0 | 0.0 | 0.0 | 86.7 | 9.4 | 0.0 | 0.0 | 3.8 |
| tasman_glacier | -43.50 | 170.30 | oceania | snow_ice | train | 10–14 | 73.6 | 15 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.7 | 0.0 | 98.3 |
| caucasus_elbrus | 43.35 | 42.45 | europe | snow_ice | val | 13–16 | 87.2 | 8 | 0.0 | 0.0 | 0.0 | 0.3 | 11.9 | 0.0 | 0.0 | 87.8 |
| nebraska_sandhills | 42.00 | -101.00 | n_america | shrub_grass | train | 2–3 | 95.8 | 16 | 0.6 | 1.7 | 0.2 | 97.5 | 0.1 | 0.0 | 0.0 | 0.0 |
| great_basin_sagebrush | 40.50 | -117.00 | n_america | shrub_grass | train | 9–13 | 16.7 | 2 | 0.0 | 0.0 | 0.0 | 1.8 | 98.2 | 0.0 | 0.0 | 0.0 |
| patagonia_steppe | -45.50 | -69.50 | s_america | shrub_grass | train | 7–17 | 74.2 | 16 | 0.0 | 0.0 | 0.0 | 77.6 | 22.4 | 0.0 | 0.0 | 0.0 |
| karoo_shrubland | -32.50 | 22.00 | africa | shrub_grass | train | 3–6 | 86.6 | 16 | 0.0 | 0.0 | 0.1 | 1.2 | 98.7 | 0.0 | 0.0 | 0.0 |
| tibet_grassland | 31.50 | 92.00 | asia | shrub_grass | train | 8–12 | 77.5 | 16 | 1.1 | 0.0 | 0.0 | 94.8 | 3.5 | 0.6 | 0.0 | 0.0 |
| saskatchewan_wheat | 51.00 | -105.00 | n_america | cropland | train | 2–3 | 83.1 | 16 | 0.1 | 77.3 | 3.6 | 18.7 | 0.0 | 0.3 | 0.0 | 0.0 |
| central_valley_ca | 36.60 | -120.00 | n_america | cropland | train | 15–18 | 55.7 | 15 | 1.0 | 84.8 | 7.6 | 2.4 | 4.2 | 0.0 | 0.0 | 0.0 |
| punjab_pakistan | 30.00 | 72.50 | asia | cropland | train | 4–8 | 83.8 | 16 | 3.9 | 95.1 | 0.5 | 0.0 | 0.5 | 0.0 | 0.0 | 0.0 |
| karawang_rice | -6.30 | 107.40 | asia | cropland | train | 4–11 | 81.0 | 16 | 15.3 | 82.3 | 2.3 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 |
| heilongjiang_maize | 46.50 | 126.50 | asia | cropland | train | 3–5 | 90.6 | 16 | 3.6 | 95.4 | 0.4 | 0.0 | 0.0 | 0.0 | 0.6 | 0.0 |
| australia_wheatbelt | -31.50 | 117.50 | oceania | cropland | train | 4–6 | 25.0 | 1 | 0.2 | 45.3 | 0.1 | 16.5 | 37.8 | 0.0 | 0.0 | 0.0 |
| hungarian_plain | 46.50 | 20.50 | europe | cropland | train | 18–21 | 92.4 | 16 | 0.9 | 91.5 | 0.8 | 6.7 | 0.0 | 0.0 | 0.0 | 0.0 |
| hudson_bay_lowland | 53.50 | -84.00 | n_america | wetland | train | 2–6 | 67.3 | 16 | 0.0 | 0.0 | 34.6 | 24.2 | 0.0 | 7.1 | 34.1 | 0.0 |
| danube_delta | 45.10 | 29.40 | europe | wetland | train | 19–24 | 94.0 | 16 | 0.2 | 0.0 | 1.7 | 8.9 | 0.6 | 13.5 | 75.1 | 0.0 |
| louisiana_marsh | 29.40 | -90.60 | n_america | wetland | train | 4–6 | 93.7 | 16 | 0.9 | 0.0 | 4.6 | 1.0 | 0.0 | 68.2 | 25.4 | 0.0 |
| niger_delta_mangrove | 4.60 | 6.50 | africa | wetland | train | 4–6 | 98.8 | 16 | 0.0 | 0.0 | 11.2 | 0.0 | 0.0 | 3.8 | 84.9 | 0.0 |
| taklamakan_desert | 38.50 | 83.00 | asia | bare | train | 4–6 | 100.0 | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| lake_balaton | 46.85 | 17.75 | europe | water | train | 21–27 | 92.7 | 16 | 1.7 | 1.7 | 8.0 | 3.3 | 0.0 | 84.3 | 1.0 | 0.0 |
| finnish_lakeland | 62.00 | 27.50 | europe | water | train | 5–36 | 65.7 | 16 | 0.1 | 1.5 | 89.4 | 0.7 | 0.0 | 8.0 | 0.2 | 0.0 |
