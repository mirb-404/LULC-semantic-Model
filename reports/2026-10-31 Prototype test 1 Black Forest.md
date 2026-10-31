# 2026-10-31 Prototype test 1: Black Forest

Review of the first rural prototype test (`Documentation/Images/Prototype_test_1/`): Google Earth view of a Black Forest clearing, `lulc_12_months.png` (`three_month_stats`) and `lulc_12_months_pca.png` (PCA variant), 2021, January start. Sections 1-5 analyse the failure; section 6 records the decisions taken because of it for the 2nd iteration, section 7 explains the training philosophy (variant names, months, PCA, terrain, one final model), section 8 the status before the build, and section 9 the dataset build on Datalab with the questions answered about the data (sources and exam compliance, previews, patches, roads, manual labels). Section 10 records the first 2nd-iteration training run (`Iter2_Test_1`) and its results; section 11 the repeat DEV run (noise), the full run (`Iter2_Test_1_Full`) and the comparison with the 1st iteration.

Previous reports in this story: `reports/Test run 6 results.md` (the 1st-iteration model), `reports/2026-10-27 Prototype 1.md` (first prototype tests in Berlin and Potsdam).

## 1. Verdict

- The `three_month_stats` maps are wrong in a **seasonal** way, not at random. Water, wetland and cropland appear in Jan-May and Dec and almost disappear in Jun-Nov. A forest with meadows does not change like that. The likely cause is snow, frozen ground and wet soil in winter. They lower C-band backscatter, so the radar looks like water or bare fields.
- The PCA variant is **more correct on this scene**: trees plus shrub & grass, almost no water, and the 12 months are stable. On this scene, PCA is not useless. Its yearly channels outweigh the 3-month window, so winter months cannot change the class. The cost is that the map can barely show real change.
- "Shrub & grass" on the open land is **correct**. WorldCover and Dynamic World call pastures and meadows grassland, so the model gives the label it was trained on.
- **Pixel-perfect maps are not possible with this input.** Sentinel-1 IW GRD is stored at 10 m, but its real resolution is about 20 × 22 m. The pipeline then applies a monthly mean and a 3 × 3 median, and the labels are only 65-75 % accurate (Venter et al. 2022). The best you can expect is clean shapes at about 20-30 m, with the right classes and no seasonal flicker.
- The comparison image covers a different area size. The Google Earth view shows a few hundred metres; the map covers 5.12 km. One farmhouse is 1-2 radar pixels. Compare on the same extent (the sanity-check tab with Esri), not by eye against a zoomed-in optical image.

## 2. Most important finding: the test area is in the training set

`data_pipeline/config.py` contains `black_forest` (lat 48.2, lon 8.2, split **train**, focus trees). If the test point is in or near that tile, the model saw this landscape during training and still predicts winter water. Two explanations are possible, and they need different fixes:

| Hypothesis | Test | If true |
|---|---|---|
| A. Train/serve skew: the prototype builds the input differently from the dataset | Run the prototype on the exact centre of the `black_forest` tile. Compare its input codes with `Dataset/s1/black_forest.tif` (band by band, same window). Compare its prediction with the notebook's prediction for that tile. | Fix `prototype/gee.py` or `pipeline.py` until the inputs match. No retraining needed. |
| B. The model never learned winter conditions | Look at the notebook's per-month test mIoU and the per-site predictions for `black_forest` in Jan / Jul. | Changes to data and training (section 3). |

Run test A first. It is cheap and decides between "bug" and "method problem".

Possible sources of skew to check in A: orbit mix (ascending and descending in both cases?), the 2021 date window per month, the incidence-angle mask, the order of mean, dB and median, the uint8 rounding, and normalisation with the training stats from the model card.

## 3. What is missing in the approach (if hypothesis B)

Ordered by expected effect.

1. **Labels are yearly, input is monthly.** Each training sample pairs one random month with the 2021 label. Winter months of snowy forest sites are a small share of the samples (few such sites, and the weighted sampler prefers other classes), so the model rarely learns that "dark in January" can still mean trees. Fixes:
   - Make the yearly map the main product: predict it once from the yearly statistics, then report monthly changes only where the monthly result disagrees over at least 2 months (temporal smoothing or HMM, already listed in `reports/2026-10-27 Prototype 1.md` §4.2).
   - Or oversample winter months of temperate, boreal and mountain sites in training.
2. **No radiometric terrain flattening.** Earth Engine's `S1_GRD` is geometrically terrain-corrected but not radiometrically (sigma0, not gamma0 RTC). Slopes facing the sensor look bright (they look like buildings: note the red spots); slopes facing away look dark (they look like water). The Black Forest is hilly. Fix: the Earth Engine angular-based flattening by Vollrath et al. 2020 (Remote Sensing 12(11) 1867), plus a layover / shadow mask. Needs a new download and retraining.
3. **The consensus labels remove the edges.** A pixel is labelled only where WorldCover and Dynamic World agree. They mostly disagree at boundaries (forest edge, field edge, roads, rivers). The loss therefore almost never sees an edge pixel, and the model learns blobs. Together with the 3 × 3 median and the overlapping softmax average, this explains the rounded shapes. Options:
   - Train on WorldCover alone (full coverage) or use soft labels from both maps instead of masking disagreements.
   - Drop the 3 × 3 median: the monthly mean already reduces speckle.
   - Use Gaussian-weighted window blending with 50 % overlap at inference.
4. **Dataset coherence.** The 176 sites are separate 10 km tiles: 79 (45 %) are city centres, and each rural site targets one class or biome. Missing: whole gradients of a region (forest, meadow, village, river) across all seasons, so the model learns the transitions and seasonal behaviour of one landscape type. Adding 10-20 temperate and mountain sites (Black Forest, Vosges, Alps foothills, Harz, Bavarian Forest, Carpathians), each with mixed land cover, would target this failure directly. Keep the 4-5 GB limit: the current dataset is 3.6 GB (4.28 GB with the terrain layers).
5. **Too many classes for radar?** Wetland, water, cropland in winter and bare are where C-band is ambiguous. If time runs out, merging Wetland into Water or Shrub & grass for the prototype is defensible in the report. It is listed in `ideas.txt` ("fix the classes too many classes").

## 3b. Objective: rural land, not cities

The product's goal is to segment the rural parts of the world. Cities are already well mapped, so the tool adds little value there. The current dataset does not match this goal.

| Current dataset | Mismatch with the rural goal |
|---|---|
| 79 of 176 sites urban (45 %); built-up is the largest class: 30.8 % of train and 54.2 % of test pixels | Most of the capacity, and most of the test score, goes to the class we care least about |
| 97 rural sites, each picked for one class or biome | The model rarely sees normal rural mosaics: field, meadow, forest edge, hamlet, stream in one tile |
| Random single month + 3-month window | Rural classes are defined by their yearly cycle (crops are ploughed, grow and are harvested; grass stays; forest is stable). A 3-month window cannot see that cycle |
| No terrain flattening | Much rural land is hilly or mountainous, where the terrain artefacts are worst |

Redesign, within the 4-5 GB budget:

1. **Rebalance the sites.** About 80 % rural mosaics, 20 % towns and peri-urban edges. Keep some built-up, because villages and roads exist in rural tiles, and the brief still asks for a screen recording of an urban area. Drop most of the dense city-centre sites: they add little that is new.
2. **Coherent site selection.** Stratify by biome and agricultural system, not by single class. For example: temperate mixed farmland and forest, mountain pasture, boreal, Mediterranean, semi-arid grazing land, tropical smallholder farms, rice paddies, savanna, wetland and delta, tundra. Use several tiles per stratum on different continents. Choose tiles with mixed classes (for example at least 3 classes covering at least 5 % each in WorldCover), so each tile teaches boundaries and transitions.
3. **Use the whole year as input.** Predict one yearly map from all 12 months (order-independent seasonal statistics, or a temporal encoder such as U-TAE). Then derive monthly maps and change from it with temporal smoothing. This targets the cropland / grassland / bare / wetland confusion directly, and removes the winter flicker seen in this test.
4. **Terrain flattening plus a layover / shadow mask** (section 3, item 2).
5. **Rural class set.** Keep Built-up as one coarse class. The classes that matter are cropland, grassland, trees, shrub, bare, water and wetland. Consider merging classes the radar cannot separate (for example shrub and grass), and say why in the report.
6. **Evaluate on rural sites.** Report test mIoU on rural sites only, and run the Esri sanity check on rural mosaics instead of Berlin.

## 4. PCA decision

- On test sites (Test_6), `three_month_stats` was better (+0.075 to +0.125 mIoU).
- In this live test, the PCA variant gave the cleaner map, because it relies on the yearly summary.
- Both results say the same thing: **yearly, order-independent information is what makes the map stable**. PCA is one way to get it, and the seasonal statistics are a better-founded way. Do not decide by removing PCA; decide by how strongly the model relies on yearly versus monthly input, and add temporal smoothing for the monthly maps.

## 5. Next steps (as planned before the decisions below)

1. Test A (skew check on the `black_forest` training tile). Decides everything below.
2. Check per-month test mIoU in the Test_6 notebook output for a winter drop.
3. Run the prototype on 3-4 more rural mixed sites with the Esri sanity tab, same extent, both variants.
4. Rebuild the site list for the rural objective (section 3b): rebalance, stratify by biome, choose mixed tiles. Re-download with terrain flattening and labels without the edge mask.
5. Retrain with full-year input. Compare against the current best model on rural test sites only.

## 6. Decisions for the 2nd iteration

The test above, the 1st-iteration results (Test_6) and the rural objective led to the following decisions. Each one names the evidence from the previous runs that triggered it. The implementation is in `data_pipeline/` (dataset) and `Notebooks/2nd_iteration.ipynb` (training); the 1st-iteration notebook is kept unchanged as `Notebooks/1st_iteration.ipynb`.

### 6.1 Product goal: rural land

**Decision.** The model is built, chosen and tested for rural land: fields, meadows, forest edges, hamlets, streams. Cities stay in the data only as far as towns and villages appear in rural land, and because the brief asks for a screen recording of an urban area.

**Why.** Cities are already well mapped. The 1st-iteration data was city-heavy (45 % of the sites, built-up the largest class), and its test score was dominated by built-up (54 % of test pixels, IoU 0.94-0.96), so a high test mIoU (0.742) said little about rural land. The Black Forest test showed the gap.

### 6.2 Radar preprocessing: terrain flattening, no median filter

**Decision.** Convert sigma0 to terrain-flattened gamma0 (Vollrath et al. 2020 volume model on the Copernicus GLO-30 DEM), mask layover, shadow and incidence angles outside 31-45°, and drop the 3 × 3 focal median.

**Evidence.** A live Earth Engine check on single July 2021 scenes (30 m sampling), ascending / descending:

| Site | Correlation VV vs slope towards the sensor, before | after | VV spatial std (dB), before | after | valid pixels |
|---|---|---|---|---|---|
| nepal_midhills | 0.82 / 0.78 | 0.05 / -0.08 | 4.1 / 4.4 | 2.4 / 3.1 | 99.7 / 97.0 % |
| andes_sacred_valley | 0.74 / 0.74 | -0.14 / -0.07 | 5.2 / 5.3 | 3.6 / 3.7 | 98.0 / 97.4 % |
| black_forest (1st-iteration train tile) | 0.76 / 0.78 | -0.09 / -0.11 | 4.3 / 3.9 | 3.0 / 2.6 | 99.1 / 99.8 % |

In the Black Forest, more than half of the VV variance in the 1st-iteration input followed the relief, not the land cover. This supports hypothesis B in section 2 (a method problem): the model had to learn hillsides as if they were a class. Test A (train/serve skew) is still worth running, but the prototype's input pipeline has to be rewritten for the 2nd iteration anyway.

The median filter is dropped because it blurred field edges, roads and streams; the monthly mean of 5-15 scenes already reduces speckle.

### 6.3 Sites: rural mosaics, city edges, fewer glaciers

**Decision.** 178 sites (1st iteration: 176):

| Group | Sites | Change |
|---|---|---|
| mixed rural mosaics | 54 (41 train, 7 val, 6 test) | new: farmed forest landscapes on all continents; 5 of them with lakes or rivers |
| single-class rural sites | 92 | kept from the 1st iteration, minus 5 training glacier sites |
| towns and city edges | 32 | 77 city centres cut to 20 smaller cities (+ Kampala, Port-au-Prince), plus 10 new city edges |

- **Mosaic rule.** A mosaic tile must have a largest WorldCover class of at most 80 % and a second class of at least 10 %, checked before download. The first rule ("at least 3 classes with 5 % each") rejected the Black Forest itself (59 % trees, 39 % grassland): WorldCover maps pasture and meadows as grassland, so a farmed forest landscape is mostly two classes plus small built-up and cropland shares. All 54 mosaic sites were checked live; 14 coordinates were moved by 0.15° to pass.
- **The test set includes the failure case**: `black_forest_villages` (47.95 N, 8.10 E), about 29 km from the 1st-iteration training tile.
- **City edges** (Man-made 30-51 % of the tile; Krakow, Des Moines, Fresno (val), Córdoba, Kumasi, Bamako, Chiang Mai, Yogyakarta, Lucknow, Christchurch): insurance that the model still works when pointed at a town, without making cities dominant again. Freiburg was rejected: about 19 km from the Black Forest test site, it would leak into the test.
- **Water mosaics** (water 22-56 %; Mecklenburg, Masuria, Brahmaputra, Paraná, Lake Kyoga): water was the smallest class, and thin rivers were the weak spot (Berlin in `reports/2026-10-27 Prototype 1.md`: water recall 59.5 %).
- **Glaciers halved** (Columbia, Vatnajökull, Mont Blanc, Jostedalsbreen, Tien Shan removed from train): permanent snow is not rural land use, and it was over-represented (7.5 % of train pixels, more than Man-made).

### 6.4 Classes: 7 instead of 8

**Decision.**

| Class | Colour | WorldCover | Includes |
|---|---|---|---|
| Man-made | `#e31a1c` red | 50 | buildings, roads, paved yards, railways |
| Crop fields | `#ffd92f` yellow | 40 | land ploughed and harvested each year |
| Trees | `#1a9850` dark green | 10, 95 | forest, orchards, plantations, mangrove |
| Grassland | `#a6d96a` light green | 20, 30, 90 | pasture, meadows, steppe, shrub and scrub, marsh |
| Bare ground | `#bf812d` brown | 60, 100 | desert, rock, sand, sparse vegetation |
| Water | `#2c7fb8` blue | 80 | lakes, rivers, sea |
| Snow & ice | `#dfe9f0` light blue-grey | 70 | glaciers, permanent snow |

**Why.**
- **Wetland & mangrove removed.** WorldCover herbaceous wetland is only 30-45 % correct, and wetland was a main false alarm in the Black Forest winter months. For the radar, mangrove is forest and marsh is low vegetation (or water when flooded). Dynamic World's flooded vegetation also maps to Grassland. The agreement rule is now strict (the wetland exception is gone).
- **Names.** "Built-up" became "Man-made": the class is not only buildings (roads and paved surfaces too), and at 10 m the radar sees settlement texture, not single buildings. "Shrub & grass" became "Grassland": the merged class is mostly pasture and meadow. Crop fields and Grassland stay separate on purpose: ploughed and harvested vs covered all year is the most useful rural distinction, and the radar sees it in the yearly cycle. Trees (dark green) and Grassland (light green) also stay separate: tall vegetation scatters the radar strongly (high VH), low vegetation does not.
- **Snow & ice stays separate from Bare ground** and gets a light blue-grey, because white (`#f0f0f0`) disappears on white figure backgrounds.
- mIoU over 7 classes is **not comparable** with the 1st iteration's 8-class mIoU.

### 6.5 Class balance: realistic shares, mild correction

**Decision.** Keep realistic class shares and correct only mildly (sqrt class weights, class-aware patch sampler, Lovász loss). Do not aim for equal shares.

**Evidence.** Estimated train shares for the 7 classes from WorldCover over full tiles, before the city-edge, water and glacier changes: Trees 32.8 %, Grassland 24.5, Crop fields 14.4, Bare ground 10.2, Snow & ice 7.5, Man-made 6.2, Water 4.5. Largest to smallest is about 7:1 (1st iteration: about 6:1 on the agreement pixels, with built-up the largest class at 30.8 %). The site changes in 6.3 raise Man-made and Water and lower Snow & ice.

**Why not equal shares.** A model learns how common each class is. Training on equal shares would make it expect about 14 % Man-made in every scene and paint villages where there are none: the same over-prediction that produced winter wetland in the Black Forest. Trees, the class the Black Forest test got wrong, are already the largest class, so more tree data would not help. The notebook now reports per-class precision next to recall: high recall with low precision means a class is over-predicted, and then its weight is lowered rather than more data added.

**Cities.** Removing city centres can make dense centres slightly worse (tree-rich suburbs, streets at an angle to the satellite, airports and rail yards), but the built-up signal is the strongest in radar. The 10 city edges insure against it, and the 4 test cities (Curitiba, Anchorage, Nairobi, Adelaide) are scored separately.

### 6.6 Labels: WorldCover everywhere, agreement weighted

**Decision.** Train on WorldCover on every pixel. Pixels where Dynamic World agrees get loss weight 1, the others 0.5 (`DISAGREE_WEIGHT`). The Lovász loss and the main scores use agreement pixels; a second score uses all pixels.

**Why.** The consensus mask of the 1st iteration ignored the pixels where the two maps disagree, and those are mostly boundaries. The loss never saw an edge, which explains the rounded blobs together with the median filter.

### 6.7 Model input: the whole year in every variant, no PCA

**Decision.** Every variant gets 9 order-independent seasonal statistics of the 12 months (VV, VH and VV-VH mean; VV and VH 2nd lowest and 2nd highest month; VH std; water months). Variants: `year_stats` (statistics only: the stable reference, it cannot show change), `month_year` (current month + statistics), `three_month_year` (3 months + statistics, the inputs of the 1st iteration's best model). Month dropout rises from 0.15 to 0.3. PCA and terrain channels are dropped.

**Why.** The Black Forest test showed the PCA model stable and the 3-month model flickering: yearly information is what keeps a dark winter month from turning forest into water. Test_6 showed the order-independent statistics beating calendar-bound PCA by 0.075-0.125 test mIoU. Month dropout forces the model to rely on the yearly statistics. Terrain channels break the Sentinel-1-only brief; the DEM is still used on the server for flattening.

### 6.8 Evaluation: rural score, smoothing and flicker

**Decision.** The main score is **rural test mIoU** (test sites that are not towns). Each test patch is predicted for all 12 months, raw and with 3-month temporal smoothing (mean softmax of months t-1, t, t+1), and **flicker** (share of pixels whose class changes between consecutive months) is reported. `BEST` is chosen by val mIoU among the variants with monthly maps; the prototype uses smoothing if it scores at least as well as raw. The 12 monthly maps of every mosaic test site, including `black_forest_villages`, are plotted.

**Why.** The labels are constant within 2021, so for most land every month-to-month change is a false change; the Black Forest maps had a lot of it, and the brief's change percentages inflate with it. Flicker measures exactly the failure seen in the test.

### 6.9 One final model

**Decision.** Each run delivers exactly one weight file. The run folder looks like this:

```
Weights/<RUN_NAME>/
  final_model.pt     <- the model to use (copy of BEST)
  model_card.json    <- how to feed it: channels, normalisation, classes, smoothing
  variants/          <- every trained variant + its training history (report material only)
```

- `DEV = True` (default, 15 epochs) trains the 3 variants once, only to decide which input works best on the new data. The variant with the best val mIoU among those with monthly maps is copied to `final_model.pt`.
- `DEV = False` (final run, up to 40 epochs) trains **only** `FINAL_VARIANT` (default `month_year`) and saves it as `final_model.pt`.

**Why.** The 1st-iteration run folders held one weight file per variant (`three_month_best.pt`, `three_month_pca_best.pt`, ...), so it was unclear which one to use. Only one of them was ever meant for use (the one named in `model_card.json`); the others were comparison material. The brief asks for a variant comparison in the results; the DEV comparison and the 1st-iteration ablation (Test_6) provide it, so the final run does not need to train several models.

## 7. Training philosophy

This section explains how the model is trained and what the variant names mean, because names like `three_month_pca_stats_terrain` hide simple ideas.

### 7.1 The task

- **Input:** Sentinel-1 radar, one composite per month: VV and VH backscatter in dB, and their difference VV - VH.
- **Output:** one class per 10 m pixel **for each of the 12 months**.
- **Label:** WorldCover exists only once per year (2021). The same yearly label is used for every month.

### 7.2 One training example

- A random 256 × 256 px piece of a site, **one random month out of all 12**, and the yearly label.
- Over thousands of steps the model sees every month of every site. **Training always uses all 12 months of data.**
- Why a random month: the model must give a correct map for any month the prototype asks for.

### 7.3 What "month" in a variant name means

The number says **how many months the model looks at to draw one monthly map**. It is not how much data it was trained on.

| Name | Input for the July map |
|---|---|
| `single_month` | July only |
| `three_month` | May, June and July (the window wraps: January uses November, December, January) |

- Why more than one month: one radar month is ambiguous. Wet soil, snow and a freshly ploughed field can all look like water. The previous months add context. In Test_5, `three_month` beat `single_month` (best val mIoU 0.599 vs 0.548).
- The limit: 3 winter months are still winter. That is the Black Forest failure (section 1).

### 7.4 Why not all 12 months as input for every map

If every monthly map sees the same 12 months, all 12 maps are identical, and the change between months is always 0. The brief asks for monthly maps and their change, so this cannot be the only input.

The solution is to combine both: **a yearly summary + the current month**. The yearly summary says "this pixel is forest"; the current month can still show a real change (a harvest, a flood, a clearing).

### 7.5 PCA (1st iteration, dropped)

- Every pixel has 36 numbers per year (12 months × VV, VH, VV - VH). Principal component analysis (PCA) compresses them into 6 numbers that describe the yearly pattern; these were added as extra input channels.
- It made the maps more stable (the PCA model was the cleaner one in the Black Forest test).
- But the 6 numbers are tied to calendar months: January in Germany behaves like July in Argentina, so PCA learns site-specific patterns. In Test_6 it lost to the seasonal statistics by 0.075-0.125 test mIoU.

### 7.6 Seasonal statistics (the yearly summary that replaces PCA)

- A hand-picked yearly summary per pixel: mean signal, darkest and brightest month (2nd lowest / 2nd highest, robust to one odd month), how much the vegetation signal varies over the year, and how many months look like water.
- It does not depend on the order of the months, so it works the same in both hemispheres.
- The 1st iteration's best model used it (`three_month_stats`, test mIoU 0.742). In the 2nd iteration it is extended to 9 statistics and used by every variant.

### 7.7 Terrain (1st iteration, side experiment only)

- Elevation, slope and height above the nearest river (HAND) from elevation models, as extra input channels. They helped water (water lies low and close to rivers: Water IoU 0.868 in Test_6).
- The brief says **Sentinel-1 only**, so terrain was always a side experiment (an ablation), never the delivered model. Dropped as an input in the 2nd iteration.
- The elevation model is still used when the dataset is built, to correct the radar for slopes (terrain flattening, section 6.2). That corrects the Sentinel-1 signal; it is not an extra input.

### 7.8 Why the 1st iteration had so many weight files

The 1st iteration trained an **ablation chain**: `three_month` → `+stats` → `+PCA` → `+PCA+stats` → `+terrain`. Each step adds one idea, and comparing them shows what each idea is worth. That comparison is the report's variant comparison. Every variant is a full model, so each produced its own weight file. Since section 6.9, the delivered model is always `final_model.pt`.

### 7.9 The 2nd-iteration variants: all three see the whole year

| Variant | Input | Monthly maps |
|---|---|---|
| `year_stats` | 9 yearly statistics only | 12 identical maps: the stability reference, cannot show change |
| `month_year` | current month + yearly statistics | 12 maps that can change; **default final model** |
| `three_month_year` | 3 months + yearly statistics | the inputs of the 1st iteration's best model, on the new data |

`month_year` is the default because it is the smallest input that still gives a separate map per month; the yearly statistics supply the context that the 3-month window was meant to give. If the DEV run shows `three_month_year` clearly ahead on rural mIoU and flicker, `FINAL_VARIANT` is changed to it.

### 7.10 How every model learns

The same for every variant (details in `Notebooks/1st_iteration.ipynb`, sections 6-9):

- **Network:** U-Net with a ResNet-34 encoder pretrained on ImageNet (transfer learning: it starts with useful edge and texture filters).
- **Loss:** weighted cross-entropy (rare classes count more; disagreement pixels count half) plus Lovász-softmax, which optimises IoU directly.
- **Augmentation:** random crop position, flips and rotations, small VV / VH brightness shifts and noise, and month dropout (the monthly input is blanked at random, so the model learns to rely on the yearly statistics).
- **Optimisation:** AdamW with weight decay, warmup + cosine learning-rate schedule, mixed precision.
- **Stability and overfitting:** an exponential moving average (EMA) of the weights is used for validation, saving and testing; early stopping when val mIoU stops improving; label smoothing and dropout.
- **Fair evaluation:** train, val and test are split by site, so val and test places are never seen in training; the final model is chosen on val, and test is only reported.

## 8. Status and next steps (2026-10-31)

Built, not yet run: the dataset scripts (`data_pipeline/`, Datalab steps in `data_pipeline/README.md`) and `Notebooks/2nd_iteration.ipynb` (pyflakes-clean). Checked live: terrain flattening (6.2) and the WorldCover mix of all new sites (6.3).

1. Build `Dataset_2nd_iteration/` on Datalab, then `make_patches.py` and `preview.py`. Check steep tiles in the previews and `skipped_sites.json`.
2. Run `Notebooks/2nd_iteration.ipynb`, DEV first. Read rural mIoU, flicker raw vs smoothed, per-class precision and recall, and the Man-made IoU on the 4 test cities (below about 0.85: add more city-edge sites).
3. Prototype updated for the 2nd iteration (done): the radar processing now comes from the shared `data_pipeline/sentinel1.py`, so the prototype's input is the training input (verified: 0 difference at `black_forest_villages`); model card driven variants, 9 statistics, 7 classes, 3-month smoothing. 1st-iteration weights still work.
4. Repeat this Black Forest test with the new model and compare.

## 9. Building the dataset on Datalab (2026-10-31)

### 9.1 Run log

| Step | What happened | Fix |
|---|---|---|
| first `download.py` | `ModuleNotFoundError: No module named 'ee'`: the active environment had no Earth Engine package | install `requirements.txt` into a separate environment |
| `uv pip install` | "Broken symlink at `.venv/bin/python3`": the root `.venv` points to a Python from an older Datalab image (same problem as on 2026-10-27); `sudo` does not help | `uv venv .venv-data`, `source .venv-data/bin/activate`, `uv pip install -r requirements.txt` (`.venv-data/` is git-ignored) |
| test and full download started at once | the three commands were pasted together, so the full `nohup` run started in parallel with the 2-site test | `pkill -f data_pipeline/download.py`, then one step at a time |
| `ee.Initialize` | "Project datalab-512920 noncommercial Earth Engine access has expired": `EE_PROJECT` pointed to the project Datalab provides | `export EE_PROJECT=<own registered project>` (the `Project_ID` in the local `.env`; also added to `~/.bashrc`). The Earth Engine login itself was still valid |
| retries | the script retried this error 10 × 10 s although a retry cannot fix it | `initialize()` now stops at once with a hint for expired / unregistered projects |
| 2-site test | `black_forest_villages` and `krakow_edge` downloaded; agreement WorldCover / Dynamic World 88.0 % and 61.9 % (city edges have more disagreement pixels, which the 1st-iteration consensus rule would have dropped) | — |
| full download | no progress information in the log | progress bar (tqdm: sites done / to go, elapsed, ETA); already downloaded sites are not counted; restart is safe |
| result | 177 of 178 sites; `java_rice_hills` skipped: June 2021 has 0 Sentinel-1 IW scenes (May and July 2 each) | correct behaviour: a month without radar cannot be used. One training site less |
| size | `Dataset_2nd_iteration/` = **3.7 GB** (budget 4-5 GB) | — |

Restarting is safe: every file is written as `<site>.tmp.tif` and renamed only when complete; a rerun skips finished files. If a run is cancelled right after a label file is written, its `_stats.json` can be missing; `download.py --relabel` rebuilds it offline.

### 9.2 Exam compliance of the data sources

The brief requires "Sentinel-1 data as its only input". The question was whether using Google Earth Engine, whose Google Earth imagery looks very good, breaks this. It does not: Earth Engine is only the platform that serves the data.

| Data | Source | Role | Compliant |
|---|---|---|---|
| model input | Sentinel-1 GRD, VV + VH, monthly (`COPERNICUS/S1_GRD`) | the only input | yes |
| training labels | ESA WorldCover, Dynamic World (made from Sentinel-2 optical) | target to learn from, never an input | yes: labels are not input |
| elevation model | Copernicus GLO-30 | corrects the radar for slopes during preprocessing | yes: the brief lists terrain correction as a preprocessing step; terrain as an input channel was removed |
| sanity check | Esri 10 m LULC | independent comparison, not used in training | yes: the brief asks for exactly this |
| Google Earth screenshot | aerial imagery | visual impression only | not evidence: different date and resolution, not a land-cover map |

For the report: the brief's preprocessing list names **speckle filtering**. The 2nd iteration has no spatial filter, but every monthly composite is the mean of 15-22 scenes; this temporal multilooking is the speckle filter. The 3 × 3 median was dropped because it blurred field edges.

### 9.3 Understanding the data (questions answered during the build)

**What the pipeline does for each site** (`download.py`, `sentinel1.py`):

1. A 10.24 × 10.24 km tile (1024 × 1024 px, 10 m) around the site coordinates, in the local UTM projection.
2. Radar, for each month of 2021: all Sentinel-1 scenes of the month, each corrected for slopes (terrain flattening) and cleaned of layover, shadow and image edges, then averaged. Saved as `s1/<site>.tif`, 24 bands (12 months × VV, VH), uint8 codes.
3. Labels: ESA WorldCover (11 classes) and Google Dynamic World (9 classes) for 2021, translated into our 7 classes (`WORLDCOVER_TO_CLASS`, `DYNAMICWORLD_TO_CLASS` in `config.py`) and sampled onto exactly the same 10 m grid.
4. Agreement: for each pixel, whether both maps give the same class. Saved as `labels/<site>.tif`, 4 bands: agreement, WorldCover, Dynamic World, Dynamic World without snow.

Who does what: ESA made the land-cover map; this project chose the classes, translated WorldCover into them, matched the labels to the radar grid pixel by pixel, computed the agreement and processed the radar. The model then learns to produce such maps from radar alone, which works through clouds and every month.

**The preview images** (`preview.py`, one PNG per site): panel 1 is the radar in January, panel 2 the radar in July (both model input, false colour R = VV, G = VH, B = VV - VH), panel 3 the WorldCover label in our class names and colours (the training target). They are **not** a model result: nothing had been trained yet. The two radar months show how the signal changes with the season while the label stays the same: fields change between bare / snow-covered and grown, forest stays similar, water stays dark. This seasonal difference is what separates crop fields from grassland and forest, and it caused the Black Forest error when a dark winter month looked like water. Bright / pink = buildings, green = vegetation (strongest for forest), dark = water, smooth sand, wet or frozen ground.

**Patches** (`make_patches.py`): 256 × 256 px squares (2.56 km), up to 16 per tile. They are the model's input size (a full tile in a batch of 64 does not fit in GPU memory) and a quality filter: a patch is kept only if at most 60 % of its pixels have no label and every month has radar for at least 95 % of its pixels. Only an index is stored (`patches/<split>_patches.csv`: site, region, site type, row, column), plus `norm_stats.json`, `class_distribution.json` and `site_summary.csv`. Val and test always use the indexed windows; training cuts its squares at random positions around them (±128 px).

**"ignore" in the preview legend** means "no label": WorldCover has no value, mostly over open sea and at data gaps. Those pixels count in neither the loss nor the scores, and patches that are more than 60 % unlabelled are dropped. In the 1st iteration "ignore" also included every pixel where the two maps disagreed (12-40 % of a tile, mostly boundaries); in the 2nd iteration those pixels are trained at half weight.

### 9.4 Observation: roads are missing from the labels

The user noticed in the previews that roads are visible in the radar but only partly present in the WorldCover label.

- Rural roads are 3-8 m wide, narrower than one 10 m pixel. WorldCover assigns each pixel its dominant cover, so most roads disappear into the surrounding trees, grassland or fields; only wide roads appear as Man-made, often with gaps. Tree crowns over roads hide them in the optical imagery WorldCover is made from.
- What the eye sees in the radar is often the corridor (a cut through forest, a tree line, a shadow strip), not the asphalt, which is below the radar's effective resolution of about 20 m.
- Consequence: the model maps wide roads partly and absorbs narrow ones into their surroundings, so rural Man-made area is somewhat underestimated; roads are a small share of rural land, so class percentages and changes barely move. A road the model finds but the label misses counts as an error (slightly pessimistic scores).
- **Decision: accept and document.** Burning OpenStreetMap roads into the labels was rejected (incomplete in rural Africa and Asia, so inconsistent labels between regions; narrow roads stay sub-pixel in the radar, so the model would learn noisy lines; new download and retraining). A separate road class was rejected for the same resolution reason.
- Sentence for the critical discussion: *Roads narrower than the 10 m pixel are mostly absent from the WorldCover labels and below the effective radar resolution (~20 m), so rural Man-made area is underestimated; wide roads are mapped partly.*

### 9.5 Decision: no manual label editing

Question: can the remaining issues be fixed by editing the training labels by hand?

- Better labels raise the ceiling: WorldCover is about 75 % correct overall (shrub and wetland much less), and the model cannot be better than its teacher. Labels could fix class errors the radar can see (a meadow labelled as crop field, a shifted forest edge).
- They cannot fix what the radar cannot see (sub-pixel roads) or input effects (snow, wet soil; handled by the yearly statistics and terrain flattening).
- **Rejected:** 177 tiles × about 1 million pixels; hand edits would follow optical imagery of other dates, differ between regions and sessions, and be hard to describe and reproduce in the report's label-matching section.
- **Instead:** rule-based label fixes only where an error is systematic (already done: agreement weighting, wetland removed, seasonal snow rule), decided after the first training results. Optional for the report: a small hand-checked "gold" test set of 3-5 patches (Black Forest, Latvia, Madagascar) checked against high-resolution imagery, reported next to the WorldCover score (about 1-2 hours; does not change training).

### 9.6 Prototype updated for the 2nd iteration

The prototype could not use 2nd-iteration weights: it built 1st-iteration sigma0 radar (a silent mismatch: wrong maps without an error), recognised models by channel counts that the new variants do not have, used 8 classes in the Esri comparison and had no smoothing.

- New `data_pipeline/sentinel1.py` holds the Sentinel-1 processing and is imported by both `download.py` and `prototype/gee.py`, so training and prototype input cannot drift apart. The `download.py` refactor was verified (identical Earth Engine graph, 0 difference).
- The model card decides everything for 2nd-iteration weights (`"iteration": 2`, `n_months`, 9 statistics, 7 classes, `temporal_smoothing`); 1st-iteration weights still work.
- Verified at `black_forest_villages`, 2021: the prototype's radar input equals the dataset's for both iterations (0 code difference each; the two iterations differ by up to 108 codes). Statistics identical to the notebook; input channels 9 / 12 / 18 for 0 / 1 / 3 months.

### 9.7 Status after the build

- `Dataset_2nd_iteration/` built (177 sites, 3.7 GB), `make_patches.py` and `preview.py` done.
- Next: print the class distribution, check the previews of steep tiles (`nepal_midhills`, `andes_sacred_valley`) and `black_forest_villages`, then run `Notebooks/2nd_iteration.ipynb` with `DEV = True`.

## 10. First 2nd-iteration training: `Iter2_Test_1` (DEV run, 2026-10-31)

`Notebooks/2nd_iteration.ipynb` with `DEV = True`: all three variants, 15 epochs each, on Datalab (RTX A6000, bfloat16, batch 64, all data on the GPU). Outputs in `Weights/Iter2_Test_1/` (`final_model.pt`, `model_card.json`, `test_metrics.json`, plots, `variants/`). The notebook ran end to end without errors.

### 10.1 Data as trained

| Split | Patches | Sites (rural) |
|---|---|---|
| train | 2218 | 139 (114) |
| val | 335 | 21 (18) |
| test | 265 | 17 (13) |

| Class | train % | val % | test % | train agreement % | loss weight |
|---|---|---|---|---|---|
| Man-made | 9.1 | 7.8 | 15.2 | 95.0 | 0.99 |
| Crop fields | 16.0 | 13.3 | 16.5 | 71.9 | 0.75 |
| Trees | 33.9 | 28.6 | 28.1 | 78.0 | 0.51 |
| Grassland | 25.5 | 31.4 | 28.4 | 55.1 | 0.59 |
| Bare ground | 7.4 | 10.2 | 7.5 | 70.7 | 1.10 |
| Water | 5.4 | 4.9 | 2.1 | 96.0 | 1.28 |
| Snow & ice | 2.8 | 3.8 | 2.3 | 93.2 | 1.78 |

- Built-up / Man-made fell from 30.8 % of the 1st-iteration training pixels to 9.1 %; trees and grassland are now the largest classes, as in real rural land.
- 73.6 % of the labelled training pixels are agreement pixels (WorldCover and Dynamic World agree). Grassland has the lowest agreement (55 %): the two maps disagree most on what is grass, shrub or crop.
- Radar channel mean / std (VV, VH, VV-VH): -9.72 / 4.52, -16.93 / 4.86, 7.21 / 2.55 dB. The 9 seasonal statistics took 78 s to precompute; data loading 30 s.

### 10.2 Training

| Variant | Inputs | Best val mIoU | Best epoch | Train-val gap | Minutes |
|---|---|---|---|---|---|
| `year_stats` | 9 | 0.714 | 15 | 0.147 | 2.4 |
| `month_year` | 12 | **0.741** | 12 | 0.106 | 2.0 |
| `three_month_year` | 18 | 0.714 | 14 | 0.141 | 2.2 |

- The train-val gap is 0.11-0.15, against 0.18-0.21 in the 1st iteration: the model generalises better with whole-year input, terrain flattening and the rural sites.
- Validation runs every 2nd epoch (the `nan` in the odd-epoch log lines is not an error). The low val score at epoch 2 (0.005-0.07) comes from the EMA weights, which are not yet warmed up.
- `month_year` stopped improving at epochs 12-15 because the cosine schedule reaches learning rate 0 at epoch 15. The 40-epoch final run should add a little (estimate 0.01-0.02).

### 10.3 Test results

| Variant | OA | mIoU (all test) | Rural mIoU | Rural mIoU smoothed | Rural mIoU, all pixels | Flicker % | Flicker smoothed % |
|---|---|---|---|---|---|---|---|
| `year_stats` | 0.850 | 0.729 | **0.646** | 0.646 | 0.565 | 0.0 | 0.0 |
| `month_year` | 0.861 | 0.714 | 0.639 | 0.641 | 0.534 | 3.0 | 1.2 |
| `three_month_year` | 0.846 | 0.722 | 0.639 | 0.640 | 0.559 | 2.5 | 1.5 |

Rural mIoU = agreement pixels of the 13 rural test sites (the main score, section 6.8); "all pixels" includes the disagreement pixels scored against WorldCover. Flicker = share of pixels whose class changes from one month to the next.

Per class, rural test sites, `month_year` (IoU / precision / recall):

| Class | IoU | Precision | Recall | Main confusion |
|---|---|---|---|---|
| Man-made | 0.72 | 0.83 | 0.85 | 6 % to crop fields |
| Crop fields | 0.66 | 0.81 | 0.78 | 20 % to grassland |
| Trees | 0.82 | 0.89 | 0.92 | 8 % to grassland |
| Grassland | 0.65 | 0.80 | 0.78 | 10 % to trees, 8 % to crop fields |
| Bare ground | 0.83 | 0.89 | 0.93 | 3 % to snow |
| Water | 0.47 | 0.56 | 0.75 | 15 % to grassland; 22 % of predicted water is trees |
| Snow & ice | 0.32 | 0.61 | 0.41 | 48 % to bare ground |

`year_stats` is better on snow (IoU 0.52) and bare ground (0.88); `month_year` is better on man-made, crop fields and grassland.

By region (`month_year`): rural and urban patches together, mIoU 0.46 (Oceania) to 0.62 (Asia); North America has the highest flicker (6.4 %, 2.4 % smoothed). By site type, wetland sites flicker most (13.9 %, 5.1 % smoothed) and snow sites score lowest (mIoU 0.16). Single-type sites score low mIoU because few classes are present; `bare` (mIoU 1.00) is trivial.

### 10.4 Black Forest: the failure of section 1 is gone

The notebook shows 12 monthly maps for four test mosaics (`months_*.png`). For `black_forest_villages`, the test site at the Black Forest location of section 1:

- every month matches WorldCover closely: forest, the grassland clearings along the valleys and the red village in the centre;
- no winter water, wetland or crop fields, no scattered red spots, no blob shapes; the maps barely change from January to December, which is correct for a forest-meadow landscape with no real change;
- the raw and smoothed rows are almost identical; smoothing only removes single-month flips.

This is the most important result of the iteration, because the Black Forest test is the failure that started it. The before (section 1) / after images should be the main evidence in the report, not the mIoU.

### 10.5 Interpretation

- **Do not compare 0.714 with the 1st-iteration 0.742.** The 1st-iteration test had 8 classes and 54 % built-up pixels, scored at IoU 0.94-0.96, which lifted its mIoU. The test set, classes and labels all changed. The score that matches the product goal is the rural mIoU (0.639), which the 1st iteration did not measure.
- **Choice of `month_year` is correct**, although `year_stats` is 0.007 higher on rural test. `year_stats` has no monthly input, so it produces the same map in all 12 months (flicker 0 by construction) and cannot show monthly change, which the prototype must show. The final model is chosen on val, and test is only reported (section 6.9); `month_year` had the best val mIoU.
- **Temporal smoothing:** flicker 3.0 % → 1.2 % at no cost in mIoU (0.639 → 0.641). Kept on for the prototype (`temporal_smoothing` in the model card).

### 10.6 Remaining weaknesses

| Weakness | Evidence | Likely cause |
|---|---|---|
| Snow vs bare ground | Snow IoU 0.32, 48 % of snow predicted as bare | Dry snow and rock both scatter weakly; snow is 2.3 % of test pixels |
| Water precision | 0.56; 22 % of predicted water is trees | Dark radar shadow behind slopes or wet forest; water is only 2.1 % of test pixels, so a few false patches cost a lot of IoU |
| Crop fields vs grassland | 20 % of crop fields predicted as grassland | Similar radar response outside the growing season; labels agree on only 55 % of grassland |
| Wetland sites | Flicker 13.9 % | Real seasonal flooding, now mapped as water or grassland month by month |

These are physical limits of C-band radar and of the labels, not pipeline errors; they go into the critical discussion.

### 10.7 Next steps

1. Final run: `DEV = False`, `FINAL_VARIANT = "month_year"`, `RUN_NAME = "Iter2_Final"` (up to 40 epochs).
2. Prototype: load `Weights/Iter2_Final/final_model.pt` with its model card; repeat the Black Forest test (section 1) away from the training tile, with the Esri sanity check, and on 3-4 other rural mosaics.
3. Use the Black Forest before/after images and the tables of 10.3 in the report.

## 11. Repeat DEV run and the full run (2026-10-31)

### 11.1 Repeat DEV run: how large is the noise?

After a stuck kernel, the DEV run was repeated by accident with identical code, data and seed (`Iter2_Test_1_20261031_144820`). The notebook ran end to end; the new progress lines in 9.2 and 9.4 worked and the 12-month test took 3-14 s per pass (the earlier hang was the kernel or the browser connection, not the code).

| Variant | Val mIoU, run 1 → run 2 | Rural test mIoU, run 1 → run 2 | Flicker %, run 1 → run 2 |
|---|---|---|---|
| `year_stats` | 0.714 → 0.717 | 0.646 → 0.643 | 0 → 0 |
| `month_year` | 0.741 → 0.723 | 0.639 → 0.624 | 3.0 → 2.7 |
| `three_month_year` | 0.714 → 0.715 | 0.639 → 0.643 | 2.5 → 2.5 |

- Identical runs differ by up to 0.02 mIoU (GPU non-determinism, random sampling and augmentation). This is as large as the differences between the variants, so one 15-epoch run cannot rank them. This repeats the 1st-iteration finding (differences below about 0.05 test mIoU are noise).
- `month_year` varies the most (in run 2 its val score peaked at epoch 10 and fell to 0.709 by epoch 15); `three_month_year` was the steadiest. `month_year` had the best val mIoU in both runs and was kept as `FINAL_VARIANT`.

### 11.2 Full run: `Iter2_Test_1_Full`

Settings: `DEV = False`, `FINAL_VARIANT = "month_year"`, up to 40 epochs, patience 10. Outputs in `Weights/Iter2_Test_1_Full/` (`final_model.pt`, `model_card.json`, `test_metrics.json`, plots; archive `Iter2_Test_1_Full.zip`).

**Training**
- Best val mIoU **0.732** at epoch 22; early stop at epoch 32 (no improvement for 10 epochs); 4.7 minutes.
- Train-val mIoU gap at the best epoch 0.122 (1st iteration 0.18-0.21).
- Val loss falls to 0.530 at epoch 22 and rises slowly to 0.568 by epoch 32 while the train loss keeps falling: mild overfitting after epoch 22. Early stopping and the saved best epoch handle it.

**Test (12-month evaluation)**

| Score | DEV run 1 (15 epochs) | DEV run 2 (15 epochs) | **Full run** |
|---|---|---|---|
| Val mIoU | 0.741 | 0.723 | 0.732 |
| Test OA, all sites | 0.861 | 0.835 | **0.872** |
| Test mIoU, all sites | 0.714 | 0.699 | **0.750** |
| **Rural test mIoU** (main score) | 0.639 | 0.624 | **0.679** |
| Rural mIoU, smoothed | 0.641 | 0.625 | **0.680** |
| Rural mIoU, all WorldCover pixels | 0.534 | 0.528 | **0.597** |
| Flicker % (raw → smoothed) | 3.0 → 1.2 | 2.7 → 1.1 | **2.4 → 0.9** |

The longer schedule added about 0.04-0.055 rural test mIoU over the two DEV runs: more than the noise level, so it is a real gain. The cosine schedule over 40 epochs keeps the learning rate higher for longer than the 15-epoch schedule did.

Per class, rural test sites (agreement pixels):

| Class | IoU | Precision | Recall | DEV run 1 IoU |
|---|---|---|---|---|
| Man-made | 0.747 | 0.823 | 0.891 | 0.721 |
| Crop fields | 0.656 | 0.808 | 0.776 | 0.660 |
| Trees | 0.833 | 0.917 | 0.901 | 0.824 |
| Grassland | 0.682 | 0.793 | 0.829 | 0.652 |
| Bare ground | 0.894 | 0.965 | 0.923 | 0.826 |
| Water | 0.492 | 0.592 | 0.744 | 0.471 |
| Snow & ice | 0.449 | 0.602 | 0.639 | 0.322 |

Every class except crop fields (unchanged) improved; snow & ice gained the most (+0.13), and bare ground +0.07.

By region (all test patches): Asia 0.656, Europe 0.654 (OA 0.960), South America 0.584, North America 0.569 (highest flicker, 4.7 % → 1.8 % smoothed), Africa 0.492, Oceania 0.441. By site type: mosaic 0.462, cropland 0.449, urban 0.485 (OA 0.920), trees 0.377, shrub_grass 0.233, snow_ice 0.222 (OA 0.545), wetland 0.223 (flicker 10.1 % → 3.6 %). Per-site-type mIoU is low partly because each group contains few classes.

**Monthly maps of unseen test mosaics**
- `black_forest_villages`: forest, valley meadows and the village are mapped as in WorldCover in all 12 months, with no seasonal false classes. The failure of section 1 stays fixed in the final model.
- `willamette_foothills` (Oregon farmland): the radar changes strongly over the year (fields bright in winter, dark in summer after harvest), but the maps stay stable: the crop field blocks, forest strips and grassland keep their shape every month. Remaining errors: part of the WorldCover crop fields (top right) mapped as grassland, and a few small water patches appear in August only, which smoothing reduces.

### 11.3 Comparison with the 1st iteration

| | 1st iteration (Test_6, `Notebooks/1st_iteration.ipynb`) | 2nd iteration (`Iter2_Test_1_Full`) |
|---|---|---|
| Goal | general LULC, city-heavy | rural land worldwide |
| Sites | 176; 79 city centres (45 %) | 177; 54 rural mosaics, 10 city edges, 20 cities kept |
| Classes | 8 (Built-up, Cropland, Trees, Shrub & grass, Bare / sparse, Water, Wetland & mangrove, Snow & ice) | 7 (Man-made, Crop fields, Trees, Grassland, Bare ground, Water, Snow & ice) |
| Built-up share | 30.8 % of train, 54.2 % of test pixels | 9.1 % of train, 15.2 % of test pixels |
| Radar | sigma0, 3×3 median filter | gamma0 terrain flattening, layover/shadow masked, no spatial filter |
| Labels | consensus only (disagreement pixels ignored) | WorldCover on all pixels, disagreement weighted 0.5 |
| Input | 3 months + optional PCA (6), 5 statistics, terrain (3) | target month + 9 order-independent seasonal statistics (12 channels) |
| Variants trained for the final run | 5 (40 epochs each, 27 min) | 1 (4.7 min) |
| Delivered weights | one `<variant>_best.pt` per variant | one `final_model.pt` |
| Best val mIoU | 0.657 (`three_month_stats`) / 0.671 (`three_month_pca_stats`, selected) | 0.732 |
| Test mIoU, all test sites | 0.742 (`three_month_stats`) / 0.617 (selected) | 0.750 |
| Rural test mIoU | not measured | 0.679 |
| Flicker between months | not measured; Black Forest showed seasonal water / wetland / crop errors | 2.4 % raw, 0.9 % smoothed |
| Train-val gap | 0.18-0.21 | 0.12 |
| Black Forest monthly maps | wrong in winter (water, wetland, crop, red spots, blobs) | stable and correct in all 12 months |

**How to read the scores.** The test sets, classes and labels differ, so the mIoU values are not a like-for-like comparison. The 1st-iteration 0.742 was lifted by Built-up (54 % of test pixels, IoU 0.95). The 2nd iteration reaches a slightly higher mIoU (0.750) on a test set with only 15 % man-made pixels and a harder rural mix, and scores 0.679 on the rural sites alone. Built-up was the 1st iteration's easiest class, and it is now a small one.

**What the changes did, step by step**

1. **PCA dropped.** In Test_6 the variant without PCA (`three_month_stats`) was best on test by 0.075-0.125 mIoU. PCA loadings are tied to calendar months, so they shift with the hemisphere and the site; the val split could not separate the PCA variants from the statistics variant (0.657-0.671) and picked the weakest one on test. The 2nd iteration uses only order-independent seasonal statistics (9 instead of 5: VV/VH/VV-VH mean, 2nd lowest and 2nd highest VV and VH, VH std, water months), which avoids both the calendar problem and the selection trap.
2. **Terrain channels dropped, terrain flattening added.** The 1st-iteration terrain channels (DEM, slope, HAND) broke the Sentinel-1-only rule and were only an ablation. The 2nd iteration corrects terrain inside the radar processing (gamma0), which is allowed: VV-slope correlation fell from 0.74-0.82 to -0.14-0.05. This removed the hillside "buildings" and "water" in mountains.
3. **Yearly context in every variant.** In Test_6, yearly features were the biggest lever (+0.09-0.12 val mIoU over plain `three_month`). Now every variant sees the whole year through the statistics, and the month only adds what changes. This removed the winter flips of the Black Forest test.
4. **Rural sites and fewer cities.** Built-up fell from 30.8 % to 9.1 % of the training pixels; trees and grassland are now the largest classes, as in real rural land. Man-made still scores 0.747 IoU on rural test sites, so villages are not lost.
5. **Labels on all pixels.** Disagreement pixels (mostly boundaries) are now trained with weight 0.5 instead of ignored, and the 3×3 median is gone. The maps show sharper edges and no blob shapes.
6. **Wetland removed, classes renamed.** Wetland had the worst labels (30-45 % correct) and was a main winter false alarm in the Black Forest test.
7. **One model.** The final run trains one variant and delivers one file. Overfitting fell (gap 0.12 vs 0.18-0.21), and the run takes 4.7 minutes.
8. **New evaluation.** Rural mIoU, the 12-month test, flicker and temporal smoothing measure what the prototype shows. The 1st iteration only scored one random month per patch.

### 11.4 What did not improve

- **Water** (IoU 0.49, precision 0.59): false water in dark forest or radar shadow; water is 2.1 % of the test pixels, so a few false patches cost a lot. In the 1st iteration only the terrain ablation (HAND) fixed water (0.868), and terrain channels are not allowed.
- **Snow & ice** (0.45): better than in the DEV runs (0.32), but snow is still confused with bare ground. The 1st-iteration PCA variant scored 0.813 on snow, because PCA encodes the calendar pattern of snow cover; the order-independent statistics lose part of this.
- **Crop fields vs grassland**: about 20 % of crop fields are mapped as grassland; the two label maps agree on only 55 % of grassland.
- **Wetland sites** flicker most (10.1 %): partly real seasonal flooding, which an annual label counts as error.
- **Single run:** the full run is one seed. From 11.1, its scores carry about ±0.02 noise.

### 11.5 Next steps

1. Prototype: load `Weights/Iter2_Test_1_Full/final_model.pt` with its `model_card.json`; repeat the Black Forest test (section 1) at the original clearing, with the Esri sanity check, and on 3-4 other rural mosaics.
2. Report: Black Forest before / after images (section 1 vs `months_black_forest_villages.png`), the comparison table of 11.3, the noise result of 11.1.
