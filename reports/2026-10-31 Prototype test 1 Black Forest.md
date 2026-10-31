# 2026-10-31 Prototype test 1: Black Forest

Review of the first rural prototype test (`Documentation/Images/Prototype_test_1/`): Google Earth view of a Black Forest clearing, `lulc_12_months.png` (`three_month_stats`) and `lulc_12_months_pca.png` (PCA variant), 2021, January start. Sections 1-5 analyse the failure; section 6 records the decisions taken because of it for the 2nd iteration, section 7 explains the training philosophy (variant names, months, PCA, terrain, one final model), and section 8 the status.

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
3. Update the prototype for the 2nd iteration: gamma0 composites without median in `prototype/gee.py`; the new model card (9 statistics, 0/1/3 months, 7 classes, smoothing) in `prototype/pipeline.py`.
4. Repeat this Black Forest test with the new model and compare.
