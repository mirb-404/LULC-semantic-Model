# 4th iteration and the label disagreement finding (2026-11-05)

**Main finding: the global land-cover maps disagree with each other about what pasture, fallow farmland, rangeland and sparse vegetation are, and this, not the network, limits the classes Crop fields, Grassland and Bare ground.**
- Adding training data in the 4th iteration did not change overall mIoU. Instead, the classes traded places.
- The prototype runs show the cause: on the same farmland the maps differ by up to 95 percentage points.
- The 5th iteration tests the fix: training labels from a majority vote of three maps.

Notebooks:
- `Notebooks/4th_iteration_multiyear.ipynb`, executed on Datalab (RTX PRO 6000, 102 GB);
- `Notebooks/5th_iteration_label_vote.ipynb`, built, not yet run.

Data pipeline: `data_pipeline/` (`build_dataset.py`, `config.py`). Earlier context: `reports/2026-11-05 3rd iteration experiment results.md`, `reports/Future scope.md`.

## 1. What the 4th iteration added

Motivation: prototype runs of `Iter2_Test_1_Full` showed three problems.
- **Madagascar highlands:** eroded laterite was mapped as Grassland and Crop fields.
- **Crop fields** were confused with Grassland.
- **Bright rock** was taken for Man-made, and **Water** appeared on dark land.

| Addition | Detail |
|---|---|
| 17 planned new training sites | bare ground outside deserts, smallholder fields, bare rock, 4 pure-class tiles (one class ≥ 60 %); **14 downloaded** (section 4) |
| radar of 2022-2024 | every training site except towns and glaciers; WorldCover 2021 as the label, Dynamic World of the same year for agreement |
| Dataset | 2021: 2440 training windows from 153 sites (before 2218 / 139); 2022-2024: 5148 windows; val 335 / test 265 unchanged; 10.35 GB |
| Class shares, 2021 training | Bare ground 7.4 → 9.0 %, Crop fields 16.0 → 17.0 %; every class grew in absolute terms (Bare ground +34 %, Crop fields +17 %) |
| Sampling | a window present in several years shares its weight between them, so the later years add variety but not class weight |

Rough total amount per class with all years, as window-equivalents (windows × share): Crop fields about 3.6×, Trees 3.4×, Grassland 3.3×, Bare ground 2.7×, Water 2.9×, Man-made 1.5×, Snow & ice 1.1× (towns and glaciers have no later years).

## 2. Results (2 seeds each; val and test unchanged, scored on WorldCover pixels confirmed by Dynamic World)

| Experiment | val mIoU | rural test mIoU | rural, all pixels | train-val gap |
|---|---|---|---|---|
| Y0 reference (= `Iter2_Test_1_Full` setup) | 0.735 | 0.663 | 0.585 | 0.125 |
| Y1 new sites | 0.739 | 0.658 | 0.549 | 0.095 |
| Y2 later years | 0.734 | 0.656 | 0.548 | 0.092 |
| Y3 both | 0.738 | 0.659 | 0.557 | 0.106 |

Rural test IoU per class:

| Class | Y0 | Y1 | Y2 | Y3 |
|---|---|---|---|---|
| Man-made | 0.780 | 0.779 | 0.709 | 0.723 |
| **Crop fields** | 0.632 | 0.660 | 0.665 | **0.696** |
| Trees | 0.802 | 0.823 | 0.814 | 0.835 |
| **Grassland** | 0.632 | 0.648 | 0.651 | **0.681** |
| Bare ground | 0.889 | 0.836 | 0.881 | 0.838 |
| Water | 0.504 | 0.474 | 0.467 | 0.495 |
| Snow & ice | 0.404 | 0.387 | 0.408 | 0.344 |

- **Overall: no change beyond noise.** Every val and rural difference is ≤ 0.007; the reference itself varies ±0.006 on val and up to 0.03 on rural test. By the decision rule, `Iter2_Test_1_Full` stays.
- **The classes traded places.**
  - Crop fields +0.064 and Grassland +0.049 (the target of the new data);
  - Bare ground -0.051, Man-made -0.057, Snow & ice -0.060.
  - mIoU averages the classes, so the gains and losses cancel.
- **More data reduced overfitting:** the train-val gap fell from 0.125 to 0.09-0.11.
- **On all pixels the new configurations are worse** (0.585 → 0.55). They follow WorldCover less where the two maps disagree.
- **Best run per region** (Y1, seed 42):
  - Africa 0.544 (`Iter2_Test_1_Full` 0.492), Oceania 0.483 (0.441), North America 0.589 (0.569);
  - Asia 0.621 (0.656), Europe 0.640 (0.654).
  - Single run, so indicative only.
- **Class bias** (offset on the log-probabilities, tuned on val): no offset for Water or Man-made improved val mIoU. Water precision stays 0.56 at recall 0.74. The false Water pixels are places where radar cannot tell water from smooth, dark ground, so a threshold cannot fix them. The bias is switched off in the final and 5th-iteration notebooks.

This is not an architecture problem: the network did not change, and the 3rd iteration showed five encoders giving the same flat result. The pattern is a data and label trade-off.

## 3. The finding: the maps define the same land differently

Prototype runs with the 4th-iteration model (`prototype/runs/20261105_*`). Model = most frequent class over 12 months; reference = Esri 10 m Annual LULC 2021 (Sentinel-2 based, Impact Observatory).

| Site | Model | Esri | Agreement | Majority baseline |
|---|---|---|---|---|
| **Kano farmland, Nigeria** (test site, chosen as cropland) | 81 % Crop fields, 10 % Man-made | **95 % Grassland ("rangeland"), 0 % crops** | **13 %** | 95 % |
| **Chilean lakes** (test site, pasture) | 74 % Grassland | **64 % Crop fields** | 40 % | 64 % |
| Madagascar highlands (test site) | 69 % Grassland, 16 % Crop fields | 77 % Grassland, 18 % Crop fields | 69 % | 77 % |
| Black Forest (near the test site) | 57 % Trees, 40 % Grassland | 56 % Trees, 37 % Grassland | 87 % | 56 % |

**Kano:**
- It lies in the "close-settled zone", one of the most densely farmed regions of Africa.
- WorldCover (our label) calls it farmland and the model says crop; Esri calls it 95 % rangeland. Dry-season Sahel fields look like rangeland in optical images.
- Here the model is likely right and the reference wrong.

**Chile** is the mirror image: Esri counts pasture as crops, while WorldCover and the model call it Grassland.

**Madagascar:** the class shares match Esri closely, but pixel by pixel the model is below the one-class baseline. The model finds the right classes in slightly wrong places.

**The training labels show the same conflict:**
- **WorldCover vs Dynamic World agreement** on the training data (2021 + later years): Grassland 55.6 %, Bare ground 64.3 %, Crop fields 70.2 %, against Water 95.3 %, Snow & ice 93.6 %, Man-made 93.4 %, Trees 82.7 %.
- **The classes the prototype gets "wrong" are exactly those with the least agreement.**
- **New sites that WorldCover labels differently from what a photo suggests** (skipped by the class-share check before download):

| Site | Intended class | WorldCover |
|---|---|---|
| `betsiboka_eroded_hills` (Madagascar) | Bare ground | **94 % Grassland**, 0.3 % bare |
| `lake_alaotra_rice` (Madagascar) | Crop fields | **97 % Grassland** (flooded rice = herbaceous wetland → Grassland) |
| `pilbara_iron_mines` (Australia) | Bare ground | **97 % Grassland** |
| `mbale_hill_farms` (Uganda) | Crop fields | 5 % crop |
| `tigray_degraded_highland` (Ethiopia) | Bare ground | 61 % crop: kept as a crop site |

- **Consequence:** the model labels Madagascar's eroded hills as Grassland because its training labels do. More data with the same labels cannot change that; new data only moves the boundaries between classes whose labels disagree, which is what section 2 shows.

**Small built-up areas:**
- Sentinel-1 resolves about 20 m, so single houses and narrow roads are smaller than one radar pixel.
- WorldCover's built-up class also misses many small rural buildings.
- In Kano the model still found 89 % of Esri's built pixels.
- This is a resolution limit, not a label error.

## 4. Second finding: Sentinel-1 coverage after 2021

After Sentinel-1B failed (December 2021), Sentinel-1A stopped imaging large regions.
- **All 12 months of 2022-2024 have no scene** at: Sahara (Algeria, Western Desert), Namib, Etosha, Matopos, Gezira, Llanos, Gran Chaco, Serra Gaúcha, Hudson Bay, Sikhote-Alin.
- **More than 2 empty months:** Mongolia and Paraná in some years.
- 106-110 of about 127 eligible sites have later-year data.
- **Consequences:**
  - the later years add the least where the model is weakest (Africa, South America);
  - the prototype gets no radar there for 2022-2024 starts;
  - Sentinel-1C restores coverage from 2025.

## 5. The fix: a three-map majority vote (5th iteration)

**Idea:** a third map decides most two-way disputes.
- WorldCover (Sentinel-1 + 2, ESA), Dynamic World (Sentinel-2, Google) and Esri (Sentinel-2, Impact Observatory) of the radar year are combined.
- The training label is the class at least two maps agree on; those pixels count fully.
- Where all three differ, the WorldCover class stays at weight 0.5, as before.
- The model input stays Sentinel-1 only.

Rule in `data_pipeline/config.vote_labels`, with Dynamic World's seasonal snow corrected as before:

| WorldCover | Dynamic World | Esri | Label | Weight |
|---|---|---|---|---|
| A | A | any | A | 1 |
| A | B | A | A | 1 |
| A | B | B | **B** | 1 |
| A | B | C | A | 0.5 |

**Implementation:**
- `download.py` downloads the Esri map of each site-year (`esri[_<year>]/`).
- `make_patches.py` writes `votes[_<year>]/` (band 1 = class where ≥ 2 maps agree, band 2 = training label) and `patches/agreement.json` (agreement per class: 2 maps vs vote). The patch windows do not change.
- `Notebooks/5th_iteration_label_vote.ipynb`:

| Experiment | Labels | Data |
|---|---|---|
| V0_worldcover | WorldCover + Dynamic World (= `Iter2_Test_1_Full`) | 2021 |
| V1_vote | three-map vote | 2021 |
| V2_vote_all_data | three-map vote | 2021-2024 + new sites |

- 2 seeds each. Every run is scored three ways:
  - WorldCover labels (comparable with all earlier runs);
  - vote labels (the decision: val vote mIoU, more than 0.02 over V0);
  - **tolerant**: a prediction counts if it matches any of the three maps, which measures how much of the "error" is map disagreement.
- **Prototype:** the reference check now shows WorldCover 2021 and Dynamic World of the analysis year next to Esri (agreement, majority baseline and map overlay for each), so a definition conflict such as Kano can be seen as one.

**Trade-off:** Esri becomes part of the training labels, so for a vote-trained model the prototype's Esri check is no longer independent. WorldCover and Dynamic World are shown as well for that reason, and the report must state it.

**Commands (Datalab, repository root):**
```
git pull
nohup python data_pipeline/build_dataset.py --workers 8 > build3.log 2>&1 & tail -f build3.log
```
Only the Esri tiles download (radar and labels exist); then patches, votes, manifest and zips are rebuilt. After that, run `Notebooks/5th_iteration_label_vote.ipynb`.

## 6. For the report's critical discussion

1. **The reference decides the score.** On the same farmland, WorldCover and Esri disagree by up to 95 percentage points (Kano). Any accuracy figure is accuracy against one map's definition; agreement with Esri must be read next to the majority baseline and the other maps.
2. **Label noise is concentrated in the rural classes the product targets:** Grassland (55.6 % agreement), Bare ground (64.3 %), Crop fields (70.2 %). These classes set the ceiling, not the network: five encoders, two optimisers and four label-handling methods changed nothing beyond noise.
3. **More data with the same labels shifts the class boundaries but does not raise the ceiling** (4th iteration: Crop fields +0.06, Bare ground -0.05, mIoU unchanged, overfitting lower).
4. **Sentinel-1 coverage is uneven over time:** 2022-2024 have no scenes over large parts of Africa and South America.
5. **Resolution:** single buildings, narrow roads and small streams are below the radar's 20 m resolution.

## 7. Exam compliance check (2026-11-05)

Checked against `Documentation/requirements_info.txt`.

| Rule | Status |
|---|---|
| Sentinel-1 as the **only model input** | Kept. Inputs: monthly VV, VH, VV-VH and 9 statistics computed from them (`prototype/pipeline.py`, all notebooks). The DEM (GLO-30) is used only inside the radar preprocessing (terrain flattening, layover/shadow mask), a step the brief lists itself ("terrain correction"). The 1st-iteration terrain-channel variant was an ablation, never delivered; the prototype warns if such weights are loaded. |
| Labels from other sources | Allowed: the brief limits the *input*, not the labels. WorldCover, Dynamic World and Esri only produce training targets or loss weights. |
| Prototype: lat/lon + start month → 12 monthly maps, % per class, month-to-month and total change in pp, map visualisation | Met (`prototype/app.py`). Year and area size are extra inputs. |
| Satellite basemap in the prototype | Display only (Esri World Imagery tiles, labelled as a static mosaic); never an input to the model. |
| **Case study checked against an independent LULC map not used for training** | **Met for `Iter2_Test_1_Full` and every WorldCover-trained model: Esri is independent.** The prototype marks each reference: "Esri (independent)", "WorldCover 2021 (training label)", "Dynamic World (agreement weights in training)". **Not met by Esri for a model trained on the 3-map vote** (5th iteration): Esri is then a training label and the prototype labels it "used in training". If a vote model became the final model, the case study would need another independent map. A candidate is GLC_FCS30D (30 m, Landsat-based, annual to 2022), but its Earth Engine asset has not been checked. Otherwise keep a WorldCover-trained final model. |
| Submission: dataset link, notebooks with outputs and training curves, prototype source + README (accounts: Earth Engine) + requirements, weights, urban screen recording | Dataset zips built (10.35 GB / 3.82 GB, link needed); notebooks with outputs (2nd, 3rd, 4th); README and requirements updated; weights link and recording still to do (`TODO.md`). |
| Report contents (data, preprocessing, model, OA/mIoU/per-class IoU-F1, curves, variant comparison, prototype, case study, critical discussion) | Material exists in `reports/`; the critical-discussion points are in section 6 above. |
