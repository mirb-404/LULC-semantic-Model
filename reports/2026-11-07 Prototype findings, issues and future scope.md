# Prototype findings, issues and future scope (2026-11-07)

Model: `Weights/Iter2_Test_1_Full` (U-Net ResNet-34, `month_year` variant, temporal smoothing over t-1, t, t+1; test rural mIoU 0.679, test mIoU 0.750). The 5th iteration (label vote) did not beat it on a label set it was not trained on (`reports/2026-11-06 5th iteration label vote results.md`), so it stays the final model.

## 1. Verdict

- The prototype meets the brief: lat/lon and a start month in, 12 monthly LULC maps out, class percentages per month, month-to-month change and the total change from the first to the last month (in percentage points), and a sanity check against an independent map not used for training (Esri 10 m Annual LULC).
- The model is good for a radar-only model. It is stable over time and close to the ceiling set by its labels. It is not a replacement for optical maps.
- The UI does not change the predictions. Smoothing is part of the declared model (`model_card.json`), the class bias is off, and the reference maps and photos are display only.

## 2. Findings

### 2.1 Case study: Willamette foothills, USA, 2021 (rural test site)

Run: `prototype/runs/20261106_175722_44.9000_-122.7000_2021-01/` (512 x 512 px = 5.1 km, 9-14 Sentinel-1 scenes per month, 100 % valid pixels).

| Reference map | Agreement with the model | Majority-class baseline |
|---|---|---|
| Esri 2021 (independent) | 68.1 % | 51.3 % |
| WorldCover 2021 (training label) | 74.8 % | 57.6 % |
| Dynamic World 2021 (agreement weights in training) | 74.7 % | 53.2 % |

- The model beats the majority-class baseline on every reference: +17 points on the independent Esri map.
- Agreement with Esri (68 %) is about the same as the agreement between the label maps themselves (WorldCover and Dynamic World agree on 73.6 % of training pixels). The model is about as close to Esri as the reference maps are to each other.
- Per class against Esri (share of Esri's pixels the model also finds): Trees 98.5 %, Water 84.1 %, Man-made 52.8 %, Crop fields 43.4 %, Grassland 4.8 %.
- **Main disagreement: Crop fields vs Grassland.** Esri calls 51 % of the area crops and 2 % grassland; the model says 22 % crops and 24 % grassland. Pasture, hay and grass seed fields (common in the Willamette Valley) look the same to radar and are labelled differently by the three maps. This is the confusion found in every iteration.
- **Stability:** the largest month-to-month change is 0.43 pp (Crop fields, Sep → Oct). The total change Jan → Dec is +0.23 pp Crop fields, −0.16 pp Grassland, −0.06 pp Trees. Flicker is 0.26-0.66 % of pixels per month. No fake land-use change from the seasons.
- Water peaks in August (0.78 % vs 0.61-0.66 % otherwise). The site is small, so this may be a low reservoir edge or speckle on dry fields; it is too small to call a real change.

### 2.2 Seasonal snow (Black Forest, 2021)

- The monthly Sentinel-2 photo shows snow in December, January and February (and March). The model map stays Trees and Grassland.
- **This is intended, not a bug.** The labels are annual land cover. Seasonal snow was removed from the labels on purpose (`data_pipeline/config.py`, `consensus_labels` and `vote_labels` replace Dynamic World's yearly snow with its snow-free class where WorldCover is not snow). Snow & ice means permanent ice and glaciers, trained on glacier sites only.
- Radar physics supports it: C-band sees through dry snow to the ground; wet snow looks dark, like water, not like ice.
- **Why this is correct for the product:** the brief asks for land-use change, and the report must discuss "seasonal effects that can look like real land-use change". A map that switched forest to snow every winter would show about −40 pp Trees in January and +40 pp in March: fake change. The radar model gives stable land cover through the winter while the optical photo turns white. This is the strongest case-study example for the seasonal-effects section.
- Seasonal snow cover is weather, not land use, and is therefore out of scope (future scope, section 4).

### 2.3 Photos for checking the map by eye

**Why Sentinel-2 is shown next to a radar-only model:**
- **Radar is hard to read by eye.** Sentinel-1 shows grey backscatter (VV/VH): bright = rough surfaces and buildings, dark = smooth water or wet snow. It does change from month to month (`radar_12_months.png`), but a user cannot tell forest from fields in it without training.
- **Sentinel-2 is true colour.** It shows the land as the eye sees it: forest green, fields brown, snow white. The user can check each monthly map against what the land looked like.
- **The basemap alone was not time-matched.** The Esri photo is a single image, often years old (2017 for a 2021 run at the Black Forest), and it does not change with Play, so places looked fixed. One Sentinel-2 photo per analysis month moves with the radar and the model map.
- **Never a model input.** The exam requires Sentinel-1 only, and Sentinel-2 fails under cloud and smog while radar does not. Radar is used for the model, the photo for display.

In short: Sentinel-2 true colour is shown for display only, because radar backscatter is hard to read by eye and the Esri basemap is not time-matched. It lets the user check each monthly map against the land surface of the same month.

- **Outside the analysis box:** Esri World Imagery Wayback, the last release before the end of the analysis year, with its capture date. Esri rarely updates rural areas: at the Black Forest the photo is from 2017 for a 2021 run, from 2013 for 2018, and from 2020 for 2025. The caption says when the photo is older than the analysis year.
- **Inside the box:** one cloud-free Sentinel-2 median per analysis month (Cloud Score+ `cs_cdf >= 0.6`). It steps with the month slider and Play, so radar, model map and photo always show the same month. Cloudy parts are transparent and the caption gives the cloud share.
- Before this change, the basemap was today's imagery while the model ran on 2021 radar. Some areas then looked wrong although the land had changed since 2021. The time-matched photo removes that confusion.
- **Disclaimer:** the photos are for orientation only. They can be old (Wayback), cloudy, or hazy from city smog (Sentinel-2, seen at Thane; section 3.2). None of this changes the model map, which uses radar only.
- Cost: about 14-15 s extra per 5 km run. Display only, never a model input, so the radar-only rule is kept.

## 3. Issues and limits

### 3.1 Model

| Issue | Evidence | Effect in the prototype |
|---|---|---|
| Crop fields vs Grassland confusion | Willamette: 22 vs 51 % crops (model vs Esri); label maps disagree on 30-44 % of pixels in these classes | Crop/Grassland shares are the least reliable numbers; changes between them under about 1 pp are noise |
| Water weaker than expected | Rural IoU about 0.49 | Small rivers and mixed 10 m pixels are missed; windy or flooded areas can flip |
| Narrow waterways missed (seen at Berlin) | Canals and smaller river arms are mapped as Man-made or Trees, not Water | Label limit first: WorldCover and Dynamic World often mark canals under about 20-30 m wide as built-up or as the bank class, so the model learned that. Radar adds to it: a canal 1-2 pixels wide is a mixed pixel, and quay walls, bridges and buildings give bright double-bounce returns that hide the dark water. Roads are not a class (they are part of Man-made). Water shares in cities are therefore too low; large lakes and wide rivers are found |
| Snow & ice weak | Rural IoU about 0.35 | Only glaciers; seasonal snow is not mapped (by design, section 2.2) |
| Weak regions | Africa 0.49, Oceania 0.44 rural mIoU (Oceania = 1 test site) | Lower trust outside Europe, Asia and the Americas |
| Scored on agreement pixels | Rural mIoU 0.679 on pixels where WorldCover and Dynamic World agree; about 0.55 on all WorldCover pixels | Report both numbers; real-world accuracy is closer to the lower one on mixed land |
| Classification errors inflate change | Every pixel that flips adds to the change, true or not | Smoothing keeps flicker under 1 %; small changes (under about 1 pp) should not be read as real |

### 3.2 Data and inputs

- **Sentinel-1B failed in December 2021.** From 2022 there are about half the scenes per month, and some sites have none (5 sites skipped in 2024, e.g. Etosha, Gezira). The prototype shows the scene count per month and warns on empty months. The model was trained on 2021 only.
- **Reference maps:** Esri covers 2017-2025 (other years use the nearest year); WorldCover exists only for 2021 (the 2020 v100 map uses a different algorithm). For other years, disagreement can be real land change, not model error.
- **Wayback photos can be years old** in rural areas (section 2.3). The Sentinel-2 photo inside the box is the time-matched view.
- **Sentinel-2 photos:** months with persistent cloud (winter in Europe, monsoon in South Asia) can be mostly transparent.
- **Smog and haze in the Sentinel-2 photos (tested on Thane, India):** over large cities the photo can look grey and washed out from air pollution, especially in the dry winter months. Cloud Score+ masks clouds, but thin haze and smog are often kept as "clear", so the median photo stays hazy and colours are hard to read. This affects only the display photo. The model input is Sentinel-1 radar, which passes through smog, haze and cloud, so the predictions are not affected. A hazy photo is not a sign of a wrong map.

### 3.3 Operation

- **Earth Engine authentication on Datalab:** `earthengine authenticate` failed through gcloud. `EE_PROJECT` must be set. Fix used: export `EE_PROJECT`, run `python check_auth.py`, then restart Streamlit (`pkill -f "streamlit run"; python launch.py`); an old Streamlit process keeps the old credentials.
- **The project ID must stay on the server.** It is read from `EE_PROJECT` or the git-ignored `.env` and never sent to the browser.
- `Weights/Iter2_Test_1_Full` is not yet on the Datalab machine.
- The month-synced photo is tested headless only (Streamlit AppTest: months 0, 5, 11, Play, Streets, photo off). A real browser run on Datalab is still open.
- Run time: about 1-2 min per 5 km run on GPU, most of it Earth Engine downloads.
- Urban areas (e.g. Thane) work but are not the product goal; the dataset is 66 % urban sites while the product targets rural land.

## 4. Future scope

Prototype-specific (model and training ideas: `reports/Future scope.md`):

- **Seasonal snow as a separate layer**, not as a land-cover class: wet-snow detection from Sentinel-1 (backscatter drop against a snow-free reference month) or a Sentinel-2 snow index, shown next to the LULC map. It would explain winter months without creating fake land-use change.
- **Confidence map:** show the maximum softmax per pixel and grey out low-confidence pixels. Report change only where both months are confident, so classification errors inflate the change less.
- **Change significance:** mark a change as real only if it exceeds the flicker of the run (e.g. 3x the median monthly flicker) and lasts at least 2 months.
- **Independent gold set:** a few hundred hand-checked points on the weak sites (Kano, Madagascar, Chilean pasture, Willamette crops vs grass). It is the only fair way to settle the Crop/Grassland question, because the three label maps disagree.
- **Multi-year runs:** 24 or 36 months to show real trends (deforestation, reservoirs), with the Sentinel-1B scene gap flagged.
- **Training on 2022+ radar** so that the half-density Sentinel-1 years match training conditions.
- **Larger areas:** tile-by-tile processing with progress per tile, beyond the 10 km limit.
- **Better water labels:** burn in narrow waterways from a vector map (e.g. OpenStreetMap waterways and canals) or a dedicated water map (JRC Global Surface Water), so the model learns canals and small rivers that the land-cover maps generalise away.
- **Haze-robust photos:** add a haze test (e.g. a blue-band or aerosol-band threshold) to the clear-pixel mask, or show the least hazy single scene of the month instead of the median, so the photo is readable over polluted cities.
- **Faster runs:** cache Earth Engine downloads per (location, month), and fetch the Sentinel-2 photos in the background after the radar.

## 5. For the final report

- Case study: Willamette (numbers in section 2.1) and the Black Forest snow example (section 2.2), with the monthly photo next to the model map.
- Seasonal effects: the snow example shows why annual labels plus radar give stable land cover, and why optical change detection would show fake change.
- Change inflation: flicker 0.26-0.66 % per month, largest monthly change 0.43 pp; changes below about 1 pp are within the noise.
- Honest scores: rural mIoU 0.679 on agreement pixels, about 0.55 on all pixels; 68 % agreement with Esri vs 51 % majority baseline.
