# SAR input features and auxiliary data for Sentinel-1 LULC segmentation

Context for the report writer: the project uses GEE `COPERNICUS/S1_GRD` (IW, VV+VH, 2021). It builds monthly linear-mean composites (incidence angle 31-45 deg, 3x3 focal median) stored as uint8 dB. Per month the model sees VV, VH and VV-VH, plus 6 PCA components of the 12-month series. Model: U-Net with a ResNet-34 encoder, 8 classes. Weak classes: wetland IoU 0.22 (confused with water), shrub/grass 0.34, cropland 0.49 (confused with trees). Every feature must be computable globally in GEE at inference time. Dataset budget is about 4-5 GB.

Research budget note: about 20 tool calls. Several primary PDFs (WorldCover PUM V2.0, IIASA) were blocked by bot protection. Gaps are listed explicitly. Items marked "(background knowledge, not verified this session)" come from prior knowledge, not from a fetched source. The writer should treat them as lower confidence.

## Q1. Temporal statistics of SAR vs raw monthly stacks or PCA

### Takeaway
The strongest controlled evidence (3,843 experiments, boreal wetlands) found that **keeping all within-year observations beat summarised temporal forms** (monthly or seasonal means, annual statistics). Temporal form mattered more than feature choice. The project already feeds a full 12-month stack. Temporal statistics are therefore cheap *complementary* channels (std, min/max, percentile range, CV) that make seasonality explicit. They are not a replacement for the stack. No source quantified their gain on top of a CNN that already sees the full stack.

### Cited Findings
- Li, Chen, White, Wulder & Hermosilla (2020), *Int. J. Applied Earth Observation and Geoinformation*, doi 10.1016/j.jag.2019.102007. The study discriminated treed vs non-treed boreal wetlands with time-series S1 data (2017). It ran 3,843 classification experiments. "Directly using all within-year observations yielded higher accuracy than using generalized temporal forms" (monthly/seasonal means, annual statistics). Temporal form mattered more than feature choice. σVV and σVH were the most useful features, ahead of four Stokes-vector-derived measures. Best OA was 0.860 ± 0.002. Most confusion was between treed wetland and non-wetland. Simulated missing observations did not significantly reduce OA when all observations were used. — [rsc4earth summary of Li et al. 2020](https://rsc4earth.de/publication/li-discriminating-2020/); [search summary](https://rsc4earth.de/publication/li-discriminating-2020/)
- Dongting Lake wetland study (IEEE JSTARS, Jan 2022). It added time-series similarity parameters and G0 statistical texture features to S1. RF reached 97.57% OA. The gain was not isolated against a baseline in the abstract. — [DOAJ record](https://doaj.org/article/28e04f0dcf4842aca33405bf94688633)
- DeLancey et al. (2020, Remote Sensing 12(1):2) used a seasonal *difference* feature `dVH = VH_winter - VH_summer` (XGBoost model) for Alberta wetlands. This is an explicit seasonal-range feature. — [DeLancey et al. 2020, MDPI PDF](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-12-00002/article_deploy/remotesensing-12-00002.pdf)
- Harmonic fitting: no Sentinel-1 backscatter study reporting harmonic amplitude/phase gains was found. Harmonic coefficients are established for optical series. Example: CCDC harmonic coefficients + RMSE fed to a DCNN on Landsat gave OA 88.7%/94% and kappa 0.87 (IJAEOG, Nov 2024). — [DOAJ record](https://doaj.org/article/4a5559470900443bbd5e08fcbac06833)
- Canadian Wetland Inventory, 2nd generation (Mahdianpari et al., GEE, RF, 10 m). The S1 VV/VH ratio feature and S2 NDVI were the most important features. OA was about 86%, ranging 76-91% across ecozones. — [INRS PDF P3777](https://espace.inrs.ca/id/eprint/10379/1/P3777.pdf)

### Inferences
- The U-Net already gets 12 monthly frames, so it can in principle learn mean/std/range itself. The Li et al. result suggests the raw stack is the right base. Under a CNN, the 6 PCA components and any added stats mostly help optimisation and inductive bias. They do not add new information.
- Seasonal range and std of VH are the most plausible helpers for **cropland vs trees**. Cropland VH swings strongly with growth and harvest, while forest VH is temporally stable. A per-pixel `VH_std`, `VH_p90 - VH_p10` and `VV_CV` are 3-4 extra channels computable in GEE with `reduce(ee.Reducer.stdDev())` / `percentile([10,50,90])`.
- For **wetland vs water**, temporal min/max matter. Seasonally flooded wetlands reach a very low VV minimum in wet months and higher values otherwise. Permanent water stays low all year. `VV_min`, `VV_max` and the "fraction of months below a water threshold" (a permanent-vs-seasonal water indicator) are logical features. No source in this session quantified that gain.
- Compute stats from the **linear** values before converting to dB. Means of dB values bias low.

### Gaps
- No ablation found for "full monthly stack + temporal stats" vs "full stack only" with a CNN.
- No S1 harmonic-regression (amplitude/phase) classification study with accuracy numbers was found.

## Q2. Polarimetric indices from dual-pol GRD; entropy/alpha; InSAR coherence

### Takeaway
From GEE GRD (intensity only) you can compute **ratio-type indices**: VH/VV, normalised difference (VV-VH)/(VV+VH), and the GRD-RVI `4·VH/(VV+VH)`. These are deterministic functions of VV and VH. The project already provides VV-VH in dB, which equals log(VV/VH), so the extra information to a CNN is small. True **DpRVI and H/alpha need the 2x2 covariance matrix from SLC**, which GEE does not host. **InSAR coherence** exists globally in GEE only as the Kellndorfer et al. 2019-2020 seasonal dataset at ~92.7 m. It is a different year (Dec 2019-Nov 2020) and much coarser than 10 m.

### Cited Findings
- DpRVI (Mandal et al., *Remote Sensing of Environment*, 2020) jointly uses the degree of polarisation and the eigenvalue spectrum of the dual-pol C2 covariance matrix. It was validated on canola, soybean and wheat in Canada and outperformed four other vegetation indices against biophysical variables. — [TU Delft record](https://research.tudelft.nl/en/publications/dual-polarimetric-radar-vegetation-index-for-crop-growth-monitori)
- SNAP/MRSLab documentation: DpRVI is computed from the dual-pol 2x2 covariance matrix (C2), i.e. from SLC data. — [ESA SNAP help: Dual-pol RVI](https://step.esa.int/main/wp-content/help/versions/9.0.0/snap-toolboxes/org.csa.rstb.rstb.op.polarimetric.tools.ui/operators/RVIOp.html)
- The GRD-compatible RVI (Sentinel Hub "RVI4S1" script) is a different index from DpRVI. It ranges 0-1 and is described as applicable globally and invariant to crop type. — [Sentinel Hub custom script](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-1/radar_vegetation_index/)
- Canadian Wetland Inventory, 2nd generation: the S1 VV/VH ratio was among the most important features. — [INRS PDF P3777](https://espace.inrs.ca/id/eprint/10379/1/P3777.pdf)
- Li et al. 2020: plain σVV/σVH beat four Stokes-vector-derived polarimetric measures for boreal wetland discrimination. — [rsc4earth](https://rsc4earth.de/publication/li-discriminating-2020/)
- Global coherence: Kellndorfer et al. 2022, *Scientific Data*, doi 10.1038/s41597-022-01189-6. GEE IDs are `Earth_Big_Data/GLOBAL_SEASONAL_S1/V2019/{COHERENCE, BACKSCATTER, INCIDENCE_LAYOVER_SHADOW, DECAY_MODEL_PARAMETERS}`. The data come from about 205,000 S1 IW images, Dec 2019-Nov 2020, covering 82°N-79°S. Coherence is the median for standardised repeat intervals (pairs up to 48 days) per season. Pixel size is about 92.7 m. Backscatter is mean RTC (bands like `winter_vv_AMP`). Licence is CC-BY-4.0. — [GEE catalog: COHERENCE](https://developers.google.com/earth-engine/datasets/catalog/Earth_Big_Data_GLOBAL_SEASONAL_S1_V2019_COHERENCE); [GEE catalog: BACKSCATTER](https://developers.google.com/earth-engine/datasets/catalog/Earth_Big_Data_GLOBAL_SEASONAL_S1_V2019_BACKSCATTER); [GEE catalog: DECAY_MODEL_PARAMETERS](https://developers.google.com/earth-engine/datasets/catalog/Earth_Big_Data_GLOBAL_SEASONAL_S1_V2019_DECAY_MODEL_PARAMETERS)
- InSAR phase/coherence time series as segmentation input: a C-band S1 pilot (Chambéry, France; Frontiers in Remote Sensing 2022) reached OA 78% (SVM) and 81% (CNN). — [Frontiers article](https://www.frontiersin.org/articles/10.3389/frsen.2022.932491)

### Inferences
- VH/VV ratio and normalised difference add little beyond what the model already sees (VV-VH in dB). If anything is added, the GRD-RVI `4·VH_lin/(VV_lin+VH_lin)` is the one with vegetation literature behind it. Expect small or no gains with a CNN.
- H/alpha and true DpRVI are **not feasible** under the "compute in GEE globally" constraint. GEE hosts only GRD intensity.
- Kellndorfer coherence is the only global coherence option. Coherence is physically very informative: urban stays high, water and forest stay low, and bare soil, crops and grass are seasonally variable. That could help built-up, bare and cropland. The drawbacks are the 92.7 m resolution (about 9x9 pixels of 10 m), the year mismatch with the 2021 labels, and the fact that it is a static layer. Adding it would be a "static auxiliary channel" upsampled to 10 m.

### Gaps
- No study found quantifying the gain from Kellndorfer coherence in a 10 m CNN land cover model.
- Did not verify whether a GRD-based approximation of DpRVI (published after Mandal 2020) exists and is validated.

## Q3. Texture features (GLCM) with a CNN

### Takeaway
No study was found that ablates GLCM inputs for a Sentinel-1 CNN. Evidence of GLCM gains comes from shallow classifiers (SVM/RF). Mixed or negative results also exist. With a U-Net/ResNet-34, whose convolutions learn spatial texture, GLCM channels are **low priority**.

### Cited Findings
- Croatian urban green infrastructure (S1, SVM): "An additional increase" in land cover classification accuracy was obtained with GLCM texture. — [CROSBI 1116295](https://www.bib.irb.hr/1116295); [Hrcak 255272](https://hrcak.srce.hr/en/255272)
- Avocado orchard study (PlanetScope/S2/Landsat, not S1): GLCM gave minimal RF improvement (F 0.9551) and hurt SVM (F 0.9453). — [WUR activity page](https://wur.nl/en/activity/texture-analysis-for-detecting-avocado-orchards-in-uruapan-mexico.htm)
- Anecdotal ESA STEP forum report: adding GLCM to dual-pol bi-temporal S1 worsened results. The thread is unresolved. — [STEP forum](https://forum.step.esa.int/t/glcm-worsening-accuracy-results/8247)
- DeLancey et al. 2020 argue CNNs learn the needed spatial cues themselves. Example: waterline edges between marsh and open water need only simple edge filters. — [DeLancey et al. 2020 PDF](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-12-00002/article_deploy/remotesensing-12-00002.pdf)

### Inferences
- Skip GLCM for the CNN. It costs extra channels and storage, and it is computationally heavy in GEE (`glcmTexture` on int images) for little expected gain.

### Gaps
- No direct S1 + CNN + GLCM ablation found.

## Q4. Auxiliary terrain: DEM, slope, aspect, HAND, TWI; legitimacy under "S1 only"

### Takeaway
Terrain layers are the **most widely used auxiliary inputs in SAR and wetland mapping**. HAND in particular gave large gains for SAR flood mapping. Large-scale wetland CNNs (Alberta) include DEM, TPI, TRI, TWI and valley-bottom-flatness. Published S1-only deep learning benchmarks have packed a DEM in as a channel (the Finland "SAR RGB-DEM" dataset). Framing: the DEM is a static auxiliary layer, not a second imaging sensor. Most papers say "SAR + topographic data". The project should state it as "S1 backscatter + static terrain context" and report an ablation with and without it.

### Cited Findings
- DeLancey, Simms, Mahdianpari, Brisco, Mahoney & Kariyeva (2020), *Remote Sensing* 12(1):2, published 18 Dec 2019. Alberta boreal wetlands (397,958 km²), CNN vs XGBoost, data from GEE.
  - CNN inputs: ARI, B2, B3, B4, DEM, NDVI, NDWI, PSRI, REIP, TPI, TRI, TWI (SAGA wetness index), VBF (multi-resolution valley bottom flatness), and S1 VH. TPI used a 750 m window.
  - The ALOS 30 m DEM was bicubically resampled to 10 m.
  - Results: CNN OA 81.3%, kappa 0.57, mean F1 0.56. XGBoost OA 75.6%, kappa 0.49, mean F1 0.52 (vs ABMI plots).
  - The paper notes that fen, bog and swamp backscatter/spectral distributions are nearly identical.
  - It does **not** report a with/without-DEM ablation in the parts read. — [DeLancey et al. 2020 PDF](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-12-00002/article_deploy/remotesensing-12-00002.pdf)
- HAND + dual-pol S1 for flood inundation (Remote Sensing 2020, rs12040644, supervised classifiers). Adding HAND improved classification accuracy by +36.0%, critical success index by +39.95% and true positive rate by +42.02% (relative to SAR only). — [search summary of MDPI rs12040644](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-12-00644/article_deploy/remotesensing-12-00644.pdf). Note: the PDF fetched at that URL turned out to be Du et al. 2020 (Delmarva U-Net with WV-3 + lidar). The HAND numbers come from the search-engine snippet and were **not verified in full text**.
- Du et al. 2020 (Remote Sensing 12(4):644): a U-Net on leaf-off WorldView-3 + lidar DEM/TWI for forested wetland inundation. Reported 95% OA (kappa 0.90) vs field data and 70% IoU vs lidar-intensity labels. The authors conclude that topographic metrics improve depressional wetland classification. The gain size was not isolated in the snippet. — [Du et al. 2020 PDF](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-12-00644/article_deploy/remotesensing-12-00644.pdf)
- Canadian Wetland Inventory, 3rd generation: added ALOS PALSAR-2 L-band, SRTM and other environmental layers. Average OA rose to about 90.5% (ecozones 84-94%), from about 86% in generation 2. Several inputs changed at once, so the DEM effect alone is not isolated. — [INRS PDF P3981](https://espace.inrs.ca/id/eprint/11958/1/P3981.pdf)
- Šćepanović et al. (IEEE JSTARS 2021, arXiv 1912.05067). Wide-area S1 land cover in Finland vs CORINE labels, seven segmentation models. The dataset "SAR RGB-DEM" packs two S1 channels plus a DEM channel into RGB imagelets (>7K). Best model FC-DenseNet, OA 90.7%. Water and forest were easiest, urban fabric hardest. No DEM ablation was reported in the extracted text. — [arXiv 1912.05067](https://arxiv.org/pdf/1912.05067v2); [Mendeley dataset](https://data.mendeley.com/datasets/gzb3y9kmf2); [JSTARS DOAJ](https://doaj.org/article/779aa9b48910447088b494dc772ba03e)
- Kellndorfer V2019 provides a global `INCIDENCE_LAYOVER_SHADOW` layer in GEE. It can serve as a layover/shadow mask channel. — [GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/Earth_Big_Data_GLOBAL_SEASONAL_S1_V2019_INCIDENCE_LAYOVER_SHADOW?hl=ja)

### Inferences
- **Wetland vs water** (IoU 0.22) is the class pair where terrain should help most:
  - Wetlands sit at low HAND, on flat ground, with high TWI.
  - Water is also low-HAND and flat, so HAND alone separates *wetland/water* from *upland*. It does not separate wetland from water. The backscatter time series must still do that.
  - The bigger HAND win is removing false "water/wetland" in **radar shadow and smooth dark surfaces on slopes** (the mechanism behind the +36% flood-mapping gain).
- **Snow**: elevation is a strong prior for snow and ice. A DEM channel likely helps the snow class and reduces confusion with bare rock.
- Cheap, global GEE layers (background knowledge, not verified this session):
  - Copernicus GLO-30 DEM (`COPERNICUS/DEM/GLO30`), NASADEM (`NASA/NASADEM_HGT/001`), SRTM (`USGS/SRTMGL1_003`, limited to 60°N-56°S, so it misses high-latitude tundra and ice).
  - `ee.Terrain.slope/aspect`.
  - MERIT Hydro (`MERIT/Hydro/v1_0_1`) has a `hnd` (HAND) band at ~90 m plus upstream area `upa`. TWI can be derived as ln(upa / tan(slope)).
  - A GLO-30-based HAND is reported in the awesome-gee-community-catalog. Its asset ID was **not** confirmed (the guessed page returned 404).
- Data cost: 2-4 static channels (elevation, slope, HAND, TWI) stored once per tile as uint8/uint16. This is negligible next to 12 months x 3 bands.
- Legitimacy framing for the exam: keep the S1-only model as the main result. Present "S1 + static terrain" as an ablation. Cite the SAR RGB-DEM dataset and the wetland inventories as precedent. Aspect should be encoded as sin/cos, or dropped. It mostly encodes SAR look-geometry effects, which incidence-angle filtering partly handles already.

### Gaps
- No S1 + CNN study found with an explicit with/without-DEM/HAND ablation reporting per-class IoU for wetland/shrub/cropland.
- HAND flood-gain numbers are not verified in full text (only the search snippet).
- GEE asset IDs for GLO-30 HAND and MERIT Hydro were not confirmed in this session.

## Q5. Speckle filtering, compositing, terrain flattening, orbit separation

### Takeaway
GEE `S1_GRD` is sigma0 without terrain flattening or speckle filtering. The **Mullissa et al. 2021 GEE S1-ARD framework** adds:
- border-noise correction,
- mono- and multi-temporal speckle filters,
- radiometric terrain normalisation (Vollrath volume/surface models) to gamma0,
- ascending/descending selection,
- linear or dB output.

For a monthly composite of several scenes, temporal averaging already reduces speckle strongly. The likely larger gain is **terrain flattening plus a layover/shadow mask**, which mainly matters in mountains (snow, bare and shrub classes). The project's incidence-angle window (31-45°) discards data and can leave gaps. Gamma0 normalisation is the standard alternative.

### Cited Findings
- Mullissa, Vollrath, Odongo-Braun, Slagter, Balling, Gou, Gorelick & Reiche (2021), "Sentinel-1 SAR Backscatter Analysis Ready Data Preparation in Google Earth Engine", *Remote Sensing* 13(10):1954, doi 10.3390/rs13101954. GEE JS/Python framework adding border noise correction, speckle filtering and radiometric terrain normalisation. Speckle filtering is not applied to ingested GEE S1 imagery because the choice is application-dependent. — [DOAJ record](https://doaj.org/article/103f5e1094994facba05434cdd914e81); [GitHub adugnag/gee_s1_ard](https://github.com/adugnag/gee_s1_ard)
- The repo supports:
  - single or dual polarisation,
  - ascending, descending or both orbits,
  - mono-temporal and multi-temporal speckle filtering (example: 9x9 multi-temporal Gamma MAP),
  - terrain normalisation with a volume scattering model (example),
  - linear or dB output. — [GitHub adugnag/gee_s1_ard](https://github.com/adugnag/gee_s1_ard)
- Terrain-flattened SAR imagery is widely considered the first SAR-derived analysis-ready dataset family (Remote Sensing 15(7):1932, 2023). — [MDPI rs15071932](https://www.mdpi.com/2072-4292/15/7/1932)
- Speckle filter vs classification accuracy (single-date, shallow classifiers):
  - Korean land/ocean study: SVM OA rose from 63.14% (unfiltered) to 73.86% with a 5x5 Lee filter. Lee beat Gamma and Frost. — [KIOST record](https://sciwatch.kiost.ac.kr/handle/2020.kiost/42247)
  - Iranian S1/S2 fusion: IDAN filter best on VH (OA 76.64%), median filter best on VV. — [Redalyc](https://www.redalyc.org/journal/3939/393976135002/393976135002.pdf)
  - Strong filters can degrade spatial and radiometric resolution and lower classification accuracy (Kampala study). — [Makerere dissertation](https://dissertations.mak.ac.ug/handle/20.500.12281/9333)
- ESA WorldCover 2021 (v200) publishes an S1 GAMMA0 yearly **median** composite of VV, VH and VH/VV (power scaled), processed from GRD with GAMMA software. The algorithm derives from the CGLS-LC100 workflow and uses S2 and S1. — [AWS registry](https://registry.opendata.aws/esa-worldcover-vito-composites); [FedEO record](https://fedeo.ceos.org/series/eo:platform/Sentinel-1/urn:eop:VITO:ESA_WorldCover_S1VVVHratio_10m_2021_V2); [IIASA ESA WorldCover 10 m 2021 v200](https://pure.iiasa.ac.at/18478)

### Inferences
- With monthly means over about 2-5 scenes per orbit, the 3x3 focal median is a mild spatial filter on top of temporal multi-looking. Switching to Refined Lee or a Quegan multi-temporal filter likely brings small gains. The single-date Lee gains (+10 pts) do not transfer to multi-scene composites.
- **Mean vs median**: WorldCover uses a median gamma0 composite. A median is more robust to outlier scenes (wet snow, rain, frozen ground). A monthly median or a linear mean is a reasonable choice. Test both if cheap.
- **Ascending/descending**: mixing orbits in one composite blends look directions, which adds variance on slopes. Options: separate ascending and descending composites (doubles storage), or one orbit plus terrain flattening. Given the 4-5 GB budget, terrain flattening is preferable to doubling channels.
- **Gamma0 terrain flattening** (Vollrath volume model, via Mullissa code) is the most valuable preprocessing upgrade for mountains, tundra and snow tiles. It can replace the fixed 31-45° incidence window. Inference must run the same code live. The Mullissa code works on the fly in GEE, but it is heavier to compute.

### Gaps
- No study found comparing Refined Lee vs Quegan vs plain temporal averaging *for monthly composites into a CNN*.
- No quantified LULC accuracy gain from gamma0 terrain flattening was found in this session.
- The exact S1 classifier features of WorldCover v200 (e.g., percentiles or stats of VV/VH) could not be confirmed. The PUM V2.0 PDF was blocked. Only the published composite description (median VV, VH, VH/VV) is verified.
- Dynamic World was not researched. Background knowledge, not verified this session: it is Sentinel-2 only, so it gives no S1 feature precedent.

## Q6. Ranking by expected gain per effort (synthesis for this project)

### Takeaway
The highest gain per effort for the weak classes is **static terrain channels (HAND, slope, elevation, TWI) plus a few temporal-statistic channels (VH std, VV min, p10-p90 range)**. Terrain flattening comes next. Polarimetric ratios, GLCM and coherence come last. These rankings are inferences built on the evidence above. No paper reports these exact combinations for this setup.

### Cited Findings
- Terrain/HAND evidence: large relative gains for SAR water mapping (HAND +36% accuracy, snippet only). Terrain layers are standard in operational wetland CNN/GEE maps. — [rs12040644](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-12-00644/article_deploy/remotesensing-12-00644.pdf); [DeLancey 2020](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-12-00002/article_deploy/remotesensing-12-00002.pdf); [CWI gen 3](https://espace.inrs.ca/id/eprint/11958/1/P3981.pdf)
- Temporal-form evidence: the full stack beats summaries, and summaries are complements. — [Li et al. 2020](https://rsc4earth.de/publication/li-discriminating-2020/)
- VV/VH ratio is a top feature in RF wetland maps. — [CWI gen 2](https://espace.inrs.ca/id/eprint/10379/1/P3777.pdf)

### Inferences
Ranked list (expected gain / effort):
1. **HAND + slope + elevation (+ optional TWI)**
   - Sources: GLO-30/NASADEM, MERIT Hydro `hnd`, `ee.Terrain`.
   - Effort: very low. 3-4 static channels.
   - Targets: wetland/water vs upland, snow, false water in shadow.
   - Frame it as an ablation under the "S1 only" rule.
2. **Temporal stat channels from the existing monthly stack**
   - Channels: VH std, VV min, VV max, VH p90-p10, VV CV.
   - Effort: low, and computable offline from the downloaded stack (no extra GEE call).
   - Targets: cropland vs trees (seasonal amplitude), seasonal wetland vs permanent water (min/max, months below a water threshold).
   - Could replace some PCA components to keep the channel count flat.
3. **Layover/shadow mask** (Kellndorfer `INCIDENCE_LAYOVER_SHADOW` or computed) or gamma0 terrain flattening (Mullissa/Vollrath). Effort: medium. Main benefit in mountain tiles.
4. **Median compositing and orbit consistency.** Effort: low to medium. Small expected gain.
5. **GRD-RVI / normalised difference.** Effort: trivial. Expected gain is near zero, because VV-VH is already present.
6. **Kellndorfer seasonal coherence (~92.7 m, 2019-20).** Effort: low. Uncertain gain because of the resolution and year mismatch. Plausible help for built-up and bare.
7. **GLCM texture.** Effort: medium to high. Expected gain ~0 with a CNN.
8. **H/alpha, true DpRVI, own InSAR coherence.** Infeasible. They need SLC, which is not in GEE.

- Shrub/grass (IoU 0.34) has the least specific aux-data fix. C-band backscatter of shrub and grass overlaps with cropland and sparse trees. Temporal amplitude and terrain context (slope/elevation for alpine grass) are the main levers. Label noise and class definition are probably as large a factor; this is outside the scope of these notes.

### Gaps
- No per-class IoU gains were found for any of these additions in a U-Net-on-S1 setting. Every ranking above is an inference and should be checked by ablation in the project.
