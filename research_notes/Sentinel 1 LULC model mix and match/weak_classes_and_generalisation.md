# Sentinel-1-only LULC: weak classes (water, snow, grassland vs crop), loss/ensemble/TTA mixes, cross-region generalisation

Context for the report writer. Project: U-Net with an ImageNet ResNet-34 encoder. Input: S1 GRD gamma0, terrain-flattened, monthly composites. 12 channels: target month VV, VH, VV-VH, plus 9 yearly stats (VV/VH/VV-VH mean; VV and VH 2nd-lowest and 2nd-highest month; VH std; share of months with VV < -18 dB). Loss: sqrt-median-frequency weighted CE + 0.5 Lovász. AdamW, EMA, month dropout 0.3.

Rural test IoU: Man-made 0.747, Crop 0.656, Trees 0.833, Grass 0.682, Bare 0.894, Water 0.492 (P 0.59 / R 0.74), Snow 0.449 (P 0.60 / R 0.64). Region mIoU: Africa 0.49, Oceania 0.44, N America 0.57, S America 0.58, Asia 0.66, Europe 0.65. Worst site types: wetland 0.22, snow_ice 0.22, shrub_grass 0.23.

Scope. These notes extend the earlier pass in `research_notes/Sentinel 1 LULC segmentation improvements/`. Topics already covered there are not repeated: Lovász (+0.3 to +2.5), EMA (+1.08), SWA (+0.77), flip TTA (0 to +1.6), the DFC2025 ensemble (+0.61), copy-paste (+6.2), HAND/DEM, temporal stats vs full stack, GLCM and coherence.

Research date: 2026-11-03. About 19 tool calls. Several PDFs were not readable (arXiv 2107.08369 PDF binary; Garg et al. 2024 RSE full text not retrieved). Items marked "(primary paper not fetched this session)" are cited from known primary URLs, but their numbers were not re-verified. Treat them as lower confidence.

## Q1. Water vs dark surfaces; snow / wet snow vs bare; incidence angle and orbit direction

### Takeaway
Two failure sets explain the water result (precision 0.59, recall 0.74, so false positives dominate). The first is low-backscatter land: smooth sand, dry bare soil, tarmac and radar shadow. The fix with the best evidence is a **temporal-persistence feature**, not a new polarisation. Low-backscatter land stays dark all year; water also stays dark, but its dark state is defined against a "definitely land" distribution. A published exclusion rule (below yearly median for more than half the year) and a sand exclusion layer cut false positives by about 50 % vs Otsu.

For snow, the standard physics is **wet-snow change detection against a reference** (Nagler method). Wet snow drops roughly 2-3 dB below a snow-free or dry-snow reference. Accuracy depends heavily on which reference is used and on a site-specific threshold. Fixed thresholds transfer badly between sites. Deep models that are fed the difference image plus VV and slope beat fixed thresholds.

Ascending + descending together gave +13 to +20 OA points in mountain terrain without terrain flattening (RF). The project already terrain-flattens, so the expected marginal gain is smaller.

### Cited Findings
**Water vs dark land**
- An EGU21 abstract (EGU21-6982) excludes pixels whose backscatter stays below the yearly median for more than half of the preceding year. It sets the water threshold at the 1st percentile of a "definitely land" distribution, separately for VV and VH. Reported result: about 50 % fewer false positives than traditional Otsu thresholding. — [EGU21-6982](https://meetingorganizer.copernicus.org/EGU21/EGU21-6982.html)
- A Remote Sensing 2018 paper (10(4):583) builds a **Sand Exclusion Layer** from S1 time series to flag non-water areas with permanently low backscatter. These are then excluded from flood detection. — [MDPI rs10040583](https://www.mdpi.com/2072-4292/10/4/583/xml)
- Garg, Dasgupta, Motagh, Martinis & Selvakumaran (RSE 2024, vol. 315, doi 10.1016/j.rse.2024.114417):
  - CEMS Global Flood Monitoring currently **excludes arid regions** because sand and calm water have similar backscatter.
  - The paper evaluates polarisation, temporal information and InSAR coherence for arid flood detection.
  - Its accuracy numbers were **not retrieved** this session. — [X-MOL record](https://x-mol.com/paper/1844392002841403392); [Bohrium/arXiv listing](https://bohrium.dp.tech/paper/arxiv/5746d041f9290599f313e1426257362a6c6dcab75790490a628ddb0d71c52011)
- **Polarisation evidence conflicts.** One MDPI 2018 paper says VV is better for calm water because of higher contrast. An ISPRS paper reports VH as most suitable after finding water vs bare-soil confusion. — [MDPI rs10040583](https://www.mdpi.com/2072-4292/10/4/583/xml); contradicted by [ISPRS Archives XLVI-4-W3-2021](https://isprs-archives.copernicus.org/articles/XLVI-4-W3-2021/45/2022/)
- An ESA STEP forum thread notes that deserts are much more temporally stable than water. Coherence can be near zero over some dark desert, so coherence alone may not separate them. This is anecdotal. — [STEP forum](https://forum.step.esa.int/t/discriminate-water-surface-and-desert-with-sentinel-1/36057)
- A Swin-Unet on S1 VH seasonal composites (MOLCA labels) reached water producer's accuracy of 0.99 in Amazonia and 0.98 in Siberia, but only **0.77 in Africa**. Africa's bare-land PA was 0.93 while Amazonia's was 0.00. Wetland PA was 0.03 / 0.16 / 0.79. — [arXiv 2503.07230](https://arxiv.org/html/2503.07230v1)
- Sen1Floods11 context:
  - Attentive U-Net with VV, VH, VV/VH: IoU 67 %, about 3 points above the benchmark.
  - Adding a **permanent-water mask and elevation** gave +4.5 IoU vs the previous benchmark. — [Yadav, Nascetti & Ban 2022, Frontiers](https://www.frontiersin.org/articles/10.3389/frsen.2022.1060144)
  - Bai et al. 2021: mIoU 52.99, OA 92.81 on the Sen1Floods11 test set, and IoU 76.74 on Bolivia. — [Remote Sensing 2021 DOAJ](https://doaj.org/article/9fd78fd0687a4424b4fca82c4c65da33)
- Navacchi, Reuß & Wagner (TU Wien) estimate the backscatter-vs-incidence-angle slope with a neural network. Normalising to a reference angle corrects strong forward-scattering effects over water and wetlands, and **orbital artefacts are significantly reduced**. — [TU Wien repositum](https://repositum.tuwien.at/handle/20.500.12708/212926?mode=full)

**Snow / wet snow**
- Nagler et al. used a ratio between snow-covered and snow-free images. The original Radarsat threshold was -3 dB; S1 practice usually uses -2 dB. — [Gong et al. 2024, Remote Sensing 16(19):3575](https://www.mdpi.com/2072-4292/16/19/3575). The S1 method is Nagler et al. 2016 (primary paper not fetched this session): [Remote Sensing 8(4):348](https://www.mdpi.com/2072-4292/8/4/348).
- Liu et al. 2022 (Geocarto Int., prairie). With an optimised threshold, OA was 80.8 % (dry-snow reference) and 79.0 % (snow-free reference). With the **conventional -2 dB threshold**, OA was only **32.1 % and 43.5 %**. -2 dB worked only on the ratio computed against a snow-free reference in cropland. Source is a search snippet; full text not read. — [search-indexed record](https://searchworks.jioinstitute.edu.in/articles/asx__172017186)
- Torralbo et al. (Sierra Nevada) found optimal thresholds between -2 and -3.88 dB depending on the period. — cited in [Gong et al. 2024](https://www.mdpi.com/2072-4292/16/19/3575)
- Gong et al. 2024 used a weakly supervised change-detection network with VV, VH and local incidence angle. Average wet-snow OA was **85.2 %** (vs Landsat). The **difference image + VV + slope** combination worked best. — [MDPI rs16193575](https://www.mdpi.com/2072-4292/16/19/3575)
- SACUNet (Mont-Blanc, 2025 HAL / IEEE TGRS) was trained on fused S1 wet-snow and S2 snow labels. Results: OA 97 %, precision 94 %, recall 97 %, **IoU 92 %**. — [HAL hal-05444067](https://memsic.ccsd.cnrs.fr/LISTIC/hal-05444067v1)
- EGU23 SESAR U-Net (EDF) reached about 80 % OA. The authors attribute the limit to coarse MODIS reference labels. — [EGU23-7240](https://meetingorganizer.copernicus.org/EGU23/EGU23-7240.html)
- A 2018 Waterloo thesis (S. Ontario) found classification more accurate than Nagler thresholding for wet snow. — [UWSpace 10012/13204](https://uwspace.uwaterloo.ca/handle/10012/13204)
- A Tupungato (Argentina) S1 study found -2 dB optimal, but notes that the parameters need local adaptation. — [CONICET 11336/130994](https://ri.conicet.gov.ar/handle/11336/130994)

**Orbit direction**
- Iran, three regions, RF. Using ascending + descending together without terrain flattening raised OA by **13-20 points** (Marand 65.33 % to 79.17 %). — [GISJ article 105953](https://gisj.sbu.ac.ir/article_105953.html?lang=en)
- Gaofen-3 analogue: geometric-distortion areas found from local incidence angle were filled from the opposite track before a 2D-CNN, giving OA 0.93. — [AGRIS record](https://agris.fao.org/search/en/records/687a8990464818917f32a897)

### Inferences
- **Water false positives (cheap).** The existing channel "share of months with VV < -18 dB" already encodes persistence, but it cannot tell permanent water from permanent dark sand or tarmac. Both have a share near 1. Add two cheap channels:
  - **Annual max of VV** (or the 90th percentile). Sand and dry soil show occasional bright months (rain moistening, roughness change), while water rarely does except under wind.
  - **VH annual mean minus the VH noise floor**, or simply **VH 2nd-highest month**. This already exists, so check that it is used.
  - Better: the **VV temporal coefficient of variation in linear power**. Wind-roughened water is more variable than stable desert, and the forum and EGU sources say deserts are very stable. The sign of this effect is site-dependent; test it.
- **Radar shadow.** Terrain flattening does not fix shadow; it is a no-data condition. Use the layover/shadow mask from preprocessing (computed from the DEM, which is allowed in preprocessing) as an **ignore mask in the loss and a no-prediction mask in the metric**. Alternatively, fill shadow pixels from the other orbit direction (the Gaofen-3 approach). This keeps the DEM out of the model input. It plausibly removes a large share of mountain "water" false positives. No quantified source found.
- **Snow.** The current yearly stats (2nd-lowest and 2nd-highest month) capture the wet-snow dip only implicitly. A Nagler-style channel fits the S1-only rule: **min-month VV (or VH) minus the median of the snow-free-season months**, in dB. Better still, give the month index (timing) of the minimum. Thresholds vary from -2 to -3.9 dB by site, so feed the **continuous difference**, not a binary mask. Gong 2024 found the difference image + VV + slope combination best; slope is DEM-derived, so under the rule use only the difference image. Dry snow is nearly transparent at C-band, so snow vs bare/rock in winter-dry areas relies on the melt-season dip. Glaciers and permanent ice have no seasonal snow-free reference, so this feature will not help there.
- **Incidence angle.** Gamma0 terrain flattening plus a 31-45 deg window already removes most of the angle dependence. Feeding the angle as a channel matters more for water and wetland (forward scattering). Expected gain: small. Mixing asc/desc in one composite may still add variance; the +13-20 point RF gain was measured **without** terrain flattening.

### Gaps
- No study found reporting S1-only CNN water IoU split by confusion source (sand vs shadow vs tarmac).
- Garg et al. 2024 numbers (best polarisation, coherence gain) not retrieved.
- No S1 study found that ablates "incidence-angle channel" in a deep segmentation model.
- No S1 study found on permanent ice / glacier vs bare rock with a CNN.

## Q2. Grassland vs cropland from C-band time series

### Takeaway
Evidence is thinner than for optical data. Simple temporal metrics (mean, std, percentiles of VH and VV) separate grassland from crops and forest well in single-region RF studies (Brazilian Amazon OA 95.3 %). Harmonic features give **about the same accuracy as raw time series** on S1. Crop-type RNNs reach F > 94 %. Within-grassland mowing detection is weak with S1 alone. Adding S1 to S2 even lowered mowing F1 (0.65 to 0.61). Grassland vs crop in this project (0.682 / 0.656) is therefore near what C-band can deliver without more temporal resolution. Biggest lever: more temporal detail (a crop's harvest drop or bare-soil period) rather than new losses.

### Cited Findings
- Brazilian Amazon (ISPRS Archives 2020). RF on S1 temporal metrics separated grassland from forest and agricultural crops. VH temporal features alone worked well; VV + VH best, **OA 95.33 %, kappa 0.93**. — [ISPRS Archives XLIII-B3-2020-259](https://isprs-archives.copernicus.org/articles/XLIII-B3-2020/259/2020/)
- Summer crops (sunflower, soybean, maize). S1 **harmonic-based phenological features vs raw time series gave comparable results**. S1 beat S2 for sunflower; S2 was better for soybean and maize. — [HAL INRAE hal-04823571](https://hal-lirmm.ccsd.cnrs.fr/RESEAU-TELEDETECTION-INRAE/hal-04823571v1)
- Camargue (France). Multitemporal S1 with deep RNNs clearly outperformed classical approaches for agricultural classes; conference version reports **F / accuracy > 94 %**. — [HAL hal-01900540](https://hal.archives-ouvertes.fr/hal-01900540)
- Amazon, S1 2017-2020: Bi-GRU F1 93.53, Bi-LSTM 93.29, best classical SVM 92.18. — [MDPI rs13245000](https://mdpi.com/2072-4292/13/24/5000/html)
- Swin-Unet on S1 VH seasonal composites, producer's accuracy:
  - Grassland: 0.88 Amazonia / 0.83 Africa / 0.86 Siberia.
  - Cropland: 0.85 / 0.73 / 0.95.
  - Shrubland: 0.50 / 0.52 / 0.00.
  - Seasonal mean "super images" beat a 5-scenes-per-season time series (OA 0.936 vs 0.872 in Africa), though tile counts differed. — [arXiv 2503.07230](https://arxiv.org/html/2503.07230v1)
- Mowing (Germany):
  - Reinermann et al. 2022: adding S1 to S2 lowered mowing F1 from 0.65 to 0.61 through false positives. PolSAR entropy was the most reliable S1 indicator. — [DOAJ Remote Sensing Mar 2022](https://doaj.org/article/1af5f3dbf03c488fbf176b716e908338)
  - Lobert et al. 2021: NDVI + cross-ratio + coherence gave F1 0.84. **SAR-only inputs were clearly worse than NDVI**. — [AGRIS record](https://agris.fao.org/search/fr/records/67a0c64c20478411b02715cf)
- Grassland use intensity (Slovenia), intensive vs extensive: OA > 88 % in all years, but with limited reference data. — [DOAJ Remote Sensing Jul 2022](https://doaj.org/article/1dca7d77501247cab64361662c78b0d7)
- Cropland accuracy rose as S1 scenes were added until early August, with only minor gains afterwards. — [VTT CRIS](https://cris.vtt.fi/en/publications/cropland-classification-using-sentinel-1-time-series-methodologic/)

### Inferences
- The model sees only the target month plus annual stats; it never sees *when* things happen. Cropland's key signal is a **timed event**: a bare-soil low followed by a rapid VH rise, then a harvest drop. Grassland is flatter, with only small mowing dips. Cheap, S1-only additions:
  - **Month index of the VH minimum and maximum**, encoded as sin/cos (handles hemispheres poorly; see Q5).
  - **First-harmonic amplitude and phase of VH** (two channels, plus the mean already present). Evidence says harmonics ≈ raw series in accuracy. They compress 12 months into 2-3 channels with timing included, which the current stats lack.
  - **Max month-to-month VH drop** (dB). A proxy for harvest or ploughing "change events".
- Alternative with less design: feed **3-4 seasonal VH/VV means** instead of the 2nd-lowest/highest stats. arXiv 2503.07230 used only four seasonal VH composites and got cropland PA 0.73-0.95.
- Expected gain: no source gives IoU deltas for these additions in a CNN. Guess: +1 to +3 IoU on crop and grass, mostly in regions with clear crop calendars. Smallholder Africa, with mixed fields at 10 m, will gain least.

### Gaps
- No S1 study found reporting harmonic amplitude/phase for grassland vs cropland with accuracy numbers.
- No S1-only deep segmentation study found with an ablation of timing features.
- Number-of-change-events features: no S1 LULC evidence found.

## Q3. Losses for rare classes and boundary quality

### Takeaway
Remote-sensing evidence for one loss over another is weak and setup-specific. The closest S1 data point (Sen1Floods11) shows **Focal-Tversky +1.9 IoU over edge-weighted CE**, with edge-weighted CE +9.7 over plain CE. Focal-Tversky outputs were badly calibrated. Boundary losses gave small gains on building benchmarks. The project already uses weighted CE + Lovász. The cheapest targeted change is a **Tversky/Focal-Tversky term tuned per class toward precision for water and snow**, because both currently have precision < recall. A recall-oriented loss (the usual Tversky setting) would make water worse.

### Cited Findings
- Cross-modal distillation on Sen1Floods11, hand-labelled split, mean over 5 runs:
  - Plain CE was worst.
  - Edge-weighted CE: +9.7 IoU points.
  - Tversky focal loss: +1.91 IoU over edge-weighted CE, about 69.7 ± 0.8 IoU.
  - Tversky focal water probabilities were **overconfident / uncalibrated**; CE was better calibrated. — [arXiv 2302.08180](https://arxiv.org/pdf/2302.08180)
- Boundary loss for RS (Bokhovkin & Burnaev 2019), added to an IoU loss: Potsdam pixel IoU > 93.8 %, INRIA 74.3 %, beating baselines. Building-type classes only. — [arXiv 1905.07852](https://ar5iv.labs.arxiv.org/html/1905.07852)
- TopPixelLoss (averaging the highest-error pixels, an OHEM-like loss) on Vaihingen with PSPNet: **about +3.0 mIoU over focal loss**. — [Chinese Space Science and Technology 2021, 41(6):85-90](https://journal26.magtechjournal.com/kjkxjs/CN/Y2021/V41/I6/85)
- CLAIRE (SAR + optical land cover) reports that Tversky alone raised rare-class IoU more than CE or focal, with weights set to favour recall. This is a self-reported ablation. — [arXiv 2509.11952](https://arxiv.org/pdf/2509.11952)
- Primary loss definitions (primary papers not fetched this session):
  - Focal Tversky: [Abraham & Khan, arXiv 1810.07842](https://arxiv.org/abs/1810.07842).
  - Unified Focal loss: [arXiv 2102.04525](https://arxiv.org/pdf/2102.04525).
  - Class-balanced loss (effective number of samples, (1-β)/(1-β^n)): [Cui et al. CVPR 2019, arXiv 1901.05555](https://arxiv.org/abs/1901.05555).
  - Boundary loss (distance maps): [Kervadec et al., arXiv 1812.07032](https://arxiv.org/abs/1812.07032).
- Earlier pass (not repeated): Lovász +0.3 in DFC2025 SAR; focal/OHEM risk amplifying label noise.

### Inferences
- **Error type decides the loss.** Water: precision 0.59 < recall 0.74. Snow: 0.60 < 0.64. Both suffer more from false positives than false negatives. Rare-class weighting (sqrt median-frequency) and recall-biased Tversky (β > α) push recall up and precision down. That is the wrong direction for water.
  - Option A: **lower** the water class weight (e.g. cap weights at 2-3x).
  - Option B: add a per-class Tversky term with α > β for water and snow, penalising false positives.
  - Option C: tune per-class logit biases / thresholds on validation after training (zero training cost). Move water's decision bias until P ≈ R. For IoU, the best operating point is usually near P ≈ R. Moving from P 0.59 / R 0.74 toward a balance could plausibly gain a few IoU points on water alone. This is a post-hoc calibration; no source quantifies it for S1.
- Class-balanced (effective number) weights with β = 0.999-0.9999 give weights between inverse frequency and uniform, similar to the current sqrt median-frequency. Expected change ≈ 0. Not a priority.
- Boundary loss: the weak classes are not boundary-limited. Water and snow errors are whole-region confusions (dark land, wet soil). Expect < 1 IoU. Low priority.
- OHEM/TopK: +3 mIoU on clean optical labels, but labels here are WorldCover-derived. The previous pass flagged noise amplification. Test only with a mild top-k (e.g. 30 % hardest pixels) plus CE on the rest.

### Gaps
- No controlled multi-loss comparison found on an S1 multiclass LULC benchmark.
- A search summary mentioned a sea-ice SAR study where Dice/focal scored slightly better on metrics but CE looked better visually. The primary source was not identified, so it is not cited.
- No RS evidence found for class-balanced (effective number) loss vs median-frequency weights.
- No study found on post-hoc per-class bias tuning for IoU in RS segmentation.

## Q4. Ensembling, TTA, multi-scale inference, CRF

### Takeaway
On top of the earlier pass (flip TTA 0 to +1.6, DFC2025 ensemble +0.61), new data points: S1 flood **TTA +5 % IoU** (paper claim), **CRF-type fusion +0.7 to +1.0 mIoU** on LoveDA/Vaihingen, and a full S1 pipeline (pseudo-labels + U-Net ensemble + CRF) going from 0.60 to 0.765 IoU. Ensembles and CRF help most on binary water/flood tasks. For multiclass LULC, expect +0.5 to +1.5 mIoU from TTA + a 3-model ensemble combined.

### Cited Findings
- Paul & Ganju (NeurIPS 2021 CCAI workshop), ETCI 2021 S1 flood detection. Baseline IoU 0.60; full pipeline (iterative pseudo-labelling from U-Net ensembles + CRF) **0.7654**. Per-component gains were not isolated in the abstract. — [arXiv 2107.08369](https://arxiv.org/abs/2107.08369)
- The same paper (per search summary of the full text): "TTA improves IoU by 5 %"; CRF described as a "crucial element" with "substantial" gains. The leaderboard shows ensemble + CRF at 0.68 vs a standard U-Net at 0.57. The PDF could not be parsed this session, so these are not verified in the table. — [arXiv 2107.08369 PDF](https://arxiv.org/pdf/2107.08369)
- ClassWise-CRF (category-specific fusion with CRF potentials):
  - LoveDA: +1.00 mIoU (val), +0.68 (test).
  - Vaihingen: +0.87 / +0.91. — [arXiv 2504.21491](https://arxiv.org/abs/2504.21491)
- LiDAR TTA +0.7 mIoU; Martian landslide hflip -0.06 points (consistent with the earlier pass: TTA gain is task-dependent). — [arXiv 2603.14132](https://arxiv.org/pdf/2603.14132)
- D4 TTA: apply the inverse transform to logits *before* averaging (common bug). — [Albumentations D4 TTA](https://albumentations.ai/docs/examples/example-tta-d4-aerial-segmentation/)

### Inferences
- Cost on a single A6000: a 12-channel ResNet-34 U-Net at 256-512 px is fast. 8x D4 TTA on a ≤ 5 GB test set takes minutes. A 3-seed ensemble costs 3x training time. Inference cost is negligible.
- **Most useful ensemble for this project:**
  1. **Multi-month ensemble.** Average softmax over the 12 target months (already suggested in the earlier pass). Labels are annual. This is a free 12-member ensemble that directly attacks snow (snow visible only in some months) and crop.
  2. **Encoder diversity.** ResNet-34 + EfficientNet-B? or ConvNeXt-T, at 2-3 models. Diverse ensembles beat same-architecture seeds in general (no RS number found).
- Multi-scale inference (0.75 / 1 / 1.25): at fixed 10 m GSD with no scale variation, expect ≈ 0. Skip.
- CRF (dense CRF on softmax, with SAR VV/VH as the "image" for pairwise terms): speckle makes appearance kernels noisy. Monthly composites reduce speckle, so it may still give +0.5 to +1 for water edges. Low priority after TTA and month ensembling. A simpler alternative is a **minimum mapping unit** (remove water blobs < N pixels). It targets the false-positive problem directly: isolated dark patches of tarmac and shadow.

### Gaps
- No S1 multiclass LULC study found that isolates TTA or CRF gains.
- The ETCI 2021 paper's per-component table was not verified.

## Q5. Cross-continent domain generalisation for SAR land cover

### Takeaway
The problem is large. A Europe-trained SEN12MS model dropped from 0.705 to 0.424 OA on Africa (-28 points, optical). Africa and Oceania (0.49 / 0.44 mIoU) follow the same pattern; Australia is called "unique". Geographic-coordinate auxiliary tasks gave **+2.7 to +4.2 mIoU** on cross-region French domains. S1-specific fixes with numbers were not found. The cheapest fixes are region-balanced sampling, an auxiliary geo-encoding task (no input change), per-site/region **standardisation of input statistics**, and regional fine-tuning if extra labels exist.

### Cited Findings
- SEN12MS, Sentinel-2 input, MODIS-derived LCCS labels (about 81 % accurate):
  - Europe-trained model: 0.705 OA on Europe, **0.424 on Africa**.
  - Out-of-continent accuracy is generally lower and highly variable.
  - Australia's landscapes are unique; no other continent's model did well there.
  - Each continent shows a distinct input-band distribution "signature".
  - No fixes tested. — [arXiv 2008.10351](https://ar5iv.labs.arxiv.org/html/2008.10351)
- GeoMultiTaskNet (FLAIR, 10 source / 3 target French departments, 0.2 m aerial):
  - Auxiliary head regresses a sinusoidal encoding of patch lat/lon (noise ±30 km), with MSE added to the segmentation loss.
  - ResNet baseline 42.51, +GeoMT **46.68 (+4.17)**, full method 47.22 (+4.71) mIoU.
  - Light U-Net: 42.05 to 44.83 (+2.78).
  - ±50 km noise erased the gain (42.68). — [arXiv 2304.07750](https://ar5iv.labs.arxiv.org/html/2304.07750)
- BELDE (2026 preprint): Europe-trained models showed a consistent drop zero-shot elsewhere; best F1 about 58 %. In-domain baseline not verified. — [arXiv 2606.20909](https://arxiv.org/pdf/2606.20909)
- Domain-adversarial training on multi-temporal Sentinel-2 (Martini et al.) and DA across sensors with CORINE labels exist as methods; gains not retrieved. — [arXiv 2104.00564](https://ar5iv.labs.arxiv.org/html/2104.00564); [arXiv 1912.05000](https://www.arxiv-vanity.com/papers/1912.05000/)
- Patel et al. (arXiv 2111.10079, from the earlier pass): SimCLR / FixMatch improved generalisation on geographically shifted test data, including S1 flood. Numbers not retrieved. — [arXiv 2111.10079](https://arxiv.org/abs/2111.10079)
- S1 VH-only Swin-Unet gave very different per-class PA by region. Africa built-up PA was 0.09 vs 0.85 Amazonia / 0.92 Siberia; Africa water PA 0.77. Models were **trained per region**, not transferred. — [arXiv 2503.07230](https://arxiv.org/html/2503.07230v1)

### Inferences
- Likely causes of low Africa / Oceania mIoU, by type:
  - **Label shift**: WorldCover/DW quality in drylands, plus shrub/grass/bare definitions.
  - **Class-prior shift**: more bare, shrub and sparse grass.
  - **Input shift**: dry soils are dark and look like water; there is no snow.
  - Label shift cannot be fixed by the model. Report per-region class confusion to separate the causes.
- Cheap fixes, ordered by cost:
  1. **Region-balanced sampling.** Sample tiles so each continent gets equal weight, or weight ∝ 1/sqrt(count). 0 cost. Typical gain in DG literature is modest; no S1 number found.
  2. **Auxiliary geo-encoding head.** Predict sin/cos of lat and lon from bottleneck features. Use an MSE weight of about 0.1 and add ±30 km noise. This does not change the input, so it stays S1-only at inference. Evidence: +2.7 to +4.2 mIoU (FLAIR, optical, within-country).
     - Alternative: feed **latitude (sin/cos) and day-of-year of the target month** as constant input channels. This tells the model the hemisphere and season; a VH low in July means different things in Europe and Australia. It breaks a strict "S1-only input" rule, so present it as metadata, not a sensor. Cost: 2-4 channels.
  3. **Hemisphere-aligned months.** Shift Southern-Hemisphere months by 6 before computing month-index features (Q2). Zero cost, and fixes the timing features for Oceania, S America and southern Africa.
  4. **Per-tile input standardisation.** Subtract the tile median of VV/VH, or histogram-match to a global reference. This removes regional offsets from soil moisture and roughness. Risk: it destroys absolute-level information that water and bare need, so apply it only to a *copy* of the channels.
  5. **Regional fine-tuning.** Fine-tune the decoder only on Africa/Oceania tiles for a few epochs, and select by regional val. Cheap on an A6000. This yields separate regional models; report them as such.
  6. **Self-training** on unlabelled Africa/Oceania tiles (earlier pass: +0.8 in DFC2025). One extra run.
- Expected sizes: SEN12MS shows that 28-point cross-continent gaps are possible. Since all continents are already in training here, the remaining gap (0.44-0.49 vs 0.65) is mostly class mix and label quality. A realistic gain from 1-3 is **+1 to +4 mIoU in the weak regions**, with little change elsewhere.

### Gaps
- No S1-specific cross-continent study with quantified fixes found.
- No evidence found on histogram matching / per-region normalisation gains for SAR segmentation.
- No climate-zone (Köppen) input study with numbers found for S1.

## Q6. Expected magnitudes (IoU points), summary for the report writer

### Takeaway
No single change is likely to add more than a few IoU points. The largest plausible gains target **water precision** (post-hoc bias / precision-oriented loss, persistence features, shadow masking) and **snow timing** (Nagler-style difference channel, month ensembling). Each may give several IoU points on those classes, but only about 1 point on mIoU. These are inferences anchored on the cited analogues.

### Cited Findings
- Focal-Tversky vs edge-weighted CE on S1 water: +1.9 IoU. — [arXiv 2302.08180](https://arxiv.org/pdf/2302.08180)
- Temporal-persistence exclusion: about -50 % false positives vs Otsu in arid flood mapping. — [EGU21-6982](https://meetingorganizer.copernicus.org/EGU21/EGU21-6982.html)
- Permanent-water + elevation auxiliaries: +4.5 IoU on Sen1Floods11 (elevation not allowed as input here). — [Frontiers 2022](https://www.frontiersin.org/articles/10.3389/frsen.2022.1060144)
- Wet snow, optimised vs fixed -2 dB threshold: about 80 % vs 32-44 % OA, so feed continuous differences. — [Gong et al. 2024 context](https://www.mdpi.com/2072-4292/16/19/3575); [Liu et al. 2022 record](https://searchworks.jioinstitute.edu.in/articles/asx__172017186)
- Geo auxiliary task: +2.7 to +4.2 mIoU. — [arXiv 2304.07750](https://ar5iv.labs.arxiv.org/html/2304.07750)
- CRF fusion: +0.7 to +1.0 mIoU. — [arXiv 2504.21491](https://arxiv.org/abs/2504.21491)
- Asc+desc in mountains (RF, no terrain flattening): +13-20 OA. — [GISJ 105953](https://gisj.sbu.ac.ir/article_105953.html?lang=en)

### Inferences
Ranked cheap changes (inference; verify by ablation, 1 seed each, report ± spread):

| # | Change | Targets | Cost | Plausible gain |
|---|---|---|---|---|
| 1 | Post-hoc per-class logit bias tuned on val (water, snow toward P ≈ R) | Water, snow | 0 training | +2 to +5 IoU water; ~+0.5-1 mIoU |
| 2 | Average softmax over all 12 target months (month ensemble) | Snow, crop, grass | 12x inference | +0.5 to +2 mIoU (untested) |
| 3 | Shadow/layover mask (from DEM in preprocessing) as ignore + no-predict | Water FP in mountains | Low | Unquantified; water precision |
| 4 | Snow channel: min-month minus snow-free-season median (dB), plus month-of-min (sin/cos, hemisphere-aligned) | Snow, crop | 2-3 channels | +2 to +5 IoU snow (guess) |
| 5 | VH first-harmonic amplitude + phase, max monthly VH drop | Crop vs grass | 3 channels | +1 to +3 IoU crop/grass (guess) |
| 6 | VV max / CV in linear power (sand/tarmac vs water) | Water FP | 1-2 channels | Unquantified |
| 7 | Region-balanced sampling + geo-encoding aux head (sin/cos lat/lon, ±30 km noise) | Africa, Oceania | Low | +1 to +4 mIoU in weak regions |
| 8 | Cap class weights (water/snow) or add precision-biased Tversky (α > β) | Water, snow | Low | ~+1-2 IoU |
| 9 | D4 TTA + 2-3 model encoder ensemble | All | 3x training | +0.5 to +1.5 mIoU |
| 10 | Dense CRF or minimum mapping unit for water | Water | Low | +0.5 to +1 |
| 11 | Boundary / class-balanced (effective number) loss | Edges / rare | Low | < +0.5 |

### Gaps
- None of the magnitudes in the table come from an S1 multiclass U-Net ablation. Only the anchors in Cited Findings are sourced. Treat the table as a prioritised test plan, not a prediction.
