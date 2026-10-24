# Sentinel-1-only LULC segmentation: published benchmarks, per-class difficulty, label-quality ceilings

Reference point (our model): U-Net ResNet-34, S1 VV+VH monthly composites (3-month window + PCA of annual series), 8 classes, global sites, labels = pixels where ESA WorldCover and Dynamic World agree. Test OA 0.83, mIoU 0.54. Per-class IoU: built-up 0.94, bare 0.71, trees 0.60, water 0.54, snow 0.49, cropland 0.49, shrub/grass 0.34, wetland 0.22.

Comparability warning used throughout: every benchmark below differs in class set, label source, reference type (expert-annotated vs. map-derived), metric (OA / AA / mIoU / PA / mAP) and temporal input. Agreement-filtered labels (ours, MOLCA) remove the hardest/ambiguous pixels and inflate scores relative to expert-annotated test sets.

## 1. SEN12MS / DFC2020: S1-only results and winning solutions

### Takeaway
On DFC2020 (expert-ish 10 m validation labels, models trained on 500 m MODIS labels), SAR-only models reach roughly 55-56% average accuracy vs ~60-68% optical-only and ~62-68% fused; the official baseline paper reports no S1-only column at all. Wetland is the weakest class even for fused models (~22% accuracy).

### Cited Findings
- DFC2020 = SEN12MS-based contest, weakly supervised: training labels are MODIS-derived land cover at 500 m native resolution; S1 input = 2 channels (VV, VH sigma0 in dB) — [Schmitt et al. 2020, arXiv 2002.08254](https://arxiv.org/pdf/2002.08254); [GRSS DFC2020 page](https://grss-ieee.org/wp-content/uploads/2019/12/dfc2020/2020%20IEEE%20GRSS%20Data%20Fusion%20Contest%20-%20GRSS%20|%20IEEE%20|%20Geoscience%20&%20Remote%20Sensing%20Society.html)
- SEN12MS: 180,662 S1/S2/land-cover patch triplets, all seasons, all inhabited continents; simplified IGBP scheme; DFC2020 class distribution lacks Savanna entirely and has no Snow/Ice pixels; Wetlands and Snow/Ice "hardly exist" in SEN12MS — [Schmitt et al. 2020](https://arxiv.org/pdf/2002.08254); [SEN12MS paper](https://arxiv.org/pdf/1906.07789)
- Official DFC2020 baselines (Table 3, class-wise accuracy) only report S2-only and S1+S2 for DeepLabv3, U-Net, k-means, RF — no S1-only column. MODIS low-res labels vs. HR validation ("LR-HR" sanity check) give Forest 51.6%, Shrubland 7.7%, Grassland 6.7%. U-Net S2-only vs S1+S2: Forest 67.3% vs 55.4%, Grassland 76.7% vs 77.2%, Shrubland 0.0% vs 3.7% — [Schmitt et al. 2020](https://arxiv.org/pdf/2002.08254) (extracted from PDF text layer)
- Semi-supervised SSCL on DFC2020, average accuracy (AA, mean±std): Std DeepLab fused 61.7±2.0 / optical-only 59.7±1.9 / SAR-only 54.8±1.2; single-channel FE DeepLab 64.2 / 59.5 / 55.9; SSCL (pretrained) 68.3±1.2 / 67.6±1.6 / 56.3±0.3 — [Semi-supervised SAR+MS land cover, arXiv 2108.09075](https://arxiv.org/html/2108.09075v2)
- Same paper, fused SSCL model: OA 71.6±0.4, mIoU 49.6±0.8; per-class accuracy Forest 73.1, Shrubland 56.5, Grassland 54.0, Wetlands 21.7, Croplands 78.2, Urban 83.1, Barren 80.6, Water 99.3 — [arXiv 2108.09075](https://arxiv.org/html/2108.09075v2)
- A 2026 DFC2020 paper reports 70.10% OA with fused S1+S2 (no SAR-only number in snippet) — [ScienceDirect S2666544126000389](https://www.sciencedirect.com/science/article/pii/S2666544126000389)

### Inferences
- SSL pretraining lifted fused/optical AA by ~6-8 pts but SAR-only only ~1.5 pts (54.8 -> 56.3): SAR-only appears capped near mid-50s AA on DFC2020 with single-date S1.
- DFC2020 S1 patches are single-date; our multi-temporal composites + PCA are a structural advantage not present in these numbers.
- Our 0.54 mIoU is above the fused SSCL mIoU (49.6) on DFC2020, but not comparable: DFC2020 trains on 500 m noisy labels and tests on HR labels; we train/test on agreement-filtered 10 m labels.

### Gaps
- Did not find the DFC2020 contest-outcome paper (Robinson et al. 2021, JSTARS) per-class table or a description of what winning teams did with S1. Not verified here — do not cite specifics.
- No per-class IoU for S1-only on DFC2020 found.

## 2. Other benchmarks: PANGAEA, reBEN/BigEarthNet-S1, SpecSAR-Former (Dynamic World labels), SSL4EO-S12, MultiSenGE

### Takeaway
Across benchmarks, S1-only consistently trails S2-only by a wide margin (reBEN: ~8 pts macro-AP; PANGAEA PASTIS-R: ~17 mIoU pts; SpecSAR-Former on Dynamic World labels: 23.6 vs 58.0 mIoU). Adding S1 to S2 gives small or no gains. Closest analogue to our setup (global, DW 9 classes, VV+VH annual mean) gives S1-only mIoU 23.6 / OA 51.1.

### Cited Findings
- **SpecSAR-Former (2024)**, labels = Dynamic World expert/non-expert annotations on 2019 S2; 9 DW classes (water, trees, grass, flooded vegetation, crops, shrub/scrub, built, bare, snow/ice); 16,893 train / 299 test patches 510×510 px; S1 = GRD IW VV+VH averaged over a 360-day window — [arXiv 2410.03962](https://arxiv.org/html/2410.03962)
- SpecSAR-Former modality ablation (Table V): S1 only (Mono) mIoU 23.61 / OA 51.07 / F1 34.90; S2 only mIoU 58.04 / OA 78.44; S2 RGB only 56.58 / 78.46; S1+S2 Mono 58.33 / 79.06; S1+S2 Dual (proposed) 59.58 / 79.48 — [arXiv 2410.03962](https://arxiv.org/html/2410.03962)
- SpecSAR-Former best model (S1+S2) per-class IoU: Water 88.87, Trees 75.85, Grass 25.10, Crops 46.55, Shrub/scrub 74.90, Flooded veg 40.36, Built 78.92, Bare 25.64, Snow/ice 80.08; Dynamic World's own map scores mIoU 52.22 / OA 72.86 on that test set — [arXiv 2410.03962](https://arxiv.org/html/2410.03962)
- **PANGAEA (Marsocci et al. 2024/2025)**: S1 used in Sen1Floods11 (S1,S2), PASTIS-R (S1,S2,SPOT-6), Crop Type Mapping-South Sudan (S1,S2,Planet), BioMassters (S1,S2); AI4SmallFarms is S2-only — [arXiv 2412.04204v2](https://arxiv.org/html/2412.04204v2)
- PANGAEA 100%-label mIoU: plain UNet baseline was Top-2 on 6 of 11 datasets (most of any model), e.g. Sen1Floods11 91.42, PASTIS 31.60, DynEarthNet 39.46, AI4Farms 46.34 — supervised UNet often beats GFMs — [arXiv 2412.04204v2](https://arxiv.org/html/2412.04204v2)
- PANGAEA PASTIS-R modality ablation (Table A3, mIoU, Linear / L-TAE temporal): CROMA joint 27.75/32.32, optical 32.06/38.51, SAR 20.68/21.61; S12-MAE SAR 18.81/14.20; paper: "multi-modality doesn't seem to bring any particular gain"; "utilizing optical data alone generally yields superior results compared to incorporating SAR information" — [arXiv 2412.04204v2](https://arxiv.org/html/2412.04204v2)
- PANGAEA Crop Type Mapping-SS (Table A12, Linear / L-TAE): CROMA joint 38.91/43.86, optical 47.02/49.38, SAR 35.39/50.44 (SAR-only with temporal L-TAE beats optical); S12-MAE SAR 28.51/21.06 — [arXiv 2412.04204v2](https://arxiv.org/html/2412.04204v2)
- TerraMind slide (metric unlabelled, likely mIoU): S1-only PASTIS 20.04, Sen1Floods11 80.39, CTM-SS 24.45; S2-only 40.20 / 89.57 / 50.90; multimodal 40.51 / 90.62 / 55.80 — [Jakubik, GRSS 2025 slides](https://www.grss-ieee.org/wp-content/uploads/2025/09/Day3_3_Johannes-Jakubik.pdf) (unverified against paper)
- **reBEN (BigEarthNet v2.0, multi-label classification, CLC-derived 19 classes)**: ResNet-50 macro AP ~62.7 S1, ~70.7 S2, ~70.8 S1+S2; MLP-Mixer Base S1 ~55.4; micro AP S1 ~74-81 vs S2 ~84-86; S1 added to S2 gives +0.1-0.5 macro AP — [reBEN, arXiv 2407.03653](https://arxiv.org/pdf/2407.03653) (via search snippet; values approximate)

### Inferences
- Gap S2-only minus S1-only: ~8 macro-AP (reBEN), ~17 mIoU (PASTIS-R), ~34 mIoU (SpecSAR / DW labels). Biggest gap occurs with expert labels and many vegetation classes.
- Temporal modelling matters most for SAR: CROMA SAR with L-TAE jumped 35.4 -> 50.4 mIoU on CTM-SS, overtaking optical. Supports our multi-temporal design.
- Our 0.54 mIoU S1-only is far above SpecSAR S1-only (0.236) on similar DW classes. Likely reasons: (a) agreement-filtered labels remove ambiguous pixels; (b) monthly composites + PCA vs a single annual mean; (c) 8 classes (shrub+grass merged) vs 9; (d) different test sets. Our number should not be claimed as SOTA versus expert-labelled benchmarks.

### Gaps
- SSL4EO-S12 downstream S1-only segmentation numbers: not found in this pass.
- MultiSenGE S1 results: not found.
- PANGAEA Sen1Floods11 S1-vs-S2 split: not in visible tables.
- Exact reBEN numbers for all models not verified against final version.

## 3. Label-quality ceiling: ESA WorldCover and Dynamic World accuracy

### Takeaway
Our labels come from two products that are each only ~74-77% correct versus independent reference, with shrub, grass and wetland the weakest (UA/PA ~30-50%). Agreement filtering raises label purity but also biases the test set toward easy pixels. Against expert references, any model trained on these labels likely cannot exceed ~75-80% OA; shrub/grass/wetland IoU near 0.3-0.5 is close to the products' own reliability.

### Cited Findings
- WorldCover 2021 v200 global OA 76.7±0.5% (2.16 M SSUs at 21,624 PSUs); WorldCover 2020 v100 ≈75% (other source: 74.4%). Uses S1+S2 — [WorldCover PVR v2.0](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PVR_V2.0.pdf); [CEOS](https://ceos.org/gst/worldcover.html)
- WorldCover v200 global per-class (Table 2), User's / Producer's accuracy %: Tree cover 80.0 / 91.9; Shrubland 49.1 / 46.9; Grassland 71.9 / 66.7; Cropland 80.6 / 79.3; Built-up 65.9 / 73.2; Bare/sparse 92.1 / 82.5; Snow & ice 93.0 / 99.1; Permanent water 89.4 / 86.4; Herbaceous wetland 30.5 / 44.6; Mangroves 74.1 / 46.2; Moss & lichen 57.5 / 46.4 — [WorldCover PVR v2.0](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PVR_V2.0.pdf) (extracted from PDF text layer)
- PVR text: shrubs, wetlands, moss/lichen "relatively lower accuracies globally"; Oceania lower due to shrubland-vs-grassland and trees-vs-grassland confusion in Australian open woodland; Siberian tundra shows shrubland-vs-trees and grassland-vs-trees confusion; continental OA all >72%, highest Asia 82.1% — [WorldCover PVR v2.0](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PVR_V2.0.pdf)
- Continental examples (PA): Africa shrubland 73.1, herbaceous wetland 45.7, cropland 58.9 (cropland under-represented in Africa); Europe shrubland 11.5, bare 28.0; South America herbaceous wetland 33.5, shrubland 46.0 — [WorldCover PVR v2.0](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PVR_V2.0.pdf)
- Dynamic World vs Expert Consensus (409 tiles, 68.1 M pixels): overall agreement 73.8%. Per class precision / recall: Water 90.6/94.3; Trees 70.2/93.2; Grass 30.1/33.8; Flooded veg 65.2/36.8; Crops 88.9/60.0; Shrub & scrub 54.7/44.7; Built 86.7/88.4; Bare 60.8/63.9; Snow & ice 71.2/94.2 — [Brown et al. 2022, Sci. Data (PMC9184477)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9184477/)
- Human ceiling: non-expert labels agree with Expert Consensus 77.8%; with "Three Expert Strict" 91.5% — [Brown et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9184477/)
- Other products vs DW expert set: LCMAP 2017 75.0%, CGLS ProbaV 2019 66.3%, ESA CCI 2018 61.6%, S2GLC Europe 59.2% — [Brown et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9184477/)
- Agreement-based labels: MOLCA (keep only pixels where all global maps agree; disagreement -> nodata) reported ~96% OA as a label set — [Russo et al. 2025, arXiv 2503.07230](https://arxiv.org/html/2503.07230v1)
- SEN12MS FROM-GLC10-based scheme validated at ~73% OA; MODIS LR labels vs HR DFC2020 give Shrubland 7.7%, Grassland 6.7% accuracy — [Schmitt et al. 2020](https://arxiv.org/pdf/2002.08254)

### Inferences
- Our weakest classes match the products' weakest: wetland (WC UA 30.5%, DW flooded-veg recall 36.8%) -> our IoU 0.22; shrub/grass (WC shrub UA 49.1%, DW grass P/R ~30%) -> our IoU 0.34. Low IoU partly reflects label noise, not only SAR limits.
- Agreement of WC and DW is not equal to ground truth: both are S2-driven (WC also S1) and can share systematic errors, so agreement-filtered labels still carry correlated error.
- Our test OA 0.83 exceeds both products' independent OA (~74-77%) only because it is measured against the agreement-filtered labels, not independent reference.
- Plausible ceiling for an honest independent evaluation: OA ~75-80%, shrub/grass/wetland F1 ~0.3-0.5.

### Gaps
- No direct published pixel-level agreement rate between WorldCover and Dynamic World found (Venter et al. 2022, Remote Sensing 14(16):4101 compares DW/WC/Esri, but the page returned 403).
- WorldCover v100 (2020) per-class table not extracted.

## 4. SAR-only global/regional LULC studies and SAR class confusions

### Takeaway
Regional S1-only deep models reach OA 0.84-0.97 when labels are agreement-filtered and classes are few; per-class producer accuracy collapses for shrubland, wetland, built-up in sparse regions. Recurrent confusions: shrubland vs grassland, wetland vs grassland, bare vs water/grass.

### Cited Findings
- Russo, Sorriso, Ullo, Gamba (2025): S1-only (GRD IW descending, VH only — adding VV gave no consistent gain), 4 seasonal mean "super images"/year after multitemporal despeckling + texture features (28 channels), Swin-Unet; labels MOLCA agreement map — [arXiv 2503.07230](https://arxiv.org/html/2503.07230v1)
- Russo et al. OA (Swin-Unet, seasonal): Amazonia 0.933, Africa 0.936, Siberia 0.974. Other models OA (Amazonia/Africa/Siberia): Attention U-Net 0.843/0.738/0.903; 3D-FCN 0.818/0.735/0.854; RF 0.622/0.745/0.677 — [arXiv 2503.07230](https://arxiv.org/html/2503.07230v1)
- Russo et al. per-class PA (Amazonia/Africa/Siberia): Forest 0.97/0.97/0.99; Shrubland 0.50/0.52/0.00; Grassland 0.88/0.83/0.86; Cropland 0.85/0.73/0.95; Wetland 0.03/0.16/0.79; Bare 0.00/0.93/0.75; Built-up 0.85/0.09/0.92; Water 0.99/0.77/0.98 — [arXiv 2503.07230](https://arxiv.org/html/2503.07230v1)
- Russo et al. confusions: shrubland confused with grassland (36% and 13% of cases); wetland often misclassified as grassland (few samples); built-up most confused where fragmented/scarce (Africa); bare overlaps water and grassland in backscatter distribution — [arXiv 2503.07230](https://arxiv.org/html/2503.07230v1)
- Russo et al.: seasonal composites beat dense time series (20 images/yr): Swin-Unet OA dropped to 0.903/0.872/0.896 with dense series — [arXiv 2503.07230](https://arxiv.org/html/2503.07230v1)
- Multitemporal S1 regional mapping with training points from coarse maps (RF, not DL): ~16% average OA gain over existing global products — [Frontiers in Remote Sensing 2023](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2023.1148328/full)
- VTT (IGARSS 2020): FC-DenseNet on wide-area SAR mosaics, CORINE 5 Level-1 classes, ~90% OA — [VTT CRIS](https://cris.vtt.fi/en/publications/classification-of-wide-area-sar-mosaics-deep-learning-approach-fo/)
- 2019 S1 C-band study (CORINE labels) compared U-Net, DeepLabV3+, PSPNet, BiSeNet, SegNet, FC-DenseNet, FRRN-B — [arXiv 1912.05067](https://arxiv.org/pdf/1912.05067v2) (numbers not extracted)
- Wetlands, S1 leaf-off time series, depth-adaptive U-Net (Delaware; emergent / scrub-shrub / forested wetland / open water): OA 0.93, mIoU 0.60 vs RF OA 0.89, mIoU 0.18 — reported in search snippet near [Remote Sensing 16(13):2427](https://www.mdpi.com/2072-4292/16/13/2427) (attribution not verified)
- Wetlands, S1+S2 CNN (Alberta): OA 80.2%, mean F1 0.58, +5% over XGBoost — search snippet near [Remote Sensing 12(1):2](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-12-00002/article_deploy/remotesensing-12-00002.pdf) (attribution not verified)
- Inundated vegetation can exceed three-quarters of total inundated area, yet common EO methods detect only open water (RadWet, 2023) — search snippet, [DOAJ Remote Sensing Mar 2023](https://doaj.org/article/13e6b9ee4cb2463195ff3e0a0b394e78) (attribution not verified)

### Inferences
- Russo's 0.93-0.97 OA is on agreement-filtered labels in 3 regions, the same label-cleaning trick as ours; it shows how much agreement filtering inflates OA. Our global 8-class OA 0.83 is in a consistent range.
- SAR physics explains confusions: shrub/grass/crop share similar volume scattering at C-band; herbaceous wetland mixes double-bounce (flooded vegetation) and specular (open water) responses and is temporally variable; snow (wet snow low backscatter) mimics water/bare. Matches our per-class ranking (shrub/grass 0.34, wetland 0.22, snow 0.49, water 0.54).
- Our built-up IoU 0.94 matches literature: built-up has strong double-bounce and is among the best SAR classes except where sparse.

### Gaps
- No global S1-only DL LULC paper with per-class IoU on independent expert labels found besides SpecSAR ablation (no per-class there).
- S1 urban-mapping-specific accuracy numbers not collected this pass.

## 5. Techniques with the biggest S1-only gains

### Takeaway
Best-documented S1 gains come from temporal aggregation (seasonal composites, temporal encoders) and model capacity/architecture; SAR-pretrained SSL encoders give small gains for SAR-only; VV adds little over VH in one study. Plain supervised U-Net remains a strong baseline.

### Cited Findings
- Seasonal composites (4/yr) + despeckling + texture: Swin-Unet OA 0.93-0.97 vs Attention U-Net 0.74-0.90 vs RF 0.62-0.75 on same data — [Russo et al. 2025](https://arxiv.org/html/2503.07230v1)
- Seasonal composites > dense series (~3-6 OA pts with Swin-Unet) — [Russo et al. 2025](https://arxiv.org/html/2503.07230v1)
- Temporal encoder: CROMA SAR-only on CTM-SS mIoU 35.39 (linear temporal) -> 50.44 (L-TAE); on PASTIS-R 20.68 -> 21.61 — [PANGAEA](https://arxiv.org/html/2412.04204v2)
- SSL pretraining (SSCL) for SAR-only: AA 54.8 -> 56.3 on DFC2020 (+1.5), vs +7.9 for optical-only — [arXiv 2108.09075](https://arxiv.org/html/2108.09075v2)
- GFMs (incl. SAR-capable CROMA, S12-MoCo/DINO/MAE/Data2Vec) do not consistently beat supervised UNet; UNet Top-2 on 6/11 datasets — [PANGAEA](https://arxiv.org/html/2412.04204v2)
- VV added to VH gave no consistent improvement (Russo project report) — [Russo et al. 2025](https://arxiv.org/html/2503.07230v1)
- Annual mean S1 VV+VH (single 360-day average) gives only 23.6 mIoU on DW expert labels — [SpecSAR-Former](https://arxiv.org/html/2410.03962)

### Inferences
- Our monthly composites + PCA of annual series is aligned with the best-documented SAR lever (temporal information). Next candidates: explicit temporal encoder (L-TAE / 3D conv over 12 monthly frames) instead of PCA; texture/despeckling channels; transformer decoder (Swin-Unet).
- DEM/polarimetric (VH/VV ratio) features: no quantified S1-only gain found in this pass; treat as plausible but unproven.

### Gaps
- No quantified DEM gain for S1-only LULC found.
- No quantified gain for dual-pol decomposition (H/alpha) on Sentinel-1 GRD found.
- SAR-specific pretrained encoders beyond CROMA/SSL4EO (e.g. SAR-only foundation models) not evaluated here.
