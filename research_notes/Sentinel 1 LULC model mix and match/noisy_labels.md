# Noisy-label training methods for land-cover segmentation from map-derived labels (Sentinel-1 LULC fit)

Scope note: 15 tool calls used. Several key PDFs (DFC2020 outcome paper, CADF-Net, WorldCover PVR) returned binary / abstract-only content, so their numbers are listed as gaps. Method-origin papers (GCE, SCE, APL, ELR, co-teaching, bootstrapping, forward correction) are cited by their canonical arXiv pages for what the method is; no remote-sensing gain is claimed for them unless a RS source is given.

## 1. Which noisy-label methods have been shown to help land-cover segmentation trained on existing maps?

### Takeaway
Direct, controlled evidence in RS land-cover segmentation is thin and the measured gains of loss-level tricks are small (+0.2 to +2 mIoU). The bigger documented levers are: more/longer training, self-training / pseudo-label refinement (+0.8 mIoU in DFC2025 SAR), cross-modal / multi-view sample selection (+1.3 to +3.8 mIoU in CromSS), and simply that deep models trained on noisy maps already beat the map itself. Uniform label smoothing hurt in one RS study.

### Cited Findings
**Symmetric cross-entropy (SCE), SAR, map/pseudo-label noise (closest analogue to our setup)**
- Liu et al. 2025 ("Learning from Noisy Pseudo-labels for All-Weather Land Cover Mapping", DFC2025 Track 1 winner, SAR-only input, OpenEarthMap-SAR with 4,333 paired training images and optical-derived pseudo-labels "full of errors"): adding SCE to CE+Lovász raised val mIoU 35.12 -> 35.31 (+0.19); CE -> CE+Lovász +0.29; self-trained pseudo-labels (DACS semi-supervised pipeline) instead of official labels +0.80; 40k -> 160k iterations +1.57 (largest single gain); ensemble +0.61 over best single model. — [arXiv 2504.13458 (HTML)](https://arxiv.org/html/2504.13458)
- Same paper: loss combo CE + Lovász + SCE is essentially our current loss (CE + Lovász) plus one term. — [arXiv 2504.13458](https://arxiv.org/abs/2504.13458v1)

**Cross-modal sample selection + label smoothing on Dynamic World / OSM labels (CromSS, Liu et al., IEEE TGRS)**
- Labels: Dynamic World 9-class, "overall accuracy of approx. 70%", 103,793 SSL4EO-S12 locations. — [CromSS arXiv 2405.01217v3](https://arxiv.org/html/2405.01217v3)
- Sample selection: per-pixel confidence from softmax at noisy label + prediction entropy, combined across S1 and S2 models, thresholded into soft weights; selection ratio decays 1 -> 0.5 over first 80 epochs. — [CromSS](https://arxiv.org/html/2405.01217v3)
- Gains of CromSS-midF over single-modal (frozen encoder mIoU): DFC2020 40.27 -> 42.52 (+2.25), SSL4EO@DW 59.30 -> 60.56 (+1.26), SSL4EO@OSM 45.29 -> 49.06 (+3.77). — [CromSS](https://arxiv.org/html/2405.01217v3)
- Sample selection did NOT beat plain multi-modal fusion on the DW-labelled test; authors attribute it to selection masks suppressing boundaries and fine details. — [CromSS](https://arxiv.org/html/2405.01217v3)
- Label smoothing ablation (frozen / fine-tuned mIoU): none 43.26/42.63 (DFC2020), 61.07/61.26 (DW), 47.24/49.80 (OSM); uniform smoothing only 41.64/42.32, 58.65/59.36, 47.67/50.57 — i.e. uniform smoothing lowered DFC2020 and DW scores by 0.3–2.4 points. Spatial-temporal (Gaussian over neighbours and 4 seasons) smoothing: 42.30/43.52, 60.98/61.61, 46.98/49.49. Combined (alpha 0.15, mu 0.05): 42.52/43.21, 60.64/60.66, 49.00/51.28. Authors claim stability across datasets, not best per dataset. — [CromSS](https://arxiv.org/html/2405.01217v3)

**Noisy-map pretraining**
- Task-specific pretraining on noisy Dynamic World / OSM masks (103,793 locations, 9 classes) then fine-tuning; OSM labels "can significantly reduce human supervision required". — [Liu et al. arXiv 2402.16164](https://arxiv.org/html/2402.16164v3)

**Teacher / pseudo-label refinement on WorldCover**
- Forest mapping study: WorldCover label noise hurts accuracy; teacher model's high-confidence pseudo-labels relabel uncertain regions (abstract only seen, no numbers). — [Sciforum 27856](https://sciforum.net/index.php/paper/view/27856)
- AIO2 (building footprints): mean-teacher + adaptive switching to stabilise training on noisy labels; names Dynamic World, WorldCover, Esri as products that need careful handling. — [arXiv 2403.01641](https://ar5iv.labs.arxiv.org/html/2403.01641)

**Early-learning correction (segmentation)**
- ADELE (CVPR 2022): per-class IoU curves detect memorisation onset; then correct labels with the model's early predictions + multiscale consistency. Tested on medical (synthetic noise) and PASCAL VOC, not RS. Authors warn: with poor-quality annotations gains may be "marginal or even negative", and structured noise may prevent early learning. — [ADELE CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Adaptive_Early-Learning_Correction_for_Segmentation_From_Noisy_Annotations_CVPR_2022_paper.html); [arXiv 2110.03740](https://arxiv.org/abs/2110.03740)

**Other RS noise-robust losses / benchmarks**
- NRN-RSSEG (Remote Sensing 2023) noise-robust network for RS segmentation; cites Feng et al. Taylor cross-entropy (TCE) for RS label noise. — [Remote Sens. 15(1):108](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-15-00108/article_deploy/remotesensing-15-00108.pdf)
- Nogueira et al. 2026: data-centric benchmark ranking RS segmentation images by pixel-label error (synthetic noise injection); cites real land-cover datasets with noise ratios 8.0 %–38.5 %; code at github.com/keillernogueira/label_noise_segmentation. — [arXiv 2603.00604](https://arxiv.org/html/2603.00604)
- Kröber et al. 2026 (ISPRS Annals): benchmark of data-centric noisy-sample filtering at synthetic noise 10–70 %. — [ISPRS Annals XI-2-2026-839](https://isprs-annals.copernicus.org/articles/XI-2-2026/839/2026/isprs-annals-XI-2-2026-839-2026.pdf)

**Method-origin references (generic, no RS gain claimed here)**
- GCE (Zhang & Sabuncu 2018): interpolates CE and MAE via exponent q. — [arXiv 1805.07836](https://arxiv.org/abs/1805.07836)
- SCE (Wang et al. 2019): CE + reverse CE. — [arXiv 1908.06112](https://arxiv.org/abs/1908.06112)
- Active-Passive Loss (Ma et al. 2020, e.g. NCE+RCE / NCE+MAE). — [arXiv 2006.13554](https://arxiv.org/abs/2006.13554)
- ELR (Liu et al. 2020): regulariser pulling predictions to temporal-ensemble targets. — [arXiv 2007.00151](https://arxiv.org/abs/2007.00151)
- Co-teaching (Han et al. 2018): two nets exchange small-loss samples. — [arXiv 1804.06872](https://arxiv.org/abs/1804.06872)
- Bootstrapping (Reed et al. 2014) — already planned. — [arXiv 1412.6596](https://arxiv.org/abs/1412.6596)
- Forward / backward loss correction with transition matrix (Patrini et al. 2017). — [arXiv 1609.03683](https://arxiv.org/abs/1609.03683)

### Inferences
- Our loss already = CE + Lovász. The DFC2025 SAR winner's increments suggest SCE adds ~+0.2 mIoU, well below our ±0.02 (2 mIoU points) run-to-run noise. Not detectable without many seeds.
- Planned "label smoothing 0.2" has negative RS evidence (CromSS uniform smoothing -0.3 to -2.4 mIoU on DW-labelled data). Expect flat or worse. A spatial (neighbourhood-Gaussian) or seasonal smoothing is the RS-specific variant; our 9 seasonal stats channels are an analogue of CromSS's temporal dimension.
- Longest documented lever in the SAR case was training length (+1.57). Our val plateau at 0.73 with 0.12 train-val gap suggests memorisation, not under-training — fits ELR/ADELE-style "stop fitting disagreement pixels late in training" more than longer training.
- Small-loss selection on disagreement pixels risks removing boundaries and rare classes (CromSS finding). Apply selection only on disagreement (weight-0.5) pixels, never on agreement pixels, and per class (class-wise quantile) to protect Grassland/Bare/Snow.
- Self-training (teacher pseudo-labels replace low-confidence disagreement pixels, then retrain) has the best evidence/cost ratio: +0.8 mIoU on SAR DFC2025, also used in WorldCover forest work.

### Gaps
- No paper found with a head-to-head GCE vs SCE vs APL vs CE on a real land-cover map label (WorldCover/DW/CORINE/NLCD). Gains of GCE/APL/ELR/co-teaching in RS land cover remain unquantified here.
- DFC2020 winning-team scores and their noise handling (IEEE JSTARS 14:3185–3199, [doi 10.1109/JSTARS.2021.3063849](https://dx.doi.org/10.1109/JSTARS.2021.3063849)) not retrieved; only abstract page was readable ([MSR page](https://www.microsoft.com/en-us/research/?p=771013)).
- DFC2021 weak-label track, GeoNRW, OpenEarthMap, MapInWild noisy-label results not searched (budget).
- Transition-matrix (forward correction) in RS land cover: no source found.

## 2. Soft labels from multiple land-cover products (WorldCover + Dynamic World probabilities + Esri) instead of hard label: evidence?

### Takeaway
Plausible and used in practice, but no peer-reviewed controlled ablation (soft multi-product vs hard single-product) with numbers was retrieved. Closest: CADF-Net (2026) fusing products with a conflict index + distillation, reported to beat the best input product (Esri); a Kenya study using confidence-weighted product fusion for labels. CromSS used DW one-hot, not DW probabilities.

### Cited Findings
- CADF-Net (Remote Sensing 18(9):1294, 2026): U-Net with knowledge distillation; sections "Conflict Index Construction", "Two-Stage Training and Adaptive Distillation", "Confidence-Guided Dynamically Weighted Ensemble"; search summary says it reports gains over the strongest input product, Esri Land Cover. Exact numbers not extractable (PDF binary). — [CADF-Net PDF](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-01294/article_deploy/remotesensing-18-01294.pdf)
- Kenya (ISPRS Archives XLVIII-4-W21-2025): confidence-weighted fusion of GlobeLand30, FROM-GLC, FROM-GLC10 to build synthetic training labels. — [ISPRS Archives](https://isprs-archives.copernicus.org/articles/XLVIII-4-W21-2025/19/2026/isprs-archives-XLVIII-4-W21-2025-19-2026.pdf)
- East European Plain study fuses Dynamic World, WorldCover, Esri, GlobeLand30, Open Land Map labels. — [DOAJ record](https://openpolar.no/Record/ftdoajarticles:oai:doaj.org%2Farticle:28f21161ac8f4542baa1e1d728ea1ba0)
- Tool doc (not a paper): pytorch_segmentation_models_trainer supports probabilistic labels from several land-cover sources plus per-pixel confidence weighting. — [docs](https://dsgoficial.github.io/pytorch_segmentation_models_trainer/docs/user-guide/soft-label-training/)
- Product accuracies are reference-dependent: global reference (MMU 250 m²) Esri 75 %, DW 72 %, WorldCover 65 %; LUCAS Europe (MMU <100 m²) WorldCover 71 %, DW 66 %, Esri 63 %. Esri global figure possibly inflated (trained on DW validation set). Esri and DW come from models trained on the same reference set → their errors are correlated. — [Venter et al. 2022, Remote Sens. 14:4101](https://www.mdpi.com/2072-4292/14/16/4101/html); [PNNL summary](https://www.pnnl.gov/publications/global-10-m-land-use-land-cover-datasets-comparison-dynamic-world-world-cover-and-esri)
- Independent 2024 study (IIASA): global OA 73.4 % ± 0.7 to 83.8 % ± 0.4, WorldCover highest, then DW, then Esri. — [RSE 2024, IIASA pure](https://pure.iiasa.ac.at/19901/1/1-s2.0-S0034425724003341-main.pdf)
- Sentinel-1+2 SpecSAR-Former trained on Dynamic World labels (16,893 patches); accuracy not retrieved. — [arXiv 2410.03962](https://arxiv.org/pdf/2410.03962)

### Inferences
- Our weight scheme (1 agree / 0.5 disagree) is already a crude 2-product fusion. Natural next step: target = normalised mix, e.g. 0.5·onehot(WC) + 0.5·DW-probability vector (DW publishes per-class probabilities), or 3-way vote with Esri. Cheap (label preprocessing + soft CE, which PyTorch CE supports with probability targets).
- Because DW and Esri errors are correlated (same training reference), a 3-product majority overcounts their shared errors; weight WorldCover higher or treat DW+Esri as one vote.
- Class remap issue: DW has 9 classes (incl. flooded vegetation, shrub & scrub); WorldCover 11; Esri 9. Mapping to our 7 classes needs explicit rules; Shrub is the main ambiguity with Grassland/Trees and likely drives the 55 % Grassland agreement.

### Gaps
- No retrieved quantitative ablation of soft multi-product labels vs hard labels for segmentation.
- No source on whether DW probability (vs DW label) targets improve calibration or mIoU.

## 3. Expected magnitude of gains and conditions

### Takeaway
On realistic map noise, loss-level robustness gives roughly +0.2 to +2 mIoU; pseudo-label refinement ~+1; multi-view/multi-modal selection +1 to +4; architectural/training-length changes often larger. Most gains are within our ±2-point seed noise, so multi-seed evaluation is needed.

### Cited Findings
- SCE +0.19, Lovász +0.29, self-training pseudo-labels +0.80, 4x iterations +1.57, ensemble +0.61 mIoU (SAR, DFC2025). — [arXiv 2504.13458](https://arxiv.org/html/2504.13458)
- Cross-modal training with selection +1.26 to +3.77 mIoU vs single modality; uniform label smoothing -0.3 to -2.4 on DW/DFC2020. — [CromSS](https://arxiv.org/html/2405.01217v3)
- ADELE: gains shrink or turn negative when annotations are poor or noise is structured. — [arXiv 2110.03740](https://arxiv.org/abs/2110.03740)

### Inferences
- Map noise (WorldCover vs DW) is structured (class-confusion and boundary-dependent), not uniform; synthetic-noise benchmark gains (often 5–20 points) will not transfer.
- Condition for any method to show: evaluate on agreement pixels across ≥3 seeds; expect differences ≤ 0.02 for single loss swaps.

### Gaps
- No RS paper found reporting GCE/APL/ELR gains on WorldCover-like labels; magnitudes for those are unknown.

## 4. Is the ceiling of training on a ~75 %-accurate map understood quantitatively?

### Takeaway
Partially. DFC2020 baseline study shows CNNs trained on very noisy MODIS labels exceed the labels' own accuracy (U-Net 48.1 % vs label 37.2 % average accuracy). Reference accuracies of the label products are known (WorldCover 2021 76.7 % OA). No paper retrieved measures "model trained on product X scored against product X" vs independent ground truth as a ceiling.

### Cited Findings
- Schmitt et al. 2020 (DFC2020 baselines, SEN12MS MODIS labels vs 10 m hand labels): label average accuracy 37.2 % (Forest 51.6, Grassland 6.7, Croplands 64.4, Urban 71.5, Water 95.1, Barren 0.3). Models trained on these labels: U-Net S2 48.1 %, U-Net S1+S2 44.6 %, DeepLabv3+ S2 44.2 %, RF ~30 %. Grassland jumps from 6.7 % (labels) to 76.7 % (U-Net). Caveat: deep numbers from best-test checkpoint (upper bound). — [arXiv 2002.08254 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2002.08254)
- Schmitt et al.: even off-the-shelf models are "significantly better than the low-resolution reference". — [arXiv 2002.08254](https://ar5iv.labs.arxiv.org/html/2002.08254)
- WorldCover 2021 v200 OA 76.7 % ± 0.5 (global), 76.5 % ± 1.3 Africa; validation by Wageningen (statistical) + IIASA (spatial); confusion matrix in PVR v2.0 Table 2 (not retrieved). — [ESA WorldCover 2021](https://worldcover2021.esa.int); [PVR V2.0](https://pure.iiasa.ac.at/18981/1/WorldCover_PVR_V2.0.pdf)
- DW labels ≈ 70 % OA (per CromSS authors). — [CromSS](https://arxiv.org/html/2405.01217v3)
- DFC2020 Track 1 setup: train on MODIS-derived noisy low-res labels only, predict 10 m; Track 2 adds few clean labels. — [arXiv 2002.08254](https://arxiv.org/pdf/2002.08254); [JSTARS outcome](https://dx.doi.org/10.1109/JSTARS.2021.3063849)
- Label super-resolution line (Malkin, Robinson, Jojic): train 1 m land cover from 30 m NLCD; naive label upsampling performs poorly (Workman et al. 2022). — [arXiv 2101.01154](https://arxiv.org/abs/2101.01154); [arXiv 2211.15790](https://ar5iv.labs.arxiv.org/html/2211.15790)

### Inferences
- Scoring on WorldCover∩DW agreement pixels measures agreement with a consensus label, not truth. The 0.73 val mIoU plateau is bounded by (a) S1-only separability and (b) residual error in consensus pixels. Agreement pixels are biased to easy, homogeneous areas, so true-accuracy can be lower than agreement-pixel mIoU.
- A model can exceed its label map accuracy (Schmitt result), so plateau ≠ hard 75 % ceiling. To check, need a small independent reference (e.g., LUCAS points for European sites, or DW expert-annotated validation tiles).
- Product-vs-product agreement at our sites (73.6 %) is itself close to each product's reported OA (65–77 %), consistent with both maps being ~75 % accurate with partly independent errors.

### Gaps
- No retrieved study reports self-consistency ceilings ("trained on X, scored on X") vs independent truth for WorldCover/DW.
- DFC2020 final contest scores not retrieved.
- WorldCover per-class Grassland producer/user accuracy not retrieved (in PVR Table 2).

## 5. Cheap (loss change only, <1 h on one GPU) vs costly

### Takeaway
Most robust losses and weighting tricks are drop-in (minutes to code, one training run each). Self-training, co-teaching, multi-product soft labels and ADELE cost 2x training or extra data export.

### Cited Findings
- SCE / GCE / APL / bootstrapping / ELR = per-pixel loss replacements; ELR needs one per-pixel EMA target buffer (memory: N_pixels x classes; 2218 x 256² x 7 floats ≈ 4 GB fp32, so store fp16 or at reduced resolution). — method papers: [GCE](https://arxiv.org/abs/1805.07836), [SCE](https://arxiv.org/abs/1908.06112), [APL](https://arxiv.org/abs/2006.13554), [ELR](https://arxiv.org/abs/2007.00151)
- CromSS confidence weighting = per-pixel softmax-at-label + entropy weights, ratio schedule 1 -> 0.5 over 80 epochs; needs a second (S2) model for its cross-modal part. — [CromSS](https://arxiv.org/html/2405.01217v3)
- Self-training pseudo-labels: teacher run + relabel + retrain (2x cost). — [arXiv 2504.13458](https://arxiv.org/html/2504.13458)
- Co-teaching: two networks trained jointly (2x GPU cost). — [arXiv 1804.06872](https://arxiv.org/abs/1804.06872)
- ADELE: per-class IoU monitoring + label correction + multiscale consistency (extra forward passes). — [ADELE](https://arxiv.org/abs/2110.03740)

### Inferences (ranked for our setup; cost = engineering, each needs one full training run on A6000)
1. Cheap — SCE or NCE+RCE (APL) on disagreement pixels only, keep CE+Lovász on agreement. Expected ≤ +0.5 mIoU.
2. Cheap — Class-wise small-loss truncation: drop top-k % highest-loss disagreement pixels per class per batch (k ramp 0 -> 30 %). Protects rare classes.
3. Cheap — DW probability soft targets on disagreement pixels (needs DW probability bands exported from GEE: 9 extra float bands per patch; data prep, not loss only).
4. Cheap-medium — ELR regulariser (EMA buffer). Targets the train-val gap / memorisation.
5. Medium — One round self-training: train current model, replace disagreement-pixel labels where teacher confidence > 0.9, retrain. Best-evidenced (+0.8 in SAR).
6. Medium — Spatial label smoothing (Gaussian neighbourhood) instead of uniform 0.2; replace the planned uniform-0.2 run.
7. Costly — Co-teaching, ADELE, CADF-Net-style distillation, noise transition matrix. Low expected payoff given structured noise and ±0.02 seed noise.

### Gaps
- Wall-clock per method not reported in RS sources; costs above are engineering estimates.
