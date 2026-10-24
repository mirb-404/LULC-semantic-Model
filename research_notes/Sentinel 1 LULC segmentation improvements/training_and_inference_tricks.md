# Training and inference tricks for noisy-label, small-data Sentinel-1 LULC segmentation

Setup under study: U-Net ResNet-34 (ImageNet), AdamW 3e-4, ReduceLROnPlateau, weighted CE + 0.5 Dice, labels = ESA WorldCover 2021 / Dynamic World 2021 agreement (~18% ignored). Test OA 0.83, mIoU 0.54; wetland 0.22, shrub/grass 0.34, cropland 0.49. Val mIoU swings 0.656 -> 0.583 between epochs. Budget: Kaggle T4, 1-2 h per run.

Research date: 2026-10-24. Tool budget limited (~15 calls). Many items below have only one reported number, or a number from a different domain. Treat cross-domain gains as rough guides.

## Q1. Losses for noisy labels and imbalance (label smoothing, SCE, GCE, bootstrapping, focal, Lovász, OHEM, soft labels, disagreement pixels)

### Takeaway
Measured gains from loss swaps are small: about +0.2 to +2.5 mIoU. Lovász-softmax has the best evidence (CVPR 2018: about +1 to +2.5 mIoU on VOC; DFC2025 SAR: about +0.3). Symmetric CE added only +0.19 mIoU in the DFC2025-winning SAR land-cover solution. The loss is not the main lever. Label quality and the val protocol likely matter more.

### Cited Findings
- Lovász-softmax (Berman, Rannen Triki, Blaschko, CVPR 2018) is a tractable surrogate for optimizing IoU. Authors report "substantially better" segmentation than cross-entropy on Pascal VOC and Cityscapes — [CVPR 2018 paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.pdf); [arXiv 1705.08790](https://arxiv.org/pdf/1705.08790)
- arXiv table (DeepLab-v2, VOC val): CE baseline about 76.4 mIoU; Lovász + equibatch (30K iterations) about 79.0; best CE with equibatch about 78.1. That suggests roughly +1 to +2.5 mIoU. Caution: the table extraction was garbled, so column alignment is uncertain — [arXiv 1705.08790](https://arxiv.org/pdf/1705.08790)
- Official PyTorch implementation: github.com/bermanmaxim/LovaszSoftmax — [GitHub mirror listing](https://github.com/HsinJhao/LovaszSoftmax)
- **DFC2025 Track 1 winner** (Liu, Wang, Guo, Duan, Kang, Li; Hunan University; arXiv 2504.13458, 18 Apr 2025). Task: all-weather land cover from SAR (OpenEarthMap-SAR) with noisy pseudo-labels derived from optical models. This is the closest analogue to the present setup — [arXiv 2504.13458](https://arxiv.org/html/2504.13458v1)
  - Ablation, val mIoU (SegFormer MiT-B3, CE baseline 32.35): CE+Lovász 32.64 (+0.29) — [same](https://arxiv.org/html/2504.13458v1)
  - MiT-B5 + their pseudo-labels, CE+Lovász 35.12; adding symmetric CE (SCE) 35.31 (+0.19). Only one isolated SCE comparison — [same](https://arxiv.org/html/2504.13458v1)
  - Longer training (40k -> 160k iterations) gave 32.35 -> 33.92 (+1.57). That beat the loss change and the larger backbone (MiT-B5 alone: 32.72, +0.37) — [same](https://arxiv.org/html/2504.13458v1)
- NRN-RSSEG (Remote Sensing, Dec 2022): noise-robust loss + attention, ISPRS Vaihingen with simulated symmetric and asymmetric noise. At noise rate 0.5, U-Net FWIoU dropped 9.75%; NRN-RSSEG dropped 3.26. It reports FWIoU, not mIoU. The exact loss was not confirmed — [MDPI Remote Sensing 15(1):108](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-15-00108/article_deploy/remotesensing-15-00108.pdf)
- Bootstrapping, defined as "combines original noisy labels and model predictions as soft reference data", is one baseline in AIO2 (building segmentation, incomplete labels, arXiv 2403.01641). The table numbers could not be extracted (PDF binary) — [AIO2 arXiv 2403.01641](https://arxiv.org/pdf/2403.01641)
- Bootstrapped Dice loss (ICPR 2020) improved road extraction under severe OSM-style label noise. No IoU numbers in snippets — [ICPR author page](https://ailb-web.ing.unimore.it/icpr/author/2966)
- Self-filtered learning (IEEE JSTARS, Jan 2023) iteratively drops high-loss samples over rounds for OSM / Microsoft building labels — [DOAJ record](https://doaj.org/article/524106097f394f87b6a68c674924e4bd)
- Bayesian spatially-correlated noisy-label segmentation (2025): CE lost about 10.9 IoU points under moderate-to-high noise — [arXiv 2504.14795](https://arxiv.org/pdf/2504.14795)
- CromSS (Cross-modal Sample Selection, LMU/MCML 2024) uses class distributions from multiple sensors to select clean samples. Conceptually close to the WorldCover/DW agreement mask — [MCML](https://mcml.ai/publications/law+24a/)
- Task-specific pretraining with noisy segmentation maps (IGARSS 2024) shows noisy maps are useful for pretraining — [arXiv 2402.16164](https://arxiv.org/html/2402.16164v3)
- Dynamic World (Sentinel-2, 10 m) public test tiles include per-class probability layers plus the top-1 label. Such probabilities could serve as soft targets — [Zenodo DW test tiles](https://zenodo.org/record/4766450)
- Soft-label training/eval for segmentation (Waterloo MSc thesis, Jamali, 2026): weighted average pooling to make soft labels at any resolution; soft-IoU relaxations for evaluation. Not tied to DW or Sentinel — [UWSpace](https://uwspace.uwaterloo.ca/items/5b50be37-7e81-4448-a0dd-c65ff8e88d35)

### Inferences
- Rank for this project: (1) add Lovász-softmax (smp `LovaszLoss`, multiclass, `ignore_index`) to CE, or replace Dice with it. Cost: about 0 extra time; expect about +0.3 to +1.5 mIoU. (2) Label smoothing 0.05-0.1 on CE: one-line change (`nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=...)`). No RS segmentation number found. (3) SCE/GCE: low cost, but measured gain is +0.19 in the closest analogue.
- The consensus mask already removes most label noise, so robust losses have less noise to fix. The remaining problem is bias: hard classes (wetland, shrub/grass) are under-represented in agreed pixels. Disagreement pixels could be added with a soft target, e.g. 0.5 WorldCover class + 0.5 DW class, or with the DW probability vector. This puts training signal into exactly the weak classes. Untested here; flag as experiment.
- Focal loss and OHEM push weight onto hard pixels. With noisy labels, hard pixels are often mislabelled pixels. So focal/OHEM may hurt. The current weighted CE already handles imbalance.
- Combining sqrt median-frequency weights with WeightedRandomSampler double-counts rarity. This may raise rare-class recall but lower precision. Worth one ablation without one of them.

### Gaps
- No exact SCE (Wang et al., ICCV 2019) or GCE (Zhang & Sabuncu, NeurIPS 2018) mIoU numbers on RS segmentation were found. Both papers report classification only.
- No found study trains on Dynamic World probabilities as soft targets. No found study trains on WorldCover-vs-DW disagreement pixels.
- Label smoothing in RS segmentation: no reported mIoU found.
- AIO2 table values (bootstrapping vs CE IoU) not retrieved.

## Q2. LR schedules, EMA/SWA, differential LR

### Takeaway
Weight averaging gives about +0.8 to +1.1 mIoU in reported segmentation cases and smooths noisy val curves at near-zero cost. Longer/better-scheduled training gave +1.57 mIoU in the DFC2025 SAR ablation. No direct OneCycle/cosine vs ReduceLROnPlateau segmentation comparison was found.

### Cited Findings
- EMA of weights (decay 0.999), used for every val checkpoint and final eval, added +1.08 mIoU over a photometric-distortion baseline. Authors say it also reduced speckle artifacts in large homogeneous regions (ICRA 2025 GOOSE 2D challenge technical report) — [arXiv 2505.11769](https://arxiv.org/pdf/2505.11769)
- Morales-Brotons et al. (TMLR 2024, classification on CIFAR/Tiny-ImageNet): EMA models generalize better and show improved robustness to noisy labels, prediction consistency and calibration — [arXiv 2411.18704](https://arxiv.org/pdf/2411.18704)
- SWA gave +0.77 mIoU over a single model on BDD100K val (DeepLabV3+). The paper also compares SWA vs deep ensembles — [arXiv 2207.11211](https://arxiv.org/pdf/2207.11211)
- DFC2025 SAR winner used an EMA teacher (alpha 0.999) in self-training. No EMA ablation reported. Longer schedule 40k -> 160k iterations: +1.57 val mIoU — [arXiv 2504.13458](https://arxiv.org/html/2504.13458v1)
- Caution: "stochastic average pooling" (arXiv 2409.16630, Potsdam/Vaihingen, about +0.2 to +0.8 mIoU) is a different method from SWA — [arXiv 2409.16630](https://arxiv.org/pdf/2409.16630)

### Inferences
- Highest value-per-effort item for the noisy val curve: keep an EMA copy (`torch.optim.swa_utils.AveragedModel` with an EMA `multi_avg_fn`, or `timm.utils.ModelEmaV2`, decay about 0.999 per step). Validate and save the EMA model. Cost: one extra model copy, negligible on T4. Remember BN: EMA of BN buffers, or recompute BN stats.
- At ~156 steps/epoch (2500/16), decay 0.999 averages over about 6 epochs. Decay 0.998-0.999 fits.
- Replace ReduceLROnPlateau, which reacts to a noisy val signal. Use a fixed schedule instead: linear warmup 1-2 epochs, then cosine to about 1e-6 over a fixed epoch count; or OneCycle. Fixed schedules decouple LR from val noise.
- Differential LR: encoder about 1e-4 to 3e-4 x 0.1, decoder 3e-4. Standard practice for ImageNet encoders on SAR. ImageNet features transfer less well to 2-channel SAR, so a 0.3-0.5 encoder multiplier may be better than 0.1. Untested.
- SWA over the last 20-25% of a cosine run is an alternative to EMA. Needs `update_bn` at the end.

### Gaps
- No controlled OneCycle vs cosine vs ReduceLROnPlateau segmentation comparison found.
- No EMA/SWA ablation on an RS segmentation benchmark (ISPRS, LoveDA, DeepGlobe) found.
- No differential-LR ablation for SAR found.

## Q3. Augmentation for SAR (mixup/cutmix, speckle, month mixing, scale)

### Takeaway
Cut-and-paste augmentation gave the largest single number found: +6.2 mIoU (37.9 -> 44.1) on DynamicEarthNet with U-Net. CutMix in semi-supervised low-label regimes can underperform. No SAR-specific speckle-augmentation ablations with mIoU were found.

### Cited Findings
- Cut-and-Paste augmentation (paste object/class regions from one image into another) raised test mIoU 37.9 -> 44.1 on DynamicEarthNet (monthly Planet time series, LULC change), U-Net. Motoi, Saraceni, Nardi, Ciarfuglia; IGARSS 2024, pp. 9802-9806; arXiv v1 8 Apr 2024 — [arXiv 2404.05693](https://arxiv.org/abs/2404.05693)
- DFC2025 SAR winner used random resized crop, flip, rotate, colour jitter, plus a resolution-alignment augmentation. No isolated augmentation ablation — [arXiv 2504.13458](https://arxiv.org/html/2504.13458v1)
- In a LoveDA 1%-label semi-supervised table, a CutMix baseline scores about 16.5 mIoU vs supervised-only about 36.0 and mean teacher about 38.7. The same 16.52 value appears in a second paper's table. Caution: the gap is so large it may reflect a weak re-implementation — [arXiv 2501.10736](https://arxiv.org/pdf/2501.10736); [arXiv 2602.00202](https://arxiv.org/pdf/2602.00202)
- ClassMix mixes two images using a mask built from predicted class segmentations; it generalizes CutMix — [ClassMix arXiv 2007.07936](https://ar5iv.labs.arxiv.org/html/2007.07936)
- A Sentinel-1 flood segmentation study used flips plus intensity scaling to make 12 augmented versions per input. No gain figure in snippet — [arXiv 2606.16302](https://arxiv.org/pdf/2606.16302)

### Inferences
- Class-aware copy-paste is the best fit for weak classes: paste wetland / shrub-grass / cropland label regions (with their SAR pixels) into other crops. Implementable in the Dataset in about 30 lines. Expected cost: small CPU overhead. Risk: SAR backscatter context mismatch at paste edges.
- Multiplicative speckle (gamma-distributed, mean 1, in linear power before dB) matches SAR physics better than additive 0.5 dB noise. In dB, multiplicative noise becomes additive, so the current ±1 dB shift + noise already approximates it. Expect a small gain at most.
- Month mixing: if 12 monthly composites exist per site, sample a random month per crop (labels are annual). This gives free extra data and invariance to seasonal backscatter. Likely strong for cropland and wetland, which change seasonally. Also consider stacking several months as channels; this is a model change, not an augmentation.
- Scale augmentation (0.75-1.25 random resize) is cheap. At 10 m Sentinel-1, the object scale is fairly fixed, so expected gain is small.

### Gaps
- No SAR-specific speckle augmentation ablation with mIoU found.
- No month-mixing augmentation study found.
- No mixup-for-segmentation result in RS found.

## Q4. Inference: TTA, sliding window blending, ensembling, temporal smoothing

### Takeaway
Overlapping sliding windows: about +1% reported, plus removal of seams. Flip TTA: inconsistent (≈0 in one orbital study, +1.6% in LiDAR). Ensembling: +0.61 mIoU over the best single model in DFC2025. HMM temporal smoothing is a documented way to remove spurious class transitions in land-cover time series.

### Cited Findings
- Sliding window with stride 32 px (75% overlap) gave about +1% over no overlap; it slows inference significantly (Audebert et al., multimodal EO segmentation) — [arXiv 1711.08681](https://arxiv.org/pdf/1711.08681)
- ResUNet-a averages predictions from 256 px windows at 64 px stride. Rationale: edge pixels have less context and lower confidence — [arXiv 1904.00592](https://arxiv.org/pdf/1904.00592)
- Albumentations tutorial: D4 (8 flips/rotations) TTA is the natural group for top-down aerial imagery. No mIoU numbers — [Albumentations D4 TTA example](https://albumentations.ai/docs/examples/example-tta-d4-aerial-segmentation/)
- Martian landslide segmentation (2026): H+V flip TTA +0.0005 mIoU (0.8278 -> 0.8283); 8-view D4 -0.0047 (0.8231) — [arXiv 2603.14132](https://arxiv.org/pdf/2603.14132)
- LiDAR (SemanticKITTI): flip TTA +1.6% mIoU for MinkUNet. Full TTA costs about 108x inference time — [arXiv 2405.14870](https://arxiv.org/pdf/2405.14870)
- DFC2025 SAR: full ensemble (simple average of model predictions) 36.64 vs best single 36.03 val mIoU (+0.61). Test: ensemble 41.08 vs best single 40.58 (table). The text says 40.08, which conflicts with the table — [arXiv 2504.13458](https://arxiv.org/html/2504.13458v1)
- HMMs for land-cover time series: Lamarche et al. (IEEE TGRS 2022) treat true land cover as hidden states behind noisy classifications. They filter errors and recover realistic change; raw per-date classifications overestimate change. Tested on annual 300 m series; also handles cloud gaps — [UCLouvain record](https://research.dial.uclouvain.be/handle/2078.5/240783)
- Gong et al. (ISPRS IJGI 2017): HMM on SVM outputs with spatial weights eliminated "the vast majority of illogical land cover transition" — [MDPI IJGI 6(10):292](https://www.mdpi.com/2220-9964/6/10/292)
- Zhu et al. (Sustainability 2024): matrix decomposition + optimized HMM vs spatio-temporal window filtering baseline — [MDPI Sustainability 16(18):8127](https://www.mdpi.com/2071-1050/16/18/8127)

### Inferences
- For 256 px training tiles, predict with 50% overlap and a Gaussian or cosine (Hann) weight map, then sum weighted softmax and normalize. This removes seams; cost about 4x inference, which is minutes on T4.
- Flip TTA (4 views: id, hflip, vflip, both) on softmax: cost 4x inference, still small. For SAR, rot90 is valid only if the model was trained with rot90 (it was). Expected gain: 0 to +1 mIoU. Always measure on val.
- Ensembling across months is cheap and principled: average softmax over the 12 monthly predictions for the annual label. Labels are annual, so this matches the target. Likely a larger gain than flip TTA. Untested.
- For a monthly map product: simplest smoothing is per-pixel majority/mode over a sliding 3-5 month window, or averaging logits over months. HMM (Viterbi) with a sticky transition matrix (e.g. 0.95 self-transition) is the principled version. Implementable in numpy in under 50 lines.
- Seed ensemble: 2-3 models at 1-2 h each exceeds one run budget. Use EMA + TTA first.

### Gaps
- No flip-TTA mIoU ablation on Sentinel-1 or optical LULC found.
- No Gaussian-vs-uniform blending ablation with numbers found.
- No monthly (12-step) HMM smoothing study with accuracy numbers found. Found studies use annual series.

## Q5. Semi-supervised / self-training on ignored pixels or unlabelled tiles

### Takeaway
Self-training with refined pseudo-labels gave +0.80 val mIoU in the DFC2025 SAR winner. Mean teacher beats supervised-only at 1% labels on LoveDA (about 38.7 vs 36.0). Full FixMatch-style training likely doubles per-step cost, which is tight on a T4 budget.

### Cited Findings
- DFC2025 SAR winner: DACS-style self-training with EMA teacher (alpha 0.999), confidence threshold 0.968, plus SCE for noise. Using their pseudo-labels instead of the official ones: 32.35 -> 33.15 val mIoU (+0.80). Combining official + own labels in the ensemble: 35.54 -> 35.70. Hardware: 4x RTX 4090 — [arXiv 2504.13458](https://arxiv.org/html/2504.13458v1)
- Patel, Sharma, Pasquarella, Gulshan (arXiv 2111.10079, Nov 2021 / rev. Jun 2022): SimCLR and FixMatch on riverbed, land cover and flood mapping (incl. Sentinel-1). They "significantly improve generalization performance when labeled data is limited", including on geographically shifted test data. Exact numbers not retrieved — [arXiv 2111.10079](https://arxiv.org/abs/2111.10079)
- LoveDA 1% labels: supervised-only about 36.0, mean teacher about 38.7 (+2.7) — [arXiv 2501.10736](https://arxiv.org/pdf/2501.10736)
- Desai et al. (WACV 2022): active learning + semi-supervised gave a 27% mIoU improvement with 2% labelled data. Optical data (UC Merced, DeepGlobe), not SAR — [WACV 2022 paper](https://openaccess.thecvf.com/content/WACV2022/papers/Desai_Active_Learning_for_Improved_Semi-Supervised_Semantic_Segmentation_in_Satellite_Images_WACV_2022_paper.pdf)
- Sentinel-1/-2 self-supervised fusion with k-means pseudo-labels (arXiv 2103.05543) — [arXiv 2103.05543](https://arxiv.org/pdf/2103.05543)

### Inferences
- Cheapest variant for T4: offline one-round self-training. (1) Train current model with EMA. (2) Predict on the 18% ignored pixels with TTA. (3) Keep pixels with confidence > about 0.9 and where the prediction equals one of the two source labels (WorldCover or DW). (4) Retrain or fine-tune. Cost: one extra run. Expect gains mainly where disagreement is concentrated (wetland, shrub/grass).
- Online mean teacher / FixMatch doubles forward passes per step. At 1-2 h budget, reduce epochs or crops. Gains in the literature are +1 to +3 mIoU at very low label fractions; here labels are not scarce, they are biased, so gains likely smaller.

### Gaps
- No found study self-trains on consensus-disagreement pixels of WorldCover/DW.
- FixMatch Sentinel-1 numbers from 2111.10079 not retrieved.

## Q6. Validation protocol (more val months, checkpoint averaging, select by loss vs mIoU)

### Takeaway
Little direct literature. EMA-evaluated checkpoints are reported to stabilize evaluation and add about +1 mIoU. The val swing (0.656 -> 0.583) is likely mostly sampling noise from a small val set, not real model change.

### Cited Findings
- GOOSE 2025 report used the EMA snapshot for every val checkpoint and final eval: +1.08 mIoU — [arXiv 2505.11769](https://arxiv.org/pdf/2505.11769)
- SWA (checkpoint averaging) +0.77 mIoU over single weights on BDD100K — [arXiv 2207.11211](https://arxiv.org/pdf/2207.11211)
- Label noise in segmentation exists on a spectrum; a 2026 data-centric benchmark ranks training samples by noise level — [arXiv 2603.00604](https://arxiv.org/html/2603.00604)

### Inferences
- Make val deterministic and larger: fixed full-tile sliding-window val (not random crops), all available months per val site. Pixel-level mIoU over many more pixels lowers variance.
- Select by a smoothed metric: EMA-model mIoU, or a 3-epoch moving average of val mIoU. Val loss on noisy labels tracks calibration, not mIoU, and with class weights it is a poor selector. Prefer mIoU on the EMA model.
- Report test results with mean ± std over 2-3 seeds if budget allows; otherwise state that single-run differences below about 1-2 mIoU are within noise.
- Split by site (134 sites), not patch, to avoid spatial leakage. Val swings partly reflect few val sites; per-site mIoU spread is useful to report.

### Gaps
- No RS paper quantifying val-mIoU variance vs val-set size found.
- No study comparing selection by val loss vs val mIoU under noisy labels found.

## Ranked summary (expected mIoU gain per effort, T4 1-2 h)

Rough expected gains come from the cited analogues. They are not guarantees for this dataset.

1. EMA weights, validate/save EMA model — about +0.5 to +1 mIoU, more stable val curve; ~0 cost ([2505.11769](https://arxiv.org/pdf/2505.11769), [2207.11211](https://arxiv.org/pdf/2207.11211)).
2. Fixed warmup + cosine schedule instead of ReduceLROnPlateau, and train longer within budget — longer schedule gave +1.57 in DFC2025 ([2504.13458](https://arxiv.org/html/2504.13458v1)); ~0 code cost.
3. Overlapping sliding window (50-75% overlap) + Gaussian blending at inference — about +1% and no seams ([1711.08681](https://arxiv.org/pdf/1711.08681)); minutes of extra inference.
4. Average softmax over months (multi-month ensembling) for annual labels — untested; free if months exist.
5. Add Lovász-softmax to the loss — +0.3 (DFC2025) to +1-2.5 (VOC) ([2504.13458](https://arxiv.org/html/2504.13458v1), [1705.08790](https://arxiv.org/pdf/1705.08790)); ~0 cost.
6. Class-aware copy-paste of weak-class regions — +6.2 on DynamicEarthNet U-Net ([2404.05693](https://arxiv.org/abs/2404.05693)); ~1 h coding; result may not transfer.
7. Flip TTA (4 views) — 0 to +1.6 ([2603.14132](https://arxiv.org/pdf/2603.14132), [2405.14870](https://arxiv.org/pdf/2405.14870)); 4x inference.
8. One-round offline self-training on ignored pixels — +0.8 in DFC2025 ([2504.13458](https://arxiv.org/html/2504.13458v1)); one extra run.
9. Label smoothing / SCE / GCE — +0.19 (SCE, DFC2025); ~0 cost; low evidence.
10. Seed/variant ensemble — +0.6 (DFC2025); costs N full runs.
11. HMM / majority temporal smoothing — for monthly map consistency, not test mIoU on annual labels ([Lamarche 2022](https://research.dial.uclouvain.be/handle/2078.5/240783)).
