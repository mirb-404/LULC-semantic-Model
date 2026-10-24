# Architectures and Pretrained Encoders for Sentinel-1-only Land Cover Segmentation

Context: current model is a U-Net with an ImageNet ResNet-34 encoder. Input is a 3-month window (VV, VH, VV-VH per month = 9 ch) plus 6 PCA channels. There are 8 classes and ~2000 patches of 256x256. Test OA is 0.83 and mIoU 0.54. Weak classes: wetland 0.22, shrub/grass 0.34, cropland 0.49. Budget: one Kaggle T4, 1-2 h per run. Output: 12 monthly maps.

Research date: 2026-10-24. 16 tool calls were used, so some items stay unverified (see Gaps).

## Q1. Temporal models for SAR time series (U-TAE, L-TAE, ConvLSTM/GRU, 3D CNN, TempCNN)

### Takeaway
Use the full time series with a temporal encoder (U-TAE / L-TAE), not one date or a few stacked dates. This gives the largest documented gains. On PASTIS, U-TAE reaches 63.1 mIoU. A single-date U-Net reaches 28.3, and a 3D-UNet 58.4. With S1-only input, U-TAE reaches 53.8-54.9 mIoU on PASTIS-R. On the S1-only BraDD-S1TS dataset, U-TAE beats 3D-UNet and ConvLSTM by 2.6-7 mIoU. U-TAE has only ~1.1M parameters, so it fits a T4 easily.

### Cited Findings
- U-TAE (Garnot & Landrieu, ICCV 2021, arXiv 2107.07933). Semantic segmentation on PASTIS (S2 time series, 5-fold CV). Scores are OA / mIoU, with parameter counts:
  - U-TAE: 83.2 / 63.1, 1.087M params
  - 3D-Unet: 81.3 / 58.4, 1.554M
  - U-ConvLSTM: 82.1 / 57.8
  - FPN-ConvLSTM: 81.6 / 57.1
  - U-BiConvLSTM: 81.8 / 55.9
  - ConvGRU: 79.8 / 54.2
  - ConvLSTM: 77.9 / 49.1

  [Source](https://ar5iv.labs.arxiv.org/html/2107.07933)
- U-TAE ablations on PASTIS (OA / mIoU):
  - Replace learned attention with mean attention: 82.8 / 60.1 (-3.0 mIoU)
  - Skip connections with temporal mean: 58.3
  - BatchNorm instead of GroupNorm: 71.9 / 36.0
  - Single-date U-Net (August): 65.6 / 28.3
  - Single-date U-Net (May): 58.1 / 20.6

  Normalization choice matters a lot: GroupNorm is used, and BatchNorm collapses performance. [Source](https://ar5iv.labs.arxiv.org/html/2107.07933)
- Inference time for one fold (~490 sequences): U-TAE 25.7 s, 3D-UNet 29.5 s, FPN-ConvLSTM 103.6 s. U-TAE is the cheapest temporal model of those compared. [Source](https://ar5iv.labs.arxiv.org/html/2107.07933)
- PASTIS-R (Garnot, Landrieu & Chehata, arXiv 2112.07558, ISPRS JPRS 2022). Semantic segmentation mIoU with U-TAE:
  - S1 descending only: 54.9
  - S1 ascending only: 53.8
  - S2 only: 63.1
  - Early fusion: 64.9
  - Late fusion with auxiliary loss and temporal dropout: 66.3

  Radar input: VV, VH, VV/VH in dB at 10 m. No speckle filter and no terrain correction. About 70 acquisitions per orbit, with orbits kept as separate series. [Source](https://arxiv.org/html/2112.07558v1)
- PASTIS-R parcel classification (OA / mIoU): S2 91.7 / 73.9, S1D 87.0 / 64.5, S1A 86.4 / 63.3. [Source](https://arxiv.org/html/2112.07558v1)
- AnySat paper (arXiv 2412.14123, Dec 2024). On BraDD-S1TS, an S1-only time-series dataset at 10 m with VV/VH/ratio:
  - U-TAE: 70.7 mIoU
  - 3D-UNet: 68.1
  - ConvLSTM: 63.7
  - AnySat fine-tuned: 80.9
  - AnySat linear probe: 78.9

  [Source](https://arxiv.org/html/2412.14123)
- PANGAEA (arXiv 2412.04204) aggregates multi-temporal inputs with an L-TAE instead of linear mapping. The authors say L-TAE "consistently enhances performance compared to the linear approach" on all datasets except DynamicEarthNet. Examples on PASTIS-R: S12-MoCo 26.09 to 34.49, Scale-MAE 11.74 to 24.55. [Source](https://arxiv.org/html/2412.04204v2)

### Inferences
- The current 3-month stack plus 6 PCA channels is a hand-made temporal summary. U-TAE learns a pixel-wise temporal weighting instead. That weighting is what lifted mIoU +4.7 over 3D-UNet and +34.8 over a single date on PASTIS. Wetland (seasonal flooding), cropland (phenology) and shrub/grass (seasonal greening) depend on seasonal patterns. They are the most likely classes to gain.
- To get 12 monthly maps from a full-year model, there are three options:
  1. Sliding window. Give U-TAE a centered window of T months (for example 5-7) and predict the label of the center month. Run it 12 times, padding or wrapping at the year edges.
  2. Per-timestep decoder. Remove the temporal collapse and keep a per-date feature map, for example by sharing the L-TAE attention masks but outputting T maps. A ConvGRU/ConvLSTM or bidirectional variant can also emit one output per step.
  3. Full-year context plus date query. Encode the whole year with U-TAE. Concatenate a learned month embedding or that month's own SAR features at the decoder, then predict that month.

  Option 1 is the cheapest to build on the existing training loop. Option 3 keeps one forward pass per year.
- Whether "monthly" labels really differ by month is an important design question. If labels are annual (ESA WorldCover / Dynamic World-style), a full-year encoder with a per-month head mainly adds smoothing. If labels are monthly (for example snow), the per-step output matters.
- Cost estimate: U-TAE has ~1.1M params, and 12 dates x 2-3 ch x 256x256 x batch 4-8 fits easily in 16 GB. 2000 patches x ~50-100 epochs is plausible within 1-2 h on a T4. This is an estimate, not a reported number.

### Gaps
- I found no published head-to-head of U-TAE against a "stacked-channel U-Net with ImageNet ResNet" on S1. The PASTIS single-date baseline is a from-scratch U-Net on one date, not a stacked-channel model.
- TempCNN (Pelletier et al.) is pixel-wise and has no spatial context. I found no S1 segmentation numbers for it in this pass.
- The official repo is github.com/VSainteuf/utae-paps (named in the PASTIS benchmark context). I did not check its license or how much it differs from the PASTIS benchmark code.

## Q2. SAR-capable pretrained encoders and foundation models

### Takeaway
On S1-only land-cover segmentation (DFC2020-S1), SAR-pretrained ViT-B models (SoftCon 52.8, CROMA 52.7, Copernicus-FM 52.4 mIoU) beat a supervised ViT-B (50.8). The gain is about 2 mIoU, which is modest. PANGAEA finds a plain UNet trained from scratch is highly competitive. It wins on Sen1Floods11 and on average beats most foundation models. The TerraMind ablation shows S1-only performance far below S2: 20.04 vs 40.20 mIoU on PASTIS. For this project, a ResNet-50 with SSL4EO-S12 S1 MoCo weights in the existing U-Net is the cheapest swap. Big ViT foundation models are unlikely to repay their cost on a T4 in 2 h.

### Cited Findings
- **Copernicus-Bench S1 results** (Copernicus-FM paper, arXiv 2503.11849, Mar 2025, v3 31 Jul 2025). Each cell is mean ± std over 3 runs.

  | Model | DFC2020-S1 LULC seg. (mIoU) | Flood-S1 (mIoU) | EuroSAT-S1 (OA) |
  |---|---|---|---|
  | Supervised ViT-S/16 | 49.9 | 78.0 | 81.7 |
  | Supervised ViT-B/16 | 50.8 | 78.3 | 81.5 |
  | Random init | 45.4 | 75.1 | 75.4 |
  | SoftCon ViT-B/14 | 52.8 | 77.2 | 83.6 |
  | CROMA ViT-B/8 | 52.7 | 77.4 | 83.9 |
  | DOFA ViT-B/16 | 49.7 | 76.0 | 81.7 |
  | Copernicus-FM ViT-B/16 | 52.4 | 77.7 | 87.2 |

  BigEarthNet-S1 mAP: SoftCon 78.7, Copernicus-FM 77.9, supervised ViT-S 71.5. Code is at github.com/zhu-xlab/Copernicus-FM. Pretraining took ~512 A100 GPU-hours. [Source](https://arxiv.org/html/2503.11849)
- **PANGAEA** (arXiv 2412.04204; IEEE GRS Magazine 2025). Parameter counts: CROMA 350.0M, DOFA 150.9M, Prithvi 125.7M, SatlasNet 121.2M, S12-MoCo/DINO/MAE/Data2Vec 53.5M each, UNet baseline 14.79M, ViT-B baseline 125.1M. Pretraining data: CROMA and DOFA used S1+S2, and the S12 models used SSL4EO-S12 S1+S2. Prithvi (HLS) and SatlasNet (S2, NAIP) have no SAR. [Source](https://arxiv.org/html/2412.04204v2)
- **PANGAEA, 100% labels, mIoU:**

  | Model | PASTIS-R | Sen1Floods11 | Crop Type Map-SS | DynamicEarthNet |
  |---|---|---|---|---|
  | UNet (from scratch) | 31.60 | 91.42 | 47.57 | 39.46 |
  | ViT (from scratch) | 38.53 | 87.66 | 44.08 | 36.83 |
  | CROMA | 32.32 | 90.89 | 49.38 | 38.29 |
  | DOFA | 30.02 | 89.37 | 51.33 | 39.29 |
  | S12-MoCo | 34.49 | 89.26 | 48.58 | 35.44 |
  | S12-DINO | 36.18 | 88.61 | 48.66 | 34.81 |
  | S12-Data2Vec | 34.32 | 88.15 | 54.03 | 35.90 |
  | Prithvi | 33.93 | 90.37 | 43.07 | 27.86 |

  The UNet reaches top-2 more often than any other model (6 times). The authors state that "utilizing optical data alone generally yields superior results compared to incorporating SAR information." [Source](https://arxiv.org/html/2412.04204v2)
- Source inconsistency: the PANGAEA paper's Table 5 gives CROMA PASTIS-R 32.32, but Table 10 (L-TAE) gives 38.51. [Source](https://arxiv.org/html/2412.04204v2)
- **TerraMind** (IBM/ESA, arXiv 2504.11171, Apr 2025). PANGAEA average mIoU over 9 datasets:
  - TerraMind v1-B: 58.35
  - TerraMind v1-L: 59.57
  - UNet baseline: 57.58
  - CROMA: 55.29
  - DOFA: 53.59
  - S12-MoCo: 53.02

  [Source](https://arxiv.org/html/2504.11171)
- **TerraMind v1-B modality ablation (mIoU):**

  | Input | PASTIS | Sen1Floods11 | CTM-SS |
  |---|---|---|---|
  | S1 only | 20.04 | 80.39 | 24.45 |
  | S2 only | 40.20 | 89.57 | 50.90 |
  | S1+S2 | 40.51 | 90.62 | 55.80 |

  [Source](https://arxiv.org/html/2504.11171)
- TerraMind's model card gives a different PANGAEA average: 66.6% mIoU over six tasks, about 3 points above CROMA. This conflicts with the 58.35 over nine datasets in the paper, so the subsets differ. [Source](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base); [Source](https://arxiv.org/html/2504.11171)
- An independent blog critique says TerraMind-L beats the supervised U-Net by only ~2 mIoU on average and loses on 5 of 9 datasets. This is opinion, not peer-reviewed. [Source](https://christopherren.substack.com/p/geospatial-foundational-disappointments)
- TerraMind loading and license:
  - License: Apache-2.0.
  - Input modalities: S1GRD and S1RTC (2-band tensor B,2,224,224), plus S2, DEM and RGB.
  - Loading through TerraTorch: `BACKBONE_REGISTRY.build('terramind_v1_base', pretrained=True, modalities=['S1GRD'])`.
  - The ViT-B output is 196 tokens x 768 for a 224 input.

  [Source](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)
- **AnySat** (arXiv 2412.14123):
  - 125M params.
  - Accepts S1 time series (VV, VH, ratio, both orbits) at 10 m.
  - Has an internal L-TAE for time and produces dense per-pixel outputs.
  - Scores 80.9 mIoU fine-tuned on BraDD-S1TS vs U-TAE 70.7.
  - Code: github.com/gastruc/AnySat.

  [Source](https://arxiv.org/html/2412.14123)
- **SSL4EO-S12** (arXiv 2211.07044). TorchGeo ships a ResNet-50 weight `SENTINEL1_ALL_MOCO`: 2 input channels, MoCo, file `resnet50_sentinel1_all_moco.pth`. [Source](https://torchgeo.readthedocs.io/en/v0.5.1/_modules/torchgeo/models/resnet.html)
- A CC-BY-4.0 Hugging Face mirror (BiliSakura/SSL4EO-S12-transformers) has 16 converted checkpoints (ViT, ResNet18/50), including S1 2-band variants. [Source](https://huggingface.co/BiliSakura/SSL4EO-S12-transformers)
- SSL4EO-S12 v1.1 (arXiv 2503.00168) is an updated multiseasonal pretraining dataset. [Source](https://arxiv.org/pdf/2503.00168)
- Copernicus-Bench includes DFC2020-SAR as the S1 dense LULC task, which is the closest public analogue to this project. [Source](https://huggingface.co/datasets/wangyi111/Copernicus-Bench/blob/bdfac676e21db01fc4c62f5961d3ca562a8232f8/readme.md)

### Inferences
- The expected gain from a SAR-pretrained encoder over an ImageNet CNN is small to moderate, roughly +1 to +3 mIoU. The DFC2020-S1 gap between SAR-pretrained and supervised ViT-B is +1.6 to +2.0. Against random init it is +7. The current baseline is ImageNet-pretrained, not random, so the realistic gain sits at the low end.
- SSL4EO-S12 S1 ResNet-50 (MoCo) is the most practical option:
  - CNN, so it keeps U-Net skip connections.
  - Native 2-channel VV/VH input.
  - About 25M encoder params.
  - Loads in one line through TorchGeo.

  Caveat: it expects 2 channels (VV, VH). The 9+6-channel input needs either per-month shared encoding (temporal batch), which pairs naturally with U-TAE-style aggregation, or an adapted first conv. Adapting the first conv loses part of the pretraining benefit.
- ViT foundation models (CROMA 350M, DOFA 151M, TerraMind-B, Copernicus-FM ViT-B, AnySat 125M) are heavier and work on 224/16 patches, which gives coarse 14x14 tokens. Fine-tuning plus a UPerNet decoder on 2000 patches x 12 dates within 2 h on a T4 is tight. Frozen-encoder or linear-probe use is feasible. TerraMind's S1-only ablation (PASTIS 20.04) is a warning that S1 alone gains little from optical-dominant pretraining.
- Prithvi and SatlasNet (S2/NAIP) have no S1 pretraining in PANGAEA, so skip them for S1-only work.

### Gaps
- Not verified this session: the current TorchGeo docs weight list (the docs URL returned 404/redirect), including ResNet-18 S1 weights, SoftCon/DeCUR S1 ResNet-50 entries, Satlas Sentinel-1 Swin weights, and Copernicus-FM/DOFA/CROMA weights in TorchGeo. SatlasPretrain Sentinel-1 weights could not be confirmed.
- Galileo (arXiv 2502.09356) and Copernicus-FM PANGAEA S1-only numbers were not found.
- No S1-only results for CROMA/DOFA/S12 on PASTIS-R. PANGAEA uses S1+S2 there, and its per-modality appendix was not retrieved.
- No T4 memory or throughput numbers were found for any foundation model.

## Q3. Encoder/decoder type comparisons for SAR segmentation (ResNet-50, ConvNeXt, EfficientNet, SegFormer, Swin, UPerNet, DeepLabv3+)

### Takeaway
Evidence for S1 specifically is thin and old. In the one S1 multi-architecture study found, the top models (SegNet, FC-DenseNet, U-Net, FRRN-B) differ by under 0.6 OA points. DeepLabv3+ and PSPNet trail by ~1-1.4 OA. Decoder choice is a second-order effect next to temporal modelling and pretraining. PANGAEA shows a conv UNet trained from scratch matches or beats ViT-based GFMs on several tasks.

### Cited Findings
- Wide-area land cover mapping with S1, Finland (arXiv 1912.05067). Setup: 20 m, summer 2018 mosaics, 5 CORINE classes, 4200 train / 2800 test 512x512 imagelets, VH/VV/ratio input, ImageNet-pretrained encoders. OA / kappa:

  | Model | OA (%) | Kappa |
  |---|---|---|
  | SegNet | 92.78 | 0.851 |
  | FC-DenseNet | 92.78 | 0.851 |
  | FRRN-B | 92.69 | |
  | U-Net (MobileNet-style) | 92.25 | 0.839 |
  | DeepLabV3+ | 91.37 | |
  | PSPNet | 91.37 | |
  | BiSeNet | 88.48 | |

  Peatland (wetland analogue) producer's accuracy ranged 42-65%. It was best with FRRN-B and FC-DenseNet (65%) and worst with DeepLabV3+ (42%). Urban was hardest everywhere. Adding a DEM gave only marginal change. [Source](https://ar5iv.labs.arxiv.org/html/1912.05067)
- PANGAEA: the authors say "learning the features through convolution is the best approach for one-class datasets." The UNet baseline (14.79M) gave the best Sen1Floods11 (91.42) and DynamicEarthNet (39.46) scores. [Source](https://arxiv.org/html/2412.04204v2)
- Copernicus-Bench DFC2020-S1: supervised ViT-B/16 (50.8) beats ViT-S/16 (49.9) by only 0.9 mIoU. Scaling the encoder gives little on S1 LULC. [Source](https://arxiv.org/html/2503.11849)

### Inferences
- Swapping ResNet-34 for ResNet-50, ConvNeXt-T or EfficientNet-B4 inside the U-Net will likely give ≤1-2 mIoU. Fine-detail decoders (U-Net, FC-DenseNet style) appear better than DeepLabv3+/PSPNet for thin or fragmented classes like wetland.
- Transformer decoders (SegFormer, UPerNet+Swin) have no S1-specific evidence here. With 2000 patches their data hunger is a risk.

### Gaps
- No S1-specific benchmark found that compares ConvNeXt, EfficientNet, SegFormer or Swin-UPerNet. The numbers above are from older (2019) or ViT-only studies.

## Q4. Ranked recommendation (expected gain per hour of effort)

### Takeaway
Rank by evidence:
1. Temporal encoder over the full series (U-TAE / L-TAE). Largest reported gains, and cheap on a T4.
2. SAR-pretrained CNN encoder (SSL4EO-S12 S1 ResNet-50) shared across dates. About +1-3 mIoU.
3. Decoder or backbone swaps. Small effect.
4. Large ViT foundation models. Unclear gain and high cost.

### Cited Findings
- Temporal attention vs alternatives: U-TAE 63.1 vs 3D-UNet 58.4 vs ConvLSTM 49.1 vs single date 28.3 mIoU (PASTIS). [Source](https://ar5iv.labs.arxiv.org/html/2107.07933)
- S1-only U-TAE on PASTIS-R: 53.8-54.9 mIoU. On BraDD-S1TS: 70.7 vs 3D-UNet 68.1 and ConvLSTM 63.7. [Source](https://arxiv.org/html/2112.07558v1); [Source](https://arxiv.org/html/2412.14123)
- SAR pretraining on S1 LULC: SoftCon/CROMA/Copernicus-FM 52.4-52.8 vs supervised ViT-B 50.8 on DFC2020-S1. [Source](https://arxiv.org/html/2503.11849)
- L-TAE temporal aggregation beats linear stacking for most GFMs in PANGAEA (for example S12-MoCo PASTIS-R 26.09 to 34.49). [Source](https://arxiv.org/html/2412.04204v2)
- PASTIS-R late fusion with temporal dropout and an auxiliary loss improves robustness. Temporal dropout alone changes S1 scores by -0.2 to -0.5, which is negligible. [Source](https://arxiv.org/html/2112.07558v1)

### Inferences (ranked; effort and gain are estimates, not reported numbers)
1. **U-TAE over 12 monthly composites (2-3 ch each), sliding-window or date-query head for monthly maps.** Effort is ~3-5 h of coding. Public code: github.com/VSainteuf/utae-paps. Expected gain is the largest, because cropland, wetland and shrub/grass are phenology-driven. The PCA channels become redundant. Use GroupNorm, not BatchNorm.
2. **Hybrid: shared SSL4EO-S12 S1 ResNet-50 (MoCo, 2-ch) encoder per date, L-TAE attention on the deepest features, attention masks reused at skips (U-TAE style), U-Net decoder.** Combines items 1 and 3. Memory is heavier: 12 x ResNet-50 passes. It likely needs batch 2-4, AMP, and possibly 6 bi-monthly dates or a frozen early stage.
3. **Drop-in: keep the current stacked-channel U-Net, switch the encoder to SSL4EO-S12 S1 ResNet-50.** Repeat or average the 2-ch conv1 weights over the 15 channels, or feed each month separately and concatenate features. Effort ~1 h. Expected gain is small (+0-2 mIoU), because the pretraining matches 2-ch single-date input.
4. **Encoder or decoder swap (ResNet-50/ConvNeXt-T/EfficientNet, UPerNet/DeepLabv3+).** Effort ~0.5-1 h with segmentation_models_pytorch. Expected gain is ≤1-2 mIoU. DeepLabv3+ may hurt thin classes such as wetland.
5. **Foundation models (TerraMind-B S1GRD via TerraTorch, Copernicus-FM, CROMA, AnySat).** Effort is high (new frameworks, 224 px patches, ViT memory). Evidence of S1-only benefit is weak (TerraMind S1-only PASTIS 20.04). AnySat is the most interesting because it handles S1 time series natively and is strong on BraDD-S1TS. At 125M params, full fine-tuning on a T4 in 2 h is uncertain. Use it as a frozen feature extractor plus light head if tried.

### Gaps
- No public study reports a gain for "ImageNet ResNet-34 U-Net with stacked months + PCA" vs U-TAE on S1 land cover. The size of item 1's gain for this dataset is unknown until tested.
- No T4 runtime numbers were found. The feasibility statements are inferred from parameter counts.
