# Temporal architectures and SAR-pretrained encoders for S1 land-cover segmentation (fit for one A6000, ~2200 patches)

Research date: 2026-11-03. This extends `research_notes/Sentinel 1 LULC segmentation improvements/architectures_and_pretrained_encoders.md` and `s1_benchmarks_and_ceilings.md`. Numbers already given there are not repeated (U-TAE/3D-UNet/ConvLSTM on PASTIS, PASTIS-R S1 54.9/53.8, BraDD-S1TS, Copernicus-Bench DFC2020-S1 table, PANGAEA tables, TerraMind ablation, AnySat). Only new or corrected evidence is listed below, plus inferences for the current setup.

Current setup: U-Net, ImageNet ResNet-34 encoder. Input has 12 channels: VV, VH and VV-VH of the target month, plus 9 yearly statistics. One map per month, followed by 3-month softmax smoothing. 7 classes, labels from WorldCover 2021. 2218 patches of 256x256. Val mIoU ~0.73, rural test mIoU 0.679.

About 16 tool calls were used. Several full-text tables could not be fetched (see Gaps).

## Q1. Temporal models on S1 time series: what beats a statistics-input U-Net?

### Takeaway
New evidence does not change the ranking: learned temporal attention (U-TAE, TSViT) beats ConvLSTM/ConvGRU and 3D-CNNs on time-series segmentation. But every TSViT/3D-UNet comparison found is S2-only. No paper compares a temporal model against a "target-month + yearly statistics" U-Net like ours on S1. One S1 study (Russo et al. 2025, already in the earlier notes) found seasonal aggregates beat dense series. Our statistics approach is therefore a reasonable, literature-backed baseline, not an obviously weak one. Expected gain from U-TAE over it is uncertain: likely 0 to +3 mIoU. It is not the +35 single-date gap reported on PASTIS.

### Cited Findings
- **TSViT (Tarasiou et al., CVPR 2023, arXiv 2301.04944). S2 ONLY.** OA / mIoU per dataset:

  | Model | PASTIS fold-1 | Germany | T31TFM-1618 |
  |---|---|---|---|
  | TSViT | 83.4 / 65.1 | 95.0 / 84.8 | 90.3 / 63.1 |
  | U-TAE | 82.9 / 62.4 | 93.1 / 77.1 | 88.9 / 58.5 |
  | UNET2D-CLSTM | 82.7 / 60.7 | 92.9 / 76.2 | — |
  | UNET3D | 82.3 / 60.4 | 92.4 / 75.2 | — |
  | FPN-CLSTM | 81.9 / 59.5 | — | — |
  | BiCGRU | 80.5 / 56.2 | — | — |

  The 5-fold PASTIS averages are TSViT 65.4 and U-TAE 63.1, so TSViT is +2.3 mIoU over U-TAE. [Source](https://ar5iv.labs.arxiv.org/html/2301.04944)
- TSViT cost: 1.7M params and 11.8 ms per sample, vs U-TAE 1.1M and 8.8 ms and UNET3D 6.2M and 11.2 ms. Measured at T=52, 24x24 px, 13 bands on a Titan Xp. TSViT works on **24x24 pixel crops** (PASTIS 128x128 tiles are cut into 24x24). The authors state that attention cost is quadratic in input size, which "can raise hardware requirements for larger inputs". 2x2 patches beat 3x3 (84.8 vs 83.6 mIoU on Germany) but raise the token count. [Source](https://ar5iv.labs.arxiv.org/html/2301.04944)
- **CNN vs transformer for SITS crop mapping (arXiv 2412.01944). S2 ONLY, 48x48 patches.** mIoU Munich / Lombardia, params, FLOPs:

  | Model | Munich mIoU | Lombardia mIoU | Params (M) | GFLOPs |
  |---|---|---|---|---|
  | 3D U-Net | 81.25 | 58.50 | 112 | 222 |
  | TSViT | 83.01 | 57.77 | 3.3 | 74 |
  | VistaFormer | 79.63 | 57.31 | 0.57 | 2.0 |
  | Swin UNETR | 75.03 | 54.43 | 4.5 | — |
  | 3D FPN | 73.02 | 55.17 | — | — |
  | 3D DeepLabv3 | 52.60 | 50.12 | — | — |

  A well-sized 3D U-Net matches or beats TSViT on one of the two datasets. [Source](https://arxiv.org/html/2412.01944)
- **TSViT on S1:** no published S1-only TSViT segmentation number found. One multi-modal crop paper (EOekoLand, arXiv 2406.16513) reports an S1-only single-modality TSViT at 65.45 mean accuracy / 85.07 OA, vs U-TAE early fusion at 67.70 / 89.75. The modality of the U-TAE column was not verifiable from the snippet. The full PDF could not be parsed. [Source](https://arxiv.org/pdf/2406.16513) (search snippet, unverified)
- **PTAViT3D (arXiv 2409.13568):** a 3D ViT for field-boundary delineation. It accepts S1 or S2 time series, with S1+S2 fused by cross-attention. The abstract claims SOTA on ePaddocks, Fields-of-the-World, PASTIS and AI4SmallFarms. A search summary says S1 is "comparable" to S2 at 10 m. The abstract gives no numbers. The task is boundaries, not land cover. [Source](https://arxiv.org/abs/2409.13568)
- **Statistics vs full series:** no study was found that benchmarks temporal-statistics input against a raw S1 time series in a deep segmentation model. The closest evidence:
  - Russo et al. 2025: seasonal S1 composites beat dense series, with the largest gain where S1 coverage is uneven (Siberia). [Source](https://arxiv.org/html/2503.07230v1)
  - S1 rapeseed mapping (Remote Sensing 2024): a 5-month window matched the full growth-cycle series (~95% F1), but cross-year transfer fell below 50% F1. Crop-specific. [Source](https://doaj.org/article/3d42496857fb41f780653569378e1ecf) (search snippet)
- **Early fusion (stacking all dates as channels):** in PASTIS-R, "early fusion" means S1+S2 concatenated per date inside U-TAE, not dates stacked as channels. No S1 result for a 36-channel date-stacked 2D U-Net was found. [Source](https://arxiv.org/html/2112.07558v1)
- **TempCNN / L-TAE pixel encoders:** no new S1 segmentation numbers found. PANGAEA's L-TAE aggregation result (already in earlier notes) remains the only S1-relevant evidence. CROMA SAR-only on CTM-SS rose from 35.4 to 50.4 mIoU with L-TAE vs linear temporal mapping. [Source](https://arxiv.org/html/2412.04204v2)
- **BreizhCrops, DynamicEarthNet, MultiSenGE:**
  - BreizhCrops is S2 pixel time series only. Not relevant to S1. (Known from the dataset definition. Not re-fetched this session.)
  - DynamicEarthNet in PANGAEA has a UNet at 39.46 mIoU. It was the one dataset where L-TAE did not beat linear aggregation. [Source](https://arxiv.org/html/2412.04204v2)
  - No S1-only DynamicEarthNet or MultiSenGE segmentation numbers were found.

### Inferences
- **Labels are annual (WorldCover 2021).** Every month has the same target. A temporal model can only help by building a better annual descriptor, not by tracking within-year change. Our 9 statistics already do that by hand. The PASTIS gains (single date 28.3 to U-TAE 63.1) measure "one date vs whole year", and our model already sees the whole year through its statistics. The relevant gap is "hand-made yearly statistics vs learned temporal attention". Literature implies that gap is a few points at most. Estimate: 0 to +3 mIoU, with Water and Snow & ice the most likely winners. Both depend on the timing of seasonal events (ice/snow months, flood months), which fixed means and order statistics may blur. VH std and the share of water months capture part of this already.
- **Cheapest temporal test, ~1 h:** an early-fusion 2D U-Net with 36 input channels (12 months x VV, VH, VV-VH), with or without the 9 statistics. It reuses the current pipeline exactly. Since labels are annual, it can emit one annual map. The monthly map then becomes a deployment choice. This directly answers "statistics vs raw stack" on our own data.
- **U-TAE at 256x256 x 12 dates fits easily on an A6000 (48 GB).** PASTIS runs 128x128 x 33-61 dates on far smaller GPUs, and our 256²x12 has a similar total voxel count. Model size is 1.1M params, so overfitting risk on 2218 patches is low. U-TAE is trained from scratch, which loses ImageNet initialisation. That may cost points on rare classes. Estimate: 10-20 min per run (inferred, not measured). Use GroupNorm (BatchNorm collapsed U-TAE in the original ablation, see earlier notes).
- **TSViT is a poor fit.** It was designed for 24x24 crops, and quadratic attention at 256x256 with 2x2 patches is impractical. Tiling into 24-48 px crops loses spatial context. Its S1 advantage is unproven. Skip it.
- **ConvLSTM/ConvGRU U-Nets** are consistently below U-TAE and slower (FPN-ConvLSTM ~4x U-TAE inference time; see earlier notes). Low priority.
- **3D-CNN U-Net** is competitive in arXiv 2412.01944 but heavy there (112M params). A small 3D U-Net (~1.5-6M) is the CNN-course-friendly alternative to U-TAE, with similar expected gain.
- **Monthly output design:** with annual labels, per-month predictions plus 3-month smoothing mostly suppress noise. A full-year model gives a temporally constant map, so flicker would be 0 by construction. If monthly maps are a product requirement, keep the target-month channels as a "query" (current design) on top of the full-year encoder.

### Gaps
- No S1-only TSViT, ConvLSTM, or 3D-UNet LULC segmentation numbers were found beyond the BraDD-S1TS numbers in the earlier notes.
- No paper compares a statistics-input U-Net against a temporal encoder on S1. The size of the gain for this dataset can only be measured by running it.
- Full tables of the EOekoLand paper (2406.16513) and the PTAViT3D paper could not be extracted.
- MultiSenGE S1 results were not found, as in the previous pass.

## Q2. SAR-pretrained / multi-sensor weights: which accept 2-channel S1, which are CNNs, what gains?

### Takeaway
TorchGeo (current stable docs) ships **five 2-channel S1 ResNet-50 weights**: MoCo (SSL4EO-S12), DeCUR, SoftCon, CLOSP and GeoCLOSP. It also ships 2-channel S1 ViT (MAE, FG-MAE, SoftCon DINOv2, CLOSP) and Swin-V2-B (Satlas) weights. **No S1 ResNet-18/34 weights are listed.** The best documented CNN is SoftCon RN50. On S1 linear probing it beats SSL4EO MoCo RN50 by +2.3 mAP on BigEarthNet-SAR and +4.7 accuracy on EuroSAT-SAR. No paper reports an S1 *segmentation* fine-tuning gain for any of these RN50 weights vs ImageNet. Prithvi has no S1 at all. Expected gain for our U-Net: 0 to +2 mIoU. It could be negative if our 12-channel input is forced into 2-channel weights.

### Cited Findings
- **TorchGeo S1 weights:** all have 2 input channels except Tessera (3 channels, architecture not stated).

  | Enum | Arch | Method | License |
  |---|---|---|---|
  | `ResNet50_Weights.SENTINEL1_GRD_MOCO` | RN50 | MoCo | CC-BY-4.0 |
  | `ResNet50_Weights.SENTINEL1_GRD_DECUR` | RN50 | DeCUR | Apache-2.0 |
  | `ResNet50_Weights.SENTINEL1_GRD_SOFTCON` | RN50 | SoftCon | CC-BY-4.0 |
  | `ResNet50_Weights.SENTINEL1_GRD_CLOSP` | RN50 | CLOSP | OpenRAIL |
  | `ResNet50_Weights.SENTINEL1_GRD_GEOCLOSP` | RN50 | GeoCLOSP | OpenRAIL |
  | ViT-S/B/L/H `SENTINEL1_GRD_MAE` / `_FGMAE` | ViT | MAE / FG-MAE | CC-BY-4.0 |
  | `ViTSmall14_DINOv2_Weights` / `ViTBase14_DINOv2_Weights` `.SENTINEL1_GRD_SOFTCON` | ViT | SoftCon | CC-BY-4.0 |
  | ViT-S/L `SENTINEL1_GRD_CLOSP` | ViT | CLOSP | OpenRAIL |
  | `Swin_V2_B_Weights.SENTINEL1_MI_SATLAS` / `_SI_SATLAS` | Swin V2-B | Satlas (multi-/single-image) | ODC-BY |
  | `Tessera_Weights.TESSERA_SENTINEL1_ENCODER` | not stated | Tessera | CC0-1.0 |

  The docs page does not list CROMA, DOFA, Copernicus-FM or Panopticon as S1-specific weights. They are sensor-agnostic or multi-modal. No ConvNeXt or ResNet-18 S1 weights are listed. [Source](https://docs.torchgeo.org/en/stable/api/models.html)
- **SoftCon (arXiv 2405.20462, TGRS), SAR linear probing.** BigEarthNet-SAR-10% mAP / EuroSAT-SAR accuracy:

  | Model | BigEarthNet-SAR-10% | EuroSAT-SAR |
  |---|---|---|
  | MoCo-v2 SSL4EO-S12 RN50 | 76.6 | 82.4 |
  | **SoftCon RN50** | **78.9** | **87.1** |
  | MAE ViT-S | 69.8 | 79.3 |
  | FG-MAE ViT-S | 71.7 | 80.7 |
  | CROMA ViT-B | 77.9 | 87.5 |
  | SoftCon ViT-S | 80.3 | 87.1 |
  | SoftCon ViT-B | 82.5 | 89.1 |

  - No ImageNet or random-init SAR baseline is given.
  - No S1 segmentation result is given; DFC2020/OSCD appear only in multispectral form.
  - The paper text quotes inconsistent deltas (e.g. "81.4%" vs the table's 82.5).
  - Pretraining: SSL4EO-S12 S1 paired with S2, with Dynamic World-derived soft labels via S2. SAR is scaled to 8-bit with SSL4EO mean/std, on 224x224 crops.
  - Compute: RN50 SAR pretraining took 4xA100 for 7 h.

  [Source](https://arxiv.org/html/2405.20462v2)
- **SoftCon/CROMA/Copernicus-FM ViT-B on DFC2020-S1 segmentation:** these are in the earlier notes (52.4-52.8 vs supervised ViT-B 50.8, random 45.4). That remains the only S1 *segmentation* comparison found that includes a supervised-pretrained baseline. [Source](https://arxiv.org/html/2503.11849)
- **DeCUR (arXiv 2309.05300):** SAR-optical is evaluated only on BigEarthNet-MM *classification*, where it beats cross-modal SimCLR by 2.4-3.4% in linear probing. Its segmentation experiments are RGB-DEM (GeoNRW) and RGB-depth only. No S1 segmentation numbers. [Source](https://ar5iv.arxiv.org/html/2309.05300)
- **Prithvi-EO-2.0** is pretrained on HLS (Harmonized Landsat-Sentinel-2) with 6 optical bands (B2, B3, B4, B8A, B11, B12). **No Sentinel-1 input.** Not usable for S1-only work. [Source](https://arxiv.org/html/2412.02732v1)
- **Galileo (arXiv 2502.09356, NASA Harvest):** a multimodal ViT whose pretraining includes S1 SAR, optical, elevation, weather and pseudo-labels, in space and time. The authors claim it beats specialists on 11 benchmarks. No S1-only segmentation number was found. Code: github.com/nasaharvest/galileo. [Source](https://arxiv.org/abs/2502.09356)
- **SMARTIES (arXiv 2506.19585, ICCV 2025):** a spectrum-aware MAE pretrained on ~500K paired S2 / S1 / Maxar RGB. It uses two dedicated S1 projection layers and amplitude only (no phase). It is evaluated on 10 datasets including SAR. Exact S1 numbers were not retrieved. ViT backbone. [Source](https://arxiv.org/html/2506.19585v1); [Supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Sumbul_SMARTIES_Spectrum-Aware_Multi-Sensor_ICCV_2025_supplemental.pdf)
- **SatMAE / SatMAE++:** in the sources reviewed they appear only as optical (fMoW-RGB/Sentinel-2) baselines, e.g. in Galileo's comparison table. No S1 weights were found. (Flag: optical only.) [Source](https://arxiv.org/abs/2502.09356)
- **CROMA, DOFA, Copernicus-FM, TerraMind, AnySat:** S1 handling and numbers are in the earlier notes. All are ViT-based with a 224 px / 8-16 px patch tokenisation. None is a CNN.

### Inferences
- **Best CNN candidate: SoftCon S1 RN50 (`ResNet50_Weights.SENTINEL1_GRD_SOFTCON`).** It has the strongest documented SAR features of the RN50 S1 weights. Second choice: SSL4EO MoCo RN50, the most widely used, which PANGAEA calls S12-MoCo. DeCUR and CLOSP RN50 have no published S1 dense-task evidence found.
- **Channel mismatch is the main risk.** The weights expect 2 channels (VV, VH) of SSL4EO-S12 GRD. That data is most likely sigma0 dB from GEE `COPERNICUS/S1_GRD`; this is inferred and not verified this session. Our input is terrain-flattened gamma0 monthly means plus 10 extra channels. Three options:
  1. **conv1 expansion:** copy VV/VH filters onto the monthly and statistic channels, zero-init or mean-init the rest. Simple, but most of what was pretrained lives in conv1-layer1 statistics that no longer match.
  2. **Per-month shared encoder plus temporal fusion (U-TAE-style).** This matches the pretraining exactly (2 channels per date), but costs 12 RN50 forward passes per sample.
  3. **Two-stream:** a pretrained RN50 on the target-month VV/VH, plus a light stream for the statistics, fused at the bottleneck.

  Renormalise inputs to the SSL4EO mean/std in every case.
- **Gain magnitude:** SAR-pretrained vs supervised ViT-B on DFC2020-S1 was +1.6 to +2.0 mIoU. SoftCon vs MoCo RN50 was +2.3 mAP (classification). Our baseline is ImageNet ResNet-34, already a strong init for U-Net. Realistic expectation for option 1: -1 to +1.5 mIoU. For option 2: +0 to +3 mIoU, mixed with the temporal-model effect. Rare classes (Water, Snow & ice) gain most from pretraining in principle, but SSL4EO-S12 samples around cities and is thin in polar and snow regions. Benefit for Snow & ice is doubtful (inference).
- **This fits the planned encoder sweep.** Add "ResNet-50 + SoftCon-S1 conv1-expanded" and "ResNet-50 + SSL4EO-MoCo-S1" next to "ResNet-50 ImageNet". The pretraining effect is then isolated at equal architecture.
- **ViT foundation models (CROMA 350M, Copernicus-FM, TerraMind, SMARTIES, Galileo, AnySat 125M)** do not fit a CNN course and have weak S1-only evidence. On an A6000 they are feasible as frozen encoders with a UPerNet head. Coarse tokens (14x14 or 16x16 for 224 px) hurt thin rivers and water edges. The PANGAEA UNet baseline matches or beats most of them. Low priority.

### Gaps
- No S1 *segmentation* fine-tuning result was found for any RN50 S1 weight (MoCo, DeCUR, SoftCon, CLOSP) against an ImageNet RN50 baseline.
- The S1 input definition of the SSL4EO-S12 GRD weights (dB sigma0 vs gamma0, value range) was not confirmed this session. Check the TorchGeo transforms or the SSL4EO-S12 repo before use.
- Galileo, SMARTIES and Copernicus-FM S1-only numbers on PANGAEA-style segmentation were not retrieved.
- CLOSP/GeoCLOSP and Tessera paper details (what Tessera's 3 channels are) were not looked up.

## Q3. Practical fit on one RTX A6000 (48 GB), 256x256 x 12 months, ~2200 patches

### Takeaway
All CNN options fit comfortably. The binding constraint is overfitting and data size, not VRAM. U-TAE (1.1M) and a small 3D U-Net are cheapest. A per-month shared RN50 is ~12x the compute of today's single-pass RN50 run but still feasible: estimated ~30-60 min per run, inferred. ViT foundation models are feasible only frozen or with low-rank fine-tuning.

### Cited Findings
- Params: U-TAE 1.1M; TSViT 1.7M; UNET3D 6.2M; UNET2D-CLSTM 2.3M. All measured at 24x24 px, T=52, Titan Xp (12 GB). [Source](https://ar5iv.labs.arxiv.org/html/2301.04944)
- PANGAEA params: S12-MoCo/DINO/MAE 53.5M; CROMA 350M; DOFA 150.9M; UNet baseline 14.8M (in the earlier notes). [Source](https://arxiv.org/html/2412.04204v2)
- SoftCon RN50 SAR pretraining: 4xA100, 7 h (pretraining only, not needed by us). [Source](https://arxiv.org/html/2405.20462v2)
- 3D U-Net at 48x48 S2: 112M params, 222 GFLOPs per sample, showing 3D U-Nets can be heavy depending on width. [Source](https://arxiv.org/html/2412.01944)

### Inferences
These are estimates, not measured.
- Current run: ~5 min. A 36-channel early-fusion U-Net costs the same (only conv1 changes). U-TAE at 256²x12 should take ~10-20 min. A per-month shared RN50 with L-TAE takes ~12x encoder FLOPs. With AMP and batch 4-8 it should run ~30-60 min, with VRAM well under 48 GB.
- Overfitting: 2218 patches from 139 sites means few independent sites. Large ViTs (100-350M) fully fine-tuned would very likely overfit site-specific texture. Prefer frozen or partial fine-tuning, and always split validation by site.
- Africa/Oceania weakness (0.49/0.44) looks like a domain or label problem. WorldCover itself has lower accuracy for Oceania shrub/grass (see earlier notes). Temporal models and pretraining are unlikely to fix it; more sites from those regions would be the lever.

### Gaps
- No published VRAM or time numbers for U-TAE/RN50-temporal at 256x256 x 12 dates. All runtime numbers above are estimates.

## Q4. Expected gain magnitudes (mIoU points vs our 0.679 rural test / 0.73 val)

### Takeaway
Ranked by expected gain per hour: (1) 36-channel early-fusion vs statistics ablation, (2) U-TAE / small 3D U-Net full-year encoder, (3) SoftCon or MoCo S1 RN50 in the encoder sweep, (4) ViT foundation models. Realistic total upside from temporal models plus SAR pretraining is about +1 to +4 mIoU. No source supports a double-digit gain over a full-year-aware baseline like ours.

### Cited Findings
- Learned temporal attention vs alternatives, S2 PASTIS: TSViT 65.4, U-TAE 63.1, UNET3D 60.4, ConvGRU-type 56.2. Spread is about 2-9 mIoU between temporal architectures. [Source](https://ar5iv.labs.arxiv.org/html/2301.04944)
- SAR pretraining on S1 LULC segmentation: +1.6 to +2.0 mIoU vs supervised ViT-B (DFC2020-S1); +7 vs random init. [Source](https://arxiv.org/html/2503.11849)
- SoftCon RN50 vs MoCo RN50, S1 linear probe: +2.3 mAP (BigEarthNet-SAR) / +4.7 acc (EuroSAT-SAR). [Source](https://arxiv.org/html/2405.20462v2)
- L-TAE vs linear temporal aggregation, SAR-only CROMA on CTM-SS: +15 mIoU; on PASTIS-R only +0.9. Strongly dataset-dependent. [Source](https://arxiv.org/html/2412.04204v2)

### Inferences
These estimates are conditional on our annual WorldCover labels and existing yearly statistics.

| Change | Expected mIoU gain | Conditions |
|---|---|---|
| 36-channel early fusion (raw 12 months), keep ResNet-34 ImageNet | -1 to +2 | Depends on whether order-independent statistics lose seasonal timing info |
| U-TAE / small 3D U-Net from scratch | 0 to +3 | Best chance on Water / Snow & ice; may lose on rare classes without ImageNet init |
| SoftCon/MoCo S1 RN50, conv1 expanded to 12 channels | -1 to +1.5 | Pretraining mismatch on channels and gamma0 vs sigma0 |
| Per-month shared S1 RN50 + L-TAE | +0 to +3 | Matches pretraining. ~12x compute. Largest engineering effort among the CNN options |
| ViT foundation models (CROMA, Copernicus-FM, TerraMind, Galileo, SMARTIES, AnySat) | uncertain, likely ≤ +2 | Frozen encoder only. Coarse tokens. Not CNN |

- None of these addresses the regional weakness in Africa (0.49) and Oceania (0.44). That needs data or label work.

### Gaps
- No direct evidence exists for our exact configuration. All gain ranges above are extrapolations from S2 or different-label benchmarks and must be confirmed by our own ablations.
