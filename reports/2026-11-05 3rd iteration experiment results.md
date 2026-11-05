# 3rd-iteration experiments: encoders, training choices, label handling (2026-11-05)

**Result: no configuration beats the final model `Iter2_Test_1_Full` by more than the run-to-run noise. The final model stays.** The three notebooks give the report its model comparison and three findings:

1. ImageNet pretraining matters more than which encoder is used.
2. The encoder choice changes rural test mIoU by at most about 0.03, which is inside the noise.
3. The reference configuration itself varies by 0.057 rural test mIoU across six identical runs, so `Iter2_Test_1_Full` (0.679) is a good draw. Its expected level is about 0.66.

Plan and settings: `reports/2026-11-02 Prototype runs and model experiments.md` sections 3-4. Notebooks:
- `Notebooks/3rd_iteration_encoders.ipynb`
- `Notebooks/3rd_iteration_training.ipynb`
- `Notebooks/3rd_iteration_labels.ipynb`

All three were executed on Datalab (RTX A6000, bf16, batch 64, all data on the GPU).

## 1. Where the results are

On Datalab, under `/home/jovyan/vault/LULC-semantic-Model/Weights/`:

| Notebook | Folder (plus `.zip` archive of the same name) | Exported `best_model.pt` (winner by mean val mIoU) | `beats_reference` |
|---|---|---|---|
| encoders | `Iter3_Encoders/` | `E1_vgg16_bn_s42` (VGG16-BN) | true (against its own reference only, see section 3) |
| training | `Iter3_Training_20261105_170703/` | `T1_sgd_s42` | false |
| labels | `Iter3_Labels_20261105_163345/` | `L1_agreement_only_s42` | false |

- **Contents of every folder:**
  - `best_model.pt` and `model_card.json` for the winner; the card contains `encoder`, so the prototype loads it.
  - `runs/<run>.pt` and `runs/<run>_history.csv` for every run.
  - `runs.csv` (one row per run) and `summary.csv` (mean over seeds).
  - `per_class_rural_IoU.csv`, `timing.json`, `comparison.png`, `val_curves.png`, `confusion_matrix_winner.png`.
- The training and labels folders carry a timestamp because the notebooks add one when a folder of the same name already exists. Folders `Iter3_Training/` and `Iter3_Labels/` from the first, parallel attempt may also exist. They may be incomplete; the timestamped folders are the complete runs.
- None of these weights are in the local repo yet (`Weights/` holds only the earlier runs).

## 2. Setup

- Every configuration changes one thing against the reference, which uses the settings of `Iter2_Test_1_Full`: U-Net with an ImageNet ResNet-34 encoder, `month_year` input (12 channels), AdamW, CE + Lovász 0.5, disagreement weight 0.5, label smoothing 0.1, 40 epochs, patience 10.
- 2 seeds (42, 7) per configuration; mean over the seeds.
- **Decision rule:** the winner is chosen by mean val mIoU. Rural test mIoU is only reported. A change counts only if it beats the reference by more than 0.02.
- Data: `Dataset_2nd_iteration`:
  - train: 2218 patches from 139 sites;
  - val: 335 patches from 21 sites;
  - test: 265 patches from 17 sites, 201 of them rural from 13 sites.
- **Time:**
  - encoders notebook: 165 min for 12 runs. It shared the GPU with the other two notebooks for part of the time, so 16 s per epoch for ResNet-34.
  - training notebook (6 runs) and labels notebook (10 runs), each rerun alone: 10 min and 19 min, about 2.5 s of training per epoch.

## 3. The reference: six identical runs

Each notebook trained its own reference, so the same configuration ran six times:

| Run | val mIoU | rural test mIoU |
|---|---|---|
| E0_resnet34 s42 / s7 | 0.728 / 0.725 | 0.672 / 0.644 |
| T0_reference s42 / s7 | 0.730 / 0.729 | 0.679 / 0.674 |
| L0_reference s42 / s7 | 0.731 / 0.736 | 0.622 / 0.656 |
| **Pooled mean (6 runs)** | **0.730** (range 0.725-0.736) | **0.658** (range 0.622-0.679) |
| `Iter2_Test_1_Full` (2026-10-31) | 0.732 | 0.679 |

- **Val is stable (±0.006); rural test is not (±0.03).** The test set has only 13 rural sites, so one or two sites decide the rare classes.
- The noise estimate of ±0.02 from the 2nd iteration was too small for rural test mIoU.
- Each notebook compared against its own reference. The pooled reference is the fairer baseline, and section 4 uses it. Example: L0 was the weakest reference pair (rural 0.639), which made every label variant look better than it is.
- `Iter2_Test_1_Full` scores 0.679, at the top of this range. In the report, quote it with the pooled mean: rural test mIoU 0.679 for the delivered model, about 0.66 ± 0.03 expected for the configuration.

## 4. Results against the pooled reference

Mean over 2 seeds. Δ = difference to the pooled reference (val 0.730, rural 0.658).

### 4.1 Encoders (lecture 07: CNN architectures)

| Experiment | Parameters | s/epoch | val mIoU (Δ) | rural test mIoU (Δ) | test mIoU all sites | flicker raw → smoothed |
|---|---|---|---|---|---|---|
| E0 ResNet-34 (reference) | 24.6 M | 16.1 | 0.727 | 0.658 | 0.742 | 2.2 → 0.9 % |
| E1 VGG16-BN | 19.1 M | 22.4 | **0.748** (+0.018) | 0.677 (+0.019) | 0.744 | 2.5 → 1.0 % |
| E2 ResNet-50 | 32.6 M | 21.2 | 0.725 (−0.005) | 0.680 (+0.022) | 0.759 | 2.7 → 1.0 % |
| E3 DenseNet-121 | 13.7 M | 23.4 | 0.741 (+0.011) | **0.689** (+0.031) | **0.768** | 2.2 → 0.9 % |
| E4 EfficientNet-B0 | 5.9 M | 18.2 | 0.742 (+0.012) | 0.644 (−0.014) | 0.729 | 3.6 → 1.3 % |
| E5 ResNet-34, no pretraining | 24.6 M | 13.6 | 0.671 (**−0.059**) | 0.570 (**−0.088**) | 0.698 | 1.0 → 0.4 % |

Rural test IoU per class:

| Class | E0 ResNet-34 | E1 VGG16-BN | E2 ResNet-50 | E3 DenseNet-121 | E4 EfficientNet-B0 | E5 scratch |
|---|---|---|---|---|---|---|
| Man-made | 0.725 | **0.820** | 0.752 | 0.779 | 0.779 | 0.459 |
| Crop fields | 0.626 | **0.695** | 0.688 | 0.691 | 0.629 | 0.634 |
| Trees | 0.822 | 0.819 | 0.831 | **0.834** | 0.795 | 0.770 |
| Grassland | 0.638 | **0.706** | 0.697 | 0.696 | 0.608 | 0.552 |
| Bare ground | 0.869 | 0.854 | 0.898 | **0.904** | 0.848 | 0.832 |
| Water | 0.470 | 0.444 | **0.482** | 0.434 | 0.471 | 0.352 |
| Snow & ice | 0.453 | 0.401 | 0.413 | **0.481** | 0.373 | 0.393 |

**Findings:**
- **Pretraining is the largest effect in the whole project's model experiments.**
  - Without ImageNet weights, ResNet-34 loses 0.06 val and 0.09 rural mIoU; Man-made falls from 0.73 to 0.46.
  - Low-level ImageNet filters (edges, textures) transfer to radar backscatter, although the input is not an optical image.
  - The scratch model has the lowest flicker (1.0 %) because it is less detailed, not because it is better.
- **The encoder choice is within the noise.**
  - VGG16-BN, ResNet-50 and DenseNet-121 are 0.02-0.03 above the pooled reference on rural test. That is inside the reference's own 0.057 range.
  - None is above 0.02 on both val and rural test.
  - All three gain mainly on Man-made, Crop fields and Grassland. None fixes Water or Snow & ice, which are limited by the radar physics (section 15.5 of SESSION_CONTEXT).
- **VGG16-BN wins by the notebook's rule** (+0.022 val against its own E0), but only by +0.018 against the pooled reference. Its rural mean (0.677) equals `Iter2_Test_1_Full` (0.679). Switching the final model would not improve the delivered result.
- **EfficientNet-B0 repeats the PCA lesson:** high val (0.742), low rural test (0.644), highest flicker (3.6 %). Val and test disagree when differences are this small.
- More parameters do not help: ResNet-50 (32.6 M) is not better than DenseNet-121 (13.7 M).

### 4.2 Training choices (lecture 05: optimisation)

| Experiment | val mIoU (Δ) | rural test mIoU (Δ) | test mIoU all sites | epochs |
|---|---|---|---|---|
| T0 reference (AdamW, CE + Lovász) | 0.729 | 0.676 | 0.761 | 30 |
| T1 SGD (Nesterov 0.9, lr 0.01, wd 1e-4) | 0.740 (+0.010) | 0.649 (−0.009) | 0.735 | 23 |
| T2 cross-entropy only (no Lovász) | 0.714 (−0.016) | 0.646 (−0.012) | 0.726 | 31 |

- **SGD vs AdamW: no difference beyond noise.** SGD stops earlier (23 epochs) and is +0.01 on val but −0.03 on rural test against its own reference.
- **Lovász helps a little, consistently:** without it, val and rural test are both lower (−0.016 / −0.012), and Snow & ice falls most (0.486 → 0.405). The direction is the same on both scores, so the loss choice of the 2nd iteration is confirmed.

### 4.3 Label handling (noisy labels)

| Experiment | val mIoU (Δ) | rural test mIoU, agreement pixels (Δ) | rural test mIoU, all WorldCover pixels | Water IoU |
|---|---|---|---|---|
| L0 reference (disagreement weight 0.5) | 0.733 | 0.639 | 0.537 | 0.426 |
| L1 agreement pixels only (weight 0) | **0.752** (+0.022) | **0.681** (+0.023) | 0.538 | 0.458 |
| L2 WorldCover on all pixels (weight 1) | 0.718 (−0.012) | 0.638 (−0.020) | 0.564 | **0.499** |
| L3 label smoothing 0.2 | 0.730 (0.000) | 0.656 (−0.002) | 0.569 | 0.438 |
| L4 soft bootstrapping β = 0.8 | 0.727 (−0.003) | 0.672 (+0.014) | **0.576** | 0.469 |

Pooled reference on all WorldCover pixels: 0.561 (E0 0.571, T0 0.576, L0 0.537).

- **L1 (agreement only) is the only change above +0.02 on both main scores. It is still not adopted:**
  - It is trained and scored on the same pixel set (pixels where WorldCover and Dynamic World agree), so the main score favours it.
  - On all WorldCover pixels, which include the class edges, it is 0.023 below the pooled reference.
  - This is the 1st-iteration label rule that produced blob-shaped maps without edges (Black Forest report section 3, item 3). The score gain comes from ignoring the hard pixels, not from better maps.
  - Its spread over seeds is small (0.008), so the gain on agreement pixels is likely real. For the report, it is a trade-off: a cleaner score against coverage of the edges.
- **L2 (WorldCover everywhere at full weight):** lower on agreement pixels, but the best Water IoU (0.499). Full weight on edge pixels helps thin rivers and shores.
- **L3 (label smoothing 0.2): no effect.** This matches the research note, which expected nothing.
- **L4 (soft bootstrapping): best on all WorldCover pixels (0.576, +0.015)** and neutral on val. It is the most promising noisy-label method, but its gain is within the noise; it needs more seeds.

## 5. Conclusions for the report

1. **The final model stays `Iter2_Test_1_Full`.** No change beats the pooled reference by more than the noise on both val and rural test. The prototype and the delivered weights need no change.
2. **Model comparison for the report:**
   - 6 encoders (lecture 07), 2 optimiser/loss choices (lecture 05) and 4 label strategies, 28 training runs in total.
   - The ranking is clear only for pretraining (large effect) and Lovász (small, consistent effect).
3. **The model is limited by labels and radar, not by architecture.** This is now measured, not only argued:
   - five encoders from 5.9 M to 32.6 M parameters are within 0.05 rural mIoU of each other;
   - Water and Snow & ice stay at 0.40-0.50 for every encoder.
4. **Report noise honestly.**
   - Rural test mIoU of one configuration varies by up to 0.057 between identical runs. Val varies by 0.011.
   - The 13 rural test sites are too few to rank configurations that differ by 0.02.
   - Quote 0.679 for the delivered model and about 0.66 ± 0.03 for the configuration.
5. **Val vs test disagreement appears again** (EfficientNet-B0, SGD): the same lesson as PCA in the 1st iteration.
6. **Future work:** `reports/Future scope.md`.
