# Prototype runs and the next experiments (2026-11-02)

Model in all runs: `Iter2_Test_1_Full/final_model.pt` (`month_year`, 3-month smoothing, rural test mIoU 0.679). Run logs: `prototype/runs/` (written by `prototype/runlog.py`). Section 1 analyses the three logged runs; section 2 asks whether the model has peaked; section 3 proposes the next experiments (to be run after review).

## 1. Logged prototype runs

| Run | Place | Scenes per month | Agreement with Esri 2021 | Majority baseline (largest Esri class) | Gain over baseline |
|---|---|---|---|---|---|
| `20261102_101102_52.5200_13.4050_2021-01` | Berlin centre | 18-24 | 96.6 % | 97.1 % (built area) | -0.5 pp |
| `20261102_101149_19.2183_72.9781_2021-01` | Thane, Mumbai metropolitan region (India) | 2-4 | 89.7 % | 75.7 % | +14.0 pp |
| `20261102_101452_19.2145_72.9841_2021-01` | Thane, 700 m further south-east | 2-4 | 88.2 % | 75.5 % | +12.7 pp |

All three: 5.12 km square, January 2021 start, 25 % patch overlap, GPU.

**Berlin** says little about the model: Esri calls 97 % of the area built, so a model that paints everything red would score 97 %. The model matches the parks (Trees 99 % of the Esri tree pixels found) and part of the Spree (Water 43 %, the river is narrow). Same conclusion as the 1st-iteration Berlin test (2026-10-27).

**Thane** is the informative case: a dense city with a tidal creek, mangroves, wetland and fields.

| Class | Model % | Esri % | Esri pixels found by the model |
|---|---|---|---|
| Man-made | 77.9-78.9 | 75.5-75.7 | 98 % |
| Crop fields | 2.3-4.0 | 2.6-4.9 | 36-43 % |
| Trees | 11.2-14.3 | 7.8-13.4 | 78-81 % |
| Grassland | 1.1-1.2 | 4.3-4.6 | 10-16 % |
| Water | 3.4-6.0 | 4.0-7.3 | 79-81 % |

- **Stable over the year.** The total change from January to December is at most 0.25 pp per class, and the 12 maps look the same, although the radar changes strongly with the monsoon (fields and open land bright in July-September). The seasonal statistics and the smoothing do their job, even with only 2-4 scenes per month (Sentinel-1A only after the loss of Sentinel-1B, one orbit direction), against 18-24 in Berlin.
- **Shapes match** the Esri map: the creek, the lakes, the tree belts along the creek and the field block east of the creek.
- **Grassland is the weakest class** (10-16 % found). The largest miss is an Esri "rangeland" area in the north-west, which the model calls Man-made. Part of this is definition: Esri rangeland covers open scrub, vacant urban land and mudflats, which our labels (WorldCover) often call built-up, bare or trees. The model also extends Trees where Esri has rangeland or crops next to the mangroves.
- **Crop fields** are half found (36-43 %): small fields between buildings are absorbed into Man-made.
- Two points 700 m apart give nearly identical tables, so the result does not depend on the exact window.

**Limits of this evidence.** All three runs are urban; the product goal is rural land. They serve the brief's urban screen recording, but they do not test the rural model. Still to run: the original Black Forest clearing (section 1 of the 2026-10-31 report) and 3-4 rural mosaics, with the Esri check.

## 2. Has the model reached its peak?

Probably close to what the **labels and the radar** allow; not necessarily what the **architecture** allows.

- Validation mIoU plateaus at about 0.73 (best epoch 22, early stop at 32), and identical runs differ by up to 0.02 mIoU (section 11.1 of the 2026-10-31 report).
- The training labels are imperfect: WorldCover is about 75 % correct overall, and WorldCover and Dynamic World agree on only 73.6 % of the labelled pixels (55 % for Grassland). A model trained on these labels cannot score much higher against them.
- The remaining errors are physical: snow vs bare ground, dark forest or radar shadow vs water, crops vs grass outside the growing season.

**So a new architecture will probably add 0.01-0.03 mIoU, not 0.1.** The experiment is still worth running:

1. The brief asks for "a comparison of model variants". A comparison of encoders and training choices from the lectures is a stronger comparison than three input variants.
2. Only one architecture has been tried (U-Net with a ResNet-34 encoder; `ideas.txt` says ResNet-50, which is not correct).
3. An honest result ("the encoder matters little, the labels and the radar set the limit") is a valid finding for the critical discussion.

## 3. Proposed experiments

All experiments keep the data, the `month_year` input (12 channels), the classes, the loss and the 40-epoch schedule of `Iter2_Test_1_Full`, and change **one thing at a time**. Every configuration is trained with **2 seeds**, because one run carries about ±0.02 noise; the winner is chosen by **mean validation mIoU**, and the test scores are only reported (the same rule as before, so the test set is not used for selection).

### 3.1 Encoders inside the same U-Net (lecture 07: classic CNN architectures)

| ID | Encoder | Lecture concept | Parameters (with decoder, approx.) | Expected effect |
|---|---|---|---|---|
| E0 | ResNet-34 (current) | residual connections | 24.5 M | baseline |
| E1 | VGG-16 with batch norm | plain deep stack of 3×3 convolutions, no residuals | about 25 M | shows what the residual connections add |
| E2 | ResNet-50 | bottleneck blocks with 1×1 convolutions | about 32 M | deeper and wider; may overfit the 2218 patches |
| E3 | DenseNet-121 | dense connections, feature reuse | about 12 M | fewer parameters, strong on small datasets |
| E4 | EfficientNet-B0 (beyond the lectures) | depthwise separable convolutions, SiLU activation | about 6 M | small and modern; a check of whether size matters |
| E5 | ResNet-34 without ImageNet weights | transfer learning (lecture 07) | 24.5 M | shows what pretraining adds for radar input |

All encoders come from torchvision with ImageNet weights (except E5); the first convolution is adapted to 12 channels in the same way as now (mean of the RGB filters). GoogLeNet / Inception (also in lecture 07) is left out: torchvision's version has auxiliary heads and no clean stage outputs for a U-Net, so it would need a hand-built encoder.

### 3.2 Training choices (lecture 05: improving deep networks), on the best encoder of 3.1

| ID | Change | Lecture concept | Question answered |
|---|---|---|---|
| T1 | SGD with momentum 0.9 (Nesterov) instead of AdamW | optimisers | is AdamW the right choice? |
| T2 | cross-entropy only, no Lovász term | sparse categorical cross-entropy | what does the IoU-based Lovász loss add? |

### 3.3 Class percentages from probabilities (evaluation only, no training)

The prototype counts the pixels of each class after `argmax`. Alternative: average the softmax probabilities over the area ("soft" percentages). Errors near class boundaries then partly cancel instead of flipping whole pixels, which should reduce the false month-to-month change. Measured on the test sites: (a) the error of the class shares against the label shares, (b) the sum of month-to-month changes on sites without real change. If soft percentages are better, the prototype switches to them (a few lines in `prototype/pipeline.py`).

### 3.4 Optional: ensemble ("mix and match")

Average the softmax of the best 2-3 models of 3.1. Ensembles usually add 0.01-0.03 mIoU and reduce flicker, but the prototype would load several weight files (2-3 × 100 MB, 2-3 × the inference time), which breaks the "one final model" decision (6.9 of the 2026-10-31 report). Only worth it if the gain is clearly above the noise.

### 3.5 Optional: compare with Google's Dynamic World in the prototype

Dynamic World (Google, Sentinel-2) is available per month, so it could be shown next to our monthly maps. It is one of our label sources (agreement weighting), so it is **not** an independent check; Esri stays the independent reference for the report.

### 3.6 Cost and changes needed

| Part | Runs | Time on the A6000 (estimate) |
|---|---|---|
| 3.1 encoders | 6 × 2 seeds | 60-90 min (ResNet-50 and DenseNet about 1.5-2 × slower per epoch) |
| 3.2 training choices | 2 × 2 seeds | about 20 min |
| 3.3 soft percentages | evaluation only | a few minutes |

Code changes before the run:
- `Notebooks/2nd_iteration.ipynb`: a generic U-Net that takes the encoder name; an experiment list (encoder, optimiser, loss, seed); a results table with mean ± spread over seeds, rural test mIoU, flicker, parameters and seconds per epoch; the encoder and training settings written to `model_card.json`.
- `prototype/model.py`: the same generic U-Net, built from the encoder named in the model card, so that any winner runs in the prototype. Existing weights keep working (no encoder in the card = ResNet-34).
- `prototype/pipeline.py`: soft percentages, if 3.3 shows they are better.

### 3.7 Decision rule

- A configuration replaces the current model only if its mean validation mIoU beats E0 by more than the seed spread (about 0.02).
- If nothing does, `Iter2_Test_1_Full` stays the final model, and the comparison table goes into the report as evidence that the labels and the radar, not the architecture, limit the result.

## 4. Notebooks for the experiments (2026-11-03)

Three notebooks, built from `Notebooks/2nd_iteration.ipynb` (sections 2-5 copied unchanged; settings, model, loss, training loop, results and export replaced). Each trains its configurations with 2 seeds, writes `runs.csv` after every run, compares by mean val mIoU against its own reference run, and exports the best run as `best_model.pt` + `model_card.json` (with `encoder` and `beats_reference`).

| Notebook | Results folder | Configurations | Runs | Time (estimate) |
|---|---|---|---|---|
| `3rd_iteration_encoders.ipynb` | `Weights/Iter3_Encoders/` | E0 ResNet-34, E1 VGG-16-BN, E2 ResNet-50, E3 DenseNet-121, E4 EfficientNet-B0, E5 ResNet-34 without pretraining | 12 | 1-1.5 h |
| `3rd_iteration_training.ipynb` | `Weights/Iter3_Training/` | T0 reference, T1 SGD + Nesterov momentum (lr 0.01, wd 1e-4), T2 cross-entropy only | 6 | 30 min |
| `3rd_iteration_labels.ipynb` | `Weights/Iter3_Labels/` | L0 reference, L1 agreement pixels only, L2 WorldCover on all pixels at full weight, L3 label smoothing 0.2, L4 soft bootstrapping (beta 0.8) | 10 | 50 min |

Order: encoders first. If an encoder beats ResNet-34 by more than 0.02, set `ENCODER` in the other two notebooks to it before running them; otherwise run them as they are (ResNet-34).

`prototype/model.py` now holds the same generic U-Net; `prototype/pipeline.py` builds the encoder named in the model card (no entry = ResNet-34, so all older weights load unchanged).

### 4.1 Missing roads in the labels

In many label tiles rural roads and tracks are not marked. Expected effect on the score: small.
- A rural road (3-8 m) is narrower than a 10 m pixel; WorldCover gives such mixed pixels the surrounding class. Only wide roads are Man-made.
- Sentinel-1 GRD resolves about 20 x 22 m (resampled to 10 m), so the radar barely sees narrow roads either: input and label lose the road together.
- Where Dynamic World marks a road and WorldCover does not, the pixel is a disagreement pixel: half weight in training, not in the Lovász term, not in the main score (agreement pixels). It lowers only `mIoU all`.
- The label errors that cost score are larger areas: Grassland vs Crop fields vs shrubs (Grassland agreement 55 %), mixed pixels at field and forest edges, seasonal wetland. The labels notebook tests training choices against these.
