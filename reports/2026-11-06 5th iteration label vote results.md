# 5th iteration: label vote results (2026-11-06)

Source: `Notebooks/5th_iteration_label_vote.ipynb` as run in commit `1654c0e` (Datalab, RTX PRO 6000, 2 seeds per configuration, 13 min). Output folder: `Weights/Iter5_LabelVote_20261106_143525/` on Datalab. Dataset build log: `build3.log`.

## 1. Verdict

The notebook says V1 (vote labels, 2021 data) beats the reference by more than the noise. That conclusion is too strong.

- V1 wins clearly only when it is scored on the vote labels, which are the labels it was trained on.
- On WorldCover labels the gain is +0.010 to +0.012, below the noise threshold of 0.02.
- On all WorldCover pixels V1 is worse (−0.014).
- More data (V2) did not help, as in the 4th iteration.

V1 is a reasonable model, but it is not shown to be better than `Iter2_Test_1_Full`. Its best seed (`V1_vote_s7`, rural WorldCover mIoU 0.675) is level with `Iter2_Test_1_Full` (0.679).

## 2. Dataset build

- The build ran in 6 min. It added the Esri tiles and the vote labels.
- The trusted share of training pixels rose from 73.6 % (WorldCover and Dynamic World agree) to 97.3 % (at least 2 of the 3 maps agree).
  - Per class: Man-made 95 → 99 %, Crop fields 71 → 97 %, Trees 78 → 98 %, Grassland 56 → 96 %, Bare ground 72 → 96 %, Water 96 → 100 %, Snow & ice 88 → 95 %.
- Much of this rise follows from the rule itself. With three maps, almost every pixel gets a 2-of-3 majority. A higher vote share does not prove the labels are more correct. It shows that a third map breaks the ties.
- Some 2024 site-years were skipped because they had no Sentinel-1 IW scenes in 2024, as expected after the loss of Sentinel-1B: `etosha_pan_edge`, `etosha_pan_pure`, `gezira_scheme_pure`, `matopos_granite` (12 empty months each), and `parana_river` (3 empty months).

## 3. Results (mean of 2 seeds)

| | V0 WorldCover | V1 vote | V2 vote + all data |
|---|---|---|---|
| val vote mIoU (decision metric) | 0.638 | **0.695** (+0.057) | 0.684 (+0.046) |
| val WorldCover mIoU | 0.723 | 0.735 (+0.012) | 0.736 (+0.013) |
| rural vote mIoU | 0.520 | **0.573** (+0.053) | 0.566 |
| rural WorldCover mIoU (agreement pixels) | 0.651 | 0.661 (+0.010) | 0.654 |
| rural mIoU, all WorldCover pixels | **0.551** | 0.537 (−0.014) | 0.502 (−0.049) |
| rural tolerant mIoU | 0.689 | **0.758** (+0.069) | 0.716 |
| test mIoU, all sites | 0.740 | 0.739 | 0.709 |
| flicker raw / smoothed % | **1.97 / 0.77** | 2.38 / 0.96 | 2.35 / 0.96 |
| train-val gap | 0.117 | 0.100 | 0.084 |
| rural spread between seeds | 0.016 | 0.029 | 0.012 |

V0 matches the earlier reference (pooled val 0.730, rural 0.658), so the run is consistent with the 3rd and 4th iterations.

## 4. What the numbers show

### 4.1 The decision metric favours the vote models

- `DECIDE_BY = "val vote mIoU"` scores every model on the vote labels.
- V0 never saw those labels. V1 and V2 were trained on them.
- A model scored on its own training labels gains automatically, so the +0.057 is largely built in.
- A fair comparison uses a label set that none of the configurations was trained on. No such set exists here. The closest substitute is to check both label sets:
  - on WorldCover, V1 gains +0.012 (val) and +0.010 (rural), which is within the noise;
  - on all WorldCover pixels, V1 loses 0.014.
- The `beats_reference: True` in V1's `model_card.json` therefore comes from the choice of metric, not from a proven improvement.

### 4.2 The tolerant score is the most interesting signal, but it is also not neutral

- Tolerant mIoU counts a prediction as correct if it matches any of the three maps. V1 gains +0.069 over V0 on it.
- The score includes Esri, and V1 was trained partly on Esri, so V1 is again favoured.
- The seed spread is large: V1 0.721 vs 0.794, V0 0.671 vs 0.706.
- Even so, both V1 seeds beat both V0 seeds on rural vote mIoU (0.563 and 0.584 vs 0.519 and 0.521) and on tolerant mIoU. The direction is consistent. What is in doubt is the size of the effect and its meaning.

### 4.3 Per class: the classes trade places again

Rural test IoU on WorldCover agreement pixels:

| Class | V0 | V1 | V2 |
|---|---|---|---|
| Man-made | 0.663 | 0.718 | **0.766** |
| Crop fields | **0.676** | 0.649 | 0.624 |
| Trees | 0.825 | 0.826 | 0.827 |
| Grassland | 0.666 | 0.670 | 0.664 |
| Bare ground | 0.885 | **0.899** | 0.874 |
| Water | **0.489** | 0.455 | 0.476 |
| Snow & ice | 0.349 | **0.409** | 0.351 |

- Man-made improves steadily: Esri and Dynamic World draw towns more widely than WorldCover.
- Crop fields gets worse on WorldCover labels but better on vote labels (0.441 → 0.508). The model is moving towards the other maps' idea of cropland, which is the intended effect.
- Grassland is flat on WorldCover labels but improves on vote labels (0.530 → 0.618).
- Water and Snow & ice stay weak (below 0.5) in every configuration. These are the radar-physics limits found in the 3rd iteration.

### 4.4 Weak regions are not fixed

Winner (`V1_vote_s7`) per region, compared with `Iter2_Test_1_Full`:

| Region | Iter2_Test_1_Full | V1_vote_s7 |
|---|---|---|
| Africa | 0.492 | 0.484 |
| Asia | 0.656 | 0.642 |
| Europe | 0.654 | 0.659 |
| N. America | 0.569 | 0.631 |
| Oceania | 0.441 | 0.400 |
| S. America | 0.584 | 0.563 |

Only N. America clearly gains. Africa and Oceania are still the weakest regions, and Oceania is 16 patches from a single site.

### 4.5 More data still does not help

- V2 (vote labels, years 2022-2024 and the new sites) is no better than V1 on any score.
- It is clearly worse on all WorldCover pixels (0.502) and on all test sites (0.709).
- This matches the 4th iteration: the limit is not the amount of data.

### 4.6 Smaller points

- Flicker is higher for both vote models (about 2.4 % vs 2.0 %, smoothed 0.96 % vs 0.77 %). The vote labels are a little noisier from map to map.
- The train-val gap is lower (0.100 and 0.084 vs 0.117). The vote labels are easier to fit consistently.
- The class bias table is all zeros (bias off), and `Weights/Iter2_Test_1_Full` is still not on the Datalab machine.

## 5. Consequences

- **Final model.** No configuration beats `Iter2_Test_1_Full` on a label set it was not trained on. Keeping `Iter2_Test_1_Full` is the safe choice. It also keeps the prototype's Esri check independent.
- **If V1 is adopted anyway**, the argument is that three maps are a better reference than one. It must be stated as a choice of reference, not as a measured gain. The Esri check in the prototype is then no longer independent, and Dynamic World should be used as the reference map.
- **A decisive test would need an independent reference**, such as a small hand-checked gold set on Kano, Madagascar and a Chilean pasture. With the 2026-11-10 deadline, this is future work.
- **For the report (critical discussion).** The label maps disagree on 30-44 % of pixels in the weak classes. A vote resolves most ties but cannot be validated without independent ground truth. Scores depend on which map is called the truth: on the same predictions, V1's rural mIoU is 0.537, 0.573, 0.661 or 0.758, depending on the reference.
