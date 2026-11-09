# Prototype run analysis (2026-11-08)

Source: `prototype/runs/` (commit `6f90cfe`), 20 runs from 2026-11-06 to 2026-11-08. All runs use the final model (`month_year`, smoothed), a 512 x 512 px area (5.1 km) and 2021 data, and every run has 100 % valid radar pixels. Repeated runs of the same place gave identical numbers (Black Forest 8 times, Mumbai and Monaco twice each), so the pipeline is deterministic.

## 1. Results per place

Agreement = share of pixels where the most frequent monthly class equals the reference map. Baseline = agreement of a map that shows only the largest reference class. Esri is the independent check; WorldCover was the training label, and Dynamic World set the training weights.

| Place | Type | Scenes / month | Esri agreement (baseline) | WorldCover | Dynamic World | Largest classes (model, year mean) | Flicker % |
|---|---|---|---|---|---|---|---|
| Black Forest, Germany | rural test site | 15-21 | **70.6** (55.7) | 91.8 (54.9) | 85.3 (58.6) | Trees 56.7, Grassland 40.1 | 0.30 |
| Willamette foothills, USA | rural test site | 9-14 | **68.1** (51.3) | 74.8 (57.6) | 74.7 (53.2) | Trees 52.1, Grassland 24.4, Crop fields 22.3 | 0.45 |
| Chilean lakes | rural test site | 7-12 | **40.3** (64.1) | 69.9 (49.0) | 72.4 (51.6) | Grassland 73.8, Trees 12.5, Crop fields 11.7 | 0.39 |
| Haut-Uele, DR Congo | rural, unseen region | 4-6 | **73.1** (55.6) | 58.3 (68.1) | 80.0 (90.5) | Trees 72.5, Grassland 27.5 | 0.89 |
| Mykonos, Greece | rural island | 8-12 | **71.5** (57.1) | 85.5 (77.0) | 58.3 (43.4) | Grassland 82.5, Water 8.7, Man-made 7.8 | 0.34 |
| Thane, India | city, creek, mangroves | 2-4 | **89.7** (75.7) | 61.9 (44.2) | 87.0 (71.5) | Man-made 78.8, Trees 14.3, Water 3.4 | 0.49 |
| Mumbai, India | city | 2-4 | **92.3** (89.9) | 68.0 (60.7) | 90.5 (85.4) | Man-made 89.0, Trees 6.1, Water 2.6 | 0.37 |
| Berlin centre, Germany | city | 18-24 | **96.6** (97.1) | 75.7 (71.0) | 97.0 (93.8) | Man-made 94.5, Trees 4.7 | 0.05 |
| Monaco | city and sea | 16-22 | **72.1** (59.9) | 84.1 (59.3) | 92.6 (59.5) | Water 59.8, Man-made 28.0, Trees 11.8 | 0.19 |
| Manchar Lake, Pakistan (26.47, 67.63) | lake | 4-9 | 94.2 (93.1) | 95.9 (96.8) | 96.4 (97.8) | Water 96.3 | 0.10 |
| Sahara, Chad/Libya (22.0, 17.0) | desert | 1-3 | 96.4 (96.4) | 100.0 (100.0) | 99.9 (99.9) | Bare ground 100.0 | 0.00 |

The Sahara and Manchar Lake runs are stored with a wrong place name ("Berlin centre" and "Nambia, Haut-Uele"). Typed coordinates did not reset the name of the previous preset or search. Fixed on 2026-11-08: typing coordinates now marks the place as custom.

## 2. Findings

### 2.1 The model beats the baseline on most mixed areas

- On 7 of 9 mixed areas (all except Chile and Berlin) the model beats the Esri baseline by 2-17 points. The best gains are on Willamette (+16.8), Haut-Uele (+17.5), Black Forest (+14.9), Thane (+14.0), Mykonos (+14.4) and Monaco (+12.2).
- Areas with one dominant class (Berlin, Mumbai, Manchar Lake, Sahara) score 92-97 %, but only 0-3 points above the baseline. A high agreement there says little about the model. The prototype warns when the gain is below 2 points.

### 2.2 Disagreements are mostly definitions, not radar errors

- **Chile, the only clear loss (40.3 vs 64.1 baseline):** Esri calls 64 % of the area crops; the model calls 74 % grassland and finds 95 % of Esri's grassland. WorldCover and Dynamic World agree with the model (70-72 %, about 20 points above their baselines). Southern Chile's lake district is mostly pasture: Esri maps managed pasture as crops, while WorldCover and Dynamic World map it as grassland. The model follows its training definition.
- **Mykonos:** Esri marks 34 % of the island as built area; the model marks 7.8 %. Mykonos has scattered houses on dry grass and rock, which Esri counts as built, while the 10 m radar sees mostly open ground. Water is 96 % correct.
- **Monaco:** Esri marks 23 % bare ground, mostly the dense city and harbour; the model calls it Man-made (28 % vs Esri 9 %). The sea is 98.9 % correct.
- **Thane and Mumbai:** WorldCover agreement is low (62-68 %) because WorldCover maps more trees and grass inside cities. Esri and Dynamic World agree with the model (87-92 %).

### 2.3 Weak points confirmed

- **Africa (Haut-Uele, DR Congo):** the forest–savanna edge, a region unlike most training data. The model finds 91 % of Esri's trees but only 51 % of its grassland. It is below the WorldCover and Dynamic World baselines (58.3 vs 68.1, 80.0 vs 90.5), which map more forest than the model. Flicker is the highest of all runs (0.89 %), and the largest total change is Grassland −1.26 pp and Trees +1.26 pp. With only 4-6 scenes per month, this is most likely noise, not real change. It matches the weak Africa score of the test set (0.49 rural mIoU).
- **Narrow water in cities:** Berlin finds only 42.7 % of Esri's water (the Spree and the canals), the label and resolution limit described in the 2026-11-07 report. Wide water is found: Monaco sea 98.9 %, Manchar Lake 96 %, Mykonos sea 96 %.
- **Few scenes in India:** Thane and Mumbai have only 2-4 Sentinel-1 scenes per month in 2021, against 15-24 in Europe. The model still agrees well with Esri and Dynamic World there, but fewer scenes give a noisier composite.

### 2.4 Stability and change

- Flicker is 0.05-0.49 % of pixels per month everywhere except Haut-Uele (0.89 %).
- The largest total change from January to December is under 1.3 pp in every run, and under 0.25 pp in 8 of 11 places.
- No run shows a land-use change that would be real in one year. The month-to-month numbers are at noise level, as expected for stable places.
- Black Forest: the Sentinel-2 photo shows snow in December to March, while the model keeps Trees and Grassland, with a total change of only ±0.12 pp. Seasonal snow does not create fake land-use change.

## 3. For the final report

- **Case study:** Black Forest (rural, snow example) and Thane (urban, required screen recording), with the Esri, WorldCover and Dynamic World numbers above.
- **Regions unlike the training data:** Haut-Uele (DR Congo) and Mykonos.
- **Reference maps disagree with each other:** Chile and Monaco show that agreement depends on the reference map's class definitions (pasture = crops in Esri; dense city = bare in Esri).
- **Change inflation:** compare each total change with the flicker of that run; Haut-Uele is the example where noise (0.89 % flicker, 4-6 scenes) produces a 1.26 pp "change".
