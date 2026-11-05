# Future scope

Ideas not tried in this project, for the report's future-work section. Each one names the evidence from our runs that motivates it and what it is expected to change. Results and run details: `reports/2026-11-05 3rd iteration experiment results.md`.

All gains below about 0.02 rural mIoU need 5 seeds and more test sites to be measured: the reference configuration itself varies by 0.057 rural mIoU between identical runs (results report, section 3).

| Idea | Why, from our results | Expected effect |
|---|---|---|
| **Encoder pretrained on radar images (SAR)**, e.g. a SoftCon ResNet-50 pretrained on Sentinel-1 (TorchGeo) | Pretraining is the only model factor with a large effect: ImageNet weights add 0.06 val and 0.09 rural mIoU over training from scratch (results report, section 4.1). Weights learned on radar instead of photos are the next step. | About +0.02 mIoU in the literature for Sentinel-1 segmentation; mostly on classes where radar texture matters (Crop fields, Grassland). |
| **Self-training**: label new, unlabelled Sentinel-1 tiles with the model's own confident predictions, then retrain on them | The model is limited by noisy labels (WorldCover and Dynamic World agree on 73.6 % of pixels; L1 and L4 in section 4.3). Confident predictions can replace wrong labels and add regions without new manual labels. | Small but cheap gain; it works best together with soft bootstrapping (L4), the most promising label method here. |
| **Extra Sentinel-1 timing channels**: date of the lowest and highest backscatter, length of the growing season, number of scenes per month | Crop fields vs Grassland is the main confusion. Crops are ploughed and harvested at one time of year; grass changes little. Our 9 seasonal statistics say how much the radar changes, but not when. | Clearer Crop/Grassland separation; the timing must be order-independent across hemispheres (the PCA lesson, 1st iteration). |
| **Radar from several years** (2021-2024 with the 2021 labels) | Sentinel-1B failed in December 2021, so later years have about half the scenes per month. The prototype runs on those years. More years also show the model wet and dry years. | Better robustness for 2022+ maps; little gain on the 2021 test set. Labels drift where land changed after 2021. The 2025 prototype test shows whether this is needed. |
| **More training sites in Africa and Oceania** | The weakest regions: Africa 0.49, Oceania 0.44 rural mIoU (`Iter2_Test_1_Full`). Oceania has only 8 training sites and 1 test site. | The most direct fix for regional generalisation; also gives more test sites, which reduces the noise of every comparison. |
| **Graph-based methods**: superpixel or segment graphs with a graph neural network (GCN, GAT) head, months as connected nodes, or graph-based post-processing | About half the errors lie within 20 m of a class boundary (Thane analysis), and the maps still show speckled pixels inside fields. A graph over segments enforces that one field gets one class and passes information between neighbouring regions and months. | Cleaner shapes and less flicker; the gain in mIoU is uncertain because the edge errors are partly a radar-resolution limit. Needs a literature check first. |

Cheaper follow-ups with the existing runs:
- **Seed ensemble:** average the softmax of the 6 reference runs. No training needed.
- **Inference-only changes:** 12-month softmax average, flip test-time augmentation, a per-class bias that lowers Water over-prediction (precision 0.59 vs recall 0.74).
- **Confirm the near-winners with 5 seeds:** soft bootstrapping (L4), DenseNet-121 and VGG16-BN.
