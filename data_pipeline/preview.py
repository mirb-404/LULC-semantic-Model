"""Save a quick-look PNG per site: S1 false colour for January and July next to the label map.

False colour: R = VV, G = VH, B = VV - VH (all in dB, contrast-stretched).
Usage: python data_pipeline/preview.py [--sites berlin paris]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import config


def false_colour(s1, month):
    vv = config.decode_db(s1[2 * (month - 1)])
    vh = config.decode_db(s1[2 * (month - 1) + 1])
    rgb = np.stack([vv, vh, vv - vh], axis=-1)
    for b, (lo, hi) in enumerate([(-20, 0), (-27, -7), (2, 15)]):
        rgb[..., b] = np.clip((rgb[..., b] - lo) / (hi - lo), 0, 1)
    rgb[np.isnan(rgb)] = 0
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="*")
    args = parser.parse_args()
    config.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap([c for _, c in config.CLASSES.values()])

    for site in config.SITES:
        name = site["name"]
        if args.sites and name not in args.sites:
            continue
        s1_path, label_path = config.S1_DIR / f"{name}.tif", config.LABEL_DIR / f"{name}.tif"
        if not s1_path.exists():
            continue
        with rasterio.open(s1_path) as src:
            s1 = src.read()
        labels = None
        if label_path.exists():
            with rasterio.open(label_path) as src:
                labels = src.read(config.LABEL_BANDS.index(config.TRAIN_LABEL_BAND) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
        axes[0].imshow(false_colour(s1, 1))
        axes[0].set_title(f"{name} S1 January {config.YEAR}")
        axes[1].imshow(false_colour(s1, 7))
        axes[1].set_title(f"{name} S1 July {config.YEAR}")
        if labels is not None:
            axes[2].imshow(labels, cmap=cmap, vmin=0, vmax=len(config.CLASSES) - 1, interpolation="nearest")
        axes[2].set_title(f"Training labels {config.YEAR} ({config.TRAIN_LABEL_BAND})")
        for ax in axes:
            ax.axis("off")
        fig.legend(handles=[Patch(color=c, label=n) for n, c in config.CLASSES.values()],
                   loc="lower center", ncol=4, fontsize=9)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(config.PREVIEW_DIR / f"{name}.png", dpi=110)
        plt.close(fig)
        print(f"saved preview {name}.png")


if __name__ == "__main__":
    main()
