"""Define the PATCH x PATCH training patches per split and compute dataset statistics.

Patches are stored as an index into the site GeoTIFFs (no pixel copies), keeping the
dataset compact; the training loader cuts each patch with a windowed read.
A patch is kept when at most 60% of its pixels are unlabelled in config.TRAIN_LABEL_BAND (WorldCover in
the 2nd iteration) and every month has valid S1 data for >= 95% of its pixels (terrain flattening masks
layover and shadow, so steep tiles lose a few pixels). Each patch carries all 12 monthly composites,
so the training loader can sample any month against the same 2021 label.

4th iteration: training sites also have radar of config.EXTRA_YEARS (same WorldCover 2021 label). Their patches
use the same windows as 2021, are checked again on that year's radar (months without any scene in the whole
tile are ignored: Sentinel-1B was lost after 2021) and go to train_extra_patches.csv. The `iteration` column
(2 = sites of the 2nd iteration, 4 = sites added in the 4th) lets the notebook train with or without the new
sites. Normalisation statistics stay those of the 2021 training patches.

Outputs in Dataset/patches/:
  <split>_patches.csv       index, site, region, focus, iteration, row, col (pixel offsets in the site tile)
  train_extra_patches.csv   index, site, year, region, focus, iteration, row, col (later years, training sites)
  norm_stats.json           per-band mean/std in dB from train patches
  class_distribution.json   pixel counts and shares per split
  site_summary.csv          per site: split, focus, kept patches, share of pixels where WorldCover and Dynamic World agree, class shares
"""
import csv
import json

import numpy as np
import rasterio

import config

P = config.PATCH
MAX_IGNORE = 0.6
MIN_VALID = 0.95
SPLITS = ("train", "val", "test")


def load_site(name, year=config.YEAR):
    with rasterio.open(config.s1_dir(year) / f"{name}.tif") as src:
        x = src.read().reshape(12, 2, src.height, src.width)
    with rasterio.open(config.label_dir(year) / f"{name}.tif") as src:
        y = src.read(config.LABEL_BANDS.index(config.TRAIN_LABEL_BAND) + 1)
    return x, y


def patch_origins(x, y):
    h, w = y.shape
    # months without any scene in the whole tile (possible after 2021) are ignored, not counted as invalid
    months = [m for m in range(12) if (x[m] != config.NODATA).any()]
    for r in range(0, h - P + 1, P):
        for c in range(0, w - P + 1, P):
            if (y[r:r + P, c:c + P] == 0).mean() > MAX_IGNORE:
                continue
            valid = (x[months, :, r:r + P, c:c + P] != config.NODATA).all(axis=1).mean(axis=(1, 2))
            if valid.min() < MIN_VALID:
                continue
            yield r, c


def main():
    config.PATCH_DIR.mkdir(parents=True, exist_ok=True)
    n_classes = len(config.CLASSES)
    class_names = [config.CLASSES[k][0] for k in range(n_classes)]
    counts = {split: np.zeros(n_classes, dtype=np.int64) for split in SPLITS}
    n_patches = {split: 0 for split in SPLITS}
    sites_in = {split: [] for split in SPLITS}
    band_sum, band_sq, band_n = np.zeros(2), np.zeros(2), np.zeros(2)
    summary = []

    writers, files = {}, {}
    for split in SPLITS:
        files[split] = open(config.PATCH_DIR / f"{split}_patches.csv", "w", newline="")
        writers[split] = csv.writer(files[split])
        writers[split].writerow(["index", "site", "region", "focus", "iteration", "row", "col"])

    for site in config.SITES:
        name, split = site["name"], site["split"]
        if not ((config.S1_DIR / f"{name}.tif").exists() and (config.LABEL_DIR / f"{name}.tif").exists()):
            print(f"{name}: not downloaded, skipping")
            continue
        x, y = load_site(name)
        site_counts = np.zeros(n_classes, dtype=np.int64)
        origins = list(patch_origins(x, y))
        for r, c in origins:
            yp = y[r:r + P, c:c + P]
            site_counts += np.bincount(yp.ravel(), minlength=n_classes)
            writers[split].writerow([n_patches[split], name, site["region"], site["focus"],
                                     site.get("iteration", 2), r, c])
            n_patches[split] += 1
            if split == "train":
                db = config.decode_db(x[:, :, r:r + P, c:c + P])
                for b in range(2):
                    v = db[:, b][~np.isnan(db[:, b])].astype(np.float64)
                    band_sum[b] += v.sum()
                    band_sq[b] += (v ** 2).sum()
                    band_n[b] += v.size
        counts[split] += site_counts
        sites_in[split].append(name)

        kept_pct = json.loads((config.LABEL_DIR / f"{name}_stats.json").read_text())["kept_pct"]
        labelled = max(site_counts[1:].sum(), 1)
        summary.append({"site": name, "split": split, "region": site["region"], "focus": site["focus"],
                        "patches": len(origins), "agree_pct": kept_pct,
                        **{class_names[k]: round(site_counts[k] / labelled * 100, 1) for k in range(1, n_classes)}})
        print(f"{split}/{name}: {len(origins)} patches")

    for f in files.values():
        f.close()

    # later years of the training sites: same windows, checked on that year's radar
    extra_counts = {year: np.zeros(n_classes, dtype=np.int64) for year in config.EXTRA_YEARS}
    with open(config.PATCH_DIR / "train_extra_patches.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "site", "year", "region", "focus", "iteration", "row", "col"])
        k = 0
        for site in config.SITES:
            if site["split"] != "train":
                continue
            for year in config.EXTRA_YEARS:
                name = site["name"]
                if not ((config.s1_dir(year) / f"{name}.tif").exists()
                        and (config.label_dir(year) / f"{name}.tif").exists()):
                    continue
                x, y = load_site(name, year)
                origins = list(patch_origins(x, y))
                for r, c in origins:
                    extra_counts[year] += np.bincount(y[r:r + P, c:c + P].ravel(), minlength=n_classes)
                    writer.writerow([k, name, year, site["region"], site["focus"], site.get("iteration", 2), r, c])
                    k += 1
                print(f"train {year}/{name}: {len(origins)} patches")
    print(f"\ntrain, later years: {k} patches ({', '.join(map(str, config.EXTRA_YEARS))})")

    with open(config.PATCH_DIR / "site_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    mean = band_sum / band_n
    std = np.sqrt(band_sq / band_n - mean ** 2)
    stats = {"bands": ["VV", "VH"], "units": "dB", "mean": mean.round(4).tolist(), "std": std.round(4).tolist()}
    (config.PATCH_DIR / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    print("norm stats:", stats)

    distribution = {}
    for split in SPLITS:
        labelled = max(counts[split][1:].sum(), 1)
        distribution[split] = {
            "sites": sites_in[split],
            "patches": n_patches[split],
            "classes": {class_names[k]: {"pixels": int(counts[split][k]),
                                         "share_of_labelled": round(float(counts[split][k] / labelled), 4)
                                         if k else None}
                        for k in range(n_classes)},
        }
        print(f"\n{split}: {n_patches[split]} patches from {len(sites_in[split])} sites")
        for k in range(1, n_classes):
            print(f"  {class_names[k]:<22} {counts[split][k] / labelled * 100:5.1f}%")
    for year, cnt in extra_counts.items():
        labelled = max(cnt[1:].sum(), 1)
        distribution[f"train_{year}"] = {
            "classes": {class_names[k]: {"pixels": int(cnt[k]),
                                         "share_of_labelled": round(float(cnt[k] / labelled), 4) if k else None}
                        for k in range(n_classes)}}
    (config.PATCH_DIR / "class_distribution.json").write_text(json.dumps(distribution, indent=2))


if __name__ == "__main__":
    main()
