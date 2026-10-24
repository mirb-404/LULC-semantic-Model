"""Download Sentinel-1 monthly composites and consensus labels per site from Google Earth Engine.

Sentinel-1 (COPERNICUS/S1_GRD) is already processed by Earth Engine with the S1 Toolbox:
orbit file, GRD border noise removal, thermal noise removal, radiometric calibration
(sigma0), terrain correction (SRTM 30 m / ASTER DEM) and dB conversion. On top of that:
  1. keep IW scenes with both VV and VH, ascending and descending passes,
  2. mask near/far-range edges (incidence angle outside 31-45 deg),
  3. average each month in linear power (temporal multilooking reduces speckle),
  4. convert back to dB and apply a 3x3 focal median (spatial speckle filter).
Output: Dataset/s1/<site>.tif, 24 uint8 bands (VV_01, VH_01, ..., VH_12), see config.decode_db.

Labels: ESA WorldCover 2021, the 2021 per-pixel mode of Dynamic World and its mode over
snow-free observations are mapped
to the target classes and sampled with nearest neighbour onto the exact S1 grid. The
consensus band (training label) is computed locally by config.consensus_labels.
`--relabel` recomputes the consensus band and stats from existing label files, offline.
Output: Dataset/labels/<site>.tif, 4 uint8 bands (consensus, worldcover, dynamicworld,
dynamicworld_snowfree).

Usage: python data_pipeline/download.py [--project <gee-project-id>] [--sites berlin lagos] [--relabel]
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

import config

CHUNK = 512


def utm_epsg(lat, lon):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def site_grid(site):
    """UTM grid of TILE_SIZE_M x TILE_SIZE_M centred on the site, snapped to the pixel size."""
    epsg = utm_epsg(site["lat"], site["lon"])
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(site["lon"], site["lat"])
    half = config.TILE_SIZE_M / 2
    x0 = round((x - half) / config.PIXEL_M) * config.PIXEL_M
    y_top = round((y + half) / config.PIXEL_M) * config.PIXEL_M
    return epsg, x0, y_top, config.TILE_SIZE_M // config.PIXEL_M


def mask_edges(img):
    angle = img.select("angle")
    return img.updateMask(angle.gt(31).And(angle.lt(45))).select(["VV", "VH"])


def month_collection(region, month):
    start = ee.Date.fromYMD(config.YEAR, month, 1)
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate(start, start.advance(1, "month"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))


def monthly_composite(region, month):
    linear = month_collection(region, month).map(
        lambda img: ee.Image(10).pow(mask_edges(img).divide(10)))
    db = linear.mean().log10().multiply(10)
    db = db.focalMedian(radius=1, kernelType="square", units="pixels")
    return db.rename([f"VV_{month:02d}", f"VH_{month:02d}"])


def s1_image(region):
    stack = ee.Image.cat([monthly_composite(region, m) for m in config.MONTHS])
    codes = stack.subtract(config.DB_MIN).divide(config.DB_STEP).round().clamp(1, 255)
    return codes.unmask(config.NODATA).toUint8()


def label_image(region):
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    wc = wc.remap(list(config.WORLDCOVER_TO_CLASS), list(config.WORLDCOVER_TO_CLASS.values()), 0)
    dw_year = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
               .filterBounds(region)
               .filterDate(f"{config.YEAR}-01-01", f"{config.YEAR + 1}-01-01")
               .select("label"))
    to_class = (list(config.DYNAMICWORLD_TO_CLASS), list(config.DYNAMICWORLD_TO_CLASS.values()), 0)
    dw = dw_year.mode().remap(*to_class)
    dw_snowfree = dw_year.map(lambda img: img.updateMask(img.neq(8))).mode().remap(*to_class)
    placeholder = ee.Image(0)
    return ee.Image.cat([placeholder, wc, dw, dw_snowfree]).rename(config.LABEL_BANDS).unmask(0).toUint8()


def write_consensus(name):
    """Fill band 1 (consensus) of a label file from its WorldCover/Dynamic World bands, save stats."""
    label_path = config.LABEL_DIR / f"{name}.tif"
    with rasterio.open(label_path, "r+") as dst:
        labels = dst.read()
        labels[0] = config.consensus_labels(labels[1], labels[2], labels[3] if len(labels) > 3 else None)
        dst.write(labels[0], 1)
    both = (labels[1] > 0) & (labels[2] > 0)
    kept = (labels[0] > 0).sum() / max(both.sum(), 1) * 100
    n_classes = len(config.CLASSES)
    stats = {"kept_pct": round(float(kept), 2),
             **{band: np.bincount(labels[i].ravel(), minlength=n_classes).tolist()
                for i, band in enumerate(config.LABEL_BANDS)}}
    (config.LABEL_DIR / f"{name}_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"{name}: {kept:.1f}% of pixels kept as training labels")


def fetch_chunk(image, bands, epsg, x0, y_top, width, height):
    request = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": width, "height": height},
            "affineTransform": {"scaleX": config.PIXEL_M, "shearX": 0, "translateX": x0,
                                "shearY": 0, "scaleY": -config.PIXEL_M, "translateY": y_top},
            "crsCode": f"EPSG:{epsg}",
        },
    }
    for attempt in range(5):
        try:
            arr = ee.data.computePixels(request)
            return np.stack([arr[b] for b in bands])
        except Exception as err:
            if attempt == 4:
                raise
            print(f"    retry after error: {err}")
            time.sleep(5 * (attempt + 1))


def download_raster(image, bands, dtype, fill, out_path, epsg, x0, y_top, n):
    data = np.full((len(bands), n, n), fill, dtype=dtype)
    offsets = [(r, c) for r in range(0, n, CHUNK) for c in range(0, n, CHUNK)]

    def job(rc):
        r, c = rc
        h, w = min(CHUNK, n - r), min(CHUNK, n - c)
        data[:, r:r + h, c:c + w] = fetch_chunk(
            image, bands, epsg, x0 + c * config.PIXEL_M, y_top - r * config.PIXEL_M, w, h)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(job, offsets), 1):
            print(f"    {out_path.parent.name}: chunk {i}/{len(offsets)}", end="\r")
    print()

    profile = dict(driver="GTiff", width=n, height=n, count=len(bands), dtype=dtype,
                   crs=f"EPSG:{epsg}", transform=from_origin(x0, y_top, config.PIXEL_M, config.PIXEL_M),
                   nodata=fill, compress="deflate",
                   tiled=True, blockxsize=256, blockysize=256)
    tmp_path = out_path.with_suffix(".tmp.tif")
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = tuple(bands)
        dst.update_tags(year=config.YEAR)
    tmp_path.replace(out_path)
    return data


def download_site(site):
    name = site["name"]
    s1_path = config.S1_DIR / f"{name}.tif"
    label_path = config.LABEL_DIR / f"{name}.tif"
    epsg, x0, y_top, n = site_grid(site)
    size_m = n * config.PIXEL_M
    region = ee.Geometry.Rectangle([x0, y_top - size_m, x0 + size_m, y_top], f"EPSG:{epsg}", False)

    if not s1_path.exists():
        counts = ee.Dictionary({str(m): month_collection(region, m).size() for m in config.MONTHS}).getInfo()
        print(f"{name}: S1 scenes per month {counts}")
        download_raster(s1_image(region), config.BAND_NAMES, "uint8", config.NODATA,
                        s1_path, epsg, x0, y_top, n)
        (config.S1_DIR / f"{name}_scene_counts.json").write_text(json.dumps(counts, indent=2))

    if not label_path.exists():
        download_raster(label_image(region), config.LABEL_BANDS, "uint8", 0,
                        label_path, epsg, x0, y_top, n)
        write_consensus(name)
    print(f"{name}: done")


def project_from_env_file():
    env_path = config.ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "Project_ID":
                return value.strip().strip("\"'")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.environ.get("EE_PROJECT") or project_from_env_file(),
                        help="Google Cloud project registered for Earth Engine "
                             "(default: EE_PROJECT env var or Project_ID in .env)")
    parser.add_argument("--sites", nargs="*", help="subset of site names (default: all)")
    parser.add_argument("--relabel", action="store_true",
                        help="only recompute consensus labels from existing label files (offline)")
    args = parser.parse_args()
    if args.relabel:
        for site in config.SITES:
            if (not args.sites or site["name"] in args.sites) and (config.LABEL_DIR / f"{site['name']}.tif").exists():
                write_consensus(site["name"])
        return
    if not args.project:
        parser.error("pass --project, set EE_PROJECT or add Project_ID to .env")

    for attempt in range(10):
        try:
            ee.Initialize(project=args.project)
            break
        except Exception as err:
            if attempt == 9:
                raise
            print(f"ee.Initialize failed ({err}); retrying")
            time.sleep(10)
    config.S1_DIR.mkdir(parents=True, exist_ok=True)
    config.LABEL_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for site in config.SITES:
        if not args.sites or site["name"] in args.sites:
            try:
                download_site(site)
            except Exception as err:
                print(f"{site['name']}: FAILED ({err}); rerun the script to retry")
                failed.append(site["name"])
    print(f"failed sites: {failed}" if failed else "all sites downloaded")


if __name__ == "__main__":
    main()
