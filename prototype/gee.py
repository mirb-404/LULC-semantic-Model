"""Google Earth Engine retrieval for the prototype.

The Sentinel-1 processing must match the training data of the model in use, generalised from the
fixed year 2021 to any (year, month); uint8 codes dB = code * 0.2 - 35 (0 = no data):
  2nd iteration: data_pipeline/sentinel1.py itself (terrain-flattened gamma0, layover / shadow masked,
                 monthly mean, no spatial filter), so training and prototype cannot drift apart.
  1st iteration: sigma0, incidence angle 31-45 deg, monthly mean in linear power, dB, 3x3 focal median
                 (the 1st-iteration data_pipeline/download.py, commit 40e1280).
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import numpy as np

import mapview

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_pipeline"))
import sentinel1  # noqa: E402
import config as pipeline_config  # noqa: E402  (class mappings of data_pipeline/config.py)

PIXEL_M = 10
DB_MIN, DB_STEP, NODATA = -35.0, 0.2, 0
CHUNK = 512
TERRAIN_BANDS = ["elevation_m", "slope_decideg", "hand_dm"]

# Esri 10 m Annual LULC (Impact Observatory, Sentinel-2 based): independent of the training labels.
ESRI_COLLECTION = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
ESRI_YEARS = (2017, 2025)
# Esri classes: 1 water, 2 trees, 4 flooded vegetation, 5 crops, 7 built area, 8 bare ground, 9 snow/ice,
# 10 clouds, 11 rangeland. Mapped to our class ids per iteration; clouds and no data -> 0.
ESRI_TO_CLASS = {
    # 1 Built-up, 2 Cropland, 3 Trees, 4 Shrub & grass, 5 Bare, 6 Water, 7 Wetland, 8 Snow
    1: {1: 6, 2: 3, 4: 7, 5: 2, 7: 1, 8: 5, 9: 8, 11: 4},
    # 1 Man-made, 2 Crop fields, 3 Trees, 4 Grassland, 5 Bare ground, 6 Water, 7 Snow & ice
    2: {1: 6, 2: 3, 4: 4, 5: 2, 7: 1, 8: 5, 9: 7, 11: 4},
}


def default_project():
    """Earth Engine project: EE_PROJECT, else Project_ID in the repository's .env (git-ignored), else ""."""
    if os.environ.get("EE_PROJECT"):
        return os.environ["EE_PROJECT"]
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "Project_ID":
                return value.strip().strip("\"'")
    return ""


def initialize(project):
    ee.Initialize(project=project)


def grid(lat, lon, n):
    """UTM grid of n x n pixels (10 m) centred on lat/lon, snapped to the pixel size."""
    epsg, x0, y_top = mapview.utm_window(lat, lon, n)
    region = ee.Geometry.Rectangle([x0, y_top - n * PIXEL_M, x0 + n * PIXEL_M, y_top], f"EPSG:{epsg}", False)
    return epsg, x0, y_top, region


def mask_edges(img):
    angle = img.select("angle")
    return img.updateMask(angle.gt(31).And(angle.lt(45))).select(["VV", "VH"])


def monthly_composite_sigma0(region, year, month):
    """1st-iteration composite: sigma0, monthly mean, 3x3 focal median."""
    linear = sentinel1.month_collection(region, year, month).map(
        lambda img: ee.Image(10).pow(mask_edges(img).divide(10)))
    db = linear.mean().log10().multiply(10)
    return db.focalMedian(radius=1, kernelType="square", units="pixels")


def s1_stack(region, months, iteration):
    """uint8 code image for a list of (year, month), bands VV/VH per month in list order."""
    if iteration >= 2:
        slope, aspect = sentinel1.slope_aspect(region)
        composites = [sentinel1.monthly_composite(region, y, m, slope, aspect) for y, m in months]
    else:
        composites = [monthly_composite_sigma0(region, y, m) for y, m in months]
    stack = ee.Image.cat([c.rename([f"VV_{y}_{m:02d}", f"VH_{y}_{m:02d}"])
                          for c, (y, m) in zip(composites, months)])
    codes = stack.subtract(DB_MIN).divide(DB_STEP).round().clamp(1, 255)
    bands = [f"{pol}_{y}_{m:02d}" for y, m in months for pol in ("VV", "VH")]
    return codes.unmask(NODATA).toUint8(), bands


def scene_counts(region, months):
    counts = ee.Dictionary({f"{y}-{m:02d}": sentinel1.month_collection(region, y, m).size()
                            for y, m in months}).getInfo()
    return [counts[f"{y}-{m:02d}"] for y, m in months]


def terrain_image(region):
    dem = sentinel1.dem_image(region)
    slope = ee.Terrain.slope(dem).multiply(10)
    hand = ee.Image("MERIT/Hydro/v1_0_1").select("hnd").multiply(10)
    stack = ee.Image.cat([dem, slope, hand]).resample("bilinear")
    return stack.round().clamp(-32767, 32767).rename(TERRAIN_BANDS).unmask(0).toInt16()


def esri_image(year, iteration):
    year = min(max(year, ESRI_YEARS[0]), ESRI_YEARS[1])
    lulc = ee.ImageCollection(ESRI_COLLECTION).filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic()
    mapping = ESRI_TO_CLASS[min(iteration, 2)]
    return lulc.remap(list(mapping), list(mapping.values()), 0).rename("esri").unmask(0).toUint8(), year


def worldcover_image():
    """ESA WorldCover 2021 (the training label of the 2nd iteration) mapped to the 7 classes of the 2nd iteration."""
    m = pipeline_config.WORLDCOVER_TO_CLASS
    return (ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
            .remap(list(m), list(m.values()), 0).rename("cls").unmask(0).toUint8())


def dynamicworld_image(region, year):
    """Google Dynamic World: the most frequent class of `year` (Sentinel-2 based), mapped to the 7 classes."""
    m = pipeline_config.DYNAMICWORLD_TO_CLASS
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region)
          .filterDate(f"{year}-01-01", f"{year + 1}-01-01").select("label").mode())
    return dw.remap(list(m), list(m.values()), 0).rename("cls").unmask(0).toUint8()


S2_BANDS = ["vis-red", "vis-green", "vis-blue"]


def sentinel2_rgb(region, start, end):
    """Cloud-free Sentinel-2 true colour of [start, end) as 8-bit RGB (bands S2_BANDS), for display only.

    Median of the clear pixels (Cloud Score+ cs_cdf >= 0.6) of Sentinel-2 L1C, so it covers 2015 onwards and
    shows the land in the same period as the radar input. Not a model input.
    """
    clear = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(region).filterDate(start, end)
          .linkCollection(clear, ["cs_cdf"])
          .map(lambda img: img.updateMask(img.select("cs_cdf").gte(0.6))))
    rgb = s2.select(["B4", "B3", "B2"]).median().visualize(min=0, max=3000, gamma=1.3)
    return rgb.unmask(0).toUint8()


def fetch(image, bands, epsg, x0, y_top, n, dtype, fill):
    """Download an n x n pixel window as (bands, n, n), in 512 px chunks with retries."""
    data = np.full((len(bands), n, n), fill, dtype=dtype)

    def job(rc):
        r, c = rc
        h, w = min(CHUNK, n - r), min(CHUNK, n - c)
        request = {
            "expression": image,
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {"width": w, "height": h},
                "affineTransform": {"scaleX": PIXEL_M, "shearX": 0, "translateX": x0 + c * PIXEL_M,
                                    "shearY": 0, "scaleY": -PIXEL_M, "translateY": y_top - r * PIXEL_M},
                "crsCode": f"EPSG:{epsg}",
            },
        }
        for attempt in range(4):
            try:
                arr = ee.data.computePixels(request)
                data[:, r:r + h, c:c + w] = np.stack([arr[b] for b in bands])
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(job, [(r, c) for r in range(0, n, CHUNK) for c in range(0, n, CHUNK)]))
    return data
