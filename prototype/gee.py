"""Google Earth Engine retrieval for the prototype.

The Sentinel-1 processing is identical to data_pipeline/download.py (the training data), only
generalised from the fixed year 2021 to any (year, month):
  IW scenes with VV and VH, both orbit directions; incidence angle 31-45 deg; monthly mean in
  linear power; back to dB; 3x3 focal median; uint8 codes dB = code * 0.2 - 35 (0 = no data).
Earth Engine's COPERNICUS/S1_GRD is already calibrated, terrain-corrected and in dB.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np
from pyproj import Transformer

PIXEL_M = 10
DB_MIN, DB_STEP, NODATA = -35.0, 0.2, 0
CHUNK = 512
TERRAIN_BANDS = ["elevation_m", "slope_decideg", "hand_dm"]

# Esri 10 m Annual LULC (Impact Observatory, Sentinel-2 based): independent of the training labels.
ESRI_COLLECTION = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
ESRI_YEARS = (2017, 2024)
# Esri class -> our class id (1 Built-up, 2 Cropland, 3 Trees, 4 Shrub & grass, 5 Bare, 6 Water,
# 7 Wetland, 8 Snow); clouds (10) and no data -> 0
ESRI_TO_CLASS = {1: 6, 2: 3, 4: 7, 5: 2, 7: 1, 8: 5, 9: 8, 11: 4}


def initialize(project):
    ee.Initialize(project=project)


def utm_epsg(lat, lon):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def grid(lat, lon, n):
    """UTM grid of n x n pixels (10 m) centred on lat/lon, snapped to the pixel size."""
    epsg = utm_epsg(lat, lon)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    half = n * PIXEL_M / 2
    x0 = round((x - half) / PIXEL_M) * PIXEL_M
    y_top = round((y + half) / PIXEL_M) * PIXEL_M
    region = ee.Geometry.Rectangle([x0, y_top - n * PIXEL_M, x0 + n * PIXEL_M, y_top], f"EPSG:{epsg}", False)
    return epsg, x0, y_top, region


def mask_edges(img):
    angle = img.select("angle")
    return img.updateMask(angle.gt(31).And(angle.lt(45))).select(["VV", "VH"])


def month_collection(region, year, month):
    start = ee.Date.fromYMD(year, month, 1)
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate(start, start.advance(1, "month"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))


def monthly_composite(region, year, month):
    linear = month_collection(region, year, month).map(
        lambda img: ee.Image(10).pow(mask_edges(img).divide(10)))
    db = linear.mean().log10().multiply(10)
    db = db.focalMedian(radius=1, kernelType="square", units="pixels")
    return db.rename([f"VV_{year}_{month:02d}", f"VH_{year}_{month:02d}"])


def s1_stack(region, months):
    """uint8 code image for a list of (year, month), bands VV/VH per month in list order."""
    stack = ee.Image.cat([monthly_composite(region, y, m) for y, m in months])
    codes = stack.subtract(DB_MIN).divide(DB_STEP).round().clamp(1, 255)
    bands = [f"{pol}_{y}_{m:02d}" for y, m in months for pol in ("VV", "VH")]
    return codes.unmask(NODATA).toUint8(), bands


def scene_counts(region, months):
    counts = ee.Dictionary({f"{y}-{m:02d}": month_collection(region, y, m).size() for y, m in months}).getInfo()
    return [counts[f"{y}-{m:02d}"] for y, m in months]


def terrain_image(region):
    dems = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").filterBounds(region).select("DEM")
    dem = dems.mosaic().setDefaultProjection(dems.first().projection())
    slope = ee.Terrain.slope(dem).multiply(10)
    hand = ee.Image("MERIT/Hydro/v1_0_1").select("hnd").multiply(10)
    stack = ee.Image.cat([dem, slope, hand]).resample("bilinear")
    return stack.round().clamp(-32767, 32767).rename(TERRAIN_BANDS).unmask(0).toInt16()


def esri_image(year):
    year = min(max(year, ESRI_YEARS[0]), ESRI_YEARS[1])
    lulc = ee.ImageCollection(ESRI_COLLECTION).filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic()
    return lulc.remap(list(ESRI_TO_CLASS), list(ESRI_TO_CLASS.values()), 0).rename("esri").unmask(0).toUint8(), year


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
