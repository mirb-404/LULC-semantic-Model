"""Download Sentinel-1 monthly composites and labels per site from Google Earth Engine (2nd iteration).

Sentinel-1 (COPERNICUS/S1_GRD) is already processed by Earth Engine with the S1 Toolbox:
orbit file, GRD border noise removal, thermal noise removal, radiometric calibration
(sigma0), geometric terrain correction (SRTM 30 m / ASTER DEM) and dB conversion. On top of that:
  1. keep IW scenes with both VV and VH, ascending and descending passes,
  2. radiometric terrain flattening: sigma0 -> gamma0 with the angular-based volume model of
     Vollrath et al. 2020 (Remote Sensing 12(11) 1867) on the Copernicus GLO-30 DEM, so slopes facing
     the sensor no longer look like buildings and slopes facing away no longer look like water,
  3. mask layover, radar shadow and near/far-range edges (incidence angle outside 31-45 deg),
  4. average each month in linear power (temporal multilooking reduces speckle),
  5. convert back to dB. No spatial speckle filter: the 3x3 focal median of the 1st iteration
     blurred field edges, roads and streams, and the monthly mean already reduces speckle.
Output: <dataset>/s1/<site>.tif, 24 uint8 bands (VV_01, VH_01, ..., VH_12), see config.decode_db.

Labels: ESA WorldCover 2021, the 2021 per-pixel mode of Dynamic World and its mode over
snow-free observations are mapped to the target classes and sampled with nearest neighbour onto
the exact S1 grid. The consensus band is computed locally by config.consensus_labels; the 2nd
iteration trains on WorldCover (config.TRAIN_LABEL_BAND) and uses the consensus band to mark
the pixels where both maps agree.
`--relabel` recomputes the consensus band and stats from existing label files, offline.
Output: <dataset>/labels/<site>.tif, 4 uint8 bands (consensus, worldcover, dynamicworld,
dynamicworld_snowfree).

Mosaic sites (focus "mosaic") are checked before downloading: WorldCover must have
a largest class of at most config.MOSAIC_MAX_SHARE and a second
class of at least config.MOSAIC_MIN_SECOND. Tiles that fail are skipped
and listed in <dataset>/skipped_sites.json; move their coordinates and rerun.

Terrain (optional, `--terrain`): Copernicus GLO-30 DEM elevation in m, slope in 0.1 deg and MERIT
Hydro height above nearest drainage (HAND) in 0.1 m, bilinear onto the S1 grid. No data (open sea) is 0.
Output: <dataset>/terrain/<site>.tif, 3 int16 bands (elevation_m, slope_decideg, hand_dm).

Rerunning skips files that exist, so an interrupted run continues where it stopped.
Usage: python data_pipeline/download.py [--project <gee-project-id>] [--sites a b] [--workers 4]
                                        [--terrain] [--relabel]
Datalab / remote machines: see data_pipeline/README.md.
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

import config

CHUNK = 512
DEG = math.pi / 180
AUTH_HINT = ("Earth Engine credentials missing or invalid. On a headless machine run\n"
             "  earthengine authenticate --auth_mode=notebook\n"
             "open the link, paste the code back, and do not tick 'read-only scopes'.")


class SkipSite(Exception):
    pass


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


def dem_image(region):
    dems = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").filterBounds(region).select("DEM")
    return dems.mosaic().setDefaultProjection(dems.first().projection())


def flatten_terrain(img, slope, aspect):
    """sigma0 (dB) scene -> terrain-flattened gamma0 in linear power, with layover, shadow and edges masked.

    Volume model of Vollrath et al. 2020. Both the look direction (aspect of the incidence-angle
    field, which rises away from the sensor) and the terrain aspect come from ee.Terrain.aspect, so
    their difference is 0 for a slope facing the sensor and alpha_r is its slope angle in range.
    """
    angle = img.select("angle")
    theta = angle.multiply(DEG)
    heading = ee.Terrain.aspect(angle).reduceRegion(ee.Reducer.mean(), img.geometry(), 1000).get("aspect")
    phi_r = ee.Image.constant(heading).multiply(DEG).subtract(aspect)
    alpha_r = slope.tan().multiply(phi_r.cos()).atan()
    ninety = ee.Image.constant(math.pi / 2)
    scf = ninety.subtract(theta).add(alpha_r).tan().divide(ninety.subtract(theta).tan())
    sigma0 = ee.Image(10).pow(img.select(["VV", "VH"]).divide(10))
    gamma0 = sigma0.divide(theta.cos()).divide(scf)
    valid = (angle.gt(31).And(angle.lt(45))
             .And(alpha_r.lt(theta))
             .And(alpha_r.gt(theta.subtract(ninety))))
    return gamma0.updateMask(valid)


def month_collection(region, month):
    start = ee.Date.fromYMD(config.YEAR, month, 1)
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate(start, start.advance(1, "month"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))


def monthly_composite(region, month, slope, aspect):
    linear = month_collection(region, month).map(lambda img: flatten_terrain(img, slope, aspect))
    db = linear.mean().log10().multiply(10)
    return db.rename([f"VV_{month:02d}", f"VH_{month:02d}"])


def s1_image(region):
    dem = dem_image(region)
    slope = ee.Terrain.slope(dem).multiply(DEG)
    aspect = ee.Terrain.aspect(dem).multiply(DEG)
    stack = ee.Image.cat([monthly_composite(region, m, slope, aspect) for m in config.MONTHS])
    codes = stack.subtract(config.DB_MIN).divide(config.DB_STEP).round().clamp(1, 255)
    return codes.unmask(config.NODATA).toUint8()


def worldcover_classes():
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    return wc.remap(list(config.WORLDCOVER_TO_CLASS), list(config.WORLDCOVER_TO_CLASS.values()), 0)


def label_image(region):
    wc = worldcover_classes()
    dw_year = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
               .filterBounds(region)
               .filterDate(f"{config.YEAR}-01-01", f"{config.YEAR + 1}-01-01")
               .select("label"))
    to_class = (list(config.DYNAMICWORLD_TO_CLASS), list(config.DYNAMICWORLD_TO_CLASS.values()), 0)
    dw = dw_year.mode().remap(*to_class)
    dw_snowfree = dw_year.map(lambda img: img.updateMask(img.neq(8))).mode().remap(*to_class)
    placeholder = ee.Image(0)
    return ee.Image.cat([placeholder, wc, dw, dw_snowfree]).rename(config.LABEL_BANDS).unmask(0).toUint8()


def terrain_image(region):
    dem = dem_image(region)
    slope = ee.Terrain.slope(dem).multiply(10)
    hand = ee.Image("MERIT/Hydro/v1_0_1").select("hnd").multiply(10)
    stack = ee.Image.cat([dem, slope, hand]).resample("bilinear")
    return stack.round().clamp(-32767, 32767).rename(config.TERRAIN_BANDS).unmask(0).toInt16()


def worldcover_shares(region):
    """{class id: share} of the WorldCover pixels in the tile (20 m sampling is enough for a check)."""
    hist = (worldcover_classes().rename("cls")
            .reduceRegion(ee.Reducer.frequencyHistogram(), region, scale=20, maxPixels=1e8)
            .get("cls").getInfo()) or {}
    hist = {int(k): v for k, v in hist.items() if int(k) > 0}
    total = max(sum(hist.values()), 1)
    return {k: round(v / total, 4) for k, v in sorted(hist.items())}


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
    print(f"{name}: {kept:.1f}% of pixels where WorldCover and Dynamic World agree")


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


def download_raster(image, bands, dtype, fill, out_path, epsg, x0, y_top, n, tags=None):
    data = np.full((len(bands), n, n), fill, dtype=dtype)
    offsets = [(r, c) for r in range(0, n, CHUNK) for c in range(0, n, CHUNK)]

    def job(rc):
        r, c = rc
        h, w = min(CHUNK, n - r), min(CHUNK, n - c)
        data[:, r:r + h, c:c + w] = fetch_chunk(
            image, bands, epsg, x0 + c * config.PIXEL_M, y_top - r * config.PIXEL_M, w, h)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(job, offsets))

    profile = dict(driver="GTiff", width=n, height=n, count=len(bands), dtype=dtype,
                   crs=f"EPSG:{epsg}", transform=from_origin(x0, y_top, config.PIXEL_M, config.PIXEL_M),
                   nodata=fill, compress="deflate",
                   tiled=True, blockxsize=256, blockysize=256)
    tmp_path = out_path.with_suffix(".tmp.tif")
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = tuple(bands)
        dst.update_tags(year=config.YEAR, **(tags or {}))
    tmp_path.replace(out_path)
    print(f"  {out_path.parent.name}/{out_path.name} written")
    return data


S1_TAGS = {"preprocessing": "S1_GRD IW VV+VH asc+desc; gamma0 terrain flattening (Vollrath 2020 volume model, "
                            "GLO-30); layover/shadow/incidence 31-45 deg masked; monthly linear mean; dB; "
                            "no spatial filter"}


def download_site(site, terrain=False):
    name = site["name"]
    s1_path = config.S1_DIR / f"{name}.tif"
    label_path = config.LABEL_DIR / f"{name}.tif"
    terrain_path = config.TERRAIN_DIR / f"{name}.tif"
    epsg, x0, y_top, n = site_grid(site)
    size_m = n * config.PIXEL_M
    region = ee.Geometry.Rectangle([x0, y_top - size_m, x0 + size_m, y_top], f"EPSG:{epsg}", False)

    if site["focus"] == "mosaic" and not s1_path.exists():
        shares = worldcover_shares(region)
        if not config.is_mosaic(shares):
            raise SkipSite(f"not a mosaic, WorldCover class shares {shares}")

    if not s1_path.exists():
        counts = ee.Dictionary({str(m): month_collection(region, m).size() for m in config.MONTHS}).getInfo()
        print(f"{name}: S1 scenes per month {counts}")
        if min(counts.values()) == 0:
            raise SkipSite(f"months without Sentinel-1 IW scenes: {counts}")
        download_raster(s1_image(region), config.BAND_NAMES, "uint8", config.NODATA,
                        s1_path, epsg, x0, y_top, n, S1_TAGS)
        (config.S1_DIR / f"{name}_scene_counts.json").write_text(json.dumps(counts, indent=2))

    if not label_path.exists():
        download_raster(label_image(region), config.LABEL_BANDS, "uint8", 0,
                        label_path, epsg, x0, y_top, n)
        write_consensus(name)

    if terrain and not terrain_path.exists():
        download_raster(terrain_image(region), config.TERRAIN_BANDS, "int16", 0,
                        terrain_path, epsg, x0, y_top, n)
    print(f"{name}: done")


def project_from_env_file():
    env_path = config.ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "Project_ID":
                return value.strip().strip("\"'")
    return None


def initialize(project):
    for attempt in range(10):
        try:
            ee.Initialize(project=project)
            ee.Number(1).getInfo()
            return
        except Exception as err:
            text = str(err)
            if any(s in text for s in ("authorize", "credentials", "invalid_scope", "invalid_grant")):
                raise SystemExit(f"{text}\n\n{AUTH_HINT}")
            if attempt == 9:
                raise
            print(f"ee.Initialize failed ({err}); retrying")
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.environ.get("EE_PROJECT") or project_from_env_file(),
                        help="Google Cloud project registered for Earth Engine "
                             "(default: EE_PROJECT env var or Project_ID in .env)")
    parser.add_argument("--sites", nargs="*", help="subset of site names (default: all)")
    parser.add_argument("--workers", type=int, default=4,
                        help="sites downloaded in parallel (each uses 4 requests at a time)")
    parser.add_argument("--terrain", action="store_true", help="also download the terrain layers")
    parser.add_argument("--relabel", action="store_true",
                        help="only recompute consensus labels from existing label files (offline)")
    args = parser.parse_args()
    sites = [s for s in config.SITES if not args.sites or s["name"] in args.sites]
    print(f"dataset: {config.DATA_DIR} | {len(sites)} sites")
    if args.relabel:
        for site in sites:
            if (config.LABEL_DIR / f"{site['name']}.tif").exists():
                write_consensus(site["name"])
        return
    if not args.project:
        parser.error("pass --project, set EE_PROJECT or add Project_ID to .env")

    initialize(args.project)
    for folder in (config.S1_DIR, config.LABEL_DIR) + ((config.TERRAIN_DIR,) if args.terrain else ()):
        folder.mkdir(parents=True, exist_ok=True)
    failed, skipped = [], {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(download_site, site, args.terrain): site["name"] for site in sites}
        for done, job in enumerate(as_completed(jobs), 1):
            name = jobs[job]
            try:
                job.result()
            except SkipSite as reason:
                print(f"{name}: SKIPPED ({reason})")
                skipped[name] = str(reason)
            except Exception as err:
                print(f"{name}: FAILED ({err}); rerun the script to retry")
                failed.append(name)
            print(f"[{done}/{len(sites)} sites]")
    skipped_path = config.DATA_DIR / "skipped_sites.json"
    earlier = json.loads(skipped_path.read_text()) if skipped_path.exists() else {}
    earlier = {k: v for k, v in earlier.items() if k not in jobs.values()}
    skipped_path.write_text(json.dumps({**earlier, **skipped}, indent=2))
    print(f"skipped sites (see skipped_sites.json): {sorted(skipped)}" if skipped else "no sites skipped")
    print(f"failed sites: {sorted(failed)}" if failed else "all other sites downloaded")


if __name__ == "__main__":
    main()
