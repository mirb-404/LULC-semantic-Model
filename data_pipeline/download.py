"""Download Sentinel-1 monthly composites and labels per site from Google Earth Engine (2nd iteration).

The Sentinel-1 processing lives in sentinel1.py, shared with the prototype.
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
Later years (4th iteration, training sites only, config.EXTRA_YEARS): <dataset>/s1_<year>/<site>.tif and
<dataset>/labels_<year>/<site>.tif (the same WorldCover 2021 label; Dynamic World of that year for the
agreement band). Up to config.MAX_EMPTY_MONTHS_EXTRA months without scenes are accepted (written as no data).
Sites with `need` (4th-iteration sites) must reach a minimum WorldCover share of a class before downloading.
Esri (5th iteration): <dataset>/esri[_<year>]/<site>.tif, the Esri 10 m Annual LULC of the same year mapped to the
target classes, the third map of the label vote (config.vote_labels, written by make_patches.py).

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
Usage: python data_pipeline/download.py [--project <gee-project-id>] [--sites a b] [--years 2021 2022]
                                        [--workers 4] [--terrain] [--relabel]
Datalab / remote machines: see data_pipeline/README.md.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from tqdm import tqdm

import config
import sentinel1

CHUNK = 512
AUTH_HINT = ("Earth Engine credentials missing or invalid. On a headless machine run\n"
             "  earthengine authenticate --auth_mode=notebook\n"
             "open the link, paste the code back, and do not tick 'read-only scopes'.")
PROJECT_HINT = ("The Cloud project cannot use Earth Engine (not registered, access expired, or API disabled).\n"
                "Use a project registered for Earth Engine: export EE_PROJECT=<id>, or pass --project <id>.\n"
                "The project Datalab provides is not registered.")


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


def s1_image(region, year):
    slope, aspect = sentinel1.slope_aspect(region)
    stack = ee.Image.cat([sentinel1.monthly_composite(region, year, m, slope, aspect)
                          .rename([f"VV_{m:02d}", f"VH_{m:02d}"]) for m in config.MONTHS])
    codes = stack.subtract(config.DB_MIN).divide(config.DB_STEP).round().clamp(1, 255)
    return codes.unmask(config.NODATA).toUint8()


def worldcover_classes():
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    return wc.remap(list(config.WORLDCOVER_TO_CLASS), list(config.WORLDCOVER_TO_CLASS.values()), 0)


def label_image(region, year):
    """WorldCover 2021 (the training label in every year) + Dynamic World of `year` (the agreement check)."""
    wc = worldcover_classes()
    dw_year = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
               .filterBounds(region)
               .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
               .select("label"))
    to_class = (list(config.DYNAMICWORLD_TO_CLASS), list(config.DYNAMICWORLD_TO_CLASS.values()), 0)
    dw = dw_year.mode().remap(*to_class)
    dw_snowfree = dw_year.map(lambda img: img.updateMask(img.neq(8))).mode().remap(*to_class)
    placeholder = ee.Image(0)
    return ee.Image.cat([placeholder, wc, dw, dw_snowfree]).rename(config.LABEL_BANDS).unmask(0).toUint8()


def esri_image(year):
    """Esri 10 m Annual LULC of `year` (2017-2025) mapped to the target classes (0 = clouds / no data)."""
    year = min(max(year, 2017), 2025)
    lulc = ee.ImageCollection(config.ESRI_COLLECTION).filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic()
    return lulc.remap(list(config.ESRI_TO_CLASS), list(config.ESRI_TO_CLASS.values()), 0).rename("esri").unmask(0).toUint8()


def terrain_image(region):
    dem = sentinel1.dem_image(region)
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


def write_consensus(name, year=config.YEAR):
    """Fill band 1 (consensus) of a label file from its WorldCover/Dynamic World bands, save stats."""
    label_path = config.label_dir(year) / f"{name}.tif"
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
    (config.label_dir(year) / f"{name}_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"{name} {year}: {kept:.1f}% of pixels where WorldCover and Dynamic World {year} agree")


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


def download_raster(image, bands, dtype, fill, out_path, epsg, x0, y_top, n, tags=None, year=config.YEAR):
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
        dst.update_tags(year=year, **(tags or {}))
    tmp_path.replace(out_path)
    print(f"  {out_path.parent.name}/{out_path.name} written")
    return data


S1_TAGS = {"preprocessing": "S1_GRD IW VV+VH asc+desc; gamma0 terrain flattening (Vollrath 2020 volume model, "
                            "GLO-30); layover/shadow/incidence 31-45 deg masked; monthly linear mean; dB; "
                            "no spatial filter"}


def download_site(site, terrain=False, year=config.YEAR):
    name = site["name"]
    s1_path = config.s1_dir(year) / f"{name}.tif"
    label_path = config.label_dir(year) / f"{name}.tif"
    terrain_path = config.TERRAIN_DIR / f"{name}.tif"
    epsg, x0, y_top, n = site_grid(site)
    size_m = n * config.PIXEL_M
    region = ee.Geometry.Rectangle([x0, y_top - size_m, x0 + size_m, y_top], f"EPSG:{epsg}", False)

    if year == config.YEAR and not s1_path.exists() and (site["focus"] == "mosaic" or site.get("need")):
        shares = worldcover_shares(region)
        if site["focus"] == "mosaic" and not config.is_mosaic(shares):
            raise SkipSite(f"not a mosaic, WorldCover class shares {shares}")
        short = {k: v for k, v in site.get("need", {}).items() if shares.get(k, 0) < v}
        if short:
            raise SkipSite(f"WorldCover shares {shares} below the required minimum {short}")
    if year != config.YEAR and not (config.s1_dir(config.YEAR) / f"{name}.tif").exists():
        raise SkipSite(f"no {config.YEAR} tile (later years only extend sites that exist in {config.YEAR})")

    if not s1_path.exists():
        counts = ee.Dictionary({str(m): sentinel1.month_collection(region, year, m).size()
                                for m in config.MONTHS}).getInfo()
        print(f"{name} {year}: S1 scenes per month {counts}")
        empty = sum(v == 0 for v in counts.values())
        # val/test tiles must be complete (comparable scores); training tiles may miss a few months, which the
        # model already handles (month dropout): Africa and later years often have a month without a scene
        allowed = 0 if site["split"] != "train" else config.MAX_EMPTY_MONTHS_EXTRA
        if empty > allowed:
            raise SkipSite(f"{year}: {empty} months without Sentinel-1 IW scenes (allowed {allowed}): {counts}")
        download_raster(s1_image(region, year), config.BAND_NAMES, "uint8", config.NODATA,
                        s1_path, epsg, x0, y_top, n, S1_TAGS, year)
        (config.s1_dir(year) / f"{name}_scene_counts.json").write_text(json.dumps(counts, indent=2))

    if not label_path.exists():
        download_raster(label_image(region, year), config.LABEL_BANDS, "uint8", 0,
                        label_path, epsg, x0, y_top, n, {"label_year": 2021, "agreement_year": year}, year)
        write_consensus(name, year)

    esri_path = config.esri_dir(year) / f"{name}.tif"
    if not esri_path.exists():  # third map for the label vote (5th iteration)
        download_raster(esri_image(year), ["esri"], "uint8", 0, esri_path, epsg, x0, y_top, n,
                        {"source": config.ESRI_COLLECTION}, year)

    if terrain and year == config.YEAR and not terrain_path.exists():
        download_raster(terrain_image(region), config.TERRAIN_BANDS, "int16", 0,
                        terrain_path, epsg, x0, y_top, n)
    print(f"{name} {year}: done")


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
            if any(s in text for s in ("access has expired", "not registered", "has not been used in project",
                                       "Earth Engine API has not been", "PERMISSION_DENIED")):
                raise SystemExit(f"project {project}: {text}\n\n{PROJECT_HINT}")
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
    parser.add_argument("--years", nargs="*", type=int,
                        help=f"subset of years (default: {config.YEAR} for every site, plus "
                             f"{', '.join(map(str, config.EXTRA_YEARS))} for training sites)")
    parser.add_argument("--workers", type=int, default=4,
                        help="tiles downloaded in parallel (each uses 4 requests at a time)")
    parser.add_argument("--terrain", action="store_true", help="also download the terrain layers")
    parser.add_argument("--relabel", action="store_true",
                        help="only recompute consensus labels from existing label files (offline)")
    args = parser.parse_args()
    sites = [s for s in config.SITES if not args.sites or s["name"] in args.sites]
    tiles = [(s, y) for s in sites for y in config.years_for(s) if not args.years or y in args.years]
    print(f"dataset: {config.DATA_DIR} | {len(sites)} sites, {len(tiles)} site-years")
    for train, held, km in config.leakage():
        print(f"warning: train site {train} is {km} km from val/test site {held}")
    if args.relabel:
        for site, year in tiles:
            if (config.label_dir(year) / f"{site['name']}.tif").exists():
                write_consensus(site["name"], year)
        return
    if not args.project:
        parser.error("pass --project, set EE_PROJECT or add Project_ID to .env")

    initialize(args.project)

    def folders(year):
        return ((config.s1_dir(year), config.label_dir(year), config.esri_dir(year))
                + ((config.TERRAIN_DIR,) if args.terrain and year == config.YEAR else ()))

    for year in sorted({y for _, y in tiles}):
        for folder in folders(year):
            folder.mkdir(parents=True, exist_ok=True)
    todo = [(s, y) for s, y in tiles if not all((f / f"{s['name']}.tif").exists() for f in folders(y))]
    print(f"{len(tiles) - len(todo)} site-years already downloaded, {len(todo)} to go")
    failed, skipped = [], {}
    for year in sorted({y for _, y in todo}):  # one year after the other: 2021 is complete before later years
        batch = [s for s, y in todo if y == year]
        with ThreadPoolExecutor(max_workers=args.workers) as pool, \
                tqdm(total=len(batch), unit="site", desc=str(year), dynamic_ncols=True) as bar:
            jobs = {pool.submit(download_site, site, args.terrain, year): site["name"] for site in batch}
            for job in as_completed(jobs):
                key = jobs[job] if year == config.YEAR else f"{jobs[job]}_{year}"
                try:
                    job.result()
                except SkipSite as reason:
                    tqdm.write(f"{key}: SKIPPED ({reason})")
                    skipped[key] = str(reason)
                except Exception as err:
                    tqdm.write(f"{key}: FAILED ({err}); rerun the script to retry")
                    failed.append(key)
                bar.update(1)
    skipped_path = config.DATA_DIR / "skipped_sites.json"
    earlier = json.loads(skipped_path.read_text()) if skipped_path.exists() else {}
    retried = {s["name"] if y == config.YEAR else f"{s['name']}_{y}" for s, y in todo}
    earlier = {k: v for k, v in earlier.items() if k not in retried}
    skipped_path.write_text(json.dumps({**earlier, **skipped}, indent=2))
    print(f"skipped (see skipped_sites.json): {sorted(skipped)}" if skipped else "nothing skipped")
    print(f"failed: {sorted(failed)}" if failed else "everything else downloaded")
    if failed:
        raise SystemExit(1)  # build_dataset.py reruns the download until nothing fails


if __name__ == "__main__":
    main()
