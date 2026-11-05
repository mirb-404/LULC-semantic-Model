"""One command for the whole dataset: download, patch index, previews, manifest and zip archives.

Steps (in this order; choose with --steps, default: download patches manifest zip):
  download   download.py: Sentinel-1 composites and labels from Earth Engine for every site and year
             (2021 for all sites, config.EXTRA_YEARS for training sites). Rerun automatically up to
             --download-attempts times while tiles fail; finished files are always skipped.
  patches    make_patches.py: patch index per split (+ train_extra_patches.csv), normalisation, class shares.
  previews   preview.py: quick-look PNGs per site (optional, not in the default steps).
  manifest   <dataset>/dataset_manifest.json: build date, tiles per year, patches per split, skipped sites,
             folder sizes, the settings of config.py.
  zip        <dataset>.zip (everything) and <dataset>_2021.zip (the 2021 part that Iter2_Test_1_Full and the
             2nd/3rd-iteration notebooks use) next to the dataset folder, for the submission link. GeoTIFFs are
             already compressed, so files are stored without recompression (fast).

Usage (from the repository root; Earth Engine login and project as in data_pipeline/README.md):
  nohup python data_pipeline/build_dataset.py --workers 8 > build.log 2>&1 &
  python data_pipeline/build_dataset.py --steps patches manifest zip      # after a finished download
  python data_pipeline/build_dataset.py --years 2021 2022 --steps download patches
"""
import argparse
import json
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import config

HERE = Path(__file__).resolve().parent
STEPS = ("download", "patches", "previews", "manifest", "zip")
DEFAULT_STEPS = ("download", "patches", "manifest", "zip")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run_script(name, args=()):
    """Runs one pipeline script with this Python; returns its exit code (output goes to this log)."""
    cmd = [sys.executable, str(HERE / name), *map(str, args)]
    log("$ " + " ".join(cmd[1:]))
    return subprocess.run(cmd, cwd=config.ROOT).returncode


def download(args):
    extra = ["--workers", args.workers]
    if args.project:
        extra += ["--project", args.project]
    if args.sites:
        extra += ["--sites", *args.sites]
    if args.years:
        extra += ["--years", *args.years]
    if args.terrain:
        extra += ["--terrain"]
    for attempt in range(1, args.download_attempts + 1):
        code = run_script("download.py", extra)
        if code == 0:
            return
        if attempt < args.download_attempts:
            log(f"download: some tiles failed (attempt {attempt}/{args.download_attempts}); retrying in 60 s")
            time.sleep(60)
    raise SystemExit("download: tiles still failing; check the log, then rerun (finished files are kept)")


def folder_size_gb(path):
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9, 3)


def manifest():
    data = config.DATA_DIR
    tiles = {}
    for year in (config.YEAR, *config.EXTRA_YEARS):
        folder = config.s1_dir(year)
        tiles[str(year)] = len([p for p in folder.glob("*.tif") if not p.name.endswith(".tmp.tif")]) \
            if folder.exists() else 0
    patches = {}
    for name in ("train", "val", "test", "train_extra"):
        path = config.PATCH_DIR / f"{name}_patches.csv"
        patches[name] = max(len(path.read_text().splitlines()) - 1, 0) if path.exists() else None
    skipped_path = data / "skipped_sites.json"
    skipped = json.loads(skipped_path.read_text()) if skipped_path.exists() else {}
    sizes = {p.name: folder_size_gb(p) for p in sorted(data.iterdir()) if p.is_dir()}
    out = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": data.name,
        "sites": len(config.SITES),
        "sites_per_split": dict(Counter(s["split"] for s in config.SITES)),
        "label": "ESA WorldCover 2021 (all years); Dynamic World of the same year for the agreement band",
        "radar_years": {"all sites": config.YEAR, "training sites": list(config.EXTRA_YEARS),
                        "no later years for": sorted(config.EXTRA_YEARS_SKIP_FOCUS)},
        "tiles_per_year": tiles,
        "esri_tiles_per_year": {str(y): len(list(config.esri_dir(y).glob("*.tif"))) if config.esri_dir(y).exists() else 0
                                for y in (config.YEAR, *config.EXTRA_YEARS)},
        "label_vote": "votes[_<year>]/: class where at least 2 of WorldCover, Dynamic World, Esri agree (config.vote_labels)",
        "patches": patches,
        "skipped": skipped,
        "size_gb": {**sizes, "total": folder_size_gb(data)},
        "tile_size_px": config.TILE_SIZE_M // config.PIXEL_M,
        "pixel_m": config.PIXEL_M,
        "patch_px": config.PATCH,
        "radar_decoding": {"db_min": config.DB_MIN, "db_step": config.DB_STEP, "nodata": config.NODATA},
        "classes": {k: v[0] for k, v in config.CLASSES.items() if k},
        "train_val_test_leakage_under_km": [list(x) for x in config.leakage()],
    }
    path = data / "dataset_manifest.json"
    path.write_text(json.dumps(out, indent=2))
    log(f"manifest: {path} | tiles per year {tiles} | patches {patches} | {out['size_gb']['total']} GB")


def write_zip(target, files):
    """Stores files (already compressed GeoTIFFs and small text files) under the dataset folder name."""
    tmp = target.with_suffix(".zip.tmp")
    total = len(files)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for i, f in enumerate(files, 1):
            zf.write(f, f.relative_to(config.DATA_DIR.parent))
            if i % 200 == 0 or i == total:
                log(f"  {target.name}: {i}/{total} files")
    tmp.replace(target)
    log(f"zip: {target} ({target.stat().st_size / 1e9:.2f} GB)")


def make_zips():
    data = config.DATA_DIR
    every = sorted(f for f in data.rglob("*") if f.is_file() and not f.name.endswith(".tmp.tif"))
    first_year = {config.S1_DIR.name, config.LABEL_DIR.name, config.PATCH_DIR.name, config.TERRAIN_DIR.name,
                  config.esri_dir(config.YEAR).name, config.vote_dir(config.YEAR).name}
    only_2021 = [f for f in every if f.relative_to(data).parts[0] in first_year or f.parent == data]
    only_2021 = [f for f in only_2021 if f.name != "train_extra_patches.csv"]
    write_zip(data.parent / f"{data.name}_2021.zip", only_2021)
    if any(config.s1_dir(y).exists() for y in config.EXTRA_YEARS):
        write_zip(data.parent / f"{data.name}.zip", every)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="*", choices=STEPS, default=list(DEFAULT_STEPS))
    parser.add_argument("--project", help="Earth Engine project (default: EE_PROJECT or Project_ID in .env)")
    parser.add_argument("--sites", nargs="*", help="subset of site names (download and previews)")
    parser.add_argument("--years", nargs="*", help="subset of years for the download")
    parser.add_argument("--workers", type=int, default=8, help="tiles downloaded in parallel (4 requests each)")
    parser.add_argument("--terrain", action="store_true", help="also download the terrain layers (not S1-only)")
    parser.add_argument("--download-attempts", type=int, default=3)
    args = parser.parse_args()

    start = time.time()
    log(f"dataset {config.DATA_DIR} | steps {args.steps}")
    for train, held, km in config.leakage():
        log(f"warning: train site {train} is {km} km from val/test site {held}")
    if "download" in args.steps:
        download(args)
    if "patches" in args.steps and run_script("make_patches.py") != 0:
        raise SystemExit("make_patches.py failed")
    if "previews" in args.steps and run_script("preview.py", ["--sites", *args.sites] if args.sites else []) != 0:
        raise SystemExit("preview.py failed")
    if "manifest" in args.steps:
        manifest()
    if "zip" in args.steps:
        make_zips()
    log(f"done in {(time.time() - start) / 60:.0f} min")


if __name__ == "__main__":
    main()
