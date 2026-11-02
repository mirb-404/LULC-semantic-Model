"""Saves every prototype run to prototype/runs/<time>_<place>/ for later analysis.

Contents of a run folder:
  run.json                 settings, model, location, months, scene counts, sanity-check score
  arrays.npz               radar codes (months, VV/VH, H, W; dB = code * 0.2 - 35, 0 = no data), predicted
                           class maps (months, H, W; 1..n classes), valid mask, reference map, terrain if used
  class_percentages.csv, class_changes.csv
  *.png                    the figures (monthly LULC maps, radar input, class shares, change, sanity check)
"""
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def run_dir(place):
    slug = re.sub(r"[^A-Za-z0-9.+-]+", "_", place).strip("_")
    folder = RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_run(folder, meta, arrays, tables, figures):
    """meta: JSON-able dict; arrays: name -> ndarray or None; tables: file name -> DataFrame;
    figures: file name -> matplotlib figure (closed after saving)."""
    (folder / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    np.savez_compressed(folder / "arrays.npz", **{k: v for k, v in arrays.items() if v is not None})
    for name, df in tables.items():
        df.to_csv(folder / name)
    for name, fig in figures.items():
        fig.savefig(folder / name, dpi=110, bbox_inches="tight")
        plt.close(fig)
