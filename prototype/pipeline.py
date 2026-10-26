"""Model loading, input features, sliding-window inference and change statistics.

The features reproduce Prototype.ipynb exactly (sections 4, 5 and 5b), using the normalisation
stored in the run's model_card.json:
  monthly: VV, VH, VV-VH in dB per month (oldest first), standardised, missing -> 0
  PCA:     pca_weights @ (36 standardised values Jan..Dec - pca_mean)
  stats:   VV min, VV max, VH std, VH 2nd highest - 2nd lowest, share of months with VV < -18 dB
  terrain: elevation, slope, log1p(HAND)
Static channels were stored as float16 during training; the same rounding is applied here.
"""
import io
import warnings

import numpy as np
import pandas as pd
import torch

from model import UNetResNet34

PATCH = 256
WATER_DB = -18.0  # notebook WATER_DB
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# The variant follows from the first layer's channel count: 3 per month + static channels.
VARIANTS = {
    3: dict(name="single_month", n_months=1, pca=False, stats=False, terrain=False),
    9: dict(name="three_month", n_months=3, pca=False, stats=False, terrain=False),
    14: dict(name="three_month_stats", n_months=3, pca=False, stats=True, terrain=False),
    15: dict(name="three_month_pca", n_months=3, pca=True, stats=False, terrain=False),
    20: dict(name="three_month_pca_stats", n_months=3, pca=True, stats=True, terrain=False),
    23: dict(name="three_month_pca_stats_terrain", n_months=3, pca=True, stats=True, terrain=True),
}
CARD_KEYS = {"pca": ["pca_mean", "pca_weights"], "stats": ["stats_mean", "stats_std"],
             "terrain": ["terrain_mean", "terrain_std"]}


def load_model(weight_bytes, card, device):
    """Build the network from a state dict and pick the variant from its input channel count."""
    state = torch.load(io.BytesIO(weight_bytes), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items() if k != "n_averaged"}
    if "stem.0.weight" not in state:
        raise ValueError("not a UNetResNet34 state dict from Prototype.ipynb (no 'stem.0.weight')")
    in_ch = state["stem.0.weight"].shape[1]
    if in_ch not in VARIANTS:
        raise ValueError(f"unknown input channel count {in_ch}; expected one of {sorted(VARIANTS)}")
    variant = VARIANTS[in_ch]
    for part, keys in CARD_KEYS.items():
        missing = [k for k in keys if variant[part] and k not in card]
        if missing:
            raise ValueError(f"model_card.json lacks {missing}, needed by {variant['name']}: "
                             "use the model card from the same run as the weights")
    n_classes = state["head.weight"].shape[0]
    net = UNetResNet34(in_ch, n_classes)
    net.load_state_dict(state)
    return net.to(device).eval(), variant


def db_lut(card):
    d = card["db_decoding"]
    lut = np.arange(256, dtype=np.float32) * d["db_step"] + d["db_min"]
    lut[d["nodata_code"]] = np.nan
    return lut


def standardise(f, mean, std):
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    return np.nan_to_num((f - mean[None, :, None, None]) / std[None, :, None, None], nan=0.0)


def month_features(codes, card):
    """codes (months, 2, H, W) -> (3 * months, H, W): VV, VH, VV-VH per month, oldest first."""
    db = db_lut(card)[codes]
    vv, vh = db[:, 0], db[:, 1]
    f = standardise(np.stack([vv, vh, vv - vh], 1), card["channel_mean"], card["channel_std"])
    return f.reshape(-1, *codes.shape[-2:]).astype(np.float32)


def stats_raw(codes, card):
    db = db_lut(card)[codes]
    vv, vh = db[:, 0], db[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vh_sorted = np.sort(np.where(np.isnan(vh), np.nanmean(vh, 0), vh), 0)
        n_valid = np.maximum((~np.isnan(vv)).sum(0), 1)
        out = np.stack([np.nanmin(vv, 0), np.nanmax(vv, 0), np.nanstd(vh, 0),
                        vh_sorted[-2] - vh_sorted[1], (vv < WATER_DB).sum(0) / n_valid])
    return out.astype(np.float32)


def terrain_raw(t):
    t = t.astype(np.float32)
    return np.stack([t[0], t[1] / 10, np.log1p(np.clip(t[2] / 10, 0, None))])


def static_features(codes_year, terrain, variant, card):
    """codes_year (12, 2, H, W) in calendar order Jan..Dec -> static channels (S, H, W)."""
    parts = []
    if variant["pca"]:
        annual = month_features(codes_year, card).reshape(36, -1)
        w, mean = np.asarray(card["pca_weights"], np.float32), np.asarray(card["pca_mean"], np.float32)
        parts.append((w @ (annual - mean[:, None])).reshape(len(w), *codes_year.shape[-2:]))
    if variant["stats"]:
        parts.append(standardise(stats_raw(codes_year, card)[None], card["stats_mean"], card["stats_std"])[0])
    if variant["terrain"]:
        parts.append(standardise(terrain_raw(terrain)[None], card["terrain_mean"], card["terrain_std"])[0])
    if not parts:
        return np.zeros((0, *codes_year.shape[-2:]), np.float32)
    return np.concatenate(parts).astype(np.float16).astype(np.float32)


def window_starts(n, stride):
    starts = list(range(0, n - PATCH + 1, stride))
    if starts[-1] != n - PATCH:
        starts.append(n - PATCH)
    return starts


def predict_map(net, x, device, stride, batch=8):
    """Sliding-window prediction of one (C, H, W) input; overlapping softmax outputs are summed."""
    _, h, w = x.shape
    probs = None
    tiles = [(r, c) for r in window_starts(h, stride) for c in window_starts(w, stride)]
    with torch.no_grad():
        for i in range(0, len(tiles), batch):
            chunk = tiles[i:i + batch]
            xb = torch.from_numpy(np.stack([x[:, r:r + PATCH, c:c + PATCH] for r, c in chunk])).to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                p = net(xb).float().softmax(1).cpu().numpy()
            if probs is None:
                probs = np.zeros((p.shape[1], h, w), np.float32)
            for (r, c), pi in zip(chunk, p):
                probs[:, r:r + PATCH, c:c + PATCH] += pi
    return probs.argmax(0).astype(np.uint8) + 1  # class ids 1..8


def run_inference(net, variant, card, codes_all, months_all, n_before, terrain, device, stride, progress=None):
    """codes_all (n_before + 12, 2, H, W) for months_all [(year, month), ...], oldest first.

    Returns (12, H, W) class ids for the 12 analysis months (codes_all[n_before:]).
    """
    analysis = codes_all[n_before:]
    order = np.argsort([m for _, m in months_all[n_before:]])  # PCA expects Jan..Dec
    static = static_features(analysis[order], terrain, variant, card)
    n = variant["n_months"]
    preds = []
    for t in range(12):
        end = n_before + t + 1
        x = np.concatenate([month_features(codes_all[end - n:end], card), static])
        preds.append(predict_map(net, x, device, stride))
        if progress:
            progress((t + 1) / 12, f"month {t + 1}/12 predicted")
    return np.stack(preds)


def class_percentages(preds, valid, class_names, labels):
    """Share of each class among valid pixels per month, in %."""
    n = max(int(valid.sum()), 1)
    rows = [[(p[valid] == k + 1).sum() / n * 100 for k in range(len(class_names))] for p in preds]
    return pd.DataFrame(rows, index=labels, columns=class_names)


def month_changes(pct):
    """Month-to-month change in percentage points, plus total change first -> last month."""
    steps = pct.diff().iloc[1:]
    steps.index = [f"{a} → {b}" for a, b in zip(pct.index[:-1], pct.index[1:])]
    total = (pct.iloc[-1] - pct.iloc[0]).rename(f"{pct.index[0]} → {pct.index[-1]}")
    return steps, total


def annual_mode(preds, n_classes):
    """Most frequent class per pixel over the 12 months (for the comparison with an annual map)."""
    counts = np.stack([(preds == k).sum(0) for k in range(1, n_classes + 1)])
    return counts.argmax(0).astype(np.uint8) + 1
