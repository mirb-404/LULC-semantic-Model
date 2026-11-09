"""Streamlit prototype: 12 monthly LULC maps from Sentinel-1 for any location.

Run from this folder:  streamlit run app.py

Layout: place and time at the top; a live map with the segmentation inside the bounding box and the radar input
of the same month beside it, both driven by one month slider (with play); analysis below (summary, year
overview, change, Esri check, downloads). The final model (Weights/Iter2_Test_1_Full) is loaded by default;
other weights, the offline dataset tiles and the patch overlap are under "Advanced" in the sidebar.
"""
import hashlib
import hmac
import io
import json
import os
import time
from collections import Counter
from pathlib import Path

import altair as alt
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch
from streamlit_folium import st_folium

import gee
import mapview
import pipeline
import runlog

ROOT = Path(__file__).resolve().parent.parent
N_BEFORE = 2  # months fetched before the start month, for the three-month window
FINAL_RUN = "Iter2_Test_1_Full"
SIZES = {"2.56 km": 256, "5.12 km": 512, "10.24 km (training tile)": 1024}
ZOOM = {256: 14, 512: 13, 1024: 12}
MAX_ZOOM = 20
OVERLAPS = {"none (fastest)": 256, "25 %": 192, "50 % (smoothest)": 128}
CUSTOM = "Custom: click the map or type coordinates"
MODEL_LAYER = "Land cover (model)"
PRESETS = {  # rural test sites of Dataset_2nd_iteration (never trained on) and the two urban cases
    CUSTOM: None,
    "Black Forest villages, Germany (rural test site)": (47.95, 8.10),
    "Willamette foothills, USA (rural test site)": (44.90, -122.70),
    "Madagascar highlands (rural test site)": (-19.30, 47.30),
    "Kano farmland, Nigeria (rural test site)": (12.20, 8.30),
    "Chilean lakes (rural test site)": (-40.30, -72.80),
    "Thane, India (city, creek, mangroves)": (19.2183, 72.9781),
    "Berlin centre, Germany (city)": (52.5200, 13.4050),
}
SATELLITE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
SATELLITE_ATTR = "Esri, Maxar, Earthstar Geographics"

st.set_page_config(page_title="Sentinel-1 Land Cover Explorer", layout="wide")

# launch.py sets an access key when the app is exposed through a public tunnel
ACCESS_KEY = os.environ.get("APP_ACCESS_KEY")
if ACCESS_KEY and not hmac.compare_digest(st.query_params.get("key", ""), ACCESS_KEY):
    st.error("Access key missing or wrong. Open the full link printed by launch.py.")
    st.stop()


# ---------------------------------------------------------------- helpers
def shift_month(year, month, k):
    i = year * 12 + month - 1 + k
    return i // 12, i % 12 + 1


def parse_degrees(text, lo, hi):
    """Decimal degrees from a text box (pasted values may carry spaces or a trailing comma), or None."""
    try:
        value = float(text.strip().strip(",;"))
    except ValueError:
        return None
    return value if lo <= value <= hi else None


def month_label(year, month):
    return f"{pipeline.MONTH_NAMES[month - 1]} {year}"


def text_color():
    """Label colour readable on the app theme: white on dark (and when unknown), dark grey on light."""
    theme = getattr(getattr(st, "context", None), "theme", None)
    return "#31333f" if getattr(theme, "type", None) == "light" else "#ffffff"


def line_color(color):
    """Very light class colours (snow) are drawn grey in charts, so they stay visible on a white background."""
    return "#9e9e9e" if sum(to_rgb(color)) / 3 > 0.85 else color


def dataset_dir(iteration):
    return ROOT / ("Dataset_2nd_iteration" if iteration >= 2 else "Dataset")


@st.cache_resource(show_spinner="Loading model ...")
def get_model(key, _weight_bytes, card_text, device_name):
    return pipeline.load_model(_weight_bytes, json.loads(card_text), torch.device(device_name))


@st.cache_data(show_spinner=False)
def gee_data(project, lat, lon, year, month, n, need_terrain, iteration):
    gee.initialize(project)
    months_all = [shift_month(year, month, k) for k in range(-N_BEFORE, 12)]
    epsg, x0, y_top, region = gee.grid(lat, lon, n)
    counts = gee.scene_counts(region, months_all)
    image, bands = gee.s1_stack(region, months_all, iteration)
    codes = gee.fetch(image, bands, epsg, x0, y_top, n, np.uint8, 0).reshape(len(months_all), 2, n, n)
    terrain = None
    if need_terrain:
        terrain = gee.fetch(gee.terrain_image(region), gee.TERRAIN_BANDS, epsg, x0, y_top, n, np.int16, 0)
    ref_year = Counter(y for y, _ in months_all[N_BEFORE:]).most_common(1)[0][0]
    try:
        esri, used_year = gee.esri_image(ref_year, iteration)
        ref = gee.fetch(esri, ["esri"], epsg, x0, y_top, n, np.uint8, 0)[0]
        ref_name = f"Esri 10 m Annual LULC {used_year} (Sentinel-2 based)"
    except Exception as err:
        ref, ref_name = None, f"Esri LULC not available: {err}"
    # the global maps define pasture, fallow and rangeland differently: show the other two as references as well
    extra_refs = {}
    if iteration >= 2:
        for name, image in ((f"WorldCover 2021 (training label)", gee.worldcover_image()),
                            (f"Dynamic World {ref_year} (agreement weights in training)",
                             gee.dynamicworld_image(region, ref_year))):
            try:
                extra_refs[name] = gee.fetch(image, ["cls"], epsg, x0, y_top, n, np.uint8, 0)[0]
            except Exception:
                pass
    # one cloud-free Sentinel-2 true-colour image per analysis month, played with the maps (display only)
    s2 = None
    try:
        image, bands = gee.sentinel2_months(region, months_all[N_BEFORE:])
        s2 = gee.fetch(image, bands, epsg, x0, y_top, n, np.uint8, 0).reshape(12, 3, n, n)
    except Exception:
        pass
    return dict(codes=codes, months_all=months_all, counts=counts, terrain=terrain, ref=ref,
                ref_name=ref_name, extra_refs=extra_refs, lat=lat, lon=lon, epsg=epsg, x0=x0, y_top=y_top, s2=s2)


def local_sites(iteration):
    return sorted(p.stem for p in (dataset_dir(iteration) / "s1").glob("*.tif") if not p.stem.endswith(".tmp"))


@st.cache_data(show_spinner=False)
def local_data(site, month, iteration):
    """A downloaded 2021 dataset tile of the model's iteration. Months before the start wrap within 2021,
    as in training."""
    import rasterio
    from pyproj import Transformer
    root = dataset_dir(iteration)
    with rasterio.open(root / "s1" / f"{site}.tif") as src:
        tile = src.read().reshape(12, 2, src.height, src.width)
        cx, cy = (src.bounds.left + src.bounds.right) / 2, (src.bounds.bottom + src.bounds.top) / 2
        lon, lat = Transformer.from_crs(src.crs, 4326, always_xy=True).transform(cx, cy)
        epsg, x0, y_top = src.crs.to_epsg(), src.bounds.left, src.bounds.top
    idx = [(month - 1 + k) % 12 for k in range(-N_BEFORE, 12)]
    terrain_path = root / "terrain" / f"{site}.tif"
    terrain = None
    if terrain_path.exists():
        with rasterio.open(terrain_path) as src:
            terrain = src.read()
    ref, ref_name, extra_refs = None, "no label file", {}
    label_path = root / "labels" / f"{site}.tif"
    if label_path.exists():
        with rasterio.open(label_path) as src:
            ref = src.read(2 if iteration >= 2 else 1)
            if iteration >= 2:
                extra_refs = {"Dynamic World 2021 (agreement weights in training)": src.read(3)}
        ref_name = (("training label (WorldCover 2021)" if iteration >= 2 else
                     "training consensus label (WorldCover ∩ Dynamic World 2021)") + ": NOT independent, test use only")
    return dict(codes=tile[idx], months_all=[(2021, i + 1) for i in idx], counts=None, terrain=terrain,
                ref=ref, ref_name=ref_name, extra_refs=extra_refs, lat=lat, lon=lon, epsg=epsg, x0=x0, y_top=y_top,
                s2=None)


def false_colour(codes_2hw, card):
    lut = pipeline.db_lut(card)
    vv, vh = lut[codes_2hw[0]], lut[codes_2hw[1]]
    rgb = np.stack([vv, vh, vv - vh], -1)
    for b, (lo, hi) in enumerate([(-20, 0), (-27, -7), (2, 15)]):
        rgb[..., b] = np.clip((rgb[..., b] - lo) / (hi - lo), 0, 1)
    return np.nan_to_num(rgb)


def maps_figure(preds, valid, labels, cmap, legend, n_classes):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12.5))
    for ax, p, lab in zip(axes.ravel(), preds, labels):
        ax.imshow(np.where(valid, p, 0), cmap=cmap, vmin=0, vmax=n_classes, interpolation="nearest")
        ax.set_title(lab)
        ax.axis("off")
    fig.legend(handles=legend, loc="lower center", ncol=len(legend))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def radar_figure(codes, labels, card):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12.5))
    for ax, c, lab in zip(axes.ravel(), codes, labels):
        ax.imshow(false_colour(c, card))
        ax.set_title(lab)
        ax.axis("off")
    fig.suptitle("Sentinel-1 input: R = VV, G = VH, B = VV-VH")
    fig.tight_layout()
    return fig


def shares_figure(pct, names, colors, labels):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, color in zip(names, colors):
        ax.plot(labels, pct[name], marker="o", color=line_color(color), label=name)
    ax.set_ylabel("% of area")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    return fig


def change_figure(total, names, colors):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(names, total.values, color=[line_color(c) for c in colors], edgecolor="black")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("pp")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def sanity_check(preds, ref, valid, names):
    """Annual mode vs reference: (annual map, agreement %, majority baseline %, per-class area table).

    The majority baseline is the agreement a map of only the largest reference class would reach."""
    annual = pipeline.annual_mode(preds, len(names))
    both = valid & (ref > 0)
    agree = (annual[both] == ref[both]).mean() * 100 if both.any() else float("nan")
    n = max(int(both.sum()), 1)
    ref_share = [(ref[both] == k + 1).sum() / n * 100 for k in range(len(names))]
    area = pd.DataFrame({"model (annual mode) %": [(annual[both] == k + 1).sum() / n * 100 for k in range(len(names))],
                         "reference %": ref_share,
                         "reference pixels found by model %": [
                             ((annual == k + 1) & (ref == k + 1) & both).sum() / max(((ref == k + 1) & both).sum(), 1) * 100
                             for k in range(len(names))]}, index=names)
    return annual, agree, max(ref_share), area


def check_figure(annual, ref, valid, cmap, legend, n_classes):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    axes[0].imshow(np.where(valid, annual, 0), cmap=cmap, vmin=0, vmax=n_classes, interpolation="nearest")
    axes[0].set_title("model: most frequent class over 12 months")
    axes[1].imshow(ref, cmap=cmap, vmin=0, vmax=n_classes, interpolation="nearest")
    axes[1].set_title("reference")
    for ax in axes:
        ax.axis("off")
    fig.legend(handles=legend, loc="lower center", ncol=len(legend))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=110, bbox_inches="tight")
    return buf.getvalue()


def class_bars(values, names, colors, unit):
    """Horizontal bars in class colours; unit '%' (share of the area) or 'pp' (change, may be negative)."""
    df = pd.DataFrame({"class": names, "value": np.round(np.asarray(values, float), 2),
                       "color": [line_color(c) for c in colors]})
    df["label"] = [f"{v:+.1f} pp" if unit == "pp" else f"{v:.1f} %" for v in df["value"]]
    df["text_x"] = df["value"].clip(lower=0)
    base = alt.Chart(df).encode(y=alt.Y("class:N", sort=list(names), title=None))
    bars = base.mark_bar().encode(
        x=alt.X("value:Q", title="percentage points" if unit == "pp" else "% of area"),
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=[alt.Tooltip("class:N"), alt.Tooltip("label:N", title="value")])
    text = base.mark_text(align="left", dx=4, color=text_color()).encode(x="text_x:Q", text="label:N")
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#888").encode(x="x:Q")
    return (bars + text + rule).properties(height=32 * len(names) + 30)


def compare_bars(before, after, names, colors, before_label, after_label):
    """Area % per class in two months as paired bars (faded = earlier, solid = later), labelled with the change."""
    rows = [dict(cls=n, month=lab, value=round(float(v), 2), color=line_color(c))
            for n, c, b, a in zip(names, colors, before, after) for lab, v in ((before_label, b), (after_label, a))]
    df = pd.DataFrame(rows)
    order = [before_label, after_label]
    bars = alt.Chart(df).mark_bar().encode(
        y=alt.Y("cls:N", sort=list(names), title=None),
        yOffset=alt.YOffset("month:N", sort=order),
        x=alt.X("value:Q", title="% of area"),
        color=alt.Color("color:N", scale=None, legend=None),
        opacity=alt.Opacity("month:N", sort=order, scale=alt.Scale(domain=order, range=[0.35, 1]),
                            legend=alt.Legend(orient="top", title=None, symbolType="square",
                                            symbolFillColor=text_color())),
        tooltip=[alt.Tooltip("cls:N", title="class"), alt.Tooltip("month:N"),
                 alt.Tooltip("value:Q", format=".1f", title="% of area")])
    change = pd.DataFrame({"cls": list(names), "x": np.maximum(before, after),
                           "label": [f"{a - b:+.1f} pp" if abs(a - b) >= 0.05 else "no change"
                                     for b, a in zip(before, after)]})
    text = alt.Chart(change).mark_text(align="left", dx=5, fontWeight="bold", color=text_color()).encode(
        y=alt.Y("cls:N", sort=list(names)), x="x:Q", text="label:N")
    return (bars + text).properties(height=44 * len(names) + 40)


def cumulative_chart(pct, names, colors, k):
    """Running change of every class since the first month (pp), one line per class; the selected month k
    is marked and labelled, later months are drawn faded."""
    since = pct[names] - pct[names].iloc[0]
    months = list(pct.index)
    long = since.reset_index(names="month").melt("month", var_name="cls", value_name="pp")
    long["shown"] = long["month"].map({m: i <= k for i, m in enumerate(months)})
    color = alt.Color("cls:N", sort=list(names), legend=alt.Legend(orient="bottom", title=None),
                      scale=alt.Scale(domain=list(names), range=[line_color(c) for c in colors]))
    x = alt.X("month:N", sort=months, title=None, axis=alt.Axis(labelAngle=-45))
    tooltip = [alt.Tooltip("month:N"), alt.Tooltip("cls:N", title="class"),
               alt.Tooltip("pp:Q", format="+.2f", title=f"change since {months[0]} (pp)")]
    base = alt.Chart(long).encode(x=x, y=alt.Y("pp:Q", title=f"change since {months[0]} (pp)"), color=color,
                                  tooltip=tooltip)
    # a line has one opacity: the whole year faded underneath, the months up to k solid on top
    lines = (base.transform_filter("datum.shown").mark_line(point=True, strokeWidth=2.5)
             + base.mark_line(strokeWidth=2, strokeOpacity=0.2))
    # value labels at month k in the class colour, spread apart so close values do not overlap
    now = long[long["month"] == months[k]].sort_values("pp").copy()
    span = max(float(since.to_numpy().max() - since.to_numpy().min()), 0.1)
    gap, ys = 0.09 * span, []
    for v in now["pp"]:
        ys.append(max(v, ys[-1] + gap) if ys else v)
    shift = (now["pp"].mean() - np.mean(ys)) if ys else 0   # recentre the stack on the real values
    now["label_y"] = np.array(ys) + shift
    now["label"] = [f"{round(v, 1) + 0:+.1f}" if abs(v) >= 0.05 else "0.0" for v in now["pp"]]
    labels = alt.Chart(now).mark_text(align="left", dx=9, fontWeight="bold", fontSize=12).encode(
        x=x, y=alt.Y("label_y:Q"), text="label:N", color=color)
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#888", strokeDash=[4, 4]).encode(y="y:Q")
    mark = alt.Chart(pd.DataFrame({"month": [months[k]]})).mark_rule(color="#ffd92f", strokeWidth=2).encode(
        x=alt.X("month:N", sort=months))
    # right padding: the labels of the last month sit right of the plot and were clipped
    return (zero + mark + lines + labels).properties(height=320, padding={"left": 5, "top": 5, "right": 48,
                                                                           "bottom": 5})


def change_table(pct, steps, names, k):
    """Per class up to month k: share in the first and in month k, the sum of all monthly gains and losses,
    and the net change (gains + losses = month k − first month)."""
    first, now = pct.index[0], pct.index[k]
    part = steps[names].iloc[:k]
    return pd.DataFrame({f"{first} %": pct[names].iloc[0], f"{now} %": pct[names].iloc[k],
                         "gained (pp)": part.clip(lower=0).sum(), "lost (pp)": part.clip(upper=0).sum(),
                         "net (pp)": pct[names].iloc[k] - pct[names].iloc[0]}).round(2)


def class_lines(pct, names, colors):
    """One small line chart per class with its own y scale, so changes of a fraction of a percent are visible."""
    months = list(pct.index)
    long = pct[names].reset_index(names="month").melt("month", var_name="cls", value_name="share")
    return alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("month:N", sort=months, title=None, axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
        y=alt.Y("share:Q", title="% of area", scale=alt.Scale(zero=False)),
        color=alt.Color("cls:N", legend=None,
                        scale=alt.Scale(domain=list(names), range=[line_color(c) for c in colors])),
        tooltip=[alt.Tooltip("month:N"), alt.Tooltip("share:Q", format=".2f", title="% of area")],
    ).properties(width=300, height=140).facet(
        facet=alt.Facet("cls:N", sort=list(names), title=None, header=alt.Header(labelFontWeight="bold",
                                                                                 labelFontSize=13)),
        columns=3).resolve_scale(y="independent")


def summarise(data, preds, card, weight_name, model_name, settings, place, place_name):
    """Everything the result views need, computed once per run (statistics, map overlays, downloads, run log)."""
    names, colors = card["class_names"], card["class_colors"]
    months = data["months_all"][N_BEFORE:]
    labels = [month_label(y, m) for y, m in months]
    valid = (data["codes"][N_BEFORE:, 0] > 0).any(0)
    pct = pipeline.class_percentages(preds, valid, names, labels)
    steps, total = pipeline.month_changes(pct)
    flick = pipeline.flicker(preds, valid)
    check = sanity_check(preds, data["ref"], valid, names) if data["ref"] is not None else None
    n = preds.shape[-1]

    # map overlays: one PNG per month (and the reference), sampled onto Web Mercator
    index, bounds = mapview.overlay_index(data["epsg"], data["x0"], data["y_top"], n)
    overlays = [mapview.png_data_url(mapview.class_overlay(np.where(valid, p, 0), index, colors, 1.0))
                for p in preds]
    s2_overlays, s2_clear = None, None
    if data.get("s2") is not None:
        s2_overlays = [mapview.png_data_url(mapview.rgb_overlay(img, index)) for img in data["s2"]]
        s2_clear = [float((img.max(0) > 0).mean() * 100) for img in data["s2"]]
    # small 12-month previews below the map (UTM grid of the run, north up); no data stays black
    step = max(1, n // 240)
    lut = mapview.palette(colors, 1.0)[:, :3]
    previews = {"Land cover (model)": [lut[np.where(valid, p, 0)[::step, ::step]] for p in preds]}
    if data.get("s2") is not None:
        previews["Photo (Sentinel-2)"] = [img[:, ::step, ::step].transpose(1, 2, 0) for img in data["s2"]]
    previews["Radar (Sentinel-1)"] = [(false_colour(c[:, ::step, ::step], card) * 255).astype(np.uint8)
                                      for c in data["codes"][N_BEFORE:]]
    ref_overlay = (mapview.png_data_url(mapview.class_overlay(data["ref"], index, colors, 1.0))
                   if data["ref"] is not None else None)
    # every reference map: (name, agreement %, majority baseline %, per-class table) and its map overlay
    # exam brief: the case study is checked against an independent map not used for training. Esri is independent
    # for models trained on WorldCover (+ Dynamic World agreement), not for models trained on the 3-map vote.
    vote_model = card.get("training_labels") == "vote"
    esri_name = "Esri (used in training: vote labels)" if vote_model else "Esri (independent)"
    if data["ref_name"].startswith("Esri 10 m"):
        data["ref_name"] += (": used in the training vote, NOT independent" if vote_model
                             else ": independent of the training labels")
    refs = ({esri_name: data["ref"]} if data["ref"] is not None else {}) | data.get("extra_refs", {})
    ref_checks = {name: sanity_check(preds, arr, valid, names) for name, arr in refs.items()}
    ref_overlays = {name: mapview.png_data_url(mapview.class_overlay(arr, index, colors, 1.0))
                    for name, arr in refs.items()}

    # figures for the downloads and the run log
    cmap = ListedColormap(["#000000"] + colors)
    legend = [Patch(color=c, label=nm) for nm, c in zip(["no data"] + names, ["#000000"] + colors)]
    figures = {"lulc_12_months.png": maps_figure(preds, valid, labels, cmap, legend, len(names)),
               "radar_12_months.png": radar_figure(data["codes"][N_BEFORE:], labels, card),
               "class_shares.png": shares_figure(pct, names, colors, labels),
               "total_change.png": change_figure(total, names, colors)}
    if check is not None:
        figures["sanity_check.png"] = check_figure(check[0], data["ref"], valid, cmap, legend, len(names))
    changes = pd.concat([steps, total.to_frame().T])
    tables = {"class_percentages.csv": pct.round(4), "class_changes.csv": changes.round(4)}
    if check is not None:
        tables["sanity_check.csv"] = check[3].round(2)
    meta = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), place_name=place_name, **settings, lat=data["lat"],
                lon=data["lon"], epsg=data["epsg"], weights=weight_name, model=model_name,
                model_card={k: card.get(k) for k in ("iteration", "variant", "encoder", "n_months", "in_channels",
                                                     "temporal_smoothing", "test_rural_mIoU", "test_mIoU")},
                class_names=names, months=labels,
                months_fetched=[month_label(y, m) for y, m in data["months_all"]],
                scene_counts=None if data["counts"] is None else [int(c) for c in data["counts"]],
                valid_share=float(valid.mean()), reference=data["ref_name"],
                reference_agreement_pct=None if check is None else float(check[1]),
                reference_majority_baseline_pct=None if check is None else float(check[2]),
                all_references={name: {"agreement_pct": float(c[1]), "majority_baseline_pct": float(c[2])}
                                for name, c in ref_checks.items()},
                flicker_pct=flick)
    files = {name: (png_bytes(fig), "image/png") for name, fig in figures.items()}
    files.update({name: (df.to_csv().encode(), "text/csv") for name, df in tables.items()})
    files["lulc_12_months.tif"] = (mapview.geotiff_bytes(np.where(valid, preds, 0), data["epsg"], data["x0"],
                                                         data["y_top"], labels), "image/tiff")
    files["run.json"] = (json.dumps(meta, indent=2, default=str).encode(), "application/json")

    # every run is saved once to prototype/runs/ for later analysis (see runlog.py)
    try:
        log_dir = runlog.run_dir(place)
        runlog.save_run(log_dir, meta, dict(radar_codes=data["codes"], lulc=preds.astype(np.uint8), valid=valid,
                                            reference=data["ref"], terrain=data["terrain"]), tables, figures)
    except Exception as err:
        log_dir = None
        st.warning(f"Could not save the run log: {err}")
    plt.close("all")

    return dict(data=data, preds=preds, valid=valid, card=card, names=names, colors=colors, labels=labels,
                pct=pct, steps=steps, total=total, flicker=flick, check=check, overlays=overlays,
                s2_overlays=s2_overlays, s2_clear=s2_clear, previews=previews,
                ref_overlay=ref_overlay, ref_checks=ref_checks, ref_overlays=ref_overlays, bounds=bounds,
                corners=mapview.utm_corners(data["epsg"], data["x0"], data["y_top"], n),
                files=files, log_dir=log_dir, model=model_name, weights=weight_name, place_name=place_name,
                size_km=n * mapview.PIXEL_M / 1000)


def base_map(center, zoom, basemap, tiles=None):
    """basemap 'Streets' = OpenStreetMap; otherwise satellite tiles (`tiles`, default today's Esri imagery)."""
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, max_zoom=MAX_ZOOM)
    # max_native_zoom: above it Leaflet enlarges the last real tiles instead of requesting deeper ones. Esri has
    # no photo beyond about zoom 17 in many rural areas and answers with a "Map data not yet available" tile.
    if basemap == "Streets":
        folium.TileLayer("OpenStreetMap", name="Streets", max_zoom=MAX_ZOOM, max_native_zoom=19).add_to(m)
    else:
        folium.TileLayer(tiles or SATELLITE_TILES, attr=SATELLITE_ATTR, name="Satellite", max_zoom=MAX_ZOOM,
                         max_native_zoom=17).add_to(m)
    return m


# network errors raise, and Streamlit does not cache a raised call, so a failed lookup is retried on the next rerun
wayback_releases = st.cache_data(ttl=24 * 3600, show_spinner=False)(mapview.wayback_releases)
wayback_capture = st.cache_data(show_spinner=False)(mapview.wayback_capture)
geocode = st.cache_data(ttl=24 * 3600, show_spinner=False)(mapview.geocode)


def satellite_for_year(year, month, lat, lon):
    """(tile URL, caption) of the newest satellite photo published before the 12 analysis months ended.

    Esri World Imagery Wayback; falls back to today's imagery (URL None) if the archive is not reachable.
    """
    y_end, m_end = shift_month(year, month, 12)
    try:
        _, url, metadata_url = mapview.wayback_release_for(wayback_releases(), f"{y_end}-{m_end:02d}-01")
    except Exception:
        return None, "Satellite photo: today's imagery (the photo archive is not reachable)."
    try:
        capture = wayback_capture(metadata_url, round(lat, 4), round(lon, 4))
    except Exception:
        capture = None
    if not capture:
        return url, f"Satellite photo: newest one available before {month_label(y_end, m_end)}."
    taken, res, source = capture
    note = f"Satellite photo taken **{taken}** ({source})"
    if int(taken[:4]) < year:
        note += f": older than the analysis year {year}, so the land may have changed since"
    return url, note + "."


def show_map(center, zoom, basemap, layers, height=560, tiles=None):
    """Draws the map with the given overlay layers; a click on the map moves the analysis area there."""
    group = folium.FeatureGroup(name="overlays")
    for layer in layers:
        layer.add_to(group)
    # st_folium only updates the overlay group of an existing map: a new key redraws it when the photo changes
    key = "map_" + hashlib.md5(f"{basemap}{tiles}".encode()).hexdigest()[:8]
    # center/zoom passed separately: st_folium pans an existing map only when these change
    out = st_folium(base_map(center, zoom, basemap, tiles), feature_group_to_add=group, key=key, height=height,
                    center=center, zoom=zoom, use_container_width=True, returned_objects=["last_clicked"],
                    pixelated=True)  # crisp 10 m pixels when zoomed in, not a blurred smear
    click = (out or {}).get("last_clicked")
    if click and click != st.session_state.get("handled_click"):
        st.session_state["handled_click"] = click
        st.session_state["pending_loc"] = (click["lat"], click["lng"])
        st.rerun()


def box_layer(corners, color, dashed=False, tooltip=None):
    return folium.Polygon(corners, color=color, weight=2.5, fill=False, dash_array="8 6" if dashed else None,
                          tooltip=tooltip)


# ---------------------------------------------------------------- location state
st.session_state.setdefault("lat_text", f"{PRESETS[list(PRESETS)[1]][0]:.4f}")
st.session_state.setdefault("lon_text", f"{PRESETS[list(PRESETS)[1]][1]:.4f}")
st.session_state.setdefault("preset", list(PRESETS)[1])
if "pending_loc" in st.session_state:  # a map click from the previous run
    plat, plon = st.session_state.pop("pending_loc")
    st.session_state["lat_text"], st.session_state["lon_text"] = f"{plat:.5f}", f"{plon:.5f}"
    st.session_state["preset"] = CUSTOM
    st.session_state.pop("place_found", None)


def apply_preset():
    point = PRESETS[st.session_state["preset"]]
    if point:
        st.session_state["lat_text"], st.session_state["lon_text"] = f"{point[0]:.4f}", f"{point[1]:.4f}"
        st.session_state.pop("place_found", None)


def go_to_place(name, lat, lon):
    st.session_state["lat_text"], st.session_state["lon_text"] = f"{lat:.5f}", f"{lon:.5f}"
    st.session_state["preset"] = CUSTOM
    st.session_state["place_found"] = name


def search_place():
    """Search box callback: look the text up (OpenStreetMap) and move the area to the best match."""
    query = st.session_state["search_text"].strip()
    st.session_state["search_hits"], st.session_state["search_error"] = [], None
    if not query:
        return
    try:
        hits = geocode(query)
    except Exception:
        st.session_state["search_error"] = "Place search is not reachable right now; use coordinates or the map."
        return
    if not hits:
        st.session_state["search_error"] = f"No place found for “{query}”."
        return
    st.session_state["search_hits"] = hits
    st.session_state["search_pick"] = 0
    go_to_place(*hits[0])


def typed_coordinates():
    """Coordinates typed by hand: the place is no longer the preset or the searched place."""
    st.session_state["preset"] = CUSTOM
    st.session_state.pop("place_found", None)


def pick_place():
    go_to_place(*st.session_state["search_hits"][st.session_state["search_pick"]])


# ---------------------------------------------------------------- sidebar: advanced
with st.sidebar:
    st.header("Advanced")
    st.caption("The final model is used by default. Change it here only to compare other runs.")
    final_dir = ROOT / "Weights" / FINAL_RUN
    model_choice = st.radio("Model", [f"Final model ({FINAL_RUN})", "Other run folder"])
    weight_bytes, card_text, weight_name = None, None, None
    if model_choice.startswith("Final"):
        if (final_dir / "final_model.pt").exists() and (final_dir / "model_card.json").exists():
            weight_bytes = (final_dir / "final_model.pt").read_bytes()
            card_text = (final_dir / "model_card.json").read_text()
            weight_name = f"{FINAL_RUN}/final_model.pt"
        else:
            st.error(f"Model files not found on the server: copy `final_model.pt` and `model_card.json` to "
                     f"`Weights/{FINAL_RUN}/`, or choose another run folder.")
    else:
        runs = sorted((p for p in (ROOT / "Weights").glob("*") if (p / "model_card.json").exists()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            st.info("No run folder with model_card.json in `Weights/`.")
        else:
            run = st.selectbox("Run folder (newest first)", runs, format_func=lambda p: p.name)
            pts = sorted(run.glob("*.pt"), key=lambda p: (p.name not in ("final_model.pt", "best_model.pt"), p.name))
            if pts:
                pt = st.selectbox("Weights file", pts, format_func=lambda p: p.name)
                weight_bytes, card_text = pt.read_bytes(), (run / "model_card.json").read_text()
                weight_name = f"{run.name}/{pt.name}"
            else:
                st.info(f"No .pt file in `Weights/{run.name}/`.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = variant = card = None
    if weight_bytes:
        try:
            card = json.loads(card_text)
            key = hashlib.sha1(weight_bytes).hexdigest() + hashlib.sha1(card_text.encode()).hexdigest()
            net, variant = get_model(key, weight_bytes, card_text, device.type)
            st.success(f"`{weight_name}` ready: U-Net ({variant['encoder']}), Sentinel-1 radar only"
                       f"{', 3-month smoothing' if variant['smoothing'] else ''}, on {device.type.upper()}")
            if variant["terrain"]:
                st.warning("This variant also uses terrain (DEM, HAND): not Sentinel-1 only.")
            if any(card.get("class_logit_bias") or []):
                st.caption("Class bias (tuned on val): " + ", ".join(
                    f"{n} {b:+.2f}" for n, b in zip(card["class_names"], card["class_logit_bias"]) if b))
        except Exception as err:
            st.error(f"Could not load the model: {err}")
            net = None

    source = st.radio("Data source", ["Google Earth Engine (any place)", "Local dataset tile (offline test)"])
    use_gee = source.startswith("Google")
    stride = OVERLAPS[st.select_slider("Patch overlap", list(OVERLAPS), value="25 %",
                                       help="More overlap gives smoother patch borders but takes longer.")]

# ---------------------------------------------------------------- header and controls
st.title("Sentinel-1 Land Cover Explorer")
st.caption("12 monthly land-cover maps from radar only (Sentinel-1), for any place on Earth. "
           + (f"Rural test accuracy (mIoU): {card['test_rural_mIoU']:.3f}."
              if card and card.get("test_rural_mIoU") else "")
           + (f" Model: {weight_name}." if weight_name and not model_choice.startswith("Final") else ""))

result = st.session_state.get("result")
with st.container(border=True):
    if use_gee:
        s = st.columns([3.2, 4.3], vertical_alignment="bottom")
        # a form: Enter always submits, also for the same text again (on_change fires only when the text changes)
        with s[0].form("search_form", border=False):
            f = st.columns([4, 1], vertical_alignment="bottom")
            f[0].text_input("Search a place", key="search_text",
                            placeholder="Town, region or address, then Enter (e.g. Berlin, Kano Nigeria)")
            f[1].form_submit_button("Search", on_click=search_place, width="stretch")
        hits = st.session_state.get("search_hits") or []
        if len(hits) > 1:
            s[1].selectbox("Matches", range(len(hits)), key="search_pick", on_change=pick_place,
                           format_func=lambda i: hits[i][0])
        if st.session_state.get("search_error"):
            s[1].warning(st.session_state["search_error"])
        c = st.columns([3.2, 1.3, 1.3, 1, 1.1, 1.5, 1.1], vertical_alignment="bottom")
        c[0].selectbox("Place", list(PRESETS), key="preset", on_change=apply_preset)
        # text boxes instead of number inputs: no +/- buttons, coordinates are usually pasted
        lat = parse_degrees(c[1].text_input("Latitude", key="lat_text", on_change=typed_coordinates), -60, 72)  # Sentinel-1 IW coverage
        lon = parse_degrees(c[2].text_input("Longitude", key="lon_text", on_change=typed_coordinates), -180, 180)
        year = c[3].number_input("Year", 2015, 2025, 2021, step=1)
        month = c[4].selectbox("Start month", range(1, 13), format_func=lambda m: pipeline.MONTH_NAMES[m - 1])
        size = SIZES[c[5].selectbox("Area", list(SIZES), index=1)]
        ready = net is not None and lat is not None and lon is not None
        run_clicked = c[6].button("Run", type="primary", disabled=not ready, width="stretch")
        if lat is None or lon is None:
            st.error("Latitude -60 to 72, longitude -180 to 180, in decimal degrees (e.g. 47.95 and 8.10).")
        else:
            st.caption(f"Analysis area: {size * 10 / 1000:.2f} km × {size * 10 / 1000:.2f} km around "
                       f"{lat:.4f}, {lon:.4f}; months {month_label(int(year), int(month))} to "
                       f"{month_label(*shift_month(int(year), int(month), 11))}. Click the map to move the area.")
    else:
        sites = local_sites(variant["iteration"] if variant else 2)
        c = st.columns([3, 1.2, 1.1], vertical_alignment="bottom")
        site = c[0].selectbox("Dataset tile (2021)", sites) if sites else None
        month = c[1].selectbox("Start month", range(1, 13), format_func=lambda m: pipeline.MONTH_NAMES[m - 1])
        run_clicked = c[2].button("Run", type="primary", disabled=net is None or not site, width="stretch")
        year = 2021
        if not sites:
            st.info("No local tiles found in the dataset folder.")
        lat = lon = None

# ---------------------------------------------------------------- run
if run_clicked:
    t0 = time.time()
    with st.status("Running ...", expanded=True) as status:
        try:
            if use_gee:
                # the project ID stays on the server (EE_PROJECT or Project_ID in .env), never in the browser
                project = gee.default_project()
                if not project:
                    raise RuntimeError("no Earth Engine project configured: set Project_ID in the repo's .env "
                                       "on the server")
                st.write("1/3 Fetching Sentinel-1 monthly composites, the reference maps and monthly Sentinel-2 photos "
                         "(display only) from Earth Engine")
                data = gee_data(project, float(lat), float(lon), int(year), int(month), size, variant["terrain"],
                                variant["iteration"])
                settings = dict(source="Google Earth Engine", start_year=int(year), start_month=int(month),
                                area_px=size, patch_stride=stride, device=device.type)
                place = f"{data['lat']:.4f}_{data['lon']:.4f}_{int(year)}-{int(month):02d}"
                place_name = (st.session_state["preset"] if st.session_state["preset"] != CUSTOM
                              else st.session_state.get("place_found") or "custom")
            else:
                st.write("1/3 Reading the local dataset tile")
                data = local_data(site, int(month), variant["iteration"])
                settings = dict(source="local dataset tile", site=site, start_month=int(month),
                                patch_stride=stride, device=device.type)
                place, place_name = f"{site}_2021-{int(month):02d}", site
            if variant["terrain"] and data["terrain"] is None:
                raise RuntimeError("this variant needs terrain, which is not available for this tile")
            st.write(f"2/3 Predicting 12 months on the {device.type.upper()}")
            bar = st.progress(0.0)
            preds = pipeline.run_inference(net, variant, card, data["codes"], data["months_all"], N_BEFORE,
                                           data["terrain"], device, stride, progress=bar.progress)
            st.write("3/3 Statistics, map layers, downloads and run log")
            model_name = variant["name"] + (" (smoothed)" if variant["smoothing"] else "")
            st.session_state["result"] = summarise(data, preds, card, weight_name, model_name, settings, place,
                                                   place_name)
            st.session_state["month_idx"] = 0
            st.session_state["playing"] = False
            status.update(label=f"Done in {time.time() - t0:.0f} s", state="complete", expanded=False)
        except Exception as err:
            message = str(err)
            if use_gee and gee.default_project():
                message = message.replace(gee.default_project(), "<project>")  # Earth Engine errors name the project
            status.update(label="Run failed", state="error", expanded=True)
            st.error(f"Run failed: {message}")
            if use_gee:
                st.info("Earth Engine needs a one-time login on the server: run `python check_auth.py` there "
                        "to see which step fails.")
            st.stop()
    result = st.session_state["result"]

# planned area of the next run (dashed); the analysed area is drawn solid
planned = None
if use_gee and lat is not None and lon is not None:
    planned = mapview.utm_corners(*mapview.utm_window(lat, lon, size), size)
if result is not None:
    rlat, rlon = result["data"]["lat"], result["data"]["lon"]
    if planned is not None and (abs(rlat - lat) > 1e-6 or abs(rlon - lon) > 1e-6 or
                                round(result["size_km"] * 100) != size):
        center, zoom = (lat, lon), ZOOM.get(size, 13)
    else:
        center, zoom = (rlat, rlon), ZOOM.get(round(result["size_km"] * 100), 13)
elif planned is not None:
    center, zoom = (lat, lon), ZOOM.get(size, 13)
else:
    center, zoom = (20.0, 0.0), 2

# ---------------------------------------------------------------- no result yet: map to choose the place
if result is None:
    basemap = st.radio("Basemap", ["Satellite", "Streets"], horizontal=True, key="basemap")
    tiles = basemap_note = None
    if basemap == "Satellite" and planned is not None:
        tiles, basemap_note = satellite_for_year(int(year), int(month), lat, lon)
    show_map(center, zoom, basemap, [box_layer(planned, "#ffd92f", True, "next analysis area")] if planned else [],
             tiles=tiles)
    if basemap_note:
        st.caption(basemap_note)
    st.info("Choose a place (search, preset, coordinates or a click on the map), then press **Run**.")
    st.stop()


# ---------------------------------------------------------------- viewer: map + radar, one month slider
def viewer():
    r = st.session_state["result"]
    labels, names, colors, data = r["labels"], r["names"], r["colors"], r["data"]
    if st.session_state.get("playing"):
        st.session_state["month_idx"] = (st.session_state.get("month_idx", 0) + 1) % 12

    # controls: month, play, map layer; the rarely changed map settings sit in a popover
    c = st.columns([5, 1.1, 2.6, 1.5], vertical_alignment="bottom")
    k = c[0].select_slider("Month", options=list(range(12)), format_func=lambda i: labels[i], key="month_idx")
    playing = st.session_state.get("playing", False)
    if c[1].button("Pause" if playing else "Play", icon=":material/pause:" if playing else ":material/play_arrow:",
                   width="stretch", help="Step through the 12 months, about one month per second."):
        st.session_state["playing"] = not playing
        st.rerun()
    ref_names = list(r.get("ref_overlays", {}))
    layers_menu = [MODEL_LAYER, *[f"Reference: {n}" for n in ref_names], "None"]
    if st.session_state.get("overlay") not in layers_menu:
        st.session_state.pop("overlay", None)
    layer = c[2].selectbox("Map layer", layers_menu, key="overlay",
                           help="The model's land-cover map of the selected month, or an annual reference map "
                                "for comparison.")
    with c[3].popover("Map options", icon=":material/tune:"):
        basemap = st.radio("Background", ["Satellite", "Streets"], horizontal=True, key="basemap")
        opacity = st.slider("Layer opacity", 0, 100, 65, 5, key="opacity", format="%d %%") / 100

    left, right = st.columns([3, 2], gap="large")
    with left:
        tiles, background = None, None
        if basemap == "Satellite":
            # the background photo follows the analysed year of the shown result
            tiles, background = satellite_for_year(*data["months_all"][N_BEFORE], data["lat"], data["lon"])
        photo_on = bool(r.get("s2_overlays")) and st.session_state.get("s2_inside", False)
        layers = []
        if photo_on:
            layers.append(folium.raster_layers.ImageOverlay(r["s2_overlays"][k], r["bounds"], opacity=1.0,
                                                            interactive=False))
        if layer == MODEL_LAYER:
            layers.append(folium.raster_layers.ImageOverlay(r["overlays"][k], r["bounds"], opacity=opacity,
                                                            interactive=False))
        elif layer.startswith("Reference: "):
            layers.append(folium.raster_layers.ImageOverlay(r["ref_overlays"][layer.removeprefix("Reference: ")],
                                                            r["bounds"], opacity=opacity, interactive=False))
        layers.append(box_layer(r["corners"], "#ffffff", tooltip="analysed area"))
        if planned is not None and planned != r["corners"]:
            layers.append(box_layer(planned, "#ffd92f", True, "next analysis area (press Run)"))
        show_map(center, zoom, basemap, layers, tiles=tiles)
        if r.get("s2_overlays"):
            st.toggle("Sentinel-2 photo of this month inside the box", key="s2_inside",
                      help="Cloud-free true-colour photo of the selected month, to check the map by eye. "
                           "Display only, not a model input. Clouds and city smog can hide parts of it.")
        notes = []
        if photo_on:
            clear = r["s2_clear"][k]
            notes.append(f"Box: Sentinel-2 photo of {labels[k]}"
                         + (f", {100 - clear:.0f} % hidden by clouds" if clear < 99.5 else ""))
        if background:
            notes.append("Background: " + background[0].lower() + background[1:])
        if notes:
            st.caption(" · ".join(notes))
        below_map = st.container()  # filled by the right column (change table), where there is more room

    with right:
        # only classes present in this area; the coloured bars double as the map legend
        present = [i for i in range(len(names)) if r["pct"].iloc[:, i].max() >= 0.05]
        shown_names, shown_colors = [names[i] for i in present], [colors[i] for i in present]
        st.markdown(f"#### Land cover · {labels[k]}")
        views = ["Area %", "Change vs last month", f"Change since {labels[0]}"]
        if st.session_state.get("bars_view") not in views:
            st.session_state["bars_view"] = views[0]
        view = st.segmented_control("Show", views, key="bars_view", label_visibility="collapsed") or views[0]
        share = r["pct"].iloc[k].values
        if view == views[0]:
            st.altair_chart(class_bars(share[present], shown_names, shown_colors, "%"), width="stretch")
            i = int(np.argmax(share))
            st.caption(f"Largest class: {names[i]}, {share[i]:.1f} % of the area.")
        elif k == 0:
            st.info(f"{labels[0]} is the first month: there is no earlier month to compare with.")
        elif view == views[2]:
            # every monthly change carried forward: the line at month k is the sum of all steps up to k
            st.altair_chart(cumulative_chart(r["pct"], shown_names, shown_colors, k), width="stretch")
            with below_map:
                st.markdown(f"**Change since {labels[0]}, per class**")
                st.dataframe(change_table(r["pct"], r["steps"], shown_names, k), width="stretch")
                st.caption("Net = gained + lost = the line at the yellow month. Gains and losses in between "
                           "cancel. Changes under about 1 pp are usually model noise, not real land change.")
        else:
            before = r["pct"].iloc[k - 1].values
            st.altair_chart(compare_bars(before[present], share[present], shown_names, shown_colors, labels[k - 1],
                                         labels[k]), width="stretch")
            diff = share - before
            i = int(np.abs(diff).argmax())
            st.caption(("No class changed by 0.1 pp or more." if abs(diff[i]) < 0.05
                        else f"Largest change: {names[i]} {diff[i]:+.1f} percentage points.")
                       + " Changes under about 1 pp are usually model noise, not real land change.")

        # what the model sees: the Sentinel-1 radar composite of the selected month
        count = data["counts"][N_BEFORE + k] if data["counts"] is not None else None
        st.markdown(f"#### Radar input · {labels[k]}")
        if count == 0:
            st.warning(f"No Sentinel-1 radar scene in {labels[k]}: this month's map is less reliable.")
        st.image(false_colour(data["codes"][N_BEFORE + k], r["card"]), width="stretch", clamp=True,
                 caption="Sentinel-1, the model's only input (R = VV, G = VH, B = VV−VH), same area as the box"
                         + (f", {count} scene{'s' if count != 1 else ''} this month" if count else ""))

    previews = r.get("previews")
    if previews:
        # all 12 months at a glance; the month shown on the map is framed in yellow
        st.markdown("#### All 12 months")
        kinds = list(previews)
        if st.session_state.get("preview_kind") not in kinds:
            st.session_state["preview_kind"] = kinds[0]
        kind = st.segmented_control("Preview", kinds, key="preview_kind", label_visibility="collapsed") or kinds[0]
        for row in range(2):
            cols = st.columns(6)
            for j, col in enumerate(cols):
                i = row * 6 + j
                img = previews[kind][i]
                if i == k:
                    pad = max(3, img.shape[0] // 40)
                    framed = np.empty((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad, 3), np.uint8)
                    framed[:] = (255, 217, 47)
                    framed[pad:-pad, pad:-pad] = img
                    img = framed
                col.image(img, width="stretch", caption=labels[i])


st.fragment(run_every=1.2 if st.session_state.get("playing") else None)(viewer)()

# ---------------------------------------------------------------- analysis
r = result
data, names, colors, labels, pct, total = r["data"], r["names"], r["colors"], r["labels"], r["pct"], r["total"]

if data["counts"] is not None:
    empty = [month_label(y, m) for (y, m), cnt in zip(data["months_all"], data["counts"]) if cnt == 0]
    if empty:
        st.warning(f"No Sentinel-1 scenes for: {', '.join(empty)}. These inputs are treated as missing.")
if r["valid"].mean() < 0.99:
    st.warning(f"{(1 - r['valid'].mean()) * 100:.1f} % of the area has no radar data and is excluded.")

st.subheader("Analysis")
st.caption(r["place_name"])
mean_share = pct.mean()
biggest = total.abs().idxmax()
m = st.columns(5)
m[0].metric("Largest class (year mean)", mean_share.idxmax(), f"{mean_share.max():.1f} % of the area",
            delta_color="off")
m[1].metric(f"Largest change {total.name}", biggest, f"{total[biggest]:+.1f} pp", delta_color="off")
m[2].metric("Flicker (mean per month)", f"{np.mean(r['flicker']):.1f} %",
            help="Share of pixels that change class from one month to the next. Most of it is model noise or "
                 "seasonal radar effects, not real land-use change.")
if data["counts"] is not None:
    counts = data["counts"][N_BEFORE:]
    m[3].metric("Scenes per month", f"{min(counts)}–{max(counts)}",
                help="Sentinel-1 scenes averaged into each monthly composite. Fewer scenes give a noisier input.")
else:
    m[3].metric("Scenes per month", "n/a", help="Not recorded for local dataset tiles.")
if r["check"] is not None:
    m[4].metric("Agreement with reference", f"{r['check'][1]:.1f} %",
                f"{r['check'][1] - r['check'][2]:+.1f} pp vs majority baseline",
                help="Most frequent monthly class vs the reference map. The majority baseline is what a map of only "
                     "the largest reference class would score.")
else:
    m[4].metric("Agreement with reference", "n/a")

tab_year, tab_check, tab_files, tab_help = st.tabs(
    ["Year overview", "Reference check", "Downloads and run log", "How to read the results"])

with tab_year:
    shown = [n for n in names if pct[n].max() >= 0.05]
    st.altair_chart(class_lines(pct, shown, [colors[names.index(n)] for n in shown]))
    st.caption("Each class has its own scale, so small changes are visible. Compare the numbers on the axes: "
               "changes under about 1 pp are usually model noise.")
    with st.expander("Exact numbers"):
        st.dataframe(pct.round(2), width="stretch")

with tab_check:
    if r.get("ref_checks"):
        st.markdown("**All reference maps.** WorldCover, Dynamic World and Esri define pasture, fallow, rangeland and "
                    "sparse vegetation differently, so they disagree with each other as well. A low agreement with one "
                    "map and a high one with another is a definition conflict, not necessarily a model error.")
        st.dataframe(pd.DataFrame({name: {"agreement %": c[1], "majority baseline %": c[2],
                                          "gain over baseline pp": c[1] - c[2]}
                                   for name, c in r["ref_checks"].items()}).T.round(1), width="stretch")
    st.write(data["ref_name"])
    if r["check"] is None:
        st.info("No reference map available.")
    else:
        annual, agree, baseline, area = r["check"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Pixel agreement", f"{agree:.1f} %")
        c2.metric("Majority baseline", f"{baseline:.1f} %",
                  help="Agreement of a map that shows only the largest reference class.")
        c3.metric("Gain over baseline", f"{agree - baseline:+.1f} pp")
        if agree - baseline < 2:
            st.warning("The model barely beats a one-class map here: the area is dominated by one class, so the "
                       "agreement says little about the model.")
        st.dataframe(area.round(1), width="stretch")
        st.image(r["files"]["sanity_check.png"][0], width="stretch")
        st.caption("The reference maps use their own class definitions (e.g. Esri 'rangeland' and 'flooded "
                   "vegetation' are mapped to our Grassland class; Esri 'built area' counts vegetation mixed with "
                   "buildings as built), so part of the disagreement is definitional. Use the 'Esri reference' "
                   "overlay above to compare on the map.")

with tab_files:
    describe = {"lulc_12_months.png": "12 monthly maps (PNG)", "lulc_12_months.tif": "12 monthly maps (GeoTIFF, UTM)",
                "radar_12_months.png": "12 radar inputs (PNG)", "class_percentages.csv": "class shares (CSV)",
                "class_changes.csv": "changes (CSV)", "class_shares.png": "class shares chart (PNG)",
                "total_change.png": "total change chart (PNG)", "sanity_check.csv": "reference check (CSV)",
                "sanity_check.png": "reference check maps (PNG)", "run.json": "run summary (JSON)"}
    cols = st.columns(3)
    for i, (name, (content, mime)) in enumerate(r["files"].items()):
        cols[i % 3].download_button(describe.get(name, name), content, name, mime, width="stretch",
                                    key=f"dl_{name}")
    if r["log_dir"]:
        st.caption(f"Run saved on the server to `{r['log_dir']}`.")

with tab_help:
    st.markdown(
        "- **What the model sees:** only Sentinel-1 radar (VV and VH backscatter), one composite per month plus a "
        "summary of the whole year. Radar works through clouds and at night.\n"
        "- **Resolution:** maps are stored at 10 m, but the radar resolves about 20 m. Single buildings, narrow "
        "roads and small streams are often missed.\n"
        "- **Known confusions:** Water vs radar shadow or dark forest (Water is over-predicted); Snow & ice vs Bare "
        "ground; Crop fields vs Grassland outside the growing season.\n"
        "- **Change is not always real:** the class percentages and their changes include model errors. Check the "
        "flicker value: if it is similar to a change, the change may be noise.\n"
        "- **Reference check:** always compare the agreement with the majority baseline. In a city, a map that "
        "shows only Man-made already agrees on most pixels.\n"
        "- **Quality of the final model:** rural test mIoU 0.679 (expected 0.66 ± 0.03 between training runs); "
        "weakest in Africa and Oceania.")
