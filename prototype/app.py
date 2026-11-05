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
OVERLAPS = {"none (fastest)": 256, "25 %": 192, "50 % (smoothest)": 128}
CUSTOM = "Custom: click the map or type coordinates"
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
        ref_name = f"Esri 10 m Annual LULC {used_year} (independent, Sentinel-2 based)"
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
    return dict(codes=codes, months_all=months_all, counts=counts, terrain=terrain, ref=ref,
                ref_name=ref_name, extra_refs=extra_refs, lat=lat, lon=lon, epsg=epsg, x0=x0, y_top=y_top)


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
                ref=ref, ref_name=ref_name, extra_refs=extra_refs, lat=lat, lon=lon, epsg=epsg, x0=x0, y_top=y_top)


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
    text = base.mark_text(align="left", dx=4).encode(x="text_x:Q", text="label:N")
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#888").encode(x="x:Q")
    return (bars + text + rule).properties(height=32 * len(names) + 30)


def year_chart(pct, names, colors):
    """Stacked bars: class shares of every month."""
    long = pct.reset_index(names="month").melt("month", var_name="class", value_name="share")
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("month:N", sort=list(pct.index), title=None),
        y=alt.Y("share:Q", title="% of area", stack="zero", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("class:N", sort=list(names),
                        scale=alt.Scale(domain=list(names), range=[line_color(c) for c in colors]),
                        legend=alt.Legend(orient="bottom", title=None)),
        order=alt.Order("class_order:Q"),
        tooltip=["month", "class", alt.Tooltip("share:Q", format=".1f", title="%")],
    ).transform_calculate(
        class_order=" + ".join(f"(datum['class'] == {json.dumps(n)}) * {i}" for i, n in enumerate(names))
    ).properties(height=380)


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
    ref_overlay = (mapview.png_data_url(mapview.class_overlay(data["ref"], index, colors, 1.0))
                   if data["ref"] is not None else None)
    # every reference map: (name, agreement %, majority baseline %, per-class table) and its map overlay
    # exam brief: the case study is checked against an independent map not used for training. Esri is independent
    # for models trained on WorldCover (+ Dynamic World agreement), not for models trained on the 3-map vote.
    esri_name = "Esri (used in training: vote labels)" if card.get("training_labels") == "vote" else "Esri (independent)"
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
                ref_overlay=ref_overlay, ref_checks=ref_checks, ref_overlays=ref_overlays, bounds=bounds,
                corners=mapview.utm_corners(data["epsg"], data["x0"], data["y_top"], n),
                files=files, log_dir=log_dir, model=model_name, weights=weight_name, place_name=place_name,
                size_km=n * mapview.PIXEL_M / 1000)


def base_map(center, zoom, basemap):
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    if basemap == "Satellite":
        folium.TileLayer(SATELLITE_TILES, attr=SATELLITE_ATTR, name="Satellite", max_zoom=19).add_to(m)
    else:
        folium.TileLayer("OpenStreetMap", name="Streets").add_to(m)
    return m


def show_map(center, zoom, basemap, layers, height=560):
    """Draws the map with the given overlay layers; a click on the map moves the analysis area there."""
    group = folium.FeatureGroup(name="overlays")
    for layer in layers:
        layer.add_to(group)
    out = st_folium(base_map(center, zoom, basemap), feature_group_to_add=group, key="map", height=height,
                    use_container_width=True, returned_objects=["last_clicked"])
    click = (out or {}).get("last_clicked")
    if click and click != st.session_state.get("handled_click"):
        st.session_state["handled_click"] = click
        st.session_state["pending_loc"] = (click["lat"], click["lng"])
        st.rerun()


def box_layer(corners, color, dashed=False, tooltip=None):
    return folium.Polygon(corners, color=color, weight=2.5, fill=False, dash_array="8 6" if dashed else None,
                          tooltip=tooltip)


def legend_html(names, colors):
    items = "".join(f"<span style='display:inline-flex;align-items:center;margin-right:14px'>"
                    f"<span style='width:13px;height:13px;border-radius:3px;background:{c};"
                    f"border:1px solid #8888;margin-right:5px'></span>{n}</span>" for n, c in zip(names, colors))
    return f"<div style='font-size:0.9rem;line-height:1.9'>{items}</div>"


# ---------------------------------------------------------------- location state
st.session_state.setdefault("lat_text", f"{PRESETS[list(PRESETS)[1]][0]:.4f}")
st.session_state.setdefault("lon_text", f"{PRESETS[list(PRESETS)[1]][1]:.4f}")
st.session_state.setdefault("preset", list(PRESETS)[1])
if "pending_loc" in st.session_state:  # a map click from the previous run
    plat, plon = st.session_state.pop("pending_loc")
    st.session_state["lat_text"], st.session_state["lon_text"] = f"{plat:.5f}", f"{plon:.5f}"
    st.session_state["preset"] = CUSTOM


def apply_preset():
    point = PRESETS[st.session_state["preset"]]
    if point:
        st.session_state["lat_text"], st.session_state["lon_text"] = f"{point[0]:.4f}", f"{point[1]:.4f}"


# ---------------------------------------------------------------- sidebar: advanced
with st.sidebar:
    st.header("Advanced")
    st.caption("The final model is used by default. Change it here only to compare other runs.")
    final_dir = ROOT / "Weights" / FINAL_RUN
    model_choice = st.radio("Model", [f"Final model ({FINAL_RUN})", "Other run folder", "Upload files"])
    weight_bytes, card_text, weight_name = None, None, None
    if model_choice.startswith("Final"):
        if (final_dir / "final_model.pt").exists() and (final_dir / "model_card.json").exists():
            weight_bytes = (final_dir / "final_model.pt").read_bytes()
            card_text = (final_dir / "model_card.json").read_text()
            weight_name = f"{FINAL_RUN}/final_model.pt"
        else:
            st.error(f"`Weights/{FINAL_RUN}/final_model.pt` and `model_card.json` not found on the server. "
                     "Copy them there or choose another option.")
    elif model_choice == "Other run folder":
        runs = sorted(p for p in (ROOT / "Weights").glob("*") if (p / "model_card.json").exists())
        if not runs:
            st.info("No run folder with model_card.json in `Weights/`.")
        else:
            run = st.selectbox("Run folder", runs, format_func=lambda p: p.name)
            pts = sorted(run.glob("*.pt"), key=lambda p: (p.name not in ("final_model.pt", "best_model.pt"), p.name))
            if pts:
                pt = st.selectbox("Weights file", pts, format_func=lambda p: p.name)
                weight_bytes, card_text = pt.read_bytes(), (run / "model_card.json").read_text()
                weight_name = f"{run.name}/{pt.name}"
    else:
        up_w = st.file_uploader("Weights (`final_model.pt`, `best_model.pt` or `<variant>_best.pt`)",
                                type=["pt", "pth"])
        up_c = st.file_uploader("`model_card.json` from the same run", type=["json"])
        if up_w and up_c:
            weight_bytes, card_text, weight_name = up_w.getvalue(), up_c.getvalue().decode("utf-8"), up_w.name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = variant = card = None
    if weight_bytes:
        try:
            card = json.loads(card_text)
            key = hashlib.sha1(weight_bytes).hexdigest() + hashlib.sha1(card_text.encode()).hexdigest()
            net, variant = get_model(key, weight_bytes, card_text, device.type)
            st.success(f"`{weight_name}`: {variant['name']}, {variant['encoder']} encoder, "
                       f"{3 * variant['n_months'] + variant['n_static']} input channels"
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
           + (f"Model: {weight_name}" + (f" (rural test mIoU {card['test_rural_mIoU']:.3f})"
                                          if card and card.get("test_rural_mIoU") else "") if card else ""))

result = st.session_state.get("result")
if result is not None and weight_name and result["weights"] != weight_name:
    # results are kept between reruns; a model change in the sidebar does not recompute them
    st.warning(f"The maps below were made with **{result['weights']}**, not with the selected **{weight_name}**. "
               "Press **Run** to compute them with the selected model.")
with st.container(border=True):
    if use_gee:
        c = st.columns([3.2, 1.3, 1.3, 1, 1.1, 1.5, 1.1], vertical_alignment="bottom")
        c[0].selectbox("Place", list(PRESETS), key="preset", on_change=apply_preset)
        # text boxes instead of number inputs: no +/- buttons, coordinates are usually pasted
        lat = parse_degrees(c[1].text_input("Latitude", key="lat_text"), -60, 72)  # Sentinel-1 IW coverage
        lon = parse_degrees(c[2].text_input("Longitude", key="lon_text"), -180, 180)
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
                st.write("1/3 Fetching Sentinel-1 monthly composites and the Esri reference map from Earth Engine")
                data = gee_data(project, float(lat), float(lon), int(year), int(month), size, variant["terrain"],
                                variant["iteration"])
                settings = dict(source="Google Earth Engine", start_year=int(year), start_month=int(month),
                                area_px=size, patch_stride=stride, device=device.type)
                place = f"{data['lat']:.4f}_{data['lon']:.4f}_{int(year)}-{int(month):02d}"
                place_name = st.session_state["preset"] if st.session_state["preset"] != CUSTOM else "custom"
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
    show_map(center, zoom, basemap, [box_layer(planned, "#ffd92f", True, "next analysis area")] if planned else [])
    st.info("Choose a place (preset, coordinates or a click on the map), then press **Run**.")
    st.stop()


# ---------------------------------------------------------------- viewer: map + radar, one month slider
def viewer():
    r = st.session_state["result"]
    labels, names, colors = r["labels"], r["names"], r["colors"]
    if st.session_state.get("playing"):
        st.session_state["month_idx"] = (st.session_state.get("month_idx", 0) + 1) % 12

    c = st.columns([4.2, 1.1, 1.8, 2.2, 1.6], vertical_alignment="bottom")
    k = c[0].select_slider("Month", options=list(range(12)), format_func=lambda i: labels[i], key="month_idx")
    playing = st.session_state.get("playing", False)
    if c[1].button("Pause" if playing else "Play", width="stretch",
                   help="Step through the 12 months, about one month per second."):
        st.session_state["playing"] = not playing
        st.rerun()
    opacity = c[2].slider("Map overlay opacity", 0, 100, 65, 5, key="opacity") / 100
    ref_names = list(r.get("ref_overlays", {}))
    layer = c[3].selectbox("Overlay", ["Model, this month", *[f"Reference: {n}" for n in ref_names], "None"],
                           key="overlay")
    basemap = c[4].radio("Basemap", ["Satellite", "Streets"], horizontal=True, key="basemap")

    left, right = st.columns([3, 2])
    with left:
        layers = []
        if layer.startswith("Model"):
            layers.append(folium.raster_layers.ImageOverlay(r["overlays"][k], r["bounds"], opacity=opacity,
                                                            interactive=False))
        elif layer.startswith("Reference: "):
            layers.append(folium.raster_layers.ImageOverlay(r["ref_overlays"][layer.removeprefix("Reference: ")],
                                                            r["bounds"], opacity=opacity,
                                                            interactive=False))
        layers.append(box_layer(r["corners"], "#ffffff", tooltip="analysed area"))
        if planned is not None and planned != r["corners"]:
            layers.append(box_layer(planned, "#ffd92f", True, "next analysis area (press Run)"))
        show_map(center, zoom, basemap, layers)
        st.markdown(legend_html(names, colors), unsafe_allow_html=True)
        st.caption("The basemap is a static, recent mosaic of mixed dates (Esri World Imagery / OpenStreetMap) for "
                   "orientation only: it does not change with the month. Only the overlay and the radar image are "
                   "from the selected month.")
        if not ref_names:
            st.caption("No reference map for this run.")

    with right:
        data = r["data"]
        count = data["counts"][N_BEFORE + k] if data["counts"] is not None else None
        st.markdown(f"**Radar input, {labels[k]}**" + (f" · {count} scene{'s' if count != 1 else ''}"
                                                         if count is not None else ""))
        if count == 0:
            st.warning("No Sentinel-1 scene in this month: the input is treated as missing.")
        st.image(false_colour(data["codes"][N_BEFORE + k], r["card"]), width="stretch", clamp=True,
                 caption="R = VV, G = VH, B = VV−VH (dB), same area as the box, north up")
        view = st.radio("Bars", ["Class share this month", "Change vs previous month"], horizontal=True,
                        key="bars_view", label_visibility="collapsed")
        if view.startswith("Class") or k == 0:
            if view.startswith("Change"):
                st.caption("The first month has no previous month; showing the class share.")
            st.altair_chart(class_bars(r["pct"].iloc[k].values, names, colors, "%"), width="stretch")
        else:
            st.altair_chart(class_bars(r["steps"].iloc[k - 1].values, names, colors, "pp"),
                            width="stretch")
            st.caption(f"{labels[k - 1]} → {labels[k]}, percentage points of the area.")


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
st.caption(f"Model of this result: {r['weights']} ({r['model']}) · {r['place_name']}")
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

tab_year, tab_change, tab_check, tab_files, tab_help = st.tabs(
    ["Year overview", "Change", "Reference check", "Downloads and run log", "How to read the results"])

with tab_year:
    st.altair_chart(year_chart(pct, names, colors), width="stretch")
    st.dataframe(pct.round(1), width="stretch")

with tab_change:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Total change, {total.name}**")
        st.altair_chart(class_bars(total.values, names, colors, "pp"), width="stretch")
    with c2:
        st.markdown("**Flicker per month step**")
        st.bar_chart(pd.Series(r["flicker"], index=list(r["steps"].index), name="% of pixels"), height=300)
    st.markdown("**Month-to-month change (percentage points)**")
    lim = max(float(r["steps"].abs().to_numpy().max()), 0.1)
    st.dataframe(r["steps"].style.format("{:+.2f}").background_gradient(cmap="RdBu", vmin=-lim, vmax=lim),
                 width="stretch")
    st.caption("Read with care: the training label is annual, so a month-to-month change is often a seasonal radar "
               "effect (snow, wet soil, harvest, flooding), not a real land-use change. Classification errors that "
               "differ between months add up in these differences.")

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
