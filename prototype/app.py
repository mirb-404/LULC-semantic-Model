"""Streamlit prototype: 12 monthly LULC maps from Sentinel-1 for any location.

Run from this folder:  streamlit run app.py
"""
import hashlib
import hmac
import io
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import pipeline

ROOT = Path(__file__).resolve().parent.parent
N_BEFORE = 2  # months fetched before the start month, for the three-month window
SIZES = {"2.56 km (256 px)": 256, "5.12 km (512 px)": 512, "10.24 km (1024 px, training tile size)": 1024}
OVERLAPS = {"none (fastest)": 256, "25 %": 192, "50 % (smoothest)": 128}

st.set_page_config(page_title="Sentinel-1 LULC", layout="wide")

# launch.py sets an access key when the app is exposed through a public tunnel
ACCESS_KEY = os.environ.get("APP_ACCESS_KEY")
if ACCESS_KEY and not hmac.compare_digest(st.query_params.get("key", ""), ACCESS_KEY):
    st.error("Access key missing or wrong. Open the full link printed by launch.py.")
    st.stop()


# ---------------------------------------------------------------- helpers
def shift_month(year, month, k):
    i = year * 12 + month - 1 + k
    return i // 12, i % 12 + 1


def month_label(year, month):
    return f"{pipeline.MONTH_NAMES[month - 1]} {year}"


def default_project():
    if os.environ.get("EE_PROJECT"):
        return os.environ["EE_PROJECT"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "Project_ID":
                return value.strip().strip("\"'")
    return ""


@st.cache_resource(show_spinner="Loading model ...")
def get_model(key, _weight_bytes, card_text, device_name):
    return pipeline.load_model(_weight_bytes, json.loads(card_text), torch.device(device_name))


@st.cache_data(show_spinner=False)
def gee_data(project, lat, lon, year, month, n, need_terrain):
    import gee
    gee.initialize(project)
    months_all = [shift_month(year, month, k) for k in range(-N_BEFORE, 12)]
    epsg, x0, y_top, region = gee.grid(lat, lon, n)
    counts = gee.scene_counts(region, months_all)
    image, bands = gee.s1_stack(region, months_all)
    codes = gee.fetch(image, bands, epsg, x0, y_top, n, np.uint8, 0).reshape(len(months_all), 2, n, n)
    terrain = None
    if need_terrain:
        terrain = gee.fetch(gee.terrain_image(region), gee.TERRAIN_BANDS, epsg, x0, y_top, n, np.int16, 0)
    ref_year = Counter(y for y, _ in months_all[N_BEFORE:]).most_common(1)[0][0]
    try:
        esri, used_year = gee.esri_image(ref_year)
        ref = gee.fetch(esri, ["esri"], epsg, x0, y_top, n, np.uint8, 0)[0]
        ref_name = f"Esri 10 m Annual LULC {used_year} (independent, Sentinel-2 based)"
    except Exception as err:
        ref, ref_name = None, f"Esri LULC not available: {err}"
    return dict(codes=codes, months_all=months_all, counts=counts, terrain=terrain, ref=ref,
                ref_name=ref_name, lat=lat, lon=lon)


def local_sites():
    return sorted(p.stem for p in (ROOT / "Dataset" / "s1").glob("*.tif"))


@st.cache_data(show_spinner=False)
def local_data(site, month):
    """A downloaded 2021 dataset tile. Months before the start wrap within 2021, as in training."""
    import rasterio
    from pyproj import Transformer
    with rasterio.open(ROOT / "Dataset" / "s1" / f"{site}.tif") as src:
        tile = src.read().reshape(12, 2, src.height, src.width)
        cx, cy = (src.bounds.left + src.bounds.right) / 2, (src.bounds.bottom + src.bounds.top) / 2
        lon, lat = Transformer.from_crs(src.crs, 4326, always_xy=True).transform(cx, cy)
    idx = [(month - 1 + k) % 12 for k in range(-N_BEFORE, 12)]
    terrain_path = ROOT / "Dataset" / "terrain" / f"{site}.tif"
    terrain = None
    if terrain_path.exists():
        with rasterio.open(terrain_path) as src:
            terrain = src.read()
    ref, ref_name = None, "no label file"
    label_path = ROOT / "Dataset" / "labels" / f"{site}.tif"
    if label_path.exists():
        with rasterio.open(label_path) as src:
            ref = src.read(1)
        ref_name = "training consensus label (WorldCover ∩ Dynamic World 2021): NOT independent, test use only"
    return dict(codes=tile[idx], months_all=[(2021, i + 1) for i in idx], counts=None, terrain=terrain,
                ref=ref, ref_name=ref_name, lat=lat, lon=lon)


def false_colour(codes_2hw, card):
    lut = pipeline.db_lut(card)
    vv, vh = lut[codes_2hw[0]], lut[codes_2hw[1]]
    rgb = np.stack([vv, vh, vv - vh], -1)
    for b, (lo, hi) in enumerate([(-20, 0), (-27, -7), (2, 15)]):
        rgb[..., b] = np.clip((rgb[..., b] - lo) / (hi - lo), 0, 1)
    return np.nan_to_num(rgb)


# ---------------------------------------------------------------- sidebar: model
st.sidebar.header("1. Model")
weight_source = st.sidebar.radio("Weights", ["Upload files", "Local Weights folder"], horizontal=True)
weight_bytes, card_text, weight_name = None, None, None
if weight_source == "Upload files":
    up_w = st.sidebar.file_uploader("Weights (`<variant>_best.pt`)", type=["pt", "pth"])
    up_c = st.sidebar.file_uploader("`model_card.json` from the same run", type=["json"])
    if up_w and up_c:
        weight_bytes, card_text, weight_name = up_w.getvalue(), up_c.getvalue().decode("utf-8"), up_w.name
else:
    runs = sorted(p for p in (ROOT / "Weights").glob("*") if (p / "model_card.json").exists())
    if not runs:
        st.sidebar.info("No run folder with model_card.json in `Weights/`. Upload files instead.")
    else:
        run = st.sidebar.selectbox("Run folder", runs, format_func=lambda p: p.name)
        pts = sorted(run.glob("*.pt"))
        if pts:
            pt = st.sidebar.selectbox("Weights file", pts, format_func=lambda p: p.name)
            weight_bytes, card_text, weight_name = pt.read_bytes(), (run / "model_card.json").read_text(), pt.name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = variant = card = None
if weight_bytes:
    try:
        card = json.loads(card_text)
        key = hashlib.sha1(weight_bytes).hexdigest() + hashlib.sha1(card_text.encode()).hexdigest()
        net, variant = get_model(key, weight_bytes, card_text, device.type)
        st.sidebar.success(f"`{weight_name}` → **{variant['name']}** ({3 * variant['n_months']} monthly + "
                           f"{int(variant['pca']) * 6 + int(variant['stats']) * 5 + int(variant['terrain']) * 3} "
                           f"static channels) on {device.type.upper()}")
        if variant["terrain"]:
            st.sidebar.warning("This variant also uses terrain (DEM, HAND): not Sentinel-1 only. "
                               "The brief's deliverable is a Sentinel-1-only model.")
    except Exception as err:
        st.sidebar.error(f"Could not load the model: {err}")
        net = None

# ---------------------------------------------------------------- sidebar: location and time
st.sidebar.header("2. Location and time")
source = st.sidebar.radio("Data", ["Google Earth Engine", "Local dataset tile (offline test)"])
if source == "Google Earth Engine":
    # the default stays on the server: a prefilled value would be sent to the browser
    project = st.sidebar.text_input("Earth Engine project ID", type="password",
                                    placeholder="blank = server default" if default_project() else "",
                                    help="Google Cloud project registered for Earth Engine. "
                                         "Blank: EE_PROJECT or Project_ID in the repo's .env") or default_project()
    c1, c2 = st.sidebar.columns(2)
    lat = c1.number_input("Latitude", -60.0, 72.0, 52.5200, step=0.01, format="%.4f")
    lon = c2.number_input("Longitude", -180.0, 180.0, 13.4050, step=0.01, format="%.4f")
    c1, c2 = st.sidebar.columns(2)
    year = c1.number_input("Start year", 2015, 2025, 2021, step=1)
    month = c2.selectbox("Start month", range(1, 13), format_func=lambda m: pipeline.MONTH_NAMES[m - 1])
    size = SIZES[st.sidebar.selectbox("Area", list(SIZES), index=1)]
else:
    sites = local_sites()
    site = st.sidebar.selectbox("Site (Dataset/s1)", sites) if sites else None
    if not sites:
        st.sidebar.info("No tiles in `Dataset/s1/`.")
    month = st.sidebar.selectbox("Start month (2021)", range(1, 13), format_func=lambda m: pipeline.MONTH_NAMES[m - 1])
stride = OVERLAPS[st.sidebar.select_slider("Patch overlap", list(OVERLAPS), value="25 %")]
run_clicked = st.sidebar.button("Run", type="primary", disabled=net is None)

# ---------------------------------------------------------------- run
st.title("Sentinel-1 land use / land cover: 12 monthly maps")
st.caption("U-Net (ResNet-34) trained on Sentinel-1 monthly composites. Classes: "
           "Built-up, Cropland, Trees, Shrub & grass, Bare, Water, Wetland, Snow & ice.")

if run_clicked:
    try:
        with st.spinner("Fetching Sentinel-1 monthly composites ..."):
            if source == "Google Earth Engine":
                if not project:
                    st.error("Enter an Earth Engine project ID.")
                    st.stop()
                data = gee_data(project, float(lat), float(lon), int(year), int(month), size, variant["terrain"])
            else:
                if not site:
                    st.stop()
                data = local_data(site, int(month))
        if variant["terrain"] and data["terrain"] is None:
            st.error("This variant needs terrain, which is not available for this tile.")
            st.stop()
        bar = st.progress(0.0, "running the model ...")
        preds = pipeline.run_inference(net, variant, card, data["codes"], data["months_all"], N_BEFORE,
                                       data["terrain"], device, stride, progress=bar.progress)
        bar.empty()
        valid = (data["codes"][N_BEFORE:, 0] > 0).any(0)
        st.session_state["result"] = dict(data=data, preds=preds, valid=valid, variant=variant["name"],
                                          card=card, weights=weight_name)
    except Exception as err:
        st.error(f"Run failed: {err}")
        if source == "Google Earth Engine":
            st.info("Earth Engine needs a one-time login: run `earthengine authenticate` in a terminal, "
                    "and the project must be registered for Earth Engine.")
        st.stop()

result = st.session_state.get("result")
if result is None:
    st.info("Choose weights and a location in the sidebar, then press **Run**.")
    st.stop()

# ---------------------------------------------------------------- results
data, preds, valid, card = result["data"], result["preds"], result["valid"], result["card"]
names, colors = card["class_names"], card["class_colors"]
cmap = ListedColormap(["#000000"] + colors)
legend = [Patch(color=c, label=n) for n, c in zip(["no data"] + names, ["#000000"] + colors)]
months = data["months_all"][N_BEFORE:]
labels = [month_label(y, m) for y, m in months]
pct = pipeline.class_percentages(preds, valid, names, labels)
steps, total = pipeline.month_changes(pct)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Location", f"{data['lat']:.4f}, {data['lon']:.4f}")
c2.metric("Period", f"{labels[0]} – {labels[-1]}")
c3.metric("Model", result["variant"])
c4.metric("Area", f"{preds.shape[-1] * 10 / 1000:.2f} km × {preds.shape[-2] * 10 / 1000:.2f} km")
st.map(pd.DataFrame({"lat": [data["lat"]], "lon": [data["lon"]]}), zoom=9, height=220)

if data["counts"] is not None:
    empty = [month_label(y, m) for (y, m), c in zip(data["months_all"], data["counts"]) if c == 0]
    if empty:
        st.warning(f"No Sentinel-1 scenes for: {', '.join(empty)}. These inputs are treated as missing.")
if valid.mean() < 0.99:
    st.warning(f"{(1 - valid.mean()) * 100:.1f} % of the area has no radar data and is excluded (black).")

tab_maps, tab_share, tab_change, tab_check, tab_radar = st.tabs(
    ["Monthly maps", "Class shares", "Change", "Sanity check", "Radar input"])

with tab_maps:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12.5))
    for ax, p, lab in zip(axes.ravel(), preds, labels):
        ax.imshow(np.where(valid, p, 0), cmap=cmap, vmin=0, vmax=len(names), interpolation="nearest")
        ax.set_title(lab)
        ax.axis("off")
    fig.legend(handles=legend, loc="lower center", ncol=len(legend))
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    st.pyplot(fig)
    png = io.BytesIO()
    fig.savefig(png, dpi=110, bbox_inches="tight")
    st.download_button("Download maps (PNG)", png.getvalue(), "lulc_12_months.png", "image/png")
    plt.close(fig)

with tab_share:
    st.subheader("Share of the area per class (%)")
    st.dataframe(pct.round(2))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, color in zip(names, colors):
        ax.plot(labels, pct[name], marker="o", color=color if color != "#f0f0f0" else "#9e9e9e", label=name)
    ax.set_ylabel("% of area")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.download_button("Download class shares (CSV)", pct.round(4).to_csv(), "class_percentages.csv", "text/csv")

with tab_change:
    st.subheader("Month-to-month change (percentage points)")
    st.dataframe(steps.round(2))
    st.subheader(f"Total change {total.name} (percentage points)")
    st.dataframe(total.round(2).to_frame("pp"))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(names, total.values, color=["#9e9e9e" if c == "#f0f0f0" else c for c in colors], edgecolor="black")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("pp")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Read with care: the label is annual, so a change from month to month is often a seasonal "
               "radar effect (snow, wet soil, harvest, flooding), not a real land-use change. Classification "
               "errors that differ between months add up in these differences and inflate the change.")
    st.download_button("Download changes (CSV)",
                       pd.concat([steps, total.to_frame().T]).round(4).to_csv(), "class_changes.csv", "text/csv")

with tab_check:
    st.subheader("Comparison with a reference map")
    st.write(data["ref_name"])
    if data["ref"] is None:
        st.info("No reference map available.")
    else:
        ref = data["ref"]
        annual = pipeline.annual_mode(preds, len(names))
        both = valid & (ref > 0)
        agree = (annual[both] == ref[both]).mean() * 100 if both.any() else float("nan")
        st.metric("Pixel agreement (most frequent monthly class vs reference)", f"{agree:.1f} %")
        n = max(int(both.sum()), 1)
        area = pd.DataFrame({"model (annual mode) %": [(annual[both] == k + 1).sum() / n * 100 for k in range(len(names))],
                             "reference %": [(ref[both] == k + 1).sum() / n * 100 for k in range(len(names))],
                             "reference pixels found by model %": [
                                 ((annual == k + 1) & (ref == k + 1) & both).sum() / max(((ref == k + 1) & both).sum(), 1) * 100
                                 for k in range(len(names))]}, index=names)
        st.dataframe(area.round(1))
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
        axes[0].imshow(np.where(valid, annual, 0), cmap=cmap, vmin=0, vmax=len(names), interpolation="nearest")
        axes[0].set_title("model: most frequent class over 12 months")
        axes[1].imshow(ref, cmap=cmap, vmin=0, vmax=len(names), interpolation="nearest")
        axes[1].set_title("reference")
        for ax in axes:
            ax.axis("off")
        fig.legend(handles=legend, loc="lower center", ncol=len(legend))
        plt.tight_layout(rect=(0, 0.06, 1, 1))
        st.pyplot(fig)
        plt.close(fig)
        st.caption("The reference maps use their own class definitions (e.g. Esri 'rangeland' ≈ our Shrub & grass, "
                   "'flooded vegetation' ≈ Wetland), so part of the disagreement is definitional.")

with tab_radar:
    k = st.select_slider("Month", options=list(range(12)), format_func=lambda i: labels[i])
    st.image(false_colour(data["codes"][N_BEFORE + k], card), caption=f"Sentinel-1 {labels[k]}: R = VV, G = VH, B = VV-VH",
             clamp=True)
