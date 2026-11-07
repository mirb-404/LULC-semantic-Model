"""Puts the UTM prediction grid on a web map (Leaflet / folium) and writes georeferenced exports.

Leaflet stretches an image overlay linearly in Web Mercator between its south-west and north-east corners.
The prediction grid is UTM (gee.grid), which is rotated against Web Mercator by the grid convergence (up to
about 2-3 degrees at the edge of a zone). Over a 10 km tile that is up to a few hundred metres at the corners,
so every overlay pixel is sampled from the UTM grid (nearest neighbour) instead of placing the UTM image as is.
"""
import base64
import io

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.image import imsave
from pyproj import Transformer

PIXEL_M = 10


def utm_epsg(lat, lon):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def utm_window(lat, lon, n):
    """UTM grid of n x n pixels (10 m) centred on lat/lon, snapped to the pixel size: (epsg, x0, y_top)."""
    epsg = utm_epsg(lat, lon)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    half = n * PIXEL_M / 2
    return epsg, round((x - half) / PIXEL_M) * PIXEL_M, round((y + half) / PIXEL_M) * PIXEL_M


def utm_corners(epsg, x0, y_top, n):
    """Lat/lon of the 4 tile corners (NW, NE, SE, SW), as [[lat, lon], ...] for a folium polygon."""
    xs = x0 + np.array([0, n, n, 0]) * PIXEL_M
    ys = y_top - np.array([0, 0, n, n]) * PIXEL_M
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(xs, ys)
    return [[float(a), float(o)] for a, o in zip(lat, lon)]


def overlay_index(epsg, x0, y_top, n, size=None):
    """Sampling index for a Web Mercator overlay of the n x n UTM grid.

    Returns (index, bounds): index (size, size) holds the flat UTM pixel (row * n + col) shown at each overlay
    pixel, -1 outside the tile; bounds [[south, west], [north, east]] for folium.ImageOverlay.
    """
    size = size or n
    corners = np.array(utm_corners(epsg, x0, y_top, n))
    south, north = corners[:, 0].min(), corners[:, 0].max()
    west, east = corners[:, 1].min(), corners[:, 1].max()
    to_merc = Transformer.from_crs(4326, 3857, always_xy=True)
    mx0, my0 = to_merc.transform(west, south)
    mx1, my1 = to_merc.transform(east, north)
    # pixel centres, top row first (images are drawn north up)
    gx = mx0 + (np.arange(size) + 0.5) * (mx1 - mx0) / size
    gy = my1 - (np.arange(size) + 0.5) * (my1 - my0) / size
    mx, my = np.meshgrid(gx, gy)
    ux, uy = Transformer.from_crs(3857, epsg, always_xy=True).transform(mx.ravel(), my.ravel())
    col = np.floor((ux - x0) / PIXEL_M).astype(np.int64)
    row = np.floor((y_top - uy) / PIXEL_M).astype(np.int64)
    inside = (row >= 0) & (row < n) & (col >= 0) & (col < n)
    index = np.where(inside, row * n + col, -1).reshape(size, size)
    return index, [[float(south), float(west)], [float(north), float(east)]]


def palette(colors, opacity):
    """RGBA lookup table: id 0 (no data) transparent, ids 1..K in the class colours at the given opacity."""
    table = np.zeros((len(colors) + 1, 4), np.uint8)
    table[1:, :3] = (np.array([to_rgb(c) for c in colors]) * 255).round()
    table[1:, 3] = round(opacity * 255)
    return table


def class_overlay(class_map, index, colors, opacity):
    """class_map (n, n) with ids 0..K -> RGBA (size, size) overlay image."""
    flat = class_map.ravel()
    ids = np.where(index >= 0, flat[np.clip(index, 0, None)], 0)
    return palette(colors, opacity)[ids]


def rgb_overlay(rgb, index):
    """rgb (3, n, n) uint8 -> RGBA (size, size) overlay image; outside the tile and all-zero pixels transparent."""
    flat = rgb.reshape(3, -1)
    pix = flat[:, np.clip(index, 0, None)].transpose(1, 2, 0)
    alpha = ((index >= 0) & (pix.max(-1) > 0)).astype(np.uint8) * 255
    return np.dstack([pix, alpha])


# Esri World Imagery Wayback: every published version of the Esri satellite basemap since 2014-02.
WAYBACK_CONFIG = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
# metadata layers 4-8 = imagery of 0.3-2.4 m, the photos drawn at the zoom levels of a 2.5-10 km analysis area
WAYBACK_METADATA_LAYERS = range(4, 9)


def _get_json(url, timeout=15, headers=None):
    import json
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=timeout) as resp:
        return json.load(resp)


# OpenStreetMap Nominatim: free place search; its usage policy asks for an identifying User-Agent and at most
# 1 request per second (one request per search here, cached by the app).
NOMINATIM = "https://nominatim.openstreetmap.org/search"


def geocode(query, limit=5):
    """Places matching a free-text query (town, region, address): [(name, lat, lon)], best match first."""
    from urllib.parse import urlencode
    url = f"{NOMINATIM}?{urlencode(dict(q=query, format='jsonv2', limit=limit))}"
    hits = _get_json(url, headers={"User-Agent": "Sentinel1-LULC-Explorer/1.0 (university prototype)"})
    return [(h["display_name"], float(h["lat"]), float(h["lon"])) for h in hits]


def wayback_releases():
    """All Wayback releases, oldest first: [(release date 'YYYY-MM-DD', folium tile URL, metadata URL)]."""
    releases = []
    for item in _get_json(WAYBACK_CONFIG).values():
        date = item["itemTitle"].rsplit(" ", 1)[-1].rstrip(")")
        url = item["itemURL"].replace("{level}", "{z}").replace("{row}", "{y}").replace("{col}", "{x}")
        releases.append((date, url, item["metadataLayerUrl"]))
    return sorted(releases)


def wayback_release_for(releases, end_date):
    """The last release published before end_date ('YYYY-MM-DD', the day after the analysis period).

    Its photos were taken before end_date, so they show the land as it was during or before the analysis
    period, never after it; the earliest release if the period is before 2014-02.
    """
    before = [r for r in releases if r[0] < end_date]
    return before[-1] if before else releases[0]


def wayback_capture(metadata_url, lat, lon):
    """Capture date ('YYYY-MM-DD'), resolution (m) and source of the release's photo at lat/lon, or None."""
    from datetime import datetime, timezone
    for layer in WAYBACK_METADATA_LAYERS:
        query = (f"{metadata_url}/{layer}/query?geometry={lon},{lat}&geometryType=esriGeometryPoint&inSR=4326"
                 "&spatialRel=esriSpatialRelIntersects&outFields=SRC_DATE2,SRC_RES,NICE_DESC"
                 "&returnGeometry=false&f=json")
        features = _get_json(query).get("features") or []
        if features:
            a = features[0]["attributes"]
            if a.get("SRC_DATE2") is None:
                continue
            date = datetime.fromtimestamp(a["SRC_DATE2"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            return date, a.get("SRC_RES"), a.get("NICE_DESC")
    return None


def png_data_url(rgba):
    buf = io.BytesIO()
    imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def geotiff_bytes(stack, epsg, x0, y_top, band_names):
    """(bands, H, W) uint8 class maps -> GeoTIFF bytes in the UTM grid of the run (0 = no data)."""
    import rasterio
    from rasterio.transform import from_origin

    profile = dict(driver="GTiff", width=stack.shape[2], height=stack.shape[1], count=stack.shape[0],
                   dtype="uint8", crs=f"EPSG:{epsg}", transform=from_origin(x0, y_top, PIXEL_M, PIXEL_M),
                   nodata=0, compress="deflate")
    with rasterio.MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(stack.astype(np.uint8))
            for i, name in enumerate(band_names, 1):
                dst.set_band_description(i, name)
        return mem.read()
