"""Shared settings for the dataset pipeline (Sentinel-1 inputs + WorldCover/Dynamic World consensus labels)."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Dataset"
S1_DIR = DATA_DIR / "s1"
LABEL_DIR = DATA_DIR / "labels"
PATCH_DIR = DATA_DIR / "patches"
PREVIEW_DIR = DATA_DIR / "previews"

YEAR = 2021
TILE_SIZE_M = 10_240
PIXEL_M = 10
PATCH = 256
NODATA = 0
DB_MIN = -35.0
DB_STEP = 0.2


def decode_db(codes):
    """uint8 codes -> float32 dB (NaN where no data)."""
    db = codes.astype(np.float32) * DB_STEP + DB_MIN
    db[codes == NODATA] = np.nan
    return db

URBAN_SITES = [
    dict(name="london",        lat=51.5074, lon=-0.1278,   region="europe",   split="train"),
    dict(name="paris",         lat=48.8566, lon=2.3522,    region="europe",   split="train"),
    dict(name="berlin",        lat=52.5200, lon=13.4050,   region="europe",   split="train"),
    dict(name="madrid",        lat=40.4168, lon=-3.7038,   region="europe",   split="train"),
    dict(name="rome",          lat=41.9028, lon=12.4964,   region="europe",   split="train"),
    dict(name="warsaw",        lat=52.2297, lon=21.0122,   region="europe",   split="train"),
    dict(name="stockholm",     lat=59.3293, lon=18.0686,   region="europe",   split="train"),
    dict(name="amsterdam",     lat=52.3676, lon=4.9041,    region="europe",   split="train"),
    dict(name="lisbon",        lat=38.7223, lon=-9.1393,   region="europe",   split="train"),
    dict(name="kyiv",          lat=50.4501, lon=30.5234,   region="europe",   split="train"),
    dict(name="innsbruck",     lat=47.2692, lon=11.4041,   region="europe",   split="train"),
    dict(name="reykjavik",     lat=64.1466, lon=-21.9426,  region="europe",   split="train"),
    dict(name="budapest",      lat=47.4979, lon=19.0402,   region="europe",   split="val"),
    dict(name="helsinki",      lat=60.1699, lon=24.9384,   region="europe",   split="val"),
    dict(name="frankfurt",     lat=50.1109, lon=8.6821,    region="europe",   split="test"),
    dict(name="athens",        lat=37.9838, lon=23.7275,   region="europe",   split="test"),
    dict(name="new_york",      lat=40.7128, lon=-74.0060,  region="n_america", split="train"),
    dict(name="los_angeles",   lat=34.0522, lon=-118.2437, region="n_america", split="train"),
    dict(name="houston",       lat=29.7604, lon=-95.3698,  region="n_america", split="train"),
    dict(name="toronto",       lat=43.6532, lon=-79.3832,  region="n_america", split="train"),
    dict(name="mexico_city",   lat=19.4326, lon=-99.1332,  region="n_america", split="train"),
    dict(name="denver",        lat=39.7392, lon=-104.9903, region="n_america", split="train"),
    dict(name="miami",         lat=25.7617, lon=-80.1918,  region="n_america", split="train"),
    dict(name="minneapolis",   lat=44.9778, lon=-93.2650,  region="n_america", split="train"),
    dict(name="new_orleans",   lat=29.9511, lon=-90.0715,  region="n_america", split="train"),
    dict(name="atlanta",       lat=33.7490, lon=-84.3880,  region="n_america", split="val"),
    dict(name="vancouver",     lat=49.2827, lon=-123.1207, region="n_america", split="val"),
    dict(name="chicago",       lat=41.8781, lon=-87.6298,  region="n_america", split="test"),
    dict(name="phoenix",       lat=33.4484, lon=-112.0740, region="n_america", split="test"),
    dict(name="anchorage",     lat=61.2181, lon=-149.9003, region="n_america", split="test"),
    dict(name="sao_paulo",     lat=-23.5505, lon=-46.6333, region="s_america", split="train"),
    dict(name="buenos_aires",  lat=-34.6037, lon=-58.3816, region="s_america", split="train"),
    dict(name="santiago",      lat=-33.4489, lon=-70.6693, region="s_america", split="train"),
    dict(name="manaus",        lat=-3.1190,  lon=-60.0217, region="s_america", split="train"),
    dict(name="brasilia",      lat=-15.7939, lon=-47.8828, region="s_america", split="train"),
    dict(name="quito",         lat=-0.1807,  lon=-78.4678, region="s_america", split="train"),
    dict(name="corumba",       lat=-19.0089, lon=-57.6510, region="s_america", split="train"),
    dict(name="lima",          lat=-12.0464, lon=-77.0428, region="s_america", split="val"),
    dict(name="bogota",        lat=4.7110,   lon=-74.0721, region="s_america", split="test"),
    dict(name="curitiba",      lat=-25.4290, lon=-49.2710, region="s_america", split="test"),
    dict(name="cairo",         lat=30.0444, lon=31.2357,   region="africa",   split="train"),
    dict(name="lagos",         lat=6.5244,  lon=3.3792,    region="africa",   split="train"),
    dict(name="johannesburg",  lat=-26.2041, lon=28.0473,  region="africa",   split="train"),
    dict(name="addis_ababa",   lat=9.0300,  lon=38.7400,   region="africa",   split="train"),
    dict(name="kinshasa",      lat=-4.4419, lon=15.2663,   region="africa",   split="train"),
    dict(name="dakar",         lat=14.7167, lon=-17.4677,  region="africa",   split="train"),
    dict(name="dar_es_salaam", lat=-6.7924, lon=39.2083,   region="africa",   split="train"),
    dict(name="khartoum",      lat=15.5007, lon=32.5599,   region="africa",   split="train"),
    dict(name="casablanca",    lat=33.5731, lon=-7.5898,   region="africa",   split="val"),
    dict(name="lusaka",        lat=-15.3875, lon=28.3228,  region="africa",   split="val"),
    dict(name="nairobi",       lat=-1.2921, lon=36.8219,   region="africa",   split="test"),
    dict(name="accra",         lat=5.6037,  lon=-0.1870,   region="africa",   split="test"),
    dict(name="delhi",         lat=28.6139, lon=77.2090,   region="asia",     split="train"),
    dict(name="mumbai",        lat=19.0760, lon=72.8777,   region="asia",     split="train"),
    dict(name="dhaka",         lat=23.8103, lon=90.4125,   region="asia",     split="train"),
    dict(name="beijing",       lat=39.9042, lon=116.4074,  region="asia",     split="train"),
    dict(name="shanghai",      lat=31.2304, lon=121.4737,  region="asia",     split="train"),
    dict(name="tokyo",         lat=35.6762, lon=139.6503,  region="asia",     split="train"),
    dict(name="jakarta",       lat=-6.2088, lon=106.8456,  region="asia",     split="train"),
    dict(name="manila",        lat=14.5995, lon=120.9842,  region="asia",     split="train"),
    dict(name="ho_chi_minh",   lat=10.8231, lon=106.6297,  region="asia",     split="train"),
    dict(name="riyadh",        lat=24.7136, lon=46.6753,   region="asia",     split="train"),
    dict(name="dubai",         lat=25.2048, lon=55.2708,   region="asia",     split="train"),
    dict(name="almaty",        lat=43.2380, lon=76.9450,   region="asia",     split="train"),
    dict(name="ulaanbaatar",   lat=47.8864, lon=106.9057,  region="asia",     split="train"),
    dict(name="khulna",        lat=22.8456, lon=89.5403,   region="asia",     split="train"),
    dict(name="seoul",         lat=37.5665, lon=126.9780,  region="asia",     split="val"),
    dict(name="karachi",       lat=24.8607, lon=67.0011,   region="asia",     split="val"),
    dict(name="kolkata",       lat=22.5726, lon=88.3639,   region="asia",     split="val"),
    dict(name="bangkok",       lat=13.7563, lon=100.5018,  region="asia",     split="test"),
    dict(name="tehran",        lat=35.6892, lon=51.3890,   region="asia",     split="test"),
    dict(name="chengdu",       lat=30.5728, lon=104.0668,  region="asia",     split="test"),
    dict(name="sydney",        lat=-33.8688, lon=151.2093, region="oceania",  split="train"),
    dict(name="melbourne",     lat=-37.8136, lon=144.9631, region="oceania",  split="train"),
    dict(name="auckland",      lat=-36.8485, lon=174.7633, region="oceania",  split="train"),
    dict(name="perth",         lat=-31.9505, lon=115.8605, region="oceania",  split="val"),
    dict(name="adelaide",      lat=-34.9285, lon=138.6007, region="oceania",  split="test"),
]

RURAL_SITES = [
    dict(name="iowa_corn_belt",     lat=42.3000, lon=-93.9000,  region="n_america", split="train", focus="cropland"),
    dict(name="punjab_india",       lat=30.6000, lon=75.3000,   region="asia",      split="train", focus="cropland"),
    dict(name="poltava_ukraine",    lat=49.3000, lon=33.5000,   region="europe",    split="train", focus="cropland"),
    dict(name="pampas_argentina",   lat=-35.0000, lon=-61.5000, region="s_america", split="train", focus="cropland"),
    dict(name="nile_delta",         lat=30.9500, lon=30.8500,   region="africa",    split="train", focus="cropland"),
    dict(name="mato_grosso_soy",    lat=-13.1000, lon=-56.1000, region="s_america", split="train", focus="cropland"),
    dict(name="mekong_delta_rice",  lat=10.3000, lon=105.5000,  region="asia",      split="train", focus="cropland"),
    dict(name="beauce_france",      lat=48.2500, lon=1.6000,    region="europe",    split="val",   focus="cropland"),
    dict(name="kansas_pivots",      lat=37.9500, lon=-100.7000, region="n_america", split="val",   focus="cropland"),
    dict(name="north_china_plain",  lat=37.0000, lon=115.5000,  region="asia",      split="test",  focus="cropland"),
    dict(name="kano_farmland",      lat=12.2000, lon=8.3000,    region="africa",    split="test",  focus="cropland"),
    dict(name="amazon_forest",      lat=-5.0000, lon=-63.0000,  region="s_america", split="train", focus="trees"),
    dict(name="congo_forest",       lat=1.0000,  lon=23.5000,   region="africa",    split="train", focus="trees"),
    dict(name="black_forest",       lat=48.2000, lon=8.2000,    region="europe",    split="train", focus="trees"),
    dict(name="borneo_forest",      lat=0.9000,  lon=114.0000,  region="asia",      split="train", focus="trees"),
    dict(name="ontario_boreal",     lat=50.0000, lon=-85.0000,  region="n_america", split="val",   focus="trees"),
    dict(name="siberia_taiga",      lat=60.0000, lon=100.0000,  region="asia",      split="test",  focus="trees"),
    dict(name="sahara_algeria",     lat=26.0000, lon=4.0000,    region="africa",    split="train", focus="bare"),
    dict(name="western_desert",     lat=27.0000, lon=29.0000,   region="africa",    split="train", focus="bare"),
    dict(name="badain_jaran",       lat=40.0000, lon=102.0000,  region="asia",      split="train", focus="bare"),
    dict(name="namib_desert",       lat=-24.7500, lon=15.3000,  region="africa",    split="train", focus="bare"),
    dict(name="atacama_desert",     lat=-24.5000, lon=-69.8000, region="s_america", split="val",   focus="bare"),
    dict(name="rub_al_khali",       lat=20.5000, lon=50.5000,   region="asia",      split="test",  focus="bare"),
    dict(name="mongolia_steppe",    lat=47.0000, lon=103.0000,  region="asia",      split="train", focus="shrub_grass"),
    dict(name="texas_shrubland",    lat=30.5000, lon=-101.5000, region="n_america", split="train", focus="shrub_grass"),
    dict(name="kazakh_steppe",      lat=50.5000, lon=70.0000,   region="asia",      split="val",   focus="shrub_grass"),
    dict(name="serengeti",          lat=-2.3300, lon=34.8300,   region="africa",    split="test",  focus="shrub_grass"),
    dict(name="sundarbans",         lat=21.9500, lon=89.1800,   region="asia",      split="train", focus="wetland"),
    dict(name="pantanal",           lat=-17.5000, lon=-56.8000, region="s_america", split="train", focus="wetland"),
    dict(name="sudd_wetland",       lat=7.5000,  lon=30.5000,   region="africa",    split="train", focus="wetland"),
    dict(name="inner_niger_delta",  lat=15.0000, lon=-4.5000,   region="africa",    split="train", focus="wetland"),
    dict(name="okavango_delta",     lat=-19.3000, lon=22.9000,  region="africa",    split="val",   focus="wetland"),
    dict(name="everglades",         lat=25.8000, lon=-80.8000,  region="n_america", split="test",  focus="wetland"),
    dict(name="aletsch_glacier",    lat=46.5000, lon=8.0300,    region="europe",    split="train", focus="snow_ice"),
    dict(name="columbia_icefield",  lat=52.2000, lon=-117.3000, region="n_america", split="train", focus="snow_ice"),
    dict(name="patagonia_icefield", lat=-49.3000, lon=-73.2000, region="s_america", split="train", focus="snow_ice"),
    dict(name="vatnajokull",        lat=64.4000, lon=-16.8000,  region="europe",    split="train", focus="snow_ice"),
    dict(name="svalbard_glacier",   lat=78.6500, lon=16.4000,   region="europe",    split="val",   focus="snow_ice"),
    dict(name="karakoram_baltoro",  lat=35.7500, lon=76.4000,   region="asia",      split="test",  focus="snow_ice"),
]

EXTRA_RURAL_SITES = [
    dict(name="sweden_taiga",         lat=64.0000, lon=17.0000,   region="europe",    split="train", focus="trees"),
    dict(name="alaska_boreal",        lat=65.5000, lon=-148.5000, region="n_america", split="train", focus="trees"),
    dict(name="quebec_boreal",        lat=49.5000, lon=-73.0000,  region="n_america", split="train", focus="trees"),
    dict(name="appalachian_forest",   lat=38.6000, lon=-79.8000,  region="n_america", split="train", focus="trees"),
    dict(name="olympic_rainforest",   lat=47.8000, lon=-123.8000, region="n_america", split="train", focus="trees"),
    dict(name="sikhote_alin_forest",  lat=46.0000, lon=135.5000,  region="asia",      split="train", focus="trees"),
    dict(name="gabon_forest",         lat=-0.5000, lon=11.5000,   region="africa",    split="train", focus="trees"),
    dict(name="tasmania_forest",      lat=-42.5000, lon=146.5000, region="oceania",   split="train", focus="trees"),
    dict(name="papua_forest",         lat=-5.0000, lon=142.0000,  region="oceania",   split="train", focus="trees"),
    dict(name="mont_blanc_massif",    lat=45.8800, lon=6.9500,    region="europe",    split="train", focus="snow_ice"),
    dict(name="jostedalsbreen",       lat=61.7000, lon=7.0000,    region="europe",    split="train", focus="snow_ice"),
    dict(name="juneau_icefield",      lat=58.6000, lon=-134.3000, region="n_america", split="train", focus="snow_ice"),
    dict(name="tien_shan_inylchek",   lat=42.2000, lon=79.9000,   region="asia",      split="train", focus="snow_ice"),
    dict(name="khumbu_himalaya",      lat=27.9500, lon=86.8500,   region="asia",      split="train", focus="snow_ice"),
    dict(name="tasman_glacier",       lat=-43.5000, lon=170.3000, region="oceania",   split="train", focus="snow_ice"),
    dict(name="caucasus_elbrus",      lat=43.3500, lon=42.4500,   region="europe",    split="val",   focus="snow_ice"),
    dict(name="nebraska_sandhills",   lat=42.0000, lon=-101.0000, region="n_america", split="train", focus="shrub_grass"),
    dict(name="great_basin_sagebrush", lat=40.5000, lon=-117.0000, region="n_america", split="train", focus="shrub_grass"),
    dict(name="patagonia_steppe",     lat=-45.5000, lon=-69.5000, region="s_america", split="train", focus="shrub_grass"),
    dict(name="karoo_shrubland",      lat=-32.5000, lon=22.0000,  region="africa",    split="train", focus="shrub_grass"),
    dict(name="tibet_grassland",      lat=31.5000, lon=92.0000,   region="asia",      split="train", focus="shrub_grass"),
    dict(name="saskatchewan_wheat",   lat=51.0000, lon=-105.0000, region="n_america", split="train", focus="cropland"),
    dict(name="central_valley_ca",    lat=36.6000, lon=-120.0000, region="n_america", split="train", focus="cropland"),
    dict(name="punjab_pakistan",      lat=30.0000, lon=72.5000,   region="asia",      split="train", focus="cropland"),
    dict(name="karawang_rice",        lat=-6.3000, lon=107.4000,  region="asia",      split="train", focus="cropland"),
    dict(name="heilongjiang_maize",   lat=46.5000, lon=126.5000,  region="asia",      split="train", focus="cropland"),
    dict(name="australia_wheatbelt",  lat=-31.5000, lon=117.5000, region="oceania",   split="train", focus="cropland"),
    dict(name="hungarian_plain",      lat=46.5000, lon=20.5000,   region="europe",    split="train", focus="cropland"),
    dict(name="hudson_bay_lowland",   lat=53.5000, lon=-84.0000,  region="n_america", split="train", focus="wetland"),
    dict(name="danube_delta",         lat=45.1000, lon=29.4000,   region="europe",    split="train", focus="wetland"),
    dict(name="louisiana_marsh",      lat=29.4000, lon=-90.6000,  region="n_america", split="train", focus="wetland"),
    dict(name="niger_delta_mangrove", lat=4.6000,  lon=6.5000,    region="africa",    split="train", focus="wetland"),
    dict(name="taklamakan_desert",    lat=38.5000, lon=83.0000,   region="asia",      split="train", focus="bare"),
    dict(name="lake_balaton",         lat=46.8500, lon=17.7500,   region="europe",    split="train", focus="water"),
    dict(name="finnish_lakeland",     lat=62.0000, lon=27.5000,   region="europe",    split="train", focus="water"),
]

BIOME_GAP_SITES = [
    dict(name="finnmark_tundra",        lat=70.0000, lon=25.5000,   region="europe",    split="train", focus="tundra"),
    dict(name="north_slope_alaska",     lat=70.0000, lon=-150.0000, region="n_america", split="train", focus="tundra"),
    dict(name="iceland_highland",       lat=64.8000, lon=-18.0000,  region="europe",    split="train", focus="tundra"),
    dict(name="kola_tundra",            lat=68.5000, lon=36.0000,   region="europe",    split="val",   focus="tundra"),
    dict(name="kruger_savanna",         lat=-24.0000, lon=31.6000,  region="africa",    split="train", focus="savanna"),
    dict(name="cerrado_goias",          lat=-13.5000, lon=-46.5000, region="s_america", split="train", focus="savanna"),
    dict(name="llanos_venezuela",       lat=7.5000,  lon=-67.5000,  region="s_america", split="train", focus="savanna"),
    dict(name="miombo_zambia",          lat=-12.5000, lon=30.5000,  region="africa",    split="train", focus="savanna"),
    dict(name="sahel_niger",            lat=12.5000, lon=3.5000,    region="africa",    split="train", focus="savanna"),
    dict(name="top_end_savanna",        lat=-14.5000, lon=132.5000, region="oceania",   split="val",   focus="savanna"),
    dict(name="dehesa_spain",           lat=39.0000, lon=-6.0000,   region="europe",    split="train", focus="mediterranean"),
    dict(name="california_chaparral",   lat=34.3000, lon=-117.4000, region="n_america", split="train", focus="mediterranean"),
    dict(name="chile_matorral",         lat=-32.8000, lon=-71.0000, region="s_america", split="train", focus="mediterranean"),
    dict(name="cape_fynbos",            lat=-34.2000, lon=19.2000,  region="africa",    split="train", focus="mediterranean"),
    dict(name="crete_maquis",           lat=35.2000, lon=24.9000,   region="europe",    split="val",   focus="mediterranean"),
    dict(name="ethiopia_highland_farms", lat=11.0000, lon=37.6000,  region="africa",    split="train", focus="smallholder"),
    dict(name="malawi_farms",           lat=-13.6000, lon=33.8000,  region="africa",    split="train", focus="smallholder"),
    dict(name="rwanda_hills",           lat=-2.2000, lon=29.7000,   region="africa",    split="train", focus="smallholder"),
    dict(name="western_kenya_farms",    lat=0.6000,  lon=34.6000,   region="africa",    split="train", focus="smallholder"),
    dict(name="kampala",                lat=0.3476,  lon=32.5825,   region="africa",    split="train", focus="urban"),
    dict(name="port_au_prince",         lat=18.5944, lon=-72.3074,  region="n_america", split="train", focus="urban"),
    dict(name="nepal_midhills",         lat=27.8000, lon=85.6000,   region="asia",      split="train", focus="steep_terrain"),
    dict(name="andes_sacred_valley",    lat=-13.3000, lon=-72.1000, region="s_america", split="train", focus="steep_terrain"),
    dict(name="gran_chaco",             lat=-22.5000, lon=-62.0000, region="s_america", split="train", focus="dry_forest"),
    dict(name="kanha_deciduous",        lat=22.3000, lon=80.6000,   region="asia",      split="train", focus="dry_forest"),
]

RURAL_SITES = RURAL_SITES + EXTRA_RURAL_SITES + BIOME_GAP_SITES

SITES = [dict(s, focus="urban") for s in URBAN_SITES] + RURAL_SITES

CLASSES = {
    0: ("ignore", "#000000"),
    1: ("Built-up", "#e31a1c"),
    2: ("Cropland", "#ffd92f"),
    3: ("Trees", "#1a9850"),
    4: ("Shrub & grass", "#a6d96a"),
    5: ("Bare / sparse", "#bf812d"),
    6: ("Water", "#2c7fb8"),
    7: ("Wetland & mangrove", "#41b6c4"),
    8: ("Snow & ice", "#f0f0f0"),
}

WORLDCOVER_TO_CLASS = {10: 3, 20: 4, 30: 4, 40: 2, 50: 1, 60: 5, 70: 8, 80: 6, 90: 7, 95: 7, 100: 5}

DYNAMICWORLD_TO_CLASS = {0: 6, 1: 3, 2: 4, 3: 7, 4: 2, 5: 4, 6: 1, 7: 5, 8: 8}

MONTHS = list(range(1, 13))
BAND_NAMES = [f"{pol}_{m:02d}" for m in MONTHS for pol in ("VV", "VH")]
LABEL_BANDS = ["consensus", "worldcover", "dynamicworld", "dynamicworld_snowfree"]

WETLAND = 7
WETLAND_COMPATIBLE_DW = (3, 4, 6, 7)


SNOW = 8


def consensus_labels(wc, dw, dw_snowfree=None):
    """Training label from class-mapped WorldCover and Dynamic World arrays (0 = ignore).

    Strict agreement for every class, except wetland (see WETLAND_COMPATIBLE_DW).
    Seasonal snow: where WorldCover is not snow but the yearly Dynamic World mode is snow,
    the mode over Dynamic World's snow-free observations (dw_snowfree) is used instead.
    """
    if dw_snowfree is not None:
        dw = np.where((dw == SNOW) & (wc != SNOW) & (dw_snowfree > 0), dw_snowfree, dw)
    label = np.where(wc == dw, wc, 0)
    label[(wc == WETLAND) & np.isin(dw, WETLAND_COMPATIBLE_DW)] = WETLAND
    return label.astype(np.uint8)
