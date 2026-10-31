"""Sentinel-1 monthly composites of the 2nd iteration, shared by download.py (training data) and the prototype.

Earth Engine's COPERNICUS/S1_GRD is calibrated (sigma0), geometrically terrain-corrected and in dB. On top:
  1. IW scenes with VV and VH, ascending and descending passes,
  2. radiometric terrain flattening: sigma0 -> gamma0 with the angular-based volume model of Vollrath et
     al. 2020 (Remote Sensing 12(11) 1867) on the Copernicus GLO-30 DEM,
  3. layover, radar shadow and incidence angles outside 31-45 deg masked,
  4. monthly mean in linear power (temporal multilooking: the speckle filter), back to dB.
No spatial speckle filter.
"""
import math

import ee

DEG = math.pi / 180


def month_collection(region, year, month):
    start = ee.Date.fromYMD(year, month, 1)
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate(start, start.advance(1, "month"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))


def dem_image(region):
    dems = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").filterBounds(region).select("DEM")
    return dems.mosaic().setDefaultProjection(dems.first().projection())


def slope_aspect(region):
    """Terrain slope and aspect in radians from the GLO-30 DEM."""
    dem = dem_image(region)
    return ee.Terrain.slope(dem).multiply(DEG), ee.Terrain.aspect(dem).multiply(DEG)


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


def monthly_composite(region, year, month, slope, aspect):
    """Terrain-flattened monthly gamma0 in dB, bands VV and VH."""
    linear = month_collection(region, year, month).map(lambda img: flatten_terrain(img, slope, aspect))
    return linear.mean().log10().multiply(10)
