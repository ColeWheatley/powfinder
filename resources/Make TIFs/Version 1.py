#!/usr/bin/env python3
"""Unified TIF generator.

This script replaces the many per-variable generator scripts. It loads the
weather dataset and terrain once and then writes GeoTIFF layers for any
requested variables.

Usage:
    python generate_tifs.py [variable [variable ...]]
If no variables are specified all known variables are generated.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Callable, Sequence

import numpy as np
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Configuration
TIMESTAMPS = [
    "2025-05-24T09:00:00", "2025-05-24T12:00:00",
    "2025-05-24T15:00:00", "2025-05-24T18:00:00",
    "2025-05-25T09:00:00", "2025-05-25T12:00:00",
    "2025-05-25T15:00:00", "2025-05-25T18:00:00",
    "2025-05-26T09:00:00", "2025-05-26T12:00:00",
    "2025-05-26T15:00:00", "2025-05-26T18:00:00",
    "2025-05-27T09:00:00", "2025-05-27T12:00:00",
    "2025-05-27T15:00:00", "2025-05-27T18:00:00",
    "2025-05-28T09:00:00", "2025-05-28T12:00:00",
    "2025-05-28T15:00:00", "2025-05-28T18:00:00",
]

WEATHER_PATH = "resources/meteo_api/weather_data_3hour.json"
ELEVATION_TIF = "resources/terrains/tirol_100m_float.tif"
COLOR_SCALE_JSON = "resources/Make TIFs/color_scales.json"
HILLSHADE_TIFS = {
    1: "resources/hillshade/hillshade_100m_period1.tif",
    2: "resources/hillshade/hillshade_100m_period2.tif",
    3: "resources/hillshade/hillshade_100m_period3.tif",
    4: "resources/hillshade/hillshade_100m_period4.tif",
}
PERIOD_FROM_HOUR = {9: 1, 12: 2, 15: 3, 18: 4}
OUTPUT_BASE = "TIFS/100m_resolution"
NODATA = -9999.0

# Variables that can be generated and the units written to the GeoTIFF
VARIABLE_UNITS = {
    "temperature_2m": "celsius",
    "dewpoint_2m": "celsius",
    "relative_humidity_2m": "percent",
    "shortwave_radiation": "W/m^2",
    "cloud_cover": "percent",
    "snowfall": "mm",
    "snow_depth": "m",
    "surface_pressure": "hPa",
    "weather_code": "wmo_code",
    "wind_speed_10m": "m/s",
    "freezing_level_height": "meters",
    "sqh": "sqh",
    "skiability": "ski",
}
ALL_VARIABLES = list(VARIABLE_UNITS.keys())

# ---------------------------------------------------------------------------
# Helper utilities

def load_weather_points(path: str, variables: Sequence[str]):
    """Load weather station data for the given variables."""
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    coords = []
    values = {v: [] for v in variables}
    times = None

    entries = data.get("coordinates", data)
    for entry in entries:
        info = entry.get("coordinate_info", entry)
        elev = info.get("elevation")
        if elev is None or str(elev) == "unknown":
            continue
        try:
            elev = float(str(elev).replace(" m", ""))
        except ValueError:
            continue
        wdata = entry.get("weather_data_3hour", entry.get("hourly"))
        if not wdata or "hourly" not in wdata:
            continue
        hourly = wdata["hourly"]
        if times is None:
            times = hourly["time"]

        coords.append((info["latitude"], info["longitude"], elev))
        for v in variables:
            arr = hourly.get(v)
            if arr is None:
                raise KeyError(f"Weather data missing '{v}'")
            values[v].append(np.array(arr, dtype=np.float32))

    coords = np.array(coords)
    for v in variables:
        values[v] = np.stack(values[v])
    return coords, values, times


def build_grid(src_dataset):
    elev = src_dataset.read(1, masked=True)
    rows, cols = np.indices(elev.shape)
    xs, ys = xy(src_dataset.transform, rows, cols)
    return elev, np.array(xs), np.array(ys)


def precompute_kdtree(latlon, transformer):
    x, y = transformer.transform(latlon[:, 1], latlon[:, 0])
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    return tree


def scale_to_byte(arr: np.ndarray, color_scale: Dict[str, float]) -> np.ndarray:
    if color_scale and "min" in color_scale and "max" in color_scale:
        tmin = color_scale["min"]
        tmax = color_scale["max"]
    else:
        valid = arr[arr != NODATA]
        tmin = float(np.nanmin(valid))
        tmax = float(np.nanmax(valid))
    clipped = np.clip(arr, tmin, tmax)
    scaled = (clipped - tmin) / (tmax - tmin)
    out = (scaled * 254 + 1).astype(np.uint8)
    out[arr == NODATA] = 0
    return out

# ---------------------------------------------------------------------------
# Physics helper functions copied from the previous individual scripts

def phys_temperature(temp, humid, snow, target_elev, source_elev, hillshade):
    lapse = -9.8 + (humid / 100.0) * 5.8
    adj = temp + lapse * ((target_elev - source_elev) / 1000.0)
    insolation = hillshade / 32767.0
    adj += (insolation - 0.5) * 2.0
    adj = np.where(snow > 0.1, adj - 1.5, adj)
    return adj


def phys_rh(temp, humid, snow, target_elev, source_elev, hillshade):
    dz_km = (target_elev - source_elev) / 1000.0
    lapse = -9.8 + (humid / 100.0) * 5.8
    t_adj = temp + lapse * dz_km
    insolation = hillshade / 32767.0
    t_adj += (insolation - 0.5) * 2.0
    t_adj = np.where(snow > 0.1, t_adj - 1.5, t_adj)
    ln_rh = np.log(np.maximum(humid, 1e-3) / 100.0)
    alpha = (17.27 * temp) / (237.7 + temp) + ln_rh
    dew = (237.7 * alpha) / (17.27 - alpha)
    dew_adj = dew - 2.0 * dz_km
    es_dew = 6.112 * np.exp((17.67 * dew_adj) / (dew_adj + 243.5))
    es_temp = 6.112 * np.exp((17.67 * t_adj) / (t_adj + 243.5))
    rh = 100.0 * es_dew / es_temp
    rh -= (insolation - 0.5) * 10.0
    rh = np.where(snow > 0.1, rh + 5.0, rh)
    return np.clip(rh, 0.0, 100.0)


def phys_dewpoint(temp, rh, target_elev, source_elev):
    a = 17.27
    b = 237.7
    alpha = np.log(rh / 100.0) + (a * temp) / (b + temp)
    dew = (b * alpha) / (a - alpha)
    return dew - 3.0 * ((target_elev - source_elev) / 1000.0)


def phys_shortwave(rad, target_elev, source_elev, hillshade):
    alt_factor = 1.0 + 0.08 * ((target_elev - source_elev) / 1000.0)
    insolation = hillshade / 32767.0
    return rad * alt_factor * insolation


def phys_surface_pressure(press, target_elev, source_elev):
    return press + (source_elev - target_elev) * 0.12


def phys_cloud_cover(cloud, humid, target_elev, source_elev):
    adj = cloud + 10.0 * ((target_elev - source_elev) / 1000.0)
    adj += (humid - 75.0) * 0.05
    return np.clip(adj, 0.0, 100.0)


def phys_snowfall(sf, humid, target_elev, source_elev, hillshade):
    enhancement = 1.0 + 0.05 * ((target_elev - source_elev) / 100.0) * (humid / 100.0)
    adj = sf * np.maximum(enhancement, 0.0)
    insolation = hillshade / 32767.0
    adj *= 1.0 + (0.5 - insolation) * 0.1
    return adj


def phys_freezing_level(level, humid, snow, hillshade):
    lapse = -9.8 + (humid / 100.0) * 5.8
    insolation = hillshade / 32767.0
    temp_shift = (insolation - 0.5) * 2.0
    temp_shift = np.where(snow > 0.1, temp_shift - 1.5, temp_shift)
    return level + temp_shift * (-1000.0 / lapse)


def weather_penalty(code):
    code = np.asarray(code)
    penalties = np.zeros_like(code, dtype=float)
    penalties = np.where(code >= 95, 0.7, penalties)
    penalties = np.where((code >= 80) & (code < 95), 0.5, penalties)
    penalties = np.where((code >= 51) & (code < 80), 0.3, penalties)
    penalties = np.where((code >= 45) & (code < 51), 0.2, penalties)
    return penalties

# ---------------------------------------------------------------------------
# Generation functions

def generate_layer(name: str, time_index: Dict[str, int], coords, values, grid_elev, grid_xy,
                   idxs, dists, profile, hillshade, color_scales):
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    grid_elev_flat = grid_elev.filled(NODATA).ravel()
    weights = 1.0 / np.maximum(dists, 1e-6) ** 2
    weights /= weights.sum(axis=1, keepdims=True)

    for ts in TIMESTAMPS:
        ti = time_index[ts]
        hour = datetime.fromisoformat(ts).hour
        period = PERIOD_FROM_HOUR.get(hour, 1)
        hs_flat = hillshade.get(period)
        if hs_flat is not None:
            hs_flat = hs_flat.filled(0).ravel()[:, None]

        def interp(arr: np.ndarray) -> np.ndarray:
            return np.sum(arr * weights, axis=1)

        elev_vals = coords[:, 2]
        e_k = elev_vals[idxs]

        if name == "temperature_2m":
            t_k = values["temperature_2m"][:, ti][idxs]
            h_k = values["relative_humidity_2m"][:, ti][idxs]
            s_k = values["snow_depth"][:, ti][idxs]
            phys = phys_temperature(t_k, h_k, s_k, grid_elev_flat[:, None], e_k, hs_flat)
            out_flat = interp(phys)
        elif name == "dewpoint_2m":
            t_k = values["temperature_2m"][:, ti][idxs]
            h_k = values["relative_humidity_2m"][:, ti][idxs]
            phys = phys_dewpoint(t_k, h_k, grid_elev_flat[:, None], e_k)
            out_flat = interp(phys)
        elif name == "relative_humidity_2m":
            t_k = values["temperature_2m"][:, ti][idxs]
            h_k = values["relative_humidity_2m"][:, ti][idxs]
            s_k = values["snow_depth"][:, ti][idxs]
            phys = phys_rh(t_k, h_k, s_k, grid_elev_flat[:, None], e_k, hs_flat)
            out_flat = interp(phys)
        elif name == "shortwave_radiation":
            r_k = values["shortwave_radiation"][:, ti][idxs]
            phys = phys_shortwave(r_k, grid_elev_flat[:, None], e_k, hs_flat)
            out_flat = interp(phys)
        elif name == "cloud_cover":
            c_k = values["cloud_cover"][:, ti][idxs]
            h_k = values["relative_humidity_2m"][:, ti][idxs]
            phys = phys_cloud_cover(c_k, h_k, grid_elev_flat[:, None], e_k)
            out_flat = interp(phys)
        elif name == "snowfall":
            sf_k = values["snowfall"][:, ti][idxs]
            h_k = values["relative_humidity_2m"][:, ti][idxs]
            phys = phys_snowfall(sf_k, h_k, grid_elev_flat[:, None], e_k, hs_flat)
            out_flat = interp(phys)
        elif name == "snow_depth":
            sd_k = values["snow_depth"][:, ti][idxs]
            out_flat = interp(sd_k)
        elif name == "surface_pressure":
            p_k = values["surface_pressure"][:, ti][idxs]
            phys = phys_surface_pressure(p_k, grid_elev_flat[:, None], e_k)
            out_flat = interp(phys)
        elif name == "weather_code":
            wc_k = values["weather_code"][:, ti][idxs]
            out_flat = interp(wc_k)
        elif name == "wind_speed_10m":
            ws_k = values["wind_speed_10m"][:, ti][idxs]
            out_flat = interp(ws_k)
        elif name == "freezing_level_height":
            fl_k = values["freezing_level_height"][:, ti][idxs]
            h_k = values["relative_humidity_2m"][:, ti][idxs]
            s_k = values["snow_depth"][:, ti][idxs]
            phys = phys_freezing_level(fl_k, h_k, s_k, hs_flat)
            out_flat = interp(phys)
        elif name == "sqh":
            sd_k = values["snow_depth"][:, ti][idxs]
            out_flat = interp(sd_k)
        elif name == "skiability":
            sd_k = values["snow_depth"][:, ti][idxs]
            ws_k = values["wind_speed_10m"][:, ti][idxs]
            wc_k = values["weather_code"][:, ti][idxs]
            weights_local = weights
            sqh = interp(sd_k)
            wind_adj = interp(ws_k)
            codes_adj = interp(weather_penalty(wc_k))
            out_flat = sqh - 0.1 * wind_adj - codes_adj
            out_flat = np.clip(out_flat, 0, None)
        else:
            raise ValueError(f"Unknown variable '{name}'")

        out_flat = np.where(grid_elev_flat == NODATA, NODATA, out_flat)
        out = out_flat.reshape(grid_elev.shape).astype(np.float32)
        out_byte = scale_to_byte(out, color_scales.get(name, {}))

        profile.update(dtype="uint8", nodata=0, compress="lzw")
        out_dir = os.path.join(OUTPUT_BASE, ts)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}.tif")
        if os.path.exists(out_path):
            os.remove(out_path)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_byte, 1)
            dst.update_tags(units=VARIABLE_UNITS[name],
                            processing_time=datetime.utcnow().isoformat(),
                            source_points_count=len(coords))
        print(f"Wrote {out_path}")

# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variables", nargs="*", default=ALL_VARIABLES,
                        help="Variables to generate")
    args = parser.parse_args(argv)

    vars_requested = args.variables

    needed_vars = set()
    for v in vars_requested:
        if v not in ALL_VARIABLES:
            parser.error(f"Unknown variable '{v}'")
        if v == "dewpoint_2m":
            needed_vars.update(["temperature_2m", "relative_humidity_2m"])
        elif v == "relative_humidity_2m":
            needed_vars.update(["temperature_2m", "relative_humidity_2m", "snow_depth"])
        elif v == "temperature_2m":
            needed_vars.update(["temperature_2m", "relative_humidity_2m", "snow_depth"])
        elif v == "shortwave_radiation":
            needed_vars.add("shortwave_radiation")
        elif v == "cloud_cover":
            needed_vars.update(["cloud_cover", "relative_humidity_2m"])
        elif v == "snowfall":
            needed_vars.update(["snowfall", "relative_humidity_2m"])
        elif v in ("snow_depth", "sqh"):
            needed_vars.add("snow_depth")
        elif v == "surface_pressure":
            needed_vars.add("surface_pressure")
        elif v == "weather_code":
            needed_vars.add("weather_code")
        elif v == "wind_speed_10m":
            needed_vars.add("wind_speed_10m")
        elif v == "freezing_level_height":
            needed_vars.update(["freezing_level_height", "relative_humidity_2m", "snow_depth"])
        elif v == "skiability":
            needed_vars.update(["snow_depth", "wind_speed_10m", "weather_code"])

    coords, values, times = load_weather_points(WEATHER_PATH, needed_vars)
    time_index = {t: i for i, t in enumerate(times)}

    color_scales = {}
    if os.path.exists(COLOR_SCALE_JSON):
        with open(COLOR_SCALE_JSON, "r", encoding="utf-8") as fp:
            color_scales = json.load(fp)

    with rasterio.open(ELEVATION_TIF) as elev_src:
        grid_elev, xs, ys = build_grid(elev_src)
        profile = elev_src.profile

    hillshade = {}
    for period, path in HILLSHADE_TIFS.items():
        if os.path.exists(path):
            with rasterio.open(path) as hs_src:
                hillshade[period] = hs_src.read(1, masked=True)

    transformer = Transformer.from_crs("EPSG:4326", profile["crs"], always_xy=True)
    tree = precompute_kdtree(coords[:, :2], transformer)
    grid_xy = np.column_stack([xs.ravel(), ys.ravel()])
    dists, idxs = tree.query(grid_xy, k=min(4, len(coords)))
    dists = dists.astype(np.float32)
    idxs = idxs.astype(np.int32)

    for var in vars_requested:
        generate_layer(var, time_index, coords, values, grid_elev, grid_xy,
                       idxs, dists, profile.copy(), hillshade, color_scales)

if __name__ == "__main__":
    main()