#!/usr/bin/env python3
"""Unified TIF generator for PowFinder.

This script consolidates the previous suite of generate_*_tifs.py scripts
into a single entry point.  All weather variables are loaded once and the
100 m grid plus KDTree search weights are reused for every variable.  Each
variable retains its original physics adjustment but they now live in one
place which greatly reduces code duplication.

Usage::

    python3 generate_tifs.py [var1 var2 ...]

If no variables are specified all known variables are rendered.  This
allows tweaking the physics for one layer without recomputing the rest.
"""

import json
import os
import sys
from datetime import datetime
from typing import Callable, Dict, List

import numpy as np
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Configuration
TIMESTAMPS = [
    "2025-05-24T09:00:00", "2025-05-24T12:00:00", "2025-05-24T15:00:00", "2025-05-24T18:00:00",
    "2025-05-25T09:00:00", "2025-05-25T12:00:00", "2025-05-25T15:00:00", "2025-05-25T18:00:00",
    "2025-05-26T09:00:00", "2025-05-26T12:00:00", "2025-05-26T15:00:00", "2025-05-26T18:00:00",
    "2025-05-27T09:00:00", "2025-05-27T12:00:00", "2025-05-27T15:00:00", "2025-05-27T18:00:00",
    "2025-05-28T09:00:00", "2025-05-28T12:00:00", "2025-05-28T15:00:00", "2025-05-28T18:00:00",
]

WEATHER_PATH = "resources/meteo_api/weather_data_3hour.json"
ELEVATION_TIF = "resources/terrains/tirol_100m_float.tif"
COLOR_SCALE_JSON = "color_scales.json"
OUTPUT_BASE = "TIFS/100m_resolution"
NODATA = -9999.0

# Hillshades for 09/12/15/18h used by several variables
HILLSHADE_TIFS = {
    1: "resources/hillshade/hillshade_100m_period1.tif",
    2: "resources/hillshade/hillshade_100m_period2.tif",
    3: "resources/hillshade/hillshade_100m_period3.tif",
    4: "resources/hillshade/hillshade_100m_period4.tif",
}
PERIOD_FROM_HOUR = {9: 1, 12: 2, 15: 3, 18: 4}

# ---------------------------------------------------------------------------
# Helper loading functions

VAR_NAMES = [
    "temperature_2m",
    "relative_humidity_2m",
    "shortwave_radiation",
    "cloud_cover",
    "snow_depth",
    "snowfall",
    "wind_speed_10m",
    "weather_code",
    "freezing_level_height",
    "surface_pressure",
]

# Additional derived variables
DERIVED_VARS = ["dewpoint_2m", "sqh", "skiability"]

ALL_VARS = VAR_NAMES + DERIVED_VARS


def load_weather(path: str):
    """Load weather json into arrays for each variable."""
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    coords: List[tuple] = []
    values: Dict[str, List[np.ndarray]] = {v: [] for v in VAR_NAMES}
    times = None

    entries = data.get("coordinates", data)
    for entry in entries:
        info = entry.get("coordinate_info", entry)
        elev = info.get("elevation")
        if elev is None:
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
        for v in VAR_NAMES:
            arr = hourly.get(v)
            if arr is None:
                values[v].append(np.zeros(len(times), dtype=np.float32))
            else:
                values[v].append(np.array(arr, dtype=np.float32))

    out = {v: np.stack(lst) for v, lst in values.items()}
    return np.array(coords, dtype=np.float32), out, times


def build_grid(src_dataset):
    elev = src_dataset.read(1, masked=True)
    rows, cols = np.indices(elev.shape)
    xs, ys = xy(src_dataset.transform, rows, cols)
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    return elev, xs, ys


def precompute_kdtree(latlon: np.ndarray, transformer: Transformer):
    x, y = transformer.transform(latlon[:, 1], latlon[:, 0])
    pts = np.column_stack([x, y])
    return cKDTree(pts)


def load_hillshade():
    shades = {}
    for period, path in HILLSHADE_TIFS.items():
        with rasterio.open(path) as src:
            shades[period] = src.read(1, masked=True)
    return shades


def get_time_index(times: List[str]):
    return {t: i for i, t in enumerate(times)}

# ---------------------------------------------------------------------------
# Physics functions from the old scripts

def physics_temperature(temp, humid, snow, target_elev, source_elev, hill):
    lapse = -9.8 + (humid / 100.0) * 5.8
    adj = temp + lapse * ((target_elev - source_elev) / 1000.0)
    ins = hill / 32767.0
    adj += (ins - 0.5) * 2.0
    adj = np.where(snow > 0.1, adj - 1.5, adj)
    return adj


def physics_relative_humidity(temp, humid, snow, target_elev, source_elev, hill):
    dz_km = (target_elev - source_elev) / 1000.0
    lapse = -9.8 + (humid / 100.0) * 5.8
    t_adj = temp + lapse * dz_km
    ins = hill / 32767.0
    t_adj += (ins - 0.5) * 2.0
    t_adj = np.where(snow > 0.1, t_adj - 1.5, t_adj)
    ln_rh = np.log(np.maximum(humid, 1e-3) / 100.0)
    alpha = (17.27 * temp) / (237.7 + temp) + ln_rh
    dew = (237.7 * alpha) / (17.27 - alpha)
    dew_adj = dew - 2.0 * dz_km
    es_dew = 6.112 * np.exp((17.67 * dew_adj) / (dew_adj + 243.5))
    es_temp = 6.112 * np.exp((17.67 * t_adj) / (t_adj + 243.5))
    rh = 100.0 * es_dew / es_temp
    rh -= (ins - 0.5) * 10.0
    rh = np.where(snow > 0.1, rh + 5.0, rh)
    return np.clip(rh, 0.0, 100.0)


def physics_shortwave_radiation(rad, target_elev, source_elev, hill):
    alt_factor = 1.0 + 0.08 * ((target_elev - source_elev) / 1000.0)
    ins = hill / 32767.0
    return rad * alt_factor * ins


def physics_cloud_cover(cloud, humid, target_elev, source_elev):
    adj = cloud + 10.0 * ((target_elev - source_elev) / 1000.0)
    adj += (humid - 75.0) * 0.05
    return np.clip(adj, 0, 100)


def physics_snowfall(sf, humid, target_elev, source_elev, hill):
    enh = 1.0 + 0.05 * ((target_elev - source_elev) / 100.0) * (humid / 100.0)
    ins = hill / 32767.0
    return sf * np.maximum(enh, 0.0) * (1.0 + (0.5 - ins) * 0.1)


def physics_dewpoint(dp, target_elev, source_elev):
    return dp + (-2.0) * ((target_elev - source_elev) / 1000.0)


def physics_pressure(p, target_elev, source_elev):
    return p + (source_elev - target_elev) * 0.12


def physics_freezing_level(level, humid, snow, hill):
    lapse = -9.8 + (humid / 100.0) * 5.8
    ins = hill / 32767.0
    temp_shift = (ins - 0.5) * 2.0
    temp_shift = np.where(snow > 0.1, temp_shift - 1.5, temp_shift)
    return level + temp_shift * (-1000.0 / lapse)


def physics_identity(v, *args):
    return v


def compute_dewpoint(temp, rh):
    a, b = 17.27, 237.7
    alpha = (a * temp) / (b + temp) + np.log(np.maximum(rh, 1e-3) / 100.0)
    return (b * alpha) / (a - alpha)


def physics_skiability(depth, wind, codes):
    penalties = np.zeros_like(codes, dtype=float)
    penalties = np.where(codes >= 95, 0.7, penalties)
    penalties = np.where((codes >= 80) & (codes < 95), 0.5, penalties)
    penalties = np.where((codes >= 51) & (codes < 80), 0.3, penalties)
    penalties = np.where((codes >= 45) & (codes < 51), 0.2, penalties)
    ski = depth - 0.1 * wind - penalties
    ski = np.clip(ski, 0, None)
    return ski

# ---------------------------------------------------------------------------
# Variable configuration table

VAR_CONFIG = {
    "temperature_2m": dict(fields=["temperature_2m", "relative_humidity_2m", "snow_depth"],
                           hillshade=True, physics=physics_temperature, units="celsius"),
    "relative_humidity_2m": dict(fields=["temperature_2m", "relative_humidity_2m", "snow_depth"],
                                 hillshade=True, physics=physics_relative_humidity, units="percent"),
    "shortwave_radiation": dict(fields=["shortwave_radiation"], hillshade=True,
                                physics=physics_shortwave_radiation, units="W/m^2"),
    "cloud_cover": dict(fields=["cloud_cover", "relative_humidity_2m"], hillshade=False,
                        physics=physics_cloud_cover, units="percent"),
    "snow_depth": dict(fields=["snow_depth"], hillshade=False, physics=physics_identity, units="m"),
    "snowfall": dict(fields=["snowfall", "relative_humidity_2m"], hillshade=True,
                     physics=physics_snowfall, units="mm"),
    "wind_speed_10m": dict(fields=["wind_speed_10m"], hillshade=False,
                           physics=physics_identity, units="m/s"),
    "weather_code": dict(fields=["weather_code"], hillshade=False,
                         physics=physics_identity, units="wmo_code"),
    "freezing_level_height": dict(fields=["freezing_level_height", "relative_humidity_2m", "snow_depth"],
                                  hillshade=True, physics=physics_freezing_level, units="meters"),
    "surface_pressure": dict(fields=["surface_pressure"], hillshade=False,
                             physics=physics_pressure, units="hPa"),
    "dewpoint_2m": dict(fields=["temperature_2m", "relative_humidity_2m"], hillshade=False,
                        physics=physics_dewpoint, units="celsius"),
    "sqh": dict(fields=["snow_depth"], hillshade=False, physics=physics_identity, units="sqh"),
    "skiability": dict(fields=["snow_depth", "wind_speed_10m", "weather_code"], hillshade=False,
                        physics=physics_skiability, units="ski"),
}

# ---------------------------------------------------------------------------


def main(vars_to_process: List[str]):
    if not os.path.exists(WEATHER_PATH):
        raise FileNotFoundError(WEATHER_PATH)

    coords, weather, times = load_weather(WEATHER_PATH)
    time_index = get_time_index(times)

    dewpoint = compute_dewpoint(weather["temperature_2m"], weather["relative_humidity_2m"])
    weather["dewpoint_2m"] = dewpoint
    weather["sqh"] = weather["snow_depth"]
    # skiability computed on the fly

    with open(COLOR_SCALE_JSON, "r", encoding="utf-8") as fp:
        color_scales = json.load(fp)

    with rasterio.open(ELEVATION_TIF) as src:
        grid_elev, xs, ys = build_grid(src)
        profile = src.profile

    transformer = Transformer.from_crs("EPSG:4326", profile["crs"], always_xy=True)
    tree = precompute_kdtree(coords[:, :2], transformer)
    grid_xy = np.column_stack([xs.ravel(), ys.ravel()])
    dists, idxs = tree.query(grid_xy, k=min(4, len(coords)))
    weights = 1.0 / np.maximum(dists, 1e-6) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    idxs = idxs.astype(np.int32)
    grid_elev_flat = grid_elev.filled(NODATA).ravel()
    profile.update(compress="lzw")

    hillshade = load_hillshade()

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    for ts in TIMESTAMPS:
        if ts not in time_index:
            raise ValueError(f"Timestamp {ts} not found in weather data")
        ti = time_index[ts]
        hour = datetime.fromisoformat(ts).hour
        period = PERIOD_FROM_HOUR.get(hour)
        hs_flat = hillshade[period].filled(0).ravel() if period in hillshade else None

        for var in vars_to_process:
            cfg = VAR_CONFIG[var]
            fields = [weather[f][:, ti] for f in cfg["fields"]]
            e_vals = coords[:, 2]
            data_k = [arr[idxs] for arr in fields]
            if cfg["hillshade"]:
                result = cfg["physics"](*data_k, grid_elev_flat[:, None], e_vals[idxs], hs_flat[:, None])
            else:
                if var == "skiability":
                    result = cfg["physics"](*data_k)
                    result = np.where(grid_elev_flat == NODATA, NODATA, result)
                else:
                    result = cfg["physics"](*data_k, grid_elev_flat[:, None], e_vals[idxs])
            out_flat = np.sum(result * weights, axis=1) if result.ndim == 2 else np.sum(result * weights, axis=1)
            out_flat = np.where(grid_elev_flat == NODATA, NODATA, out_flat)
            out = out_flat.reshape(grid_elev.shape).astype(np.float32)

            scale = color_scales.get(var, {})
            if scale and "min" in scale and "max" in scale:
                vmin = scale["min"]
                vmax = scale["max"]
            else:
                valid = out[out != NODATA]
                vmin = float(np.nanmin(valid)) if valid.size else 0.0
                vmax = float(np.nanmax(valid)) if valid.size else 1.0

            clipped = np.clip(out, vmin, vmax)
            scaled = (clipped - vmin) / (vmax - vmin)
            out_byte = (scaled * 254 + 1).astype(np.uint8)
            out_byte[out == NODATA] = 0

            profile.update(dtype="uint8", nodata=0)
            out_dir = os.path.join(OUTPUT_BASE, ts)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{var}.tif")
            if os.path.exists(out_path):
                os.remove(out_path)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(out_byte, 1)
                dst.update_tags(units=cfg["units"], processing_time=datetime.utcnow().isoformat(),
                                source_points_count=len(coords))
            print(f"{ts} - wrote {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        unknown = [a for a in args if a not in VAR_CONFIG]
        if unknown:
            sys.exit(f"Unknown variable(s): {', '.join(unknown)}")
        vars_to_run = args
    else:
        vars_to_run = list(VAR_CONFIG.keys())
    main(vars_to_run)