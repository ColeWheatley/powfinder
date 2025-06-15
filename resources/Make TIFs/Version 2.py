#!/usr/bin/env python3
"""Unified GeoTIFF generator for PowFinder.

This script replaces the many per-variable generators with a single
implementation.  All layers share the same grid, KD-tree lookup and
color-scale handling which greatly speeds up generation.

Usage:
    python generate_tifs.py [layer ...]
If no layer names are supplied all available layers are generated.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from scipy.spatial import cKDTree

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
COLOR_SCALE_JSON = "color_scales.json"
OUTPUT_BASE = "TIFS/100m_resolution"

HILLSHADE_TIFS = {
    1: "resources/hillshade/hillshade_100m_period1.tif",
    2: "resources/hillshade/hillshade_100m_period2.tif",
    3: "resources/hillshade/hillshade_100m_period3.tif",
    4: "resources/hillshade/hillshade_100m_period4.tif",
}
PERIOD_FROM_HOUR = {9: 1, 12: 2, 15: 3, 18: 4}

NODATA = -9999.0
DEWPOINT_LAPSE_RATE = -3.0  # °C per km


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_grid(src_dataset):
    elev = src_dataset.read(1, masked=True)
    rows, cols = np.indices(elev.shape)
    xs, ys = xy(src_dataset.transform, rows, cols)
    return elev, np.array(xs), np.array(ys)


def precompute_kdtree(latlon, transformer):
    x, y = transformer.transform(latlon[:, 1], latlon[:, 0])
    pts = np.column_stack([x, y])
    return cKDTree(pts)


def idw(values: np.ndarray, dists: np.ndarray) -> np.ndarray:
    w = 1.0 / np.maximum(dists, 1e-6) ** 2
    w /= w.sum(axis=1, keepdims=True)
    return np.sum(values * w, axis=1)


def load_weather_points(path: str, variables: set[str]):
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    coords = []
    arrays = {v: [] for v in variables}
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
        for v in variables:
            arrays[v].append(np.array(hourly.get(v, []), dtype=np.float32))

    stacked = {k: np.stack(v) for k, v in arrays.items()}
    return np.array(coords), stacked, times


# ---------------------------------------------------------------------------
# Physics functions copied from the original per-layer scripts
# ---------------------------------------------------------------------------

def physics_temperature(temp, humid, snow, target_elev, source_elev, hillshade):
    lapse = -9.8 + (humid / 100.0) * 5.8
    adj = temp + lapse * ((target_elev - source_elev) / 1000.0)
    insolation = hillshade / 32767.0
    adj += (insolation - 0.5) * 2.0
    adj = np.where(snow > 0.1, adj - 1.5, adj)
    return adj


def physics_relative_humidity(temp, humid, snow, target_elev, source_elev, hillshade):
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


def compute_dewpoint(temp, rh):
    a = 17.27
    b = 237.7
    alpha = np.log(rh / 100.0) + (a * temp) / (b + temp)
    return (b * alpha) / (a - alpha)


def physics_dewpoint(dewpoint, target_elev, source_elev):
    return dewpoint + DEWPOINT_LAPSE_RATE * ((target_elev - source_elev) / 1000.0)


def physics_cloud_cover(cloud, humid, target_elev, source_elev):
    adj = cloud + 10.0 * ((target_elev - source_elev) / 1000.0)
    adj += (humid - 75.0) * 0.05
    return np.clip(adj, 0, 100)


def physics_snowfall(snowfall, humid, target_elev, source_elev, hillshade):
    enhancement = 1.0 + 0.05 * ((target_elev - source_elev) / 100.0) * (humid / 100.0)
    adj = snowfall * np.maximum(enhancement, 0.0)
    insolation = hillshade / 32767.0
    adj *= 1.0 + (0.5 - insolation) * 0.1
    return adj


def physics_pressure(pressure, target_elev, source_elev):
    return pressure + (source_elev - target_elev) * 0.12


def physics_radiation(rad, target_elev, source_elev, hillshade):
    alt_factor = 1.0 + 0.08 * ((target_elev - source_elev) / 1000.0)
    insolation = hillshade / 32767.0
    return rad * alt_factor * insolation


def physics_freezing_level(level, humid, snow, hillshade):
    lapse = -9.8 + (humid / 100.0) * 5.8
    insolation = hillshade / 32767.0
    temp_shift = (insolation - 0.5) * 2.0
    temp_shift = np.where(snow > 0.1, temp_shift - 1.5, temp_shift)
    adj = level + temp_shift * (-1000.0 / lapse)
    return adj


# Skiability helper ------------------------------------------------------------

def weather_penalty(code):
    code = np.asarray(code)
    penalties = np.zeros_like(code, dtype=float)
    penalties = np.where(code >= 95, 0.7, penalties)
    penalties = np.where((code >= 80) & (code < 95), 0.5, penalties)
    penalties = np.where((code >= 51) & (code < 80), 0.3, penalties)
    penalties = np.where((code >= 45) & (code < 51), 0.2, penalties)
    return penalties


# ---------------------------------------------------------------------------
# Layer configuration
# ---------------------------------------------------------------------------

LAYER_CONFIG = {
    "temperature_2m": {
        "inputs": ["temperature_2m", "relative_humidity_2m", "snow_depth"],
        "hillshade": True,
        "physics": physics_temperature,
        "units": "celsius",
        "color": "temperature_2m",
        "outfile": "temperature_2m.tif",
    },
    "relative_humidity_2m": {
        "inputs": ["temperature_2m", "relative_humidity_2m", "snow_depth"],
        "hillshade": True,
        "physics": physics_relative_humidity,
        "units": "percent",
        "color": "relative_humidity_2m",
        "outfile": "relative_humidity_2m.tif",
    },
    "dewpoint_2m": {
        "inputs": ["temperature_2m", "relative_humidity_2m"],
        "hillshade": False,
        "physics": physics_dewpoint,
        "units": "celsius (dewpoint)",
        "color": "dewpoint_2m",
        "outfile": "dewpoint_2m.tif",
    },
    "cloud_cover": {
        "inputs": ["cloud_cover", "relative_humidity_2m"],
        "hillshade": False,
        "physics": physics_cloud_cover,
        "units": "percent",
        "color": "cloud_cover",
        "outfile": "cloud_cover.tif",
    },
    "snowfall": {
        "inputs": ["snowfall", "relative_humidity_2m"],
        "hillshade": True,
        "physics": physics_snowfall,
        "units": "mm",
        "color": "snowfall",
        "outfile": "snowfall.tif",
    },
    "surface_pressure": {
        "inputs": ["surface_pressure"],
        "hillshade": False,
        "physics": physics_pressure,
        "units": "hPa",
        "color": "surface_pressure",
        "outfile": "surface_pressure.tif",
    },
    "shortwave_radiation": {
        "inputs": ["shortwave_radiation"],
        "hillshade": True,
        "physics": physics_radiation,
        "units": "W/m^2",
        "color": "shortwave_radiation",
        "outfile": "shortwave_radiation.tif",
    },
    "freezing_level_height": {
        "inputs": ["freezing_level_height", "relative_humidity_2m", "snow_depth"],
        "hillshade": True,
        "physics": physics_freezing_level,
        "units": "meters",
        "color": "freezing_level_height",
        "outfile": "freezing_level_height.tif",
    },
    "snow_depth": {
        "inputs": ["snow_depth"],
        "hillshade": False,
        "physics": lambda v, *a: v,
        "units": "m",
        "color": "snow_depth",
        "outfile": "snow_depth.tif",
    },
    "wind_speed_10m": {
        "inputs": ["wind_speed_10m"],
        "hillshade": False,
        "physics": lambda v, *a: v,
        "units": "m/s",
        "color": "wind_speed_10m",
        "outfile": "wind_speed_10m.tif",
    },
    "weather_code": {
        "inputs": ["weather_code"],
        "hillshade": False,
        "physics": lambda v, *a: v,
        "units": "wmo_code",
        "color": "weather_code",
        "outfile": "weather_code.tif",
    },
    "sqh": {
        "inputs": ["snow_depth"],
        "hillshade": False,
        "physics": lambda v, *a: v,
        "units": "sqh",
        "color": "sqh",
        "outfile": "sqh.tif",
    },
    "skiability": {
        "inputs": ["snow_depth", "wind_speed_10m", "weather_code"],
        "hillshade": False,
        "physics": None,  # handled separately
        "units": "ski",
        "color": "skiability",
        "outfile": "skiability.tif",
    },
}


# ---------------------------------------------------------------------------

def load_color_scales(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def generate_layer(name: str, config: dict, arrays: dict, coords: np.ndarray, *,
                    grid_elev: np.ndarray, dists: np.ndarray, idxs: np.ndarray,
                    grid_elev_flat: np.ndarray, hillshade: dict, profile: dict,
                    color_scales: dict, time_index: dict):
    print(f"\n--- Generating {name} ---")
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    inputs = [arrays[v] for v in config["inputs"]]
    elev_vals = coords[:, 2]

    color_scale = color_scales.get(config["color"], {})

    for ts in TIMESTAMPS:
        if ts not in time_index:
            raise ValueError(f"Timestamp {ts} not found in weather data")
        ti = time_index[ts]
        hour = datetime.fromisoformat(ts).hour
        period = PERIOD_FROM_HOUR.get(hour)

        vals = [arr[:, ti] for arr in inputs]
        neigh = [v[idxs] for v in vals]
        e_k = elev_vals[idxs]

        if config["hillshade"]:
            hs_flat = hillshade[period].filled(0).ravel()[:, None]
        else:
            hs_flat = np.zeros_like(grid_elev_flat)[:, None]

        if name == "dewpoint_2m":
            neigh[0] = compute_dewpoint(neigh[0], neigh[1])
        if name == "skiability":
            d_k, w_k, c_k = neigh
            penalties = weather_penalty(c_k)
            w = 1.0 / np.maximum(dists, 1e-6) ** 2
            w /= w.sum(axis=1, keepdims=True)
            sqh = np.sum(d_k * w, axis=1)
            wind_adj = np.sum(w_k * w, axis=1)
            codes_adj = np.sum(penalties * w, axis=1)
            out_flat = sqh - 0.1 * wind_adj - codes_adj
        else:
            phys_inputs = neigh + [grid_elev_flat[:, None], e_k]
            if config["hillshade"]:
                phys_inputs.append(hs_flat)
            result = config["physics"](*phys_inputs)
            out_flat = idw(result, dists)

        out_flat = np.where(grid_elev_flat == NODATA, NODATA, out_flat)
        out = out_flat.reshape(grid_elev.shape).astype(np.float32)

        if color_scale and "min" in color_scale and "max" in color_scale:
            vmin = color_scale["min"]
            vmax = color_scale["max"]
        else:
            valid = out[out != NODATA]
            vmin = float(np.nanmin(valid)) if len(valid) > 0 else 0.0
            vmax = float(np.nanmax(valid)) if len(valid) > 0 else 1.0

        clipped = np.clip(out, vmin, vmax)
        scaled = (clipped - vmin) / (vmax - vmin)
        out_byte = (scaled * 254 + 1).astype(np.uint8)
        out_byte[out == NODATA] = 0

        profile.update(dtype="uint8", nodata=0, compress="lzw")
        out_dir = os.path.join(OUTPUT_BASE, ts)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, config["outfile"])
        if os.path.exists(out_path):
            os.remove(out_path)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_byte, 1)
            dst.update_tags(
                units=config["units"],
                processing_time=datetime.utcnow().isoformat(),
                source_points_count=len(coords),
            )
        print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate PowFinder GeoTIFF layers")
    parser.add_argument("layers", nargs="*", help="Layers to generate (default: all)")
    args = parser.parse_args()

    layers = args.layers or list(LAYER_CONFIG.keys())
    unknown = set(layers) - LAYER_CONFIG.keys()
    if unknown:
        raise SystemExit(f"Unknown layer(s): {', '.join(sorted(unknown))}")

    required_vars = set()
    for l in layers:
        required_vars.update(LAYER_CONFIG[l]["inputs"])

    coords, arrays, times = load_weather_points(WEATHER_PATH, required_vars)
    time_index = {t: i for i, t in enumerate(times)}

    with rasterio.open(ELEVATION_TIF) as elev_src:
        grid_elev, xs, ys = build_grid(elev_src)
        profile = elev_src.profile

    transformer = Transformer.from_crs("EPSG:4326", profile["crs"], always_xy=True)
    tree = precompute_kdtree(coords[:, :2], transformer)

    grid_xy = np.column_stack([xs.ravel(), ys.ravel()])
    dists, idxs = tree.query(grid_xy, k=min(4, len(coords)))
    dists = dists.astype(np.float32)
    idxs = idxs.astype(np.int32)

    grid_elev_flat = grid_elev.filled(NODATA).ravel()

    hillshade = {}
    for period, path in HILLSHADE_TIFS.items():
        with rasterio.open(path) as hs_src:
            hillshade[period] = hs_src.read(1, masked=True)

    color_scales = load_color_scales(COLOR_SCALE_JSON)

    for layer in layers:
        generate_layer(
            layer,
            LAYER_CONFIG[layer],
            arrays,
            coords,
            grid_elev=grid_elev,
            dists=dists,
            idxs=idxs,
            grid_elev_flat=grid_elev_flat,
            hillshade=hillshade,
            profile=profile.copy(),
            color_scales=color_scales,
            time_index=time_index,
        )


if __name__ == "__main__":
    main()