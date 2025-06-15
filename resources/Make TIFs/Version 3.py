#!/usr/bin/env python3
"""Unified generator for PowFinder GeoTIFF layers.

This consolidates the many individual scripts that produced each weather
layer into a single program.  The workflow mirrors the old scripts so the
output directory layout and colour mapping remain compatible with the rest
of the project.  Each variable has its own small physics function which
matches the previous implementation.

Running ``python generate_tifs.py`` will create all TIF layers in
``TIFS/100m_resolution/<timestamp>/``.  If a layer already exists and the
physics function has not changed since the last run it will be skipped so
minor tweaks do not trigger a full rebuild of unrelated files.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Configuration shared with the original scripts
TIMESTAMPS = [
    "2025-05-24T09:00:00",
    "2025-05-24T12:00:00",
    "2025-05-24T15:00:00",
    "2025-05-24T18:00:00",
    "2025-05-25T09:00:00",
    "2025-05-25T12:00:00",
    "2025-05-25T15:00:00",
    "2025-05-25T18:00:00",
    "2025-05-26T09:00:00",
    "2025-05-26T12:00:00",
    "2025-05-26T15:00:00",
    "2025-05-26T18:00:00",
    "2025-05-27T09:00:00",
    "2025-05-27T12:00:00",
    "2025-05-27T15:00:00",
    "2025-05-27T18:00:00",
    "2025-05-28T09:00:00",
    "2025-05-28T12:00:00",
    "2025-05-28T15:00:00",
    "2025-05-28T18:00:00",
]

WEATHER_PATH = Path("resources/meteo_api/weather_data_3hour.json")
ELEVATION_TIF = Path("resources/terrains/tirol_100m_float.tif")
COLOR_SCALE_JSON = Path("resources/Make TIFs/color_scales.json")
OUTPUT_BASE = Path("TIFS/100m_resolution")
NODATA = -9999.0

HILLSHADE_TIFS = {
    1: Path("resources/hillshade/hillshade_100m_period1.tif"),
    2: Path("resources/hillshade/hillshade_100m_period2.tif"),
    3: Path("resources/hillshade/hillshade_100m_period3.tif"),
    4: Path("resources/hillshade/hillshade_100m_period4.tif"),
}
PERIOD_FROM_HOUR = {9: 1, 12: 2, 15: 3, 18: 4}

# ---------------------------------------------------------------------------
# Weather loading


def _parse_elevation(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.endswith(" m"):
        value = value[:-2].strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_weather(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], Iterable[str]]:
    """Return coordinates array, variable arrays dict and timestamp list."""
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    variables = {
        "temperature_2m": [],
        "relative_humidity_2m": [],
        "snow_depth": [],
        "snowfall": [],
        "wind_speed_10m": [],
        "weather_code": [],
        "shortwave_radiation": [],
        "surface_pressure": [],
        "cloud_cover": [],
        "freezing_level_height": [],
    }
    coords = []
    times = None

    entries = data.get("coordinates", data)
    for entry in entries:
        info = entry.get("coordinate_info", entry)
        elev = _parse_elevation(info.get("elevation"))
        if elev is None:
            continue
        wdata = entry.get("weather_data_3hour", entry.get("hourly"))
        if not wdata or "hourly" not in wdata:
            continue
        hourly = wdata["hourly"]
        if times is None:
            times = hourly["time"]
        coords.append((info["latitude"], info["longitude"], elev))
        for key in variables:
            if key in hourly:
                variables[key].append(np.array(hourly[key], dtype=np.float32))
            else:
                # Fill missing with zeros
                variables[key].append(np.zeros(len(hourly["time"]), dtype=np.float32))

    arrs = {k: np.stack(v) for k, v in variables.items()}
    return np.array(coords), arrs, times


# ---------------------------------------------------------------------------
# Grid helpers


def build_grid(src: rasterio.io.DatasetReader) -> Tuple[np.ma.MaskedArray, np.ndarray, np.ndarray]:
    elev = src.read(1, masked=True)
    rows, cols = np.indices(elev.shape)
    xs, ys = xy(src.transform, rows, cols)
    return elev, np.array(xs), np.array(ys)


def precompute_kdtree(latlon: np.ndarray, transformer: Transformer) -> cKDTree:
    x, y = transformer.transform(latlon[:, 1], latlon[:, 0])
    pts = np.column_stack([x, y])
    return cKDTree(pts)


def time_index(times: Iterable[str]) -> Dict[str, int]:
    return {t: i for i, t in enumerate(times)}

# ---------------------------------------------------------------------------
# Physics functions equivalent to the old scripts


DEWPOINT_LAPSE_RATE = -3.0  # °C per km


def physics_temperature(temp, humid, snow, tgt_elev, src_elev, hill):
    lapse = -9.8 + (humid / 100.0) * 5.8
    adj = temp + lapse * ((tgt_elev - src_elev) / 1000.0)
    ins = hill / 32767.0
    adj += (ins - 0.5) * 2.0
    adj = np.where(snow > 0.1, adj - 1.5, adj)
    return adj


def physics_dewpoint(td, tgt_elev, src_elev):
    return td + DEWPOINT_LAPSE_RATE * ((tgt_elev - src_elev) / 1000.0)


def physics_relative_humidity(temp, humid, snow, tgt_elev, src_elev, hill):
    dz_km = (tgt_elev - src_elev) / 1000.0
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


def physics_shortwave(rad, tgt_elev, src_elev, hill):
    alt_factor = 1.0 + 0.08 * ((tgt_elev - src_elev) / 1000.0)
    ins = hill / 32767.0
    return rad * alt_factor * ins


def physics_snowfall(sf, humid, tgt_elev, src_elev, hill):
    enh = 1.0 + 0.05 * ((tgt_elev - src_elev) / 100.0) * (humid / 100.0)
    adj = sf * np.maximum(enh, 0.0)
    ins = hill / 32767.0
    adj *= 1.0 + (0.5 - ins) * 0.1
    return adj


def physics_cloud_cover(cloud, humid, tgt_elev, src_elev):
    adj = cloud + 10.0 * ((tgt_elev - src_elev) / 1000.0)
    adj += (humid - 75.0) * 0.05
    return np.clip(adj, 0.0, 100.0)


def physics_pressure(p, tgt_elev, src_elev):
    return p + (src_elev - tgt_elev) * 0.12

# No-op physics used by several layers

def physics_identity(value, *_, **__):
    return value


# Skiability helper

def weather_penalty(code: np.ndarray) -> np.ndarray:
    code = np.asarray(code)
    penalties = np.zeros_like(code, dtype=float)
    penalties = np.where(code >= 95, 0.7, penalties)
    penalties = np.where((code >= 80) & (code < 95), 0.5, penalties)
    penalties = np.where((code >= 51) & (code < 80), 0.3, penalties)
    penalties = np.where((code >= 45) & (code < 51), 0.2, penalties)
    return penalties


# ---------------------------------------------------------------------------
# Variable configuration


class Variable:
    def __init__(
        self,
        key: str,
        deps: Tuple[str, ...],
        physics,
        units: str,
        needs_hillshade: bool = False,
    ) -> None:
        self.key = key
        self.deps = deps or (key,)
        self.physics = physics
        self.units = units
        self.needs_hillshade = needs_hillshade
        self.fingerprint = hashlib.sha1(
            inspect.getsource(physics).encode("utf-8")
        ).hexdigest()


VARIABLES = [
    Variable("temperature_2m", ("temperature_2m", "relative_humidity_2m", "snow_depth"), physics_temperature, "celsius", True),
    Variable("dewpoint_2m", ("temperature_2m", "relative_humidity_2m"), physics_dewpoint, "celsius"),
    Variable("relative_humidity_2m", ("temperature_2m", "relative_humidity_2m", "snow_depth"), physics_relative_humidity, "percent", True),
    Variable("shortwave_radiation", ("shortwave_radiation",), physics_shortwave, "W/m^2", True),
    Variable("snowfall", ("snowfall", "relative_humidity_2m"), physics_snowfall, "mm", True),
    Variable("snow_depth", ("snow_depth",), physics_identity, "m"),
    Variable("wind_speed_10m", ("wind_speed_10m",), physics_identity, "m/s"),
    Variable("weather_code", ("weather_code",), physics_identity, "wmo_code"),
    Variable("surface_pressure", ("surface_pressure",), physics_pressure, "hPa"),
    Variable("cloud_cover", ("cloud_cover", "relative_humidity_2m"), physics_cloud_cover, "percent"),
    Variable("freezing_level_height", ("freezing_level_height", "relative_humidity_2m", "snow_depth"), physics_temperature, "meters", True),
    Variable("sqh", ("snow_depth",), physics_identity, "sqh"),
    Variable("skiability", ("snow_depth", "wind_speed_10m", "weather_code"), physics_identity, "ski"),
]

# Map for quick lookup
VAR_BY_KEY = {v.key: v for v in VARIABLES}


# ---------------------------------------------------------------------------
# Generation utilities


def should_skip(var: Variable) -> bool:
    fp_file = OUTPUT_BASE / f"{var.key}.hash"
    if not fp_file.exists():
        return False
    saved = fp_file.read_text().strip()
    if saved != var.fingerprint:
        return False
    for ts in TIMESTAMPS:
        tif = OUTPUT_BASE / ts / f"{var.key}.tif"
        if not tif.exists():
            return False
    return True


def save_fingerprint(var: Variable) -> None:
    fp_file = OUTPUT_BASE / f"{var.key}.hash"
    fp_file.write_text(var.fingerprint)


# ---------------------------------------------------------------------------


def main() -> None:
    if not WEATHER_PATH.exists():
        raise FileNotFoundError(WEATHER_PATH)

    coords, arrays, times = load_weather(WEATHER_PATH)
    tindex = time_index(times)

    color_scale = {}
    if COLOR_SCALE_JSON.exists():
        with COLOR_SCALE_JSON.open("r", encoding="utf-8") as fp:
            color_scale = json.load(fp)

    with rasterio.open(ELEVATION_TIF) as elev_src:
        grid_elev, xs, ys = build_grid(elev_src)
        profile = elev_src.profile

    hillshade = {}
    for p, path in HILLSHADE_TIFS.items():
        with rasterio.open(path) as hs:
            hillshade[p] = hs.read(1, masked=True)

    transformer = Transformer.from_crs("EPSG:4326", profile["crs"], always_xy=True)
    tree = precompute_kdtree(coords[:, :2], transformer)
    grid_xy = np.column_stack([xs.ravel(), ys.ravel()])
    dists, idxs = tree.query(grid_xy, k=min(4, len(coords)))
    dists = dists.astype(np.float32)
    idxs = idxs.astype(np.int32)

    grid_elev_flat = grid_elev.filled(NODATA).ravel()
    profile.update(compress="lzw")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    for var in VARIABLES:
        if should_skip(var):
            print(f"Skipping {var.key} (up to date)")
            continue
        for ts in TIMESTAMPS:
            if ts not in tindex:
                raise ValueError(f"Timestamp {ts} not found in weather data")
            ti = tindex[ts]
            hour = datetime.fromisoformat(ts).hour
            period = PERIOD_FROM_HOUR.get(hour)
            hs = hillshade[period].filled(0).ravel() if var.needs_hillshade else None

            dep_arrays = [arrays[d][:, ti] for d in var.deps]
            e_vals = coords[:, 2]
            dep_k = [arr[idxs] for arr in dep_arrays]
            e_k = e_vals[idxs]

            if var.key == "skiability":
                depth_k, wind_k, code_k = dep_k
                penalties = weather_penalty(code_k)
                w = 1.0 / np.maximum(dists, 1e-6) ** 2
                w /= w.sum(axis=1, keepdims=True)
                sqh = np.sum(depth_k * w, axis=1)
                wind_adj = np.sum(wind_k * w, axis=1)
                code_adj = np.sum(penalties * w, axis=1)
                out_flat = sqh - 0.1 * wind_adj - code_adj
                out_flat = np.clip(out_flat, 0, None)
            else:
                tgt = grid_elev_flat[:, None]
                if var.needs_hillshade:
                    phys = var.physics(*dep_k, tgt, e_k, hs[:, None])
                elif len(dep_k) == 1:
                    phys = var.physics(dep_k[0], tgt, e_k)
                else:
                    phys = var.physics(*dep_k, tgt, e_k)
                w = 1.0 / np.maximum(dists, 1e-6) ** 2
                w /= w.sum(axis=1, keepdims=True)
                out_flat = np.sum(phys * w, axis=1)

            out_flat = np.where(grid_elev_flat == NODATA, NODATA, out_flat)
            out = out_flat.reshape(grid_elev.shape).astype(np.float32)

            scale = color_scale.get(var.key, {})
            if "min" in scale and "max" in scale:
                vmin, vmax = scale["min"], scale["max"]
            else:
                valid = out[out != NODATA]
                vmin = float(np.nanmin(valid)) if valid.size else 0.0
                vmax = float(np.nanmax(valid)) if valid.size else 1.0
            clipped = np.clip(out, vmin, vmax)
            scaled = (clipped - vmin) / (vmax - vmin) if vmax != vmin else clipped
            out_byte = (scaled * 254 + 1).astype(np.uint8)
            out_byte[out == NODATA] = 0

            profile.update(dtype="uint8", nodata=0)
            out_dir = OUTPUT_BASE / ts
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{var.key}.tif"
            if out_path.exists():
                out_path.unlink()
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(out_byte, 1)
                dst.update_tags(
                    units=var.units,
                    processing_time=datetime.utcnow().isoformat(),
                    source_points_count=len(coords),
                )
            print(f"Wrote {out_path}")
        save_fingerprint(var)


if __name__ == "__main__":
    main()