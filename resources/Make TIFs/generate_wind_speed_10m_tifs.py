#!/usr/bin/env python3
"""Generate wind speed GeoTIFFs by interpolating weather data to a 100m grid."""

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
NODATA = -9999.0


def load_weather_points(path):
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    coords, winds = [], []
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
        winds.append(np.array(hourly["wind_speed_10m"], dtype=np.float32))

    return np.array(coords), np.stack(winds), times


def build_grid(src):
    elev = src.read(1, masked=True)
    rows, cols = np.indices(elev.shape)
    xs, ys = xy(src.transform, rows, cols)
    return elev, np.array(xs), np.array(ys)


def precompute_kdtree(latlon, transformer):
    x, y = transformer.transform(latlon[:, 1], latlon[:, 0])
    pts = np.column_stack([x, y])
    return cKDTree(pts)


def get_time_indices(all_times):
    return {t: i for i, t in enumerate(all_times)}


def main():
    if not os.path.exists(WEATHER_PATH):
        raise FileNotFoundError(WEATHER_PATH)

    coords, winds, times = load_weather_points(WEATHER_PATH)
    time_index = get_time_indices(times)

    color_scale = None
    if os.path.exists(COLOR_SCALE_JSON):
        with open(COLOR_SCALE_JSON, "r", encoding="utf-8") as fp:
            scales = json.load(fp)
        color_scale = scales.get("wind_speed_10m", {})

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
    profile.update(compress="lzw")

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    for ts in TIMESTAMPS:
        if ts not in time_index:
            raise ValueError(f"Timestamp {ts} not found in weather data")
        ti = time_index[ts]
        vals = winds[:, ti]
        elev_vals = coords[:, 2]
        v_k = vals[idxs]
        e_k = elev_vals[idxs]
        w = 1.0 / np.maximum(dists, 1e-6) ** 2
        w /= w.sum(axis=1, keepdims=True)
        out_flat = np.sum(v_k * w, axis=1)
        out_flat = np.where(grid_elev_flat == NODATA, NODATA, out_flat)
        out = out_flat.reshape(grid_elev.shape).astype(np.float32)

        if color_scale and "min" in color_scale and "max" in color_scale:
            tmin = color_scale["min"]
            tmax = color_scale["max"]
        else:
            tmin = float(np.nanmin(out[out != NODATA]))
            tmax = float(np.nanmax(out[out != NODATA]))

        clipped = np.clip(out, tmin, tmax)
        scaled = (clipped - tmin) / (tmax - tmin)
        out_byte = (scaled * 254 + 1).astype(np.uint8)
        out_byte[out == NODATA] = 0

        profile.update(dtype="uint8", nodata=0)

        out_dir = os.path.join(OUTPUT_BASE, ts)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "wind_speed_10m.tif")
        if os.path.exists(out_path):
            os.remove(out_path)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_byte, 1)
            dst.update_tags(units="m/s", processing_time=datetime.utcnow().isoformat(),
                            source_points_count=len(coords))
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
