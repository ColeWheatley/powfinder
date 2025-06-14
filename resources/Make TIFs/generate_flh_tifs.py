#!/usr/bin/env python3
"""Generate Freezing Level GeoTIFFs by interpolating weather data to a 100m grid.

This script reads the aggregated 3-hour weather dataset and creates
interpolated freezing level rasters for a set of predefined timestamps.
The output structure is::

    TIFS/100m_resolution/<timestamp>/freezing_level_height.tif

The interpolation uses up to four nearest weather points with inverse
distance weighting and physics-based adjustments (variable lapse rate,
hillshade illumination, snow cooling).
"""

import json
import os
from datetime import datetime

import numpy as np
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from scipy.spatial import cKDTree

# --------------------------- Configuration -----------------------------------
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

WEATHER_PATH = "resources/meteo_api/weather_data_3hour.json"
ELEVATION_TIF = "resources/terrains/tirol_100m_float.tif"
# Color scale configuration for freezing level range
COLOR_SCALE_JSON = "color_scales.json"
# Pre-rendered hillshade for four 3-hour periods (period1=9am, ... period4=6pm)
HILLSHADE_TIFS = {
    1: "resources/hillshade/hillshade_100m_period1.tif",
    2: "resources/hillshade/hillshade_100m_period2.tif",
    3: "resources/hillshade/hillshade_100m_period3.tif",
    4: "resources/hillshade/hillshade_100m_period4.tif",
}
# Map display hour to hillshade period index
PERIOD_FROM_HOUR = {9: 1, 12: 2, 15: 3, 18: 4}
OUTPUT_BASE = "TIFS/100m_resolution"
NODATA = -9999.0

# Use EPSG:4326 (lat/lon) for weather data

# --------------------------- Helpers ----------------------------------------

def load_weather_points(path):
    """Load weather points and return structured arrays."""
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    coords = []
    levels = []
    humidity = []
    snow_depth = []
    times = None

    entries = data.get("coordinates", data)
    for entry in entries:
        info = entry.get("coordinate_info", entry)

        # Parse elevation robustly
        elev_str = info.get("elevation", "unknown")
        if elev_str == "unknown":
            continue  # Skip this weather point
        if isinstance(elev_str, str) and elev_str.endswith(" m"):
            elev_str = elev_str[:-2].strip()
        try:
            elevation = float(elev_str)
        except ValueError:
            print(f"Skipping point with unparseable elevation: {elev_str}")
            continue

        wdata = entry.get("weather_data_3hour", entry.get("hourly"))
        if not wdata or "hourly" not in wdata:
            continue
        hourly = wdata["hourly"]
        if times is None:
            times = hourly["time"]

        coords.append((info["latitude"], info["longitude"], elevation))
        levels.append(np.array(hourly["freezing_level_height"], dtype=np.float32))
        humidity.append(np.array(hourly["relative_humidity_2m"], dtype=np.float32))
        snow_depth.append(np.array(hourly["snow_depth"], dtype=np.float32))

    return (np.array(coords), np.stack(levels), np.stack(humidity),
            np.stack(snow_depth), times)


def build_grid(src_dataset):
    """Return arrays of grid x/y coordinates and elevation."""
    elev = src_dataset.read(1, masked=True)
    rows, cols = np.indices(elev.shape)
    xs, ys = xy(src_dataset.transform, rows, cols)
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    return elev, xs, ys


def precompute_kdtree(latlon, transformer):
    """Build KDTree of weather points in the DEM CRS."""
    # Transformer with always_xy=True expects lon, lat order
    x, y = transformer.transform(latlon[:, 1], latlon[:, 0])
    pts = np.column_stack([x, y])
    return cKDTree(pts), pts


def get_time_indices(all_times):
    return {t: i for i, t in enumerate(all_times)}


def apply_flh_physics(level, humid, snow, hillshade):
    """Apply physics adjustments to freezing level data."""
    lapse = -9.8 + (humid / 100.0) * 5.8

    insolation = hillshade / 32767.0
    temp_shift = (insolation - 0.5) * 2.0
    temp_shift = np.where(snow > 0.1, temp_shift - 1.5, temp_shift)

    adj = level + temp_shift * (-1000.0 / lapse)
    return adj


# --------------------------- Main Processing ---------------------------------

def main():
    if not os.path.exists(WEATHER_PATH):
        raise FileNotFoundError(WEATHER_PATH)

    coords, levels, humids, snows, times = load_weather_points(WEATHER_PATH)
    time_index = get_time_indices(times)

    # Load color scale configuration for reference range
    color_scale = None
    if os.path.exists(COLOR_SCALE_JSON):
        with open(COLOR_SCALE_JSON, "r", encoding="utf-8") as fp:
            scales = json.load(fp)
        color_scale = scales.get("freezing_level_height", {})
        print(
            f"Freezing level range: {color_scale.get('min')} to {color_scale.get('max')} m"
        )

    with rasterio.open(ELEVATION_TIF) as elev_src:
        grid_elev, xs, ys = build_grid(elev_src)
        profile = elev_src.profile

    hillshade = {}
    for period, path in HILLSHADE_TIFS.items():
        with rasterio.open(path) as hs_src:
            hillshade[period] = hs_src.read(1, masked=True)

    transformer = Transformer.from_crs("EPSG:4326", profile["crs"], always_xy=True)
    tree, pts_proj = precompute_kdtree(coords[:, :2], transformer)

    grid_xy = np.column_stack([xs.ravel(), ys.ravel()])
    dists, idxs = tree.query(grid_xy, k=min(4, len(coords)))
    dists = dists.astype(np.float32)
    idxs = idxs.astype(np.int32)

    grid_elev_flat = grid_elev.filled(NODATA).ravel()

    # profile.update(dtype=rasterio.float32, nodata=NODATA, compress="lzw")
    profile.update(compress="lzw")

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    for ts in TIMESTAMPS:
        if ts not in time_index:
            raise ValueError(f"Timestamp {ts} not found in weather data")
        ti = time_index[ts]
        hour = datetime.fromisoformat(ts).hour
        period = PERIOD_FROM_HOUR.get(hour)
        if period is None:
            raise ValueError(f"No hillshade period for hour {hour}")

        flh_vals = levels[:, ti]
        h_vals = humids[:, ti]
        s_vals = snows[:, ti]
        elev_vals = coords[:, 2]

        # gather nearest station data
        flh_k = flh_vals[idxs]
        h_k = h_vals[idxs]
        s_k = s_vals[idxs]
        e_k = elev_vals[idxs]

        hs_flat = hillshade[period].filled(0).ravel()
        lapse_adj = apply_flh_physics(flh_k, h_k, s_k, hs_flat[:, None])

        w = 1.0 / np.maximum(dists, 1e-6) ** 2
        w /= w.sum(axis=1, keepdims=True)

        out_flat = np.sum(lapse_adj * w, axis=1)
        out_flat = np.where(grid_elev_flat == NODATA, NODATA, out_flat)
        out = out_flat.reshape(grid_elev.shape).astype(np.float32)
        print(f"{ts}: min {out.min():.1f} max {out.max():.1f}")

        # Map floating point temperature to 0-255 byte range using color scale
        if color_scale and "min" in color_scale and "max" in color_scale:
            tmin = color_scale["min"]
            tmax = color_scale["max"]
        else:
            # fallback to min/max from data
            tmin = float(np.nanmin(out[out != NODATA]))
            tmax = float(np.nanmax(out[out != NODATA]))

        # Clip and scale
        clipped = np.clip(out, tmin, tmax)
        scaled = (clipped - tmin) / (tmax - tmin)
        out_byte = (scaled * 255).astype(np.uint8)
        out_byte[out == NODATA] = 0  # set nodata to 0 for byte

        profile.update(dtype="uint8", nodata=0)

        out_dir = os.path.join(OUTPUT_BASE, ts)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "freezing_level_height.tif")

        # Remove existing file if it exists to ensure clean overwrite
        if os.path.exists(out_path):
            os.remove(out_path)
            print(f"Removed existing {out_path}")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_byte, 1)
            dst.update_tags(units="meters", processing_time=datetime.utcnow().isoformat(),
                            source_points_count=len(coords))

        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
