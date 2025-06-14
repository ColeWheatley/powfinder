#!/usr/bin/env python3
"""Generate RH GeoTIFFs by interpolating weather data to a 100m grid.

This script mirrors ``generate_t2m_tifs.py`` but outputs relative
humidity at 2 m (``relative_humidity_2m.tif``). Weather values are aggregated over
3‑hour periods and interpolated onto the terrain grid using inverse
distance weighting with simple physics adjustments. The resulting
directory structure is::

    TIFS/100m_resolution/<timestamp>/relative_humidity_2m.tif

Humidity is influenced by elevation, solar exposure and underlying snow
cover. The algorithm keeps the source dew‑point roughly constant with
altitude and applies a sunlight drying factor.
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
# Colour scale configuration for relative humidity range
COLOR_SCALE_JSON = "color_scales.json"
OUTPUT_BASE = "TIFS/100m_resolution"
NODATA = -9999.0

# Use EPSG:4326 (lat/lon) for weather data

# --------------------------- Helpers ----------------------------------------

def load_weather_points(path):
    """Load weather points and return structured arrays."""
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    coords = []
    temps = []
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
        temps.append(np.array(hourly["temperature_2m"], dtype=np.float32))
        humidity.append(np.array(hourly["relative_humidity_2m"], dtype=np.float32))
        snow_depth.append(np.array(hourly["snow_depth"], dtype=np.float32))

    return (np.array(coords), np.stack(temps), np.stack(humidity),
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


def apply_physics(temp, humid, target_elev, source_elev):
    """Return relative humidity adjusted for terrain effects."""
    # Calculate temperature change due to lapse rate
    lapse = -9.8 + (humid / 100.0) * 5.8
    dz_km = (target_elev - source_elev) / 1000.0
    delta_t = lapse * dz_km
    
    # ADD: Temperature-coupling - RH drops when moist air warms
    rh_adj = humid * np.exp(-0.07 * delta_t)
    
    # Ensure RH stays within physical bounds
    rh_adj = np.clip(rh_adj, 0.0, 100.0)
    
    return rh_adj


# --------------------------- Main Processing ---------------------------------

def main():
    if not os.path.exists(WEATHER_PATH):
        raise FileNotFoundError(WEATHER_PATH)

    coords, temps, humids, snows, times = load_weather_points(WEATHER_PATH)
    time_index = get_time_indices(times)

    # Load color scale configuration for reference range
    color_scale = None
    if os.path.exists(COLOR_SCALE_JSON):
        with open(COLOR_SCALE_JSON, "r", encoding="utf-8") as fp:
            scales = json.load(fp)
        color_scale = scales.get("relative_humidity_2m", {})
        if color_scale:
            print(
                "Humidity scale range:"
                f" {color_scale.get('min')} to {color_scale.get('max')} %"
            )

    with rasterio.open(ELEVATION_TIF) as elev_src:
        grid_elev, xs, ys = build_grid(elev_src)
        profile = elev_src.profile

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

        t_vals = temps[:, ti]
        h_vals = humids[:, ti]
        elev_vals = coords[:, 2]

        # gather nearest station data
        t_k = t_vals[idxs]
        h_k = h_vals[idxs]
        e_k = elev_vals[idxs]

        lapse_adj = apply_physics(t_k, h_k, grid_elev_flat[:, None], e_k)

        w = 1.0 / np.maximum(dists, 1e-6) ** 2
        w /= w.sum(axis=1, keepdims=True)

        out_flat = np.sum(lapse_adj * w, axis=1)
        out_flat = np.where(grid_elev_flat == NODATA, NODATA, out_flat)
        out = out_flat.reshape(grid_elev.shape).astype(np.float32)
        print(f"{ts}: min {out.min():.1f} max {out.max():.1f}")

        # Map floating point humidity to 0-255 byte range using colour scale
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
        out_path = os.path.join(out_dir, "relative_humidity_2m.tif")

        # Remove existing file if it exists to ensure clean overwrite
        if os.path.exists(out_path):
            os.remove(out_path)
            print(f"Removed existing {out_path}")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_byte, 1)
            dst.update_tags(units="percent", processing_time=datetime.utcnow().isoformat(),
                            source_points_count=len(coords))

        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
