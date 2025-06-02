
#!/usr/bin/env python3
"""
physics_extrapolate.py
----------------------

Predict the ten target weather variables at an arbitrary coordinate by
extrapolating from 1–2 nearest weather points using simple, *tunable* physics.

Uses 3-hour aggregated weather data for efficient computation.

CLI
---
    python physics_extrapolate.py 47.2750 11.4100 2025-05-23T12:00

Input files (static, same folder):
    physics_params.json   – coefficients dictionary
    ../meteo_api/weather_data_3hour.json  – 3-hour aggregated weather dataset
    resources/terrains/dem_25m_wgs84.tif  – elevation look-up

Stdout:
    JSON dict   {lat, lon, time,
                 temperature_2m, relative_humidity_2m, shortwave_radiation,
                 cloud_cover, snow_depth, snowfall,
                 wind_speed_10m, weather_code,
                 freezing_level_height, surface_pressure}

Any stderr is considered a warning; fatal errors exit 1.
"""

import sys, json, math, csv, pathlib, datetime, os
from typing import Dict, Any, List
import rasterio
import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
DEM_PATH = THIS_DIR.parent / "terrains" / "dem_25m_wgs84.tif"
PARAM_PATH = THIS_DIR / "physics_params.json"
WEATHER_3H_PATH = THIS_DIR.parent / "meteo_api" / "weather_data_3hour.json"

# Global weather data cache
_weather_data_cache = None


# ------------------------------------------------------------------ helpers
def load_params() -> Dict[str, float]:
    with open(PARAM_PATH, "r", encoding="utf-8") as fp:
        return json.load(fp)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))

def read_dem_elevation(lat, lon, dataset):
    row, col = dataset.index(lon, lat)
    try:
        elev = dataset.read(1)[row, col]
        if elev == dataset.nodata:
            return None
        return float(elev)
    except IndexError:
        return None

def parse_time(ts: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(ts)

# ------------------------------------------------------------------ load 3-hour weather data
def load_weather_data():
    """Load and cache the 3-hour aggregated weather dataset."""
    global _weather_data_cache

    if _weather_data_cache is not None:
        return _weather_data_cache

    if not WEATHER_3H_PATH.exists():
        print(f"Weather data file not found: {WEATHER_3H_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(WEATHER_3H_PATH, 'r', encoding='utf-8') as fp:
            data = json.load(fp)

        # Extract coordinates with weather data
        weather_points = []
        for coord_entry in data['coordinates']:
            if coord_entry['status'] == 'collected' and coord_entry.get('weather_data_3hour'):
                coord_info = coord_entry['coordinate_info']
                weather_points.append({
                    'lat': coord_info['latitude'],
                    'lon': coord_info['longitude'],
                    'elevation': coord_info['elevation'],
                    'weather_data': coord_entry['weather_data_3hour']
                })

        _weather_data_cache = weather_points
        return weather_points

    except Exception as e:
        print(f"Error loading weather data: {e}", file=sys.stderr)
        sys.exit(1)

def nearest_sources(lat, lon, weather_points, k=2):
    scored = [(haversine_km(lat, lon, p["lat"], p["lon"]), p) for p in weather_points]
    scored.sort(key=lambda x: x[0])
    return [p for _, p in scored[:k]]

def find_3hour_period_index(target_time, time_periods):
    """Find the best matching 3-hour period for the target time."""
    target_dt = target_time

    # Convert time period strings to datetime objects
    period_dts = []
    for t_str in time_periods:
        # Handle both 'Z' and timezone offset formats
        if t_str.endswith('Z'):
            t_str = t_str[:-1] + '+00:00'
        period_dts.append(datetime.datetime.fromisoformat(t_str))

    # Find the period that contains the target time
    # Each period covers 3 hours: period_time to period_time + 3 hours
    best_idx = 0
    best_diff = float('inf')

    for i, period_start in enumerate(period_dts):
        period_end = period_start + datetime.timedelta(hours=3)

        # Check if target falls within this period
        if period_start <= target_dt < period_end:
            return i

        # If no exact match, find closest period
        diff = min(abs((target_dt - period_start).total_seconds()),
                  abs((target_dt - period_end).total_seconds()))
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    return best_idx

# ------------------------------------------------------------------ physics rules
def apply_physics(target_elev, source_elev, source_vars, params):
    dz_km = (target_elev - source_elev) / 1000.0

    out = {}

    # temperature lapse
    out["temperature_2m"] = source_vars["temperature_2m"] + params["lapse_rate_degC_per_km"] * dz_km

    # humidity lapse (percentage points per km)
    out["relative_humidity_2m"] = max(0, min(
        100,
        source_vars["relative_humidity_2m"] +
        params["humidity_lapse_pct_per_km"] * dz_km))

    # radiation scaling (shadow handled later on raster)
    out["shortwave_radiation"] = source_vars["shortwave_radiation"] * params["radiation_clear_fraction"]

    # cloud cover carried over unchanged
    out["cloud_cover"] = source_vars["cloud_cover"]

    # snowfall orographic factor (simple linear)
    out["snowfall"] = max(
        0,
        source_vars["snowfall"] *
        (1 + params["snowfall_orographic_factor"] * dz_km))

    # snow depth accumulated (keep original for now)
    out["snow_depth"] = source_vars["snow_depth"] + out["snowfall"]

    # wind – assume same
    out["wind_speed_10m"] = source_vars["wind_speed_10m"]

    # weather code pass-through
    out["weather_code"] = source_vars["weather_code"]

    # freezing level – shift by lapse
    out["freezing_level_height"] = max(
        0,
        source_vars["freezing_level_height"] - dz_km * 1000)

    # surface pressure via hypsometric (approx.) ΔP ≈ −12 hPa per 100 m
    out["surface_pressure"] = max(
        300,
        source_vars["surface_pressure"] - dz_km * 120)

    return out

# ------------------------------------------------------------------ main
def main():
    if not sys.stdin.isatty():
        try:
            # Attempt to read from stdin if it's not a TTY
            input_data_raw = sys.stdin.buffer.read()
            if not input_data_raw:
                print("Error: stdin is not a TTY but no data received.", file=sys.stderr)
                sys.exit(1)
            input_data = json.loads(input_data_raw.decode('utf-8'))
            lat = float(input_data['lat'])
            lon = float(input_data['lon'])
            target_time_str = input_data['time']
            if not isinstance(target_time_str, str):
                raise ValueError("time must be an ISO timestamp string")
            target_time = parse_time(target_time_str)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            print(f"Error reading from stdin: {e}. Expected JSON with 'lat' (float), 'lon' (float), 'time' (ISO string).", file=sys.stderr)
            sys.exit(1)
    elif len(sys.argv) == 4:
        # Fallback to argv if stdin is a TTY and args are provided
        lat = float(sys.argv[1])
        lon = float(sys.argv[2])
        target_time = parse_time(sys.argv[3])
    else:
        # Print usage if stdin is a TTY and no/wrong args are provided
        print("Usage: python physics_extrapolate.py LAT LON ISO_TIMESTAMP", file=sys.stderr)
        print("Alternatively, pipe a JSON object with keys 'lat', 'lon', 'time' (ISO string) to stdin.", file=sys.stderr)
        sys.exit(1)

    params = load_params()
    weather_points = load_weather_data()

    with rasterio.open(DEM_PATH) as dem:
        target_elev = read_dem_elevation(lat, lon, dem)
        if target_elev is None:
            print("Target outside DEM or nodata", file=sys.stderr)
            sys.exit(1)

        # choose up to 2 nearest weather points
        sources = nearest_sources(lat, lon, weather_points, k=2)
        if not sources:
            print("No weather sources found", file=sys.stderr)
            sys.exit(1)

        # read source values from 3-hour data
        source_vars_agg = {}
        weight_sum = 0.0
        for s in sources:
            dist_km = haversine_km(lat, lon, s["lat"], s["lon"])
            w = 1.0 / max(dist_km, 0.01)  # inverse-distance weight

            # Find the matching 3-hour period
            weather_3h = s["weather_data"]["hourly"]
            period_idx = find_3hour_period_index(target_time, weather_3h["time"])

            # build variable dict for this source (updated parameter names)
            sv = {k: weather_3h[k][period_idx] if period_idx < len(weather_3h[k]) else 0.0
                  for k in ("temperature_2m","relative_humidity_2m","shortwave_radiation",
                           "cloud_cover","snow_depth","snowfall",
                           "wind_speed_10m","weather_code",
                           "freezing_level_height","surface_pressure")}

            # weight–average
            if not source_vars_agg:
                source_vars_agg = {k:0.0 for k in sv.keys()}
            for k,v in sv.items():
                if v is not None:  # Handle None values in aggregated data
                    source_vars_agg[k] += v * w
            weight_sum += w

            # save source elevation for lapse calculations
            s["elev"] = s["elevation"]  # Use elevation from coordinate info

        # Normalize by total weight
        for k in source_vars_agg.keys():
            source_vars_agg[k] /= weight_sum

        # Average source elevation for lapse rate calculations
        source_elev = sum(s["elev"] for s in sources) / len(sources)

    # apply physics
    predicted_vars = apply_physics(target_elev, source_elev, source_vars_agg, params)

    # final output structure
    output_json = {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "time": target_time.isoformat(),
        "variable_dict": predicted_vars
    }
    json.dump(output_json, sys.stdout, separators=(",",":"))
    sys.stdout.write("\n")

if __name__ == "__main__":
    main()