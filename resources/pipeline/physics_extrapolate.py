
#!/usr/bin/env python3
"""
physics_extrapolate.py
----------------------

Predict the ten target weather variables at an arbitrary coordinate by
extrapolating from 1–2 nearest raw-API points using simple, *tunable* physics.

CLI
---
    python physics_extrapolate.py 47.2750 11.4100 2025-05-23T12:00

Input files (static, same folder):
    physics_params.json   – coefficients dictionary
    raw_api_index.csv     – lat,lon,api_json_path
    resources/terrains/dem_25m_wgs84.tif  – elevation look-up

Stdout:
    JSON dict   {lat, lon, time,
                 temperature_2m, relative_humidity_2m, shortwave_radiation,
                 cloud_cover_total, snow_depth, snowfall,
                 wind_speed_10m, weather_code,
                 freezing_level_height, surface_pressure}

Any stderr is considered a warning; fatal errors exit 1.
"""

import sys, json, math, csv, pathlib, datetime
from typing import Dict, Any, List
import rasterio
import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
DEM_PATH = THIS_DIR.parent / "terrains" / "dem_25m_wgs84.tif"
PARAM_PATH = THIS_DIR / "physics_params.json"
INDEX_PATH = THIS_DIR / "raw_api_index.csv"


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

# ------------------------------------------------------------------ load API index
def load_api_index() -> List[Dict[str, Any]]:
    rows = []
    with open(INDEX_PATH, newline='', encoding='utf-8') as fp:
        for r in csv.DictReader(fp):
            rows.append({"lat": float(r["lat"]),
                         "lon": float(r["lon"]),
                         "file": r["file"]})
    return rows

def nearest_sources(lat, lon, all_rows, k=2):
    scored = [(haversine_km(lat, lon, r["lat"], r["lon"]), r) for r in all_rows]
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:k]]

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
    out["cloud_cover_total"] = source_vars["cloud_cover_total"]

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
    if len(sys.argv) != 4:
        print("Usage: physics_extrapolate.py LAT LON ISO_TIMESTAMP", file=sys.stderr)
        sys.exit(1)

    lat, lon = float(sys.argv[1]), float(sys.argv[2])
    target_time = parse_time(sys.argv[3])

    params = load_params()
    api_rows = load_api_index()

    with rasterio.open(DEM_PATH) as dem:
        target_elev = read_dem_elevation(lat, lon, dem)
        if target_elev is None:
            print("Target outside DEM or nodata", file=sys.stderr)
            sys.exit(1)

        # choose up to 2 nearest API points
        sources = nearest_sources(lat, lon, api_rows, k=2)
        if not sources:
            print("No API sources found", file=sys.stderr)
            sys.exit(1)

        # read source values
        source_vars_agg = {}
        weight_sum = 0.0
        for s in sources:
            dist_km = haversine_km(lat, lon, s["lat"], s["lon"])
            w = 1.0 / max(dist_km, 0.01)  # inverse-distance weight
            with open(s["file"], "r", encoding="utf-8") as fp:
                full_json = json.load(fp)

            # pick index matching target_time
            times = [datetime.datetime.fromisoformat(t) for t in full_json["hourly"]["time"]]
            try:
                idx = times.index(target_time)
            except ValueError:
                idx = 0  # fallback to first hour

            # build variable dict for this source
            sv = {k: full_json["hourly"][k][idx] for k in (
                    "temperature_2m","relative_humidity_2m","shortwave_radiation",
                    "cloud_cover_total","snow_depth","snowfall",
                    "wind_speed_10m","weather_code",
                    "freezing_level_height","surface_pressure")}

            # weight–average
            if not source_vars_agg:
                source_vars_agg = {k:0.0 for k in sv.keys()}
            for k,v in sv.items():
                source_vars_agg[k] += v * w
            weight_sum += w

            # save first source elevation for lapse
            s["elev"] = read_dem_elevation(s["lat"], s["lon"], dem)

        for k in source_vars_agg.keys():
            source_vars_agg[k] /= weight_sum

        source_elev = sum(s["elev"] for s in sources)/len(sources)

    # apply physics
    pred = apply_physics(target_elev, source_elev, source_vars_agg, params)

    # final output
    pred.update({
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "time": target_time.isoformat()
    })
    json.dump(pred, sys.stdout, separators=(",",":"))
    sys.stdout.write("\n")

if __name__ == "__main__":
    main()