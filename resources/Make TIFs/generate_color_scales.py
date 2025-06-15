#!/usr/bin/env python3
"""
generate_color_scales.py
------------------------
Scan weather data → build a master colour/opacity spec for the 10 core
weather variables.  Output goes to `color_scales.json`.

Assumptions
-----------
* Weather data lives in `resources/meteo_api/weather_data_3hour.json`
  (the 165 MB file already produced) OR in `predictions.csv`
* All values in the JSON are already 3-hour aggregates.
* Terrain points are ≥2 300 m; we add simple lapse/pressure buffers so
  valley pixels will not clip the colour ramp.

Usage
-----
    python generate_color_scales.py          # auto-detect JSON vs. CSV

Result
------
`color_scales.json`, e.g.

{
  "temperature_2m": {
    "min": -25,
    "max": 20,
    "palette": ["#0000ff", "#ffffff", "#ff0000"],
    "interpolation": "linear"
  },
  ...
}
"""

import json, csv, pathlib, sys
from collections import defaultdict
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[2]  # up to repo root
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent  # current script directory
JSON_PATH = ROOT / "resources" / "meteo_api" / "weather_data_3hour.json"
# CSV_PATH  = ROOT / "resources" / "pipeline"   / "predictions.csv"  # Removed - pipeline directory deleted
OUT_PATH  = SCRIPT_DIR / "color_scales.json"  # output in same folder as script

VAR_ORDER = [
    "temperature_2m", "relative_humidity_2m", "shortwave_radiation",
    "cloud_cover", "snow_depth", "snowfall",
    "wind_speed_10m", "weather_code",
    "freezing_level_height", "surface_pressure"
]

# How to colour each variable ----------------------------------------------
DEFAULT_SPECS = {
    "temperature_2m":       dict(palette=["#0000ff","#ffffff","#ff0000"]),
    "relative_humidity_2m": dict(palette=["#ffffff","#00bfff"], opacity=True),
    "shortwave_radiation":  dict(palette=["#000000","#ffff00"]),
    "cloud_cover":          dict(palette=["#ffffff"], opacity=True),
    "snow_depth":           dict(palette=["#f0f8ff","#4169e1"]),
    "snowfall":             dict(palette=["#f0f8ff","#00008b"]),
    "wind_speed_10m":       dict(palette=["#e0ffff","#008080"]),
    "weather_code":         dict(palette=["#ffd700","#808080","#4169e1",
                                          "#ffffff","#ff4500"],
                                 discrete=True),
    "freezing_level_height":dict(palette=["#8b4513","#00ff7f"]),
    "surface_pressure":     dict(palette=["#87cefa","#8b0000"]),
}

# Read weather values -------------------------------------------------------
def read_values():
    vals = defaultdict(list)
    if JSON_PATH.exists():
        with open(JSON_PATH,"r",encoding="utf-8") as fp:
            j = json.load(fp)
        for c in j["coordinates"]:
            hh = c["weather_data_3hour"]["hourly"]
            for k in VAR_ORDER:
                if k in hh:
                    vals[k].extend(hh[k])
    else:
        print("❌ No weather dataset found.", file=sys.stderr)
        sys.exit(1)
    return vals

vals = read_values()

# Calculate min/max and add buffers where needed ----------------------------
result = {}
for var in VAR_ORDER:
    data = [v for v in vals[var] if v is not None]
    if not data:
        continue
    data_min, data_max = min(data), max(data)

    spec = dict(DEFAULT_SPECS[var])  # copy palette / flags
    spec["interpolation"] = "linear"

    # Handle each variable's specific range logic
    if var == "temperature_2m":
        # Small buffer for ~100m elevation difference (DEM 461m vs data 540m)
        elevation_buffer = 0.1 * 6.5  # 100m * 6.5°C/km = 0.65°C
        cushion = 2  # Reduced cushion since data covers full range
        spec.update({
            "min": round(data_min - cushion, 2),
            "api_max": round(data_max, 2),
            "elevation_buffer": round(elevation_buffer, 2),
            "max": round(data_max + elevation_buffer + cushion, 2)
        })
    elif var == "surface_pressure":
        # Small buffer for ~100m elevation difference
        pressure_buffer = 0.1 * 12  # 100m * 12 hPa/100m = 1.2 hPa
        cushion = 5  # Small pressure cushion
        spec.update({
            "min": round(data_min - cushion, 2),
            "api_max": round(data_max, 2),
            "pressure_buffer": round(pressure_buffer, 2),
            "max": round(data_max + pressure_buffer + cushion, 2)
        })
    elif var in {"snowfall", "snow_depth"}:
        # Minimal headroom since data covers full elevation range
        spec.update({
            "min": 0.0,
            "api_max": round(data_max, 2),
            "max": round(data_max * 1.1, 2)  # Just 10% headroom
        })
    elif var == "relative_humidity_2m":
        # Humidity can go to zero, but data covers full range already
        spec.update({
            "min": 0.0,
            "api_max": round(data_max, 2),
            "max": round(data_max, 2)  # No buffer needed
        })
    else:
        # Use data range as-is
        spec.update({
            "min": round(data_min, 2),
            "max": round(data_max, 2)
        })
    
    result[var] = spec

with open(OUT_PATH,"w",encoding="utf-8") as fp:
    json.dump(result, fp, indent=2)

print(f"✅ Wrote {OUT_PATH.relative_to(SCRIPT_DIR)}")