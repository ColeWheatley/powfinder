#!/usr/bin/env python3
"""Unified GeoTIFF generator with caching and SQH integration.

This script replaces the previous per-variable generators. Weather data
is loaded once and interpolation weights are reused for every layer.
Fingerprint files prevent regeneration when the physics have not
changed. Powder snow depth and quality are forward integrated to
produce SQH and skiability layers.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple
import gc
try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

import numpy as np
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Configuration

# Frontend timestamps (for PNG generation)
FRONTEND_TIMESTAMPS = [
    "2025-05-24T09:00:00", "2025-05-24T12:00:00", "2025-05-24T15:00:00", "2025-05-24T18:00:00",
    "2025-05-25T09:00:00", "2025-05-25T12:00:00", "2025-05-25T15:00:00", "2025-05-25T18:00:00",
    "2025-05-26T09:00:00", "2025-05-26T12:00:00", "2025-05-26T15:00:00", "2025-05-26T18:00:00",
    "2025-05-27T09:00:00", "2025-05-27T12:00:00", "2025-05-27T15:00:00", "2025-05-27T18:00:00",
    "2025-05-28T09:00:00", "2025-05-28T12:00:00", "2025-05-28T15:00:00", "2025-05-28T18:00:00",
]

def generate_all_timestamps(start_time: str, end_time: str) -> List[str]:
    """Generate all 3-hour timestamps for the full time series."""
    timestamps = []
    current = datetime.fromisoformat(start_time)
    end_dt = datetime.fromisoformat(end_time)
    
    while current <= end_dt:
        timestamps.append(current.isoformat())
        current += timedelta(hours=3)
    
    return timestamps

# Generate ALL timestamps for TIF creation (including historical)
ALL_TIMESTAMPS = generate_all_timestamps("2025-05-14T00:00:00", "2025-05-28T18:00:00")

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
# Helpers

def load_weather(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], Iterable[str]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    variables = {
        "temperature_2m": [],
        "relative_humidity_2m": [],
        "shortwave_radiation": [],
        "cloud_cover": [],
        "snow_depth": [],
        "snowfall": [],
        "wind_speed_10m": [],
        "weather_code": [],
        "freezing_level_height": [],
        "surface_pressure": [],
    }
    coords = []
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
        for key in variables:
            arr = hourly.get(key)
            if arr is None:
                variables[key].append(np.zeros(len(times), dtype=np.float32))
            else:
                variables[key].append(np.array(arr, dtype=np.float32))

    arrays = {k: np.stack(v) for k, v in variables.items()}
    return np.array(coords, dtype=np.float32), arrays, times


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
# Physics

def physics_temperature(temp, humid, snow, tgt_elev, src_elev, hill):
    lapse = -9.8 + (humid / 100.0) * 5.8
    adj = temp + lapse * ((tgt_elev - src_elev) / 1000.0)
    ins = hill / 32767.0
    adj += (ins - 0.5) * 2.0
    adj = np.where(snow > 0.1, adj - 1.5, adj)
    return adj


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
    ins = hill / 32767.0
    return sf * np.maximum(enh, 0.0) * (1.0 + (0.5 - ins) * 0.1)


def physics_dewpoint(temp, humid, tgt_elev, src_elev):
    # Calculate dewpoint from temperature and humidity using Magnus formula
    a, b = 17.27, 237.7
    alpha = (a * temp) / (b + temp) + np.log(np.maximum(humid, 1e-3) / 100.0)
    dp = (b * alpha) / (a - alpha)
    # Apply elevation adjustment
    return dp - 2.0 * ((tgt_elev - src_elev) / 1000.0)


def physics_pressure(p, tgt_elev, src_elev):
    return p + (src_elev - tgt_elev) * 0.12


def physics_identity(v, *args):
    return v


def physics_cloud_cover(cloud, humid, tgt_elev, src_elev):
    """Cloud cover with humidity adjustment."""
    # Slightly increase cloud cover in high humidity conditions
    return np.clip(cloud + (humid - 50) * 0.1, 0, 100)


def compute_dewpoint(temp, rh):
    a, b = 17.27, 237.7
    alpha = (a * temp) / (b + temp) + np.log(np.maximum(rh, 1e-3) / 100.0)
    return (b * alpha) / (a - alpha)


def update_powder_state(prev_mm, new_snow_mm, temp_C, rad_Wm2):
    """Return new powder depth [mm] and quality [0–1]."""
    DEPTH_MAX_MM = 1000           # 100 cm → 1000 mm
    SETTLING_FRAC = 0.05          # 5 % depth loss per 3 h
    MELT_MM_PER_DEG = 0.4         # very crude melt parameter
    
    # 1. settle (compaction)
    depth_mm = prev_mm * (1 - SETTLING_FRAC)
    
    # 2. simple melt when temps near/above freezing
    if temp_C > -2:
        melt_mm = max(temp_C + 2, 0) * MELT_MM_PER_DEG
        depth_mm = max(depth_mm - melt_mm, 0)
    
    # 3. add fresh snowfall
    depth_mm += new_snow_mm
    
    # 4. cap at sane upper bound
    return min(depth_mm, DEPTH_MAX_MM)

def physics_powder(prev_depth, prev_quality, snowfall, temp, rad, wind, cloud_cover, relative_humidity, timestamp, grid_elev_flat, hours=3):
    """Forward integrate powder conditions, gracefully handling missing data."""
    DEPTH_MAX_MM = 1000  # 100 cm max depth in mm
    
    # Create copies to avoid modifying input arrays - prev_depth is already in mm
    depth_mm = np.array(prev_depth, dtype=np.float32, copy=True)
    quality = np.array(prev_quality, dtype=np.float32, copy=True)

    nodata_mask = depth_mm == NODATA

    # Add new snow (if any)
    new_snow_mm = 0
    if snowfall is not None and not np.all(snowfall == NODATA):
        valid_snowfall = snowfall[snowfall != NODATA]
        if len(valid_snowfall) > 0:
            snowfall_mean = np.mean(valid_snowfall)
            snowfall_max = np.max(valid_snowfall)
            print(f"[DEBUG] Raw snowfall at {timestamp}: max={snowfall_max:.2f}mm, mean={snowfall_mean:.2f}mm")
            
            # snowfall is already in mm from API
            new_snow_mm = np.where(snowfall == NODATA, 0, snowfall)
    else:
        print(f"Warning: No snowfall data for {timestamp}, skipping accumulation")

    # Apply physics using vectorized update_powder_state logic
    # 1. Settling (compaction) - 5% loss per 3h
    depth_mm = depth_mm * (1 - 0.05)
    
    # 2. Melt when temps near/above freezing
    if temp is not None and not np.all(temp == NODATA):
        melt_mm = np.where(temp > -2, np.maximum(temp + 2, 0) * 0.4, 0)
        depth_mm = np.maximum(depth_mm - melt_mm, 0)
    
    # 3. Add fresh snowfall
    depth_mm = depth_mm + new_snow_mm
    
    # 4. Cap at sane upper bound
    depth_mm = np.minimum(depth_mm, DEPTH_MAX_MM)
    
    # Additional wind scour effect
    if wind is not None and not np.all(wind == NODATA):
        # Wind scouring: 0.02 * wind_speed² mm per step for exposed areas
        wind_scour = 0.02 * np.square(wind) * hours / 3.0  # normalize to 3h step
        depth_mm = np.maximum(0, depth_mm - wind_scour)

    # Update quality based on conditions
    if snowfall is not None and not np.all(snowfall == NODATA):
        quality = np.where(new_snow_mm > 1, 1.0, quality * 0.98)

    if rad is not None and not np.all(rad == NODATA):
        quality = np.where(rad > 200, quality * 0.95, quality)
    if temp is not None and not np.all(temp == NODATA):
        quality = np.where(temp > -2, quality * 0.96, quality)

    hour = datetime.fromisoformat(timestamp).hour
    is_night = (hour < 6) or (hour > 21)
    if is_night:
        if cloud_cover is not None and not np.all(cloud_cover == NODATA):
            quality = np.where(cloud_cover > 70, quality * 0.96, quality * 0.98)
        else:
            quality *= 0.98

    if (relative_humidity is not None and not np.all(relative_humidity == NODATA)) and (
        temp is not None and not np.all(temp == NODATA)
    ):
        quality = np.where((relative_humidity > 90) & (temp < -5),
                          np.minimum(quality * 1.02, 1.0), quality)

    depth_mm[nodata_mask] = NODATA
    quality[nodata_mask] = NODATA

    return depth_mm, np.clip(quality, 0.0, 1.0)

def physics_skiability(wind, codes, tgt_elev, src_elev):
    # Use a base skiability of 1.0 and apply penalties
    penalties = np.zeros_like(codes, dtype=float)
    penalties = np.where(codes >= 95, 0.7, penalties)
    penalties = np.where((codes >= 80) & (codes < 95), 0.5, penalties)
    penalties = np.where((codes >= 51) & (codes < 80), 0.3, penalties)
    penalties = np.where((codes >= 45) & (codes < 51), 0.2, penalties)
    ski = 1.0 - 0.1 * wind - penalties
    ski = np.clip(ski, 0, None)
    
    # Preserve NODATA from inputs
    nodata_mask = (wind == NODATA) | (codes == NODATA)
    ski[nodata_mask] = NODATA
    return ski

# ---------------------------------------------------------------------------
# Variable configuration

class Variable:
    def __init__(
        self,
        key: str,
        deps: Tuple[str, ...],
        physics: Callable,
        units: str,
        needs_hillshade: bool = False,
        outputs: List[str] | None = None,
        depends_on_previous: bool = False,
    ) -> None:
        self.key = key
        self.deps = deps or (key,)
        self.physics = physics
        self.units = units
        self.needs_hillshade = needs_hillshade
        self.outputs = outputs or [key]
        self.depends_on_previous = depends_on_previous
        self.fingerprint = hashlib.sha1(inspect.getsource(physics).encode("utf-8")).hexdigest()


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
    Variable(
        "powder",
        ("snowfall", "temperature_2m", "shortwave_radiation", "wind_speed_10m"),
        physics_powder,
        "cm",
        False,
        outputs=["powder_depth", "powder_quality"],
        depends_on_previous=True,
    ),
    Variable("skiability", ("wind_speed_10m", "weather_code"), physics_skiability, "ski"),
]

VAR_BY_KEY = {v.key: v for v in VARIABLES}

# Weather variables that can be processed independently
WEATHER_VARS = [
    "temperature_2m",
    "dewpoint_2m",
    "relative_humidity_2m",
    "shortwave_radiation",
    "snowfall",
    "snow_depth",
    "wind_speed_10m",
    "weather_code",
    "surface_pressure",
    "cloud_cover",
    "freezing_level_height",
]



# ---------------------------------------------------------------------------
# Caching

def should_skip(var: Variable) -> bool:
    fp_file = OUTPUT_BASE / f"{var.key}.hash"
    if not fp_file.exists():
        return False
    if fp_file.read_text().strip() != var.fingerprint:
        return False
    for ts in ALL_TIMESTAMPS:
        for out in var.outputs:
            tif = OUTPUT_BASE / ts / f"{out}.tif"
            if not tif.exists():
                return False
    return True


def save_fingerprint(var: Variable) -> None:
    (OUTPUT_BASE / f"{var.key}.hash").write_text(var.fingerprint)

# ---------------------------------------------------------------------------
# Interpolation helper

def interpolate(var: Variable, ti: int, hill: np.ndarray | None, arrays: Dict[str, np.ndarray], idxs: np.ndarray, weights: np.ndarray, grid_elev_flat: np.ndarray, coords_elev: np.ndarray, grid_shape: tuple) -> np.ndarray:
    dep_arrays = [arrays[d][:, ti] for d in var.deps if d in arrays]
    data_k = [arr[idxs] for arr in dep_arrays]
    if var.needs_hillshade:
        result = var.physics(*data_k, grid_elev_flat[:, None], coords_elev[idxs], hill[:, None])
    elif len(data_k) == 1:
        result = var.physics(data_k[0], grid_elev_flat[:, None], coords_elev[idxs])
    else:
        result = var.physics(*data_k, grid_elev_flat[:, None], coords_elev[idxs])
    out_flat = np.sum(result * weights, axis=1)
    out_flat = np.where(grid_elev_flat == NODATA, NODATA, out_flat)
    return out_flat.reshape(grid_shape)

# ---------------------------------------------------------------------------

def main(vars_to_process: List[str]) -> None:
    start_time = time.time()
    tifs_created = 0
    print(f"🚀 Starting TIF generation at {datetime.now().strftime('%H:%M:%S')}")
    
    if not WEATHER_PATH.exists():
        raise FileNotFoundError(WEATHER_PATH)

    coords, arrays, times = load_weather(WEATHER_PATH)
    tindex = time_index(times)

    dewpoint = compute_dewpoint(arrays["temperature_2m"], arrays["relative_humidity_2m"])
    arrays["dewpoint_2m"] = dewpoint

    with COLOR_SCALE_JSON.open("r", encoding="utf-8") as fp:
        color_scales = json.load(fp)

    with rasterio.open(ELEVATION_TIF) as elev_src:
        grid_elev, xs, ys = build_grid(elev_src)
        profile = elev_src.profile

    transformer = Transformer.from_crs("EPSG:4326", profile["crs"], always_xy=True)
    tree = precompute_kdtree(coords[:, :2], transformer)
    grid_xy = np.column_stack([xs.ravel(), ys.ravel()])
    dists, idxs = tree.query(grid_xy, k=min(4, len(coords)))
    weights = 1.0 / np.maximum(dists, 1e-6) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    idxs = idxs.astype(np.int32)
    grid_elev_flat = grid_elev.filled(NODATA).ravel()
    coords_elev = coords[:, 2]
    profile.update(compress="lzw")

    hillshade = {}
    for p, path in HILLSHADE_TIFS.items():
        if path.exists():
            with rasterio.open(path) as hs:
                hillshade[p] = hs.read(1, masked=True).ravel()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Sort timestamps chronologically for proper forward integration
    sorted_timestamps = sorted(ALL_TIMESTAMPS, key=lambda x: datetime.fromisoformat(x))


    # Define which variables are needed for powder physics
    POWDER_REQUIRED_VARS = [
        "snowfall",
        "temperature_2m",
        "shortwave_radiation",
        "wind_speed_10m",
        "cloud_cover",
        "relative_humidity_2m",
        "powder",
        "skiability",
    ]

    # Order variables: weather first, then powder, then skiability
    ordered_vars = [v for v in WEATHER_VARS if v in vars_to_process]
    if "powder" in vars_to_process:
        ordered_vars.append("powder")
    if "skiability" in vars_to_process:
        ordered_vars.append("skiability")

    # Determine which variables can be skipped entirely based on fingerprints
    skip_map = {k: should_skip(VAR_BY_KEY[k]) for k in ordered_vars}

    prev_depth = None
    prev_quality = None

    for ts in sorted_timestamps:
        if ts not in tindex:
            continue

        ti = tindex[ts]
        hour = datetime.fromisoformat(ts).hour
        period = PERIOD_FROM_HOUR.get(hour)
        hs = hillshade.get(period, np.zeros_like(grid_elev_flat))

        # Determine which weather variables are actually needed
        needed_vars = set()
        for var_key in WEATHER_VARS:
            if var_key in POWDER_REQUIRED_VARS or ts in FRONTEND_TIMESTAMPS:
                if var_key in vars_to_process:
                    needed_vars.add(var_key)
                elif var_key in POWDER_REQUIRED_VARS:
                    needed_vars.add(var_key)
        
        # Add dependencies for skiability
        if "skiability" in vars_to_process:
            needed_vars.add("wind_speed_10m")
            needed_vars.add("weather_code")

        weather_data = {}
        for var_key in needed_vars:
            weather_data[var_key] = interpolate(
                VAR_BY_KEY[var_key],
                ti,
                hs,
                arrays,
                idxs,
                weights,
                grid_elev_flat,
                coords_elev,
                grid_elev.shape,
            )

        out_dir = OUTPUT_BASE / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        for var_key in ordered_vars:
            if skip_map.get(var_key):
                continue
            
            # Determine if we should process this variable for this timestamp
            should_process = False
            
            if ts in FRONTEND_TIMESTAMPS:
                # Always process ALL variables for frontend timestamps
                should_process = True
            elif var_key in POWDER_REQUIRED_VARS:
                # Process powder-required variables for all timestamps (historical)
                should_process = True
            else:
                # Skip non-powder variables for historical timestamps
                should_process = False
                
            if not should_process:
                continue

            var = VAR_BY_KEY[var_key]

            if var.depends_on_previous:
                if prev_depth is None:
                    prev_depth, prev_quality = load_powder_state(ts, grid_elev.shape, sorted_timestamps)

                depth, qual = physics_powder(
                    prev_depth,
                    prev_quality,
                    weather_data["snowfall"].ravel(),
                    weather_data["temperature_2m"].ravel(),
                    weather_data["shortwave_radiation"].ravel(),
                    weather_data["wind_speed_10m"].ravel(),
                    weather_data["cloud_cover"].ravel(),
                    weather_data["relative_humidity_2m"].ravel(),
                    ts,
                    grid_elev_flat,
                )
                print(
                    f"[DEBUG] Powder state at {ts}: depth mean {np.nanmean(depth):.2f}, quality mean {np.nanmean(qual):.2f}"
                )

                prev_depth, prev_quality = depth, qual

                results = {
                    "powder_depth": depth.reshape(grid_elev.shape),
                    "powder_quality": qual.reshape(grid_elev.shape),
                }
            elif var_key in WEATHER_VARS:
                results = {var.outputs[0]: weather_data[var_key]}
            elif var_key == "skiability":
                ski = physics_skiability(
                    weather_data["wind_speed_10m"],
                    weather_data["weather_code"],
                    grid_elev,
                    grid_elev,
                )
                results = {"skiability": ski.reshape(grid_elev.shape)}

            else:
                data = interpolate(
                    var,
                    ti,
                    hs,
                    arrays,
                    idxs,
                    weights,
                    grid_elev_flat,
                    coords_elev,
                    grid_elev.shape,
                )
                results = {var.outputs[0]: data}

            # Process results for this variable
            for out_name, data in results.items():
                scale = color_scales.get(out_name, {})
                if "min" in scale and "max" in scale:
                    vmin = scale["min"]
                    vmax = scale["max"]
                else:
                    valid = data[data != NODATA]
                    vmin = float(np.nanmin(valid)) if valid.size else 0.0
                    vmax = float(np.nanmax(valid)) if valid.size else 1.0
                clipped = np.clip(data, vmin, vmax)
                scaled = (clipped - vmin) / (vmax - vmin) if vmax != vmin else clipped
                out_byte = (scaled * 254 + 1).astype(np.uint8)
                out_byte[data == NODATA] = 0
                profile.update(dtype="uint8", nodata=0)
                out_path = out_dir / f"{out_name}.tif"
                if out_path.exists():
                    out_path.unlink()
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(out_byte, 1)
                    dst.update_tags(units=var.units, processing_time=datetime.utcnow().isoformat(), source_points_count=len(coords))
                print(f"[DEBUG] {out_name} byte range: {out_byte.min()}-{out_byte.max()}")
                tifs_created += 1

        # Release memory for this timestamp
        del weather_data
        gc.collect()
        if psutil:
            mem_mb = psutil.Process().memory_info().rss / 1024 ** 2
            print(f"[DEBUG] Memory after {ts}: {mem_mb:.1f} MB")

    for var_key in ordered_vars:
        if not skip_map.get(var_key):
            save_fingerprint(VAR_BY_KEY[var_key])
    
    # Print timing summary
    end_time = time.time()
    total_time = end_time - start_time
    first_date = sorted_timestamps[0].split('T')[0]
    last_date = sorted_timestamps[-1].split('T')[0]
    
    print(f"\n✅ Generated {tifs_created} TIFs from {first_date} to {last_date}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    if tifs_created > 0:
        print(f"⚡ Average: {total_time/tifs_created:.3f}s per TIF")

def load_powder_state(timestamp: str, grid_shape: tuple, sorted_timestamps: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load previous powder state from TIFs, or initialize to zero."""
    # Find previous timestamp
    current_idx = sorted_timestamps.index(timestamp)
    if current_idx == 0:
        # First timestamp - initialize to zero
        flat_size = np.prod(grid_shape)
        return np.zeros(flat_size, dtype=np.float32), np.zeros(flat_size, dtype=np.float32)
    
    prev_timestamp = sorted_timestamps[current_idx - 1]
    depth_path = OUTPUT_BASE / prev_timestamp / "powder_depth.tif"
    quality_path = OUTPUT_BASE / prev_timestamp / "powder_quality.tif"
    
    if depth_path.exists() and quality_path.exists():
        with rasterio.open(depth_path) as dsrc, rasterio.open(quality_path) as qsrc:
            # Convert from uint8 back to physical units
            depth_bytes = dsrc.read(1).astype(np.float32)
            quality_bytes = qsrc.read(1).astype(np.float32)
            
            # CRITICAL FIX: Properly decode from byte values to physical units
            # depth_mm = byte_val / 255 * DEPTH_MAX_MM (not just byte_val!)
            DEPTH_MAX_MM = 1000  # 100 cm in mm
            depth_mm = depth_bytes / 255.0 * DEPTH_MAX_MM  # Convert bytes to mm
            quality_phys = quality_bytes / 255.0  # Convert bytes to 0-1 range
            
            return depth_mm.ravel(), quality_phys.ravel()
    else:
        # Fallback to zero initialization if previous state not found
        flat_size = np.prod(grid_shape)
        return np.zeros(flat_size, dtype=np.float32), np.zeros(flat_size, dtype=np.float32)

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args:
        unknown = [a for a in args if a not in VAR_BY_KEY]
        if unknown:
            raise SystemExit(f"Unknown variable(s): {', '.join(unknown)}")
        vars_to_run = args
    else:
        vars_to_run = [v.key for v in VARIABLES]
    main(vars_to_run)

