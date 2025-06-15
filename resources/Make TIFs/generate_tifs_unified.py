#!/usr/bin/env python3
"""Unified GeoTIFF generator for PowFinder with SQH support.

This script consolidates all TIF generation into a single pipeline, including
forward-integrated powder tracking (SQH). It replaces the previous suite of
individual generator scripts.

Usage:
    python generate_tifs_unified.py [variable ...]
    
If no variables specified, generates all variables for all timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import xy
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

# Paths
WEATHER_PATH = Path("resources/meteo_api/weather_data_3hour.json")
ELEVATION_TIF = Path("resources/terrains/tirol_100m_float.tif")
COLOR_SCALE_JSON = Path("resources/Make TIFs/color_scales.json")
OUTPUT_BASE = Path("TIFS/100m_resolution")

# Hillshade files for the 4 daylight periods
HILLSHADE_TIFS = {
    1: Path("resources/hillshade/hillshade_100m_period1.tif"),  # 09:00
    2: Path("resources/hillshade/hillshade_100m_period2.tif"),  # 12:00
    3: Path("resources/hillshade/hillshade_100m_period3.tif"),  # 15:00
    4: Path("resources/hillshade/hillshade_100m_period4.tif"),  # 18:00
}

# ---------------------------------------------------------------------------
# Variable Class


class Variable:
    """Configuration for a weather variable or derived layer."""

    def __init__(
        self,
        key: str,
        physics: Callable,
        unit: str,
        weather_deps: Optional[Tuple[str, ...]] = None,
        tif_deps: Optional[Tuple[str, ...]] = None,
        needs_hill: bool = False,
        outputs: Optional[List[str]] = None,
        depends_on_previous: bool = False,
    ):
        self.key = key
        self.physics = physics
        self.unit = unit
        self.weather_deps = list(weather_deps or [])
        self.tif_deps = list(tif_deps or [])
        self.needs_hill = needs_hill
        self.outputs = outputs or [key]
        self.depends_on_previous = depends_on_previous
        
        # Compute fingerprint of physics function for caching
        self.fingerprint = hashlib.sha1(
            inspect.getsource(physics).encode("utf-8")
        ).hexdigest()


# ---------------------------------------------------------------------------
# Physics Functions

def apply_adiabatic_lapse(temperature: np.ndarray, ref_elevation: np.ndarray, 
                         target_elevation: np.ndarray) -> np.ndarray:
    """Apply temperature lapse rate adjustment."""
    lapse_rate = 6.5  # °C per 1000m
    elevation_diff = target_elevation - ref_elevation
    return temperature - (elevation_diff * lapse_rate / 1000)


def physics_temperature(deps: Dict[str, np.ndarray], elev: np.ndarray, 
                       hill: Optional[np.ndarray]) -> np.ndarray:
    """Temperature with elevation adjustment and solar heating."""
    temp = deps["temperature_2m"].copy()
    ref_elev = deps["elevation"]
    
    # Apply lapse rate
    temp = apply_adiabatic_lapse(temp, ref_elev, elev)
    
    # Solar heating if hillshade available
    if hill is not None and "shortwave_radiation" in deps:
        insolar = hill * deps["shortwave_radiation"]
        solar_heating = insolar * 0.005  # 0.5°C per 100 W/m²
        temp += solar_heating
    
    return temp


def physics_freezing_level(deps: Dict[str, np.ndarray], elev: np.ndarray,
                          hill: Optional[np.ndarray]) -> np.ndarray:
    """Freezing level height adjustment."""
    fl = deps["freezing_level_height"].copy()
    
    # Temperature-based adjustment
    if "temperature_2m" in deps:
        temp_adjustment = (deps["temperature_2m"] - 10) * 20
        fl += temp_adjustment
    
    return fl


def physics_relative_humidity(deps: Dict[str, np.ndarray], elev: np.ndarray,
                             hill: Optional[np.ndarray]) -> np.ndarray:
    """Relative humidity with elevation adjustment."""
    rh = deps["relative_humidity_2m"].copy()
    ref_elev = deps["elevation"]
    
    # Increase RH with elevation
    elevation_factor = (elev - ref_elev) * 0.002
    rh = rh * (1 + elevation_factor)
    
    return np.clip(rh, 0, 100)


def physics_cloud_cover(deps: Dict[str, np.ndarray], elev: np.ndarray,
                       hill: Optional[np.ndarray]) -> np.ndarray:
    """Cloud cover - pass through."""
    return deps["cloud_cover"].copy()


def physics_shortwave_radiation(deps: Dict[str, np.ndarray], elev: np.ndarray,
                               hill: Optional[np.ndarray]) -> np.ndarray:
    """Shortwave radiation modulated by hillshade."""
    radiation = deps["shortwave_radiation"].copy()
    
    if hill is not None:
        # Simple multiplication by hillshade factor
        radiation = radiation * hill
    
    return radiation


def physics_snow_depth(deps: Dict[str, np.ndarray], elev: np.ndarray,
                      hill: Optional[np.ndarray]) -> np.ndarray:
    """Total snow depth from weather model."""
    return deps["snow_depth"].copy()


def physics_snowfall(deps: Dict[str, np.ndarray], elev: np.ndarray,
                    hill: Optional[np.ndarray]) -> np.ndarray:
    """Snowfall with elevation enhancement."""
    snowfall = deps["snowfall"].copy()
    ref_elev = deps["elevation"]
    
    # Orographic enhancement
    elevation_factor = np.maximum(0, (elev - ref_elev) / 1000)
    snowfall = snowfall * (1 + elevation_factor * 0.3)
    
    return snowfall


def physics_wind_speed(deps: Dict[str, np.ndarray], elev: np.ndarray,
                      hill: Optional[np.ndarray]) -> np.ndarray:
    """Wind speed with elevation adjustment."""
    wind = deps["wind_speed_10m"].copy()
    ref_elev = deps["elevation"]
    
    # Increase wind with elevation
    elevation_factor = (elev - ref_elev) / 1000
    wind = wind * (1 + elevation_factor * 0.2)
    
    return wind


def physics_dewpoint(deps: Dict[str, np.ndarray], elev: np.ndarray,
                    hill: Optional[np.ndarray]) -> np.ndarray:
    """Calculate dewpoint from temperature and RH."""
    # Get adjusted temperature and RH
    temp = physics_temperature(deps, elev, hill)
    rh = physics_relative_humidity(deps, elev, hill)
    
    # Magnus formula
    a, b = 17.27, 237.7
    alpha = ((a * temp) / (b + temp)) + np.log(rh / 100.0)
    dewpoint = (b * alpha) / (a - alpha)
    
    return dewpoint


def physics_weather_code(deps: Dict[str, np.ndarray], elev: np.ndarray,
                        hill: Optional[np.ndarray]) -> np.ndarray:
    """Weather code - pass through."""
    return deps["weather_code"].copy()


def physics_surface_pressure(deps: Dict[str, np.ndarray], elev: np.ndarray,
                           hill: Optional[np.ndarray]) -> np.ndarray:
    """Surface pressure with elevation adjustment."""
    pressure = deps["surface_pressure"].copy()
    ref_elev = deps["elevation"]
    
    # Barometric formula
    elevation_diff = elev - ref_elev
    pressure = pressure * np.exp(-elevation_diff / 8500)
    
    return pressure


def physics_powder(deps: Dict[str, np.ndarray], elev: np.ndarray,
                  hill: Optional[np.ndarray],
                  previous_state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                  dt_hours: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward-integrate powder conditions.
    
    Returns:
        (powder_depth, powder_quality) both as numpy arrays
    """
    if previous_state is None:
        # Initialize - no powder exists at start
        shape = elev.shape
        powder_depth = np.zeros(shape)
        powder_quality = np.zeros(shape)
    else:
        powder_depth, powder_quality = previous_state
        powder_depth = powder_depth.copy()
        powder_quality = powder_quality.copy()
    
    # Get weather variables with physics adjustments
    temp = physics_temperature(deps, elev, hill)
    snowfall = physics_snowfall(deps, elev, hill)
    wind = physics_wind_speed(deps, elev, hill)
    radiation = physics_shortwave_radiation(deps, elev, hill) if hill is not None else 0
    
    # Add new snowfall (mm to cm)
    new_snow_cm = snowfall * 0.1
    powder_depth += new_snow_cm
    
    # Reset quality if significant new snow
    significant_snow = new_snow_cm > 1.0
    powder_quality = np.where(significant_snow, 1.0, powder_quality)
    
    # Degradation factors
    
    # Temperature effects
    melt_rate = np.maximum(0, temp) * 0.02 * dt_hours
    powder_depth = np.maximum(0, powder_depth - melt_rate)

    # Wind scouring
    wind_scour = wind * 0.01 * dt_hours
    powder_depth = np.maximum(0, powder_depth - wind_scour)

    # Natural compaction
    compaction = 0.005 * dt_hours
    powder_depth = np.maximum(0, powder_depth - compaction)
    
    # Quality degradation
    quality_decay = 0.98  # 2% per 3 hours base rate
    
    # Temperature effects on quality
    temp_factor = np.where(temp > -2, 0.96, 1.0)  # Faster degradation when warm
    
    # Solar effects on quality
    if hill is not None:
        solar_factor = np.where(radiation > 200, 0.95, 1.0)  # Sun crust
    else:
        solar_factor = 1.0
    
    # Apply all quality factors
    powder_quality *= quality_decay * temp_factor * solar_factor
    
    # Zero out quality where there's no powder
    powder_quality = np.where(powder_depth < 0.1, 0.0, powder_quality)
    
    # Ensure bounds
    powder_quality = np.clip(powder_quality, 0.0, 1.0)
    powder_depth = np.maximum(0.0, powder_depth)
    
    return powder_depth, powder_quality


def physics_skiability(deps: Dict[str, np.ndarray], elev: np.ndarray,
                      hill: Optional[np.ndarray],
                      powder_state: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Skiability combines powder quality with current weather conditions.
    Only relevant for future timestamps.
    """
    powder_depth, powder_quality = powder_state
    
    # Start with powder quality
    skiability = powder_quality.copy()
    
    # Get current weather
    temp = physics_temperature(deps, elev, hill)
    wind = physics_wind_speed(deps, elev, hill)
    weather_code = deps["weather_code"]
    cloud_cover = deps["cloud_cover"]
    
    # Weather adjustments
    
    # Temperature bonus/penalty
    ideal_temp = -5.0
    temp_factor = 1.0 - np.abs(temp - ideal_temp) * 0.01  # -1% per degree from ideal
    temp_factor = np.clip(temp_factor, 0.7, 1.1)
    
    # Wind penalty
    wind_factor = 1.0 - (wind / 100)  # -1% per m/s
    wind_factor = np.clip(wind_factor, 0.5, 1.0)
    
    # Visibility/weather penalty
    # Weather codes: 0-49 good, 50-69 ok, 70+ bad
    weather_factor = np.where(weather_code < 50, 1.0,
                             np.where(weather_code < 70, 0.9, 0.7))
    
    # Apply all factors
    skiability *= temp_factor * wind_factor * weather_factor
    
    # Ensure bounds [0, 1]
    skiability = np.clip(skiability, 0.0, 1.0)
    
    # Zero out where no powder
    skiability = np.where(powder_depth < 1.0, 0.0, skiability)

    return skiability


def physics_sqh(powder_state: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Return powder quality for SQH visualisation."""
    _, quality = powder_state
    return quality.copy()


# ---------------------------------------------------------------------------
# Variable Definitions

VARIABLES = {
    "temperature_2m": Variable(
        key="temperature_2m",
        physics=physics_temperature,
        unit="celsius",
        weather_deps=("temperature_2m", "elevation", "shortwave_radiation"),
        needs_hill=True
    ),
    
    "freezing_level_height": Variable(
        key="freezing_level_height",
        physics=physics_freezing_level,
        unit="meters",
        weather_deps=("freezing_level_height", "temperature_2m")
    ),
    
    "relative_humidity_2m": Variable(
        key="relative_humidity_2m",
        physics=physics_relative_humidity,
        unit="percent",
        weather_deps=("relative_humidity_2m", "elevation")
    ),
    
    "cloud_cover": Variable(
        key="cloud_cover",
        physics=physics_cloud_cover,
        unit="percent",
        weather_deps=("cloud_cover",)
    ),
    
    "shortwave_radiation": Variable(
        key="shortwave_radiation",
        physics=physics_shortwave_radiation,
        unit="watts_per_m2",
        weather_deps=("shortwave_radiation",),
        needs_hill=True
    ),
    
    "snow_depth": Variable(
        key="snow_depth",
        physics=physics_snow_depth,
        unit="cm",
        weather_deps=("snow_depth",)
    ),
    
    "snowfall": Variable(
        key="snowfall",
        physics=physics_snowfall,
        unit="mm",
        weather_deps=("snowfall", "elevation")
    ),
    
    "wind_speed_10m": Variable(
        key="wind_speed_10m",
        physics=physics_wind_speed,
        unit="meters_per_second",
        weather_deps=("wind_speed_10m", "elevation")
    ),
    
    "dewpoint_2m": Variable(
        key="dewpoint_2m",
        physics=physics_dewpoint,
        unit="celsius",
        weather_deps=("temperature_2m", "relative_humidity_2m", "elevation", "shortwave_radiation"),
        needs_hill=True
    ),
    
    "weather_code": Variable(
        key="weather_code",
        physics=physics_weather_code,
        unit="code",
        weather_deps=("weather_code",)
    ),
    
    "surface_pressure": Variable(
        key="surface_pressure",
        physics=physics_surface_pressure,
        unit="hPa",
        weather_deps=("surface_pressure", "elevation")
    ),
    
    "powder": Variable(
        key="powder",
        physics=physics_powder,
        unit="powder",
        weather_deps=("temperature_2m", "snowfall", "wind_speed_10m", "shortwave_radiation", "elevation"),
        needs_hill=True,
        outputs=["powder_depth", "powder_quality"],
        depends_on_previous=True
    ),

    "sqh": Variable(
        key="sqh",
        physics=physics_sqh,
        unit="quality",
        tif_deps=("powder_depth", "powder_quality"),
    ),
    
    "skiability": Variable(
        key="skiability",
        physics=physics_skiability,
        unit="quality",
        weather_deps=("temperature_2m", "wind_speed_10m", "weather_code", "cloud_cover", "elevation", "shortwave_radiation"),
        tif_deps=("powder_depth", "powder_quality"),
        needs_hill=True
    ),
}


def topo_sort(var_names: List[str]) -> List[str]:
    """Topologically sort variables based on tif dependencies."""
    # Map output name -> variable key
    output_to_var = {}
    for v in VARIABLES.values():
        for out in v.outputs:
            output_to_var[out] = v.key

    # Expand dependencies
    needed = set()
    stack = list(var_names)
    while stack:
        name = stack.pop()
        if name in needed:
            continue
        needed.add(name)
        var = VARIABLES[name]
        for dep in var.tif_deps:
            dep_var = output_to_var.get(dep)
            if dep_var and dep_var not in needed:
                stack.append(dep_var)

    # Build graph
    in_deg = {n: 0 for n in needed}
    edges = {n: set() for n in needed}
    for n in needed:
        var = VARIABLES[n]
        for dep in var.tif_deps:
            dep_var = output_to_var.get(dep)
            if dep_var and dep_var in needed:
                edges[dep_var].add(n)
                in_deg[n] += 1

    # Kahn's algorithm
    queue = [n for n, d in in_deg.items() if d == 0]
    ordered = []
    while queue:
        cur = queue.pop(0)
        ordered.append(cur)
        for nxt in edges[cur]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)

    if len(ordered) != len(needed):
        raise RuntimeError("Cyclic dependency detected")

    # Keep only requested vars in resulting order
    return [v for v in ordered if v in var_names]


# ---------------------------------------------------------------------------
# Utility Functions

def load_weather_data() -> Dict[str, Any]:
    """Load weather data from JSON."""
    print(f"Loading weather data from {WEATHER_PATH}")
    with open(WEATHER_PATH, "r") as f:
        return json.load(f)


def load_elevation() -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load elevation data and metadata."""
    with rasterio.open(ELEVATION_TIF) as src:
        elevation = src.read(1)
        metadata = {
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height
        }
    return elevation, metadata


def load_hillshade(period: int) -> Optional[np.ndarray]:
    """Load hillshade for given period, or None if not available."""
    if period not in HILLSHADE_TIFS:
        return None
    
    path = HILLSHADE_TIFS[period]
    if not path.exists():
        return None
    
    with rasterio.open(path) as src:
        return src.read(1)


def get_period_from_timestamp(timestamp: str) -> Optional[int]:
    """Get hillshade period (1-4) from timestamp hour."""
    hour = int(timestamp.split("T")[1].split(":")[0])
    period_map = {9: 1, 12: 2, 15: 3, 18: 4}
    return period_map.get(hour)


def build_kdtree(weather_data: Dict[str, Any]) -> Tuple[cKDTree, np.ndarray]:
    """Build KDTree for spatial interpolation."""
    lons = np.array([float(coord["coordinate_info"]["longitude"]) for coord in weather_data["coordinates"]])
    lats = np.array([float(coord["coordinate_info"]["latitude"]) for coord in weather_data["coordinates"]])
    
    # Project to Web Mercator (EPSG:3857) for web mapping compatibility
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform(lons, lats)
    points = np.column_stack([xs, ys])
    
    return cKDTree(points), points


def interpolate_to_grid(
    values: np.ndarray,
    kdtree: cKDTree,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    k: int = 4
) -> np.ndarray:
    """Interpolate point values to grid using IDW."""
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    distances, indices = kdtree.query(grid_points, k=k)
    
    # Inverse distance weighting
    weights = 1.0 / (distances + 1e-10)
    weights /= weights.sum(axis=1, keepdims=True)
    
    interpolated = np.sum(values[indices] * weights, axis=1)
    return interpolated.reshape(grid_x.shape)


def should_regenerate(var: Variable, timestamp: str, output_dir: Path) -> bool:
    """Check if variable needs regeneration based on fingerprint."""
    # Check all outputs for this variable
    for output_name in var.outputs:
        fp_file = output_dir / f"{output_name}.hash"
        tif_file = output_dir / f"{output_name}.tif"
        
        if not tif_file.exists():
            return True
        
        if not fp_file.exists():
            return True
        
        stored_fp = fp_file.read_text().strip()
        if stored_fp != var.fingerprint:
            return True
    
    return False


def save_fingerprint(var: Variable, output_dir: Path):
    """Save fingerprint for all outputs of a variable."""
    for output_name in var.outputs:
        fp_file = output_dir / f"{output_name}.hash"
        fp_file.write_text(var.fingerprint)


def write_geotiff(
    data: np.ndarray,
    output_path: Path,
    metadata: Dict[str, Any],
    unit: str
):
    """Write data to GeoTIFF file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=metadata["height"],
        width=metadata["width"],
        count=1,
        dtype=np.float32,
        crs=metadata["crs"],
        transform=metadata["transform"],
        compress="lzw"
    ) as dst:
        dst.write(data.astype(np.float32), 1)
        dst.update_tags(unit=unit)


def load_previous_powder_state(timestamp_idx: int, timestamps: List[str],
                              output_base: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load the previous powder state for forward integration."""
    if timestamp_idx == 0:
        return None
    
    prev_timestamp = timestamps[timestamp_idx - 1]
    prev_dir = output_base / prev_timestamp
    
    depth_path = prev_dir / "powder_depth.tif"
    quality_path = prev_dir / "powder_quality.tif"
    
    if not depth_path.exists() or not quality_path.exists():
        return None
    
    with rasterio.open(depth_path) as src:
        depth = src.read(1)
    
    with rasterio.open(quality_path) as src:
        quality = src.read(1)
    
    return depth, quality


def load_powder_state_for_timestamp(timestamp: str, output_base: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load powder state for the given timestamp if available."""
    ts_dir = output_base / timestamp
    depth_path = ts_dir / "powder_depth.tif"
    quality_path = ts_dir / "powder_quality.tif"
    if not depth_path.exists() or not quality_path.exists():
        return None
    with rasterio.open(depth_path) as src:
        depth = src.read(1)
    with rasterio.open(quality_path) as src:
        quality = src.read(1)
    return depth, quality


def get_all_timestamps_from_weather(weather_data: Dict[str, Any]) -> List[str]:
    """Extract all available timestamps from weather data."""
    # Get timestamps from first coordinate
    first_coord = weather_data["coordinates"][0]
    return sorted(first_coord["weather_data_3hour"]["hourly"]["time"])


# ---------------------------------------------------------------------------
# Main Processing

def process_timestamp(
    timestamp: str,
    timestamp_idx: int,
    all_timestamps: List[str],
    weather_data: Dict[str, Any],
    elevation: np.ndarray,
    metadata: Dict[str, Any],
    kdtree: cKDTree,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    variables: List[str],
    powder_state: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Process all variables for a single timestamp."""
    
    output_dir = OUTPUT_BASE / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load hillshade if available
    period = get_period_from_timestamp(timestamp)
    hillshade = load_hillshade(period) if period else None
    
    # Prepare interpolated weather data
    weather_grids = {}
    
    # Extract point values for this timestamp
    time_idx = None
    for coord in weather_data["coordinates"]:
        times = coord["weather_data_3hour"]["hourly"]["time"]
        if timestamp in times:
            time_idx = times.index(timestamp)
            break
    
    if time_idx is None:
        print(f"Warning: Timestamp {timestamp} not found in weather data")
        return powder_state
    
    for var_name in ["temperature_2m", "relative_humidity_2m", "cloud_cover",
                     "shortwave_radiation", "snow_depth", "snowfall",
                     "wind_speed_10m", "weather_code", "freezing_level_height",
                     "surface_pressure"]:
        values = []
        for coord in weather_data["coordinates"]:
            hourly_data = coord["weather_data_3hour"]["hourly"]
            if var_name in hourly_data and len(hourly_data[var_name]) > time_idx:
                values.append(hourly_data[var_name][time_idx])
            else:
                # Use a sensible default or skip
                print(f"Warning: Missing {var_name} for {timestamp}")
                values.append(0)
        
        values = np.array(values)
        weather_grids[var_name] = interpolate_to_grid(values, kdtree, grid_x, grid_y)
    
    # Add elevation grid for lapse rate calculations
    # Parse elevation strings like "2505 m" to float values
    location_elevations = []
    for coord in weather_data["coordinates"]:
        elev_str = coord["coordinate_info"]["elevation"]
        try:
            if isinstance(elev_str, str):
                if elev_str.endswith(" m"):
                    elev_val = float(elev_str.replace(" m", ""))
                elif elev_str == "unknown":
                    elev_val = 2500.0  # Default for unknown
                else:
                    elev_val = float(elev_str)  # Try parsing as number string
            elif isinstance(elev_str, (int, float)):
                elev_val = float(elev_str)
            else:
                elev_val = 2500.0
        except (ValueError, TypeError):
            print(f"Warning: Could not parse elevation '{elev_str}', using 2500m default")
            elev_val = 2500.0
        location_elevations.append(elev_val)
    
    location_elevations = np.array(location_elevations)
    weather_grids["elevation"] = interpolate_to_grid(location_elevations, kdtree, grid_x, grid_y)
    
    # Process each requested variable
    updated_powder_state = powder_state
    
    for var_name in variables:
        if var_name not in VARIABLES:
            print(f"Unknown variable: {var_name}")
            continue

        var = VARIABLES[var_name]
        
        # Check if regeneration needed
        if not should_regenerate(var, timestamp, output_dir):
            print(f"  Skipping {var_name} (unchanged)")
            continue
        
        print(f"  Generating {var_name}")
        
        # Prepare dependencies
        deps = {dep: weather_grids[dep] for dep in var.weather_deps if dep in weather_grids}
        tif_deps_data = {}
        for dep in var.tif_deps:
            if dep in ("powder_depth", "powder_quality") and updated_powder_state is not None:
                depth, quality = updated_powder_state
                if dep == "powder_depth":
                    tif_deps_data[dep] = depth
                else:
                    tif_deps_data[dep] = quality
            else:
                loaded = load_powder_state_for_timestamp(timestamp, OUTPUT_BASE)
                if loaded is not None:
                    d, q = loaded
                    if dep == "powder_depth":
                        tif_deps_data[dep] = d
                    elif dep == "powder_quality":
                        tif_deps_data[dep] = q
                else:
                    print(f"    Warning: Missing dependency {dep} for {var.key}")
                    tif_deps_data[dep] = np.zeros_like(elevation)
        
        # Handle special cases
        if var.key == "powder":
            # Load previous state if needed
            if var.depends_on_previous:
                prev_state = load_previous_powder_state(timestamp_idx, all_timestamps, OUTPUT_BASE)
            else:
                prev_state = None
            
            # Compute powder
            result = var.physics(deps, elevation, hillshade, prev_state)
            updated_powder_state = result
            
            # Write both outputs
            write_geotiff(result[0], output_dir / "powder_depth.tif", metadata, "cm")
            write_geotiff(result[1], output_dir / "powder_quality.tif", metadata, "quality")
            
        elif var.key == "skiability":
            if updated_powder_state is None:
                loaded = load_powder_state_for_timestamp(timestamp, OUTPUT_BASE)
                if loaded is None:
                    print(f"    Warning: No powder state for skiability, skipping")
                    continue
                powder_state_for_use = loaded
            else:
                powder_state_for_use = updated_powder_state

            result = var.physics({**deps, **tif_deps_data}, elevation, hillshade, powder_state_for_use)
            write_geotiff(result, output_dir / f"{var.key}.tif", metadata, var.unit)

        elif var.key == "sqh":
            if updated_powder_state is None:
                powder_state_for_use = load_powder_state_for_timestamp(timestamp, OUTPUT_BASE)
                if powder_state_for_use is None:
                    print(f"    Warning: No powder state for SQH, skipping")
                    continue
            else:
                powder_state_for_use = updated_powder_state

            result = var.physics(powder_state_for_use)
            write_geotiff(result, output_dir / f"{var.key}.tif", metadata, var.unit)

        else:
            result = var.physics({**deps, **tif_deps_data}, elevation, hillshade)
            write_geotiff(result, output_dir / f"{var.key}.tif", metadata, var.unit)
        
        # Save fingerprint
        save_fingerprint(var, output_dir)
    
    return updated_powder_state


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate PowFinder TIF layers")
    parser.add_argument("variables", nargs="*", help="Variables to generate (default: all)")
    args = parser.parse_args()
    
    # Determine which variables to process
    if args.variables:
        variables = args.variables
    else:
        variables = list(VARIABLES.keys())

    variables = topo_sort(variables)
    print(f"Generating TIFs for: {', '.join(variables)}")
    
    # Load data
    weather_data = load_weather_data()
    elevation, metadata = load_elevation()
    
    # Get all timestamps from weather data
    all_timestamps = get_all_timestamps_from_weather(weather_data)
    print(f"Found {len(all_timestamps)} timestamps in weather data")
    
    # Build spatial interpolation tree
    kdtree, points = build_kdtree(weather_data)
    
    # Create coordinate grids
    height, width = elevation.shape
    transform = metadata["transform"]
    
    # Generate coordinate arrays
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    xs, ys = xy(transform, rows, cols)
    grid_x = np.array(xs)
    grid_y = np.array(ys)
    
    # Process all timestamps
    powder_state = None
    
    for idx, timestamp in enumerate(all_timestamps):
        print(f"\nProcessing {timestamp} ({idx + 1}/{len(all_timestamps)})")
        
        # Process this timestamp and get updated powder state
        powder_state = process_timestamp(
            timestamp, idx, all_timestamps,
            weather_data, elevation, metadata,
            kdtree, grid_x, grid_y,
            variables, powder_state
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()