#!/usr/bin/env python3
"""
analyze_tif_ranges.py
--------------------
Utility script to analyze TIF files and find their actual min/max values
in physical units (°C, %, hPa, etc.) for updating color_scales.json ranges.
Converts scaled 0-255 TIF values back to real physical values using color scales.

Usage:
    python analyze_tif_ranges.py [variable_name]
    
    If variable_name is provided, analyzes only that variable.
    If no variable is provided, analyzes all variables found.

Examples:
    python analyze_tif_ranges.py dewpoint_2m
    python analyze_tif_ranges.py  # analyzes all variables
"""

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import rasterio


def load_color_scales():
    """Load color scales configuration."""
    script_dir = pathlib.Path(__file__).parent
    color_scales_path = script_dir / "color_scales.json"
    
    if not color_scales_path.exists():
        print(f"Warning: {color_scales_path} not found. Physical unit conversion unavailable.")
        return {}
        
    with open(color_scales_path, 'r') as f:
        return json.load(f)


def get_variable_units(variable_name):
    """Get the appropriate units for a variable."""
    units_map = {
        'temperature_2m': '°C',
        'dewpoint_2m': '°C',
        'relative_humidity_2m': '%',
        'surface_pressure': 'hPa',
        'cloud_cover': '%',
        'shortwave_radiation': 'W/m²',
        'snowfall': 'mm',
        'freezing_level_height': 'm',
        'wind_speed_10m': 'm/s',
        'weather_code': ''
    }
    return units_map.get(variable_name, '')


def convert_to_physical_values(scaled_values, variable_name, color_scales):
    """
    Convert 0-255 scaled values back to physical units using color scale ranges.
    
    Args:
        scaled_values: numpy array of 0-255 values from TIF
        variable_name: name of the variable
        color_scales: loaded color scales configuration
        
    Returns:
        numpy array of physical values, or original values if no conversion available
    """
    if variable_name not in color_scales:
        return scaled_values
        
    spec = color_scales[variable_name]
    if 'min' not in spec or 'max' not in spec:
        return scaled_values
        
    # Convert 0-255 back to physical range
    # scaled = (physical - min) / (max - min) * 255
    # physical = (scaled / 255) * (max - min) + min
    
    phys_min = spec['min']
    phys_max = spec['max']
    
    # Normalize 0-255 to 0-1, then scale to physical range
    normalized = scaled_values / 255.0
    physical = normalized * (phys_max - phys_min) + phys_min
    
    return physical


def analyze_variable_tifs(variable_name, base_dir="TIFS/100m_resolution", color_scales=None):
    """
    Analyze all TIF files for a specific variable across all timestamps.
    
    Args:
        variable_name: Name of the variable (e.g., 'dewpoint_2m', 'temperature_2m')
        base_dir: Base directory containing timestamp subdirectories
        color_scales: Color scales configuration for unit conversion
        
    Returns:
        dict: Statistics including global_min, global_max, timestamp_count, valid_pixel_count
    """
    base_path = pathlib.Path(base_dir)
    
    global_min = float('inf')
    global_max = float('-inf')
    timestamp_count = 0
    total_valid_pixels = 0
    units = get_variable_units(variable_name)
    
    print(f"\nAnalyzing {variable_name} ({units}):")
    print("-" * 50)
    
    for timestamp_dir in sorted(base_path.iterdir()):
        if not timestamp_dir.is_dir():
            continue
            
        tif_path = timestamp_dir / f"{variable_name}.tif"
        if not tif_path.exists():
            continue
            
        timestamp_count += 1
        
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            nodata = src.nodata
            
            # Create mask for valid data (not nodata, not NaN, not -9999)
            if nodata is not None:
                valid_mask = (data != nodata) & ~np.isnan(data) & (data != -9999)
            else:
                valid_mask = ~np.isnan(data) & (data != -9999)
            
            if not np.any(valid_mask):
                print(f"  {timestamp_dir.name}: No valid data")
                continue
                
            valid_data = data[valid_mask]
            
            # Convert to physical values if possible
            if color_scales:
                physical_data = convert_to_physical_values(valid_data, variable_name, color_scales)
            else:
                physical_data = valid_data
                
            timestamp_min = float(np.min(physical_data))
            timestamp_max = float(np.max(physical_data))
            valid_pixels = np.sum(valid_mask)
            
            print(f"  {timestamp_dir.name}: min={timestamp_min:.2f}{units}, max={timestamp_max:.2f}{units}, pixels={valid_pixels}")
            
            global_min = min(global_min, timestamp_min)
            global_max = max(global_max, timestamp_max)
            total_valid_pixels += valid_pixels
    
    if timestamp_count == 0:
        print(f"  No TIF files found for {variable_name}")
        return None
        
    stats = {
        'variable': variable_name,
        'global_min': global_min,
        'global_max': global_max,
        'timestamp_count': timestamp_count,
        'total_valid_pixels': total_valid_pixels,
        'units': units
    }
    
    print(f"\nSUMMARY for {variable_name}:")
    print(f"  Global Min: {global_min:.2f}{units}")
    print(f"  Global Max: {global_max:.2f}{units}")
    print(f"  Timestamps: {timestamp_count}")
    print(f"  Total Valid Pixels: {total_valid_pixels:,}")
    
    return stats


def find_all_variables(base_dir="TIFS/100m_resolution"):
    """Find all unique variable names in the TIF directories."""
    base_path = pathlib.Path(base_dir)
    variables = set()
    
    for timestamp_dir in base_path.iterdir():
        if not timestamp_dir.is_dir():
            continue
        for tif_file in timestamp_dir.glob("*.tif"):
            variables.add(tif_file.stem)
    
    return sorted(variables)


def generate_color_scale_update(stats_dict):
    """
    Generate suggested updates for color_scales.json based on analyzed ranges.
    
    Args:
        stats_dict: Dictionary mapping variable names to their statistics
    """
    print("\n" + "="*70)
    print("SUGGESTED COLOR_SCALES.JSON UPDATES (Physical Units):")
    print("="*70)
    
    for variable, stats in stats_dict.items():
        if stats is None:
            continue
            
        # Use exact range for precise color mapping
        units = stats.get('units', '')
        
        print(f"\n\"{variable}\": {{")
        print(f"  \"min\": {stats['global_min']:.2f},")
        print(f"  \"max\": {stats['global_max']:.2f}")
        print(f"  // Actual range: {stats['global_min']:.2f}{units} to {stats['global_max']:.2f}{units}")
        print(f"  // Based on {stats['timestamp_count']} timestamps, {stats['total_valid_pixels']:,} pixels")
        print(f"}},")


def main():
    # Load color scales for unit conversion
    color_scales = load_color_scales()
    
    if len(sys.argv) > 1:
        # Analyze specific variable
        variable_name = sys.argv[1]
        stats = analyze_variable_tifs(variable_name, color_scales=color_scales)
        if stats:
            generate_color_scale_update({variable_name: stats})
    else:
        # Analyze all variables
        print("Finding all variables...")
        variables = find_all_variables()
        print(f"Found variables: {', '.join(variables)}")
        
        all_stats = {}
        for variable in variables:
            stats = analyze_variable_tifs(variable, color_scales=color_scales)
            all_stats[variable] = stats
        
        generate_color_scale_update(all_stats)


if __name__ == "__main__":
    main()
