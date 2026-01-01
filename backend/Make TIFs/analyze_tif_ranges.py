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
        'weather_code': '',
        'skiability': '',  # dimensionless 0-1 scale
        'powder_depth': 'mm',  # powder depth in millimeters
        'sqh': '',  # dimensionless
        'snow_depth': 'm',  # snow depth in meters
        'elevation': 'm',  # elevation in meters
        'aspect': '°',  # aspect in degrees
        'slope': '°'  # slope in degrees
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


def analyze_variable_tifs(variable_name, base_dir=None, color_scales=None):
    """
    Analyze all TIF files for a specific variable across all timestamps.
    
    Args:
        variable_name: Name of the variable (e.g., 'dewpoint_2m', 'temperature_2m')
        base_dir: Base directory containing timestamp subdirectories (defaults to script-relative path)
        color_scales: Color scales configuration for unit conversion
        
    Returns:
        dict: Statistics including global_min, global_max, timestamp_count, valid_pixel_count
    """
    if base_dir is None:
        # Use script-relative path for robustness
        script_dir = pathlib.Path(__file__).parent
        base_dir = script_dir / ".." / ".." / "TIFS" / "100m_resolution"
    
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


def find_all_variables(base_dir=None):
    """Find all unique variable names in the TIF directories."""
    if base_dir is None:
        # Use script-relative path for robustness
        script_dir = pathlib.Path(__file__).parent
        base_dir = script_dir / ".." / ".." / "TIFS" / "100m_resolution"
    
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


def analyze_variable_with_histogram(variable_name, base_dir=None, color_scales=None, num_bins=10):
    """
    Analyze variable with histogram/distribution statistics.
    """
    if base_dir is None:
        # Use script-relative path for robustness
        script_dir = pathlib.Path(__file__).parent
        base_dir = script_dir / ".." / ".." / "TIFS" / "100m_resolution"
    
    base_path = pathlib.Path(base_dir)
    
    all_valid_data = []
    timestamp_count = 0
    units = get_variable_units(variable_name)
    
    print(f"\nAnalyzing {variable_name} ({units}):")
    print("-" * 50)
    
    # Collect all valid data
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
            
            # Create mask for valid data
            if nodata is not None:
                valid_mask = (data != nodata) & ~np.isnan(data) & (data != -9999) & (data > 0)
            else:
                valid_mask = ~np.isnan(data) & (data != -9999) & (data > 0)
            
            if np.any(valid_mask):
                valid_data = data[valid_mask]
                
                # Convert to physical values if possible
                if color_scales and variable_name in color_scales:
                    physical_data = convert_to_physical_values(valid_data, variable_name, color_scales)
                else:
                    # For variables without color scales, normalize 0-255 to 0-1
                    physical_data = valid_data / 255.0
                    
                all_valid_data.extend(physical_data)
    
    if not all_valid_data:
        print(f"  No valid data found for {variable_name}")
        return None
        
    # Convert to numpy array for statistics
    all_data = np.array(all_valid_data)
    
    # Calculate statistics
    stats = {
        'variable': variable_name,
        'min': float(np.min(all_data)),
        'max': float(np.max(all_data)),
        'mean': float(np.mean(all_data)),
        'median': float(np.median(all_data)),
        'std': float(np.std(all_data)),
        'percentiles': {
            '1%': float(np.percentile(all_data, 1)),
            '5%': float(np.percentile(all_data, 5)),
            '25%': float(np.percentile(all_data, 25)),
            '50%': float(np.percentile(all_data, 50)),
            '75%': float(np.percentile(all_data, 75)),
            '95%': float(np.percentile(all_data, 95)),
            '99%': float(np.percentile(all_data, 99))
        },
        'timestamp_count': timestamp_count,
        'total_pixels': len(all_data),
        'units': units
    }
    
    # Create histogram
    hist, bin_edges = np.histogram(all_data, bins=num_bins)
    
    print(f"\nDistribution for {variable_name}:")
    print(f"  Min: {stats['min']:.3f}{units}")
    print(f"  Max: {stats['max']:.3f}{units}")
    print(f"  Mean: {stats['mean']:.3f}{units}")
    print(f"  Median: {stats['median']:.3f}{units}")
    print(f"  Std Dev: {stats['std']:.3f}{units}")
    print(f"\nPercentiles:")
    for pct, val in stats['percentiles'].items():
        print(f"  {pct}: {val:.3f}{units}")
    
    print(f"\nHistogram ({num_bins} bins):")
    for i, count in enumerate(hist):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        pct = (count / len(all_data)) * 100
        bar = '█' * int(pct / 2)  # Scale bar to fit
        print(f"  [{bin_start:.3f}-{bin_end:.3f}): {count:8d} ({pct:5.1f}%) {bar}")
    
    # Find peak bin
    peak_bin_idx = np.argmax(hist)
    peak_start = bin_edges[peak_bin_idx]
    peak_end = bin_edges[peak_bin_idx + 1]
    print(f"\n  Peak bin: [{peak_start:.3f}-{peak_end:.3f}) with {hist[peak_bin_idx]} pixels ({(hist[peak_bin_idx]/len(all_data)*100):.1f}%)")
    
    return stats


def main():
    # Load color scales for unit conversion
    color_scales = load_color_scales()
    
    if len(sys.argv) > 1:
        # Analyze specific variable with histogram
        variable_name = sys.argv[1]
        
        # Use histogram analysis for skiability and other key variables
        if variable_name in ['skiability', 'powder_depth', 'powder_quality', 'sqh']:
            stats = analyze_variable_with_histogram(variable_name, color_scales=color_scales, num_bins=20)
        else:
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
