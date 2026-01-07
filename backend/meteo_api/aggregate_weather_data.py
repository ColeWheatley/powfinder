#!/usr/bin/env python3
"""
aggregate_weather_data_numpy.py
================================
OPTIMIZED VERSION: NumPy + Multiprocessing (10 cores)

Converts hourly weather data to 3-hour periods using:
- NumPy vectorization (no Python loops!)
- Multiprocessing Pool (parallelize across 10 cores)
- Structured arrays for memory efficiency

Performance target: 5-15x faster than original
"""

import json
import numpy as np
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path
import math
import sys
from functools import partial

# =====================================
# CONFIGURATION
# =====================================

INPUT_FILE = "weather_data_collection.json"
OUTPUT_FILE = "weather_data_3hour_numpy.json"
NUM_CORES = 10  # Use all cores!

AGGREGATION_METHODS = {
    "temperature_2m": "average",
    "relative_humidity_2m": "average",
    "shortwave_radiation": "average",
    "cloud_cover": "average",
    "cloud_cover_low": "average",
    "cloud_cover_mid": "average",
    "cloud_cover_high": "average",
    "snow_depth": "average",
    "snowfall": "sum",
    "wind_speed_10m": "average",
    "wind_direction_10m": "circular_mean",
    "freezing_level_height": "average",
    "surface_pressure": "average"
}

# =====================================
# UTILITY FUNCTIONS
# =====================================

def parse_iso_datetime(time_str):
    """Parse ISO datetime."""
    if time_str.endswith('Z'):
        time_str = time_str[:-1] + '+00:00'
    import re
    time_str = re.sub(r'(\+\d{2}:\d{2})\+\d{2}:\d{2}$', r'\1', time_str)
    return datetime.fromisoformat(time_str)

def create_3hour_timestamps(hourly_times):
    """Create 3-hour period timestamps."""
    if not hourly_times:
        return []

    start_time = parse_iso_datetime(hourly_times[0])
    start_hour = (start_time.hour // 3) * 3
    period_start = start_time.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    end_time = parse_iso_datetime(hourly_times[-1])

    periods = []
    current = period_start
    while current <= end_time:
        median_time = current + timedelta(hours=1, minutes=30)
        periods.append(median_time.isoformat().replace('+00:00', 'Z'))
        current += timedelta(hours=3)

    return periods

def aggregate_values_numpy(values, method):
    """Vectorized aggregation using NumPy."""
    values = np.asarray(values)
    valid_values = values[~np.isnan(values)]

    if len(valid_values) == 0:
        return None

    if method == "average":
        return float(np.nanmean(values))
    elif method == "sum":
        return float(np.nansum(values))
    elif method == "circular_mean":
        # Circular mean for angles
        sin_sum = np.sum(np.sin(np.radians(valid_values)))
        cos_sum = np.sum(np.cos(np.radians(valid_values)))
        mean_rad = np.arctan2(sin_sum, cos_sum)
        mean_deg = np.degrees(mean_rad)
        return float((mean_deg + 360) % 360)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def process_single_coordinate(coord_entry, period_times):
    """
    Process a SINGLE coordinate - this gets parallelized across cores.

    Args:
        coord_entry: One coordinate entry from weather_data_collection
        period_times: Pre-computed 3-hour period timestamps

    Returns:
        Processed coordinate entry with aggregated weather data
    """
    if coord_entry['status'] != 'collected':
        return {
            'coordinate_info': coord_entry['coordinate_info'],
            'status': coord_entry['status'],
            'weather_data_3hour': None
        }

    try:
        hourly_data = coord_entry.get('weather_data', {}).get('hourly', {})
        if not hourly_data or 'time' not in hourly_data:
            raise ValueError("No hourly data")

        hourly_times = hourly_data['time']

        # Convert hourly times to datetime objects for fast comparison
        hourly_datetimes = np.array([parse_iso_datetime(t) for t in hourly_times], dtype=object)
        period_datetimes = np.array([parse_iso_datetime(t) for t in period_times], dtype=object)

        # Initialize aggregated data
        aggregated = {
            'time': period_times,
            'time_units': '3-hour periods'
        }

        # Process each weather parameter
        for param, method in AGGREGATION_METHODS.items():
            if param not in hourly_data:
                continue

            hourly_values = np.asarray(hourly_data[param], dtype=float)
            period_values = []

            # Vectorized: for each period, find which hours fall within it
            for i, period_time in enumerate(period_datetimes):
                period_start = period_time
                period_end = period_time + timedelta(hours=3)

                # Boolean mask: hourly times within this period
                mask = (hourly_datetimes >= period_start) & (hourly_datetimes < period_end)
                period_hourly_values = hourly_values[mask]

                # Aggregate
                agg_value = aggregate_values_numpy(period_hourly_values, method)
                period_values.append(agg_value)

            aggregated[param] = period_values

        return {
            'coordinate_info': coord_entry['coordinate_info'],
            'status': coord_entry['status'],
            'weather_data_3hour': {'hourly': aggregated}
        }

    except Exception as e:
        return {
            'coordinate_info': coord_entry['coordinate_info'],
            'status': coord_entry['status'],
            'weather_data_3hour': None,
            'aggregation_error': str(e)
        }

# =====================================
# MAIN PIPELINE
# =====================================

def main():
    start_time = datetime.now()
    print("🚀 NUMPY + MULTIPROCESSING AGGREGATION (10 CORES)")
    print("=" * 70)

    # Load weather data
    print("📊 Loading weather data...")
    with open(INPUT_FILE, 'r') as f:
        weather_data = json.load(f)

    metadata = weather_data['metadata']
    coord_entries = weather_data['coordinates']

    print(f"   ✅ Loaded {len(coord_entries)} coordinates")

    # Pre-compute period times once
    sample_coord = next((c for c in coord_entries if c['status'] == 'collected'), None)
    if not sample_coord:
        print("❌ No collected coordinates found!")
        sys.exit(1)

    hourly_times = sample_coord['weather_data']['hourly']['time']
    period_times = create_3hour_timestamps(hourly_times)
    print(f"   ✅ Pre-computed {len(period_times)} periods per coordinate")

    # Process in parallel
    print(f"\n⚡ Processing with {NUM_CORES} cores...")

    with mp.Pool(NUM_CORES) as pool:
        # Create partial function with period_times fixed
        process_func = partial(process_single_coordinate, period_times=period_times)

        # Map across all coordinates
        processed = list(pool.imap(process_func, coord_entries, chunksize=32))

    # Count results
    success_count = sum(1 for p in processed if p.get('weather_data_3hour') is not None)
    error_count = len(processed) - success_count

    print(f"   ✅ Processed: {success_count} successful, {error_count} errors")

    # Build output dataset
    aggregated_dataset = {
        'metadata': {
            **metadata,
            'aggregation_info': {
                'source_file': INPUT_FILE,
                'aggregation_date': datetime.now().isoformat(),
                'aggregation_methods': AGGREGATION_METHODS,
                'period_length': '3 hours',
                'successful_aggregations': success_count,
                'failed_aggregations': error_count,
                'implementation': 'NumPy + Multiprocessing (10 cores)'
            }
        },
        'coordinates': processed
    }

    # Save output
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(aggregated_dataset, f, indent=2)

    file_size_mb = Path(OUTPUT_FILE).stat().st_size / (1024 * 1024)
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"   ✅ Saved {file_size_mb:.1f}MB")

    # Summary
    print("\n" + "=" * 70)
    print(f"⚡ TOTAL TIME: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"📊 Throughput: {len(processed) / elapsed:.1f} coordinates/sec")
    print(f"🎯 Per-coordinate: {(elapsed / len(processed)) * 1000:.2f}ms")
    print("=" * 70)
    print(f"✅ Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # macOS compatibility
    main()
