#!/usr/bin/env python3
"""
Weather Data Aggregation Script
==============================

Converts hourly weather data into 3-hour periods for terrain analysis integration.

USAGE:
    python aggregate_weather_data.py

INPUT:
    weather_data_collection.json (hourly weather data from collect_weather_data.py)

OUTPUT:
    weather_data_3hour.json (3-hour aggregated weather data)

Aggregation Methods:
- Temperature/Snow Depth/Humidity/Pressure: Average over 3-hour period
- Snowfall: Sum (total accumulation over 3 hours)
- Weather Code: Median (most representative condition)
- Wind Speed/Cloud Cover/Radiation: Average over 3-hour period
- Freezing Level: Average over 3-hour period

Time Periods:
The 24-hour day is divided into 8 periods of 3 hours each, using median times:
- 01:30 (00:00-03:00), 04:30 (03:00-06:00), 07:30 (06:00-09:00), 10:30 (09:00-12:00)
- 13:30 (12:00-15:00), 16:30 (15:00-18:00), 19:30 (18:00-21:00), 22:30 (21:00-24:00)

Using median times provides the most representative timestamp for shadows and other calculations.

EXAMPLE:
    # First collect hourly weather data
    python collect_weather_data.py
    
    # Then aggregate to 3-hour periods
    python aggregate_weather_data.py
    
    # Result: weather_data_3hour.json with terrain-analysis-ready data
"""

import json
import statistics
from datetime import datetime, timedelta
import sys
from pathlib import Path
import re

# =====================================
# CONFIGURATION
# =====================================

INPUT_FILE = "weather_data_collection.json"
OUTPUT_FILE = "weather_data_3hour.json"

# Aggregation method for each parameter
AGGREGATION_METHODS = {
    "temperature_2m": "average",
    "relative_humidity_2m": "average", 
    "shortwave_radiation": "average",
    "cloud_cover": "average",
    "snow_depth": "average",
    "snowfall": "sum",  # Accumulation over 3 hours
    "wind_speed_10m": "average",
    "weather_code": "median",  # Most representative condition
    "freezing_level_height": "average",
    "surface_pressure": "average"
}

def load_weather_data():
    """Load the hourly weather data collection."""
    print(f"📊 Loading weather data from {INPUT_FILE}...")
    
    if not Path(INPUT_FILE).exists():
        print(f"❌ Input file {INPUT_FILE} not found!")
        return None
    
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
        
        collected_count = data['metadata']['collected_count']
        total_count = data['metadata']['total_coordinates']
        print(f"   ✅ Loaded {collected_count}/{total_count} coordinate weather datasets")
        
        return data
    
    except Exception as e:
        print(f"   ❌ Error loading weather data: {e}")
        return None

def parse_iso_datetime(time_str):
    """Safely parse ISO datetime string with various timezone formats."""
    # Remove 'Z' and replace with UTC offset
    if time_str.endswith('Z'):
        time_str = time_str[:-1] + '+00:00'
    
    # Handle case where timezone offset might be duplicated
    # Remove any double timezone offsets like '+00:00+00:00'
    time_str = re.sub(r'(\+\d{2}:\d{2})\+\d{2}:\d{2}$', r'\1', time_str)
    
    return datetime.fromisoformat(time_str)

def aggregate_values(values, method):
    """Aggregate a list of values using the specified method."""
    # Filter out None values
    valid_values = [v for v in values if v is not None]
    
    if not valid_values:
        return None
    
    if method == "average":
        return sum(valid_values) / len(valid_values)
    elif method == "sum":
        return sum(valid_values)
    elif method == "median":
        return statistics.median(valid_values)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def create_3hour_timestamps(hourly_times):
    """Create 3-hour period timestamps from hourly timestamps using median times."""
    if not hourly_times:
        return []
    
    # Parse first timestamp to get starting point
    start_time = parse_iso_datetime(hourly_times[0])
    
    # Round down to nearest 3-hour boundary (00:00, 03:00, 06:00, etc.)
    start_hour = (start_time.hour // 3) * 3
    period_start = start_time.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    
    # Parse last timestamp to get ending point
    end_time = parse_iso_datetime(hourly_times[-1])
    
    # Generate 3-hour periods using median times (start + 1.5 hours)
    periods = []
    current = period_start
    while current <= end_time:
        # Use median time of the 3-hour period (start + 1.5 hours)
        median_time = current + timedelta(hours=1, minutes=30)
        periods.append(median_time.isoformat().replace('+00:00', 'Z'))
        current += timedelta(hours=3)
    
    return periods

def aggregate_hourly_to_3hour(hourly_data):
    """Convert hourly weather data to 3-hour aggregated data."""
    if not hourly_data or 'hourly' not in hourly_data:
        return None
    
    hourly = hourly_data['hourly']
    hourly_times = hourly.get('time', [])
    
    if not hourly_times:
        return None
    
    # Create 3-hour period timestamps
    period_times = create_3hour_timestamps(hourly_times)
    
    # Initialize aggregated data structure
    aggregated = {
        'time': period_times,
        'time_units': '3-hour periods'
    }
    
    # Process each weather parameter
    for param, method in AGGREGATION_METHODS.items():
        if param not in hourly:
            continue
        
        hourly_values = hourly[param]
        period_values = []
        
        # Group hourly values into 3-hour periods
        for i, period_time in enumerate(period_times):
            period_start = parse_iso_datetime(period_time)
            period_end = period_start + timedelta(hours=3)
            
            # Find hourly values within this 3-hour period
            period_hourly_values = []
            
            for j, hourly_time in enumerate(hourly_times):
                hour_dt = parse_iso_datetime(hourly_time)
                
                # Include hourly values that fall within this 3-hour period
                if period_start <= hour_dt < period_end:
                    if j < len(hourly_values) and hourly_values[j] is not None:
                        period_hourly_values.append(hourly_values[j])
            
            # Aggregate the values for this period
            aggregated_value = aggregate_values(period_hourly_values, method)
            period_values.append(aggregated_value)
        
        aggregated[param] = period_values
    
    return {'hourly': aggregated}

def process_all_coordinates(weather_data):
    """Process all coordinates and aggregate their weather data."""
    print(f"🔄 Aggregating weather data to 3-hour periods...")
    
    processed_coordinates = []
    success_count = 0
    error_count = 0
    
    for coord_entry in weather_data['coordinates']:
        if coord_entry['status'] != 'collected':
            # Skip coordinates that don't have weather data (no original weather data in output)
            processed_coordinates.append({
                'coordinate_info': coord_entry['coordinate_info'],
                'status': coord_entry['status'],
                'weather_data_3hour': None
            })
            continue
        
        try:
            # Aggregate the hourly weather data
            aggregated_data = aggregate_hourly_to_3hour(coord_entry['weather_data'])
            
            # Create new coordinate entry with aggregated data (remove original hourly data)
            new_coord_entry = {
                'coordinate_info': coord_entry['coordinate_info'],
                'status': coord_entry['status'],
                'weather_data_3hour': aggregated_data
            }
            
            processed_coordinates.append(new_coord_entry)
            success_count += 1
            
            if success_count % 100 == 0:
                print(f"   📈 Processed {success_count} coordinates...")
        
        except Exception as e:
            print(f"   ⚠️  Error processing coordinate {coord_entry['coordinate_info']['id']}: {e}")
            
            # Keep essential info but mark aggregation as failed (no original weather data)
            processed_coordinates.append({
                'coordinate_info': coord_entry['coordinate_info'],
                'status': coord_entry['status'],
                'weather_data_3hour': None,
                'aggregation_error': str(e)
            })
            error_count += 1
    
    print(f"   ✅ Aggregation complete: {success_count} successful, {error_count} errors")
    
    # Create new dataset structure
    aggregated_dataset = {
        'metadata': {
            **weather_data['metadata'],
            'aggregation_info': {
                'source_file': INPUT_FILE,
                'aggregation_date': datetime.now().isoformat(),
                'aggregation_methods': AGGREGATION_METHODS,
                'period_length': '3 hours',
                'successful_aggregations': success_count,
                'failed_aggregations': error_count
            }
        },
        'coordinates': processed_coordinates
    }
    
    return aggregated_dataset

def save_aggregated_data(aggregated_dataset):
    """Save the aggregated dataset to file."""
    print(f"💾 Saving aggregated data to {OUTPUT_FILE}...")
    
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(aggregated_dataset, f, indent=2)
        
        # Calculate file size
        file_size = Path(OUTPUT_FILE).stat().st_size / (1024 * 1024)  # MB
        print(f"   ✅ Saved {file_size:.1f}MB to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"   ❌ Error saving aggregated data: {e}")
        return False
    
    return True

def print_aggregation_summary(aggregated_dataset):
    """Print summary of aggregation results."""
    print("\n📊 AGGREGATION SUMMARY")
    print("=" * 40)
    
    metadata = aggregated_dataset['metadata']
    agg_info = metadata['aggregation_info']
    
    print(f"Source File: {agg_info['source_file']}")
    print(f"Period Length: {agg_info['period_length']}")
    print(f"Total Coordinates: {metadata['total_coordinates']}")
    print(f"Successful Aggregations: {agg_info['successful_aggregations']}")
    print(f"Failed Aggregations: {agg_info['failed_aggregations']}")
    
    print(f"\nAggregation Methods:")
    for param, method in agg_info['aggregation_methods'].items():
        print(f"  • {param}: {method}")
    
    # Sample coordinate to show period count
    sample_coord = None
    for coord in aggregated_dataset['coordinates']:
        if coord.get('weather_data_3hour') and coord['weather_data_3hour']:
            sample_coord = coord
            break
    
    if sample_coord:
        period_count = len(sample_coord['weather_data_3hour']['hourly']['time'])
        print(f"\nTime Periods per Coordinate: {period_count}")
        # Get date range from actual time data
        time_data = sample_coord['weather_data_3hour']['hourly']['time']
        if time_data:
            print(f"Data Coverage: {time_data[0]} to {time_data[-1]}")

def main():
    """Main aggregation function."""
    print("📈 PowFinder Weather Data Aggregation")
    print("=" * 50)
    
    # Load hourly weather data
    weather_data = load_weather_data()
    if not weather_data:
        sys.exit(1)
    
    # Process and aggregate all coordinates
    aggregated_dataset = process_all_coordinates(weather_data)
    
    # Save aggregated data
    if not save_aggregated_data(aggregated_dataset):
        sys.exit(1)
    
    # Print summary
    print_aggregation_summary(aggregated_dataset)
    
    print(f"\n✅ Weather data aggregation complete!")
    print(f"Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
