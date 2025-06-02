#!/usr/bin/env python3
"""
Script to rename 3-hourly weather data bins from start times to median times.
Converts: 12:00, 15:00, 18:00, 21:00, 00:00, 03:00, 06:00, 09:00
To:       13:30, 16:30, 19:30, 22:30, 01:30, 04:30, 07:30, 10:30
"""

import json
from datetime import datetime, timedelta
import sys

def convert_start_to_median_time(start_time_str):
    """Convert a bin start time to median time (start + 1.5 hours)."""
    try:
        # Parse the datetime
        dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        # Add 1.5 hours to get median
        median_dt = dt + timedelta(hours=1, minutes=30)
        # Convert back to string in same format
        if start_time_str.endswith('Z'):
            return median_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            return median_dt.isoformat()
    except Exception as e:
        print(f"Error converting time {start_time_str}: {e}", file=sys.stderr)
        return start_time_str

def main():
    input_file = "weather_data_3hour.json"
    output_file = "weather_data_3hour_median.json"
    
    print(f"Reading {input_file}...")
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found!", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading {input_file}: {e}", file=sys.stderr)
        return 1
    
    
    # Check data structure
    if 'coordinates' in data:
        coordinates = data['coordinates']
        print(f"Found 'coordinates' key with {len(coordinates)} entries")
    elif isinstance(data, list):
        coordinates = data
        print(f"Data is a direct list with {len(coordinates)} entries")
    else:
        # Look for entries that have 'latitude', 'longitude', and 'hourly' keys
        coordinates = []
        for key, value in data.items():
            if isinstance(value, dict) and 'latitude' in value and 'longitude' in value and 'hourly' in value:
                coordinates.append(value)
        if coordinates:
            print(f"Found {len(coordinates)} coordinate entries in data structure")
        else:
            print("Error: Could not find coordinate data in expected format", file=sys.stderr)
            return 1
    total_coords = len(coordinates)
    print(f"Processing {total_coords} coordinates...")
    
    converted_count = 0
    
    for i, coord_data in enumerate(coordinates):
        if i % 1000 == 0:
            print(f"Progress: {i}/{total_coords} coordinates processed...")
        
        if 'weather_data_3hour' in coord_data and 'hourly' in coord_data['weather_data_3hour'] and 'time' in coord_data['weather_data_3hour']['hourly']:
            old_times = coord_data['weather_data_3hour']['hourly']['time']
            new_times = []
            
            for time_str in old_times:
                new_time = convert_start_to_median_time(time_str)
                new_times.append(new_time)
            
            coord_data['weather_data_3hour']['hourly']['time'] = new_times
            converted_count += 1
    
    print(f"Converted timestamps for {converted_count} coordinates")
    print(f"Writing to {output_file}...")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, separators=(',', ':'))  # Compact format to save space
        print(f"Successfully created {output_file}")
        
        # Show example of conversion
        if converted_count > 0:
            first_coord = coordinates[0]
            if 'weather_data_3hour' in first_coord and 'hourly' in first_coord['weather_data_3hour'] and 'time' in first_coord['weather_data_3hour']['hourly']:
                sample_times = first_coord['weather_data_3hour']['hourly']['time'][:4]  # Show first 4 timestamps
                print(f"Example converted timestamps: {sample_times}")
                
    except Exception as e:
        print(f"Error writing {output_file}: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
