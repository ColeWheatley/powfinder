#!/usr/bin/env python3
"""
MOCK physics_extrapolate.py for testing process_task_queue.py
"""
import sys
import json
import datetime

WEATHER_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "shortwave_radiation",
    "cloud_cover", "snowfall", "snow_depth", "wind_speed_10m",
    "weather_code", "freezing_level_height", "surface_pressure"
]

def main():
    if sys.stdin.isatty():
        if len(sys.argv) != 4:
            print(f"Usage: {sys.argv[0]} <lat> <lon> <ISO_timestamp>", file=sys.stderr)
            print("Mock physics_extrapolate.py: This script is intended to be called with JSON input via stdin for process_task_queue.py, or with CLI args.", file=sys.stderr)
            # Provide a minimal valid output if called directly for some reason / for basic testing
            lat, lon, time_str = 47.0, 11.0, "2025-05-23T12:00:00"
        else:
            try:
                lat = float(sys.argv[1])
                lon = float(sys.argv[2])
                time_str = sys.argv[3]
            except ValueError as e:
                print(f"Mock physics_extrapolate.py: Error parsing command line arguments: {e}", file=sys.stderr)
                sys.exit(1)
    else:
        try:
            input_data = json.loads(sys.stdin.buffer.read().decode('utf-8'))
            lat = float(input_data['lat'])
            lon = float(input_data['lon'])
            time_str = input_data['time']
        except Exception as e:
            print(f"Mock physics_extrapolate.py: Error reading from stdin: {e}", file=sys.stderr)
            sys.exit(1)

    # Ensure target_time (dt_obj) is a datetime.datetime object
    try:
        # Handle ISO 8601 timestamps, specifically the 'Z' suffix for UTC
        if time_str.endswith('Z'):
            # Replace 'Z' with '+00:00' for datetime.fromisoformat compatibility
            time_str_parsed = time_str[:-1] + '+00:00'
        else:
            time_str_parsed = time_str
        dt_obj = datetime.datetime.fromisoformat(time_str_parsed)
        hour_component = dt_obj.hour
    except ValueError as e:
        print(f"Mock physics_extrapolate.py: Error parsing time string '{time_str}': {e}", file=sys.stderr)
        # Fallback to a default hour if parsing fails, though ideally, input should be validated upstream
        hour_component = 12
        # Use a fixed ISO string for output if parsing failed, to maintain output structure
        time_str = "1970-01-01T12:00:00+00:00"


    variable_dict = {}
    for i, var_name in enumerate(WEATHER_VARIABLES):
        # Create some variation based on input
        variable_dict[var_name] = round(
            (lat % 10) + (lon % 10) + hour_component + (i * 0.1), 2
        )
    
    # Override some specific variables for more controlled testing if needed
    variable_dict["temperature_2m"] = round(10 + (lat % 1) + (hour_component / 6), 2) # Temp between 10-14
    variable_dict["relative_humidity_2m"] = round(60 + (lon % 1) * 10, 2) # Humidity between 60-70
    variable_dict["weather_code"] = int(lat % 5) # Integer code 0-4
    
    # Ensure the output time string is in ISO format and represents the parsed datetime
    # This is especially important if the input time_str was modified (e.g. 'Z' replacement)
    # or if parsing failed and we're using a default.
    if isinstance(dt_obj, datetime.datetime):
        output_time_str = dt_obj.isoformat()
    else: # Should not happen if logic above is correct, but as a fallback
        output_time_str = time_str


    output_json = {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "time": output_time_str,
        "variable_dict": variable_dict
    }

    json.dump(output_json, sys.stdout, separators=(",",":"))
    sys.stdout.write("\n")

if __name__ == "__main__":
    main()
