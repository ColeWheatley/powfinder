import json
import subprocess
import sys
import csv
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

# --- Configuration ---
TARGET_TIMESTAMPS = [
    "2025-05-23T07:30:00",
    "2025-05-23T10:30:00",
    "2025-05-23T13:30:00",
    "2025-05-23T16:30:00"
]
WEATHER_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "shortwave_radiation",
    "cloud_cover", "snow_depth", "snowfall", "wind_speed_10m",
    "weather_code", "freezing_level_height", "surface_pressure"
]

# --- Paths ---
# Assuming this script is in resources/pipeline/
THIS_DIR = Path(__file__).parent.resolve()
TASK_QUEUE_PATH = THIS_DIR / "task_queue.json"
PREDICTIONS_CSV_PATH = THIS_DIR / "predictions.csv"
PHYSICS_EXTRAPOLATE_SCRIPT_PATH = THIS_DIR / "physics_extrapolate.py"
RESUME_LOG_FILE = THIS_DIR / "processed_tasks.log"
RAW_WEATHER_DATA_FILE = THIS_DIR.parent / "meteo_api" / "weather_data_collection.json" # Adjusted path
RESIDUALS_CSV_FILE = THIS_DIR / "residuals.csv"


def load_raw_weather_data(file_path):
    """Loads raw weather data and structures it for quick lookup by (lat, lon)."""
    data_by_coord = {}
    try:
        with open(file_path, 'r') as f:
            raw_data_collection = json.load(f)
        for entry in raw_data_collection:
            # Using float for lat/lon keys for consistency, ensure they are rounded if needed
            key = (float(entry['latitude']), float(entry['longitude']))
            data_by_coord[key] = entry
        print(f"Successfully loaded raw weather data for {len(data_by_coord)} locations from {file_path}")
    except FileNotFoundError:
        print(f"Warning: Raw weather data file not found at {file_path}. Validation tasks will be skipped.", file=sys.stderr)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {file_path}. Validation tasks will be skipped.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: An error occurred loading raw weather data from {file_path}: {e}. Validation tasks will be skipped.", file=sys.stderr)
    return data_by_coord

def load_processed_log(file_path):
    """Loads the set of already processed task keys."""
    processed_keys = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                processed_keys.add(line.strip())
        print(f"Loaded {len(processed_keys)} entries from resume log {file_path}")
    except FileNotFoundError:
        print(f"Resume log {file_path} not found. Starting fresh.")
    return processed_keys

def aggregate_hourly_to_3hourly(raw_hourly_data_for_coord, target_dt_object, weather_variables_list):
    """
    Aggregates hourly data to 3-hourly averages around the target_dt_object.
    target_dt_object should be timezone-aware (UTC).
    """
    aggregated_values = {}
    if not raw_hourly_data_for_coord or 'hourly' not in raw_hourly_data_for_coord or 'time' not in raw_hourly_data_for_coord['hourly']:
        print(f"Warning: Insufficient raw hourly data for coord to aggregate for {target_dt_object.isoformat()}.", file=sys.stderr)
        return None

    hourly_times_str = raw_hourly_data_for_coord['hourly']['time']

    # Ensure raw hourly times are parsed as UTC datetime objects
    hourly_datetimes = []
    for t_str in hourly_times_str:
        try:
            # Assume raw times are ISO format and UTC. Add 'Z' if naive.
            if 'Z' not in t_str and '+' not in t_str:
                 dt = datetime.fromisoformat(t_str + 'Z') # Assume UTC if naive
            else:
                 dt = datetime.fromisoformat(t_str)
            hourly_datetimes.append(dt.astimezone(timezone.utc) if dt.tzinfo is None else dt)
        except ValueError:
            print(f"Warning: Could not parse time '{t_str}' from raw_hourly_data. Skipping this time point.", file=sys.stderr)
            continue

    timestamps_to_average = [
        target_dt_object - timedelta(hours=1),
        target_dt_object,
        target_dt_object + timedelta(hours=1)
    ]

    for var_name in weather_variables_list:
        if var_name not in raw_hourly_data_for_coord['hourly']:
            aggregated_values[var_name] = None # Variable not available in raw data
            print(f"Warning: Variable '{var_name}' not found in raw hourly data for aggregation.", file=sys.stderr)
            continue

        raw_var_values = raw_hourly_data_for_coord['hourly'][var_name]
        values_for_var = []

        for required_ts in timestamps_to_average:
            try:
                idx = hourly_datetimes.index(required_ts)
                if raw_var_values[idx] is not None: # Check for null/None values
                    values_for_var.append(float(raw_var_values[idx]))
            except ValueError: # Timestamp not found in hourly_datetimes
                pass # This hour is missing, will average available ones
            except (TypeError, IndexError): # Value is None or index out of bounds
                print(f"Warning: Problem with value for {var_name} at {required_ts} in raw data.", file=sys.stderr)
                pass


        if values_for_var:
            aggregated_values[var_name] = sum(values_for_var) / len(values_for_var)
        else:
            aggregated_values[var_name] = None # No valid data points found for this variable

    return aggregated_values


def main():
    print("Starting task processing...")

    raw_weather_data_map = load_raw_weather_data(RAW_WEATHER_DATA_FILE)
    processed_task_keys = load_processed_log(RESUME_LOG_FILE)

    # 1. Load task queue
    try:
        with open(TASK_QUEUE_PATH, 'r') as f:
            task_queue = json.load(f)
    except FileNotFoundError:
        print(f"Error: Task queue file not found at {TASK_QUEUE_PATH}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {TASK_QUEUE_PATH}", file=sys.stderr)
        sys.exit(1)

    if not task_queue:
        print("No tasks found in the queue. Exiting.")
        sys.exit(0)

    # 2. All tasks will be processed
    tasks_to_process = task_queue # Process all tasks
    total_tasks = len(tasks_to_process)
    print(f"Found {total_tasks} tasks. Each task will be processed for {len(TARGET_TIMESTAMPS)} timestamps.")

    # 3. Prepare CSV files and write headers if needed
    # Predictions CSV
    pred_csv_exists = PREDICTIONS_CSV_PATH.exists()
    pred_csv_is_empty = pred_csv_exists and PREDICTIONS_CSV_PATH.stat().st_size == 0
    pred_write_header = not pred_csv_exists or pred_csv_is_empty

    pred_csv_header = ["lat", "lon", "time", "task_type"] + WEATHER_VARIABLES

    # Residuals CSV
    res_csv_exists = RESIDUALS_CSV_FILE.exists()
    res_csv_is_empty = res_csv_exists and RESIDUALS_CSV_FILE.stat().st_size == 0
    res_write_header = not res_csv_exists or res_csv_is_empty

    residuals_header = ["lat", "lon", "time", "task_type"]
    for var in WEATHER_VARIABLES:
        residuals_header.extend([f"pred_{var}", f"actual_{var}", f"error_{var}"])

    try:
        with open(PREDICTIONS_CSV_PATH, 'a', newline='') as pred_csvfile, \
             open(RESIDUALS_CSV_FILE, 'a', newline='') as res_csvfile, \
             open(RESUME_LOG_FILE, 'a') as resume_log:

            pred_csv_writer = csv.writer(pred_csvfile)
            if pred_write_header:
                pred_csv_writer.writerow(pred_csv_header)
                print(f"Written header to {PREDICTIONS_CSV_PATH}")

            res_csv_writer = csv.writer(res_csvfile)
            if res_write_header:
                res_csv_writer.writerow(residuals_header)
                print(f"Written header to {RESIDUALS_CSV_FILE}")

            # 4. Main loop for processing selected tasks
            predictions_written_count = 0
            residuals_written_count = 0

            for i, task_item in enumerate(tasks_to_process):
                lat = task_item.get('lat') # Assume float
                lon = task_item.get('lon') # Assume float
                task_type = task_item.get('task', 'unknown') # Default task_type if missing

                if lat is None or lon is None:
                    print(f"Warning: Skipping task {i+1}/{total_tasks} ('{task_type}') due to missing 'lat' or 'lon': {task_item}.", file=sys.stderr)
                    continue

                for target_timestamp_str in TARGET_TIMESTAMPS:
                    log_entry_key = f"{float(lat)}_{float(lon)}_{target_timestamp_str}_{task_type}"

                    if log_entry_key in processed_task_keys:
                        print(f"Skipping already processed task {i+1}/{total_tasks} ('{task_type}') for lat: {lat}, lon: {lon}, timestamp: {target_timestamp_str}")
                        continue

                    print(f"Processing task {i+1}/{total_tasks} ('{task_type}') for lat: {lat}, lon: {lon}, timestamp: {target_timestamp_str}")

                    # Convert target_timestamp_str to datetime object for aggregation and potential use
                    try:
                        # Ensure target_timestamp_str is treated as UTC if naive
                        if 'Z' not in target_timestamp_str and '+' not in target_timestamp_str:
                            target_dt = datetime.fromisoformat(target_timestamp_str + 'Z')
                        else:
                            target_dt = datetime.fromisoformat(target_timestamp_str)
                        target_dt = target_dt.astimezone(timezone.utc) if target_dt.tzinfo is None else target_dt
                    except ValueError as e:
                        print(f"Error parsing target_timestamp_str '{target_timestamp_str}': {e}. Skipping this timestamp.", file=sys.stderr)
                        continue

                    # 5. Call physics_extrapolate.py
                    input_payload = json.dumps({
                        "lat": lat,
                        "lon": lon,
                        "time": target_timestamp_str # physics_extrapolate expects ISO string
                    })

                    prediction_result = None
                    try:
                        process = subprocess.run(
                            [sys.executable, str(PHYSICS_EXTRAPOLATE_SCRIPT_PATH)],
                            input=input_payload,
                            capture_output=True,
                            text=True,
                            check=False
                        )

                        if process.returncode != 0:
                            print(f"Error calling physics_extrapolate.py for task {task_item}, timestamp {target_timestamp_str}:", file=sys.stderr)
                            print(f"  Return code: {process.returncode}", file=sys.stderr)
                            print(f"  Stderr: {process.stderr.strip()}", file=sys.stderr)
                            continue

                        if process.stderr:
                             print(f"Warning: physics_extrapolate.py produced stderr for task {task_item}, timestamp {target_timestamp_str}: {process.stderr.strip()}", file=sys.stderr)

                        prediction_result = json.loads(process.stdout)
                    except json.JSONDecodeError:
                        print(f"Error: Could not decode JSON output from physics_extrapolate.py for task {task_item}, timestamp {target_timestamp_str}.", file=sys.stderr)
                        print(f"  Raw stdout: {process.stdout}", file=sys.stderr)
                        continue
                    except Exception as e: # Catch other subprocess related errors
                        print(f"Subprocess execution error for task {task_item}, timestamp {target_timestamp_str}: {e}", file=sys.stderr)
                        continue

                    if not prediction_result: # If subprocess failed or JSON parsing failed
                        continue

                    # 7. Store prediction (this part is common to all tasks)
                    output_lat_pred = prediction_result.get('lat')
                    output_lon_pred = prediction_result.get('lon')
                    output_time_pred = prediction_result.get('time')
                    predicted_variable_dict = prediction_result.get('variable_dict', {})

                    if output_lat_pred is None or output_lon_pred is None or output_time_pred is None:
                        print(f"Error: Output from physics_extrapolate.py for task {task_item}, ts {target_timestamp_str} is missing lat/lon/time.", file=sys.stderr)
                        continue

                    pred_row_values = [output_lat_pred, output_lon_pred, output_time_pred, task_type]
                        for var_name in WEATHER_VARIABLES:
                        pred_row_values.append(predicted_variable_dict.get(var_name, None))

                    pred_csv_writer.writerow(pred_row_values)
                    pred_csvfile.flush()
                    predictions_written_count += 1

                    # 8. Handle "validate" tasks
                    if task_type == "validate":
                        coord_key = (float(lat), float(lon)) # Ensure float for lookup
                        raw_hourly_data = raw_weather_data_map.get(coord_key)

                        if not raw_hourly_data:
                            print(f"Warning: No raw weather data found for validation task at {coord_key}, ts {target_timestamp_str}. Skipping residual calculation.", file=sys.stderr)
                        else:
                            actual_aggregated_vars = aggregate_hourly_to_3hourly(raw_hourly_data, target_dt, WEATHER_VARIABLES)

                            if actual_aggregated_vars:
                                residuals_row = [output_lat_pred, output_lon_pred, output_time_pred, task_type]
                                all_residuals_calculable = True
                                for var_name in WEATHER_VARIABLES:
                                    pred_val = predicted_variable_dict.get(var_name)
                                    actual_val = actual_aggregated_vars.get(var_name)
                                    residuals_row.append(pred_val)
                                    residuals_row.append(actual_val)
                                    if pred_val is not None and actual_val is not None:
                                        try:
                                            residuals_row.append(float(pred_val) - float(actual_val))
                                        except (ValueError, TypeError):
                                            residuals_row.append(None) # Error in calculation
                                            all_residuals_calculable = False
                                    else:
                                        residuals_row.append(None) # Missing pred or actual
                                        all_residuals_calculable = False

                                if all_residuals_calculable: # Only write if all residuals could be computed
                                    res_csv_writer.writerow(residuals_row)
                                    res_csvfile.flush()
                                    residuals_written_count += 1
                                else:
                                    print(f"Warning: Could not calculate all residuals for task {task_item}, ts {target_timestamp_str}. Row not written to residuals.csv.", file=sys.stderr)

                            else:
                                print(f"Warning: Could not aggregate actual weather data for validation task at {coord_key}, ts {target_timestamp_str}. Skipping residual calculation.", file=sys.stderr)

                    # If all successful, log it for resumability
                    resume_log.write(log_entry_key + "\n")
                    resume_log.flush()
                    processed_task_keys.add(log_entry_key)

            print(f"Finished processing. Total predictions written: {predictions_written_count}. Total residuals written: {residuals_written_count}.")

    except IOError as e:
        print(f"IOError during file operations: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"A critical error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # For testing, we might need a dummy task_queue.json and physics_extrapolate.py
    # Ensure physics_extrapolate.py is executable or called with python interpreter

    # Create a dummy task_queue.json and weather_data_collection.json if they don't exist for local testing
    if not TASK_QUEUE_PATH.exists():
        print(f"Creating dummy {TASK_QUEUE_PATH.name} for local testing.")
        dummy_tasks_content = [
            {"lat": 47.2, "lon": 11.3, "task": "validate"}, # For validation
            {"lat": 47.5, "lon": 11.8, "task": "grid_forecast"},
            {"lat": 47.2, "lon": 11.3, "task": "another_task_type"} # Duplicate coords, diff task
        ]
        with open(TASK_QUEUE_PATH, 'w') as f:
            json.dump(dummy_tasks_content, f)

    if not RAW_WEATHER_DATA_FILE.exists():
        print(f"Creating dummy {RAW_WEATHER_DATA_FILE.name} for local testing.")
        # Create dummy weather_data_collection.json - must align with TARGET_TIMESTAMPS and WEATHER_VARIABLES for aggregation
        # For target "2025-05-23T10:30:00", we need 09:30, 10:30, 11:30 data.
        # Let's make one entry match a validation task: lat 47.2, lon 11.3
        dummy_raw_weather_content = [{
            "latitude": 47.2, "longitude": 11.3, "elevation": 500,
            "hourly_units": {var: "unit" for var in WEATHER_VARIABLES},
            "hourly": {
                "time": [
                    (datetime.fromisoformat("2025-05-23T07:30:00Z") - timedelta(hours=1)).isoformat(), # 06:30
                    "2025-05-23T07:30:00Z", # 07:30
                    (datetime.fromisoformat("2025-05-23T07:30:00Z") + timedelta(hours=1)).isoformat(), # 08:30
                    (datetime.fromisoformat("2025-05-23T10:30:00Z") - timedelta(hours=1)).isoformat(), # 09:30
                    "2025-05-23T10:30:00Z", # 10:30
                    (datetime.fromisoformat("2025-05-23T10:30:00Z") + timedelta(hours=1)).isoformat(), # 11:30
                    (datetime.fromisoformat("2025-05-23T13:30:00Z") - timedelta(hours=1)).isoformat(), # 12:30
                    "2025-05-23T13:30:00Z", # 13:30
                    (datetime.fromisoformat("2025-05-23T13:30:00Z") + timedelta(hours=1)).isoformat(), # 14:30
                    (datetime.fromisoformat("2025-05-23T16:30:00Z") - timedelta(hours=1)).isoformat(), # 15:30
                    "2025-05-23T16:30:00Z", # 16:30
                    (datetime.fromisoformat("2025-05-23T16:30:00Z") + timedelta(hours=1)).isoformat()  # 17:30
                ],
                # Populate dummy data for each weather variable for all these times
                **{var: [10+i*0.1+j*0.01 for j in range(12)] for i, var in enumerate(WEATHER_VARIABLES)}
            }
        }]
        RAW_WEATHER_DATA_FILE.parent.mkdir(parents=True, exist_ok=True) # Ensure meteo_api dir exists
        with open(RAW_WEATHER_DATA_FILE, 'w') as f:
            json.dump(dummy_raw_weather_content, f)

    main()
