#!/usr/bin/env python3
"""
progressive_grid_scheduler.py
-----------------------------

Generates a prioritized task queue for weather extrapolation.
Validation points are processed first, followed by grid points at various resolutions.
"""

import json
import math
import pathlib
import sys

# Define paths relative to the script's location
THIS_DIR = pathlib.Path(__file__).resolve().parent
ALL_POINTS_PATH = THIS_DIR.parent / "meteo_api" / "all_points.json"
GRID_BOUNDS_PATH = THIS_DIR / "grid_bounds.json" # In the same directory as the script

def create_task_queue():
    """
    Main function to generate the task queue.
    """
    tasks = []

    # 1. Load Inputs
    try:
        with open(GRID_BOUNDS_PATH, "r", encoding="utf-8") as fp:
            grid_bounds_config = json.load(fp)
        print(f"Successfully loaded grid bounds configuration from {GRID_BOUNDS_PATH}")
    except FileNotFoundError:
        print(f"Error: Grid bounds file not found at {GRID_BOUNDS_PATH}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {GRID_BOUNDS_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(ALL_POINTS_PATH, "r", encoding="utf-8") as fp:
            all_points_data = json.load(fp)
        print(f"Successfully loaded all points data from {ALL_POINTS_PATH}")
    except FileNotFoundError:
        print(f"Error: All points file not found at {ALL_POINTS_PATH}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {ALL_POINTS_PATH}", file=sys.stderr)
        sys.exit(1)

    # 2. Add Validation Points
    validation_points_added = 0
    if "coordinates" in all_points_data:
        validation_tasks = []
        for point in all_points_data["coordinates"]:
            if point.get("is_validation", False):
                validation_tasks.append({
                    "lat": float(point["latitude"]),
                    "lon": float(point["longitude"]),
                    "task": "validate"
                })
                validation_points_added += 1
        # Sort validation tasks for deterministic order
        validation_tasks.sort(key=lambda p: (p["lat"], p["lon"]))
        tasks.extend(validation_tasks)
    print(f"Added {validation_points_added} validation points to the queue.")

    # 3. Add Grid Points
    min_lat = grid_bounds_config["min_lat"]
    max_lat = grid_bounds_config["max_lat"]
    min_lon = grid_bounds_config["min_lon"]
    max_lon = grid_bounds_config["max_lon"]
    grid_levels_m = grid_bounds_config["grid_levels_m"]
    output_file_name = grid_bounds_config.get("output_file", "task_queue.json")

    avg_lat_rad = math.radians((min_lat + max_lat) / 2.0)

    total_grid_points_added = 0

    for resolution_m in grid_levels_m:
        task_type = f"grid_{resolution_m}"
        grid_level_tasks = []

        lat_spacing_deg = resolution_m / 111000.0
        # Handle potential division by zero or invalid cos value if avg_lat is near poles, though unlikely for typical weather grids
        if math.cos(avg_lat_rad) == 0:
            print(f"Warning: Cosine of average latitude ({avg_lat_rad}) is zero. Skipping longitude grid for resolution {resolution_m}m.", file=sys.stderr)
            lon_spacing_deg = 0 # Or handle as error
        else:
            lon_spacing_deg = resolution_m / (111000.0 * math.cos(avg_lat_rad))

        if lat_spacing_deg == 0 or lon_spacing_deg == 0:
            print(f"Warning: Zero spacing degree calculated for resolution {resolution_m}m. Skipping this level.", file=sys.stderr)
            continue

        num_lat = int(math.ceil((max_lat - min_lat) / lat_spacing_deg))
        num_lon = int(math.ceil((max_lon - min_lon) / lon_spacing_deg))

        current_level_points = 0
        for i in range(num_lat + 1): # +1 to include the max_lat boundary if perfectly aligned
            cell_lat = min_lat + i * lat_spacing_deg
            if cell_lat > max_lat + lat_spacing_deg / 2: # Ensure we don't go too far beyond max_lat
                continue
            # Clamp to max_lat if we exceed it due to step
            cell_lat = min(cell_lat, max_lat)

            for j in range(num_lon + 1): # +1 to include the max_lon boundary
                cell_lon = min_lon + j * lon_spacing_deg
                if cell_lon > max_lon + lon_spacing_deg / 2: # Ensure we don't go too far beyond max_lon
                    continue
                # Clamp to max_lon
                cell_lon = min(cell_lon, max_lon)

                grid_level_tasks.append({
                    "lat": round(cell_lat, 6), # Round for cleaner output
                    "lon": round(cell_lon, 6),
                    "task": task_type
                })
                current_level_points +=1

        # Sort grid tasks for this level for deterministic order
        grid_level_tasks.sort(key=lambda p: (p["lat"], p["lon"]))
        tasks.extend(grid_level_tasks)
        print(f"Added {current_level_points} points for {task_type} resolution.")
        total_grid_points_added += current_level_points

    print(f"Total grid points added: {total_grid_points_added}")

    # 4. Write Output
    output_path = THIS_DIR / output_file_name
    try:
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(tasks, fp, indent=2)
        print(f"Successfully wrote task queue to {output_path}")
    except IOError:
        print(f"Error: Could not write task queue to {output_path}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    create_task_queue()
