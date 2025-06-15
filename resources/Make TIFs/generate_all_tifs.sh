#!/bin/bash

# TIF Generation Scripts - Comment out any you don't want to run
# =============================================================
scripts=(
    "generate_cloud_cover_tifs.py"
    "generate_dewpoint_2m_tifs.py" 
    "generate_freezing_level_height_tifs.py"
    "generate_relative_humidity_2m_tifs.py"
    "generate_shortwave_radiation_tifs.py"
    "generate_snowfall_tifs.py"
    "generate_surface_pressure_tifs.py"
    "generate_temperature_2m_tifs.py"
    "generate_weather_code_tifs.py"
    "generate_sqh_tifs.py"
    "generate_skiability_tifs.py"
)

# Change to the project root directory so relative paths work
cd /Users/cole/dev/PowFinder

# Run all scripts in parallel
echo "Starting TIF generation for ${#scripts[@]} scripts in parallel..."
echo "Using all available CPU cores for maximum performance"
echo "=================================================="

# Start all scripts in background
pids=()
for script in "${scripts[@]}"; do
    if [[ ! $script =~ ^# ]]; then  # Skip commented lines
        echo "Starting: $script"
        python3 "resources/Make TIFs/$script" &
        pids+=($!)  
    else
        echo "Skipping: $script (commented out)"
    fi
done

# Wait for all background processes to complete
echo ""
echo "Waiting for all scripts to complete..."
for pid in "${pids[@]}"; do
    wait $pid
    echo "Process $pid completed"
done

echo ""
echo "All TIF generation scripts completed!"
echo "You can now run render_pngs.py to generate PNGs from the new TIFs"
