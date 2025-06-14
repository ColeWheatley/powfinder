#!/usr/bin/env python3
"""
Query temperature TIF files at specific locations to compare:
- Innsbruck downtown (47.2364, 11.3884) - valley location
- Patscherkoffel peak (47.1846, 11.4873) - mountain peak
"""

import rasterio
import numpy as np
from rasterio.transform import rowcol
from pyproj import Transformer
import os
import json

# Locations to query
locations = {
    "Innsbruck Downtown": {"lat": 47.2364, "lon": 11.3884, "description": "Valley location"},
    "Patscherkoffel Peak": {"lat": 47.1846, "lon": 11.4873, "description": "Mountain peak"}
}

# Available timestamps
timestamps = [
    "2025-05-24T09:00:00",
    "2025-05-24T12:00:00",
    "2025-05-24T15:00:00",
    "2025-05-24T18:00:00"
]

def query_temperature_at_location(tif_path, lat, lon):
    """Query temperature at a specific lat/lon from a TIF file."""
    try:
        with rasterio.open(tif_path) as src:
            # Transform from WGS84 to the TIF's CRS if needed
            if src.crs != 'EPSG:4326':
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
            else:
                x, y = lon, lat
            
            # Get pixel coordinates
            row, col = rowcol(src.transform, x, y)
            
            # Check if coordinates are within the raster bounds
            if 0 <= row < src.height and 0 <= col < src.width:
                # Read the temperature value
                temp_data = src.read(1)
                temp_value = temp_data[row, col]
                
                # Check for nodata
                if temp_value == src.nodata or np.isnan(temp_value):
                    return None
                    
                return float(temp_value)
            else:
                return None
                
    except Exception as e:
        print(f"Error reading {tif_path}: {e}")
        return None

def get_tif_range(tif_path):
    """Get the temperature range from a TIF file."""
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1, masked=True)  # Read as masked array
            if data.count() > 0:  # If there's valid data
                return float(data.min()), float(data.max())
            else:
                return None, None
    except Exception as e:
        print(f"Error reading range from {tif_path}: {e}")
        return None, None

def get_weather_data_range():
    """Get temperature range from the original weather JSON data."""
    weather_path = "resources/meteo_api/weather_data_3hour.json"
    if not os.path.exists(weather_path):
        return None, None
    
    try:
        with open(weather_path, 'r') as f:
            data = json.load(f)
        
        all_temps = []
        for coord in data['coordinates']:
            if 'weather_data_3hour' in coord:
                temps = coord['weather_data_3hour']['hourly']['temperature_2m']
                all_temps.extend([t for t in temps if t is not None])
        
        if all_temps:
            return min(all_temps), max(all_temps)
        else:
            return None, None
            
    except Exception as e:
        print(f"Error reading weather data: {e}")
        return None, None

def main():
    print("=== Temperature Comparison Analysis ===\n")
    
    # First, check the overall temperature ranges
    print("1. Overall Temperature Ranges:")
    
    # Get weather data range
    weather_min, weather_max = get_weather_data_range()
    if weather_min is not None:
        print(f"   Original weather data: {weather_min:.1f}°C to {weather_max:.1f}°C")
    else:
        print("   Could not read weather data range")
    
    # Check a sample TIF file range
    sample_tif = "TIFS/100m_resolution/2025-05-23T12:00:00/t2m.tif"
    if os.path.exists(sample_tif):
        tif_min, tif_max = get_tif_range(sample_tif)
        if tif_min is not None:
            print(f"   Sample TIF (2025-05-23T12:00): {tif_min:.1f}°C to {tif_max:.1f}°C")
        else:
            print("   Could not read TIF range")
    
    print("\n2. Location-specific Temperature Queries:")
    
    # Query each location for each timestamp
    for timestamp in timestamps:
        tif_path = f"TIFS/100m_resolution/{timestamp}/t2m.tif"
        
        if not os.path.exists(tif_path):
            print(f"   {timestamp}: TIF file not found")
            continue
            
        print(f"\n   {timestamp}:")
        
        for location_name, location_data in locations.items():
            lat, lon = location_data["lat"], location_data["lon"]
            temp = query_temperature_at_location(tif_path, lat, lon)
            
            if temp is not None:
                print(f"     {location_name}: {temp:.1f}°C ({location_data['description']})")
            else:
                print(f"     {location_name}: No data (outside coverage area)")
    
    print("\n3. Analysis:")
    print("   - Compare valley vs mountain temperatures")
    print("   - Check if TIF range extends beyond weather data range")
    print("   - Evaluate interpolation effectiveness")

if __name__ == "__main__":
    main()
