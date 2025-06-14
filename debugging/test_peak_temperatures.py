#!/usr/bin/env python3
"""
Peak Temperature Test Script
===========================

Tests temperature values at 5 specific Tirol peaks by:
1. Finding exact coordinates from tirol_peaks.geojson
2. Querying TIF temperature at peak location
3. Querying TIF temperature 100m south of peak
4. Finding API temperature from weather_data_3hour.json
5. Comparing all three values

Target peaks: Glungezer, Patscherkoffel, Rosskugel, Wildspitze, Lanser Kopf
Test timestamp: 2025-05-24T12:00:00 (noon)
"""

import json
import rasterio
import numpy as np
from rasterio.transform import rowcol
from pyproj import Transformer
import math

# Configuration
TIF_PATH = "TIFS/100m_resolution/2025-05-24T12:00:00/t2m.tif"
WEATHER_JSON = "resources/meteo_api/weather_data_3hour.json"
PEAKS_JSON = "resources/meteo_api/tirol_peaks.geojson"
TARGET_TIME = "2025-05-24T12:00:00"
OFFSET_METERS = 100  # Distance south to test

# Target peaks
TARGET_PEAKS = ['Glungezer', 'Patscherkoffel', 'Rosskugel', 'Wildspitze', 'Lanser Kopf']

def load_peak_coordinates():
    """Load exact coordinates of target peaks from GeoJSON."""
    with open(PEAKS_JSON, 'r') as f:
        peaks_data = json.load(f)
    
    found_peaks = {}
    
    # First try exact matches
    for feature in peaks_data['features']:
        props = feature['properties']
        if 'name' not in props:
            continue
            
        name = props['name']
        if name in TARGET_PEAKS:
            found_peaks[name] = {
                'lat': feature['geometry']['coordinates'][1],
                'lon': feature['geometry']['coordinates'][0], 
                'elevation': int(props['ele']),
                'source': 'exact_match'
            }
    
    # Try partial matches for missing peaks
    missing = [p for p in TARGET_PEAKS if p not in found_peaks]
    for feature in peaks_data['features']:
        props = feature['properties']
        if 'name' not in props:
            continue
            
        name = props['name']
        for target in missing:
            if target.lower() in name.lower():
                found_peaks[target] = {
                    'lat': feature['geometry']['coordinates'][1],
                    'lon': feature['geometry']['coordinates'][0], 
                    'elevation': int(props['ele']),
                    'source': f'partial_match:{name}'
                }
                missing.remove(target)
                break
    
    return found_peaks

def calculate_offset_coordinates(lat, lon, offset_meters, bearing_degrees):
    """Calculate new coordinates offset by distance and bearing."""
    # Convert to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_degrees)
    
    # Earth radius in meters
    earth_radius = 6371000
    
    # Calculate new coordinates
    lat2_rad = math.asin(
        math.sin(lat_rad) * math.cos(offset_meters / earth_radius) +
        math.cos(lat_rad) * math.sin(offset_meters / earth_radius) * math.cos(bearing_rad)
    )
    
    lon2_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(offset_meters / earth_radius) * math.cos(lat_rad),
        math.cos(offset_meters / earth_radius) - math.sin(lat_rad) * math.sin(lat2_rad)
    )
    
    return math.degrees(lat2_rad), math.degrees(lon2_rad)

def query_tif_temperature(tif_path, lat, lon):
    """Query temperature from TIF at specific coordinates."""
    try:
        with rasterio.open(tif_path) as src:
            # Transform coordinates if needed
            if src.crs != 'EPSG:4326':
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
            else:
                x, y = lon, lat
            
            # Get pixel coordinates
            row, col = rowcol(src.transform, x, y)
            
            # Check bounds
            if 0 <= row < src.height and 0 <= col < src.width:
                temp_data = src.read(1)
                temp_value = temp_data[row, col]
                
                # Check for nodata
                if temp_value != src.nodata and not np.isnan(temp_value):
                    return float(temp_value)
            
            return None
            
    except Exception as e:
        print(f"Error querying TIF: {e}")
        return None

def find_api_temperature(peak_name, lat, lon, elevation):
    """Find API temperature for the closest weather station."""
    with open(WEATHER_JSON, 'r') as f:
        weather_data = json.load(f)
    
    closest_coord = None
    min_distance = float('inf')
    
    # Find closest weather coordinate
    for coord in weather_data['coordinates']:
        if 'weather_data_3hour' not in coord or not coord['weather_data_3hour']:
            continue
            
        coord_lat = coord['coordinate_info']['latitude']
        coord_lon = coord['coordinate_info']['longitude']
        
        # Calculate rough distance
        distance = ((lat - coord_lat) ** 2 + (lon - coord_lon) ** 2) ** 0.5
        
        if distance < min_distance:
            min_distance = distance
            closest_coord = coord
    
    if not closest_coord:
        return None, None, None
    
    # Get temperature at target time
    times = closest_coord['weather_data_3hour']['hourly']['time']
    if TARGET_TIME not in times:
        return None, None, None
        
    time_idx = times.index(TARGET_TIME)
    temperature = closest_coord['weather_data_3hour']['hourly']['temperature_2m'][time_idx]
    
    # Return temperature and station info
    station_lat = closest_coord['coordinate_info']['latitude']
    station_lon = closest_coord['coordinate_info']['longitude']
    station_elev = closest_coord['coordinate_info']['elevation']
    station_distance = min_distance * 111000  # Rough conversion to meters
    
    return temperature, (station_lat, station_lon, station_elev), station_distance

def scale_rgb_to_temperature(rgb_value, min_temp=-17.5, max_temp=25.62):
    """Convert RGB value (0-255) back to actual temperature."""
    return min_temp + (rgb_value / 255.0) * (max_temp - min_temp)

def main():
    print("=== Peak Temperature Test ===")
    print(f"Target timestamp: {TARGET_TIME}")
    print(f"TIF file: {TIF_PATH}")
    print("=" * 50)
    
    # Load peak coordinates
    peaks = load_peak_coordinates()
    
    if not peaks:
        print("❌ No peaks found!")
        return
    
    print(f"Found {len(peaks)} peaks:")
    for name, data in peaks.items():
        print(f"  {name}: ({data['lat']:.6f}, {data['lon']:.6f}) - {data['elevation']}m ({data['source']})")
    
    print("\n" + "=" * 80)
    
    # Test each peak
    for peak_name, peak_data in peaks.items():
        print(f"\n🏔️  {peak_name.upper()}")
        print(f"   Coordinates: ({peak_data['lat']:.6f}, {peak_data['lon']:.6f})")
        print(f"   Elevation: {peak_data['elevation']}m")
        
        lat, lon = peak_data['lat'], peak_data['lon']
        
        # 1. Query TIF at peak location
        tif_temp_peak = query_tif_temperature(TIF_PATH, lat, lon)
        
        # 2. Query TIF 100m south (bearing 180°)
        lat_south, lon_south = calculate_offset_coordinates(lat, lon, OFFSET_METERS, 180)
        tif_temp_south = query_tif_temperature(TIF_PATH, lat_south, lon_south)
        
        # 3. Find API temperature from closest weather station
        api_temp, station_info, station_distance = find_api_temperature(peak_name, lat, lon, peak_data['elevation'])
        
        # Display results
        print(f"   📍 Peak location:")
        if tif_temp_peak is not None:
            converted_temp_peak = scale_rgb_to_temperature(tif_temp_peak)
            print(f"      TIF (RGB): {tif_temp_peak:.0f}")
            print(f"      TIF Temperature: {converted_temp_peak:.1f}°C")
        else:
            print(f"      TIF: No data")
        
        print(f"   📍 100m South ({lat_south:.6f}, {lon_south:.6f}):")
        if tif_temp_south is not None:
            converted_temp_south = scale_rgb_to_temperature(tif_temp_south)
            print(f"      TIF (RGB): {tif_temp_south:.0f}")
            print(f"      TIF Temperature: {converted_temp_south:.1f}°C")
        else:
            print(f"      TIF: No data")
        
        print(f"   📡 API Weather Data:")
        if api_temp is not None and station_info is not None:
            station_lat, station_lon, station_elev = station_info
            print(f"      Station: ({station_lat:.6f}, {station_lon:.6f}) - {station_elev}m")
            print(f"      Distance: {station_distance:.0f}m")
            print(f"      API Temperature: {api_temp:.1f}°C")
        else:
            print(f"      No API data found")
        
        # Analysis
        print(f"   📊 Analysis:")
        if tif_temp_peak is not None and api_temp is not None:
            converted_temp_peak = scale_rgb_to_temperature(tif_temp_peak)
            physics_error = converted_temp_peak - api_temp
            print(f"      Physics Error: {physics_error:+.1f}°C (TIF - API)")
        
        if tif_temp_peak is not None and tif_temp_south is not None:
            converted_temp_peak = scale_rgb_to_temperature(tif_temp_peak)
            converted_temp_south = scale_rgb_to_temperature(tif_temp_south)
            spatial_diff = converted_temp_south - converted_temp_peak
            print(f"      Spatial Gradient: {spatial_diff:+.1f}°C (South - Peak)")
        
        print("   " + "-" * 60)

if __name__ == "__main__":
    main()
