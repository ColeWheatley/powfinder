#!/usr/bin/env python3
"""
Weather Data Collection Script
=============================

Collects weather data from Open-Meteo API for:
1. First 3000 peaks from tirol_peaks.geojson
2. All 2000 random coordinates from random_coordinates.json

A subset of the random coordinates is used as a validation dataset and flagged as such. 

Outputs comprehensive weather dataset for May 14-28, 2025 analysis.
"""

import json
import requests
import time
from urllib.parse import urlencode
import sys
from pathlib import Path

# =====================================
# CONFIGURATION PARAMETERS
# =====================================

# API configuration
BASE_URL = "https://api.open-meteo.com/v1/forecast"
MODEL = "icon-d2"
START_DATE = "2025-05-14"
END_DATE = "2025-05-28"
TIMEZONE = "Europe/Vienna"

# Weather parameters to collect
HOURLY_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m", 
    "shortwave_radiation",
    "cloud_cover",
    "snow_depth",
    "snowfall",
    "wind_speed_10m",
    "weather_code",
    "freezing_level_height",
    "surface_pressure"
]

# Rate limiting (aggressive but safe for 5000/hour target)
REQUESTS_PER_HOUR = 7200  # Target 7200/hour for comfort margin
DELAY_BETWEEN_REQUESTS = 3600 / REQUESTS_PER_HOUR  # 0.5 seconds between requests

# Retry configuration
MAX_RETRIES = 5  # Maximum retry attempts per coordinate
RETRY_DELAY = 10  # Seconds to wait before retry on error

# Batch configuration
BATCH_SIZE = 10  # Process in small batches for progress tracking
PEAKS_LIMIT = 3000  # First 3000 peaks
RANDOM_LIMIT = 2000  # All 2000 random coordinates

# File paths
PEAKS_FILE = "tirol_peaks.geojson"
RANDOM_FILE = "all_points.json"
OUTPUT_FILE = "weather_data_collection.json"

def load_peak_coordinates():
    """Load first 3000 peak coordinates from GeoJSON file."""
    print(f"📍 Loading peaks from {PEAKS_FILE}...")
    
    try:
        with open(PEAKS_FILE, 'r') as f:
            geojson_data = json.load(f)
        
        coordinates = []
        for i, feature in enumerate(geojson_data['features'][:PEAKS_LIMIT]):
            if i >= PEAKS_LIMIT:
                break
                
            geometry = feature['geometry']
            if geometry['type'] == 'Point':
                lon, lat = geometry['coordinates']
                
                # Get elevation from properties if available
                props = feature.get('properties', {})
                elevation = props.get('ele', props.get('elevation', 'unknown'))
                name = props.get('name', f'Peak_{i+1}')
                
                coordinates.append({
                    'id': f'peak_{i+1}',
                    'name': name,
                    'latitude': lat,
                    'longitude': lon,
                    'elevation': elevation,
                    'source': 'peaks'
                })
        
        print(f"   ✅ Loaded {len(coordinates)} peak coordinates")
        return coordinates
        
    except Exception as e:
        print(f"   ❌ Error loading peaks: {e}")
        return []

def load_random_coordinates():
    """Load all coordinates from all_points.json (includes validation flagging)."""
    print(f"📍 Loading coordinates from {RANDOM_FILE}...")
    
    try:
        with open(RANDOM_FILE, 'r') as f:
            data = json.load(f)
        
        coordinates = []
        for coord in data['coordinates'][:RANDOM_LIMIT]:
            coordinates.append({
                'id': f'random_{coord["id"]}',
                'name': f'Random_Point_{coord["id"]}',
                'latitude': coord['latitude'],
                'longitude': coord['longitude'],
                'elevation': coord['elevation'],
                'source': 'random',
                'is_validation': coord.get('is_validation', False)
            })
        
        validation_count = sum(1 for c in coordinates if c['is_validation'])
        print(f"   ✅ Loaded {len(coordinates)} coordinates ({validation_count} validation points)")
        return coordinates
        
    except Exception as e:
        print(f"   ❌ Error loading coordinates: {e}")
        return []

def build_api_params(latitude, longitude):
    """Build Open-Meteo API parameters for given coordinates."""
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'model': MODEL,
        'hourly': HOURLY_PARAMS,  # Pass as list, let requests handle formatting
        'start_date': START_DATE,
        'end_date': END_DATE,
        'timezone': TIMEZONE
    }
    
    return params

def initialize_collection_file(all_coordinates):
    """Initialize or load existing collection file with coordinate index."""
    if Path(OUTPUT_FILE).exists():
        print(f"📂 Loading existing collection from {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, 'r') as f:
                existing_data = json.load(f)
            
            # Validate structure
            if 'coordinates' in existing_data and len(existing_data['coordinates']) == len(all_coordinates):
                print(f"   ✅ Found existing collection with {len(existing_data['coordinates'])} coordinates")
                return existing_data
            else:
                print(f"   ⚠️  Existing file has different structure, creating new collection")
        except Exception as e:
            print(f"   ❌ Error loading existing file: {e}")
            print(f"   🔄 Creating new collection")
    
    # Create new collection structure
    print(f"🆕 Creating new collection index for {len(all_coordinates)} coordinates...")
    
    collection_data = {
        'metadata': {
            'created_at': time.time(),
            'total_coordinates': len(all_coordinates),
            'collected_count': 0,
            'failed_count': 0,
            'api_config': {
                'model': MODEL,
                'start_date': START_DATE,
                'end_date': END_DATE,
                'timezone': TIMEZONE,
                'parameters': HOURLY_PARAMS
            }
        },
        'coordinates': []
    }
    
    # Initialize coordinate entries
    for i, coord in enumerate(all_coordinates):
        coord_entry = {
            'index': i,
            'coordinate_info': coord,
            'status': 'pending',  # pending, collected, failed
            'weather_data': None,
            'error_info': None,
            'attempts': 0,
            'last_attempt': None
        }
        collection_data['coordinates'].append(coord_entry)
    
    # Save initial structure
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(collection_data, f, indent=2)
    
    print(f"   ✅ Collection index initialized and saved")
    return collection_data

def get_pending_coordinates(collection_data):
    """Get list of coordinates that still need to be collected."""
    pending = []
    for coord_entry in collection_data['coordinates']:
        if coord_entry['status'] == 'pending':
            pending.append(coord_entry)
    return pending

def update_coordinate_entry(collection_data, index, status, weather_data=None, error_info=None, attempt_count=1):
    """Update a specific coordinate entry in the collection."""
    coord_entry = collection_data['coordinates'][index]
    coord_entry['status'] = status
    coord_entry['attempts'] = attempt_count
    coord_entry['last_attempt'] = time.time()
    
    if weather_data:
        coord_entry['weather_data'] = weather_data
    if error_info:
        coord_entry['error_info'] = error_info
    
    # Update metadata counts
    collected_count = sum(1 for c in collection_data['coordinates'] if c['status'] == 'collected')
    failed_count = sum(1 for c in collection_data['coordinates'] if c['status'] == 'failed')
    
    collection_data['metadata']['collected_count'] = collected_count
    collection_data['metadata']['failed_count'] = failed_count
    collection_data['metadata']['last_updated'] = time.time()

def save_collection_progress(collection_data):
    """Save current collection progress to file."""
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(collection_data, f, indent=2)

def fetch_weather_data_resumable(coord_entry):
    """Fetch weather data for a coordinate entry with retry logic."""
    coordinate = coord_entry['coordinate_info']
    current_attempt = coord_entry['attempts'] + 1
    
    params = build_api_params(coordinate['latitude'], coordinate['longitude'])
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        weather_data = response.json()
        
        # Validate that we got actual weather data
        if 'hourly' not in weather_data:
            raise ValueError("Invalid response: missing hourly data")
        
        # Add metadata to response
        weather_data['api_url'] = f"{BASE_URL}?{urlencode(params)}"
        weather_data['collected_at'] = time.time()
        weather_data['attempt'] = current_attempt
        
        return True, weather_data, None
        
    except Exception as e:
        error_info = {
            'error': str(e),
            'api_url': f"{BASE_URL}?{urlencode(params)}",
            'failed_at': time.time(),
            'attempt': current_attempt
        }
        return False, None, error_info

def process_pending_coordinates(collection_data):
    """Process all pending coordinates with resumable logic."""
    pending_coords = get_pending_coordinates(collection_data)
    total_pending = len(pending_coords)
    
    if total_pending == 0:
        print("   ✅ All coordinates already collected!")
        return
    
    print(f"   🔄 Processing {total_pending} pending coordinates...")
    
    for i, coord_entry in enumerate(pending_coords):
        coord_info = coord_entry['coordinate_info']
        coord_index = coord_entry['index']
        
        print(f"   📡 [{i+1}/{total_pending}] {coord_info['id']} ({coord_info['name'][:30]}...)...", end='')
        
        # Try to fetch weather data with retries
        attempt = 0
        success = False
        
        while attempt < MAX_RETRIES and not success:
            attempt += 1
            
            if attempt > 1:
                print(f" ⚠️ (retry {attempt}/{MAX_RETRIES}, waiting {RETRY_DELAY}s...)", end='')
                time.sleep(RETRY_DELAY)
            
            success, weather_data, error_info = fetch_weather_data_resumable(coord_entry)
            
            if success:
                # Update coordinate entry as collected
                update_coordinate_entry(collection_data, coord_index, 'collected', 
                                      weather_data=weather_data, attempt_count=attempt)
                print(" ✅")
                break
            else:
                if attempt == MAX_RETRIES:
                    # Mark as failed after all retries
                    update_coordinate_entry(collection_data, coord_index, 'failed', 
                                          error_info=error_info, attempt_count=attempt)
                    print(" ❌")
        
        # Save progress every 10 coordinates
        if (i + 1) % 10 == 0:
            save_collection_progress(collection_data)
            collected = collection_data['metadata']['collected_count']
            failed = collection_data['metadata']['failed_count']
            total = collection_data['metadata']['total_coordinates']
            print(f"      💾 Progress saved: {collected}/{total} collected, {failed} failed")
        
        # Rate limiting (skip delay on last item)
        if i < total_pending - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Final save
    save_collection_progress(collection_data)

def process_coordinates_batch(coordinates, batch_start, batch_size):
    """Legacy function - replaced by resumable processing."""
    pass

def main():
    """Main weather data collection function with resumable capability."""
    print("🌤️  PowFinder Weather Data Collection")
    print("=" * 50)
    
    # Load coordinates
    peak_coords = load_peak_coordinates()
    random_coords = load_random_coordinates()
    
    all_coordinates = peak_coords + random_coords
    total_count = len(all_coordinates)
    
    if total_count == 0:
        print("❌ No coordinates loaded. Exiting.")
        return
    
    # Initialize or load collection file
    collection_data = initialize_collection_file(all_coordinates)
    
    # Check current status
    collected_count = collection_data['metadata']['collected_count']
    failed_count = collection_data['metadata']['failed_count']
    pending_count = total_count - collected_count - failed_count
    
    print(f"\n📊 Collection Status:")
    print(f"   📍 Total coordinates: {total_count}")
    print(f"   ✅ Already collected: {collected_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   ⏳ Pending: {pending_count}")
    
    if pending_count == 0:
        print(f"\n🎉 All coordinates already collected!")
        return
    
    print(f"\n🎯 Collection Parameters:")
    print(f"   🚀 Request rate: {REQUESTS_PER_HOUR} requests/hour ({DELAY_BETWEEN_REQUESTS:.1f}s between requests)")
    print(f"   🔄 Retry policy: Up to {MAX_RETRIES} attempts with {RETRY_DELAY}s delays")
    print(f"   ⏱️  Estimated time for pending: {pending_count * DELAY_BETWEEN_REQUESTS / 60:.1f} minutes")
    
    # Process pending coordinates immediately
    print(f"\n🚀 Starting resumable collection...")
    process_pending_coordinates(collection_data)
    
    # Final summary
    final_collected = collection_data['metadata']['collected_count']
    final_failed = collection_data['metadata']['failed_count']
    
    print(f"\n🎉 Weather data collection session complete!")
    print(f"   ✅ Successfully collected: {final_collected}")
    print(f"   ❌ Failed after retries: {final_failed}")
    print(f"   📁 Data saved to: {OUTPUT_FILE}")
    
    # Check if we got all datapoints
    if final_collected == total_count:
        print(f"   🎯 SUCCESS: All {total_count} coordinates collected successfully!")
        print(f"   🌟 System ready for weather analysis!")
    else:
        missing = total_count - final_collected
        print(f"   ⚠️  WARNING: Missing {missing} datapoints - system may not work correctly.")
        print(f"   💡 Run script again to retry failed coordinates.")
    
    if final_failed > 0:
        print(f"\n⚠️  {final_failed} coordinates failed after all retries.")
        print(f"   💡 Run script again to retry failed coordinates.")

if __name__ == "__main__":
    main()
