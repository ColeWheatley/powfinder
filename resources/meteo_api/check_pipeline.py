#!/usr/bin/env python3
"""
Weather Pipeline Validation
==========================

Quick validation script to check the weather data collection and aggregation pipeline.
"""

import json
from pathlib import Path

def check_pipeline_files():
    """Check if the pipeline files exist and show their status."""
    
    files_to_check = [
        ("all_points.json", "Coordinate generation"),
        ("weather_data_collection.json", "Hourly weather data"),
        ("weather_data_3hour.json", "3-hour aggregated data")
    ]
    
    print("🔍 PowFinder Weather Pipeline Status")
    print("=" * 45)
    
    for filename, description in files_to_check:
        filepath = Path(filename)
        
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {description}")
            print(f"   📁 {filename} ({size_mb:.1f}MB)")
            
            # Show data summary
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                if filename == "all_points.json":
                    coord_count = len(data.get('coordinates', []))
                    validation_count = sum(1 for c in data['coordinates'] if c.get('is_validation', False))
                    print(f"   📊 {coord_count} coordinates ({validation_count} validation points)")
                
                elif filename == "weather_data_collection.json":
                    metadata = data.get('metadata', {})
                    collected = metadata.get('collected_count', 0)
                    total = metadata.get('total_coordinates', 0)
                    print(f"   📊 {collected}/{total} coordinates collected")
                
                elif filename == "weather_data_3hour.json":
                    metadata = data.get('metadata', {})
                    agg_info = metadata.get('aggregation_info', {})
                    successful = agg_info.get('successful_aggregations', 0)
                    total = metadata.get('total_coordinates', 0)
                    print(f"   📊 {successful}/{total} coordinates aggregated")
                    
                    # Sample coordinate info
                    sample_coord = None
                    for coord in data['coordinates']:
                        if coord.get('weather_data_3hour'):
                            sample_coord = coord
                            break
                    
                    if sample_coord:
                        periods = len(sample_coord['weather_data_3hour']['hourly']['time'])
                        print(f"   📊 {periods} time periods per coordinate")
            
            except Exception as e:
                print(f"   ⚠️  Could not read file: {e}")
        else:
            print(f"❌ {description}")
            print(f"   📁 {filename} (missing)")
    
    print("\n📋 NEXT STEPS")
    print("=" * 20)
    
    if not Path("all_points.json").exists():
        print("1. Generate coordinates: python generate_random_coordinates.py")
    elif not Path("weather_data_collection.json").exists():
        print("1. Collect weather data: python collect_weather_data.py")
    elif not Path("weather_data_3hour.json").exists():
        print("1. Aggregate weather data: python aggregate_weather_data.py")
    else:
        print("✅ Pipeline complete! Ready for terrain analysis integration.")
        print("   Next: Implement physics-based extrapolation pipeline")

if __name__ == "__main__":
    check_pipeline_files()
