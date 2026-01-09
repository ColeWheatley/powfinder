#!/usr/bin/env python3
"""
Test Weather Data Aggregation
=============================

Creates a small test dataset to validate the aggregation script functionality.
"""

import json
from datetime import datetime, timedelta

def create_test_weather_data():
    """Create a small test weather dataset for validation."""
    
    # Create 24 hours of hourly data (starting at midnight)
    start_time = datetime(2025, 5, 14, 0, 0, 0)
    hourly_times = []
    
    for i in range(24):
        time_point = start_time + timedelta(hours=i)
        hourly_times.append(time_point.isoformat() + 'Z')
    
    # Create mock weather data with predictable patterns
    test_data = {
        'metadata': {
            'total_coordinates': 2,
            'collected_count': 2,
            'failed_count': 0,
            'start_date': '2025-05-14',
            'end_date': '2025-05-14'
        },
        'coordinates': [
            {
                'coordinate_info': {
                    'id': 'test_1',
                    'name': 'Test Point 1',
                    'latitude': 47.2692,
                    'longitude': 11.4041,
                    'elevation': 2000,
                    'source': 'test'
                },
                'status': 'collected',
                'weather_data': {
                    'hourly': {
                        'time': hourly_times,
                        'temperature_2m': [i for i in range(24)],  # 0°C to 23°C
                        'snowfall': [1.0 if i % 6 == 0 else 0.0 for i in range(24)],  # Snow every 6 hours
                        'weather_code': [0, 1, 2, 3] * 6,  # Repeating pattern
                        'wind_speed_10m': [10.0] * 24,  # Constant wind
                        'relative_humidity_2m': [50.0 + i for i in range(24)]  # 50% to 73%
                    }
                }
            },
            {
                'coordinate_info': {
                    'id': 'test_2',
                    'name': 'Test Point 2', 
                    'latitude': 47.0000,
                    'longitude': 11.0000,
                    'elevation': 1500,
                    'source': 'test'
                },
                'status': 'failed',
                'error_info': 'Test failure case'
            }
        ]
    }
    
    # Save test data
    with open('test_weather_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print("✅ Created test_weather_data.json")
    
    # Expected results for validation
    print("\n📊 Expected 3-Hour Aggregation Results:")
    print("Time periods: 8 periods (00:00, 03:00, 06:00, ..., 21:00)")
    print("Temperature averages: [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5]")
    print("Snowfall sums: [4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0]")
    print("Weather code medians: [1, 1, 1, 1, 1, 1, 1, 1]")

if __name__ == "__main__":
    create_test_weather_data()
