#!/usr/bin/env python3

from pysolar.solar import get_altitude, get_azimuth
from datetime import datetime
import pytz

# Same configuration as our shadow script
CENTER_LAT = 47.0  # °N - Center of Tirol
CENTER_LON = 11.0  # °E - Center of Tirol
TIROL_TZ = pytz.timezone('Europe/Vienna')
TARGET_DATE = datetime(2025, 5, 23).date()

# Calculate sun angles for 16:30
dt_1630 = datetime(TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day, 
                   16, 30, tzinfo=TIROL_TZ)

altitude = get_altitude(CENTER_LAT, CENTER_LON, dt_1630)
azimuth = get_azimuth(CENTER_LAT, CENTER_LON, dt_1630)

print(f"May 23rd, 2025 at 16:30 (Center of Tirol):")
print(f"Sun altitude: {altitude:.2f}°")
print(f"Sun azimuth: {azimuth:.2f}°")

# Calculate ski-relevant shadow distance
# Elevation difference: 3,768m (Wildspitze) - 2,000m (ski base) = 1,768m
from math import tan, radians

elevation_diff = 1768  # meters
shadow_length = elevation_diff / tan(radians(altitude))

print(f"\nSki-relevant shadow calculation:")
print(f"Elevation difference: {elevation_diff}m")
print(f"Maximum shadow length: {shadow_length:.0f}m = {shadow_length/1000:.1f}km")
