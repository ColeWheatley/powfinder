from pysolar.solar import get_altitude, get_azimuth
# Ensure output directory exists
import os
HILLSHADE_OUT_DIR = '/Users/cole/dev/PowFinder/resources/hillshade/'
os.makedirs(HILLSHADE_OUT_DIR, exist_ok=True)
from datetime import datetime
import pytz
import os
from whitebox.whitebox_tools import WhiteboxTools

# --- HILLSHADE MODEL CONFIGURATION ---

# Input/output paths for all resolutions
DEM_PATHS = {
    '5m': '/Users/cole/dev/PowFinder/resources/terrains/tirol_5m_highprec.tif',
    '25m': '/Users/cole/dev/PowFinder/resources/terrains/tirol_25m_float.tif',
    '100m': '/Users/cole/dev/PowFinder/resources/terrains/tirol_100m_float.tif',
}
RESOLUTIONS = ['5m', '25m', '100m']

# Median times for each 3-hour window (CET) - May 23rd 2025 for skiing hintercasting demo
TIROL_TZ = pytz.timezone('Europe/Vienna')
# Using May 23rd, 2025 for retroactive skiing condition analysis (skiing day was May 24th)
target_date = datetime(2025, 5, 23).date()
MEDIAN_TIMES = [
    datetime(target_date.year, target_date.month, target_date.day, 7, 30, tzinfo=TIROL_TZ),
    datetime(target_date.year, target_date.month, target_date.day, 10, 30, tzinfo=TIROL_TZ),
    datetime(target_date.year, target_date.month, target_date.day, 13, 30, tzinfo=TIROL_TZ),
    datetime(target_date.year, target_date.month, target_date.day, 16, 30, tzinfo=TIROL_TZ)
]

# Tirol center for sun angle calculation
CENTER_LAT = 47.267
CENTER_LON = 11.393

def get_sun_angles(dt, lat, lon):
    """Return (azimuth, altitude) in degrees for given datetime and location."""
    az = get_azimuth(lat, lon, dt)
    alt = get_altitude(lat, lon, dt)
    return az, alt

def main():
    wbt = WhiteboxTools()
    wbt.set_working_dir(os.getcwd())
    for res in RESOLUTIONS:
        dem_path = DEM_PATHS[res]
        for i, dt in enumerate(MEDIAN_TIMES):
            az, alt = get_sun_angles(dt, CENTER_LAT, CENTER_LON)
            print(f"{res} Period {i+1}: {dt.strftime('%H:%M')} azimuth={az:.2f} altitude={alt:.2f}")
            out_path = os.path.join(HILLSHADE_OUT_DIR, f'hillshade_{res}_period{i+1}.tif')
            if not os.path.exists(out_path):
                wbt.hillshade(
                    dem=dem_path,
                    output=out_path,
                    azimuth=az,
                    altitude=alt,
                    zfactor=1.0
                )
                print(f"  Saved: {out_path}")
            else:
                print(f"  Skipping (already exists): {out_path}")

if __name__ == "__main__":
    main()
# --- HILLSHADE MODEL CONFIGURATION ---

# Center of Tirol (approximate)
CENTER_LAT = 47.267
CENTER_LON = 11.393

# Get current date in Tirol timezone
TIROL_TZ = pytz.timezone('Europe/Vienna')
now = datetime.now(TIROL_TZ)
today = now.date()

# Define 3-hour windows (CET)
HILLSHADE_WINDOWS = [
    (6, 9),
    (9, 12),
    (12, 15),
    (15, 18)
]

import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.windows import Window
from datetime import datetime, timedelta
import pytz
from pyproj import Proj, Transformer
import math

# Solar position calculation
from pysolar.solar import get_altitude, get_azimuth

# For progress bars and logging
from tqdm import tqdm
import logging
