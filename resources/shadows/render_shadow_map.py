#!/usr/bin/env python3
"""
render_shadow_map_grass.py

Computes terrain-cast shadows using GRASS GIS r.sun module with optimizations.
Generates shadow maps for 4 time periods on May 23rd, 2025 for the Tirol region.

Optimizations:
- Uses r.horizon with 2.7km max distance (ski-relevant shadow analysis)
- Uses r.sun with incidout parameter for NULL-based shadow detection  
- 15° azimuth steps for horizon computation balance speed vs accuracy
- Binary shadow output (0=shadow, 255=sunlit) for efficient storage

Based on GRASS documentation:
- r.horizon pre-computes limited-distance terrain obstruction
- r.sun with horizon data produces incidence angles with NULL = shadow
- Shadows occur where incidence angle is NULL (terrain blocks sun)
- Binary output optimized for tiling and database queries

Output format is optimized for:
- Future tiling for OpenLayers display
- Database queries for insolar calculations (hillshade × shadow × clouds)
- Compatibility with existing EPSG:3857 projection and 5m resolution
"""

import os
import sys
import subprocess
import tempfile
import time
from datetime import datetime
import pytz
from pysolar.solar import get_altitude, get_azimuth
import numpy as np
from osgeo import gdal, osr
from pyproj import Proj, Transformer

# --- CONFIGURATION ---

# Shadow processing optimization
MAX_SHADOW_DIST = 2700  # metres - based on ski-relevant elevation analysis
HSTEP = 15              # azimuth step for horizon map (degrees)

# Input DEM path (5m resolution only)
DEM_PATH = '/Users/cole/dev/PowFinder/resources/terrains/tirol_5m_float.tif'
CROPPED_DEM_PATH = '/Users/cole/dev/PowFinder/resources/shadows/tirol_5m_test_crop.tif'

# Output directory
SHADOW_OUT_DIR = '/Users/cole/dev/PowFinder/resources/shadows/'

# Center of Tirol coordinates for full processing
CENTER_LAT = 47.0  # °N - Center of Tirol
CENTER_LON = 11.0  # °E - Center of Tirol

# Timezone
TIROL_TZ = pytz.timezone('Europe/Vienna')

# Target date for shadow calculation
TARGET_DATE = datetime(2025, 5, 23).date()

# Time periods (ALL periods for production)
TIME_PERIODS = [
    (7, 30),   # 07:30
    (10, 30),  # 10:30
    (13, 30),  # 13:30  
    (16, 30)   # 16:30
]

# Create datetime objects for each time period
SHADOW_TIMES = [
    datetime(TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day, 
             hour, minute, tzinfo=TIROL_TZ)
    for hour, minute in TIME_PERIODS
]

# Shadow output mode
SHADOW_MODE = 'binary'  # 'binary' for 0/1, 'gradient' for 0-255 based on incident angle

# --- FUNCTIONS ---

def get_sun_angles(dt, lat, lon):
    """
    Calculate sun azimuth and altitude for given datetime and location.
    
    Args:
        dt: datetime object with timezone
        lat: latitude in degrees
        lon: longitude in degrees
    
    Returns:
        tuple: (azimuth, altitude) in degrees
    """
    az = get_azimuth(lat, lon, dt)
    alt = get_altitude(lat, lon, dt)
    return az, alt

def setup_grass_environment(dem_path, location_name='shadow_calc'):
    """
    Set up GRASS GIS environment for processing.
    
    Args:
        dem_path: path to input DEM
        location_name: name for GRASS location
    
    Returns:
        tuple: (gisdb_path, location_path, mapset)
    """
    # Create temporary GRASS database
    gisdb = tempfile.mkdtemp(prefix='grassdata_')
    location = os.path.join(gisdb, location_name)
    mapset = 'PERMANENT'
    
    # Create location from DEM
    grass_cmd = '/Applications/GRASS-8.4.app/Contents/Resources/bin/grass'
    
    # First create the location
    cmd = [
        grass_cmd, '-c', dem_path, location, '-e'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Failed to create GRASS location: {result.stderr}")
        return None, None, None
    
    # Verify the location was created properly
    if not os.path.exists(os.path.join(location, 'PERMANENT')):
        print(f"ERROR: PERMANENT mapset not created in {location}")
        return None, None, None
    
    return gisdb, location, mapset

def generate_shadow_map_grass(dem_path, output_path, azimuth, altitude,
                             gisdb, location, mapset, dt, mode='binary'):
    """
    Generate shadow map using GRASS GIS r.sun module.
    
    Args:
        dem_path: path to input DEM
        output_path: path for output shadow map
        azimuth: sun azimuth in degrees
        altitude: sun altitude in degrees
        gisdb: GRASS database path
        location: GRASS location path
        mapset: GRASS mapset name
        dt: datetime object for the calculation time
        mode: 'binary' or 'gradient' shadow output
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set up GRASS environment variables
        grass_cmd = '/Applications/GRASS-8.4.app/Contents/Resources/bin/grass'
        
        # Use full path to location/mapset (DEM and horizon already imported)
        grass_location = f"{location}/PERMANENT"
        
        # Calculate solar parameters for the specific time
        # r.sun needs day of year and time
        day_of_year = dt.timetuple().tm_yday  # May 23, 2025 is day 143
        decimal_time = dt.hour + dt.minute / 60.0  # Convert to decimal hours
        
        # Run r.sun with horizon data and incidence angle output
        print(f"  Running r.sun with horizon optimization...")
        incid_map = 'incid'
        sun_cmd = [
            grass_cmd, grass_location, '--exec',
            'r.sun',
            'elevation=dem',
            f'horizon_basename=hor{HSTEP}',
            f'horizon_step={HSTEP}',
            f'day={day_of_year}',
            f'time={decimal_time}',
            'beam_rad=',      # omit - we don't want irradiance
            'diff_rad=',
            'refl_rad=',
            'glob_rad=',
            f'incidout={incid_map}',  # solar incidence angle with NULLs = shadow
            '--overwrite'
        ]
        
        result = subprocess.run(sun_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: Failed to run r.sun: {result.stderr}")
            return False
        
        # Debug: Get statistics of raw incidence angle values
        stats_cmd = [
            grass_cmd, grass_location, '--exec',
            'r.univar', f'map={incid_map}'
        ]
        
        result = subprocess.run(stats_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Incidence angle stats: {result.stdout.strip()}")
        
        # Create binary shadow map from incidence angle
        # NULL values in incidence map = shadow (terrain blocks sun)
        # Valid values = sunlit (sun reaches ground)
        shadow_cmd = [
            grass_cmd, grass_location, '--exec',
            'r.mapcalc',
            f'shadow_byte = isnull({incid_map}) ? 0 : 255',
            '--overwrite'
        ]
        
        result = subprocess.run(shadow_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: Failed to create shadow map: {result.stderr}")
            return False
        
        # Export shadow map as GeoTIFF without color table
        export_cmd = [
            grass_cmd, grass_location, '--exec',
            'r.out.gdal',
            'input=shadow_byte',
            f'output={output_path}',
            'type=Byte',  # 8-bit unsigned
            'format=GTiff',
            'createopt=COMPRESS=LZW,TILED=YES',
            '--overwrite'
        ]
        
        result = subprocess.run(export_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: Failed to export shadow map: {result.stderr}")
            return False
        
        # Remove any color table to ensure grayscale display
        # Use gdal_calc.py to force pure grayscale values
        temp_calc = f'{output_path}.calc.tif'
        calc_cmd = [
            'gdal_calc.py',
            '-A', output_path,
            '--outfile', temp_calc,
            '--calc', 'A',  # Just copy values as-is
            '--type', 'Byte',
            '--co', 'COMPRESS=LZW',
            '--co', 'TILED=YES'
        ]
        
        result = subprocess.run(calc_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Replace original with clean version (no color table, no georeferencing)
            import shutil
            shutil.move(temp_calc, output_path)
            print(f"  Removed color table and georeferencing for Mac Preview compatibility")
        else:
            print(f"  Warning: Could not remove color table: {result.stderr}")
            # Clean up temp file if it exists
            if os.path.exists(temp_calc):
                os.remove(temp_calc)
        
        return True
        
    except Exception as e:
        print(f"  ERROR: Exception in shadow generation: {str(e)}")
        return False

def verify_output_format(file_path):
    """
    Verify the format of the output shadow map using GDAL.
    
    Args:
        file_path: path to the shadow map file
    
    Returns:
        dict: format information
    """
    try:
        ds = gdal.Open(file_path)
        if ds is None:
            return None
        
        band = ds.GetRasterBand(1)
        
        info = {
            'driver': ds.GetDriver().ShortName,
            'dtype': gdal.GetDataTypeName(band.DataType),
            'dtype_size': gdal.GetDataTypeSize(band.DataType),
            'width': ds.RasterXSize,
            'height': ds.RasterYSize,
            'projection': ds.GetProjection(),
            'geotransform': ds.GetGeoTransform(),
            'min': band.GetMinimum(),
            'max': band.GetMaximum(),
            'stats': band.GetStatistics(True, True)  # [min, max, mean, stddev]
        }
        
        # Check projection
        srs = osr.SpatialReference()
        srs.ImportFromWkt(info['projection'])
        info['epsg'] = srs.GetAttrValue('AUTHORITY', 1)
        
        # Calculate resolution
        gt = info['geotransform']
        info['pixel_width'] = abs(gt[1])
        info['pixel_height'] = abs(gt[5])
        
        ds = None
        return info
        
    except Exception as e:
        print(f"ERROR verifying output: {str(e)}")
        return None

def crop_dem_around_center(input_dem_path, output_dem_path, center_lat, center_lon, size_km=2):
    """
    Crop a 2km x 2km area around the center coordinates from the input DEM.
    
    Args:
        input_dem_path: path to input DEM
        output_dem_path: path for cropped output DEM
        center_lat: center latitude in degrees
        center_lon: center longitude in degrees  
        size_km: size of square crop in kilometers
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Cropping {size_km}km x {size_km}km area around {center_lat:.3f}°N, {center_lon:.3f}°E...")
        
        # Open input DEM
        ds = gdal.Open(input_dem_path)
        if ds is None:
            print(f"ERROR: Could not open input DEM: {input_dem_path}")
            return False
        
        # Get spatial reference and geotransform
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        gt = ds.GetGeoTransform()
        
        print(f"Input DEM projection: {srs.GetAttrValue('AUTHORITY', 1) if srs.GetAttrValue('AUTHORITY') else 'Unknown'}")
        print(f"Input DEM pixel size: {abs(gt[1]):.1f}m x {abs(gt[5]):.1f}m")
        print(f"Input DEM dimensions: {ds.RasterXSize} x {ds.RasterYSize}")
        
        # Transform center coordinates to DEM coordinate system
        # Create transformer from WGS84 to DEM CRS
        src_crs = "EPSG:4326"  # WGS84
        dst_crs = srs.ExportToProj4()
        
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        center_x, center_y = transformer.transform(center_lon, center_lat)
        
        print(f"Center in DEM coordinates: {center_x:.0f}, {center_y:.0f}")
        
        # Calculate crop bounds in DEM coordinates
        half_size_m = (size_km * 1000) / 2  # Convert km to meters
        min_x = center_x - half_size_m
        max_x = center_x + half_size_m
        min_y = center_y - half_size_m
        max_y = center_y + half_size_m
        
        print(f"Crop bounds: {min_x:.0f}, {min_y:.0f}, {max_x:.0f}, {max_y:.0f}")
        
        # Convert to pixel coordinates
        inv_gt = gdal.InvGeoTransform(gt)
        px_min_x, px_min_y = gdal.ApplyGeoTransform(inv_gt, min_x, max_y)  # Note: max_y for min pixel row
        px_max_x, px_max_y = gdal.ApplyGeoTransform(inv_gt, max_x, min_y)  # Note: min_y for max pixel row
        
        # Round and ensure integers
        px_min_x, px_min_y = int(round(px_min_x)), int(round(px_min_y))
        px_max_x, px_max_y = int(round(px_max_x)), int(round(px_max_y))
        
        # Calculate crop dimensions
        crop_width = px_max_x - px_min_x
        crop_height = px_max_y - px_min_y
        
        print(f"Pixel crop window: {px_min_x}, {px_min_y}, {crop_width}, {crop_height}")
        print(f"Cropped dimensions: {crop_width} x {crop_height} pixels")
        
        # Use gdal_translate to crop
        crop_cmd = [
            'gdal_translate',
            '-srcwin', str(px_min_x), str(px_min_y), str(crop_width), str(crop_height),
            '-co', 'COMPRESS=LZW',
            '-co', 'TILED=YES',
            input_dem_path,
            output_dem_path
        ]
        
        print("Running gdal_translate...")
        result = subprocess.run(crop_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: gdal_translate failed: {result.stderr}")
            return False
        
        # Verify output
        crop_ds = gdal.Open(output_dem_path)
        if crop_ds is None:
            print("ERROR: Could not verify cropped output")
            return False
        
        crop_gt = crop_ds.GetGeoTransform()
        actual_width_m = crop_ds.RasterXSize * abs(crop_gt[1])
        actual_height_m = crop_ds.RasterYSize * abs(crop_gt[5])
        
        print(f"SUCCESS: Cropped DEM saved to {output_dem_path}")
        print(f"Cropped area: {actual_width_m/1000:.2f}km x {actual_height_m/1000:.2f}km")
        print(f"Cropped dimensions: {crop_ds.RasterXSize} x {crop_ds.RasterYSize} pixels")
        
        crop_ds = None
        ds = None
        
        return True
        
    except Exception as e:
        print(f"ERROR: Exception in crop_dem_around_center: {str(e)}")
        return False

def main():
    """Main function for shadow map generation with timing and full DEM processing."""
    print("=== Shadow Map Generation Production ===")
    print(f"Target date: {TARGET_DATE}")
    print(f"Production mode: Full Tirol DEM, all time periods")
    print()
    
    # Record overall start time
    overall_start_time = time.time()
    
    # Check if input DEM exists
    if not os.path.exists(DEM_PATH):
        print(f"ERROR: Input DEM not found: {DEM_PATH}")
        sys.exit(1)
    
    # Check if GRASS GIS is installed
    grass_cmd = '/Applications/GRASS-8.4.app/Contents/Resources/bin/grass'
    try:
        result = subprocess.run([grass_cmd, '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR: GRASS GIS not found. Please install GRASS GIS.")
            sys.exit(1)
        print(f"GRASS GIS version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("ERROR: GRASS GIS not found. Please install GRASS GIS.")
        sys.exit(1)
    
    # Ensure output directory exists
    os.makedirs(SHADOW_OUT_DIR, exist_ok=True)
    print(f"Output directory: {SHADOW_OUT_DIR}")
    print()
    
    # Step 1: Crop DEM (COMMENTED OUT - using full DEM for production)
    crop_start_time = time.time()
    # print("=== Step 1: Cropping DEM around center ===")
    # if crop_dem_around_center(DEM_PATH, CROPPED_DEM_PATH, CENTER_LAT, CENTER_LON, 2):
    #     print("SUCCESS: Cropped DEM around center")
    # else:
    #     print("ERROR: Failed to crop DEM")
    #     sys.exit(1)
    
    # For production, use full DEM instead of cropped version
    working_dem_path = DEM_PATH  # Use full DEM
    
    crop_time = time.time() - crop_start_time
    # print(f"Crop time: {crop_time:.2f} seconds")
    print("Using full DEM for production run")
    print()
    
    # Step 2: Set up GRASS environment  
    grass_start_time = time.time()
    print("=== Step 2: Setting up GRASS GIS environment ===")
    gisdb, location, mapset = setup_grass_environment(working_dem_path)
    if gisdb is None:
        print("ERROR: Failed to set up GRASS environment")
        sys.exit(1)
    print(f"GRASS database: {gisdb}")
    grass_setup_time = time.time() - grass_start_time
    print(f"GRASS setup time: {grass_setup_time:.2f} seconds")
    print()
    
    # Step 2.5: Pre-compute horizon map (once for all time periods)
    horizon_start_time = time.time()
    print("=== Step 2.5: Computing horizon map for shadow optimization ===")
    
    grass_cmd = '/Applications/GRASS-8.4.app/Contents/Resources/bin/grass'
    grass_location = f"{location}/PERMANENT"
    
    # Import DEM into GRASS
    import_cmd = [
        grass_cmd, grass_location, '--exec',
        'r.in.gdal', f'input={working_dem_path}', 'output=dem', '--overwrite'
    ]
    
    result = subprocess.run(import_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: Failed to import DEM")
        print(result.stderr)
        sys.exit(1)
    
    # Set computational region
    region_cmd = [
        grass_cmd, grass_location, '--exec',
        'g.region', 'raster=dem'
    ]
    result = subprocess.run(region_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: Failed to set region")
        print(result.stderr)
        sys.exit(1)
    
    # Compute limited-distance horizon map for shadow optimization
    print(f"Computing horizon map (max distance: {MAX_SHADOW_DIST}m, step: {HSTEP}°)...")
    horizon_cmd = [
        grass_cmd, grass_location, '--exec',
        'r.horizon',
        'elevation=dem',
        f'output=hor{HSTEP}',
        f'step={HSTEP}',
        f'maxdistance={MAX_SHADOW_DIST}',
        '--overwrite'
    ]
    
    result = subprocess.run(horizon_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: Failed to compute horizon")
        print(result.stderr)
        sys.exit(1)
    
    horizon_time = time.time() - horizon_start_time
    print(f"Horizon computation time: {horizon_time:.2f} seconds")
    print("SUCCESS: Horizon map computed for all shadow calculations")
    print()
    
    # Step 3: Process shadow map
    shadow_start_time = time.time()
    print("=== Step 3: Generating shadow maps ===")
    
    # Process each time period
    successful = 0
    skipped = 0
    failed = 0
    
    for i, dt in enumerate(SHADOW_TIMES):
        period_num = i + 1
        time_str = dt.strftime('%H:%M')
        
        # Calculate sun angles
        azimuth, altitude = get_sun_angles(dt, CENTER_LAT, CENTER_LON)
        
        print(f"Period {period_num} ({time_str}):")
        print(f"  Sun azimuth: {azimuth:.2f}°")
        print(f"  Sun altitude: {altitude:.2f}°")
        
        # Check if sun is below horizon
        if altitude <= 0:
            print(f"  WARNING: Sun below horizon (altitude={altitude:.2f}°)")
            print(f"  Skipping shadow calculation for this period")
            skipped += 1
            continue
        
        # Define output filename (overwrite existing)
        mode_suffix = '_gradient' if SHADOW_MODE == 'gradient' else ''
        output_filename = f'shadow_5m_period{period_num}_{time_str.replace(":", "")}{mode_suffix}.tif'
        output_path = os.path.join(SHADOW_OUT_DIR, output_filename)
        
        print(f"  Generating shadow map (overwriting if exists)...")
        period_start_time = time.time()
        
        if generate_shadow_map_grass(working_dem_path, output_path, azimuth, altitude,
                                   gisdb, location, mapset, dt, SHADOW_MODE):
            period_time = time.time() - period_start_time
            print(f"  SUCCESS: Saved to {output_filename}")
            print(f"  Processing time: {period_time:.2f} seconds")
            
            # Verify output format
            print(f"  Verifying output format...")
            info = verify_output_format(output_path)
            if info:
                print(f"    Data type: {info['dtype']} ({info['dtype_size']} bits)")
                print(f"    Value range: {info['stats'][0]:.0f} - {info['stats'][1]:.0f}")
                print(f"    Resolution: {info['pixel_width']:.1f}m x {info['pixel_height']:.1f}m")
                print(f"    Projection: EPSG:{info['epsg']}")
            
            successful += 1
        else:
            failed += 1
        
        print()
    
    shadow_time = time.time() - shadow_start_time
    overall_time = time.time() - overall_start_time
    
    # Clean up GRASS database
    try:
        import shutil
        shutil.rmtree(gisdb)
        print("Cleaned up temporary GRASS database")
    except:
        print(f"Warning: Could not clean up GRASS database at {gisdb}")
    
    # Performance summary and estimation
    print("\n=== Performance Summary ===")
    print(f"Total time: {overall_time:.2f} seconds")
    print(f"  - DEM cropping: {crop_time:.2f} seconds")
    print(f"  - GRASS setup: {grass_setup_time:.2f} seconds") 
    print(f"  - Horizon computation: {horizon_time:.2f} seconds")
    print(f"  - Shadow processing: {shadow_time:.2f} seconds")
    print()
    
    if successful > 0:
        time_per_period = shadow_time / successful
        print(f"Time per period (2km x 2km): {time_per_period:.2f} seconds")
        
        # Estimate full processing time
        # Original DEM is much larger - estimate scaling
        crop_info = verify_output_format(CROPPED_DEM_PATH)
        if crop_info:
            crop_pixels = crop_info['width'] * crop_info['height']
            
            # Estimate full DEM size (roughly 100x larger area)
            estimated_full_pixels = crop_pixels * 100  # Very rough estimate
            scaling_factor = estimated_full_pixels / crop_pixels
            
            estimated_time_per_period = time_per_period * scaling_factor
            estimated_total_time = estimated_time_per_period * 4  # 4 time periods
            
            print(f"Estimated scaling factor: {scaling_factor:.1f}x")
            print(f"Estimated time per period (full DEM): {estimated_time_per_period/60:.1f} minutes")
            print(f"Estimated total time (4 periods): {estimated_total_time/3600:.1f} hours")
        print()
    
    # Summary
    print("=== Generation Summary ===")
    print(f"Total time periods: {len(SHADOW_TIMES)}")
    print(f"Successfully generated: {successful}")
    print(f"Skipped (existing): {skipped}")
    print(f"Failed: {failed}")
    print()
    
    print("=== Production Files Created ===")
    print(f"Full DEM used: {DEM_PATH}")
    if successful > 0:
        print(f"Shadow maps: Check {SHADOW_OUT_DIR} for shadow_5m_* files")
    
    return successful > 0

if __name__ == "__main__":
    main()