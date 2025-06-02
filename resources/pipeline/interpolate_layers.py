import json
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import Affine
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
import geopandas
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
from shapely.geometry import Point # For potential coordinate transformations

# --- Configuration ---
WEATHER_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "shortwave_radiation",
    "cloud_cover", "snow_depth", "snowfall", "wind_speed_10m",
    "weather_code", "freezing_level_height", "surface_pressure"
]
TARGET_RESOLUTION_M = 50  # meters
ALTITUDE_THRESHOLD_M = 2300 # meters for high-altitude mask
NODATA_VALUE = -9999.0 # Common nodata value for float rasters

# --- Paths ---
THIS_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = THIS_DIR / "generated_layers"
PREDICTIONS_CSV_PATH = THIS_DIR / "predictions.csv"
# DEM_PATH = THIS_DIR.parent / "terrains" / "dem_25m_wgs84.tif" # Higher res option
DEM_PATH = THIS_DIR.parent / "terrains" / "tirol_100m_web.tif" # Provided option
BOUNDARY_GEOJSON_PATH = THIS_DIR.parent / "meteo_api" / "tirol_peaks.geojson" # Points, needs convex hull

def create_target_grid_profile(dem_profile, target_resolution):
    """Creates a rasterio profile for the target grid based on DEM extent and new resolution."""
    dem_affine = dem_profile['transform']
    dem_width = dem_profile['width']
    dem_height = dem_profile['height']

    # Calculate new width and height based on target resolution
    # Assumes original DEM pixels are square
    scale_x = dem_affine.a / target_resolution
    scale_y = -dem_affine.e / target_resolution # dem_affine.e is negative

    new_width = int(dem_width * scale_x)
    new_height = int(dem_height * scale_y)

    # New affine: top-left corner remains same, pixel size changes
    new_affine = Affine(target_resolution, dem_affine.b, dem_affine.c,
                        dem_affine.d, -target_resolution, dem_affine.f)

    target_profile = dem_profile.copy()
    target_profile.update({
        'transform': new_affine,
        'width': new_width,
        'height': new_height,
        'nodata': NODATA_VALUE
    })
    return target_profile, new_width, new_height, new_affine

def create_masks(dem_path, boundary_geojson_path, target_profile, altitude_threshold):
    """Creates high-altitude and boundary masks on the target grid."""
    high_alt_mask = None
    boundary_mask = None

    # High-altitude mask
    try:
        with rasterio.open(dem_path) as dem_src:
            dem_data = dem_src.read(1)

            # Create mask based on threshold
            alt_mask_orig_res = dem_data > altitude_threshold

            # Resample this mask to target_profile resolution/shape
            high_alt_mask = np.empty((target_profile['height'], target_profile['width']), dtype=np.bool_)
            reproject(
                source=alt_mask_orig_res.astype(rasterio.uint8), # Needs to be numeric for reproject
                destination=high_alt_mask,
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                dst_transform=target_profile['transform'],
                dst_crs=target_profile['crs'],
                resampling=Resampling.nearest
            )
            high_alt_mask = high_alt_mask.astype(bool)
            print(f"Successfully created high-altitude mask (>{altitude_threshold}m).")

    except Exception as e:
        print(f"Warning: Could not create high-altitude mask from {dem_path}: {e}", file=sys.stderr)

    # Boundary mask (simplified: convex hull of peaks)
    try:
        gdf = geopandas.read_file(boundary_geojson_path)
        if gdf.crs != target_profile['crs']:
             print(f"Warning: Boundary GeoJSON CRS ({gdf.crs}) differs from target CRS ({target_profile['crs']}). Attempting to transform.", file=sys.stderr)
             gdf = gdf.to_crs(target_profile['crs'])

        # Create a convex hull of all points to form a boundary polygon
        # Ensure there are enough points to form a polygon
        if not gdf.empty and len(gdf) >= 3 :
            boundary_shape = [gdf.unary_union.convex_hull] # unary_union to handle multi-part geometries first

            boundary_mask_arr = rasterize(
                shapes=boundary_shape,
                out_shape=(target_profile['height'], target_profile['width']),
                transform=target_profile['transform'],
                fill=0,  # outside polygon
                default_value=1,  # inside polygon
                dtype=np.uint8
            )
            boundary_mask = boundary_mask_arr.astype(bool)
            print(f"Successfully created boundary mask from {boundary_geojson_path}.")
        else:
            print(f"Warning: Not enough points in {boundary_geojson_path} to form a convex hull. Skipping boundary mask.", file=sys.stderr)

    except Exception as e:
        print(f"Warning: Could not create boundary mask from {boundary_geojson_path}: {e}", file=sys.stderr)

    # Combine masks
    if high_alt_mask is not None and boundary_mask is not None:
        final_mask = np.logical_and(~high_alt_mask, boundary_mask) # Valid areas are NOT high AND inside boundary
    elif boundary_mask is not None: # Only boundary mask
        final_mask = boundary_mask
    elif high_alt_mask is not None: # Only altitude mask (inverted, so valid areas are NOT high)
        final_mask = ~high_alt_mask
    else:
        final_mask = np.ones((target_profile['height'], target_profile['width']), dtype=bool) # All valid
        print("Warning: No masks created. Interpolation will cover the full extent.", file=sys.stderr)

    return final_mask


def process_variable_timeslice(var_name, current_time_slice_data, dem_profile, target_grid_profile, combined_mask):
    """Interpolates data for a single variable and time slice, applies mask, saves GeoTIFF."""
    print(f"Processing: Variable '{var_name}', Time Slice '{current_time_slice_data['time'].iloc[0]}'")

    points_lat = current_time_slice_data['lat'].values
    points_lon = current_time_slice_data['lon'].values
    values = current_time_slice_data[var_name].values

    # Ensure no NaNs in coordinates or values for interpolation
    valid_data_idx = ~np.isnan(points_lat) & ~np.isnan(points_lon) & ~np.isnan(values)
    points_lat = points_lat[valid_data_idx]
    points_lon = points_lon[valid_data_idx]
    values = values[valid_data_idx]

    if len(values) < 3: # griddata needs at least 3 points for non-nearest methods
        print(f"Warning: Less than 3 valid data points for {var_name} at this time slice. Skipping interpolation.", file=sys.stderr)
        return

    # TODO: CRS Transformation of prediction points (lon, lat) to DEM CRS
    # Assuming predictions are WGS84 (EPSG:4326) for now.
    # If dem_profile['crs'] is different, transformation is needed here.
    # For simplicity, this example proceeds assuming CRSs are compatible or transformation is handled upstream.
    # Example using geopandas for transformation (if points were in a GeoDataFrame):
    # gdf_points = geopandas.GeoDataFrame(current_time_slice_data, geometry=geopandas.points_from_xy(current_time_slice_data.lon, current_time_slice_data.lat), crs="EPSG:4326")
    # if gdf_points.crs != dem_profile['crs']:
    #     gdf_points = gdf_points.to_crs(dem_profile['crs'])
    # points_x = gdf_points.geometry.x.values
    # points_y = gdf_points.geometry.y.values
    # For now, using lon/lat directly as x/y, assuming they are in target grid's CRS units:
    points_x = points_lon
    points_y = points_lat


    # Create target grid coordinates
    tgt_width = target_grid_profile['width']
    tgt_height = target_grid_profile['height']
    tgt_affine = target_grid_profile['transform']

    # Grid cell centers
    cols = np.arange(tgt_width)
    rows = np.arange(tgt_height)
    grid_x_centers, grid_y_centers = rasterio.transform.xy(tgt_affine, rows, cols, offset='center')

    # Flatten for griddata
    grid_x_flat = grid_x_centers.flatten()
    grid_y_flat = grid_y_centers.flatten()

    print(f"  Interpolating {len(values)} points onto a {tgt_width}x{tgt_height} grid...")
    try:
        interpolated_flat = griddata(
            (points_x, points_y), values, (grid_x_flat, grid_y_flat),
            method='cubic', fill_value=NODATA_VALUE
        )
        # If 'cubic' fails due to triangulation, try 'linear' or 'nearest'
    except Exception as e:
        print(f"    Cubic interpolation failed ({e}), trying linear.", file=sys.stderr)
        try:
            interpolated_flat = griddata(
                (points_x, points_y), values, (grid_x_flat, grid_y_flat),
                method='linear', fill_value=NODATA_VALUE
            )
        except Exception as e_lin:
            print(f"    Linear interpolation also failed ({e_lin}), using nearest.", file=sys.stderr)
            interpolated_flat = griddata(
                (points_x, points_y), values, (grid_x_flat, grid_y_flat),
                method='nearest' # fill_value not used for 'nearest' but good to have default
            )
            # For nearest, ensure NODATA_VALUE is applied where there's no nearest point (should be rare)
            # This is more complex, usually griddata handles this by assigning nearest value.

    interpolated_grid = interpolated_flat.reshape((tgt_height, tgt_width))

    # Apply combined mask
    interpolated_grid[~combined_mask] = NODATA_VALUE

    # Write GeoTIFF
    time_str = pd.to_datetime(current_time_slice_data['time'].iloc[0]).strftime('%Y%m%dT%H%M%S')
    output_filename = OUTPUT_DIR / f"{var_name}_{time_str}.tif"

    out_profile = target_grid_profile.copy()
    out_profile['dtype'] = interpolated_grid.dtype
    out_profile['nodata'] = NODATA_VALUE

    try:
        with rasterio.open(output_filename, 'w', **out_profile) as dst:
            dst.write(interpolated_grid.astype(out_profile['dtype']), 1)
        print(f"  Successfully wrote GeoTIFF: {output_filename}")
    except Exception as e:
        print(f"  Error writing GeoTIFF {output_filename}: {e}", file=sys.stderr)

    # Optional: Generate PNG
    try:
        plt.figure(figsize=(10, 8))
        # Create a masked array for plotting to handle nodata correctly
        masked_array = np.ma.masked_where(interpolated_grid == NODATA_VALUE, interpolated_grid)
        plt.imshow(masked_array, cmap='viridis') # Choose a suitable colormap
        plt.colorbar(label=var_name)
        plt.title(f"{var_name} at {time_str}")
        plt.xlabel("Pixel Column")
        plt.ylabel("Pixel Row")
        png_filename = OUTPUT_DIR / f"{var_name}_{time_str}.png"
        plt.savefig(png_filename)
        plt.close()
        print(f"  Successfully wrote PNG: {png_filename}")
    except Exception as e:
        print(f"  Error generating PNG for {var_name} at {time_str}: {e}", file=sys.stderr)


def main():
    print("Starting raster layer interpolation...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load predictions
    try:
        predictions_df = pd.read_csv(PREDICTIONS_CSV_PATH)
        if predictions_df.empty:
            print(f"Predictions file {PREDICTIONS_CSV_PATH} is empty. Exiting.", file=sys.stderr)
            return
    except FileNotFoundError:
        print(f"Error: Predictions file not found at {PREDICTIONS_CSV_PATH}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading predictions CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Get DEM profile (used as basis for target grid)
    try:
        with rasterio.open(DEM_PATH) as dem_src:
            dem_profile_orig = dem_src.profile
            print(f"Using DEM: {DEM_PATH} with CRS {dem_profile_orig['crs']}")
    except Exception as e:
        print(f"Critical Error: Could not open DEM file {DEM_PATH}: {e}", file=sys.stderr)
        sys.exit(1)

    # Create target grid profile (50m resolution)
    target_grid_profile, _, _, _ = create_target_grid_profile(dem_profile_orig, TARGET_RESOLUTION_M)
    print(f"Target grid: {target_grid_profile['width']}x{target_grid_profile['height']} @ {TARGET_RESOLUTION_M}m, CRS {target_grid_profile['crs']}")

    # Create masks
    print("Creating masks...")
    combined_mask = create_masks(DEM_PATH, BOUNDARY_GEOJSON_PATH, target_grid_profile, ALTITUDE_THRESHOLD_M)

    # Process unique time slices (start with the first one for this subtask)
    unique_times = predictions_df['time'].unique()
    if not unique_times.any():
        print("No time slices found in predictions. Exiting.", file=sys.stderr)
        return

    # --- FOCUS ON FIRST TIME SLICE AND ONE VARIABLE FOR INITIAL IMPLEMENTATION ---
    time_to_process = unique_times[0]
    print(f"\nProcessing for time slice: {time_to_process}")
    current_time_slice_data_full = predictions_df[predictions_df['time'] == time_to_process]

    if current_time_slice_data_full.empty:
        print(f"No data for time slice {time_to_process}. Skipping.", file=sys.stderr)
        return

    # Process one variable first (e.g., temperature_2m)
    # var_to_process = "temperature_2m"
    # if var_to_process in WEATHER_VARIABLES and var_to_process in current_time_slice_data_full.columns:
    #     process_variable_timeslice(var_to_process, current_time_slice_data_full, dem_profile_orig, target_grid_profile, combined_mask)
    # else:
    #    print(f"Variable {var_to_process} not found in data or WEATHER_VARIABLES list.", file=sys.stderr)

    # Loop through all defined weather variables for the selected time slice
    for var_name in WEATHER_VARIABLES:
        if var_name in current_time_slice_data_full.columns:
            process_variable_timeslice(var_name, current_time_slice_data_full[['lat', 'lon', 'time', var_name]], dem_profile_orig, target_grid_profile, combined_mask)
        else:
            print(f"Warning: Variable column '{var_name}' not found in predictions.csv. Skipping.", file=sys.stderr)

    # To process all time slices and all variables, uncomment and adapt:
    # for time_val in unique_times:
    #     print(f"\nProcessing for time slice: {time_val}")
    #     current_time_slice_data_full = predictions_df[predictions_df['time'] == time_val]
    #     if current_time_slice_data_full.empty:
    #         print(f"No data for time slice {time_val}. Skipping.", file=sys.stderr)
    #         continue
    #     for var_name in WEATHER_VARIABLES:
    #         if var_name in current_time_slice_data_full.columns:
    #             process_variable_timeslice(var_name, current_time_slice_data_full[['lat', 'lon', 'time', var_name]], dem_profile_orig, target_grid_profile, combined_mask)
    #         else:
    #             print(f"Warning: Variable column '{var_name}' not found in predictions.csv for time {time_val}. Skipping.", file=sys.stderr)

    print("\nFinished raster layer interpolation.")

if __name__ == "__main__":
    # Dummy files for testing if main inputs are missing
    if not PREDICTIONS_CSV_PATH.exists():
        print(f"Creating dummy {PREDICTIONS_CSV_PATH.name} for local testing.")
        header = ["lat", "lon", "time", "task_type"] + WEATHER_VARIABLES
        dummy_data = []
        times = ["2025-05-23T10:30:00", "2025-05-23T13:30:00"]
        for t in times:
            for i in range(10): # 10 points per time slice
                row = [47.0 + i*0.1, 11.0 + i*0.1, t, "grid_forecast"]
                row.extend([10+i*0.5+j for j in range(len(WEATHER_VARIABLES))]) # Dummy values
                dummy_data.append(row)
        pd.DataFrame(dummy_data, columns=header).to_csv(PREDICTIONS_CSV_PATH, index=False)

    # A proper DEM and boundary file are harder to create as dummies.
    # The script will likely fail if they are not present and correctly formatted.
    # For DEM_PATH, ensure it points to a valid GeoTIFF.
    # For BOUNDARY_GEOJSON_PATH, ensure it points to a valid GeoJSON.
    # This script assumes rasterio, geopandas, etc. are installed.

    main()
