#!/usr/bin/env python3

"""
Random Coordinate Generator for Weather Sampling
===============================================

Generates random lat/lon coordinates within Tirol's DEM boundaries for weather data collection.
These points will be used alongside peak locations to create a comprehensive weather dataset
for testing extrapolation algorithms.

The script:
1. Reads T    print(f"\n🎉 Random coordinate generation complete!")
    print(f"📁 Output file: {OUTPUT_FILE}")
    print(f"🌐 Ready for weather API integration") DEM boundaries and resolution
2. Generates random coordinates within the bounding box
3. Validates points against elevation and proximity constraints
4. Snaps valid coordinates to exact DEM grid points
5. Outputs results to all_points.json with validation flagging

Usage:
    python generate_random_coordinates.py

Requirements:
    pip install rasterio shapely numpy
"""

import numpy as np
import rasterio
from rasterio.transform import xy
import json
import random
from math import sqrt, degrees, radians, cos, sin, atan2
import sys
from pathlib import Path

# =====================================
# CONFIGURATION PARAMETERS
# =====================================

# DEM file to use for boundary detection and elevation validation
DEM_FILE = "../terrains/tirol_5m_float.tif"

# Minimum elevation threshold (meters) - focus on higher altitude areas
MIN_ELEVATION = 2300.0

# Minimum distance between points (meters) to avoid redundancy
MIN_DISTANCE_M = 250.0

# Number of random points to generate
TARGET_POINTS = 2000

# Number of points to reserve for validation (will be flagged is_validation=True)
VALIDATION_POINTS = 200

# Random seed for reproducibility (because memes)
RANDOM_SEED = 42069

# Output file for generated coordinates
OUTPUT_FILE = "all_points.json"

# Random seed for reproducible results (set to None for true randomness)
RANDOM_SEED = 42

# =====================================
# UTILITY FUNCTIONS
# =====================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth.
    Returns distance in meters.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # Radius of Earth in meters
    R = 6371000
    return R * c

def snap_to_grid(lat, lon, transform, shape, crs):
    """
    Snap a lat/lon coordinate to the nearest DEM grid point.
    Returns the snapped coordinates in lat/lon and grid indices.
    """
    from rasterio.warp import transform as rasterio_transform
    
    # Convert lat/lon (WGS84) to DEM projection
    proj_x, proj_y = rasterio_transform('EPSG:4326', crs, [lon], [lat])
    proj_x, proj_y = proj_x[0], proj_y[0]
    
    # Convert projected coords to raster coordinates
    col, row = ~transform * (proj_x, proj_y)
    
    # Round to nearest integer (snap to grid)
    col_int = int(round(col))
    row_int = int(round(row))
    
    # Ensure we're within bounds
    col_int = max(0, min(shape[1] - 1, col_int))
    row_int = max(0, min(shape[0] - 1, row_int))
    
    # Convert back to projected coordinates
    snapped_proj_x, snapped_proj_y = transform * (col_int, row_int)
    
    # Convert back to lat/lon (WGS84)
    snapped_lon, snapped_lat = rasterio_transform(crs, 'EPSG:4326', [snapped_proj_x], [snapped_proj_y])
    snapped_lon, snapped_lat = snapped_lon[0], snapped_lat[0]
    
    return snapped_lat, snapped_lon, row_int, col_int

def is_too_close(lat, lon, existing_points, min_distance_m):
    """
    Check if a point is too close to any existing points.
    """
    for existing_lat, existing_lon in existing_points:
        distance = haversine_distance(lat, lon, existing_lat, existing_lon)
        if distance < min_distance_m:
            return True
    return False

class RandomCoordinateGenerator:
    def __init__(self, dem_file, min_elevation, min_distance_m, target_points, random_seed=None):
        """Initialize the coordinate generator with DEM data."""
        self.dem_file = dem_file
        self.min_elevation = min_elevation
        self.min_distance_m = min_distance_m
        self.target_points = target_points
        
        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Load DEM data
        print(f"📍 Loading DEM: {dem_file}")
        try:
            with rasterio.open(dem_file) as src:
                self.dem_data = src.read(1)  # Read first band
                self.transform = src.transform
                self.crs = src.crs
                self.bounds = src.bounds
                self.shape = self.dem_data.shape
                
            # Convert projected bounds to lat/lon bounds for random generation
            from rasterio.warp import transform as rasterio_transform
            
            # Get corner coordinates in projected system
            left, bottom, right, top = self.bounds.left, self.bounds.bottom, self.bounds.right, self.bounds.top
            
            # Transform corners to lat/lon
            lons, lats = rasterio_transform(self.crs, 'EPSG:4326', [left, right, left, right], [bottom, bottom, top, top])
            
            # Find actual lat/lon bounds
            self.lon_min = min(lons)
            self.lon_max = max(lons)
            self.lat_min = min(lats)
            self.lat_max = max(lats)
            
            # Get valid data mask (not NaN)
            self.valid_mask = ~np.isnan(self.dem_data)
            
            print(f"   ✅ DEM loaded: {self.shape[0]}x{self.shape[1]} pixels")
            print(f"   📏 Bounds: {self.bounds}")
            print(f"   🌍 Lat/Lon bounds: {self.lat_min:.4f} to {self.lat_max:.4f} lat, {self.lon_min:.4f} to {self.lon_max:.4f} lon")
            print(f"   🗻 Elevation range: {np.nanmin(self.dem_data):.1f}m to {np.nanmax(self.dem_data):.1f}m")
            print(f"   ✅ Valid pixels: {np.sum(self.valid_mask):,} / {self.dem_data.size:,}")
            
        except Exception as e:
            print(f"❌ Error loading DEM: {e}")
            sys.exit(1)
    
    def generate_random_point(self):
        """Generate a random lat/lon point within DEM bounds."""
        # Generate random coordinates within lat/lon bounding box
        lon = random.uniform(self.lon_min, self.lon_max)
        lat = random.uniform(self.lat_min, self.lat_max)
        return lat, lon
    
    def get_elevation_at_point(self, lat, lon):
        """Get elevation at a specific lat/lon coordinate."""
        try:
            from rasterio.warp import transform as rasterio_transform
            
            # Convert lat/lon to projected coordinates
            proj_x, proj_y = rasterio_transform('EPSG:4326', self.crs, [lon], [lat])
            proj_x, proj_y = proj_x[0], proj_y[0]
            
            # Convert to raster coordinates
            col, row = ~self.transform * (proj_x, proj_y)
            col_int = int(round(col))
            row_int = int(round(row))
            
            # Check if within bounds
            if (0 <= row_int < self.shape[0] and 0 <= col_int < self.shape[1]):
                elevation = self.dem_data[row_int, col_int]
                return elevation if not np.isnan(elevation) else None
            else:
                return None
        except Exception:
            return None
    
    def is_valid_point(self, lat, lon, existing_points):
        """
        Check if a point meets all validation criteria:
        1. Has valid elevation (inside Tirol)
        2. Above minimum elevation threshold
        3. Not too close to existing points
        """
        # Check if point has valid elevation
        elevation = self.get_elevation_at_point(lat, lon)
        if elevation is None:
            return False, "outside_bounds"
        
        # Check elevation threshold
        if elevation < self.min_elevation:
            return False, f"too_low_{elevation:.1f}m"
        
        # Check distance to existing points
        if is_too_close(lat, lon, existing_points, self.min_distance_m):
            return False, "too_close"
        
        return True, f"valid_{elevation:.1f}m"
    
    def generate_coordinates(self):
        """Generate the target number of valid random coordinates."""
        print(f"\n🎯 Generating {self.target_points} random coordinates...")
        print(f"   📏 Min elevation: {self.min_elevation}m")
        print(f"   📐 Min distance: {self.min_distance_m}m")
        
        valid_points = []
        existing_coords = []  # For distance checking
        
        # Statistics
        attempts = 0
        rejection_stats = {
            "outside_bounds": 0,
            "too_low": 0,
            "too_close": 0
        }
        
        # Generate points
        while len(valid_points) < self.target_points:
            attempts += 1
            
            # Generate random point
            lat, lon = self.generate_random_point()
            
            # Validate point
            is_valid, reason = self.is_valid_point(lat, lon, existing_coords)
            
            if is_valid:
                # Snap to DEM grid
                snapped_lat, snapped_lon, row, col = snap_to_grid(lat, lon, self.transform, self.shape, self.crs)
                elevation = self.dem_data[row, col]
                
                # Create point record
                point = {
                    "id": len(valid_points) + 1,
                    "latitude": round(snapped_lat, 6),
                    "longitude": round(snapped_lon, 6),
                    "elevation": round(float(elevation), 1),
                    "grid_row": int(row),
                    "grid_col": int(col),
                    "original_lat": round(lat, 6),
                    "original_lon": round(lon, 6),
                    "type": "random",
                    "is_validation": len(valid_points) < VALIDATION_POINTS
                }
                
                valid_points.append(point)
                existing_coords.append((snapped_lat, snapped_lon))
                
                # Progress update
                if len(valid_points) % 100 == 0:
                    success_rate = len(valid_points) / attempts * 100
                    print(f"   ✅ Generated {len(valid_points)}/{self.target_points} points "
                          f"(success rate: {success_rate:.1f}%)")
            
            else:
                # Track rejection reasons
                if reason.startswith("too_low"):
                    rejection_stats["too_low"] += 1
                elif reason == "outside_bounds":
                    rejection_stats["outside_bounds"] += 1
                elif reason == "too_close":
                    rejection_stats["too_close"] += 1
            
            # Safety check to avoid infinite loops
            if attempts > self.target_points * 100:
                print(f"⚠️  Reached maximum attempts ({attempts}). Generated {len(valid_points)} points.")
                break
        
        # Final statistics
        success_rate = len(valid_points) / attempts * 100 if attempts > 0 else 0
        print(f"\n📊 Generation Statistics:")
        print(f"   ✅ Valid points: {len(valid_points)}")
        print(f"   🎯 Total attempts: {attempts}")
        print(f"   📈 Success rate: {success_rate:.1f}%")
        print(f"   📊 Rejections:")
        print(f"      🗺️  Outside bounds: {rejection_stats['outside_bounds']}")
        print(f"      ⛰️  Below {self.min_elevation}m: {rejection_stats['too_low']}")
        print(f"      📐 Too close (<{self.min_distance_m}m): {rejection_stats['too_close']}")
        
        return valid_points
    
    def save_coordinates(self, points, output_file):
        """Save coordinates to JSON file."""
        print(f"\n💾 Saving coordinates to {output_file}...")
        
        # Create output data structure
        output_data = {
            "metadata": {
                "generated_at": np.datetime64('now').astype(str),
                "dem_file": self.dem_file,
                "min_elevation_m": self.min_elevation,
                "min_distance_m": self.min_distance_m,
                "target_points": self.target_points,
                "actual_points": len(points),
                "dem_bounds": {
                    "left": self.bounds.left,
                    "right": self.bounds.right,
                    "top": self.bounds.top,
                    "bottom": self.bounds.bottom
                },
                "dem_shape": list(self.shape),
                "crs": str(self.crs)
            },
            "coordinates": points
        }
        
        # Save to file
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"   ✅ Saved {len(points)} coordinates to {output_file}")
            
            # Summary statistics
            elevations = [p["elevation"] for p in points]
            print(f"   📊 Elevation range: {min(elevations):.1f}m to {max(elevations):.1f}m")
            print(f"   📊 Average elevation: {np.mean(elevations):.1f}m")
            
        except Exception as e:
            print(f"   ❌ Error saving file: {e}")

def main():
    """Main execution function."""
    print("🗻 PowFinder Random Coordinate Generator")
    print("=" * 50)
    
    # Check if DEM file exists
    dem_path = Path(DEM_FILE)
    if not dem_path.exists():
        print(f"❌ DEM file not found: {DEM_FILE}")
        print("   Please ensure the Tirol DEM is available at the specified path.")
        sys.exit(1)
    
    # Initialize generator
    generator = RandomCoordinateGenerator(
        dem_file=DEM_FILE,
        min_elevation=MIN_ELEVATION,
        min_distance_m=MIN_DISTANCE_M,
        target_points=TARGET_POINTS,
        random_seed=RANDOM_SEED
    )
    
    # Generate coordinates
    coordinates = generator.generate_coordinates()
    
    # Save results
    generator.save_coordinates(coordinates, OUTPUT_FILE)
    
    print(f"\n🎉 Random coordinate generation complete!")
    print(f"📁 Output file: {OUTPUT_FILE}")
    print(f"�️  Includes validation flagging (replaces flag_validation_points.py step)")
    print(f"�🌐 Ready for progressive_grid_scheduler.py pipeline integration")

if __name__ == "__main__":
    main()
