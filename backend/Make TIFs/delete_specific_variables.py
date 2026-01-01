#!/usr/bin/env python3
"""
delete_specific_variables.py
----------------------------
Utility script to delete TIF and PNG files for specific variables.
Useful for re-running pipeline for specific variables without regenerating everything.

Usage:
    python delete_specific_variables.py [--tifs-only] [--pngs-only]
    
    No flags: Delete both TIFs and PNGs
    --tifs-only: Delete only TIF files (useful for color scale testing)
    --pngs-only: Delete only PNG files (useful for render testing)
"""

import argparse
import os
import pathlib
import shutil

# Get the directory containing this script
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()

# Define paths relative to script location
TIFS_DIR = SCRIPT_DIR / ".." / ".." / "TIFS" / "100m_resolution"

# Variables to delete TIFs for (comment out ones you want to KEEP)
TIFS_TO_DELETE = [
    # Weather variables
    # "temperature_2m",
    # "dewpoint_2m", 
    # "relative_humidity_2m",
    # "surface_pressure",
    # "cloud_cover",
    # "shortwave_radiation",
    # "snowfall",
    # "freezing_level_height",
    # "wind_speed_10m",
    # "weather_code",
    # "snow_depth",
    
    # Terrain variables
    # "elevation",
    # "aspect", 
    # "slope",
    
    # Derived variables - UNCOMMENTED = WILL BE DELETED
    # "powder_depth",
    # "powder_quality", 
    # "sqh",
    # "skiability",
]

# Variables to delete PNGs for (comment out ones you want to KEEP)
PNGS_TO_DELETE = [
    # Weather variables
    # "temperature_2m",
    # "dewpoint_2m", 
    # "relative_humidity_2m",
    # "surface_pressure",
    # "cloud_cover",
    # "shortwave_radiation",
    # "snowfall",
    # "freezing_level_height",
    # "wind_speed_10m",
    # "weather_code",
    # "snow_depth",
    
    # Terrain variables
    # "elevation",
    # "aspect", 
    # "slope",
    
    # Derived variables - UNCOMMENTED = WILL BE DELETED
    # "powder_depth",
    # "powder_quality", 
    "sqh",
    "skiability",
]

def delete_tif_files(variables_list, base_tif_dir=None):
    """Delete TIF files for specified variables."""
    if base_tif_dir is None:
        base_tif_dir = TIFS_DIR
    else:
        base_tif_dir = pathlib.Path(base_tif_dir)
    
    base_path = base_tif_dir.resolve()
    
    if not base_path.exists():
        print(f"TIF directory {base_path} does not exist")
        return 0, 0
    
    deleted_count = 0
    total_timestamps = 0
    
    print("🗑️  Deleting TIF files...")
    
    for timestamp_dir in sorted(base_path.iterdir()):
        if not timestamp_dir.is_dir():
            continue
            
        total_timestamps += 1
        
        for variable in variables_list:
            tif_path = timestamp_dir / f"{variable}.tif"
            if tif_path.exists():
                try:
                    tif_path.unlink()
                    deleted_count += 1
                    print(f"  ✓ Deleted {timestamp_dir.name}/{variable}.tif")
                except Exception as e:
                    print(f"  ✗ Failed to delete {tif_path}: {e}")
    
    print(f"\n📊 TIF Summary:")
    print(f"  Variables to delete: {len(variables_list)}")
    print(f"  Timestamps processed: {total_timestamps}")
    print(f"  TIF files deleted: {deleted_count}")
    
    return deleted_count, total_timestamps


def delete_png_tiles(variables_list, base_tif_dir=None):
    """Delete PNG files for specified variables from TIF timestamp directories."""
    if base_tif_dir is None:
        base_tif_dir = TIFS_DIR
    else:
        base_tif_dir = pathlib.Path(base_tif_dir)
    
    base_path = base_tif_dir.resolve()
    
    if not base_path.exists():
        print(f"TIF directory {base_path} does not exist")
        return 0, 0
    
    deleted_count = 0
    total_timestamps = 0
    
    print("\n🗑️  Deleting PNG files...")
    
    for timestamp_dir in sorted(base_path.iterdir()):
        if not timestamp_dir.is_dir() or timestamp_dir.name == "terrainPNGs":
            continue
            
        total_timestamps += 1
        
        for variable in variables_list:
            png_path = timestamp_dir / f"{variable}.png"
            if png_path.exists():
                try:
                    png_path.unlink()
                    deleted_count += 1
                    print(f"  ✓ Deleted {timestamp_dir.name}/{variable}.png")
                except Exception as e:
                    print(f"  ✗ Failed to delete {png_path}: {e}")
    
    print(f"\n📊 PNG Summary:")
    print(f"  Variables to delete: {len(variables_list)}")
    print(f"  Timestamps processed: {total_timestamps}")
    print(f"  PNG files deleted: {deleted_count}")
    
    return deleted_count, total_timestamps


def main():
    parser = argparse.ArgumentParser(description="Delete TIF and/or PNG files for specific variables")
    parser.add_argument("--tifs-only", action="store_true", help="Delete only TIF files")
    parser.add_argument("--pngs-only", action="store_true", help="Delete only PNG files")
    args = parser.parse_args()
    
    # Determine what to delete
    delete_tifs = not args.pngs_only
    delete_pngs = not args.tifs_only
    
    print("=" * 60)
    print("🧹 VARIABLE CLEANUP UTILITY")
    print("=" * 60)
    
    if delete_tifs and delete_pngs:
        print(f"TIFs to delete: {', '.join(TIFS_TO_DELETE)}")
        print(f"PNGs to delete: {', '.join(PNGS_TO_DELETE)}")
    elif delete_tifs:
        print(f"TIFs to delete: {', '.join(TIFS_TO_DELETE)}")
    elif delete_pngs:
        print(f"PNGs to delete: {', '.join(PNGS_TO_DELETE)}")
    
    print()
    
    # Confirm deletion
    response = input("⚠️  Are you sure you want to delete these files? (y/N): ")
    if response.lower() != 'y':
        print("❌ Deletion cancelled.")
        return
    
    total_tif_deleted = 0
    total_png_deleted = 0
    
    # Delete TIF files
    if delete_tifs:
        tif_deleted, _ = delete_tif_files(TIFS_TO_DELETE)
        total_tif_deleted = tif_deleted
    
    # Delete PNG tiles
    if delete_pngs:
        png_deleted, _ = delete_png_tiles(PNGS_TO_DELETE)
        total_png_deleted = png_deleted
    
    print("\n" + "=" * 60)
    print("✅ CLEANUP COMPLETE")
    print("=" * 60)
    if delete_tifs and delete_pngs:
        print(f"Deleted {total_tif_deleted} TIF files and {total_png_deleted} PNG directories")
    elif delete_tifs:
        print(f"Deleted {total_tif_deleted} TIF files")
    elif delete_pngs:
        print(f"Deleted {total_png_deleted} PNG directories")
    
    print("You can now re-run the pipeline:")
    if delete_tifs:
        print("  python generate_tifs.py")
    if delete_pngs:
        print("  python render_pngs.py")


if __name__ == "__main__":
    main()
