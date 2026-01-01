import argparse
import os
import math
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import numpy as np
from pathlib import Path
from tqdm import tqdm

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_tiles(input_path, output_dir, target_res=2.5, tile_size=512, base_zoom_level=16):
    """
    Generates tiles from a DEM.
    
    Args:
        input_path (str): Path to the source DEM.
        output_dir (str): output directory root.
        target_res (float): Target resolution in meters (default 2.5m).
        tile_size (int): Pixel size of tiles (default 512).
        base_zoom_level (int): The Zoom level label for the target resolution.
                               Downsampled levels will use base_zoom_level - 1, etc.
    """
    
    with rasterio.open(input_path) as src:
        print(f"Source: {input_path}")
        print(f"Source Res: {src.res}")
        print(f"Source Profile: {src.profile}")
        
        # We need to construct a VRT that resamples to the target resolution
        # calculate the scale factor
        src_res_x = src.res[0]
        # We assume 2.5m is the target. If source is 5m, scale is 2.
        scale_factor = src_res_x / target_res
        
        upscaled_width = int(src.width * scale_factor)
        upscaled_height = int(src.height * scale_factor)
        
        target_transform = src.transform * src.transform.scale(
            (src.width / upscaled_width),
            (src.height / upscaled_height)
        )
        
        vrt_options = {
            'resampling': Resampling.bilinear,
            'transform': target_transform,
            'height': upscaled_height,
            'width': upscaled_width,
            'crs': src.crs # Keep original CRS for now (likely EPSG:31254)
        }
        
        print(f"Target Resolution: {target_res}m")
        print(f"Base Zoom Level: {base_zoom_level}")
        print(f"Generating Levels...")

        # We will generate tiles for the base level and potentially lower levels.
        # But efficiently: usually one generates the base tiles, then downsamples those.
        # Given providing "super high - medium - and super low", let's do 3 levels.
        # Level 0 (Target): 2.5m (User calls this 'even larger 2.5m') -> Zoom 16
        # Level 1: 5m -> Zoom 15
        # Level 2: 10m -> Zoom 14
        
        levels = [
            {'zoom': base_zoom_level, 'res': target_res},
            {'zoom': base_zoom_level - 1, 'res': target_res * 2},
            {'zoom': base_zoom_level - 2, 'res': target_res * 4},
        ]
        
        for level in levels:
            zoom = level['zoom']
            res = level['res']
            print(f"\n--- Processing Zoom {zoom} (Res: {res}m) ---")
            
            # Calculate VRT for this specific level
            lvl_scale = src_res_x / res
            lvl_width = int(src.width * lvl_scale)
            lvl_height = int(src.height * lvl_scale)
            
            lvl_transform = src.transform * src.transform.scale(
                (src.width / lvl_width),
                (src.height / lvl_height)
            )
            
            # Create a VRT for this level
            with WarpedVRT(src, transform=lvl_transform, 
                           width=lvl_width, height=lvl_height, 
                           resampling=Resampling.bilinear) as vrt:
                
                # Calculate grid
                cols = math.ceil(vrt.width / tile_size)
                rows = math.ceil(vrt.height / tile_size)
                
                print(f"Grid: {cols} x {rows} tiles")
                
                zoom_dir = os.path.join(output_dir, str(zoom))
                
                # Iterate tiles
                for col in tqdm(range(cols), desc=f"Zoom {zoom} Columns"):
                    # Create column directory (X)
                    col_dir = os.path.join(zoom_dir, str(col))
                    ensure_dir(col_dir)
                    
                    for row in range(rows):
                        # Define window
                        x_off = col * tile_size
                        y_off = row * tile_size
                        
                        # Calculate intersection with VRT bounds
                        req_window = Window(x_off, y_off, tile_size, tile_size)
                        ds_window = Window(0, 0, vrt.width, vrt.height)
                        read_window = req_window.intersection(ds_window)
                        
                        if read_window.width == 0 or read_window.height == 0:
                            continue

                        # Read data
                        data = vrt.read(1, window=read_window)
                        
                        # Check if tile has valid data (not just nodata)
                        if np.all(data == -9999):
                            continue
                            
                        # Pad if necessary
                        if data.shape != (tile_size, tile_size):
                            full_tile = np.full((tile_size, tile_size), -9999, dtype='float32')
                            
                            # Calculate placement
                            # read_window.row_off is global. 
                            # We need the offset relative to the requested tile top-left (y_off)
                            r_start = int(read_window.row_off - y_off)
                            c_start = int(read_window.col_off - x_off)
                            
                            r_end = r_start + data.shape[0]
                            c_end = c_start + data.shape[1]
                            
                            full_tile[r_start:r_end, c_start:c_end] = data
                            data = full_tile
                        
                        tile_path = os.path.join(col_dir, f"{row}.tif")
                        
                        # Write tile
                        out_meta = vrt.profile.copy()
                        out_meta.update({
                            'driver': 'GTiff',
                            'height': tile_size,
                            'width': tile_size,
                            'transform': vrt.window_transform(req_window), # Use the transform of the full tile
                            'compress': 'lzw',
                            'count': 1,
                            'dtype': 'float32'
                        })
                        
                        with rasterio.open(tile_path, "w", **out_meta) as dest:
                            dest.write(data, 1)

def main():
    parser = argparse.ArgumentParser(description="Generate tiled DEMs from source.")
    parser.add_argument("--input", default="/Users/cole/dev/PowFinder/resources/terrains/DGM_Tirol_5m_epsg31254_2006_2020.tif", help="Path to master DEM")
    parser.add_argument("--output", default="tiles_dem", help="Output directory")
    parser.add_argument("--zoom", type=int, default=16, help="Base Zoom Level (for 2.5m)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        # Try finding it in current dir or known locations
        proposed = os.path.join(os.getcwd(), "resources/terrains", os.path.basename(args.input))
        if os.path.exists(proposed):
            args.input = proposed
            print(f"Found at: {args.input}")
        else:
            return

    ensure_dir(args.output)
    
    generate_tiles(args.input, args.output, base_zoom_level=args.zoom)

if __name__ == "__main__":
    main()
