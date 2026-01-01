import os
import argparse
import numpy as np
import rasterio
from data_utils import compute_slope_aspect # Placeholder or inline
from PIL import Image
from tqdm import tqdm

def process_tile(dem_path, sat_path, output_path, resolution=2.5):
    """
    Generates a binary hex chunk from a DEM tile and Satellite Image.
    """
    # Load DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform
        # dem shape: (rows, cols)

    # Load Satellite
    # If sat_path provided and exists, use it. Else white.
    if sat_path and os.path.exists(sat_path):
        try:
            sat_img = Image.open(sat_path).convert('RGB')
            # Resize sat to match DEM if needed (naive)
            if sat_img.size != (dem.shape[1], dem.shape[0]):
                sat_img = sat_img.resize((dem.shape[1], dem.shape[0]))
            sat = np.array(sat_img)
        except Exception as e:
            print(f"Error loading sat {sat_path}: {e}")
            sat = np.full((dem.shape[0], dem.shape[1], 3), 255, dtype=np.uint8)
    else:
        # Default white
        sat = np.full((dem.shape[0], dem.shape[1], 3), 255, dtype=np.uint8)

    rows, cols = dem.shape
    
    # Calculate Slope/Aspect for the whole tile
    # Using simple gradient (central difference)
    # np.gradient returns (dy, dx)
    # resolution is 2.5m
    dy, dx = np.gradient(dem, resolution)
    
    # Slope (degrees)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    
    # Aspect (degrees, 0=North, 90=East)
    # aspect = arctan2(-dx, dy) ? Standard definitions vary.
    # PyVista tool creates 'true_slope_deg' only.
    # We'll calculate standard aspect.
    aspect_rad = np.arctan2(-dx, dy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = (aspect_deg + 360) % 360

    hex_data = []
    
    # Sampling Logic
    # 2.5m Resolution. Hex centers are 5m apart in X (approx).
    # Rows: Staggered.
    # Even Rows (0, 2, 4...) -> Cols (0, 2, 4...)
    # Odd Rows (1, 3, 5...)  -> Cols (1, 3, 5...)
    
    # Validate with User Logic: 
    # "Even Columns: Center aligns... Odd Columns: Center is staggered"
    # "Upscale... to 2.5m... ensures every hex center... snaps to a discrete data point"
    
    for r in range(rows):
        # Determine starting column
        # If r is even (0, 2), start at 0, step 2.
        # If r is odd (1, 3), start at 1, step 2.
        start_c = 1 if (r % 2 == 1) else 0
        
        for c in range(start_c, cols, 2):
            z = dem[r, c]
            if z == -9999 or np.isnan(z): 
                continue
            
            # Local coordinates within the tile (meters)
            x_local = c * resolution
            y_local = r * resolution
            
            # Color (normalized 0-1)
            r_val, g_val, b_val = sat[r, c] / 255.0
            
            s_val = slope_deg[r, c]
            a_val = aspect_deg[r, c]
            
            # Record: x, y, z, r, g, b, slope, aspect
            hex_data.append([x_local, y_local, z, r_val, g_val, b_val, s_val, a_val])
            
    # Convert to float32 buffer
    if not hex_data:
        return
        
    arr = np.array(hex_data, dtype=np.float32)
    
    # Write to binary file
    with open(output_path, 'wb') as f:
        f.write(arr.tobytes())

def batch_process(dem_root, sat_root, out_root):
    # Traverse DEM tiles
    for root, dirs, files in os.walk(dem_root):
        for file in files:
            if file.endswith(".tif"):
                dem_path = os.path.join(root, file)
                
                # Derive relative path to find corresponding satellite/output
                rel_path = os.path.relpath(dem_path, dem_root)
                # rel_path = "16/5/10.tif"
                
                # Check satellite
                # Assuming sat structure: sat_root/16/5/10.webp
                base_name = os.path.splitext(rel_path)[0] # "16/5/10"
                sat_path = os.path.join(sat_root, base_name + ".webp")
                
                out_path = os.path.join(out_root, base_name + ".bin")
                
                # Ensure output dir
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                
                process_tile(dem_path, sat_path, out_path)
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", required=True, help="DEM tiles root")
    parser.add_argument("--sat", required=True, help="Satellite tiles root")
    parser.add_argument("--out", required=True, help="Output root")
    args = parser.parse_args()
    
    batch_process(args.dem, args.sat, args.out)

if __name__ == "__main__":
    main()
