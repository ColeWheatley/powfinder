
import os
import glob
import rasterio
import struct
import numpy as np
import math
import multiprocessing
from rasterio.windows import from_bounds

# --- CONSTANTS ---
# Use the debug directory requested by user
SAT_DIR = "/Users/cole/dev/PowFinder/hex_backend/debug/aerials"
DEM_PATH = "/Users/cole/dev/PowFinder/hex_backend/DGM_Tirol_5m_epsg31254_2006_2020.tif"
OUTPUT_BASE_DIR = "/Users/cole/dev/PowFinder/frontend/hexagons/app/tiles_bin"

RESOLUTIONS = [2.5, 5, 8, 10, 12, 15, 20, 25, 30, 50]

def process_tile(tif_path, resolution):
    # Spacing based on resolution (resolution is Y_SPACING)
    y_spacing = float(resolution)
    hex_radius = y_spacing / math.sqrt(3)
    x_spacing = 1.5 * hex_radius
    
    offsets = [
        (0, y_spacing),                 # 0: N
        (x_spacing, y_spacing/2.0),     # 1: NE
        (x_spacing, -y_spacing/2.0),    # 2: SE
        (0, -y_spacing),                # 3: S
        (-x_spacing, -y_spacing/2.0),   # 4: SW
        (-x_spacing, y_spacing/2.0)     # 5: NW
    ]
    
    print(f"Processing {os.path.basename(tif_path)} at {resolution}m...")
    
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        tile_x = int(bounds.left)
        tile_y = int(bounds.top) 
    
    res_dir = os.path.join(OUTPUT_BASE_DIR, f"res_{resolution}")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir, exist_ok=True)
        
    out_name = f"tile_{tile_x}_{tile_y}.bin"
    out_path = os.path.join(res_dir, out_name)
    
    # DEM
    with rasterio.open(DEM_PATH) as dem:
        BUFFER = 100.0
        window = from_bounds(
            bounds.left - BUFFER, bounds.bottom - BUFFER,
            bounds.right + BUFFER, bounds.top + BUFFER,
            dem.transform
        )
        dem_data = dem.read(1, window=window)
        dem_trans = dem.window_transform(window)

    def get_z(x, y):
        c, r = ~dem_trans * (x, y)
        r, c = int(round(r)), int(round(c))
        if 0 <= r < dem_data.shape[0] and 0 <= c < dem_data.shape[1]:
             val = dem_data[r, c]
             if val < -1000: return np.nan
             return val
        return np.nan

    hexes = []
    base_z = 1000.0 

    # Grid Gen
    right = 1250
    top = 1000
    
    x_steps = []
    curr_x = 0.0
    while curr_x <= right + 1.1:
        x_steps.append(curr_x)
        curr_x += x_spacing
        
    y_steps = []
    curr_y = 0.0
    while curr_y <= top + 1.1:
        y_steps.append(curr_y)
        curr_y += y_spacing

    for col_idx, dx in enumerate(x_steps):
        x_center = bounds.left + dx
        y_shift = (y_spacing / 2.0) if (col_idx % 2 == 1) else 0.0
        
        for dy in y_steps:
            y_center = (bounds.bottom + dy) - y_shift
            
            z = get_z(x_center, y_center)
            z_safe = z if not np.isnan(z) else base_z

            neighbors_z = []
            for off in offsets:
                nx = x_center + off[0]
                ny = y_center + off[1]
                nz = get_z(nx, ny)
                neighbors_z.append(nz if not np.isnan(nz) else (z_safe - 5.0))
            
            hexes.append({
                'z': z_safe - base_z,
                'n': [n - base_z for n in neighbors_z]
            })

    # --- BORDER PROTECTION ---
    abs_zs = [h['z'] + base_z for h in hexes]
    has_zeros = any(z <= 1.0 for z in abs_zs)
    
    if has_zeros:
        valid_zs = [z for z in abs_zs if z > 1.0]
        if valid_zs:
            avg_z = sum(valid_zs) / len(valid_zs)
            print(f"  !!! BORDER DETECTED !!! Flattening tile {tile_x}_{tile_y} to {avg_z:.1f}m")
            b_val = avg_z
            for h in hexes:
                h['z'] = 0.0 
                h['n'] = [0.0] * 6
        else:
            b_val = 0.0
            for h in hexes:
                h['z'] = 0.0
                h['n'] = [0.0] * 6
        save_base = b_val
    else:
        save_base = base_z

    final_abs_zs = [h['z'] + save_base for h in hexes]
    if final_abs_zs:
        min_z = min(final_abs_zs)
        avg_z = sum(final_abs_zs) / len(final_abs_zs)
    else:
        min_z = save_base
        avg_z = save_base

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<f', save_base))
        f.write(struct.pack('<f', min_z))
        f.write(struct.pack('<f', avg_z))
        for h in hexes:
            data = [h['z']] + h['n']
            # We encode as float16 to save space
            # Use np.half which is float16
            f.write(np.array(data, dtype=np.half).tobytes())

    return f"SUCCESS: {out_name} @ {resolution}m"

def process_tile_wrapper(args):
    return process_tile(*args)

def main():
    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR)
    
    # Check if SAT_DIR exists, if not, try to find the "stubai" one
    actual_sat_dir = SAT_DIR
    if not os.path.exists(actual_sat_dir):
        # Fallback to the original aerial_tifs if debug isn't ready
        actual_sat_dir = "/Users/cole/dev/PowFinder/hex_backend/aerial_tifs"
        print(f"Warning: {SAT_DIR} not found. Falling back to {actual_sat_dir}")

    tifs = sorted(glob.glob(os.path.join(actual_sat_dir, "*.tif")))
    
    # Handle the "stubai" request if it's a subdirectory
    stubai_dir = os.path.join(actual_sat_dir, "stubai")
    if os.path.exists(stubai_dir):
        actual_sat_dir = stubai_dir
        tifs = sorted(glob.glob(os.path.join(actual_sat_dir, "*.tif")))
        print(f"Found stubai directory: {actual_sat_dir}")
    elif actual_sat_dir == "/Users/cole/dev/PowFinder/hex_backend/aerial_tifs":
        # Limit to two tiles for testing if we are using the main directory
        tifs = tifs[:2]
        print(f"BETA TEST: Limiting to first two tiles: {[os.path.basename(t) for t in tifs]}")

    if not tifs:
        print(f"No TIFs found in {actual_sat_dir}")
        return

    print(f"Baking resolutions: {RESOLUTIONS}")
    print(f"Using {multiprocessing.cpu_count()} workers...")
    
    tasks = []
    for res in RESOLUTIONS:
        for tif in tifs:
            tasks.append((tif, res))
            
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(pool.imap_unordered(process_tile_wrapper, tasks))
        for r in results:
            if "SUCCESS" not in r:
                print(r)

    print("Baking complete.")

if __name__ == "__main__":
    main()
