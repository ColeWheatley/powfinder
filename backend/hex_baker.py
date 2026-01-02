
import os
import glob
import rasterio
import struct
import numpy as np
import math
import multiprocessing
from rasterio.windows import from_bounds

# --- CONSTANTS ---
HEX_RADIUS = 5.7735
X_SPACING = 1.5 * HEX_RADIUS          # 8.66025
Y_SPACING = math.sqrt(3) * HEX_RADIUS # 10.0

# Paths
DEM_PATH = "/Users/cole/dev/PowFinder/backend/terrains/DGM_Tirol_5m_epsg31254_2006_2020.tif"
SAT_DIR = "/Users/cole/dev/PowFinder/backend/aerial_tifs"
OUTPUT_DIR = "/Users/cole/dev/PowFinder/frontend/piston_viewer/tiles_bin"

OFFSETS = [
    (0, 10),       # 0: N
    (8.66025, 5),  # 1: NE
    (8.66025, -5), # 2: SE
    (0, -10),      # 3: S
    (-8.66025, -5),# 4: SW
    (-8.66025, 5)  # 5: NW
]
# NOTE: Viewer face indices are clockwise: 0=N, 1=NE, 2=SE, 3=S, 4=SW, 5=NW.
# Neighbor array order remains [N, NE, SE, S, SW, NW]; keep in sync with shader mapping.

def process_tile(tif_path):
    print(f"Processing {os.path.basename(tif_path)}...")
    
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        tile_x = int(bounds.left)
        tile_y = int(bounds.top) 
    
    out_name = f"tile_{tile_x}_{tile_y}.bin"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    
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
    # FIX: Use a CONSTANT GLOBAL BASELINE (1000m) for all 80 tiles.
    # This ensures perfect horizontal/vertical stitching and eliminates floating strips.
    base_z = 1000.0 

    # Grid Gen
    right = 1250
    top = 1000
    
    x_steps = []
    curr_x = 0.0
    while curr_x <= right + 1.1:
        x_steps.append(curr_x)
        curr_x += X_SPACING
        
    y_steps = []
    curr_y = 0.0
    while curr_y <= top + 1.1:
        y_steps.append(curr_y)
        curr_y += 10.0

    for col_idx, dx in enumerate(x_steps):
        x_center = bounds.left + dx
        y_shift = 5.0 if (col_idx % 2 == 1) else 0.0
        
        for dy in y_steps:
            y_center = (bounds.bottom + dy) - y_shift
            
            z = get_z(x_center, y_center)
            z_safe = z if not np.isnan(z) else base_z

            neighbors_z = []
            for off in OFFSETS:
                nx = x_center + off[0]
                ny = y_center + off[1]
                nz = get_z(nx, ny)
                # For neighbors on the edge of the DEM, tilt them slightly down
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
            print(f"  !!! BORDER DETECTED (Italy) !!! Flattening tile to {avg_z:.1f}m")
            b_val = avg_z
            for h in hexes:
                h['z'] = 0.0 # Height relative to b_val
                h['n'] = [0.0] * 6
        else:
            print(f"  !!! DEAD TILE (Italy) !!! Flattening to 0m")
            b_val = 0.0
            for h in hexes:
                h['z'] = 0.0
                h['n'] = [0.0] * 6
        
        # Save this tile with its specific flattened base_z instead of global 1000
        # This prevents it from dropping into a hole since its relative z is 0.0
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
            f.write(np.array(data, dtype=np.half).tobytes())

    return f"SUCCESS: {out_name}"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    tifs = sorted(glob.glob(os.path.join(SAT_DIR, "*.tif")))
    print(f"Baking PISTON V4 (GLOBAL BASELINE 1000m) with {multiprocessing.cpu_count()} workers...")
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(pool.imap_unordered(process_tile, tifs))
        for r in results:
            if "SUCCESS" not in r:
                print(r)

    print("Baking complete.")

if __name__ == "__main__":
    main()
