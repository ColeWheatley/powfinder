
import os
import glob
import rasterio
import struct
import numpy as np
import math
from rasterio.windows import from_bounds

# --- CONSTANTS ---
HEX_RADIUS = 5.7735
X_SPACING = 1.5 * HEX_RADIUS          # 8.66025
Y_SPACING = math.sqrt(3) * HEX_RADIUS # 10.0

# Paths
DEM_PATH = "/Users/cole/dev/PowFinder/backend/terrains/DGM_Tirol_5m_epsg31254_2006_2020.tif"
SAT_DIR = "/Users/cole/dev/PowFinder/backend/aerial_tifs"
OUTPUT_DIR = "/Users/cole/dev/PowFinder/frontend/piston_viewer/tiles_bin"

# NEIGHBOR OFFSETS (Odd-Q / Vertical Staggered)
# N: (0, 10)
# S: (0, -10)
# Even Col (Shift 0): NE(+8.66, +5), SE(+8.66, -5), NW(-8.66, +5), SW(-8.66, -5)
# Odd Col  (Shift 5): NE(+8.66, +5), SE(+8.66, -5), NW(-8.66, +5), SW(-8.66, -5)
# Wait, let's verify Odd/Even shifts.
# Grid Generation: 
# x += 8.66. 
# y_shift = 5 if col%2==1 else 0. 
# y_center = (Top - dy) - y_shift.
#
# Neighbor N: Same Col. dy decreases by 10. -> y_center increases by 10. Correct.
# Neighbor S: Same Col. dy increases by 10. -> y_center decreases by 10. Correct.
#
# Neighbor NE (Col+1, Row?):
# If Even (y_shift=0): Me(y). Neighbor(Odd, y_shift=5).
#   To match y+5? 
#   If I am Even at Y=100.
#   Neighbor Odd is at X+8.66. Its grid Ys are ... 95, 105...
#   NE should be Y=105.
#   So NE is (x+8.66, y+5).
#   SE is (x+8.66, y-5).
# If Odd (y_shift=5): Me at Y=105.
#   Neighbor Even is at X+8.66. Its grid Ys are ... 100, 110...
#   NE should be Y=110. (y+5).
#   SE should be Y=100. (y-5).
# CONCLUSION: Offsets are constant regardless of column parity because the grid y-shift handles the staggering.
# N: (0, 10)
# NE: (8.66, 5)
# SE: (8.66, -5)
# S: (0, -10)
# SW: (-8.66, -5)
# NW: (-8.66, 5)

OFFSETS = [
    (0, 10),      # 0: N
    (8.66025, 5), # 1: NE
    (8.66025, -5),# 2: SE
    (0, -10),     # 3: S
    (-8.66025, -5),# 4: SW
    (-8.66025, 5) # 5: NW
]

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
    base_z = None

    # Grid Gen (Matches Main.js logic)
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

    count = 0
    for col_idx, dx in enumerate(x_steps):
        x_center = bounds.left + dx
        y_shift = 5.0 if (col_idx % 2 == 1) else 0.0
        
        for dy in y_steps:
            # Main.js logic: y goes 0 to 1000. RealY = 0 is Top.
            y_center = (bounds.top - dy) - y_shift
            
            z = get_z(x_center, y_center)
            if base_z is None and not np.isnan(z): base_z = z
            
            ref_z = z if not np.isnan(z) else (base_z or 0.0)
            z_safe = z if not np.isnan(z) else ref_z

            # Sample 6 Neighbors
            neighbors_z = []
            for off in OFFSETS:
                nx = x_center + off[0]
                ny = y_center + off[1]
                nz = get_z(nx, ny)
                # If neighbor is missing, assume it is very low (so I render a wall down to it? 
                # Or assume it is same height?
                # If edge of map, assuming same height prevents ugly walls.
                # Assuming low creates cliffs.
                # Let's assume SAME HEIGHT - 5m to create a small edge but not infinite.
                neighbors_z.append(nz if not np.isnan(nz) else (z_safe - 5.0))
            
            hexes.append({
                'z': z_safe - (base_z or 0.0),
                'n': [n - (base_z or 0.0) for n in neighbors_z]
            })
            count += 1

    if not hexes: return f"SKIP: {tif_path}"

    # --- PACKING BINARY V4 ---
    # Z (half) + 6 * NeighborZ (half) = 7 * 2 = 14 bytes per hex.
    
    with open(out_path, 'wb') as f:
        f.write(struct.pack('f', base_z or 0.0))
        for h in hexes:
            data = [h['z']] + h['n'] # [z, n0, n1, n2, n3, n4, n5]
            f.write(np.array(data, dtype=np.half).tobytes())

    return f"SUCCESS: {out_name} ({len(hexes)} hexes)"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    tifs = sorted(glob.glob(os.path.join(SAT_DIR, "*.tif")))
    print(f"Baking PISTON V4 (6-Face Downward): {len(tifs)} targets...")
    for tif in tifs:
        print(process_tile(tif))

if __name__ == "__main__":
    main()
