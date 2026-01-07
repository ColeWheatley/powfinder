
import os
import glob
import rasterio
import struct
import numpy as np
import math
import multiprocessing
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

# --- CONSTANTS ---
# Use the debug directory requested by user
SAT_DIR = "/Users/cole/dev/PowFinder/hex_backend/debug/aerials"
DEM_PATH = "/Users/cole/dev/PowFinder/hex_backend/DGM_Tirol_5m_epsg31254_2006_2020.tif"
OUTPUT_BASE_DIR = "/Users/cole/dev/PowFinder/frontend/hexagons/app/tiles_bin"

RESOLUTIONS = [
    1, 2, 3, 4, 5, 6, 7, 7.7136, 8, 9, 10, 11, 12, 13, 14, 15, 
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
    30, 35, 40, 45, 50
]

LOD_RATIO = math.sqrt(7) # ~2.646

# Mapping of method names to Rasterio Resampling enums
RESAMPLING_MAP = {
    'bilinear': Resampling.bilinear,
    'bicubic': Resampling.cubic,
    'lanczos': Resampling.lanczos
}

def process_tile_res_v3(args):
    """
    Standalone task for ONE tile at ONE resolution with ONE resampling method.
    """
    tif_path, res, method_name = args
    resampling_enum = RESAMPLING_MAP.get(method_name, Resampling.bilinear)
    fname = os.path.basename(tif_path)
    
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        tile_x = int(bounds.left)
        tile_y = int(bounds.top)

    target_res = res * LOD_RATIO if fname == "dop_2121-07_2023.tif" else float(res)
    y_spacing = target_res
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
    
    # Bilinear Upsample window to 0.5m (using the selected method for original upsample)
    UPSAMPLE_RES = 0.5
    BUFFER = 100.0 
    
    with rasterio.open(DEM_PATH) as dem:
        window = from_bounds(
            bounds.left - BUFFER, bounds.bottom - BUFFER,
            bounds.right + BUFFER, bounds.top + BUFFER,
            dem.transform
        )
        out_height = int(window.height * (dem.res[0] / UPSAMPLE_RES))
        out_width = int(window.width * (dem.res[1] / UPSAMPLE_RES))
        
        dem_data = dem.read(1, window=window, out_shape=(out_height, out_width), resampling=resampling_enum)
        mask = (dem_data < -1000)
        if np.any(mask):
            valid = dem_data[~mask]
            fill_val = np.mean(valid) if valid.size > 0 else 1000.0
            dem_data[mask] = fill_val
        
        dem_trans = dem.window_transform(window)
        rescale = dem.res[0] / UPSAMPLE_RES
        dem_trans_scaled = dem_trans * dem_trans.scale(1.0/rescale, 1.0/rescale)

    padded = np.zeros((dem_data.shape[0] + 1, dem_data.shape[1] + 1), dtype=np.float64)
    padded[1:, 1:] = dem_data
    sat = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    def get_avg_z(cx, cy, r):
        c0s, r0s = ~dem_trans_scaled * (cx - r, cy + (math.sqrt(3)*r/2.0))
        c1s, r1s = ~dem_trans_scaled * (cx + r, cy - (math.sqrt(3)*r/2.0))
        pc0, pc1 = int(round(c0s)) + 1, int(round(c1s)) + 1
        pr0, pr1 = int(round(r0s)) + 1, int(round(r1s)) + 1
        pr0, pr1 = max(0, min(sat.shape[0]-1, pr0)), max(0, min(sat.shape[0]-1, pr1))
        pc0, pc1 = max(0, min(sat.shape[1]-1, pc0)), max(0, min(sat.shape[1]-1, pc1))
        if pr0 > pr1: pr0, pr1 = pr1, pr0
        if pc0 > pc1: pc0, pc1 = pc1, pc0
        area = (pr1 - pr0) * (pc1 - pc0)
        if area <= 0: return padded[max(1, min(sat.shape[0]-1, pr0)), max(1, min(sat.shape[1]-1, pc0))]
        return (sat[pr1, pc1] - sat[pr0, pc1] - sat[pr1, pc0] + sat[pr0, pc0]) / area

    base_z = 1000.0
    hexes = []
    right, top_lim = 1250, 1000
    x_steps = np.arange(0, right + 1.1, x_spacing)
    y_steps = np.arange(0, top_lim + 1.1, y_spacing)

    for col_idx, dx in enumerate(x_steps):
        x_c = bounds.left + dx
        y_s = (y_spacing / 2.0) if (col_idx % 2 == 1) else 0.0
        for dy in y_steps:
            y_c = (bounds.bottom + dy) - y_s
            z = get_avg_z(x_c, y_c, hex_radius)
            z_s = z if not np.isnan(z) else base_z
            neighbors_z = []
            slopes = []
            for off in offsets:
                nz = get_avg_z(x_c + off[0], y_c + off[1], hex_radius)
                nz_s = nz if not np.isnan(nz) else (z_s - 5.0)
                neighbors_z.append(nz_s)
                dz = abs(z_s - nz_s)
                slope = math.degrees(math.atan(dz / y_spacing))
                slopes.append(slope)
            hexes.append({'z': z_s - base_z, 'n': [n - base_z for n in neighbors_z], 's': slopes})

    # Border check
    abs_zs = [h['z'] + base_z for h in hexes]
    save_base = base_z
    if any(z <= 1.0 for z in abs_zs):
        valid = [z for z in abs_zs if z > 1.0]
        save_base = sum(valid)/len(valid) if valid else 0.0
        for h in hexes: 
            h['z'], h['n'], h['s'] = 0.0, [0.0]*6, [0.0]*6

    final_abs = [h['z'] + save_base for h in hexes]
    min_z = min(final_abs) if final_abs else save_base
    avg_z = sum(final_abs)/len(final_abs) if final_abs else save_base

    res_dir = os.path.join(OUTPUT_BASE_DIR, method_name, f"res_{res}")
    os.makedirs(res_dir, exist_ok=True)
    out_path = os.path.join(res_dir, f"tile_{tile_x}_{tile_y}.bin")
    
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<f', save_base))
        f.write(struct.pack('<f', min_z))
        f.write(struct.pack('<f', avg_z))
        f.write(struct.pack('<f', y_spacing))
        for h in hexes:
            f.write(np.array([h['z']] + h['n'] + h['s'], dtype=np.half).tobytes())

    return f"{fname} @ {res}m ({method_name})"

def main():
    if not os.path.exists(OUTPUT_BASE_DIR): os.makedirs(OUTPUT_BASE_DIR)
    tifs = sorted(glob.glob(os.path.join("./aerial_tifs", "*.tif")))
    target = ["dop_2121-05_2023.tif", "dop_2121-06_2023.tif", "dop_2121-07_2023.tif"]
    tifs = [t for t in tifs if os.path.basename(t) in target]
    
    methods = ['bilinear', 'bicubic', 'lanczos']
    
    tasks = []
    for method in methods:
        for tif in tifs:
            for res in RESOLUTIONS:
                tasks.append((tif, res, method))
            
    print(f"Baking {len(tasks)} tasks (3 methods x 3 tiles x 31 resolutions)...")
    # Using slightly more workers since we have a lot of small tasks now
    with multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count())) as pool:
        for i, res in enumerate(pool.imap_unordered(process_tile_res_v3, tasks)):
            if i % 25 == 0: print(f"Progress: {i}/{len(tasks)} | Last: {res}")
            
    print("\nAll baking complete.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
