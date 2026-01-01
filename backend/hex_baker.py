import os
import glob
import rasterio
import numpy as np
import struct
from rasterio.windows import from_bounds

# --- CONFIG ---
DEM_PATH = "/Users/cole/dev/PowFinder/backend/terrains/DGM_Tirol_5m_epsg31254_2006_2020.tif"
SAT_DIR = "/Users/cole/dev/PowFinder/frontend/hexes_beta/aerial_dem/original_tifs"
OUTPUT_DIR = "/Users/cole/dev/PowFinder/frontend/piston_viewer/tiles_bin"

# Hex Geometry (10m Flat-Topped)
DX = 10 * (np.sqrt(3)/2)  # Column spacing (~8.66m)
DY = 10                   # Row spacing
STAGGER = 5               # Vertical stagger

def get_piston_slope(z_center, z_neigh, dist=10.0):
    # Rise over run -> degrees
    angle = np.degrees(np.arctan2(z_neigh - z_center, dist))
    # Map signed -90/+90 to 0-180 for a safe Uint8
    return int(max(0, min(255, angle + 90)))

def process_tile(tif_path):
    with rasterio.open(tif_path) as sat:
        b = sat.bounds
        # Snap the tile bounds to the Global 10m grid
        left = int(np.floor(b.left / 10) * 10)
        right = int(np.ceil(b.right / 10) * 10)
        bottom = int(np.floor(b.bottom / 10) * 10)
        top = int(np.ceil(b.top / 10) * 10)
        
    out_name = f"tile_{left}_{top}.bin"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    
    # Open DEM with a Halo to handle neighbors correctly for slopes
    HALO = 30 
    with rasterio.open(DEM_PATH) as dem:
        window = from_bounds(left - HALO, bottom - HALO, right + HALO, top + HALO, dem.transform)
        # Check if window is empty or outside
        if window.width <= 0 or window.height <= 0: return f"FAIL: {tif_path} (Out of Bounds)"
        
        # We need Float32 for math, will convert to Float16 later
        data = dem.read(1, window=window).astype(np.float32)
        trans = dem.window_transform(window)
        data[data < -100] = np.nan # Clean NoData

    def sample_z(ex, ey):
        row, col = ~trans * (ex, ey)
        # Using a simple 2x2 bilinear sample for sub-pixel DEM precision
        c1, r1 = int(np.floor(col)), int(np.floor(row))
        c2, r2 = min(c1 + 1, data.shape[1]-1), min(r1 + 1, data.shape[0]-1)
        if 0 <= r1 < data.shape[0] and 0 <= c1 < data.shape[1]:
            dx, dy = col - c1, row - r1
            v11, v21 = data[r1, c1], data[r1, c2]
            v12, v22 = data[r2, c1], data[r2, c2]
            if np.any(np.isnan([v11, v21, v12, v22])): return v11 # Fallback
            return (v11*(1-dx) + v21*dx)*(1-dy) + (v12*(1-dx) + v22*dx)*dy
        return np.nan

    hexes = []
    base_z = None
    
    # Global Loop: Snapped to the 10m grid
    for x in np.arange(left, right + DX, DX):
        col_idx = int(round((x - left) / DX))
        y_shift = STAGGER if (col_idx % 2 == 1) else 0
        
        for y in np.arange(bottom, top + DY, DY):
            real_y = y + y_shift
            z = sample_z(x, real_y)
            if np.isnan(z): continue
            
            if base_z is None: base_z = float(z)
            
            # Calculate 3 Axial Slopes (Vectors pointing at three visible faces)
            # Neighbor 1: 30 deg, Neighbor 2: 150 deg, Neighbor 3: 270 deg
            z_s1 = sample_z(x + DX/2, real_y + STAGGER)
            z_s2 = sample_z(x - DX/2, real_y + STAGGER)
            z_s3 = sample_z(x, real_y - DY)
            
            hexes.append({
                'z': float(z - base_z),
                's1': get_piston_slope(z, z_s1),
                's2': get_piston_slope(z, z_s2),
                's3': get_piston_slope(z, z_s3)
            })

    if not hexes: return f"SKIP: {tif_path} (Empty)"

    # Write 5-byte Packet Binary
    with open(out_path, 'wb') as f:
        # 4 Byte Header: Tile Root Elevation
        f.write(struct.pack('f', base_z))
        for h in hexes:
            # 2 Byte: Relative Z (Float16)
            f.write(np.array([h['z']], dtype=np.half).tobytes())
            # 3 Bytes: Slope Degrees (mapped)
            f.write(struct.pack('BBB', h['s1'], h['s2'], h['s3']))
            
    return f"SUCCESS: {tif_path} -> {out_name}"

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    tifs = glob.glob(os.path.join(SAT_DIR, "*.tif"))
    print(f"Starting Hex Bake: {len(tifs)} tiles...")
    
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_tile, tifs))
    
    for r in results: print(r)

if __name__ == "__main__":
    main()
