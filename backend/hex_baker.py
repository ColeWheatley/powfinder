import os
import glob
import rasterio
import numpy as np
import struct
from rasterio.windows import from_bounds
import concurrent.futures
from PIL import Image

# --- CONFIG ---
DEM_PATH = "/Users/cole/dev/PowFinder/backend/terrains/DGM_Tirol_5m_epsg31254_2006_2020.tif"
SAT_DIR = "/Users/cole/dev/PowFinder/frontend/hexes_beta/aerial_dem/original_tifs"
OUTPUT_DIR = "/Users/cole/dev/PowFinder/frontend/piston_viewer/tiles_bin"

# Hex Geometry (10m Flat-Topped)
DX = 10 * (np.sqrt(3)/2)  # ~8.66m
DY = 10                   # ~10m
STAGGER = 5               # 5m shift

def get_piston_slope(z_center, z_neigh, dist=10.0):
    angle = np.degrees(np.arctan2(z_neigh - z_center, dist))
    return int(max(0, min(255, angle + 90)))

def get_triangle_avg_color(sat_data, trans, cx, cy, angle_deg, radius=5.0):
    """
    Samples a triangular region pointing from (cx, cy) towards angle_deg.
    Simple approximation: Sample a small window in that direction.
    """
    import math
    # Point towards the midpoint of the side
    rad = math.radians(angle_deg)
    # Move slightly into the triangle
    sample_x = cx + math.cos(rad) * 3.0
    sample_y = cy + math.sin(rad) * 3.0
    
    # Get pixel coordinates
    row, col = ~trans * (sample_x, sample_y)
    r, c = int(row), int(col)
    
    # Sample a 5x5 window (1m x 1m at 20cm res)
    window_size = 5
    half = window_size // 2
    
    r1, r2 = max(0, r-half), min(sat_data.shape[1], r+half+1)
    c1, c2 = max(0, c-half), min(sat_data.shape[2], c+half+1)
    
    patch = sat_data[:, r1:r2, c1:c2]
    if patch.size == 0: return [128, 128, 128]
    
    avg = np.mean(patch, axis=(1, 2)).astype(int)
    return avg.tolist()

def process_tile(tif_path):
    print(f"Processing {os.path.basename(tif_path)}...")
    try:
        with rasterio.open(tif_path) as sat:
            b = sat.bounds
            sat_trans = sat.transform
            # Load Sat data (it's 20cm, so it's big, but we only read what we need)
            # Actually, loading the whole thing in memory for one process is faster
            sat_data = sat.read() # (3, H, W)
            
            left = int(np.floor(b.left / 10) * 10)
            right = int(np.ceil(b.right / 10) * 10)
            bottom = int(np.floor(b.bottom / 10) * 10)
            top = int(np.ceil(b.top / 10) * 10)
            
        out_name = f"tile_{left}_{top}.bin"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        # Load DEM with Halo
        HALO = 30 
        with rasterio.open(DEM_PATH) as dem:
            window = from_bounds(left - HALO, bottom - HALO, right + HALO, top + HALO, dem.transform)
            if window.width <= 0 or window.height <= 0: return f"FAIL: {tif_path} (Out of Bounds)"
            dem_data = dem.read(1, window=window).astype(np.float32)
            dem_trans = dem.window_transform(window)
            dem_data[dem_data < -100] = np.nan

        def sample_z(ex, ey):
            row, col = ~dem_trans * (ex, ey)
            r, c = int(row), int(col)
            if 0 <= r < dem_data.shape[0] and 0 <= c < dem_data.shape[1]:
                return dem_data[r, c]
            return np.nan

        hexes = []
        base_z = None
        
        # We sample the 3 South-facing sides
        # S1: 30 deg, S2: 150 deg, S3: 270 deg (Relative to Standard Math Axis)
        # Note: In our 31254 projection, Northing is North, Easting is East.
        
        for x in np.arange(left, right + DX, DX):
            col_idx = int(round((x - left) / DX))
            y_shift = STAGGER if (col_idx % 2 == 1) else 0
            
            for y in np.arange(bottom, top + DY, DY):
                real_y = y + y_shift
                z = sample_z(x, real_y)
                if np.isnan(z): continue
                
                if base_z is None: base_z = float(z)
                
                # Slopes
                z1 = sample_z(x + DX/2, real_y + STAGGER)
                z2 = sample_z(x - DX/2, real_y + STAGGER)
                z3 = sample_z(x, real_y - DY)
                
                # Side Colors (Averaging touching triangles from both hexes)
                c1a = get_triangle_avg_color(sat_data, sat_trans, x, real_y, 30)
                c1b = get_triangle_avg_color(sat_data, sat_trans, x + DX/2, real_y + STAGGER, 210)
                rgb1 = [(a + b) // 2 for a, b in zip(c1a, c1b)]
                
                c2a = get_triangle_avg_color(sat_data, sat_trans, x, real_y, 150)
                c2b = get_triangle_avg_color(sat_data, sat_trans, x - DX/2, real_y + STAGGER, 330)
                rgb2 = [(a + b) // 2 for a, b in zip(c2a, c2b)]
                
                c3a = get_triangle_avg_color(sat_data, sat_trans, x, real_y, 270)
                c3b = get_triangle_avg_color(sat_data, sat_trans, x, real_y - DY, 90)
                rgb3 = [(a + b) // 2 for a, b in zip(c3a, c3b)]
                
                # Sample south-facing neighbor heights for wall termination
                # These are the hexes that our south-facing walls extend down to
                nz1 = sample_z(x + DX, real_y - STAGGER)  # SE neighbor
                nz2 = sample_z(x - DX, real_y - STAGGER)  # SW neighbor
                nz3 = sample_z(x, real_y - DY)            # S neighbor (same as z3)

                hexes.append({
                    'z': float(z - base_z),
                    's1': get_piston_slope(z, z1),
                    's2': get_piston_slope(z, z2),
                    's3': get_piston_slope(z, z3),
                    'rgb1': rgb1, 'rgb2': rgb2, 'rgb3': rgb3,
                    'nz1': float(nz1 - base_z) if not np.isnan(nz1) else float(z - base_z),
                    'nz2': float(nz2 - base_z) if not np.isnan(nz2) else float(z - base_z),
                    'nz3': float(nz3 - base_z) if not np.isnan(nz3) else float(z - base_z),
                })

        if not hexes: return f"SKIP: {tif_path}"

        # Packet: 20 Bytes
        # [2B Float16 Z] [3B Slopes] [3B RGB1] [3B RGB2] [3B RGB3] [6B Neighbor Heights]
        with open(out_path, 'wb') as f:
            f.write(struct.pack('f', base_z))
            for h in hexes:
                f.write(np.array([h['z']], dtype=np.half).tobytes())
                f.write(struct.pack('BBB', h['s1'], h['s2'], h['s3']))
                f.write(struct.pack('BBB', *h['rgb1']))
                f.write(struct.pack('BBB', *h['rgb2']))
                f.write(struct.pack('BBB', *h['rgb3']))
                # Neighbor heights for south-facing wall termination
                f.write(np.array([h['nz1']], dtype=np.half).tobytes())
                f.write(np.array([h['nz2']], dtype=np.half).tobytes())
                f.write(np.array([h['nz3']], dtype=np.half).tobytes())
                
        return f"SUCCESS: {out_name} ({len(hexes)} hexes)"
    except Exception as e:
        return f"ERROR: {tif_path} -> {str(e)}"

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    tifs = sorted(glob.glob(os.path.join(SAT_DIR, "*.tif")))
    print(f"Baking CINEMATIC Tiles: {len(tifs)} targets...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_tile, tifs))
    
    for r in results: print(r)

if __name__ == "__main__":
    main()
