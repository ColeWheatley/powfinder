
import os
import glob
import rasterio
import struct
import numpy as np
import math
from rasterio.windows import from_bounds

# --- CONSTANTS (Matching main.js) ---
# Flat-Topped Hexagons in Odd-Q (Vertical Columns) layout
HEX_RADIUS = 5.7735
X_SPACING = 1.5 * HEX_RADIUS          # 8.66025
Y_SPACING = math.sqrt(3) * HEX_RADIUS # 10.0

# Paths
DEM_PATH = "/Users/cole/dev/PowFinder/backend/terrains/DGM_Tirol_5m_epsg31254_2006_2020.tif"
SAT_DIR = "/Users/cole/dev/PowFinder/backend/aerial_tifs"
OUTPUT_DIR = "/Users/cole/dev/PowFinder/frontend/piston_viewer/tiles_bin"

class SafeSampler:
    def __init__(self, tifs_dir):
        self.tifs = []
        print(f"Indexing TIFs in {tifs_dir}...")
        for path in glob.glob(os.path.join(tifs_dir, "*.tif")):
            with rasterio.open(path) as src:
                self.tifs.append({
                    'path': path,
                    'bounds': src.bounds, # box(left, bottom, right, top)
                    'src': src
                })
        # Pre-load data for speed (assuming fits in RAM)
        for t in self.tifs:
            with rasterio.open(t['path']) as handle:
                t['data'] = handle.read() 
                t['transform'] = handle.transform
        print(f"Loaded {len(self.tifs)} TIFs into memory.")


    def sample(self, x, y):
        # Find TIF containing x,y
        for t in self.tifs:
            b = t['bounds']
            # Coordinates are EPSG:31254 (East, North)
            if x >= b.left and x <= b.right and y >= b.bottom and y <= b.top:
                # ~transform * (x, y) returns (col, row)
                # Note: Rasterio row index 0 is at the top (max Y).
                # Affine transform for top-down TIFs correctly maps max Y to row 0.
                f_col, f_row = ~t['transform'] * (x, y)
                col, row = int(f_col), int(f_row)
                
                # data.shape is (Bands, Rows, Cols)
                if 0 <= row < t['data'].shape[1] and 0 <= col < t['data'].shape[2]:
                    # RGB
                    return t['data'][:3, row, col].tolist()
        return [0, 0, 0] # Default Black if out of bounds


# Global Sampler
sampler = None

def init_sampler():
    global sampler
    if not sampler:
        sampler = SafeSampler(SAT_DIR)

def process_tile(tif_path):
    print(f"Processing {os.path.basename(tif_path)}...")
    
    # 1. Determine Tile Bounds
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        tile_x = int(bounds.left)
        tile_y = int(bounds.top) 

    out_name = f"tile_{tile_x}_{tile_y}.bin"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    
    # 2. DEM Access
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
        r, c = ~dem_trans * (x, y)
        r, c = int(round(r)), int(round(c))
        if 0 <= r < dem_data.shape[0] and 0 <= c < dem_data.shape[1]:
             val = dem_data[r, c]
             if val < -100: return np.nan
             return val
        return np.nan

    # 3. Grid Generation (Odd-Q)
    c_min = int((bounds.left - 50) / X_SPACING)
    c_max = int((bounds.right + 50) / X_SPACING)
    r_min = int((bounds.bottom - 50) / Y_SPACING)
    r_max = int((bounds.top + 50) / Y_SPACING)

    hexes = []
    base_z = None

    for c in range(c_min, c_max + 1):
        x_center = c * X_SPACING
        
        # Verify X
        if x_center < bounds.left - 10 or x_center > bounds.right + 10:
            continue
            
        y_shift = (Y_SPACING / 2.0) if (c % 2 != 0) else 0.0
        
        for r in range(r_min, r_max + 1):
            y_center = r * Y_SPACING + y_shift
            
            # Verify Y
            if y_center < bounds.bottom - 10 or y_center > bounds.top + 10:
                continue
                
            z = get_z(x_center, y_center)
            if np.isnan(z): continue
            
            if base_z is None: base_z = z

            # Neighbors (Odd-Q logic matching Main.js)
            # S: (c, r-1)
            # SE: Even(c+1, r-1), Odd(c+1, r)
            # SW: Even(c-1, r-1), Odd(c-1, r)
            
            # South Coord
            cx_s = x_center
            cy_s = (r - 1) * Y_SPACING + y_shift
            
            # SE Coord
            r_se = r - 1 if (c % 2 == 0) else r
            cx_se = (c + 1) * X_SPACING
            cy_se = r_se * Y_SPACING + ((Y_SPACING / 2.0) if ((c + 1) % 2 != 0) else 0.0)

            # SW Coord
            r_sw = r - 1 if (c % 2 == 0) else r
            cx_sw = (c - 1) * X_SPACING
            cy_sw = r_sw * Y_SPACING + ((Y_SPACING / 2.0) if ((c - 1) % 2 != 0) else 0.0)

            # Sample Heights
            z_s = get_z(cx_s, cy_s)
            z_se = get_z(cx_se, cy_se)
            z_sw = get_z(cx_sw, cy_sw)
            
            def fill_nan(val, reference):
                return val if not np.isnan(val) else (reference - 50.0)
            
            z_s_safe = fill_nan(z_s, z)
            z_se_safe = fill_nan(z_se, z)
            z_sw_safe = fill_nan(z_sw, z)

            # Colors (Top/Bot pairs)
            def get_color_pair(target_cx, target_cy):
                dx = target_cx - x_center
                dy = target_cy - y_center
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 0.001: return [0,0,0], [0,0,0]
                nx, ny = dx/dist, dy/dist
                
                # Sample 4m (Top edge) and 6m (Bot edge / Neighbor Top)
                c1 = sampler.sample(x_center + nx * 4.0, y_center + ny * 4.0)
                c2 = sampler.sample(x_center + nx * 6.0, y_center + ny * 6.0)
                return c1, c2

            rgb_s_top, rgb_s_bot = get_color_pair(cx_s, cy_s)
            rgb_se_top, rgb_se_bot = get_color_pair(cx_se, cy_se)
            rgb_sw_top, rgb_sw_bot = get_color_pair(cx_sw, cy_sw)
            
            # Black void check
            if np.isnan(z_s): rgb_s_bot = [0,0,0]
            if np.isnan(z_se): rgb_se_bot = [0,0,0]
            if np.isnan(z_sw): rgb_sw_bot = [0,0,0]
            
            hexes.append({
                'z': float(z - base_z),
                'z_s': float(z_s_safe - base_z),
                'z_se': float(z_se_safe - base_z),
                'z_sw': float(z_sw_safe - base_z),
                'c_s': (rgb_s_top, rgb_s_bot),
                'c_se': (rgb_se_top, rgb_se_bot),
                'c_sw': (rgb_sw_top, rgb_sw_bot)
            })

    if not hexes: return f"SKIP: {tif_path}"

    # Pack 26 bytes per hex
    with open(out_path, 'wb') as f:
        f.write(struct.pack('f', base_z))
        for h in hexes:
            f.write(np.array([h['z'], h['z_s'], h['z_se'], h['z_sw']], dtype=np.half).tobytes())
            f.write(struct.pack('BBB', *h['c_s'][0]))
            f.write(struct.pack('BBB', *h['c_s'][1]))
            f.write(struct.pack('BBB', *h['c_se'][0]))
            f.write(struct.pack('BBB', *h['c_se'][1]))
            f.write(struct.pack('BBB', *h['c_sw'][0]))
            f.write(struct.pack('BBB', *h['c_sw'][1]))
            
    return f"SUCCESS: {out_name} ({len(hexes)} hexes)"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    init_sampler()
    tifs = sorted(glob.glob(os.path.join(SAT_DIR, "*.tif")))
    print(f"Baking PISTON Tiles V2: {len(tifs)} targets...")
    for tif in tifs:
        print(process_tile(tif))

if __name__ == "__main__":
    main()
