
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

    # 3. Grid Generation (Matching Main.js exactly)
    # x loop: 0 to 1250 step HEX_DX (8.66025)
    # y loop: 0 to 1000 step 10.0
    
    hexes = []
    base_z = None

    # We use a small epsilon for the float range to match the <= right + 1 logic
    right = 1250
    top = 1000
    
    # Calculate how many steps we expect
    x_steps = []
    curr_x = 0.0
    while curr_x <= right + 1.1: # 1.1 is safety to catch the last step
        x_steps.append(curr_x)
        curr_x += X_SPACING
        
    y_steps = []
    curr_y = 0.0
    while curr_y <= top + 1.1:
        y_steps.append(curr_y)
        curr_y += 10.0

    print(f"Generating dense grid: {len(x_steps)} columns, {len(y_steps)} rows. Total: {len(x_steps)*len(y_steps)}")

    for col_idx, dx in enumerate(x_steps):
        x_center = bounds.left + dx
        y_shift = 5.0 if (col_idx % 2 == 1) else 0.0
        
        for dy in y_steps:
            y_center = (bounds.bottom + dy) + y_shift # Start from bottom to match 0 to 1000 in main.js
            # Actually, main.js uses -realY, so it goes from 0 to -1000. 
            # In world space, that's top down.
            # If main.js loop is y=0 to 1000 and it places at -y, then y=0 is TOP (bounds.top).
            # So y_center should be bounds.top - dy - y_shift.
            
            y_center = (bounds.top - dy) - y_shift
            
            z = get_z(x_center, y_center)
            
            if base_z is None and not np.isnan(z): 
                base_z = z

            # Neighbor coords (Relative to current x_center, y_center)
            # S: same x, y-10
            # SE: x + 8.66, y-5 (if even-to-odd) or y+5 (if odd-to-even) ? 
            # Let's just use the same logic as baker had but with these centers.
            
            # S Coord
            cx_s = x_center
            cy_s = y_center - 10.0
            
            # SE/SW depend on odd/even column.
            # In Main.js: colIdx % 2 === 1 ? 5 : 0
            # If current col is even (y_shift=0), neighbor (c+1) is odd (y_shift=5).
            # Neighbor SE is at (c+1, r) which has center (x+8.66, y+5). 
            # Wait, Main.js doesn't explicitly calculate neighbor coordinates, it just assumes indices?
            # No, the shader handles the neighbor offsets. The baker just needs to provide the heights.
            
            # Re-calculating neighbors for the current y_center
            # This logic needs to be robust.
            
            # SE Coord
            cx_se = x_center + X_SPACING
            # If we are even (shift 0), neighbor is odd (shift 5). 
            # y_center was bounds.top - dy - 0. 
            # Neighbor y_center_se would be bounds.top - dy - 5.
            # So cy_se = y_center - 5.0
            
            # If we are odd (shift 5), neighbor is even (shift 0).
            # y_center was bounds.top - dy - 5.
            # Neighbor y_center_se would be bounds.top - dy - 0.
            # So cy_se = y_center + 5.0
            
            if col_idx % 2 == 0:
                cy_se = y_center - 5.0
                cy_sw = y_center - 5.0
            else:
                cy_se = y_center + 5.0
                cy_sw = y_center + 5.0
            
            cx_sw = x_center - X_SPACING

            # Sample Heights
            z_s = get_z(cx_s, cy_s)
            z_se = get_z(cx_se, cy_se)
            z_sw = get_z(cx_sw, cy_sw)
            
            # Safe heights for NaN
            reference_z = z if not np.isnan(z) else (base_z if base_z is not None else 0.0)
            
            z_safe = z if not np.isnan(z) else reference_z
            z_s_safe = z_s if not np.isnan(z_s) else (z_safe - 5.0)
            z_se_safe = z_se if not np.isnan(z_se) else (z_safe - 5.0)
            z_sw_safe = z_sw if not np.isnan(z_sw) else (z_safe - 5.0)

            # Colors (Top/Bot pairs)
            def get_color_pair(target_cx, target_cy, current_z, neighbor_z):
                if np.isnan(current_z): return [0,0,0], [0,0,0]
                
                dx = target_cx - x_center
                dy = target_cy - y_center
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 0.001: return [0,0,0], [0,0,0]
                nx, ny = dx/dist, dy/dist
                
                c1 = sampler.sample(x_center + nx * 4.0, y_center + ny * 4.0)
                c2 = sampler.sample(x_center + nx * 6.0, y_center + ny * 6.0)
                
                if np.isnan(neighbor_z): c2 = [0,0,0]
                return c1, c2

            rgb_s_top, rgb_s_bot = get_color_pair(cx_s, cy_s, z, z_s)
            rgb_se_top, rgb_se_bot = get_color_pair(cx_se, cy_se, z, z_se)
            rgb_sw_top, rgb_sw_bot = get_color_pair(cx_sw, cy_sw, z, z_sw)
            
            hexes.append({
                'z': float(z_safe - (base_z or 0.0)),
                'z_s': float(z_s_safe - (base_z or 0.0)),
                'z_se': float(z_se_safe - (base_z or 0.0)),
                'z_sw': float(z_sw_safe - (base_z or 0.0)),
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
