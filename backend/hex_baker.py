
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

class SafeSampler:
    def __init__(self, tifs_dir):
        self.tifs = []
        print(f"Indexing TIFs in {tifs_dir}...")
        for path in glob.glob(os.path.join(tifs_dir, "*.tif")):
            with rasterio.open(path) as src:
                self.tifs.append({
                    'path': path,
                    'bounds': src.bounds,
                    'src': src,
                    'transform': src.transform
                })
        
        # Pre-load data. 
        # CAUTION: If TIFs are huge, this might OOM. 
        # Assuming they fit since they worked before.
        for t in self.tifs:
            try:
                with rasterio.open(t['path']) as handle:
                    # Read as CHW (3, H, W)
                    t['data'] = handle.read() 
                    print(f"Loaded {os.path.basename(t['path'])}: {t['data'].shape}")
            except Exception as e:
                print(f"Failed to load {t['path']}: {e}")

    def sample(self, x, y):
        # Find TIF
        for t in self.tifs:
            b = t['bounds']
            if x >= b.left and x <= b.right and y >= b.bottom and y <= b.top:
                f_col, f_row = ~t['transform'] * (x, y)
                col, row = int(f_col), int(f_row)
                
                d = t.get('data')
                if d is None: return [0,0,0]

                if 0 <= row < d.shape[1] and 0 <= col < d.shape[2]:
                    return d[:3, row, col].tolist()
        return [0, 0, 0]

sampler = None
def init_sampler():
    global sampler
    if not sampler:
        sampler = SafeSampler(SAT_DIR)

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
        
        # DEBUG: Check range
        valid_mask = dem_data > -10000
        if valid_mask.any():
            d_min, d_max = dem_data[valid_mask].min(), dem_data[valid_mask].max()
            print(f"  [DEM DEBUG] Chunk Stats: Min={d_min}, Max={d_max}, Mean={dem_data[valid_mask].mean():.1f}, Type={dem_data.dtype}")
        else:
            print(f"  [DEM DEBUG] Chunk is all NoData/Invalid")

    def get_z(x, y):
        r, c = ~dem_trans * (x, y)
        r, c = int(round(r)), int(round(c))
        if 0 <= r < dem_data.shape[0] and 0 <= c < dem_data.shape[1]:
             val = dem_data[r, c]
             if val < -100: return np.nan
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
            
            # --- SOUTH-PUSH NEIGHBORS ---
            # We strictly care about S, SE, SW neighbors to form the 3 bottom walls.
            
            # 1. SOUTH Neighbor (Same X, Y - 10)
            cx_s = x_center
            cy_s = y_center - 10.0
            
            # 2. SE / SW Coordinates
            # Even Col (Shift 0): SE is Odd (Shift 5). Y-5.
            # Odd Col (Shift 5): SE is Even (Shift 0). Y+5.
            if col_idx % 2 == 0:
                cy_se = y_center - 5.0
                cy_sw = y_center - 5.0
            else:
                cy_se = y_center + 5.0
                cy_sw = y_center + 5.0
                
            cx_se = x_center + X_SPACING
            cx_sw = x_center - X_SPACING
            
            z_s  = get_z(cx_s, cy_s)
            z_se = get_z(cx_se, cy_se)
            z_sw = get_z(cx_sw, cy_sw)
            
            ref_z = z if not np.isnan(z) else (base_z or 0.0)
            z_safe = z if not np.isnan(z) else ref_z
            z_s_safe  = z_s  if not np.isnan(z_s)  else (z_safe - 2.0)
            z_se_safe = z_se if not np.isnan(z_se) else (z_safe - 2.0)
            z_sw_safe = z_sw if not np.isnan(z_sw) else (z_safe - 2.0)

            # --- CURTAIN COLOR SAMPLING ---
            # We want the color at the boundary, not the center.
            # Center to S-Edge is vector (0, -1). 
            # Boundary is roughly radius away? Flat top radius is ~5.77. 
            # Height (center to edge) is ~5.0.
            
            def get_boundary_colors(target_cx, target_cy, my_z, neighbor_z):
                # Vector from Me to Neighbor
                vx = target_cx - x_center
                vy = target_cy - y_center
                dist = math.sqrt(vx*vx + vy*vy) # Should be ~10 or 11.5
                if dist < 0.1: return [0,0,0], [0,0,0], 0

                nx, ny = vx/dist, vy/dist
                
                # Sample Points: 
                # Inner (My Edge): 4m out
                # Outer (Neighbor Edge): 6m out (or dist - 4m)
                
                # S-Neighbor is 10m away. Edge is at 5m.
                # SE-Neighbor is 10m away (vertical diff 5, horiz 8.66 -> dist 10).
                # Actually dist is 10.0 for all immediate neighbors in hex grid.
                
                c_edge_top = sampler.sample(x_center + nx * 4.5, y_center + ny * 4.5)
                c_edge_bot = sampler.sample(x_center + nx * 5.5, y_center + ny * 5.5)
                
                # Calculate Slope
                dz = abs(my_z - neighbor_z)
                # Run is 10m (center to center)? Or 0m (vertical wall)?
                # Piston viewer draws vertical walls. The "Slope" is conceptual terrain steepness.
                # Slope = atan(dz / dist_between_centers).
                slope_deg = math.degrees(math.atan2(dz, 10.0))
                
                return c_edge_top, c_edge_bot, int(slope_deg)

            rgb_s_top, rgb_s_bot, s_s = get_boundary_colors(cx_s, cy_s, z_safe, z_s_safe)
            rgb_se_top, rgb_se_bot, s_se = get_boundary_colors(cx_se, cy_se, z_safe, z_se_safe)
            rgb_sw_top, rgb_sw_bot, s_sw = get_boundary_colors(cx_sw, cy_sw, z_safe, z_sw_safe)

            hexes.append({
                'z': float(z_safe - (base_z or 0.0)),
                'nz_s': float(z_s_safe - (base_z or 0.0)),
                'nz_se': float(z_se_safe - (base_z or 0.0)),
                'nz_sw': float(z_sw_safe - (base_z or 0.0)),
                'c_s': (rgb_s_top, rgb_s_bot, s_s),
                'c_se': (rgb_se_top, rgb_se_bot, s_se),
                'c_sw': (rgb_sw_top, rgb_sw_bot, s_sw)
            })
            count += 1

    if not hexes: return f"SKIP: {tif_path}"

    # Analyze Slopes
    all_slopes = []
    for h in hexes:
        all_slopes.append(h['c_s'][2])
        all_slopes.append(h['c_se'][2])
        all_slopes.append(h['c_sw'][2])
    
    avg_s = sum(all_slopes) / len(all_slopes)
    max_s = max(all_slopes)
    print(f"  > Slopes: Avg={avg_s:.1f}°, Max={max_s:.1f}°. >45° count: {len([s for s in all_slopes if s > 45])}")

    # --- PACKING BINARY V3 ---
    # 4 Neighbors (inc Z) * 2 bytes (half) = 8 bytes
    # 3 Faces * (4 bytes Top + 3 bytes Bot) = 21 bytes
    # Total = 29 bytes per hex.
    
    with open(out_path, 'wb') as f:
        f.write(struct.pack('f', base_z or 0.0))
        for h in hexes:
            # Heights (float16)
            f.write(np.array([h['z'], h['nz_s'], h['nz_se'], h['nz_sw']], dtype=np.half).tobytes())
            
            # S Face: RGBA_Top (A=Slope), RGB_Bot
            c, cb, slope = h['c_s']
            f.write(struct.pack('BBBB', *c, min(slope, 255)))
            f.write(struct.pack('BBB', *cb))
            
            # SE Face
            c, cb, slope = h['c_se']
            f.write(struct.pack('BBBB', *c, min(slope, 255)))
            f.write(struct.pack('BBB', *cb))
            
            # SW Face
            c, cb, slope = h['c_sw']
            f.write(struct.pack('BBBB', *c, min(slope, 255)))
            f.write(struct.pack('BBB', *cb))

    return f"SUCCESS: {out_name} ({len(hexes)} hexes)"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    init_sampler()
    tifs = sorted(glob.glob(os.path.join(SAT_DIR, "*.tif")))
    print(f"Baking PISTON V3 (South-Push + Slope): {len(tifs)} targets...")
    for tif in tifs:
        print(process_tile(tif))

if __name__ == "__main__":
    main()
