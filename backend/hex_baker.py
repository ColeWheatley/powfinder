
import os
import rasterio
import numpy as np
import struct
from rasterio.windows import from_bounds
import concurrent.futures

# --- CONFIGURATION (Coordinate System Truth) ---
# We use a Flat-Topped Hexagon Grid.
# Orientation: Flat tops aligned horizontally.
# Vertices: 0, 60, 120, 180, 240, 300 degrees.
# Faces (Normals): 30, 90, 150, 210 (SW), 270 (S), 330 (SE).
# Camera View: From South (+Z in ThreeJS, -Y in Geospatial) looking North.
# Visible Faces: South-West (210), South (270), South-East (330).

# Hex Size
HEX_RADIUS = 5.7735  # For width (flat-to-flat) of 10m. R = 10 / sqrt(3).
HEX_WIDTH = 10.0     # Flat-to-flat
HEX_HEIGHT = 11.547  # Point-to-point (2 * R)

# Grid Spacing (Odd-R layout generally, but we'll use calculated offsets)
# Horizontal spacing: HEX_WIDTH (10m)
# Vertical spacing: HEX_RADIUS * 1.5 (~8.66m)
COL_STEP = 10.0
ROW_STEP = HEX_RADIUS * 1.5

DEM_PATH = "/Users/cole/dev/PowFinder/backend/terrains/DGM_Tirol_5m_epsg31254_2006_2020.tif"
SAT_DIR = "/Users/cole/dev/PowFinder/frontend/hexes_beta/aerial_dem/original_tifs"
OUTPUT_DIR = "/Users/cole/dev/PowFinder/frontend/piston_viewer/tiles_bin"

def get_color_at_point(sat_data, sat_trans, x, y):
    """Samples the satellite image at a specific point."""
    row, col = ~sat_trans * (x, y)
    r, c = int(row), int(col)
    
    # Check bounds
    if 0 <= r < sat_data.shape[1] and 0 <= c < sat_data.shape[2]:
        return sat_data[:, r, c].tolist()
    return [100, 100, 100]

def process_tile(tif_path):
    print(f"Processing {os.path.basename(tif_path)}...")
    try:
        # 1. Load Satellite Data (Source of Bounds)
        with rasterio.open(tif_path) as sat:
            sat_bounds = sat.bounds
            sat_trans = sat.transform
            sat_data = sat.read() # Load RGB
            
            # Snap bounds to grid to ensure consistent alignment across tiles
            # We align to origin (0,0)
            
            # Find min row/col indices that cover the bounds
            min_row_idx = int(np.floor(sat_bounds.bottom / ROW_STEP))
            max_row_idx = int(np.ceil(sat_bounds.top / ROW_STEP))
            min_col_idx = int(np.floor(sat_bounds.left / COL_STEP))
            max_col_idx = int(np.ceil(sat_bounds.right / COL_STEP))
            
            # Output filename based on rounded coords for ID
            tile_id_x = int(sat_bounds.left)
            tile_id_y = int(sat_bounds.top)
            
        out_name = f"tile_{tile_id_x}_{tile_id_y}.bin"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        # 2. Load DEM Data (with context for neighbors)
        # We need a buffer since we simulate 'neighbors' which might be outside the sat tile
        BUFFER = 50.0 
        with rasterio.open(DEM_PATH) as dem:
            window = from_bounds(
                sat_bounds.left - BUFFER, sat_bounds.bottom - BUFFER,
                sat_bounds.right + BUFFER, sat_bounds.top + BUFFER,
                dem.transform
            )
            dem_data = dem.read(1, window=window)
            dem_trans = dem.window_transform(window)
            
        def sample_z(wx, wy):
            """Bilinear or Nearest sample from DEM"""
            r, c = ~dem_trans * (wx, wy)
            r, c = int(round(r)), int(round(c))
            if 0 <= r < dem_data.shape[0] and 0 <= c < dem_data.shape[1]:
                val = dem_data[r, c]
                if val < -100: return np.nan # No Data
                return val
            return np.nan

        # 3. Generate Hexes
        hexes = []
        base_z = None
        
        # Grid Generation Loop matches the "Odd-Q" or similar offset logic compatible with flat/pointy mixing
        # We used Flat-Topped Geometry.
        # "Odd-Q" vertical layout? No, usually Odd-Q is for vertical columns. 
        # For horizontally aligned Flat-Tops, we use "Odd-R" (offset rows).
        # Shift every odd row by half width.
        
        for r_idx in range(min_row_idx, max_row_idx + 1):
            y_center = r_idx * ROW_STEP
            
            # Offset for odd rows
            x_offset = (COL_STEP / 2.0) if (r_idx % 2 != 0) else 0.0
            
            for c_idx in range(min_col_idx, max_col_idx + 1):
                x_center = c_idx * COL_STEP + x_offset
                
                # Check if this hex center is inside the satellite tile (with small margin)
                if not (sat_bounds.left - 5 <= x_center <= sat_bounds.right + 5 and 
                        sat_bounds.bottom - 5 <= y_center <= sat_bounds.top + 5):
                    continue

                z_center = sample_z(x_center, y_center)
                if np.isnan(z_center): continue
                
                if base_z is None: base_z = z_center

                # Calculate Neighbor Coordinates for the 3 South Faces
                # We need exact geometry centers to sample their Z (for piston length)
                
                # S Neighbor (Bottom): Same X, Y - stride?
                # In Odd-R Flat Top:
                # S neighbor is at (c, r-1) ? No, coords are staggered.
                # Let's use Geometry offsets.
                # S Face Normal is 270 deg (South). Neighbor is directly South?
                # Flat Topped Grid:
                # Neighbors are at 30, 90, 150... No, those are faces.
                # The Neighbors of a Flat Topped Hex are at 30, 90, 150... ? 
                # NO. Neighbors are at 30, 90... that's Pointy Topped.
                
                # FLAT TOPPED Neighbors are at: 0 deg (Invalid?), 60, 120...
                # Vector to neighbor:
                # N: (0, +1.732 R) approx?
                # Vertical neighbors (N/S) exist in Flat Topped? 
                # Yes, but they are clearly N and S.
                # Wait, Flat Topped grid looks like bricks.
                # Direct N and S neighbors exist.
                # Their centers are at X_same, Y +/- (2*H_tri + Side) ??
                # Dist Y = 1.5 * R. Dist X = 0? No, offset.
                
                # Let's derive simpler:
                # South Neighbor (S, 270 deg):
                #   y_s = y_center - ROW_STEP
                #   x_s = x_center +/- offset?
                #   Actually, in Odd-R grid, the rows below are shifted.
                #   We essentially share edge with two hexes below ("South East" and "South West").
                #   There is NO direct South neighbor in a standard brick grid?
                #   Wait.
                #       ___
                #      /   \
                #      \___/
                #      /   \
                #      \___/
                #   This stack is Pointy Topped. (Vertical columns).
                #   
                #   Flat Topped:
                #     / \
                #    |   | 
                #     \ /
                #   
                #   If I drew Flat Topped:
                #    __    __
                #   /  \__/  \
                #   \__/  \__/
                #
                #   Neighbors of center are:
                #   - North-East, North-West
                #   - East, West
                #   - South-East, South-West
                #   THERE IS NO "SOUTH" Neighbor in Flat Topped Grid!
                #   The "South" direction points to a VERTEX.
                
                #   User said "3 faces per hex... S, SE, SW".
                #   The faces of a Flat Topped Hex are:
                #     North (Top Edge)
                #     NW, NE
                #     South (Bottom Edge)
                #     SW, SE
                
                #   Wait, Flat Topped:
                #   Top edge is horizontal. Normal is (0,1) i.e. 90 deg (North).
                #   Bottom edge is horizontal. Normal is (0,-1) i.e. 270 deg (South).
                #   So there IS a South Face.
                #   But is there a South Neighbor?
                #   The neighbor sharing the South Face is directly South.
                #   Center: (0,0).
                #   South neighbor center: (0, -1.732 * R).
                #   
                #   But does a hex grid allow direct South neighbors?
                #   Usually Hex Grid = Honeycomb.
                #   Every node has degree 3? No, dual is triangles.
                #   Each hex has 6 neighbors.
                #   If Flat Topped:
                #     Neighbors are at angles: 30, 90, 150, 210, 270, 330.
                #     Wait.
                #     If neighbors are at 90 (N) and 270 (S), then we have columns aligned.
                #     If we have columns aligned, e.g. (0,0), (0,H), (0,2H)...
                #     Then the hexes to the side must be interleaved.
                #     (W, H/2).
                #   This is the layout!
                #   Centers:
                #     C(0,0).
                #     N(0, dY). S(0, -dY).
                #     NE(dX, dY/2). SE(dX, -dY/2).
                #     NW(-dX, dY/2). SW(-dX, -dY/2).
                #   This configuration matches "Flat Topped" visual (pointy sides, flat top/bottom).
                #   Neighbors at 90, 270, 30, 330, 150, 210.
                
                #   Coordinates for this layout:
                #   y_stride = sqrt(3) * R  (~8.66 for R=5)
                #   x_stride = 1.5 * R      (~7.5 for R=5) *CHECK THIS*
                
                #   Let's check distance:
                #   Center to NE (dX, dY/2). Distance must be sqrt(3)*R.
                #   dX^2 + dY^2/4 = 3R^2.
                #   If dY = sqrt(3)R:
                #     dX^2 + 3R^2/4 = 3R^2
                #     dX^2 = 2.25 R^2
                #     dX = 1.5 R.
                #   So:
                #   X Steps: 1.5 * R.
                #   Y Steps: sqrt(3) * R.
                #   Offset: Odd columns shifted vertical by Y_Step / 2.
                
                #   Wait, earlier I chose Odd-R (Rows).
                #   If I used Odd-R (Horizontal Strips):
                #     Rows are continuous bands of hexes.
                #     Centers (x, 0), (x+W, 0). W = sqrt(3) R.
                #     Next row: (x+W/2, H). H = 1.5 R.
                #     Neighbors:
                #       E, W (0, 180).
                #       NE, NW (60, 120).
                #       SE, SW (300, 240).
                #     NO "South" (270) neighbor.
                
                #   THE USER WANTS 3 FACES: SW, S, SE.
                #   This implies the SOUTH face exists.
                #   Therefore, we MUST use the layout with a South Neighbor.
                #   This layout is: COLUMNS (pointy side) with OFFSET.
                #   Also called "Odd-Q" (Column based).
                #   Layout:
                #     Cols are vertical stacks.
                #     Hexes have flat Tops/Bottoms.
                #     Col 0: (0,0), (0, H), (0, 2H)...
                #     Col 1: (W, H/2), (W, 1.5H)...
                
                #   So:
                #   H (vertical spacing) = sqrt(3) * R.
                #   W (horizontal spacing) = 1.5 * R.
                #   Neighbors of (0,0):
                #     N (0, H), S (0, -H) -> Neighbors at 90, 270.
                #     SE (W, -H/2). SW (-W, -H/2).
                #     NE (W, H/2). NW (-W, H/2).
                #   Angles:
                #     S: 270.
                #     SE: atan2(-H/2, W) = atan2(-sq3/2, 1.5) = atan2(-0.866, 1.5) = -30 deg (330).
                #     SW: atan2(-H/2, -W) = 210 deg.
                
                #   This perfectly matches "S, SE, SW".
                #   So we proceed with this layout: VERTICAL COLUMNS, FLAT TOPPED HEXES.
                
                #   Grid definition:
                #   Col Index `c`. Row Index `r`.
                #   x = c * 1.5 * R
                #   y = r * sqrt(3) * R - (sqrt(3)*R/2 if c is odd).
                #   Or simply:
                #   y = r * sqrt(3) * R + (c%2) * (sqrt(3)*R/2).
            
            # Re-defining constants for this layout
            R = HEX_RADIUS
            X_SPACING = 1.5 * R
            Y_SPACING = np.sqrt(3) * R
            
            # Recalculate loop bounds based on this new geometry
            padding = 2
            c_min = int(sat_bounds.left / X_SPACING) - padding
            c_max = int(sat_bounds.right / X_SPACING) + padding
            r_min = int(sat_bounds.bottom / Y_SPACING) - padding
            r_max = int(sat_bounds.top / Y_SPACING) + padding
            
        # Re-start loop with correct grid logic
        for c in range(c_min, c_max + 1):
            for r in range(r_min, r_max + 1):
                 # Grid calc
                cx = c * X_SPACING
                cy = r * Y_SPACING + ((c % 2) * (Y_SPACING / 2.0))
                
                # Skip if outside
                if not (sat_bounds.left - 5 <= cx <= sat_bounds.right + 5 and 
                        sat_bounds.bottom - 5 <= cy <= sat_bounds.top + 5):
                    continue
                
                z = sample_z(cx, cy)
                if np.isnan(z): continue
                
                if base_z is None: base_z = z
                
                # Neighbors (World Coords)
                # S: (c, r-1)
                cx_s, cy_s = cx, cy - Y_SPACING
                z_s = sample_z(cx_s, cy_s)
                
                # SE: (c+1, r) if c_even else (c+1, r-1)
                # Let's just calculate geometrically to be robust
                cx_se = cx + X_SPACING
                cy_se = cy - (Y_SPACING / 2.0)
                z_se = sample_z(cx_se, cy_se)
                
                # SW: (c-1, r) if c_even else (c-1, r-1)
                cx_sw = cx - X_SPACING
                cy_sw = cy - (Y_SPACING / 2.0)
                z_sw = sample_z(cx_sw, cy_sw)
                
                # Determine "Skirt Length" and Colors
                # If neighbor is missing (NaN), assume infinite drop (render full wall down to 0 or something)
                # But typically we clamp. Let's assume z_neigh = z - 20 if nan
                
                def safe_z(val, default):
                    return val if not np.isnan(val) else (default - 100.0) # Drop buffer

                # Lengths are simply Z - NeighborZ.
                # If Z < NeighborZ, length is negative, meaning wall is hidden (handled in shader).
                
                # Colors: Sample midway
                # Vector to face center is half-vector to neighbor
                rgb_s = get_color_at_point(sat_data, sat_trans, (cx + cx_s)/2, (cy + cy_s)/2)
                rgb_se = get_color_at_point(sat_data, sat_trans, (cx + cx_se)/2, (cy + cy_se)/2)
                rgb_sw = get_color_at_point(sat_data, sat_trans, (cx + cx_sw)/2, (cy + cy_sw)/2)
                
                # Pack
                # We save: Z_center, Z_S, Z_SE, Z_SW, RGBs
                hexes.append({
                    'z': float(z - base_z),
                    'z_s': float(safe_z(z_s, z) - base_z),
                    'z_se': float(safe_z(z_se, z) - base_z),
                    'z_sw': float(safe_z(z_sw, z) - base_z),
                    'rgb_s': rgb_s,
                    'rgb_se': rgb_se,
                    'rgb_sw': rgb_sw
                })
        
        if not hexes: return f"SKIP: {tif_path}"
        
        # Binary Format:
        # Header: BaseZ (float32)
        # Body: [Z (f16), Z_S (f16), Z_SE (f16), Z_SW (f16), RGB_S (3B), RGB_SE (3B), RGB_SW (3B)] = 8 + 9 = 17 bytes
        
        with open(out_path, 'wb') as f:
            f.write(struct.pack('f', base_z))
            for h in hexes:
                f.write(np.array([h['z'], h['z_s'], h['z_se'], h['z_sw']], dtype=np.half).tobytes())
                f.write(struct.pack('BBB', *h['rgb_s']))
                f.write(struct.pack('BBB', *h['rgb_se']))
                f.write(struct.pack('BBB', *h['rgb_sw']))
                
        return f"SUCCESS: {out_name} ({len(hexes)} hexes)"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"ERROR: {tif_path} -> {str(e)}"

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    tifs = sorted(glob.glob(os.path.join(SAT_DIR, "*.tif")))
    print(f"Baking PISTON Tiles: {len(tifs)} targets...")
    
    # Run in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_tile, tifs))
    
    for r in results: print(r)

if __name__ == "__main__":
    import glob
    main()
