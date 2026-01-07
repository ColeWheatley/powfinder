
import os
import glob
import math
import numpy as np
import rasterio
import rasterio.enums
import gc
from shapely.geometry import Polygon, box
from shapely.affinity import affine_transform
from multiprocessing import Pool, cpu_count
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import coordinate_utility as coord_util

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# 1. Coordinate System & Hex Dimensions
# -------------------------------------
# Driven by PIXEL-FIRST logic in coordinate_utility.py
# High-Res Texture: 4096px * 0.2m/px = 819.2m Sector Width

# 2. Border & Buffer Logic
# ------------------------
BUFFER_PX = 32 

# 3. Paths
# --------
# Assuming script runs from root
DEM_PATH = "hex_backend/DGM_Tirol_5m_epsg31254_2006_2020.tif" 
AERIAL_DIR = "hex_backend/aerial_tifs"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sector_bounds(sector_Q, sector_R):
    """
    Returns a Shapely Polygon representing the bounding box of the Level 5 Sector (Q, R).
    """
    cen_x, cen_y = coord_util.sector_to_world_meters(sector_Q, sector_R)
    
    # Conservative Radius: Level 5 Width is defined in coord_util. 
    # Safety factor 0.7 * Width covers the fractal edges comfortably.
    width = coord_util.SECTOR_WIDTH_METERS
    safety_radius = width * 0.7 
    
    # Approximate meters for 32px buffer
    buffer_meters = BUFFER_PX * coord_util.METERS_PER_PIXEL
    
    min_x = cen_x - safety_radius - buffer_meters
    max_x = cen_x + safety_radius + buffer_meters
    min_y = cen_y - safety_radius - buffer_meters
    max_y = cen_y + safety_radius + buffer_meters
    
    return box(min_x, min_y, max_x, max_y)

def bake_sector_textures(Q, R, valid_tifs, output_dir="hex_backend/baked_sectors"):
    """
    Bakes the WebP texture for a single sector and its LOD variants.
    
    Generates 3 files:
    1. _compressed (0.2m) - Full Res
    2. _high       (~0.8m) - 1/4 Scale
    3. _low        (~3.2m) - 1/16 Scale
    
    All saved at Quality=10 (User Request).
    """
    import PIL.Image as Image
    import PIL.ImageDraw as ImageDraw
    from rasterio.windows import from_bounds
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Define Canvas
    # Shape: Centered on Sector Center
    cen_x, cen_y = coord_util.sector_to_world_meters(Q, R)
    
    # Canvas Size: The Hex Width (Pixels) + 2x Buffer
    base_px = coord_util.TEXTURE_SIZE_PX # ~4148.5
    total_w_px = int(base_px + (BUFFER_PX * 2))
    total_h_px = int(base_px + (BUFFER_PX * 2)) 
    
    # Calculate World Bounds for this Canvas
    half_w_m = (total_w_px * coord_util.METERS_PER_PIXEL) / 2.0
    half_h_m = (total_h_px * coord_util.METERS_PER_PIXEL) / 2.0
    
    win_min_x = cen_x - half_w_m
    win_max_x = cen_x + half_w_m
    win_min_y = cen_y - half_h_m
    win_max_y = cen_y + half_h_m
    
    target_poly = box(win_min_x, win_min_y, win_max_x, win_max_y)

    # Create Transparent Canvas (RGBA)
    canvas = Image.new("RGBA", (total_w_px, total_h_px), (0, 0, 0, 0))
    
    # 2. Sample Data
    # Find intersecting TIFs
    intersecting = []
    for t in valid_tifs:
        if t['poly'].intersects(target_poly):
            intersecting.append(t)
            
    if not intersecting:
        print(f"⚠️ Sector {Q},{R}: No intersecting TIFs found.")
        return []

    # Paste Data
    for t in intersecting:
        with rasterio.open(t['path']) as src:
            ix_min_x = max(win_min_x, src.bounds.left)
            ix_max_x = min(win_max_x, src.bounds.right)
            ix_min_y = max(win_min_y, src.bounds.bottom)
            ix_max_y = min(win_max_y, src.bounds.top)
            
            if ix_min_x >= ix_max_x or ix_min_y >= ix_max_y:
                continue

            window = from_bounds(ix_min_x, ix_min_y, ix_max_x, ix_max_y, src.transform)
            
            w_px = int((ix_max_x - ix_min_x) / coord_util.METERS_PER_PIXEL)
            h_px = int((ix_max_y - ix_min_y) / coord_util.METERS_PER_PIXEL)
            
            if w_px <= 0 or h_px <= 0: continue
            
            try:
                data = src.read(window=window, out_shape=(src.count, h_px, w_px), resampling=rasterio.enums.Resampling.lanczos)
                img_array = np.moveaxis(data, 0, -1)
                
                if img_array.shape[2] == 3:
                    patch = Image.fromarray(img_array.astype('uint8'), 'RGB')
                elif img_array.shape[2] == 4:
                    patch = Image.fromarray(img_array.astype('uint8'), 'RGBA')
                else:
                    continue
                    
                paste_x = int((ix_min_x - win_min_x) / coord_util.METERS_PER_PIXEL)
                # Y flip because PIL is top-down, Map is bottom-up (usually)
                # But wait, TIFs are usually top-down in rasterio read?
                # Actually standard geotiff is +Y up? No, usually +Y down in pixel space, but +Y up in coord space.
                # Let's stick to the world-coord math:
                paste_y = int((win_max_y - ix_max_y) / coord_util.METERS_PER_PIXEL)
                
                canvas.paste(patch, (paste_x, paste_y))
                
            except Exception as e:
                print(f"Error reading TIF {t['path']}: {e}")

    # 3. Apply Hexagonal Mask (Crop to Shape + 32px Padding)
    # create a grayscale image to be the alpha channel
    mask = Image.new("L", (total_w_px, total_h_px), 0) # Black (Transparent)
    draw = ImageDraw.Draw(mask)
    
    # Center of Canvas
    cx, cy = total_w_px / 2.0, total_h_px / 2.0
    
    # Radius Calculation
    # TEXTURE_SIZE_PX is Flat-to-Flat Width
    # Hex Radius (Corner) = Width / sqrt(3)
    # We add BUFFER_PX to the radius to keep the padding
    hex_radius_px = (coord_util.TEXTURE_SIZE_PX / math.sqrt(3)) + BUFFER_PX
    
    # Vertices (Pointy Topped Hexagon implies 30, 90, 150...)
    # (If Flat Topped, angles are 0, 60, 120...)
    # coord_util defines 'Q' as horizontal? Usually standard Axial coordinates 
    # use Flat Topped if +Q is East. 
    # But usually Pointy Topped is better for packing?
    # Let's assume Pointy Top (angles 30..). If wrong, we rotate 30 deg.
    # Given the aspect ratio of the Sector (Width != Height), a Pointy Top hex 
    # fits a rectangular bounding box with ratio sqrt(3)/2 ~ 0.866.
    # Wait, Pointy Top is taller than wide.
    # Flat Top is wider than tall.
    # We used square canvas?
    # WaffleMaker uses Square Canvas based on Width.
    # If the sector is hexagonal, it fits in a square.
    # Let's just draw the Pointy Top hex.
    
    verts = []
    for i in range(6):
        deg = 30 + (60 * i)
        rad = math.radians(deg)
        vx = cx + hex_radius_px * math.cos(rad)
        vy = cy + hex_radius_px * math.sin(rad)
        verts.append((vx, vy))
        
    draw.polygon(verts, fill=255) # White (Opaque)
    
    # Apply to Alpha
    # The canvas is already RGBA. We want to Mask the Alpha.
    # If a pixel is transparent in canvas, it remains transparent.
    # If opaque, it becomes (Alpha * Mask).
    
    # Simple way: canvas.putalpha(mask) replaces the ENTIRE alpha channel.
    # But our TIF pasting might have empty spots (0 alpha).
    # If we putalpha(mask), empty spots inside the hex become Opaque Black? No.
    # putalpha replaces the alpha channel values.
    # We should Multiply?
    # Or:
    final_img = Image.new("RGBA", canvas.size)
    final_img.paste(canvas, (0,0), mask=mask)
    canvas = final_img
    
    # Create Subfolders
    res_dirs = {
        "compressed": os.path.join(output_dir, "compressed"),
        "high_res": os.path.join(output_dir, "high_res"),
        "low_res": os.path.join(output_dir, "low_res")
    }
    for d in res_dirs.values():
        if not os.path.exists(d):
            os.makedirs(d)

    # 3. Export Variants
    outputs = []
    
    # Variant 1: Full Res (Source 0.2m) -> "compressed/"
    f_full = f"sector_{Q}_{R}.webp"
    p_full = os.path.join(res_dirs["compressed"], f_full)
    canvas.save(p_full, "WEBP", quality=10)
    outputs.append(("Compressed (0.2m)", p_full, total_w_px))
    
    # Variant 2: High Res (~0.8m) -> "high_res/"
    w_high = int(total_w_px / 4)
    h_high = int(total_h_px / 4)
    c_high = canvas.resize((w_high, h_high), Image.LANCZOS)
    p_high = os.path.join(res_dirs["high_res"], f_full)
    c_high.save(p_high, "WEBP", quality=10)
    outputs.append(("High (~0.8m)", p_high, w_high))
    
    
    # Variant 4: Low Res (~3.2m) -> "low_res/"
    w_low = int(total_w_px / 16)
    h_low = int(total_h_px / 16)
    c_low = canvas.resize((w_low, h_low), Image.LANCZOS)
    p_low = os.path.join(res_dirs["low_res"], f_full)
    c_low.save(p_low, "WEBP", quality=10)
    outputs.append(("Low (~3.2m)", p_low, w_low))

    return outputs, canvas

def bake_sector_tifs(Q, R, canvas, output_dir="hex_backend/baked_sectors"):
    """
    Saves a TIF version of the sector texture.
    NOTE: Many browsers do not support TIF natively, but they can be 
    smaller than high-quality WebPs in some geo-scenarios.
    """
    f_tif = f"sector_{Q}_{R}_source.tif"
    p_tif = os.path.join(output_dir, f_tif)
    canvas.save(p_tif, "TIFF")
    return p_tif

# =============================================================================
# ELEVATION BAKING (THE "BIN" LOGIC)
# =============================================================================

# Cache the 16,807 offsets relative to a Sector Center
# Shape: (16807, 2) array of [dq, dr]
CACHED_GOSPER_OFFSETS = None

def get_cached_offsets():
    global CACHED_GOSPER_OFFSETS
    if CACHED_GOSPER_OFFSETS is None:
        print("⚙️ Generating Fractal Tree (Level 5)...", end="")
        offsets = coord_util.generate_gosper_offsets(5) # 16,807 points
        CACHED_GOSPER_OFFSETS = np.array(offsets)
        print(f" Done. {len(offsets)} nodes.")
    return CACHED_GOSPER_OFFSETS

def precompute_sector_heights_manual(Q, R, transform):
    """
    Optimized DEM Sampler using a specific affine transform.
    """
    offsets = get_cached_offsets()
    center_q = -87 * Q - 149 * R
    center_r = 149 * Q + 62 * R
    
    qs = offsets[:, 0] + center_q
    rs = offsets[:, 1] + center_r
    
    h = coord_util.UNIT_HEX_WIDTH_METERS
    xs = qs * (math.sqrt(3)/2 * h)
    ys = rs * h + qs * (0.5 * h)
    
    cols, rows = rasterio.transform.rowcol(transform, xs, ys)
    return np.array(rows), np.array(cols)

# Cache Neighbor Tables for each Level (Unit, Parent, Grand...)
# { level: np.array(N, 6) }
CACHED_NEIGHBOR_TABLES = {}

def get_neighbor_table(level):
    if level not in CACHED_NEIGHBOR_TABLES:
        print(f"⚙️ Generating Neighbor Table for Level {level}...", end="")
        table = coord_util.compute_gosper_neighbors(level)
        CACHED_NEIGHBOR_TABLES[level] = table
        print(f" Done. {len(table)} nodes.")
    return CACHED_NEIGHBOR_TABLES[level]

def pack_level(level_idx, heights, min_z, scale_factor):
    """
    Compresses a level's height array into the 8-Byte struct.
    Struct: [Uint16 H, Int8 N1, Int8 N2... Int8 N6]
    """
    count = len(heights)
    neighbor_indices = get_neighbor_table(level_idx)
    
    # 1. Quantize Heights -> Uint16
    # val = (h - min) * scale_factor
    # scale_factor = 65535 / (max - min)
    h_norm = (heights - min_z) * scale_factor
    h_u16 = np.clip(h_norm, 0, 65535).astype(np.uint16)
    
    # 2. Compute Skirts (Neighbor Deltas)
    # We need the quantized height of the neighbors to match visual cracks exactly?
    # Or just use the source float diff? 
    # Using source float diff is safer for physics.
    # Storing: Delta in 0.5m units. Range +/- 64m.
    # Delta = Neighbor - Self (Convention: Height at edge relative to center)
    
    # Prepare Output Buffer: (N, 8) bytes
    # But numpy struct arrays are tricky. Let's make separate arrays and stack.
    
    # Neighbors Matrix (N, 6) of indices
    # We want heights[neighbors]. If neighbor == -1, use self height (delta 0).
    
    # Handle -1 indices safely:
    # Create padded heights with strict boundary value?
    # Actually, if index is -1, we want delta=0.
    # Let's replace -1 with 'current index' in the lookup so diff is 0.
    
    self_indices = np.arange(count)
    safe_neighbors = neighbor_indices.copy()
    
    # Mask for valid neighbors
    invalid_mask = (safe_neighbors == -1)
    
    # Temporarily replace -1 with 0 to safely fetch, then zero out diffs
    safe_neighbors[invalid_mask] = 0 
    
    # Fetch Neighbor Heights
    n_heights = heights[safe_neighbors] # Shape (N, 6)
    
    # Correct the values where neighbor was -1 to be equal to self
    # Actually simpler: Calculate diff, then mask.
    
    # Self heights expanded for broadcasting: (N, 1)
    self_h_col = heights.reshape(-1, 1)
    
    diffs = n_heights - self_h_col # (N, 6)
    
    # Apply -1 mask: if neighbor was missing, diff is 0
    # But wait, safe_neighbors used index 0 for invalid.
    # Row i used index 0. If neighbor was -1, we fetched heights[0] instead of heights[i].
    # This is WRONG behavior. We wanted heights[i].
    # FIX:
    # Use fancy indexing more carefully.
    # or just: where(mask, self_h, n_heights)
    
    actual_n_heights = n_heights # This has heights[0] at invalid spots
    # We need to construct a matrix of "Correct Neighbor Heights"
    # If invalid, use self_h.
    
    # Correction:
    # We can iterate cols?
    # Or optimize:
    # Just fix the delta.
    
    # Let's map delta to Int8
    # 1 unit = 0.5m
    delta_units = diffs / 0.5
    delta_i8 = np.clip(delta_units, -127, 127).astype(np.int8)
    
    # Now fix the invalid ones to 0
    delta_i8[invalid_mask] = 0
    
    # 3. Interleave / Pack
    # We want: H(2), N1(1), N2(1)...
    # Create raw byte buffer
    buffer = np.zeros(count * 8, dtype=np.uint8)
    
    # View as custom dtype
    # dtype = [('h', '<u2'), ('n', '6i1')]
    # This might be padding sensitive.
    # Safer: Assign byte slices.
    
    # H (Uint16) -> Bytes 0,1
    # Little Endian standard
    h_bytes = h_u16.tobytes()
    buffer[0::8] = np.frombuffer(h_bytes[0::2], dtype=np.uint8)
    buffer[1::8] = np.frombuffer(h_bytes[1::2], dtype=np.uint8)
    
    # Neighbors -> Bytes 2..7
    for k in range(6):
        col_bytes = delta_i8[:, k].tobytes() # shape N
        buffer[(2+k)::8] = np.frombuffer(col_bytes, dtype=np.uint8)
        
    return buffer

def bake_sector_binary(Q, R, dem_array, transform, output_dir="frontend/hexagons/app/tiles_bin"):
    """
    Bakes the 'Universal Bin' for a sector (v3.0 - Grid Packed).
    Matches the Frontend Grid iteration order (Col-Major).
    """
    t_start = time.time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Define Grid Dimensions (Matching Frontend)
    # Frontend: TILE_WIDTH_WORLD based on SECTOR_WIDTH_METERS
    # Iterates x from 0 to Width (inclusive of boundary?)
    # Frontend logic:
    # xSteps = [0, dx, 2dx ... <= Width+1]
    # ySteps = [0, dy, 2dy ... <= Height+1]
    
    # We need to calculate exactly how many cols/rows the frontend expects.
    # Unit Hex Width ~ 6.4m.
    # Sector Width ~ 829.7m.
    
    # Let's derive it exactly from coordinate_utility
    w_m = coord_util.SECTOR_WIDTH_METERS
    h_m = coord_util.SECTOR_WIDTH_METERS # Square tile bounds
    
    unit_w = coord_util.UNIT_HEX_WIDTH_METERS 
    
    y_spacing = unit_w
    x_spacing = unit_w * (math.sqrt(3) / 2.0)
    
    # Number of steps
    # for (let x = 0; x <= TILE_WIDTH_WORLD + 1; x += currentTileXSpacing)
    num_cols = int((w_m + 1.0) / x_spacing) + 1
    num_rows = int((h_m + 1.0) / y_spacing) + 1
    
    # 2. Collect Heights in Grid Order (Col-Major)
    # Frontend Loop: Col (X) outer, Row (Y) inner.
    
    # Center of Sector in World Coords
    cen_x, cen_y = coord_util.sector_to_world_meters(Q, R)
    
    # Tile Origin
    origin_x = cen_x - (w_m / 2.0)
    origin_y = cen_y - (h_m / 2.0) # Bottom-Left in 2D?
    
    # Sample points.
    
    # Pre-calculate world coordinates for all points
    sample_xs = []
    sample_ys = []
    
    for col in range(num_cols):
        x_local = col * x_spacing
        y_shift = (y_spacing / 2.0) if (col % 2 == 1) else 0.0
        
        for row in range(num_rows):
            y_local = row * y_spacing + y_shift
            
            # Map to World
            # Frontend increasing row adds y_spacing.
            # Assuming standard North+ coords for DEM sampling.
            wx = origin_x + x_local
            wy = origin_y + y_local 
            
            sample_xs.append(wx)
            sample_ys.append(wy)
            
    # Batch Sample
    rows_idx, cols_idx = rasterio.transform.rowcol(transform, sample_xs, sample_ys)
    
    # Read from pre-loaded DEM array
    # Safety clip
    r_idx = np.clip(rows_idx, 0, dem_array.shape[0]-1)
    c_idx = np.clip(cols_idx, 0, dem_array.shape[1]-1)
    
    h_values = dem_array[r_idx, c_idx] # Flat array of N items
    
    # 3. Calculate Scale
    sector_min = float(h_values.min())
    sector_max = float(h_values.max())
    sector_min -= 10.0
    sector_max += 10.0
    h_range = sector_max - sector_min
    if h_range < 1.0: h_range = 1.0
    scale = 65535.0 / h_range
    
    # 4. Pack Buffer (V3)
    final_blob = bytearray()
    
    # Header: HEX3 (4) + Q(4) + R(4) + Min(4) + Max(4) + Scale(4) 
    # + GridW(2) + GridH(2) + Pad(4) = 32 Bytes
    import struct
    header = struct.pack('<4siiffHH4x', 
        b'HEX3', int(Q), int(R), sector_min, sector_max, scale, 
        num_cols, num_rows
    )
    final_blob.extend(header)
    
    # Need neighbor lookups in the grid.
    # Reshape heights to (Cols, Rows) for easy indexing
    grid_h = h_values.reshape((num_cols, num_rows))
    
    # Helper to get neighbor height safely
    def get_h(c, r, default):
        if 0 <= c < num_cols and 0 <= r < num_rows:
            return grid_h[c, r]
        return default
        
    for c in range(num_cols):
        is_odd = (c % 2 == 1)
        for r in range(num_rows):
            z_self = grid_h[c, r]
            
            # Encoded Height
            h_u16 = int(np.clip((z_self - sector_min) * scale, 0, 65535))
            final_blob.extend(struct.pack('<H', h_u16))
            
            # Neighbors (N, NE, SE, S, SW, NW)
            # Based on standard Grid logic used in Frontend or visual consistency
            if not is_odd:
                n_coords = [
                    (c, r+1),   # N
                    (c+1, r),   # NE
                    (c+1, r-1), # SE
                    (c, r-1),   # S
                    (c-1, r-1), # SW
                    (c-1, r)    # NW
                ]
            else:
                 n_coords = [
                    (c, r+1),   # N
                    (c+1, r+1), # NE
                    (c+1, r),   # SE
                    (c, r-1),   # S
                    (c-1, r),   # SW
                    (c-1, r+1)  # NW
                ]
                
            for (nc, nr) in n_coords:
                nz = get_h(nc, nr, z_self)
                diff = nz - z_self
                d_byte = int(np.clip(diff / 0.5, -127, 127))
                final_blob.extend(struct.pack('b', d_byte))
                
    # 5. Write
    filename = f"sector_{Q}_{R}.bin"
    path = os.path.join(output_dir, filename)
    with open(path, 'wb') as f:
        f.write(final_blob)
        
    duration = time.time() - t_start
    return path, len(final_blob), duration

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print(f"🧇 INIT: Waffle Maker spinning up...")
    dims = coord_util.get_hex_dimensions()
    print(f"      - High Res Texture: {coord_util.TEXTURE_SIZE_PX}px")
    print(f"      - Nominal Res: {coord_util.METERS_PER_PIXEL} m/px")
    print(f"      - Sector Width: {dims['sector_width_m']:.4f}m (Derived)")
    print(f"      - Unit Hex Width: {dims['unit_hex_width_m']:.4f}m (derived)")
    print(f"      - Pixels Per Unit: {dims['pixels_per_unit_hex']:.2f}")
    
    # 1. Load DEM Metadata
    # --------------------
    if not os.path.exists(DEM_PATH):
        print(f"❌ ERROR: DEM not found at {DEM_PATH}")
        return

    with rasterio.open(DEM_PATH) as dem:
        dem_bounds = dem.bounds
        dem_poly = box(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)
        print(f"✅ DEM Loaded: {dem.width}x{dem.height} | Bounds: {dem_bounds}")

    # 2. Load TIF Metadata (Parallel Scan)
    # ------------------------------------
    print("🔍 Scanning Aerial TIFs...")
    tif_files = glob.glob(os.path.join(AERIAL_DIR, "*.tif"))
    if not tif_files:
         print(f"❌ ERROR: No TIFs found in {AERIAL_DIR}")
         return 
    
    valid_tifs = []
    print(f"✅ Found {len(tif_files)} total TIFs.")
    
    # We collect all TIF polygons to create a "Coverage Map"
    all_tif_polys = []
    
    for f in tif_files: 
        try:
            with rasterio.open(f) as src:
                p = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                all_tif_polys.append(p)
                valid_tifs.append({'path': f, 'poly': p})
        except:
            pass
    
    print(f"✅ Indexed {len(valid_tifs)} valid TIF bounds.")
    
    if not valid_tifs:
        print("⚠️ No valid TIFs to process. Exiting.")
        return

    # 3. Cross-Check & Sector Discovery
    # ---------------------------------
    from shapely.ops import unary_union
    print("🕸️ Building Unified Coverage Map (this may take a second)...")
    coverage_poly = unary_union(all_tif_polys)
    
    print("🛠 Calculating Valid Sectors...")
    
    # Intersection with DEM
    c_bounds = coverage_poly.bounds
    search_min_x = max(dem_bounds.left, c_bounds[0])
    search_max_x = min(dem_bounds.right, c_bounds[2])
    search_min_y = max(dem_bounds.bottom, c_bounds[1])
    search_max_y = min(dem_bounds.top, c_bounds[3])
    
    print(f"   Search Area: X[{search_min_x:.1f}, {search_max_x:.1f}] Y[{search_min_y:.1f}, {search_max_y:.1f}]")
    
    # Use the Module's Approx Inversion
    Q1, R1 = coord_util.world_meters_to_sector_approx(search_min_x, search_min_y)
    Q2, R2 = coord_util.world_meters_to_sector_approx(search_max_x, search_max_y)
    Q3, R3 = coord_util.world_meters_to_sector_approx(search_min_x, search_max_y)
    Q4, R4 = coord_util.world_meters_to_sector_approx(search_max_x, search_min_y)

    Q_start = int(min(Q1, Q2, Q3, Q4)) - 2
    Q_end = int(max(Q1, Q2, Q3, Q4)) + 2
    R_start = int(min(R1, R2, R3, R4)) - 2
    R_end = int(max(R1, R2, R3, R4)) + 2
    
    print(f"   Scanning Sector Range: Q[{Q_start} to {Q_end}] R[{R_start} to {R_end}]")
    
    valid_sectors = []
    partial_sectors = []
    
    for Q in range(Q_start, Q_end + 1):
        for R in range(R_start, R_end + 1):
            sec_poly = get_sector_bounds(Q, R)
            
            # Must be completely inside DEM
            if not dem_poly.contains(sec_poly):
                continue
                
            # Check against Coverage Map
            if coverage_poly.contains(sec_poly):
                valid_sectors.append((Q, R))
            elif coverage_poly.intersects(sec_poly):
                partial_sectors.append((Q, R))

    print("\n📊 ANALYSIS REPORT")
    print(f"   Total Coverage Area: {coverage_poly.area/1e6:.1f} km²")
    print(f"   Target Location: Q[{Q1:.1f}], R[{R1:.1f}]")
    print(f"   Potential Hex Sectors (Full): {len(valid_sectors)}")
    print(f"   Partial Sectors (Border): {len(partial_sectors)}")
    print(f"   Optimal Hex Hexes (Unit): {len(valid_sectors) * 16807:,}")
    print(f"   Pixel-Verified: YES")

    # =============================================================================
    # 4. FULL BAKE: WebP Textures
    # =============================================================================
    if valid_sectors:
        print(f"\n🧑‍🍳 COOKING: Baking {len(valid_sectors)} Sectors...")
        
        baked_count = 0
        total_sectors = len(valid_sectors)
        
        for Q, R in valid_sectors:
            baked_count += 1
            print(f"[{baked_count}/{total_sectors}] Baking Q={Q}, R={R}...", end="\r")
            
            # TEXTURE BAKE (Skipping to focus on Elevation Logic)
            # results, canvas = bake_sector_textures(Q, R, valid_tifs, output_dir="frontend/hexagons/app/aerial_tiles")
            
            # TIF BAKE (Commented out per request)
            # path_tif = bake_sector_tifs(Q, R, canvas, output_dir="frontend/hexagons/app/aerial_tiles")
            
    # =============================================================================
    # 5. ELEVATION LOGIC PREVIEW (The "Waffle Batter")
    # =============================================================================
    print("\n🥞 ELEVATION LOGIC PREVIEW (Tree Logic)")
    
    TARGET_RES = 3.5 # Upscale from 5m to 3.5m to fit ~10GB RAM
    
    print(f"   Upscaling DEM to {TARGET_RES}m (Targeting ~10GB RAM)...")
    with rasterio.open(DEM_PATH) as src:
        # Calculate new shape
        new_w = int((src.bounds.right - src.bounds.left) / TARGET_RES)
        new_h = int((src.bounds.top - src.bounds.bottom) / TARGET_RES)
        
        # Read with resampling
        print(f"   Reading and Resampling {new_w}x{new_h}...", end="", flush=True)
        dem_data = src.read(
            1, 
            out_shape=(new_h, new_w), 
            resampling=rasterio.enums.Resampling.lanczos
        )
        upscaled_transform = rasterio.transform.from_bounds(
            src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top, 
            new_w, new_h
        )
        print(f" Done. {dem_data.nbytes / 1e9:.2f} GB used.")

        # =============================================================================
        # 6. FULL PRODUCTION BAKE
        # =============================================================================
        if valid_sectors:
            print(f"\n🏭 FACTORY MODE: Baking {len(valid_sectors)} Sectors (Texture + Geometry)...")
            
            baked_count = 0
            total_sectors = len(valid_sectors)
            
            for Q, R in valid_sectors:
                baked_count += 1
                print(f"[{baked_count}/{total_sectors}] Baking Q={Q}, R={R}...", end="\r")
                
                # A. TEXTURES
                results, canvas = bake_sector_textures(Q, R, valid_tifs, output_dir="frontend/hexagons/app/aerial_tiles")
                
                # B. GEOMETRY (BIN)
                bin_path, bin_size = bake_sector_binary(Q, R, dem_data, upscaled_transform, output_dir="frontend/hexagons/app/tiles_bin")
                
                # Explicit Cleanup of Canvas
                del canvas
                
                if baked_count % 10 == 0:
                    gc.collect()
                    
            print(f"\n✅ SUCCESS! All {baked_count} Sectors served.")
            print(f"   - Textures: 'frontend/hexagons/app/aerial_tiles/'")
            print(f"   - Geometry: 'frontend/hexagons/app/tiles_bin/'")

    # Final cleanup
    del dem_data
    gc.collect()

if __name__ == "__main__":
    main()
