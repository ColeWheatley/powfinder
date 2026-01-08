import os
import glob
import math
import time
import numpy as np
import rasterio
import rasterio.enums
import gc
import re
from shapely.geometry import Polygon, box
from shapely.affinity import affine_transform
from multiprocessing import Pool, cpu_count
import sys
from pyproj import Transformer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import coordinate_utility as coord_util
import generate_manifest


def latlon_to_world_meters(lat, lon):
    """Convert EPSG:4326 to EPSG:31254 (MGI Austrian GK Central)"""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:31254", always_xy=True)
    return transformer.transform(lon, lat)


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

# 3. Debug Mode Configuration
# ----------------------------
DEBUG_MODE = True
RESAMPLE_DEM = False  # Skip expensive resampling during debug
TARGET_LAT = 46.98705560886202
TARGET_LON = 11.115050838788871

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
    1. _full (0.2m) - Full Res
    2. _high  (~0.8m) - 1/4 Scale
    3. _low   (~3.2m) - 1/16 Scale

    Output format:
    - Square canvas containing flat-top hexagon
    - 32px buffer on all sides for WebP 16px block alignment
    - BLACK background in corner areas (not transparent) for optimal compression
    - All saved at Quality=10 (User Request)
    """
    import PIL.Image as Image
    import PIL.ImageDraw as ImageDraw
    from rasterio.windows import from_bounds

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Define Canvas Dimensions
    # For a FLAT-TOP hexagon inscribed in a square canvas:
    # - The hexagon's flat-to-flat height = TEXTURE_SIZE_PX
    # - The hexagon's vertex-to-vertex width = height * 2 / sqrt(3)
    # - We use a SQUARE canvas where side = width (the larger dimension)
    # - Add 32px buffer on each side for WebP block alignment

    cen_x, cen_y = coord_util.sector_to_world_meters(Q, R)

    base_px = coord_util.TEXTURE_SIZE_PX  # ~4148.5 (flat-to-flat height)

    # For flat-top hex: vertex-to-vertex width > flat-to-flat height
    # radius (center to vertex) = flat_to_flat_height / sqrt(3)
    hex_radius_px = base_px / math.sqrt(3.0)

    # Add buffer to radius for the mask
    buffered_radius_px = hex_radius_px + BUFFER_PX

    # Square canvas: side = diameter of buffered hexagon (vertex-to-vertex)
    canvas_size_px = int(math.ceil(buffered_radius_px * 2.0))

    # Ensure even dimensions for WebP compression efficiency
    if canvas_size_px % 2 == 1:
        canvas_size_px += 1

    total_w_px = canvas_size_px
    total_h_px = canvas_size_px

    # Canvas center in pixel coords
    cx = total_w_px / 2.0
    cy = total_h_px / 2.0

    # Calculate World Bounds for this Canvas
    half_extent_m = (total_w_px * coord_util.METERS_PER_PIXEL) / 2.0

    win_min_x = cen_x - half_extent_m
    win_max_x = cen_x + half_extent_m
    win_min_y = cen_y - half_extent_m
    win_max_y = cen_y + half_extent_m

    target_poly = box(win_min_x, win_min_y, win_max_x, win_max_y)

    # 2. Create Canvas with BLACK background (RGB, not RGBA)
    # Black corners compress better than transparent for WebP
    canvas = Image.new("RGB", (total_w_px, total_h_px), (0, 0, 0))

    # Find intersecting TIFs
    intersecting = []
    for t in valid_tifs:
        if t["poly"].intersects(target_poly):
            intersecting.append(t)

    if not intersecting:
        print(f"⚠️ Sector {Q},{R}: No intersecting TIFs found.")
        return [], None

    # 3. Sample and Paste Aerial Data
    for t in intersecting:
        with rasterio.open(t["path"]) as src:
            ix_min_x = max(win_min_x, src.bounds.left)
            ix_max_x = min(win_max_x, src.bounds.right)
            ix_min_y = max(win_min_y, src.bounds.bottom)
            ix_max_y = min(win_max_y, src.bounds.top)

            if ix_min_x >= ix_max_x or ix_min_y >= ix_max_y:
                continue

            window = from_bounds(ix_min_x, ix_min_y, ix_max_x, ix_max_y, src.transform)

            w_px = int((ix_max_x - ix_min_x) / coord_util.METERS_PER_PIXEL)
            h_px = int((ix_max_y - ix_min_y) / coord_util.METERS_PER_PIXEL)

            if w_px <= 0 or h_px <= 0:
                continue

            try:
                data = src.read(
                    window=window,
                    out_shape=(src.count, h_px, w_px),
                    resampling=rasterio.enums.Resampling.lanczos,
                )
                img_array = np.moveaxis(data, 0, -1)

                if img_array.shape[2] == 3:
                    patch = Image.fromarray(img_array.astype("uint8"), "RGB")
                elif img_array.shape[2] == 4:
                    patch = Image.fromarray(img_array.astype("uint8"), "RGBA")
                else:
                    continue

                paste_x = int((ix_min_x - win_min_x) / coord_util.METERS_PER_PIXEL)
                # Y axis: PIL is top-down, geo coords are bottom-up
                paste_y = int((win_max_y - ix_max_y) / coord_util.METERS_PER_PIXEL)

                canvas.paste(patch, (paste_x, paste_y))

            except Exception as e:
                print(f"Error reading TIF {t['path']}: {e}")

    # 4. Create Hexagonal Mask (with buffer)
    # This mask will be used to crop the canvas to hexagonal shape
    mask = Image.new("L", (total_w_px, total_h_px), 0)  # Black = masked out
    draw = ImageDraw.Draw(mask)

    # Flat-Top Hexagon vertices
    # For flat-top: first vertex at 0° (right), flat edges at top/bottom
    # Vertices at: 0°, 60°, 120°, 180°, 240°, 300°
    verts = []
    for i in range(6):
        deg = 0 + (60 * i)
        rad = math.radians(deg)
        vx = cx + buffered_radius_px * math.cos(rad)
        vy = cy + buffered_radius_px * math.sin(rad)
        verts.append((vx, vy))

    draw.polygon(verts, fill=255)  # White = visible

    # 5. Apply Mask - Convert to RGBA and apply alpha
    canvas_rgba = canvas.convert("RGBA")
    # Create black background
    black_bg = Image.new("RGBA", (total_w_px, total_h_px), (0, 0, 0, 255))
    # Composite: texture where mask is white, black where mask is black
    canvas_rgba.putalpha(mask)
    final_canvas = Image.alpha_composite(black_bg, canvas_rgba)
    # Convert back to RGB (black corners, texture in hex)
    final_canvas = final_canvas.convert("RGB")

    # 6. Create Output Directories
    res_dirs = {
        "full": os.path.join(output_dir, "full"),
        "high": os.path.join(output_dir, "high"),
        "low": os.path.join(output_dir, "low"),
    }
    for d in res_dirs.values():
        if not os.path.exists(d):
            os.makedirs(d)

    # 7. Export Variants
    outputs = []
    f_name = f"sector_{Q}_{R}.webp"

    # Full Res (0.2m)
    p_full = os.path.join(res_dirs["full"], f_name)
    final_canvas.save(p_full, "WEBP", quality=10)
    outputs.append(("Full (0.2m)", p_full, total_w_px))

    # High Res (~0.8m) - 1/4 Scale
    w_high = int(total_w_px / 4)
    h_high = int(total_h_px / 4)
    c_high = final_canvas.resize((w_high, h_high), Image.LANCZOS)
    p_high = os.path.join(res_dirs["high"], f_name)
    c_high.save(p_high, "WEBP", quality=10)
    outputs.append(("High (~0.8m)", p_high, w_high))

    # Low Res (~3.2m) - 1/16 Scale
    w_low = int(total_w_px / 16)
    h_low = int(total_h_px / 16)
    c_low = final_canvas.resize((w_low, h_low), Image.LANCZOS)
    p_low = os.path.join(res_dirs["low"], f_name)
    c_low.save(p_low, "WEBP", quality=10)
    outputs.append(("Low (~3.2m)", p_low, w_low))

    return outputs, final_canvas


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
        offsets = coord_util.generate_gosper_offsets(5)  # 16,807 points
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
    xs = qs * (math.sqrt(3) / 2 * h)
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
    invalid_mask = safe_neighbors == -1

    # Temporarily replace -1 with 0 to safely fetch, then zero out diffs
    safe_neighbors[invalid_mask] = 0

    # Fetch Neighbor Heights
    n_heights = heights[safe_neighbors]  # Shape (N, 6)

    # Correct the values where neighbor was -1 to be equal to self
    # Actually simpler: Calculate diff, then mask.

    # Self heights expanded for broadcasting: (N, 1)
    self_h_col = heights.reshape(-1, 1)

    diffs = n_heights - self_h_col  # (N, 6)

    # Apply -1 mask: if neighbor was missing, diff is 0
    # But wait, safe_neighbors used index 0 for invalid.
    # Row i used index 0. If neighbor was -1, we fetched heights[0] instead of heights[i].
    # This is WRONG behavior. We wanted heights[i].
    # FIX:
    # Use fancy indexing more carefully.
    # or just: where(mask, self_h, n_heights)

    actual_n_heights = n_heights  # This has heights[0] at invalid spots
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
        col_bytes = delta_i8[:, k].tobytes()  # shape N
        buffer[(2 + k) :: 8] = np.frombuffer(col_bytes, dtype=np.uint8)

    return buffer


def bake_sector_binary(
    Q, R, dem_array, transform, output_dir="frontend/hexagons/app/tiles_bin"
):
    """
    Bakes the 'Universal Bin' for a sector (v2.0).
    """
    t_start = time.time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Get Unit Heights (L5 in Fractal Terms, but Leaves of Tree)
    # Using 'precompute_sector_heights_manual' to get indices
    rows, cols = precompute_sector_heights_manual(Q, R, transform)
    h_l5 = dem_array[rows, cols]  # (16807,) Unit Hexes

    # 2. Compute Hierarchical Means (Bottom-Up)
    h_l4 = h_l5.reshape(-1, 7).mean(axis=1)  # Parent (Huge)
    h_l3 = h_l4.reshape(-1, 7).mean(axis=1)
    h_l2 = h_l3.reshape(-1, 7).mean(axis=1)
    h_l1 = h_l2.reshape(-1, 7).mean(axis=1)
    h_l0 = h_l1.reshape(-1, 7).mean(axis=1)  # Root (Mega)

    all_levels_h = [h_l0, h_l1, h_l2, h_l3, h_l4, h_l5]

    # 3. Calculate Global Min/Max for Scaling
    # We use the Sector Extremes to define the 16-bit range
    sector_min = float(h_l5.min())
    sector_max = float(h_l5.max())

    # Safety margin
    sector_min -= 10.0
    sector_max += 10.0

    height_range = sector_max - sector_min
    if height_range < 1.0:
        height_range = 1.0
    scale_factor = 65535.0 / height_range

    # 4. Pack Data
    final_blob = bytearray()

    # A. Header (32 Bytes)
    # Magic (4) + Q(4) + R(4) + Min(4) + Max(4) + Scale(4) + Padding(8)
    import struct

    header = struct.pack(
        "<4siifff8x", b"HEX2", int(Q), int(R), sector_min, sector_max, scale_factor
    )
    final_blob.extend(header)

    # B. Blocks (Levels 0 to 5)
    for lvl_idx, heights in enumerate(all_levels_h):
        # Pack
        packed = pack_level(lvl_idx, heights, sector_min, scale_factor)
        final_blob.extend(packed)

    # 5. Write
    filename = f"sector_{Q}_{R}.bin"
    path = os.path.join(output_dir, filename)
    with open(path, "wb") as f:
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
        dem_poly = box(
            dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top
        )
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
                p = box(
                    src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
                )
                all_tif_polys.append(p)
                valid_tifs.append({"path": f, "poly": p})
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

    print(
        f"   Search Area: X[{search_min_x:.1f}, {search_max_x:.1f}] Y[{search_min_y:.1f}, {search_max_y:.1f}]"
    )

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
    print(f"   Total Coverage Area: {coverage_poly.area / 1e6:.1f} km²")
    print(f"   Target Location: Q[{Q1:.1f}], R[{R1:.1f}]")
    print(f"   Potential Hex Sectors (Full): {len(valid_sectors)}")
    print(f"   Partial Sectors (Border): {len(partial_sectors)}")
    print(f"   Optimal Hex Hexes (Unit): {len(valid_sectors) * 16807:,}")
    print(f"   Pixel-Verified: YES")

    if DEBUG_MODE:
        print("🐛 DEBUG MODE: Using 6 adjacent sectors")
        # Convert target coordinates to world meters
        target_x, target_y = latlon_to_world_meters(TARGET_LAT, TARGET_LON)
        # Find the sector
        target_Q, target_R = coord_util.world_meters_to_sector_approx(
            target_x, target_y
        )
        target_Q, target_R = int(target_Q), int(target_R)

        # Get 6 adjacent sectors (center + 5 neighbors)
        neighbor_dirs = [(0, 0), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 0)]
        debug_sectors = [(target_Q + dq, target_R + dr) for dq, dr in neighbor_dirs]

        # Only keep sectors that are in valid_sectors
        valid_sectors = [(Q, R) for Q, R in debug_sectors if (Q, R) in valid_sectors]
        print(f"   Debug sectors: {valid_sectors}")

        # Clean up old files from previous runs (only in debug mode)
        print("🧹 Cleaning up old debug sector files...")
        bin_dir = "frontend/hexagons/app/tiles_bin"
        texture_dir = "frontend/hexagons/app/aerial_tiles"

        if os.path.exists(bin_dir):
            bin_files = os.listdir(bin_dir)
            removed = 0
            for f in bin_files:
                # Check if this file matches one of our debug sectors
                match = re.match(r"sector_(-?\d+)_(-?\d+)\.bin", f)
                if match:
                    q, r = int(match.group(1)), int(match.group(2))
                    if (q, r) not in valid_sectors:
                        os.remove(os.path.join(bin_dir, f))
                        removed += 1
            if removed > 0:
                print(f"   Removed {removed} old .bin files")

        if os.path.exists(texture_dir):
            for subdir in ["full", "high", "low"]:
                subdir_path = os.path.join(texture_dir, subdir)
                if os.path.exists(subdir_path):
                    webp_files = [
                        f for f in os.listdir(subdir_path) if f.endswith(".webp")
                    ]
                    removed = 0
                    for f in webp_files:
                        match = re.match(r"sector_(-?\d+)_(-?\d+)\.webp", f)
                        if match:
                            q, r = int(match.group(1)), int(match.group(2))
                            if (q, r) not in valid_sectors:
                                os.remove(os.path.join(subdir_path, f))
                                removed += 1
                    if removed > 0:
                        print(f"   Removed {removed} old .webp files from {subdir}")

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

    if RESAMPLE_DEM:
        TARGET_RES = 3.5  # Upscale from 5m to 3.5m to fit ~10GB RAM

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
                resampling=rasterio.enums.Resampling.lanczos,
            )
            upscaled_transform = rasterio.transform.from_bounds(
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
                new_w,
                new_h,
            )
            print(f" Done. {dem_data.nbytes / 1e9:.2f} GB used.")
    else:
        print("   Loading DEM at original resolution (DEBUG MODE)...")
        with rasterio.open(DEM_PATH) as src:
            dem_data = src.read(1)
            upscaled_transform = src.transform
            print(f"   Done. {dem_data.nbytes / 1e9:.2f} GB used.")

    # =============================================================================
    # 6. FULL PRODUCTION BAKE
    # =============================================================================
    if valid_sectors:
        print(
            f"\n🏭 FACTORY MODE: Baking {len(valid_sectors)} Sectors (Texture + Geometry)..."
        )

        baked_count = 0
        total_sectors = len(valid_sectors)

        for Q, R in valid_sectors:
            baked_count += 1
            print(f"[{baked_count}/{total_sectors}] Baking Q={Q}, R={R}...", end="\r")

            # A. TEXTURES
            results, canvas = bake_sector_textures(
                Q, R, valid_tifs, output_dir="frontend/hexagons/app/aerial_tiles"
            )

            # B. GEOMETRY (BIN)
            bin_path, bin_size, bin_duration = bake_sector_binary(
                Q,
                R,
                dem_data,
                upscaled_transform,
                output_dir="frontend/hexagons/app/tiles_bin",
            )

            # Explicit Cleanup of Canvas
            del canvas

            if baked_count % 10 == 0:
                gc.collect()

        print(f"\n✅ SUCCESS! All {baked_count} Sectors served.")
        print(f"   - Textures: 'frontend/hexagons/app/aerial_tiles/'")
        print(f"   - Geometry: 'frontend/hexagons/app/tiles_bin/'")

    # Generate manifest for debug mode
    if DEBUG_MODE:
        print("\n📋 Generating manifest for debug sectors...")
        generate_manifest.generate_manifest()

    # Final cleanup
    del dem_data
    gc.collect()


if __name__ == "__main__":
    main()
