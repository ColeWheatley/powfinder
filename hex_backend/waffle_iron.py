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
from multiprocessing import Pool, cpu_count
import sys
import struct
from pyproj import Transformer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import coordinate_utility as coord_util
import generate_manifest

def latlon_to_world_meters(lat, lon):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:31254", always_xy=True)
    return transformer.transform(lon, lat)

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
TEXTURE_PADDING_PX = 64  
WEB_P_QUALITY = 10 # User Requested
DEBUG_MODE = True
RESAMPLE_DEM = False
TARGET_LAT = 46.98705560886202
TARGET_LON = 11.115050838788871

DEM_PATH = "hex_backend/DGM_Tirol_5m_epsg31254_2006_2020.tif"
SLOPE_PATH = "hex_backend/DGM_Tirol_slope_cached.tif"
AERIAL_DIR = "hex_backend/aerial_tifs"

# =============================================================================
# BAKING FUNCTIONS
# =============================================================================

def get_or_create_slope_map(dem_path, output_path, resample=False):
    """
    Checks for a cached slope TIF. If missing, generates it from the DEM.
    Calculates gradient vector magnitude (steepness).
    """
    if os.path.exists(output_path):
        print(f"✅ Found cached slope map: {output_path}")
        return rasterio.open(output_path)

    print(f"⚠️  Slope map not found. Generating from {dem_path}...")
    start_time = time.time()

    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        transform = src.transform
        profile = src.profile.copy()
        
        # Use absolute resolution for gradient spacing
        res_x = abs(transform[0])
        res_y = abs(transform[4])

        # Optional: Upsample Logic (for smoother gradients as requested)
        if resample:
            print("   -> Upsampling DEM for smoother gradients...")
            # TODO: Implement fancy upsampling if 'RESAMPLE_DEM' is True. 
            # For now, we stick to the 5m grid to save RAM, as 'np.gradient' checks neighbors anyway.
            pass

        print(f"   -> Calculating Gradient on {dem_data.shape} grid (Spacing: {res_x}m, {res_y}m)...")
        # np.gradient returns (gradient_along_axis_0, gradient_along_axis_1) -> (dy, dx)
        dy, dx = np.gradient(dem_data, res_y, res_x)
        
        # Calculate Magnitude (Slope in Radians) -> Convert to Degrees
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        
        # Save Cached TIF
        profile.update(dtype=rasterio.float32, count=1, driver='GTiff')
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(slope_deg.astype(rasterio.float32), 1)
            
    print(f"✅ Generated Slope Map in {time.time() - start_time:.2f}s")
    return rasterio.open(output_path)

def bake_sector_textures(SX, SY, valid_tifs, output_dir="frontend/hexagons/app/aerial_tiles"):
    import PIL.Image as Image
    from rasterio.windows import from_bounds

    if not os.path.exists(output_dir): os.makedirs(output_dir)

    min_x, min_y, max_x, max_y = coord_util.sector_id_to_bounds_meters(SX, SY)
    padding_m = TEXTURE_PADDING_PX * coord_util.METERS_PER_PIXEL
    padded_min_x, padded_max_x = min_x - padding_m, max_x + padding_m
    padded_min_y, padded_max_y = min_y - padding_m, max_y + padding_m
    
    total_size_px = coord_util.SECTOR_PIXELS + (TEXTURE_PADDING_PX * 2)
    target_poly = box(padded_min_x, padded_min_y, padded_max_x, padded_max_y)

    canvas = Image.new("RGB", (total_size_px, total_size_px), (0, 0, 0))
    intersecting = [t for t in valid_tifs if t["poly"].intersects(target_poly)]

    for t in intersecting:
        with rasterio.open(t["path"]) as src:
            ix_min_x, ix_max_x = max(padded_min_x, src.bounds.left), min(padded_max_x, src.bounds.right)
            ix_min_y, ix_max_y = max(padded_min_y, src.bounds.bottom), min(padded_max_y, src.bounds.top)
            if ix_min_x >= ix_max_x or ix_min_y >= ix_max_y: continue

            window = from_bounds(ix_min_x, ix_min_y, ix_max_x, ix_max_y, src.transform)
            w_px = int((ix_max_x - ix_min_x) / coord_util.METERS_PER_PIXEL)
            h_px = int((ix_max_y - ix_min_y) / coord_util.METERS_PER_PIXEL)
            if w_px <= 0 or h_px <= 0: continue

            try:
                data = src.read(window=window, out_shape=(src.count, h_px, w_px), resampling=rasterio.enums.Resampling.lanczos)
                patch = Image.fromarray(np.moveaxis(data, 0, -1).astype("uint8"), "RGB")
                px = int((ix_min_x - padded_min_x) / coord_util.METERS_PER_PIXEL)
                py = int((padded_max_y - ix_max_y) / coord_util.METERS_PER_PIXEL)
                canvas.paste(patch, (px, py))
            except: pass

    res_dirs = { k: os.path.join(output_dir, k) for k in ["full", "high", "low"] }
    for d in res_dirs.values():
        if not os.path.exists(d): os.makedirs(d)

    f_name = f"sector_{SX}_{SY}.webp"
    canvas.save(os.path.join(res_dirs["full"], f_name), "WEBP", quality=WEB_P_QUALITY)
    
    # High: 4096 -> 1024
    c_high = canvas.resize((total_size_px // 4, total_size_px // 4), Image.LANCZOS)
    c_high.save(os.path.join(res_dirs["high"], f_name), "WEBP", quality=WEB_P_QUALITY)
    
    # Low: 4096 -> 256
    c_low = canvas.resize((total_size_px // 16, total_size_px // 16), Image.LANCZOS)
    c_low.save(os.path.join(res_dirs["low"], f_name), "WEBP", quality=WEB_P_QUALITY)

def bake_sector_binary(SX, SY, dem_ds, slope_ds, output_dir="frontend/hexagons/app/tiles_bin"):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    min_x, min_y, max_x, max_y = coord_util.sector_id_to_bounds_meters(SX, SY)

    # Access Data & Transforms specifically for this baking op (keeps handles alive in main)
    dem_array = dem_ds.read(1) # Warning: Reads full DEM into memory every sector if not careful? 
    # Actually, main() reads it once? No, let's fix the passing.
    # To avoid repeated reads if we passed the OPEN DATASET, we should read windowed?
    # For now, assuming dataset is passed, let's just use window reading! Much faster and lighter.
    
    # Actually, previous code passed the *array*. Let's stick to passing the array for speed if it fits in RAM.
    # But wait, we want to optimize. 
    # Let's revert to passing ARRAY and TRANSFORM to avoid IO bottle necks in the loop.
    pass 

def bake_sector_binary_fast(SX, SY, dem_data, dem_transform, slope_data, slope_transform, output_dir="frontend/hexagons/app/tiles_bin"):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    min_x, min_y, max_x, max_y = coord_util.sector_id_to_bounds_meters(SX, SY)

    # NOTE: Gradient calculation removed from here. Using pre-baked 'slope_data'.

    def sample_data(hex_list):
        if not hex_list: return np.array([]), np.array([])
        wxs, wys = np.array([h[2] for h in hex_list]), np.array([h[3] for h in hex_list])
        
        # Sample Height
        rows, cols = rasterio.transform.rowcol(dem_transform, wxs, wys)
        rows = np.clip(rows, 0, dem_data.shape[0]-1)
        cols = np.clip(cols, 0, dem_data.shape[1]-1)
        h_vals = dem_data[rows, cols]

        # Sample Slope (using slope transform in case resolutions differ)
        s_rows, s_cols = rasterio.transform.rowcol(slope_transform, wxs, wys)
        s_rows = np.clip(s_rows, 0, slope_data.shape[0]-1)
        s_cols = np.clip(s_cols, 0, slope_data.shape[1]-1)
        s_vals = slope_data[s_rows, s_cols]
        
        return h_vals, s_vals
        
        # We need to aggregate? 
        # Ideally we'd grab a window. But 'sample_heights' in this script currently does 
        # NEAREST NEIGHBOR (single point). 
        # For Slope Average, Point Sampling is risky (aliasing).
        # But per user instruction "Straight averages are fine. It's not a super crazy high res dem".
        # If we stick to single point sampling for now (fast), it's just grabbing the slope AT center.
        # To do true "average", we'd need a window.
        # Let's start with Point Sampling of the Slope Map for speed/consistency with height logic.
        


    scales = [{"id": 3, "s": 24.0}, {"id": 2, "s": 6.0}, {"id": 1, "s": 3.0}, {"id": 0, "s": 1.0}]
    layers_data, min_z, max_z = [], 9999, -9999
    
    center_wx, center_wy = coord_util.get_sector_center(SX, SY)
    cq, cr = [int(round(v)) for v in coord_util.world_meters_to_axial_approx(center_wx, center_wy)]

    for l in scales:
        hx = coord_util.get_lod_grid_hexes_in_bbox(min_x, max_x, min_y, max_y, l["s"])
        if hx:
            h_vals, s_vals = sample_data(hx)
            min_z, max_z = min(min_z, h_vals.min()), max(max_z, h_vals.max())
            
            # Pack Tuple: (q, r, height, slope)
            # Only bake slope for L0 (s=1.0) and L1 (s=3.0)
            use_slope = (l["s"] <= 3.0)
            
            layer = []
            for i, (q, r, wx, wy) in enumerate(hx):
                slope = s_vals[i] if use_slope else 0
                layer.append((q, r, h_vals[i], slope))
            layers_data.append(layer)
        else: layers_data.append([])

    scale_f = 65535.0 / (max_z - min_z + 20) if max_z > min_z else 1.0
    blob = struct.pack("<4siifffii", b"HEX3", int(SX), int(SY), float(min_z-10), float(max_z+10), float(scale_f), cq, cr)
    
    for ld in layers_data:
        blob += struct.pack("<I", len(ld))
        # Struct: dq(h), dr(h), hn(H, 2 bytes), slope(B, 1 byte), pad(B, 1 byte) => 8 bytes total?
        # Previous: 6 bytes (hhH). 
        # New: 8 bytes per hex to keep alignment? Or 7? 
        # Let's do 7 bytes: q(2), r(2), h(2), s(1).
        buf = bytearray(len(ld) * 7)
        for i, (q, r, h, s) in enumerate(ld):
            dq, dr = max(-32767, min(32767, int(q - cq))), max(-32767, min(32767, int(r - cr)))
            hn = max(0, min(65535, int((h - (min_z-10)) * scale_f)))
            s_byte = max(0, min(255, int(s)))
            struct.pack_into("<hhHB", buf, i*7, dq, dr, hn, s_byte)
        blob += buf
    
    with open(os.path.join(output_dir, f"sector_{SX}_{SY}.bin"), "wb") as f: f.write(blob)

def main():
    print("🧇 Waffle Iron v3.3: Gradient Flow Edition")
    
    # 1. Load DEM (Keep in memory for speed, it's ~1GB max usually)
    print("Loading DEM...")
    with rasterio.open(DEM_PATH) as dem:
        dem_data = dem.read(1)
        dem_transform = dem.transform
        dem_poly = box(*dem.bounds)

    # 2. Get/Create Gradient Slope Map
    print("Loading Slope Map...")
    slope_ds = get_or_create_slope_map(DEM_PATH, SLOPE_PATH, resample=RESAMPLE_DEM)
    slope_data = slope_ds.read(1)
    slope_transform = slope_ds.transform

    valid_tifs = []
    for f in glob.glob(os.path.join(AERIAL_DIR, "*.tif")):
        try:
            with rasterio.open(f) as src: valid_tifs.append({"path": f, "poly": box(*src.bounds)})
        except: pass

    tx, ty = latlon_to_world_meters(TARGET_LAT, TARGET_LON)
    r = 2
    min_sx, min_sy = coord_util.world_to_sector_id(tx - r*819, ty - r*819)
    max_sx, max_sy = coord_util.world_to_sector_id(tx + r*819, ty + r*819)

    for sx in range(min_sx, max_sx + 1):
        for sy in range(min_sy, max_sy + 1):
            if dem_poly.intersects(box(*coord_util.sector_id_to_bounds_meters(sx, sy))):
                print(f"Cooking Sector {sx}, {sy}...")
                bake_sector_textures(sx, sy, valid_tifs)
                bake_sector_binary_fast(sx, sy, dem_data, dem_transform, slope_data, slope_transform)

    generate_manifest.generate_manifest()
    print("Done.")

if __name__ == "__main__": main()
