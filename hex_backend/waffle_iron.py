# 🧇 Waffle Iron v3.3 - Performance & Quality Update
# - Optimized with High-Res Slope Caching (BIGTIFF support)
# - Upsampling: 2x (5m -> 2.5m) using Lanczos Resampling
# - Performance: Cache generation ~185s, Sector Baking ~15s (25 sectors)
# - Logic: Uses vector magnitude (sqrt(dx^2 + dy^2)) for true fall-line gradients.

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
DEBUG_MODE = False
RESAMPLE_DEM = False
TARGET_LAT = 46.98705560886202
TARGET_LON = 11.115050838788871

DEM_PATH = "hex_backend/DGM_Tirol_5m_epsg31254_2006_2020.tif"
SLOPE_PATH = "hex_backend/DGM_Tirol_slope_cached.tif"
AERIAL_DIR = "hex_backend/aerial_tifs"

# =============================================================================
# BAKING FUNCTIONS
# =============================================================================

def get_or_create_slope_map(dem_path, output_path, upsample_factor=1):
    """
    Checks for a cached slope TIF. If missing, generates it from the DEM.
    Calculates gradient vector magnitude (steepness) using windowed processing
    to avoid OOM errors on large datasets.
    """
    if os.path.exists(output_path):
        print(f"✅ Found cached slope map: {output_path}")
        return rasterio.open(output_path)

    print(f"⚠️  Slope map not found. Generating from {dem_path}...")
    start_time = time.time()

    with rasterio.open(dem_path) as src:
        # Calculate new dimensions
        new_width = src.width * upsample_factor
        new_height = src.height * upsample_factor
        
        # New transform
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        # Output Profile
        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.float32, 
            count=1, 
            driver='GTiff',
            width=new_width,
            height=new_height,
            transform=new_transform,
            compress='lzw',
            tiled=True,
            blockxsize=512,
            blockysize=512,
            predictor=3,
            BIGTIFF='YES'
        )
        
        # Grid resolutions
        res_x = abs(new_transform[0])
        res_y = abs(new_transform[4])

        print(f"   -> Processing in windows (Total: {new_width}x{new_height}, Res: {res_x:.2f}m)...")
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            # We process the output in blocks
            # To calculate gradient at edges, we need a 1-pixel padding from the DEM
            # But since we are upsampling, we need to be careful.
            
            # Simpler approach: Process based on destination blocks
            for jt, window in dst.block_windows(1):
                # 'window' is in destination coordinates
                # We need to read slightly more from the source DEM to avoid edge artifacts
                # 2 pixels padding in source is plenty for np.gradient
                pad = 2
                
                # Back-project window to source coordinates
                # Since it's a simple scale, we can just divide
                src_window = rasterio.windows.Window(
                    window.col_off / upsample_factor - pad,
                    window.row_off / upsample_factor - pad,
                    window.width / upsample_factor + 2*pad,
                    window.height / upsample_factor + 2*pad
                )
                
                # Clamp to src bounds
                src_window = src_window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                
                # Read and Resample this small chunk
                # Calculate the exact shape we expect for this window
                # Note: window.width/height are the sizes in the DESTINATION file
                chunk_dem = src.read(
                    1, 
                    window=src_window, 
                    out_shape=(int(src_window.height * upsample_factor), int(src_window.width * upsample_factor)),
                    resampling=rasterio.enums.Resampling.lanczos
                )
                
                if chunk_dem.size == 0: continue

                # Calculate local gradient
                dy, dx = np.gradient(chunk_dem, res_y, res_x)
                slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
                slope_deg = np.degrees(slope_rad).astype(np.float32)
                
                # We need to crop out the padding we added (if any)
                # The src_window might have been shifted by 'pad'
                # Find where the actual destination 'window' starts inside this resampled chunk
                
                # Calculate the offset within the resampled chunk_dem
                # src_window.col_off is the starting column in the original TIF
                # (src_window.col_off * upsample_factor) is the starting column in 'infinite upsampled space'
                # window.col_off is the starting column in the actual destination TIF
                
                off_x = int(round(window.col_off - (src_window.col_off * upsample_factor)))
                off_y = int(round(window.row_off - (src_window.row_off * upsample_factor)))
                
                # Slice out only the part that belongs in the destination window
                final_chunk = slope_deg[off_y:off_y+int(window.height), off_x:off_x+int(window.width)]
                
                dst.write(final_chunk, 1, window=window)
                
                if jt[0] % 20 == 0 and jt[1] == 0:
                    total_blocks = (dst.height // 512 + 1) * (dst.width // 512 + 1)
                    current_block = jt[0] * (dst.width // 512 + 1) + jt[1]
                    print(f"   -> Progress: {100 * current_block / total_blocks:.1f}%")

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

def bake_sector_binary(SX, SY, dem_data, dem_transform, slope_data, slope_transform, output_dir="frontend/hexagons/app/tiles_bin"):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    min_x, min_y, max_x, max_y = coord_util.sector_id_to_bounds_meters(SX, SY)

    # NOTE: Gradient calculation removed from here. Using pre-baked 'slope_data'.

    # Function sample_data removed as it is superseded by inline vectorized sampling.
        


    scales = [{"id": 3, "s": 24.0}, {"id": 2, "s": 6.0}, {"id": 1, "s": 3.0}, {"id": 0, "s": 1.0}]
    layers_data, min_z, max_z = [], 9999, -9999
    
    center_wx, center_wy = coord_util.get_sector_center(SX, SY)
    cq, cr = [int(round(v)) for v in coord_util.world_meters_to_axial_approx(center_wx, center_wy)]

    for l in scales:
        S = l["s"]
        # Calculate Center in Scale (Layer Center Q/R)
        lcq, lcr = [int(round(v)) for v in coord_util.world_meters_to_axial_scale(center_wx, center_wy, S)]
        
        hx = coord_util.get_lod_grid_hexes_in_bbox(min_x, max_x, min_y, max_y, S)
        if hx:
            # Vectorized Sampling for Center + 6 Neighbors
            # Neighbor Order: N, NE, SE, S, SW, NW (Aligned with Axial +y=North)
            n_offsets = [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)]
            
            # Precompute World Delta per Axial Step for this Scale
            # Note: This reproduces the axial_to_world logic inline
            w_h = coord_util.UNIT_HEX_WIDTH_METERS * S
            dx_dq = (math.sqrt(3)/2) * w_h
            dy_dq = 0.5 * w_h
            dy_dr = w_h 
            
            # Arrays of Center WX, WY
            c_wx = np.array([h[2] for h in hx])
            c_wy = np.array([h[3] for h in hx])
            
            # Create list of arrays to sample: [Center, N, NE, SE, S, SW, NW]
            sample_wxs = [c_wx]
            sample_wys = [c_wy]
            
            for (ndq, ndr) in n_offsets:
                 # Calculate relative world offset for neighbor
                 odx = ndq * dx_dq
                 ody = ndr * dy_dr + ndq * dy_dq
                 sample_wxs.append(c_wx + odx)
                 sample_wys.append(c_wy + ody)
                 
            # Flatten to sample
            flat_wxs = np.concatenate(sample_wxs)
            flat_wys = np.concatenate(sample_wys)
            
            # Sample Heights
            rows, cols = rasterio.transform.rowcol(dem_transform, flat_wxs, flat_wys)
            rows = np.clip(rows, 0, dem_data.shape[0]-1)
            cols = np.clip(cols, 0, dem_data.shape[1]-1)
            all_h = dem_data[rows, cols]
            
            # Reshape: (7, N)
            stacked_h = all_h.reshape(7, len(hx))
            
            # Sample Slope (Center only)
            s_rows, s_cols = rasterio.transform.rowcol(slope_transform, c_wx, c_wy)
            s_rows = np.clip(s_rows, 0, slope_data.shape[0]-1)
            s_cols = np.clip(s_cols, 0, slope_data.shape[1]-1)
            s_vals = slope_data[s_rows, s_cols]

            # Update Min/Max Z from Center vals
            min_z, max_z = min(min_z, stacked_h[0].min()), max(max_z, stacked_h[0].max())
            
            # Process Layers
            use_slope = (S <= 3.0)
            
            layer = []
            num_hexes = len(hx)
            
            # Vectorized calc for deltas? 
            # D = Center - Neighbor.
            # But we need math.ceil logic.
            # Can use numpy:
            # center (1, N) broadcast against neighbors (6, N)
            
            c_h_b = stacked_h[0] # (N,)
            n_h_b = stacked_h[1:] # (6, N)
            
            # diffs = c_h - n_h
            diffs = c_h_b - n_h_b # (6, N)
            
            # Logic: if diff <= 0 -> 0. else ceil(diff). Clamp 0-255.
            # ceil: np.ceil
            d_ceil = np.ceil(diffs)
            d_clamped = np.clip(d_ceil, 0, 255).astype(np.uint8)
            
            # Handle special floor case? 
            # If d_ceil >= 254 -> 255. (Infinite/Floor)
            # Actually user said "bake in neighbor Z".
            # If the gap is huge (cliff), we probably want to extend to the neighbor.
            # So 255 is just a max limit.
            
            # Transpose to iterate by hex
            d_final = d_clamped.T # (N, 6)
            
            for i in range(num_hexes):
                q, r = hx[i][0], hx[i][1]
                slope = s_vals[i] if use_slope else 0
                layer.append({
                    'q': q, 'r': r, 
                    'l': hx[i], # raw tuple
                    'lcq': lcq, 'lcr': lcr,
                    'h': stacked_h[0, i],
                    's': slope,
                    'deltas': d_final[i] # array of 6 uint8
                })
            layers_data.append(layer)
        else: layers_data.append([])

    scale_f = 65535.0 / (max_z - min_z + 20) if max_z > min_z else 1.0
    blob = struct.pack("<4siifffii", b"HEX3", int(SX), int(SY), float(min_z-10), float(max_z+10), float(scale_f), cq, cr)
    
    # REFACTORED LOOP
    for l_idx, ld in enumerate(layers_data):
        S = scales[l_idx]["s"]
        blob += struct.pack("<I", len(ld))
        buf = bytearray(len(ld) * 13)
        # Directions for neighbors (North, NE, SE, South, SW, NW) 
        # based on Axial (Flat Top): 
        # N: q, r-1
        # NE: q+1, r-1
        # SE: q+1, r
        # S: q, r+1
        # SW: q-1, r+1
        # NW: q-1, r
        nb_offsets = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)]
        
        for i, item in enumerate(ld):
            q, r = item['q'], item['r']
            h = item['h']
            s = item['s']
            lcq, lcr = item['lcq'], item['lcr']

            dq, dr = max(-32767, min(32767, int(q - lcq))), max(-32767, min(32767, int(r - lcr)))
            hn = max(0, min(65535, int((h - (min_z-10)) * scale_f)))
            s_byte = max(0, min(255, int(s)))
            
            # Neighborhood Search
            deltas = []
            for oq, or_ in nb_offsets:
                nq, nr = q + oq, r + or_
                # Sample DEM at neighbor centered world pos
                # we can use the formula directly since coord_util has it
                # world_x = (q * (math.sqrt(3)/2) * h_eff) ...
                h_eff = coord_util.UNIT_HEX_WIDTH_METERS * S
                wx = (nq * (math.sqrt(3)/2) * h_eff)
                wy = (nr * h_eff + nq * 0.5 * h_eff)
                
                # Fast sample (Nearest)
                row, col = rasterio.transform.rowcol(dem_transform, wx, wy)
                row = max(0, min(dem_data.shape[0]-1, row))
                col = max(0, min(dem_data.shape[1]-1, col))
                nb_h = dem_data[row, col]
                
                # Delta is positive if WE are higher than NEIGHBOR (skirt drops)
                # Max drop 255m
                d_val = max(0, min(255, int(round(h - nb_h))))
                deltas.append(d_val)
                
            struct.pack_into("<hhHB6B", buf, i*13, dq, dr, hn, s_byte, *deltas)
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
    # Upsample by factor of 2 (5m -> 2.5m) for smoother gradients
    upsample = 2 if not DEBUG_MODE else 1
    slope_ds = get_or_create_slope_map(DEM_PATH, SLOPE_PATH, upsample_factor=upsample)
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
                bake_sector_binary(sx, sy, dem_data, dem_transform, slope_data, slope_transform)

    generate_manifest.generate_manifest()
    print("Done.")

if __name__ == "__main__": main()
