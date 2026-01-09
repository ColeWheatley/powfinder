
import os
import glob
import rasterio
from rasterio.windows import from_bounds
import numpy as np
from PIL import Image

# Config
INPUT_DIR = "/Users/cole/dev/PowFinder/backend/aerial_tifs"
BASE_OUTPUT_DIR = "/Users/cole/dev/PowFinder/frontend/piston_viewer/tiles_sat"
PADDING_METERS = 20.0 

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_tiles():
    # Define Levels
    high_dir = os.path.join(BASE_OUTPUT_DIR, "high_res")
    med_dir = os.path.join(BASE_OUTPUT_DIR, "med_res")
    low_dir = os.path.join(BASE_OUTPUT_DIR, "low_res")
    
    ensure_dir(high_dir)
    ensure_dir(med_dir)
    ensure_dir(low_dir)

    # 1. Index all TIFs
    tifs = []
    tif_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))
    print(f"Found {len(tif_files)} TIFs in {INPUT_DIR}. Indexing...")

    for path in tif_files:
        with rasterio.open(path) as src:
            tifs.append({
                'path': path,
                'bounds': src.bounds,
                'transform': src.transform,
            })

    print("Indexing complete. Processing...")

    # 2. Process each TIF
    for main_tif in tifs:
        L, B, R, T = main_tif['bounds']
        pL, pB, pR, pT = L - PADDING_METERS, B - PADDING_METERS, R + PADDING_METERS, T + PADDING_METERS

        res_x = main_tif['transform'][0]
        res_y = -main_tif['transform'][4]
        
        width_px = int(round((pR - pL) / res_x))
        height_px = int(round((pT - pB) / res_y))

        canvas = np.zeros((height_px, width_px, 3), dtype=np.uint8)

        # 3. Contributors
        contributors = []
        for candidate in tifs:
            cL, cB, cR, cT = candidate['bounds']
            if not (cL > pR or cR < pL or cB > pT or cT < pB):
                contributors.append(candidate)

        # 4. Composite
        pasted_any = False
        for c in contributors:
            with rasterio.open(c['path']) as c_src:
                cL, cB, cR, cT = c['bounds']
                iL, iB, iR, iT = max(pL, cL), max(pB, cB), min(pR, cR), min(pT, cT)

                if iL >= iR or iB >= iT: continue

                window = from_bounds(iL, iB, iR, iT, c_src.transform)
                w_px = int(round((iR - iL) / res_x))
                h_px = int(round((iT - iB) / res_y))

                data = c_src.read([1, 2, 3], window=window, boundless=True, fill_value=0, out_shape=(3, h_px, w_px))
                data = data.transpose(1, 2, 0)

                col_off = int(round((iL - pL) / res_x))
                row_off = int(round((pT - iT) / res_y))
                
                h, w, _ = data.shape
                r1, r2 = row_off, row_off + h
                c1, c2 = col_off, col_off + w
                
                cr1, cr2 = max(0, r1), min(height_px, r2)
                cc1, cc2 = max(0, c1), min(width_px, c2)
                
                dr1, dr2 = cr1 - r1, h - (r2 - cr2)
                dc1, dc2 = cc1 - c1, w - (c2 - cc2)
                
                if cr2 > cr1 and cc2 > cc1:
                    canvas[cr1:cr2, cc1:cc2] = data[dr1:dr2, dc1:dc2]
                    pasted_any = True

        # 5. Save Levels
        tile_x = int(L)
        tile_y = int(T)
        out_base = f"tile_{tile_x}_{tile_y}"
        
        if pasted_any:
            img = Image.fromarray(canvas)
            
            # HIGH: Oversized TIF (Uncompressed)
            high_path = os.path.join(high_dir, out_base + ".tif")
            img.save(high_path, 'TIFF', compression=None)

            # MED: Native Res, WebP Q=5
            med_path = os.path.join(med_dir, out_base + ".webp")
            img.save(med_path, 'WEBP', quality=5)

            # LOW: 1m Res (~0.2 -> 1.0 => 1/5 scale), WebP Q=5
            low_w = int(width_px * 0.2)
            low_h = int(height_px * 0.2)
            img_low = img.resize((low_w, low_h), Image.Resampling.LANCZOS)
            low_path = os.path.join(low_dir, out_base + ".webp")
            img_low.save(low_path, 'WEBP', quality=5)
            
            print(f"Saved {out_base} [High/Med/Low] (padded {PADDING_METERS}m)")
        else:
            print(f"Skipping {out_base} - No data")

if __name__ == "__main__":
    generate_tiles()
