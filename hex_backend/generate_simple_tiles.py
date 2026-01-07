
import os
import glob
import rasterio
from PIL import Image
import multiprocessing
import shutil

# Config
INPUT_DIR = "/Users/cole/dev/PowFinder/hex_backend/aerial_tifs"
BASE_OUTPUT_DIR = "/Users/cole/dev/PowFinder/frontend/hexagons/app/tiles_sat"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_single_tif(path):
    try:
        filename = os.path.basename(path)
        low_dir = os.path.join(BASE_OUTPUT_DIR, "low_res")
        med_dir = os.path.join(BASE_OUTPUT_DIR, "med_res")
        high_dir = os.path.join(BASE_OUTPUT_DIR, "high_res")
        
        with rasterio.open(path) as src:
            bounds = src.bounds
            tile_x = int(bounds.left)
            tile_y = int(bounds.top)
            out_base = f"tile_{tile_x}_{tile_y}"
            
            # 1. HIGH: Original TIF only (No WebP)
            high_path_tif = os.path.join(high_dir, out_base + ".tif")
            shutil.copy(path, high_path_tif)
            
            # Load image for med/low
            data = src.read([1, 2, 3]).transpose(1, 2, 0)
            img = Image.fromarray(data)
            
            # 2. MED: Native Res, WebP Quality 5
            med_path = os.path.join(med_dir, out_base + ".webp")
            img.save(med_path, 'WEBP', quality=5)
            
            # 3. LOW: 1m scale (1/5th), WebP Quality 5
            low_size = (img.width // 5, img.height // 5)
            img_low = img.resize(low_size, Image.Resampling.LANCZOS)
            low_path = os.path.join(low_dir, out_base + ".webp")
            img_low.save(low_path, 'WEBP', quality=5)
            
        return f"Done: {out_base}"
    except Exception as e:
        return f"Error processing {path}: {str(e)}"

def generate_simple_tiles():
    ensure_dir(os.path.join(BASE_OUTPUT_DIR, "low_res"))
    ensure_dir(os.path.join(BASE_OUTPUT_DIR, "med_res"))
    ensure_dir(os.path.join(BASE_OUTPUT_DIR, "high_res"))

    tif_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))
    print(f"Found {len(tif_files)} TIFs. Converting (High=TIF, Med=Q5 WebP, Low=1m Q5 WebP)...")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(pool.imap_unordered(process_single_tif, tif_files))
        for r in results:
            print(r)

    print("Conversion complete.")

if __name__ == "__main__":
    generate_simple_tiles()
