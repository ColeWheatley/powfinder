
import os
import glob
import rasterio
from PIL import Image

# Config
INPUT_DIR = "/Users/cole/dev/PowFinder/backend/aerial_tifs"
BASE_OUTPUT_DIR = "/Users/cole/dev/PowFinder/frontend/piston_viewer/tiles_sat"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_simple_tiles():
    # Define Levels
    med_dir = os.path.join(BASE_OUTPUT_DIR, "med_res")
    ensure_dir(med_dir)

    tif_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))
    print(f"Found {len(tif_files)} TIFs. Converting to WebP (1:1)...")

    for path in tif_files:
        filename = os.path.basename(path)
        name_no_ext = os.path.splitext(filename)[0]
        # Expected name format: tile_X_Y.tif or similar. 
        # If the input names are consistent, we just swap extension.
        
        print(f"Processing {filename}...")
        
        with rasterio.open(path) as src:
            # Read full image
            # Transpose to (H, W, C)
            data = src.read([1, 2, 3]).transpose(1, 2, 0)
            
            img = Image.fromarray(data)
            
            # Save MED: Native Res, WebP Q=75 (Better quality than 5)
            # Use Q=5 if size is critical, but 5 is very low. User had 5.
            # Let's stick to 5-20 range if low bandwidth is needed, but 75 is standard.
            # User used Q=5. I'll use 20 for a bit better quality without huge size.
            out_path = os.path.join(med_dir, name_no_ext + ".webp")
            img.save(out_path, 'WEBP', quality=20)
            
            # We can add Low/High if needed, but Med is the core one.
            
    print("Conversion complete.")

if __name__ == "__main__":
    generate_simple_tiles()
