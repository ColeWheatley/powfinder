
import os
import json
import re

TILES_DIR = 'frontend/piston_viewer/tiles_bin'
OUTPUT_FILE = 'frontend/piston_viewer/tile_manifest.json'

def generate_manifest():
    files = os.listdir(TILES_DIR)
    tiles = []
    
    # Pattern to match tile_X_Y.bin
    pattern = re.compile(r'tile_(\d+)_(\d+)\.bin')
    
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for f in files:
        match = pattern.match(f)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            
            # Skip the 'oddball' tile at the lower left
            if x == 53750:
                continue
                
            tiles.append({'x': x, 'y': y})
            
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y
            
    manifest = {
        'tiles': tiles,
        'bounds': {
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y
        },
        'tile_size': {
            'x': 1250,
            'y': 1000
        }
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Generated manifest with {len(tiles)} tiles.")
    print(f"Bounds: X[{min_x}, {max_x}], Y[{min_y}, {max_y}]")

if __name__ == "__main__":
    generate_manifest()
