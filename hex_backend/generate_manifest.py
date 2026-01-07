
import os
import json
import re

TILES_DIR = '../frontend/hexagons/app/tiles_bin/res_10'
OUTPUT_FILE = '../frontend/hexagons/app/tile_manifest.json'

def generate_manifest():
    global TILES_DIR
    if not os.path.exists(TILES_DIR):
        # Fallback to see if any res_ folder exists
        parent = os.path.dirname(TILES_DIR)
        if os.path.exists(parent):
            subs = [d for d in os.listdir(parent) if d.startswith('res_')]
            if subs:
                TILES_DIR = os.path.join(parent, subs[0])
    
    print(f"Indexing tiles from {TILES_DIR}...")
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
