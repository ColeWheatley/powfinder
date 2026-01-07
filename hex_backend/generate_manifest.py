import os
import json
import re
import sys
import coordinate_utility as coord_util

# CONFIG
BINARY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/hexagons/app/tiles_bin"))
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/hexagons/app/tile_manifest.json"))

def generate_manifest():
    print(f"🔍 Manifest Generator looking in: {BINARY_DIR}")
    
    if not os.path.exists(BINARY_DIR):
        print("❌ Error: Binary directory not found.")
        return

    files = os.listdir(BINARY_DIR)
    sectors = []
    
    # Pattern: sector_Q_R.bin ("sector_277_-234.bin")
    pattern = re.compile(r'sector_(-?\d+)_(-?\d+)\.bin')
    
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for f in files:
        match = pattern.match(f)
        if match:
            # Parse Q, R
            Q = int(match.group(1))
            R = int(match.group(2))
            
            # Convert to World Center (for Frontend positioning)
            cx, cy = coord_util.sector_to_world_meters(Q, R)
            
            # Append to list
            # We treat 'x','y' in the manifest as the CENTER of the tile/sector
            sectors.append({
                'q': Q,
                'r': R,
                'x': cx,
                'y': cy
            })
            
            # Update Bounds (Approximate, based on center)
            # Actually, frontend uses bounds to center camera.
            if cx < min_x: min_x = cx
            if cx > max_x: max_x = cx
            if cy < min_y: min_y = cy
            if cy > max_y: max_y = cy
            
    # Calculate approx bounds size for camera
    margin = 1000.0
            
    manifest = {
        'tiles': sectors, # Rename to 'sectors'? Frontend expects 'tiles' array.
        'type': 'sector_hex',
        'bounds': {
            'min_x': min_x - margin,
            'max_x': max_x + margin,
            'min_y': min_y - margin,
            'max_y': max_y + margin
        },
        'sector_radius_m': coord_util.SECTOR_WIDTH_METERS / 2.0
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"✅ Generated manifest for {len(sectors)} sectors.")
    print(f"   Bounds: X[{min_x:.0f}, {max_x:.0f}], Y[{min_y:.0f}, {max_y:.0f}]")
    print(f"   Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_manifest()
