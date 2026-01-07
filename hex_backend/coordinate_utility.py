
import math
import os
import glob
try:
    from pyproj import Transformer
except ImportError:
    Transformer = None

# =============================================================================
# CONSTANTS (Synced with waffle_maker.py)
# =============================================================================

# Level 5 Sector: 1000m Flat-to-Flat
SECTOR_WIDTH_FLAT_TO_FLAT = 1000.0
# Unit Hex: ~7.71m Flat-to-Flat
LEVEL_5_SCALE_FACTOR = 7.0 ** 2.5  
UNIT_HEX_FLAT_TO_FLAT = SECTOR_WIDTH_FLAT_TO_FLAT / LEVEL_5_SCALE_FACTOR 
UNIT_HEX_RADIUS = UNIT_HEX_FLAT_TO_FLAT / math.sqrt(3)

# CRS
EPSG_WORLD = "EPSG:31254" # MGI Austrian GK Central
EPSG_GPS = "EPSG:4326"   # WGS84

# =============================================================================
# CORE CONVERSIONS
# =============================================================================

def axial_to_world(q, r):
    """
    Converts hex axial coordinates (q, r) to World Meters (EPSG:31254).
    Uses the North-Zero, Flat-Topped standard from waffle_maker.py.
    """
    h = UNIT_HEX_FLAT_TO_FLAT
    # x = q * (sqrt(3)/2 * h)
    # y = r * h + q * (0.5 * h)
    x = q * (math.sqrt(3)/2) * h
    y = r * h + q * 0.5 * h
    return x, y

def world_to_axial(x, y):
    """
    Converts World Meters (EPSG:31254) to hex axial coordinates (q, r).
    Returns floating point q, r (use hex_round for integer hex).
    """
    h = UNIT_HEX_FLAT_TO_FLAT
    q = x / (h * math.sqrt(3) / 2)
    r = (y - q * 0.5 * h) / h
    return q, r

def hex_round(q, r):
    """ Rounds floating point axial coordinates to the nearest hex integer. """
    # Convert axial to cube
    x = q
    z = r
    y = -x - z
    
    rx = round(x)
    ry = round(y)
    rz = round(z)
    
    x_diff = abs(rx - x)
    y_diff = abs(ry - y)
    z_diff = abs(rz - rz) # wait, z_diff = abs(rz - z)
    z_diff = abs(rz - z)
    
    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry
        
    return int(rx), int(rz)

def world_to_gps(x, y):
    """ Converts EPSG:31254 (MGI Austrian GK Central) to EPSG:4326 (WGS84). """
    if Transformer is None:
        raise ImportError("pyproj is required for GPS conversions.")
    transformer = Transformer.from_crs(EPSG_WORLD, EPSG_GPS, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon

def gps_to_world(lat, lon):
    """ Converts EPSG:4326 (WGS84) to EPSG:31254 (MGI Austrian GK Central). """
    if Transformer is None:
        raise ImportError("pyproj is required for GPS conversions.")
    transformer = Transformer.from_crs(EPSG_GPS, EPSG_WORLD, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def gps_to_hex(lat, lon):
    """ Helper: GPS -> World -> Axial Hex (Rounded) """
    x, y = gps_to_world(lat, lon)
    q, r = world_to_axial(x, y)
    return hex_round(q, r)

def hex_to_gps(q, r):
    """ Helper: Hex -> World -> GPS """
    x, y = axial_to_world(q, r)
    return world_to_gps(x, y)

def distance_meters(q1, r1, q2, r2):
    """ Euclidean distance in meters between two hex centers. """
    x1, y1 = axial_to_world(q1, r1)
    x2, y2 = axial_to_world(q2, r2)
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def hex_distance(q1, r1, q2, r2):
    """ Distance in hex steps (Manhattan distance on hex grid). """
    return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) // 2

# =============================================================================
# TIROL GRID & FILE LOGIC
# =============================================================================

def get_grid_id_for_world_coords(x, y):
    """
    Inverses the logic from check_tirol_grid.py to find which Grid ID (e.g. 2121-53)
    covers a specific world coordinate.
    """
    # Inverse of:
    # base_x = (xx - 16) * 10000
    # base_y = (yy - 1) * 10000 + 2000
    
    xx = int(x // 10000) + 16
    yy = int((y - 2000) // 10000) + 1
    
    base_x = (xx - 16) * 10000
    base_y = (yy - 1) * 10000 + 2000
    
    dx = x - base_x
    dy = y - base_y
    
    # Each 10km block is 8x8 tiles of 1250m x 1000m
    col = int(dx // 1250)
    row = int((8000 - dy) // 1000)
    
    # ss is 1-64 normally
    ss = row * 8 + col + 1
    return f"{xx}{yy}-{ss:02d}"

def get_expected_filenames(q, r, year="2023"):
    """ Returns the expected TIF and TFW filename for a hex. """
    x, y = axial_to_world(q, r)
    grid_id = get_grid_id_for_world_coords(x, y)
    return f"dop_{grid_id}_{year}.tif", f"dop_{grid_id}_{year}.tfw"

def find_local_tif(q, r, search_dir="hex_backend/aerial_tifs"):
    """
    Finds a local TIF file that covers the given hex. 
    Matches by Grid ID, ignoring year if necessary.
    """
    x, y = axial_to_world(q, r)
    grid_id = get_grid_id_for_world_coords(x, y)
    
    # Search for any file with this grid ID
    pattern = f"*{grid_id}*.tif"
    matches = glob.glob(os.path.join(search_dir, pattern))
    
    if matches:
        return matches[0]
    return None

# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Coordinate Utility - Usage Examples:")
        print("  hex_to_gps <q> <r>         -> Lat, Lon")
        print("  gps_to_hex <lat> <lon>     -> q, r")
        print("  dist <q1> <r1> <q2> <r2>   -> Distance in meters")
        print("  hex_dist <q1> <r1> <q2> <r2> -> Distance in hex steps")
        print("  find_file <q> <r>          -> Grid ID and expected filename")
        print("  find_local <q> <r>         -> Path to local TIF if it exists")
        sys.exit(0)
        
    cmd = sys.argv[1]
    
    try:
        if cmd == "hex_to_gps":
            q, r = int(sys.argv[2]), int(sys.argv[3])
            lat, lon = hex_to_gps(q, r)
            print(f"Hex ({q}, {r}) -> GPS: {lat:.6f}, {lon:.6f}")
            
        elif cmd == "gps_to_hex":
            lat, lon = float(sys.argv[2]), float(sys.argv[3])
            q, r = gps_to_hex(lat, lon)
            print(f"GPS ({lat}, {lon}) -> Hex: {q}, {r}")
            
        elif cmd == "dist":
            q1, r1 = int(sys.argv[2]), int(sys.argv[3])
            q2, r2 = int(sys.argv[4]), int(sys.argv[5])
            d = distance_meters(q1, r1, q2, r2)
            print(f"Distance: {d:.2f} meters")

        elif cmd == "hex_dist":
            q1, r1 = int(sys.argv[2]), int(sys.argv[3])
            q2, r2 = int(sys.argv[4]), int(sys.argv[5])
            d = hex_distance(q1, r1, q2, r2)
            print(f"Hex Distance: {d} steps")
            
        elif cmd == "find_file":
            q, r = int(sys.argv[2]), int(sys.argv[3])
            tif, tfw = get_expected_filenames(q, r)
            print(f"Hex ({q}, {r}) -> Expected: {tif}")
            
        elif cmd == "find_local":
            q, r = int(sys.argv[2]), int(sys.argv[3])
            path = find_local_tif(q, r)
            if path:
                print(f"Found: {path}")
            else:
                print("Not found locally.")
            
    except Exception as e:
        print(f"Error: {e}")
