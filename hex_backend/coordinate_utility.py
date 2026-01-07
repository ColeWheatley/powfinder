
import math

# =============================================================================
# PIXEL-FIRST CONSTANTS
# =============================================================================
# The fundamental truth of the universe is the Pixel.
# We define everything relative to the High-Res Texture capabilities.

TEXTURE_SIZE_PX = 4096       # The size of the Sector Texture (Power of Two)
METERS_PER_PIXEL = 0.2       # The "Tirol Truth" (Nominal resolution)

# Gosper Fractal Scale Factor for Level 5 (7^2.5)
LEVEL_5_SCALE_FACTOR = 7.0 ** 2.5  # ~129.6418

# Derived Dimensions
SECTOR_WIDTH_METERS = TEXTURE_SIZE_PX * METERS_PER_PIXEL  # 819.2m
UNIT_HEX_WIDTH_METERS = SECTOR_WIDTH_METERS / LEVEL_5_SCALE_FACTOR # ~6.319m

# Directions
NORTH = 0
NORTH_EAST = 1
SOUTH_EAST = 2
SOUTH = 3
SOUTH_WEST = 4
NORTH_WEST = 5

def get_hex_dimensions():
    """
    Returns the calculated dimensions of the hexes based on pixel constants.
    """
    return {
        'sector_width_m': SECTOR_WIDTH_METERS,
        'unit_hex_width_m': UNIT_HEX_WIDTH_METERS,
        'unit_hex_radius_m': UNIT_HEX_WIDTH_METERS / math.sqrt(3),
        'pixels_per_unit_hex': TEXTURE_SIZE_PX / LEVEL_5_SCALE_FACTOR # ~31.59 px
    }

def axial_to_world_meters(q, r):
    """
    Converts Universal Unit Axial (q, r) to World Meters (x, y).
    North-Zero Standard.
    """
    h = UNIT_HEX_WIDTH_METERS
    # Basis vectors: +r is North, +q is NE
    world_x = (q * (math.sqrt(3)/2) * h)
    world_y = (r * h + q * 0.5 * h)
    return world_x, world_y
    
def sector_to_world_meters(Q, R):
    """
    Converts Level 5 Sector (Q, R) -> World Meters center point.
    Uses Matrix 7^5 Transformation.
    """
    # 1. Transform Sector (Q, R) -> Universal Unit Center (q, r)
    center_q = -87 * Q - 149 * R
    center_r = 149 * Q + 62 * R
    return axial_to_world_meters(center_q, center_r)

def world_meters_to_sector_approx(x, y):
    """
    Approximate inverse to find which Sector (Q, R) a point (x, y) falls in.
    Useful for scanning bounds.
    """
    h = UNIT_HEX_WIDTH_METERS
    w_inv = 1.0 / (math.sqrt(3)/2 * h)
    
    q_approx = x * w_inv
    r_approx = (y - q_approx * 0.5 * h) / h
    
    # Inverse Matrix 7^5 (Det = 16807)
    det = 16807
    Q = (62 * q_approx + 149 * r_approx) / det
    R = (-149 * q_approx - 87 * r_approx) / det
    
    return Q, R
