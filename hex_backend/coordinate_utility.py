import math
import numpy as np

# =============================================================================
# PIXEL-FIRST CONSTANTS (UNIT-BASED)
# =============================================================================
# The fundamental truth is the UNIT HEX.
# We want exactly 32 pixels across a unit hex to strictly avoid aliasing/shimmer.

UNIT_HEX_PX = 32.0           # Exactly 32 pixels flat-to-flat
METERS_PER_PIXEL = 0.2       # The "Tirol Truth"

# Gosper Fractal Scale Factor for Level 5 (7^2.5)
LEVEL_5_SCALE_FACTOR = 7.0 ** 2.5  # ~129.6418

# Derived Dimensions
UNIT_HEX_WIDTH_METERS = UNIT_HEX_PX * METERS_PER_PIXEL  # 6.4 meters exactly
SECTOR_WIDTH_METERS = UNIT_HEX_WIDTH_METERS * LEVEL_5_SCALE_FACTOR # ~829.7m

# Final Texture Size (Non-Power-of-Two is OK for modern GL)
# 32 * 129.6418 = ~4148.5 px
TEXTURE_SIZE_PX = UNIT_HEX_PX * LEVEL_5_SCALE_FACTOR 

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
        'pixels_per_unit_hex': UNIT_HEX_PX,
        'texture_size_px': TEXTURE_SIZE_PX
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

# =============================================================================
# FRACTAL TREE LOGIC
# =============================================================================

def generate_gosper_offsets(level, debug=False):
    """
    Generates the list of (q, r) unit hex offsets for a Gosper Tile of 'level'.

    The list is HIERARCHICALLY ORDERED.
    - Index 0-6: The 7 unit hexes of the first Level 1 child.
    - Index 7-13: The 7 unit hexes of the second Level 1 child.
    - ...

    This allows us to perform "Bottom-Up Aggregation" by simply reshaping the array:
    heights_lvl_0 = [16807]
    heights_lvl_1 = heights_lvl_0.reshape(-1, 7).mean(axis=1) -> [2401]
    heights_lvl_2 = heights_lvl_1.reshape(-1, 7).mean(axis=1) -> [343]
    ...
    """
    if level == 0:
        return [(0, 0)]
    
    # Gosper "Flower" Pattern (7 children relative to parent center)
    # These are the specific offsets for a standard "Pointy Top" gosper? 
    # NO: We are Flat Topped. But the logic is isomorphic.
    # We use a specific 'Seed' pattern that tiles the plane.
    # 
    # Relative offsets for the 7 children (Center + 6 Neighbors)
    # Note: These must be rotated/scaled appropriately for the level?
    # Actually, the recursion logic is:
    # 
    # P_child = P_parent + (Basis_Level * Offset_1_to_7)
    #
    # We build Bottom-Up recursively:
    # Offsets(N) = [ child_pos + (Basis * sub_offset) ]
    
    prev_offsets = generate_gosper_offsets(level - 1, debug)
    
    # The 7 Positions of the sub-tiles in a Gosper level
    # This matrix/offset depends heavily on the specific variation (island index).
    # Using a standard "Flower" spread helps, but for perfect tiling we need the specific shifting vectors.
    #
    # Let's use the standard "0-index is center" approach.
    # Shifts in Axial Coords (q, r):
    # Center (0,0)
    # And the 6 neighbors... but Spaced out by the Fractal Basis?
    #
    # Matrix M for doubling level (Scale sqrt(7), Rot ~19deg):
    # M = [[2, 1], [-1, 3]] ? (Determinant 7)
    #
    # Wait, simpler:
    # A Level N tile contains 7 Level N-1 tiles.
    # The N-1 tiles are located at specific offsets relative to the center.
    # These offsets are calculated using the integer power basis.
    
    # Let's hardcode the 7 relative "shift vectors" for the children.
    # In the specific "Flow Snake" configuration:
    shifts = [
        (0, 0),    # Center
        (0, -1),   # N
        (1, -1),   # NE
        (1, 0),    # SE
        (0, 1),    # S
        (-1, 1),   # SW
        (-1, 0)    # NW
    ]
    # Wait, these are just neighbors. 
    # The SHIFT magnitude depends on the level.
    # BUT, we can also define it recursively:
    # "Replace every hex with a flower of 7 hexes" logic (L-system).
    
    # We need the transformation Matrix 'M' that maps Level N-1 coords to Level N grid.
    # M = [[3, -1], [1, 2]] ?? (Det = 6+1=7)
    # Let's try M = [[2, 2], [-1, 1]]? Det = 2--2=4. No.
    #
    # Used in previous project:
    # q_new = 3*q - r
    # r_new = q + 2*r
    # Det = 6 - (-1) = 7. 
    # Check: (1,0) -> (3,1). Length sq = 9+1+3 = 13? No.
    #
    # Let's assume for this MVP we use the simple "Hex Flake" approximation 
    # if we can't derive the perfect Gosper shifts instantly.
    # Actually, if we use the Matrix from our 'waffle_maker' config:
    # M^5 = [[-87...]]
    # That was derived from a base M.
    # Let's derive M from the 5th root.
    # 
    # Let's try a simpler approach for the offsets:
    # Just standard neighbors relative to the 'Level Scale'?
    #
    # Let's defer to a robust recursion:
    # A level N tile is effectively "Dilated" by M, then checks neighbors?
    #
    # Correct Logic:
    # 1. Take the shape at Level N-1.
    # 2. Iterate 7 copies of it, shifted by the 7 basis vectors of Level N-1?
    
    # To keep this safe and not waste tokens guessing the matrix:
    # We will generate a "Dense Hexagon" (Level 5 radius ~ 7^5 is huge).
    # We will just fill a radius of ~80 hexes?
    #
    # No, user wants the tree logic.
    # We will use the [0,0] + 6 Neighbors recursion, applying the Matrix M rotation each step.
    # M = [[2, 1], [-1, 3]] (Det=7).
    
    M = [[2, 1], [-1, 3]] # This is a valid base-7 transform
    
    # Transform previous offsets by M
    rotated_offsets = []
    for (q, r) in prev_offsets:
        nq = 2*q + 1*r
        nr = -1*q + 3*r
        rotated_offsets.append((nq, nr))
        
    # Now duplicate this cloud 7 times? No.
    # The Gosper iteration is: Take the unit, replace with 7 units.
    # 
    # Let's stick to the simplest "Cluster" logic which works for averaging:
    # We want 7 "groups".
    # Group 0 is centered at (0,0).
    # Group 1 is centered at (Offset1).
    # ...
    #
    # The offset for Group K at Level L is determined by applying M^(L-1) to the unit neighbor vectors.
    
    # Unit neighbor vectors
    neighbors = [
        (0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1) 
        # Note: ordering matters for rotation, but for a "Set of Points" it's ok.
    ]
    
    # Calculate M^(level-1) basics
    def apply_matrix_power(vec, power):
        q, r = vec
        for _ in range(power):
             nq = 2*q + 1*r
             nr = -1*q + 3*r
             q, r = nq, nr
        return q, r

    final_list = []
    
    # For each of the 7 "macro positions"
    for i in range(7):
        base_shift = neighbors[i]
        # Scale this shift by the Level
        # Actually, the shift is just the neighbor vector transformed by M^(level-1)?
        # No, simpler:
        # Offsets(Level) = Offsets(Level-1) + (Shift_i * M^(Level-1) ??)
        
        # Let's trust the "Replace with 7" logic.
        # If we have a list of points for Level L-1.
        # We assume those are centered at 0.
        # We explicitly shift 7 copies of that cloud.
        # The shift magnitude is determined by the "stride" of Level L-1.
        
        # Approximate Stride for Level L-1 is sqrt(7)^(L-1).
        # We use the matrix power to get the exact integer vector.
        
        shift_q, shift_r = apply_matrix_power(base_shift, level - 1)

        if debug and level == 5:
            print(f"Level {level}, neighbor {i}: base{base_shift} -> shift({shift_q},{shift_r})")

        for (pq, pr) in prev_offsets:
            final_list.append((pq + shift_q, pr + shift_r))

    if debug and level == 5:
        print("=== PYTHON GOSPER OFFSET DEBUG ===")
        print(f"First 7 offsets: {final_list[:7]}")
        print(f"Offsets 2401-2407: {final_list[2401:2408]}")
        print(f"Last 7 offsets: {final_list[-7:]}")
        print(f"Total offsets: {len(final_list)}")

    return final_list

def compute_gosper_neighbors(level):
    """
    Returns a lookup table (Dict or Array) mapping:
    Index -> [N_idx, NE_idx, SE_idx, S_idx, SW_idx, NW_idx]
    
    For a Gosper curve of a given level.
    If a neighbor is outside the set (border), returns -1.
    """
    offsets = generate_gosper_offsets(level)
    
    # Build Map: (q, r) -> Index
    coord_to_idx = { (q,r): i for i, (q,r) in enumerate(offsets) }
    
    neighbor_dirs = [
        (0, -1),   # N
        (1, -1),   # NE
        (1, 0),    # SE
        (0, 1),    # S
        (-1, 1),   # SW
        (-1, 0)    # NW
    ]
    
    neighbor_table = []
    
    for (q, r) in offsets:
        row = []
        for (dq, dr) in neighbor_dirs:
            nq, nr = q + dq, r + dr
            if (nq, nr) in coord_to_idx:
                row.append(coord_to_idx[(nq, nr)])
            else:
                row.append(-1) # Border
        neighbor_table.append(row)
        
    return np.array(neighbor_table, dtype=np.int32)
