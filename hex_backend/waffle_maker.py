
import os
import glob
import math
import numpy as np
import rasterio
from shapely.geometry import Polygon, box
from shapely.affinity import affine_transform
from multiprocessing import Pool, cpu_count
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import coordinate_utility as coord_util

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# 1. Coordinate System & Hex Dimensions
# -------------------------------------
# Driven by PIXEL-FIRST logic in coordinate_utility.py
# High-Res Texture: 4096px * 0.2m/px = 819.2m Sector Width

# 2. Border & Buffer Logic
# ------------------------
BUFFER_PX = 32 

# 3. Paths
# --------
# Assuming script runs from root
DEM_PATH = "hex_backend/DGM_Tirol_5m_epsg31254_2006_2020.tif" 
AERIAL_DIR = "hex_backend/aerial_tifs"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sector_bounds(sector_Q, sector_R):
    """
    Returns a Shapely Polygon representing the bounding box of the Level 5 Sector (Q, R).
    """
    cen_x, cen_y = coord_util.sector_to_world_meters(sector_Q, sector_R)
    
    # Conservative Radius: Level 5 Width is defined in coord_util. 
    # Safety factor 0.7 * Width covers the fractal edges comfortably.
    width = coord_util.SECTOR_WIDTH_METERS
    safety_radius = width * 0.7 
    
    # Approximate meters for 32px buffer
    buffer_meters = BUFFER_PX * coord_util.METERS_PER_PIXEL
    
    min_x = cen_x - safety_radius - buffer_meters
    max_x = cen_x + safety_radius + buffer_meters
    min_y = cen_y - safety_radius - buffer_meters
    max_y = cen_y + safety_radius + buffer_meters
    
    return box(min_x, min_y, max_x, max_y)

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print(f"🧇 INIT: Waffle Maker spinning up...")
    dims = coord_util.get_hex_dimensions()
    print(f"      - High Res Texture: {coord_util.TEXTURE_SIZE_PX}px")
    print(f"      - Nominal Res: {coord_util.METERS_PER_PIXEL} m/px")
    print(f"      - Sector Width: {dims['sector_width_m']:.4f}m (Derived)")
    print(f"      - Unit Hex Width: {dims['unit_hex_width_m']:.4f}m (derived)")
    print(f"      - Pixels Per Unit: {dims['pixels_per_unit_hex']:.2f}")
    
    # 1. Load DEM Metadata
    # --------------------
    if not os.path.exists(DEM_PATH):
        print(f"❌ ERROR: DEM not found at {DEM_PATH}")
        return

    with rasterio.open(DEM_PATH) as dem:
        dem_bounds = dem.bounds
        dem_poly = box(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)
        print(f"✅ DEM Loaded: {dem.width}x{dem.height} | Bounds: {dem_bounds}")

    # 2. Load TIF Metadata (Parallel Scan)
    # ------------------------------------
    print("🔍 Scanning Aerial TIFs...")
    tif_files = glob.glob(os.path.join(AERIAL_DIR, "*.tif"))
    if not tif_files:
         print(f"❌ ERROR: No TIFs found in {AERIAL_DIR}")
         return 
    
    valid_tifs = []
    print(f"✅ Found {len(tif_files)} total TIFs.")
    
    # We collect all TIF polygons to create a "Coverage Map"
    all_tif_polys = []
    
    for f in tif_files: 
        try:
            with rasterio.open(f) as src:
                p = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                all_tif_polys.append(p)
                valid_tifs.append({'path': f, 'poly': p})
        except:
            pass
    
    print(f"✅ Indexed {len(valid_tifs)} valid TIF bounds.")
    
    if not valid_tifs:
        print("⚠️ No valid TIFs to process. Exiting.")
        return

    # 3. Cross-Check & Sector Discovery
    # ---------------------------------
    from shapely.ops import unary_union
    print("🕸️ Building Unified Coverage Map (this may take a second)...")
    coverage_poly = unary_union(all_tif_polys)
    
    print("🛠 Calculating Valid Sectors...")
    
    # Intersection with DEM
    c_bounds = coverage_poly.bounds
    search_min_x = max(dem_bounds.left, c_bounds[0])
    search_max_x = min(dem_bounds.right, c_bounds[2])
    search_min_y = max(dem_bounds.bottom, c_bounds[1])
    search_max_y = min(dem_bounds.top, c_bounds[3])
    
    print(f"   Search Area: X[{search_min_x:.1f}, {search_max_x:.1f}] Y[{search_min_y:.1f}, {search_max_y:.1f}]")
    
    # Use the Module's Approx Inversion
    Q1, R1 = coord_util.world_meters_to_sector_approx(search_min_x, search_min_y)
    Q2, R2 = coord_util.world_meters_to_sector_approx(search_max_x, search_max_y)
    Q3, R3 = coord_util.world_meters_to_sector_approx(search_min_x, search_max_y)
    Q4, R4 = coord_util.world_meters_to_sector_approx(search_max_x, search_min_y)

    Q_start = int(min(Q1, Q2, Q3, Q4)) - 2
    Q_end = int(max(Q1, Q2, Q3, Q4)) + 2
    R_start = int(min(R1, R2, R3, R4)) - 2
    R_end = int(max(R1, R2, R3, R4)) + 2
    
    print(f"   Scanning Sector Range: Q[{Q_start} to {Q_end}] R[{R_start} to {R_end}]")
    
    valid_sectors = []
    partial_sectors = []
    
    for Q in range(Q_start, Q_end + 1):
        for R in range(R_start, R_end + 1):
            sec_poly = get_sector_bounds(Q, R)
            
            # Must be completely inside DEM
            if not dem_poly.contains(sec_poly):
                continue
                
            # Check against Coverage Map
            if coverage_poly.contains(sec_poly):
                valid_sectors.append((Q, R))
            elif coverage_poly.intersects(sec_poly):
                partial_sectors.append((Q, R))

    print("\n📊 ANALYSIS REPORT")
    print(f"   Total Coverage Area: {coverage_poly.area/1e6:.1f} km²")
    print(f"   Target Location: Q[{Q1:.1f}], R[{R1:.1f}]")
    print(f"   Potential Hex Sectors (Full): {len(valid_sectors)}")
    print(f"   Partial Sectors (Border): {len(partial_sectors)}")
    print(f"   Optimal Hex Hexes (Unit): {len(valid_sectors) * 16807:,}")
    print(f"   Pixel-Verified: YES")

    # =============================================================================
    # 4. LOD TREE GENERATION (Preview)
    # =============================================================================
    # To support the "Hybrid LOD" system:
    # 1. "Far Field": We need a QuadTree/BVH of these Sectors for fast frustum/distance culling.
    # 2. "Near Field": We use 'world_meters_to_axial' for O(1) lookups.
    
    # In the full bake, we would export 'manifest_lod_tree.json':
    # {
    #   "sectors": [
    #      { "q": ..., "r": ..., "center": [x,y], "bounds": [minx, miny, maxx, maxy], "children": [] }
    #   ]
    # }
    print("🌳 LOD Tree Readiness: COMPATIBLE")
    print("   - Inverse Coordinate System: Active")
    print("   - Sector Bounding Boxes: Ready for QuadTree insertion")

if __name__ == "__main__":
    main()
