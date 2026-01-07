#!/usr/bin/env python3
"""
render_webps.py
---------------
Convert TIFs directly to WebP format with aggressive compression (60-70%).

This script:
1. Reads TIFs with weather/terrain data
2. Applies color mapping from color_scales.json
3. Outputs WebP files with 60-70% quality
4. Skips PNG intermediate (faster, saves space)

Usage:
    python render_webps.py

Output:
    TIFS/100m_resolution/<timestamp>/<variable>.webp
    TIFS/100m_resolution/terrainWebPs/<variable>.webp
"""

import json
import os
import pathlib
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import sys

# Configuration
SCRIPT_DIR = pathlib.Path(__file__).parent
BASE = SCRIPT_DIR / "../../TIFS/100m_resolution"
TERRAIN_WEBP_DIR = BASE / "terrainWebPs"
COLOR_SCALES_PATH = SCRIPT_DIR / "color_scales.json"
WEBP_QUALITY = 65  # 60-70% quality for aggressive compression

# Load color scales
try:
    CS = json.load(open(COLOR_SCALES_PATH))
except Exception as e:
    print(f"❌ Error loading color_scales.json: {e}")
    sys.exit(1)

# Frontend timestamps (4 per day: 9am, 12pm, 3pm, 6pm)
FRONTEND_TIMESTAMPS = [
    # Dec 25-31, 2025
    "2025-12-25T09:00:00", "2025-12-25T12:00:00", "2025-12-25T15:00:00", "2025-12-25T18:00:00",
    "2025-12-26T09:00:00", "2025-12-26T12:00:00", "2025-12-26T15:00:00", "2025-12-26T18:00:00",
    "2025-12-27T09:00:00", "2025-12-27T12:00:00", "2025-12-27T15:00:00", "2025-12-27T18:00:00",
    "2025-12-28T09:00:00", "2025-12-28T12:00:00", "2025-12-28T15:00:00", "2025-12-28T18:00:00",
    "2025-12-29T09:00:00", "2025-12-29T12:00:00", "2025-12-29T15:00:00", "2025-12-29T18:00:00",
    "2025-12-30T09:00:00", "2025-12-30T12:00:00", "2025-12-30T15:00:00", "2025-12-30T18:00:00",
    "2025-12-31T09:00:00", "2025-12-31T12:00:00", "2025-12-31T15:00:00", "2025-12-31T18:00:00",
    # Jan 1-7, 2026
    "2026-01-01T09:00:00", "2026-01-01T12:00:00", "2026-01-01T15:00:00", "2026-01-01T18:00:00",
    "2026-01-02T09:00:00", "2026-01-02T12:00:00", "2026-01-02T15:00:00", "2026-01-02T18:00:00",
    "2026-01-03T09:00:00", "2026-01-03T12:00:00", "2026-01-03T15:00:00", "2026-01-03T18:00:00",
    "2026-01-04T09:00:00", "2026-01-04T12:00:00", "2026-01-04T15:00:00", "2026-01-04T18:00:00",
    "2026-01-05T09:00:00", "2026-01-05T12:00:00", "2026-01-05T15:00:00", "2026-01-05T18:00:00",
    "2026-01-06T09:00:00", "2026-01-06T12:00:00", "2026-01-06T15:00:00", "2026-01-06T18:00:00",
    "2026-01-07T09:00:00", "2026-01-07T12:00:00", "2026-01-07T15:00:00", "2026-01-07T18:00:00",
]

# Terrain TIF file mappings
TERRAIN_TIFS = {
    "elevation": SCRIPT_DIR / "../terrains/tirol_100m_float.tif",
    "slope": SCRIPT_DIR / "../terrains/tirol_slope_100m_float.tif",
    "aspect": SCRIPT_DIR / "../terrains/tirol_aspect_100m_web.tif",
}

def build_cmap(palette_hex):
    """Build colormap from hex colors or RGBA strings"""
    colors = []
    for color in palette_hex:
        if color.startswith('rgba('):
            # Parse rgba(r,g,b,a) format
            rgba_str = color[5:-1]  # Remove 'rgba(' and ')'
            r, g, b, a = map(float, rgba_str.split(','))
            colors.append((r/255, g/255, b/255, a))
        else:
            # Assume hex format
            colors.append(color)
    return LinearSegmentedColormap.from_list("pal", colors, N=256)

def tif_to_webp(tif_path: pathlib.Path, out_webp: pathlib.Path, spec: dict):
    """Convert TIF to WebP with color mapping."""
    cmap = build_cmap(spec["palette"])
    var = tif_path.stem

    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1).astype(float)
            mask = (data == src.nodata) | np.isnan(data)
            # Normalize data to 0-1 for colormap
            norm = np.clip(data / 255.0, 0, 1)

        # Apply colormap to get RGBA
        rgba = cmap(norm)
        # Set nodata areas to transparent
        rgba[mask, 3] = 0.0

        # Convert to PIL Image and save as WebP
        img = Image.fromarray((rgba * 255).astype(np.uint8), 'RGBA')
        out_webp.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_webp, 'WEBP', quality=WEBP_QUALITY, method=6)

        # Calculate compression ratio
        file_size_mb = out_webp.stat().st_size / (1024 * 1024)
        return True, file_size_mb

    except Exception as e:
        print(f"      ❌ Error converting {tif_path.name}: {e}")
        return False, 0

def render_terrain_webp(var: str, tif_path: pathlib.Path):
    """Render terrain TIF to WebP."""
    spec = CS.get(var)
    if not spec:
        print(f"  ⚠️  No colour spec for {var}")
        return False

    if not tif_path.exists():
        print(f"  ⚠️  Terrain TIF not found: {tif_path}")
        return False

    TERRAIN_WEBP_DIR.mkdir(exist_ok=True)
    webp_path = TERRAIN_WEBP_DIR / f"{var}.webp"

    success, size = tif_to_webp(tif_path, webp_path, spec)
    if success:
        print(f"  ✓ terrain/{var}.webp ({size:.2f}MB @ {WEBP_QUALITY}% quality)")
        return True
    return False

def render_sqh_webp(depth_tif: pathlib.Path, qual_tif: pathlib.Path, out_webp: pathlib.Path, spec: dict):
    """Combine powder depth (opacity) and quality (color) into WebP."""
    cmap = build_cmap(spec["palette"])
    vmin, vmax = spec.get("min", 0), spec.get("max", 1)

    try:
        with rasterio.open(depth_tif) as dsrc, rasterio.open(qual_tif) as qsrc:
            depth = dsrc.read(1).astype(float)
            qual = qsrc.read(1).astype(float)
            mask = (depth == dsrc.nodata) | np.isnan(depth)

            # Convert from scaled uint8 back to physical units
            depth_phys = (depth - 1) / 254.0 * 100.0
            qual_phys = (qual - 1) / 254.0

            # Normalize for visualization
            depth_norm = np.clip(depth_phys / 100.0, 0, 1)
            qual_norm = np.clip((qual_phys - vmin) / (vmax - vmin), 0, 1)

        rgba = cmap(qual_norm)
        rgba[..., 3] = depth_norm
        rgba[mask, 3] = 0.0

        img = Image.fromarray((rgba * 255).astype(np.uint8), 'RGBA')
        out_webp.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_webp, 'WEBP', quality=WEBP_QUALITY, method=6)

        file_size_mb = out_webp.stat().st_size / (1024 * 1024)
        return True, file_size_mb

    except Exception as e:
        print(f"      ❌ Error rendering SQH WebP: {e}")
        return False, 0

def main():
    print("🌐 Converting TIFs to WebP (Direct TIF → WebP, no PNG intermediate)")
    print("=" * 70)
    print(f"Quality: {WEBP_QUALITY}% (Aggressive Compression)")
    print(f"Output: {BASE}")
    print("=" * 70)

    updated_count = 0
    skipped_count = 0
    total_size_mb = 0

    # Process weather/composite layers (ONLY for frontend timestamps)
    print("\n🌤️  Processing weather and composite layers...")
    for ts_dir in sorted(BASE.iterdir()):
        if not ts_dir.is_dir() or ts_dir.name in ["terrainPNGs", "terrainWebPs"]:
            continue
        if ts_dir.name not in FRONTEND_TIMESTAMPS:
            continue

        for tif in sorted(ts_dir.glob("*.tif")):
            var = tif.stem
            webp_path = tif.parent / f"{var}.webp"

            # Skip powder depth/quality individual files (handled as SQH)
            if var in ["powder_depth", "powder_quality"]:
                # Generate SQH WebP from powder depth and quality
                if var == "powder_depth":
                    qual_path = tif.parent / "powder_quality.tif"
                    sqh_webp = tif.parent / "sqh.webp"
                    if qual_path.exists():
                        spec = CS.get("sqh", {})
                        if spec:
                            success, size = render_sqh_webp(tif, qual_path, sqh_webp, spec)
                            if success:
                                print(f"  → sqh.webp ({size:.2f}MB @ {WEBP_QUALITY}%)")
                                total_size_mb += size
                                updated_count += 1
                continue

            spec = CS.get(var)
            if not spec:
                print(f"  ⚠️  No colour spec for {var}")
                skipped_count += 1
                continue

            success, size = tif_to_webp(tif, webp_path, spec)
            if success:
                print(f"  → {var}.webp ({size:.2f}MB @ {WEBP_QUALITY}%)")
                total_size_mb += size
                updated_count += 1
            else:
                skipped_count += 1

    # Process terrain layers
    print("\n🏔️  Processing terrain layers...")
    for var, tif_path in TERRAIN_TIFS.items():
        if render_terrain_webp(var, tif_path):
            # Get file size for counting
            webp_size = (TERRAIN_WEBP_DIR / f"{var}.webp").stat().st_size / (1024 * 1024)
            total_size_mb += webp_size
            updated_count += 1
        else:
            skipped_count += 1

    print("\n" + "=" * 70)
    print(f"✅ Conversion Complete")
    print(f"   Files created: {updated_count}")
    print(f"   Files skipped: {skipped_count}")
    print(f"   Total size: {total_size_mb:.1f}MB")
    print(f"   Quality: {WEBP_QUALITY}% (Compression: ~{100-WEBP_QUALITY}%)")
    print("=" * 70)

if __name__ == "__main__":
    main()
