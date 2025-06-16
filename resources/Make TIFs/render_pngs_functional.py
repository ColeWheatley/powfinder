#!/usr/bin/env python3
"""
render_pngs.py
--------------
Walk every temperature_2m.tif, relative_humidity_2m.tif, etc. inside
TIFS/100m_resolution/<timestamp>/ and emit a colour-mapped PNG
alongside each TIF.

Also handles terrain layers (elevation, slope, aspect) and outputs
them to TIFS/100m_resolution/terrainPNGs/

• Reads palettes/min/max from color_scales.json
• nodata turns into alpha=0 (fully transparent) so the PNG is clipped
  neatly to Tirol.
"""

import json, os, pathlib
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# Get the directory where this script is located
SCRIPT_DIR = pathlib.Path(__file__).parent
BASE = SCRIPT_DIR / "../../TIFS/100m_resolution"
TERRAIN_PNG_DIR = BASE / "terrainPNGs"
COLOR_SCALES_PATH = SCRIPT_DIR / "color_scales.json"
CS = json.load(open(COLOR_SCALES_PATH))

# Only publish a subset of timestamps to the web frontend
TIMESTAMPS = [
    "2025-05-24T09:00:00", "2025-05-24T12:00:00", "2025-05-24T15:00:00", "2025-05-24T18:00:00",
    "2025-05-25T09:00:00", "2025-05-25T12:00:00", "2025-05-25T15:00:00", "2025-05-25T18:00:00",
    "2025-05-26T09:00:00", "2025-05-26T12:00:00", "2025-05-26T15:00:00", "2025-05-26T18:00:00",
    "2025-05-27T09:00:00", "2025-05-27T12:00:00", "2025-05-27T15:00:00", "2025-05-27T18:00:00",
    "2025-05-28T09:00:00", "2025-05-28T12:00:00", "2025-05-28T15:00:00", "2025-05-28T18:00:00",
]

# Terrain TIF file mappings
TERRAIN_TIFS = {
    "elevation": SCRIPT_DIR / "../terrains/tirol_100m_float.tif",
    "slope": SCRIPT_DIR / "../terrains/tirol_slope_100m_float.tif",
    "aspect": SCRIPT_DIR / "../terrains/tirol_aspect_100m_float.tif",
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

def render_sqh_png(depth_tif: pathlib.Path, qual_tif: pathlib.Path, out_png: pathlib.Path, spec: dict):
    """Combine powder depth (opacity) and quality (colour) into a single PNG."""
    cmap = build_cmap(spec["palette"])
    vmin, vmax = spec.get("min", 0), spec.get("max", 1)

    with rasterio.open(depth_tif) as dsrc, rasterio.open(qual_tif) as qsrc:
        depth = dsrc.read(1).astype(float)
        qual = qsrc.read(1).astype(float)
        mask = (depth == dsrc.nodata) | np.isnan(depth)
        depth_norm = np.clip(depth / 255.0, 0, 1)
        qual_norm = np.clip((qual / 255.0 - vmin) / (vmax - vmin), 0, 1)

    rgba = cmap(qual_norm)
    rgba[..., 3] = depth_norm
    rgba[mask, 3] = 0.0
    plt.imsave(out_png, rgba)

def needs_update(tif_path: pathlib.Path, png_path: pathlib.Path, var_name: str) -> bool:
    """Check if PNG needs to be updated based on TIF modification time and color scale changes."""
    if not png_path.exists():
        return True
    
    png_mtime = png_path.stat().st_mtime
    
    # Check TIF timestamp
    tif_mtime = tif_path.stat().st_mtime
    if tif_mtime > png_mtime:
        return True
    
    # Check color scale timestamp for this specific variable
    spec = CS.get(var_name, {})
    if 'last_updated' in spec:
        from datetime import datetime
        import re
        timestamp = spec['last_updated']
        # Handle [Europe/Berlin] format by converting to standard format
        if '[Europe/Berlin]' in timestamp:
            timestamp = timestamp.replace('[Europe/Berlin]', '+02:00')
        elif 'Z' in timestamp:
            timestamp = timestamp.replace('Z', '+00:00')
        color_time = datetime.fromisoformat(timestamp).timestamp()
        if color_time > png_mtime:
            return True
    
    return False

def terrain_needs_update(png_path: pathlib.Path, var_name: str) -> bool:
    """Check if terrain PNG needs to be updated based on individual color scale timestamp."""
    if not png_path.exists():
        return True
    
    png_mtime = png_path.stat().st_mtime
    
    # Check color scale timestamp for this specific terrain variable
    spec = CS.get(var_name, {})
    if 'last_updated' in spec:
        from datetime import datetime
        import re
        timestamp = spec['last_updated']
        # Handle [Europe/Berlin] format by converting to standard format
        if '[Europe/Berlin]' in timestamp:
            timestamp = timestamp.replace('[Europe/Berlin]', '+02:00')
        elif 'Z' in timestamp:
            timestamp = timestamp.replace('Z', '+00:00')
        color_time = datetime.fromisoformat(timestamp).timestamp()
        if color_time > png_mtime:
            return True
    
    return False
    
    color_scales_mtime = COLOR_SCALES_PATH.stat().st_mtime
    png_mtime = png_path.stat().st_mtime
    
    return color_scales_mtime > png_mtime

def tif_to_png(tif_path: pathlib.Path):
    var = tif_path.stem  # Just use the stem directly

    spec = CS.get(var)
    if not spec:        # skip un-known variable names
        print(f"  ⚠️  No colour spec for {var} (from file {tif_path.name})")
        return

    # Special handling for SQH which combines powder depth and quality
    if var == "sqh":
        depth_path = tif_path.parent / "powder_depth.tif"
        quality_path = tif_path.parent / "powder_quality.tif"
        if not depth_path.exists() or not quality_path.exists():
            print(f"  ⚠️  Missing powder_depth/quality for {tif_path.parent.name}")
            return
        render_sqh_png(depth_path, quality_path, tif_path.parent / "sqh.png", spec)
        return
    
    # PNG name will be: temperature_2m.png
    png_name = var + ".png"
    out_png = tif_path.parent / png_name
    
    # Check if update is needed
    if not needs_update(tif_path, out_png, var):
        print(f"  ✓ {tif_path.relative_to(BASE)} → {png_name} (up to date)")
        return
    
    print(f"  → {tif_path.relative_to(BASE)} → {png_name} (updating)")

    cmap = build_cmap(spec["palette"])
    vmin, vmax = spec["min"], spec["max"]

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        mask = (data == src.nodata) | np.isnan(data)
        # TIF data is already 0-255, so normalize to 0-1 for colormap
        norm = np.clip(data / 255.0, 0, 1)

    rgba = cmap(norm)
    # Preserve colormap alpha, but set nodata areas to transparent
    rgba[mask, 3] = 0.0  # Only set nodata pixels to transparent
    
    plt.imsave(out_png, rgba)             # matplotlib writes RGBA PNG
    print(f"      ✓ PNG updated: {out_png.name}")

def render_terrain_png(var: str, tif_path: pathlib.Path):
    """Render a terrain TIF to PNG with proper data handling."""
    spec = CS.get(var)
    if not spec:
        print(f"  ⚠️  No colour spec for {var}")
        return
    
    # Create terrain PNG directory if it doesn't exist
    TERRAIN_PNG_DIR.mkdir(exist_ok=True)
    
    png_path = TERRAIN_PNG_DIR / f"{var}.png"
    
    # Check if update is needed based on color_scales.json
    if not terrain_needs_update(png_path, var):
        print(f"  ✓ terrain/{var}.png (up to date)")
        return
        
    print(f"  → terrain/{var}.png (updating)")

    cmap = build_cmap(spec["palette"])
    vmin, vmax = spec["min"], spec["max"]

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        mask = (data == src.nodata) | np.isnan(data)
        # Terrain data is in raw units, normalize to 0-1 for colormap
        norm = np.clip((data.astype(float) - vmin) / (vmax - vmin), 0, 1)

    rgba = cmap(norm)
    rgba[..., 3] = (~mask).astype(float)  # 1.0 where data, 0.0 where nodata
    
    plt.imsave(png_path, rgba)
    print(f"      ✓ PNG updated: {png_path.name}")

def main():
    updated_count = 0
    skipped_count = 0
    
    # Process weather/composite layers (timestamp-based TIFs)
    print("🌤️  Processing weather and composite layers...")
    for ts_dir in BASE.iterdir():
        if not ts_dir.is_dir() or ts_dir.name == "terrainPNGs":
            continue
        if ts_dir.name not in TIMESTAMPS:
            continue
        for tif in ts_dir.glob("*.tif"):
            var = tif.stem  # Get variable name from filename
            png_path = tif.parent / f"{tif.stem}.png"
            if needs_update(tif, png_path, var):
                tif_to_png(tif)
                updated_count += 1
            else:
                skipped_count += 1
    
    # Process terrain layers
    print("\n🏔️  Processing terrain layers...")
    terrain_updated = 0
    terrain_skipped = 0
    
    for var, tif_path in TERRAIN_TIFS.items():
        if tif_path.exists():
            png_path = TERRAIN_PNG_DIR / f"{var}.png"
            if terrain_needs_update(png_path, var):
                render_terrain_png(var, tif_path)
                terrain_updated += 1
            else:
                terrain_skipped += 1
                print(f"  ✓ terrain/{var}.png (up to date)")
        else:
            print(f"  ⚠️  Terrain TIF not found: {tif_path}")
    
    print(f"\n📊 Summary:")
    print(f"   Weather/Composite: {updated_count} updated, {skipped_count} skipped")
    print(f"   Terrain: {terrain_updated} updated, {terrain_skipped} skipped")

if __name__ == "__main__":
    main()