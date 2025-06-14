#!/usr/bin/env python3
"""
render_pngs.py
--------------
Walk every temperature_2m.tif, relative_humidity_2m.tif, etc. inside
TIFS/100m_resolution/<timestamp>/ and emit a colour-mapped PNG
alongside each TIF.

• Reads palettes/min/max from color_scales.json
• nodata turns into alpha=0 (fully transparent) so the PNG is clipped
  neatly to Tirol.
"""

import json, os, pathlib
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

BASE = pathlib.Path("../../TIFS/100m_resolution")
# Get color_scales.json from the same directory as this script
SCRIPT_DIR = pathlib.Path(__file__).parent
CS = json.load(open(SCRIPT_DIR / "color_scales.json"))

def build_cmap(palette_hex):
    return LinearSegmentedColormap.from_list("pal", palette_hex, N=256)

def tif_to_png(tif_path: pathlib.Path):
    var = tif_path.stem  # Just use the stem directly
    
    spec = CS.get(var)
    if not spec:        # skip un-known variable names
        print(f"  ⚠️  No colour spec for {var} (from file {tif_path.name})")
        return
    print(f"  → {tif_path.relative_to(BASE)} …")

    cmap = build_cmap(spec["palette"])
    vmin, vmax = spec["min"], spec["max"]

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        mask = (data == src.nodata) | np.isnan(data)
        # TIF data is already 0-255, so normalize to 0-1 for colormap
        norm = np.clip(data / 255.0, 0, 1)

    rgba = cmap(norm)
    rgba[..., 3] = (~mask).astype(float)  # 1.0 where data, 0.0 where nodata
    
    # PNG name will be: temperature_2m.png
    png_name = var + ".png"
    out_png = tif_path.parent / png_name
    plt.imsave(out_png, rgba)             # matplotlib writes RGBA PNG
    print(f"      PNG saved as {out_png.name}")

def main():
    for ts_dir in BASE.iterdir():
        if not ts_dir.is_dir(): continue
        for tif in ts_dir.glob("*.tif"):
            tif_to_png(tif)

if __name__ == "__main__":
    main()