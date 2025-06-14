#!/usr/bin/env python3
"""
tile_pngs.py
------------
Take every *.png produced by render_pngs.py and cut it into standard
256×256 Web-mercator-friendly tiles.

Tiles saved as:

  tiles/<timestamp>/<variable>/<x>_<y>.png
"""

import os, pathlib
from PIL import Image

PNG_BASE  = pathlib.Path("TIFS/100m_resolution")
TILE_ROOT = pathlib.Path("tiles")
TILE_SIZE = 256

def slice_png(png_path: pathlib.Path):
    var       = png_path.stem
    timestamp = png_path.parent.name
    out_dir   = TILE_ROOT / timestamp / var
    out_dir.mkdir(parents=True, exist_ok=True)

    im = Image.open(png_path).convert("RGBA")
    w, h = im.size
    xtiles = (w + TILE_SIZE - 1) // TILE_SIZE
    ytiles = (h + TILE_SIZE - 1) // TILE_SIZE

    for x in range(xtiles):
        for y in range(ytiles):
            left   = x * TILE_SIZE
            upper  = y * TILE_SIZE
            right  = min(left + TILE_SIZE, w)
            lower  = min(upper + TILE_SIZE, h)
            tile   = im.crop((left, upper, right, lower))
            # Skip fully transparent tiles
            if not tile.getbbox():
                continue
            tile.save(out_dir / f"{x}_{y}.png")

    print(f"✔️  {var} @ {timestamp}: {xtiles*ytiles} tiles written")

def main():
    for ts_dir in PNG_BASE.iterdir():
        if not ts_dir.is_dir(): continue
        for png in ts_dir.glob("*.png"):
            slice_png(png)

if __name__ == "__main__":
    main()