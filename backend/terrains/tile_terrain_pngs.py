#!/usr/bin/env python3
"""Slice terrain PNGs into 256x256 tiles."""

import pathlib
from PIL import Image

PNG_DIR = pathlib.Path(__file__).parent / "png"
TILE_ROOT = pathlib.Path("tiles/terrain")
TILE_SIZE = 256


def slice_png(png_path):
    var = png_path.stem
    out_dir = TILE_ROOT / var
    out_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(png_path).convert("RGBA")
    w, h = im.size
    xtiles = (w + TILE_SIZE - 1) // TILE_SIZE
    ytiles = (h + TILE_SIZE - 1) // TILE_SIZE
    for x in range(xtiles):
        for y in range(ytiles):
            left = x * TILE_SIZE
            upper = y * TILE_SIZE
            right = min(left + TILE_SIZE, w)
            lower = min(upper + TILE_SIZE, h)
            tile = im.crop((left, upper, right, lower))
            if not tile.getbbox():
                continue
            tile.save(out_dir / f"{x}_{y}.png")


def main():
    for png in PNG_DIR.glob("*.png"):
        slice_png(png)

if __name__ == "__main__":
    main()
