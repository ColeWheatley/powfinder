#!/usr/bin/env python3
"""Render terrain TIFs (DEM, slope, aspect) to colour PNGs."""

import json
import pathlib
import rasterio
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

SCRIPT_DIR = pathlib.Path(__file__).parent
TIF_DIR = SCRIPT_DIR
PNG_DIR = SCRIPT_DIR / "png"
COLOR_SCALES = json.load(open(SCRIPT_DIR.parent / "Make TIFs" / "color_scales.json"))

TIFS = {
    "elevation": TIF_DIR / "tirol_100m_web.tif",
    "slope": TIF_DIR / "tirol_slope_100m_web.tif",
    "aspect": TIF_DIR / "tirol_aspect_100m_web.tif",
}

PNG_DIR.mkdir(exist_ok=True)


def build_cmap(pal):
    return LinearSegmentedColormap.from_list("pal", pal, N=256)


def render(tif_path, var):
    spec = COLOR_SCALES.get(var)
    if not spec:
        print(f"No color scale for {var}")
        return
    cmap = build_cmap(spec["palette"])
    vmin = spec.get("min", 0)
    vmax = spec.get("max", 1)

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        mask = data == src.nodata
        norm = np.clip((data.astype(float) - vmin) / (vmax - vmin), 0, 1)

    rgba = cmap(norm)
    rgba[..., 3] = (~mask).astype(float)

    out_png = PNG_DIR / f"{var}.png"
    plt.imsave(out_png, rgba)
    print("Wrote", out_png)


def main():
    for var, path in TIFS.items():
        if path.exists():
            render(path, var)

if __name__ == "__main__":
    main()
