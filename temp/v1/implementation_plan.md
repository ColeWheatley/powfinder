# Piston Viewer Optimization & Fix Plan

## Problem Diagnosis
1.  **Texture Mismatch & Ratio Issues**: The `generate_oversized_tiles.py` script adds `20m` of padding to satellite textures to "extend" them. However, the frontend fragment shader's UV calculation (trying to compensate for this) creates aspect ratio distortions and misalignment between the DEM geometry and the texture.
2.  **Missing Slope Colors**: The "steep slope" coloring is failing because slope data is **never calculated or stored** in `hex_baker.py`. The frontend currently defaults all slopes to `90.0`, resulting in incorrect coloring.
3.  **Inefficient Geometry**: The frontend generates a full 6-sided hexagon (plus top/bottom), but only paints/needs the South, South-East, and South-West faces due to the camera angle (looking form North). The user specifically requested to "render the minimum number of triangles".
4.  **Complexity**: The "extender" logic adds unnecessary complexity and "bugs" at the edges.

## Proposed Solution: "Structural Simplification"

### 1. Remove Texture Padding (Fixes Mismatch)
We will abandon the "Oversized/Extender" approach.
-   **Action**: Create a new `generate_simple_tiles.py` script that converts raw TIFs to WebP 1:1, preserving exact bounds.
-   **Frontend**: Update `main.js` to remove the padding UV math. This ensures the texture perfectly matches the DEM grid (0 to 1 mapping).
-   **Edge Case**: Hexes at the very edge of the tile will source colors/pixels from the clampped edge of the texture. This is visually acceptable and far better than the current ratio distortion.

### 2. Implement "South-Facing Flux" Geometry (Saves GPU)
We will modify the geometry generation to strictly produce the requested faces.
-   **Frontend**: Rewrite `createHexGeometry` to generate **only** faces: Top, Bottom, SE, S, SW.
-   **Savings**: Removes N, NE, NW faces. Reduces triangle count for walls by 50%.
-   **Result**: "OpenGl expects to only render it from one direction" -> We assume the camera never looks from the South. If the user rotates 180, the back faces will be missing (invisible). This is the "Compromise" the user alluded to.

### 3. Calculate & Store Slope Data (Fixes Colors)
We will add slope data to the pipeline.
-   **Backend**: Update `hex_baker.py` to calculate slope (degrees) for each of the 3 faces (S, SE, SW).
    -   Formula: `slope = degrees(atan(abs(z - z_neighbor) / 10.0))`
-   **Binary Format**: Pack this slope into the unused "Alpha" byte of the color pairs?
    -   Current: `rgb_s_top` (3 bytes).
    -   New: `rgba_s_top` (4 bytes), where A = slope (uint8).
    -   Total size increase: 3 bytes per hex.
-   **Frontend**: Update `main.js` to parse this extra byte and pass it to the shader as `vFaceSlope`.

## Execution Steps
1.  **Stop** `generate_oversized_tiles.py`.
2.  **Run** new `generate_simple_tiles.py`.
3.  **Update** `hex_baker.py` to calculate/pack slopes.
4.  **Re-bake** the binary files.
5.  **Update** `main.js` (Geometry + Shader + Loader).

This plan directly addresses all user grievances: bugs from extending, missing slope colors, and GPU efficiency.
