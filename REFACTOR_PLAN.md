# Refactor Plan: Precision Piston

## 1. The "Cardinal" Coordinate Standard
We will establish a strict, universal coordinate system across the Python baker, the binary format, and the JavaScript shaders.

*   **World Axis**:
    *   **North (-Z)**: 0 degrees / Top.
    *   **East (+X)**: 90 degrees / Right.
*   **Indices (0-5) Clockwise**:
    *   `0`: **North** (-Z)
    *   `1`: **North-East** (+X, -Z)
    *   `2`: **South-East** (+X, +Z)
    *   `3`: **South** (+Z)
    *   `4`: **South-West** (-X, +Z)
    *   `5`: **North-West** (-X, -Z)
*   **Rule**: All arrays (neighbors, face indices, lighting calculations) MUST follow this order.

## 2. Hexagonal Tiling Strategy ("Mega Hexes")
To solve the meshing gaps, we will abandon rectangular tiles in favor of **Fractal Hex Tiles**.

*   **Structure**:
    *   **Unit Hex**: The visible 10m Piston.
    *   **Chunk (Tile)**: A large hexagon containing a fixed arrangement of Unit Hexes (e.g., radius 7 hexes = ~169 units).
    *   **Addressing**: Use **Axial Coordinates (q, r)** for file naming (`tile_q_r.bin`).
*   **Benefit**: Tiles interlock perfectly like puzzle pieces. No "gear teeth" edges.
*   **Textures**: We will continue using rectangular WebPs.
    *   *Implementation*: The `tile_q_r.bin` will define a `boundingBox` (min/max X/Z). The corresponding WebP will be a rectangle covering this box. The shader will comfortably sample the texture; transparent pixels will naturally fall outside the hex geometry.

## 3. Data Compression & Binary Format (v2.0)
We can optimize the payload while increasing precision.

**Proposed Schema (Per Tile):**
*   **Header**:
    *   `MapBaseElevation` (Float32): Global offset (e.g., 1000m) to keep precise floats small.
    *   `TileQ`, `TileR`: Coordinates.
*   **Per Hex (14 bytes)**:
    *   **Center Z** (Float16): Absolute elevation relative to `MapBaseElevation`.
    *   **Neighbor Zs** (6 × Float16): **Absolute** elevation of neighbors relative to `MapBaseElevation`.
        *   *Why Absolute?*: Calculating offsets (`Center - Neighbor`) in the shader for millions of hexes costs ALU cycles. Passing absolute values allows the vertex shader to simply read `neighborZ` and process skirt logic immediately. Matches current refactor direction.

## 4. Rendering & Geometry (The "Native" Coordinate Fix)
We will replace the ad-hoc `faceDirs` matching with an explicit, mathematically verified array.

*   **Explicit Attributes**:
    *   `position`: Vertices generated manually.
    *   `faceIndex`: An un-interpolated float attribute.
*   **Implementation Detail**:
    *   The `faceDirs` in `createHexGeometry` must optimally match the standard:
        *   `0: (0, -1)` (N)
        *   `1: (0.866, -0.5)` (NE)
        *   `2: (0.866, 0.5)` (SE)
        *   `3: (0, 1)` (S)
        *   `4: (-0.866, 0.5)` (SW)
        *   `5: (-0.866, -0.5)` (NW)

## 5. Hierarchical Data (LOD)
To address the "Huge Hexes" issue:

*   **LOD 0 (Close)**: User loads `tile_q_r.bin` (Full 10m resolution).
*   **LOD 1 (Far)**: We generate a separate dataset of **Super Hexes** (lower res).
    *   Instead of loading 49 high-res files, the viewer loads 1 "Super Tile" that covers the same area with larger, flatter hexes.
    *   This avoids the "raising in the air" issue by simply rendering larger pistons at the mean elevation of their children.
