# Refactor Plan: Operation "Precision Piston"

## 1. The "Cardinal" Coordinate Standard
A strict, hard-coded standard across Python and JS.
*   **Zero**: North (Positive Y in World/Baker, Negative Z in Three.js).
*   **Direction**: Clockwise.
*   **Indices**: 0: N, 1: NE, 2: SE, 3: S, 4: SW, 5: NW.
*   **Implementation**:
    *   **Baker**: `OFFSETS` array aligned to this.
    *   **Frontend**: `createHexGeometry` explicitly maps face vertices to these slots.

## 2. Fractal Hex Tiling
Switch from rectangular tiles to hexagonal chunks (LOD-compatible).
*   **Tile Shape**: Large hexagons composed of $7^n$ unit hexes.
*   **Addressing**: Axial Coordinates (q, r).

## 3. Binary Format v2.0
*   **Elevation**: Delta encoded (Float32 base + Uint16/Float16 offset) for high precision.
*   **Layer Data**: Decoupled binary streams for Snow, Wind, etc.

## 4. Rendering & Performance
*   **Unified WebGL Pipeline (Goodbye CSS)**:
    *   **The Problem**: Syncing DOM/CSS 2D transforms to a 3D camera causes mathematical lag ("filthy" transitions) and UI complexity.
    *   **The Solution**: Move all "2D" states into the WebGL renderer.
    *   **State 1 (Flat Mode)**: When viewed top-down, tiles are rendered as single, static textured quads at $z=0$. This is computationally identical to 2D rendering.
    *   **State 2 (Transition)**: As the camera tilts, the flat quads are replaced by the "Instanced Hex Mesh". Because they share the same world coordinate origin, the handoff is invisible.
    *   **The Reveal**: Use the `uHeightFactor` to "shatter" the flat map into individual pistons that rise to their real elevations.
*   **Instancing**: Use `InstancedMesh` with an "Ideal Hex" and `aNeighborSlot` attribute.
*   **Ideal Hex**: Optimized indexed geometry (~19 vertices vs ~36).
    *   **Flat Tops**: Strictly horizontal tops (Normal: 0,1,0). No tilting or cross-product calculations on geometry.
    *   **Baked Normals**: Use a single integer `FaceID (0-5)` attribute per vertex. Shader looks up cardinal normals from a constant table instead of storing 3 floats per vertex.
    *   **Variable Framerates & On-Demand Rendering**:
    *   **State**: The renderer should be "reactive."
    *   **Triggers**: Render only when:
        *   User controls are active (panning, tilting, zooming).
        *   Piston animations are running (e.g., during 2D -> 3D transition).
        *   New data (Tiles/Textures) has finished loading.
    *   **Benefit**: Massive battery saving on mobile by dropping to 0 FPS when the map is static.
    *   **Occlusion Culling (Future)**: Calculate a low-resolution occlusion map to identify and skip rendering for hexes that are fully occluded (e.g., hidden in the center of a dense cluster).

## 5. Mobile Optimizations
*   **Adaptive Frame Capping**: Detect OS; if Android/iOS, hard-cap the rendering at 60 FPS (even on 120Hz ProMotion displays) to prioritize battery life over extreme smoothness.
*   **Dynamic UI**: Auto-minimize performance metrics/debug panels on smaller screens to maximize map visibility.
*   **Orientation-Aware Load**: Detect Portrait vs. Horizontal orientation on initial load and adjust the initial camera FOV or starting zoom to fit the device aspect ratio.

## 6. Camera 
*   **Pivot-Based Rig**: The camera orbits a `PivotPoint` locked to the terrain surface at screen-center.
*   **Surface-Aware Translation**: As pistons rise, the `PivotPoint` rises with them, ensuring the spot the user is looking at remains stable in the viewport during the 2D->3D transition.
*   **Dynamic Culling**: Use frustum culling on tiles (and future occlusion maps) to minimize vertex density.
