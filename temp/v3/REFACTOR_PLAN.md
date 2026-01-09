# Refactor Plan: Operation "Precision Piston"

## 0. Sequential Baking Pipeline
1.  **Index Data Sources**: Load all available TIFs and DEM files.
2.  **Coverage Analysis**: Intersection of Satellite + DEM to define valid Level 5 (1km) "Sector" hexes.
3.  **No-Rotation Policy**: All hexes are strictly North-facing.
4.  **Sequential Bake**: Process by Sector. If a Sector has both data types, pack 16,807 unit hexes.

## 1. The "Cardinal" Coordinate Standard
A strict, hard-coded standard across Python and JS.
*   **Zero**: North (Positive Y in World/Baker, Negative Z in Three.js).
*   **Direction**: Clockwise.
*   **Indices**: 0: N, 1: NE, 2: SE, 3: S, 4: SW, 5: NW.
*   **Implementation**:
    *   **Baker**: `OFFSETS` array aligned to this. Use **Lanczos Resampling** for ridge/cliff sharpness (Bicubic as fallback) when sampling the DEM for hex centers.
    *   **Frontend**: `createHexGeometry` explicitly maps face vertices to these slots.

## 2. Fractal Hex Tiling
Switch from rectangular tiles to hexagonal chunks (LOD-compatible).
*   **Tile Shape**: Large hexagons composed of $7^n$ unit hexes.
*   **Addressing**: Axial Coordinates (q, r).

## 3. Binary Format v2.0
*   **Elevation**: Delta encoded (Float32 base + Uint16/Float16 offset) for high precision.
*   **Layer Data**: Decoupled binary streams for Snow, Wind, etc.
*   **Multi-Res Gradient Baking**: When baking color/slope data for different hex resolutions (Small/Med/Large), logic must account for the changing horizontal distance ($X$) in the rise/run calculation. Higher-res "sub-skirts" can be blended mapped to lower-res parent hexes to maintain sharp visual gradients even at distance.

## 4. Rendering & Performance
*   **Unified WebGL Pipeline (Goodbye CSS)**:
    *   **The Problem**: Syncing DOM/CSS 2D transforms to a 3D camera causes mathematical lag ("filthy" transitions) and UI complexity.
    *   **The Solution**: Move all "2D" states into the WebGL renderer.
    *   **State 1 (Flat Mode)**: When viewed top-down, tiles are rendered as single, static textured quads at $z=0$. This is computationally identical to 2D rendering.
    *   **State 2 (Transition)**: As the camera tilts, the flat quads are replaced by the "Instanced Hex Mesh". Because they share the same world coordinate origin, the handoff is invisible.
    *   **The Reveal**: Use the `uHeightFactor` to "shatter" the flat map into individual pistons that rise to their real elevations.
*   **Overshoot Texturing (WebP Optimization)**: 
    *   **Justification**: WebP uses 16x16 macroblocks. High-contrast edges (Satellite vs. Black padding) cause artifacts if they fall mid-block.
    *   **Implementation**: Crop satellite textures with a **32px overshoot** past the hex boundary. The 3D hex mesh acts as a clean "cookie cutter," hiding the compression-muddled edge while gaining the file-size benefits of the black padding.
    *   **Compression Note**: We utilize an aggressive **10% quality** WebP setting to keep mobile loads fast. Interestingly, tests show that **Original Resolution TIFs are actually smaller** than 90% quality WebPs, though WebP remains the target for browser compatibility.
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
*   **Android Battery API**: Query Android Battery Status API to detect low-battery mode; automatically reduce frame rate, texture quality, and LOD aggressiveness when device battery is critically low.

## 6. Camera 
*   **Pivot-Based Rig**: The camera orbits a `PivotPoint` locked to the terrain surface at screen-center.
*   **Surface-Aware Translation**: As pistons rise, the `PivotPoint` rises with them, ensuring the spot the user is looking at remains stable in the viewport during the 2D->3D transition.
*   **Dynamic Culling**: Use frustum culling on tiles (and future occlusion maps) to minimize vertex density.
*   **3D-Aware LOD Strategy (The "Peak & Valley" Check)**: 
    *   **Architecture Clarification (Vertices vs. Textures)**:
        *   **The Bottleneck**: It's **Vertices**, not VRAM. We have millions of hexes.
        *   **Texture Strategy**: Lazy load. Network constrained. But once loaded, render freely (~1MB files are cheap).
            *   *Compressed (High Res)*: For nearby terrain.
            *   *Low Res*: For distant horizons.
        *   **Geometry Strategy**: Aggressive Vertex Reduction.
            *   *Unit Hexes*: Render dense geometry ONLY where needed (Slope > Threshold or Camera < Distance).
            *   *Sector Hexes*: Render huge ~800m hexes for distant valleys or flat areas.
    *   **The Check (Conservative Heuristic)**:
        1.  **Proximal Check**: Is the camera within distance $D$ of the closest point on the tile's bounding cylinder? (`Distance - Radius`). If NO, stop (Low Res/Cull).
        2.  **Elevation Check**: Since we are in the mountains, check vertical distance. 
            *   Is the camera close to `maxZ`? (User is flying near a peak). If YES -> Force High-Res recursion.
            *   Is the camera far above `minZ`? (User is high above a deep valley). If YES -> Allow Low-Res.
    *   **Efficiency**: This "Dual Lookup" replaces iterating thousands of child hexes. If the *entire* 1km tile bounding volume fits the "Low Res" criteria (far away AND deep down), we render 1 big hex (Sector Mesh). If ANY part of it (the peak) is close, we drill down (Unit Mesh).
