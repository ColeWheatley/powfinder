PowFinder Piston: Technical Development Guidelines
Version: Beta 0.1 Design Phase Core Concept: A hybrid 3D visualization tool for ski touring that blends high-resolution winter satellite imagery with a procedurally generated "Hex Piston" mesh to visualize terrain, slope, and snow quality.

1. Architectural Overview
Philosophy
Static & Serverless: No active backend server (Python/Node) required for the viewer.

Pre-Computed: All heavy lifting (DEM interpolation, Hex generation) is done offline.

Browser-First: The client handles rendering. The cloud (S3) just serves static assets.

Tech Stack
Data Processing (Offline): Python (Rasterio, NumPy, PyVista for prototyping).

Hosting: AWS S3 (Static file hosting for web assets and data tiles).

Frontend Engine: Three.js (WebGL).

Constraint: Must support GPU acceleration on mobile (Safari/Chrome).

Key Class: THREE.InstancedMesh (Crucial for rendering 100k+ hexes).

Satellite Source: Tiris (Tirol GIS) gdi_winter WMTS layer.

Source Link: https://tiris.maps.arcgis.com/apps/webappviewer/index.html?id=5849fe1df5994dc8a3c1e4675682d2fd

Note: Do NOT scrape Reality Maps or Strava. Use the official government winter feed.

2. Data Pipeline & Math
The "Hex" Decision
We are using Hexagons instead of Squares.

Why: Hexes have 6 equidistant neighbors (vs. 4 for squares). This provides 3 axes for gradient calculation, resulting in smoother, more accurate slope derivation for skiing terrain.

Layout: "Odd-q" vertical layout (columns are aligned, rows are staggered).

### The "Magic Alignment" (Flat-Topped 30° Stagger)
Based on visual testing, the most efficient layout for aligning hexes with a 5m source DEM (interpolated to 2.5m) is **Flat-Topped** with a horizontal stagger.

**Geometry Values (for 5m Hex centers):**
- **Orientation**: Flat-Topped (Edges at N/S, Vertices at E/W).
- **Horizontal Gap (ΔX)**: $4.33\text{m}$ ($width * \cos(30^\circ)$).
- **Vertical Stagger (ΔY)**: $2.5\text{m}$ (Exactly 50% of the center-to-center distance).
- **Vertical Spacing (Same Column)**: $5.0\text{m}$ (Matches DEM resolution).

**Why this is the "Truth":**
On a **2.5m uniform interpolation grid**, every hex center lands on a perfect pixel center:
- **P-Column**: Centers at $Y = 0, 5, 10, 15...$ (Even Pixels).
- **N-Column**: Centers at $Y = 2.5, 7.5, 12.5...$ (Odd Pixels).
- **Horizontal**: Centers at $X = 0, 4.33, 8.66...$

This eliminates sub-pixel sampling drift and ensures the "Pistons" align perfectly with the tiled scanlines of the WebP assets.

Tiling Strategy
Format: Pre-generated static files (JSON or Binary) stored on S3.

Structure: Standard XYZ or Quadtree tiling scheme matching the map tiles.

Content: Each tile file contains an array of Piston data: [x, y, z, r, g, b, slope, aspect].

3. Visualization Logic
Hybrid Rendering Modes
The visual style changes based on the terrain angle to optimize for aesthetics and information density.

The "Pathetic Attempt" (Low Angle):

Condition: Slope < 25° (configurable).

Visual: Render as flat/transparent or paint the satellite texture onto the sides of short hexes.

Goal: Keep the "base" of the mountain looking like the satellite photo; don't distract with 3D noise on flat valley floors.

The "Pistons" (High Angle):

Condition: Slope > 25°.

Visual: 3D Hexagonal columns extruding vertically.

Texture: Top cap = Satellite color. Sides = Gradient/Slope color.

The "Windfucked" Shader (Snow Quality)
Custom coloring logic to visualize potential powder vs. wind-scoured ice.

Input: Aspect (Compass direction) + Slope.

Logic:

Powder (Good): North/North-East aspects (Protected). Mix in Neon Pink.

Windfucked (Bad): North-West aspects (Windward). Mix in Cyan/White.

Melt-Freeze: South aspects.

4. Frontend & Interaction (Three.js)
Camera & Controls
Controller: MapControls (OrbitControls variation for panning on XZ plane).

Input Support:

1 Finger: Pan.

2 Fingers: Pinch to Zoom, Rotate, and Tilt.

Constraints:

maxPolarAngle: Set to ~85°. Prevents the user from dipping the camera below the horizon line.

Collision System (The "Glider")
To prevent the camera from clipping through mountains when zooming:

Logic: Raycast vertically down from the camera position to the terrain mesh.

Action: If CameraHeight < TerrainHeight + Buffer, force CameraHeight up. Smoothly glide over peaks.

The "Invisible Transition" (2D/3D Snap)
Eliminate visual artifacts (fringing) when looking top-down.

Layer 1: Flat Plane with high-res Satellite WebP tiles (Always rendered).

Layer 2: The 3D Piston Mesh.

Transition Logic: Monitor Camera Angle (Polar).

0° - 45° (Side View): Pistons fully visible.

45° - 75° (Transition): Smooth opacity fade of Pistons. I think this might end up being done by calculating the avg color of the top cap and fading that color to be the side color porgressively for more and more steep parts. 

75° - 90° (Top Down): Pistons.visible = false. User sees only the high-quality 2D satellite map. maybe Keep original sattelite imagery just rendered undeneath the piston mesh so we can just SNAP and the pistons are gone. 

5. Current Implementation Tasks
Immediate To-Do (Python/Data)
DEM Interpolation: Upscale current DEMs to 2.5m resolution using Bilinear interpolation.

Hex Gen Script: Write the script to sample the 2.5m DEM and gdi_winter imagery to generate the JSON/Bin chunks for the InstancedMesh.

Immediate To-Do (Web/JS)
Tiling Upgrade: Ensure the 2D web map uses tiled WebP images (replacing large single PNGs).

Three.js Boilerplate: Set up the scene with MapControls and the "Invisible Transition" logic.

Tile Alignment: Verify that the 3D hex grid aligns perfectly (pixel-perfect) with the 2D satellite underlay.

Refer to piston_tool.py for prototyping logic but implementation is hexes instead of squares in three.js. 