import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { TIFFLoader } from 'three/addons/loaders/TIFFLoader.js';
import { LODController } from './lod_controller.js';

// --- HEX COORDINATE SYSTEM (from coordinate_utility.py) ---
const UNIT_HEX_PX = 32.0;
const METERS_PER_PIXEL = 0.2;
const UNIT_HEX_WIDTH_METERS = UNIT_HEX_PX * METERS_PER_PIXEL; // 6.4m
const LEVEL_5_SCALE_FACTOR = Math.pow(7, 2.5); // ~129.6418
const SECTOR_WIDTH_METERS = UNIT_HEX_WIDTH_METERS * LEVEL_5_SCALE_FACTOR; // ~829.7m

// --- CONFIG ---
// Used for culling and camera centering.
// Hexagons are 829m wide. Using that for bounds.
const TILE_WIDTH_WORLD = SECTOR_WIDTH_METERS;
const TILE_HEIGHT_WORLD = SECTOR_WIDTH_METERS; // Approximate square for culling
const SCALE_Z = 1.0;

// Convert world meters (x, y) to approximate sector (Q, R)
function worldToSectorApprox(worldX, worldY) {
    const h = UNIT_HEX_WIDTH_METERS;
    const w_inv = 1.0 / (Math.sqrt(3) / 2 * h);

    const q_approx = worldX * w_inv;
    const r_approx = (worldY - q_approx * 0.5 * h) / h;

    // Inverse Matrix 7^5 (Det = 16807)
    const det = 16807;
    const Q = Math.round((62 * q_approx + 149 * r_approx) / det);
    const R = Math.round((-149 * q_approx - 87 * r_approx) / det);

    return { Q, R };
}
const DEFAULT_RENDER_DISTANCE = 10000;
const FLOOR_MODE = 'view-min';
const RESOLUTIONS = [
    1, 2, 3, 4, 5, 6, 7, 7.7136, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    30, 35, 40, 45, 50
];
// Options: view-min, view-avg, camera-tile-min, camera-tile-avg, camera-tile-base, global-min, global-avg, global-base
const LOCK_FLOOR_ON_RISE = true;
const FLOOR_LOCK_THRESHOLD = 0.02;
const TILE_BOUNDS_MIN_Y = -10000;
const TILE_BOUNDS_MAX_Y = 10000;
const DEM_FLIP_NS = false;
const BORDER_WALLS_ALWAYS = true;
const DEBUG_FIXED_WALLS = false;
const DEBUG_FIXED_WALL_DEPTH = 10.0;
const DEBUG_VIOLET_SOUTH = false;
const DEBUG_NEIGHBOR_SLOT_TEST = false;
const DEBUG_SLOT_INDEX = 1;
const DEBUG_SLOT_WALL_DEPTH = 10.0;
const DEBUG_OTHER_WALL_DEPTH = 2.0;
const LIGHTING_DEFAULTS = {
    aoFloor: 0.0,
    aoPower: 1.0,
    lambert: 0.0,
    rim: 0.0,
    rimPower: 2.2,
    spec: 0.0,
    specPower: 30.0,
    slopeLight: 0.0,
};

class PistonViewer {
    constructor() {
        console.log("Initializing PistonViewer (Multi-Tile)...");
        this.container = document.getElementById('canvas-container');
        this.scene = new THREE.Scene();
        // this.scene.background = new THREE.Color(0xff2aa1); // debug pink
        this.scene.background = new THREE.Color(0x000000);

        // Initial Camera Setup
        this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 10, 50000);
        this.camera.position.set(0, 5000, 0);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        this.controls = new MapControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 100;
        this.controls.maxDistance = 50000;
        this.controls.maxPolarAngle = Math.PI / 2.1;

        const ambient = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambient);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(500, 2000, 500);
        this.scene.add(dirLight);

        // State
        this.currentRes = 7.7136; // Level 0 Unit
        this.currentResMethod = 'bilinear';
        this.hexWidth = this.currentRes;
        this.hexDx = this.hexWidth * (Math.sqrt(3) / 2);

        this.lodThresholds = [200, 500, 1000, 2000];

        // Resize Handler
        window.addEventListener('resize', this.onResize.bind(this));

        // Shared Geometry
        const side = this.hexWidth / Math.sqrt(3);
        this.hexGeometry = this.createHexGeometry(side);
        // NOTE: megahex optimization - this needs updated when we switch to hextiles
        // to use large pre-baked hex quads instead of a rectangular plane for minimal performance loss.
        this.flatGeometry = new THREE.PlaneGeometry(TILE_WIDTH_WORLD, TILE_HEIGHT_WORLD);
        this.flatGeometry.rotateX(-Math.PI / 2); // Lay flat on XZ plane
        this.flatGeometry.translate(TILE_WIDTH_WORLD / 2, 0, -TILE_HEIGHT_WORLD / 2); // Align with hex mesh origin (top-left)

        // State
        this.tiles = [];
        this.manifest = null;
        this.loadingTiles = new Set(); // Track tiles currently in flight
        this.essentialTilesLoaded = 0;
        this.essentialTilesTarget = 0;
        this.loaderHidden = false;
        this.materialsToUpdate = [];
        this.cssMapActive = false; // Unified WebGL Pipeline
        this.cssWorld = null;
        this.webglActive = true;
        this.heightFactor = 0.0;
        this.transSettings = {
            flatThresh: 5.0,
            riseStart: 6.0,
            riseEnd: 25.0,
            curve: 1.0
        };
        this.worldOrigin = { x: 0, y: 0 };
        this.floorMode = FLOOR_MODE;
        this.floorState = { locked: false, value: 0.0, lastFactor: 0.0 };
        this.globalStats = { min: Infinity, max: -Infinity, avgSum: 0.0, baseSum: 0.0, count: 0 };
        this.frustum = new THREE.Frustum();
        this.projScreenMatrix = new THREE.Matrix4();
        this.renderSettings = {
            renderDistance: DEFAULT_RENDER_DISTANCE,
        };
        this.fpsState = { lastSample: performance.now(), frames: 0 };
        this.fpsEl = document.getElementById('fps-counter');
        this.tilesLoadedCount = 0;
        this.lodController = new LODController();
        this.lodStats = {
            accumulator: 0,
            count: 0,
            lastUpdate: 0,
            accLoop: 0,
            accTree: 0,
            accHash: 0
        };

        // === DEBUG: Render Analysis ===
        this.hexCountEl = document.getElementById('hex-count');
        this.triCountEl = document.getElementById('tri-count');
        this.drawStatsEl = document.getElementById('draw-stats');
        this.tileHeightEl = document.getElementById('tile-height');
        this.cameraHeightEl = document.getElementById('camera-height');
        this.frametimeCanvas = document.getElementById('frametime-graph');
        this.frametimeCtx = this.frametimeCanvas ? this.frametimeCanvas.getContext('2d') : null;
        this.frametimeBuffer = new Array(120).fill(16.67); // 120 frames of history
        this.frametimeIndex = 0;
        this.lastFrameTime = performance.now();
        this.displayRefreshRate = 60; // Will be detected
        this.statsUpdateState = { lastUpdate: 0, interval: 500 }; // Update stats every 500ms
        this.lastTriCount = { theoretical: 0, actual: 0 };
        // === END DEBUG ===

        // Start
        this.initDebugConsole();
        this.initMinimizeButton();
        this.initLODSliders();
        this.detectRefreshRate();
        this.updateFogAndClip();
        this.initWorld();
        this.animate();
    }

    initDebugConsole() {
        this.consoleEl = document.getElementById('console-output');
        const btn = document.getElementById('copy-log-btn');
        if (btn) {
            btn.addEventListener('click', () => {
                if (this.consoleEl) {
                    const text = this.consoleEl.innerText;
                    navigator.clipboard.writeText(text).then(() => {
                        this.log("Log copied to clipboard", "success");
                    });
                }
            });
        }
        this.log("System Ready. PistonViewer Initialized.", "success");
        this.log(`Config: Render Dist: ${this.renderSettings.renderDistance}, Floor Lock: ${LOCK_FLOOR_ON_RISE}`);
    }

    log(msg, type = "info") {
        if (!this.consoleEl) return;
        const line = document.createElement('div');
        line.className = `log-line ${type}`;

        const now = new Date();
        const time = `${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${now.getMilliseconds().toString().padStart(3, '0')}`;

        const span = document.createElement('span');
        span.className = 'log-time';
        span.textContent = `[${time}]`;

        line.appendChild(span);
        line.appendChild(document.createTextNode(msg));

        this.consoleEl.appendChild(line);
        this.consoleEl.scrollTop = this.consoleEl.scrollHeight;
    }

    // === DEBUG: Minimize button functionality ===
    initMinimizeButton() {
        const btn = document.getElementById('minimize-btn');
        const panel = document.getElementById('main-panel');
        if (btn && panel) {
            btn.addEventListener('click', () => {
                panel.classList.toggle('minimized');
                btn.textContent = panel.classList.contains('minimized') ? '+' : '−';
            });
        }
    }

    clearAllTiles() {
        for (const tile of this.tiles) {
            if (tile.mesh) {
                this.scene.remove(tile.mesh);
                if (tile.mesh.geometry) tile.mesh.geometry.dispose();
                if (tile.mesh.material) {
                    if (tile.mesh.material.map) tile.mesh.material.map.dispose();
                    tile.mesh.material.dispose();
                }
            }
        }
        this.tiles = [];
        this.materialsToUpdate = [];
        this.loadingTiles.clear();
        this.tilesLoadedCount = 0;
        this.essentialTilesLoaded = 0;
    }

    initLODSliders() {
        for (let i = 0; i < 4; i++) {
            const slider = document.getElementById(`lod${i}-slider`);
            const label = document.getElementById(`lod${i}-val`);
            if (slider && label) {
                slider.addEventListener('input', () => {
                    const val = slider.value;
                    label.textContent = val >= 1000 ? `${(val / 1000).toFixed(1)}km` : `${val}m`;
                });
                slider.addEventListener('change', () => {
                    this.lodThresholds[i] = parseInt(slider.value);
                    this.log(`LOD Level ${i} threshold updated to ${this.lodThresholds[i]}m.`, "info");

                    if (this.lodController) {
                        this.lodController.setThresholds({
                            lod0: this.lodThresholds[0],
                            lod1: this.lodThresholds[1],
                            lod2: this.lodThresholds[2],
                            lod3: this.lodThresholds[3]
                        });
                    }
                });
            }
        }
    }

    // === END DEBUG ===

    // === DEBUG: Detect display refresh rate ===
    detectRefreshRate() {
        let lastTime = performance.now();
        let frameCount = 0;
        const samples = [];

        const measure = (currentTime) => {
            const delta = currentTime - lastTime;
            if (delta > 0) {
                samples.push(1000 / delta);
            }
            lastTime = currentTime;
            frameCount++;

            if (frameCount < 60) {
                requestAnimationFrame(measure);
            } else {
                // Calculate median FPS from samples
                samples.sort((a, b) => a - b);
                const median = samples[Math.floor(samples.length / 2)];

                // Round to common refresh rates: 60, 75, 90, 120, 144, 165, 240
                const commonRates = [60, 75, 90, 120, 144, 165, 240];
                this.displayRefreshRate = commonRates.reduce((prev, curr) =>
                    Math.abs(curr - median) < Math.abs(prev - median) ? curr : prev
                );

                this.log(`Display refresh rate detected: ${this.displayRefreshRate}Hz`, "info");
            }
        };

        requestAnimationFrame(measure);
    }
    // === END DEBUG ===

    // === DEBUG: Render statistics update ===
    updateRenderStats(now) {
        // Throttle updates to avoid DOM thrashing
        if (now - this.statsUpdateState.lastUpdate < this.statsUpdateState.interval) {
            return;
        }
        this.statsUpdateState.lastUpdate = now;

        // Count visible hexagons
        let visibleHexCount = 0;
        for (const tile of this.tiles) {
            if (tile.mesh && tile.mesh.visible) {
                visibleHexCount += tile.mesh.count;
            }
        }

        // Get GPU stats
        const info = this.renderer.info.render;
        const actualTriangles = info.triangles;
        const drawCalls = info.calls;
        const geometries = this.renderer.info.memory.geometries;
        const textures = this.renderer.info.memory.textures;

        // Calculate theoretical triangles (12 per hex)
        const theoreticalTriangles = visibleHexCount * 12;
        const efficiency = theoreticalTriangles > 0
            ? Math.round((theoreticalTriangles / actualTriangles) * 100)
            : 100;

        // Update DOM
        if (this.hexCountEl) {
            this.hexCountEl.textContent = visibleHexCount.toLocaleString() + ' VISIBLE';
        }

        if (this.triCountEl) {
            const theoStr = (theoreticalTriangles / 1000000).toFixed(2) + 'M';
            const actualStr = (actualTriangles / 1000000).toFixed(2) + 'M';
            this.triCountEl.textContent = `${theoStr} / ${actualStr} (${efficiency}%)`;
        }

        if (this.drawStatsEl) {
            this.drawStatsEl.textContent = `${drawCalls} | G:${geometries} | T:${textures}`;
        }

        // Log significant changes
        const triDiff = Math.abs(actualTriangles - this.lastTriCount.actual);
        if (triDiff > actualTriangles * 0.1) { // >10% change
            this.log(`Render: ${visibleHexCount.toLocaleString()} hex | Efficiency: ${efficiency}% | Draws: ${drawCalls}`, "info");
            this.lastTriCount = { theoretical: theoreticalTriangles, actual: actualTriangles };
        }
    }
    // === END DEBUG ===

    // === DEBUG: Frametime graph rendering ===
    drawFrametimeGraph() {
        if (!this.frametimeCtx) return;

        const ctx = this.frametimeCtx;
        const width = this.frametimeCanvas.width;
        const height = this.frametimeCanvas.height;

        // Clear canvas
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(0, 0, width, height);

        // Calculate current FPS (average of recent frames)
        const recentFrames = 30;
        let sum = 0;
        for (let i = 0; i < recentFrames; i++) {
            const idx = (this.frametimeIndex - 1 - i + 120) % 120;
            sum += this.frametimeBuffer[idx];
        }
        const avgFrameTime = sum / recentFrames;
        const currentFps = Math.round(1000 / avgFrameTime);

        // Draw FRAMETIME label
        ctx.font = '9px monospace';
        ctx.fillStyle = '#666';
        ctx.textAlign = 'left';
        ctx.fillText('FRAMETIME', 6, 11);

        // Draw FPS counter (current / display cap)
        ctx.font = 'bold 11px monospace';
        ctx.fillStyle = '#74b9ff';
        ctx.textAlign = 'right';
        ctx.fillText(`${currentFps}/${this.displayRefreshRate}fps`, width - 6, 11);

        // Draw target line (16.67ms for 60fps)
        const targetMs = 16.67;
        const targetY = height - (targetMs / 33.33) * height; // Scale to 33.33ms max
        ctx.strokeStyle = 'rgba(46, 204, 113, 0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, targetY);
        ctx.lineTo(width, targetY);
        ctx.stroke();

        // Draw frametime bars (oldest on left, newest on right)
        const barWidth = width / 120;
        for (let i = 0; i < 120; i++) {
            const idx = (this.frametimeIndex + i) % 120;
            const frameTime = this.frametimeBuffer[idx];
            const barHeight = Math.min((frameTime / 33.33) * height, height);
            const x = i * barWidth;
            const y = height - barHeight;

            // Color code: green if under 16.67ms, yellow if under 20ms, red if over
            if (frameTime < 16.67) {
                ctx.fillStyle = 'rgba(116, 185, 255, 0.8)';
            } else if (frameTime < 20) {
                ctx.fillStyle = 'rgba(241, 196, 15, 0.8)';
            } else {
                ctx.fillStyle = 'rgba(231, 76, 60, 0.8)';
            }

            ctx.fillRect(x, y, barWidth, barHeight);
        }
    }
    // === END DEBUG ===

    updateFloorUniforms() {
        for (const mat of this.materialsToUpdate) {
            const shader = mat.userData.shader;
            if (!shader) continue;
            shader.uniforms.uFloorOffset.value = this.floorState.value;
        }
    }

    updateGlobalStats(stats) {
        if (!stats) return;
        if (Number.isFinite(stats.min)) {
            this.globalStats.min = Math.min(this.globalStats.min, stats.min);
        }
        if (Number.isFinite(stats.max)) {
            this.globalStats.max = Math.max(this.globalStats.max, stats.max || -Infinity);
        }
        if (Number.isFinite(stats.avg)) {
            this.globalStats.avgSum += stats.avg;
        }
        if (Number.isFinite(stats.base)) {
            this.globalStats.baseSum += stats.base;
        }
        this.globalStats.count += 1;
    }
    getTilesInView() {
        this.camera.updateMatrixWorld();
        this.projScreenMatrix.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
        this.frustum.setFromProjectionMatrix(this.projScreenMatrix);

        const target = this.controls.target;
        const renderDistance = this.renderSettings.renderDistance;
        const renderDistanceSq = renderDistance * renderDistance;
        const inView = [];

        for (const tile of this.tiles) {
            if (!tile.stats || !tile.bounds) continue;
            const dx = target.x - tile.center.x;
            const dz = target.z - tile.center.z;
            if ((dx * dx + dz * dz) > renderDistanceSq) continue;
            if (this.frustum.intersectsBox(tile.bounds)) {
                inView.push(tile);
            }
        }
        return inView;
    }

    getTileUnderTarget() {
        const x = this.controls.target.x;
        const z = this.controls.target.z;
        for (const tile of this.tiles) {
            if (!tile.bounds) continue;
            if (x >= tile.bounds.min.x && x <= tile.bounds.max.x &&
                z >= tile.bounds.min.z && z <= tile.bounds.max.z) {
                return tile;
            }
        }
        return null;
    }

    reduceMin(tiles, getValue) {
        let min = Infinity;
        for (const tile of tiles) {
            const value = getValue(tile);
            if (Number.isFinite(value)) min = Math.min(min, value);
        }
        return Number.isFinite(min) ? min : null;
    }

    reduceAvg(tiles, getValue) {
        let sum = 0.0;
        let count = 0;
        for (const tile of tiles) {
            const value = getValue(tile);
            if (Number.isFinite(value)) {
                sum += value;
                count += 1;
            }
        }
        return count ? (sum / count) : null;
    }

    computeFloorCandidates() {
        const inView = this.getTilesInView();
        const tiles = inView.length ? inView : this.tiles.filter((tile) => tile.stats);
        const viewMin = this.reduceMin(tiles, (tile) => tile.stats.min);
        const viewAvg = this.reduceAvg(tiles, (tile) => tile.stats.avg);
        const cameraTile = this.getTileUnderTarget();
        const cameraMin = cameraTile?.stats?.min ?? null;
        const cameraAvg = cameraTile?.stats?.avg ?? null;
        const cameraBase = cameraTile?.stats?.base ?? null;
        const globalMin = this.globalStats.count ? this.globalStats.min : viewMin;
        const globalAvg = this.globalStats.count ? (this.globalStats.avgSum / this.globalStats.count) : viewAvg;
        const globalBase = this.globalStats.count ? (this.globalStats.baseSum / this.globalStats.count) : cameraBase;

        return {
            viewMin,
            viewAvg,
            cameraMin,
            cameraAvg,
            cameraBase,
            globalMin,
            globalAvg,
            globalBase
        };
    }

    pickFloorValue() {
        const candidates = this.computeFloorCandidates();
        let value = null;

        switch (this.floorMode) {
            case 'view-avg':
                value = candidates.viewAvg;
                break;
            case 'camera-tile-min':
                value = candidates.cameraMin;
                break;
            case 'camera-tile-avg':
                value = candidates.cameraAvg;
                break;
            case 'camera-tile-base':
                value = candidates.cameraBase;
                break;
            case 'global-min':
                value = candidates.globalMin;
                break;
            case 'global-avg':
                value = candidates.globalAvg;
                break;
            case 'global-base':
                value = candidates.globalBase;
                break;
            case 'view-min':
            default:
                value = candidates.viewMin;
                break;
        }

        if (!Number.isFinite(value)) {
            value = candidates.viewMin ?? candidates.globalMin ?? 0.0;
        }
        return Number.isFinite(value) ? value : 0.0;
    }

    updateFloorState(heightFactor) {
        const threshold = FLOOR_LOCK_THRESHOLD;
        if (heightFactor <= threshold) {
            this.floorState.locked = false;
            return;
        }

        if (LOCK_FLOOR_ON_RISE) {
            if (!this.floorState.locked && this.floorState.lastFactor <= threshold) {
                this.floorState.value = this.pickFloorValue();
                this.floorState.locked = true;
                this.updateFloorUniforms();
            }
        } else {
            const nextValue = this.pickFloorValue();
            if (Number.isFinite(nextValue) && nextValue !== this.floorState.value) {
                this.floorState.value = nextValue;
                this.updateFloorUniforms();
            }
        }
    }

    maintainCameraAltitudeDuringAnimation(heightFactor) {
        const camX = this.camera.position.x;
        const camZ = this.camera.position.z;
        const BUFFER = 25.0;

        // Convert camera scene position back to world meters
        // Scene: posX = x - worldOrigin.x, posZ = -(y - worldOrigin.y)
        // Inverse: x = posX + worldOrigin.x, y = worldOrigin.y - posZ
        const worldX = camX + this.worldOrigin.x;
        const worldY = this.worldOrigin.y - camZ;

        // Convert world meters to sector Q, R
        const { Q, R } = worldToSectorApprox(worldX, worldY);

        // Find tile by Q, R match
        let foundTile = null;
        for (const tile of this.tiles) {
            if (tile.q === Q && tile.r === R) {
                foundTile = tile;
                break;
            }
        }

        // Update camera height display regardless
        if (this.cameraHeightEl) {
            this.cameraHeightEl.textContent = this.camera.position.y.toFixed(1) + 'm';
        }

        if (!foundTile) {
            if (this.tileHeightEl) {
                this.tileHeightEl.textContent = `NO TILE (Q${Q},R${R})`;
            }
            return;
        }

        // Check if tile has valid stats
        if (!foundTile.stats || !Number.isFinite(foundTile.stats.max)) {
            if (this.tileHeightEl) {
                this.tileHeightEl.textContent = 'NO STATS';
            }
            return;
        }

        const baseElev = foundTile.stats.max - this.floorState.value;
        const animatedElev = baseElev * heightFactor;
        const minCamY = Math.max(100.0, animatedElev + BUFFER);

        // DEBUG: Update tile height display
        if (this.tileHeightEl) {
            this.tileHeightEl.textContent = `${animatedElev.toFixed(1)}m (Q${Q},R${R})`;
        }

        // Only push camera up if below minimum - don't override zoom controls
        if (this.camera.position.y < minCamY) {
            this.camera.position.y = minCamY;
        }
    }

    updateFogAndClip() {
        const dist = this.renderSettings.renderDistance;
        const fogStart = Math.max(10, dist * 0.6);
        const fogEnd = Math.max(fogStart + 10, dist);

        if (!this.scene.fog) {
            this.scene.fog = new THREE.Fog(this.scene.background, fogStart, fogEnd);
        }
        this.scene.fog.color.copy(this.scene.background);
        this.scene.fog.near = fogStart;
        this.scene.fog.far = fogEnd;

        this.camera.far = Math.max(fogEnd + 500, 2000);
        this.camera.updateProjectionMatrix();
    }

    async initWorld() {
        try {
            const res = await fetch(`tile_manifest.json?v=${Date.now()}`);
            if (!res.ok) throw new Error('Failed to load tile_manifest.json');
            this.manifest = await res.json();

            console.log(`Manifest loaded. Found ${this.manifest.tiles.length} tiles.`);
            this.log(`Manifest loaded. Tiles: ${this.manifest.tiles.length} | Bounds: [${this.manifest.bounds.min_x},${this.manifest.bounds.min_y}] to [${this.manifest.bounds.max_x},${this.manifest.bounds.max_y}]`);

            // Calculate Global Bounds
            const { min_x, min_y, max_x, max_y } = this.manifest.bounds;
            this.worldOrigin = { x: min_x, y: min_y };

            // Center Camera on the map
            const mapWidth = (max_x - min_x) + TILE_WIDTH_WORLD;
            const mapHeight = (max_y - min_y) + TILE_HEIGHT_WORLD;

            const centerX = mapWidth / 2;
            const centerZ = -mapHeight / 2;

            // Start zoomed out (approx x2 from previous 800)
            this.camera.position.set(centerX, 1600, centerZ);
            this.controls.target.set(centerX, 0, centerZ);
            this.controls.update();

            // Determine "Essential" tiles (those in the initial viewport)
            const initialInView = this.manifest.tiles.filter(tileDef => {
                const tx = tileDef.x - this.worldOrigin.x + (TILE_WIDTH_WORLD / 2);
                const tz = -(tileDef.y - this.worldOrigin.y) - (TILE_HEIGHT_WORLD / 2);
                const distSq = (centerX - tx) ** 2 + (centerZ - tz) ** 2;
                return distSq < (this.renderSettings.renderDistance ** 2);
            });

            this.essentialTilesTarget = Math.max(1, initialInView.length);
            this.log(`Proximity Engine: ${this.essentialTilesTarget} tiles required for first paint.`, "info");

            // Initialize LOD Science
            if (this.lodController) {
                // Ensure tiles have q, r (assuming manifest has them or we calculate them? 
                // Currently manifest has x, y (World Meters). 
                // LODController calculates its own q,r from x,y, so raw x,y is fine for QuadTree insertion.
                this.lodController.init(this.manifest.bounds, this.manifest.tiles);
            }

            // CSS Map is dead. Unified pipeline lives.

            // Only trigger updateLOD to start loading what's in range
            this.updateLOD();

        } catch (err) {
            console.error("Error initializing world:", err);
            this.log(`Error initializing world: ${err.message}`, "error");
        }
    }

    async loadSingleTile(tileDef) {
        const { x, y, q, r } = tileDef;
        // Use Q_R as the unique key if available, else fallback provided manifest is old
        const tileKey = (q !== undefined && r !== undefined) ? `${q}_${r}` : `${x}_${y}`;

        if (this.loadingTiles.has(tileKey)) return;
        this.loadingTiles.add(tileKey);

        const posX = x - this.worldOrigin.x;
        const posZ = -(y - this.worldOrigin.y);

        const t0 = performance.now();

        // New Sector-based paths
        // If q,r missing, this will fail 404, which is expected if manifest isn't updated.
        const cb = Date.now();
        const binUrl = (q !== undefined) ? `tiles_bin/sector_${q}_${r}.bin?v=${cb}` : `tiles_bin/tile_${x}_${y}.bin?v=${cb}`;
        const lowTexUrl = (q !== undefined) ? `aerial_tiles/low_res/sector_${q}_${r}.webp?v=${cb}` : `tiles_sat/low_res/tile_${x}_${y}.webp?v=${cb}`;
        const highTexUrl = (q !== undefined) ? `aerial_tiles/high_res/sector_${q}_${r}.webp?v=${cb}` : `tiles_sat/high_res/tile_${x}_${y}.webp?v=${cb}`;

        // 1. Load Low Res Texture
        const texLoader = new THREE.TextureLoader();
        const texture = await texLoader.loadAsync(lowTexUrl);
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.flipY = false;

        // 2. Create Material
        const material = new THREE.MeshBasicMaterial({
            map: texture,
            side: THREE.DoubleSide,
            fog: true,
            alphaTest: 0.1
        });
        this.setupMaterialShader(material);
        this.materialsToUpdate.push(material);

        // 3. Fetch Binary Data
        const response = await fetch(binUrl);
        const buffer = await response.arrayBuffer();
        const parsed = this.parseBinary(buffer);
        const hexData = parsed.hexes;

        // 4. Create Flat Mesh for Unified Pipeline
        const flatMesh = new THREE.Mesh(this.flatGeometry, material);
        flatMesh.position.set(posX, 0, posZ);
        this.scene.add(flatMesh);
        const stats = parsed.stats;

        // 4. Create Instanced Mesh (with Cloned Geometry!)
        material.userData.uTileResolution = parsed.ySpacing;
        const mesh = this.createInstancedMesh(hexData, material, parsed.ySpacing);

        // 5. Position Mesh
        mesh.position.set(posX, 0, posZ);
        mesh.updateMatrixWorld();

        this.scene.add(mesh);

        const centerX = posX + (TILE_WIDTH_WORLD / 2);
        const centerZ = posZ - (TILE_HEIGHT_WORLD / 2);
        const bounds = new THREE.Box3(
            new THREE.Vector3(posX, TILE_BOUNDS_MIN_Y, posZ - TILE_HEIGHT_WORLD),
            new THREE.Vector3(posX + TILE_WIDTH_WORLD, TILE_BOUNDS_MAX_Y, posZ)
        );

        const tileObj = {
            x, y, q, r,
            mesh,
            flatMesh,
            material,
            stats,
            center: { x: centerX, z: centerZ },
            bounds,
            urls: { low: lowTexUrl, high: highTexUrl },
            highResLoaded: false,
            highResLoading: false
        };
        this.tiles.push(tileObj);
        this.updateGlobalStats(stats);
        const t1 = performance.now();
        this.tilesLoadedCount++;
        this.loadingTiles.delete(tileKey);

        if (this.tilesLoadedCount <= 5) {
            this.log(`[Ready] Tile ${x},${y} | ${(t1 - t0).toFixed(1)}ms | Meshes: ${hexData.length}`, "success");
        }

        // Handle dynamic loading screen
        if (!this.loaderHidden) {
            // Check if this tile was part of the initial "essential" set
            this.essentialTilesLoaded++;
            if (this.essentialTilesLoaded >= this.essentialTilesTarget) {
                this.hideLoader();
            }
        }

        if (this.tilesLoadedCount === this.manifest.tiles.length) {
            this.log(`System: All ${this.tilesLoadedCount} tiles rendered.`, "success");
            const geoms = this.renderer.info.memory.geometries;
            const textures = this.renderer.info.memory.textures;
            this.log(`Final State: Geoms: ${geoms} | Tex: ${textures} | Manifest: ${this.manifest.tiles.length} tiles`, "info");
        }
    }

    hideLoader() {
        if (this.loaderHidden) return;
        this.loaderHidden = true;
        const loader = document.getElementById('loader');
        if (loader) {
            this.log("First paint ready. Fading loader.", "success");
            loader.style.transition = 'opacity 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
            loader.style.opacity = '0';
            setTimeout(() => { loader.style.display = 'none'; }, 600);
        }
    }

    parseBinary(buffer) {
        const view = new DataView(buffer);

        // 1. Detect Version
        // Check for 'HEX2' Magic (0x32584548)
        // Little endian read of first 4 bytes
        const magic = view.getUint32(0, true);
        if (magic === 0x32584548) {
            return this.parseBinaryV2(view);
        }

        // --- Legacy Fallback ---
        let baseElevation = view.getFloat32(0, true);
        let minElevation = baseElevation;
        let avgElevation = baseElevation;
        let ySpacing = this.currentRes; // Fallback
        let headerSize = 4;
        let payloadStride = 14;

        if ((buffer.byteLength - 16) % 26 === 0) {
            baseElevation = view.getFloat32(0, true);
            minElevation = view.getFloat32(4, true);
            avgElevation = view.getFloat32(8, true);
            ySpacing = view.getFloat32(12, true);
            headerSize = 16;
            payloadStride = 26;
        } else if ((buffer.byteLength - 16) % 14 === 0) {
            baseElevation = view.getFloat32(0, true);
            minElevation = view.getFloat32(4, true);
            avgElevation = view.getFloat32(8, true);
            ySpacing = view.getFloat32(12, true);
            headerSize = 16;
            payloadStride = 14;
        } else if ((buffer.byteLength - 12) % 14 === 0) {
            baseElevation = view.getFloat32(0, true);
            minElevation = view.getFloat32(4, true);
            avgElevation = view.getFloat32(8, true);
            headerSize = 12;
            payloadStride = 14;
        } else if ((buffer.byteLength - 4) % 14 === 0) {
            headerSize = 4;
            payloadStride = 14;
        }

        const hexData = [];
        let offset = headerSize;
        let minAbs = Infinity;
        let maxAbs = -Infinity;
        let sumAbs = 0.0;
        let count = 0;

        while (offset < buffer.byteLength) {
            const z = this.decodeFloat16(view.getUint16(offset, true));
            const neighbors = [];
            for (let i = 0; i < 6; i++) {
                neighbors.push(this.decodeFloat16(view.getUint16(offset + 2 + i * 2, true)) + baseElevation);
            }

            let slopes = [0, 0, 0, 0, 0, 0];
            if (payloadStride === 26) {
                for (let i = 0; i < 6; i++) {
                    slopes[i] = this.decodeFloat16(view.getUint16(offset + 14 + i * 2, true));
                }
            }

            const zAbs = z + baseElevation;
            hexData.push({
                z: zAbs,
                n_n: neighbors[0],
                n_ne: neighbors[1],
                n_se: neighbors[2],
                n_s: neighbors[3],
                n_sw: neighbors[4],
                n_nw: neighbors[5],
                s: slopes
            });

            if (headerSize === 4) {
                minAbs = Math.min(minAbs, zAbs);
                maxAbs = Math.max(maxAbs, zAbs);
                sumAbs += zAbs;
                count += 1;
            }
            offset += payloadStride;
        }

        if (headerSize === 4) {
            if (!Number.isFinite(minAbs)) minAbs = baseElevation;
            if (!Number.isFinite(maxAbs)) maxAbs = baseElevation;
            const avgAbs = count ? (sumAbs / count) : baseElevation;
            return {
                hexes: hexData,
                stats: { base: baseElevation, min: minAbs, max: maxAbs, avg: avgAbs },
                ySpacing
            };
        }
        return {
            hexes: hexData,
            stats: { base: baseElevation, min: minElevation, max: maxAbs, avg: avgElevation },
            ySpacing
        };
    }

    parseBinaryV2(view) {
        // HEADER (32 Bytes)
        // Magic (4) Already checked
        const Q = view.getInt32(4, true);
        const R = view.getInt32(8, true);
        const minZ = view.getFloat32(12, true);
        const maxZ = view.getFloat32(16, true);
        const scale = view.getFloat32(20, true);
        // Padding 8 bytes at 24..32

        let offset = 32;

        // Blocks: 
        // 0: Mega (1)
        // 1: Huge (7)
        // 2: Great (49)
        // 3: Grand (343)
        // 4: Parent (2401)
        // 5: Unit (16807)

        // For now, hardcode reading Level 5 (Unit)
        // Calculate offset to L5
        // 1 + 7 + 49 + 343 + 2401 = 2801 items * 8 bytes = 22408 bytes
        offset += 22408;

        const count = 16807;
        const hexData = new Array(count);

        const unitScale = 0.5; // Neighbor delta units

        // This loop works well enough for 16k items
        // Since attributes are Float32Arrays later, we convert here.

        for (let i = 0; i < count; i++) {
            const baseOff = offset + (i * 8);

            // Height (Uint16) -> Float
            const hRaw = view.getUint16(baseOff, true);
            const zAbs = minZ + hRaw / scale;

            // Neighbors (6 x Int8)
            // Stored as relative delta in 0.5m units
            const neighbors = [];
            for (let k = 0; k < 6; k++) {
                const delta = view.getInt8(baseOff + 2 + k);
                const val = zAbs + (delta * unitScale);
                neighbors.push(val);
            }

            // Mapping to N_N, N_NE...
            // Standard order: N, NE, SE, S, SW, NW
            hexData[i] = {
                z: zAbs,
                n_n: neighbors[0],
                n_ne: neighbors[1],
                n_se: neighbors[2],
                n_s: neighbors[3],
                n_sw: neighbors[4],
                n_nw: neighbors[5],
                s: [0, 0, 0, 0, 0, 0] // Slopes implicit now
            };
        }

        return {
            hexes: hexData,
            stats: { base: minZ, min: minZ, max: maxZ, avg: (minZ + maxZ) / 2 },
            ySpacing: this.currentRes // We trust the renderer knows ~6.4m
        };
    }

    createInstancedMesh(hexes, material, tileYSpacing) {
        const numHexes = hexes.length;
        const currentTileYSpacing = tileYSpacing || this.hexWidth;
        const currentTileXSpacing = currentTileYSpacing * (Math.sqrt(3) / 2);

        // FIX: Clone geometry so each tile has unique instance attributes
        // ALSO: Create a fresh hex geometry for this specific spacing
        const side = currentTileYSpacing / Math.sqrt(3);
        const tileGeom = this.createHexGeometry(side);
        const mesh = new THREE.InstancedMesh(tileGeom, material, numHexes);

        const matrix = new THREE.Matrix4();
        const instanceNZ_1 = new Float32Array(numHexes * 4);
        const instanceNZ_2 = new Float32Array(numHexes * 4);
        const instanceBorder = new Float32Array(numHexes);

        const xSteps = [];
        for (let x = 0; x <= TILE_WIDTH_WORLD + 1; x += currentTileXSpacing) xSteps.push(x);
        const ySteps = [];
        for (let y = 0; y <= TILE_HEIGHT_WORLD + 1; y += currentTileYSpacing) ySteps.push(y);

        const rowCount = ySteps.length;
        const colCount = xSteps.length;

        for (let col = 0; col < colCount; col++) {
            const x = xSteps[col];
            const yShift = (col % 2 === 1) ? (currentTileYSpacing / 2.0) : 0;
            for (let row = 0; row < rowCount; row++) {
                const instanceIdx = (col * rowCount) + row;
                if (instanceIdx >= numHexes) continue;

                const dataRow = DEM_FLIP_NS ? (rowCount - 1 - row) : row;
                const dataIdx = (col * rowCount) + dataRow;
                if (dataIdx >= numHexes) continue;

                const realY = ySteps[row] + yShift;
                const h = hexes[dataIdx];

                // Position within the tile
                matrix.makeTranslation(x, 0, -realY);
                mesh.setMatrixAt(instanceIdx, matrix);

                instanceBorder[instanceIdx] = (BORDER_WALLS_ALWAYS && (
                    row === 0 || row === rowCount - 1 || col === 0 || col === colCount - 1
                )) ? 1.0 : 0.0;

                instanceNZ_1[instanceIdx * 4] = h.n_n * SCALE_Z;
                instanceNZ_1[instanceIdx * 4 + 1] = h.n_ne * SCALE_Z;
                instanceNZ_1[instanceIdx * 4 + 2] = h.n_se * SCALE_Z;
                instanceNZ_1[instanceIdx * 4 + 3] = h.n_s * SCALE_Z;

                instanceNZ_2[instanceIdx * 4] = h.n_sw * SCALE_Z;
                instanceNZ_2[instanceIdx * 4 + 1] = h.n_nw * SCALE_Z;
                instanceNZ_2[instanceIdx * 4 + 2] = h.z * SCALE_Z;
                instanceNZ_2[instanceIdx * 4 + 3] = 0.0;
            }
        }

        mesh.instanceMatrix.needsUpdate = true;
        mesh.geometry.setAttribute('instanceNZ_1', new THREE.InstancedBufferAttribute(instanceNZ_1, 4));
        mesh.geometry.setAttribute('instanceNZ_2', new THREE.InstancedBufferAttribute(instanceNZ_2, 4));

        const instanceSlope_1 = new Float32Array(numHexes * 4);
        const instanceSlope_2 = new Float32Array(numHexes * 4);

        for (let i = 0; i < numHexes; i++) {
            const h = hexes[i];
            instanceSlope_1[i * 4] = h.s[0];
            instanceSlope_1[i * 4 + 1] = h.s[1];
            instanceSlope_1[i * 4 + 2] = h.s[2];
            instanceSlope_1[i * 4 + 3] = h.s[3];

            instanceSlope_2[i * 4] = h.s[4];
            instanceSlope_2[i * 4 + 1] = h.s[5];
            instanceSlope_2[i * 4 + 2] = 0.0;
            instanceSlope_2[i * 4 + 3] = 0.0;
        }

        mesh.geometry.setAttribute('instanceSlope_1', new THREE.InstancedBufferAttribute(instanceSlope_1, 4));
        mesh.geometry.setAttribute('instanceSlope_2', new THREE.InstancedBufferAttribute(instanceSlope_2, 4));
        mesh.geometry.setAttribute('instanceBorder', new THREE.InstancedBufferAttribute(instanceBorder, 1));
        mesh.frustumCulled = true;

        return mesh;
    }

    setupMaterialShader(material) {
        material.userData.shader = null;
        material.onBeforeCompile = (shader) => {
            material.userData.shader = shader;
            shader.uniforms.uHeightFactor = { value: 0.0 };
            shader.uniforms.uFloorOffset = { value: this.floorState.value };
            shader.uniforms.uTileResolution = { value: material.userData.uTileResolution || 10.0 };
            shader.uniforms.uTileSize = { value: SECTOR_WIDTH_METERS };
            shader.uniforms.uTextureFlipY = { value: 0.0 };

            shader.uniforms.uAoFloor = { value: LIGHTING_DEFAULTS.aoFloor };
            shader.uniforms.uAoPower = { value: LIGHTING_DEFAULTS.aoPower };
            shader.uniforms.uLambertStrength = { value: LIGHTING_DEFAULTS.lambert };
            shader.uniforms.uRimStrength = { value: LIGHTING_DEFAULTS.rim };
            shader.uniforms.uRimPower = { value: LIGHTING_DEFAULTS.rimPower };
            shader.uniforms.uSpecStrength = { value: LIGHTING_DEFAULTS.spec };
            shader.uniforms.uSpecPower = { value: LIGHTING_DEFAULTS.specPower };
            shader.uniforms.uSlopeLight = { value: LIGHTING_DEFAULTS.slopeLight };

            shader.vertexShader = shader.vertexShader.replace(
                '#include <common>',
                `#include <common>
                uniform float uHeightFactor;
                uniform float uFloorOffset;
                uniform float uTileResolution; // Declared uniform
                attribute vec4 instanceNZ_1; 
                attribute vec4 instanceNZ_2;
                attribute vec4 instanceSlope_1;
                attribute vec4 instanceSlope_2;
                attribute float instanceBorder;
                attribute float faceIndex;
                
                varying vec3 vLocalPos;
                varying vec3 vWorldPos;
                varying vec3 vObjNormal;
                varying float vFaceSlope;
                varying float vGrad;
                varying float vIsHidden;
                varying float vFaceId;
                `
            ).replace(
                '#include <begin_vertex>',
                `#include <begin_vertex>
                vIsHidden = 0.0;
                vFaceId = faceIndex;
                int face = int(faceIndex + 0.5);
                float myZ = instanceNZ_2.z - uFloorOffset;
                float neighborZ = 0.0;
                bool isWall = true;
                
                // Face index mapping (verified in-scene):
                // 0=N, 1=NE, 2=SE, 3=S, 4=SW, 5=NW (clockwise)
                // Face-specific skirt logic (Manifold Crust):
                // Each face renders down to the height of ITS specific neighbor.
                // This seals the gap perfectly while avoiding redundant deep-geometry.
                if (face == 0) neighborZ = instanceNZ_1.x - uFloorOffset;
                else if (face == 1) neighborZ = instanceNZ_1.y - uFloorOffset;
                else if (face == 2) neighborZ = instanceNZ_1.z - uFloorOffset;
                else if (face == 3) neighborZ = instanceNZ_1.w - uFloorOffset;
                else if (face == 4) neighborZ = instanceNZ_2.x - uFloorOffset;
                else if (face == 5) neighborZ = instanceNZ_2.y - uFloorOffset;
                else neighborZ = myZ;

                isWall = (face < 6); 




                if (${DEBUG_FIXED_WALLS ? 'true' : 'false'}) {
                    neighborZ = myZ - ${DEBUG_FIXED_WALL_DEPTH.toFixed(1)};
                }

                float animMyZ = myZ * uHeightFactor;
                float animNeighborZ = neighborZ * uHeightFactor;

                if (isWall) {
                    if (${DEBUG_FIXED_WALLS ? 'true' : 'false'}) {
                        if (position.y > 0.5) transformed.y = animMyZ;
                        else transformed.y = animNeighborZ;
                    } else {
                        if (instanceBorder < 0.5 && myZ <= neighborZ + 0.01) { 
                            vIsHidden = 1.0;
                            transformed = vec3(0.0);
                        } else {
                            if (position.y > 0.5) transformed.y = animMyZ;
                            else transformed.y = animNeighborZ;
                        }
                    }
                } else {
                    transformed.y = animMyZ; // Top Cap
                }

                // UV Calculation
                #ifdef USE_INSTANCING
                    vLocalPos = (instanceMatrix * vec4(transformed, 1.0)).xyz;
                #else
                    vLocalPos = transformed;
                #endif
                vWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;

                vObjNormal = normal;

                float dz = abs(myZ - neighborZ);
                
                // Use per-face slope from attributes
                float slopeVal = 0.0;
                if (face < 1) slopeVal = instanceSlope_1.x;
                else if (face < 2) slopeVal = instanceSlope_1.y;
                else if (face < 3) slopeVal = instanceSlope_1.z;
                else if (face < 4) slopeVal = instanceSlope_1.w;
                else if (face < 5) slopeVal = instanceSlope_2.x;
                else if (face < 6) slopeVal = instanceSlope_2.y;
                
                vFaceSlope = slopeVal;
                
                vGrad = 1.0;
                if (isWall && position.y < 0.5) vGrad = 0.0;
                `
            );

            shader.fragmentShader = shader.fragmentShader.replace(
                '#include <common>',
                `#include <common>
                uniform float uTextureFlipY;
                uniform float uAoFloor;
                uniform float uAoPower;
                uniform float uLambertStrength;
                uniform float uRimStrength;
                uniform float uRimPower;
                uniform float uSpecStrength;
                uniform float uSpecPower;
                uniform float uSlopeLight;
                uniform float uTileSize;
                varying vec3 vLocalPos;
                varying vec3 vWorldPos;
                varying vec3 vObjNormal;
                varying float vFaceSlope;
                varying float vGrad;
                varying float vIsHidden;
                varying float vFaceId;
                `
            ).replace(
                '#include <map_fragment>',
                `if (vIsHidden > 0.5) discard;
                
                // UV Mapping
                float u = vLocalPos.x / uTileSize;
                float v = -vLocalPos.z / uTileSize;
                float texY = (uTextureFlipY > 0.5) ? v : (1.0 - v);
                vec2 myUV = vec2(clamp(u, 0.001, 0.999), clamp(texY, 0.001, 0.999));
                
                vec4 texColor = texture2D(map, myUV);
                diffuseColor = texColor;

                if (abs(vObjNormal.y) < 0.9) {
                    float verticalShade = mix(0.32, 1.0, pow(vGrad, 1.4));
                    vec3 sideBase = texColor.rgb * verticalShade;
                    
                    vec3 lightDir = normalize(vec3(0.45, 0.80, 0.35));
                    vec3 nrm = normalize(vObjNormal);
                    vec3 viewDir = normalize(cameraPosition - vWorldPos);
                    float lambert = clamp(dot(nrm, lightDir), 0.0, 1.0);
                    float rim = pow(1.0 - clamp(dot(nrm, viewDir), 0.0, 1.0), uRimPower);
                    vec3 halfDir = normalize(lightDir + viewDir);
                    float spec = pow(max(dot(nrm, halfDir), 0.0), uSpecPower) * uSpecStrength;
                    float ao = mix(uAoFloor, 1.0, pow(vGrad, uAoPower));
                    float light = mix(1.0 - uLambertStrength, 1.0, lambert);
                    light += uRimStrength * rim + spec;
                    light = clamp(light, 0.0, 1.5);
                    sideBase *= light * ao;

                    vec3 cGreen = vec3(0.0, 1.0, 0.0);
                    vec3 cBlue = vec3(0.0009, 0.0027, 0.1119);
                    vec3 cYellow = vec3(1.0, 0.85, 0.0);
                    vec3 cRed = vec3(0.85, 0.0, 0.0);

                    vec3 slopeColor = cGreen;
                    if (vFaceSlope >= 30.0 && vFaceSlope < 40.0) slopeColor = cGreen;
                    else if (vFaceSlope >= 40.0 && vFaceSlope < 50.0) slopeColor = cYellow;
                    else if (vFaceSlope >= 50.0 && vFaceSlope < 60.0) slopeColor = cRed;
                    else if (vFaceSlope >= 60.0) slopeColor = cBlue;
                    slopeColor *= mix(1.0, light, uSlopeLight) * ao;
                    
                    if (${DEBUG_VIOLET_SOUTH ? 'true' : 'false'}) {
                        if (vFaceId < 0.5) diffuseColor.rgb = vec3(1.0, 0.1, 0.1); // S
                        else if (vFaceId < 1.5) diffuseColor.rgb = vec3(1.0, 0.6, 0.0); // SE
                        else if (vFaceId < 2.5) diffuseColor.rgb = vec3(1.0, 1.0, 0.1); // NE
                        else if (vFaceId < 3.5) diffuseColor.rgb = vec3(0.8, 0.1, 0.9); // N
                        else if (vFaceId < 4.5) diffuseColor.rgb = vec3(0.1, 1.0, 1.0); // NW
                        else if (vFaceId < 5.5) diffuseColor.rgb = vec3(0.1, 0.9, 0.2); // SW
                    } else if (vFaceSlope >= 30.0) {
                        diffuseColor.rgb = slopeColor;
                    } else {
                        diffuseColor.rgb = sideBase;
                    }
                }
                `
            );
        };
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();

        // === DEBUG: Frametime tracking ===
        const now = performance.now();
        const frameTime = now - this.lastFrameTime;
        this.lastFrameTime = now;
        this.frametimeBuffer[this.frametimeIndex] = frameTime;
        this.frametimeIndex = (this.frametimeIndex + 1) % 120;
        // === END DEBUG ===

        // Camera altitude is now maintained by maintainCameraAltitudeDuringAnimation()
        // which properly accounts for heightFactor scaling

        // === DEBUG: Update performance metrics ===
        this.updateRenderStats(now);
        this.drawFrametimeGraph();
        // === END DEBUG ===

        const angle = this.controls.getPolarAngle();
        const angleDeg = angle * 180 / Math.PI;

        const ts = this.transSettings;
        const FLAT_THRESHOLD = ts.flatThresh;
        const TRANSITION_START = ts.riseStart;
        const TRANSITION_END = ts.riseEnd;

        // Calculate raw linear progress (0.0 to 1.0)
        let linearFactor = (angleDeg - TRANSITION_START) / (TRANSITION_END - TRANSITION_START);
        linearFactor = Math.min(1.0, Math.max(0.0, linearFactor));

        // Apply non-linear curve (power easing)
        // Clamping to 0.0 first ensures the curve always starts at the origin
        const curvedFactor = Math.pow(linearFactor, ts.curve);

        // Final height factor with 0.1% "pancake" floor for visual structure
        const heightFactor = Math.max(0.001, curvedFactor);

        const isFlatMode = angleDeg < FLAT_THRESHOLD;

        this.updateFloorState(heightFactor);
        this.maintainCameraAltitudeDuringAnimation(heightFactor);

        for (const tile of this.tiles) {
            if (tile.flatMesh) tile.flatMesh.visible = isFlatMode;
            if (tile.mesh) tile.mesh.visible = !isFlatMode;
        }

        for (const mat of this.materialsToUpdate) {
            if (mat.userData.shader) {
                mat.userData.shader.uniforms.uHeightFactor.value = heightFactor;
            }
        }

        this.updateLOD();

        // LOD Science Benchmark
        if (this.lodController && this.lodController.initialized) {
            const bench = this.lodController.runBenchmark(this.camera.position); // returns { times: { loop, tree, hash }, results }

            // Accumulate
            this.lodStats.accLoop += bench.times.loop;
            this.lodStats.accTree += bench.times.tree;
            this.lodStats.accHash += bench.times.hash;
            this.lodStats.count++;

            // Update UI at 2Hz
            if (now - this.lodStats.lastUpdate > 500) {
                const count = this.lodStats.count;
                const elLoop = document.getElementById('bench-loop');
                const elTree = document.getElementById('bench-tree');
                const elHash = document.getElementById('bench-hash');

                if (elLoop) elLoop.textContent = (this.lodStats.accLoop / count).toFixed(5);
                if (elTree) elTree.textContent = (this.lodStats.accTree / count).toFixed(5);
                if (elHash) elHash.textContent = (this.lodStats.accHash / count).toFixed(5);

                this.lodStats.accLoop = 0;
                this.lodStats.accTree = 0;
                this.lodStats.accHash = 0;
                this.lodStats.count = 0;
                this.lodStats.lastUpdate = now;
            }
        }

        if (this.webglActive) {
            this.renderer.render(this.scene, this.camera);
        }
        this.floorState.lastFactor = heightFactor;
    }

    initCSSMap() {
        this.cssWorld = document.getElementById('css-world');
        if (!this.cssWorld) return;

        this.log("Initializing CSS Map...", "info");

        // Create DOM tiles
        for (const tileDef of this.manifest.tiles) {
            const div = document.createElement('div');
            div.className = 'css-tile';
            div.style.width = `${TILE_WIDTH_WORLD}px`;
            div.style.height = `${TILE_HEIGHT_WORLD}px`;

            // Texture
            const lowTexUrl = `aerial_tiles/low_res/sector_${tileDef.q}_${tileDef.r}.webp`;
            div.style.backgroundImage = `url(${lowTexUrl})`;

            // Position: Map 3D (X,Z) to CSS (X,Y)
            // tileDef.x/y is the world center of the sector.
            // tx/ty is the top-left offset for the CSS div.
            const tx = (tileDef.x - this.worldOrigin.x) - (TILE_WIDTH_WORLD / 2);
            const ty = -(tileDef.y - this.worldOrigin.y) - (TILE_HEIGHT_WORLD / 2);

            // Place tile flat on the XY plane (which represents the Ground)
            div.style.left = `${tx}px`;
            div.style.top = `${ty}px`;

            this.cssWorld.appendChild(div);
        }

        // Initial visibility
        const layer = document.getElementById('css-map-layer');
        if (layer) layer.style.opacity = '1';

        this.renderer.domElement.style.transition = 'opacity 0.5s';
        this.renderer.domElement.style.opacity = '0';
    }


    // Unified WebGL Pipeline Replaced CSS Map

    updateFps() {
        if (!this.fpsEl) return;
        const now = performance.now();
        this.fpsState.frames += 1;
        const elapsed = now - this.fpsState.lastSample;
        if (elapsed < 500) return;
        const fps = (this.fpsState.frames * 1000) / elapsed;
        const dist = this.camera.position.distanceTo(this.controls.target);
        this.fpsEl.textContent = `FPS: ${fps.toFixed(0)} | Zoom: ${dist.toFixed(0)}`;
        this.fpsState.frames = 0;
        this.fpsState.lastSample = now;
    }

    updateLOD() {
        if (!this.manifest) return;
        const target = this.controls.target;
        const CAM_HIGH_RES_DIST = 600;
        const CAM_MED_RES_DIST = 1500;
        const renderDistance = this.renderSettings.renderDistance;
        const renderDistanceSq = renderDistance * renderDistance;

        // 1. Check manifest for tiles that need to be loaded
        for (const tileDef of this.manifest.tiles) {
            const cx = (tileDef.x - this.worldOrigin.x) + 625;
            const cz = -(tileDef.y - this.worldOrigin.y) - 500;
            const distSq = (target.x - cx) ** 2 + (target.z - cz) ** 2;

            if (distSq < renderDistanceSq) {
                // Check if already loaded
                const isLoaded = this.tiles.some(t => t.x === tileDef.x && t.y === tileDef.y);
                if (!isLoaded) {
                    this.loadSingleTile(tileDef).catch(e => console.error("Lazy Load Error:", e));
                }
            }
        }

        const texLoader = new THREE.TextureLoader();
        const tiffLoader = new TIFFLoader();

        // 2. Handle LOD for already loaded tiles
        for (const tile of this.tiles) {
            const cx = (tile.x - this.worldOrigin.x) + 625;
            const cz = -(tile.y - this.worldOrigin.y) - 500;
            const distSq = (target.x - cx) ** 2 + (target.z - cz) ** 2;

            if (distSq > renderDistanceSq) {
                if (tile.mesh.visible) tile.mesh.visible = false;
                continue;
            }
            if (!tile.mesh.visible) tile.mesh.visible = true;

            if (distSq < CAM_HIGH_RES_DIST ** 2) {
                if (!tile.highResLoaded && !tile.highResLoading) {
                    tile.highResLoading = true;
                    // Note: In a real production app we'd want to handle the 
                    // switch from WebP to TIFF or High-WebP here.
                    texLoader.load(tile.urls.high, (tex) => {
                        tex.colorSpace = THREE.SRGBColorSpace;
                        tex.flipY = false;
                        tile.material.map = tex;
                        tile.material.needsUpdate = true;
                        tile.highResLoaded = true;
                        tile.highResLoading = false;
                    });
                }
            }
        }
    }

    onResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    createHexGeometry(radius) {
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const normals = [];
        const faceIndices = [];
        const corners = [];
        for (let i = 0; i < 6; i++) {
            const angle = (i * Math.PI / 3);
            corners.push({
                x: Math.cos(angle) * radius,
                z: Math.sin(angle) * radius
            });
        }
        const faceDirs = [
            { x: 0.0, z: -1.0 },      // 0: N
            { x: 0.866, z: -0.5 },    // 1: NE
            { x: 0.866, z: 0.5 },     // 2: SE
            { x: 0.0, z: 1.0 },       // 3: S
            { x: -0.866, z: 0.5 },    // 4: SW
            { x: -0.866, z: -0.5 },   // 5: NW
        ];
        const pushWall = (p1, p2) => {
            const v0 = { x: p1.x, y: 0, z: p1.z };
            const v1 = { x: p2.x, y: 1, z: p2.z };
            const v2 = { x: p2.x, y: 0, z: p2.z };
            const out = { x: p1.x + p2.x, z: p1.z + p2.z };
            let best = 0; let max = -Infinity;
            for (let i = 0; i < 6; i++) {
                const d = out.x * faceDirs[i].x + out.z * faceDirs[i].z;
                if (d > max) { max = d; best = i; }
            }
            positions.push(p1.x, 0, p1.z, p2.x, 1, p2.z, p2.x, 0, p2.z);
            positions.push(p1.x, 0, p1.z, p1.x, 1, p1.z, p2.x, 1, p2.z);
            for (let j = 0; j < 6; j++) faceIndices.push(best);
        };
        for (let i = 0; i < 6; i++) {
            const p1 = corners[i];
            const p2 = corners[(i + 1) % 6];
            pushWall(p1, p2);
            const ang = Math.atan2((p1.z + p2.z) / 2, (p1.x + p2.x) / 2);
            const nx = Math.cos(ang), nz = Math.sin(ang);
            for (let k = 0; k < 6; k++) normals.push(nx, 0, nz);
        }
        for (let i = 0; i < 6; i++) {
            const p1 = corners[i];
            const p2 = corners[(i + 1) % 6];
            positions.push(0, 1, 0, p2.x, 1, p2.z, p1.x, 1, p1.z);
            normals.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
            faceIndices.push(6, 6, 6);
            positions.push(0, 0, 0, p1.x, 0, p1.z, p2.x, 0, p2.z);
            normals.push(0, -1, 0, 0, -1, 0, 0, -1, 0);
            faceIndices.push(7, 7, 7);
        }
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
        geometry.setAttribute('faceIndex', new THREE.Float32BufferAttribute(faceIndices, 1));
        return geometry;
    }

    decodeFloat16(binary) {
        const exponent = (binary & 0x7C00) >> 10;
        const fraction = binary & 0x03FF;
        return (binary >> 15 ? -1 : 1) * (
            exponent ?
                (exponent === 0x1F ? (fraction ? NaN : Infinity) : Math.pow(2, exponent - 15) * (1 + fraction / 1024)) :
                6.103515625e-5 * (fraction / 1024)
        );
    }
}

new PistonViewer();
