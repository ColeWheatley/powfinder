import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { TIFFLoader } from 'three/addons/loaders/TIFFLoader.js';

// --- CONFIG ---
const TILE_WIDTH_WORLD = 1250;
const TILE_HEIGHT_WORLD = 1000;
const SCALE_Z = 1.0;
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
        this.resolutions = RESOLUTIONS;
        this.currentResIndex = 7; // 7.7136m
        this.currentRes = this.resolutions[this.currentResIndex];
        this.currentResMethod = 'bilinear';
        this.hexWidth = this.currentRes;
        this.hexDx = this.hexWidth * (Math.sqrt(3) / 2);

        // Resize Handler
        window.addEventListener('resize', this.onResize.bind(this));

        // Shared Geometry
        const side = this.hexWidth / Math.sqrt(3);
        this.hexGeometry = this.createHexGeometry(side);

        // State
        this.tiles = [];
        this.manifest = null;
        this.loadingTiles = new Set(); // Track tiles currently in flight
        this.essentialTilesLoaded = 0;
        this.essentialTilesTarget = 0;
        this.loaderHidden = false;
        this.materialsToUpdate = [];
        this.cssMapActive = true;
        this.cssWorld = null;
        this.webglActive = false;
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

        // === DEBUG: Render Analysis ===
        this.hexCountEl = document.getElementById('hex-count');
        this.triCountEl = document.getElementById('tri-count');
        this.drawStatsEl = document.getElementById('draw-stats');
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
        this.initResSlider();
        this.initResMethod();
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

    initResSlider() {
        const slider = document.getElementById('res-slider');
        const valLabel = document.getElementById('res-val');
        if (slider && valLabel) {
            slider.addEventListener('input', () => {
                const index = parseInt(slider.value);
                const res = this.resolutions[index];
                valLabel.textContent = `${res}m`;
            });

            slider.addEventListener('change', async () => {
                const index = parseInt(slider.value);
                const res = this.resolutions[index];
                this.log(`Changing resolution to ${res}m...`, "info");

                this.currentResIndex = index;
                this.currentRes = res;
                this.hexWidth = res;
                this.hexDx = this.hexWidth * (Math.sqrt(3) / 2);

                this.clearAllTiles();
                this.updateLOD();
            });
        }
    }

    initResMethod() {
        const resMethod = document.getElementById('res-method');
        if (resMethod) {
            resMethod.addEventListener('change', (e) => {
                this.currentResMethod = e.target.value;
                this.log(`Resampling method changed to ${this.currentResMethod}. Reloading...`, "info");
                this.clearAllTiles();
                this.updateLOD();
            });
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
            const res = await fetch('tile_manifest.json');
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

            // Build CSS Map
            this.initCSSMap();

            // Only trigger updateLOD to start loading what's in range
            this.updateLOD();

        } catch (err) {
            console.error("Error initializing world:", err);
            this.log(`Error initializing world: ${err.message}`, "error");
        }
    }

    async loadSingleTile(tileDef) {
        const { x, y } = tileDef;
        const tileKey = `${x}_${y}`;
        if (this.loadingTiles.has(tileKey)) return;
        this.loadingTiles.add(tileKey);

        const posX = x - this.worldOrigin.x;
        const posZ = -(y - this.worldOrigin.y);

        const t0 = performance.now();

        const binUrl = `tiles_bin/${this.currentResMethod}/res_${this.currentRes}/tile_${x}_${y}.bin`;
        const lowTexUrl = `tiles_sat/low_res/tile_${x}_${y}.webp`;
        const medTexUrl = `tiles_sat/med_res/tile_${x}_${y}.webp`;
        const highTexUrl = `tiles_sat/high_res/tile_${x}_${y}.tif`;

        // 1. Load Low Res Texture
        const texLoader = new THREE.TextureLoader();
        const texture = await texLoader.loadAsync(lowTexUrl);
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.flipY = false;

        // 2. Create Material
        const material = new THREE.MeshBasicMaterial({
            map: texture,
            side: THREE.DoubleSide,
            fog: true
        });
        this.setupMaterialShader(material);
        this.materialsToUpdate.push(material);

        // 3. Fetch Binary Data
        const response = await fetch(binUrl);
        const buffer = await response.arrayBuffer();
        const parsed = this.parseBinary(buffer);
        const hexData = parsed.hexes;
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
            x, y,
            mesh,
            material,
            stats,
            center: { x: centerX, z: centerZ },
            bounds,
            highResLoaded: false,
            highResLoading: false,
            medResLoaded: false,
            medResLoading: false,
            urls: { med: medTexUrl, high: highTexUrl }
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
                vLocalPos = (instanceMatrix * vec4(transformed, 1.0)).xyz;
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
                float u = vLocalPos.x / 1250.0;
                float v = -vLocalPos.z / 1000.0;
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

        // Keep camera above surface
        const camX = this.camera.position.x;
        const camZ = this.camera.position.z;
        let localMax = -10000.0;
        for (const tile of this.tiles) {
            if (tile.bounds && camX >= tile.bounds.min.x && camX <= tile.bounds.max.x &&
                camZ >= tile.bounds.min.z && camZ <= tile.bounds.max.z) {
                localMax = (tile.stats.max - this.floorState.value);
                break;
            }
        }

        const minCamY = Math.max(100.0, localMax + 100.0);
        if (this.camera.position.y < minCamY) {
            this.camera.position.y = minCamY;
        }

        // === DEBUG: Update performance metrics ===
        this.updateRenderStats(now);
        this.drawFrametimeGraph();
        // === END DEBUG ===

        const angle = this.controls.getPolarAngle();
        let factor = angle / (20 * Math.PI / 180);
        factor = Math.min(1.0, Math.max(0.01, factor));

        this.updateFloorState(factor);
        for (const mat of this.materialsToUpdate) {
            if (mat.userData.shader) {
                mat.userData.shader.uniforms.uHeightFactor.value = factor;
            }
        }

        this.updateCSSMap(angle, factor);
        this.updateLOD();
        if (this.webglActive) {
            this.renderer.render(this.scene, this.camera);
        }
        this.floorState.lastFactor = factor;
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
            const lowTexUrl = `tiles_sat/low_res/tile_${tileDef.x}_${tileDef.y}.webp`;
            div.style.backgroundImage = `url(${lowTexUrl})`;

            // Position: Map 3D (X,Z) to CSS (X,Y)
            // 3D: X is Right, Z is Down (towards viewer)
            // CSS: Left is Right, Top is Down

            const tx = (tileDef.x - this.worldOrigin.x);
            // In 3D logic: posZ = -(y - originY).
            // So if y increases (North), Z is negative (Up/Away).
            // In CSS, Up/Away is negative Y? No, Top=0 is top.
            // We want y-increasing (North) to be Up (Negative Top).
            // So CSS Y = -(y - originY) matches 3D Z.
            const ty = -(tileDef.y - this.worldOrigin.y);

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

    updateCSSMap(angle, factor) {
        if (!this.cssWorld) return;

        const pos = this.camera.position;
        const tgt = this.controls.target;
        const dist = pos.distanceTo(tgt);

        // Perspective Match
        // FOV 60 vertical.
        // At distance D, the view height is 2 * D * tan(30deg) ~= 1.15 * D.
        // In CSS, with perspective 800px, at distance 800px, 1px = 1px.
        // We want to scale such that our world units match pixels.
        // Scale = 800 / dist is a good approximation for 'constant size'.
        // But we want the map to 'zoom'.
        // Actually, if we translate the world by Z, it scales automatically in perspective.
        // BUT, we are trying to match Three.js OrbitControls zoom which moves the camera.
        // In CSS we move the World.

        // Let's rely on Scale.
        // Zoom 1 (Dist 1000): Scale 1.
        // Zoom 2 (Dist 500): Scale 2.
        const scale = 800.0 / dist;

        const deg = angle * 180 / Math.PI; // 0 is Top-Down in OrbitControls? 

        // Handoff Thresholds
        // Note: OrbitControls polar angle: 0 is Top (North Pole), PI/2 is Horizon.
        // So 2 degrees is very close to top-down.

        // Debug
        // if (Math.random() < 0.01) this.log(`Tilt: ${deg.toFixed(1)} Dist: ${dist.toFixed(0)}`, "info");

        if (deg > 5.0 && !this.webglActive) {
            this.webglActive = true;
            this.renderer.domElement.style.opacity = '1';
            this.cssWorld.parentElement.style.opacity = '0';
            this.log("Transition: CSS -> 3D", "info");
        }
        else if (deg < 2.0 && this.webglActive) {
            this.webglActive = false;
            this.renderer.domElement.style.opacity = '0';
            this.cssWorld.parentElement.style.opacity = '1';
            this.log("Transition: 3D -> CSS", "info");
        }

        // Optimization: Don't update hidden DOM
        // if (this.webglActive && this.renderer.domElement.style.opacity === '1') return;
        // Actually we should keep extended sync for smooth fade out

        const tx = -tgt.x;
        const ty = -tgt.z; // 3D Z maps to CSS Y (Top)

        // Rotation
        // OrbitControls Azimuth: Angle around Y axis.
        const azimuth = this.controls.getAzimuthalAngle();
        const rotZ = azimuth * 180 / Math.PI;
        // In CSS, rotating screen Z rotates the map like a wheel. correct.

        // Tilt
        // 3D: Camera tilts from Y axis.
        // CSS: We rotate the Plane around X axis.
        // 0 deg tilt = Flat.
        const rotX = deg;

        // Transform Composition
        // 1. Center the target point: translate(tx, ty)
        // 2. Rotate World around target: rotateZ (azimuth) -> rotateX (tilt)
        // 3. Scale (Zoom)

        this.cssWorld.style.transform =
            `scale(${scale}) ` +
            `rotateX(${rotX}deg) ` +
            `rotateZ(${rotZ}deg) ` +
            `translate3d(${tx}px, ${ty}px, 0px)`;

        // Add more logging for debugging
        // if (Math.random() < 0.05) { // Log occasionally to avoid spam
        //     this.log(`CSS Map Transform: scale=${scale.toFixed(2)}, rotX=${rotX.toFixed(1)}deg, rotZ=${rotZ.toFixed(1)}deg, translate3d(${tx.toFixed(0)}px, ${ty.toFixed(0)}px, 0px)`, "debug");
        // }
    }

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
                    tiffLoader.load(tile.urls.high, (tex) => {
                        tex.colorSpace = THREE.SRGBColorSpace;
                        tex.flipY = false;
                        tile.material.map = tex;
                        tile.material.needsUpdate = true;
                        tile.highResLoaded = true;
                        tile.highResLoading = false;
                    });
                }
            } else if (distSq < CAM_MED_RES_DIST ** 2) {
                if (!tile.highResLoaded && !tile.medResLoaded && !tile.medResLoading) {
                    tile.medResLoading = true;
                    texLoader.load(tile.urls.med, (tex) => {
                        tex.colorSpace = THREE.SRGBColorSpace;
                        tex.flipY = false;
                        if (!tile.highResLoaded) {
                            tile.material.map = tex;
                            tile.material.needsUpdate = true;
                        }
                        tile.medResLoaded = true;
                        tile.medResLoading = false;
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
