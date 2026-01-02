import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { TIFFLoader } from 'three/addons/loaders/TIFFLoader.js';

// --- CONFIG ---
const TILE_WIDTH_WORLD = 1250;
const TILE_HEIGHT_WORLD = 1000;
const HEX_WIDTH = 10;
const HEX_DX = HEX_WIDTH * (Math.sqrt(3) / 2);
const SCALE_Z = 1.0;
const DEFAULT_RENDER_DISTANCE = 10000;
const FLOOR_MODE = 'view-min';
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

        // Resize Handler
        window.addEventListener('resize', this.onResize.bind(this));

        // Shared Geometry
        const side = HEX_WIDTH / Math.sqrt(3);
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

        // Start
        this.initDebugConsole();
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

        const binUrl = `tiles_bin/tile_${x}_${y}.bin`;
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
        const mesh = this.createInstancedMesh(hexData, material);

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
        let headerSize = 4;

        if ((buffer.byteLength - 12) % 14 === 0) {
            baseElevation = view.getFloat32(0, true);
            minElevation = view.getFloat32(4, true);
            avgElevation = view.getFloat32(8, true);
            headerSize = 12;
        } else if ((buffer.byteLength - 4) % 14 === 0) {
            headerSize = 4;
        }

        const hexData = [];
        let offset = headerSize;
        let minAbs = Infinity;
        let maxAbs = -Infinity;
        let sumAbs = 0.0;
        let count = 0;

        while (offset < buffer.byteLength) {
            const z = this.decodeFloat16(view.getUint16(offset, true));
            const n_n = this.decodeFloat16(view.getUint16(offset + 2, true));
            const n_ne = this.decodeFloat16(view.getUint16(offset + 4, true));
            const n_se = this.decodeFloat16(view.getUint16(offset + 6, true));
            const n_s = this.decodeFloat16(view.getUint16(offset + 8, true));
            const n_sw = this.decodeFloat16(view.getUint16(offset + 10, true));
            const n_nw = this.decodeFloat16(view.getUint16(offset + 12, true));

            const zAbs = z + baseElevation;
            hexData.push({
                z: zAbs,
                n_n: n_n + baseElevation,
                n_ne: n_ne + baseElevation,
                n_se: n_se + baseElevation,
                n_s: n_s + baseElevation,
                n_sw: n_sw + baseElevation,
                n_nw: n_nw + baseElevation
            });

            if (headerSize === 4) {
                minAbs = Math.min(minAbs, zAbs);
                maxAbs = Math.max(maxAbs, zAbs);
                sumAbs += zAbs;
                count += 1;
            }
            offset += 14;
        }

        if (headerSize === 4) {
            if (!Number.isFinite(minAbs)) minAbs = baseElevation;
            if (!Number.isFinite(maxAbs)) maxAbs = baseElevation;
            const avgAbs = count ? (sumAbs / count) : baseElevation;
            return { hexes: hexData, stats: { base: baseElevation, min: minAbs, max: maxAbs, avg: avgAbs } };
        }
        // If header size is 12, we don't have max in the header, so we should really compute it above anyway
        // or just return min/avg as provided. Let's assume max is useful and compute it.
        return { hexes: hexData, stats: { base: baseElevation, min: minElevation, max: maxAbs, avg: avgElevation } };
    }

    createInstancedMesh(hexes, material) {
        const numHexes = hexes.length;

        // FIX: Clone geometry so each tile has unique instance attributes
        const mesh = new THREE.InstancedMesh(this.hexGeometry.clone(), material, numHexes);

        const matrix = new THREE.Matrix4();
        const instanceNZ_1 = new Float32Array(numHexes * 4);
        const instanceNZ_2 = new Float32Array(numHexes * 4);
        const instanceBorder = new Float32Array(numHexes);

        const xSteps = [];
        for (let x = 0; x <= TILE_WIDTH_WORLD + 1; x += HEX_DX) xSteps.push(x);
        const ySteps = [];
        for (let y = 0; y <= TILE_HEIGHT_WORLD + 1; y += 10) ySteps.push(y);

        const rowCount = ySteps.length;
        const colCount = xSteps.length;

        for (let col = 0; col < colCount; col++) {
            const x = xSteps[col];
            const yShift = (col % 2 === 1) ? 5 : 0;
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
                attribute vec4 instanceNZ_1; 
                attribute vec4 instanceNZ_2;
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
                // Uniform "Max Depth" Skirt Logic:
                // Find the lowest neighbor Z among all 6 neighbors to ensure skirts cover everything.
                float minNeighborZ = myZ;
                minNeighborZ = min(minNeighborZ, instanceNZ_1.x - uFloorOffset); // N
                minNeighborZ = min(minNeighborZ, instanceNZ_1.y - uFloorOffset); // NE
                minNeighborZ = min(minNeighborZ, instanceNZ_1.z - uFloorOffset); // SE
                minNeighborZ = min(minNeighborZ, instanceNZ_1.w - uFloorOffset); // S
                minNeighborZ = min(minNeighborZ, instanceNZ_2.x - uFloorOffset); // SW
                minNeighborZ = min(minNeighborZ, instanceNZ_2.y - uFloorOffset); // NW
                
                neighborZ = minNeighborZ;
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
                vFaceSlope = degrees(atan(dz, 10.0));
                
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
                    vec3 cBlue = vec3(0.0, 0.15, 0.5);
                    vec3 cYellow = vec3(1.0, 0.85, 0.0);
                    vec3 cRed = vec3(0.85, 0.0, 0.0);

                    vec3 slopeColor = cGreen;
                    if (vFaceSlope >= 35.0 && vFaceSlope < 40.0) slopeColor = cBlue;
                    else if (vFaceSlope >= 40.0 && vFaceSlope < 45.0) slopeColor = cYellow;
                    else if (vFaceSlope >= 45.0) slopeColor = cRed;
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

        const minCamY = 100.0;
        if (this.camera.position.y < minCamY) {
            this.camera.position.y = minCamY;
        }

        this.updateFps();

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
