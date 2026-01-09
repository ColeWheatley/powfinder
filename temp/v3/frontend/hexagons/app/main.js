import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';

// --- HEX COORDINATE SYSTEM (Rectangular Sectors) ---
const UNIT_HEX_PX = 32.0;
const METERS_PER_PIXEL = 0.2;
const UNIT_HEX_WIDTH_METERS = UNIT_HEX_PX * METERS_PER_PIXEL; // 6.4m
const SECTOR_WIDTH_METERS = 819.2; // 4096px

function worldToSectorID(worldX, worldY) {
    const sx = Math.floor(worldX / SECTOR_WIDTH_METERS);
    const sy = Math.floor(worldY / SECTOR_WIDTH_METERS);
    return { Q: sx, R: sy };
}

// --- CONFIG ---
const TILE_WIDTH_WORLD = SECTOR_WIDTH_METERS;
const TILE_HEIGHT_WORLD = SECTOR_WIDTH_METERS;
const SCALE_Z = 1.0;
// --- DEBUG OVERRIDE ---
// Expanding render distance to 40km to ensure all available tiles are loaded.
const DEFAULT_RENDER_DISTANCE = 40000;
const FLOOR_MODE = 'view-min';
const LOCK_FLOOR_ON_RISE = true;
const FLOOR_LOCK_THRESHOLD = 0.02;
const TILE_BOUNDS_MIN_Y = -10000;
const TILE_BOUNDS_MAX_Y = 10000;

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
        console.log("Initializing PistonViewer (Priority Radial + LOD)...");
        this.container = document.getElementById('canvas-container');
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

        this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 10, 50000);
        this.camera.position.set(0, 800, 0);

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

        this.scene.add(new THREE.AmbientLight(0xffffff, 0.4));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(500, 2000, 500);
        this.scene.add(dirLight);

        // [Geo-Close, Geo-Mid, Geo-Far, Geo-Horizon]
        // Corresponding to Binary Layers: 3 (Unit), 2 (Scale 3), 1 (Scale 6), 0 (Scale 24)
        this.geoThresholds = [1500, 3000, 6000, 12000];

        // Texture High-Res Load Distance
        this.texThreshold = 2000;

        window.addEventListener('resize', this.onResize.bind(this));

        // Shared Geometry
        const side = UNIT_HEX_WIDTH_METERS / Math.sqrt(3);
        this.hexGeometry = this.createHexGeometry(side);
        this.flatGeometry = new THREE.PlaneGeometry(TILE_WIDTH_WORLD, TILE_HEIGHT_WORLD);
        this.flatGeometry.rotateX(-Math.PI / 2);

        this.tiles = new Map(); // Key: "q_r" -> Tile Object
        this.manifest = null;
        this.loadingTiles = new Set();
        this.loadQueue = [];
        this.isProcessingQueue = false;

        this.loaderHidden = false;
        this.materialsToUpdate = [];
        this.heightFactor = 0.0;
        this.transSettings = { flatThresh: 5.0, riseStart: 6.0, riseEnd: 25.0, curve: 1.0 };
        this.worldOrigin = { x: 0, y: 0 };
        this.floorMode = FLOOR_MODE;
        this.floorState = { locked: false, value: 0.0, lastFactor: 0.0 };
        this.globalStats = { min: Infinity, max: -Infinity, avgSum: 0.0, baseSum: 0.0, count: 0 };
        this.frustum = new THREE.Frustum();
        this.projScreenMatrix = new THREE.Matrix4();
        this.renderSettings = { renderDistance: DEFAULT_RENDER_DISTANCE };

        // Debug/Stats
        this.fpsState = { lastSample: performance.now(), frames: 0 };
        this.fpsEl = document.getElementById('fps-counter');
        this.hexCountEl = document.getElementById('hex-count');
        this.tileHeightEl = document.getElementById('tile-height');
        this.cameraHeightEl = document.getElementById('camera-height');
        this.statsUpdateState = { lastUpdate: 0, interval: 500 };

        this.initDebugConsole();
        this.initMinimizeButton();
        this.initLODSliders();
        this.updateFogAndClip();
        this.initWorld();
        this.animate();
    }

    log(msg, type = "info") {
        const el = document.getElementById('console-output');
        if (!el) return;
        const line = document.createElement('div');
        line.className = `log-line ${type}`;
        line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        el.appendChild(line);
        el.scrollTop = el.scrollHeight;
    }

    initDebugConsole() {
        this.log("PistonViewer Initialized.", "success");
    }

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

    initLODSliders() {
        // Geo LODs
        ['geo-lod0', 'geo-lod1', 'geo-lod2', 'geo-lod3'].forEach((id, i) => {
            const s = document.getElementById(`${id}-slider`);
            const v = document.getElementById(`${id}-val`);
            if (s) {
                // Initialize UI
                s.value = this.geoThresholds[i];
                if (v) v.textContent = s.value + "m";

                s.addEventListener('input', () => {
                    this.geoThresholds[i] = parseInt(s.value);
                    if (v) v.textContent = s.value + "m";
                });
            }
        });

        // Tex LOD
        const tSlider = document.getElementById('tex-lod0-slider');
        const tVal = document.getElementById('tex-lod0-val');
        if (tSlider) {
            tSlider.value = this.texThreshold;
            if (tVal) tVal.textContent = tSlider.value + "m";
            tSlider.addEventListener('input', () => {
                this.texThreshold = parseInt(tSlider.value);
                if (tVal) tVal.textContent = tSlider.value + "m";
            });
        }
    }

    createHexGeometry(radius) {
        const geometry = new THREE.CylinderGeometry(radius, radius, 1, 6);
        geometry.rotateY(Math.PI / 6); // Flat top
        return geometry;
    }

    onResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    updateFogAndClip() {
        const dist = this.renderSettings.renderDistance;
        const fogEnd = dist;
        const fogStart = dist * 0.6;
        if (!this.scene.fog) this.scene.fog = new THREE.Fog(0x000000, fogStart, fogEnd);
        this.scene.fog.near = fogStart;
        this.scene.fog.far = fogEnd;
        this.camera.far = dist + 2000;
        this.camera.updateProjectionMatrix();
    }

    async initWorld() {
        try {
            const res = await fetch('tile_manifest.json');
            this.manifest = await res.json();
            const { min_x, min_y, max_x, max_y } = this.manifest.bounds;
            this.worldOrigin = { x: min_x, y: min_y };

            const centerX = (min_x + max_x) / 2 - min_x;
            const centerZ = -((min_y + max_y) / 2 - min_y);
            this.camera.position.set(centerX, 800, centerZ);
            this.controls.target.set(centerX, 0, centerZ);
            this.controls.update();

            this.essentialTilesTarget = 1;
            this.updateLOD();
        } catch (e) { this.log("Manifest error: " + e.message, "error"); }
    }

    parseBinaryV3(buffer) {
        const view = new DataView(buffer);
        const minZ = view.getFloat32(12, true);
        const maxZ = view.getFloat32(16, true);
        const scale = view.getFloat32(20, true);
        let offset = 32;
        const layers = []; // [L3(Coarse), L2, L1, L0(Fine)]

        for (let l = 0; l < 4; l++) {
            const count = view.getUint32(offset, true);
            offset += 4;
            const layer = [];
            for (let i = 0; i < count; i++) {
                const dq = view.getInt16(offset, true);
                const dr = view.getInt16(offset + 2, true);
                const hn = view.getUint16(offset + 4, true);
                offset += 6;
                layer.push({ dq, dr, h: minZ + (hn / scale) });
            }
            layers.push(layer);
        }
        return { layers, stats: { min: minZ, max: maxZ, avg: (minZ + maxZ) / 2, base: minZ } };
    }

    createInstancedMeshV3(allLayers, lodIndex, material) {
        // lodIndex: 0=Fine(1x), 1=3x, 2=6x, 3=24x.
        // allLayers: [24x, 6x, 3x, 1x] (from backend)

        // Map LOD to Layer Index
        // LOD 0 (Fine) -> Layer 3 (Scale 1)
        // LOD 1        -> Layer 2 (Scale 3)
        // LOD 2        -> Layer 1 (Scale 6)
        // LOD 3 (Far)  -> Layer 0 (Scale 24)
        const layerIdx = 3 - Math.min(3, Math.max(0, lodIndex));
        const hexes = allLayers[layerIdx];

        const scaleTable = [24.0, 6.0, 3.0, 1.0];
        const scale = scaleTable[layerIdx];
        const num = hexes.length;

        // Clone geometry and SCALE it
        const geometry = this.hexGeometry.clone();
        geometry.scale(scale, 1, scale);

        const mesh = new THREE.InstancedMesh(geometry, material, num);
        const matrix = new THREE.Matrix4();
        const nz1 = new Float32Array(num * 4);
        const nz2 = new Float32Array(num * 4);

        const h = UNIT_HEX_WIDTH_METERS;
        const dx_dq = (Math.sqrt(3) / 2) * (h * scale);
        const dy_dq = 0.5 * (h * scale);
        const dy_dr = (h * scale);

        for (let i = 0; i < num; i++) {
            const d = hexes[i];
            const lx = d.dq * dx_dq;
            const ly = d.dr * dy_dr + d.dq * dy_dq;

            // Backend Y is North. Frontend -Z is North.
            matrix.makeTranslation(lx, 0, -ly);
            mesh.setMatrixAt(i, matrix);

            nz1[i * 4] = d.h; nz1[i * 4 + 1] = d.h; nz1[i * 4 + 2] = d.h; nz1[i * 4 + 3] = d.h;
            nz2[i * 4] = d.h; nz2[i * 4 + 1] = d.h; nz2[i * 4 + 2] = d.h; nz2[i * 4 + 3] = 0.0;
        }
        mesh.instanceMatrix.needsUpdate = true;
        mesh.geometry.setAttribute('instanceNZ_1', new THREE.InstancedBufferAttribute(nz1, 4));
        mesh.geometry.setAttribute('instanceNZ_2', new THREE.InstancedBufferAttribute(nz2, 4));

        mesh.geometry.setAttribute('instanceSlope_1', new THREE.InstancedBufferAttribute(new Float32Array(num * 4), 4));
        mesh.geometry.setAttribute('instanceSlope_2', new THREE.InstancedBufferAttribute(new Float32Array(num * 4), 4));
        mesh.geometry.setAttribute('instanceBorder', new THREE.InstancedBufferAttribute(new Float32Array(num), 1));

        mesh.frustumCulled = true;
        return mesh;
    }

    setupMaterialShader(material) {
        material.onBeforeCompile = (shader) => {
            material.userData.shader = shader;
            shader.uniforms.uHeightFactor = { value: 0.0 };
            shader.uniforms.uFloorOffset = { value: this.floorState.value };
            shader.uniforms.uTileSize = { value: SECTOR_WIDTH_METERS };

            // UV Padding correction (64px padding on 4096px base)
            const pad = 64.0;
            const size = 4096.0;
            const total = size + pad * 2;
            shader.uniforms.uUvScale = { value: size / total };
            shader.uniforms.uUvOffset = { value: pad / total };

            shader.vertexShader = shader.vertexShader.replace('#include <common>', `
                #include <common>
                uniform float uHeightFactor;
                uniform float uFloorOffset;
                attribute vec4 instanceNZ_1;
                attribute vec4 instanceNZ_2;
                varying vec3 vLocalPos;
                varying float vGrad;
            `).replace('#include <begin_vertex>', `
                #include <begin_vertex>
                float myH = instanceNZ_2.z - uFloorOffset;
                float animH = myH * uHeightFactor;
                if (position.y > 0.0) transformed.y = animH;
                else transformed.y = 0.0;

                #ifdef USE_INSTANCING
                    vLocalPos = (instanceMatrix * vec4(transformed, 1.0)).xyz;
                #else
                    vLocalPos = transformed;
                #endif
                vGrad = (position.y > 0.5) ? 1.0 : 0.0;
            `);

            shader.fragmentShader = shader.fragmentShader.replace('#include <common>', `
                #include <common>
                uniform float uTileSize;
                uniform float uUvScale;
                uniform float uUvOffset;
                varying vec3 vLocalPos;
            `).replace('#include <map_fragment>', `
                float u = (vLocalPos.x / uTileSize) + 0.5;
                float v = (-vLocalPos.z / uTileSize) + 0.5;
                
                u = clamp(u * uUvScale + uUvOffset, 0.0, 1.0);
                v = clamp(v * uUvScale + uUvOffset, 0.0, 1.0);
                diffuseColor = texture2D(map, vec2(u, v));
            `);
        };
    }

    updateGlobalStats(stats) {
        if (!stats) return;
        this.globalStats.min = Math.min(this.globalStats.min, stats.min);
        this.globalStats.max = Math.max(this.globalStats.max, stats.max);
        this.globalStats.avgSum += stats.avg;
        this.globalStats.baseSum += stats.base;
        this.globalStats.count++;
    }

    updateRenderStats(now) {
        if (now - this.statsUpdateState.lastUpdate < 500) return;
        this.statsUpdateState.lastUpdate = now;
        let count = 0;
        for (const t of this.tiles.values()) if (t.mesh && t.mesh.visible) count += t.mesh.count;
        const countEl = document.getElementById('hex-count');
        if (countEl) countEl.textContent = count.toLocaleString() + " VISIBLE";
    }

    // --- CORE LOOP ---

    updateLOD() {
        if (!this.manifest) return;

        const target = this.controls.target;
        const distSqLimit = this.renderSettings.renderDistance ** 2;

        // 1. Sort Manifest by Distance to Camera
        const sortedManifest = this.manifest.tiles.map(t => {
            const lx = t.x - this.worldOrigin.x;
            const lz = -(t.y - this.worldOrigin.y);
            const dx = target.x - lx;
            const dz = target.z - lz;
            return { ...t, d2: dx * dx + dz * dz, lx, lz };
        }).sort((a, b) => a.d2 - b.d2);

        // 2. Identify Tasks
        for (const t of sortedManifest) {
            const key = `${t.q}_${t.r}`;
            const tile = this.tiles.get(key);
            const dist = Math.sqrt(t.d2);

            if (t.d2 > distSqLimit) {
                if (tile) this.unloadTile(key); // Out of range
                continue;
            }

            // Determine Desired Geometry LOD
            // 0=High(Close), 1=Mid, 2=Low, 3=Lowest(Far)
            let desiredGeoLOD = 3;
            if (dist < this.geoThresholds[0]) desiredGeoLOD = 0;
            else if (dist < this.geoThresholds[1]) desiredGeoLOD = 1;
            else if (dist < this.geoThresholds[2]) desiredGeoLOD = 2;

            // Determine Desired Texture LOD
            const desiredTexFull = (dist < this.texThreshold);

            if (!tile) {
                // Not loaded? Queue it.
                if (!this.loadQueue.find(q => q.key === key)) {
                    this.loadQueue.push({ key, t, desiredGeoLOD, desiredTexFull });
                }
            } else {
                // Already loaded. Check updates.
                // Geo Swap?
                if (tile.currentGeoLOD !== desiredGeoLOD && !tile.isTransitioning) {
                    this.swapGeometry(tile, desiredGeoLOD);
                }
                // Texture Upgrade?
                if (desiredTexFull && !tile.isFullTex && !tile.loadingTex) {
                    this.upgradeTexture(tile);
                }
            }
        }

        this.processQueue();
        this.checkInitialLoad(sortedManifest);
    }

    checkInitialLoad(sorted) {
        if (this.loaderHidden) return;
        // Are the closest 4 tiles visual?
        let operational = 0;
        for (let i = 0; i < Math.min(4, sorted.length); i++) {
            const t = sorted[i];
            const tile = this.tiles.get(`${t.q}_${t.r}`);
            if (tile && tile.mesh) operational++;
        }
        if (operational >= Math.min(4, sorted.length)) this.hideLoader();
    }

    processQueue() {
        if (this.isProcessingQueue || this.loadQueue.length === 0) return;

        // Ensure queue is limited and we pick sorted
        // The queue might have mixed push orders if camera jumps. 
        // Ideally we'd sort, but shifting is fast. The main loop fills it sorted.
        if (this.loadQueue.length > 50) this.loadQueue = this.loadQueue.slice(0, 50);

        const task = this.loadQueue.shift();
        this.isProcessingQueue = true;

        this.loadNewTile(task.t, task.desiredGeoLOD, task.desiredTexFull).then(() => {
            this.isProcessingQueue = false;
            if (this.loadQueue.length > 0) requestAnimationFrame(() => this.processQueue());
        });
    }

    async loadNewTile(t, geoLOD, loadFullTexNow) {
        const key = `${t.q}_${t.r}`;
        if (this.tiles.has(key)) return;

        try {
            // 1. Initial Texture (Low Res for speed)
            const lowTexUrl = `aerial_tiles/low/sector_${t.q}_${t.r}.webp`;
            const texLoader = new THREE.TextureLoader();
            const texture = await texLoader.loadAsync(lowTexUrl);
            texture.colorSpace = THREE.SRGBColorSpace;
            texture.flipY = true;

            const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
            this.setupMaterialShader(material);
            this.materialsToUpdate.push(material);

            // 2. Binary Data
            const binUrl = `tiles_bin/sector_${t.q}_${t.r}.bin`;
            const buffer = await (await fetch(binUrl)).arrayBuffer();
            const parsed = this.parseBinaryV3(buffer);

            // 3. Create Geometry
            const flatMesh = new THREE.Mesh(this.flatGeometry, material);
            flatMesh.position.set(t.lx, 0, t.lz);
            this.scene.add(flatMesh);

            const mesh = this.createInstancedMeshV3(parsed.layers, geoLOD, material);
            mesh.position.set(t.lx, 0, t.lz);
            this.scene.add(mesh);

            const half = TILE_WIDTH_WORLD / 2;
            const bounds = new THREE.Box3(
                new THREE.Vector3(t.lx - half, TILE_BOUNDS_MIN_Y, t.lz - half),
                new THREE.Vector3(t.lx + half, TILE_BOUNDS_MAX_Y, t.lz + half)
            );

            const tileObj = {
                q: t.q, r: t.r, lx: t.lx, lz: t.lz,
                mesh, flatMesh, material, bounds,
                hexDataLayers: parsed.layers,
                stats: parsed.stats,
                currentGeoLOD: geoLOD,
                isFullTex: false,
                loadingTex: false,
                isTransitioning: false
            };
            this.tiles.set(key, tileObj);
            this.updateGlobalStats(parsed.stats);

            // 5. Upgrade Texture Immediately if close?
            if (loadFullTexNow) this.upgradeTexture(tileObj);

        } catch (e) { console.error("Tile Load Error", key, e); }
    }

    async upgradeTexture(tile) {
        tile.loadingTex = true;
        const url = `aerial_tiles/full/sector_${tile.q}_${tile.r}.webp`;
        try {
            const texLoader = new THREE.TextureLoader();
            const fullTex = await texLoader.loadAsync(url);
            fullTex.colorSpace = THREE.SRGBColorSpace;
            fullTex.flipY = true;

            tile.material.map = fullTex;
            tile.material.needsUpdate = true;
            tile.isFullTex = true;
        } catch (e) { }
        tile.loadingTex = false;
    }

    swapGeometry(tile, newLOD) {
        tile.isTransitioning = true;
        if (tile.mesh) {
            this.scene.remove(tile.mesh);
            tile.mesh.geometry.dispose();
        }

        const mesh = this.createInstancedMeshV3(tile.hexDataLayers, newLOD, tile.material);
        mesh.position.set(tile.lx, 0, tile.lz);

        // Ensure visibility state matches current animation frame
        const angle = this.controls.getPolarAngle() * 180 / Math.PI;
        mesh.visible = (angle >= 5.5);

        this.scene.add(mesh);
        tile.mesh = mesh;
        tile.currentGeoLOD = newLOD;
        tile.isTransitioning = false;
    }

    unloadTile(key) {
        const tile = this.tiles.get(key);
        if (!tile) return;

        this.scene.remove(tile.mesh);
        this.scene.remove(tile.flatMesh);
        tile.mesh.geometry.dispose();
        tile.flatMesh.geometry.dispose();
        tile.material.map.dispose();
        tile.material.dispose();

        this.tiles.delete(key);
    }

    hideLoader() {
        if (this.loaderHidden) return;
        const loader = document.getElementById('loader');
        if (loader) { loader.style.display = 'none'; this.loaderHidden = true; }
    }

    maintainCameraAltitudeDuringAnimation(h) {
        const target = this.controls.target;
        const wx = target.x + this.worldOrigin.x;
        const wy = this.worldOrigin.y - target.z;

        const q_r = worldToSectorID(wx, wy);
        const key = `${q_r.Q}_${q_r.R}`;
        const tile = this.tiles.get(key);

        // Update Readouts
        const secEl = document.getElementById('sector-val');
        if (secEl) secEl.textContent = `${q_r.Q}, ${q_r.R}`;

        const worldEl = document.getElementById('world-val');
        if (worldEl) worldEl.textContent = `${wx.toFixed(0)}, ${wy.toFixed(0)}`;

        // Approximate Hex (Axial)
        const h_size = UNIT_HEX_WIDTH_METERS;
        const aq = wx / (Math.sqrt(3) / 2 * h_size);
        const ar = (wy - (aq * 0.5 * h_size)) / h_size;
        const hexEl = document.getElementById('hex-val');
        if (hexEl) hexEl.textContent = `${Math.round(aq)}, ${Math.round(ar)}`;

        if (tile && tile.stats) {
            const animatedH = (tile.stats.max - this.floorState.value) * h;
            const minCamY = animatedH + 50.0;
            if (this.camera.position.y < minCamY) this.camera.position.y = minCamY;
            const thEl = document.getElementById('tile-height');
            if (thEl) thEl.textContent = `${animatedH.toFixed(1)}m`;
        }
        const chEl = document.getElementById('camera-height');
        if (chEl) chEl.textContent = `${this.camera.position.y.toFixed(0)}m`;
    }

    updateFloorState(h) {
        if (LOCK_FLOOR_ON_RISE && h > FLOOR_LOCK_THRESHOLD && !this.floorState.locked) {
            this.floorState.value = this.pickFloorValue();
            this.floorState.locked = true;
            this.updateFloorUniforms();
        } else if (!LOCK_FLOOR_ON_RISE) {
            this.floorState.value = this.pickFloorValue();
            this.updateFloorUniforms();
        }
    }

    pickFloorValue() {
        const inView = this.getTilesInView();
        const validTiles = inView.length ? inView : Array.from(this.tiles.values());
        let min = Infinity;
        for (const t of validTiles) if (t.stats && t.stats.min < min) min = t.stats.min;
        return Number.isFinite(min) ? min : 0;
    }

    getTilesInView() {
        this.camera.updateMatrixWorld();
        this.projScreenMatrix.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
        this.frustum.setFromProjectionMatrix(this.projScreenMatrix);
        return Array.from(this.tiles.values()).filter(t => this.frustum.intersectsBox(t.bounds));
    }

    updateFloorUniforms() {
        for (const m of this.materialsToUpdate) {
            if (m.userData.shader) m.userData.shader.uniforms.uFloorOffset.value = this.floorState.value;
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        const now = performance.now();
        this.updateRenderStats(now);

        const angle = this.controls.getPolarAngle() * 180 / Math.PI;
        const linear = Math.min(1, Math.max(0, (angle - 6.0) / (25.0 - 6.0)));
        const h = linear;
        const flat = angle < 5.5;

        this.updateFloorState(h);
        this.maintainCameraAltitudeDuringAnimation(h);

        for (const t of this.tiles.values()) {
            if (t.flatMesh) t.flatMesh.visible = flat;
            if (t.mesh) t.mesh.visible = !flat;
        }

        for (const m of this.materialsToUpdate) {
            if (m.userData.shader) m.userData.shader.uniforms.uHeightFactor.value = h;
        }

        this.updateLOD();
        this.renderer.render(this.scene, this.camera);
        this.floorState.lastFactor = h;
    }
}

new PistonViewer();
