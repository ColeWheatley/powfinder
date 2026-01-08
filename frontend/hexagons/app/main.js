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
        console.log("Initializing PistonViewer (HexRect V3)...");
        this.container = document.getElementById('canvas-container');
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

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

        this.scene.add(new THREE.AmbientLight(0xffffff, 0.4));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(500, 2000, 500);
        this.scene.add(dirLight);

        // DEBUG: Setting thresholds very high (20km+) to prevent any LOD-based culling or 
        // downsampling while we diagnose why the pistons aren't raising.
        this.lodThresholds = [20000, 20001, 20002, 20003];
        window.addEventListener('resize', this.onResize.bind(this));

        // Shared Geometry
        const side = UNIT_HEX_WIDTH_METERS / Math.sqrt(3);
        this.hexGeometry = this.createHexGeometry(side);
        this.flatGeometry = new THREE.PlaneGeometry(TILE_WIDTH_WORLD, TILE_HEIGHT_WORLD);
        this.flatGeometry.rotateX(-Math.PI / 2);

        this.tiles = [];
        this.manifest = null;
        this.loadingTiles = new Set();
        this.essentialTilesLoaded = 0;
        this.essentialTilesTarget = 0;
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
        this.tilesLoadedCount = 0;
        this.hexCountEl = document.getElementById('hex-count');
        this.tileHeightEl = document.getElementById('tile-height');
        this.cameraHeightEl = document.getElementById('camera-height');
        this.lastFrameTime = performance.now();
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
        ['geo-lod0', 'geo-lod1', 'geo-lod2', 'geo-lod3'].forEach((id, i) => {
            const s = document.getElementById(`${id}-slider`);
            if (s) s.addEventListener('change', () => this.lodThresholds[i] = parseInt(s.value));
        });
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
            this.camera.position.set(centerX, 2000, centerZ);
            this.controls.target.set(centerX, 0, centerZ);
            this.controls.update();

            this.essentialTilesTarget = 1;
            this.updateLOD();
        } catch (e) { this.log("Manifest error: " + e.message, "error"); }
    }

    async loadSingleTile(tileDef) {
        const { x, y, q, r } = tileDef;
        const tileKey = `${q}_${r}`;
        if (this.loadingTiles.has(tileKey)) return;
        this.loadingTiles.add(tileKey);

        const localX = x - this.worldOrigin.x;
        const localZ = -(y - this.worldOrigin.y);

        try {
            const cb = Date.now();
            const fullTexUrl = `aerial_tiles/full/sector_${q}_${r}.webp?v=${cb}`;
            const binUrl = `tiles_bin/sector_${q}_${r}.bin?v=${cb}`;

            const texLoader = new THREE.TextureLoader();
            const texture = await texLoader.loadAsync(fullTexUrl);
            texture.colorSpace = THREE.SRGBColorSpace;
            texture.flipY = true; // FIXED: Use natural GL orientation

            const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
            this.setupMaterialShader(material);
            this.materialsToUpdate.push(material);

            const buffer = await (await fetch(binUrl)).arrayBuffer();
            const parsed = this.parseBinaryV3(buffer);

            const flatMesh = new THREE.Mesh(this.flatGeometry, material);
            flatMesh.position.set(localX, 0, localZ);
            this.scene.add(flatMesh);

            const mesh = this.createInstancedMeshV3(parsed.hexes, material);
            mesh.position.set(localX, 0, localZ);
            this.scene.add(mesh);

            const half = TILE_WIDTH_WORLD / 2;
            const bounds = new THREE.Box3(
                new THREE.Vector3(localX - half, TILE_BOUNDS_MIN_Y, localZ - half),
                new THREE.Vector3(localX + half, TILE_BOUNDS_MAX_Y, localZ + half)
            );

            const tileObj = {
                q, r, x, y, mesh, flatMesh, material, bounds,
                stats: parsed.stats, center: { x: localX, z: localZ }
            };
            this.tiles.push(tileObj);
            this.updateGlobalStats(parsed.stats);
            this.tilesLoadedCount++;
            this.hideLoader();
        } catch (e) {
            console.error(e);
            this.log(`Error loading tile ${q},${r}`, "error");
        } finally {
            this.loadingTiles.delete(tileKey);
        }
    }

    hideLoader() {
        if (this.loaderHidden) return;
        const loader = document.getElementById('loader');
        if (loader) { loader.style.display = 'none'; this.loaderHidden = true; }
    }

    parseBinaryV3(buffer) {
        const view = new DataView(buffer);
        const minZ = view.getFloat32(12, true);
        const maxZ = view.getFloat32(16, true);
        const scale = view.getFloat32(20, true);
        let offset = 32;
        const layers = [];
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
        return { hexes: layers[3], stats: { min: minZ, max: maxZ, avg: (minZ + maxZ) / 2, base: minZ } };
    }

    createInstancedMeshV3(hexes, material) {
        const num = hexes.length;
        const geometry = this.hexGeometry.clone();
        const mesh = new THREE.InstancedMesh(geometry, material, num);
        const matrix = new THREE.Matrix4();
        const nz1 = new Float32Array(num * 4);
        const nz2 = new Float32Array(num * 4);

        const h = UNIT_HEX_WIDTH_METERS;
        const dx_dq = (Math.sqrt(3) / 2) * h;
        const dy_dq = 0.5 * h;
        const dy_dr = h;

        for (let i = 0; i < num; i++) {
            const d = hexes[i];
            const lx = d.dq * dx_dq;
            const ly = d.dr * dy_dr + d.dq * dy_dq;
            matrix.makeTranslation(lx, 0, -ly);
            mesh.setMatrixAt(i, matrix);

            nz1[i * 4] = d.h; nz1[i * 4 + 1] = d.h; nz1[i * 4 + 2] = d.h; nz1[i * 4 + 3] = d.h;
            nz2[i * 4] = d.h; nz2[i * 4 + 1] = d.h; nz2[i * 4 + 2] = d.h; nz2[i * 4 + 3] = 0.0;
        }
        mesh.instanceMatrix.needsUpdate = true;
        mesh.geometry.setAttribute('instanceNZ_1', new THREE.InstancedBufferAttribute(nz1, 4));
        mesh.geometry.setAttribute('instanceNZ_2', new THREE.InstancedBufferAttribute(nz2, 4));
        // Dummies
        mesh.geometry.setAttribute('instanceSlope_1', new THREE.InstancedBufferAttribute(new Float32Array(num * 4), 4));
        mesh.geometry.setAttribute('instanceSlope_2', new THREE.InstancedBufferAttribute(new Float32Array(num * 4), 4));
        mesh.geometry.setAttribute('instanceBorder', new THREE.InstancedBufferAttribute(new Float32Array(num), 1));
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
        for (const t of this.tiles) if (t.mesh && t.mesh.visible) count += t.mesh.count;
        const countEl = document.getElementById('hex-count');
        if (countEl) countEl.textContent = count.toLocaleString() + " VISIBLE";
    }

    updateLOD() {
        if (!this.manifest) return;
        const target = this.controls.target;
        const distSq = this.renderSettings.renderDistance ** 2;
        for (const t of this.manifest.tiles) {
            const lx = t.x - this.worldOrigin.x;
            const lz = -(t.y - this.worldOrigin.y);
            const d = (target.x - lx) ** 2 + (target.z - lz) ** 2;
            if (d < distSq) {
                if (!this.tiles.some(tile => tile.q === t.q && tile.r === t.r)) {
                    this.loadSingleTile(t);
                }
            }
        }
    }

    maintainCameraAltitudeDuringAnimation(h) {
        const target = this.controls.target;
        const q_r = worldToSectorID(target.x + this.worldOrigin.x, this.worldOrigin.y - target.z);
        const tile = this.tiles.find(t => t.q === q_r.Q && t.r === q_r.R);
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
        const tiles = inView.length ? inView : this.tiles;
        let min = Infinity;
        for (const t of tiles) if (t.stats && t.stats.min < min) min = t.stats.min;
        return Number.isFinite(min) ? min : 0;
    }

    getTilesInView() {
        this.camera.updateMatrixWorld();
        this.projScreenMatrix.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
        this.frustum.setFromProjectionMatrix(this.projScreenMatrix);
        return this.tiles.filter(t => this.frustum.intersectsBox(t.bounds));
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
        const h = Math.max(0.001, linear);
        const flat = angle < 5.5;

        this.updateFloorState(h);
        this.maintainCameraAltitudeDuringAnimation(h);

        for (const t of this.tiles) {
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
