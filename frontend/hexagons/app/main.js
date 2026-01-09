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
// Default render distance: 20km (configurable via UI slider)
const DEFAULT_RENDER_DISTANCE = 20000;
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
        this.scene.background = new THREE.Color(0xFF00FF); // Debug Pink

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

        this.gradientMode = 1.0;
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

        // Frametime Graph
        this.frametimeCanvas = document.getElementById('frametime-graph');
        this.frametimeCtx = this.frametimeCanvas ? this.frametimeCanvas.getContext('2d') : null;
        this.frametimeBuffer = new Array(640).fill(16.67); // 60fps baseline
        this.frametimeLastTime = performance.now();

        // LOD Pause Toggle
        this.lodPaused = false;
        this.debugRings = { unit: true, small: true, medium: true, large: true };

        this.initDebugConsole();
        this.initMinimizeButton();
        this.initCollapsibleSections();
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

    initCollapsibleSections() {
        document.querySelectorAll('.collapsible-header').forEach(header => {
            header.addEventListener('click', () => {
                const section = header.parentElement;
                section.classList.toggle('collapsed');
            });
        });
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

        // Render Distance
        const rdSlider = document.getElementById('render-distance-slider');
        const rdVal = document.getElementById('render-distance-val');
        if (rdSlider) {
            rdSlider.value = this.renderSettings.renderDistance / 1000; // Convert to km
            if (rdVal) rdVal.textContent = (this.renderSettings.renderDistance / 1000) + "km";
            rdSlider.addEventListener('input', () => {
                this.renderSettings.renderDistance = parseInt(rdSlider.value) * 1000; // Convert back to meters
                if (rdVal) rdVal.textContent = rdSlider.value + "km";
                this.updateFogAndClip();
            });
        }

        // Gradient Toggle
        const terrainBtn = document.getElementById('gradient-terrain');
        const gradientBtn = document.getElementById('gradient-slope');

        if (terrainBtn && gradientBtn) {
            terrainBtn.addEventListener('click', () => {
                this.gradientMode = 0.0;
                terrainBtn.style.background = '#74b9ff';
                terrainBtn.style.color = '#fff';
                gradientBtn.style.background = 'transparent';
                gradientBtn.style.color = '#ccc';
            });

            gradientBtn.addEventListener('click', () => {
                this.gradientMode = 1.0;
                gradientBtn.style.background = '#74b9ff';
                gradientBtn.style.color = '#fff';
                terrainBtn.style.background = 'transparent';
                terrainBtn.style.color = '#ccc';
            });
        }

        // LOD Pause Toggle
        const lodPauseToggle = document.getElementById('lod-pause-toggle');
        if (lodPauseToggle) {
            lodPauseToggle.addEventListener('change', (e) => {
                this.lodPaused = e.target.checked;
                this.log(this.lodPaused ? "LOD Updates PAUSED" : "LOD Updates RESUMED", "info");
            });
        }

        // Debug Ring Toggles
        ['unit', 'small', 'medium', 'large'].forEach(key => {
            const el = document.getElementById(`ring-${key}-toggle`);
            if (el) {
                el.addEventListener('change', (e) => {
                    this.debugRings[key] = e.target.checked;
                });
            }
        });
    }

    createHexGeometry(radius) {
        const geometry = new THREE.CylinderGeometry(radius, radius, 1, 6);
        geometry.rotateY(Math.PI / 6);
        const nonIndexed = geometry.toNonIndexed();

        // Add "IsTop" Attribute
        const pos = nonIndexed.attributes.position;
        const count = pos.count;
        const isTop = new Float32Array(count);

        for (let i = 0; i < count; i++) {
            // Logic: Top Cap has y=0.5 (height=1/2). Side Tops have y=0.5. Side Bottoms have y=-0.5.
            // Wait, Cylinder is centered at 0. Height=1 -> 0.5 to -0.5.
            // We want Texture on Top Cap. Slope on Sides (including Skirts).
            // Top Cap vertices usually have Normal.y = 1.
            // Side Top vertices have Normal.y = 0.

            // Let's rely on Normals from the non-indexed geo?
            // Actually, we can just use Normal.Y > 0.5 check inside the loop here to bake it.
            const ny = nonIndexed.attributes.normal.getY(i);
            isTop[i] = (ny > 0.9) ? 1.0 : 0.0;
        }

        nonIndexed.setAttribute('aIsTop', new THREE.BufferAttribute(isTop, 1));
        return nonIndexed;
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
        if (!this.scene.fog) this.scene.fog = new THREE.Fog(0xFF00FF, fogStart, fogEnd); // Match Bg
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

            /*
            const centerX = (min_x + max_x) / 2 - min_x;
            const centerZ = -((min_y + max_y) / 2 - min_y);
            this.camera.position.set(centerX, 800, centerZ);
            this.controls.target.set(centerX, 0, centerZ);
            */
            // DEBUG: Circular Building Fault Line
            const debugX = 59817 - this.worldOrigin.x;
            const debugZ = -(206664 - this.worldOrigin.y);
            this.camera.position.set(debugX, 208, debugZ + 100); // Offset Z slightly to look at it?
            this.controls.target.set(debugX, 12, debugZ);
            this.controls.update();

            this.essentialTilesTarget = 1;
            this.updateLOD();
        } catch (e) { this.log("Manifest error: " + e.message, "error"); }
    }

    worldToAxialScale(x, y, s) {
        const h = UNIT_HEX_WIDTH_METERS * s;
        const A = (Math.sqrt(3) / 2) * h;
        const q = x / A;
        const r = (y - (q * 0.5 * h)) / h;
        return { q, r };
    }

    parseBinaryV3(buffer) {
        const view = new DataView(buffer);
        const minZ = view.getFloat32(12, true);
        const maxZ = view.getFloat32(16, true);
        const scale = view.getFloat32(20, true);
        const sx = view.getInt32(4, true); // Header stores Sector X
        const sy = view.getInt32(8, true); // Header stores Sector Y

        let offset = 32;
        const layers = []; // [L3(Scale24), L2(Scale6), L1(Scale3), L0(Scale1)]
        const scales = [24.0, 6.0, 3.0, 1.0];

        // Calculate Sector Center in World Meters
        const minX = sx * SECTOR_WIDTH_METERS;
        const minY = sy * SECTOR_WIDTH_METERS;
        const cenX = minX + SECTOR_WIDTH_METERS * 0.5;
        const cenY = minY + SECTOR_WIDTH_METERS * 0.5;

        for (let l = 0; l < 4; l++) {
            const count = view.getUint32(offset, true);
            offset += 4;
            const layer = [];

            // Calculate Layer Center (lcq, lcr) for this scale
            const sc = scales[l];
            const rawC = this.worldToAxialScale(cenX, cenY, sc);
            const lcq = Math.round(rawC.q);
            const lcr = Math.round(rawC.r);

            for (let i = 0; i < count; i++) {
                // dq, dr are RELATIVE to lcq, lcr
                const dq = view.getInt16(offset, true);
                const dr = view.getInt16(offset + 2, true);
                const hn = view.getUint16(offset + 4, true);
                const slope = view.getUint8(offset + 6);

                // Read 6 neighbor deltas
                const n0 = view.getUint8(offset + 7);
                const n1 = view.getUint8(offset + 8);
                const n2 = view.getUint8(offset + 9);
                const n3 = view.getUint8(offset + 10);
                const n4 = view.getUint8(offset + 11);
                const n5 = view.getUint8(offset + 12);

                offset += 13;

                // Store accurate reconstructed global Axial if needed
                layer.push({
                    dq, dr,
                    q: lcq + dq, r: lcr + dr,
                    h: minZ + (hn / scale),
                    s: slope,
                    nbs: [n0, n1, n2, n3, n4, n5]
                });
            }
            layers.push(layer);
        }
        return { layers, stats: { min: minZ, max: maxZ, avg: (minZ + maxZ) / 2, base: minZ }, center: { q: 0, r: 0 } };
    }

    createInstancedMeshV3(allLayers, lodIndex, material) {
        // lodIndex: 0=Fine(1x), 1=3x, 2=6x, 3=24x.
        // allLayers: [24x, 6x, 3x, 1x] (from backend)

        // Map LOD to Layer Index
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

        // Buffers
        const nz1 = new Float32Array(num * 4); // h, h, h, h
        const nz2 = new Float32Array(num * 4); // h, h, h, 0
        const slopes = new Float32Array(num);  // Single float per instance

        // Neighbors
        const nbA = new Uint8Array(num * 4);
        const nbB = new Uint8Array(num * 2);

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

            const hh = d.h;
            nz1[i * 4] = hh; nz1[i * 4 + 1] = hh; nz1[i * 4 + 2] = hh; nz1[i * 4 + 3] = hh;
            nz2[i * 4] = hh; nz2[i * 4 + 1] = hh; nz2[i * 4 + 2] = hh; nz2[i * 4 + 3] = 0.0;
            slopes[i] = d.s;

            nbA[i * 4 + 0] = d.nbs[0]; nbA[i * 4 + 1] = d.nbs[1]; nbA[i * 4 + 2] = d.nbs[2]; nbA[i * 4 + 3] = d.nbs[3];
            nbB[i * 2 + 0] = d.nbs[4]; nbB[i * 2 + 1] = d.nbs[5];
        }
        mesh.instanceMatrix.needsUpdate = true;

        mesh.geometry.setAttribute('instanceNZ_1', new THREE.InstancedBufferAttribute(nz1, 4));
        mesh.geometry.setAttribute('instanceNZ_2', new THREE.InstancedBufferAttribute(nz2, 4));
        mesh.geometry.setAttribute('instanceSlope', new THREE.InstancedBufferAttribute(slopes, 1));

        // Pass Neighbors as Normalized (0-1)
        mesh.geometry.setAttribute('instanceNbA', new THREE.InstancedBufferAttribute(nbA, 4, true));
        mesh.geometry.setAttribute('instanceNbB', new THREE.InstancedBufferAttribute(nbB, 2, true));

        mesh.frustumCulled = true;
        return mesh;
    }

    setupMaterialShader(material) {
        material.onBeforeCompile = (shader) => {
            material.userData.shader = shader;
            shader.uniforms.uHeightFactor = { value: 0.0 };
            shader.uniforms.uGradientMode = { value: 1.0 };
            shader.uniforms.uFloorOffset = { value: this.floorState.value };
            shader.uniforms.uTileSize = { value: SECTOR_WIDTH_METERS };
            shader.uniforms.uCameraPos = { value: new THREE.Vector3() };
            // T0, T1, T2, Tmax
            shader.uniforms.uDebugRadii = { value: new Float32Array([0, 0, 0, 0]) };
            shader.uniforms.uDebugRingsEnabled = { value: new THREE.Vector4(0, 0, 0, 0) };

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
                attribute float instanceSlope;
                
                attribute vec4 instanceNbA; 
                attribute vec2 instanceNbB;
                attribute float aIsTop; // 1.0 = Top Cap, 0.0 = Side/Skirt
                
                varying vec3 vLocalPos; // Relative to Tile (for UVs)
                varying vec3 vWorldPos; // Absolute World (for Rings)
                varying float vSlope;
                varying float vIsTop;
                varying vec3 vMyNormal;
                
                float getDelta(int idx) {
                    if (idx == 0) return instanceNbA.x;
                    if (idx == 1) return instanceNbA.y;
                    if (idx == 2) return instanceNbA.z;
                    if (idx == 3) return instanceNbA.w;
                    if (idx == 4) return instanceNbB.x;
                    if (idx == 5) return instanceNbB.y;
                    return 0.0;
                }
            `).replace('#include <begin_vertex>', `
                #include <begin_vertex>
                float myH = instanceNZ_2.z - uFloorOffset;
                float animH = myH * uHeightFactor;
                
                // Angle increases CCW: S(0) -> E(90) -> N(180).
                // S(3), SE(2), NE(1), N(0).
                // Using Normal for robust direction on non-indexed geometry
                float angle = atan(normal.x, normal.z); 
                float rawIdx = (angle / 6.28318) * 6.0;
                
                float fIdx = mod(3.0 - rawIdx, 6.0);
                if (fIdx < 0.0) fIdx += 6.0;
                int neighborIdx = int(fIdx + 0.5) % 6;
                
                float normD = getDelta(neighborIdx);
                float deltaM = normD * 255.0;
                
                if (position.y > 0.0) {
                    // It's a Top Vertex (includes Cap and Side-Top)
                    transformed.y = animH;
                } else {
                    // Bottom Vertex (Skirt Bottom)
                    if (deltaM < 0.1) {
                        // Tightened Threshold (0.5m -> 0.1m)
                        // If delta < 10cm, collapse skirt.
                        transformed.y = animH; 
                    } else if (deltaM > 254.0) {
                        transformed.y = 0.0; 
                    } else {
                        transformed.y = animH - (deltaM * uHeightFactor);
                    }
                }

                #ifdef USE_INSTANCING
                    vLocalPos = (instanceMatrix * vec4(transformed, 1.0)).xyz;
                    vWorldPos = (modelMatrix * instanceMatrix * vec4(transformed, 1.0)).xyz;
                #else
                    vLocalPos = transformed;
                    vWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
                #endif
                
                vIsTop = aIsTop;
                vSlope = instanceSlope;
                vMyNormal = normal;
            `);

            shader.fragmentShader = shader.fragmentShader.replace('#include <common>', `
                #include <common>
                uniform float uTileSize;
                uniform float uUvScale;
                uniform float uUvOffset;
                uniform float uGradientMode;
                uniform vec3 uCameraPos;
                uniform float uDebugRadii[4];
                uniform vec4 uDebugRingsEnabled;
                varying vec3 vLocalPos;
                varying vec3 vWorldPos;
                varying float vSlope;
                varying float vIsTop;
                varying vec3 vMyNormal;

                vec3 gradientColor(float s) {
                    // Green: 30-35
                    // Yellow: 35-40
                    // Orange: 40-45
                    // Red: 45-55
                    // Violet: > 55
                    
                    if (s < 30.0) return vec3(0.0); // Transparent/Texture?
                    if (s < 35.0) return vec3(0.2, 0.8, 0.2); // Green
                    if (s < 40.0) return vec3(0.9, 0.9, 0.2); // Yellow
                    if (s < 45.0) return vec3(1.0, 0.6, 0.0); // Orange
                    if (s < 55.0) return vec3(0.9, 0.2, 0.2); // Red
                    return vec3(0.6, 0.2, 0.8); // Violet
                }
            `).replace('#include <map_fragment>', `
                float u = (vLocalPos.x / uTileSize) + 0.5;
                float v = (-vLocalPos.z / uTileSize) + 0.5;
                
                // Apply Padding Scale/Offset first
                u = u * uUvScale + uUvOffset;
                v = v * uUvScale + uUvOffset;
                
                // NOW check bounds. This allows us to use the padding area safeely.
                bool outOfBounds = (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0);
                
                // Clamp to prevent texture wrapping artifacts at the very edge
                u = clamp(u, 0.0, 1.0);
                v = clamp(v, 0.0, 1.0);
                vec4 texColor = texture2D(map, vec2(u, v));
                
                // Fallback for huge hexes exceeding padding (Debug Pink)
                if (outOfBounds) texColor = vec4(1.0, 0.0, 1.0, 1.0);

                // --- LIGHTING ---
                // Sun is approx 15 degrees East of South (South is +Z, East is +X)
                // sin(15) = 0.258, cos(15) = 0.966
                vec3 sunDir = normalize(vec3(0.258, 1.0, 0.966));
                float light = dot(vMyNormal, sunDir);
                // Ambient + Diffuse
                float lighting = clamp(light * 0.6 + 0.6, 0.4, 1.0);

                if (vIsTop < 0.5) {
                    // SIDE / SKIRT
                    vec3 baseColor;
                    if (uGradientMode > 0.5 && vSlope >= 30.0) {
                        baseColor = gradientColor(vSlope);
                    } else {
                        // Low Slope / Distant LOD: Use texture but darken it significantly
                        baseColor = texColor.rgb * 0.6; 
                    }
                    diffuseColor = vec4(baseColor * lighting, 1.0);
                } else {
                   // TOP (Texture)
                   diffuseColor = vec4(texColor.rgb * lighting, 1.0);
                }

                // RADIAL DEBUG RINGS
                float dist = distance(vWorldPos, uCameraPos);
                float ringWidth = 80.0;
                
                // Logic Table: 
                // Unit (x): T0
                // Small (y): T0 + T1
                // Medium (z): T1 + T2
                // Large (w): T2 + Tmax
                
                bool shouldDraw = false;
                float targetRadius = 0.0;
                vec3 ringColor = vec3(1.0);
                
                // Unit (Cyan)
                if (uDebugRingsEnabled.x > 0.5 && abs(dist - uDebugRadii[0]) < ringWidth) { 
                    shouldDraw = true; targetRadius = uDebugRadii[0]; ringColor = vec3(0.0, 1.0, 1.0); 
                }
                // Small (Green)
                if (uDebugRingsEnabled.y > 0.5) {
                    if (abs(dist - uDebugRadii[0]) < ringWidth) { shouldDraw = true; targetRadius = uDebugRadii[0]; ringColor = vec3(0.2, 1.0, 0.2); }
                    if (abs(dist - uDebugRadii[1]) < ringWidth) { shouldDraw = true; targetRadius = uDebugRadii[1]; ringColor = vec3(0.2, 1.0, 0.2); }
                }
                // Medium (Yellow)
                if (uDebugRingsEnabled.z > 0.5) {
                    if (abs(dist - uDebugRadii[1]) < ringWidth) { shouldDraw = true; targetRadius = uDebugRadii[1]; ringColor = vec3(1.0, 1.0, 0.2); }
                    if (abs(dist - uDebugRadii[2]) < ringWidth) { shouldDraw = true; targetRadius = uDebugRadii[2]; ringColor = vec3(1.0, 1.0, 0.2); }
                }
                // Large (Red)
                if (uDebugRingsEnabled.w > 0.5) {
                    if (abs(dist - uDebugRadii[2]) < ringWidth) { shouldDraw = true; targetRadius = uDebugRadii[2]; ringColor = vec3(1.0, 0.2, 0.2); }
                    if (abs(dist - uDebugRadii[3]) < ringWidth) { shouldDraw = true; targetRadius = uDebugRadii[3]; ringColor = vec3(1.0, 0.2, 0.2); }
                }
                
                if (shouldDraw) {
                    float norm = abs(dist - targetRadius) / ringWidth;
                    float alpha = mix(0.1, 0.6, smoothstep(0.0, 1.0, norm)); 
                    diffuseColor.rgb = mix(diffuseColor.rgb, ringColor, alpha);
                    if (norm > 0.94) diffuseColor.rgb += ringColor * 0.5; // Boundary glow
                }
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

    updateFrametimeGraph() {
        if (!this.frametimeCtx) return;

        const now = performance.now();
        const frametime = now - this.frametimeLastTime;
        this.frametimeLastTime = now;

        // Update buffer (shift left, add new value on right)
        this.frametimeBuffer.shift();
        this.frametimeBuffer.push(frametime);

        const ctx = this.frametimeCtx;
        const width = this.frametimeCanvas.width;
        const height = this.frametimeCanvas.height;

        // Clear canvas
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(0, 0, width, height);

        // Draw grid lines
        ctx.strokeStyle = '#222';
        ctx.lineWidth = 1;
        // 16.67ms line (60fps)
        const y60 = height - (16.67 / 50) * height;
        ctx.beginPath();
        ctx.moveTo(0, y60);
        ctx.lineTo(width, y60);
        ctx.stroke();
        // 33.33ms line (30fps)
        const y30 = height - (33.33 / 50) * height;
        ctx.beginPath();
        ctx.moveTo(0, y30);
        ctx.lineTo(width, y30);
        ctx.stroke();

        // Draw frametime graph
        ctx.strokeStyle = '#74b9ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < this.frametimeBuffer.length; i++) {
            const ft = Math.min(this.frametimeBuffer[i], 50); // Cap at 50ms for display
            const x = i;
            const y = height - (ft / 50) * height;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw labels
        ctx.fillStyle = '#666';
        ctx.font = '10px monospace';
        ctx.fillText('16.67ms (60fps)', 5, y60 - 3);
        ctx.fillText('33.33ms (30fps)', 5, y30 - 3);
    }

    // --- CORE LOOP ---

    updateLOD() {
        if (!this.manifest || this.lodPaused) return;

        const camPos = this.camera.position;
        const distLimit = this.renderSettings.renderDistance;

        // 1. Sort Manifest by Distance to Camera (Surface Distance)
        const sortedManifest = this.manifest.tiles.map(t => {
            const minX = t.x - this.worldOrigin.x;
            const maxX = minX + SECTOR_WIDTH_METERS;
            // Manifest Y is North (World -Z). 
            // In ThreeJS: Z goes -inf to +inf. 
            // t.y is typically huge positive (UTM).
            // Our World Origin shift: lz = -(t.y - origin.y).
            // So MinZ = -(t.y - origin.y + SECTOR) -> Further negative? 
            // Wait, t.y increases North. So more North = More Negative Z.
            // min_z (Three) = -((t.y + SECTOR) - origin.y)
            // max_z (Three) = -(t.y - origin.y)

            const lzVal = -(t.y - this.worldOrigin.y);

            const box = new THREE.Box3(
                new THREE.Vector3(minX, TILE_BOUNDS_MIN_Y, lzVal - SECTOR_WIDTH_METERS),
                new THREE.Vector3(maxX, TILE_BOUNDS_MAX_Y, lzVal)
            );

            // For logic consistency later
            t.d = box.distanceToPoint(camPos);
            t.lx = minX;
            t.lz = lzVal - SECTOR_WIDTH_METERS; // Approximate for placement? No wait, logic uses lx/lz for placement.
            // Original lx = t.x - origin.x
            // Original lz = -(t.y - origin.y) = This is the "Top Left" corner in Z?
            // Let's keep original lx/lz definitions

            t.lx = minX;
            t.lz = lzVal; // This matches original logic: lz = -(t.y - wy)

            return t;
        }).sort((a, b) => a.d - b.d);

        // 2. Identify Tasks
        for (const t of sortedManifest) {
            const key = `${t.q}_${t.r}`;
            const tile = this.tiles.get(key);
            const dist = t.d;

            if (dist > distLimit) {
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
                center: parsed.center,
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
        const aq = Math.round(wx / (Math.sqrt(3) / 2 * h_size));
        const ar = Math.round((wy - (aq * 0.5 * h_size)) / h_size);

        const hexEl = document.getElementById('hex-val');
        if (hexEl) hexEl.textContent = `${aq}, ${ar}`;

        if (tile && tile.center) {
            // Find specific hex height
            const dq = aq - tile.center.q;
            const dr = ar - tile.center.r;

            let groundH = tile.stats.min; // Fallback
            let found = false;

            // Search Active Layers (start from finest L3 -> index 3)
            // Or just search all? Finest is best.
            for (let l = 3; l >= 0; l--) {
                const layer = tile.hexDataLayers[l];
                if (!layer) continue;
                // Simple linear search (fast enough for 1 hex per frame)
                for (const hx of layer) {
                    if (hx.dq === dq && hx.dr === dr) {
                        groundH = hx.h;
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }

            // If not found (maybe gap?), fall back to average, not MAX.
            if (!found) groundH = tile.stats.avg;

            const animatedH = (groundH - this.floorState.value) * h;
            const minCamY = animatedH + 50.0;

            // Soft constraint: only push if below
            if (this.camera.position.y < minCamY) this.camera.position.y = minCamY;

            const thEl = document.getElementById('tile-height');
            if (thEl) thEl.textContent = `${animatedH.toFixed(1)}m`;
        }
        const chEl = document.getElementById('camera-height');
        if (chEl) chEl.textContent = `${this.camera.position.y.toFixed(0)}m`;
    }

    updateFloorState(h) {
        const currentMin = this.pickFloorValue();

        if (LOCK_FLOOR_ON_RISE && h > FLOOR_LOCK_THRESHOLD) {
            // Logic: Only update if we found a LOWER floor (prevent sinking), but don't raise it (prevent jitter).
            if (!this.floorState.locked || currentMin < this.floorState.value) {
                this.floorState.value = currentMin;
            }
            this.floorState.locked = true;
            this.updateFloorUniforms();
        } else if (!LOCK_FLOOR_ON_RISE) {
            this.floorState.value = currentMin;
            this.updateFloorUniforms();
        } else {
            // Not yet locked (flat mode), just track freely
            this.floorState.value = currentMin;
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
        this.updateFps();
        this.updateFrametimeGraph();

        const angle = this.controls.getPolarAngle() * 180 / Math.PI;
        const linear = Math.min(1, Math.max(0, (angle - 5.5) / (25.0 - 5.5)));
        const h = linear;
        const flat = angle < 5.5;

        this.updateFloorState(h);
        this.maintainCameraAltitudeDuringAnimation(h);

        for (const t of this.tiles.values()) {
            if (t.flatMesh) t.flatMesh.visible = flat;
            if (t.mesh) t.mesh.visible = !flat;
        }

        for (const m of this.materialsToUpdate) {
            if (m.userData.shader) {
                m.userData.shader.uniforms.uHeightFactor.value = h;
                m.userData.shader.uniforms.uGradientMode.value = this.gradientMode;
                m.userData.shader.uniforms.uCameraPos.value.copy(this.camera.position);

                m.userData.shader.uniforms.uDebugRadii.value.set([
                    this.geoThresholds[0],
                    this.geoThresholds[1],
                    this.geoThresholds[2],
                    this.renderSettings.renderDistance
                ]);
                m.userData.shader.uniforms.uDebugRingsEnabled.value.set(
                    this.debugRings.unit ? 1 : 0,
                    this.debugRings.small ? 1 : 0,
                    this.debugRings.medium ? 1 : 0,
                    this.debugRings.large ? 1 : 0
                );
            }
        }

        this.updateLOD();
        this.renderer.render(this.scene, this.camera);
        this.floorState.lastFactor = h;
    }
}

new PistonViewer();
