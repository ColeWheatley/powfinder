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
        this.controls.addEventListener('change', () => { this.needsRender = true; });

        this.needsRender = true;
        this.lastLODCamPos = new THREE.Vector3().copy(this.camera.position);

        // Granular LOD Ranges for Stacked Rendering
        this.lodRanges = {
            unitEnd: 1200,
            smallStart: 1000,
            smallEnd: 3500,
            mediumStart: 3000,
            mediumEnd: 8500,
            largeStart: 8000
        };

        // Legacy/Sorting Support
        this.geoThresholds = [1200, 3500, 8500, 25000];

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
        this.upgradeQueue = [];
        this.isProcessingTile = false;
        this.isUpgradingTex = false;

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
        // UNIT END
        const unitEnd = document.getElementById('lod-unit-end');
        if (unitEnd) {
            unitEnd.addEventListener('input', () => {
                this.lodRanges.unitEnd = parseInt(unitEnd.value);
                document.getElementById('lod-unit-end-val').textContent = unitEnd.value;
                this.needsRender = true;
            });
        }

        // SMALL
        const smallStart = document.getElementById('lod-small-start');
        const smallEnd = document.getElementById('lod-small-end');
        if (smallStart && smallEnd) {
            smallStart.addEventListener('input', () => {
                this.lodRanges.smallStart = parseInt(smallStart.value);
                document.getElementById('lod-small-start-val').textContent = smallStart.value + 'm';
                this.needsRender = true;
            });
            smallEnd.addEventListener('input', () => {
                this.lodRanges.smallEnd = parseInt(smallEnd.value);
                document.getElementById('lod-small-end-val').textContent = smallEnd.value + 'm';
                this.needsRender = true;
            });
        }

        // MEDIUM
        const medStart = document.getElementById('lod-medium-start');
        const medEnd = document.getElementById('lod-medium-end');
        if (medStart && medEnd) {
            medStart.addEventListener('input', () => {
                this.lodRanges.mediumStart = parseInt(medStart.value);
                document.getElementById('lod-medium-start-val').textContent = medStart.value + 'm';
                this.needsRender = true;
            });
            medEnd.addEventListener('input', () => {
                this.lodRanges.mediumEnd = parseInt(medEnd.value);
                document.getElementById('lod-medium-end-val').textContent = medEnd.value + 'm';
                this.needsRender = true;
            });
        }

        // LARGE
        const largeStart = document.getElementById('lod-large-start');
        if (largeStart) {
            largeStart.addEventListener('input', () => {
                this.lodRanges.largeStart = parseInt(largeStart.value);
                document.getElementById('lod-large-start-val').textContent = largeStart.value + 'm';
                this.needsRender = true;
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


    }

    createHexGeometry(radius) {
        // 1. CAP GEOMETRY (Top Face Only)
        const capGeo = new THREE.CircleGeometry(radius, 6);
        capGeo.rotateX(-Math.PI / 2); // Lay flat

        // Add dummy aSideId to Cap (required for shared shader)
        const capLen = capGeo.attributes.position.count;
        capGeo.setAttribute('aSideId', new THREE.Float32BufferAttribute(new Float32Array(capLen).fill(0), 1));

        // 2. PARTIAL SKIRT GEOMETRY (SE, S, SW Only)
        // Manual construction to ensure clean Side IDs and no overhead
        // Flat Top: SE(2), S(3), SW(4).
        // Angles:
        // 0: E, 1: SE, 2: SW, 3: W, 4: NW, 5: NE (Standard CircleGeo order??)
        // Let's verify standard ThreeJS Circle/Cyl order:
        // Vert 0: (1, 0, 0) -> East
        // Vert 1: (0.5, 0, 0.866) -> SouthEast (Z+)
        // Vert 2: (-0.5, 0, 0.866) -> SouthWest
        // Vert 3: (-1, 0, 0) -> West
        // Vert 4: (-0.5, 0, -0.866) -> NorthWest
        // Vert 5: (0.5, 0, -0.866) -> NorthEast

        // Seg 0 (Verts 0-1): East Face (E -> SE). This is SE Face? No, average is ESE.
        // Wait, Map: N=0, NE=1, SE=2, S=3, SW=4, NW=5.
        // N is -Z. S is +Z. E is +X.
        // CircleGeo 0 is +X.
        // So Vert 0 is "East".
        // Vert 5 is "NorthEast".
        // Vert 4 is "NorthWest".
        // Vert 3 is "West".
        // Vert 2 is "SouthWest".
        // Vert 1 is "SouthEast".

        // Segments (Counter-Clockwise in Theta, but indices might be different):
        // Face 0: 0 -> 1 (East -> SE). This is SE Face? No, average is ESE.
        // Let's look at the edges required for SE, S, SW neighbors.
        // Neighbor SE (Index 2): Direction (1, -1) -> Angle ~ -30 deg? (North is +90? No).
        // Standard Map: N(0,-1) usually? No, here N is -Z.
        // SE is (+X, +Z).
        // Edge SE is the edge connecting East Vertex and SouthEast Vertex? No.
        // It's the edge perpendicular to the SE direction.
        // SE Direction: (+1, +1) approx.
        // The Edge "facing" SE is the one between E(0) and S(approx).

        // Let's rely on the visual check:
        // We want the "Bottom Right", "Bottom", "Bottom Left" faces on screen.
        // These are Verts 0->1, 1->2, 2->3.
        // 0->1: East to SouthEast. (SE Face)
        // 1->2: SouthEast to SouthWest. (South Face)
        // 2->3: SouthWest to West. (SW Face)

        // This matches our indices 2(SE), 3(S), 4(SW) perfectly if we treat 0 as start.
        // So we build 3 quads connecting:
        // Quad 0 (SE): Top(0,1) -> Bottom(0,1)
        // Quad 1 (S):  Top(1,2) -> Bottom(1,2)
        // Quad 2 (SW): Top(2,3) -> Bottom(2,3)

        const vertices = [];
        const indices = [];
        const sideIDs = [];

        const angles = [
            0,                  // 0: East
            Math.PI / 3,        // 1: SE
            2 * Math.PI / 3,    // 2: SW
            Math.PI             // 3: West
        ];

        let vIdx = 0;
        for (let i = 0; i < 3; i++) {
            const th1 = angles[i];
            const th2 = angles[i + 1];

            const x1 = Math.cos(th1) * radius; const z1 = Math.sin(th1) * radius;
            const x2 = Math.cos(th2) * radius; const z2 = Math.sin(th2) * radius;

            // Top (Y=0), Bottom (Y=-1)
            // 4 Verts per quad to allow distinct attributes if needed,
            // though we could share. Separate is safer for flat shading/normals.

            // BL, BR, TR, TL order for CCW face?
            // Top Edge: (x1,0,z1) -> (x2,0,z2)
            // Bottom Edge: (x1,-1,z1) -> (x2,-1,z2)

            // Push Vertices
            vertices.push(x1, 0, z1);   // 0: Top Left (Start)
            vertices.push(x2, 0, z2);   // 1: Top Right (End)
            vertices.push(x1, -1, z1);  // 2: Btm Left
            vertices.push(x2, -1, z2);  // 3: Btm Right

            // Faces (Standard Two-Triangle Quad)
            // 2, 1, 0
            // 2, 3, 1
            indices.push(vIdx + 2, vIdx + 1, vIdx + 0);
            indices.push(vIdx + 2, vIdx + 3, vIdx + 1);

            // Side ID (0, 1, 2)
            for (let k = 0; k < 4; k++) sideIDs.push(i);

            vIdx += 4;
        }

        const skirtGeo = new THREE.BufferGeometry();
        skirtGeo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        skirtGeo.setAttribute('aSideId', new THREE.Float32BufferAttribute(sideIDs, 1));
        skirtGeo.setIndex(indices);
        skirtGeo.computeVertexNormals(); // Nice to have for lighting

        return { capGeo, skirtGeo };
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

            // PRE-ALLOCATE GEOMETRIES
            const side = UNIT_HEX_WIDTH_METERS / Math.sqrt(3);
            const geos = this.createHexGeometry(side);
            this.capGeometry = geos.capGeo;
            this.skirtGeometry = geos.skirtGeo;

            this.flatGeometry = new THREE.PlaneGeometry(TILE_WIDTH_WORLD, TILE_HEIGHT_WORLD);
            this.flatGeometry.rotateX(-Math.PI / 2);

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
        // Header: HEX4 check
        const sig = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
        if (sig !== 'HEX4') {
            console.error(`Invalid Binary Signature: Expected HEX4, got ${sig}`);
            return { layers: [[], [], [], []], stats: { min: 0, max: 0, avg: 0, base: 0 }, center: { q: 0, r: 0 } };
        }

        const minZ = view.getFloat32(12, true);
        const maxZ = view.getFloat32(16, true);
        const scale = view.getFloat32(20, true);
        const sx = view.getInt32(4, true);
        const sy = view.getInt32(8, true);

        let offset = 32;
        const layers = [];
        const scales = [24.0, 6.0, 3.0, 1.0];

        const minX = sx * SECTOR_WIDTH_METERS;
        const minY = sy * SECTOR_WIDTH_METERS;
        const cenX = minX + SECTOR_WIDTH_METERS * 0.5;
        const cenY = minY + SECTOR_WIDTH_METERS * 0.5;

        for (let l = 0; l < 4; l++) {
            const count = view.getUint32(offset, true);
            offset += 4;
            const layer = [];
            const sc = scales[l];
            const rawC = this.worldToAxialScale(cenX, cenY, sc);
            const lcq = Math.round(rawC.q);
            const lcr = Math.round(rawC.r);

            for (let i = 0; i < count; i++) {
                // HEX4 16-Byte Layout
                const dq = view.getInt8(offset);
                const dr = view.getInt8(offset + 1);
                const hn = view.getUint16(offset + 2, true);

                const d1 = view.getInt16(offset + 4, true);
                const d2 = view.getInt16(offset + 6, true);
                const d3 = view.getInt16(offset + 8, true);

                const s1 = view.getUint8(offset + 10);
                const s2 = view.getUint8(offset + 11);
                const s3 = view.getUint8(offset + 12);

                const nx = view.getUint8(offset + 13);
                const nz = view.getUint8(offset + 14);
                // offset 15 is pad

                offset += 16;

                layer.push({
                    dq, dr,
                    q: lcq + dq, r: lcr + dr,
                    h: minZ + (hn / scale),
                    deltas: [d1, d2, d3],
                    slopes: [s1, s2, s3], // Array of 3
                    norm: [nx, nz]
                });
            }
            layers.push(layer);
        }
        return {
            layers,
            sx, sy,
            stats: { min: minZ, max: maxZ, avg: (minZ + maxZ) / 2, base: minZ },
            center: { q: 0, r: 0 }
        };
    }

    // Updated signature to accept tileX, tileZ
    createInstancedMeshV3(allLayers, lodIndex, material, sx, sy) {
        const layerIdx = 3 - Math.min(3, Math.max(0, lodIndex));
        const hexes = allLayers[layerIdx];
        if (!hexes || hexes.length === 0) return null;

        const scaleTable = [24.0, 6.0, 3.0, 1.0];
        const scale = scaleTable[layerIdx];
        const num = hexes.length;
        const h_eff = UNIT_HEX_WIDTH_METERS * scale;

        const dx = (Math.sqrt(3) / 2) * h_eff;
        const dy = h_eff;
        const dy_q = 0.5 * h_eff;

        const sectorMinX = sx * SECTOR_WIDTH_METERS;
        const sectorMaxY = (sy + 1) * SECTOR_WIDTH_METERS;

        // Clone material...
        const instMat = material.clone();
        if (!instMat.userData) instMat.userData = {};
        instMat.userData.lodIdx = layerIdx;
        instMat.userData.isClone = true;
        this.setupMaterialShader(instMat);

        const capG = this.capGeometry.clone();
        const skirtG = this.skirtGeometry.clone();
        capG.scale(scale, 1, scale);
        skirtG.scale(scale, 1, scale);

        const capMesh = new THREE.InstancedMesh(capG, instMat, num);
        const skirtMesh = new THREE.InstancedMesh(skirtG, instMat, num);
        const matrix = new THREE.Matrix4();

        const nz1 = new Float32Array(num * 4);
        const nz2 = new Float32Array(num * 4);

        const slopesVec = new Float32Array(num * 3);
        const deltasVec = new Float32Array(num * 3);
        const normsVec = new Float32Array(num * 2);

        let activeSkirts = 0;
        for (let i = 0; i < num; i++) {
            const hx = hexes[i];
            // GLOBAL METER POS
            const gx = hx.q * dx;
            const gy = hx.r * dy + hx.q * dy_q;

            // LOCAL POS (Relative to Container Origin)
            // Container X is Center (minX/t.lx) -> lx must be -409..+409
            // Container Z is Center (lzVal/t.lz) -> lz must be -409..+409

            const lx = (gx - sectorMinX) - SECTOR_WIDTH_METERS * 0.5;
            const lz = (sectorMaxY - gy) - SECTOR_WIDTH_METERS * 0.5;

            matrix.makeTranslation(lx, 0, lz);
            capMesh.setMatrixAt(i, matrix);
            skirtMesh.setMatrixAt(i, matrix);

            const hh = hx.h;
            nz1[i * 4] = hh; nz1[i * 4 + 1] = hh; nz1[i * 4 + 2] = hh; nz1[i * 4 + 3] = hh;
            nz2[i * 4] = hh; nz2[i * 4 + 1] = hh; nz2[i * 4 + 2] = hh; nz2[i * 4 + 3] = 0.0;

            slopesVec[i * 3 + 0] = hx.slopes[0];
            slopesVec[i * 3 + 1] = hx.slopes[1];
            slopesVec[i * 3 + 2] = hx.slopes[2];

            deltasVec[i * 3 + 0] = hx.deltas[0];
            deltasVec[i * 3 + 1] = hx.deltas[1];
            deltasVec[i * 3 + 2] = hx.deltas[2];

            normsVec[i * 2 + 0] = hx.norm[0] / 255.0;
            normsVec[i * 2 + 1] = hx.norm[1] / 255.0;

            if (hx.deltas.some(v => v !== 0)) activeSkirts++;
        }

        capMesh.instanceMatrix.needsUpdate = true;
        skirtMesh.instanceMatrix.needsUpdate = true;

        [capMesh, skirtMesh].forEach(m => {
            m.geometry.setAttribute('instanceNZ_1', new THREE.InstancedBufferAttribute(nz1, 4));
            m.geometry.setAttribute('instanceNZ_2', new THREE.InstancedBufferAttribute(nz2, 4));
            m.geometry.setAttribute('instanceSlopes', new THREE.InstancedBufferAttribute(slopesVec, 3));
            m.geometry.setAttribute('instanceDeltas', new THREE.InstancedBufferAttribute(deltasVec, 3));
            m.geometry.setAttribute('instanceNormal', new THREE.InstancedBufferAttribute(normsVec, 2));
        });

        const group = new THREE.Group();
        group.add(capMesh);
        group.add(skirtMesh);
        group.userData.activeSkirts = activeSkirts;
        group.frustumCulled = false;
        return group;
    }

    setupMaterialShader(material) {
        material.onBeforeCompile = (shader) => {
            material.userData.shader = shader; // Correctly targets 'material' (which is 'instMat' when called on clone)
            shader.uniforms.uHeightFactor = { value: 0.0 };
            shader.uniforms.uGradientMode = { value: 1.0 };
            shader.uniforms.uFloorOffset = { value: this.floorState.value };
            shader.uniforms.uTileSize = { value: SECTOR_WIDTH_METERS };
            shader.uniforms.uCameraPos = { value: new THREE.Vector3() };
            shader.uniforms.uLodRadii = { value: new THREE.Vector2(0.0, 100000.0) }; // Min, Max

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
                uniform vec3 uCameraPos;
                uniform vec2 uLodRadii;
                
                attribute vec4 instanceNZ_1;
                attribute vec4 instanceNZ_2;
                
                // NEW: Vec3 for Slopes/Deltas, Vec2 for Normal
                attribute vec3 instanceSlopes;
                attribute vec3 instanceDeltas; 
                attribute vec2 instanceNormal; // (Nx, Nz)
                
                attribute float aSideId;
                
                varying vec3 vLocalPos;
                varying vec3 vWorldPos;
                varying float vSlope;
                varying float vIsTop;
                varying vec3 vMyNormal;
            `).replace('#include <begin_vertex>', `
                #include <begin_vertex>
                float myH = instanceNZ_2.z - uFloorOffset;
                float animH = myH * uHeightFactor;
                
                bool isCap = (normal.y > 0.9);
                vIsTop = isCap ? 1.0 : 0.0;
                
                if (isCap) {
                    // CAP
                    transformed.y = 0.0 + animH; 
                    vSlope = 0.0; // Caps follow texture color usually, or flat slope
                    
                    // Decode Normal from [0, 1] -> [-1, 1]
                    float nx = instanceNormal.x * 2.0 - 1.0; 
                    float nz = instanceNormal.y * 2.0 - 1.0;
                    float ny_sq = 1.0 - nx*nx - nz*nz;
                    float ny = sqrt(max(0.0, ny_sq));
                    
                    vMyNormal = normalize(vec3(nx, ny, nz));
                    
                } else {
                    // SKIRT
                    if (position.y > -0.1) {
                         transformed.y = animH;
                    } else {
                         // Select Delta based on Side ID (0=SE, 1=S, 2=SW)
                         float dVal = (aSideId < 0.5) ? instanceDeltas.x : 
                                      (aSideId < 1.5) ? instanceDeltas.y : instanceDeltas.z;
                         
                         // Fix: Convert Decimeters (Int16) to Meters (Float)
                         dVal *= 0.1;
 
                         transformed.y = animH - (dVal * uHeightFactor);
                    }
                    
                    // Pick Slope for Gradient
                    float sVal = (aSideId < 0.5) ? instanceSlopes.x : 
                                 (aSideId < 1.5) ? instanceSlopes.y : instanceSlopes.z;
                    vSlope = sVal;
                    
                    vMyNormal = normal; // Skirt flat normal
                }

                #ifdef USE_INSTANCING
                    vLocalPos = (instanceMatrix * vec4(transformed, 1.0)).xyz;
                    vWorldPos = (modelMatrix * instanceMatrix * vec4(transformed, 1.0)).xyz;
                #else
                    vLocalPos = transformed;
                    vWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
                #endif

                float dist = distance(vWorldPos, uCameraPos);
                if (dist < uLodRadii.x || dist > uLodRadii.y) {
                    transformed = vec3(0.0);
                }
            `);

            shader.fragmentShader = shader.fragmentShader.replace('#include <common>', `
                #include <common>
                uniform float uTileSize;
                uniform float uUvScale;
                uniform float uUvOffset;
                uniform float uGradientMode;
                uniform vec3 uCameraPos;
                uniform vec2 uLodRadii;
                varying vec3 vLocalPos;
                varying vec3 vWorldPos;
                varying float vSlope;
                varying float vIsTop;

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

                // --- LIGHTING (Simplified for Performance) ---
                float lighting = 1.0;

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

        let capCount = 0;
        let skirtCount = 0;

        for (const t of this.tiles.values()) {
            if (t.mesh && t.mesh.isGroup) {
                // Caps are always first child, skirts second
                const capMesh = t.mesh.children[0];
                const skirtMesh = t.mesh.children[1];

                if (capMesh && capMesh.visible) capCount += capMesh.count;
                if (skirtMesh && skirtMesh.visible) skirtCount += (t.mesh.userData.activeSkirts || 0);
            }
        }

        const countEl = document.getElementById('hex-count');
        if (countEl) {
            countEl.innerHTML = `
                <span style="color: #00d2ff">${capCount.toLocaleString()} TOPS</span> | 
                <span style="color: #ff7675">${skirtCount.toLocaleString()} SKIRTS</span>
            `;
        }
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

        // Limit updates per frame
        const maxUpdates = 1;
        let updates = 0;

        // Camera Direction for Frustum Weighting
        const camDir = new THREE.Vector3();
        this.camera.getWorldDirection(camDir);
        camDir.y = 0; // Horizontal bias
        camDir.normalize();

        // 2. Identify Tasks with Directional Bias
        for (const t of sortedManifest) {
            const key = `${t.q}_${t.r}`;
            const tile = this.tiles.get(key);

            // Calculate Box Center for Direction Check
            const boxCenter = new THREE.Vector3(
                t.x - this.worldOrigin.x + SECTOR_WIDTH_METERS * 0.5,
                0,
                -(t.y - this.worldOrigin.y) - SECTOR_WIDTH_METERS * 0.5
            );

            // Direction to Tile
            const toTile = new THREE.Vector3().subVectors(boxCenter, camPos);
            toTile.y = 0;
            toTile.normalize();

            // Dot Product: 1.0 = Front, -1.0 = Back
            const dot = camDir.dot(toTile);

            // 1. GEO Frustum Culling (Aggressive)
            // +/- 70 degrees (cos(70) ~= 0.34)
            const isBehindGeo = (dot < 0.34);

            // 2. TEXTURE Frustum Culling (Generous + Buffer)
            // Widened cone: +/- 100 degrees (cos(100) ~= -0.2)
            // Plus: 1000m Proximity Buffer (approx 1 tile width)
            const isEffectivelyFrontTex = (dot > -0.2) || (t.d < 1000);

            if (t.d > distLimit) { // Use actual distance for culling
                if (tile) this.unloadTile(key); // Out of range
                continue;
            }

            // Determine Nominal LOD based on Distance
            let nominalLOD = 0;
            if (t.d < this.geoThresholds[0]) nominalLOD = 3;
            else if (t.d < this.geoThresholds[1]) nominalLOD = 2;
            else if (t.d < this.geoThresholds[2]) nominalLOD = 1;

            // Frustum Override: Force Large (0) if behind (Geo only)
            let targetLOD = nominalLOD;
            if (isBehindGeo) {
                targetLOD = 0;
            }

            // Handle New Loads
            if (!tile && !this.loadingTiles.has(key)) {
                this.loadingTiles.add(key);
                this.loadQueue.push({ t, targetLOD, loadFullTexNow: isEffectivelyFrontTex });
            } else if (tile) {
                // Geo Visibility Update
                if (!tile.isTransitioning) {
                    this.swapGeometry(tile, targetLOD);
                }

                // Texture Upgrade Logic
                if (isEffectivelyFrontTex && !tile.isFullTex && !tile.loadingTex && !tile.queuedForUpgrade) {
                    tile.queuedForUpgrade = true;
                    this.upgradeQueue.push(tile);
                }
            }
        }

        this.processQueues();

        // Queue processing handled by async loaders mostly now, 
        // but we still have an initial load checker.
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

    processQueues() {
        if (this.isProcessingTile || this.isUpgradingTex) return;

        // PRIORITIZE: Load new tiles (low-res) first
        if (this.loadQueue.length > 0) {
            const task = this.loadQueue.shift();
            const key = `${task.t.q}_${task.t.r}`;

            // Hygiene: Skip if already loaded or too far now
            if (this.tiles.has(key) || task.t.d > this.renderSettings.renderDistance + 1000) {
                this.loadingTiles.delete(key);
                return this.processQueues();
            }

            this.isProcessingTile = true;
            this.loadNewTile(task.t, task.targetLOD, task.loadFullTexNow).finally(() => {
                this.isProcessingTile = false;
                this.processQueues();
            });
            return;
        }

        // SECONDARY: Upgrade textures (high-res)
        if (this.upgradeQueue.length > 0) {
            const tile = this.upgradeQueue.shift();
            tile.queuedForUpgrade = false;
            const key = `${tile.q}_${tile.r}`;

            // Hygiene: Skip if tile unloaded or already full
            if (!this.tiles.has(key) || tile.isFullTex) {
                return this.processQueues();
            }

            // Optional: Skip if no longer in "effectively front" zone? 
            // For now, let's just do it if it's still in the tiles map.

            this.isUpgradingTex = true;
            this.upgradeTexture(tile).finally(() => {
                this.isUpgradingTex = false;
                this.processQueues();
            });
        }
    }

    async loadNewTile(t, geoLOD, loadFullTexNow) {
        const key = `${t.q}_${t.r}`;
        if (this.tiles.has(key)) return;

        try {
            // 1. Initial Texture
            const lowTexUrl = `aerial_tiles/low/sector_${t.q}_${t.r}.webp`;
            const texLoader = new THREE.TextureLoader();
            const texture = await texLoader.loadAsync(lowTexUrl);
            texture.colorSpace = THREE.SRGBColorSpace;
            texture.flipY = true;

            const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
            const angle = this.controls.getPolarAngle() * 180 / Math.PI;
            const isVis = (angle >= 5.5);

            const flatMesh = new THREE.Mesh(this.flatGeometry, material);
            flatMesh.position.set(t.lx, 0, t.lz);
            flatMesh.visible = !isVis; // Hide flat if tilted
            this.scene.add(flatMesh);

            const containerGroup = new THREE.Group();
            containerGroup.position.set(t.lx, 0, t.lz);

            const activeMaterials = [];
            // Register base material too so flatMesh rises
            this.materialsToUpdate.push(material);

            this.setupMaterialShader(material);

            // 2. Binary Data
            const binUrl = `tiles_bin/sector_${t.q}_${t.r}.bin?v=6`; // CACHE BUST v6
            const buffer = await (await fetch(binUrl)).arrayBuffer();
            const parsed = this.parseBinaryV3(buffer);



            // Load ALL 4 Scales (3=Unit, 2=Small, 1=Med, 0=Large)
            [0, 1, 2, 3].forEach(level => {
                const meshGroup = this.createInstancedMeshV3(parsed.layers, level, material, parsed.sx, parsed.sy);
                if (meshGroup) {
                    containerGroup.add(meshGroup);
                    // Extract the cloned material from this group and register it for updates
                    if (meshGroup.children.length > 0) {
                        const m = meshGroup.children[0].material;
                        this.materialsToUpdate.push(m);
                        activeMaterials.push(m);
                    }
                }
            });


            containerGroup.visible = isVis; // Parent controls all

            this.scene.add(containerGroup);
            this.needsRender = true; // Visual change

            const half = TILE_WIDTH_WORLD / 2;
            const bounds = new THREE.Box3(
                new THREE.Vector3(t.lx - half, TILE_BOUNDS_MIN_Y, t.lz - half),
                new THREE.Vector3(t.lx + half, TILE_BOUNDS_MAX_Y, t.lz + half)
            );

            const tileObj = {
                q: t.q, r: t.r, lx: t.lx, lz: t.lz,
                mesh: containerGroup,
                flatMesh, material, bounds,
                hexDataLayers: parsed.layers,
                stats: parsed.stats,
                center: parsed.center,
                currentGeoLOD: -1, // Stacked mode
                isFullTex: false,
                loadingTex: false,
                queuedForUpgrade: false, // NEW
                isTransitioning: false,
                clonedMaterials: activeMaterials // Store for cleanup
            };
            this.tiles.set(key, tileObj);
            this.updateGlobalStats(parsed.stats);

            if (loadFullTexNow && !tileObj.isFullTex && !tileObj.loadingTex && !tileObj.queuedForUpgrade) {
                tileObj.queuedForUpgrade = true;
                this.upgradeQueue.push(tileObj);
            }

            this.loadingTiles.delete(key); // Cleanup flight tracker on success

        } catch (e) {
            console.error("Tile Load Error", key, e);
            this.loadingTiles.delete(key); // Allow retry
        }
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
            this.needsRender = true; // Visual change

            // CRITICAL: Also update all cloned materials!
            if (tile.clonedMaterials) {
                tile.clonedMaterials.forEach(m => {
                    m.map = fullTex;
                    m.needsUpdate = true;
                });
            }

            tile.isFullTex = true;
        } catch (e) { }
        tile.loadingTex = false;
    }

    swapGeometry(tile, newLOD) {
        // Stacked Mode: No need to swap geometry!
        // The shader handles LOD via uLodRadii.
        // We just ensure visibility matches camera angle.
        const angle = this.controls.getPolarAngle() * 180 / Math.PI;
        const isVis = (angle >= 5.5);
        if (tile.mesh) tile.mesh.visible = isVis;
    }

    unloadTile(key) {
        const tile = this.tiles.get(key);
        if (!tile) return;

        this.scene.remove(tile.mesh);
        this.scene.remove(tile.flatMesh);

        if (tile.mesh.isGroup) {
            // Remove from update loop
            if (tile.clonedMaterials) {
                tile.clonedMaterials.forEach(m => {
                    const idx = this.materialsToUpdate.indexOf(m);
                    if (idx > -1) this.materialsToUpdate.splice(idx, 1);
                    m.dispose();
                });
            }
        }

        // Deep Cleanup of Stacked Group
        tile.mesh.traverse(obj => {
            if (obj.isMesh) {
                if (obj.geometry) obj.geometry.dispose();
                if (obj.material && obj.material.userData && obj.material.userData.isClone) {
                    obj.material.dispose();
                }
            }
        });

        if (tile.flatMesh.geometry) tile.flatMesh.geometry.dispose();
        if (tile.material.map) tile.material.map.dispose();
        tile.material.dispose();

        this.tiles.delete(key);
        this.loadingTiles.delete(key);
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
        const moved = this.controls.update();

        // 1. Check if render is actually needed
        // If damping is active (moved=true) or logic set a flag, proceed.
        if (!moved && !this.needsRender) return;

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
            if (flat) {
                if (t.flatMesh) t.flatMesh.visible = true;
                if (t.mesh) t.mesh.visible = false;
            } else {
                if (t.flatMesh) t.flatMesh.visible = false;
                if (t.mesh) t.mesh.visible = true;
            }
        }

        for (const m of this.materialsToUpdate) {
            if (m.userData.shader) {
                m.userData.shader.uniforms.uHeightFactor.value = h;
                m.userData.shader.uniforms.uGradientMode.value = this.gradientMode;
                if (!m.userData.shader.uniforms.uCameraPos) {
                    m.userData.shader.uniforms.uCameraPos = { value: new THREE.Vector3() };
                }
                const uCam = m.userData.shader.uniforms.uCameraPos;
                if (!uCam.value || !uCam.value.copy) {
                    uCam.value = new THREE.Vector3();
                }
                uCam.value.copy(this.camera.position);

                if (m.userData.lodIdx !== undefined) {
                    const idx = m.userData.lodIdx;
                    let minD = 0.0, maxD = 100000.0;

                    // Use granular ranges
                    if (idx === 3) { // Unit
                        minD = 0.0;
                        maxD = this.lodRanges.unitEnd;
                    } else if (idx === 2) { // Small
                        minD = this.lodRanges.smallStart;
                        maxD = this.lodRanges.smallEnd;
                    } else if (idx === 1) { // Medium
                        minD = this.lodRanges.mediumStart;
                        maxD = this.lodRanges.mediumEnd;
                    } else if (idx === 0) { // Large
                        minD = this.lodRanges.largeStart;
                        maxD = this.renderSettings.renderDistance + 500.0;
                    }

                    if (!m.userData.shader.uniforms.uLodRadii || !m.userData.shader.uniforms.uLodRadii.value || !m.userData.shader.uniforms.uLodRadii.value.set) {
                        m.userData.shader.uniforms.uLodRadii = { value: new THREE.Vector2(0, 100000.0) };
                    }
                    m.userData.shader.uniforms.uLodRadii.value.set(minD, maxD);
                }
            }
        }

        // 2. Decide if we should update LOD (only if camera moved > 50m)
        // FORCE update if loader is visible (to ensure initial check runs)
        const camDist = this.camera.position.distanceTo(this.lastLODCamPos);
        if (camDist > 50 || this.needsLODUpdate || !this.loaderHidden) {
            this.updateLOD();
            if (camDist > 50) this.lastLODCamPos.copy(this.camera.position);
            this.needsLODUpdate = false;
        }

        // Keep the loading pipes moving
        this.processQueues();

        this.renderer.render(this.scene, this.camera);
        this.needsRender = false;
        this.floorState.lastFactor = h;
    }
}


new PistonViewer();
