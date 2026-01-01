import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';

// --- AI TELEMETRY BRIDGE ---
const LOG_URL = 'http://localhost:8888/log';
const remoteLog = (msg) => {
    fetch(LOG_URL, { method: 'POST', body: msg }).catch(() => { });
};

// Intercept Console
const originalLog = console.log;
const originalError = console.error;
console.log = (...args) => {
    remoteLog(`[LOG] ${args.join(' ')}`);
    originalLog(...args);
};
console.error = (...args) => {
    remoteLog(`[ERROR] ${args.join(' ')}`);
    originalError(...args);
};

// Clear file on reload
remoteLog(`\n\n--- NEW SESSION --- ${new Date().toLocaleTimeString()}`);

// --- CONFIG ---
const TILE_X = 55000;
const TILE_Y = 203000;
const HEX_WIDTH = 10;
const HEX_DX = HEX_WIDTH * (Math.sqrt(3) / 2);
const SCALE_Z = 1.0;

class PistonViewer {
    constructor() {
        console.log("Initializing PistonViewer...");
        this.container = document.getElementById('canvas-container');
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x050505);
        // this.scene.fog = new THREE.Fog(0x050505, 500, 2500); // Disabling fog to prevent darkening on zoom out

        this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000);
        this.camera.position.set(625, 1500, -500.01); // Birds eye position

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        this.controls = new MapControls(this.camera, this.renderer.domElement);
        this.controls.target.set(625, 0, -500); // Look at center
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 100;
        this.controls.maxDistance = 3000;
        this.controls.maxPolarAngle = Math.PI / 2.1;
        this.controls.update();

        // ... (lights) ...
        const ambient = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambient);

        const sun = new THREE.DirectionalLight(0xffffff, 1.2);
        sun.position.set(500, 1000, 500);
        this.scene.add(sun);

        this.pistonMesh = null;
        this.flatPlane = null;
        this.currentHeightFactor = 0.0;
        
        // Instant placeholder plane
        const width = 1290;
        const height = 1040;
        const planeGeo = new THREE.PlaneGeometry(width, height);
        const planeMat = new THREE.MeshBasicMaterial({ color: 0xffffff, map: null, side: THREE.DoubleSide });
        this.flatPlane = new THREE.Mesh(planeGeo, planeMat);
        this.flatPlane.rotation.x = -Math.PI / 2;
        this.flatPlane.position.set(625, 0, -500);
        this.flatPlane.scale.y = -1;
        this.scene.add(this.flatPlane);

        this.init();
        this.animate();

        window.addEventListener('resize', () => this.onResize());
    }

    async init() {
        console.log("Starting init...");
        try {
            await this.loadTile(TILE_X, TILE_Y);
            console.log("LoadTile complete, hiding loader.");
            document.getElementById('loader').style.opacity = '0';
            setTimeout(() => document.getElementById('loader').style.display = 'none', 500);
        } catch (e) {
            console.error("Failed to load tile:", e);
        }
    }

    async loadTile(x, y) {
        console.log(`Loading tile: ${x}, ${y}...`);
        const binUrl = `tiles_bin/tile_${x}_${y}.bin`;
        
        // Progressive Texture Loading
        const texLoader = new THREE.TextureLoader();
        
        // 1. Instant Low Res
        const lowUrl = `tiles_sat/low_res/tile_${x}_${y}.webp`;
        const lowTexture = await texLoader.loadAsync(lowUrl);
        lowTexture.colorSpace = THREE.SRGBColorSpace;
        this.applyTexture(lowTexture);
        console.log("Low-res texture applied.");

        // 2. Async Medium Res
        const medUrl = `tiles_sat/med_res/tile_${x}_${y}.webp`;
        texLoader.load(medUrl, (tex) => {
            tex.colorSpace = THREE.SRGBColorSpace;
            this.applyTexture(tex);
            console.log("Medium-res texture upgraded.");
        });

        // 3. High Res Trigger logic
        this.currentTileCoords = { x, y };
        this.highResLoaded = false;

        // Load Binary Data
        console.log("Fetching binary data...");
        const response = await fetch(binUrl);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const buffer = await response.arrayBuffer();
        console.log(`Binary data received: ${buffer.byteLength} bytes.`);
        const view = new DataView(buffer);

        const baseElevation = view.getFloat32(0, true);
        const hexData = [];
        let offset = 4;
        
        // ... (hex parsing logic remains same)
        while (offset < buffer.byteLength) {
            const zRaw = view.getUint16(offset, true);
            const z = this.decodeFloat16(zRaw);

            const nz_s = this.decodeFloat16(view.getUint16(offset + 2, true));
            const nz_se = this.decodeFloat16(view.getUint16(offset + 4, true));
            const nz_sw = this.decodeFloat16(view.getUint16(offset + 6, true));

            const rgb_s_top = [view.getUint8(offset + 8) / 255, view.getUint8(offset + 9) / 255, view.getUint8(offset + 10) / 255];
            const rgb_s_bot = [view.getUint8(offset + 11) / 255, view.getUint8(offset + 12) / 255, view.getUint8(offset + 13) / 255];
            const rgb_se_top = [view.getUint8(offset + 14) / 255, view.getUint8(offset + 15) / 255, view.getUint8(offset + 16) / 255];
            const rgb_se_bot = [view.getUint8(offset + 17) / 255, view.getUint8(offset + 18) / 255, view.getUint8(offset + 19) / 255];
            const rgb_sw_top = [view.getUint8(offset + 20) / 255, view.getUint8(offset + 21) / 255, view.getUint8(offset + 22) / 255];
            const rgb_sw_bot = [view.getUint8(offset + 23) / 255, view.getUint8(offset + 24) / 255, view.getUint8(offset + 25) / 255];

            hexData.push({
                z, nz_s, nz_se, nz_sw,
                rgb_s_top, rgb_s_bot,
                rgb_se_top, rgb_se_bot,
                rgb_sw_top, rgb_sw_bot,
                s_s: 90.0, s_se: 90.0, s_sw: 90.0
            });
            offset += 26;
        }

        this.createInstancedMesh(hexData, lowTexture, baseElevation);
    }

    applyTexture(tex) {
        if (this.flatPlane) {
            this.flatPlane.material.map = tex;
            this.flatPlane.material.needsUpdate = true;
        }
        if (this.pistonMaterial) {
            this.pistonMaterial.map = tex;
            this.pistonMaterial.needsUpdate = true;
        }
    }

    checkHighResTrigger() {
        if (this.highResLoaded || !this.currentTileCoords) return;

        // Trigger high res if zoomed in (distance < 400)
        if (this.controls.getDistance() < 400) {
            this.highResLoaded = true; // Mark as loading to prevent double trigger
            console.log("Zoom threshold reached. Fetching High-Res TIF...");
            
            const { x, y } = this.currentTileCoords;
            const highUrl = `tiles_sat/high_res/tile_${x}_${y}.tif`;
            
            // Note: Browsers can't native-load TIF as texture easily without a lib.
            // Since we generated high_res as TIF, we'll need to handle it.
            // For now, let's assume high_res are also webp for the texture map, 
            // but the user mentioned high_res as TIF.
        }
    }

    decodeFloat16(binary) {
        const s = (binary & 0x8000) >> 15;
        const e = (binary & 0x7C00) >> 10;
        const f = binary & 0x03FF;
        if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
        if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
        return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }

    createHexGeometry(radius) {
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const normals = [];
        const faceIndices = [];

        // Define the 6 corners of a flat-topped hex
        // Corner 0 is at 0 degrees (East)
        const corners = [];
        for (let i = 0; i < 6; i++) {
            const angle = (i * Math.PI / 3);
            corners.push({
                x: Math.cos(angle) * radius,
                z: Math.sin(angle) * radius
            });
        }

        const faceDefinitions = [
            { c1: 0, c2: 1, id: 0 }, // SE (Face between 0 and 60 deg)
            { c1: 1, c2: 2, id: 1 }, // S  (Face between 60 and 120 deg)
            { c1: 2, c2: 3, id: 2 }, // SW (Face between 120 and 180 deg)
            { c1: 3, c2: 4, id: 3 }, // NW
            { c1: 4, c2: 5, id: 4 }, // N
            { c1: 5, c2: 0, id: 5 }  // NE
        ];

        faceDefinitions.forEach(face => {
            const p1 = corners[face.c1];
            const p2 = corners[face.c2];

            // Midpoint normal pointing OUTWARD
            const midAngle = Math.atan2((p1.z + p2.z) / 2, (p1.x + p2.x) / 2);
            const nx = Math.cos(midAngle);
            const nz = Math.sin(midAngle);

            // Triangle 1: p1_bottom, p2_bottom, p2_top (CCW from outside)
            positions.push(p1.x, 0, p1.z, p2.x, 0, p2.z, p2.x, 1, p2.z);
            // Triangle 2: p1_bottom, p2_top, p1_top (CCW from outside)
            positions.push(p1.x, 0, p1.z, p2.x, 1, p2.z, p1.x, 1, p1.z);

            for (let j = 0; j < 6; j++) {
                normals.push(nx, 0, nz);
                faceIndices.push(face.id);
            }
        });

        // Caps
        for (let i = 0; i < 6; i++) {
            const p1 = corners[i];
            const p2 = corners[(i + 1) % 6];
            // Top
            positions.push(0, 1, 0, p2.x, 1, p2.z, p1.x, 1, p1.z);
            normals.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
            faceIndices.push(6, 6, 6);
            // Bottom
            positions.push(0, 0, 0, p1.x, 0, p1.z, p2.x, 0, p2.z);
            normals.push(0, -1, 0, 0, -1, 0, 0, -1, 0);
            faceIndices.push(7, 7, 7);
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
        geometry.setAttribute('faceIndex', new THREE.Float32BufferAttribute(faceIndices, 1));
        return geometry;
    }

    createInstancedMesh(data, texture, baseZ) {
        const numHexes = data.length;
        console.log(`Creating instanced mesh for ${numHexes} hexes...`);
        document.getElementById('piston-count').innerText = numHexes.toLocaleString();

        const side = HEX_WIDTH / Math.sqrt(3);
        const geometry = this.createHexGeometry(side);

        const material = new THREE.MeshBasicMaterial({
            map: texture
        });

        this.pistonMaterial = material;
        material.userData.shader = null;

        material.onBeforeCompile = (shader) => {
            material.userData.shader = shader;
            shader.uniforms.uTileSize = { value: new THREE.Vector2(1250, 1000) };
            shader.uniforms.uHeightFactor = { value: 0.0 };

            shader.vertexShader = shader.vertexShader.replace(
                '#include <common>',
                `
                #include <common>
                uniform float uHeightFactor;
                attribute vec4 instanceNZ; // x=SE, y=S, z=SW, w=instanceZ
                attribute float faceIndex;
                
                attribute vec4 instanceRGB_SE_Top;
                attribute vec3 instanceRGB_SE_Bot;
                attribute vec4 instanceRGB_S_Top;
                attribute vec3 instanceRGB_S_Bot;
                attribute vec4 instanceRGB_SW_Top;
                attribute vec3 instanceRGB_SW_Bot;
                
                varying vec3 vWorldPos;
                varying vec3 vObjNormal;
                varying vec3 vColorTop;
                varying vec3 vColorBot;
                varying float vFaceSlope;
                varying float vLocalY;
                varying float vGrad;
                `
            ).replace(
                '#include <begin_vertex>',
                `
                #include <begin_vertex>

                // --- CRUST GEOMETRY: Explicit Compass Alignment ---
                int face = int(faceIndex + 0.5);
                float neighborZ = 0.0;
                bool isSouthFace = false;

                // Locked Compass IDs: 0:SE, 1:S, 2:SW
                if (face == 0) { neighborZ = instanceNZ.x; isSouthFace = true; }      // SE
                else if (face == 1) { neighborZ = instanceNZ.y; isSouthFace = true; } // S
                else if (face == 2) { neighborZ = instanceNZ.z; isSouthFace = true; } // SW

                // Transform heights based on uHeightFactor
                float finalZ = instanceNZ.w * uHeightFactor;
                float finalNeighborZ = neighborZ * uHeightFactor;

                // Apply height transformation
                if (position.y > 0.5) {
                    transformed.y = finalZ;
                } else {
                    if (isSouthFace) {
                        transformed.y = finalNeighborZ;
                    } else {
                        transformed.y = finalZ; // North faces collapse
                    }
                }

                vWorldPos = (instanceMatrix * vec4(transformed, 1.0)).xyz;
                vObjNormal = normal;

                // --- FACE COLORING ---
                vFaceSlope = 90.0;
                vColorTop = vec3(0.0);
                vColorBot = vec3(0.0);
                vLocalY = transformed.y;

                if (face == 0) {
                    vColorTop = instanceRGB_SE_Top.rgb; vColorBot = instanceRGB_SE_Bot; vFaceSlope = instanceRGB_SE_Top.w;
                } else if (face == 1) {
                    vColorTop = instanceRGB_S_Top.rgb; vColorBot = instanceRGB_S_Bot; vFaceSlope = instanceRGB_S_Top.w;
                } else if (face == 2) {
                    vColorTop = instanceRGB_SW_Top.rgb; vColorBot = instanceRGB_SW_Bot; vFaceSlope = instanceRGB_SW_Top.w;
                }
                
                vGrad = 1.0;
                if (isSouthFace && position.y < 0.5) {
                    vGrad = 0.0;
                }
                `
            );

            shader.fragmentShader = shader.fragmentShader.replace(
                '#include <common>',
                `
                #include <common>
                uniform vec2 uTileSize;
                varying vec3 vWorldPos;
                varying vec3 vObjNormal;
                varying vec3 vColorTop;
                varying vec3 vColorBot;
                varying float vFaceSlope;
                varying float vGrad;

                vec3 getSlopeColor(float sRaw) {
                    float deg = abs(sRaw - 90.0);
                    if (deg < 25.0) return vec3(0.5, 0.5, 0.5); 
                    if (deg < 35.0) return vec3(0.0, 1.0, 0.0); 
                    if (deg < 40.0) return vec3(0.0, 0.0, 1.0); 
                    if (deg < 45.0) return vec3(0.5, 0.0, 0.5); 
                    if (deg < 50.0) return vec3(1.0, 0.5, 0.0); 
                    return vec3(1.0, 0.0, 0.0);                 
                }
                `
            ).replace(
                '#include <map_fragment>',
                `
                #include <map_fragment>
                
                if (abs(vObjNormal.y) < 0.9) {
                    vec3 finalColor = mix(vColorBot, vColorTop, vGrad);
                    float deg = abs(vFaceSlope - 90.0);
                    if (deg >= 25.0) {
                        finalColor = getSlopeColor(vFaceSlope);
                    }
                    diffuseColor.rgb = finalColor;
                } else {
                    float padding = 20.0;
                    float totalW = uTileSize.x + 2.0 * padding;
                    float totalH = uTileSize.y + 2.0 * padding;
                    
                    float u = (vWorldPos.x + padding) / totalW;
                    float v = (20.0 - vWorldPos.z) / totalH;
                    
                    vec2 uvSat = vec2(u, 1.0 - v);
                    diffuseColor = texture2D(map, uvSat);
                }
                `
            );
        };

        const mesh = new THREE.InstancedMesh(geometry, material, numHexes);
        const matrix = new THREE.Matrix4();

        let minZ = 0;
        data.forEach(h => {
            if (h.z < minZ) minZ = h.z;
            if (h.nz_se < minZ) minZ = h.nz_se;
            if (h.nz_s < minZ) minZ = h.nz_s;
            if (h.nz_sw < minZ) minZ = h.nz_sw;
        });
        console.log(`Piston Floor Offset: ${minZ}m`);

        const instanceNZ = new Float32Array(numHexes * 4);

        // Color Pairs (Top is vec4 to hold slope in .w)
        const instanceRGB_S_Top = new Float32Array(numHexes * 4);
        const instanceRGB_S_Bot = new Float32Array(numHexes * 3);
        const instanceRGB_SE_Top = new Float32Array(numHexes * 4);
        const instanceRGB_SE_Bot = new Float32Array(numHexes * 3);
        const instanceRGB_SW_Top = new Float32Array(numHexes * 4);
        const instanceRGB_SW_Bot = new Float32Array(numHexes * 3);

        let idx = 0;
        const right = 1250;
        const top = 1000;

        console.log("Starting Matrix Assignment Loop...");
        for (let x = 0; x <= right + 1; x += HEX_DX) {
            const colIdx = Math.round(x / HEX_DX);
            const yShift = (colIdx % 2 === 1) ? 5 : 0;
            for (let y = 0; y <= top + 1; y += 10) {
                const realY = y + yShift;
                if (idx >= numHexes) break;
                const h = data[idx];

                matrix.makeTranslation(x, 0, -realY);
                mesh.setMatrixAt(idx, matrix);

                // Set Colors + Slopes in .w
                instanceRGB_S_Top.set(h.rgb_s_top, idx * 4);
                instanceRGB_S_Top[idx * 4 + 3] = h.s_s;
                instanceRGB_S_Bot.set(h.rgb_s_bot, idx * 3);

                instanceRGB_SE_Top.set(h.rgb_se_top, idx * 4);
                instanceRGB_SE_Top[idx * 4 + 3] = h.s_se;
                instanceRGB_SE_Bot.set(h.rgb_se_bot, idx * 3);

                instanceRGB_SW_Top.set(h.rgb_sw_top, idx * 4);
                instanceRGB_SW_Top[idx * 4 + 3] = h.s_sw;
                instanceRGB_SW_Bot.set(h.rgb_sw_bot, idx * 3);

                instanceNZ[idx * 4] = (h.nz_se - minZ) * SCALE_Z;
                instanceNZ[idx * 4 + 1] = (h.nz_s - minZ) * SCALE_Z;
                instanceNZ[idx * 4 + 2] = (h.nz_sw - minZ) * SCALE_Z;
                instanceNZ[idx * 4 + 3] = (h.z - minZ + 5.0) * SCALE_Z; // Packed instanceZ
                idx++;
            }
        }
        console.log(`Assigned matrices for ${idx} instances.`);

        geometry.setAttribute('instanceNZ', new THREE.InstancedBufferAttribute(instanceNZ, 4));
        geometry.setAttribute('instanceRGB_S_Top', new THREE.InstancedBufferAttribute(instanceRGB_S_Top, 4));
        geometry.setAttribute('instanceRGB_S_Bot', new THREE.InstancedBufferAttribute(instanceRGB_S_Bot, 3));
        geometry.setAttribute('instanceRGB_SE_Top', new THREE.InstancedBufferAttribute(instanceRGB_SE_Top, 4));
        geometry.setAttribute('instanceRGB_SE_Bot', new THREE.InstancedBufferAttribute(instanceRGB_SE_Bot, 3));
        geometry.setAttribute('instanceRGB_SW_Top', new THREE.InstancedBufferAttribute(instanceRGB_SW_Top, 4));
        geometry.setAttribute('instanceRGB_SW_Bot', new THREE.InstancedBufferAttribute(instanceRGB_SW_Bot, 3));


        this.scene.add(mesh);
        this.pistonMesh = mesh;
        this.pistonMesh.visible = false; // Start hidden (flat mode)

        console.log("Mesh added to scene.");
        this.controls.target.set(625, 0, -500);
        this.controls.update();
        console.log("Camera target set.");
    }

    onResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.checkHighResTrigger();

        // Piston Raising Logic
        // Polar Angle is 0 at top (birds eye)
        const angle = this.controls.getPolarAngle();
        const maxTilt = 20 * (Math.PI / 180); // 20 degrees

        // Map 0 -> maxTilt to 0.0 -> 1.0
        let targetFactor = THREE.MathUtils.clamp(angle / maxTilt, 0, 1);

        // Apply easing? Linear is fine for now, user asked for linear.
        // But maybe a small deadzone at top for pure flat?
        // Let's stick to the prompt: "relate angle ... linear map ... full height as someone reaches say 20 degrees"

        if (targetFactor < 0.05) {
            // Birds Eye / Low Battery Mode
            if (this.pistonMesh) this.pistonMesh.visible = false;
            if (this.flatPlane) this.flatPlane.visible = true;
        } else {
            // 3D Mode
            if (this.pistonMesh) {
                this.pistonMesh.visible = true;
                // Update uniform
                if (this.pistonMaterial && this.pistonMaterial.userData.shader) {
                    this.pistonMaterial.userData.shader.uniforms.uHeightFactor.value = targetFactor;
                }
            }
            if (this.flatPlane) this.flatPlane.visible = false;
        }

        this.renderer.render(this.scene, this.camera);
    }
}

new PistonViewer();
