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
        this.controls.maxDistance = 5000; // Increased to allow viewing large maps
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
            const loader = document.getElementById('loader');
            loader.classList.add('hide');
            setTimeout(() => loader.style.display = 'none', 500);
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
        const BYTES_PER_HEX = 29;

        console.log("Parsing V3 Binary (South-Push + Slope)...");

        while (offset < buffer.byteLength) {
            // 1. Geometry (4 x half-float) = 8 bytes
            const zRaw = view.getUint16(offset, true);
            const z = this.decodeFloat16(zRaw);
            const nz_s = this.decodeFloat16(view.getUint16(offset + 2, true));
            const nz_se = this.decodeFloat16(view.getUint16(offset + 4, true));
            const nz_sw = this.decodeFloat16(view.getUint16(offset + 6, true));

            offset += 8;

            // 2. South Attributes (RGBA_Top, RGB_Bot) = 7 bytes
            const rgb_s_top = [view.getUint8(offset) / 255, view.getUint8(offset + 1) / 255, view.getUint8(offset + 2) / 255, view.getUint8(offset + 3)]; // .w is Slope
            const rgb_s_bot = [view.getUint8(offset + 4) / 255, view.getUint8(offset + 5) / 255, view.getUint8(offset + 6) / 255];
            offset += 7;

            // 3. SE Attributes = 7 bytes
            const rgb_se_top = [view.getUint8(offset) / 255, view.getUint8(offset + 1) / 255, view.getUint8(offset + 2) / 255, view.getUint8(offset + 3)];
            const rgb_se_bot = [view.getUint8(offset + 4) / 255, view.getUint8(offset + 5) / 255, view.getUint8(offset + 6) / 255];
            offset += 7;

            // 4. SW Attributes = 7 bytes
            const rgb_sw_top = [view.getUint8(offset) / 255, view.getUint8(offset + 1) / 255, view.getUint8(offset + 2) / 255, view.getUint8(offset + 3)];
            const rgb_sw_bot = [view.getUint8(offset + 4) / 255, view.getUint8(offset + 5) / 255, view.getUint8(offset + 6) / 255];
            offset += 7;

            hexData.push({
                z, nz_s, nz_se, nz_sw,
                rgb_s_top, rgb_s_bot,
                rgb_se_top, rgb_se_bot,
                rgb_sw_top, rgb_sw_bot
            });
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

        // SOUTH-PUSH OPTIMIZATION: Only generate S, SW, SE faces (and Caps)
        const faceDefinitions = [
            { c1: 0, c2: 1, id: 0 }, // SE (Face between 0 and 60 deg)
            { c1: 1, c2: 2, id: 1 }, // S  (Face between 60 and 120 deg)
            { c1: 2, c2: 3, id: 2 }, // SW (Face between 120 and 180 deg)
            // NW, N, NE are REMOVED. Use neighbors.
        ];

        faceDefinitions.forEach(face => {
            const p1 = corners[face.c1];
            const p2 = corners[face.c2];

            // Midpoint normal pointing OUTWARD (South-ish)
            const midAngle = Math.atan2((p1.z + p2.z) / 2, (p1.x + p2.x) / 2);
            const nx = Math.cos(midAngle);
            const nz = Math.sin(midAngle);

            // Triangle 1: p1_bottom, p2_top, p2_bottom
            positions.push(p1.x, 0, p1.z, p2.x, 1, p2.z, p2.x, 0, p2.z);
            // Triangle 2: p1_bottom, p1_top, p2_top
            positions.push(p1.x, 0, p1.z, p1.x, 1, p1.z, p2.x, 1, p2.z);

            for (let j = 0; j < 6; j++) {
                normals.push(nx, 0, nz);
                faceIndices.push(face.id);
            }
        });

        // Caps (Top and Bottom)
        for (let i = 0; i < 6; i++) {
            const p1 = corners[i];
            const p2 = corners[(i + 1) % 6];
            // Top
            positions.push(0, 1, 0, p2.x, 1, p2.z, p1.x, 1, p1.z);
            normals.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
            faceIndices.push(6, 6, 6);
            // Bottom (Optimization: Remove bottom cap if never seen? 
            // Keep for now in case of flying underneath)
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
            map: texture,
            side: THREE.DoubleSide // Crucial for "South Push" walls that might face North
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
                varying float vGrad;
                varying float vIsHidden;
                `
            ).replace(
                '#include <begin_vertex>',
                `
                #include <begin_vertex>
                vIsHidden = 0.0;

                int face = int(faceIndex + 0.5);
                float neighborZ = instanceNZ.w; // Default to self if not a wall
                bool isWall = false;

                // 0:SE, 1:S, 2:SW. w holds instance Top Z.
                float myZ = instanceNZ.w;

                if (face == 0) { neighborZ = instanceNZ.x; isWall = true; } // SE
                else if (face == 1) { neighborZ = instanceNZ.y; isWall = true; } // S
                else if (face == 2) { neighborZ = instanceNZ.z; isWall = true; } // SW

                float animMyZ = myZ * uHeightFactor;
                float animNeighborZ = neighborZ * uHeightFactor;

                // Optimization: Collapse wall if height diff is tiny to save rasterizer
                if (isWall && abs(animMyZ - animNeighborZ) < 0.01) {
                     vIsHidden = 1.0;
                     transformed = vec3(0.0); // Collapse to point
                } else {
                    if (position.y > 0.5) {
                        transformed.y = animMyZ;
                    } else {
                        if (isWall) {
                            transformed.y = animNeighborZ;
                        } else {
                            transformed.y = animMyZ; // Top caps (face 6) stay at Top
                        }
                    }
                }

                vWorldPos = (instanceMatrix * vec4(transformed, 1.0)).xyz;
                vObjNormal = normal;

                // --- COLORS ---
                vFaceSlope = 0.0;
                vColorTop = vec3(0.0);
                vColorBot = vec3(0.0);
                vGrad = 1.0; // Default Top

                if (face == 0) {
                    vColorTop = instanceRGB_SE_Top.rgb; vColorBot = instanceRGB_SE_Bot; vFaceSlope = instanceRGB_SE_Top.w;
                } else if (face == 1) {
                    vColorTop = instanceRGB_S_Top.rgb; vColorBot = instanceRGB_S_Bot; vFaceSlope = instanceRGB_S_Top.w;
                } else if (face == 2) {
                    vColorTop = instanceRGB_SW_Top.rgb; vColorBot = instanceRGB_SW_Bot; vFaceSlope = instanceRGB_SW_Top.w;
                }
                
                if (isWall && position.y < 0.5) {
                    vGrad = 0.0; // Bottom Vertex
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
                varying float vIsHidden;
                `
            ).replace(
                '#include <map_fragment>',
                `
                if (vIsHidden > 0.5) discard;
                #include <map_fragment>
                
                // Check if Normal is vertical (Top Cap)
                if (abs(vObjNormal.y) < 0.9) {
                    // It's a Wall
                    vec3 cinColor = mix(vColorBot, vColorTop, vGrad);
                    
                    // Safety Slope Colors
                    if (vFaceSlope > 30.0) {
                        vec3 safeColor = vec3(0.0, 1.0, 0.0); // 30-35 Green
                        if (vFaceSlope >= 35.0) safeColor = vec3(1.0, 1.0, 0.0); // Yellow
                        if (vFaceSlope >= 40.0) safeColor = vec3(0.0, 0.0, 1.0); // Blue
                        if (vFaceSlope >= 45.0) safeColor = vec3(0.5, 0.0, 0.5); // Purple
                        if (vFaceSlope >= 50.0) safeColor = vec3(1.0, 0.0, 0.0); // Red
                        
                        diffuseColor.rgb = safeColor;
                    } else {
                        // Cinematic Gradient (Tree -> Rock)
                        diffuseColor.rgb = cinColor;
                    }
                    
                    // Lighting Fix for DoubleSide
                    if (!gl_FrontFacing) {
                        // If we see the back, normal should flip for lighting math
                        // (ThreeJS standard materials usually handle this if side=DoubleSide, 
                        // but custom shader tweaks might need care. For MeshBasicMaterial, lighting is irrelevant, 
                        // but if we add lights later, this matters. Currently Basic = Unlit.)
                    }

                } else {
                    // Top Cap: Sample Satellite WebP
                    // UV Calculation - NO PADDING logic now (1:1 mapping)
                    // WorldPos: X (0 to 1250), Z (0 to -1000)
                    // UV: u = X/1250, v = -Z/1000
                    
                    float u = vWorldPos.x / 1250.0;
                    float v = -vWorldPos.z / 1000.0;
                    
                    // Clamp to prevent wraparound artifacts at edges
                    u = clamp(u, 0.001, 0.999);
                    v = clamp(v, 0.001, 0.999);
                    
                    vec4 texColor = texture2D(map, vec2(u, 1.0 - v)); // Flip V for Texture
                    diffuseColor = texColor;
                }
                `
            );
        };

        const mesh = new THREE.InstancedMesh(geometry, material, numHexes);
        const matrix = new THREE.Matrix4();

        let minZ = Infinity;
        let maxZ = -Infinity;
        data.forEach(h => {
            if (h.z < minZ) minZ = h.z;
            if (h.nz_se < minZ) minZ = h.nz_se;
            if (h.nz_s < minZ) minZ = h.nz_s;
            if (h.nz_sw < minZ) minZ = h.nz_sw;

            if (h.z > maxZ) maxZ = h.z;
        });
        console.log(`Piston Range: ${minZ}m to ${maxZ}m`);
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
                instanceRGB_S_Top.set(h.rgb_s_top, idx * 4); // .w already contains slope
                instanceRGB_S_Bot.set(h.rgb_s_bot, idx * 3);

                instanceRGB_SE_Top.set(h.rgb_se_top, idx * 4);
                instanceRGB_SE_Bot.set(h.rgb_se_bot, idx * 3);

                instanceRGB_SW_Top.set(h.rgb_sw_top, idx * 4);
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

        const localMaxHeight = (maxZ - minZ) * SCALE_Z;
        console.log(`Max Local Height from Floor: ${localMaxHeight}`);

        // Position Camera: Center X/Z, Height = Max Peak + 2000m
        this.camera.position.set(625, localMaxHeight + 2000, -500);
        this.controls.target.set(625, 0, -500); // 0 is the floor (minZ)
        this.controls.maxDistance = 20000; // Allow huge zoom out
        this.controls.update();

        console.log(`Camera Set: Pos [625, ${this.camera.position.y}, -500], Target [625, 0, -500]`);
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
