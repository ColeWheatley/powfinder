import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { TIFFLoader } from 'three/addons/loaders/TIFFLoader.js';

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
        this.controls.maxDistance = 20000;
        this.controls.maxPolarAngle = Math.PI / 2.1;
        this.controls.update();

        const ambient = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambient);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(500, 1000, 500);
        this.scene.add(dirLight);

        window.addEventListener('resize', this.onResize.bind(this));

        // Load Initial Tile
        this.loadTile(TILE_X, TILE_Y);

        this.animate();
    }

    async loadTile(x, y) {
        console.log(`Loading tile: ${x}, ${y}...`);
        const binUrl = `tiles_bin/tile_${x}_${y}.bin`;

        // Progressive Texture Loading
        const texLoader = new THREE.TextureLoader();
        const lowUrl = `tiles_sat/low_res/tile_${x}_${y}.webp`;
        const lowTexture = await texLoader.loadAsync(lowUrl);
        lowTexture.colorSpace = THREE.SRGBColorSpace;

        const medUrl = `tiles_sat/med_res/tile_${x}_${y}.webp`;
        // Don't await med-res, let it load in background
        texLoader.load(medUrl, (tex) => {
            tex.colorSpace = THREE.SRGBColorSpace;
            this.medTexture = tex;
            this.applyTexture(tex);
        });

        this.currentTileCoords = { x, y };
        this.highResLoaded = false;
        this.highResLoading = false;

        console.log("Fetching binary data...");
        const response = await fetch(binUrl);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const buffer = await response.arrayBuffer();
        console.log(`Binary data received: ${buffer.byteLength} bytes.`);
        const view = new DataView(buffer);

        const baseElevation = view.getFloat32(0, true);
        const hexData = [];
        let offset = 4;
        const BYTES_PER_HEX = 14;

        console.log("Parsing V4 Binary (6-Face Heights)... Base Elevation:", baseElevation);

        let minLocalZ = Infinity;
        let maxLocalZ = -Infinity;

        while (offset < buffer.byteLength) {
            // Z (2 bytes) + 6 Neighbors (12 bytes)
            const z = this.decodeFloat16(view.getUint16(offset, true));
            const n_n = this.decodeFloat16(view.getUint16(offset + 2, true));
            const n_ne = this.decodeFloat16(view.getUint16(offset + 4, true));
            const n_se = this.decodeFloat16(view.getUint16(offset + 6, true));
            const n_s = this.decodeFloat16(view.getUint16(offset + 8, true));
            const n_sw = this.decodeFloat16(view.getUint16(offset + 10, true));
            const n_nw = this.decodeFloat16(view.getUint16(offset + 12, true));

            if (z < minLocalZ) minLocalZ = z;
            if (z > maxLocalZ) maxLocalZ = z;

            hexData.push({
                z, n_n, n_ne, n_se, n_s, n_sw, n_nw
            });
            offset += 14;
        }

        console.log(`Parsed ${hexData.length} hexes. Local Range: ${minLocalZ.toFixed(2)} to ${maxLocalZ.toFixed(2)}`);
        this.createInstancedMesh(hexData, lowTexture, baseElevation, minLocalZ, maxLocalZ);
    }

    applyTexture(tex) {
        if (this.pistonMaterial) {
            console.log("Applying texture to material...");
            this.pistonMaterial.map = tex;
            this.pistonMaterial.needsUpdate = true;
        }
    }

    async checkHighResTrigger() {
        if (this.highResLoaded || this.highResLoading || !this.pistonMesh) return;

        const dist = this.camera.position.distanceTo(this.controls.target);
        if (dist < 600) {
            this.highResLoading = true;
            console.log("Zoom threshold reached. Upgrading to High-Res...");

            const { x, y } = this.currentTileCoords;
            const highUrl = `tiles_sat/high_res/tile_${x}_${y}.tif`;

            const tiffLoader = new TIFFLoader();
            tiffLoader.load(highUrl, (tex) => {
                tex.colorSpace = THREE.SRGBColorSpace;
                this.applyTexture(tex);
                this.highResLoaded = true;
                this.highResLoading = false;
                console.log("High-res textures active.");
            }, undefined, (err) => {
                console.error("Error loading high-res TIFF:", err);
                this.highResLoading = false;
            });
        }
    }

    decodeFloat16(binary) {
        const exponent = (binary & 0x7C00) >> 10;
        const fraction = binary & 0x03FF;
        return (binary >> 15 ? -1 : 1) * (
            exponent ?
                (
                    exponent === 0x1F ?
                        (fraction ? NaN : Infinity) :
                        Math.pow(2, exponent - 15) * (1 + fraction / 1024)
                ) :
                6.103515625e-5 * (fraction / 1024)
        );
    }

    createHexGeometry(radius) {
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const normals = [];
        const faceIndices = [];

        // Corners: 0 (East), 60 (SE), 120 (SW), 180 (W), 240 (NW), 300 (NE)
        const corners = [];
        for (let i = 0; i < 6; i++) {
            const angle = (i * Math.PI / 3);
            corners.push({
                x: Math.cos(angle) * radius,
                z: Math.sin(angle) * radius
            });
        }

        const faceDirs = [
            { x: 0.0, z: 1.0 },   // N
            { x: 0.866, z: 0.5 }, // NE
            { x: 0.866, z: -0.5 },// SE
            { x: 0.0, z: -1.0 },  // S
            { x: -0.866, z: -0.5 },// SW
            { x: -0.866, z: 0.5 }, // NW
        ];

        const resolveFaceId = (out) => {
            let best = 0;
            let bestDot = -Infinity;
            for (let i = 0; i < faceDirs.length; i++) {
                const d = faceDirs[i];
                const dot = out.x * d.x + out.z * d.z;
                if (dot > bestDot) {
                    bestDot = dot;
                    best = i;
                }
            }
            return best;
        };

        const pushWall = (p1, p2) => {
            const v0 = { x: p1.x, y: 0, z: p1.z };
            const v1 = { x: p2.x, y: 1, z: p2.z };
            const v2 = { x: p2.x, y: 0, z: p2.z };
            const e1 = [v1.x - v0.x, v1.y - v0.y, v1.z - v0.z];
            const e2 = [v2.x - v0.x, v2.y - v0.y, v2.z - v0.z];
            const n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            const out = { x: p1.x + p2.x, z: p1.z + p2.z };
            const faceId = resolveFaceId(out);
            const flip = (n[0] * out.x + n[2] * out.z) < 0;

            if (!flip) {
                positions.push(p1.x, 0, p1.z, p2.x, 1, p2.z, p2.x, 0, p2.z);
                positions.push(p1.x, 0, p1.z, p1.x, 1, p1.z, p2.x, 1, p2.z);
            } else {
                positions.push(p1.x, 0, p1.z, p2.x, 0, p2.z, p2.x, 1, p2.z);
                positions.push(p1.x, 0, p1.z, p2.x, 1, p2.z, p1.x, 1, p1.z);
            }

            for (let j = 0; j < 6; j++) {
                faceIndices.push(faceId);
            }
        };

        for (let i = 0; i < 6; i++) {
            const p1 = corners[i];
            const p2 = corners[(i + 1) % 6];

            // Midpoint normal (approx)
            const midAngle = Math.atan2((p1.z + p2.z) / 2, (p1.x + p2.x) / 2);
            const nx = Math.cos(midAngle);
            const nz = Math.sin(midAngle);

            pushWall(p1, p2);
            for (let j = 0; j < 6; j++) normals.push(nx, 0, nz);
        }

        // Caps
        for (let i = 0; i < 6; i++) {
            const p1 = corners[i];
            const p2 = corners[(i + 1) % 6];
            positions.push(0, 1, 0, p2.x, 1, p2.z, p1.x, 1, p1.z);
            normals.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
            faceIndices.push(6, 6, 6); // Top

            positions.push(0, 0, 0, p1.x, 0, p1.z, p2.x, 0, p2.z);
            normals.push(0, -1, 0, 0, -1, 0, 0, -1, 0);
            faceIndices.push(7, 7, 7); // Bot
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
        geometry.setAttribute('faceIndex', new THREE.Float32BufferAttribute(faceIndices, 1));
        return geometry;
    }

    createInstancedMesh(data, texture, baseZ, minZ, maxZ) {
        const numHexes = data.length;
        console.log(`Creating instanced mesh for ${numHexes} hexes...`);
        document.getElementById('piston-count').innerText = numHexes.toLocaleString();

        const side = HEX_WIDTH / Math.sqrt(3);
        const geometry = this.createHexGeometry(side);

        const material = new THREE.MeshBasicMaterial({
            map: texture,
            side: THREE.FrontSide
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
                attribute vec4 instanceNZ_1; 
                attribute vec4 instanceNZ_2;
                attribute float faceIndex;
                
                varying vec3 vWorldPos;
                varying vec3 vObjNormal;
                varying float vFaceSlope;
                varying float vGrad;
                varying float vIsHidden;
                varying float vFaceIndex;
                `
            ).replace(
                '#include <begin_vertex>',
                `
                #include <begin_vertex>
                vIsHidden = 0.0;
                vFaceIndex = faceIndex;

                int face = int(faceIndex + 0.5);
                float neighborZ = 0.0;
                float myZ = instanceNZ_2.z;
                bool isWall = true;

                if (face == 0) neighborZ = instanceNZ_1.x; // N
                else if (face == 1) neighborZ = instanceNZ_1.y; // NE
                else if (face == 2) neighborZ = instanceNZ_1.z; // SE
                else if (face == 3) neighborZ = instanceNZ_1.w; // S
                else if (face == 4) neighborZ = instanceNZ_2.x; // SW
                else if (face == 5) neighborZ = instanceNZ_2.y; // NW
                else {
                    isWall = false; 
                    neighborZ = myZ;
                }

                float animMyZ = myZ * uHeightFactor;
                float animNeighborZ = neighborZ * uHeightFactor;

                if (isWall) {
                    // Logic: I only draw if I am TALLER than neighbor.
                    // If myZ <= neighborZ, I collapse.
                    if (myZ <= neighborZ + 0.01) { 
                        vIsHidden = 1.0;
                        transformed = vec3(0.0);
                    } else {
                        if (position.y > 0.5) transformed.y = animMyZ;
                        else transformed.y = animNeighborZ;
                    }
                } else {
                    transformed.y = animMyZ; // Top Cap
                }

                vWorldPos = (instanceMatrix * vec4(transformed, 1.0)).xyz;
                vObjNormal = normal;

                float dz = abs(myZ - neighborZ);
                vFaceSlope = degrees(atan(dz, 10.0));
                
                vGrad = 1.0;
                if (isWall && position.y < 0.5) vGrad = 0.0;
                `
            );

            shader.fragmentShader = shader.fragmentShader.replace(
                '#include <common>',
                `
                #include <common>
                uniform vec2 uTileSize;
                varying vec3 vWorldPos;
                varying vec3 vObjNormal;
                varying float vFaceSlope;
                varying float vGrad;
                varying float vIsHidden;
                varying float vFaceIndex;

                vec2 getUV(vec3 pos) {
                    float u = pos.x / 1250.0;
                    float v = -pos.z / 1000.0;
                    return vec2(clamp(u, 0.001, 0.999), clamp(v, 0.001, 0.999));
                }
                `
            ).replace(
                '#include <map_fragment>',
                `
                if (vIsHidden > 0.5) discard;
                
                #include <map_fragment>
                
                if (abs(vObjNormal.y) < 0.9) {
                    vec2 uv = getUV(vWorldPos);
                    vec3 texColor = texture2D(map, vec2(uv.x, 1.0 - uv.y)).rgb;

                    float verticalShade = mix(0.32, 1.0, pow(vGrad, 1.4));
                    vec3 sideBase = texColor * verticalShade;

                    vec3 lightDir = normalize(vec3(0.45, 0.80, 0.35));
                    vec3 viewDir = normalize(cameraPosition - vWorldPos);
                    vec3 nrm = normalize(vObjNormal);
                    float lambert = clamp(dot(nrm, lightDir), 0.0, 1.0);
                    float rim = pow(1.0 - clamp(dot(nrm, viewDir), 0.0, 1.0), 2.2);
                    float ao = mix(0.1, 1.0, pow(vGrad, 0.9));
                    float faceShade = 0.35 + 0.55 * lambert + 0.18 * rim;
                    sideBase *= clamp(faceShade, 0.3, 1.1) * ao;

                    vec3 cGreen = vec3(0.0, 1.0, 0.0);
                    vec3 cBlue = vec3(0.0009, 0.0027, 0.1119);
                    vec3 cYellow = vec3(1.0, 0.85, 0.0);
                    vec3 cRed = vec3(0.85, 0.0, 0.0);

                    vec3 slopeColor = cGreen;
                    if (vFaceSlope >= 35.0 && vFaceSlope < 40.0) slopeColor = cBlue;
                    else if (vFaceSlope >= 40.0 && vFaceSlope < 45.0) slopeColor = cYellow;
                    else if (vFaceSlope >= 45.0) slopeColor = cRed;
                    slopeColor *= ao;
                    
                    if (vFaceSlope >= 30.0) diffuseColor.rgb = slopeColor;
                    else diffuseColor.rgb = sideBase;
                } else {
                    vec2 uv = getUV(vWorldPos);
                    vec4 texColor = texture2D(map, vec2(uv.x, 1.0 - uv.y));
                    diffuseColor = texColor;
                }
                `
            );
        };

        const mesh = new THREE.InstancedMesh(geometry, material, numHexes);
        const matrix = new THREE.Matrix4();

        // Use local normalization for piston heights
        const floorOffset = minZ;

        const instanceNZ_1 = new Float32Array(numHexes * 4);
        const instanceNZ_2 = new Float32Array(numHexes * 4);

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

                // NZ_1: N, NE, SE, S (offset by minZ)
                instanceNZ_1[idx * 4] = (h.n_n - floorOffset) * SCALE_Z;
                instanceNZ_1[idx * 4 + 1] = (h.n_ne - floorOffset) * SCALE_Z;
                instanceNZ_1[idx * 4 + 2] = (h.n_se - floorOffset) * SCALE_Z;
                instanceNZ_1[idx * 4 + 3] = (h.n_s - floorOffset) * SCALE_Z;

                // NZ_2: SW, NW, Z, padding (offset by minZ)
                instanceNZ_2[idx * 4] = (h.n_sw - floorOffset) * SCALE_Z;
                instanceNZ_2[idx * 4 + 1] = (h.n_nw - floorOffset) * SCALE_Z;
                instanceNZ_2[idx * 4 + 2] = (h.z - floorOffset) * SCALE_Z;
                instanceNZ_2[idx * 4 + 3] = 0.0;

                idx++;
            }
        }
        console.log(`Assigned matrices for ${idx} instances.`);

        geometry.setAttribute('instanceNZ_1', new THREE.InstancedBufferAttribute(instanceNZ_1, 4));
        geometry.setAttribute('instanceNZ_2', new THREE.InstancedBufferAttribute(instanceNZ_2, 4));

        this.scene.add(mesh);
        this.pistonMesh = mesh;
        this.pistonMesh.visible = true;
        this.pistonMesh.frustumCulled = false;

        // Ensure matrices are sent to GPU
        mesh.instanceMatrix.needsUpdate = true;

        const localMaxHeight = (maxZ - minZ) * SCALE_Z;
        console.log(`Max Local Height from Floor: ${localMaxHeight}`);

        this.camera.position.set(625, localMaxHeight + 1000, -500);
        this.controls.target.set(625, 0, -500);
        this.controls.update();

        console.log(`Camera and Controls Set.`);

        // Hide Loader
        const loader = document.getElementById('loader');
        if (loader) {
            loader.style.transition = 'opacity 0.5s';
            loader.style.opacity = '0';
            setTimeout(() => {
                loader.style.display = 'none';
            }, 500);
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.checkHighResTrigger();

        // Piston Animation Logic
        if (this.pistonMesh) {
            const angle = this.controls.getPolarAngle();
            
            // MapControls: 0 is Top-Down.
            // We want: 0 deg (Top) -> Flat.
            // 20 deg -> Full Height.
            let factor = angle / (20 * Math.PI / 180); 
            factor = Math.min(1.0, Math.max(0.01, factor)); // Minimum 0.01 to keep tops visible

            // Smooth transition - update shader uniform
            if (this.pistonMaterial && this.pistonMaterial.userData.shader) {
                this.pistonMaterial.userData.shader.uniforms.uHeightFactor.value = factor;
            }

            this.pistonMesh.visible = true;
        }

        this.renderer.render(this.scene, this.camera);
    }

    onResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

new PistonViewer();
