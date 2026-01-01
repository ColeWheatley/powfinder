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
        this.scene.fog = new THREE.Fog(0x050505, 500, 2500);

        this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000);
        this.camera.position.set(625, 800, 800);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        this.controls = new MapControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 100;
        this.controls.maxDistance = 3000;
        this.controls.maxPolarAngle = Math.PI / 2.1;

        // Boost speeds for better responsiveness
        this.controls.zoomSpeed = 1.5;
        this.controls.panSpeed = 1.2;

        // Map behavior: 1 finger pan, 2 finger rotate/zoom
        this.controls.touches = {
            ONE: THREE.TOUCH.PAN,
            TWO: THREE.TOUCH.DOLLY_ROTATE
        };

        const ambient = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambient);

        const sun = new THREE.DirectionalLight(0xffffff, 1.2);
        sun.position.set(500, 1000, 500);
        this.scene.add(sun);

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
        const webpUrl = `tiles_sat/tile_${x}_${y}.webp`;

        // Load Texture
        console.log("Fetching texture...");
        const texLoader = new THREE.TextureLoader();
        const texture = await texLoader.loadAsync(webpUrl);
        console.log("Texture loaded successfully.");
        texture.colorSpace = THREE.SRGBColorSpace;

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

        // Header is 4 bytes, each hex is 14 bytes now
        while (offset < buffer.byteLength) {
            const zRaw = view.getUint16(offset, true);
            const z = this.decodeFloat16(zRaw);

            const s1 = view.getUint8(offset + 2);
            const s2 = view.getUint8(offset + 3);
            const s3 = view.getUint8(offset + 4);

            const rgb1 = [view.getUint8(offset + 5) / 255, view.getUint8(offset + 6) / 255, view.getUint8(offset + 7) / 255];
            const rgb2 = [view.getUint8(offset + 8) / 255, view.getUint8(offset + 9) / 255, view.getUint8(offset + 10) / 255];
            const rgb3 = [view.getUint8(offset + 11) / 255, view.getUint8(offset + 12) / 255, view.getUint8(offset + 13) / 255];

            if (hexData.length < 5) {
                console.log(`First Hex Sample: Z=${z.toFixed(2)}, S1=${s1}, S2=${s2}, S3=${s3}`);
            }

            hexData.push({ z, s1, s2, s3, rgb1, rgb2, rgb3 });
            offset += 14;
        }
        console.log(`Total HexData Parsed: ${hexData.length}`);

        this.createInstancedMesh(hexData, texture, baseElevation);
    }

    decodeFloat16(binary) {
        const s = (binary & 0x8000) >> 15;
        const e = (binary & 0x7C00) >> 10;
        const f = binary & 0x03FF;
        if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
        if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
        return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }

    createInstancedMesh(data, texture, baseZ) {
        const numHexes = data.length;
        console.log(`Creating instanced mesh for ${numHexes} hexes...`);
        document.getElementById('piston-count').innerText = numHexes.toLocaleString();

        const side = HEX_WIDTH / Math.sqrt(3);
        const geometry = new THREE.CylinderGeometry(side, side, 1, 6);
        geometry.translate(0, 0.5, 0);
        geometry.rotateY(Math.PI / 2);

        const material = new THREE.MeshStandardMaterial({
            map: texture,
            metalness: 0,
            roughness: 0.8
        });

        material.onBeforeCompile = (shader) => {
            shader.uniforms.uTileSize = { value: new THREE.Vector2(1250, 1000) };
            shader.vertexShader = `
                attribute float instanceZ;
                attribute vec3 instanceSlope;
                attribute vec3 instanceRGB1;
                attribute vec3 instanceRGB2;
                attribute vec3 instanceRGB3;
                varying vec3 vSlope;
                varying vec3 vWorldPos;
                varying vec3 vObjNormal;
                varying vec3 vSideColor;
                varying float vFaceSlope;
                ${shader.vertexShader}
            `.replace(
                '#include <begin_vertex>',
                `
                #include <begin_vertex>
                transformed.y *= instanceZ;
                vSlope = instanceSlope;
                vWorldPos = (instanceMatrix * vec4(transformed, 1.0)).xyz;
                vObjNormal = normal;

                // Pick the correct baked color and slope based on face normal
                float ang = atan(normal.z, normal.x);
                if (ang > -0.5 && ang < 1.5) { 
                    vSideColor = instanceRGB1; 
                    vFaceSlope = instanceSlope.x; 
                }
                else if (ang > 1.5 || ang < -2.5) { 
                    vSideColor = instanceRGB2; 
                    vFaceSlope = instanceSlope.y; 
                }
                else { 
                    vSideColor = instanceRGB3; 
                    vFaceSlope = instanceSlope.z; 
                }
                `
            );

            shader.fragmentShader = `
                uniform vec2 uTileSize;
                varying vec3 vSlope;
                varying vec3 vWorldPos;
                varying vec3 vObjNormal;
                varying vec3 vSideColor;
                varying float vFaceSlope;

                vec3 getSlopeColor(float sRaw) {
                    float deg = abs(sRaw - 90.0);
                    if (deg < 25.0) return vec3(0.5, 0.5, 0.5); // Should be ignored by override
                    if (deg < 35.0) return vec3(0.0, 1.0, 0.0); // Green
                    if (deg < 40.0) return vec3(0.0, 0.0, 1.0); // Blue
                    if (deg < 45.0) return vec3(0.5, 0.0, 0.5); // Purple
                    if (deg < 50.0) return vec3(1.0, 0.5, 0.0); // Orange
                    return vec3(1.0, 0.0, 0.0);                 // Red
                }
                
                ${shader.fragmentShader}
            `.replace(
                '#include <map_fragment>',
                `
                #include <map_fragment>
                
                if (abs(vObjNormal.y) < 0.9) {
                    // Start with the baked satellite color
                    vec3 finalColor = vSideColor;
                    
                    // Override if slope is steep (>= 25 degrees)
                    float deg = abs(vFaceSlope - 90.0);
                    if (deg >= 25.0) {
                        finalColor = getSlopeColor(vFaceSlope);
                    }
                    
                    diffuseColor.rgb = finalColor;
                } else {
                    // Top cap - use WebP Texture
                    vec2 uvSat = vec2(vWorldPos.x / uTileSize.x, 1.0 + (vWorldPos.z / uTileSize.y));
                    diffuseColor = texture2D(map, uvSat);
                }
                `
            );
        };

        const mesh = new THREE.InstancedMesh(geometry, material, numHexes);
        const matrix = new THREE.Matrix4();
        // Determine a safe floor (min height)
        let minZ = 0;
        data.forEach(h => { if (h.z < minZ) minZ = h.z; });
        console.log(`Piston Floor Offset: ${minZ}m`);
        const instanceZ = new Float32Array(numHexes);
        const instanceSlope = new Float32Array(numHexes * 3);
        const instanceRGB1 = new Float32Array(numHexes * 3);
        const instanceRGB2 = new Float32Array(numHexes * 3);
        const instanceRGB3 = new Float32Array(numHexes * 3);

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

                // Position the piston at Y=0 (the floor)
                matrix.setPosition(x, 0, -realY);
                mesh.setMatrixAt(idx, matrix);

                // Height is relative to the floor
                // We add a 5m "under-base" so we never see gaps
                instanceZ[idx] = (h.z - minZ + 5.0) * SCALE_Z;

                instanceSlope[idx * 3] = h.s1;
                instanceSlope[idx * 3 + 1] = h.s2;
                instanceSlope[idx * 3 + 2] = h.s3;

                instanceRGB1.set(h.rgb1, idx * 3);
                instanceRGB2.set(h.rgb2, idx * 3);
                instanceRGB3.set(h.rgb3, idx * 3);
                idx++;
            }
        }
        console.log(`Assigned matrices for ${idx} instances.`);

        geometry.setAttribute('instanceZ', new THREE.InstancedBufferAttribute(instanceZ, 1));
        geometry.setAttribute('instanceSlope', new THREE.InstancedBufferAttribute(instanceSlope, 3));
        geometry.setAttribute('instanceRGB1', new THREE.InstancedBufferAttribute(instanceRGB1, 3));
        geometry.setAttribute('instanceRGB2', new THREE.InstancedBufferAttribute(instanceRGB2, 3));
        geometry.setAttribute('instanceRGB3', new THREE.InstancedBufferAttribute(instanceRGB3, 3));

        this.scene.add(mesh);
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
        this.renderer.render(this.scene, this.camera);
    }
}

new PistonViewer();
