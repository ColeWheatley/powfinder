import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { TIFFLoader } from 'three/addons/loaders/TIFFLoader.js';

// --- CONFIG ---
const TILE_WIDTH_WORLD = 1250;
const TILE_HEIGHT_WORLD = 1000;
const HEX_WIDTH = 10;
const HEX_DX = HEX_WIDTH * (Math.sqrt(3) / 2);
const SCALE_Z = 1.0;

class PistonViewer {
    constructor() {
        console.log("Initializing PistonViewer (Multi-Tile)...");
        this.container = document.getElementById('canvas-container');
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x050505);

        // Initial Camera Setup - Will be reset in initWorld
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
        this.controls.maxDistance = 50000; // Increased max distance
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
        this.materialsToUpdate = [];
        this.worldOrigin = { x: 0, y: 0 };
        this.lightingSettings = {
            aoFloor: 0.1,
            aoPower: 0.9,
            lambert: 0.7,
            rim: 0.18,
            rimPower: 2.2,
            spec: 0.0,
            specPower: 30.0,
            slopeLight: 0.0,
        };
        this.fpsState = { lastSample: performance.now(), frames: 0 };
        this.fpsEl = null;

        // Start
        this.initLightingControls();
        this.initWorld();
        this.animate();
    }

    initLightingControls() {
        this.textureFlipY = false;

        // Debug: Texture Flip Key Listener
        window.addEventListener('keydown', (e) => {
            if (e.key === 'f' || e.key === 'F') {
                this.textureFlipY = !this.textureFlipY;
                console.log("Toggling Texture Flip Y:", this.textureFlipY);
                for (const mat of this.materialsToUpdate) {
                    if (mat.userData.shader) {
                        mat.userData.shader.uniforms.uTextureFlipY.value = this.textureFlipY ? 1.0 : 0.0;
                    }
                }
            }
        });

        this.fpsEl = document.getElementById('fps-counter');

        const panel = document.getElementById('shader-controls');
        if (!panel) return;

        const bind = (id, key, format) => {
            const input = document.getElementById(id);
            if (!input) return;
            const output = document.getElementById(`${id}-val`);
            const fmt = format || ((val) => val.toFixed(2));
            const update = () => {
                const value = parseFloat(input.value);
                this.lightingSettings[key] = value;
                if (output) output.textContent = fmt(value);
                this.updateLightingUniforms();
            };
            input.addEventListener('input', update);
            update();
        };

        bind('ao-floor', 'aoFloor');
        bind('ao-power', 'aoPower');
        bind('lambert', 'lambert');
        bind('rim', 'rim');
        bind('rim-power', 'rimPower');
        bind('spec', 'spec');
        bind('spec-power', 'specPower', (val) => val.toFixed(0));
        bind('slope-light', 'slopeLight');
    }

    updateLightingUniforms() {
        for (const mat of this.materialsToUpdate) {
            const shader = mat.userData.shader;
            if (!shader) continue;
            shader.uniforms.uAoFloor.value = this.lightingSettings.aoFloor;
            shader.uniforms.uAoPower.value = this.lightingSettings.aoPower;
            shader.uniforms.uLambertStrength.value = this.lightingSettings.lambert;
            shader.uniforms.uRimStrength.value = this.lightingSettings.rim;
            shader.uniforms.uRimPower.value = this.lightingSettings.rimPower;
            shader.uniforms.uSpecStrength.value = this.lightingSettings.spec;
            shader.uniforms.uSpecPower.value = this.lightingSettings.specPower;
            shader.uniforms.uSlopeLight.value = this.lightingSettings.slopeLight;
        }
    }

    async initWorld() {
        try {
            const res = await fetch('tile_manifest.json');
            if (!res.ok) throw new Error('Failed to load tile_manifest.json');
            this.manifest = await res.json();

            console.log(`Manifest loaded. Found ${this.manifest.tiles.length} tiles.`);

            // Calculate Global Bounds
            const { min_x, min_y, max_x, max_y } = this.manifest.bounds;
            this.worldOrigin = { x: min_x, y: min_y };

            // Center Camera on the map
            const mapWidth = (max_x - min_x) + TILE_WIDTH_WORLD;
            const mapHeight = (max_y - min_y) + TILE_HEIGHT_WORLD;

            const centerX = mapWidth / 2;
            const centerZ = -mapHeight / 2;

            // FIX: Start HIGH UP (Birds Eye) looking straight down
            // This ensures `angle = 0` -> `factor = 0` -> Pistons Flat
            this.camera.position.set(centerX, 800, centerZ);
            this.controls.target.set(centerX, 0, centerZ);
            this.controls.update();

            // Load All Tiles
            for (const tile of this.manifest.tiles) {
                this.loadSingleTile(tile).catch(e => console.error(`Error loading tile ${tile.x}_${tile.y}:`, e));
            }

            // Hide Loader
            const loader = document.getElementById('loader');
            if (loader) {
                setTimeout(() => {
                    loader.style.transition = 'opacity 0.5s';
                    loader.style.opacity = '0';
                    setTimeout(() => { loader.style.display = 'none'; }, 500);
                }, 1500);
            }

        } catch (err) {
            console.error("Error initializing world:", err);
        }
    }

    async loadSingleTile(tileDef) {
        const { x, y } = tileDef;

        // World Position:
        // X grows East (Right). Y grows North (Up).
        // In 3D: X is +X, Y is -Z.
        const posX = x - this.worldOrigin.x;
        const posZ = -(y - this.worldOrigin.y);

        const binUrl = `tiles_bin/tile_${x}_${y}.bin`;
        const lowTexUrl = `tiles_sat/low_res/tile_${x}_${y}.webp`;
        const medTexUrl = `tiles_sat/med_res/tile_${x}_${y}.webp`;
        const highTexUrl = `tiles_sat/high_res/tile_${x}_${y}.tif`;

        const texLoader = new THREE.TextureLoader();
        const texture = await texLoader.loadAsync(lowTexUrl);
        texture.colorSpace = THREE.SRGBColorSpace;
        // Optimization: Don't flipY in texture loader, handle in shader
        texture.flipY = false;

        const material = new THREE.MeshBasicMaterial({
            map: texture,
            side: THREE.FrontSide
        });

        this.setupMaterialShader(material);
        this.materialsToUpdate.push(material);

        const response = await fetch(binUrl);
        const buffer = await response.arrayBuffer();
        const hexData = this.parseBinary(buffer);

        const mesh = this.createInstancedMesh(hexData, material);
        mesh.position.set(posX, 0, posZ);
        mesh.updateMatrixWorld();

        this.scene.add(mesh);

        const tileObj = {
            x, y,
            mesh,
            material,
            highResLoaded: false,
            highResLoading: false,
            urls: { med: medTexUrl, high: highTexUrl }
        };
        this.tiles.push(tileObj);

        // Lazy Load Med Res
        texLoader.load(medTexUrl, (tex) => {
            tex.colorSpace = THREE.SRGBColorSpace;
            tex.flipY = false;
            if (!tileObj.highResLoaded) {
                material.map = tex;
                material.needsUpdate = true;
            }
        });
    }

    parseBinary(buffer) {
        const view = new DataView(buffer);
        const hexData = [];
        let offset = 4;

        while (offset < buffer.byteLength) {
            const z = this.decodeFloat16(view.getUint16(offset, true));
            const n_n = this.decodeFloat16(view.getUint16(offset + 2, true));
            const n_ne = this.decodeFloat16(view.getUint16(offset + 4, true));
            const n_se = this.decodeFloat16(view.getUint16(offset + 6, true));
            const n_s = this.decodeFloat16(view.getUint16(offset + 8, true));
            const n_sw = this.decodeFloat16(view.getUint16(offset + 10, true));
            const n_nw = this.decodeFloat16(view.getUint16(offset + 12, true));
            hexData.push({ z, n_n, n_ne, n_se, n_s, n_sw, n_nw });
            offset += 14;
        }
        return hexData;
    }

    createInstancedMesh(hexes, material) {
        const numHexes = hexes.length;
        // FIX: Clone geometry so each tile has unique instance attributes
        const mesh = new THREE.InstancedMesh(this.hexGeometry.clone(), material, numHexes);

        const matrix = new THREE.Matrix4();
        const instanceNZ_1 = new Float32Array(numHexes * 4);
        const instanceNZ_2 = new Float32Array(numHexes * 4);

        let idx = 0;
        const floorOffset = 0.0;

        for (let x = 0; x <= TILE_WIDTH_WORLD + 1; x += HEX_DX) {
            const colIdx = Math.round(x / HEX_DX);
            const yShift = (colIdx % 2 === 1) ? 5 : 0;
            for (let y = 0; y <= TILE_HEIGHT_WORLD + 1; y += 10) {
                if (idx >= numHexes) break;

                const realY = y + yShift;
                const h = hexes[idx];

                matrix.makeTranslation(x, 0, -realY);
                mesh.setMatrixAt(idx, matrix);

                instanceNZ_1[idx * 4] = (h.n_n - floorOffset) * SCALE_Z;
                instanceNZ_1[idx * 4 + 1] = (h.n_ne - floorOffset) * SCALE_Z;
                instanceNZ_1[idx * 4 + 2] = (h.n_se - floorOffset) * SCALE_Z;
                instanceNZ_1[idx * 4 + 3] = (h.n_s - floorOffset) * SCALE_Z;

                instanceNZ_2[idx * 4] = (h.n_sw - floorOffset) * SCALE_Z;
                instanceNZ_2[idx * 4 + 1] = (h.n_nw - floorOffset) * SCALE_Z;
                instanceNZ_2[idx * 4 + 2] = (h.z - floorOffset) * SCALE_Z;
                instanceNZ_2[idx * 4 + 3] = 0.0;

                idx++;
            }
        }

        mesh.instanceMatrix.needsUpdate = true;
        mesh.geometry.setAttribute('instanceNZ_1', new THREE.InstancedBufferAttribute(instanceNZ_1, 4));
        mesh.geometry.setAttribute('instanceNZ_2', new THREE.InstancedBufferAttribute(instanceNZ_2, 4));
        mesh.frustumCulled = true;

        return mesh;
    }

    setupMaterialShader(material) {
        material.userData.shader = null;
        material.onBeforeCompile = (shader) => {
            material.userData.shader = shader;
            shader.uniforms.uHeightFactor = { value: 0.0 };
            shader.uniforms.uTextureFlipY = { value: 0.0 }; // Default 0 (Standard 1-y)

            // User Lighting Vars
            shader.uniforms.uAoFloor = { value: this.lightingSettings.aoFloor };
            shader.uniforms.uAoPower = { value: this.lightingSettings.aoPower };
            shader.uniforms.uLambertStrength = { value: this.lightingSettings.lambert };
            shader.uniforms.uRimStrength = { value: this.lightingSettings.rim };
            shader.uniforms.uRimPower = { value: this.lightingSettings.rimPower };
            shader.uniforms.uSpecStrength = { value: this.lightingSettings.spec };
            shader.uniforms.uSpecPower = { value: this.lightingSettings.specPower };
            shader.uniforms.uSlopeLight = { value: this.lightingSettings.slopeLight };

            shader.vertexShader = shader.vertexShader.replace(
                '#include <common>',
                `#include <common>
                uniform float uHeightFactor;
                attribute vec4 instanceNZ_1; 
                attribute vec4 instanceNZ_2;
                attribute float faceIndex;
                
                varying vec3 vLocalPos;
                varying vec3 vWorldPos;
                varying vec3 vObjNormal;
                varying float vFaceSlope;
                varying float vGrad;
                varying float vIsHidden;
                `
            ).replace(
                '#include <begin_vertex>',
                `#include <begin_vertex>
                vIsHidden = 0.0;
                
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
                else { isWall = false; neighborZ = myZ; }

                float animMyZ = myZ * uHeightFactor;
                float animNeighborZ = neighborZ * uHeightFactor;

                if (isWall) {
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

                // UV Calc: Relative to Instanced Mesh (Tile) Origin (0,0) to (1250, -1000)
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
                `
            ).replace(
                '#include <map_fragment>',
                `if (vIsHidden > 0.5) discard;
                
                // UV Mapping
                float u = vLocalPos.x / 1250.0;
                float v = -vLocalPos.z / 1000.0;
                
                // Texture Y logic: 
                // Default (uTextureFlipY=0.0): 1.0 - v (Standard GL)
                // Flipped (uTextureFlipY=1.0): v 
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
                    if (vFaceSlope >= 35.0 && vFaceSlope < 40.0) slopeColor = cBlue;
                    else if (vFaceSlope >= 40.0 && vFaceSlope < 45.0) slopeColor = cYellow;
                    else if (vFaceSlope >= 45.0) slopeColor = cRed;
                    slopeColor *= mix(1.0, light, uSlopeLight) * ao;
                    
                    if (vFaceSlope >= 30.0) diffuseColor.rgb = slopeColor;
                    else diffuseColor.rgb = sideBase;
                }
                `
            );
            this.updateLightingUniforms();
        };
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.updateFps();

        const angle = this.controls.getPolarAngle();
        let factor = angle / (20 * Math.PI / 180);
        factor = Math.min(1.0, Math.max(0.01, factor));

        for (const mat of this.materialsToUpdate) {
            if (mat.userData.shader) {
                mat.userData.shader.uniforms.uHeightFactor.value = factor;
            }
        }

        this.updateTextureLOD();
        this.renderer.render(this.scene, this.camera);
    }

    updateFps() {
        if (!this.fpsEl) return;
        const now = performance.now();
        this.fpsState.frames += 1;
        const elapsed = now - this.fpsState.lastSample;
        if (elapsed < 500) return;
        const fps = (this.fpsState.frames * 1000) / elapsed;
        this.fpsEl.textContent = fps.toFixed(0);
        this.fpsState.frames = 0;
        this.fpsState.lastSample = now;
    }

    updateTextureLOD() {
        const target = this.controls.target;
        const dist = this.camera.position.distanceTo(target);
        if (dist > 1500) return;

        let closest = null;
        let minDist = Infinity;

        for (const tile of this.tiles) {
            const cx = (tile.x - this.worldOrigin.x) + 625;
            const cz = -(tile.y - this.worldOrigin.y) - 500;
            const d = (target.x - cx) ** 2 + (target.z - cz) ** 2;
            if (d < minDist) {
                minDist = d;
                closest = tile;
            }
        }

        if (closest && !closest.highResLoaded && !closest.highResLoading) {
            console.log(`Loading High-Res for ${closest.x}, ${closest.y}`);
            closest.highResLoading = true;
            const tiffLoader = new TIFFLoader();
            tiffLoader.load(closest.urls.high, (tex) => {
                tex.colorSpace = THREE.SRGBColorSpace;
                tex.flipY = false; // Shader handles flip
                closest.material.map = tex;
                closest.material.needsUpdate = true;
                closest.highResLoaded = true;
                closest.highResLoading = false;
            });
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
            { x: 0.0, z: 1.0 }, { x: 0.866, z: 0.5 }, { x: 0.866, z: -0.5 },
            { x: 0.0, z: -1.0 }, { x: -0.866, z: -0.5 }, { x: -0.866, z: 0.5 },
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
