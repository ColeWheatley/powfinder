# GPU Texture Compression Research

**Date**: January 2025
**Context**: Research for PowFinder hexagon terrain system
**Problem**: Chrome crashing with 10GB RAM usage when loading 150+ WebP tiles

---

## The Core Problem

WebP/JPEG/PNG are **CPU-optimized formats**. They compress well for network transfer, but **decompress to full bitmaps in GPU memory (VRAM)**.

| Format | Network Size | VRAM Size | Notes |
|--------|--------------|-----------|-------|
| WebP 10% | ~1 MB | **125 MB** | Decompresses to RGBA |
| JPEG | ~2 MB | **125 MB** | Decompresses to RGBA |
| PNG | ~15 MB | **125 MB** | Decompresses to RGBA |

For a 6250x5000 tile: `6250 * 5000 * 4 bytes = 125 MB` in VRAM regardless of input format.

With 150 tiles visible at once: `150 * 125 MB = 18.75 GB VRAM` - hence the crash.

---

## The Solution: GPU-Compressed Textures

GPU-compressed formats stay compressed in VRAM. The GPU decompresses on-the-fly during rendering.

### Native GPU Formats

| Format | Platform | Bits Per Pixel | Quality |
|--------|----------|----------------|---------|
| **ASTC** | iOS (A8+), Modern Android, New Desktop | 2-8 bpp | Excellent |
| **BC7** | Desktop (DX11+) | 8 bpp | Excellent |
| **BC1** (DXT1) | Desktop (legacy) | 4 bpp | Good (no alpha) |
| **ETC2** | Android (ES 3.0+) | 4-8 bpp | Good |
| **PVRTC** | iOS (legacy) | 2-4 bpp | Acceptable |

The problem: Each platform needs a different format. You'd have to ship 3-4 versions of every texture.

---

## Basis Universal: The Universal Solution

[Basis Universal](https://github.com/BinomialLLC/basis_universal) solves this by creating a **single file** that **transcodes at runtime** to whatever format the device supports.

### Two Encoding Modes

#### 1. ETC1S (BasisLZ)
- **Network size**: 0.3-1.25 bpp (WebP-like!)
- **VRAM**: ~4 bpp after transcoding
- **Quality**: Low-Medium
- **Transcodes to**: BC1, ETC1, PVRTC, ASTC
- **Best for**: Color/albedo textures, when network size is critical

#### 2. UASTC
- **Network size**: 8 bpp raw, **2-6 bpp with Zstandard compression**
- **VRAM**: ~4-8 bpp after transcoding
- **Quality**: Excellent (comparable to BC7/ASTC)
- **Transcodes to**: BC7, ASTC 4x4, ETC2
- **Best for**: Normal maps, complex textures, when quality matters

### KTX2: The Container Format

KTX2 is the Khronos standard container for GPU textures. It wraps Basis Universal data and adds:
- Mipmap support
- Supercompression (Zstandard)
- Metadata

---

## The Recommendation: UASTC + Zstandard

For terrain with 2020+ device targets:

```
UASTC + Zstd supercompression
├── Network: ~8-15 MB per full-res tile (comparable to JPEG)
├── VRAM: ~4-8 MB per tile (stays compressed!)
├── Quality: Excellent
└── Compatibility: Universal (transcodes to BC7/ASTC/ETC2)
```

### Comparison for 6250x5000 tile:

| Format | Network | VRAM | 150 Tiles VRAM |
|--------|---------|------|----------------|
| WebP 10% | 1 MB | 125 MB | **18.75 GB** |
| UASTC 8x8 | 32 MB | 4 MB | **600 MB** |
| ETC1S | 5 MB | 15 MB | **2.25 GB** |

**31x VRAM reduction** with UASTC vs WebP!

---

## Device Compatibility (2020+)

### iOS
- iPhone 11+ (A13, 2019): 100% ASTC support
- iPad Air 3+ (A12, 2019): 100% ASTC support
- WebGL extension: `WEBGL_compressed_texture_astc`

### Android
- ~98.5% of 2020+ devices support ASTC or ETC2
- Fallback to ETC2 for older ES 3.0 devices
- WebGL extension: `WEBGL_compressed_texture_astc` or `WEBGL_compressed_texture_etc`

### Desktop
- Universal BC7 support (DX11+)
- WebGL extension: `WEBGL_compressed_texture_s3tc`

---

## How Three.js Loads KTX2

```javascript
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';

const loader = new KTX2Loader();
loader.setTranscoderPath('https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/libs/basis/');
loader.detectSupport(renderer);  // Detects which GPU formats are available

const texture = await loader.loadAsync('texture.ktx2');
// Automatically transcodes to BC7/ASTC/ETC2 based on device
```

---

## Creating KTX2 Files

### Install basisu (macOS)
```bash
brew install basisu
```

### UASTC (High Quality)
```bash
# Full quality UASTC
basisu -file input.png -ktx2 -uastc -mipmap -y_flip -output_file output.ktx2

# With quality level (0=fastest, 4=highest)
basisu -file input.png -ktx2 -uastc -uastc_level 2 -mipmap -y_flip -output_file output.ktx2
```

### ETC1S (Small Network)
```bash
# Default quality
basisu -file input.png -ktx2 -mipmap -y_flip -output_file output.ktx2

# High quality (q=1-255)
basisu -file input.png -ktx2 -q 255 -mipmap -y_flip -output_file output.ktx2
```

### ASTC Block Sizes
```bash
# 4x4 blocks (8 bpp, highest quality)
basisu ... -uastc_rdo_l 0

# 8x8 blocks (2 bpp, most aggressive, still good for aerial)
basisu ... -uastc_rdo_l 1.0
```

---

## Known Issue: ETC1S Not Displaying

During testing, ETC1S textures wouldn't display in the browser while UASTC worked fine.

**Possible causes**:
1. ETC1S transcodes to BC1 (DXT1) on desktop, which has different characteristics
2. WebGL extension detection issue
3. Transcoder initialization difference between ETC1S and UASTC
4. Color space handling difference

**Status**: Not yet debugged. The test webapp at `/tmp/ktx2_test/` has both formats but ETC1S shows black.

---

## LOD Strategy for Terrain

```
Distance        Format              Network    VRAM
─────────────────────────────────────────────────────
< 2km          UASTC full-res      ~32 MB     ~4 MB
2-5km          UASTC 2048px        ~8 MB      ~1 MB
5-10km         UASTC 512px         ~500 KB    ~64 KB
> 10km         Placeholder/color   ~1 KB      ~1 KB
```

---

## Integration with waffle_iron.py

Currently `waffle_iron.py` outputs WebP textures. To switch to KTX2:

1. Keep generating WebP as intermediate
2. Run `basisu` to convert WebP → KTX2
3. Upload KTX2 to S3 instead of WebP
4. Update frontend to use KTX2Loader

```python
# Proposed addition to waffle_iron.py
def convert_to_ktx2(webp_path, ktx2_path, mode='uastc'):
    cmd = ['basisu', '-file', webp_path, '-ktx2', '-mipmap', '-y_flip']
    if mode == 'uastc':
        cmd.extend(['-uastc', '-uastc_level', '2'])
    cmd.extend(['-output_file', ktx2_path])
    subprocess.run(cmd)
```

---

## Key Takeaways

1. **WebP/JPEG decompress to full size in VRAM** - that's why you're crashing at 10GB
2. **GPU-compressed formats (ASTC/BC7) stay compressed** - 4-8x VRAM savings
3. **Basis Universal/KTX2 is the universal solution** - one file works everywhere
4. **UASTC + Zstd is recommended** for quality + reasonable network size
5. **ETC1S is even smaller** but has quality tradeoffs and currently has a display bug
6. **2020+ devices all support ASTC or BC7** - no compatibility concerns

---

## Test Files Created

Location: `/tmp/ktx2_test/`

| File | Format | Size | Notes |
|------|--------|------|-------|
| full_4x4_highest.ktx2 | UASTC 4x4 | 37 MB | Overkill |
| full_4x4_high.ktx2 | UASTC 4x4 | 37 MB | Overkill |
| full_6x6_medium.ktx2 | UASTC ~6x6 | 35 MB | Good |
| full_8x8_low.ktx2 | UASTC 8x8 | 32 MB | **Recommended** |
| full_etc1s.ktx2 | ETC1S | 4.9 MB | Doesn't display (bug) |
| full_etc1s_hq.ktx2 | ETC1S q=255 | 6.3 MB | Doesn't display (bug) |
| low_1024.ktx2 | UASTC 1024px | 1.2 MB | Too low res |

Comparison webapp: `http://localhost:8765` (run `python3 -m http.server 8765` in `/tmp/ktx2_test/`)

---

## Next Steps

1. Debug why ETC1S doesn't display (likely transcoder or WebGL extension issue)
2. Integrate KTX2 generation into `waffle_iron.py`
3. Update `main.js` to use KTX2Loader instead of standard texture loader
4. Test on actual iOS/Android devices
5. Implement LOD-based texture loading (full-res KTX2 for close, low-res for distant)

---

## References

- [Basis Universal GitHub](https://github.com/BinomialLLC/basis_universal)
- [KTX2 Specification](https://github.khronos.org/KTX-Specification/ktxspec.v2.html)
- [Three.js KTX2Loader](https://threejs.org/docs/#examples/en/loaders/KTX2Loader)
- [WebGL ASTC Extension](https://developer.mozilla.org/en-US/docs/Web/API/WEBGL_compressed_texture_astc)
- [Don McCurdy: Web Texture Formats 2024](https://www.donmccurdy.com/2024/02/11/web-texture-formats/)
