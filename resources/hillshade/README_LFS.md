# Hillshade Data Files - Git LFS Required

This folder contains pre-rendered hillshade visualizations for terrain display across different seasonal periods.

## File Organization

### ✅ Available on GitHub (Regular Git)
- `hillshade_100m_period*.tif` - 100m resolution hillshades (~45MB total)
  - Web-optimized for frontend terrain visualization
  - 4 seasonal periods for dynamic display

### 🔒 Requires Git LFS (Large File Storage)  
- `hillshade_25m_period*.tif` - 25m resolution hillshades (~200MB each)
- `hillshade_5m_period*.tif` - 5m resolution hillshades (~2GB+ each)

## Git LFS Setup

To download the high-resolution hillshade files:

```bash
# Install Git LFS (if not already installed)
git lfs install

# Pull LFS files
git lfs pull

# Track hillshade patterns for LFS
git lfs track "resources/hillshade/*25m*.tif"
git lfs track "resources/hillshade/*5m*.tif"
```

## Seasonal Periods

The hillshade files are rendered for 4 distinct seasonal periods:

1. **Period 1**: Winter solstice lighting (December 21)
2. **Period 2**: Spring equinox lighting (March 21) 
3. **Period 3**: Summer solstice lighting (June 21)
4. **Period 4**: Autumn equinox lighting (September 21)

Each period captures different sun angles and shadow patterns for realistic terrain visualization.

## Technical Specifications

- **Projection**: Web Mercator (EPSG:3857)
- **Format**: GeoTIFF with internal compression
- **Bit Depth**: 8-bit grayscale for optimal web performance
- **Hillshade Algorithm**: Standard GDAL implementation
  - Azimuth: 315° (northwest lighting)
  - Altitude: 45° elevation
  - Z-factor: Adjusted for latitude

## File Sizes

| Resolution | File Size (per period) | Total (4 periods) |
|-----------|------------------------|-------------------|
| 100m      | ~11MB                 | ~45MB            |
| 25m       | ~50MB                 | ~200MB           |
| 5m        | ~500MB                | ~2GB             |

## Usage in Frontend

The 100m hillshade files are used by `gridService.js` for terrain visualization:

```javascript
// Dynamic hillshade loading based on season
const period = getCurrentSeason(); // 1-4
const hillshadeUrl = `resources/hillshade/hillshade_100m_period${period}.tif`;
```

## Rendering Pipeline

Generated using `render_hillshade.py`:
1. Loads terrain DEM files
2. Calculates hillshade for each seasonal sun position
3. Exports optimized GeoTIFF files
4. Creates multiple resolution versions

For rendering instructions, see `render_hillshade.py` in this directory.
