# Terrain Data Files - Git LFS Required

This folder contains Digital Elevation Model (DEM) and derived terrain analysis files for the Tirol region.

## File Organization

### ✅ Available on GitHub (Regular Git)
- `tirol_100m_*.tif` - 100m resolution files (~46MB total)
  - Web-optimized versions for frontend display
  - Aspect and slope analysis at 100m resolution

### 🔒 Requires Git LFS (Large File Storage)
- `tirol_25m_*.tif` - 25m resolution files (~500MB-1GB each)
- `tirol_5m_*.tif` - 5m resolution files (~10GB+ each)
- `DGM_Tirol_5m_epsg31254_2006_2020.tif` - Original source DEM (>20GB)

## Git LFS Setup

To download the large terrain files, you need Git LFS installed:

```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone repository with LFS files
git lfs pull

# Or track specific patterns for new LFS files
git lfs track "*.tif"
git lfs track "resources/terrains/*25m*.tif"
git lfs track "resources/terrains/*5m*.tif"
git lfs track "resources/terrains/DGM_Tirol_*.tif"
```

## File Descriptions

- **tirol_5m_float.tif**: Original high-resolution DEM (5m, float32)
- **tirol_5m_web.tif**: Web-optimized 5m DEM (int16, compressed)
- **tirol_25m_*.tif**: Medium-resolution files for analysis
- **tirol_100m_*.tif**: Low-resolution files for frontend display
- **tirol_aspect_*.tif**: Terrain aspect analysis
- **tirol_slope_*.tif**: Terrain slope analysis

## Storage Requirements

- **GitHub Pro LFS**: 2GB free storage allocation
- **Local Development**: ~30GB+ for full dataset
- **Production**: Use 100m files for web display, 5m/25m for analysis

## Processing Pipeline

These files are generated from the original DGM using:
1. `projection_resolution_change.sh` - Resampling and format conversion
2. GDAL tools for aspect/slope analysis
3. Compression and web optimization

For processing instructions, see the shell scripts in this directory.
