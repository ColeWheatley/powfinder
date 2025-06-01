# Shadow Data Files - Git LFS Required

This folder contains shadow analysis and calculation tools for terrain-based shadow mapping.

## File Organization

### 🔒 All Shadow Files Require Git LFS
- `*.tif` - Shadow map raster files (typically 1GB+ each)
- Generated shadow analysis for specific dates/times
- High-resolution shadow casting calculations

### ✅ Python Scripts (Regular Git)
- `render_shadow_map.py` - Main shadow rendering pipeline
- `sun_calc.py` - Solar position calculations

## Git LFS Setup

Shadow files are computationally expensive and large:

```bash
# Install Git LFS
git lfs install

# Track all shadow TIF files
git lfs track "resources/shadows/*.tif"

# Pull existing shadow files
git lfs pull
```

## Shadow Analysis Pipeline

The shadow mapping system consists of:

### 1. Solar Position Calculation (`sun_calc.py`)
- Calculates sun position for any date/time/location
- Accounts for seasonal variations
- Provides azimuth and elevation angles

### 2. Shadow Rendering (`render_shadow_map.py`)
- Uses high-resolution terrain data (5m DEM)
- Calculates shadow casting based on sun position
- Generates binary shadow maps (shadow/no-shadow)
- Exports as compressed GeoTIFF

## File Naming Convention

Shadow files follow this pattern:
```
shadow_YYYYMMDD_HHMM_azimuth_elevation.tif
```

Example: `shadow_20231221_1200_180_30.tif`
- Date: December 21, 2023
- Time: 12:00 (noon)
- Sun azimuth: 180° (due south)
- Sun elevation: 30°

## Technical Specifications

- **Input DEM**: 5m resolution terrain data
- **Output Resolution**: Matches input DEM (5m)
- **Projection**: Web Mercator (EPSG:3857)
- **Data Type**: Boolean (0=shadow, 1=sunlight)
- **Compression**: LZW compression for file size optimization

## Storage Requirements

- **Single shadow map**: ~1-2GB (5m resolution)
- **Daily analysis** (multiple times): ~10-20GB
- **Seasonal analysis**: ~100GB+

## Usage in Analysis Pipeline

Shadow data is used for:
1. **Snow persistence modeling** - Areas in shadow melt slower
2. **Temperature analysis** - Shadow affects local temperatures  
3. **Skiing conditions** - Shadow impacts snow quality
4. **Avalanche risk** - Shadow affects snow stability

## Performance Notes

- Shadow calculation is CPU-intensive
- Use multiprocessing for batch processing
- Consider time-of-day sampling (every 2-4 hours)
- Cache frequently used shadow maps

## Dependencies

```python
# Required packages
rasterio      # Geospatial raster I/O
numpy         # Numerical computations  
pyproj        # Coordinate transformations
astral        # Solar position calculations
gdal          # Geospatial data processing
```

For detailed usage, see the Python scripts in this directory.
