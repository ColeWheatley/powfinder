# PowFinder AI Coding Guidelines

## Architecture Overview
PowFinder is a ski touring weather visualization app with offline data processing and browser-based rendering. Backend Python scripts collect weather data, perform physics-based interpolation, and generate GeoTIFFs/PNGs. Frontend uses OpenLayers (2D maps) and Three.js (3D hex piston viewer) to display pre-computed tiles from S3.

**Key Components:**
- `backend/`: Python data pipeline (weather collection, TIF generation, terrain processing)
- `frontend/`: Vanilla JS viewers (piston_viewer with Three.js, hexagons with OpenLayers)
- `TIFS/`: Generated visualization assets (weather PNGs, terrain layers)
- `resources/meteo_api/`: Weather data (5,000 coordinates, 81MB JSON)

## Critical Workflows
- **Data Pipeline**: `conda activate powfinder` → Run scripts in `backend/Make TIFs/` (e.g., `python generate_tifs.py`) → Outputs to `TIFS/100m_resolution/[timestamp]/`
- **Frontend Dev**: `python -m http.server 3000` → Access `http://localhost:3000` → Edit `frontend/*/main.js`
- **Deployment**: `./sync_to_s3.sh` → Selective S3 sync excluding large TIFs
- **Validation**: `python debugging/test_peak_temperatures.py` → Compare TIF values vs. API data

## Project Conventions
- **Paths**: Use `Path(__file__).parent` for script-relative paths; project root via `SCRIPT_DIR.parent.parent`
- **Weather Interpolation**: Physics-based (lapse rate -9.8°C/km, hillshade ±1°C, snow cooling -1.5°C) using inverse distance weighting from 4 nearest points
- **Color Scales**: Defined in `backend/Make TIFs/color_scales.json`; temperature range -17.5°C to 25.6°C mapped to 0-255 uint8
- **Tile Structure**: XYZ tiling with WebP textures; binary files for 3D hex data (14 bytes/hex: Z + 6 neighbor heights)
- **Timestamps**: Frontend uses 4 daily periods (09:00, 12:00, 15:00, 18:00); full pipeline generates all 3-hour intervals
- **Dependencies**: Backend uses rasterio/geopandas for GIS, scipy for interpolation; Frontend uses geotiff/node-fetch for dynamic loading

## Common Patterns
- **TIF Generation**: Load weather JSON → Build KDTree for spatial queries → Interpolate grid → Apply color scaling → Write GeoTIFF (see `generate_tifs.py` lines 50-200)
- **3D Hex Geometry**: Flat-topped hexes with 4.33m horizontal gap, 5m vertical spacing; only render S/SE/SW faces for efficiency (see `piston_viewer/main.js` createHexGeometry)
- **Resource Loading**: Progressive texture loading (low → med → high res); cache in THREE.js for performance
- **Error Handling**: Validate API responses; use try/catch for file I/O; log detailed errors to console

## Key Files
- `Powfinder_Readme.txt`: Complete architecture and deployment guide
- `backend/Make TIFs/generate_tifs.py`: Core TIF generation with physics models
- `frontend/piston_viewer/main.js`: Three.js 3D viewer with instanced meshes
- `resources/meteo_api/weather_data_3hour.json`: Source weather dataset (81MB)
- `sync_to_s3.sh`: Production deployment script</content>
<parameter name="filePath">/Users/cole/dev/PowFinder/.github/copilot-instructions.md