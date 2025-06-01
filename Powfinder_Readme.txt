Powfinder

⚠️ NOTE: Large terrain and shadow data files (~32GB+) are excluded from this Git repository due to size constraints. See resources/README_DATA_FILES.md for setup instructions.

## Repository Structure:
* **Frontend**: `index.html`, `style.css`, `frontend.js` - Interactive web interface
* **Core Services**: `gridService.js` - Data management and grid handling  
* **Meteorological API**: `resources/meteo_api/` - Weather data API and peak locations
* **Hillshade Processing**: `resources/hillshade/render_hillshade.py` - Solar illumination modeling
* **Shadow Processing**: `resources/shadows/render_shadow_map.py` - Binary terrain obstruction mapping
* **Terrain Data**: `resources/terrains/` - Multi-resolution DEM files (excluded from Git)
* **Configuration**: Model configurations and processing parameters

Project Overview and Rationale

This project aims to create a high-quality, interactive visualization tool that helps ski-tourers identify ideal conditions for powder skiing within the region of Tirol, Austria. The central goal is to visually integrate multiple terrain and meteorological factors to provide a quick, intuitive, and detailed reference for skiers, enabling them to pinpoint optimal locations and times for their tours.

We utilize precise terrain data (Digital Elevation Models�DEM) alongside weather forecasts and historical data (primarily from Open-Meteo) to compute derived layers such as slope, aspect, shadow exposure, wind vulnerability, snowfall, and ultimately an index of overall skiability (referred to as the Snow Quality Heuristic, or SQH). Skiablity is SQH adjusted by day-of weather. It�s done by integrating all the weather up to appoint to determine snow quality and depth and then the weather at the point to see if it�s sunny cloud or windy. 

The project�s end-user interface is an interactive web map (built previously with OpenLayers), allowing users to dynamically explore, click to query conditions, and evaluate touring conditions across Tirol with minimal latency.

End Goals

At a high level, the visualization tool should enable:
* Rapid identification of optimal powder-skiing locations and times.
* Intuitive exploration of terrain features including slope steepness, aspect (direction facing), wind vulnerability, and shadow exposure (sunlight exposure).
* Easy access to, current, and short-term future weather conditions relevant to powder skiing.
* Calculation and visualization of integrated skiability metrics.

Practically, we aim for:
* Fluid, responsive performance in a web environment.
* Efficient storage and retrieval of multiple data layers.
* Ability to update forecasts frequently (daily or sub-daily).
* Scalability to add or modify data layers or calculations later.
* Minimal server resource consumption, leveraging precomputed rasters and efficient caching techniques.

Data Layers and Sources

1. Terrain (DEM) Data
* Source: Original 5-meter resolution Digital Elevation Model (DEM) of Tirol (DGM Tirol, EPSG:31254).
* Processed Resolutions: 5m, 20m, 100m  reprojected into web-friendly WGS84 projection.
* Processed Terrain Layers:
o Elevation: Raw elevation data at multiple resolutions.
o Slope: Computed slope angle (in degrees) from elevation at all resolutions.
o Aspect: Computed compass direction slope faces (North, South, etc.) at all resolutions.
o Hillshade: Computed solar illumination (dot product of sun vector and surface normal) at multiple resolutions - models direct sunlight.
o Shadow Maps: Binary terrain obstruction shadows using GRASS GIS r.sun beam radiation - identifies areas blocked by terrain features at specific times.
o Wind Vulnerability: (Next step) Computed via terrain analysis or ray-casting for dominant wind directions (planned at coarser resolution).

2. Weather Data
* Source: Open-Meteo API (forecast and historical weather).
* Variables: Temperature (2m and altitude-adjusted), snowfall, snow depth, wind speed, wind direction, shortwave radiation, relative humidity, dew point, among others.
* Time Resolution: Forecast data at 3-hour intervals, historical data at hourly or 3-hour intervals.
* Spatial Resolution: Forecasts acquired primarily at peak points (high-altitude locations) and selectively extrapolated to surrounding terrain.

3. Derived Metrics (SQH and Skiability)
* Snow Quality Heuristic (SQH): Integrates snowfall, temperature, settling, wind scouring, and solar radiation to approximate snowpack quality and depth.
* Skiability Index: Further integrates day-of conditions (wind, visibility, sunshine) to give a single, intuitive metric for skiing suitability.

Current State of Project (serverside-refactor branch)

## Architecture Refactor Status:
* ✅ **Client-side processing moved to server**: Weather API calls, terrain evaluation, snow quality modeling, and weather extrapolation now handled server-side
* ✅ **Hillshade modeling implemented**: Python scripts for solar illumination (dot product of sun and surface normal) at multiple resolutions (5m, 25m, 100m) across 4 time periods
* ✅ **Shadow mapping system optimized**: GRASS GIS-based binary shadow casting with horizon optimization for ski-relevant terrain analysis at 4 time periods (07:30, 10:30, 13:30, 16:30)
* ✅ **Performance optimization implemented**: Horizon pre-computation with 2.7km distance limiting based on ski-terrain analysis (max shadow distance for 1,768m elevation difference)
* ✅ **Terrain processing pipeline**: Complete DEM processing at 5m, 25m, 100m resolutions with slope/aspect calculations
* ✅ **Data management**: Large terrain/shadow files (~32GB) properly excluded from Git repository
* ✅ **File organization**: Weather API and peak data moved to `resources/meteo_api/`, separate directories for hillshade and shadow processing
* 🔄 **Shadow map production**: Optimized shadow processing currently running with horizon pre-computation for dramatic performance improvement
* 🔄 **Frontend streamlined**: Interactive web map with OpenLayers maintained but simplified for preprocessed data consumption
* 📅 **Demo date set**: Using May 23rd, 2025 for retroactive skiing condition analysis (skiing confirmation: May 24th)

## Immediate Next Steps:
* 🔄 **Shadow map production**: Optimized shadow processing currently running in background with horizon pre-computation and 2.7km distance limiting for 15x performance improvement
* **Performance validation**: Measure actual speedup from horizon optimization once current shadow processing completes
* **Server API development**: Create REST endpoints to serve preprocessed weather and terrain data
* **Data preprocessing pipeline**: Automate weather fetching, extrapolation, and snow quality calculations for MacBook demo
* **Frontend integration**: Update client to consume server-processed data instead of doing calculations in browser

## Longer-Term Goals:
* **MacBook demo optimization**: Performance tuning for local classroom demonstration
* **Advanced terrain analysis**: Wind vulnerability and exposure calculations
* **Enhanced visualization**: Improved map rendering and user interaction features
* **Data validation**: Cross-reference with actual skiing conditions from May 24th field test

Tiling Strategy and Resolution Management

Rationale:

Due to significant differences in required spatial resolution for various terrain features and weather variables, a multi-resolution approach was adopted:
* 5m Resolution: Reserved for high-altitude, steep, skiing-relevant terrain.
* 20m Resolution: Medium-detail data�likely the core spatial resolution for weather and derived products (slope/aspect/shadow).
* 100m Resolution: Low-detail context data�useful for zoomed-out views or flat, non-ski-relevant terrain areas.

Elevation-Based Tile Flagging Strategy:

Tiles at higher resolutions (5m, 50m) are selectively generated only where the lowest elevation within a tile is greater than a chosen threshold (~2300m). This strategy ensures we never serve unnecessary high-resolution data for irrelevant terrain. Tiles flagged as relevant are stored and indexed for fast retrieval.

Future Considerations and Improvements
* Data Update Frequency: Implement automatic daily or sub-daily fetching of fresh weather data from Open-Meteo API. Long term goal. Right now manual fetching.
* Interactive Features: Provide sliders or toggles to dynamically adjust displayed variables or time-frames.
* Advanced Extrapolation Models: Upgrade from basic physics-based extrapolation to more advanced models informed by field observations and ML algorithms.
* Offline and Mobile Support: Consider progressive web app (PWA) features, caching strategies, and efficient data formats (such as Zarr or Cloud-optimized GeoTIFF).

Technical Stack and Current Architecture

## Data Processing (MacBook Local):
* **Python Environment**: `conda activate powfinder` (Python 3.11.12)
* **Key Dependencies**: rasterio, geopandas, pysolar, whitebox, pyproj, matplotlib, grass-session (GRASS GIS)
* **Python/GDAL**: Terrain processing, hillshade generation, raster calculations
* **GRASS GIS**: Binary shadow map generation using r.sun beam radiation analysis
* **Storage**: Multi-resolution GeoTIFF files (5m, 25m, 100m) with optimized projections
* **Processing Architecture**: Separate hillshade (solar illumination) and shadow (terrain obstruction) pipelines with horizon optimization
* **Hillshade System**: Solar illumination modeling using dot product calculations for direct sunlight simulation
* **Shadow System**: Binary terrain obstruction mapping using GRASS GIS r.sun with horizon pre-computation and 2.7km distance limiting for ski-relevant terrain analysis
* **Optimization Implementation**: Horizon calculated once with 15° azimuth steps, then reused across all 4 time periods for dramatic performance improvement
* **Weather Integration**: Open-Meteo API with physics-based extrapolation for May 23rd conditions

## Frontend (Client-Side):
* **Mapping**: OpenLayers for interactive map visualization
* **Grid System**: Unified grid management for efficient data handling
* **Weather API**: Streamlined weather data consumption from `resources/meteo_api/`
* **Caching**: Local storage optimization for map performance

## Development Status:
* **Repository**: Clean separation of large data files (~32GB) from codebase
* **Branching**: serverside-refactor branch ready for MacBook demo deployment
* **Processing Scripts**: Complete terrain and hillshade generation pipeline
* **Demo Date**: May 23rd, 2025 configured for retroactive skiing analysis

Current Action Items (Priority Order):

## Phase 1 - MacBook Demo Infrastructure:
* ✅ Complete terrain data processing pipeline
* ✅ Implement hillshade generation system  
* ✅ Develop binary shadow mapping system (terrain obstruction)
* ✅ Optimize shadow processing with horizon pre-computation and distance limiting
* 🔄 Complete shadow map production for all time periods (currently running optimized process)
* 🔄 Create local API for data serving
* 🔄 Set up May 23rd weather data preprocessing
* 🔄 Configure snow quality calculations for demo date

## Phase 2 - Demo Integration:
* 🔄 Update client to consume preprocessed data
* 🔄 Optimize map rendering for MacBook performance
* 🔄 Validate against May 24th skiing field conditions
* 🔄 Prepare classroom demonstration interface

## Phase 3 - Enhancement Features:
* 🔄 Wind vulnerability analysis integration
* 🔄 Advanced terrain obstruction modeling
* 🔄 Enhanced visualization and interaction
* 🔄 Performance optimization for real-time demo

