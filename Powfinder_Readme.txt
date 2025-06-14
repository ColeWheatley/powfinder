Powfinder

⚠️ NOTE: Large terrain and shadow data files (~32GB+) are excluded from this Git repository due to size constraints. See resources/README_DATA_FILES.md for setup instructions.

## Repository Structure:
* **Frontend**: `index.html`, `style.css`, `main.js` - Interactive web interface with OpenLayers
* **TIF Generation**: `resources/Make TIFs/` - Temperature interpolation and color scale management
* **Meteorological API**: `resources/meteo_api/` - Weather data API and peak locations (GeoJSON format from OpenStreetMap)
* **Hillshade Processing**: `resources/hillshade/render_hillshade.py` - Solar illumination modeling
* **Shadow Processing**: `resources/shadows/render_shadow_map.py` - Binary terrain obstruction mapping
* **Terrain Data**: `resources/terrains/` - Multi-resolution DEM files (excluded from Git)
* **Pipeline Utilities**: `resources/pipeline/` - Weather processing utilities
* **Debugging Tools**: `debugging/` - Temperature validation and diagnostic scripts

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
* Processed Resolutions: 5m, 25m, 100m  reprojected into web-friendly WGS84 projection.
* Processed Terrain Layers:
o Elevation: Raw elevation data at multiple resolutions.
o Slope: Computed slope angle (in degrees) from elevation at all resolutions.
o Aspect: Computed compass direction slope faces (North, South, etc.) at all resolutions.
o Hillshade: Computed solar illumination (dot product of sun vector and surface normal) at multiple resolutions - models direct sunlight.

2. Weather Data
* Source: Open-Meteo API (forecast and historical weather).
* Variables: temperature_2m, relative_humidity_2m, shortwave_radiation, cloud_cover, snow_depth, snowfall, wind_speed_10m, weather_code, freezing_level_height, surface_pressure
* Time Resolution: Hourly API data averaged over 3-hour periods to match shadow time periods (07:30, 10:30, 13:30, 16:30)
* Spatial Resolution: Forecasts acquired at peak points (high-altitude locations) and random terrain sampling, then extrapolated to surrounding terrain.
* Date Range: May 14-28, 2025 (5 days forward from May 23rd for realistic forecast duration, then 14 days back due to API historical limits)

## Weather API Strategy and Implementation:
### Comprehensive Sampling System:
* **Peak-Based Sampling**: 3,000 highest peaks from OpenStreetMap `tirol_peaks.geojson` provide natural ski-touring locations
* **Random Terrain Sampling**: 2,000 scientifically generated random coordinates covering Tirol's DEM boundaries
* **Combined Coverage**: Total of 5,000 strategic sampling points for comprehensive weather monitoring
* **API Allocation**: 5,000 calls/hour limit allows strategic sampling of most relevant locations

### Random Coordinate Generation:
* **Elevation Filtering**: All 2,000 random points above 2,300m elevation (ski-relevant terrain)
* **Proximity Control**: Minimum 250m separation between points to avoid redundancy
* **Grid Alignment**: Coordinates snapped to 5m DEM grid for precise terrain matching
* **Boundary Validation**: All points guaranteed within Tirol DEM boundaries
* **Quality Metrics**: Average elevation 2,634m, range 2,300-3,522m, 9.7% generation success rate

### API Implementation Status:
* ✅ **Weather API Operational**: Successfully tested and debugged Open-Meteo integration with robust resumable collection system
* ✅ **Data Collection Infrastructure**: Comprehensive weather collection script with retry logic and internet interruption handling
* ✅ **Peak Data Enhanced**: Using 3,000 highest peaks from comprehensive OpenStreetMap dataset
* ✅ **Random Coordinates Generated**: 2,000 scientific sampling points with elevation/proximity controls and reproducible seeding
* ✅ **Coordinate Validation**: All 5,000 coordinates validated and ready for weather data collection
* ✅ **API Rate Management**: Confirmed 5,000 calls/hour allocation with optimized rate limiting (0.5s delays)
* ✅ **Weather Data Collection**: Robust resumable collection system ready for full 5,000-coordinate dataset
* 🔄 **Physics Extrapolation Pipeline**: Weather extrapolation system development for comprehensive coverage
* 🔄 **Server API Development**: REST endpoints for efficient weather data serving

### Technical Details:
* **File Locations**: 
  - Peak data: `resources/meteo_api/tirol_peaks.geojson` (3,000 highest peaks)
  - Random coordinates: `resources/meteo_api/random_coordinates.json` (2,000 points)
  - Weather collection: `resources/meteo_api/collect_weather_data.py` (resumable collection system)
* **Coordinate Generation**: Python script with GDAL/OSR coordinate transformation, DEM validation, and reproducible seeding (RANDOM_SEED = 42069)
* **Weather Parameters**: temperature_2m, relative_humidity_2m, shortwave_radiation, cloud_cover, snow_depth, snowfall, wind_speed_10m, weather_code, freezing_level_height, surface_pressure
* **Time Coverage**: May 14-28, 2025 (5 days forward + 14 days back from May 23rd analysis date due to API forecast/historical limits)

3. Derived Metrics (SQH and Skiability)
* Snow Quality Heuristic (SQH): Integrates snowfall, temperature, settling, wind scouring, and solar radiation to approximate snowpack quality and depth. (Implementation after raw weather data displays successfully)
* Skiability Index: Further integrates day-of conditions (wind, visibility, sunshine) to give a single, intuitive metric for skiing suitability. (Implementation after raw weather data displays successfully)

Current State of Project (prototyping branch)

## TIF Generation and Visualization Pipeline:
* ✅ **Comprehensive TIF Generation**: Complete suite of generation scripts for all weather variables (temperature_2m, relative_humidity_2m, shortwave_radiation, cloud_cover, snowfall, freezing_level_height, surface_pressure, weather_code, dewpoint_2m)
* ✅ **Consistent Variable Naming**: All TIF output files and scripts renamed to match color_scales.json variable names for frontend compatibility
* ✅ **Physics-Based Processing**: Temperature TIFs include sophisticated physics (hillshade normalization, lapse rate calculations, snow effects) - other variables use simplified interpolation
* ⚠️ **Physics Model Status**: Temperature TIF generation uses validated physics model; other weather variables use basic interpolation (physics models may need refinement)
* ✅ **Color Scale Management**: Consistent visualization ranges across all weather variables including estimated dewpoint range (-20°C to 20°C)
* ✅ **Frontend Integration**: Point-based weather visualization with OpenLayers, synchronized to May 24th, 2025 reference date
* ✅ **API Integration**: Direct Open-Meteo API calls for arbitrary map locations with proper parameter handling
* ✅ **Physics Debugging**: Fixed major hillshade normalization bug (corrected from 0-255 to int16 0-32767 range)
* ✅ **Validation Tools**: Peak temperature testing scripts for TIF accuracy validation (moved to `debugging/` folder)
* ✅ **Weather Data Pipeline**: Complete 165MB weather dataset covering 5,000 coordinates (3,000 peaks + 2,000 random points)
* ✅ **Terrain processing pipeline**: Complete DEM processing at 5m, 25m, 100m resolutions with slope/aspect calculations
* ✅ **Hillshade modeling implemented**: Solar illumination calculations at multiple resolutions across 4 time periods
* ✅ **Data management**: Large terrain/shadow files (~32GB) properly excluded from Git repository
* ✅ **File organization**: Weather API and peak data moved to `resources/meteo_api/`, separate directories for hillshade and shadow processing
* ✅ **Weather API system**: Operational Open-Meteo integration with robust resumable collection system (3,000 highest peaks + 2,000 random coordinates)
* ✅ **Random coordinate generation**: 2,000 generated with proximity controls, DEM validation, and reproducible seeding
* 🔄 **Frontend streamlined**: Interactive web map with OpenLayers maintained but simplified for preprocessed data consumption
* 📅 **Demo date set**: Using May 24th, 2025 for retroactive skiing condition analysis not the currentdate or the beginning of the data

## Immediate Next Steps:
* **Snow Quality Metrics**: Implement SQH and skiability calculations using validated temperature TIFs
* **Enhanced Visualization**: Optional smooth interpolation layer overlay for spatial context



Tiling Strategy and Resolution Management

Rationale:

Due to significant differences in required spatial resolution for various terrain features and weather variables, a multi-resolution approach was adopted:
* 5m Resolution: Reserved for high-altitude, steep, skiing-relevant terrain.
* 25m Resolution: Medium-detail data�likely the core spatial resolution for weather and derived products (slope/aspect/shadow).
* 100m Resolution: Low-detail context data�useful for zoomed-out views or flat, non-ski-relevant terrain areas.

Elevation-Based Tile Flagging Strategy:
Tiles at higher resolutions (5m, 25m) are selectively generated only where the lowest elevation within a tile is greater than a chosen threshold (~2300m). This strategy ensures we never serve unnecessary high-resolution data for irrelevant terrain. Tiles flagged as relevant are stored and indexed for fast retrieval.

Future Considerations and Improvements

* Advanced Extrapolation Models: Upgrade from basic physics-based extrapolation to more advanced models informed by field observations and ML algorithms. We will eventually use the residuals and reserved validaiton points to build a machine learning model. 
* Possibly long term integrating real past readings from weather stations to improve the accuracy of the extrapolation from the nueral network beyond that of ICON model

Technical Stack and Current Architecture

## Data Processing (MacBook Local):
* **Python Environment**: `conda activate powfinder` (Python 3.11.12)
* **Key Dependencies**: rasterio, geopandas, pysolar, whitebox, pyproj, matplotlib, grass-session, requests, scipy, scikit-learn
* **Python/GDAL**: Terrain processing, hillshade generation, raster calculations
* **GRASS GIS**: Binary shadow map generation using r.sun beam radiation analysis
* **Storage**: Multi-resolution GeoTIFF files (5m, 25m, 100m) with optimized projections
* **Processing Architecture**: Separate hillshade (solar illumination) and shadow (terrain obstruction) pipelines with horizon optimization
* **Hillshade System**: Solar illumination modeling using dot product calculations for direct sunlight simulation
* **Shadow System**: Binary terrain obstruction mapping using GRASS GIS r.sun with horizon pre-computation and 2.7km distance limiting for ski-relevant terrain analysis
* **Optimization Implementation**: Horizon calculated once with 15° azimuth steps, then reused across all 4 time periods for dramatic performance improvement
## Frontend (Client-Side):
* **Mapping**: OpenLayers for interactive point-based weather visualization
* **Date Synchronization**: Frontend aligned to May 24th, 2025 as "Today" reference date  
* **API Integration**: Direct Open-Meteo API calls for arbitrary map locations with consistent parameter handling
* **Popup Enhancement**: Improved weather data display with proper units, labels, and formatting

## Development Status:
* **Repository**: Clean separation of large data files (~32GB) from codebase with proper .gitignore
* **Code Organization**: Vestigial pipeline scripts removed, debugging tools moved to dedicated folder
* **Branch Strategy**: Using prototyping branch for active TIF generation and visualization development

Current Action Items (Priority Order):

* ✅ Complete terrain data processing pipeline
* ✅ Implement hillshade generation system with proper int16 normalization
* ✅ Complete weather API debugging and parameter validation
* ✅ Build robust resumable weather collection system
* ✅ Generate 5,000 validated coordinates (3,000 peaks + 2,000 random points)
* ✅ Execute full weather data collection for all 5,000 coordinates (May 14-28, 2025) - Complete with 165MB weather_data_3hour.json
* ✅ Implement TIF generation pipeline with physics-based interpolation and color scale management
* ✅ Debug and fix physics calculations (hillshade normalization, lapse rate adjustments)
* ✅ Integrate frontend with consistent date mapping and API calls
* ✅ Create diagnostic tools for TIF validation and temperature accuracy testing
* 🔄 Create snow quality heuristic and skiability calculations

**Pipeline Status**: Fully operational and validated with excellent performance metrics. All scripts implemented and tested on May 23rd, 2025 data with physics-based weather extrapolation generating accurate predictions.

## TIF Generation Pipeline

**Current Implementation**: Simplified physics-based temperature interpolation system integrated into frontend workflow.

**Key Components**:
- **TIF Generation**: `resources/Make TIFs/generate_t2m_tifs.py` - Creates temperature TIFs with physics-based interpolation
- **Color Scale Management**: `resources/Make TIFs/generate_color_scales.py` and `color_scales.json` - Ensures consistent visualization
- **Validation Tools**: `debugging/test_peak_temperatures.py` - Validates TIF accuracy against API weather data

**TIF Generation Workflow**:
```bash
conda activate powfinder  # Python 3.11.12 environment
cd /Users/cole/dev/PowFinder/resources/Make\ TIFs

# Generate color scales (one-time setup)
python generate_color_scales.py

# Generate temperature TIFs for all timestamps
python generate_t2m_tifs.py
```

**Technical Details**:
- **Physics Model**: Variable lapse rate (-9.8 + humidity adjustment), hillshade illumination (±1°C), snow cooling (-1.5°C)
- **Interpolation**: Inverse distance weighting using up to 4 nearest weather points from 5,000-coordinate dataset
- **Output Format**: uint8 TIFs (0-255) representing temperature range -17.5°C to 25.62°C
- **Validation**: Peak temperature testing shows excellent accuracy after hillshade normalization fixes

**Data Flow**:
1. Weather data: `weather_data_3hour.json` (5,000 coordinates)
2. Terrain data: `tirol_100m_float.tif` + hillshade TIFs  
3. Physics interpolation: Generate temperature values for each grid cell
4. Color scaling: Convert temperatures to 0-255 range for consistent visualization
5. Output: TIFs stored in `TIFS/100m_resolution/<timestamp>/temperature_2m.tif`
Creates task_queue.json with 1,474 processing tasks in priority order (validation points first, then progressive grid densification).

### 3. Execute Main Pipeline
```bash
python process_task_queue.py
```
**Main orchestrator script** that:
- Processes each task in task_queue.json
- Calls physics_extrapolate.py for weather predictions  
- Generates predictions.csv with extrapolated weather data
- Creates residuals.csv with validation metrics for model performance
- Handles 4 target timestamps: 06:00, 09:00, 12:00, 15:00 on May 23rd, 2025
- Safe to resume (maintains .done log for interrupted runs)

### 4. Analyze Model Performance
```bash
python analyze_residuals.py
```
Generates validation dashboard:
- residual_summary.json with MAE/RMSE statistics
- histogram_errors.png showing error distributions
- scatter_error_vs_elev.png for elevation-based analysis

### 5. Parameter Optimization (Optional)
```bash
python tune_physics_params.py
```
Optimizes physics_params.json coefficients using grid search to minimize validation RMSE.

### 6. Generate Map Layers (Ready for Implementation)
```bash
python interpolate_layers.py
```
Creates GeoTIFF and PNG map layers for web visualization (50m resolution, Tirol boundary).

### Key Pipeline Files
- **Input**: `weather_data_3hour.json` (165MB weather data)
- **Configuration**: `physics_params.json` (physics model parameters)
- **Task Management**: `task_queue.json` (1,474 processing tasks)
- **Output**: `predictions.csv` (weather predictions), `residuals.csv` (validation data)
- **Validation**: `residual_summary.json` (model performance metrics)