Powfinder

⚠️ NOTE: Large terrain and shadow data files (~32GB+) are excluded from this Git repository due to size constraints. See resources/README_DATA_FILES.md for setup instructions.

📊 **Weather Data:** Comprehensive weather dataset documented in `weather_data_summary.md` - 5,000 coordinates with 14 days of high-resolution meteorological data (165MB total).

🗓️ **Prototype Reference Date:** For development/testing purposes, the application uses **May 25, 2025 at 12:00 PM** as the current "now" reference point. Data from May 14-24 represents historical conditions (for SQH snow quality integration), May 25 evening onwards represents forecast data, and May 26-28 are future forecasts for testing forward prediction capabilities.

## Repository Structure:
* **Frontend**: `index.html`, `style.css`, `main.js` - Interactive web interface with weather point visualization
* **Weather Data**: `weather_data_summary.md` - Complete documentation of 5,000-coordinate weather dataset
* **Meteorological API**: `resources/meteo_api/` - Weather data collection & aggregation scripts  
* **Weather Pipeline**: `resources/pipeline/` - Physics-based extrapolation and analysis tools
* **Hillshade Processing**: `resources/hillshade/render_hillshade.py` - Solar illumination modeling
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
* Processed Resolutions: 5m, 25m, 100m  reprojected into web-friendly WGS84 projection.
* Processed Terrain Layers:
o Elevation: Raw elevation data at multiple resolutions.
o Slope: Computed slope angle (in degrees) from elevation at all resolutions.
o Aspect: Computed compass direction slope faces (North, South, etc.) at all resolutions.
o Hillshade: Computed solar illumination (dot product of sun vector and surface normal) at multiple resolutions - models direct sunlight.
o Shadow Maps: Binary terrain obstruction shadows using GRASS GIS r.sun beam radiation - identifies areas blocked by terrain features at specific times.

2. Weather Data
* Source: Open-Meteo API (forecast and historical weather).
* Variables: temperature_2m, relative_humidity_2m, shortwave_radiation, cloud_cover, snow_depth, snowfall, wind_speed_10m, weather_code, freezing_level_height, surface_pressure
* API Note: `cloud_cover` parameter represents total cloud coverage (sum of low/mid/high cloud layers), labeled as "Cloud Cover Total" in Open-Meteo UI
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
* 🔄 **Weather Data Collection**: Robust resumable collection system ready for full 5,000-coordinate dataset
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

Current State of Project (serverside-refactor branch)

## Architecture Refactor Status:
* ✅ **Client-side processing moved to server**: Weather API calls, terrain evaluation, snow quality modeling, and weather extrapolation now handled server-side
* ✅ **Hillshade modeling implemented**: Python scripts for solar illumination (dot product of sun and surface normal) at multiple resolutions (5m, 25m, 100m) across 4 time periods
* ✅ **Shadow mapping system optimized**: GRASS GIS-based binary shadow casting with horizon optimization for ski-relevant terrain analysis at 4 time periods (07:30, 10:30, 13:30, 16:30)
* ✅ **Performance optimization implemented**: Horizon pre-computation with 2.7km distance limiting based on ski-terrain analysis (max shadow distance for 1,768m elevation difference)
* ✅ **Terrain processing pipeline**: Complete DEM processing at 5m, 25m, 100m resolutions with slope/aspect calculations
* ✅ **Data management**: Large terrain/shadow files (~32GB) properly excluded from Git repository
* ✅ **File organization**: Weather API and peak data moved to `resources/meteo_api/`, separate directories for hillshade and shadow processing
* ✅ **Weather API system**: Operational Open-Meteo integration with robust resumable collection system (3,000 highest peaks + 2,000 random coordinates)
* ✅ **Random coordinate generation**: 2,000 scientifically generated sampling points above 2,300m with proximity controls, DEM validation, and reproducible seeding
* 🔄 **Shadow map production**: Optimized shadow processing currently running with horizon pre-computation for dramatic performance improvement
* 🔄 **Frontend streamlined**: Interactive web map with OpenLayers maintained but simplified for preprocessed data consumption
* 📅 **Demo date set**: Using May 23rd, 2025 for retroactive skiing condition analysis (skiing confirmation: May 24th)

## Immediate Next Steps:
* 🔄 **Shadow map production**: Optimized shadow processing currently running in background with horizon pre-computation and 2.7km distance limiting for 15x performance improvement
* **Performance validation**: Measure actual speedup from horizon optimization once current shadow processing completes
* **Server API development**: Create REST endpoints to serve preprocessed weather and terrain data
* **Data preprocessing pipeline**: Automate weather fetching, extrapolation, and snow quality calculations for MacBook demo
* **Frontend integration**: Update client to consume server-processed data instead of doing calculations in browser


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
* **Weather Integration**: Open-Meteo API with resumable collection system and physics-based extrapolation for May 14-28, 2025 conditions

## Frontend (Client-Side):
* **Mapping**: OpenLayers for interactive map visualization
* **Grid System**: Unified grid management for efficient data handling
* **Weather API**: Streamlined weather data consumption from `resources/meteo_api/`

## Development Status:
* **Repository**: Clean separation of large data files (~32GB) from codebase by excluding TIF files in `.gitignore`
# liberal use of branching to save progressive development states

Current Action Items (Priority Order):

* ✅ Complete terrain data processing pipeline
* ✅ Implement hillshade generation system  
* ✅ Develop binary shadow mapping system (terrain obstruction)
* ✅ Optimize shadow processing with horizon pre-computation and distance limiting
* ✅ Complete weather API debugging and parameter validation
* ✅ Build robust resumable weather collection system
* ✅ Generate 5,000 validated coordinates (3,000 peaks + 2,000 random points)
* ✅ Execute full weather data collection for all 5,000 coordinates (May 14-28, 2025) - Complete with 165MB weather_data_3hour.json
* ✅ Develop physics-based weather extrapolation system (6-script pipeline) - Operational and validated
* ✅ Create progressive grid scheduler for multi-resolution weather mapping - Complete with 1,474 task queue
* ✅ Implement physics-based weather predictions - Generating accurate predictions with excellent validation metrics
* ✅ Build parameter tuning system for physics-based extrapolation - Available via tune_physics_params.py
* 🔄 Implement weather map visualization system (gradient images for all variables) - Ready for interpolate_layers.py
* 🔄 Create local API for data serving
* 🔄 Integrate raw weather data display before derived metrics implementation

Instructions for upcoming 6 script pipeline: 
## Weather-Extrapolation Processing Pipeline (6 fully-interoperable scripts) - ✅ OPERATIONAL

**Pipeline Status**: Fully operational and validated with excellent performance metrics. All scripts implemented and tested on May 23rd, 2025 data with physics-based weather extrapolation generating accurate predictions.

**Performance Validation**: Recent pipeline run shows exceptional accuracy:
- Temperature: 0.43°C Mean Absolute Error
- Snow Depth: 0.01cm Mean Absolute Error  
- Snowfall: 0.00004 Mean Absolute Error
- All weather variables performing within expected tolerances

All scripts live in **`resources/pipeline/`** and communicate **only** via the
explicitly named files below. Every file is JSON, CSV or GeoTIFF so that any
developer can swap scripts without breakage.

| No | Script (CLI entry-point)                | Input(s)                                              | Output(s)                                                        | Purpose (2-line definition)                                                                                                                | Status |
|----|----------------------------------------|-------------------------------------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|---------|
| 1  | `progressive_grid_scheduler.py`        | • `all_points.json`  – array of `{lat,lon,elev,type,is_validation}`<br>• `grid_bounds.json` – `{minLat,maxLat,minLon,maxLon}` | `task_queue.json` – ordered list of tasks:<br>`{"lat":…,"lon":…,"task":"validate" \| "grid_1000" \| "grid_500" \| "grid_100" \| "grid_25"}` | Build a **priority queue**: all validation points first, then 1 km grid, then 500 m, 100 m, 25 m.  Guarantees deterministic order and idempotent reruns. | ✅ Complete |
| 2  | `physics_extrapolate.py`               | • single task (lat,lon) via CLI args **or** stdin JSON<br>• `physics_params.json` (coefficients: lapse_rate, rad_scale, etc.)<br>• `raw_api_index.csv` – rows `{lat,lon,file}` linking pings to JSON<br>• DEM tiles (rasterio readable) | writes **stdout JSON**:<br>`{"lat":…,"lon":…,"variable_dict":{temp_2m:…, …}}` | Find ≤2 nearest non validation API points, apply parameterised physics (elevation lapse, humidity lapse, radiation mask) to predict the ten target variables at the target coordinate. | ✅ Complete |
| 3  | `process_task_queue.py`                | • `task_queue.json`<br>• directory `raw_api/` containing original API JSONs (names from `raw_api_index.csv`) | • `predictions.csv` – every processed task, cols `{lat,lon,time,var1…var10}`<br>• `residuals.csv` – rows for `task=="validate"` with extra `actual_*` and `error_*` columns | Iterates queue.  For each task: calls `physics_extrapolate.py`.  If it is a validation point, loads the matching raw API JSON, calculates residuals, logs both.  Safe to resume (keeps a `.done` log). | ✅ Complete |
| 4  | `analyze_residuals.py`                 | • `residuals.csv`                                     | • `residual_summary.json` (MAE, RMSE per variable & elevation band)<br>• `histogram_errors.png`, `scatter_error_vs_elev.png` | Compute and plot error diagnostics; writes summary JSON for dashboards and optimisation. | ✅ Complete |
| 5  | `tune_physics_params.py`  *(optional)* | • `residuals.csv`<br>• `physics_params.json`           | • overwrites `physics_params.json` with improved coefficients   | Grid-search / optimiser that minimises RMSE on residuals, ready for a second processing pass. | ✅ Available |
| 6  | `interpolate_layers.py`                | • `predictions.csv`  (dense & sparse points)<br>• `high_altitude_mask.tif` (>2 300 m = 1, else 0)<br>• `tirol_boundary.geojson` | • One GeoTIFF **per variable × time slice** (e.g. `t2m_20250523T1200.tif`) at 50 m<br>• Matching colour-mapped PNGs (same name, `.png`) | IDW/RBF interpolation onto a 50 m raster, masked to >2 300 m & Tirol border, then colour-renders each raster.  These images become ready-made map layers. | 🔄 Ready |

**Current Pipeline Data**:
- **Weather Data**: 165MB aggregated weather_data_3hour.json covering 5,000 coordinates
- **Task Queue**: 1,474 processing tasks generated and validated
- **Target Timestamps**: 4 time periods on May 23rd, 2025 (06:00, 09:00, 12:00, 15:00)
- **Predictions Output**: predictions.csv with physics-extrapolated weather data
- **Validation Results**: residuals.csv with excellent model performance metrics

**Data/parameter conventions**

*   All scripts read **`physics_params.json`** (same directory) which defines tunable coefficients:  
    ```json
    {
      "lapse_rate_degC_per_km": -6.5,
      "humidity_lapse_pct_per_km": -5,
      "radiation_clear_fraction": 1.0,
      "snowfall_orographic_factor": 0.1
    }
    ```
    Add new keys freely; every script must ignore unknown keys.

*   Raw API files are one-per-coordinate:<br>
    `raw_api/47.26890_11.40123.json` *(unchanged by pipeline)*

*   DEM is consumed read-only (via rasterio) from `resources/terrains/dem_25m_wgs84.tif`.

Running the full chain in order:

```bash
python progressive_grid_scheduler.py
python process_task_queue.py        # (internally spawns physics_extrapolate)
python analyze_residuals.py
python tune_physics_params.py       # optional
python interpolate_layers.py
```

## Pipeline Workflow and Usage

### Prerequisites
```bash
conda activate powfinder  # Python 3.11.12 environment
cd /Users/cole/dev/PowFinder/resources/pipeline
```

### 1. Check Weather Data Status
```bash
python check_pipeline.py
```
Validates that weather_data_3hour.json is available (165MB, 5,000 coordinates) and displays data summary.

### 2. Generate Task Queue (if needed)
```bash
python progressive_grid_scheduler.py
```
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