# PowFinder - Interactive Ski Touring Weather Visualization

🎿 **Live Demo**: [https://wheatley.cloud/](https://wheatley.cloud/)  
🔧 **Dev Preview**: [https://wheatley.cloud/dev/](https://wheatley.cloud/dev/) (local development iframe)

⚠️ **Note**: Large terrain and weather data files (~32GB+) are excluded from Git repository due to size constraints. The application uses optimized web-friendly assets served from S3.

## 🌟 Project Overview

PowFinder is a high-performance interactive web application that helps ski-tourers identify optimal powder skiing conditions in Tirol, Austria. By integrating precise terrain data with real-time weather forecasts, it provides an intuitive visual interface for exploring skiing conditions across multiple data layers.

### ✨ Key Features

- **🗺️ Interactive Mapping**: Dual-mode visualization (smooth interpolated layers + precise point data)
- **⏰ Real-time Weather**: Auto-loading weather data with 2-second background initialization
- **🏔️ Terrain Analysis**: Elevation, slope, aspect, and hillshade visualization
- **📊 Multi-layer Data**: 16 different weather and terrain parameters
- **⚡ High Performance**: Background PNG preloading and optimized resource management  
- **📱 Responsive Design**: Clean, modern interface with keyboard shortcuts
- **🎯 Precise Validation**: Click anywhere for detailed weather comparisons

## 🚀 Current Status: FULLY OPERATIONAL

✅ **Production Ready**: Complete web application deployed and optimized  
✅ **Weather Data**: Auto-loading 16MB frontend-optimized weather dataset  
✅ **Resource Management**: Clean S3 deployment with optimized asset structure  
✅ **Performance**: Sub-2-second initial load, background data preloading  
✅ **User Experience**: Intuitive controls, keyboard shortcuts, detailed popups  

## 📊 Data Layers

### 🌤️ Weather Parameters (Time-based)
- **Temperature** (°C) - 2m above ground temperature
- **Humidity** (%) - Relative humidity 
- **Radiation** (W/m²) - Shortwave solar radiation
- **Cloud Cover** (%) - Total cloud coverage
- **Snow Depth** (m) - Current snow depth
- **Snowfall** (mm) - Fresh snowfall accumulation
- **Wind Speed** (m/s) - 10m wind speed
- **Weather Code** - Categorical weather conditions
- **Freezing Level** (m) - Height of 0°C isotherm
- **Surface Pressure** (hPa) - Atmospheric pressure
- **Dewpoint** (°C) - Dewpoint temperature

### 🏔️ Terrain Parameters (Static)
- **Elevation** (m) - Digital elevation model
- **Aspect** (°) - Slope direction (compass bearing)
- **Slope** (°) - Slope steepness

### 🎿 Composite Indices (Time-based)
- **Skiability** - Integrated skiing suitability metric
- **SQH** (Snow Quality Heuristic) - Advanced snow quality assessment

## 🏗️ Technical Architecture

### 🌐 Frontend (Production)
- **Framework**: Vanilla JavaScript with OpenLayers 7.4.0
- **Deployment**: AWS S3 + CloudFlare CDN
- **Performance**: 
  - 2-second weather data auto-loading
  - 5-second background PNG preloading
  - Canvas-based image caching system
  - Optimized resource bundling

### 📁 Resource Structure
```
/ (Root - Core web files)
├── index.html              # Main application
├── main.js                 # Application logic  
├── style.css               # Styling
├── favicon.png             # Site icon
└── package.json            # Dependencies

TIFS/100m_resolution/       # Image assets (S3)
├── terrainPNGs/            # Static terrain images
│   ├── elevation.png
│   ├── aspect.png
│   └── slope.png
└── [timestamp]/            # Weather visualization PNGs
    ├── temperature_2m.png
    ├── cloud_cover.png
    └── [other variables].png

resources/                  # Data files (S3)
├── meteo_api/
│   ├── tirol_peaks.geojson        # 3,000 peaks
│   └── weather_data_frontend.json # 16MB weather dataset
└── Make TIFs/
    └── color_scales.json          # Visualization parameters
```

### 💾 Data Pipeline (Development)
- **Python Environment**: `conda activate powfinder` (Python 3.11+)
- **Key Dependencies**: rasterio, geopandas, requests, numpy, scipy
- **Weather Source**: Open-Meteo API with 5,000-point sampling strategy
- **Terrain Processing**: Multi-resolution GeoTIFF generation (5m/25m/100m)
- **Optimization**: Physics-based interpolation with validation system

## 🎮 User Interface

### 🖱️ Controls
- **Layer Selection**: Click any weather/terrain parameter
- **Time Navigation**: Day/time controls with 4 daily timestamps
- **Mode Toggle**: Switch between smooth interpolation and point data
- **Map Interaction**: Click anywhere for detailed weather popup
- **Settings Panel**: Spacebar or drawer icon for layer management

### ⌨️ Keyboard Shortcuts  
- `Space` - Toggle settings panel
- `Escape` - Close popups/panels  
- `↑/↓` - Cycle through layers
- `←/→` - Navigate time
- `Right Shift` - Toggle visualization mode

### 📍 Interactive Features
- **Smart Popups**: Click map for detailed weather comparison
- **API Integration**: Real-time Open-Meteo API calls for validation
- **Peak Detection**: Automatic mountain peak identification
- **Distance Calculation**: Nearest data point with distance measurement
- **Delta Analysis**: Compare interpolated vs. API weather data

## 🗓️ Time Coverage

**Reference Period**: May 24-28, 2025 (5 days)  
**Daily Timestamps**: 09:00, 12:00, 15:00, 18:00 (4 per day)  
**Total Timestamps**: 20 time periods  
**Reference Date**: May 24th, 2025 as "Today"  

## 🚀 Deployment

### Production Deployment
```bash
# Simplified S3 sync (only essential files)
aws s3 sync . s3://wheatley.cloud/ \
  --include "index.html" \
  --include "main.js" \
  --include "style.css" \
  --include "favicon.png" \
  --include "package.json" \
  --include "web-resources/*" \
  --include "dev/*" \
  --exclude "*" \
  --delete
```

### Development Setup
```bash
# Clone repository
git clone https://github.com/ColeWheatley/powfinder.git
cd powfinder

# Switch to development branch
git checkout web-friendly

# Local development server
python -m http.server 3000
# Access: http://localhost:3000
```

## 🛠️ Development Pipeline

### Data Collection (Complete)
```bash
conda activate powfinder
cd resources/meteo_api

# Weather data collection (5,000 coordinates)
python collect_weather_data.py
# Output: weather_data_3hour.json (81MB dataset)
```

### TIF Generation (Complete)
```bash
cd resources/Make\ TIFs

# Generate color scales
python generate_color_scales.py

# Create visualization TIFs  
python generate_tifs_unified.py
# Output: TIFS/100m_resolution/[timestamp]/[variable].png
```

### Terrain Processing (Complete)
```bash
cd resources/terrains

# Multi-resolution DEM processing
python tile_terrain_pngs.py
# Output: terrainPNGs/[elevation|aspect|slope].png
```

## 📈 Performance Metrics

### 🚀 Load Times
- **Initial Page Load**: < 1 second
- **Weather Data Loading**: 2 seconds (16MB dataset)
- **PNG Preloading**: 5 seconds background (260+ images)
- **Map Interaction**: Instant (cached resources)

### 💾 Resource Optimization
- **Frontend Bundle**: ~50KB (HTML/CSS/JS)
- **Weather Dataset**: 16MB (optimized from 81MB source)
- **Image Assets**: Canvas caching with smart preloading
- **S3 Deployment**: Minimal file sync strategy

### 🎯 Data Coverage
- **Weather Points**: 5,000 coordinates (3,000 peaks + 2,000 random)
- **Terrain Resolution**: 100m visualization (source: 5m DEM)
- **Time Resolution**: 3-hour intervals, 4 times daily
- **Spatial Coverage**: Complete Tirol region boundary

## 🔧 Configuration

### Resource Paths (main.js)
```javascript
// All resource paths centrally managed
const RES = './';

// Image assets
TIFS/100m_resolution/[timestamp]/[variable].png    // Weather layers
TIFS/100m_resolution/terrainPNGs/[variable].png    // Terrain layers

// Data files  
resources/meteo_api/tirol_peaks.geojson            // Peak locations
resources/meteo_api/weather_data_frontend.json     // Weather dataset
resources/Make TIFs/color_scales.json              // Visualization config
```

### Color Scales
- **Temperature**: -17.5°C to 25.6°C (blue to red gradient)
- **Elevation**: 0m to 3,800m (green to white gradient)  
- **Wind Speed**: 0-20 m/s (calm to extreme)
- **Snow Depth**: 0-5m (transparent to deep blue)
- **Weather Code**: Categorical (clear/fog/rain/snow/thunder)

## 🐛 Debugging & Validation

### Development Tools
```bash
cd debugging/

# Validate TIF accuracy
python test_peak_temperatures.py

# Query specific temperatures  
python query_temperatures.py
```

### Browser Console
- Weather data loading progress
- PNG preloading statistics  
- Error handling with detailed messages
- Performance timing logs

## 🌟 Key Achievements

✅ **Complete Refactor**: Clean separation of code and data assets  
✅ **Performance Optimization**: 2-second weather data loading  
✅ **Resource Management**: Optimized S3 deployment strategy  
✅ **User Experience**: Intuitive dual-mode visualization  
✅ **Data Pipeline**: Robust 5,000-point weather collection  
✅ **Validation System**: Real-time API comparison functionality  
✅ **Responsive Design**: Modern interface with keyboard shortcuts  
✅ **Production Ready**: Fully deployed and operational  

## 🚀 Future Enhancements

### Planned Features
- **🤖 ML Enhancement**: Machine learning model for improved weather extrapolation
- **📱 Mobile Optimization**: Touch controls and responsive layout improvements
- **🔄 Real-time Updates**: Live weather data integration
- **📊 Advanced Metrics**: Enhanced SQH and skiability calculations
- **🗺️ Extended Coverage**: Additional Austrian regions

### Technical Roadmap
- **⚡ Performance**: WebGL rendering for large datasets
- **🔌 API Enhancement**: Custom weather API endpoints
- **📦 Bundle Optimization**: Code splitting and lazy loading
- **🏗️ Infrastructure**: CDN optimization and edge caching

## 📞 Support & Contributing

**Repository**: [github.com/ColeWheatley/powfinder](https://github.com/ColeWheatley/powfinder)  
**Branch**: `web-friendly` (active development)  
**Issues**: GitHub issue tracker  
**Documentation**: This README + inline code comments  

---

*Last Updated: June 16, 2025*  
*Version: Production Release*  
*Status: ✅ FULLY OPERATIONAL*
* ✅ **Physics Debugging**: Fixed major hillshade normalization bug (corrected from 0-255 to int16 0-32767 range)
* ✅ **Validation Tools**: Peak temperature testing scripts for TIF accuracy validation (moved to `debugging/` folder)
* ✅ **Weather Data Pipeline**: Complete 81MB weather dataset covering 5,000 coordinates (3,000 peaks + 2,000 random points) for 5 days (May 24-28, 2025), 20 timestamps total - Now available on GitHub
* ✅ **Terrain processing pipeline**: Complete DEM processing at 5m, 25m, 100m resolutions with slope/aspect calculations
* ✅ **Hillshade modeling implemented**: Solar illumination calculations at multiple resolutions across 4 time periods
* ✅ **Data management**: Large terrain/shadow files (~32GB) properly excluded from Git repository
* ✅ **File organization**: Weather API and peak data moved to `resources/meteo_api/`, separate directories for hillshade and shadow processing
* ✅ **Weather API system**: Operational Open-Meteo integration with robust resumable collection system (3,000 highest peaks + 2,000 random coordinates)
* ✅ **Random coordinate generation**: 2,000 generated with proximity controls, DEM validation, and reproducible seeding
* 🔄 **Frontend streamlined**: Interactive web map with OpenLayers maintained but simplified for preprocessed data consumption
* 📅 **Demo date set**: Using May 24th, 2025 as "Today" for retroactive skiing condition analysis, covering 5 days through May 28th (20 timestamps total)

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
* **Date Synchronization**: Frontend aligned to May 24th, 2025 as "Today" reference date, covering 5 days (May 24-28) with 20 timestamps total  
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
* ✅ Execute full weather data collection for all 5,000 coordinates (May 24-28, 2025) - Complete with 81MB weather_data_3hour.json covering 5 days and 20 timestamps (optimized from 165MB backup)
* ✅ Implement TIF generation pipeline with physics-based interpolation and color scale management
* ✅ Debug and fix physics calculations (hillshade normalization, lapse rate adjustments)
* ✅ Integrate frontend with consistent date mapping and API calls
* ✅ Create diagnostic tools for TIF validation and temperature accuracy testing
* 🔄 Create snow quality heuristic and skiability calculations

**Pipeline Status**: Fully operational and validated with excellent performance metrics. All scripts implemented and tested on May 23rd, 2025 data with physics-based weather extrapolation generating accurate predictions.

## TIF Generation Pipeline

**Current Implementation**: Simplified physics-based temperature interpolation system integrated into frontend workflow.

**Key Components**:
- **TIF Generation**: `resources/Make TIFs/generate_tifs.py` - Unified generator with caching and SQH integration
- **Color Scale Management**: `resources/Make TIFs/generate_color_scales.py` and `color_scales.json` - Ensures consistent visualization
- **Validation Tools**: `debugging/test_peak_temperatures.py` - Validates TIF accuracy against API weather data

**TIF Generation Workflow**:
```bash
conda activate powfinder  # Python 3.11.12 environment
cd /Users/cole/dev/PowFinder/resources/Make\ TIFs

# Generate color scales (one-time setup)
python generate_color_scales.py

# Generate all GeoTIFF layers
python generate_tifs.py
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
- **Input**: `weather_data_3hour.json` (81MB weather data, available on GitHub)
- **Configuration**: `physics_params.json` (physics model parameters)
- **Task Management**: `task_queue.json` (1,474 processing tasks)
- **Output**: `predictions.csv` (weather predictions), `residuals.csv` (validation data)
- **Validation**: `residual_summary.json` (model performance metrics)