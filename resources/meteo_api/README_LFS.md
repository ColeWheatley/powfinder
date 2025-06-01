# Weather Data Collection Pipeline - Git LFS Required

This folder contains the complete meteorological data collection and aggregation pipeline for PowFinder.

## File Organization

### ✅ Scripts and Configuration (Regular Git)
- `generate_random_coordinates.py` - Coordinate generation with validation flags
- `collect_weather_data.py` - Hourly weather data collection from Open-Meteo API
- `aggregate_weather_data.py` - Convert hourly data to 3-hour periods for terrain analysis
- `check_pipeline.py` - Pipeline status validation and monitoring
- `test_aggregation.py` - Testing script for aggregation validation
- `all_points.json` - Unified coordinate file (2000 random + 3000 peaks = 5000 points)
- `tirol_peaks.geojson` - Peak location data from Tirol region
- `dwd_api.pdf` - API documentation

### 🔒 Large Weather Datasets (Git LFS)
- `weather_data_collection.json` - Full hourly weather dataset (~365MB) - **Local only, too large for sharing**
- `weather_data_3hour.json` - Aggregated 3-hour weather data (~165MB) - **Tracked with Git LFS**
- Any additional weather cache files (*.cache, large *.json files) - **Local only**

### Storage Strategy Notes
- The raw hourly data (`weather_data_collection.json`) remains local only due to bandwidth restrictions
- The aggregated 3-hour data (`weather_data_3hour.json`) is shared via Git LFS for pipeline testing
- All scripts and configuration files are tracked in regular Git
- Git LFS enables sharing of the optimized 3-hour dataset without bandwidth issues

## Git LFS Setup

Weather data files are large and require Git LFS:

```bash
# Install Git LFS
git lfs install

# Track weather data files
git lfs track "resources/meteo_api/weather_data_collection.json"
git lfs track "resources/meteo_api/weather_data_3hour.json"
git lfs track "resources/meteo_api/*.cache"

# Pull existing weather data
git lfs pull
```

## Data Collection Pipeline

### 1. Coordinate Generation
```bash
python generate_random_coordinates.py
```
- **Input**: Random coordinate generation + 3000 peaks from tirol_peaks.geojson
- **Output**: `all_points.json` with 5000 coordinates
- **Validation Points**: 200 points (10% of random coordinates) marked for model validation
- **Features**: Elevation lookup, validation flagging, unified format

### 2. Hourly Weather Data Collection
```bash
python collect_weather_data.py
```
- **API**: Open-Meteo weather service (ICON-D2 model)
- **Rate Limit**: 7200 calls/hour with 0.5s delays
- **Parameters**: 
  - temperature_2m, relative_humidity_2m, shortwave_radiation
  - cloud_cover, snow_depth, snowfall, wind_speed_10m
  - weather_code, freezing_level_height, surface_pressure
- **Time Range**: May 14-28, 2025 (configurable)
- **Resume Support**: Can restart interrupted collections
- **Error Handling**: Retries and robust connection management

### 3. Weather Data Aggregation (NEW)
```bash
python aggregate_weather_data.py
```
- **Input**: `weather_data_collection.json` (hourly data)
- **Output**: `weather_data_3hour.json` (3-hour periods)
- **Aggregation Methods**:
  - **Average**: Temperature, humidity, pressure, wind speed, radiation, cloud cover, freezing level, snow depth
  - **Sum**: Snowfall (total accumulation over 3 hours)
  - **Median**: Weather code (most representative condition)
- **Time Periods**: 8 periods per day (00:00-03:00, 03:00-06:00, ..., 21:00-24:00)

### 4. Pipeline Validation
```bash
python check_pipeline.py
```
- **Purpose**: Monitor pipeline status and data integrity
- **Features**: File size checking, coordinate counting, progress tracking
- **Output**: Current pipeline status and next steps
## Weather Data Structure

### Hourly Data Format (`weather_data_collection.json`)
Each coordinate contains hourly weather data:
```json
{
  "coordinate_info": {
    "id": "random_1",
    "name": "Random_Point_1", 
    "latitude": 47.2692,
    "longitude": 11.4041,
    "elevation": 1560,
    "source": "random",
    "is_validation": false
  },
  "status": "collected",
  "weather_data": {
    "hourly": {
      "time": ["2025-05-14T00:00Z", "2025-05-14T01:00Z", ...],
      "temperature_2m": [1.2, 1.1, 1.0, ...],
      "snowfall": [0.0, 0.1, 0.0, ...],
      "weather_code": [0, 1, 2, ...],
      "wind_speed_10m": [3.2, 3.5, 3.1, ...],
      "snow_depth": [45, 45, 46, ...]
    }
  }
}
```

### 3-Hour Aggregated Data Format (`weather_data_3hour.json`)
Same structure but with 3-hour aggregated values:
```json
{
  "weather_data_3hour": {
    "hourly": {
      "time": ["2025-05-14T00:00Z", "2025-05-14T03:00Z", "2025-05-14T06:00Z", ...],
      "temperature_2m": [1.1, 2.3, 4.1, ...],  // Average over 3 hours
      "snowfall": [0.3, 0.0, 1.2, ...],        // Sum over 3 hours  
      "weather_code": [1, 2, 1, ...],           // Median over 3 hours
      "wind_speed_10m": [3.3, 4.1, 2.8, ...]   // Average over 3 hours
    }
  }
}
```

## File Sizes and Storage

| File | Size | Storage | Description |
|------|------|---------|-------------|
| `all_points.json` | ~500KB | GitHub | 5000 coordinate locations and metadata |
| `weather_data_collection.json` | ~365MB | **Local Only** | Full hourly weather dataset (15 days × 24 hours × 5000 points) |
| `weather_data_3hour.json` | ~165MB | **Git LFS** | Aggregated 3-hour periods (15 days × 8 periods × 5000 points) |

### Storage Strategy
- **Large raw datasets** (`weather_data_collection.json`) remain local only - too large for sharing
- **Processed datasets** (`weather_data_3hour.json`) are shared via Git LFS for pipeline testing
- **Configuration files** and scripts are tracked in regular Git

## API Integration

### Open-Meteo API Configuration
- **Endpoint**: `https://api.open-meteo.com/v1/forecast`
- **Model**: ICON-D2 (high-resolution European model)
- **Free Tier**: 10,000 calls/day
- **Rate Limiting**: 7200 calls/hour (0.5s delays)
- **Timezone**: Europe/Vienna

### Weather Parameters Collected
```python
HOURLY_PARAMS = [
    "temperature_2m",           # Air temperature at 2m height
    "relative_humidity_2m",     # Relative humidity at 2m
    "shortwave_radiation",      # Solar radiation
    "cloud_cover",              # Cloud coverage percentage
    "snow_depth",               # Snow depth in cm
    "snowfall",                 # Snowfall amount in cm/hour
    "wind_speed_10m",           # Wind speed at 10m height
    "weather_code",             # Weather condition code
    "freezing_level_height",    # Height of 0°C isotherm
    "surface_pressure"          # Atmospheric pressure
]
```

## Validation Strategy

- **Training Set**: 4800 points (4600 random + 200 validation + 3000 peaks)
- **Validation Set**: 200 random points (flagged with `is_validation: true`)
- **Purpose**: Model performance evaluation and terrain analysis validation
- **Selection**: Every 10th random coordinate is marked for validation

## Usage in Physics Pipeline

The 3-hour aggregated weather data integrates with:

1. **Snow Accumulation Models**: Snowfall sums over 3-hour periods
2. **Temperature Interpolation**: Average temperatures for thermal modeling
3. **Wind Exposure Analysis**: Average wind speeds for exposure calculations  
4. **Avalanche Risk Assessment**: Combined weather parameters for stability analysis
5. **Terrain Analysis**: Aligned with 3-hour terrain processing windows

**Important**: Physics scripts should use `weather_data_3hour.json` for terrain analysis integration, not the hourly data.

## Pipeline Status Monitoring

### Check Pipeline Script
The `check_pipeline.py` script provides:

- **File Status**: Checks existence and sizes of pipeline files
- **Data Validation**: Counts coordinates, validates JSON structure
- **Progress Tracking**: Shows collection/aggregation completion rates
- **Next Steps**: Indicates what pipeline step to run next
- **Error Detection**: Identifies missing or corrupted files

### Example Output:
```
🔍 PowFinder Weather Pipeline Status
=============================================
✅ Coordinate generation
   📁 all_points.json (0.5MB)
   📊 5000 coordinates (200 validation points)
✅ Hourly weather data  
   📁 weather_data_collection.json (330MB)
   📊 4500/5000 coordinates collected
✅ 3-hour aggregated data
   📁 weather_data_3hour.json (220MB) 
   📊 4500/5000 coordinates aggregated
   📊 120 time periods per coordinate

📋 NEXT STEPS: Pipeline complete! Ready for terrain analysis.
```

## Recovery and Resumption

Both collection and aggregation scripts support robust recovery:

```python
# collect_weather_data.py automatically:
# - Detects existing weather_data_collection.json
# - Resumes from last successful coordinate
# - Preserves all previously collected data
# - Handles connection failures gracefully

# aggregate_weather_data.py automatically:
# - Processes all coordinates with collected weather data
# - Skips coordinates with failed/missing data
# - Creates complete aggregated dataset
```

## Dependencies

```bash
# Core Python libraries (included in most installations)
requests      # HTTP API calls to Open-Meteo
json          # Data serialization and file I/O  
statistics    # Median calculation for weather codes
datetime      # Time period processing and timezone handling
pathlib       # File system operations
re            # Regular expressions for datetime parsing
```

## Complete Pipeline Execution

```bash
# Full pipeline from start to finish:

# 1. Generate coordinates (5000 points with validation flags)
python generate_random_coordinates.py

# 2. Collect hourly weather data (may take 45-60 minutes)
python collect_weather_data.py  

# 3. Aggregate to 3-hour periods (takes ~2-3 minutes)
python aggregate_weather_data.py

# 4. Validate pipeline completion
python check_pipeline.py

# Result: weather_data_3hour.json ready for terrain analysis
```
