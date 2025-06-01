# Weather Data Collection - Git LFS Required

This folder contains meteorological data collection scripts and large weather datasets.

## File Organization

### ✅ Scripts and Configuration (Regular Git)
- `collect_weather_data.py` - Main weather data collection script
- `generate_random_coordinates.py` - Coordinate generation with validation flags
- `all_points.json` - Unified coordinate file (5000 points)
- `tirol_peaks.geojson` - Peak location data
- `debug_*.py` - Debug and testing scripts
- `dwd_api.pdf` - API documentation

### 🔒 Large Weather Datasets (Git LFS Required)
- `weather_data_collection.json` - Full weather dataset (~268MB)
- Any additional weather cache files (*.cache, *.json)

## Git LFS Setup

Weather data files can become very large with historical data:

```bash
# Install Git LFS
git lfs install

# Track weather data files
git lfs track "resources/meteo_api/weather_data_collection.json"
git lfs track "resources/meteo_api/*.cache"

# Pull existing weather data
git lfs pull
```

## Data Collection Pipeline

### 1. Coordinate Generation
- **Input**: 3000 peak coordinates + 2000 random coordinates
- **Output**: `all_points.json` with validation flags
- **Validation Points**: 200 points marked for model validation

### 2. Weather Data Collection
- **API**: Open-Meteo weather service
- **Rate Limit**: 5000 calls/hour (1 call/coordinate)
- **Parameters**: Temperature, precipitation, wind, snow depth
- **Time Range**: Configurable historical periods

### 3. Data Processing
- **Resume Support**: Can restart interrupted collections
- **Error Handling**: Retries and connection management  
- **Data Validation**: Checks for missing/invalid responses

## Weather Data Structure

Each coordinate point contains:
```json
{
  "id": "unique_identifier",
  "latitude": 47.2692,
  "longitude": 11.4041,
  "elevation": 1560,
  "type": "peak|random",
  "is_validation": true|false,
  "weather_data": {
    "dates": ["2023-01-01", "2023-01-02", ...],
    "temperature": [1.2, 2.1, ...],
    "precipitation": [0.0, 1.5, ...],
    "wind_speed": [3.2, 4.1, ...],
    "snow_depth": [45, 47, ...]
  }
}
```

## File Sizes and Storage

| File Type | Size Range | Description |
|-----------|------------|-------------|
| Coordinates | ~500KB | Point locations and metadata |
| Weather Data | 50-300MB | Full historical weather dataset |
| Cache Files | 1-10MB | Temporary processing files |

## API Integration

### Open-Meteo API
- **Endpoint**: `https://api.open-meteo.com/v1/forecast`
- **Free Tier**: 10,000 calls/day
- **Parameters**: 
  - Historical weather data
  - Multiple meteorological variables
  - Hourly and daily aggregations

### Rate Limiting
```python
# Built-in rate limiting in collect_weather_data.py
RATE_LIMIT = 5000  # calls per hour
DELAY_BETWEEN_CALLS = 0.72  # seconds (3600/5000)
```

## Validation Strategy

- **Training Set**: 4800 points (96%)
- **Validation Set**: 200 points (4%)
- **Purpose**: Model performance evaluation
- **Selection**: First 200 random points flagged

## Usage in Physics Pipeline

Weather data feeds into:
1. **Snow accumulation models**
2. **Temperature interpolation** 
3. **Wind exposure analysis**
4. **Precipitation patterns**
5. **Avalanche risk assessment**

## Recovery and Resumption

The collection script supports resuming interrupted downloads:
```python
# Automatically detects existing data
# Resumes from last successful coordinate
# Preserves all previously collected data
```

## Dependencies

```python
requests      # HTTP API calls
json          # Data serialization
time          # Rate limiting
sys           # Progress tracking
```

For detailed usage, see the Python scripts in this directory.
