# Weather Data Summary

## Overview
This document describes the comprehensive weather dataset collected for PowFinder skiing condition analysis in Tirol, Austria.

## Data Collection Pipeline

### Phase 1: Raw Data Collection (`collect_weather_data.py`)

**API Source:** Open-Meteo API  
**Model:** ICON-D2 (German Weather Service high-resolution model)  
**Date Range:** May 14-28, 2025 (14 days)  
**Timezone:** Europe/Vienna  
**Temporal Resolution:** Hourly data  

**Coordinates Collected:**
- **Total Points:** 5,000 coordinates
- **Peak Points:** 3,000 (highest peaks from OpenStreetMap Tirol)
- **Random Points:** 2,000 (strategically generated above 2,300m elevation)
- **Success Rate:** 100% (5,000 collected, 0 failed)

**Weather Parameters Requested:**
1. `temperature_2m` (°C) - Air temperature at 2 meters
2. `relative_humidity_2m` (%) - Relative humidity at 2 meters  
3. `shortwave_radiation` (W/m²) - Solar radiation
4. `cloud_cover` (%) - Total cloud coverage (sum of low/mid/high cloud layers)
5. `snow_depth` (m) - Snow depth on ground
6. `snowfall` (mm) - Snowfall amount
7. `wind_speed_10m` (m/s) - Wind speed at 10 meters
8. `weather_code` (WMO code) - Weather condition codes
9. `freezing_level_height` (m) - Altitude of 0°C isotherm
10. `surface_pressure` (hPa) - Atmospheric pressure at surface

**Rate Limiting:** 0.5 seconds between requests (7,200 requests/hour target)

### Phase 2: 3-Hour Aggregation (`aggregate_weather_data.py`)

**Purpose:** Convert hourly data to 3-hour periods for terrain analysis integration

**Temporal Structure:**
- **Input:** Hourly data (24 points per day × 14 days = 336 timestamps per coordinate)
- **Output:** 3-hour periods (8 periods per day × 14 days = 112 timestamps per coordinate)
- **Period Alignment:** 00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00

**Aggregation Methods:**
| Parameter | Method | Rationale |
|-----------|---------|-----------|
| `temperature_2m` | Average | Representative temperature over 3-hour period |
| `relative_humidity_2m` | Average | Mean humidity conditions |
| `shortwave_radiation` | Average | Mean solar radiation exposure |
| `cloud_cover` | Average | Mean cloud coverage |
| `snow_depth` | Average | Representative snow depth |
| `snowfall` | **Sum** | Total accumulation over 3 hours |
| `wind_speed_10m` | Average | Mean wind conditions |
| `weather_code` | **Median** | Most representative weather condition |
| `freezing_level_height` | Average | Mean freezing level |
| `surface_pressure` | Average | Mean atmospheric pressure |

**Timestamp Handling:**
- **Design Intent:** Use median times (e.g., 01:30 for 00:00-03:00 period)
- **Actual Implementation:** Period start times (00:00, 03:00, 06:00, etc.)
- **Interpretation:** Each timestamp represents the START of a 3-hour aggregation period

## Dataset Statistics

### Spatial Coverage
- **Coordinate Distribution:** 5,000 points across Tirol
- **Elevation Range:** 2,300m - 3,522m (ski-relevant terrain)
- **Point Density:** Strategic sampling ensuring comprehensive coverage
- **Validation Subset:** Subset of random coordinates flagged for model validation

### Temporal Coverage  
- **Total Duration:** 336 hours (14 days)
- **Aggregated Periods:** 112 three-hour periods per coordinate
- **Start Date:** 2025-05-14T00:00:00
- **End Date:** 2025-05-28T21:00:00

### Data Volume
- **Raw File Size:** 81MB (`weather_data_3hour.json`) - Optimized from original 165MB
- **Backup File Size:** 165MB (`weather_data_3hour_backup.json`) - Original unoptimized version
- **Total Data Points:** 5.6 million (5,000 coordinates × 112 periods × 10 variables)
- **Backup File:** `weather_data_3hour_backup.json` (local safety copy)

## Data Structure

### Top-Level JSON Structure
```json
{
  "metadata": {
    "created_at": 1748799606.665358,
    "total_coordinates": 5000,
    "collected_count": 5000,
    "failed_count": 0,
    "api_config": { ... },
    "aggregation_info": { ... }
  },
  "coordinates": [ ... ]
}
```

### Per-Coordinate Structure
```json
{
  "coordinate_info": {
    "elevation": 2962,
    "type": "peak|random",
    "is_validation": false
  },
  "status": "collected",
  "weather_data_3hour": {
    "hourly": {
      "time": ["2025-05-14T00:00:00", ...],
      "time_units": "3-hour periods",
      "temperature_2m": [-2.43, -3.00, ...],
      "relative_humidity_2m": [89.33, 97.67, ...],
      ...
    }
  }
}
```

## Data Quality & Validation

### Collection Success
- **100% Success Rate:** All 5,000 coordinates successfully collected
- **No Missing Data:** Complete temporal coverage for all points
- **API Reliability:** ICON-D2 model provided consistent high-quality forecasts

### Aggregation Validation
- **Temporal Alignment:** All periods properly aligned to 3-hour boundaries
- **Method Consistency:** Aggregation methods applied uniformly across all coordinates
- **Data Integrity:** No data loss during hourly-to-3hour conversion

### Known Issues
1. **Timestamp Documentation:** Comments mention median times but implementation uses period start times

## Usage in PowFinder Pipeline

### Integration Points
1. **Physics-Based Extrapolation:** Input for spatial weather interpolation
2. **Snow Quality Modeling:** Temperature, snowfall, and wind data for powder assessment  
3. **Terrain Analysis:** Integration with elevation, slope, and aspect data
4. **Visualization:** Direct rendering of weather variables on interactive maps

### Coordinate Categories
- **Peak Points:** Natural ski-touring destinations
- **Random Points:** Comprehensive terrain coverage
- **Validation Points:** Model performance assessment
- **Grid Points:** Systematic spatial sampling

## File Management

### Storage Strategy
- **Local Storage:** Complete dataset maintained locally (81MB active, 165MB backup)
- **Git Tracking:** Main file (81MB) now tracked in Git repository (under GitHub's 100MB limit)
- **Backup Strategy:** Original 165MB backup file maintained locally but excluded from Git
- **Cloud Sync:** Main weather data file now available on GitHub for AI agent access

### Access Patterns
- **Read-Only:** Weather data is immutable once collected
- **Direct Access:** JSON structure allows efficient coordinate/time lookup
- **Streaming:** Large dataset supports streaming processing for memory efficiency

---

*Dataset collected: May-June 2025*  
*Analysis period: May 14-28, 2025*  
*Total processing time: ~45 minutes for 5,000 coordinates*
