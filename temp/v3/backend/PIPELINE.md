# PowFinder Data Pipeline - NumPy Optimized

## Overview

Fast, efficient weather data processing pipeline for Tirol ski forecasting.

```
Open-Meteo API
     ↓
collect_weather_data.py (7000 coords, 584MB JSON)
     ↓
[✨ NUMPY-OPTIMIZED] aggregate_weather_data.py (47.6 seconds, 10-core parallel)
     ↓
weather_data_3hour.json (264MB, 112 periods per coordinate)
     ↓
train_and_generate.py (LightGBM interpolation)
     ↓
TIFS/100m_resolution_ml/ (interpolated weather TIFs)
     ↓
render_webps.py (direct TIF → WebP, 60-70% quality)
     ↓
frontend/web-resources/images/ (WebP tiles)
     ↓
S3 → wheatley.cloud
```

## Key Optimizations

### Aggregation: 20x Faster! ⚡

**Before (aggregate_weather_data_original_slow.py):**
- Single-threaded
- Triple-nested Python loops
- ~16 minutes for 7000 coordinates

**After (aggregate_weather_data.py - NumPy version):**
- 10-core parallelization with multiprocessing.Pool
- Vectorized aggregation using NumPy
- **47.6 seconds** for 7000 coordinates
- **20x speedup**

### Implementation Details

```python
# Parallelization strategy
with mp.Pool(10) as pool:
    processed = list(pool.imap(process_single_coordinate, coords, chunksize=32))

# Vectorized operations
hourly_datetimes = np.array([...], dtype=object)
mask = (hourly_datetimes >= start) & (hourly_datetimes < end)
agg_values = np.nanmean(hourly_values[mask])  # No loops!
```

**Memory**: 14GB peak (loading full dataset into arrays)
**CPU**: 98%+ utilization across all cores

## Pipeline Commands

```bash
# 1. Collect fresh weather data (interactive, ~3-5 min)
cd backend/meteo_api
python3 collect_weather_data.py

# 2. Aggregate to 3-hour periods (FAST: 47.6 seconds)
python3 aggregate_weather_data.py
# Outputs: weather_data_3hour.json (264MB)

# 3. Train LightGBM interpolation model
cd ../ml_interpolator
python3 train_and_generate.py
# Outputs: TIFS/100m_resolution_ml/<timestamp>/*.tif

# 4. Render TIFs directly to WebP (no PNG intermediate)
cd ../Make\ TIFs
python3 render_webps.py
# Outputs: frontend/web-resources/images/weather/<timestamp>/*.webp

# 5. Deploy to S3
cd ../../
./sync_to_s3.sh
```

## File Locations

| File | Size | Purpose |
|------|------|---------|
| `meteo_api/weather_data_collection.json` | 584MB | Raw hourly data from Open-Meteo |
| `meteo_api/weather_data_3hour.json` | 264MB | Aggregated 3-hour periods |
| `TIFS/100m_resolution_ml/` | ~1GB | Interpolated weather TIFs |
| `frontend/web-resources/images/` | ~500MB | WebP tiles for browser |

## Performance Benchmarks

```
Aggregation (7000 coords × 13 vars × 112 periods):
  NumPy:     47.6 seconds (10-core)
  Original:  ~960 seconds (single-thread)
  Speedup:   20.2x

LightGBM Training (7 variables):
  Est. ~4-10 minutes

WebP Rendering (56 timestamps × 12 vars + terrain):
  Est. ~5-10 minutes

Total Pipeline:
  ~15-30 minutes (was 60+ minutes)
```

## Configuration

### Aggregation
- **Cores**: 10 (adjust `NUM_CORES` in script)
- **Chunksize**: 32 (batch size per worker)
- **Method**: Multiprocessing.Pool with imap

### LightGBM
- **Variables**: temperature, humidity, radiation, cloud_cover, snowfall, wind_speed, freezing_level
- **Model**: Gradient Boosting Decision Trees
- **Rounds**: 100 per variable per timestamp
- **Learning Rate**: 0.1

### WebP Rendering
- **Quality**: 65% (60-70% compression)
- **Method**: Direct TIF → PIL Image → WebP (no PNG intermediate)

## Notes

- Keep `aggregate_weather_data_original_slow.py` as reference/backup
- NumPy version uses structured arrays for memory efficiency
- Multiprocessing spawns 10 worker processes; monitor memory on load
- WebP format reduces bandwidth by ~60% vs PNG while maintaining visual quality

## Future Improvements

- [ ] GPU acceleration for TIF processing
- [ ] Incremental aggregation (only process new data)
- [ ] Caching of static terrain features
- [ ] Distributed processing across multiple machines
