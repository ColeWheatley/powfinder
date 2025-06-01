#!/bin/bash

# Input file
INPUT_TIF="resources/terrains/DGM_Tirol_5m_epsg31254_2006_2020.tif"

# Output directory
OUTPUT_DIR="resources/terrains"
mkdir -p "$OUTPUT_DIR"

# Target projection (Web Mercator)
TARGET_EPSG="EPSG:3857"

echo "Processing Tirol DEM to 3 resolutions..."

# 1. High-precision 5m (local use only - terrain calculations)

if [ ! -f "$OUTPUT_DIR/tirol_5m_float.tif" ]; then
    echo "Creating high-precision 5m resolution (no compression, PREDICTOR=1)..."
    gdalwarp -t_srs "$TARGET_EPSG" \
             -tr 5 5 \
             -r bilinear \
             -ot Float32 \
             -co COMPRESS=NONE \
             -co PREDICTOR=1 \
             "$INPUT_TIF" \
             "$OUTPUT_DIR/tirol_5m_float.tif"
fi

# 1. Low-precision 5m (web delivery, UInt16)
if [ ! -f "$OUTPUT_DIR/tirol_5m_web.tif" ]; then
    echo "Creating low-precision 5m resolution..."
    # First reproject and resample
    gdalwarp -t_srs "$TARGET_EPSG" \
             -tr 5 5 \
             -r average \
             -ot Float32 \
             "$INPUT_TIF" \
             "$OUTPUT_DIR/temp_5m.tif"

    # Then scale to full 8-bit range and convert to Byte
    gdal_translate -ot Byte \
                   -scale 0 4096 0 255  \
                   -co COMPRESS=DEFLATE \
                   -co PREDICTOR=2 \
                   "$OUTPUT_DIR/temp_5m.tif" \
                   "$OUTPUT_DIR/tirol_5m_web.tif"

    # Clean up temp file
    rm "$OUTPUT_DIR/temp_5m.tif"
else
    echo "Low-precision 5m already exists, skipping..."
fi

# 2. 25m full-precision (for analysis)
if [ -f "$OUTPUT_DIR/tirol_25m_float.tif" ]; then
    echo "25m full-precision already exists, skipping..."
else
    echo "Creating 25m full-precision Float32..."
    gdalwarp -t_srs "$TARGET_EPSG" \
             -tr 25 25 \
             -r average \
             -ot Float32 \
             -co COMPRESS=NONE \
             -co PREDICTOR=1 \
             "$INPUT_TIF" \
             "$OUTPUT_DIR/tirol_25m_float.tif"
fi

# 2b. Low-precision 25m (web delivery, Byte, 8-bit)
if [ ! -f "$OUTPUT_DIR/tirol_25m_web.tif" ]; then
    echo "Creating low-precision 25m resolution (8-bit Byte)..."
    # First reproject and resample
    gdalwarp -t_srs "$TARGET_EPSG" \
             -tr 25 25 \
             -r average \
             -ot Float32 \
             "$INPUT_TIF" \
             "$OUTPUT_DIR/temp_25m.tif"

    # Then scale to 8-bit range and convert to Byte
    gdal_translate -ot Byte \
                   -scale 0 4096 0 255 \
                   -co COMPRESS=DEFLATE \
                   -co PREDICTOR=2 \
                   "$OUTPUT_DIR/temp_25m.tif" \
                   "$OUTPUT_DIR/tirol_25m_web.tif"

    # Clean up temp file
    rm "$OUTPUT_DIR/temp_25m.tif"
else
    echo "Low-precision 25m already exists, skipping..."
fi

# 3. 100m full-precision (for analysis)
if [ -f "$OUTPUT_DIR/tirol_100m_float.tif" ]; then
    echo "100m full-precision already exists, skipping..."
else
    echo "Creating 100m full-precision Float32..."
    gdalwarp -t_srs "$TARGET_EPSG" \
             -tr 100 100 \
             -r average \
             -ot Float32 \
             -co COMPRESS=NONE \
             -co PREDICTOR=1 \
             "$INPUT_TIF" \
             "$OUTPUT_DIR/tirol_100m_float.tif"
fi

# 3b. Low-precision 100m (web delivery, Byte, 8-bit)
if [ ! -f "$OUTPUT_DIR/tirol_100m_web.tif" ]; then
    echo "Creating low-precision 100m resolution (8-bit Byte)..."
    # First reproject and resample
    gdalwarp -t_srs "$TARGET_EPSG" \
             -tr 100 100 \
             -r average \
             -ot Float32 \
             "$INPUT_TIF" \
             "$OUTPUT_DIR/temp_100m.tif"

    gdal_translate -ot Byte \
                   -scale 0 4096 0 255 \
                   -co COMPRESS=DEFLATE \
                   -co PREDICTOR=2 \
                   "$OUTPUT_DIR/temp_100m.tif" \
                   "$OUTPUT_DIR/tirol_100m_web.tif"

    # Clean up temp file
    rm "$OUTPUT_DIR/temp_100m.tif"
else
    echo "Low-precision 100m already exists, skipping..."
fi

# 4. Generate slope and aspect from high-precision 5m data
echo ""
echo "Generating slope and aspect data..."

# 5m slope (temporary, high precision)
if [ -f "$OUTPUT_DIR/temp_slope_5m.tif" ]; then
    echo "5m slope already exists, skipping..."
else
    echo "Creating 5m slope..."
    gdaldem slope "$OUTPUT_DIR/tirol_5m_float.tif" \
                  "$OUTPUT_DIR/temp_slope_5m.tif" \
                  -co COMPRESS=DEFLATE \
                  -co PREDICTOR=3
fi

# 5m aspect (temporary, high precision)
if [ -f "$OUTPUT_DIR/temp_aspect_5m.tif" ]; then
    echo "5m aspect already exists, skipping..."
else
    echo "Creating 5m aspect..."
    gdaldem aspect "$OUTPUT_DIR/tirol_5m_float.tif" \
                   "$OUTPUT_DIR/temp_aspect_5m.tif" \
                   -co COMPRESS=DEFLATE \
                   -co PREDICTOR=3
fi

# Downscale slope to 25m (0-90 degrees, use 8-bit)
if [ ! -f "$OUTPUT_DIR/tirol_slope_25m_web.tif" ]; then
    echo "Creating 25m slope for web..."
    gdal_translate -ot Byte \
                   -scale 0 90 0 255 \
                   -co COMPRESS=DEFLATE \
                   -co PREDICTOR=2 \
                   "$OUTPUT_DIR/temp_slope_5m.tif" \
                   "$OUTPUT_DIR/temp_slope_25m_scaled.tif"
    
    gdalwarp -tr 25 25 \
             -r average \
             "$OUTPUT_DIR/temp_slope_25m_scaled.tif" \
             "$OUTPUT_DIR/tirol_slope_25m_web.tif"
    
    rm "$OUTPUT_DIR/temp_slope_25m_scaled.tif"
else
    echo "25m slope already exists, skipping..."
fi

# Downscale aspect to 25m (0-360 degrees, use 8-bit)
if [ ! -f "$OUTPUT_DIR/tirol_aspect_25m_web.tif" ]; then
    echo "Creating 25m aspect for web..."
    gdal_translate -ot Byte \
                   -scale 0 360 0 255 \
                   -co COMPRESS=DEFLATE \
                   -co PREDICTOR=2 \
                   "$OUTPUT_DIR/temp_aspect_5m.tif" \
                   "$OUTPUT_DIR/temp_aspect_25m_scaled.tif"
    
    gdalwarp -tr 25 25 \
             -r average \
             "$OUTPUT_DIR/temp_aspect_25m_scaled.tif" \
             "$OUTPUT_DIR/tirol_aspect_25m_web.tif"
    
    rm "$OUTPUT_DIR/temp_aspect_25m_scaled.tif"
else
    echo "25m aspect already exists, skipping..."
fi

# Downscale slope to 100m (0-90 degrees, use 8-bit)
if [ ! -f "$OUTPUT_DIR/tirol_slope_100m_web.tif" ]; then
    echo "Creating 100m slope for web..."
    gdal_translate -ot Byte \
                   -scale 0 90 0 255 \
                   -co COMPRESS=DEFLATE \
                   -co PREDICTOR=2 \
                   "$OUTPUT_DIR/temp_slope_5m.tif" \
                   "$OUTPUT_DIR/temp_slope_100m_scaled.tif"
    
    gdalwarp -tr 100 100 \
             -r average \
             "$OUTPUT_DIR/temp_slope_100m_scaled.tif" \
             "$OUTPUT_DIR/tirol_slope_100m_web.tif"
    
    rm "$OUTPUT_DIR/temp_slope_100m_scaled.tif"
else
    echo "100m slope already exists, skipping..."
fi

# Downscale aspect to 100m (0-360 degrees, use 8-bit)
if [ ! -f "$OUTPUT_DIR/tirol_aspect_100m_web.tif" ]; then
    echo "Creating 100m aspect for web..."
    gdal_translate -ot Byte \
                   -scale 0 360 0 255 \
                   -co COMPRESS=DEFLATE \
                   -co PREDICTOR=2 \
                   "$OUTPUT_DIR/temp_aspect_5m.tif" \
                   "$OUTPUT_DIR/temp_aspect_100m_scaled.tif"
    
    gdalwarp -tr 100 100 \
             -r average \
             "$OUTPUT_DIR/temp_aspect_100m_scaled.tif" \
             "$OUTPUT_DIR/tirol_aspect_100m_web.tif"
    
    rm "$OUTPUT_DIR/temp_aspect_100m_scaled.tif"
else
    echo "100m aspect already exists, skipping..."
fi

# Clean up temporary 5m slope/aspect files (keep only if you want to preserve them)
echo "Cleaning up temporary high-resolution slope/aspect files..."
rm -f "$OUTPUT_DIR/temp_slope_5m.tif"
rm -f "$OUTPUT_DIR/temp_aspect_5m.tif"

echo ""
echo "Processing complete!"
echo "Output files:"
echo "  Elevation:"
echo "    - High-precision 5m:  $OUTPUT_DIR/tirol_5m_float.tif"
echo "    - Low-precision 25m:  $OUTPUT_DIR/tirol_25m_web.tif" 
echo "    - Low-precision 100m: $OUTPUT_DIR/tirol_100m_web.tif"
echo "  Slope:"
echo "    - 25m web:  $OUTPUT_DIR/tirol_slope_25m_web.tif"
echo "    - 100m web: $OUTPUT_DIR/tirol_slope_100m_web.tif"
echo "  Aspect:"
echo "    - 25m web:  $OUTPUT_DIR/tirol_aspect_25m_web.tif"
echo "    - 100m web: $OUTPUT_DIR/tirol_aspect_100m_web.tif"

# Show file sizes
echo ""
echo "File sizes:"
ls -lh "$OUTPUT_DIR"/*.tif