#!/bin/bash

# Configuration
BUCKET_NAME="wheatley.cloud"
PROJECT_PATH="powfinder/app"

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Syncing PowFinder to s3://${BUCKET_NAME}/${PROJECT_PATH}/...${NC}"

# Sync command
# --delete: Removes files in S3 that are not in local (keeps them 1:1)
# Includes core web files and resources, excludes data processing and temp files
aws s3 sync . "s3://${BUCKET_NAME}/${PROJECT_PATH}/" \
    --exclude "*" \
    --include "index.html" \
    --include "landing.html" \
    --include "powfinder.html" \
    --include "main.js" \
    --include "style.css" \
    --include "favicon.png" \
    --include "clientside.mp4" \
    --include "hillshade series.png" \
    --include "package.json" \
    --include "web-resources/*" \
    --include "dev/*" \
    --exclude "*/.DS_Store" \
    --exclude ".DS_Store" \
    --exclude "*.git*" \
    --exclude ".claude*" \
    --delete

echo -e "${GREEN}Sync complete!${NC}"

