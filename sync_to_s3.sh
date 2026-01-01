#!/bin/bash

# Configuration
BUCKET_NAME="wheatley.cloud"
PROJECT_PATH="powfinder"

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Syncing PowFinder to s3://${BUCKET_NAME}/${PROJECT_PATH}/...${NC}"

# Sync the frontend folder directly to the project path
# This mirrors the local frontend/ structure to wheatley.cloud/powfinder/
aws s3 sync ./frontend "s3://${BUCKET_NAME}/${PROJECT_PATH}/" \
    --delete \
    --exclude "*/.DS_Store" \
    --exclude ".DS_Store"

echo -e "${GREEN}Sync complete!${NC}"