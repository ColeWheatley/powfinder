#!/bin/bash

# Configuration
BUCKET_NAME="wheatley-cloud-eu"
PROJECT_PATH="powfinder"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Password Protection
REQUIRED_PASS="dumbass"

echo -e "$RED WARNING: This script uses --delete. It will WIPE anything in s3://$BUCKET_NAME/$PROJECT_PATH/ that is not in the local ./frontend/ folder. $NC"
read -p "Enter validation password: " USER_PASS

if [ "$USER_PASS" != "$REQUIRED_PASS" ]; then
    echo -e "$RED Incorrect password. Aborting. $NC"
    exit 1
fi

echo -e "$GREEN Password accepted. Syncing PowFinder to s3://$BUCKET_NAME/$PROJECT_PATH/... $NC"

# Sync the frontend folder
# Braincells engaged: We only sync the ./frontend dir.
# If a file exists on S3/powfinder/ but NOT in ./frontend/, it WILL BE DELETED.
# EXCEPT: hexagons/ is preserved (managed separately)
aws s3 sync ./frontend "s3://$BUCKET_NAME/$PROJECT_PATH/" \
    --delete \
    --exclude "*/.DS_Store" \
    --exclude ".DS_Store" \
    --exclude "hexagons/*"

echo -e "$GREEN Sync complete! $NC"