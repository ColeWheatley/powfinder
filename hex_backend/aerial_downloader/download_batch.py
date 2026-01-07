"""
TIROL ORTHOFOTO DOWNLOADER
Source: https://gis.tirol.gv.at/geo/dop/m28/ (e.g., dop_2121-53_2023.tif)

Project History & Coverage:
We have defined three main bounding boxes to cover the key ski areas:
1. Southern / Central: Refined by grid anchors 2123-44, 2323-47, and 2121-76.
2. Ischgl / Silvretta: Anchored by 1421-56, 1621-50, and 1624-66.
3. Arlberg / In-Between: The zone between 2223-49 and 2024-13.

Methodology:
- Check .tfw files (worldfiles) first via check_tirol_grid.py to mapping availability.
- Update queue via update_queue.py to merge boxes and filter existing disk files.
- SEQUENTIAL download with 2s delay per file to avoid bot-capping.

Stats:
- Total unique tiles available in these zones: ~976.
- Average size: 10-15MB per TIF. Total set ~14.5 GB.
"""

import os
import json
import requests
import time

QUEUE_FILE = "../download_queue.json"
OUTPUT_DIR = "../aerial_tifs"
YEAR = "2023"

def download_tif(grid_id):
    filename = f"dop_{grid_id}_{YEAR}.tif"
    url = f"https://gis.tirol.gv.at/geo/dop/m28/{filename}"
    target_path = os.path.join(OUTPUT_DIR, filename)
    
    if os.path.exists(target_path):
        return grid_id, "exists"
        
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return grid_id, "done"
        else:
            return grid_id, f"error_{response.status_code}"
    except Exception as e:
        return grid_id, f"failed_{str(e)}"

import time

def main():
    if not os.path.exists(QUEUE_FILE):
        print(f"No queue file found at {QUEUE_FILE}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(QUEUE_FILE, 'r') as f:
        data = json.load(f)
    
    initial_queue = data.get('queue', [])
    
    # Re-index what is on disk right now
    downloaded_on_disk = set()
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith('.tif'):
                parts = f.split('_')
                if len(parts) >= 2:
                    downloaded_on_disk.add(parts[1])
    
    queue = [gid for gid in initial_queue if gid not in downloaded_on_disk]
    total = len(queue)
    already_had = len(initial_queue) - total
    
    if already_had > 0:
        print(f"Re-indexed disk: Skipping {already_had} files already present.")
    
    if total == 0:
        print("Queue is empty (all tiles already on disk).")
        return

    print(f"Starting sequential download of {total} files with a 2-second delay...")
    
    for idx, gid in enumerate(queue):
        gid, status = download_tif(gid)
        done_count = idx + 1
        
        if status == "done":
            print(f"[{done_count}/{total}] ✓ {gid} downloaded")
            time.sleep(2) # Delay to stay under the radar
        elif status == "exists":
            print(f"[{done_count}/{total}] - {gid} already exists")
        else:
            print(f"[{done_count}/{total}] ✗ {gid} failed: {status}")
            time.sleep(1)

    print("\nBatch download complete.")

if __name__ == "__main__":
    main()
