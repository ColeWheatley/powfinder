import os
import argparse
import rasterio
from rasterio.warp import transform_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import numpy as np
from PIL import Image
from tqdm import tqdm

# Config for Basemap.at (as used in piston_tool.py). 
# User mentioned Tiris Winter, but provided no URL. Using this as default/placeholder.
XYZ_URL = "https://maps.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/${z}/${y}/${x}.jpeg"

GDAL_WMS_XML = f"""<GDAL_WMS>
    <Service name="TMS">
        <ServerUrl>{XYZ_URL}</ServerUrl>
    </Service>
    <DataWindow>
        <UpperLeftX>-20037508.34</UpperLeftX>
        <UpperLeftY>20037508.34</UpperLeftY>
        <LowerRightX>20037508.34</LowerRightX>
        <LowerRightY>-20037508.34</LowerRightY>
        <TileLevel>20</TileLevel>
        <TileCountX>1</TileCountX>
        <TileCountY>1</TileCountY>
        <YOrigin>top</YOrigin>
    </DataWindow>
    <Projection>EPSG:3857</Projection>
    <BlockSizeX>256</BlockSizeX>
    <BlockSizeY>256</BlockSizeY>
    <BandsCount>3</BandsCount>
    <Cache />
</GDAL_WMS>
"""

def fetch_satellite(dem_tiles_dir, output_dir):
    """
    Iterates over DEM tiles, determines their bounds, and fetches corresponding satellite imagery.
    """
    # Create valid WMS config file
    wms_config_path = "satellite_source.xml"
    with open(wms_config_path, "w") as f:
        f.write(GDAL_WMS_XML)
        
    # Open the WMS source
    # We keep it open ? Or open per tile? 
    # GDAL handles caching, but opening remote datasets can be slow.
    # WarpedVRT is needed to reproject WMS (3857) to Match DEM (31254).
    
    # WE need the CRS of the DEM tiles.
    # We'll read the first tile found to get profile.
    
    # Find all tiles
    tasks = []
    print("Scanning DEM tiles...")
    for root, dirs, files in os.walk(dem_tiles_dir):
        for file in files:
            if file.endswith(".tif"):
                tasks.append(os.path.join(root, file))
    
    if not tasks:
        print("No DEM tiles found.")
        return

    print(f"Found {len(tasks)} tiles.")
    
    # Read profile from first tile
    with rasterio.open(tasks[0]) as src:
        profile = src.profile
        crs = src.crs
        
    # Open WMS with WarpedVRT to target CRS
    # Note: Bounds of WMS are global, so VRT works.
    with rasterio.open(wms_config_path) as src_wms:
        print("Connected to Satellite Source.")
        
        # Iterate tasks
        for dem_path in tqdm(tasks):
            # Determine relative path for output
            rel_path = os.path.relpath(dem_path, dem_tiles_dir)
            out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".webp")
            
            if os.path.exists(out_path):
                continue
                
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            with rasterio.open(dem_path) as dem_ds:
                bounds = dem_ds.bounds
                width = dem_ds.width
                height = dem_ds.height
                transform = dem_ds.transform
                
                # Setup VRT for this specific tile area?
                # Or use WarpedVRT on the whole WMS and read window?
                # WarpedVRT for specific window is more efficient usually?
                
                vrt_options = {
                    'resampling': Resampling.bilinear,
                    'crs': crs,
                    'transform': transform,
                    'height': height,
                    'width': width
                }
                
                # Fetch
                try:
                    with WarpedVRT(src_wms, **vrt_options) as vrt:
                        data = vrt.read()
                        
                        # data is (B, H, W). Save as Image.
                        # Move bands to last dim
                        img_data = np.moveaxis(data, 0, -1)
                        if img_data.shape[2] > 3:
                            img_data = img_data[:,:,:3] # Drop alpha if 4 bands
                            
                        # Convert to uint8
                        img = Image.fromarray(img_data.astype('uint8'))
                        img.save(out_path, "WEBP", quality=85)
                        
                except Exception as e:
                    print(f"Error fetching {dem_path}: {e}")

    if os.path.exists(wms_config_path):
        os.remove(wms_config_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", required=True, help="Path to DEM tiles directory")
    parser.add_argument("--out", required=True, help="Path to Output Satellite directory")
    args = parser.parse_args()
    
    fetch_satellite(args.dem, args.out)

if __name__ == "__main__":
    main()
