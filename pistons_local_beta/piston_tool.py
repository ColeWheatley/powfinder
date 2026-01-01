import os
import argparse
import sys
import numpy as np
import rasterio
import pyvista as pv
from PIL import Image
import json
import asyncio
from playwright.async_api import async_playwright
from rasterio.windows import from_bounds
from rasterio.warp import transform
from thefuzz import process
import io

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_DEM = "/Users/cole/dev/PowFinder/backend/terrains/DGM_Tirol_5m_epsg31254_2006_2020.tif"
PLACES_JSON = os.path.join(BASE_DIR, "places.json")
CACHE_DIR = os.path.join(BASE_DIR, "data")

# --- DATA ACQUISITION LOGIC ---

def get_coords(input_str):
    if "," in input_str:
        try:
            lat, lon = map(float, input_str.split(","))
            return lat, lon, 2.0, f"{lat:.4f}_{lon:.4f}_2km"
        except ValueError:
            pass

    if not os.path.exists(PLACES_JSON):
        print(f"Error: {PLACES_JSON} not found.")
        sys.exit(1)
        
    with open(PLACES_JSON, 'r') as f:
        places = json.load(f)
    
    names = [p['name'] for p in places]
    best_match, score = process.extractOne(input_str, names)
    
    if score > 70:
        place = next(p for p in places if p['name'] == best_match)
        r = place.get('radius_km', 2.0)
        print(f"Fuzzy matched '{input_str}' to '{best_match}' (Radius: {r}km)")
        return place['lat'], place['lon'], r, f"{place['lat']:.4f}_{place['lon']:.4f}_{r}km"
    
    print(f"Error: Could not parse '{input_str}' as coordinates or find a match in places.json")
    sys.exit(1)

async def capture_satellite(lat, lon, output_path, radius_km):
    async with async_playwright() as p:
        pixels_per_km = 2400 
        view_size = min(int(radius_km * pixels_per_km), 8000)
        
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page(viewport={'width': view_size, 'height': view_size})
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                body, html, #map {{ margin: 0; padding: 0; height: 100%; width: 100%; background: #000; }}
                .leaflet-control-container {{ display: none; }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map = L.map('map', {{
                    center: [{lat}, {lon}],
                    zoom: 18,
                    zoomControl: false,
                    attributionControl: false,
                    fadeAnimation: false
                }});
                L.tileLayer('https://maps.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/{{z}}/{{y}}/{{x}}.jpeg', {{
                    maxZoom: 20
                }}).addTo(map);
            </script>
        </body>
        </html>
        """
        temp_html = "/private/tmp/piston_capture_temp.html"
        with open(temp_html, "w") as f: f.write(html_content)
        await page.goto(f"file://{temp_html}")
        
        print(f"Waiting for satellite tiles (Radius {radius_km}km, Reactive Mode)...")
        max_wait, start_time, final_img_bytes = 25, asyncio.get_event_loop().time(), None
        while (asyncio.get_event_loop().time() - start_time) < max_wait:
            screenshot_bytes = await page.screenshot()
            img = Image.open(io.BytesIO(screenshot_bytes))
            data = np.array(img)
            black_pixels = np.all(data == [0, 0, 0], axis=-1)
            num_black = np.sum(black_pixels)
            if num_black < 1000:
                print(f"✓ Map fully loaded in {int(asyncio.get_event_loop().time() - start_time)}s.")
                final_img_bytes = screenshot_bytes
                break
            print(f"  ... {num_black} black pixels remaining.")
            await asyncio.sleep(1.5)
            final_img_bytes = screenshot_bytes
            
        img = Image.open(io.BytesIO(final_img_bytes))
        img.save(output_path, "WEBP", quality=80)
        await browser.close()
        if os.path.exists(temp_html): os.remove(temp_html)
        print(f"✓ Satellite imagery finalized.")

def crop_dem(lat, lon, output_path, radius_km):
    with rasterio.open(MASTER_DEM) as src:
        xs, ys = transform('EPSG:4326', src.crs, [lon], [lat])
        center_x, center_y = xs[0], ys[0]
        offset = (radius_km * 1000) / 2
        window = from_bounds(center_x-offset, center_y-offset, center_x+offset, center_y+offset, src.transform)
        kwargs = src.meta.copy()
        kwargs.update({'height': int(window.height), 'width': int(window.width), 'transform': src.window_transform(window)})
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst.write(src.read(window=window))
    print(f"✓ DEM cropped ({radius_km}km).")

def get_slope_color(deg):
    if deg < 25: return [0.5, 0.5, 0.5]
    if deg < 35: return [0, 1, 0]
    if deg < 40: return [0, 0, 1]
    if deg < 45: return [0.5, 0, 0.5]
    if deg < 50: return [1, 0.5, 0]
    return [1, 0, 0]

def launch_viewer(target_dir):
    dem_path = os.path.join(target_dir, "dem.tif")
    sat_path_webp = os.path.join(target_dir, "satellite.webp")
    sat_path_png = os.path.join(target_dir, "satellite.png")
    sat_path = sat_path_webp if os.path.exists(sat_path_webp) else sat_path_png

    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(np.float32)
    tile_size = 5
    heights = dem_data - np.nanmin(dem_data)
    rows, cols = dem_data.shape

    sat_img = Image.open(sat_path).convert('RGB')
    sat_array = np.array(sat_img)
    sat_texture = pv.numpy_to_texture(sat_array)

    dy, dx = np.gradient(heights, tile_size)
    true_slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    plotter = pv.Plotter(window_size=(1200, 900))
    cap_pts, cap_faces = [], []
    for r in range(rows):
        for c in range(cols):
            h = heights[r, c]
            x0, x1 = c * tile_size, (c + 1) * tile_size
            y0, y1 = (rows - r - 1) * tile_size, (rows - r) * tile_size
            s_idx = len(cap_pts)
            cap_pts.extend([[x0, y0, h], [x1, y0, h], [x1, y1, h], [x0, y1, h]])
            cap_faces.extend([4, s_idx, s_idx + 1, s_idx + 2, s_idx + 3])

    cap_mesh = pv.PolyData(np.array(cap_pts), np.array(cap_faces))
    tex_coords = np.zeros((len(cap_pts), 2))
    for i, p in enumerate(cap_pts):
        tex_coords[i] = [p[0] / (cols * tile_size), p[1] / (rows * tile_size)]
    cap_mesh.active_texture_coordinates = tex_coords
    plotter.add_mesh(cap_mesh, texture=sat_texture, lighting=False)

    wall_pts, wall_faces, wall_colors = [], [], []
    for r in range(rows):
        for c in range(cols):
            h_curr, color = heights[r, c], get_slope_color(true_slope_deg[r, c])
            x0, x1 = c * tile_size, (c + 1) * tile_size
            y0, y1 = (rows - r - 1) * tile_size, (rows - r) * tile_size
            if r < rows - 1:
                h_neigh = heights[r+1, c]
                if h_curr != h_neigh:
                    s_idx = len(wall_pts)
                    wall_pts.extend([[x0, y0, h_curr], [x1, y0, h_curr], [x1, y0, h_neigh], [x0, y0, h_neigh]])
                    wall_faces.extend([4, s_idx, s_idx + 1, s_idx + 2, s_idx + 3]), wall_colors.append(color)
            if c < cols - 1:
                h_neigh = heights[r, c+1]
                if h_curr != h_neigh:
                    s_idx = len(wall_pts)
                    wall_pts.extend([[x1, y0, h_curr], [x1, y1, h_curr], [x1, y1, h_neigh], [x1, y0, h_neigh]])
                    wall_faces.extend([4, s_idx, s_idx + 1, s_idx + 2, s_idx + 3]), wall_colors.append(color)

    if wall_pts:
        wall_mesh = pv.PolyData(np.array(wall_pts), np.array(wall_faces))
        wall_mesh.cell_data['colors'] = np.array(wall_colors)
        plotter.add_mesh(wall_mesh, scalars='colors', rgb=True, preference='cell')

    plotter.add_title(f"PowFinder Piston Model: {target_dir}")
    plotter.set_background("black")
    print("READY")
    plotter.show()

def main():
    places = []
    if os.path.exists(PLACES_JSON):
        with open(PLACES_JSON, 'r') as f: places = json.load(f)

    parser = argparse.ArgumentParser(description="PowFinder Piston Tool")
    place_group = parser.add_mutually_exclusive_group(required=True)
    for p in places:
        flag = f"--{p['name'].lower().replace(' ', '')}"
        place_group.add_argument(flag, action="store_true", help=f"View {p['name']}")
    place_group.add_argument("--coords", help="Manual 'lat,lon' input")
    args = parser.parse_args()

    lat, lon, radius_km, loc_id = None, None, None, None
    if args.coords:
        lat, lon, radius_km, loc_id = get_coords(args.coords)
    else:
        for p in places:
            flag_name = p['name'].lower().replace(' ', '')
            if getattr(args, flag_name):
                lat, lon = p['lat'], p['lon']
                radius_km = p.get('radius_km', 2.0)
                loc_id = f"{lat:.4f}_{lon:.4f}_{radius_km}km"
                print(f"--- Selected Place: {p['name']} ({radius_km}km) ---")
                break

    if lat is None: sys.exit(1)
    target_dir = os.path.join(CACHE_DIR, loc_id)
    dem_out, sat_out = os.path.join(target_dir, "dem.tif"), os.path.join(target_dir, "satellite.webp")

    if not (os.path.exists(dem_out) and os.path.exists(sat_out)):
        os.makedirs(target_dir, exist_ok=True)
        print(f"--- Downloading new data for {loc_id} ---")
        crop_dem(lat, lon, dem_out, radius_km)
        asyncio.run(capture_satellite(lat, lon, sat_out, radius_km))
    else:
        print(f"--- Loading cached data for {loc_id} ---")

    launch_viewer(target_dir)

if __name__ == "__main__": main()
