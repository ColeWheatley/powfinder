#!/usr/bin/env python3
"""
Calculate EPSG:31254 and hex/sector coordinates for Tirol ski areas
"""
import sys
import json
from pyproj import Transformer
sys.path.append('hex_backend')
from coordinate_utility import world_to_sector_id, world_meters_to_axial_approx, round_axial

# GPS Coordinates (WGS84 - EPSG:4326)
ski_areas = {
    "Kappl": {"lat": 47.06689, "lon": 10.35909},
    "St. Anton am Arlberg": {"lat": 47.1275, "lon": 10.2637},
    "Stubaier Gletscher": {"lat": 47.0833, "lon": 11.1667},
    "Schlick 2000": {"lat": 47.1089, "lon": 11.3181},
    "Axamer Lizum": {"lat": 47.19569, "lon": 11.30265},
    "Patscherkofel": {"lat": 47.208775, "lon": 11.462048},
    "Nordkette (Innsbruck)": {"lat": 47.285794, "lon": 11.398938},
    "Glungezer": {"lat": 47.207598, "lon": 11.528409},
    "Seefeld Rosshütte": {"lat": 47.3302, "lon": 11.1879},  # Main area
    "Ischgl": {"lat": 47.011845, "lon": 10.288420},
    "Serfaus-Fiss-Ladis": {"lat": 47.0378, "lon": 10.6058},  # Serfaus estimate
    "Muttereralm": {"lat": 47.23, "lon": 11.38},
    "Bergeralm - Steinach am Brenner": {"lat": 47.0888716, "lon": 11.4714428},
    "Obergurgl-Hochgurgl": {"lat": 46.8692, "lon": 11.0283}  # Obergurgl estimate
}

# Create transformer from WGS84 to EPSG:31254 (Austria GK West)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:31254", always_xy=True)

results = []

print("Transforming coordinates...\n")
for name, coords in ski_areas.items():
    lat, lon = coords["lat"], coords["lon"]

    # Transform to EPSG:31254
    x, y = transformer.transform(lon, lat)

    # Calculate sector ID
    sx, sy = world_to_sector_id(x, y)

    # Calculate hex coordinates (approximate)
    q_float, r_float = world_meters_to_axial_approx(x, y)
    q, r = round_axial(q_float, r_float)

    result = {
        "name": name,
        "gps": {
            "lat": lat,
            "lon": lon
        },
        "epsg_31254": {
            "x": round(x, 1),
            "y": round(y, 1)
        },
        "sector": {
            "q": sx,
            "r": sy
        },
        "hex_axial_approx": {
            "q": q,
            "r": r
        }
    }

    results.append(result)

    print(f"{name}:")
    print(f"  GPS: {lat:.6f}, {lon:.6f}")
    print(f"  EPSG:31254: {x:.1f}, {y:.1f}")
    print(f"  Sector: {sx}, {sy}")
    print(f"  Hex: {q}, {r}")
    print()

# Create final JSON structure
output = {
    "ski_areas": results,
    "coordinate_system_info": {
        "projection": "EPSG:31254 (MGI / Austria GK West)",
        "unit_hex_width_meters": 6.4,
        "sector_width_meters": 819.2,
        "meters_per_pixel": 0.2,
        "unit_hex_px": 32.0,
        "hex_orientation": "flat_top",
        "note": "Hex coordinates are approximate calculations based on the coordinate conversion system. Sector coordinates (q, r) represent rectangular bins. Hex axial coordinates represent the hexagonal grid position."
    }
}

# Write to file
with open('skigebiete.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\n✓ Updated skigebiete.json")
