// main.js  – peak-driven powder visualiser
// ----------------------------------------
import { getSnowMetrics }          from './snowModel.js';
import { getRawWeather, rawCache, clearWeatherAPICache } from './weatherAPI.js'; // Import clearWeatherAPICache
import { getExtrapolatedWeatherAtPoint } from './weatherExtrapolation.js';

/* ---------- constants ---------- */
const CACHE_TTL_MS   = 60 * 60 * 1000;      // 1 h
const DEBOUNCE_MS    = 120;                 // map move debounce
const OFFSET_M       = 250;                 // N/S/E/W offset distance (m)
const STAGGER_MS     = 20;                  // Reduced stagger for faster anchor processing
const MIN_HEIGHT_CM  = 1;                   // hide points with < 1 cm fresh (used for opacity check)
const MAX_ACCUM_CM_FOR_OPACITY = 50;      // cm of snow for max opacity
const GRID_RESOLUTION = 10;               // Number of cells for grid in queryViewport (e.g., 10x10)
const NUM_CLOSEST_ANCHORS_FOR_EXTRAPOLATION = 5; 

// New color constants for point styling
const COLOR_LIGHT_BLUE = [173, 216, 230]; // For SQH score < 0.33
const COLOR_INDIGO = [75, 0, 130];       // For SQH score < 0.66
const COLOR_HOT_PINK = [255, 105, 180];   // For SQH score >= 0.66

// Color Scales for Other Data Layers
const THERMAL_SCALE = [[0,0,255], [255,255,0], [255,0,0]]; // Blue-Yellow-Red (e.g., for Temperature)
const SNOW_SCALE = [[200,200,255], [0,0,150]];          // Light Blue to Dark Blue (e.g., for Snowfall, Snow Depth)
const HUMIDITY_SCALE = [[224,224,224], [0,0,255]];       // Grey to Blue (e.g. for relative humidity)
const WIND_SPEED_SCALE = [[0,255,0], [255,255,0], [255,0,0]]; // Green-Yellow-Red
const PRESSURE_SCALE = [[255,0,255], [0,255,255]];       // Magenta to Cyan

const ASPECT_COLORS = {                                 // For Aspect
  'north': [255,0,0], 'east': [0,255,0], 'south': [0,0,255], 'west': [255,255,0],
  'northeast': [255,165,0], 'southeast': [0,128,128], // Orange, Teal
  'southwest': [128,0,128], 'northwest': [255,192,203], // Purple, Pink
  'flat': [128,128,128], 'default': [128,128,128]
};
const SLOPE_SCALE = [[0,255,0], [255,255,0], [255,0,0]]; // Green-Yellow-Red (0-15, 15-35, >35 deg)
const ELEVATION_SCALE = [[0,128,0], [139,69,19], [255,255,255]]; // Green-Brown-White

// Predefined min/max for color scaling (simplification)
const TEMP_MIN_C = -20; const TEMP_MAX_C = 30;
const SNOW_DEPTH_MIN_CM = 0; const SNOW_DEPTH_MAX_CM = 300;
const SNOWFALL_MIN_MM = 0; const SNOWFALL_MAX_MM = 20; // Per hour, adjust as needed
const WIND_MIN_KMH = 0; const WIND_MAX_KMH = 100;
const ELEVATION_MIN_M = 500; const ELEVATION_MAX_M = 4000; // Tirol range
const DEWPOINT_MIN_C = -25; const DEWPOINT_MAX_C = 25;
const HUMIDITY_MIN_RH = 0; const HUMIDITY_MAX_RH = 100;
const RADIATION_MIN_W_M2 = 0; const RADIATION_MAX_W_M2 = 1000;
const PRESSURE_MIN_HPA = 950; const PRESSURE_MAX_HPA = 1050;


/* ---------- state variables ---------- */
let snowDisplayMode = 'skiability';
let activeLayer = 'sqh'; // Default active layer
let activeLayerType = 'snow'; // Default active layer type
let targetDate = new Date(); // Initialize with current date and time

/* ---------- caches and data stores ---------- */
let metricsCache = {}; // key → {finalScore, recentAccumCm, ts, lat, lon} for anchor points
let anchorPointsData = {}; // key → { lat, lon, elevation, slope, aspect, weatherData: rawHourlyAPIData }

const savedMetrics = localStorage.getItem('powMetrics');
if (savedMetrics) { try { metricsCache = JSON.parse(savedMetrics); } catch{} }

function key(lat,lon){ return `${lat.toFixed(5)},${lon.toFixed(5)}`; }
function saveMetricsCache(){ localStorage.setItem('powMetrics',JSON.stringify(metricsCache)); }
function parseKey(k) { const parts = k.split(','); return { lat: parseFloat(parts[0]), lon: parseFloat(parts[1]) }; }


/* ---------- peak list ---------- */
let peaks = [];
const peaksPromise = fetch('./refined_tirol_peaks.json')
  .then(r=>r.json())
  .then(arr=>{ peaks = arr; });

/* ---------- OpenLayers setup ---------- */
const map = new ol.Map({
  target: 'map',
  layers: [ new ol.layer.Tile({ source: new ol.source.OSM() }) ],
  view  : new ol.View({
            center: ol.proj.fromLonLat([11.4041, 47.2692]), // Innsbruck
            zoom  : 11
          })
});

const pointSource = new ol.source.Vector();
const pointLayer  = new ol.layer.Vector({
  source: pointSource,
  style: feature => {
    const featureId = feature.getId() || 'unknown_id';
    let styleLogInfo = `Styling feature ${featureId}, ActiveLayer=${activeLayer}, Type=${activeLayerType}. `;
    if (activeLayer === 'sqh') {
        styleLogInfo += `SQH score=${feature.get('finalScore')}, accum=${feature.get('recentAccumCm')}.`;
    } else if (activeLayerType === 'weather' || activeLayerType === 'terrain') {
        styleLogInfo += `${activeLayer}=${feature.get(activeLayer)}.`;
    }
    console.log(styleLogInfo);

    let value = feature.get(activeLayer); 
    let color;
    let opacity = 0.8; 

    if (activeLayer === 'sqh' || (activeLayerType === 'snow' && activeLayer === 'sqh')) { 
      const finalScore = feature.get('finalScore');
      const recentAccumCm = feature.get('recentAccumCm');

      if (recentAccumCm == null || recentAccumCm < MIN_HEIGHT_CM || finalScore == null) {
        return null;
      }
      if (finalScore < 0.33) color = COLOR_LIGHT_BLUE;
      else if (finalScore < 0.66) color = COLOR_INDIGO;
      else color = COLOR_HOT_PINK;
      opacity = Math.max(0.2, Math.min(recentAccumCm / MAX_ACCUM_CM_FOR_OPACITY, 1));

    } else if (activeLayerType === 'terrain') {
      if (value == null) return null;
      switch (activeLayer) {
        case 'elevation':
          color = interpolateColor(value, ELEVATION_MIN_M, ELEVATION_MAX_M, ELEVATION_SCALE);
          break;
        case 'slope':
          if (value < 15) color = SLOPE_SCALE[0];
          else if (value < 35) color = SLOPE_SCALE[1];
          else color = SLOPE_SCALE[2];
          break;
        case 'aspect':
          color = ASPECT_COLORS[value.toLowerCase()] || ASPECT_COLORS.default;
          break;
        default: return null; // Unknown terrain layer
      }
    } else if (activeLayerType === 'weather') {
      if (value == null) return null;
      switch (activeLayer) {
        case 'temperature_2m':
        case 'temperature_850hPa':
        case 'temperature_500hPa':
          color = interpolateColor(value, TEMP_MIN_C, TEMP_MAX_C, THERMAL_SCALE);
          break;
        case 'snowfall':
          color = interpolateColor(value, SNOWFALL_MIN_MM, SNOWFALL_MAX_MM, SNOW_SCALE);
          break;
        case 'snow_depth':
          color = interpolateColor(value, SNOW_DEPTH_MIN_CM, SNOW_DEPTH_MAX_CM, SNOW_SCALE);
          break;
        case 'dewpoint_2m':
          color = interpolateColor(value, DEWPOINT_MIN_C, DEWPOINT_MAX_C, THERMAL_SCALE); // Can reuse thermal or specific
          break;
        case 'relativehumidity_2m':
          color = interpolateColor(value, HUMIDITY_MIN_RH, HUMIDITY_MAX_RH, HUMIDITY_SCALE);
          break;
        case 'windspeed_10m':
          color = interpolateColor(value, WIND_MIN_KMH, WIND_MAX_KMH, WIND_SPEED_SCALE);
          break;
        case 'shortwave_radiation':
          color = interpolateColor(value, RADIATION_MIN_W_M2, RADIATION_MAX_W_M2, THERMAL_SCALE, true); // Single color to max for radiation
          break;
        case 'surface_pressure':
            color = interpolateColor(value, PRESSURE_MIN_HPA, PRESSURE_MAX_HPA, PRESSURE_SCALE);
            break;
        default: return null; // Unknown weather layer
      }
    } else {
      return null; // Unknown layer type or default not met
    }

    if (!color) return null; 

    const [r, g, b] = color;
    const finalFillColor = `rgba(${r},${g},${b},${opacity.toFixed(2)})`;
    // Log calculated color and opacity
    console.log(`  Style for ${featureId}: finalFillColor=${finalFillColor}, opacity=${opacity.toFixed(2)}`);

    return new ol.style.Style({
      image: new ol.style.Circle({ radius: 6, fill: new ol.style.Fill({ color: finalFillColor }), stroke: null })
    });
  }
});
map.addLayer(pointLayer);


// Import getElevation from terrain_evaluator.js
// Assuming terrain_evaluator.js is in the same directory and exports getElevation
import { getElevation } from './terrain_evaluator.js';


/* ---------- helper functions ---------- */
function getHourIndexForTargetDate(timeArray, date) { 
  if (!timeArray || timeArray.length === 0) return -1;
  const targetMs = date.getTime();
  let closestIdx = -1;
  for (let i = 0; i < timeArray.length; i++) {
    if (new Date(timeArray[i]).getTime() <= targetMs) {
      closestIdx = i;
    } else {
      break; 
    }
  }
  if (closestIdx === -1 && timeArray.length > 0) return 0;
  return closestIdx;
}

function getVisiblePeaks(mapExtent, allPeaks) { 
    const [minLon,minLat,maxLon,maxLat] = mapExtent;
    return allPeaks.filter(p => {
        const [lon,lat] = p.coordinates;
        return lon >= minLon && lon <= maxLon && lat >= minLat && lat <= maxLat;
    });
}

function generateGridForExtent(mapExtent, gridRes) { 
    const [minLon,minLat,maxLon,maxLat] = mapExtent;
    const lonStep = (maxLon - minLon) / gridRes;
    const latStep = (maxLat - minLat) / gridRes;
    const grid = [];
    for (let i = 0; i < gridRes; i++) {
        for (let j = 0; j < gridRes; j++) {
            grid.push({ 
                lon: minLon + i * lonStep + lonStep / 2, 
                lat: minLat + j * latStep + latStep / 2 
            });
        }
    }
    return grid;
}

const M_PER_DEG = 111320; 
function dLat(m){ return m / M_PER_DEG; } 
function dLon(lat,m){ return m / (M_PER_DEG*Math.cos(lat*Math.PI/180)); } 

function lerp(a, b, t) { return a + (b - a) * t; } 

// Helper to interpolate color from a scale
function interpolateColor(value, minVal, maxVal, scale, singleMaxColor = false) {
  if (value == null) return [128, 128, 128]; // Default grey for missing values
  const t = Math.max(0, Math.min(1, (value - minVal) / (maxVal - minVal)));

  if (singleMaxColor && scale.length === 2) { // e.g. radiation: transparent to yellow
    const from = scale[0]; // Assuming this is a "zero" or transparent/base color
    const to = scale[1];
    return [
      lerp(from[0], to[0], t),
      lerp(from[1], to[1], t),
      lerp(from[2], to[2], t),
    ];
  }
  
  const segments = scale.length - 1;
  const segment = Math.floor(t * segments);
  if (segment >= segments) return scale[segments]; // Value is max or above
  if (segment < 0) return scale[0]; // Value is min or below (should be caught by Math.max(0,..))

  const segmentT = (t * segments) - segment;
  const from = scale[segment];
  const to = scale[segment + 1];

  return [
    lerp(from[0], to[0], segmentT),
    lerp(from[1], to[1], segmentT),
    lerp(from[2], to[2], segmentT),
  ];
}


// Helper to calculate distance between two lat/lon points (Haversine)
function haversineDistance(lat1, lon1, lat2, lon2) {
    const R = 6371e3; // metres
    const φ1 = lat1 * Math.PI/180;
    const φ2 = lat2 * Math.PI/180;
    const Δφ = (lat2-lat1) * Math.PI/180;
    const Δλ = (lon2-lon1) * Math.PI/180;

    const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
              Math.cos(φ1) * Math.cos(φ2) *
              Math.sin(Δλ/2) * Math.sin(Δλ/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c; // in metres
}


/* ---------- fetching for anchor points ---------- */
async function fetchMetricsForAnchor(lat, lon) { 
  const k = key(lat, lon);
  const now = Date.now();

  if (metricsCache[k] && (now - metricsCache[k].ts < CACHE_TTL_MS)) {
    if (!rawCache[k]) { 
        try {
            await getRawWeather(lat, lon); 
        } catch (err) {
            console.warn(`Error fetching raw weather for cached metrics ${k}:`, err);
            delete metricsCache[k]; // Invalidate if raw data can't be fetched
        }
    }
    return metricsCache[k]; 
  }

  try {
    const rawWeatherData = await getRawWeather(lat, lon); 
    if (!rawWeatherData || !rawWeatherData.hourly) {
        console.warn(`Failed to get raw weather for anchor ${k}`);
        return null;
    }
    const { finalScore, recentAccumCm } = computeSnowMetrics(rawWeatherData.hourly, targetDate, 24, snowDisplayMode);
    metricsCache[k] = { finalScore, recentAccumCm, ts: now, lat, lon };
    saveMetricsCache(); 
    return metricsCache[k];
  } catch (err) {
    console.warn(`fetchMetricsForAnchor failed for ${k}:`, err);
    return null;
  }
}


/* ---------- main routine ---------- */
async function queryViewport() {
  console.log('Starting queryViewport...');
  await peaksPromise; // Ensure peaks are loaded

  const view = map.getView();
  const olExtent = view.calculateExtent(map.getSize());
  const mapExtent = ol.proj.transformExtent(olExtent, 'EPSG:3857', 'EPSG:4326'); // minLon, minLat, maxLon, maxLat

  // Phase 1: Fetch/Update Anchor Point Data
  console.log('Phase 1: Fetching/Updating Anchor Point Data...');
  const visiblePeaksArray = getVisiblePeaks(mapExtent, peaks);
  const uniqueAnchorPointKeys = new Set();

  visiblePeaksArray.forEach(p => {
    const [lon, lat] = p.coordinates;
    uniqueAnchorPointKeys.add(key(lat, lon));
    // Add offsets
    const dLatDeg = dLat(OFFSET_M);
    const dLonDeg = dLon(lat, OFFSET_M);
    uniqueAnchorPointKeys.add(key(lat + dLatDeg, lon));
    uniqueAnchorPointKeys.add(key(lat - dLatDeg, lon));
    uniqueAnchorPointKeys.add(key(lat, lon + dLonDeg));
    uniqueAnchorPointKeys.add(key(lat, lon - dLonDeg));
  });
  
  console.log(`Identified ${uniqueAnchorPointKeys.size} unique anchor points (peaks + offsets).`);

  let anchorFetchPromises = [];
  let idx = 0;
  for (const pointKey of uniqueAnchorPointKeys) {
      const { lat: pLat, lon: pLon } = parseKey(pointKey);
      // Stagger the initiation of fetch operations
      const promise = new Promise(resolve => setTimeout(async () => {
          try {
              // console.log(`Processing anchor: ${pointKey}`);
              const terrain = await getElevation(pLat, pLon);
              await fetchMetricsForAnchor(pLat, pLon); // Ensures rawCache and metricsCache are populated

              if (terrain && terrain.elevation != null && rawCache[pointKey] && rawCache[pointKey].hourly) {
                  anchorPointsData[pointKey] = {
                      lat: pLat,
                      lon: pLon,
                      elevation: terrain.elevation,
                      slope: terrain.slope,
                      aspect: terrain.aspect,
                      weatherData: rawCache[pointKey].hourly
                  };
                  // console.log(`Stored data for anchor ${pointKey}`);
              } else {
                  // console.warn(`Missing terrain or weather data for anchor ${pointKey}, removing from anchorPointsData.`);
                  delete anchorPointsData[pointKey]; // Ensure only complete anchors are kept
              }
          } catch (error) {
              console.warn(`Error processing anchor ${pointKey}:`, error);
              delete anchorPointsData[pointKey]; // Remove incomplete anchor on error
          } finally {
              resolve();
          }
      }, idx * STAGGER_MS));
      anchorFetchPromises.push(promise);
      idx++;
  }
  
  await Promise.all(anchorFetchPromises);
  console.log(`Phase 1 Complete. ${Object.keys(anchorPointsData).length} anchors processed and stored.`);

  // Phase 2: Process Grid Points for Display (Extrapolation using Anchors)
  console.log('Phase 2: Processing Grid Points for Display...');
  pointSource.clear();
  const gridCells = generateGridForExtent(mapExtent, GRID_RESOLUTION);

  if (Object.keys(anchorPointsData).length === 0) {
    console.warn("No valid anchor points available for extrapolation. Skipping grid display.");
    return;
  }
  
  const validAnchors = Object.values(anchorPointsData).filter(anchor => anchor.weatherData);
  if (validAnchors.length === 0) {
    console.warn("No anchors with weatherData available. Skipping grid display.");
    return;
  }
  console.log(`Using ${validAnchors.length} valid anchors for extrapolation.`);

  let featuresAdded = 0;
  for (const cell of gridCells) {
    try {
      const gridLat = cell.lat;
      const gridLon = cell.lon;
      const gridPointTerrain = await getElevation(gridLat, gridLon);

      if (!gridPointTerrain || gridPointTerrain.elevation == null) {
        // console.warn(`Skipping grid cell ${key(gridLat, gridLon)} due to missing terrain.`);
        continue;
      }

      const sortedAnchors = validAnchors
        .map(anchor => ({ ...anchor, distance: haversineDistance(gridLat, gridLon, anchor.lat, anchor.lon) }))
        .sort((a, b) => a.distance - b.distance);
      
      const closestSourceAnchors = sortedAnchors.slice(0, NUM_CLOSEST_ANCHORS_FOR_EXTRAPOLATION);

      if (closestSourceAnchors.length === 0) { // Should be at least 1 if validAnchors is not empty
        // console.warn(`No source anchors found for grid cell ${key(gridLat, gridLon)}`);
        continue;
      }
      
      // Format for extrapolation function
      const sourcePointsForExtrapolation = closestSourceAnchors.map(anchor => ({
          lat: anchor.lat,
          lon: anchor.lon,
          elevation: anchor.elevation,
          slope: anchor.slope,
          aspect: anchor.aspect,
          weather: anchor.weatherData // This is the raw hourly API data object
      }));

      const extrapolatedWeather = getExtrapolatedWeatherAtPoint(gridPointTerrain, sourcePointsForExtrapolation);
      if (!extrapolatedWeather || !extrapolatedWeather.time || extrapolatedWeather.time.length === 0) {
        // console.warn(`Extrapolation failed for grid cell ${key(gridLat, gridLon)}`);
        continue;
      }

      const metrics = computeSnowMetrics(extrapolatedWeather, targetDate, 24, snowDisplayMode);
      const hourIdx = getHourIndexForTargetDate(extrapolatedWeather.time, targetDate);

      const featureData = {
        finalScore: metrics?.finalScore,
        recentAccumCm: metrics?.recentAccumCm,
        elevation: gridPointTerrain.elevation,
        slope: gridPointTerrain.slope,
        aspect: gridPointTerrain.aspect,
        lat: gridLat,
        lon: gridLon,
      };
      if (hourIdx !== -1) {
        featureData.temperature_2m = extrapolatedWeather.temperature_2m?.[hourIdx];
        featureData.snowfall = extrapolatedWeather.snowfall?.[hourIdx];
        featureData.snow_depth = extrapolatedWeather.snow_depth?.[hourIdx];
        featureData.dewpoint_2m = extrapolatedWeather.dewpoint_2m?.[hourIdx];
        featureData.relativehumidity_2m = extrapolatedWeather.relativehumidity_2m?.[hourIdx];
        featureData.windspeed_10m = extrapolatedWeather.windspeed_10m?.[hourIdx];
        featureData.shortwave_radiation = extrapolatedWeather.shortwave_radiation?.[hourIdx];
        featureData.temperature_850hPa = extrapolatedWeather.temperature_850hPa?.[hourIdx];
        featureData.temperature_500hPa = extrapolatedWeather.temperature_500hPa?.[hourIdx];
        featureData.surface_pressure = extrapolatedWeather.surface_pressure?.[hourIdx];
      }

      const feature = new ol.Feature({
        geometry: new ol.geom.Point(ol.proj.fromLonLat([gridLon, gridLat])),
        ...featureData
      });
      feature.setId(key(gridLat, gridLon));
      
      // Enhanced log before adding feature
      console.log(`Feature data for ${key(gridLat, gridLon)}: score=${metrics.finalScore?.toFixed(2)}, accumCm=${metrics.recentAccumCm?.toFixed(1)}, elev=${gridPointTerrain.elevation}, temp@targetHour=${extrapolatedWeather.temperature_2m?.[hourIdx]?.toFixed(1)}`);
      
      pointSource.addFeature(feature);
      featuresAdded++;
    } catch (error) {
      console.warn(`Error processing grid cell ${cell.lat},${cell.lon}:`, error);
    }
  }
  console.log(`Phase 2 Complete. Added ${featuresAdded} features to map.`);
}


/* ---------- interaction ---------- */
let debounce = null;
map.on('moveend', ()=>{
  clearTimeout(debounce);
  debounce = setTimeout(queryViewport, DEBOUNCE_MS);
});
queryViewport();                                     // initial load

// click-info & highlight -----------------------------------------------------
const highlightSrc = new ol.source.Vector();
map.addLayer(new ol.layer.Vector({
  source: highlightSrc,
  style : new ol.style.Style({
            stroke: new ol.style.Stroke({ color:'#ff4081', width:3 }),
            fill  : null
          })
}));

map.on('singleclick',async evt=>{
  const info = document.getElementById('info-box');
  info.innerText = 'Fetching data...';
  info.style.display = 'block';
  highlightSrc.clear();

  const coordinate = evt.coordinate;
  const lonLat = ol.proj.toLonLat(coordinate);
  const clickLat = lonLat[1];
  const clickLon = lonLat[0];

  let terrainData = null;
  try {
    terrainData = await getElevation(clickLat, clickLon);
  } catch (error) {
    console.error("Error fetching terrain data:", error);
  }

  let directWeatherData = null;
  try {
    directWeatherData = await getRawWeather(clickLat, clickLon); // Fresh API call
  } catch (error) {
    console.error("Error fetching live weather data:", error);
  }

  let infoText = `Lat: ${clickLat.toFixed(4)}, Lon: ${clickLon.toFixed(4)}\n`;

  if (terrainData) {
    infoText += `Elev: ${terrainData.elevation?.toFixed(0)}m, Slope: ${terrainData.slope?.toFixed(1)}°, Aspect: ${terrainData.aspect}\n`;
  } else {
    infoText += "Terrain data unavailable.\n";
  }

  if (directWeatherData && directWeatherData.hourly && directWeatherData.hourly.time && directWeatherData.hourly.time.length > 0) {
    let hourIdxForTargetDate = -1;
    const targetMs = targetDate.getTime();
    for (let i = 0; i < directWeatherData.hourly.time.length; i++) {
      if (new Date(directWeatherData.hourly.time[i]).getTime() <= targetMs) {
        hourIdxForTargetDate = i;
      } else {
        break;
      }
    }
     if (hourIdxForTargetDate === -1 && directWeatherData.hourly.time.length > 0) {
        hourIdxForTargetDate = 0; // Default to earliest if targetDate is before
    }


    if (hourIdxForTargetDate !== -1) {
        const weatherTime = new Date(directWeatherData.hourly.time[hourIdxForTargetDate]);
        infoText += `Data for: ${weatherTime.toLocaleString()}\n`;

        const infoWeather = {
            temperature_2m: directWeatherData.hourly.temperature_2m?.[hourIdxForTargetDate],
            snowfall: directWeatherData.hourly.snowfall?.[hourIdxForTargetDate],
            snow_depth: directWeatherData.hourly.snow_depth?.[hourIdxForTargetDate],
            dewpoint_2m: directWeatherData.hourly.dewpoint_2m?.[hourIdxForTargetDate],
            relativehumidity_2m: directWeatherData.hourly.relativehumidity_2m?.[hourIdxForTargetDate],
            windspeed_10m: directWeatherData.hourly.windspeed_10m?.[hourIdxForTargetDate],
            shortwave_radiation: directWeatherData.hourly.shortwave_radiation?.[hourIdxForTargetDate],
            temperature_850hPa: directWeatherData.hourly.temperature_850hPa?.[hourIdxForTargetDate],
            temperature_500hPa: directWeatherData.hourly.temperature_500hPa?.[hourIdxForTargetDate],
            surface_pressure: directWeatherData.hourly.surface_pressure?.[hourIdxForTargetDate],
        };

        infoText += `Temp: ${infoWeather.temperature_2m?.toFixed(1)}°C, Dewpt: ${infoWeather.dewpoint_2m?.toFixed(1)}°C, RH: ${infoWeather.relativehumidity_2m?.toFixed(0)}%\n`;
        infoText += `Wind: ${infoWeather.windspeed_10m?.toFixed(1)}km/h, Rad: ${infoWeather.shortwave_radiation?.toFixed(0)}W/m²\n`;
        infoText += `Snowfall (1h): ${infoWeather.snowfall?.toFixed(1)}cm, Snow Depth: ${infoWeather.snow_depth?.toFixed(0)}cm\n`;
        infoText += `T850: ${infoWeather.temperature_850hPa?.toFixed(1)}°C, T500: ${infoWeather.temperature_500hPa?.toFixed(1)}°C, Pres: ${infoWeather.surface_pressure?.toFixed(0)}hPa\n`;
        
        const accumHours = 24; // Use a constant for now
        const snowMetrics = computeSnowMetrics(directWeatherData.hourly, targetDate, accumHours, snowDisplayMode);
        const modeLabel = snowDisplayMode === 'skiability' ? 'Skiability' : 'SQH Score';
        infoText += `${modeLabel}: ${snowMetrics.finalScore?.toFixed(2)}, Recent Snow: ${snowMetrics.recentAccumCm?.toFixed(1)}cm\n`;

    } else {
         infoText += "Could not match target time in weather data.\n";
    }

  } else {
    infoText += "Weather data unavailable for this point/time.\n";
  }

  info.innerText = infoText;

  // Highlight
  const circ = new ol.Feature( new ol.geom.Circle(coordinate, 60) ); // Use map projection coordinate
  highlightSrc.addFeature(circ);
});

// ESC to clear highlight/info
window.addEventListener('keydown',e=>{
  if (e.key==='Escape'){
    highlightSrc.clear();
    document.getElementById('info-box').style.display='none';
  }
});

/* ---------- event listeners ---------- */
// Snow display mode
document.querySelectorAll('input[name="snowMode"]').forEach(radio => {
  radio.addEventListener('change', event => {
    snowDisplayMode = event.target.value; 
    console.log('Snow display mode set to:', snowDisplayMode);
    if (activeLayer === 'sqh' || (activeLayerType === 'snow' && activeLayer === 'sqh')) {
        pointSource.clear();
        queryViewport(); 
    }
  });
});

// New Layer Item Listeners
document.querySelectorAll('.layer-item').forEach(item => {
    item.addEventListener('click', () => {
        const newActiveLayer = item.dataset.layerName;
        const newActiveLayerType = item.dataset.layerType;

        // Remove 'active' class from all items
        document.querySelectorAll('.layer-item').forEach(otherItem => {
            otherItem.classList.remove('active');
        });
        // Add 'active' class to clicked item
        item.classList.add('active');

        activeLayer = newActiveLayer;
        activeLayerType = newActiveLayerType;
        
        console.log('Active layer changed to:', activeLayer, 'Type:', activeLayerType);
        pointSource.clear();
        queryViewport();
    });
});

// Time input - ID changed to 'timeInput' in HTML
document.getElementById('timeInput').addEventListener('input', event => {
  const timeVal = event.target.value;
  if (timeVal) {
    const [hours, minutes] = timeVal.split(':').map(Number);
    const newDate = new Date(targetDate); // Keep current date, just update time
    newDate.setHours(hours, minutes, 0, 0);
    targetDate = newDate;
    console.log('Target date updated to:', targetDate.toLocaleString());
    
    if (activeLayerType === 'weather' || activeLayer === 'sqh' || (activeLayerType === 'snow' && activeLayer === 'sqh')) {
        pointSource.clear();
        queryViewport();
    } else {
        console.log('Time changed, but active layer is not time-dependent. View not refreshed.');
    }
  }
});

// Removed updateMapDisplay function as its role is now directly handled by queryViewport in event listeners.

/* ---------- cache utilities ---------- */
window.clearCache = async function(){ // Made async
  localStorage.removeItem('powMetrics'); 
  // localStorage.removeItem('elevCache'); // Removed as IndexedDB handles this now
  
  metricsCache = {}; 
  anchorPointsData = {}; 
  
  clearWeatherAPICache(); // For Open-Meteo data
  
  // Clear Terrain Evaluator (IndexedDB) cache
  if (window.clearTerrainEvaluatorCache) {
    try {
      await window.clearTerrainEvaluatorCache(); // Added await
      console.log("Terrain evaluator cache (IndexedDB) cleared successfully.");
    } catch (e) {
      console.error("Error clearing terrain evaluator cache (IndexedDB):", e);
    }
  } else {
    console.warn("clearTerrainEvaluatorCache function not found on window.");
  }
  
  pointSource.clear(); 
  queryViewport(); 
  console.log('🗑️ All caches (metrics, anchor points, weather, terrain) cleared and viewport refreshed.');
};

// Set default active layer on load
function setDefaultActiveLayer() {
    const defaultLayerElement = document.querySelector('.layer-item[data-layer-name="sqh"]');
    if (defaultLayerElement) {
        defaultLayerElement.classList.add('active');
        activeLayer = defaultLayerElement.dataset.layerName;
        activeLayerType = defaultLayerElement.dataset.layerType;
    }
    queryViewport(); // Initial load of data
}

setDefaultActiveLayer(); // Call this instead of direct queryViewport()