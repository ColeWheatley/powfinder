// main.js  – peak-driven powder visualiser
// ----------------------------------------
import { getSnowMetrics }          from './snowModel.js';
import { getRawWeather } from './weatherAPI.js';
import { 
    openDB, // Added import for openDB
    getElevationData, addElevationData, 
    getRawWeatherData, addRawWeatherData, 
    getDerivedData, addDerivedData, 
    clearElevationStore, clearRawWeatherStore, clearDerivedStore,
    countDerivedItemsInView // Ensure this is imported
} from './dbService.js';
import { getExtrapolatedWeatherAtPoint } from './weatherExtrapolation.js';
import { 
    COLOR_LIGHT_BLUE, COLOR_INDIGO, COLOR_HOT_PINK,
    THERMAL_SCALE, SNOW_SCALE, HUMIDITY_SCALE, WIND_SPEED_SCALE, PRESSURE_SCALE,
    ASPECT_COLORS, SLOPE_CATEGORY_COLORS, ELEVATION_SCALE, SLOPE_GRADIENT_SCALE, // Import SLOPE_GRADIENT_SCALE
    interpolateColorLinear, 
    getDynamicScaleGlobal, // Renamed from getDynamicScaleInfo
    calculateDynamicScaleFromFeatures
} from './colorManager.js';

// Import dbService constants if available, otherwise use strings. For this task, using strings.
const DERIVED_STORE_NAME = 'DerivedStore'; // Define if not imported
const ELEVATION_STORE_NAME = 'ElevationStore'; // Define for global elevation scaling

/* ---------- constants ---------- */
const CACHE_TTL_MS   = 60 * 60 * 1000;      // 1 h (re-defined here, also in weatherAPI.js)
const DEBOUNCE_MS    = 120;                 // map move debounce
const OFFSET_M       = 250;                 // N/S/E/W offset distance (m)
const STAGGER_MS     = 20;                  // Reduced stagger for faster anchor processing
const MIN_HEIGHT_CM  = 1;                   // hide points with < 1 cm fresh (used for opacity check)
const MAX_ACCUM_CM_FOR_OPACITY = 50;      // cm of snow for max opacity
const GRID_RESOLUTION = 10;               // Number of cells for grid in queryViewport (e.g., 10x10)
const NUM_CLOSEST_ANCHORS_FOR_EXTRAPOLATION = 5; 

// Predefined min/max for color scaling (simplification) - These remain in main.js
const PREDEFINED_MIN_MAX = {
    temperature_2m: { min: -20, max: 30, scale: THERMAL_SCALE },
    snow_depth: { min: 0, max: 300, scale: SNOW_SCALE },
    snowfall: { min: 0, max: 20, scale: SNOW_SCALE }, // Per hour
    windspeed_10m: { min: 0, max: 100, scale: WIND_SPEED_SCALE },
    elevation: { min: 500, max: 4000, scale: ELEVATION_SCALE }, // Tirol range
    dewpoint_2m: { min: -25, max: 25, scale: THERMAL_SCALE },
    relativehumidity_2m: { min: 0, max: 100, scale: HUMIDITY_SCALE },
    shortwave_radiation: { min: 0, max: 1000, scale: THERMAL_SCALE }, // Can use THERMAL or a dedicated one
    surface_pressure: { min: 950, max: 1050, scale: PRESSURE_SCALE },
    temperature_850hPa: { min: -25, max: 20, scale: THERMAL_SCALE },
    temperature_500hPa: { min: -40, max: -5, scale: THERMAL_SCALE }
};


/* ---------- state variables ---------- */
let snowDisplayMode = 'skiability'; 
let activeLayer = 'skiability';     
let activeLayerType = 'snow_composite'; 
// Initialize targetDate to tomorrow at 12 PM
let targetDate = new Date();
targetDate.setDate(targetDate.getDate() + 1); // Tomorrow
targetDate.setHours(12, 0, 0, 0); // 12 PM
let currentDynamicMinMax = { min: 0, max: 1, calculated: false };    
let isPaused = true; // Initialize isPaused state

/* ---------- DOM Elements ---------- */
// const dateSelector = document.getElementById('dateSelector'); // Old element
// const timeInput = document.getElementById('timeInput'); // Old element
const dayControlButton = document.getElementById('day-control-button');
const timeControlButton = document.getElementById('time-control-button');
const clearCacheGridButton = document.getElementById('clear-cache-grid-button');
// const infoBoxCloseButton = document.getElementById('info-box-close-button'); // Removed

/* ---------- Layer Display Names ---------- */
const LAYER_DISPLAY_NAMES = {
    'skiability': 'Skiability',
    'sqh_raw': 'SQH',
    'temperature_2m': 'Temp (2m)',
    'elevation': 'Elevation',
    'windspeed_10m': 'Wind (10m)',
    'shortwave_radiation': 'Radiation',
    'snowfall': 'Snowfall',
    'snow_depth': 'Snow Depth',
    'dewpoint_2m': 'Dewpoint',
    'relativehumidity_2m': 'Rel. Humidity',
    'temperature_850hPa': 'Temp (850hPa)',
    'temperature_500hPa': 'Temp (500hPa)',
    'surface_pressure': 'Pressure',
    'slope': 'Slope',
    'aspect': 'Aspect'
};

function getShortLayerName(layerId) {
    return LAYER_DISPLAY_NAMES[layerId] || layerId; // Fallback to layerId
}


/* ---------- Helper Functions for Date/Time Display ---------- */
function getDayName(date) { // Remains for full day name when needed by getDisplayDay
    const days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    return days[date.getDay()];
}

function getDisplayDay(dateToCheck) {
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);

    // Normalize dates to midnight for accurate comparison
    const todayMidnight = new Date(today.getFullYear(), today.getMonth(), today.getDate());
    const tomorrowMidnight = new Date(tomorrow.getFullYear(), tomorrow.getMonth(), tomorrow.getDate());
    const dateToCheckMidnight = new Date(dateToCheck.getFullYear(), dateToCheck.getMonth(), dateToCheck.getDate());

    if (dateToCheckMidnight.getTime() === todayMidnight.getTime()) {
        return "Today";
    } else if (dateToCheckMidnight.getTime() === tomorrowMidnight.getTime()) {
        return "Tomorrow";
    } else {
        return getDayName(dateToCheck); // Uses existing getDayName() for "Friday", etc.
    }
}

function formatTimeForButton(date) {
    let hours = date.getHours();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12;
    hours = hours ? hours : 12; // the hour '0' should be '12'
    return `${hours} ${ampm}`;
}

function updateDayTimeButtonText() {
    if (dayControlButton) {
        dayControlButton.textContent = getDisplayDay(targetDate); // Updated
    }
    if (timeControlButton) {
        timeControlButton.textContent = formatTimeForButton(targetDate);
    }
}

function updateInfoBoxTitle() {
    const info = document.getElementById('info-box');
    if (!info) return;

    info.classList.remove('info-box-selecting'); // Remove selector class when setting title

    let title = getShortLayerName(activeLayer);
    const layersWithoutDateTime = ['elevation', 'slope', 'aspect'];
    if (!layersWithoutDateTime.includes(activeLayer)) {
        title += ` - ${getDisplayDay(targetDate)} ${formatTimeForButton(targetDate)}`; // Updated
    }
    
    // Using a specific class for the title text for potential future styling
    let titleDiv = info.querySelector('.info-box-title-text');
    if (!titleDiv) {
        // Clear everything if the title div isn't there (e.g. after map click data)
        // This ensures that when we switch from map click data back to layer info,
        // the old map click data is gone.
        info.innerHTML = ''; 
        titleDiv = document.createElement('div');
        titleDiv.className = 'info-box-title-text';
        info.appendChild(titleDiv);
    }
    titleDiv.textContent = title;
    info.style.display = 'block'; // Ensure it's visible
}

// Helper function to create a gradient legend element
function createGradientLegendElement(minValue, maxValue, colorScaleArray, labelText, unit = "") {
    const legendItem = document.createElement('div');
    legendItem.className = 'legend-item';

    const label = document.createElement('div');
    label.className = 'legend-label';
    label.textContent = labelText;
    legendItem.appendChild(label);

    const bar = document.createElement('div');
    bar.className = 'legend-bar';
    const gradientColors = colorScaleArray.map(rgb => `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`);
    bar.style.background = `linear-gradient(to right, ${gradientColors.join(', ')})`;
    legendItem.appendChild(bar);

    const minLabel = document.createElement('span');
    minLabel.className = 'legend-min-max';
    minLabel.textContent = `${minValue.toFixed(1)}${unit}`;
    legendItem.appendChild(minLabel);

    const maxLabel = document.createElement('span');
    maxLabel.className = 'legend-min-max';
    maxLabel.textContent = `${maxValue.toFixed(1)}${unit}`;
    maxLabel.style.float = 'right'; // Align to the right
    legendItem.appendChild(maxLabel);
    
    // Clear float for subsequent elements
    const clearer = document.createElement('div');
    clearer.style.clear = 'both';
    legendItem.appendChild(clearer);

    return legendItem;
}

// Function to render the legend in the InfoBox
function renderInfoBoxLegend(infoBoxElement, currentActiveLayer, scaleInfo, currentLayerType) {
    // Clear existing legend elements
    const existingLegends = infoBoxElement.querySelectorAll('.legend-container');
    existingLegends.forEach(legend => legend.remove());

    // Exclusion conditions
    if (currentActiveLayer === 'skiability' || currentActiveLayer === 'aspect' || !scaleInfo.calculated) {
        return; 
    }

    const legendContainer = document.createElement('div');
    legendContainer.className = 'legend-container';

    if (currentActiveLayer === 'sqh_raw') {
        // Snow Height Legend (using predefined for snow_depth as recentAccumCm might not have a global scale)
        const snowDepthScaleInfo = PREDEFINED_MIN_MAX.snow_depth;
        if (snowDepthScaleInfo) {
            const snowHeightLegend = createGradientLegendElement(
                snowDepthScaleInfo.min, 
                snowDepthScaleInfo.max, 
                snowDepthScaleInfo.scale, 
                "Snow Height", "cm"
            );
            legendContainer.appendChild(snowHeightLegend);
        }

        // Snow Quality Legend (using scaleInfo for finalScore)
        // The scale for finalScore is implicitly defined by COLOR_LIGHT_BLUE, COLOR_INDIGO, COLOR_HOT_PINK
        // For simplicity, we'll use a representative gradient.
        const qualityScale = [COLOR_LIGHT_BLUE, COLOR_INDIGO, COLOR_HOT_PINK]; 
        const qualityLegend = createGradientLegendElement(
            scaleInfo.min, // min of finalScore
            scaleInfo.max, // max of finalScore
            qualityScale, 
            "Snow Quality Index"
        );
        legendContainer.appendChild(qualityLegend);

    } else if (scaleInfo.scale && Array.isArray(scaleInfo.scale) && scaleInfo.scale.length > 0) {
        // General continuous layers
        let unit = "";
        if (currentActiveLayer.includes("temperature") || currentActiveLayer.includes("dewpoint")) unit = "°C";
        else if (currentActiveLayer === "elevation") unit = "m";
        else if (currentActiveLayer.includes("snow")) unit = "cm"; // snowfall, snow_depth
        else if (currentActiveLayer === "windspeed_10m") unit = "km/h";
        else if (currentActiveLayer === "relativehumidity_2m") unit = "%";
        else if (currentActiveLayer === "shortwave_radiation") unit = "W/m²";
        else if (currentActiveLayer === "surface_pressure") unit = "hPa";


        const generalLegend = createGradientLegendElement(
            scaleInfo.min, 
            scaleInfo.max, 
            scaleInfo.scale, 
            getShortLayerName(currentActiveLayer),
            unit
        );
        legendContainer.appendChild(generalLegend);
    }

    if (legendContainer.hasChildNodes()) {
        infoBoxElement.appendChild(legendContainer);
    }
}


/* ---------- caches and data stores (now primarily in dbService.js) ---------- */
// let metricsCache = {}; // Removed, replaced by DerivedStore
// let anchorPointsData = {}; // Removed, data will be fetched from stores as needed

function key(lat,lon){ return `${lat.toFixed(5)},${lon.toFixed(5)}`; } // Keep for DB keys
function parseKey(k) { const parts = k.split(','); return { lat: parseFloat(parts[0]), lon: parseFloat(parts[1]) }; } // Keep for parsing DB keys


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
// Styling function for pointLayer
const pointLayerStyleFunction = (feature) => {
    const featureId = feature.getId() || 'unknown_id';
    let currentOpacity = 0.8; // Default opacity
    let colorToUse;
    const value = feature.get(activeLayer);

    if (activeLayer === 'skiability' || activeLayer === 'sqh_raw') {
        const finalScore = feature.get('finalScore');
        const recentAccumCm = feature.get('recentAccumCm');
        if (recentAccumCm == null || recentAccumCm < MIN_HEIGHT_CM || finalScore == null) return null;
        
        // Color categories for skiability are fixed, not using currentDynamicMinMax for color here.
        // Opacity is dynamic based on recentAccumCm.
        if (finalScore < 0.33) colorToUse = COLOR_LIGHT_BLUE;
        else if (finalScore < 0.66) colorToUse = COLOR_INDIGO;
        else colorToUse = COLOR_HOT_PINK;
        currentOpacity = Math.max(0.2, Math.min(recentAccumCm / MAX_ACCUM_CM_FOR_OPACITY, 1));

    } else if (activeLayerType === 'terrain') {
        if (value == null && activeLayer !== 'aspect') return null; // Allow null for aspect if 'flat' or 'default' is handled
        switch (activeLayer) {
            case 'elevation':
                // Use dynamic scale if calculated (from DerivedStore), otherwise predefined
                const elevScaleInfo = currentDynamicMinMax.calculated ? currentDynamicMinMax : PREDEFINED_MIN_MAX.elevation;
                colorToUse = interpolateColorLinear(value, elevScaleInfo.min, elevScaleInfo.max, ELEVATION_SCALE);
                break;
            case 'slope':
                if (value != null) {
                    // Use fixed 0-90 range for slope coloring
                    colorToUse = interpolateColorLinear(value, 0, 90, SLOPE_GRADIENT_SCALE);
                } else {
                    colorToUse = ASPECT_COLORS.default; // Default color if slope value is null
                }
                break;
            case 'aspect':
                // Aspect uses fixed colors, not dynamic scaling.
                colorToUse = ASPECT_COLORS[(value || 'default').toString().toLowerCase()] || ASPECT_COLORS.default;
                break;
            default: return null;
        }
    } else if (activeLayerType === 'weather') {
        if (value == null) return null;
        const scaleDef = PREDEFINED_MIN_MAX[activeLayer];
        if (!scaleDef) return null; // Should not happen if layer selectors are correct

        // For weather layers, currentDynamicMinMax is typically from features or predefined.
        let minToUse, maxToUse;
        if (currentDynamicMinMax.calculated) { 
            minToUse = currentDynamicMinMax.min;
            maxToUse = currentDynamicMinMax.max;
        } else { // Fallback to predefined for weather
            minToUse = scaleDef.min;
            maxToUse = scaleDef.max;
        }
        colorToUse = interpolateColorLinear(value, minToUse, maxToUse, scaleDef.scale, activeLayer === 'shortwave_radiation');
    } else {
        return null; // Should not happen
    }

    if (!colorToUse) return null; // If color couldn't be determined

    const [r, g, b] = colorToUse;
    const finalFillColor = `rgba(${r},${g},${b},${currentOpacity.toFixed(2)})`;
    
    return new ol.style.Style({
        image: new ol.style.Circle({
            radius: 6,
            fill: new ol.style.Fill({ color: finalFillColor }),
            stroke: null // No stroke for cleaner look
        })
    });
};

const pointLayer = new ol.layer.Vector({
  source: pointSource,
  style: pointLayerStyleFunction
});
map.addLayer(pointLayer);

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
  if (closestIdx === -1 && timeArray.length > 0) return 0; // Default to earliest if targetDate is before
  return closestIdx;
}

function getVisiblePeaks(mapExtent, allPeaks) { 
    const [minLon,minLat,maxLon,maxLat] = mapExtent; // Assuming mapExtent is [minLon, minLat, maxLon, maxLat]
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

function haversineDistance(lat1, lon1, lat2, lon2) {
    const R = 6371e3; 
    const φ1 = lat1 * Math.PI/180;
    const φ2 = lat2 * Math.PI/180;
    const Δφ = (lat2-lat1) * Math.PI/180;
    const Δλ = (lon2-lon1) * Math.PI/180;
    const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) + Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ/2) * Math.sin(Δλ/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c; 
}

async function fetchMetricsForAnchor(lat, lon, peakName, isAdjacent) { 
  const derivedKey = key(lat, lon);
  let cachedData = null;
  try {
    cachedData = await getDerivedData(derivedKey);
    if (cachedData && (Date.now() - cachedData.ts < CACHE_TTL_MS)) {
      // console.log(`DerivedStore Cache hit for ${derivedKey} (Peak: ${peakName}, Adj: ${isAdjacent})`);
      return cachedData;
    }
  } catch (dbError) {
    console.error(`Error fetching derived data for ${derivedKey} from DB:`, dbError);
  }

  // If not cached or expired, or if fetching failed previously, try to fetch new data
  // Respect isPaused state if we need to fetch new raw weather data
  if (isPaused && (!cachedData || Date.now() - cachedData.ts >= CACHE_TTL_MS)) {
      console.log(`Paused: Skipping new data processing for anchor ${derivedKey} (Peak: ${peakName}, Adj: ${isAdjacent})`);
      return cachedData; // Return stale data if available, or null
  }

  // If not cached or expired, fetch raw weather and compute metrics
  const rawWeatherData = await getRawWeather(lat, lon, { peakName, isAdjacent }); 
  if (!rawWeatherData || !rawWeatherData.hourly) { 
    console.warn(`Failed to get raw weather for anchor ${derivedKey} (Peak: ${peakName}, Adj: ${isAdjacent})`); 
    return null; 
  }
  
  const { finalScore, recentAccumCm } = computeSnowMetrics(rawWeatherData.hourly, targetDate, 24, snowDisplayMode);
  const derivedEntry = { 
    id: derivedKey, 
    finalScore, 
    recentAccumCm, 
    ts: Date.now(), 
    lat, 
    lon 
  };

  try {
    await addDerivedData(derivedEntry);
    // console.log(`Added derived data to DB for ${derivedKey} (Peak: ${peakName}, Adj: ${isAdjacent})`);
  } catch (dbError) {
    console.error(`Error adding derived data for ${derivedKey} to DB:`, dbError);
  }
  
  return derivedEntry;
}

async function queryViewport() {
  console.log('Starting queryViewport...');
  await peaksPromise; 
  const view = map.getView();
  const olExtent = view.calculateExtent(map.getSize());
  const mapExtent = ol.proj.transformExtent(olExtent, 'EPSG:3857', 'EPSG:4326'); 
  
  console.log('Phase 1: Ensuring Anchor Point Data in DB Stores...');
  const visiblePeaksArray = getVisiblePeaks(mapExtent, peaks);
  const uniqueAnchorPointKeys = new Set();
  
  // Populate uniqueAnchorPointKeys with main peaks and their offsets
  visiblePeaksArray.forEach(p => {
    const [lon, lat] = p.coordinates;
    uniqueAnchorPointKeys.add({ id: key(lat, lon), peakName: p.name, isAdjacent: false, lat, lon });
    const dLatDeg = dLat(OFFSET_M); 
    const dLonDeg = dLon(lat, OFFSET_M);
    uniqueAnchorPointKeys.add({ id: key(lat + dLatDeg, lon), peakName: p.name, isAdjacent: true, lat: lat + dLatDeg, lon });
    uniqueAnchorPointKeys.add({ id: key(lat - dLatDeg, lon), peakName: p.name, isAdjacent: true, lat: lat - dLatDeg, lon });
    uniqueAnchorPointKeys.add({ id: key(lat, lon + dLonDeg), peakName: p.name, isAdjacent: true, lat: lat, lon: lon + dLonDeg });
    uniqueAnchorPointKeys.add({ id: key(lat, lon - dLonDeg), peakName: p.name, isAdjacent: true, lat: lat, lon: lon - dLonDeg });
  });
  console.log(`Identified ${uniqueAnchorPointKeys.size} unique anchor points (peaks + offsets) for DB population.`);

  let anchorProcessingPromises = [];
  let idx = 0;
  // This set is to avoid processing the same lat/lon key multiple times if peaks overlap with offsets etc.
  const processedAnchorLatLonKeys = new Set(); 

  for (const anchorInfo of uniqueAnchorPointKeys) {
      if (processedAnchorLatLonKeys.has(anchorInfo.id)) {
          continue;
      }
      processedAnchorLatLonKeys.add(anchorInfo.id);

      const promise = new Promise(resolve => setTimeout(async () => {
          try {
              // Ensure elevation is in ElevationStore
              await getElevation(anchorInfo.lat, anchorInfo.lon); // This function now handles its own caching via dbService
              
              // Ensure raw weather is in RawWeatherStore and derived metrics are in DerivedStore
              await fetchMetricsForAnchor(anchorInfo.lat, anchorInfo.lon, anchorInfo.peakName, anchorInfo.isAdjacent);
              
          } catch (error) { 
              console.warn(`Error processing anchor ${anchorInfo.id} (Peak: ${anchorInfo.peakName}, Adj: ${anchorInfo.isAdjacent}):`, error); 
          } 
          finally { resolve(); }
      }, idx * STAGGER_MS));
      anchorProcessingPromises.push(promise); 
      idx++;
  }
  await Promise.all(anchorProcessingPromises);
  console.log(`Phase 1 Complete. Anchor point data population/update tasks finished for ${processedAnchorLatLonKeys.size} unique lat/lon keys.`);
  
  console.log('Phase 2: Processing Grid Points for Display...');
  pointSource.clear();
  const gridCells = generateGridForExtent(mapExtent, GRID_RESOLUTION);
  
  let derivedPointsInViewCount = 0;
  try {
    // Actual call to countDerivedItemsInView
    derivedPointsInViewCount = await countDerivedItemsInView(mapExtent[1], mapExtent[3], mapExtent[0], mapExtent[2]);
    console.log(`Derived points already in view (DB): ${derivedPointsInViewCount}`);
  } catch (e) {
    console.warn("countDerivedItemsInView function failed, defaulting to 0:", e);
    derivedPointsInViewCount = 0; // Default to 0 on error to allow extrapolation
  }

  let featuresForDynamicScaling = [];
  let featuresAdded = 0;

  // Create a list of all potential anchor *keys* from uniqueAnchorPointKeys (which is a Set of objects)
  // These keys will be used to fetch data from DB for extrapolation sources.
  const allPotentialAnchorDbKeys = Array.from(uniqueAnchorPointKeys).map(anchorObj => anchorObj.id);

  if (!isPaused) {
    for (const cell of gridCells) {
      try {
        const gridLat = cell.lat; 
        const gridLon = cell.lon;
      const gridCellKey = key(gridLat, gridLon);
      
      const gridPointTerrain = await getElevation(gridLat, gridLon); // getElevation handles its own DB interaction
      if (!gridPointTerrain || gridPointTerrain.elevation == null) {
        // console.log(`Skipping grid cell ${gridCellKey}: no terrain data.`);
        continue;
      }

      let derivedCellData = null;
      try {
        derivedCellData = await getDerivedData(gridCellKey);
      } catch (dbError) {
        console.error(`queryViewport: Error getting derived data for grid cell ${gridCellKey} from DB:`, dbError);
        // Attempt to continue and extrapolate if possible
      }

      // Check if we need to extrapolate and populate derivedCellData
      // For now, we'll always try to extrapolate if not found, and refine with play button state later
      if (!derivedCellData && derivedPointsInViewCount < 1000) { 
        // console.log(`No derived data for ${gridCellKey}, attempting extrapolation. Current count: ${derivedPointsInViewCount}`);
        
        // Find closest source anchors based on distance to this grid cell
        // We need full anchor objects (lat, lon) to calculate distance, then use their keys to fetch from DB
        const sortedPotentialAnchors = allPotentialAnchorDbKeys.map(anchorDbKey => {
            const { lat: anchorLat, lon: anchorLon } = parseKey(anchorDbKey); // Parse key to get lat/lon
            return { 
                id: anchorDbKey, 
                lat: anchorLat, 
                lon: anchorLon, 
                distance: haversineDistance(gridLat, gridLon, anchorLat, anchorLon) 
            };
        }).sort((a, b) => a.distance - b.distance);
        
        const closestSourceAnchorInfos = sortedPotentialAnchors.slice(0, NUM_CLOSEST_ANCHORS_FOR_EXTRAPOLATION);

        if (closestSourceAnchorInfos.length > 0) {
            const sourcePointsForExtrapolation = [];
            for (const anchorInfo of closestSourceAnchorInfos) {
                let anchorTerrain = null;
                let anchorWeatherRaw = null;
                try {
                    anchorTerrain = await getElevationData(anchorInfo.id); // Fetch from ElevationStore
                } catch (dbError) {
                    console.error(`queryViewport: Error getting elevation data for anchor ${anchorInfo.id} from DB:`, dbError);
                    continue; // Skip this anchor if DB read fails
                }
                try {
                    anchorWeatherRaw = await getRawWeatherData(anchorInfo.id); // Fetch from RawWeatherStore
                } catch (dbError) {
                    console.error(`queryViewport: Error getting raw weather data for anchor ${anchorInfo.id} from DB:`, dbError);
                    continue; // Skip this anchor if DB read fails
                }
                
                if (anchorTerrain && anchorWeatherRaw && anchorWeatherRaw.apiResponse && anchorWeatherRaw.apiResponse.hourly) {
                    sourcePointsForExtrapolation.push({
                        lat: anchorInfo.lat,
                        lon: anchorInfo.lon,
                        elevation: anchorTerrain.elevation,
                        slope: anchorTerrain.slope, 
                        aspect: anchorTerrain.aspect,
                        weather: anchorWeatherRaw.apiResponse.hourly // Pass the raw hourly data
                    });
                } else {
                    // console.warn(`Missing terrain or weather data for potential source anchor ${anchorInfo.id}`);
                }
            }

            if (sourcePointsForExtrapolation.length > 0) {
                const extrapolatedWeather = getExtrapolatedWeatherAtPoint(gridPointTerrain, sourcePointsForExtrapolation);
                if (extrapolatedWeather && extrapolatedWeather.time && extrapolatedWeather.time.length > 0) {
                    const metrics = computeSnowMetrics(extrapolatedWeather, targetDate, 24, snowDisplayMode);
                    derivedCellData = {
                        id: gridCellKey,
                        lat: gridLat,
                        lon: gridLon,
                        elevation: gridPointTerrain.elevation, // Store base elevation from gridPointTerrain
                        slope: gridPointTerrain.slope,         // Store slope
                        aspect: gridPointTerrain.aspect,       // Store aspect
                        finalScore: metrics.finalScore,        // From computeSnowMetrics
                        recentAccumCm: metrics.recentAccumCm,  // From computeSnowMetrics
                        weatherData: extrapolatedWeather,      // Full extrapolated weather data
                        sourceAnchorIds: closestSourceAnchorInfos.map(a => a.id),
                        ts: Date.now() // Timestamp for this derived data
                    };
                    try {
                        await addDerivedData(derivedCellData);
                        derivedPointsInViewCount++; // Increment local count
                        // console.log(`Extrapolated and saved derived data for ${gridCellKey}. New count: ${derivedPointsInViewCount}`);
                    } catch (dbError) {
                        console.error(`queryViewport: Error adding derived data for grid cell ${gridCellKey} to DB:`, dbError);
                        derivedCellData = null; // Nullify if DB write fails, so it's not added to map
                    }
                } else {
                    // console.log(`Extrapolation failed for ${gridCellKey}.`);
                }
            } else {
                // console.log(`Not enough valid source points for extrapolation for ${gridCellKey}.`);
            }
        } else {
            // console.log(`No closest source anchors found for ${gridCellKey}.`);
        }
      } else if (derivedCellData) {
        // console.log(`Using existing derived data for ${gridCellKey}`);
      } else {
        // console.log(`Skipping extrapolation for ${gridCellKey} due to derivedPointsInViewCount: ${derivedPointsInViewCount}`);
      }

      // If we have derivedCellData (either fetched or newly created), create a feature
      if (derivedCellData) {
        const featureData = {
            finalScore: derivedCellData.finalScore,
            recentAccumCm: derivedCellData.recentAccumCm,
            elevation: derivedCellData.elevation, // Use elevation from derivedCellData (which came from gridPointTerrain)
            slope: derivedCellData.slope,
            aspect: derivedCellData.aspect,
            lat: gridLat,
            lon: gridLon
        };

        // Populate weather variables for the active layer from derivedCellData.weatherData
        if (derivedCellData.weatherData && derivedCellData.weatherData.time) {
            const hourIdx = getHourIndexForTargetDate(derivedCellData.weatherData.time, targetDate);
            if (hourIdx !== -1) {
                for (const varName in PREDEFINED_MIN_MAX) {
                    if (derivedCellData.weatherData[varName] && derivedCellData.weatherData[varName][hourIdx] !== undefined) {
                        featureData[varName] = derivedCellData.weatherData[varName][hourIdx];
                    }
                }
            }
        }
        
        const feature = new ol.Feature({ 
            geometry: new ol.geom.Point(ol.proj.fromLonLat([gridLon, gridLat])), 
            ...featureData 
        });
        feature.setId(gridCellKey);
        featuresForDynamicScaling.push(feature);
        featuresAdded++;
      }
    } catch (error) { 
        console.warn(`Error processing grid cell ${cell.lat},${cell.lon}:`, error); 
      }
    }
  } else {
    console.log("Paused: Skipping grid extrapolation loop.");
    // If paused and no features were prepared from anchors (which might still load from DB)
    // and featuresForDynamicScaling is empty, clear the map.
    if (featuresForDynamicScaling.length === 0) {
        console.log("Paused: No features to display, clearing point source.");
        pointSource.clear(); // Clear map if nothing to show
    }
  }
  
  // Dynamic scaling logic updated
  let scaleVariableName = activeLayer;
  if (activeLayer === 'skiability' || activeLayer === 'sqh_raw') {
    scaleVariableName = 'finalScore'; // Actual data field for skiability layers
  } else if (activeLayer === 'temperature_2m' || activeLayer === 'dewpoint_2m' || 
             activeLayer === 'windspeed_10m' || activeLayer === 'shortwave_radiation' ||
             activeLayer === 'temperature_850hPa' || activeLayer === 'temperature_500hPa' ||
             activeLayer === 'snowfall' || activeLayer === 'snow_depth' || activeLayer === 'surface_pressure' ||
             activeLayer === 'relativehumidity_2m') {
    // For weather variables, they are nested under weatherData in DerivedStore
    // However, getDynamicScaleInfo is currently set to only work on DerivedStore with direct props or one level (e.g. weatherData.temperature_2m)
    // If we want global scale for these, DerivedStore items need these props directly or getDynamicScaleInfo needs deeper path resolution.
    // For now, we'll assume these use feature-based scaling or predefined.
    // The new getDynamicScaleInfo for 'DerivedStore' expects variableName to be a direct prop or one level deep.
    // Let's try to use the new one if it's a weather layer that *could* be in DerivedStore (hypothetically)
    // but rely on its internal check for storeName === 'DerivedStore'
    // This structure means we'll call it, it will likely return calculated:false, and we fallback.
    scaleVariableName = `weatherData.${activeLayer}`; // Path for items in DerivedStore
  }


  // Determine if global scaling from DerivedStore should be attempted
  // Determine dynamic scaling
  if (activeLayer === 'elevation') {
    currentDynamicMinMax = await getDynamicScaleGlobal('elevation', dbService, ELEVATION_STORE_NAME);
    if (!currentDynamicMinMax.calculated) {
      console.log(`Global DB scale for elevation failed, falling back to viewport-based.`);
      if (featuresForDynamicScaling.length > 0) {
        currentDynamicMinMax = calculateDynamicScaleFromFeatures(featuresForDynamicScaling, 'elevation');
        currentDynamicMinMax.source = 'viewport'; 
      } else {
        console.log(`Viewport-based scale for elevation failed, falling back to predefined.`);
        currentDynamicMinMax = { ...PREDEFINED_MIN_MAX.elevation, calculated: false, source: 'predefined' };
      }
    }
  } else if (activeLayer === 'skiability' || activeLayer === 'sqh_raw') {
    const dbQueryVariableName = 'finalScore'; // 'finalScore' is the field in DerivedStore
    currentDynamicMinMax = await getDynamicScaleGlobal(dbQueryVariableName, dbService, DERIVED_STORE_NAME);
    if (!currentDynamicMinMax.calculated) {
      console.log(`Global DB scale for ${dbQueryVariableName} failed, falling back to viewport-based.`);
      if (featuresForDynamicScaling.length > 0) {
        // calculateDynamicScaleFromFeatures expects the feature's direct property name.
        // For skiability/sqh_raw, features have 'finalScore' property from derivedCellData.
        currentDynamicMinMax = calculateDynamicScaleFromFeatures(featuresForDynamicScaling, 'finalScore');
        currentDynamicMinMax.source = 'viewport';
      } else {
        console.log(`Viewport-based scale for ${dbQueryVariableName} failed, falling back to predefined.`);
        // PREDEFINED_MIN_MAX uses 'skiability' or 'sqh_raw' as keys, not 'finalScore'
        const predefinedKey = activeLayer; 
        currentDynamicMinMax = PREDEFINED_MIN_MAX[predefinedKey] 
            ? { ...PREDEFINED_MIN_MAX[predefinedKey], calculated: false, source: 'predefined' }
            : { min: 0, max: 1, calculated: false, source: 'defaultPredefined' };
      }
    }
  } else if (activeLayerType === 'weather') {
    // For weather layers, use feature-based scaling or predefined, not global DB for now.
    // (Global DB for weather could be complex due to time series data in RawWeatherStore)
    if (featuresForDynamicScaling.length > 0) {
        currentDynamicMinMax = calculateDynamicScaleFromFeatures(featuresForDynamicScaling, activeLayer);
        currentDynamicMinMax.source = 'viewport';
    } else {
        const predefined = PREDEFINED_MIN_MAX[activeLayer];
        currentDynamicMinMax = predefined 
            ? { ...predefined, calculated: false, source: 'predefined' } 
            : { min: 0, max: 1, calculated: false, source: 'defaultPredefined' };
    }
  } else if (activeLayerType === 'terrain' && activeLayer === 'slope') { 
    // Slope uses a fixed 0-90 scale for coloring, so currentDynamicMinMax is not strictly needed for color.
    // However, we can still calculate it from features for potential display in info or legend.
    if (featuresForDynamicScaling.length > 0) {
        currentDynamicMinMax = calculateDynamicScaleFromFeatures(featuresForDynamicScaling, 'slope');
        currentDynamicMinMax.source = 'viewport';
    } else {
        currentDynamicMinMax = { min: 0, max: 90, calculated: false, source: 'fixedRange' }; // Default for slope
    }
  } else { // Other terrain layers like aspect, or if logic missed something
    console.log(`Layer ${activeLayer} does not use dynamic scaling by default. Using predefined if available.`);
    const predefined = PREDEFINED_MIN_MAX[activeLayer];
    currentDynamicMinMax = predefined 
        ? { ...predefined, calculated: false, source: 'predefined' }
        : { min: 0, max: 1, calculated: false, source: 'defaultPredefined' };
  }

  // Log the final decision for scaling
  if (currentDynamicMinMax.calculated) {
    console.log(`Using dynamic scale for ${activeLayer}: min=${currentDynamicMinMax.min.toFixed(2)}, max=${currentDynamicMinMax.max.toFixed(2)} (Source: ${currentDynamicMinMax.source || 'unknown'})`);
  } else {
    console.log(`Using predefined/default scale for ${activeLayer}: min=${currentDynamicMinMax.min}, max=${currentDynamicMinMax.max} (Source: ${currentDynamicMinMax.source || 'unknownPredefined'})`);
  }
  
  updateInfoBoxTitle(); // Update title first
  const infoBoxElement = document.getElementById('info-box');
  if (infoBoxElement) {
      renderInfoBoxLegend(infoBoxElement, activeLayer, currentDynamicMinMax, activeLayerType);
  }


  pointSource.addFeatures(featuresForDynamicScaling);
  pointSource.changed(); 

  console.log(`Phase 2 Complete. Added ${featuresAdded} features to map. Dynamic scale updated for ${activeLayer}.`);

  // Call fillMissingDataSpirally after Phase 2 - unconditional for now
  // Assuming 'peaks' is available globally from the peaksPromise
  if (typeof fillMissingDataSpirally === 'function' && peaks.length > 0) {
    // Pass the actual isPaused state
    fillMissingDataSpirally(map, pointSource, peaks, allPotentialAnchorDbKeys, mapExtent, isPaused); 
  } else {
    console.warn("fillMissingDataSpirally function not available or no peaks loaded for spiral fill.");
  }
}

const SPIRAL_GRID_RESOLUTION_MULTIPLIER = 2; // For a denser grid in spiral fill
const MAX_SPIRAL_FILL_POINTS_PER_CALL = 20; // Max points to fill in one call
const MAX_SPIRAL_ITERATIONS_PER_PEAK = 100; // Max spiral steps around a peak to avoid infinite loops

async function fillMissingDataSpirally(mapInstance, olPointSource, allPeaks, anchorKeysForExtrapolationSources, mapExtent, isPaused) {
  console.log("Attempting spiral fill for missing data...");
  if (isPaused) {
    console.log("Spiral Fill: Paused, skipping.");
    return;
  }

  // mapExtent is now passed as a parameter, so we don't need to recalculate it here.
  // const view = mapInstance.getView();
  // const olExtent = view.calculateExtent(mapInstance.getSize());
  // const mapExtent = ol.proj.transformExtent(olExtent, 'EPSG:3857', 'EPSG:4326');
  
  const visiblePeaks = getVisiblePeaks(mapExtent, allPeaks);
  const olExtent = view.calculateExtent(mapInstance.getSize());
  const mapExtent = ol.proj.transformExtent(olExtent, 'EPSG:3857', 'EPSG:4326');
  
  const visiblePeaks = getVisiblePeaks(mapExtent, allPeaks);
  if (visiblePeaks.length === 0) {
    console.log("Spiral Fill: No visible peaks, skipping.");
    return;
  }

  const processingGridRes = GRID_RESOLUTION * SPIRAL_GRID_RESOLUTION_MULTIPLIER;
  const processingGrid = generateGridForExtent(mapExtent, processingGridRes);
  if (processingGrid.length === 0) {
    console.log("Spiral Fill: Processing grid is empty, skipping.");
    return;
  }

  let pointsFilledThisCall = 0;
  const visitedSpiralCells = new Set(); // Tracks 'lat,lon' strings of visited spiral cells globally

  // anchorKeysForExtrapolationSources is now directly passed (it was already an array of keys).
  // No need to convert from allAnchorPointDbPrimaryKeys here.

  for (const peak of visiblePeaks) {
    if (pointsFilledThisCall >= MAX_SPIRAL_FILL_POINTS_PER_CALL) break;

    const [peakLon, peakLat] = peak.coordinates;

    // Find the closest cell in processingGrid to this peak's actual coordinates
    let closestGridCell = null;
    let minDistSq = Infinity;
    for (const cell of processingGrid) {
      const distSq = (cell.lon - peakLon)**2 + (cell.lat - peakLat)**2;
      if (distSq < minDistSq) {
        minDistSq = distSq;
        closestGridCell = cell;
      }
    }

    if (!closestGridCell) continue;

    // Spiral logic starts here (simple expanding box for now)
    // Convert closestGridCell lat/lon to indices in the conceptual processingGrid for easier spiral
    // This assumes processingGrid is ordered and consistent, which generateGridForExtent provides.
    // However, direct coordinate manipulation for spiral steps is more robust.
    
    let currentLayer = 0; // Spiral layer
    let spiralIterationsForThisPeak = 0;

    // Start spiral from the actual coordinates of the closestGridCell
    let currentSpiralLat = closestGridCell.lat;
    let currentSpiralLon = closestGridCell.lon;
    const lonStep = (mapExtent[2] - mapExtent[0]) / processingGridRes; // lon step for processing grid
    const latStep = (mapExtent[3] - mapExtent[1]) / processingGridRes; // lat step for processing grid


    // Process the starting cell first
    const startCellKey = key(currentSpiralLat, currentSpiralLon);
    if (!visitedSpiralCells.has(startCellKey)) {
      visitedSpiralCells.add(startCellKey);
      // Pass mapExtent to processSpiralCell
      const filled = await processSpiralCell(startCellKey, currentSpiralLat, currentSpiralLon, olPointSource, anchorKeysForExtrapolationSources, mapExtent);
      if (filled) pointsFilledThisCall++;
      spiralIterationsForThisPeak++;
    }
    
    // Expanding box spiral logic
    for (let l = 1; l < Math.max(processingGridRes, MAX_SPIRAL_ITERATIONS_PER_PEAK / 4) ; l++) { // Iterate through layers
        if (pointsFilledThisCall >= MAX_SPIRAL_FILL_POINTS_PER_CALL || spiralIterationsForThisPeak >= MAX_SPIRAL_ITERATIONS_PER_PEAK) break;

        // Top row (from left to right)
        for (let i = -l + 1; i <= l; i++) {
            if (pointsFilledThisCall >= MAX_SPIRAL_FILL_POINTS_PER_CALL || spiralIterationsForThisPeak >= MAX_SPIRAL_ITERATIONS_PER_PEAK) break;
            const lon = closestGridCell.lon + i * lonStep;
            const lat = closestGridCell.lat - l * latStep; // Top means decreasing latitude
            const cellKey = key(lat, lon);
            if (!visitedSpiralCells.has(cellKey) && lon >= mapExtent[0] && lon <= mapExtent[2] && lat >= mapExtent[1] && lat <= mapExtent[3]) {
                visitedSpiralCells.add(cellKey);
                // Pass mapExtent to processSpiralCell
                const filled = await processSpiralCell(cellKey, lat, lon, olPointSource, anchorKeysForExtrapolationSources, mapExtent);
                if (filled) pointsFilledThisCall++;
                spiralIterationsForThisPeak++;
            }
        }
        // Right col (from top to bottom)
        for (let i = -l + 1; i <= l; i++) {
            if (pointsFilledThisCall >= MAX_SPIRAL_FILL_POINTS_PER_CALL || spiralIterationsForThisPeak >= MAX_SPIRAL_ITERATIONS_PER_PEAK) break;
            const lon = closestGridCell.lon + l * lonStep;
            const lat = closestGridCell.lat + i * latStep;
            const cellKey = key(lat, lon);
             if (!visitedSpiralCells.has(cellKey) && lon >= mapExtent[0] && lon <= mapExtent[2] && lat >= mapExtent[1] && lat <= mapExtent[3]) {
                visitedSpiralCells.add(cellKey);
                // Pass mapExtent to processSpiralCell
                const filled = await processSpiralCell(cellKey, lat, lon, olPointSource, anchorKeysForExtrapolationSources, mapExtent);
                if (filled) pointsFilledThisCall++;
                spiralIterationsForThisPeak++;
            }
        }
        // Bottom row (from right to left)
        for (let i = l - 1; i >= -l; i--) {
            if (pointsFilledThisCall >= MAX_SPIRAL_FILL_POINTS_PER_CALL || spiralIterationsForThisPeak >= MAX_SPIRAL_ITERATIONS_PER_PEAK) break;
            const lon = closestGridCell.lon + i * lonStep;
            const lat = closestGridCell.lat + l * latStep;
            const cellKey = key(lat, lon);
            if (!visitedSpiralCells.has(cellKey) && lon >= mapExtent[0] && lon <= mapExtent[2] && lat >= mapExtent[1] && lat <= mapExtent[3]) {
                visitedSpiralCells.add(cellKey);
                // Pass mapExtent to processSpiralCell
                const filled = await processSpiralCell(cellKey, lat, lon, olPointSource, anchorKeysForExtrapolationSources, mapExtent);
                if (filled) pointsFilledThisCall++;
                spiralIterationsForThisPeak++;
            }
        }
        // Left col (from bottom to top)
        for (let i = l - 1; i >= -l + 1; i--) { // Avoid double counting corner with top row
            if (pointsFilledThisCall >= MAX_SPIRAL_FILL_POINTS_PER_CALL || spiralIterationsForThisPeak >= MAX_SPIRAL_ITERATIONS_PER_PEAK) break;
            const lon = closestGridCell.lon - l * lonStep;
            const lat = closestGridCell.lat + i * latStep;
            const cellKey = key(lat, lon);
            if (!visitedSpiralCells.has(cellKey) && lon >= mapExtent[0] && lon <= mapExtent[2] && lat >= mapExtent[1] && lat <= mapExtent[3]) {
                visitedSpiralCells.add(cellKey);
                // Pass mapExtent to processSpiralCell
                const filled = await processSpiralCell(cellKey, lat, lon, olPointSource, anchorKeysForExtrapolationSources, mapExtent);
                if (filled) pointsFilledThisCall++;
                spiralIterationsForThisPeak++;
            }
        }
    }
  } // End loop over visiblePeaks

  if (pointsFilledThisCall > 0) {
    console.log(`Spiral Fill: Added ${pointsFilledThisCall} new points to the map.`);
    olPointSource.changed(); // Trigger map refresh if points were added
  } else {
    console.log("Spiral Fill: No new points added in this call.");
  }
}

async function processSpiralCell(cellKey, lat, lon, olPointSource, anchorKeysForExtrapolationSources, mapExtent) {
    let existingData = null;
    try {
        existingData = await getDerivedData(cellKey);
        if (existingData) {
            // console.log(`Spiral Fill: Data already exists for ${cellKey}, skipping.`);
            return false; // Not filled in this call
        }
    } catch (dbError) {
        console.error(`Spiral Fill: Error checking existing derived data for ${cellKey}. Returning false.`, dbError);
        return false; 
    }

    // Implement 1,000 Point Viewport Limit
    try {
        const derivedPointsInViewport = await countDerivedItemsInView(mapExtent[1], mapExtent[3], mapExtent[0], mapExtent[2]);
        if (derivedPointsInViewport >= 1000) {
            console.log(`Spiral Fill: Viewport limit (1000 points) reached. Skipping further processing for cell ${cellKey}. Points in view: ${derivedPointsInViewport}`);
            return false; // Point not filled
        }
    } catch (dbError) { // Renamed 'e' to 'dbError' for clarity
        console.warn(`Spiral Fill: Error counting derived items in view for cell ${cellKey}. Proceeding with caution.`, dbError);
        // Allow to proceed if count fails, as it's an optimization
    }
    
    let gridPointTerrain = null;
    try {
        gridPointTerrain = await getElevation(lat, lon); 
    } catch (terrainError) { // Catch errors specifically from getElevation or its underlying calls
        console.error(`Spiral Fill: Error getting terrain data (via getElevation) for ${cellKey}:`, terrainError);
        return false; // Cannot proceed without terrain
    }
    if (!gridPointTerrain || gridPointTerrain.elevation == null) {
        // console.log(`Spiral Fill: No terrain data for ${cellKey}`);
        return false;
    }

    // Find closest source anchors
    const sortedPotentialAnchors = anchorKeysForExtrapolationSources.map(anchorDbKey => {
        const { lat: anchorLat, lon: anchorLon } = parseKey(anchorDbKey);
        return { 
            id: anchorDbKey, 
            lat: anchorLat, 
            lon: anchorLon, 
            distance: haversineDistance(lat, lon, anchorLat, anchorLon) 
        };
    }).sort((a, b) => a.distance - b.distance);
    
    const closestSourceAnchorInfos = sortedPotentialAnchors.slice(0, NUM_CLOSEST_ANCHORS_FOR_EXTRAPOLATION);

    if (closestSourceAnchorInfos.length === 0) {
        // console.log(`Spiral Fill: No source anchors found for ${cellKey}`);
        return false;
    }

    const sourcePointsForExtrapolation = [];
    for (const anchorInfo of closestSourceAnchorInfos) {
        let anchorTerrain = null;
        let anchorWeatherRaw = null;
        try {
            anchorTerrain = await getElevationData(anchorInfo.id);
        } catch (dbError) {
            console.error(`Spiral Fill: Error getting elevation data for source anchor ${anchorInfo.id} from DB:`, dbError);
            continue; // Skip this source point if DB error
        }
        try {
            anchorWeatherRaw = await getRawWeatherData(anchorInfo.id);
        } catch (dbError) {
            console.error(`Spiral Fill: Error getting raw weather data for source anchor ${anchorInfo.id} from DB:`, dbError);
            continue; // Skip this source point if DB error
        }
        
        if (anchorTerrain && anchorWeatherRaw && anchorWeatherRaw.apiResponse && anchorWeatherRaw.apiResponse.hourly) {
            sourcePointsForExtrapolation.push({
                lat: anchorInfo.lat, lon: anchorInfo.lon, 
                elevation: anchorTerrain.elevation, slope: anchorTerrain.slope, aspect: anchorTerrain.aspect, 
                weather: anchorWeatherRaw.apiResponse.hourly
            });
        }
    }

    if (sourcePointsForExtrapolation.length === 0) {
        // console.log(`Spiral Fill: Not enough valid source data for extrapolation for ${cellKey}`);
        return false;
    }

    const extrapolatedWeather = getExtrapolatedWeatherAtPoint(gridPointTerrain, sourcePointsForExtrapolation);
    if (!extrapolatedWeather || !extrapolatedWeather.time || extrapolatedWeather.time.length === 0) {
        // console.log(`Spiral Fill: Extrapolation failed for ${cellKey}`);
        return false;
    }

    const metrics = computeSnowMetrics(extrapolatedWeather, targetDate, 24, snowDisplayMode);
    const derivedCellData = {
        id: cellKey, lat, lon,
        elevation: gridPointTerrain.elevation, slope: gridPointTerrain.slope, aspect: gridPointTerrain.aspect,
        finalScore: metrics.finalScore, recentAccumCm: metrics.recentAccumCm,
        weatherData: extrapolatedWeather,
        sourceAnchorIds: closestSourceAnchorInfos.map(a => a.id),
        ts: Date.now()
    };
    
    try {
        await addDerivedData(derivedCellData);
        console.log(`Spiral Fill: Successfully processed and added data for ${cellKey}`);
    } catch (dbError) {
        console.error(`Spiral Fill: Error adding derived data for cell ${cellKey} to DB:`, dbError);
        return false; // Point not fully processed if DB write fails
    }

    // Add to map
    const featureData = {
        finalScore: derivedCellData.finalScore, recentAccumCm: derivedCellData.recentAccumCm,
        elevation: derivedCellData.elevation, slope: derivedCellData.slope, aspect: derivedCellData.aspect,
        lat, lon
    };
    if (derivedCellData.weatherData && derivedCellData.weatherData.time) {
        const hourIdx = getHourIndexForTargetDate(derivedCellData.weatherData.time, targetDate);
        if (hourIdx !== -1) {
            for (const varName in PREDEFINED_MIN_MAX) {
                if (derivedCellData.weatherData[varName] && derivedCellData.weatherData[varName][hourIdx] !== undefined) {
                    featureData[varName] = derivedCellData.weatherData[varName][hourIdx];
                }
            }
        }
    }
    const feature = new ol.Feature({ geometry: new ol.geom.Point(ol.proj.fromLonLat([lon, lat])), ...featureData });
    feature.setId(cellKey);
    olPointSource.addFeature(feature);
    return true; // Point was filled
}


let debounce = null;
map.on('moveend', ()=>{ clearTimeout(debounce); debounce = setTimeout(queryViewport, DEBOUNCE_MS); });

const highlightSrc = new ol.source.Vector();
map.addLayer(new ol.layer.Vector({ source: highlightSrc, style : new ol.style.Style({ stroke: new ol.style.Stroke({ color:'#ff4081', width:3 }), fill: null }) }));

map.on('singleclick',async evt=>{
  const info = document.getElementById('info-box');
  info.classList.remove('info-box-selecting'); // Remove selector class for map data display
  info.innerHTML = ''; // Clear previous content, including any day/time selectors
  info.style.display = 'block'; 
  highlightSrc.clear();

  // Ensure map clicks don't interfere with drawer toggle if drawer is open and info-box is used for selection
  // This is more of a UX consideration for later; for now, map click always shows point data.

  function addInfoItem(text) {
      if (!text) return; // Don't add empty items
      const itemDiv = document.createElement('div');
      itemDiv.textContent = text;
      info.appendChild(itemDiv);
  }

  addInfoItem('Fetching data...'); // Initial message

  const coordinate = evt.coordinate; const lonLat = ol.proj.toLonLat(coordinate);
  const clickLat = lonLat[1]; const clickLon = lonLat[0];
  
  // Clear "Fetching data..." and add actual data once available
  info.innerHTML = ''; // Clear "Fetching data..."

  addInfoItem(`Lat: ${clickLat.toFixed(4)}, Lon: ${clickLon.toFixed(4)}`);

  let terrainData = null;
  try { terrainData = await getElevation(clickLat, clickLon); } catch (error) { console.error("Error fetching terrain data:", error); }
  
  if (terrainData) {
      addInfoItem(`Elev: ${terrainData.elevation?.toFixed(0)}m, Slope: ${terrainData.slope?.toFixed(1)}°, Aspect: ${terrainData.aspect}`);
  } else {
      addInfoItem("Terrain data unavailable.");
  }

  let directWeatherData = null;
  try { directWeatherData = await getRawWeather(clickLat, clickLon); } catch (error) { console.error("Error fetching live weather data:", error); }

  if (directWeatherData && directWeatherData.hourly && directWeatherData.hourly.time && directWeatherData.hourly.time.length > 0) {
    let hourIdxForTargetDate = getHourIndexForTargetDate(directWeatherData.hourly.time, targetDate);
    if (hourIdxForTargetDate !== -1) {
        const weatherTime = new Date(directWeatherData.hourly.time[hourIdxForTargetDate]);
        addInfoItem(`Data for: ${weatherTime.toLocaleString()}`);

        const infoWeather = {};
        for (const varName in PREDEFINED_MIN_MAX) {
            if (directWeatherData.hourly[varName] && directWeatherData.hourly[varName][hourIdxForTargetDate] !== undefined) {
                infoWeather[varName] = directWeatherData.hourly[varName][hourIdxForTargetDate];
            }
        }

        let tempGroup = [];
        if (infoWeather.temperature_2m !== undefined) tempGroup.push(`Temp: ${infoWeather.temperature_2m.toFixed(1)}°C`);
        if (infoWeather.dewpoint_2m !== undefined) tempGroup.push(`Dewpt: ${infoWeather.dewpoint_2m.toFixed(1)}°C`);
        if (tempGroup.length > 0) addInfoItem(tempGroup.join(', '));
        
        let humidityPressureGroup = [];
        if (infoWeather.relativehumidity_2m !== undefined) humidityPressureGroup.push(`RH: ${infoWeather.relativehumidity_2m.toFixed(0)}%`);
        if (infoWeather.surface_pressure !== undefined) humidityPressureGroup.push(`Pres: ${infoWeather.surface_pressure.toFixed(0)}hPa`);
        if (humidityPressureGroup.length > 0) addInfoItem(humidityPressureGroup.join(', '));

        let windRadGroup = [];
        if (infoWeather.windspeed_10m !== undefined) windRadGroup.push(`Wind: ${infoWeather.windspeed_10m.toFixed(1)}km/h`);
        if (infoWeather.shortwave_radiation !== undefined) windRadGroup.push(`Rad: ${infoWeather.shortwave_radiation.toFixed(0)}W/m²`);
        if (windRadGroup.length > 0) addInfoItem(windRadGroup.join(', '));
        
        let snowGroup = [];
        if (infoWeather.snowfall !== undefined) snowGroup.push(`Snowfall (1h): ${infoWeather.snowfall.toFixed(1)}cm`);
        if (infoWeather.snow_depth !== undefined) snowGroup.push(`Snow Depth: ${infoWeather.snow_depth.toFixed(0)}cm`);
        if (snowGroup.length > 0) addInfoItem(snowGroup.join(', '));

        let tempDataHighAlt = [];
        if (infoWeather.temperature_850hPa !== undefined) tempDataHighAlt.push(`T850: ${infoWeather.temperature_850hPa.toFixed(1)}°C`);
        if (infoWeather.temperature_500hPa !== undefined) tempDataHighAlt.push(`T500: ${infoWeather.temperature_500hPa.toFixed(1)}°C`);
        if (tempDataHighAlt.length > 0) addInfoItem(tempDataHighAlt.join(', '));

        const accumHours = 24; 
        const snowMetrics = computeSnowMetrics(directWeatherData.hourly, targetDate, accumHours, snowDisplayMode);
        const modeLabel = snowDisplayMode === 'skiability' ? 'Skiability' : 'SQH Score';
        addInfoItem(`${modeLabel}: ${snowMetrics.finalScore?.toFixed(2)}, Recent Snow: ${snowMetrics.recentAccumCm?.toFixed(1)}cm`);

    } else { 
        addInfoItem("Could not match target time in weather data."); 
    }
  } else { 
      addInfoItem("Weather data unavailable for this point/time."); 
  }

  const circ = new ol.Feature( new ol.geom.Circle(coordinate, 60) ); 
  highlightSrc.addFeature(circ);
});

window.addEventListener('keydown',e=>{ 
    if (e.key==='Escape'){ 
        highlightSrc.clear();
        const infoBoxElement = document.getElementById('info-box');
        // When escape is pressed, we want to show the layer title and legend,
        // not the point-specific data.
        // queryViewport will typically be called by layer changes or map moves,
        // but here we explicitly update the title and legend for the current active layer.
        updateInfoBoxTitle(); 
        if (infoBoxElement) {
            renderInfoBoxLegend(infoBoxElement, activeLayer, currentDynamicMinMax, activeLayerType);
        }
    }
});

// Old updateTargetDateAndRefresh function is no longer needed as date/time are handled by new controls.

document.querySelectorAll('.layer-item').forEach(item => {
    // Ensure the new control buttons don't get treated as layer selectors
    if (item.id === 'day-control-button' || item.id === 'time-control-button' || item.id === 'clear-cache-grid-button') { // Corrected ID
        return;
    }
    item.addEventListener('click', () => {
        const clickedLayerName = item.dataset.layerName;
        const clickedLayerType = item.dataset.layerType;
        document.querySelectorAll('.layer-item').forEach(other => other.classList.remove('active'));
        item.classList.add('active');
        if (clickedLayerName === 'skiability') {
            snowDisplayMode = 'skiability'; activeLayer = 'skiability'; activeLayerType = 'snow_composite';
        } else if (clickedLayerName === 'sqh_raw') {
            snowDisplayMode = 'sqh_raw'; activeLayer = 'sqh_raw'; activeLayerType = 'snow_composite';
        } else {
            activeLayer = clickedLayerName; activeLayerType = clickedLayerType;
        }
        console.log(`Active layer: ${activeLayer}, Type: ${activeLayerType}, Mode: ${snowDisplayMode}`);
        
        queryViewport();
        updateInfoBoxTitle(); // Update title after layer change
    });
});

window.clearCache = async function(){ 
  try {
    await clearElevationStore();
    console.log("Elevation store cleared.");
    await clearRawWeatherStore();
    console.log("Raw weather store cleared.");
    await clearDerivedStore();
    console.log("Derived data store cleared.");
    // metricsCache = {}; // Removed
    // anchorPointsData = {}; // Removed
    
    // The individual clear functions (clearTerrainEvaluatorCache, clearWeatherAPICache)
    // are now effectively handled by the specific store clearing functions above
    // if they were solely interacting with those stores.
    // window.clearTerrainEvaluatorCache might still be needed if it does more than just clear its store.
    // Let's assume for now that clearElevationStore covers it.
    // Same for clearWeatherAPICache and clearRawWeatherStore.

  } catch (error) {
    console.error("Error during cache clearing process:", error);
  }
  pointSource.clear(); 
  queryViewport(); 
  console.log('All DB stores cleared and viewport refreshed.');
};

async function initializeApp() { // Made initializeApp async
    try {
        await openDB(); // Call openDB at the beginning
        console.log("Database initialized successfully.");
    } catch (error) {
        console.error("Failed to initialize database. Application cannot start.", error);
        // Optionally, display a message to the user in the UI
        const mapElement = document.getElementById('map');
        if (mapElement) {
            mapElement.innerHTML = '<p style="text-align:center;padding:20px;">Could not initialize the database. Please try refreshing the page or clearing your browser cache.</p>';
        }
        return; // Prevent further execution
    }

    // targetDate is already initialized globally to tomorrow at 12 PM.
    updateDayTimeButtonText(); // Update button text based on initial targetDate
    updateInfoBoxTitle(); // Set initial info-box title

    console.log("Initial targetDate:", targetDate.toLocaleString());

    // Setup Clear Cache button
    if (clearCacheGridButton) {
        clearCacheGridButton.addEventListener('click', clearCache);
    } else {
        console.warn("Clear Cache grid button not found.");
    }

    // Setup Play/Pause button
    const playPauseButton = document.getElementById('play-pause-button');
    if (playPauseButton) {
        playPauseButton.textContent = isPaused ? 'Play' : 'Pause';
        playPauseButton.style.backgroundColor = isPaused ? '#4CAF50' : '#f44336'; // Green for Play, Red for Pause

        playPauseButton.addEventListener('click', () => {
            isPaused = !isPaused;
            playPauseButton.textContent = isPaused ? 'Play' : 'Pause';
            playPauseButton.style.backgroundColor = isPaused ? '#4CAF50' : '#f44336';
            console.log(`Play/Pause toggled. isPaused: ${isPaused}`);
            if (!isPaused) {
                queryViewport(); // Re-run queryViewport when resuming
            }
        });
    } else {
        console.warn("Play/Pause button not found.");
    }

    // Setup Info Box Close Button - REMOVED
    // if (infoBoxCloseButton) {
    //     infoBoxCloseButton.addEventListener('click', () => {
    //         const info = document.getElementById('info-box');
    //         if (info) {
    //             info.innerHTML = '';
    //             info.style.display = 'none';
    //         }
    //         highlightSrc.clear(); // Also clear any map highlight

    //         // Reset active layer to skiability
    //         activeLayer = 'skiability';
    //         activeLayerType = 'snow_composite';
    //         snowDisplayMode = 'skiability';

    //         // Update visual active state on grid items
    //         document.querySelectorAll('.layer-item.active').forEach(item => item.classList.remove('active'));
    //         const skiabilityLayerItem = document.querySelector('.layer-item[data-layer-name="skiability"]');
    //         if (skiabilityLayerItem) {
    //             skiabilityLayerItem.classList.add('active');
    //         } else {
    //             console.warn("Skiability layer item not found for visual reset.");
    //         }
            
    //         console.log(`Info box closed. Active layer reset to: ${activeLayer}`);
    //         queryViewport(); // Refresh the map display
    //         if (infoBoxCloseButton) { // Hide the button itself
    //             infoBoxCloseButton.style.display = 'none';
    //         }
    //     });
    // } else {
    //     console.warn("Info box close button not found.");
    // }

    // Setup default active layer
    const defaultLayerElement = document.querySelector('.layer-item[data-layer-name="skiability"]');
    if (defaultLayerElement) {
        defaultLayerElement.classList.add('active');
        activeLayer = 'skiability'; activeLayerType = 'snow_composite'; snowDisplayMode = 'skiability';
    } else { 
        // Fallback if the element isn't found, though it should be there.
        activeLayer = 'skiability'; activeLayerType = 'snow_composite'; snowDisplayMode = 'skiability';
        console.warn("Default layer item 'skiability' not found by selector.");
    }
    // Ensure initial state of infoBoxCloseButton is hidden as skiability is default - REMOVED
    // if (infoBoxCloseButton) {
    //     infoBoxCloseButton.style.display = 'none';
    // }
    console.log(`Initial active layer: ${activeLayer}, Type: ${activeLayerType}, Mode: ${snowDisplayMode}`);
    
    // Attach the new toggleDrawer function
    const drawerBar = document.getElementById('drawer-bar');
    if (drawerBar) {
        drawerBar.addEventListener('click', toggleDrawer);
    } else {
        console.warn("Drawer bar element not found for attaching toggleDrawer.");
    }

    queryViewport(); // Initial data load
}

// Moved and enhanced from index.html
function toggleDrawer() {
    const drawerContainer = document.getElementById('drawer-container');
    const info = document.getElementById('info-box');

    const drawerIsCurrentlyOpen = drawerContainer.classList.contains('open');

    if (drawerIsCurrentlyOpen) { 
        if (info && info.classList.contains('info-box-selecting')) {
            updateInfoBoxTitle(); 
            console.log("Drawer closing with selector active - reverted info-box to title.");
        }
    }
    
    drawerContainer.classList.toggle('open');
}

// Event listeners for Day/Time controls (simplified for brevity, assuming they exist)
const setupControls = () => {
    if (dayControlButton) {
        dayControlButton.addEventListener('click', () => {
            const info = document.getElementById('info-box');
            info.innerHTML = ''; 
            info.classList.add('info-box-selecting'); 
            info.style.display = 'block';
            highlightSrc.clear(); 

            const ul = document.createElement('ul');
            ul.id = 'info-list';
            ul.classList.add('info-list-selecting');

            const today = new Date();
            for (let i = 0; i < 5; i++) {
                const dayOption = new Date(today);
                dayOption.setDate(today.getDate() + i);
                dayOption.setHours(targetDate.getHours(), targetDate.getMinutes(), targetDate.getSeconds(), targetDate.getMilliseconds());

                const li = document.createElement('li');
                li.classList.add('list-item'); 
                li.textContent = getDisplayDay(dayOption);
                
                if (dayOption.getFullYear() === targetDate.getFullYear() &&
                    dayOption.getMonth() === targetDate.getMonth() &&
                    dayOption.getDate() === targetDate.getDate()) {
                    li.classList.add('active');
                }

                li.addEventListener('click', () => {
                    targetDate.setFullYear(dayOption.getFullYear(), dayOption.getMonth(), dayOption.getDate());
                    updateDayTimeButtonText();
                    queryViewport();
                    updateInfoBoxTitle(); 
                });
                ul.appendChild(li);
            }
            info.appendChild(ul);
        });
    } else {
        console.warn("Day control button not found.");
    }

    if (timeControlButton) {
        timeControlButton.addEventListener('click', () => {
            const info = document.getElementById('info-box');
            info.innerHTML = ''; 
            info.classList.add('info-box-selecting');
            info.style.display = 'block';
            highlightSrc.clear(); 

            let selectedHour12 = null; 
            let selectedAmPm = null;   

            const ul = document.createElement('ul');
            ul.id = 'info-list';
            ul.classList.add('info-list-selecting');

            for (let i = 1; i <= 12; i++) {
                const li = document.createElement('li');
                li.classList.add('list-item'); 
                li.textContent = i.toString();

                let currentTargetHour12 = targetDate.getHours() % 12;
                currentTargetHour12 = currentTargetHour12 ? currentTargetHour12 : 12;
                if (i === currentTargetHour12) {
                    li.classList.add('active');
                    selectedHour12 = i; 
                }

                li.addEventListener('click', () => {
                    ul.querySelectorAll('li').forEach(el => { 
                        if (parseInt(el.textContent) >=1 && parseInt(el.textContent) <=12) el.classList.remove('active');
                    });
                    li.classList.add('active');
                    selectedHour12 = i;
                    if (selectedAmPm) { 
                        commitTimeChange();
                    }
                });
                ul.appendChild(li);
            }
            
            ['AM', 'PM'].forEach(period => {
                const li = document.createElement('li');
                li.classList.add('list-item');
                li.textContent = period;

                const currentTargetAmPm = targetDate.getHours() >= 12 ? 'PM' : 'AM';
                if (period === currentTargetAmPm) {
                    li.classList.add('active');
                    selectedAmPm = period;
                }

                li.addEventListener('click', () => {
                     ul.querySelectorAll('li').forEach(el => { 
                        if (el.textContent === "AM" || el.textContent === "PM") el.classList.remove('active');
                    });
                    li.classList.add('active');
                    selectedAmPm = period;
                    if (selectedHour12 !== null) { 
                        commitTimeChange();
                    }
                });
                ul.appendChild(li);
            });
            
            info.appendChild(ul);

            function commitTimeChange() {
                if (selectedHour12 === null || !selectedAmPm) return;

                let newHour24 = selectedHour12;
                if (selectedAmPm === 'PM' && selectedHour12 < 12) newHour24 += 12;
                else if (selectedAmPm === 'AM' && selectedHour12 === 12) newHour24 = 0;
                
                targetDate.setHours(newHour24, 0, 0, 0); 
                updateDayTimeButtonText();
                queryViewport();
                updateInfoBoxTitle(); 
            }
        });
    } else {
        console.warn("Time control button not found.");
    }
};

setupControls(); // Call to attach listeners

initializeApp();