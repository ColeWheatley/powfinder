// frontend.js - Frontend application logic
// Uses unified grid system with enhanced physics and triple controls

import {
  openDB as openGridDB,
  getOrCreateGridCell,
  getCellsInBounds,
  updateCellTerrain,
  updateCellWeather,
  updateCellDerived,
  countCellsWithData,
  clearAllCells
} from './gridService.js';

import { getRawWeather } from './weatherAPI.js';
import { getElevation } from './terrain_evaluator.js';
import { extrapolateWeather } from './weatherExtrapolationEnhanced.js';
import { calculateSnowQuality, calculateSkiability } from './snowQualityModel.js';
import { setActive as setShadowActive, calculateShadow } from './shadowCalculator.js';

import { 
  COLOR_LIGHT_BLUE, COLOR_INDIGO, COLOR_HOT_PINK,
  THERMAL_SCALE, SNOW_SCALE, HUMIDITY_SCALE, WIND_SPEED_SCALE, PRESSURE_SCALE,
  ASPECT_COLORS, ELEVATION_SCALE, SLOPE_GRADIENT_SCALE,
  interpolateColorLinear
} from './colorManager.js';

// Constants
const GRID_LEVELS = [
  { level: 0, step: 0.01 },    // ~1.1km grid - coarsest
  { level: 1, step: 0.005 },   // ~550m grid
  { level: 2, step: 0.0025 },  // ~275m grid
  { level: 3, step: 0.00125 }, // ~140m grid
  { level: 4, step: 0.0006 },  // ~70m grid - finest
];
const PRIORITY_ELEVATION = 2300; // meters - prioritize above this
const API_STAGGER_MS = 10; // ms between API calls
const COMPUTE_BATCH_SIZE = 20; // cells per computation batch
const MAX_EXTRAPOLATION_DISTANCE = 2000; // meters
const TARGET_GRID_POINTS = 500; // target number of points to display

// Control states
const controls = {
  apiActive: false,
  computeActive: false,
  shadowActive: false
};

// State variables
let map;
let pointSource;
let pointLayer;
let peaks = [];
let targetDate = new Date();
targetDate.setDate(targetDate.getDate() + 1);
targetDate.setHours(12, 0, 0, 0);

let activeLayer = 'skiability';
let activeLayerType = 'snow_composite';
let processingQueue = [];
let isProcessing = false;

// Initialize database
async function initializeDatabase() {
  console.log('Initializing grid database...');
  await openGridDB();
  console.log('Grid database ready');
}

// Load peaks data
async function loadPeaks() {
  try {
    const response = await fetch('./refined_tirol_peaks.json');
    peaks = await response.json();
    console.log(`Loaded ${peaks.length} peaks`);
  } catch (error) {
    console.error('Failed to load peaks:', error);
    peaks = [];
  }
}

// Initialize map
function initializeMap() {
  pointSource = new ol.source.Vector();
  
  pointLayer = new ol.layer.Vector({
    source: pointSource,
    style: createStyleFunction()
  });
  
  map = new ol.Map({
    target: 'map',
    layers: [
      new ol.layer.Tile({ source: new ol.source.OSM() }),
      pointLayer
    ],
    view: new ol.View({
      center: ol.proj.fromLonLat([11.1581, 46.9731]), // Zuckerhütl
      zoom: 15
    })
  });
  
  // Map event handlers
  map.on('moveend', debounce(onMapMoveEnd, 300));
  map.on('singleclick', onMapClick);
}

// Style function for points
function createStyleFunction() {
  return (feature) => {
    const cell = feature.get('cell');
    if (!cell) return null;
    
    let color, opacity = 0.8;
    
    if (activeLayer === 'skiability' || activeLayer === 'sqh_raw') {
      const quality = cell.derived?.snowQuality;
      const height = cell.derived?.snowHeight;
      
      if (quality == null || height == null || height < 1) return null;
      
      // Color based on quality
      if (quality < 33) color = COLOR_LIGHT_BLUE;
      else if (quality < 66) color = COLOR_INDIGO;
      else color = COLOR_HOT_PINK;
      
      // Opacity based on snow height
      opacity = Math.max(0.2, Math.min(height / 50, 1));
      
    } else if (activeLayerType === 'terrain') {
      const value = cell.terrain?.[activeLayer];
      if (value == null) return null;
      
      switch (activeLayer) {
        case 'elevation':
          color = interpolateColorLinear(value, 500, 4000, ELEVATION_SCALE);
          break;
        case 'slope':
          color = interpolateColorLinear(value, 0, 90, SLOPE_GRADIENT_SCALE);
          break;
        case 'aspect':
          color = ASPECT_COLORS[value] || ASPECT_COLORS.default;
          break;
        case 'shadow':
          const shadowValue = 1 - (value || 0); // Invert so 0=no shadow, 1=full shadow
          color = [shadowValue * 255, shadowValue * 255, shadowValue * 255];
          break;
      }
      
    } else if (activeLayerType === 'weather') {
      const weatherData = cell.rawWeather?.data || cell.extrapolatedWeather?.data;
      if (!weatherData) return null;
      
      const value = getWeatherValueAtTime(weatherData, activeLayer, targetDate);
      if (value == null) return null;
      
      color = getWeatherColor(activeLayer, value);
    }
    
    if (!color) return null;
    
    // Visual distinction for data source
    let strokeWidth = 0;
    let strokeColor = null;
    
    if (cell.isPeak) {
      strokeWidth = 2;
      strokeColor = 'rgba(255, 255, 255, 0.8)';
    } else if (cell.isAnchor) {
      strokeWidth = 1;
      strokeColor = 'rgba(255, 255, 255, 0.5)';
    } else if (cell.extrapolatedWeather?.data && !cell.rawWeather?.data) {
      strokeWidth = 1;
      strokeColor = 'rgba(0, 0, 0, 0.3)';
    }
    
    const [r, g, b] = color;
    return new ol.style.Style({
      image: new ol.style.Circle({
        radius: 6,
        fill: new ol.style.Fill({ 
          color: `rgba(${r},${g},${b},${opacity})` 
        }),
        stroke: strokeWidth > 0 ? new ol.style.Stroke({
          color: strokeColor,
          width: strokeWidth
        }) : null
      })
    });
  };
}

// Get weather value at specific time
function getWeatherValueAtTime(weatherData, variable, date) {
  if (!weatherData.time || !weatherData[variable]) return null;
  
  const targetMs = date.getTime();
  let idx = -1;
  
  for (let i = 0; i < weatherData.time.length; i++) {
    if (new Date(weatherData.time[i]).getTime() <= targetMs) {
      idx = i;
    } else {
      break;
    }
  }
  
  return idx >= 0 ? weatherData[variable][idx] : null;
}

// Get color for weather value
function getWeatherColor(variable, value) {
  const scales = {
    temperature_2m: { min: -20, max: 30, scale: THERMAL_SCALE },
    temperature_850hPa: { min: -25, max: 20, scale: THERMAL_SCALE },
    temperature_500hPa: { min: -40, max: -5, scale: THERMAL_SCALE },
    dewpoint_2m: { min: -25, max: 25, scale: THERMAL_SCALE },
    relativehumidity_2m: { min: 0, max: 100, scale: HUMIDITY_SCALE },
    windspeed_10m: { min: 0, max: 100, scale: WIND_SPEED_SCALE },
    shortwave_radiation: { min: 0, max: 1000, scale: THERMAL_SCALE },
    surface_pressure: { min: 950, max: 1050, scale: PRESSURE_SCALE },
    snowfall: { min: 0, max: 20, scale: SNOW_SCALE },
    snow_depth: { min: 0, max: 300, scale: SNOW_SCALE }
  };
  
  const scale = scales[variable];
  if (!scale) return null;
  
  return interpolateColorLinear(value, scale.min, scale.max, scale.scale);
}

// Map move handler
async function onMapMoveEnd() {
  const view = map.getView();
  const extent = view.calculateExtent(map.getSize());
  const [minLon, minLat, maxLon, maxLat] = ol.proj.transformExtent(extent, 'EPSG:3857', 'EPSG:4326');
  
  console.log('Map moved, processing viewport...');
  
  // Queue viewport processing
  queueViewportProcessing(minLat, maxLat, minLon, maxLon);
}

// Queue viewport for processing
async function queueViewportProcessing(minLat, maxLat, minLon, maxLon) {
  const bounds = { minLat, maxLat, minLon, maxLon };
  const centerLat = (minLat + maxLat) / 2;
  const centerLon = (minLon + maxLon) / 2;
  
  // Build progressive queue
  const queue = [];
  
  // First: Add peaks (highest priority)
  const visiblePeaks = peaks.filter(p => {
    const [lon, lat] = p.coordinates;
    return lat >= minLat && lat <= maxLat && lon >= minLon && lon <= maxLon;
  });
  
  for (const peak of visiblePeaks) {
    const [lon, lat] = peak.coordinates;
    queue.push({
      lat, lon,
      priority: 0,
      type: 'peak',
      peakName: peak.name,
      isPeak: true
    });
    
    // Add anchor points
    const offset = 0.00225; // ~250m
    queue.push({ lat: lat + offset, lon, priority: 0, type: 'anchor', peakName: peak.name, isAnchor: true });
    queue.push({ lat: lat - offset, lon, priority: 0, type: 'anchor', peakName: peak.name, isAnchor: true });
    queue.push({ lat, lon: lon + offset, priority: 0, type: 'anchor', peakName: peak.name, isAnchor: true });
    queue.push({ lat, lon: lon - offset, priority: 0, type: 'anchor', peakName: peak.name, isAnchor: true });
  }
  
  // Then: Add progressive grid levels
  const processedKeys = new Set();
  
  for (const gridLevel of GRID_LEVELS) {
    const levelCells = [];
    
    // Generate grid at this level
    for (let lat = minLat; lat <= maxLat; lat += gridLevel.step) {
      for (let lon = minLon; lon <= maxLon; lon += gridLevel.step) {
        // Snap to grid
        const gridLat = Math.round(lat / gridLevel.step) * gridLevel.step;
        const gridLon = Math.round(lon / gridLevel.step) * gridLevel.step;
        const key = `${gridLat.toFixed(5)},${gridLon.toFixed(5)}`;
        
        // Skip if already processed at coarser level
        if (processedKeys.has(key)) continue;
        processedKeys.add(key);
        
        // Calculate distance from center for better visual loading
        const distFromCenter = Math.sqrt(
          Math.pow(gridLat - centerLat, 2) + Math.pow(gridLon - centerLon, 2)
        );
        
        levelCells.push({
          lat: gridLat,
          lon: gridLon,
          priority: gridLevel.level + 1,
          type: 'grid',
          level: gridLevel.level,
          distFromCenter
        });
      }
    }
    
    // Sort by distance from center within each level
    levelCells.sort((a, b) => a.distFromCenter - b.distFromCenter);
    queue.push(...levelCells);
  }
  
  // Sort entire queue by priority (peaks first, then coarse to fine grid)
  queue.sort((a, b) => {
    if (a.priority !== b.priority) return a.priority - b.priority;
    return (a.distFromCenter || 0) - (b.distFromCenter || 0);
  });
  
  console.log(`Generated progressive queue: ${queue.filter(c => c.type === 'peak').length} peaks, ${queue.filter(c => c.type === 'anchor').length} anchors, ${queue.filter(c => c.type === 'grid').length} grid cells`);
  
  // Replace processing queue
  processingQueue = queue;
  
  // Start processing if not already running
  if (!isProcessing) {
    processQueue();
  }
}

// Process queue
async function processQueue() {
  if (isProcessing) return;
  isProcessing = true;
  
  // Update UI to show processing
  updateProcessingIndicator(true);
  
  while (processingQueue.length > 0) {
    const batch = processingQueue.splice(0, COMPUTE_BATCH_SIZE);
    
    // Separate batch by type
    const peakBatch = batch.filter(c => c.type === 'peak' || c.type === 'anchor');
    const gridBatch = batch.filter(c => c.type === 'grid');
    
    // Process peaks and anchors first (they get weather data)
    if (peakBatch.length > 0 && controls.apiActive) {
      for (const cell of peakBatch) {
        await processPeakPoint(cell.lat, cell.lon, cell.peakName, cell.isPeak, cell.isAnchor);
        // Stagger API calls
        if (cell.type === 'peak' || cell.type === 'anchor') {
          await new Promise(resolve => setTimeout(resolve, API_STAGGER_MS));
        }
      }
    }
    
    // Process grid cells (terrain first, then extrapolation)
    if (gridBatch.length > 0) {
      // Get terrain for grid cells
      if (controls.computeActive) {
        await processBatchTerrain(gridBatch);
      }
      
      // Extrapolate weather for grid cells
      if (controls.computeActive) {
        await processExtrapolation(gridBatch);
      }
      
      // Calculate shadows if enabled
      if (controls.shadowActive) {
        await processShadows(gridBatch);
      }
    }
    
    // Update display after each batch
    await updateDisplay();
    
    // Small delay to prevent blocking
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  
  isProcessing = false;
  updateProcessingIndicator(false);
  console.log('Processing complete');
}

// Update processing indicator
function updateProcessingIndicator(active) {
  // Add pulsing effect to active controls when processing
  const apiBtn = document.getElementById('api-control');
  const computeBtn = document.getElementById('compute-control');
  const shadowBtn = document.getElementById('shadow-control');
  
  if (active) {
    if (controls.apiActive && apiBtn) apiBtn.style.animation = 'pulse 1s infinite';
    if (controls.computeActive && computeBtn) computeBtn.style.animation = 'pulse 1s infinite';
    if (controls.shadowActive && shadowBtn) shadowBtn.style.animation = 'pulse 1s infinite';
  } else {
    if (apiBtn) apiBtn.style.animation = '';
    if (computeBtn) computeBtn.style.animation = '';
    if (shadowBtn) shadowBtn.style.animation = '';
  }
}

// Process terrain for batch
async function processBatchTerrain(batch) {
  // Process terrain one at a time to reduce load
  for (const { lat, lon } of batch) {
    const cell = await getOrCreateGridCell(lat, lon);
    
    if (!cell.terrain.elevation) {
      const terrainData = await getElevation(lat, lon);
      if (terrainData) {
        await updateCellTerrain(cell.id, terrainData);
      }
      // Small delay between terrain fetches to prevent overload
      await new Promise(resolve => setTimeout(resolve, 20));
    }
  }
}


// Process a peak point
async function processPeakPoint(lat, lon, peakName, isPeak, isAnchor) {
  const cell = await getOrCreateGridCell(lat, lon);
  
  // Update metadata
  cell.isPeak = isPeak;
  cell.isAnchor = isAnchor;
  
  // Get weather data if needed
  if (!cell.rawWeather.data || Date.now() - cell.rawWeather.fetchedAt > 3600000) {
    const weatherData = await getRawWeather(lat, lon, { peakName, isAdjacent: isAnchor });
    
    if (weatherData) {
      await updateCellWeather(cell.id, weatherData, 'api');
      
      // Calculate derived data
      if (controls.computeActive) {
        await calculateDerivedData(cell);
      }
    }
  }
}

// Process extrapolation
async function processExtrapolation(batch) {
  // Get cells with raw weather data nearby
  const bounds = {
    minLat: Math.min(...batch.map(c => c.lat)) - 0.02,
    maxLat: Math.max(...batch.map(c => c.lat)) + 0.02,
    minLon: Math.min(...batch.map(c => c.lon)) - 0.02,
    maxLon: Math.max(...batch.map(c => c.lon)) + 0.02
  };
  
  const allCells = await getCellsInBounds(bounds.minLat, bounds.maxLat, bounds.minLon, bounds.maxLon);
  const sourceCells = allCells.filter(c => c.rawWeather?.data);
  
  if (sourceCells.length === 0) return;
  
  // Process each cell in batch
  for (const { lat, lon } of batch) {
    const cell = await getOrCreateGridCell(lat, lon);
    
    // Skip if already has weather data
    if (cell.rawWeather.data || cell.extrapolatedWeather.data) continue;
    
    // Skip if terrain not available
    if (!cell.terrain.elevation) continue;
    
    // Find nearest source cells
    const nearestSources = findNearestSources(cell, sourceCells, 5, MAX_EXTRAPOLATION_DISTANCE);
    
    if (nearestSources.length >= 3) {
      const extrapolated = extrapolateWeather(cell, nearestSources, targetDate);
      
      if (extrapolated) {
        await updateCellWeather(cell.id, {
          data: extrapolated,
          sourcePoints: nearestSources.map(s => s.id)
        }, 'extrapolated');
        
        // Calculate derived data
        await calculateDerivedData(cell);
      }
    }
  }
}

// Find nearest source cells
function findNearestSources(targetCell, sourceCells, maxCount, maxDistance) {
  const distances = sourceCells.map(source => ({
    cell: source,
    distance: haversineDistance(targetCell.lat, targetCell.lon, source.lat, source.lon)
  }));
  
  return distances
    .filter(d => d.distance <= maxDistance)
    .sort((a, b) => a.distance - b.distance)
    .slice(0, maxCount)
    .map(d => d.cell);
}

// Calculate derived data
async function calculateDerivedData(cell) {
  const weatherData = cell.rawWeather?.data || cell.extrapolatedWeather?.data;
  if (!weatherData) return;
  
  const result = calculateSnowQuality(weatherData.hourly || weatherData, cell.terrain, targetDate);
  const skiability = calculateSkiability(result.quality, result.height, result.rawFactors);
  
  await updateCellDerived(cell.id, {
    snowQuality: result.quality,
    snowHeight: result.height,
    skiability: skiability
  }, targetDate);
}

// Process shadows
async function processShadows(batch) {
  const cellsWithTerrain = [];
  
  for (const { lat, lon } of batch) {
    const cell = await getOrCreateGridCell(lat, lon);
    if (cell.terrain.elevation) {
      cellsWithTerrain.push(cell);
    }
  }
  
  if (cellsWithTerrain.length === 0) return;
  
  // Calculate shadows for all cells
  const shadowPromises = cellsWithTerrain.map(async (cell) => {
    const shadow = await calculateShadow(cell.lat, cell.lon, cell.terrain.elevation, targetDate);
    if (shadow !== null) {
      cell.terrain.shadow = shadow;
      await updateCellTerrain(cell.id, { shadow });
    }
  });
  
  await Promise.all(shadowPromises);
}

// Update display
async function updateDisplay() {
  const view = map.getView();
  const extent = view.calculateExtent(map.getSize());
  const [minLon, minLat, maxLon, maxLat] = ol.proj.transformExtent(extent, 'EPSG:3857', 'EPSG:4326');
  
  // Get cells in viewport
  const cells = await getCellsInBounds(minLat, maxLat, minLon, maxLon);
  
  // Clear existing features
  pointSource.clear();
  
  // Filter cells based on active layer and data availability
  let displayCells = cells;
  
  if (activeLayerType === 'terrain') {
    // For terrain, show cells that have elevation data
    displayCells = cells.filter(cell => cell.terrain?.elevation !== null);
  } else if (activeLayerType === 'weather') {
    // For weather, show cells that have any weather data
    displayCells = cells.filter(cell => cell.rawWeather?.data || cell.extrapolatedWeather?.data);
  } else {
    // For snow composite, show cells that have derived data
    displayCells = cells.filter(cell => cell.derived?.snowQuality !== null);
  }
  
  // Create features for all display cells
  const features = displayCells.map(cell => {
    return new ol.Feature({
      geometry: new ol.geom.Point(ol.proj.fromLonLat([cell.lon, cell.lat])),
      cell: cell
    });
  });
  
  pointSource.addFeatures(features);
  
  // Update status with more detail
  const peakCount = cells.filter(c => c.isPeak).length;
  const anchorCount = cells.filter(c => c.isAnchor).length;
  const terrainCount = cells.filter(c => c.terrain?.elevation !== null).length;
  const weatherCount = cells.filter(c => c.rawWeather?.data || c.extrapolatedWeather?.data).length;
  
  console.log(`Display updated - Total: ${cells.length} | Peaks: ${peakCount} | Anchors: ${anchorCount} | Terrain: ${terrainCount} | Weather: ${weatherCount} | Showing: ${displayCells.length}`);
}


// Map click handler
async function onMapClick(evt) {
  const coordinate = evt.coordinate;
  const [lon, lat] = ol.proj.toLonLat(coordinate);
  
  console.log(`Clicked at: ${lat.toFixed(5)}, ${lon.toFixed(5)}`);
  
  // Get or create cell at click location
  const cell = await getOrCreateGridCell(lat, lon);
  
  // Fetch fresh data if API is active
  if (controls.apiActive) {
    const weatherData = await getRawWeather(lat, lon, { manual: true });
    if (weatherData) {
      await updateCellWeather(cell.id, weatherData, 'api');
      
      if (controls.computeActive) {
        await calculateDerivedData(cell);
      }
    }
  }
  
  // Update display
  await updateDisplay();
}

// Toggle drawer
function toggleDrawer() {
  const drawerContainer = document.getElementById('drawer-container');
  const info = document.getElementById('info-box');
  
  const drawerIsCurrentlyOpen = drawerContainer.classList.contains('open');
  
  if (drawerIsCurrentlyOpen) {
    if (info && info.classList.contains('info-box-selecting')) {
      updateInfoBoxTitle();
    }
  }
  
  drawerContainer.classList.toggle('open');
}

// Update info box title
function updateInfoBoxTitle() {
  const info = document.getElementById('info-box');
  if (!info) return;
  
  info.classList.remove('info-box-selecting');
  
  let title = getShortLayerName(activeLayer);
  const layersWithoutDateTime = ['elevation', 'slope', 'aspect', 'shadow'];
  if (!layersWithoutDateTime.includes(activeLayer)) {
    title += ` - ${getDisplayDay(targetDate)} ${formatTimeForButton(targetDate)}`;
  }
  
  let titleDiv = info.querySelector('.info-box-title-text');
  if (!titleDiv) {
    info.innerHTML = '';
    titleDiv = document.createElement('div');
    titleDiv.className = 'info-box-title-text';
    info.appendChild(titleDiv);
  }
  titleDiv.textContent = title;
  info.style.display = 'block';
}

// Get short layer name
function getShortLayerName(layerId) {
  const names = {
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
    'aspect': 'Aspect',
    'shadow': 'Shadow'
  };
  return names[layerId] || layerId;
}

// Get display day
function getDisplayDay(date) {
  const today = new Date();
  const tomorrow = new Date(today);
  tomorrow.setDate(today.getDate() + 1);
  
  const todayMidnight = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  const tomorrowMidnight = new Date(tomorrow.getFullYear(), tomorrow.getMonth(), tomorrow.getDate());
  const dateMidnight = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  
  if (dateMidnight.getTime() === todayMidnight.getTime()) {
    return "Today";
  } else if (dateMidnight.getTime() === tomorrowMidnight.getTime()) {
    return "Tomorrow";
  } else {
    const days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    return days[date.getDay()];
  }
}

// Format time for button
function formatTimeForButton(date) {
  let hours = date.getHours();
  const ampm = hours >= 12 ? 'PM' : 'AM';
  hours = hours % 12;
  hours = hours ? hours : 12;
  return `${hours} ${ampm}`;
}

// Control handlers
function setupControls() {
  // Drawer toggle
  const drawerBar = document.getElementById('drawer-bar');
  if (drawerBar) {
    drawerBar.addEventListener('click', toggleDrawer);
  }
  
  // API control
  document.getElementById('api-control').addEventListener('click', (e) => {
    controls.apiActive = !controls.apiActive;
    e.target.classList.toggle('active', controls.apiActive);
    console.log(`API requests: ${controls.apiActive ? 'ACTIVE' : 'PAUSED'}`);
  });
  
  // Compute control
  document.getElementById('compute-control').addEventListener('click', (e) => {
    controls.computeActive = !controls.computeActive;
    e.target.classList.toggle('active', controls.computeActive);
    console.log(`Computations: ${controls.computeActive ? 'ACTIVE' : 'PAUSED'}`);
  });
  
  // Shadow control
  document.getElementById('shadow-control').addEventListener('click', (e) => {
    controls.shadowActive = !controls.shadowActive;
    e.target.classList.toggle('active', controls.shadowActive);
    setShadowActive(controls.shadowActive);
    console.log(`Shadow calculations: ${controls.shadowActive ? 'ACTIVE' : 'PAUSED'}`);
  });
  
  // Layer selection
  document.querySelectorAll('.layer-item[data-layer-name]').forEach(item => {
    item.addEventListener('click', () => {
      activeLayer = item.dataset.layerName;
      activeLayerType = item.dataset.layerType;
      
      // Update UI
      document.querySelectorAll('.layer-item').forEach(el => el.classList.remove('active'));
      item.classList.add('active');
      
      // Update info box
      updateInfoBoxTitle();
      
      // Refresh display
      updateDisplay();
    });
  });
  
  // Set initial active layer
  const skiabilityItem = document.querySelector('.layer-item[data-layer-name="skiability"]');
  if (skiabilityItem) {
    skiabilityItem.classList.add('active');
  }
  
  // Time controls
  document.getElementById('day-control-button').addEventListener('click', () => {
    // Cycle through days
    targetDate.setDate(targetDate.getDate() + 1);
    if (targetDate > new Date(Date.now() + 14 * 24 * 60 * 60 * 1000)) {
      targetDate = new Date();
    }
    updateTimeDisplay();
    updateDisplay();
  });
  
  document.getElementById('time-control-button').addEventListener('click', () => {
    // Cycle through hours
    targetDate.setHours((targetDate.getHours() + 3) % 24);
    updateTimeDisplay();
    updateDisplay();
  });
  
  // Clear cache
  document.getElementById('clear-cache-grid-button').addEventListener('click', async () => {
    if (confirm('Clear all cached data?')) {
      // Clear all stores
      await clearAllCells();
      pointSource.clear();
      console.log('Cache cleared');
    }
  });
}

// Update time display
function updateTimeDisplay() {
  const dayButton = document.getElementById('day-control-button');
  const timeButton = document.getElementById('time-control-button');
  
  if (dayButton) {
    dayButton.textContent = getDisplayDay(targetDate);
  }
  
  if (timeButton) {
    timeButton.textContent = formatTimeForButton(targetDate);
  }
}

// Utility functions
function haversineDistance(lat1, lon1, lat2, lon2) {
  const R = 6371e3;
  const φ1 = lat1 * Math.PI / 180;
  const φ2 = lat2 * Math.PI / 180;
  const Δφ = (lat2 - lat1) * Math.PI / 180;
  const Δλ = (lon2 - lon1) * Math.PI / 180;
  
  const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
            Math.cos(φ1) * Math.cos(φ2) *
            Math.sin(Δλ/2) * Math.sin(Δλ/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  
  return R * c;
}

function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Initialize application
async function initialize() {
  try {
    await initializeDatabase();
    await loadPeaks();
    initializeMap();
    setupControls();
    updateTimeDisplay();
    updateInfoBoxTitle();
    
    console.log('PowFinder initialized - use control buttons to start processing');
  } catch (error) {
    console.error('Failed to initialize:', error);
  }
}

// Start the application
initialize();