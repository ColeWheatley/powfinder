import 'https://cdn.jsdelivr.net/npm/ol@v7.4.0/dist/ol.js';

// Resource base path - single source of truth for all asset URLs
const RES = './web-resources/';

// Create base map layers
const osmLayer = new ol.layer.Tile({ 
  source: new ol.source.OSM({
    cacheSize: 16384  // 4x default cache size for smooth zooming
  })
});

const lightBaseLayer = new ol.layer.Tile({
  source: new ol.source.XYZ({
    url: 'https://{a-d}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png'
  })
});

const hillshadeLayer = new ol.layer.Tile({
  source: new ol.source.XYZ({
    url: `${RES}images/terrain/hillshade.png`
  }),
  opacity: 0.3,
  visible: false  // Initially hidden
});

const labelsLayer = new ol.layer.Tile({
  source: new ol.source.Stamen({
    layer: 'toner-labels'  // Just black text, zero background
  }),
  visible: false  // Initially hidden
});

const map = new ol.Map({
  target: 'map',
  layers: [osmLayer, lightBaseLayer, hillshadeLayer, labelsLayer],
  view: new ol.View({
    center: ol.proj.fromLonLat([11.3558, 47.0108]), // Zuckerhütl coordinates
    zoom: 12 // Closer zoom to see mountain detail
  })
});

const src = new ol.source.Vector();
const layer = new ol.layer.Vector({ source: src });
map.addLayer(layer);
// Tiled PNG layer for interpolated data
const TILE_EXTENT = [1116554.4631222049, 5871439.8889177805,
                     1454674.4631222049, 6072574.8889177805];
const tileGrid = new ol.tilegrid.TileGrid({
  extent: TILE_EXTENT,
  origin: [TILE_EXTENT[0], TILE_EXTENT[3]],
  resolutions: [100],
  tileSize: 256
});
const tileLayer = new ol.layer.Tile({visible:false}); // Back to hidden by default
map.addLayer(tileLayer);
const popup = document.getElementById('popup');
const popupContent = document.getElementById('popup-content');
const popupCloseButton = document.getElementById('popup-close-button');
const overlay = new ol.Overlay({ element: popup, positioning: 'bottom-center', stopEvent: false });
map.addOverlay(overlay);

// Declare toggle button early to avoid usage before declaration
const toggleBtn = document.getElementById('mode-toggle');

let times = [], points = [], varName = 'sqh'; // Default to SQH layer
let variables = [
  'temperature_2m', 'relative_humidity_2m', 'shortwave_radiation', 
  'cloud_cover', 'snow_depth', 'snowfall', 'wind_speed_10m', 
  'freezing_level_height', 'surface_pressure', 
  'dewpoint_2m', 'skiability', 'sqh'
];
let colorScales = {};

// Weather data lazy loading
let weatherData = null;
let weatherDataLoading = false;

// Available tile timestamps - these match the actual tile directory names
let availableTimestamps = [
  "2025-05-24T09:00:00", "2025-05-24T12:00:00", "2025-05-24T15:00:00", "2025-05-24T18:00:00",
  "2025-05-25T09:00:00", "2025-05-25T12:00:00", "2025-05-25T15:00:00", "2025-05-25T18:00:00",
  "2025-05-26T09:00:00", "2025-05-26T12:00:00", "2025-05-26T15:00:00", "2025-05-26T18:00:00",
  "2025-05-27T09:00:00", "2025-05-27T12:00:00", "2025-05-27T15:00:00", "2025-05-27T18:00:00",
  "2025-05-28T09:00:00", "2025-05-28T12:00:00", "2025-05-28T15:00:00", "2025-05-28T18:00:00"
];

const hourOffsets = [0,1,2,3]; // Index into the 4 daily timestamps (9am, 12pm, 3pm, 6pm)
let dayIdx = 0, hourIdx = 0; // Start at day 0 (May 24th) as "Today"
let currentMin = 0, currentMax = 1;
let peaks = [];
let drawerOpen = false;

const varLabels = {
  temperature_2m: 'Temperature',
  relative_humidity_2m: 'Relative Humidity',
  shortwave_radiation: 'Radiation',
  cloud_cover: 'Cloud Cover',
  snow_depth: 'Snow Depth',
  snowfall: 'Snowfall',
  wind_speed_10m: 'Wind Speed',
  freezing_level_height: 'Freezing Level',
  surface_pressure: 'Surface Pressure',
  dewpoint_2m: 'Dewpoint',
  elevation: 'Elevation',
  aspect: 'Aspect',
  slope: 'Slope',
  skiability: 'Skiability',
  sqh: 'SQH'
};

const varUnits = {
  temperature_2m: '°C',
  relative_humidity_2m: '%',
  shortwave_radiation: 'W/m²',
  cloud_cover: '%',
  snow_depth: 'm',
  snowfall: 'mm',
  wind_speed_10m: 'm/s',
  freezing_level_height: 'm',
  surface_pressure: 'hPa',
  dewpoint_2m: '°C',
  elevation: 'm',
  aspect: '°',
  slope: '°',
  skiability: '',
  sqh: ''
};

const dayBtn = document.getElementById('day-control-button');
const timeBtn = document.getElementById('time-control-button');
const infoBox = document.getElementById('info-box');

if (infoBox) {
  infoBox.addEventListener('click', (e) => {
    if (e.target.closest('.layer-item')) {
      e.stopPropagation();
    }
  });

}

// load peak list for name lookups
const peaksPromise = fetch(`${RES}data/tirol_peaks.geojson`)
  .then(r => {
    if (!r.ok) throw new Error(`Missing ${RES}data/tirol_peaks.geojson - status ${r.status}`);
    return r.json();
  })
  .then(g => {
    peaks = g.features.map(f => ({
      name: f.properties.name,
      ele: f.properties.ele,
      lat: f.geometry.coordinates[1],
      lon: f.geometry.coordinates[0]
    }));
  }).catch(() => {});

const colorScalePromise = fetch(`${RES}data/color_scales.json`)
  .then(r => {
    if (!r.ok) throw new Error(`Missing ${RES}data/color_scales.json - status ${r.status}`);
    return r.json();
  })
  .then(d => { colorScales = d; })
  .catch(e => console.error('Color scale load failed', e));

// Lazy load weather data for point mode validation
function loadWeatherData() {
  if (weatherData || weatherDataLoading) return Promise.resolve(weatherData);
  
  weatherDataLoading = true;
  console.log('Loading weather data for point validation...');
  
  return fetch(`${RES}data/weather_data_frontend.json`)
    .then(r => r.json())
    .then(d => {
      const sample = d.coordinates.find(c => c.weather_data_3hour);
      if (sample) {
        times = sample.weather_data_3hour.hourly.time.map(t => new Date(t));
        
        // Update availableTimestamps from the loaded data
        if (sample.weather_data_3hour.hourly.time && sample.weather_data_3hour.hourly.time.length > 0) {
          availableTimestamps = sample.weather_data_3hour.hourly.time;
          console.log('Updated available timestamps from data:', availableTimestamps.length, 'timestamps');
          // Update buttons to reflect potentially new dates
          updateButtons();
        }

        variables = Object.keys(sample.weather_data_3hour.hourly)
          .filter(k => k !== 'time' && k !== 'time_units');
        points = d.coordinates.filter(c => c.weather_data_3hour).map(c => ({
          lat: c.coordinate_info.latitude ?? c.latitude,
          lon: c.coordinate_info.longitude ?? c.longitude,
          info: c.coordinate_info || {},
          w: c.weather_data_3hour.hourly
        }));
        weatherData = d;
        weatherDataReady = true;
        console.log('Weather data loaded successfully (16MB frontend optimized)');
      }
      weatherDataLoading = false;
      return weatherData;
    })
    .catch(e => {
      console.error('Weather data load failed', e);
      weatherDataLoading = false;
      return null;
    });
}

// Simplified initial load - just load essentials, weather data loads separately
const weatherPromise = Promise.resolve(); // Don't load weather data initially

Promise.all([colorScalePromise, weatherPromise, peaksPromise]).then(() => {
  updateButtons();
  draw();
});

// Load weather data after 2 seconds - prioritize over PNG preloading
setTimeout(() => {
  loadWeatherData();
}, 2000);

function formatDay(d){
  const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  
  const dayName = days[d.getDay()];
  const monthName = months[d.getMonth()];
  const date = d.getDate();
  
  let suffix = 'th';
  if (date === 1 || date === 21 || date === 31) suffix = 'st';
  else if (date === 2 || date === 22) suffix = 'nd';
  else if (date === 3 || date === 23) suffix = 'rd';
  
  return `${dayName}, ${monthName} ${date}${suffix}`;
}

function formatTime(d){
  const h = d.getHours();
  const period = h >= 12 ? 'pm' : 'am';
  const hour = ((h + 11) % 12) + 1;
  return `${hour}${period}`;
}

function parseColor(str){
  if(str.startsWith('rgba')){
    const m = str.match(/rgba\((\d+),(\d+),(\d+),(\d*\.?\d+)\)/);
    return [parseInt(m[1]), parseInt(m[2]), parseInt(m[3]), parseFloat(m[4])];
  }
  const hex = str.replace('#','');
  const full = hex.length===3 ? hex.split('').map(c=>c+c).join('') : hex;
  const num = parseInt(full,16);
  return [num>>16 & 255, num>>8 & 255, num & 255, 1];
}

function interpColor(c1,c2,t){
  return [
    Math.round(c1[0] + (c2[0]-c1[0])*t),
    Math.round(c1[1] + (c2[1]-c1[1])*t),
    Math.round(c1[2] + (c2[2]-c1[2])*t),
    c1[3] + (c2[3]-c1[3])*t
  ];
}

function color(val, varName){
  const spec = colorScales[varName];
  if(!spec) return '#ff00ff';

  const min = spec.min;
  const max = spec.max;
  const palette = spec.palette;
  const t = Math.max(0, Math.min(1, (val - min)/(max - min)));
  const scaled = t * (palette.length - 1);
  const idx = Math.floor(scaled);
  const frac = scaled - idx;
  const c1 = parseColor(palette[idx]);
  const c2 = parseColor(palette[Math.min(idx+1, palette.length-1)]);
  const c = interpColor(c1,c2,frac);
  if(spec.opacity || c[3] !== 1){
    return `rgba(${c[0]},${c[1]},${c[2]},${c[3].toFixed(2)})`;
  }
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// Cache canvases for sampled PNGs
const canvasCache = {};

function loadCanvas(url){
  return new Promise((resolve, reject) => {
    if(canvasCache[url]) return resolve(canvasCache[url]);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width = img.width;
      c.height = img.height;
      c.getContext('2d').drawImage(img,0,0);
      canvasCache[url] = c;
      resolve(c);
    };
    img.onerror = () => reject(new Error('load fail'));
    img.src = url;
  });
}

function colorToValue(pixel, varName){
  const spec = colorScales[varName];
  if(!spec) return null;
  if(pixel[3] === 0) return null;

  const palette = spec.palette.map(parseColor);
  if(spec.discrete){
    let best=0, bestDist=Infinity;
    for(let i=0;i<palette.length;i++){
      const c=palette[i];
      const d=(c[0]-pixel[0])**2+(c[1]-pixel[1])**2+(c[2]-pixel[2])**2;
      if(d<bestDist){bestDist=d;best=i;}
    }
    const t=best/(palette.length-1);
    return spec.min + t*(spec.max-spec.min);
  }

  let bestVal=null,bestDist=Infinity;
  for(let i=0;i<palette.length-1;i++){
    const c1=palette[i], c2=palette[i+1];
    const dx=c2[0]-c1[0], dy=c2[1]-c1[1], dz=c2[2]-c1[2];
    const dot=(pixel[0]-c1[0])*dx+(pixel[1]-c1[1])*dy+(pixel[2]-c1[2])*dz;
    const len2=dx*dx+dy*dy+dz*dz;
    const t=Math.max(0,Math.min(1,len2?dot/len2:0));
    const r=c1[0]+dx*t, g=c1[1]+dy*t, b=c1[2]+dz*t;
    const d=(pixel[0]-r)**2+(pixel[1]-g)**2+(pixel[2]-b)**2;
    if(d<bestDist){bestDist=d;bestVal=i+t;}
  }
  if(bestVal==null) return null;
  const t=bestVal/(palette.length-1);
  return spec.min + t*(spec.max-spec.min);
}

async function getPngValue(varName, tsStr, lat, lon){
  const layerType = getLayerType(varName);
  let imageUrl;
  if(layerType === 'terrain'){
    imageUrl = `${RES}images/terrain/${varName}.png`;
  }else{
    imageUrl = `${RES}images/weather/${tsStr}/${varName}.png`;
  }
  try{
    const canvas = await loadCanvas(imageUrl);
    const coord = ol.proj.fromLonLat([lon, lat]);
    const xRatio = (coord[0]-TILE_EXTENT[0])/(TILE_EXTENT[2]-TILE_EXTENT[0]);
    const yRatio = (TILE_EXTENT[3]-coord[1])/(TILE_EXTENT[3]-TILE_EXTENT[1]);
    const x = Math.round(xRatio*canvas.width);
    const y = Math.round(yRatio*canvas.height);
    if(x<0||x>=canvas.width||y<0||y>=canvas.height) return null;
    const data = canvas.getContext('2d').getImageData(x,y,1,1).data;
    return colorToValue(data, varName);
  }catch(e){
    return null;
  }
}

function updatePopupRow(variable, value, type) {
  const row = document.getElementById(`row-${variable}`);
  if (!row) return;

  const cellIndex = type === 'source' ? 1 : 2; // 1 for Source, 2 for Extrapolated
  const cell = row.cells[cellIndex];
  
  if (value === null || value === undefined) {
    cell.innerHTML = 'N/A';
    cell.dataset.value = '';
  } else {
    const unit = varUnits[variable] || '';
    cell.innerHTML = typeof value === 'number' ? value.toFixed(1) + unit : value + unit;
    cell.dataset.value = value;
  }

  // Calculate Delta if both values are present
  const sourceCell = row.cells[1];
  const mapCell = row.cells[2];
  const deltaCell = row.cells[3];

  const sourceVal = parseFloat(sourceCell.dataset.value);
  const mapVal = parseFloat(mapCell.dataset.value);

  if (!isNaN(sourceVal) && !isNaN(mapVal)) {
    deltaCell.innerHTML = (mapVal - sourceVal).toFixed(1);
  }
}

async function fetchMapData(lat, lon, tsStr) {
  const vars = [...variables];
  if (!vars.includes('elevation')) vars.push('elevation');
  if (!vars.includes('aspect')) vars.push('aspect');
  if (!vars.includes('slope')) vars.push('slope');

  // Fetch all PNG values in parallel
  const promises = vars.map(async (v) => {
    try {
      const val = await getPngValue(v, tsStr, lat, lon);
      updatePopupRow(v, val, 'map');
    } catch (e) {
      console.warn(`Failed to fetch PNG value for ${v}`, e);
      updatePopupRow(v, null, 'map');
    }
  });

  await Promise.all(promises);
}

async function fetchApiData(lat, lon, tsStr) {
  const selectedDate = new Date(tsStr);
  const dateStr = selectedDate.toISOString().split('T')[0];
  
  // Variables supported by Open-Meteo
  const apiVars = [
    'temperature_2m','relative_humidity_2m','shortwave_radiation',
    'cloud_cover','snow_depth','snowfall','wind_speed_10m',
    'weather_code','freezing_level_height','surface_pressure',
    'dewpoint_2m'
  ];

  const apiParams = new URLSearchParams({
    latitude: lat.toFixed(6),
    longitude: lon.toFixed(6),
    hourly: apiVars.join(','),
    start_date: dateStr,
    end_date: dateStr,
    timezone: 'Europe/Vienna'
  });

  // Determine if we need Forecast or Archive API
  // Forecast API usually covers recent past (3-5 days) and future.
  // Archive API covers everything else.
  // Simple check: if date is older than 5 days ago, use Archive.
  const diffTime = new Date() - selectedDate;
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24)); 
  
  let baseUrl = 'https://api.open-meteo.com/v1/forecast';
  if (diffDays > 5) {
    baseUrl = 'https://archive-api.open-meteo.com/v1/archive';
    // Archive API doesn't use specific high-res models like icon-d2 generally, 
    // or handles them differently. We'll omit 'model' to let it use best match/reanalysis.
  } else {
    apiParams.append('model', 'icon-d2'); // Use high-res for recent/future
  }

  const apiUrl = `${baseUrl}?${apiParams}`;

  try {
    const r = await fetch(apiUrl);
    if (!r.ok) throw new Error('API request failed');
    const apiData = await r.json();

    const apiTimes = apiData.hourly.time;
    // Find closest time index
    let closestIndex = 0;
    let minDiff = Math.abs(new Date(apiTimes[0]).getTime() - new Date(tsStr).getTime());
    
    for(let i=1; i < apiTimes.length; i++){
      const d = Math.abs(new Date(apiTimes[i]).getTime() - new Date(tsStr).getTime());
      if(d < minDiff){ minDiff = d; closestIndex = i; }
    }

    // Update table with API values
    apiVars.forEach(v => {
      if (apiData.hourly[v]) {
        updatePopupRow(v, apiData.hourly[v][closestIndex], 'source');
      }
    });
    
    // API provides elevation too
    if (apiData.elevation) {
      updatePopupRow('elevation', apiData.elevation, 'source');
    }

    // Update all rows to N/A if API doesn't provide them
    const allPopupVars = [...variables, 'elevation', 'aspect', 'slope'];
    allPopupVars.forEach(v => {
      if (!apiVars.includes(v) && v !== 'elevation') {
        updatePopupRow(v, null, 'source');
      }
    });

  } catch (e) {
    console.error('API Fetch Error:', e);
    // Set all source columns to "N/A" on error
    const allVars = [...variables, 'elevation', 'aspect', 'slope'];
    allVars.forEach(v => updatePopupRow(v, null, 'source'));
  }
}

function showAsyncPopup(lat, lon, tsStr, coordinate) {
  const vars = [...variables];
  if(!vars.includes('elevation')) vars.push('elevation');
  if(!vars.includes('aspect')) vars.push('aspect');
  if(!vars.includes('slope')) vars.push('slope');

  // Build Table Skeleton
  let html = `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
               <div style="font-weight:bold;">${lat.toFixed(4)}, ${lon.toFixed(4)}</div>
               <div style="font-size:0.9em;color:#666;">${tsStr.replace('T', ' ')}</div>
             </div>`;
  
  html += `<table class="popup-table">
    <thead>
      <tr>
        <th>Variable</th>
        <th>API Source</th>
        <th>Map Value</th>
        <th>Delta</th>
      </tr>
    </thead>
    <tbody>`;

  vars.forEach(v => {
    html += `<tr id="row-${v}">
      <td>${varLabels[v] || v}</td>
      <td data-value="" class="cell-source"><div class="spinner"></div></td>
      <td data-value="" class="cell-map"><div class="spinner"></div></td>
      <td class="cell-delta"></td>
    </tr>`;
  });

  html += `</tbody></table>`;
  
  popupContent.innerHTML = html;
  overlay.setPosition(coordinate);
  popup.style.display = 'block';

  // Trigger parallel fetching
  fetchApiData(lat, lon, tsStr);
  fetchMapData(lat, lon, tsStr);
}

function haversine(lat1, lon1, lat2, lon2){
  const R=6371000;
  const rad=Math.PI/180;
  const dLat=(lat2-lat1)*rad;
  const dLon=(lon2-lon1)*rad;
  const a=Math.sin(dLat/2)**2+Math.cos(lat1*rad)*Math.cos(lat2*rad)*Math.sin(dLon/2)**2;
  return R*2*Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}



function draw(){
  const timestampIdx = dayIdx * 4 + hourIdx; // 4 times per day
  if(timestampIdx >= availableTimestamps.length) return;
  const spec = colorScales[varName] || {};
  currentMin = spec.min ?? 0;
  currentMax = spec.max ?? 1;
  
  const layerType = getLayerType(varName);
  
  if(!isPointMode){
    // Smooth mode: hide points, show image layer
    src.clear();
    layer.setVisible(false);
    updateTileLayer();
    showLayerInfoBox();
    return;
  }

  // Point mode: Load weather data if needed, then show points
  if (!weatherDataReady) {
    // Show loading message and load data
    src.clear();
    layer.setVisible(false);
    console.log('Loading weather data for point mode...');
    loadWeatherData().then(() => {
      if (weatherDataReady) {
        draw(); // Retry after data loads
      }
    });
    showLayerInfoBox();
    return;
  }

  // Point mode: show points, hide image layer
  if (tileLayer.imageLayer) {
    tileLayer.imageLayer.setVisible(false);
  }
  
  // For terrain layers, we don't have point data, so switch to smooth mode
  if (layerType === 'terrain') {
    isPointMode = false;
    toggleBtn.classList.add('smooth');
    updateBaseMapLayers();
    layer.setVisible(false);
    updateTileLayer();
    showLayerInfoBox();
    return;
  }
  
  layer.setVisible(true);

  const feats = points.map(p=>{
    const v = p.w[varName]?.[timestampIdx];
    const f = new ol.Feature(new ol.geom.Point(ol.proj.fromLonLat([p.lon,p.lat])));
    f.set('v', v);
    f.set('data', {p, timestampIdx});
    return f;
  });
  src.clear();
  feats.forEach(f=>{
    const v=f.get('v');
    if(typeof v!=='number') return;
    f.setStyle(new ol.style.Style({image:new ol.style.Circle({radius:6,fill:new ol.style.Fill({color:color(v,varName)})})}));
    src.addFeature(f);
  });
  showLayerInfoBox();
}

function updateButtons(){
  const timestampIdx = dayIdx * 4 + hourIdx; // 4 times per day
  if(timestampIdx >= availableTimestamps.length) return;
  const t = new Date(availableTimestamps[timestampIdx]);
  dayBtn.textContent = formatDay(t);
  timeBtn.textContent = formatTime(t);
}

function showLayerInfoBox(){
  const info = document.getElementById('info-box');
  const timestampIdx = dayIdx * 4 + hourIdx; // 4 times per day
  if(timestampIdx >= availableTimestamps.length) return;
  const t = new Date(availableTimestamps[timestampIdx]);
  if(!info || !t) return;
  info.classList.remove('info-box-selecting');
  const spec = colorScales[varName] || {palette:['#0000ff','#ff0000']};
  
  const layerType = getLayerType(varName);
  
  const label = `${varLabels[varName] ?? varName}`;
  const timeLabel = `${formatDay(t)} at ${formatTime(t)}`;
  
  // Update mobile-only floating date/time
  const mobDt = document.getElementById('mobile-datetime');
  if(mobDt) mobDt.textContent = timeLabel;

  const unit = varUnits[varName] ?? '';
  info.classList.remove('info-box-selecting');
  
  const barStyle = `background:linear-gradient(to right,${spec.palette.join(',')})`;
  info.innerHTML = `
    <div class="info-box-left desktop-only">
      ${label} on ${timeLabel}
    </div>
    <div class="info-box-right">
      <span class="desktop-only">${currentMin.toFixed(1)}${unit}</span>
      <div class="legend-bar" style="${barStyle}">
          <span class="legend-value left mobile-only">${currentMin.toFixed(1)}${unit}</span>
          <span class="legend-overlay-title mobile-only">${label}</span>
          <span class="legend-value right mobile-only">${currentMax.toFixed(1)}${unit}</span>
      </div>
      <span class="desktop-only">${currentMax.toFixed(1)}${unit}</span>
    </div>
  `;
  
  info.style.display = 'block';
}
function updateTileLayer(){
  const timestampIdx = dayIdx * 4 + hourIdx; // 4 times per day
  if(timestampIdx >= availableTimestamps.length) return;
  const ts = availableTimestamps[timestampIdx];
  
  // Remove any existing image layer
  if (tileLayer.imageLayer) {
    map.removeLayer(tileLayer.imageLayer);
  }
  
  // Determine the image URL based on layer type
  let imageUrl;
  const layerType = getLayerType(varName);
  
  if (layerType === 'terrain') {
    // Static terrain layers
    imageUrl = `${RES}images/terrain/${varName}.png`;
  } else if (layerType === 'snow_composite' || layerType === 'weather') {
    // Time-dependent layers (weather data, skiability, SQH)
    imageUrl = `${RES}images/weather/${ts}/${varName}.png`;
  } else {
    console.warn(`Unknown layer type for ${varName}`);
    return;
  }
  
  const src = new ol.source.ImageStatic({
    url: imageUrl,
    imageExtent: TILE_EXTENT, // Use the same extent as before
    projection: 'EPSG:3857' // Web Mercator
  });
  
  const imageLayer = new ol.layer.Image({
    source: src,
    opacity: 0.7 // Make it semi-transparent so we can see the base map
  });
  
  // Store reference and add to map
  tileLayer.imageLayer = imageLayer;
  map.addLayer(imageLayer);
}

// Helper function to determine layer type
function getLayerType(layerName) {
  const terrainLayers = ['elevation', 'aspect', 'slope'];
  const snowCompositeLayers = ['skiability', 'sqh'];
  
  if (terrainLayers.includes(layerName)) {
    return 'terrain';
  } else if (snowCompositeLayers.includes(layerName)) {
    return 'snow_composite';
  } else {
    return 'weather';
  }
}

// Helper function to check if layer supports point mode
function supportsPointMode(layerName) {
  const layerType = getLayerType(layerName);
  // Only weather layers support point mode
  return layerType === 'weather';
}

// Helper function to update toggle button state
function updateToggleButtonState() {
  const supportsPoints = supportsPointMode(varName);
  
  if (supportsPoints) {
    // Enable the toggle button
    toggleBtn.disabled = false;
    toggleBtn.classList.remove('disabled');
    toggleBtn.title = 'Toggle between point and smooth visualization';
  } else {
    // Disable the toggle button and force smooth mode
    toggleBtn.disabled = true;
    toggleBtn.classList.add('disabled');
    toggleBtn.title = 'This layer only supports smooth visualization';
    isPointMode = false;
    toggleBtn.classList.add('smooth');
  }
}


function isMobile() {
  return window.innerWidth <= 768;
}

function showDaySelector(){
  if (isMobile()) {
    const layerGrid = document.getElementById('layer-grid');
    const selectorGrid = document.getElementById('mobile-selector-grid');
    if (!layerGrid || !selectorGrid) return;

    selectorGrid.innerHTML = '';
    selectorGrid.style.display = 'grid'; // Ensure grid layout
    layerGrid.style.display = 'none';

    for(let i=0; i < 5; i++){ 
      const timestampIdx = i * 4;
      if(timestampIdx >= availableTimestamps.length) continue;
      const d = new Date(availableTimestamps[timestampIdx]);
      const div = document.createElement('div');
      div.className = 'layer-item';
      div.textContent = formatDay(d);
      if (i === dayIdx) div.classList.add('active');
      div.onclick = () => {
        dayIdx = i;
        updateButtons();
        draw();
        // Return to layer view
        selectorGrid.style.display = 'none';
        layerGrid.style.display = 'grid';
      };
      selectorGrid.appendChild(div);
    }
    return;
  }

  // Desktop Behavior
  const info = document.getElementById('info-box');
  info.innerHTML='';
  info.classList.add('info-box-selecting');
  const dayRow = document.createElement('div');
  dayRow.className = 'date-selector-row';
  info.appendChild(dayRow);

  for(let i=0; i < 5; i++){ 
    const timestampIdx = i * 4; 
    if(timestampIdx >= availableTimestamps.length) continue;
    const d = new Date(availableTimestamps[timestampIdx]);
    const div = document.createElement('div');
    div.className = 'layer-item';
    div.style.setProperty('--selector-basis', '15%');
    div.textContent = formatDay(d);
    if (i === dayIdx) div.classList.add('active');
    div.onclick = () => {
      dayIdx = i;
      updateButtons();
      draw();
      showDaySelector();
    };
    dayRow.appendChild(div);
  }
  info.style.display='block';
}

function showTimeSelector(){
  if (isMobile()) {
    const layerGrid = document.getElementById('layer-grid');
    const selectorGrid = document.getElementById('mobile-selector-grid');
    if (!layerGrid || !selectorGrid) return;

    selectorGrid.innerHTML = '';
    selectorGrid.style.display = 'grid';
    layerGrid.style.display = 'none';

    for(let i=0; i<4; i++){
      const tsStr = availableTimestamps[dayIdx*4 + i];
      if(!tsStr) continue;
      const d = new Date(tsStr);
      const div = document.createElement('div');
      div.className = 'layer-item';
      div.textContent = formatTime(d);
      if(i === hourIdx) div.classList.add('active');
      div.onclick = () => {
        hourIdx = i;
        updateButtons();
        draw();
        // Return to layer view
        selectorGrid.style.display = 'none';
        layerGrid.style.display = 'grid';
      };
      selectorGrid.appendChild(div);
    }
    return;
  }

  // Desktop Behavior
  const info = document.getElementById('info-box');
  info.innerHTML='';
  info.classList.add('info-box-selecting');
  const timeRow = document.createElement('div');
  timeRow.className = 'time-selector-row';
  info.appendChild(timeRow);

  for(let i=0; i<4; i++){
    const tsStr = availableTimestamps[dayIdx*4 + i];
    if(!tsStr) continue;
    const d = new Date(tsStr);
    const div = document.createElement('div');
    div.className = 'layer-item';
    div.style.setProperty('--selector-basis', '20%');
    div.textContent = formatTime(d);
    if(i === hourIdx) div.classList.add('active');
    div.onclick = () => {
      hourIdx = i;
      updateButtons();
      draw();
      showTimeSelector();
    };
    timeRow.appendChild(div);
  }
  info.style.display='block';
}

dayBtn.onclick=()=>{showDaySelector();};
timeBtn.onclick=()=>{showTimeSelector();};

// Toggle button functionality
let isPointMode = false; // Default to smooth mode for web
let weatherDataReady = false;

// Function to update base map layers based on mode
function updateBaseMapLayers() {
  if (isPointMode) {
    // Point mode: Show OSM, hide smooth mode layers
    osmLayer.setVisible(true);
    lightBaseLayer.setVisible(false);
    hillshadeLayer.setVisible(false);
    labelsLayer.setVisible(false);
  } else {
    // Smooth mode: Show light base + hillshade + labels, hide OSM
    osmLayer.setVisible(false);
    lightBaseLayer.setVisible(true);
    hillshadeLayer.setVisible(true);
    labelsLayer.setVisible(true);
  }
}

// Initialize with smooth mode base map
updateBaseMapLayers();

// Update toggle button to reflect smooth mode
if (toggleBtn) {
  toggleBtn.classList.add('smooth');
}

function toggleMode() {
  // Don't do anything if the button is disabled
  if (toggleBtn.disabled) return;
  
  isPointMode = !isPointMode;
  updateBaseMapLayers();
  
  if (isPointMode) {
    toggleBtn.classList.remove('smooth');
    // Hide any existing image layer
    if (tileLayer.imageLayer) {
      tileLayer.imageLayer.setVisible(false);
    }
    tileLayer.setVisible(false);
    draw(); // This will show the points
  } else {
    toggleBtn.classList.add('smooth');
    // Hide the vector points
    layer.setVisible(false);
    // Show/update the image layer
    draw(); // This will update the image layer
  }
  console.log('Mode:', isPointMode ? 'Point' : 'Smooth');
}

toggleBtn.onclick = toggleMode;

// Handle all layer types: weather, terrain, and snow_composite
document.querySelectorAll('.layer-item[data-layer-name]').forEach(btn=>{
  if(btn.dataset.layerName===varName) btn.classList.add('active'); // SQH will be active
  btn.onclick=()=>{
    document.querySelectorAll('.layer-item[data-layer-name]').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    varName=btn.dataset.layerName;
    updateToggleButtonState(); // Update toggle button based on new layer
    draw();
  };
});

function findPeak(lat,lon){
  return peaks.find(p=>Math.abs(p.lat-lat)<1e-4 && Math.abs(p.lon-lon)<1e-4);
}

map.on('singleclick', async evt => {
  if(drawerOpen){
    toggleDrawer();
    return;
  }
  overlay.setPosition(undefined);

  const clicked = ol.proj.toLonLat(evt.coordinate);
  const [lon,lat] = clicked;
  const tsStr = availableTimestamps[dayIdx*4 + hourIdx];
  
  if(!tsStr){
    popupContent.textContent = 'Time data unavailable';
    overlay.setPosition(evt.coordinate);
    popup.style.display='block';
    return;
  }

  // Immediately show popup with spinners
  showAsyncPopup(lat, lon, tsStr, evt.coordinate);
});

// Drawer toggle functionality
window.toggleDrawer = function() {
  const cont = document.getElementById('drawer-container');
  cont.classList.toggle('open');
  drawerOpen = cont.classList.contains('open');
  popup.style.display='none';
  if(!drawerOpen) showLayerInfoBox();
};

// Event listener for 'Escape' key press
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    if (drawerOpen) {
      toggleDrawer();
    }
    if (popup.style.display !== 'none') {
      popup.style.display = 'none';
    }
  } else if (event.key === ' ') {
    // Spacebar toggles settings panel
    toggleDrawer();
    event.preventDefault();
  } else if (event.key === 'ArrowUp') {
    // Up arrow cycles to previous layer
    const layers = Array.from(document.querySelectorAll('.layer-item[data-layer-name]'));
    const currentIndex = layers.findIndex(layer => layer.dataset.layerName === varName);
    const prevIndex = currentIndex > 0 ? currentIndex - 1 : layers.length - 1;
    const prevLayer = layers[prevIndex];
    
    // Update active layer
    layers.forEach(layer => layer.classList.remove('active'));
    prevLayer.classList.add('active');
    varName = prevLayer.dataset.layerName;
    updateToggleButtonState();
    draw();
    event.preventDefault();
  } else if (event.key === 'ArrowDown') {
    // Down arrow cycles to next layer
    const layers = Array.from(document.querySelectorAll('.layer-item[data-layer-name]'));
    const currentIndex = layers.findIndex(layer => layer.dataset.layerName === varName);
    const nextIndex = currentIndex < layers.length - 1 ? currentIndex + 1 : 0;
    const nextLayer = layers[nextIndex];
    
    // Update active layer
    layers.forEach(layer => layer.classList.remove('active'));
    nextLayer.classList.add('active');
    varName = nextLayer.dataset.layerName;
    updateToggleButtonState();
    draw();
    event.preventDefault();
  } else if (event.key === 'ArrowRight') {
    const idx = dayIdx * 4 + hourIdx;
    if (idx < availableTimestamps.length - 1) {
      hourIdx++;
      if (hourIdx > 3) { hourIdx = 0; dayIdx++; }
      updateButtons();
      draw();
    }
    event.preventDefault();
  } else if (event.key === 'ArrowLeft') {
    const idx = dayIdx * 4 + hourIdx;
    if (idx > 0) {
      hourIdx--;
      if (hourIdx < 0) { hourIdx = 3; dayIdx--; }
      updateButtons();
      draw();
    }
    event.preventDefault();
  } else if (event.key === 'Shift' && event.location === KeyboardEvent.DOM_KEY_LOCATION_RIGHT) {
    // Right shift key toggles between point/smooth mode
    toggleMode();
    event.preventDefault();
  }
});

// Popup close button functionality
if (popupCloseButton) {
  popupCloseButton.onclick = () => {
    popup.style.display = 'none';
    overlay.setPosition(undefined); // Also clear the overlay's position
  };
}

// Initialize toggle button state based on the default layer
updateToggleButtonState();

// Background PNG preloading system
let preloadingActive = false;
let preloadedCount = 0;
let totalPngsToPreload = 0;

// Generate all PNG URLs to preload
function generateAllPngUrls() {
  const allUrls = [];
  
  // Terrain PNGs (always available)
  const terrainVars = ['elevation', 'aspect', 'slope'];
  terrainVars.forEach(varName => {
    allUrls.push(`${RES}images/terrain/${varName}.png`);
  });
  
  // Weather and composite PNGs (time-based)
  const weatherVars = ['temperature_2m', 'relative_humidity_2m', 'shortwave_radiation', 
                      'cloud_cover', 'snow_depth', 'snowfall', 'wind_speed_10m', 
                      'freezing_level_height', 'surface_pressure', 
                      'dewpoint_2m', 'skiability', 'sqh'];
  
  availableTimestamps.forEach(timestamp => {
    weatherVars.forEach(varName => {
      allUrls.push(`${RES}images/weather/${timestamp}/${varName}.png`);
    });
  });
  
  // Sort alphabetically for systematic loading
  return allUrls.sort();
}

function preloadPngsInBackground() {
  if (preloadingActive) return;
  const allUrls = generateAllPngUrls();
  preloadingActive = true;
  totalPngsToPreload = allUrls.length;
  preloadedCount = 0;
  
  console.log(`Starting background preload of ${totalPngsToPreload} PNGs...`);
  
  // Preload with delay to avoid overwhelming the browser
  let urlIndex = 0;
  
  function preloadNext() {
    if (urlIndex >= allUrls.length) {
      console.log(`✅ Preloaded all ${preloadedCount} PNGs successfully!`);
      preloadingActive = false;
      return;
    }
    
    const url = allUrls[urlIndex++];
    loadCanvas(url)
      .then(() => {
        preloadedCount++;
        if (preloadedCount % 10 === 0) {
          console.log(`📥 Preloaded ${preloadedCount}/${totalPngsToPreload} PNGs (${Math.round(preloadedCount/totalPngsToPreload*100)}%)`);
        }
        // Small delay between requests to avoid overwhelming the browser
        setTimeout(preloadNext, 50);
      })
      .catch(err => {
        // Silently continue on error - some PNGs might not exist
        setTimeout(preloadNext, 50);
      });
  }
  
  preloadNext();
}

// Start preloading when the app initializes
window.addEventListener('load', () => {
  // Load weather data after 2 seconds for better UX
  setTimeout(() => {
    console.log('Auto-loading weather data...');
    loadWeatherData();
  }, 2000);
  
  // Give weather data priority - start PNG preloading after weather data has had time to load
  setTimeout(preloadPngsInBackground, 5000);
});