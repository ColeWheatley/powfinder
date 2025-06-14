import 'https://cdn.jsdelivr.net/npm/ol@v7.4.0/dist/ol.js';

const map = new ol.Map({
  target: 'map',
  layers: [new ol.layer.Tile({ 
    source: new ol.source.OSM({
      cacheSize: 8192  // 4x default cache size for smooth zooming
    })
  })],
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

let times = [], points = [], varName = 'temperature_2m';
let variables = [];
let colorScales = {};

// Available tile timestamps - these match the actual tile directory names
const availableTimestamps = [
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
  weather_code: 'Weather Code',
  freezing_level_height: 'Freezing Level',
  surface_pressure: 'Surface Pressure'
};

const varUnits = {
  temperature_2m: '°C',
  relative_humidity_2m: '%',
  shortwave_radiation: 'W/m²',
  cloud_cover: '%',
  snow_depth: 'm',
  snowfall: 'mm',
  wind_speed_10m: 'm/s',
  weather_code: '',
  freezing_level_height: 'm',
  surface_pressure: 'hPa'
};

const dayBtn = document.getElementById('day-control-button');
const timeBtn = document.getElementById('time-control-button');

// load peak list for name lookups
const peaksPromise = fetch('resources/meteo_api/tirol_peaks.geojson')
  .then(r => r.json())
  .then(g => {
    peaks = g.features.map(f => ({
      name: f.properties.name,
      ele: f.properties.ele,
      lat: f.geometry.coordinates[1],
      lon: f.geometry.coordinates[0]
    }));
  }).catch(() => {});

const colorScalePromise = fetch('resources/Make%20TIFs/color_scales.json')
  .then(r => r.json())
  .then(d => { colorScales = d; })
  .catch(e => console.error('Color scale load failed', e));

const weatherPromise = fetch('resources/meteo_api/weather_data_3hour.json').then(r => r.json()).then(d => {
  const sample = d.coordinates.find(c => c.weather_data_3hour);
  if (!sample) return;
  times = sample.weather_data_3hour.hourly.time.map(t => new Date(t));
  variables = Object.keys(sample.weather_data_3hour.hourly)
    .filter(k => k !== 'time' && k !== 'time_units');
  points = d.coordinates.filter(c => c.weather_data_3hour).map(c => ({
    lat: c.coordinate_info.latitude ?? c.latitude,
    lon: c.coordinate_info.longitude ?? c.longitude,
    info: c.coordinate_info || {},
    w: c.weather_data_3hour.hourly
  }));
}).catch(e => console.error('Weather data load failed', e));

Promise.all([colorScalePromise, weatherPromise, peaksPromise]).then(() => {
  updateButtons();
  draw();
});

function formatDay(d){
  // Use May 24th as reference "Today" 
  const referenceDate = new Date('2025-05-24T00:00:00');
  const diff = Math.round((d - referenceDate) / 86400000);
  if(diff === 0) return 'Today';
  if(diff === 1) return 'Tomorrow';
  if(diff === -1) return 'Yesterday';
  return d.toLocaleDateString('en-US',{weekday:'long'});
}

function formatTime(d){
  const h = d.getHours();
  const period = h >= 12 ? 'pm' : 'am';
  const hour = ((h + 11) % 12) + 1;
  return `${hour}${period}`;
}

function hexToRgb(hex){
  hex = hex.replace('#','');
  if(hex.length===3) hex = hex.split('').map(c=>c+c).join('');
  const num = parseInt(hex,16);
  return [num>>16 & 255, num>>8 & 255, num & 255];
}

function interpColor(c1,c2,t){
  return [
    Math.round(c1[0] + (c2[0]-c1[0])*t),
    Math.round(c1[1] + (c2[1]-c1[1])*t),
    Math.round(c1[2] + (c2[2]-c1[2])*t)
  ];
}

function color(val, varName){
  const spec = colorScales[varName];
  if(!spec) return '#ff00ff';

  if(varName === 'weather_code') {
    const pal = spec.palette;
    if(val <= 3) return pal[0];
    if(val <= 48) return pal[1];
    if(val <= 67) return pal[2];
    if(val <= 77) return pal[3];
    return pal[4];
  }

  const min = spec.min;
  const max = spec.max;
  const palette = spec.palette;
  const t = Math.max(0, Math.min(1, (val - min)/(max - min)));
  const scaled = t * (palette.length - 1);
  const idx = Math.floor(scaled);
  const frac = scaled - idx;
  const c1 = hexToRgb(palette[idx]);
  const c2 = hexToRgb(palette[Math.min(idx+1, palette.length-1)]);
  const c = interpColor(c1,c2,frac);
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function draw(){
  const timestampIdx = dayIdx * 4 + hourIdx; // 4 times per day
  if(timestampIdx >= availableTimestamps.length) return;
  const spec = colorScales[varName] || {};
  currentMin = spec.min ?? 0;
  currentMax = spec.max ?? 1;
  
  if(!isPointMode){
    // Smooth mode: hide points, show image layer
    src.clear();
    layer.setVisible(false);
    updateTileLayer();
    showLayerInfoBox();
    return;
  }

  // Point mode: show points, hide image layer
  if (tileLayer.imageLayer) {
    tileLayer.imageLayer.setVisible(false);
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
  const barStyle = `background:linear-gradient(to right,${spec.palette.join(',')})`;
  const label = `${varLabels[varName] ?? varName} ${formatDay(t)} at ${formatTime(t)}`;
  const unit = varUnits[varName] ?? '';
  info.classList.remove('info-box-selecting');
  info.innerHTML = `
    <div class="info-box-left">
      ${label}
    </div>
    <div class="info-box-right">
      <span>${currentMin.toFixed(1)}${unit}</span>
      <div class="legend-bar" style="${barStyle}"></div>
      <span>${currentMax.toFixed(1)}${unit}</span>
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
  
  // For now, use a simple image layer instead of tiles
  // This will display the full PNG image for the selected variable and time
  const imageUrl = `TIFS/100m_resolution/${ts}/${varName}.png`;
  
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


function showDaySelector(){
  const info = document.getElementById('info-box');
  info.innerHTML='';
  info.classList.add('info-box-selecting');
  const dayRow = document.createElement('div');
  dayRow.className = 'date-selector-row';
  info.appendChild(dayRow);

  for(let i=0; i < 5; i++){ // 5 days available
    const timestampIdx = i * 4; // First timestamp of each day
    if(timestampIdx >= availableTimestamps.length) continue;
    const d = new Date(availableTimestamps[timestampIdx]);
    const div = document.createElement('div');
    div.className = 'layer-item';
    div.textContent = formatDay(d);
    if (i === dayIdx) div.classList.add('active');
    div.onclick = () => { dayIdx = i; updateButtons(); draw(); };
    dayRow.appendChild(div);
  }
  info.style.display='block';
}

function showTimeSelector(){
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
    div.textContent = formatTime(d);
    if(i === hourIdx) div.classList.add('active');
    div.onclick = () => { hourIdx = i; updateButtons(); draw(); };
    timeRow.appendChild(div);
  }
  info.style.display='block';
}

dayBtn.onclick=()=>{showDaySelector();};
timeBtn.onclick=()=>{showTimeSelector();};

// Toggle button functionality
let isPointMode = true; // Back to starting in point mode
const toggleBtn = document.getElementById('mode-toggle');

toggleBtn.onclick = () => {
  isPointMode = !isPointMode;
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
};

document.querySelectorAll('.layer-item[data-layer-type="weather"]').forEach(btn=>{
  if(btn.dataset.layerName===varName) btn.classList.add('active');
  btn.onclick=()=>{
    // Ignore clicks on not-implemented items
    if(btn.classList.contains('not-implemented')) return;
    
    document.querySelectorAll('.layer-item[data-layer-type="weather"]').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    varName=btn.dataset.layerName;draw();
  };
});

function findPeak(lat,lon){
  return peaks.find(p=>Math.abs(p.lat-lat)<1e-4 && Math.abs(p.lon-lon)<1e-4);
}

map.on('singleclick', evt=>{
  if(drawerOpen){
    toggleDrawer();
    return;
  }
  overlay.setPosition(undefined);
  const feature = map.forEachFeatureAtPixel(evt.pixel,f=>f);

  const handleNoData = () => {
    const clickedCoord = ol.proj.toLonLat(evt.coordinate);
    const [lon, lat] = clickedCoord;

    // Get current selected timestamp
    const tsStr = availableTimestamps[dayIdx*4 + hourIdx];
    if(!tsStr){
        popupContent.textContent = 'Time data unavailable for API request.';
        overlay.setPosition(evt.coordinate);
        popup.style.display = 'block';
        return;
    }

    // Get the selected date and time from the frontend interface
    const selectedDate = new Date(tsStr);
    const dateStr = selectedDate.toISOString().split('T')[0]; // YYYY-MM-DD format

    // Build Open-Meteo API URL with same parameters as collection script
    // Use the date range that covers our selected time
    const apiParams = new URLSearchParams({
        latitude: lat.toFixed(6),
        longitude: lon.toFixed(6),
        model: 'icon-d2',
        hourly: [
            'temperature_2m', 'relative_humidity_2m', 'shortwave_radiation',
            'cloud_cover', 'snow_depth', 'snowfall', 'wind_speed_10m',
            'weather_code', 'freezing_level_height', 'surface_pressure'
        ].join(','),
        start_date: dateStr,
        end_date: dateStr,
        timezone: 'Europe/Vienna'
    });

    const apiUrl = `https://api.open-meteo.com/v1/forecast?${apiParams}`;

    popupContent.textContent = 'Loading weather data...';
    overlay.setPosition(evt.coordinate);
    popup.style.display = 'block';

    fetch(apiUrl)
      .then(response => {
        if (!response.ok) {
          throw new Error(`API request failed: ${response.status}`);
        }
        return response.json();
      })
      .then(apiData => {
        if (!apiData.hourly || !apiData.hourly.time) {
          throw new Error('Invalid API response structure');
        }

        // Find the closest time index to our selected time
        const targetTime = selectedDate.toISOString();
        const apiTimes = apiData.hourly.time;
        let closestIndex = 0;
        let minDiff = Math.abs(new Date(apiTimes[0]).getTime() - selectedDate.getTime());

        for (let i = 1; i < apiTimes.length; i++) {
          const diff = Math.abs(new Date(apiTimes[i]).getTime() - selectedDate.getTime());
          if (diff < minDiff) {
            minDiff = diff;
            closestIndex = i;
          }
        }

        // Extract weather values for the closest time
        const weatherData = {};
        Object.keys(apiData.hourly).forEach(param => {
          if (param !== 'time' && apiData.hourly[param][closestIndex] !== null) {
            weatherData[param] = apiData.hourly[param][closestIndex];
          }
        });

        // Format and display
        let formattedHtml = `<div><strong>Location:</strong> ${lat.toFixed(4)}, ${lon.toFixed(4)}</div>`;
        formattedHtml += `<div><strong>Time:</strong> ${apiTimes[closestIndex]}</div><br>`;

        Object.entries(weatherData).forEach(([key, value]) => {
          const label = varLabels[key] || key;
          const unit = varUnits[key] || '';
          const displayValue = typeof value === 'number' ? value.toFixed(1) : value;
          formattedHtml += `<div>${label}: ${displayValue}${unit}</div>`;
        });

        popupContent.innerHTML = formattedHtml;
      })
      .catch(error => {
        console.error('Weather API error:', error);
        popupContent.textContent = `Failed to fetch weather data: ${error.message}`;
      });
  };

  if(!feature){
    handleNoData();
    return;
  }
  const data = feature.get('data');
  if(!data){
    handleNoData();
    return;
  }
  const {p,idx} = data;
  const lines=[];
  if(p.info && p.info.type){
    if(p.info.type==='peak'){
      const peak=findPeak(p.lat,p.lon);
      if(peak){
        lines.push(`<strong>Peak:</strong> ${peak.name} (${peak.ele}m)`);
      }else{
        lines.push('<strong>Peak point</strong>');
      }
    }else{
      lines.push('<strong>Random point</strong>');
    }
  }
  lines.push(`<strong>Location:</strong> ${p.lat.toFixed(4)}, ${p.lon.toFixed(4)}<br>`);
  
  variables.forEach(v=>{
    const val=p.w[v]?.[idx];
    if(val!=null) {
      const label = varLabels[v] || v;
      const unit = varUnits[v] || '';
      const displayValue = typeof val === 'number' ? val.toFixed(1) : val;
      lines.push(`${label}: ${displayValue}${unit}`);
    }
  });
  popupContent.innerHTML=lines.join('<br>');
  overlay.setPosition(evt.coordinate);
  popup.style.display='block';

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
    // Check if the drawer is open
    if (drawerOpen) {
      toggleDrawer();
    }
    // Check if the popup is visible
    if (popup.style.display !== 'none') {
      popup.style.display = 'none';
    }
  }
});

// Popup close button functionality
if (popupCloseButton) {
  popupCloseButton.onclick = () => {
    popup.style.display = 'none';
    overlay.setPosition(undefined); // Also clear the overlay's position
  };
}