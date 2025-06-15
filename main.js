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
  weather_code: '',
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
    imageUrl = `TIFS/100m_resolution/terrainPNGs/${varName}.png`;
  }else{
    imageUrl = `TIFS/100m_resolution/${tsStr}/${varName}.png`;
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

function haversine(lat1, lon1, lat2, lon2){
  const R=6371000;
  const rad=Math.PI/180;
  const dLat=(lat2-lat1)*rad;
  const dLon=(lon2-lon1)*rad;
  const a=Math.sin(dLat/2)**2+Math.cos(lat1*rad)*Math.cos(lat2*rad)*Math.sin(dLon/2)**2;
  return R*2*Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

function nearestDataPoint(lat,lon){
  let best=null,dist=Infinity;
  for(const p of points){
    const d=haversine(lat,lon,p.lat,p.lon);
    if(d<dist){dist=d;best=p;}
  }
  return best?{point:best,distance:dist}:null;
}

async function showResidualPopup(lat, lon, srcVals, srcLabel, tsStr, nearest, coordinate){
  const vars = [...variables];
  if(!vars.includes('elevation')) vars.push('elevation');
  const pngVals={};
  for(const v of vars){
    pngVals[v] = await getPngValue(v, tsStr, lat, lon);
  }

  let middle='';
  if(nearest){
    if(nearest.point.info && nearest.point.info.type==='peak'){
      const pk=findPeak(nearest.point.lat, nearest.point.lon);
      const name = pk?pk.name:(nearest.point.info.name||'Peak');
      middle = `${name} - ${Math.round(nearest.distance)}m away`;
    }else{
      middle = `Validation Ping - ${Math.round(nearest.distance)}m away`;
    }
  }

  let html=`<div style="display:flex;justify-content:space-between;align-items:center;">
               <div>${lat.toFixed(4)}, ${lon.toFixed(4)}</div>
               <div>${middle}</div>
             </div>`;
  html += `<table class="popup-table"><tr><th>Variable</th><th>${srcLabel}</th><th>Extrapolated</th><th>Delta</th></tr>`;
  vars.forEach(v=>{
    const srcVal = srcVals[v];
    const pngVal = pngVals[v];
    const unit = varUnits[v]||'';
    const srcDisp = srcVal!=null ? (typeof srcVal==='number'?srcVal.toFixed(1):srcVal)+unit : 'N/A';
    const pngDisp = pngVal!=null ? pngVal.toFixed(1)+unit : 'N/A';
    const delta = (srcVal!=null && pngVal!=null) ? (pngVal - srcVal).toFixed(1) : '';
    html += `<tr><td>${varLabels[v]||v}</td><td>${srcDisp}</td><td>${pngDisp}</td><td>${delta}</td></tr>`;
  });
  html += '</table>';
  popupContent.innerHTML = html;
  overlay.setPosition(coordinate);
  popup.style.display='block';
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

  // Point mode: show points, hide image layer
  if (tileLayer.imageLayer) {
    tileLayer.imageLayer.setVisible(false);
  }
  
  // For terrain layers, we don't have point data, so switch to smooth mode
  if (layerType === 'terrain') {
    isPointMode = false;
    toggleBtn.classList.add('smooth');
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
  
  // For terrain layers, don't show time info
  let label;
  if (layerType === 'terrain') {
    label = `${varLabels[varName] ?? varName}`;
  } else {
    label = `${varLabels[varName] ?? varName} ${formatDay(t)} at ${formatTime(t)}`;
  }
  
  const unit = varUnits[varName] ?? '';
  info.classList.remove('info-box-selecting');
  if(varName === 'weather_code'){
    const cats = [
      {range:'0-3', label:'Clear/cloudy', color:spec.palette[0]},
      {range:'45-48', label:'Fog', color:spec.palette[1]},
      {range:'51-86', label:'Rain/showers', color:spec.palette[2]},
      {range:'71-77', label:'Snow', color:spec.palette[3]},
      {range:'95-99', label:'Thunder', color:spec.palette[4]}
    ];
    info.innerHTML = `<div class="info-box-left">${label}</div>`;
    const row = document.createElement('div');
    row.className = 'date-selector-row';
    cats.forEach(c=>{
      const div = document.createElement('div');
      div.className = 'layer-item';
      const flexMap = {3:30,4:20,5:15,6:12};
      const basis = flexMap[cats.length] ?? (100/cats.length);
      div.style.setProperty('--selector-basis', `${basis}%`);
      div.innerHTML = `<span style="background:${c.color};width:1em;height:1em;display:inline-block;margin-right:0.5em"></span>${c.label}`;
      row.appendChild(div);
    });
    info.appendChild(row);
  }else{
    const barStyle = `background:linear-gradient(to right,${spec.palette.join(',')})`;
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
  }
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
    imageUrl = `TIFS/100m_resolution/terrainPNGs/${varName}.png`;
  } else if (layerType === 'snow_composite' || layerType === 'weather') {
    // Time-dependent layers (weather data, skiability, SQH)
    imageUrl = `TIFS/100m_resolution/${ts}/${varName}.png`;
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
  const toggleBtn = document.getElementById('mode-toggle');
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
let isPointMode = true; // Back to starting in point mode
const toggleBtn = document.getElementById('mode-toggle');

toggleBtn.onclick = () => {
  // Don't do anything if the button is disabled
  if (toggleBtn.disabled) return;
  
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

// Handle all layer types: weather, terrain, and snow_composite
document.querySelectorAll('.layer-item[data-layer-name]').forEach(btn=>{
  if(btn.dataset.layerName===varName) btn.classList.add('active');
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

map.on('singleclick', async evt=>{
  if(drawerOpen){
    toggleDrawer();
    return;
  }
  overlay.setPosition(undefined);
  const feature = map.forEachFeatureAtPixel(evt.pixel,f=>f);

  const clicked = ol.proj.toLonLat(evt.coordinate);
  const [lon,lat] = clicked;
  const tsStr = availableTimestamps[dayIdx*4 + hourIdx];
  if(!tsStr){
    popupContent.textContent = 'Time data unavailable';
    overlay.setPosition(evt.coordinate);
    popup.style.display='block';
    return;
  }

  const nearest = nearestDataPoint(lat,lon);

  if(!feature || !feature.get('data')){
    // Fetch from API
    const selectedDate = new Date(tsStr);
    const dateStr = selectedDate.toISOString().split('T')[0];
    const apiParams = new URLSearchParams({
      latitude: lat.toFixed(6),
      longitude: lon.toFixed(6),
      model: 'icon-d2',
      hourly: [
        'temperature_2m','relative_humidity_2m','shortwave_radiation','cloud_cover','snow_depth','snowfall','wind_speed_10m','weather_code','freezing_level_height','surface_pressure'
      ].join(','),
      start_date: dateStr,
      end_date: dateStr,
      timezone: 'Europe/Vienna'
    });

    const apiUrl = `https://api.open-meteo.com/v1/forecast?${apiParams}`;
    popupContent.textContent='Loading weather data...';
    overlay.setPosition(evt.coordinate);
    popup.style.display='block';
    try{
      const apiData = await fetch(apiUrl).then(r=>{if(!r.ok) throw new Error('API'); return r.json();});
      const apiTimes = apiData.hourly.time;
      let closestIndex=0, minDiff=Math.abs(new Date(apiTimes[0]).getTime()-new Date(tsStr).getTime());
      for(let i=1;i<apiTimes.length;i++){
        const d=Math.abs(new Date(apiTimes[i]).getTime()-new Date(tsStr).getTime());
        if(d<minDiff){minDiff=d;closestIndex=i;}
      }
      const srcVals={};
      variables.forEach(v=>{ if(apiData.hourly[v]) srcVals[v]=apiData.hourly[v][closestIndex]; });
      srcVals.elevation = apiData.elevation;
      await showResidualPopup(lat,lon,srcVals,'API',tsStr,nearest,evt.coordinate);
    }catch(e){
      popupContent.textContent='Failed to fetch weather data';
    }
    return;
  }

  const {p,timestampIdx} = feature.get('data');
  const srcVals={};
  variables.forEach(v=>{ srcVals[v]=p.w[v]?.[timestampIdx]; });
  srcVals.elevation = p.info?.elevation;
  await showResidualPopup(lat,lon,srcVals,'JSON',tsStr,nearest,evt.coordinate);
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