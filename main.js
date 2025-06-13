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
const popup = document.getElementById('popup');
const popupContent = document.getElementById('popup-content');
const overlay = new ol.Overlay({ element: popup, positioning: 'bottom-center', stopEvent: false });
map.addOverlay(overlay);

let times = [], points = [], varName = 'temperature_2m';
let variables = [];
const hourOffsets = [3,4,5,6]; // 9am, noon, 3pm, 6pm
let dayIdx = 0, hourIdx = 0;
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
fetch('resources/meteo_api/tirol_peaks.geojson')
  .then(r => r.json())
  .then(g => {
    peaks = g.features.map(f => ({
      name: f.properties.name,
      ele: f.properties.ele,
      lat: f.geometry.coordinates[1],
      lon: f.geometry.coordinates[0]
    }));
  }).catch(() => {});

fetch('resources/meteo_api/weather_data_3hour.json').then(r => r.json()).then(d => {
  const sample = d.coordinates.find(c => c.weather_data_3hour);
  if (!sample) return;
  times = sample.weather_data_3hour.hourly.time.slice(0, 40).map(t => new Date(t));
  variables = Object.keys(sample.weather_data_3hour.hourly)
    .filter(k => k !== 'time' && k !== 'time_units');
  points = d.coordinates.filter(c => c.weather_data_3hour).map(c => ({
    lat: c.coordinate_info.latitude ?? c.latitude,
    lon: c.coordinate_info.longitude ?? c.longitude,
    info: c.coordinate_info || {},
    w: c.weather_data_3hour.hourly
  }));
  updateButtons();
  draw();
}).catch(e => console.error('Weather data load failed', e));

function formatDay(d){
  const today = times[0];
  const diff = Math.round((d - today) / 86400000);
  if(diff === 0) return 'Today';
  if(diff === 1) return 'Tomorrow';
  return d.toLocaleDateString('en-US',{weekday:'long'});
}

function formatTime(d){
  const h = d.getHours();
  const period = h >= 12 ? 'pm' : 'am';
  const hour = ((h + 11) % 12) + 1;
  return `${hour}${period}`;
}

function color(val, min, max, varName){
  // Special handling for weather_code (WMO codes)
  if(varName === 'weather_code') {
    // WMO weather codes: 0=clear, 1-3=partly cloudy, 45-48=fog, 51-67=rain, 71-77=snow, 80-99=storms
    if(val <= 3) return '#FFD700'; // Clear/partly cloudy - yellow/gold
    if(val <= 48) return '#808080'; // Fog - gray
    if(val <= 67) return '#4169E1'; // Rain - blue
    if(val <= 77) return '#FFFFFF'; // Snow - white
    return '#FF4500'; // Storms - red-orange
  }
  
  // Normal gradient for other variables
  const t = Math.max(0, Math.min(1, (val - min)/(max-min)));
  const r = Math.round(255*t);
  const b = Math.round(255*(1-t));
  return `rgb(${r},0,${b})`;
}

function draw(){
  const idx = dayIdx*8 + hourOffsets[hourIdx];
  if(!times[idx]) return;
  const values = [];
  const feats = points.map(p=>{
    const v = p.w[varName]?.[idx];
    values.push(v);
    const f = new ol.Feature(new ol.geom.Point(ol.proj.fromLonLat([p.lon,p.lat])));
    f.set('v', v);
    f.set('data', {p, idx});
    return f;
  });
  const nums = values.filter(v=>typeof v==='number');
  const min = Math.min(...nums);
  const max = Math.max(...nums);
  currentMin = min; currentMax = max;
  src.clear();
  feats.forEach(f=>{
    const v=f.get('v');
    if(typeof v!=='number') return;
    f.setStyle(new ol.style.Style({image:new ol.style.Circle({radius:6,fill:new ol.style.Fill({color:color(v,min,max,varName)})})}));
    src.addFeature(f);
  });
  showLayerInfoBox();
}

function updateButtons(){
  const idx = dayIdx*8 + hourOffsets[hourIdx];
  const t = times[idx];
  if(!t) return;
  dayBtn.textContent = formatDay(t);
  timeBtn.textContent = formatTime(t);
}

function showLayerInfoBox(){
  const info = document.getElementById('info-box');
  const idx = dayIdx*8 + hourOffsets[hourIdx];
  const t = times[idx];
  if(!info || !t) return;
  info.classList.remove('info-box-selecting');
  const barStyle = `background:linear-gradient(to right,rgb(0,0,255),rgb(255,0,0))`;
  const label = `${varLabels[varName] ?? varName} ${formatDay(t)} at ${formatTime(t)}`;
  const unit = varUnits[varName] ?? '';
  info.innerHTML =
    `<div class="info-label">${label}</div>`+
    `<div class="legend"><span>${currentMin.toFixed(1)}${unit}</span>`+
    `<div class="legend-bar" style="${barStyle}"></div>`+
    `<span>${currentMax.toFixed(1)}${unit}</span></div>`;
  info.style.display = 'block';
}

function showDaySelector(){
  const info = document.getElementById('info-box');
  info.innerHTML='';
  info.classList.add('info-box-selecting');
  const days = [];
  for(let i=0;i<Math.min(4,Math.floor(times.length/8));i++){
    const d=times[i*8];
    days.push({label:formatDay(d),idx:i});
  }
  days.forEach(d=>{
    const div=document.createElement('div');
    div.className='layer-item';
    if(d.idx===dayIdx) div.classList.add('active');
    div.textContent=d.label;
    div.onclick=()=>{dayIdx=d.idx;updateButtons();draw();};
    info.appendChild(div);
  });
  info.style.display='block';
}

function showTimeSelector(){
  const info = document.getElementById('info-box');
  info.innerHTML='';
  info.classList.add('info-box-selecting');
  for(let i=0;i<4;i++){
    const d=times[dayIdx*8 + hourOffsets[i]];
    if(!d) continue;
    const label=formatTime(d);
    const div=document.createElement('div');
    div.className='layer-item';
    if(i===hourIdx) div.classList.add('active');
    div.textContent=label;
    div.onclick=()=>{hourIdx=i;updateButtons();draw();};
    info.appendChild(div);
  }
  info.style.display='block';
}

dayBtn.onclick=()=>{showDaySelector();};
timeBtn.onclick=()=>{showTimeSelector();};

// Toggle button functionality
let isPointMode = true; // Start in point mode
const toggleBtn = document.getElementById('mode-toggle');

toggleBtn.onclick = () => {
  isPointMode = !isPointMode;
  if (isPointMode) {
    toggleBtn.classList.remove('smooth');
  } else {
    toggleBtn.classList.add('smooth');
  }
  // TODO: Implement smooth/interpolated visualization mode
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
  if(!feature){
    popupContent.textContent='No data here';
    overlay.setPosition(evt.coordinate);
    popup.style.display='block';
    return;
  }
  const data = feature.get('data');
  if(!data){
    popupContent.textContent='No data here';
    overlay.setPosition(evt.coordinate);
    popup.style.display='block';
    return;
  }
  const {p,idx} = data;
  const lines=[];
  if(p.info && p.info.type){
    if(p.info.type==='peak'){
      const peak=findPeak(p.lat,p.lon);
      if(peak){
        lines.push(`Peak: ${peak.name} (${peak.ele}m)`);
      }else{
        lines.push('Peak point');
      }
    }else{
      lines.push('Random point');
    }
  }
  variables.forEach(v=>{
    const val=p.w[v]?.[idx];
    if(val!=null) lines.push(`${v}: ${val}`);
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
