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

let times = [], points = [], varName = 'temperature_2m';
let dayIdx = 0, hourIdx = 0;
const dayBtn = document.getElementById('day-control-button');
const timeBtn = document.getElementById('time-control-button');

fetch('resources/meteo_api/weather_data_3hour.json').then(r => r.json()).then(d => {
  const sample = d.coordinates.find(c => c.weather_data_3hour);
  if (!sample) return;
  times = sample.weather_data_3hour.hourly.time.slice(0, 40).map(t => new Date(t));
  points = d.coordinates.filter(c => c.weather_data_3hour).map(c => ({
    lat: c.coordinate_info.latitude ?? c.latitude,
    lon: c.coordinate_info.longitude ?? c.longitude,
    w: c.weather_data_3hour.hourly
  }));
  updateButtons();
  draw();
}).catch(e => console.error('Weather data load failed', e));

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
  const idx = dayIdx*8 + hourIdx;
  if(!times[idx]) return;
  const values = [];
  const feats = points.map(p=>{
    const v = p.w[varName]?.[idx];
    values.push(v);
    const f = new ol.Feature(new ol.geom.Point(ol.proj.fromLonLat([p.lon,p.lat])));
    f.set('v', v);
    return f;
  });
  const nums = values.filter(v=>typeof v==='number');
  const min = Math.min(...nums);
  const max = Math.max(...nums);
  src.clear();
  feats.forEach(f=>{
    const v=f.get('v');
    if(typeof v!=='number') return;
    f.setStyle(new ol.style.Style({image:new ol.style.Circle({radius:6,fill:new ol.style.Fill({color:color(v,min,max,varName)})})}));
    src.addFeature(f);
  });
}

function updateButtons(){
  const idx = dayIdx*8 + hourIdx;
  const t = times[idx];
  if(!t) return;
  dayBtn.textContent = t.toISOString().split('T')[0];
  timeBtn.textContent = t.toISOString().split('T')[1].slice(0,5);
}

dayBtn.onclick=()=>{dayIdx=(dayIdx+1)%Math.min(5,Math.floor(times.length/8));updateButtons();draw();};
timeBtn.onclick=()=>{hourIdx=(hourIdx+1)%8;updateButtons();draw();};

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

// Drawer toggle functionality
window.toggleDrawer = function() {
  document.getElementById('drawer-container').classList.toggle('open');
};
