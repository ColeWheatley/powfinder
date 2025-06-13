import 'https://cdn.jsdelivr.net/npm/ol@v7.4.0/dist/ol.js';

const map = new ol.Map({
  target: 'map',
  layers: [new ol.layer.Tile({ source: new ol.source.OSM() })],
  view: new ol.View({
    center: ol.proj.fromLonLat([11.3, 47.2]),
    zoom: 8
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

function color(val, min, max){
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
    f.setStyle(new ol.style.Style({image:new ol.style.Circle({radius:3,fill:new ol.style.Fill({color:color(v,min,max)})})}));
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

document.querySelectorAll('.layer-item[data-layer-type="weather"]').forEach(btn=>{
  if(btn.dataset.layerName===varName) btn.classList.add('active');
  btn.onclick=()=>{
    document.querySelectorAll('.layer-item[data-layer-type="weather"]').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    varName=btn.dataset.layerName;draw();
  };
});
