#!/usr/bin/env node

/**
 * weatherAPI.js
 *
 * Fetch raw weather data every 250 m from Innsbruck eastward for 7 km,
 * using Open-Meteo’s ICON-D2 model.  
 * Dumps each full JSON response (per location) into `raw_responses.txt`  
 * with a 150 ms delay between requests to avoid rate limits.
 *
 * Usage:
 *   node weatherAPI.js
 *
 * Assumes Node.js v18+ (built-in fetch). If using an older Node, install `node-fetch`:
 *   npm install node-fetch
 */

import fs from 'fs/promises';

// (If you're on Node < 18, uncomment the following two lines and run `npm install node-fetch`)
// import fetch from 'node-fetch';
// global.fetch = fetch;

const API_BASE = 'https://api.open-meteo.com/v1/forecast';

// =======================================
//  Configuration
// =======================================

// Innsbruck center coordinate (approximate)
const BASE_LAT = 47.2692;
const BASE_LON = 11.4041;

// Eastward extent: 7 km
const DISTANCE_EAST_KM = 7;

// Sampling interval: 250 m = 0.25 km
const SAMPLE_INTERVAL_KM = 0.25;

// Convert kilometers to degrees longitude at this latitude
// 1° lon ≈ 111.32 km × cos(lat)
const KM_PER_DEG_LON = 111.32 * Math.cos(BASE_LAT * Math.PI / 180);
const DELTA_LON = SAMPLE_INTERVAL_KM / KM_PER_DEG_LON;
const TOTAL_STEPS = Math.round(DISTANCE_EAST_KM / SAMPLE_INTERVAL_KM);

// New choice of variables (including a high-altitude pressure level)
const HOURLY_VARS = [
  'temperature_2m',
  'temperature_300hPa',
  'snowfall',
  'snow_depth',
  'dewpoint_2m',
  'relativehumidity_2m',
  'windspeed_10m',
  'shortwave_radiation'
].join(',');

// Time window: 14 days past + 4 days forecast
const PAST_DAYS = 14;
const FORECAST_DAYS = 4;

// Timezone
const TZ = 'Europe%2FVienna';

// Output file (overwritten each run)
const OUTPUT_FILE = 'raw_responses.txt';

// Delay between calls (ms)
const DELAY_MS = 150;

// =======================================
//  Helper Functions
// =======================================

/**
 * Sleep for the specified number of milliseconds.
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Build API URL for given lat/lon.
 */
function buildUrl(lat, lon) {
  return (
    `${API_BASE}` +
    `?latitude=${lat.toFixed(5)}` +
    `&longitude=${lon.toFixed(5)}` +
    `&hourly=${HOURLY_VARS}` +
    `&past_days=${PAST_DAYS}&forecast_days=${FORECAST_DAYS}` +
    `&model=icon-d2` +
    `&timezone=${TZ}`
  );
}

/**
 * Append a JSON object as a line to OUTPUT_FILE.
 */
async function appendResponse(lat, lon, jsonObj) {
  const entry = {
    timestamp: new Date().toISOString(),
    lat: lat,
    lon: lon,
    model: jsonObj.generationtime_ms !== undefined
      ? (jsonObj.model_run || 'unknown')
      : 'unknown',
    response: jsonObj
  };
  const line = JSON.stringify(entry) + '\n';
  await fs.appendFile(OUTPUT_FILE, line, 'utf8');
}

// =======================================
//  Main Fetch Loop
// =======================================

async function main() {
  // Overwrite existing output file
  await fs.writeFile(OUTPUT_FILE, '', 'utf8');
  console.log(`⬇️  Starting bulk fetch → writing to ${OUTPUT_FILE}`);

  for (let i = 0; i <= TOTAL_STEPS; i++) {
    const lon = BASE_LON + i * DELTA_LON;
    const lat = BASE_LAT;

    const url = buildUrl(lat, lon);
    console.log(`→ [${i}/${TOTAL_STEPS}] Requesting: lat=${lat.toFixed(5)}, lon=${lon.toFixed(5)}`);

    try {
      const res = await fetch(url);
      if (!res.ok) {
        console.error(`  ❌ HTTP ${res.status} ${res.statusText} for (${lat.toFixed(5)},${lon.toFixed(5)})`);
      } else {
        const data = await res.json();
        await appendResponse(lat, lon, data);
        console.log(`  ✅ Saved response for (${lat.toFixed(5)},${lon.toFixed(5)})`);
      }
    } catch (err) {
      console.error(`  ❌ Fetch error for (${lat.toFixed(5)},${lon.toFixed(5)}):`, err);
    }

    // Stagger next request
    if (i < TOTAL_STEPS) await sleep(DELAY_MS);
  }

  console.log('✅ Bulk fetch complete.');
}

main().catch(err => {
  console.error('Fatal error in bulk fetch:', err);
  process.exit(1);
});