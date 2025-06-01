// weatherAPI.js
// Fetch raw weather + elevation data only (–14d .. +14d window)
// Exposes getRawWeather(lat, lon, context) → full JSON response
import { openDB, getRawWeatherData, addRawWeatherData, clearRawWeatherStore } from './dbService.js';

// Initialize DB on module load
let dbInitialized = false;
async function ensureDBInitialized() {
  if (!dbInitialized) {
    await openDB();
    dbInitialized = true;
  }
}

const API_BASE = 'https://api.open-meteo.com/v1/forecast';
const CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour

// 9 hourly vars for snowpack modeling
const HOURLY_VARS = [
  'snowfall',
  'snow_depth',
  'temperature_2m',
  'dewpoint_2m',
  'relativehumidity_2m',
  'windspeed_10m',
  'shortwave_radiation',
  'temperature_850hPa',
  'temperature_500hPa',
  // 'surface_pressure' // Removed as per earlier decision, confirm if needed
].join(',');

// Helper function to stagger requests
function staggerRequest(delayMs) {
  return new Promise(resolve => setTimeout(resolve, delayMs));
}

// Build ID from lat/lon with 5-decimal precision
function makeKey(lat, lon) {
  return `${lat.toFixed(5)},${lon.toFixed(5)}`;
}

/**
 * Fetches and caches raw weather data for a point.
 * @param {number} lat
 * @param {number} lon
 * @param {object} [context={}] - Optional context for logging, e.g., { peakName: 'Everest', isAdjacent: true }
 * @returns {Promise<Object|null>} Full JSON response with .hourly, .elevation, etc., or null on error.
 */
export async function getRawWeather(lat, lon, context = {}) {
  await ensureDBInitialized();
  const id = makeKey(lat, lon);

  try {
    const storedData = await getRawWeatherData(id);
    if (storedData && (Date.now() - storedData.fetchTimestamp) < CACHE_TTL_MS) {
      let logMessage = `☑️ DB Cache hit for ${id}`;
      if (context.peakName) {
        logMessage += ` (Peak: ${context.peakName}${context.isAdjacent ? ', Adjacent' : ''})`;
      }
      console.log(logMessage);
      return storedData.apiResponse;
    }
  } catch (dbError) {
    console.error(`Error accessing IndexedDB for ${id}:`, dbError);
    // Proceed to fetch from network if DB read fails
  }

  await staggerRequest(50); // 50ms stagger

  let fetchLogMessage = `🌐 fetching weather for ${id}`;
  if (context.peakName && context.isAdjacent) {
    fetchLogMessage = `🌐 fetching ADJACENT weather for peak: ${context.peakName} (${id})`;
  } else if (context.peakName) {
    fetchLogMessage = `🌐 fetching weather for peak: ${context.peakName} (${id})`;
  }
  console.log(fetchLogMessage);

  const url = `${API_BASE}`
    + `?latitude=${lat.toFixed(5)}` // Ensure precision matches key
    + `&longitude=${lon.toFixed(5)}` // Ensure precision matches key
    + `&hourly=${HOURLY_VARS}`
    + `&past_days=14&forecast_days=14` // 28-day span
    + `&timezone=Europe%2FVienna`; // Consider making timezone configurable if needed

  try {
    const res = await fetch(url);
    if (!res.ok) {
      console.error(`Weather API error for ${id}: ${res.status} ${res.statusText}`);
      // Potentially return a specific error object or null based on requirements
      return null; 
    }
    const data = await res.json();

    const weatherEntry = {
      id: id,
      apiResponse: data,
      fetchTimestamp: Date.now()
    };

    try {
      await addRawWeatherData(weatherEntry);
    } catch (dbWriteError) {
      console.error(`Error writing raw weather data to DB for ${id}:`, dbWriteError);
      // Still return data as it was successfully fetched, the primary goal of getRawWeather
    }
    return data; // which is weatherEntry.apiResponse
  } catch (fetchError) {
    console.error(`Failed to fetch weather data for ${id}:`, fetchError);
    return null; // Return null or throw, depending on desired error handling
  }
}

export async function clearWeatherAPICache() {
  try {
    await clearRawWeatherStore();
    console.log('WeatherAPI raw weather store cleared via dbService.');
  } catch (error) {
    console.error('Error clearing WeatherAPI raw weather store:', error);
  }
}

// Make clearWeatherAPICache globally accessible
window.clearWeatherAPICache = clearWeatherAPICache;
