// gridService.js - Unified Grid Index for persistent storage
// All terrain, weather, and derived data in a single store

const DB_NAME = 'powFinderGridDB';
const DB_VERSION = 1;
const GRID_STORE_NAME = 'GridStore';

let db;

// Grid cell structure
function createGridCell(lat, lon) {
  return {
    id: `${lat.toFixed(5)},${lon.toFixed(5)}`,
    lat: lat,
    lon: lon,
    
    // Terrain data
    terrain: {
      elevation: null,
      slope: null,
      aspect: null,
      shadow: null,
      fetchedAt: null
    },
    
    // Weather data
    rawWeather: {
      data: null,
      fetchedAt: null,
      source: null // 'api' or 'extrapolated'
    },
    
    extrapolatedWeather: {
      data: null,
      computedAt: null,
      sourcePoints: []
    },
    
    // Derived metrics
    derived: {
      snowHeight: null,
      snowQuality: null,
      skiability: null,
      computedAt: null,
      targetDate: null
    },
    
    // Metadata
    isPeak: false,
    isAnchor: false,
    lastUpdated: Date.now()
  };
}

// Open database
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      db = event.target.result;
      console.log('Creating GridStore...');

      if (!db.objectStoreNames.contains(GRID_STORE_NAME)) {
        const store = db.createObjectStore(GRID_STORE_NAME, { keyPath: 'id' });
        
        // Create indices for efficient queries
        store.createIndex('lat', 'lat', { unique: false });
        store.createIndex('lon', 'lon', { unique: false });
        store.createIndex('isPeak', 'isPeak', { unique: false });
        store.createIndex('isAnchor', 'isAnchor', { unique: false });
        
        console.log(`${GRID_STORE_NAME} created with indices.`);
      }
    };

    request.onsuccess = (event) => {
      db = event.target.result;
      console.log('Grid database opened successfully.');
      resolve(db);
    };

    request.onerror = (event) => {
      console.error('Error opening grid database:', event.target.error);
      reject(event.target.error);
    };
  });
}

// Get or create grid cell
async function getOrCreateGridCell(lat, lon) {
  const id = `${lat.toFixed(5)},${lon.toFixed(5)}`;
  
  try {
    let cell = await getGridCell(id);
    if (!cell) {
      cell = createGridCell(lat, lon);
      await setGridCell(cell);
    }
    return cell;
  } catch (error) {
    console.error(`Error getting/creating grid cell ${id}:`, error);
    return null;
  }
}

// Basic CRUD operations
function setGridCell(cell) {
  return new Promise((resolve, reject) => {
    if (!db) {
      return reject('Database not initialized. Call openDB() first.');
    }
    
    cell.lastUpdated = Date.now();
    
    const transaction = db.transaction([GRID_STORE_NAME], 'readwrite');
    const store = transaction.objectStore(GRID_STORE_NAME);
    const request = store.put(cell);

    request.onsuccess = () => {
      resolve(request.result);
    };

    request.onerror = (event) => {
      console.error(`Error saving grid cell ${cell.id}:`, event.target.error);
      reject(event.target.error);
    };
  });
}

function getGridCell(id) {
  return new Promise((resolve, reject) => {
    if (!db) {
      return reject('Database not initialized. Call openDB() first.');
    }
    
    const transaction = db.transaction([GRID_STORE_NAME], 'readonly');
    const store = transaction.objectStore(GRID_STORE_NAME);
    const request = store.get(id);

    request.onsuccess = () => {
      resolve(request.result);
    };

    request.onerror = (event) => {
      console.error(`Error getting grid cell ${id}:`, event.target.error);
      reject(event.target.error);
    };
  });
}

// Get cells within bounds
function getCellsInBounds(minLat, maxLat, minLon, maxLon) {
  return new Promise((resolve, reject) => {
    if (!db) {
      return reject('Database not initialized. Call openDB() first.');
    }
    
    const transaction = db.transaction([GRID_STORE_NAME], 'readonly');
    const store = transaction.objectStore(GRID_STORE_NAME);
    const cells = [];
    
    const request = store.openCursor();
    
    request.onsuccess = (event) => {
      const cursor = event.target.result;
      if (cursor) {
        const cell = cursor.value;
        if (cell.lat >= minLat && cell.lat <= maxLat &&
            cell.lon >= minLon && cell.lon <= maxLon) {
          cells.push(cell);
        }
        cursor.continue();
      } else {
        resolve(cells);
      }
    };

    request.onerror = (event) => {
      console.error('Error getting cells in bounds:', event.target.error);
      reject(event.target.error);
    };
  });
}

// Get cells by type
function getCellsByType(type) {
  return new Promise((resolve, reject) => {
    if (!db) {
      return reject('Database not initialized. Call openDB() first.');
    }
    
    const transaction = db.transaction([GRID_STORE_NAME], 'readonly');
    const store = transaction.objectStore(GRID_STORE_NAME);
    const index = store.index(type);
    const cells = [];
    
    const request = index.openCursor(IDBKeyRange.only(true));
    
    request.onsuccess = (event) => {
      const cursor = event.target.result;
      if (cursor) {
        cells.push(cursor.value);
        cursor.continue();
      } else {
        resolve(cells);
      }
    };

    request.onerror = (event) => {
      console.error(`Error getting cells by type ${type}:`, event.target.error);
      reject(event.target.error);
    };
  });
}

// Update specific fields
async function updateCellTerrain(id, terrainData) {
  const cell = await getGridCell(id);
  if (!cell) return null;
  
  cell.terrain = {
    ...cell.terrain,
    ...terrainData,
    fetchedAt: Date.now()
  };
  
  return setGridCell(cell);
}

async function updateCellWeather(id, weatherData, source = 'api') {
  const cell = await getGridCell(id);
  if (!cell) return null;
  
  if (source === 'api') {
    cell.rawWeather = {
      data: weatherData,
      fetchedAt: Date.now(),
      source: 'api'
    };
  } else {
    cell.extrapolatedWeather = {
      data: weatherData.data,
      computedAt: Date.now(),
      sourcePoints: weatherData.sourcePoints || []
    };
  }
  
  return setGridCell(cell);
}

async function updateCellDerived(id, derivedData, targetDate) {
  const cell = await getGridCell(id);
  if (!cell) return null;
  
  cell.derived = {
    ...derivedData,
    computedAt: Date.now(),
    targetDate: targetDate
  };
  
  return setGridCell(cell);
}

// Clear functions
function clearAllCells() {
  return new Promise((resolve, reject) => {
    if (!db) {
      return reject('Database not initialized. Call openDB() first.');
    }
    
    const transaction = db.transaction([GRID_STORE_NAME], 'readwrite');
    const store = transaction.objectStore(GRID_STORE_NAME);
    const request = store.clear();

    request.onsuccess = () => {
      console.log('All grid cells cleared.');
      resolve();
    };

    request.onerror = (event) => {
      console.error('Error clearing grid cells:', event.target.error);
      reject(event.target.error);
    };
  });
}

// Count cells with data
function countCellsWithData(minLat, maxLat, minLon, maxLon, dataType) {
  return new Promise((resolve, reject) => {
    if (!db) {
      return reject('Database not initialized. Call openDB() first.');
    }
    
    const transaction = db.transaction([GRID_STORE_NAME], 'readonly');
    const store = transaction.objectStore(GRID_STORE_NAME);
    const request = store.openCursor();
    let count = 0;

    request.onsuccess = (event) => {
      const cursor = event.target.result;
      if (cursor) {
        const cell = cursor.value;
        if (cell.lat >= minLat && cell.lat <= maxLat &&
            cell.lon >= minLon && cell.lon <= maxLon) {
          
          let hasData = false;
          switch(dataType) {
            case 'terrain':
              hasData = cell.terrain.elevation !== null;
              break;
            case 'rawWeather':
              hasData = cell.rawWeather.data !== null;
              break;
            case 'extrapolated':
              hasData = cell.extrapolatedWeather.data !== null;
              break;
            case 'derived':
              hasData = cell.derived.snowQuality !== null;
              break;
            default:
              hasData = true;
          }
          
          if (hasData) count++;
        }
        cursor.continue();
      } else {
        resolve(count);
      }
    };

    request.onerror = (event) => {
      console.error('Error counting cells:', event.target.error);
      reject(event.target.error);
    };
  });
}

// Export functions
export {
  openDB,
  createGridCell,
  getOrCreateGridCell,
  setGridCell,
  getGridCell,
  getCellsInBounds,
  getCellsByType,
  updateCellTerrain,
  updateCellWeather,
  updateCellDerived,
  clearAllCells,
  countCellsWithData
};