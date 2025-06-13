// Constants
const DB_NAME = 'weatherAppDB';
const DB_VERSION = 1;
const ELEVATION_STORE_NAME = 'ElevationStore';
const RAW_WEATHER_STORE_NAME = 'RawWeatherStore';
const DERIVED_STORE_NAME = 'DerivedStore';

let db;

// openDB function
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      db = event.target.result;
      console.log('Upgrading database...');

      if (!db.objectStoreNames.contains(ELEVATION_STORE_NAME)) {
        db.createObjectStore(ELEVATION_STORE_NAME, { keyPath: 'id' });
        console.log(`${ELEVATION_STORE_NAME} created.`);
      }
      if (!db.objectStoreNames.contains(RAW_WEATHER_STORE_NAME)) {
        db.createObjectStore(RAW_WEATHER_STORE_NAME, { keyPath: 'id' });
        console.log(`${RAW_WEATHER_STORE_NAME} created.`);
      }
      if (!db.objectStoreNames.contains(DERIVED_STORE_NAME)) {
        db.createObjectStore(DERIVED_STORE_NAME, { keyPath: 'id' });
        console.log(`${DERIVED_STORE_NAME} created.`);
      }
    };

    request.onsuccess = (event) => {
      db = event.target.result;
      console.log('Database opened successfully.');
      resolve(db);
    };

    request.onerror = (event) => {
      console.error('Error opening database:', event.target.error);
      reject(event.target.error);
    };
  });
}

// Generic CRUD operations
function setItem(storeName, item) {
  return new Promise((resolve, reject) => {
    if (!db) {
      console.error('Database not initialized. Call openDB() first.');
      return reject('Database not initialized.');
    }
    const transaction = db.transaction([storeName], 'readwrite');
    const store = transaction.objectStore(storeName);
    const request = store.put(item);

    request.onsuccess = () => {
      console.log(`Item added to ${storeName}:`, item);
      resolve(request.result);
    };

    request.onerror = (event) => {
      console.error(`Error adding item to ${storeName}:`, event.target.error);
      reject(event.target.error);
    };
  });
}

function getItem(storeName, id) {
  return new Promise((resolve, reject) => {
    if (!db) {
      console.error('Database not initialized. Call openDB() first.');
      return reject('Database not initialized.');
    }
    const transaction = db.transaction([storeName], 'readonly');
    const store = transaction.objectStore(storeName);
    const request = store.get(id);

    request.onsuccess = () => {
      console.log(`Item retrieved from ${storeName} with id ${id}:`, request.result);
      resolve(request.result);
    };

    request.onerror = (event) => {
      console.error(`Error getting item from ${storeName} with id ${id}:`, event.target.error);
      reject(event.target.error);
    };
  });
}

function deleteItem(storeName, id) {
  return new Promise((resolve, reject) => {
    if (!db) {
      console.error('Database not initialized. Call openDB() first.');
      return reject('Database not initialized.');
    }
    const transaction = db.transaction([storeName], 'readwrite');
    const store = transaction.objectStore(storeName);
    const request = store.delete(id);

    request.onsuccess = () => {
      console.log(`Item deleted from ${storeName} with id ${id}.`);
      resolve(request.result);
    };

    request.onerror = (event) => {
      console.error(`Error deleting item from ${storeName} with id ${id}:`, event.target.error);
      reject(event.target.error);
    };
  });
}

function getAllItems(storeName) {
  return new Promise((resolve, reject) => {
    if (!db) {
      console.error('Database not initialized. Call openDB() first.');
      return reject('Database not initialized.');
    }
    const transaction = db.transaction([storeName], 'readonly');
    const store = transaction.objectStore(storeName);
    const request = store.getAll();

    request.onsuccess = () => {
      console.log(`All items retrieved from ${storeName}:`, request.result);
      resolve(request.result);
    };

    request.onerror = (event) => {
      console.error(`Error getting all items from ${storeName}:`, event.target.error);
      reject(event.target.error);
    };
  });
}

function clearStore(storeName) {
  return new Promise((resolve, reject) => {
    if (!db) {
      console.error('Database not initialized. Call openDB() first.');
      return reject('Database not initialized.');
    }
    const transaction = db.transaction([storeName], 'readwrite');
    const store = transaction.objectStore(storeName);
    const request = store.clear();

    request.onsuccess = () => {
      console.log(`${storeName} cleared.`);
      resolve(request.result);
    };

    request.onerror = (event) => {
      console.error(`Error clearing ${storeName}:`, event.target.error);
      reject(event.target.error);
    };
  });
}

// Specific exported functions
// Elevation Store
function getElevationData(gridCellId) {
  return getItem(ELEVATION_STORE_NAME, gridCellId);
}

function addElevationData(data) {
  return setItem(ELEVATION_STORE_NAME, data);
}

function clearElevationStore() {
  return clearStore(ELEVATION_STORE_NAME);
}

// Raw Weather Store
function getRawWeatherData(id) {
  return getItem(RAW_WEATHER_STORE_NAME, id);
}

function addRawWeatherData(data) {
  return setItem(RAW_WEATHER_STORE_NAME, data);
}

function clearRawWeatherStore() {
  return clearStore(RAW_WEATHER_STORE_NAME);
}

// Derived Store
function getDerivedData(id) {
  return getItem(DERIVED_STORE_NAME, id);
}

function addDerivedData(data) {
  return setItem(DERIVED_STORE_NAME, data);
}

function clearDerivedStore() {
  return clearStore(DERIVED_STORE_NAME);
}

// Function to count items in DerivedStore within a lat/lon bounding box
function countDerivedItemsInView(minLat, maxLat, minLon, maxLon) {
  return new Promise((resolve, reject) => {
    if (!db) {
      console.error('Database not initialized. Call openDB() first.');
      return reject('Database not initialized.');
    }
    const transaction = db.transaction([DERIVED_STORE_NAME], 'readonly');
    const store = transaction.objectStore(DERIVED_STORE_NAME);
    const request = store.openCursor();
    let count = 0;

    request.onsuccess = (event) => {
      const cursor = event.target.result;
      if (cursor) {
        const item = cursor.value;
        // Check if the item has lat and lon properties and falls within the bounds
        if (item && typeof item.lat === 'number' && typeof item.lon === 'number' &&
            item.lat >= minLat && item.lat <= maxLat &&
            item.lon >= minLon && item.lon <= maxLon) {
          count++;
        }
        cursor.continue();
      } else {
        // No more entries
        console.log(`Counted ${count} items in ${DERIVED_STORE_NAME} within view.`);
        resolve(count);
      }
    };

    request.onerror = (event) => {
      console.error(`Error counting items in ${DERIVED_STORE_NAME}:`, event.target.error);
      reject(event.target.error);
    };
  });
}


// Export functions to be used by other modules
export {
  openDB,
  getElevationData,
  addElevationData,
  clearElevationStore,
  getRawWeatherData,
  addRawWeatherData,
  clearRawWeatherStore,
  getDerivedData,
  addDerivedData,
  clearDerivedStore,
  countDerivedItemsInView, // Export the new function
  // Export generic functions if they need to be used directly
  setItem,
  getItem,
  deleteItem,
  getAllItems,
  clearStore
};
