// colorManager.js

// For SQH/Skiability layers
export const COLOR_LIGHT_BLUE = [173, 216, 230];
export const COLOR_INDIGO = [75, 0, 130];
export const COLOR_HOT_PINK = [255, 105, 180];

// General Color Scales
export const THERMAL_SCALE = [[0,0,255], [255,255,0], [255,0,0]]; // Blue-Yellow-Red (e.g., for Temperature)
export const SNOW_SCALE = [[200,200,255], [0,0,150]];          // Light Blue to Dark Blue (e.g., for Snowfall, Snow Depth)
export const HUMIDITY_SCALE = [[224,224,224], [0,0,255]];       // Grey to Blue (e.g. for relative humidity)
export const WIND_SPEED_SCALE = [[0,255,0], [255,255,0], [255,0,0]]; // Green-Yellow-Red
export const PRESSURE_SCALE = [[255,0,255], [0,255,255]];       // Magenta to Cyan
export const ELEVATION_SCALE = [[0,128,0], [139,69,19], [255,255,255]]; // Green-Brown-White


// Specific Color Maps
export const ASPECT_COLORS = {                                 // For Aspect
  'north': [255,0,0], 'east': [0,255,0], 'south': [0,0,255], 'west': [255,255,0],
  'northeast': [255,165,0], 'southeast': [0,128,128], 
  'southwest': [128,0,128], 'northwest': [255,192,203], 
  'flat': [128,128,128], 'default': [128,128,128]
};

export const SLOPE_CATEGORY_COLORS = { // As per user requirements - 7 categories
    '0-15': [0, 0, 255],      // Blue (flat/boring)
    '15-20': [64, 224, 208],   // Blue-green (e.g., Turquoise)
    '20-30': [0, 255, 0],      // Green (sweet spot)
    '30-35': [173, 255, 47],   // Yellow-green
    '35-45': [255, 165, 0],    // Orange (steep, advanced, avalanche-prone)
    '45-50': [255, 0, 0],      // Red (very steep)
    '>50': [139, 0, 0]       // Dark red (cliff/unskiable)
};

export const SLOPE_GRADIENT_SCALE = [[0, 255, 0], [255, 255, 0], [255, 0, 0]]; // Green-Yellow-Red for continuous slope


// Helper functions
export function lerp(a, b, t) {
    return a + (b - a) * t;
}

// Generic interpolation for scales like THERMAL_SCALE
export function interpolateColorLinear(value, minVal, maxVal, colorScale) {
    if (value == null) return [128, 128, 128]; // Default for no data
    
    // Clamp value to minVal/maxVal to handle out-of-range inputs gracefully
    const clampedValue = Math.max(minVal, Math.min(maxVal, value));
    const t = (clampedValue - minVal) / (maxVal - minVal);

    const segments = colorScale.length - 1;
    if (segments < 1) {
        return colorScale[0] || [0,0,0]; // Should not happen with valid scales like THERMAL_SCALE
    }
    
    // Ensure t is within [0, 1] even with potential floating point inaccuracies
    const clampedT = Math.max(0, Math.min(1, t)); 
    
    const segmentIndex = Math.floor(clampedT * segments);
    
    // Ensure segment index is within bounds [0, segments - 1]
    const safeSegment = Math.max(0, Math.min(segmentIndex, segments - 1));

    const segmentT = (clampedT * segments) - safeSegment;

    const c1 = colorScale[safeSegment];
    // Ensure c2 is the last color if safeSegment is already the last one
    const c2 = colorScale[Math.min(safeSegment + 1, segments)];


    return [
        Math.round(lerp(c1[0], c2[0], segmentT)),
        Math.round(lerp(c1[1], c2[1], segmentT)),
        Math.round(lerp(c1[2], c2[2], segmentT))
    ];
}

// Renamed original function to be used by main.js as a fallback for feature-based scaling
export function calculateDynamicScaleFromFeatures(featuresArray, variableName, options = {}) {
    let min = Infinity;
    let max = -Infinity;
    let validDataFound = false;

    if (!featuresArray || featuresArray.length === 0) {
        // Return a very basic default if no features to analyze
        return { min: 0, max: 1, calculated: false }; 
    }

    for (const feature of featuresArray) {
        const value = feature.get(variableName);
        if (value != null && typeof value === 'number' && !isNaN(value)) {
            min = Math.min(min, value);
            max = Math.max(max, value);
            validDataFound = true;
        }
    }

    if (!validDataFound) {
        // console.warn(`No valid data found for ${variableName} to calculate dynamic scale.`);
        return { min: 0, max: 1, calculated: false }; // Fallback if no numeric data
    }
    
    // Handle case where all values are the same
    if (min === max) {
        min = min - 0.5; // Avoid min === max for scaling, create a small range
        max = max + 0.5;
        if (min === 0 && max === 0) { // Special case for all zeros
            max = 1;
        }
    }

    // Placeholder for future options like percentile cutoffs
    // if (options.usePercentiles) { /* ... calculate percentiles ... */ }

    return { min, max, calculated: true };
}


// New function for global dynamic scaling from any store.
// variableName: e.g., 'elevation', 'finalScore', 'recentAccumCm', or 'weatherData.temperature_2m'
// dbService: imported dbService object (expected to have getAllItems method)
// storeName: e.g., 'DerivedStore', 'ElevationStore'
// options: { path: 'path.to.variable' } (optional, if variableName itself is not the direct key or path)
export async function getDynamicScaleGlobal(variableName, dbService, storeName, options = {}) {
    if (!dbService) {
        console.warn(`getDynamicScaleGlobal: dbService instance not provided. Cannot fetch global scale for ${variableName} from ${storeName}.`);
        return { calculated: false };
    }
    if (!storeName) {
        console.warn(`getDynamicScaleGlobal: storeName not provided. Cannot fetch global scale for ${variableName}.`);
        return { calculated: false };
    }

    let min = Infinity;
    let max = -Infinity;
    let validDataFound = false;

    try {
        const allItems = await dbService.getAllItems(storeName);

        if (!allItems || allItems.length === 0) {
            console.warn(`No items found in ${storeName} for dynamic scaling of ${variableName}.`);
            return { min: 0, max: 1, calculated: false };
        }

        for (const item of allItems) {
            let value;
            // Use options.path if provided, otherwise variableName is the direct key or path.
            const path = options.path || variableName;
            let value;

            if (path.includes('.')) {
                const parts = path.split('.');
                let currentVal = item;
                for (const part of parts) {
                    if (currentVal && typeof currentVal === 'object' && part in currentVal) {
                        currentVal = currentVal[part];
                    } else {
                        currentVal = undefined;
                        break;
                    }
                }
                value = currentVal;
            } else {
                value = item[path]; // If no path, variableName is the key
            }
            
            if (value != null && typeof value === 'number' && !isNaN(value)) {
                min = Math.min(min, value);
                max = Math.max(max, value);
                validDataFound = true;
            }
        }

        if (!validDataFound) {
        // console.warn(`No valid numeric data found for ${path} in ${storeName} for dynamic scaling.`);
        return { calculated: false };
        }

        if (min === max) {
            min = min - 0.5; // Avoid min === max for scaling
            max = max + 0.5;
            if (min === 0 && max === 0) max = 1; // Special case for all zeros
        }
        
        // console.log(`Global dynamic scale for ${variableName} (from ${storeName}): min=${min.toFixed(2)}, max=${max.toFixed(2)}`);
        return { min, max, calculated: true };

    } catch (error) {
        console.error(`Error fetching or processing data from ${storeName} for dynamic scaling of ${variableName}:`, error);
        return { min: 0, max: 1, calculated: false };
    }
}
