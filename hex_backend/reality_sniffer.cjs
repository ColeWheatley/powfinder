const { chromium } = require('playwright');

(async () => {
  console.log("Starting Sniffer...");
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  // Sniff network requests
  page.on('request', request => {
    const url = request.url();
    // Look for common tile patterns or realitymaps domains
    if (url.includes('tile') || url.includes('realitymaps') || url.includes('.webp') || url.includes('.jpg')) {
      if (!url.includes('bundle.js') && !url.includes('favicon')) {
        console.log(`[TILE CANDIDATE]: ${url}`);
      }
    }
  });

  try {
    console.log("Navigating to RealityMaps...");
    await page.goto('https://og.realitymaps.de/RealityMaps/', { waitUntil: 'networkidle' });

    // 1. Wait for the app to load
    await page.waitForTimeout(5000); 

    console.log("Attempting to toggle Winter Satellite and 2D mode via JS API...");
    
    // We use the internal API we discovered in the bundle investigation
    await page.evaluate(() => {
        const viewer = window.rm3dApi.viewer;
        
        // 1. Switch to 2D mode
        viewer.cameraView.transitionToSceneMode('2d');
        
        // 2. Try to find the winter layer and enable it
        // From bundle: mapStyle.props.mainImagery might be the key
        // Or we toggle the specific layer in mapStyle
        const layers = viewer.mapStyle.getLayers();
        const winterLayer = layers.find(l => l.name.toLowerCase().includes('winter') || l.id?.toLowerCase().includes('winter'));
        
        if (winterLayer) {
            winterLayer.apply(true);
            console.log("Winter layer found and applied.");
        } else {
            // Fallback: try setting props
            viewer.mapStyle.props.mainImagery = 'winter'; 
            viewer.mapStyle.props.slopes = false; // Kill slopes while we are at it
        }
    });

    console.log("Waiting for tiles to load (10s)...");
    await page.waitForTimeout(10000);

    console.log("Sniffing complete.");
  } catch (err) {
    console.error("Error during sniffing:", err);
  } finally {
    await browser.close();
  }
})();
