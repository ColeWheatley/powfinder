// Loading screen animation sequence
const ANIMATION_DURATION = 1500; // 1.5 seconds total for the loading screen

// Get a random satellite tile from the low_res directory
function getRandomSatelliteTile() {
    // Using a known tile from the tiles_sat/low_res directory
    // In production, you'd fetch the list dynamically
    const tiles = [
        'tile_53750_203000.webp',
        'tile_55000_203000.webp',
        'tile_55000_204000.webp',
        'tile_55000_205000.webp',
        'tile_55000_206000.webp'
    ];

    const randomTile = tiles[Math.floor(Math.random() * tiles.length)];
    return `../piston_viewer/tiles_sat/low_res/${randomTile}`;
}

// Initialize the loading sequence
function initLoadingSequence() {
    const loadingScreen = document.getElementById('loading-screen');
    const satDisplay = document.getElementById('sat-display');
    const satImage = document.getElementById('sat-image');

    // Set the satellite image source
    satImage.src = getRandomSatelliteTile();

    // After animation duration, fade out loading screen and show satellite
    setTimeout(() => {
        loadingScreen.classList.add('fade-out');

        // Wait for fade out to complete, then show satellite image
        setTimeout(() => {
            loadingScreen.style.display = 'none';
            satDisplay.classList.add('show');
        }, 500);
    }, ANIMATION_DURATION);
}

// Start the sequence when the page loads
window.addEventListener('load', initLoadingSequence);

// Preload the satellite image for smooth transition
window.addEventListener('DOMContentLoaded', () => {
    const img = new Image();
    img.src = getRandomSatelliteTile();
});

// 3D Skew effect on satellite image with scroll/two-finger drag
function init3DSkewEffect() {
    const satImage = document.getElementById('sat-image');
    const satDisplay = document.getElementById('sat-display');

    let currentRotateX = 0;
    let currentRotateY = 0;
    let targetRotateX = 0;
    let targetRotateY = 0;
    let isAnimating = false;

    // Maximum rotation in degrees
    const MAX_ROTATION = 15;

    // Smoothly interpolate to target rotation
    function animate() {
        if (!isAnimating) return;

        // Ease towards target
        currentRotateX += (targetRotateX - currentRotateX) * 0.1;
        currentRotateY += (targetRotateY - currentRotateY) * 0.1;

        // Apply transform
        satImage.style.transform = `rotateX(${currentRotateX}deg) rotateY(${currentRotateY}deg)`;

        // Continue animation if we haven't reached target
        if (Math.abs(targetRotateX - currentRotateX) > 0.1 ||
            Math.abs(targetRotateY - currentRotateY) > 0.1) {
            requestAnimationFrame(animate);
        } else {
            isAnimating = false;
        }
    }

    // Handle wheel events (trackpad two-finger scroll)
    satDisplay.addEventListener('wheel', (e) => {
        e.preventDefault();

        // Calculate rotation based on scroll delta
        const deltaX = e.deltaX;
        const deltaY = e.deltaY;

        // Map scroll to rotation (smaller multiplier for more control)
        targetRotateY += deltaX * 0.05;
        targetRotateX -= deltaY * 0.05;

        // Clamp rotation
        targetRotateX = Math.max(-MAX_ROTATION, Math.min(MAX_ROTATION, targetRotateX));
        targetRotateY = Math.max(-MAX_ROTATION, Math.min(MAX_ROTATION, targetRotateY));

        // Start animation if not already running
        if (!isAnimating) {
            isAnimating = true;
            requestAnimationFrame(animate);
        }
    }, { passive: false });

    // Handle touch events for mobile
    let touchStartX = 0;
    let touchStartY = 0;

    satDisplay.addEventListener('touchstart', (e) => {
        if (e.touches.length === 2) {
            // Two finger touch
            touchStartX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
            touchStartY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
        }
    }, { passive: true });

    satDisplay.addEventListener('touchmove', (e) => {
        if (e.touches.length === 2) {
            e.preventDefault();

            const touchX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
            const touchY = (e.touches[0].clientY + e.touches[1].clientY) / 2;

            const deltaX = touchX - touchStartX;
            const deltaY = touchY - touchStartY;

            targetRotateY += deltaX * 0.1;
            targetRotateX -= deltaY * 0.1;

            targetRotateX = Math.max(-MAX_ROTATION, Math.min(MAX_ROTATION, targetRotateX));
            targetRotateY = Math.max(-MAX_ROTATION, Math.min(MAX_ROTATION, targetRotateY));

            touchStartX = touchX;
            touchStartY = touchY;

            if (!isAnimating) {
                isAnimating = true;
                requestAnimationFrame(animate);
            }
        }
    }, { passive: false });

    // Reset on double-click
    satDisplay.addEventListener('dblclick', () => {
        targetRotateX = 0;
        targetRotateY = 0;

        if (!isAnimating) {
            isAnimating = true;
            requestAnimationFrame(animate);
        }
    });
}

// Initialize 3D effect after satellite image is shown
const originalInit = initLoadingSequence;
window.removeEventListener('load', originalInit);
window.addEventListener('load', () => {
    originalInit();
    // Wait for satellite to appear before enabling 3D effect
    setTimeout(() => {
        init3DSkewEffect();
    }, ANIMATION_DURATION + 500);
});
