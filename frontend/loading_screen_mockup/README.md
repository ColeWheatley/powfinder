# Loading Screen Mockup

A lightweight, vanilla JavaScript loading screen with race car animation.

## Features

- Race car animation zooming in from the left with speed lines
- "Good code loads fast." gradient text (matching Powfinder 3D aesthetic)
- Flashing "Fetching high-res bestagons..." message
- Hexagon pattern background (subtle, animated)
- Displays random satellite tile after ~3.5 seconds
- Uses Outfit font and pink-to-blue gradient from main app

## Files

- `index.html` - Main HTML structure
- `style.css` - All animations and styling
- `script.js` - Loading sequence logic
- `README.md` - This file

## Usage

Simply open `index.html` in a browser. The sequence will:
1. Show race car zooming in (0-1.5s)
2. Display "Good code loads fast." (1s)
3. Flash "Fetching high-res bestagons..." (2s onwards)
4. Fade to satellite image (3.5s)

## Integration Notes

To integrate into the main piston_viewer:
- Copy the CSS animations to `style.css`
- Replace the `#loader` div in `index.html` with the new loading screen HTML
- Add the JavaScript logic to `main.js` or keep as separate file
- Adjust timing as needed for actual load performance
