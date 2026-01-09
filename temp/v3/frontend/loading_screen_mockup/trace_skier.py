#!/usr/bin/env python3
"""
Trace skier silhouette to SVG using edge detection and contour tracing
"""

import cv2
import numpy as np
from pathlib import Path

def trace_to_svg(image_path, output_path, simplification=0.5):
    """
    Convert a silhouette image to SVG paths

    Args:
        image_path: Path to input image
        output_path: Path to output SVG file
        simplification: Higher = simpler paths (epsilon for contour approximation)
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to get binary image (assuming dark silhouette on light background)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find ALL contours including holes (use RETR_CCOMP for hierarchy)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in image")

    # Get image dimensions for SVG viewBox
    height, width = binary.shape

    # Convert contour to SVG path
    def contour_to_path(contour):
        """Convert OpenCV contour to SVG path string"""
        if len(contour) == 0:
            return ""

        # Start path
        path = f"M {contour[0][0][0]},{contour[0][0][1]}"

        # Add lines to each point
        for point in contour[1:]:
            x, y = point[0]
            path += f" L {x},{y}"

        # Close path
        path += " Z"
        return path

    # Process all contours with hierarchy
    # hierarchy[i] = [next, previous, first_child, parent]
    # Outer contours have parent = -1
    # Inner contours (holes) have parent >= 0

    all_paths = []
    epsilon = simplification

    for i, contour in enumerate(contours):
        # Simplify the contour
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Skip very small contours (noise)
        if cv2.contourArea(contour) < 50:
            continue

        path_data = contour_to_path(approx)
        all_paths.append(path_data)

    # Combine all paths
    combined_path = " ".join(all_paths)

    # Create SVG with fill-rule to handle holes properly
    svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
    <defs>
        <linearGradient id="skierGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#ff6b9d;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#74b9ff;stop-opacity:1" />
        </linearGradient>
    </defs>
    <path d="{combined_path}" fill="url(#skierGradient)" fill-rule="evenodd" stroke="none"/>
</svg>"""

    # Write SVG file
    with open(output_path, 'w') as f:
        f.write(svg_content)

    print(f"✓ SVG created: {output_path}")
    print(f"  Image size: {width}x{height}")
    print(f"  Total contours: {len(all_paths)}")
    print(f"  Total path points: {sum(len(c) for c in contours if cv2.contourArea(c) >= 50)}")
    return output_path


if __name__ == "__main__":
    # Input and output paths
    input_image = Path(__file__).parent / "skier_input.png"
    output_svg = Path(__file__).parent / "skier_traced.svg"

    print(f"Tracing: {input_image}")
    trace_to_svg(input_image, output_svg, simplification=0.5)
    print(f"\nDone! Check {output_svg}")
