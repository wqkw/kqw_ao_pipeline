#!/usr/bin/env python3
"""
Script to tile images from ref_moodboard into a 3x2 grid with black separators.
"""

import os
from PIL import Image
import glob

def create_moodboard_tile():
    input_dir = "data/ref_moodboard"
    output_path = "data/moodboard_tile.png"

    # Get all PNG files from the input directory
    image_files = glob.glob(os.path.join(input_dir, "*.png"))

    if not image_files:
        print(f"No PNG files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Calculate grid dimensions for approximately 3:2 ratio
    num_images = len(image_files)

    # Find optimal grid dimensions
    import math

    # Start with square root and adjust to get close to 3:2 ratio (2 wide, 3 tall)
    # For 3:2 ratio, we want cols/rows ≈ 2/3, so rows should be about 1.5 * cols
    sqrt_images = math.sqrt(num_images)

    # Try different column counts around the square root
    best_cols = max(1, int(sqrt_images))
    best_rows = (num_images + best_cols - 1) // best_cols
    best_diff = float('inf')
    target_ratio = 2/3  # cols/rows ratio we want

    for cols in range(max(1, int(sqrt_images * 0.5)), min(num_images, int(sqrt_images * 2)) + 1):
        rows = (num_images + cols - 1) // cols  # Ceiling division
        current_ratio = cols / rows
        diff = abs(current_ratio - target_ratio)
        if diff < best_diff:
            best_diff = diff
            best_cols = cols
            best_rows = rows

    cols = best_cols
    rows = best_rows
    print(f"Using {cols}x{rows} grid for {num_images} images")

    # Load and resize images to reduce file size
    images = []
    target_width = 400  # Reduced from original size

    for img_path in image_files:  # Use all images
        with Image.open(img_path) as img:
            # Calculate height to maintain aspect ratio
            aspect_ratio = img.height / img.width
            target_height = int(target_width * aspect_ratio)

            # Resize image
            resized_img = img.resize((target_width, target_height), Image.LANCZOS)
            images.append(resized_img)

    # Calculate dimensions for the tiled image
    separator_width = 5

    # Assume all images have similar dimensions (use first image as reference)
    img_width = images[0].width
    img_height = images[0].height

    # Calculate total dimensions
    total_width = (cols * img_width) + ((cols - 1) * separator_width)
    total_height = (rows * img_height) + ((rows - 1) * separator_width)

    # Create the final tiled image with black background
    tiled_image = Image.new('RGB', (total_width, total_height), color='black')

    # Place images in grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        x = col * (img_width + separator_width)
        y = row * (img_height + separator_width)

        tiled_image.paste(img, (x, y))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the tiled image
    tiled_image.save(output_path, 'PNG', optimize=True)
    print(f"Tiled moodboard saved to {output_path}")
    print(f"Final dimensions: {total_width}x{total_height}")

if __name__ == "__main__":
    create_moodboard_tile()