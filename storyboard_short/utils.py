"""
Utilities for Short Storyboard Generation Pipeline

This module provides file I/O utilities for:
- Saving generated images with systematic naming
- Saving artifact checkpoints for pipeline resumption
- Creating shot collages
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Union, Optional

from .artifact import ShortStoryboardSpec

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


def save_image_to_data(image_bytes: bytes, project_name: str, image_type: str, item_name: str) -> str:
    """Save image to data folder with systematic naming convention.

    Args:
        image_bytes: The image data as bytes
        project_name: Name of the project (e.g., 'test_short_001')
        image_type: Type of image ('final', 'shot', etc.)
        item_name: Name of the item being generated

    Returns:
        Local file path to the saved image
    """
    # Sanitize names for file system compatibility
    def sanitize_name(name: str) -> str:
        sanitized = re.sub(r'[^\w\-_]', '_', name)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized.lower()

    # Create directory structure: data/{project_name}/images/
    images_dir = os.path.join("data", project_name, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Sanitize names
    clean_type = sanitize_name(image_type)
    clean_item = sanitize_name(item_name)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate filename: {type}_{item_name}_{timestamp}.png
    filename = f"{clean_type}_{clean_item}_{timestamp}.png"
    filepath = os.path.join(images_dir, filename)

    # Save image
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    return filepath


def save_artifact_checkpoint(artifact: ShortStoryboardSpec, step) -> None:
    """Save intermediate artifact state for pipeline resumption.

    Args:
        artifact: The current storyboard artifact
        step: GenerationStep enum value
    """
    # Create data directory using artifact name
    checkpoint_dir = os.path.join("data", artifact.name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save artifact with step name
    checkpoint_path = os.path.join(checkpoint_dir, f"artifact_after_{step.value}.json")
    with open(checkpoint_path, "w") as f:
        json.dump(artifact.model_dump(), f, indent=2)

    print(f"Checkpoint saved: {checkpoint_path}")


def create_shots_collage(artifact_or_json: Union[ShortStoryboardSpec, dict, str], output_filename: Optional[str] = None) -> str:
    """Create a collage image of all shots with scene names and descriptions.

    Args:
        artifact_or_json: ShortStoryboardSpec artifact, dict representation, or path to JSON file
        output_filename: Optional custom output filename (defaults to "shots_collage_{timestamp}.png")

    Returns:
        Path to the saved collage image

    Raises:
        ImportError: If PIL/Pillow is not installed
        ValueError: If no shots with images are found
    """
    if Image is None:
        raise ImportError("PIL/Pillow is required for creating collages. Install with: pip install Pillow")

    # Convert input to ShortStoryboardSpec
    if isinstance(artifact_or_json, str):
        with open(artifact_or_json, 'r') as f:
            artifact_dict = json.load(f)
        artifact = ShortStoryboardSpec(**artifact_dict)
    elif isinstance(artifact_or_json, dict):
        artifact = ShortStoryboardSpec(**artifact_or_json)
    else:
        artifact = artifact_or_json

    # Collect all shots with images
    shots_with_images = []
    for scene in artifact.scenes or []:
        if scene.shots:
            for shot in scene.shots:
                if shot.image_path:
                    image_path = Path(shot.image_path)
                    if image_path.exists():
                        shots_with_images.append((scene.name, shot, image_path))

    if not shots_with_images:
        raise ValueError("No shots with images found in artifact")

    print(f"📸 Creating collage for {len(shots_with_images)} shots...")

    # Load all images and find dimensions
    images = []
    max_height = 0
    total_width = 0

    for scene_name, shot, image_path in shots_with_images:
        img = Image.open(image_path)
        images.append((scene_name, shot, img))
        max_height = max(max_height, img.height)
        total_width += img.width

    # Calculate dimensions for text area
    text_height = 450  # Height for scene name + shot name + description
    padding = 30
    shot_padding = 20  # Padding between shots

    # Create new image for collage
    collage_width = total_width + (len(images) - 1) * shot_padding
    collage_height = max_height + text_height + padding * 2
    collage = Image.new('RGB', (collage_width, collage_height), color='white')
    draw = ImageDraw.Draw(collage)

    # Try to load a font
    try:
        scene_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 38)
        shot_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        desc_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        try:
            scene_font = ImageFont.truetype("arial.ttf", 32)
            shot_font = ImageFont.truetype("arial.ttf", 28)
            desc_font = ImageFont.truetype("arial.ttf", 20)
        except:
            scene_font = ImageFont.load_default()
            shot_font = ImageFont.load_default()
            desc_font = ImageFont.load_default()

    # Paste images and add descriptions
    x_offset = 0
    for i, (scene_name, shot, img) in enumerate(images):
        # Paste image
        y_offset = (max_height - img.height) // 2  # Center vertically
        collage.paste(img, (x_offset, y_offset))

        # Add text below image
        text_y = max_height + padding

        # Draw scene name (bold-looking)
        draw.text((x_offset + 10, text_y), scene_name, fill='black', font=scene_font)
        text_y += 45

        # Draw shot name
        draw.text((x_offset + 10, text_y), shot.name, fill='#444444', font=shot_font)
        text_y += 40

        # Draw shot description (wrapped)
        shot_desc = shot.description or ""
        words = shot_desc.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=desc_font)
            text_width = bbox[2] - bbox[0]

            if text_width > img.width - 20:  # Leave margin
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
                    current_line = []
            else:
                current_line.append(word)

        if current_line:
            lines.append(' '.join(current_line))

        # Draw wrapped lines (limit to fit in text area)
        line_y = text_y
        max_lines = 8  # Limit lines to fit in available space
        for line in lines[:max_lines]:
            draw.text((x_offset + 10, line_y), line, fill='#666666', font=desc_font)
            line_y += 32

        x_offset += img.width + shot_padding

    # Save collage
    output_dir = Path("data") / artifact.name
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"shots_collage_{timestamp}.png"

    output_path = output_dir / output_filename
    collage.save(output_path)

    print(f"✅ Collage saved: {output_path}")
    return str(output_path)


def artifact_to_shot_list_string(artifact: ShortStoryboardSpec) -> str:
    """Convert artifact to a formatted string listing all shots with details.

    Args:
        artifact: ShortStoryboardSpec artifact

    Returns:
        Formatted string with shot names, descriptions, and camera angles
    """
    output_lines = []

    if not artifact.scenes:
        return "No scenes found in artifact."

    for scene_idx, scene in enumerate(artifact.scenes, 1):
        output_lines.append(f"SCENE {scene_idx}: {scene.name}")
        output_lines.append("=" * 60)
        output_lines.append("")

        if not scene.shots:
            output_lines.append("No shots in this scene.")
            output_lines.append("")
            continue

        for shot_idx, shot in enumerate(scene.shots, 1):
            output_lines.append(f"Shot {shot_idx}: {shot.name}")
            output_lines.append(f"Description: {shot.description}")
            if shot.camera_angle:
                output_lines.append(f"Camera Angle: {shot.camera_angle}")
            output_lines.append("")

    return "\n".join(output_lines)
