"""
Utilities for Storyboard Generation Pipeline

This module provides file I/O utilities for:
- Saving generated images with systematic naming
- Saving multi-option outputs (logline, lore, narrative) to markdown
- Saving artifact checkpoints for pipeline resumption
"""

import os
import re
import json
from datetime import datetime
from typing import List, Optional, Set, Union
from pathlib import Path

from .artifact import StoryboardSpec

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
        project_name: Name of the project (e.g., 'crimson_rebellion')
        image_type: Type of image ('component', 'shot', 'stage_setting')
        item_name: Name of the item being generated

    Returns:
        Local file path to the saved image
    """
    # Sanitize names for file system compatibility
    def sanitize_name(name: str) -> str:
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^\w\-_]', '_', name)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Convert to lowercase for consistency
        return sanitized.lower()

    # Create directory structure: data/{project_name}/images/
    images_dir = os.path.join("data", project_name, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Sanitize names
    clean_project = sanitize_name(project_name)
    clean_type = sanitize_name(image_type)
    clean_item = sanitize_name(item_name)

    # Generate timestamp in YYYYMMDD_HHMMSS format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate filename: {type}_{item_name}_{timestamp}.png
    filename = f"{clean_type}_{clean_item}_{timestamp}.png"
    filepath = os.path.join(images_dir, filename)

    # Save image
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    return filepath


def save_multi_option_output(project_name: str, step, options: List[str], user_choice: int) -> None:
    """Save multi-option outputs (logline/lore/narrative) to markdown file.

    Args:
        project_name: Name of the project
        step: GenerationStep enum value (LOGLINE, LORE, or NARRATIVE)
        options: List of generated options
        user_choice: Index of the option that was selected (0-based)
    """
    # Create data directory
    data_dir = os.path.join("data", project_name)
    os.makedirs(data_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create markdown content
    md_content = f"# {step.value.upper()} Options\n\n"
    md_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += "---\n\n"

    for i, option in enumerate(options):
        selected = " ✓ SELECTED" if i == user_choice else ""
        md_content += f"## Option {i+1}{selected}\n\n"
        md_content += f"{option}\n\n"
        md_content += "---\n\n"

    # Save to file
    filename = f"{step.value}_options_{timestamp}.md"
    filepath = os.path.join(data_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"📝 Options saved: {filepath}")


def save_artifact_checkpoint(artifact: StoryboardSpec, step) -> None:
    """Save intermediate artifact state for pipeline resumption.

    Args:
        artifact: The current storyboard artifact
        step: GenerationStep enum value
    """
    # Create data directory for project
    checkpoint_dir = os.path.join("data", artifact.name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save artifact with step name
    checkpoint_path = os.path.join(checkpoint_dir, f"artifact_after_{step.value}.json")
    with open(checkpoint_path, "w") as f:
        json.dump(artifact.model_dump(), f, indent=2)

    print(f"Checkpoint saved: {checkpoint_path}")


def get_valid_moodboard_filenames(moodboard_dir: str = "data/ref_moodboard") -> Set[str]:
    """Get set of valid moodboard filenames.

    Args:
        moodboard_dir: Path to moodboard directory

    Returns:
        Set of valid filenames (just the filename, not full path)
    """
    moodboard_path = Path(moodboard_dir)
    if not moodboard_path.exists():
        print(f"Warning: Moodboard directory {moodboard_dir} does not exist")
        return set()

    valid_files = {
        f.name for f in moodboard_path.glob("*.png")
        if f.name != "moodboard_tile.png"
    }
    return valid_files


def validate_moodboard_filename(
    filename: Optional[str],
    moodboard_dir: str = "data/ref_moodboard",
    component_name: str = "",
    component_type: str = ""
) -> Optional[str]:
    """Validate and fix moodboard reference filename.

    Args:
        filename: Filename to validate (can be just filename or full path)
        moodboard_dir: Path to moodboard directory
        component_name: Name of component for logging
        component_type: Type of component for logging

    Returns:
        Full path to valid moodboard image, or None if filename is invalid and no fallback
    """
    if not filename:
        return None

    # Strip path if full path provided - just get the filename
    filename_only = Path(filename).name

    # Get valid filenames
    valid_filenames = get_valid_moodboard_filenames(moodboard_dir)

    if not valid_filenames:
        print(f"⚠️  Warning: No valid moodboard files found in {moodboard_dir}")
        return None

    # Check if filename exists
    if filename_only in valid_filenames:
        # Valid! Return full path
        return f"{moodboard_dir}/{filename_only}"

    # Invalid filename - log warning
    print(f"⚠️  Invalid moodboard filename for {component_type} '{component_name}': '{filename_only}'")
    print(f"    This file does not exist in {moodboard_dir}")
    print(f"    Falling back to moodboard_tile.png")

    # Fallback to moodboard_tile.png
    fallback_path = f"{moodboard_dir}/moodboard_tile.png"
    if Path(fallback_path).exists():
        return fallback_path

    # No fallback available
    print(f"    Warning: Fallback moodboard_tile.png also not found")
    return None


def create_stage_shots_collage(artifact_or_json: Union[StoryboardSpec, dict, str], output_filename: Optional[str] = None) -> str:
    """Create a collage image of all stage setting shots with descriptions.

    Args:
        artifact_or_json: StoryboardSpec artifact, dict representation, or path to JSON file
        output_filename: Optional custom output filename (defaults to "{project_name}_stage_shots_collage.png")

    Returns:
        Path to the saved collage image

    Raises:
        ImportError: If PIL/Pillow is not installed
        ValueError: If no stage setting shots with images are found
    """
    if Image is None:
        raise ImportError("PIL/Pillow is required for creating collages. Install with: pip install Pillow")

    # Convert input to StoryboardSpec
    if isinstance(artifact_or_json, str):
        # It's a file path
        with open(artifact_or_json, 'r') as f:
            artifact_dict = json.load(f)
        artifact = StoryboardSpec(**artifact_dict)
    elif isinstance(artifact_or_json, dict):
        # It's a dictionary
        artifact = StoryboardSpec(**artifact_or_json)
    else:
        # It's already a StoryboardSpec
        artifact = artifact_or_json

    # Collect scenes with stage setting shots that have images
    scenes_with_images = []
    for scene in artifact.scenes or []:
        if scene.stage_setting_shot and scene.stage_setting_shot.image_path:
            image_path = Path(scene.stage_setting_shot.image_path)
            if image_path.exists():
                scenes_with_images.append((scene, image_path))

    if not scenes_with_images:
        raise ValueError("No stage setting shots with images found in artifact")

    print(f"📸 Creating collage for {len(scenes_with_images)} stage setting shots...")

    # Load all images and find dimensions
    images = []
    max_height = 0
    total_width = 0

    for scene, image_path in scenes_with_images:
        img = Image.open(image_path)
        images.append((scene, img))
        max_height = max(max_height, img.height)
        total_width += img.width

    # Calculate dimensions for text area
    text_height = 400  # Larger height for full description text
    padding = 30
    scene_padding = 20  # Padding between scenes

    # Create new image for collage
    collage_width = total_width + (len(images) - 1) * scene_padding
    collage_height = max_height + text_height + padding * 2
    collage = Image.new('RGB', (collage_width, collage_height), color='white')
    draw = ImageDraw.Draw(collage)

    # Try to load a font - large sizes
    try:
        # Try different font sizes and paths
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 42)
        desc_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    except:
        try:
            title_font = ImageFont.truetype("arial.ttf", 36)
            desc_font = ImageFont.truetype("arial.ttf", 22)
        except:
            # Fallback to default
            title_font = ImageFont.load_default()
            desc_font = ImageFont.load_default()

    # Paste images and add descriptions
    x_offset = 0
    for i, (scene, img) in enumerate(images):
        # Paste image
        y_offset = (max_height - img.height) // 2  # Center vertically
        collage.paste(img, (x_offset, y_offset))

        # Add scene name and description
        text_y = max_height + padding
        scene_name = scene.name
        scene_desc = scene.description or ""

        # Draw scene name (larger, bold-looking)
        draw.text((x_offset + 10, text_y), scene_name, fill='black', font=title_font)

        # Draw description (full text, wrapped)
        # Word wrap the description based on actual image width
        words = scene_desc.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            # Use textbbox to get actual text width
            bbox = draw.textbbox((0, 0), test_line, font=desc_font)
            text_width = bbox[2] - bbox[0]

            if text_width > img.width - 20:  # Leave some margin
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

        # Draw all wrapped lines (no limit)
        line_y = text_y + 50
        for line in lines:
            draw.text((x_offset + 10, line_y), line, fill='#333333', font=desc_font)
            line_y += 30

        x_offset += img.width + scene_padding

    # Save collage
    output_dir = Path("data") / artifact.name
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"stage_shots_collage_{timestamp}.png"

    output_path = output_dir / output_filename
    collage.save(output_path)

    print(f"✅ Collage saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Test the collage function on test_rbl artifact
    test_artifact_path = "data/test_rbl/artifact_after_stage_shot_images.json"
    print(f"🎬 Creating stage shots collage from {test_artifact_path}")
    collage_path = create_stage_shots_collage(test_artifact_path)
    print(f"📸 Collage created at: {collage_path}")