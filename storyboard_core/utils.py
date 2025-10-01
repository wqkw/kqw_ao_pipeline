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
from typing import List, Optional, Set
from pathlib import Path

from .artifact import StoryboardSpec


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