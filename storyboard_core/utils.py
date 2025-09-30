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
from typing import List

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