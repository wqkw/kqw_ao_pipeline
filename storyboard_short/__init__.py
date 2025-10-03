"""
Short Storyboard Generation Pipeline

Simplified pipeline for generating quick scene sequences from 1-3 input images.
"""

from .artifact import (
    ShortStoryboardSpec,
    SceneSpec,
    ShotSpec,
    StrictModel
)

from .pipeline import (
    GenerationStep,
    run_generation_pipeline,
    generate_step,
    generate_batch_shots,
    generate_shot_images
)

from .artifact_adapters import (
    get_completion_status,
    get_next_steps,
    extract_context_dto,
    context_dto_to_string,
    create_output_dto,
    patch_artifact
)

from .utils import (
    save_artifact_checkpoint,
    save_image_to_data,
    create_shots_collage
)

__all__ = [
    # Core models
    "ShortStoryboardSpec",
    "SceneSpec",
    "ShotSpec",
    "StrictModel",

    # Pipeline
    "GenerationStep",
    "run_generation_pipeline",
    "generate_step",
    "generate_batch_shots",
    "generate_shot_images",

    # Adapters
    "get_completion_status",
    "get_next_steps",
    "extract_context_dto",
    "context_dto_to_string",
    "create_output_dto",
    "patch_artifact",

    # Utils
    "save_artifact_checkpoint",
    "save_image_to_data",
    "create_shots_collage",
]
