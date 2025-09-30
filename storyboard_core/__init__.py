"""
Storyboard Generation Pipeline Core Module

This module provides the core functionality for generating storyboards through a
multi-step pipeline process, including artifact management, context adaptation,
and orchestration of generation steps.
"""

from .artifact import (
    StoryboardSpec,
    SceneSpec,
    CharacterSpec,
    LocationSpec,
    PropSpec,
    ShotSpec,
    StrictModel
)

from .pipeline import (
    GenerationStep,
    run_generation_pipeline,
    generate_step,
    generate_batch_components,
    generate_batch_stage_shots,
    generate_stage_shot_images,
    generate_batch_shot_descriptions,
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
    save_multi_option_output
)

__all__ = [
    # Core models
    "StoryboardSpec",
    "SceneSpec",
    "CharacterSpec",
    "LocationSpec",
    "PropSpec",
    "ShotSpec",
    "StrictModel",

    # Pipeline
    "GenerationStep",
    "run_generation_pipeline",

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
    "save_multi_option_output",

    # Generation functions
    "generate_step",
    "generate_batch_components",
    "generate_batch_stage_shots",
    "generate_stage_shot_images",
    "generate_batch_shot_descriptions",
    "generate_shot_images"
]