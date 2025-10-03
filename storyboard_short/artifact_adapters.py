"""
Artifact Adapters - DTO management for ShortStoryboardSpec

This module handles:
- Dynamic DTO creation for inputs/outputs
- Context extraction from the artifact
- Patching the artifact with LLM outputs
- State evaluation and next step determination
"""

from __future__ import annotations

from typing import List, Dict, Type, TYPE_CHECKING
from pydantic import BaseModel, Field, create_model

from .artifact import ShortStoryboardSpec, SceneSpec, ShotSpec, StrictModel

if TYPE_CHECKING:
    from .pipeline import GenerationStep


# ---------- Field Extraction Helper ----------

def get_field_desc(model_class: Type[BaseModel], field_name: str) -> str:
    """Extract field description from a Pydantic model."""
    field_info = model_class.model_fields[field_name]
    return field_info.description or f"{field_name} from {model_class.__name__}"


# ---------- Context DTO Extraction ----------

def extract_context_dto(artifact: ShortStoryboardSpec, step, **kwargs) -> BaseModel:
    """Extract context data from artifact and return populated DTO.

    Args:
        artifact: The current storyboard artifact
        step: GenerationStep enum value
        **kwargs: Additional context data to include

    Returns:
        Populated context DTO with all required fields
    """
    from .pipeline import GenerationStep

    if step == GenerationStep.IMAGE_RECOGNITION:
        fields = {
            "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
            "image_references": (List[str], Field(..., description="List of file paths to input images"))
        }
        ContextModel = create_model("ImageRecognitionContext", **fields, __base__=StrictModel)
        data = kwargs.copy()
        data["image_references"] = artifact.image_references
        return ContextModel(**data)

    elif step == GenerationStep.SCENES:
        fields = {
            "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
            "image_descriptions": (List[str], Field(..., description="AI-generated descriptions of input images"))
        }
        ContextModel = create_model("ScenesContext", **fields, __base__=StrictModel)
        data = kwargs.copy()
        data["image_descriptions"] = artifact.image_descriptions or []
        return ContextModel(**data)

    elif step == GenerationStep.SHOTS:
        fields = {
            "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
            "scenes": (List[SceneSpec], Field(..., description="Scenes to generate shots for"))
        }
        ContextModel = create_model("ShotsContext", **fields, __base__=StrictModel)
        data = kwargs.copy()
        data["scenes"] = artifact.scenes or []
        return ContextModel(**data)

    elif step == GenerationStep.SHOT_IMAGES:
        fields = {
            "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
            "scenes": (List[SceneSpec], Field(..., description="Scenes with shots to generate images for")),
            "image_references": (List[str], Field(..., description="Original input images for style reference"))
        }
        ContextModel = create_model("ShotImagesContext", **fields, __base__=StrictModel)
        data = kwargs.copy()
        data["scenes"] = artifact.scenes or []
        data["image_references"] = artifact.image_references
        return ContextModel(**data)

    else:
        raise ValueError(f"Unknown generation step: {step}")


def context_dto_to_string(context_dto: BaseModel) -> str:
    """Convert context DTO to string by enumerating field names and values.

    Args:
        context_dto: Populated context DTO

    Returns:
        Formatted string with all field names and values
    """
    lines = []
    for field_name, value in context_dto.model_dump().items():
        if isinstance(value, dict):
            value_str = "\n  ".join([f"{k}: {v}" for k, v in value.items()])
            lines.append(f"{field_name}:\n  {value_str}")
        elif isinstance(value, list):
            if len(value) > 0 and len(value) <= 10:
                value_str = "\n  - " + "\n  - ".join([str(v) for v in value])
                lines.append(f"{field_name}:{value_str}")
            else:
                lines.append(f"{field_name}: {len(value)} items")
        else:
            lines.append(f"{field_name}: {value}")

    return "\n".join(lines)


# ---------- Output DTO Creation ----------

def create_output_dto(step) -> Type[StrictModel] | None:
    """Create output DTO for each step.

    Args:
        step: GenerationStep enum value

    Returns:
        Pydantic model class or None for image generation steps
    """
    from .pipeline import GenerationStep

    fields = {}

    if step == GenerationStep.IMAGE_RECOGNITION:
        fields["image_descriptions"] = (List[str], Field(..., description="Detailed description of each input image in order"))

    elif step == GenerationStep.SCENES:
        fields["scene_names"] = (List[str], Field(..., description="List of scene names (1-2 scenes)"))
        fields["scene_descriptions"] = (List[str], Field(..., description="List of scene descriptions (1-2 scenes, same order as names)"))

    elif step == GenerationStep.SHOTS:
        # Define shot structure inline (without image_path)
        ShotDescriptionFields = {
            "name": (str, Field(..., description="Descriptive name for this shot")),
            "description": (str, Field(..., description="Visual description of the shot")),
            "camera_angle": (str, Field(..., description="Camera angle and movement"))
        }
        ShotDescriptionModel = create_model("ShotDescription", **ShotDescriptionFields, __base__=StrictModel)
        fields["shots"] = (List[ShotDescriptionModel], Field(..., description="Exactly 10 camera shots for the scene (descriptions only, no image paths)"))

    # Image generation steps don't need structured output
    elif step in [GenerationStep.SHOT_IMAGES, GenerationStep.FINAL_IMAGE]:
        return None

    else:
        raise ValueError(f"Unknown generation step: {step}")

    return create_model(
        f"{step.value.title().replace('_', '')}Output",
        **fields,
        __base__=StrictModel
    )


# ---------- Patchers (Update Artifact from LLM Outputs) ----------

def patch_artifact(artifact: ShortStoryboardSpec, step, output: BaseModel) -> ShortStoryboardSpec:
    """Update the artifact with LLM output.

    Args:
        artifact: The current storyboard artifact
        step: GenerationStep enum value
        output: LLM output as Pydantic model

    Returns:
        Updated artifact
    """
    from .pipeline import GenerationStep

    if step == GenerationStep.IMAGE_RECOGNITION:
        artifact.image_descriptions = output.image_descriptions

    elif step == GenerationStep.SCENES:
        # Create SceneSpec objects from names and descriptions (without shots)
        scenes = []
        for name, desc in zip(output.scene_names, output.scene_descriptions):
            scene = SceneSpec(name=name, description=desc, shots=None)
            scenes.append(scene)
        artifact.scenes = scenes

    # SHOTS and SHOT_IMAGES and FINAL_IMAGE are handled directly in pipeline
    # (batch generation, image generation, collage creation)

    return artifact


# ---------- Dependencies ----------

def get_dependencies():
    """Get step dependency graph."""
    from .pipeline import GenerationStep

    return {
        GenerationStep.IMAGE_RECOGNITION: [],
        GenerationStep.SCENES: [GenerationStep.IMAGE_RECOGNITION],
        GenerationStep.SHOTS: [GenerationStep.SCENES],
        GenerationStep.SHOT_IMAGES: [GenerationStep.SHOTS],
        GenerationStep.FINAL_IMAGE: [GenerationStep.SHOT_IMAGES],
    }


# ---------- State Evaluation ----------

def get_completion_status(artifact: ShortStoryboardSpec) -> Dict:
    """Check which generation steps have been completed.

    Args:
        artifact: The current storyboard artifact

    Returns:
        Dictionary mapping GenerationStep to completion status (bool)
    """
    from .pipeline import GenerationStep

    # Check if all scenes have shots
    all_scenes_have_shots = False
    if artifact.scenes and len(artifact.scenes) > 0:
        all_scenes_have_shots = all(
            scene.shots is not None and len(scene.shots) > 0
            for scene in artifact.scenes
        )

    # Check if all shots have images
    all_shots_have_images = False
    if artifact.scenes and len(artifact.scenes) > 0 and all_scenes_have_shots:
        all_shots_have_images = all(
            all(shot.image_path is not None for shot in scene.shots)
            for scene in artifact.scenes
            if scene.shots
        )

    return {
        GenerationStep.IMAGE_RECOGNITION: artifact.image_descriptions is not None and len(artifact.image_descriptions) > 0,
        GenerationStep.SCENES: artifact.scenes is not None and len(artifact.scenes) > 0,
        GenerationStep.SHOTS: all_scenes_have_shots,
        GenerationStep.SHOT_IMAGES: all_shots_have_images,
        GenerationStep.FINAL_IMAGE: artifact.final_image_path is not None,
    }


def get_next_steps(artifact: ShortStoryboardSpec) -> List:
    """Get the next available generation steps based on completed dependencies.

    Args:
        artifact: The current storyboard artifact

    Returns:
        List of GenerationStep enum values that can be executed next
    """
    completion_status = get_completion_status(artifact)
    available_steps = []
    dependencies = get_dependencies()

    for step, deps in dependencies.items():
        if not completion_status[step]:  # Step not completed
            if all(completion_status[dep] for dep in deps):  # All dependencies completed
                available_steps.append(step)

    return available_steps
