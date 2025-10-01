"""
Artifact Adapters - Projection and extraction logic for StoryboardSpec

This module handles all interactions with the StoryboardSpec artifact, including:
- Dynamic DTO creation for inputs/outputs
- Context extraction from the artifact
- Patching the artifact with LLM outputs
- State evaluation and dependency management
"""

from __future__ import annotations

from typing import List, Dict, Type, Any, TYPE_CHECKING
from pydantic import BaseModel, Field, create_model
from enum import Enum

from .artifact import StoryboardSpec, SceneSpec, CharacterSpec, LocationSpec, PropSpec, ShotSpec, StrictModel

if TYPE_CHECKING:
    from .pipeline import GenerationStep


# ---------- Field Extraction Helper ----------

def get_field_desc(model_class: Type[BaseModel], field_name: str) -> str:
    """Extract field description from a Pydantic model."""
    field_info = model_class.model_fields[field_name]
    return field_info.description or f"{field_name} from {model_class.__name__}"


# ---------- Context DTO Extraction ----------

def extract_context_dto(artifact: StoryboardSpec, step, **kwargs) -> BaseModel:
    """Extract context data from artifact and return populated DTO.

    Args:
        artifact: The current storyboard artifact
        step: GenerationStep enum value
        **kwargs: Additional context data to include

    Returns:
        Populated context DTO with all required fields
    """
    from .pipeline import GenerationStep

    # Each step is self-contained: define fields, create model, extract data, return instance
    if step == GenerationStep.LOGLINE:
        fields = {
            "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
            "moodboard_path": (str, Field(..., description=get_field_desc(StoryboardSpec, "moodboard_path")))
        }
        ContextModel = create_model("LoglineContext", **fields, __base__=StrictModel)
        data = kwargs.copy()
        data["moodboard_path"] = artifact.moodboard_path
        return ContextModel(**data)
    elif step == GenerationStep.LORE:
        fields = {
            "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
            "moodboard_path": (str, Field(..., description=get_field_desc(StoryboardSpec, "moodboard_path"))),
            "logline": (str, Field(..., description=get_field_desc(StoryboardSpec, "logline")))
        }
        ContextModel = create_model("LoreContext", **fields, __base__=StrictModel)
        data = kwargs.copy()
        data["moodboard_path"] = artifact.moodboard_path
        data["logline"] = artifact.logline
        return ContextModel(**data)
    elif step == GenerationStep.NARRATIVE:
        fields = {
            "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
            "logline": (str, Field(..., description=get_field_desc(StoryboardSpec, "logline"))),
            "lore": (str, Field(..., description=get_field_desc(StoryboardSpec, "lore")))
        }
        ContextModel = create_model("NarrativeContext", **fields, __base__=StrictModel)
        data = kwargs.copy()
        data["logline"] = artifact.logline
        data["lore"] = artifact.lore
        return ContextModel(**data)
    elif step == GenerationStep.SCENES:
        fields = {
            "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
            "logline": (str, Field(..., description=get_field_desc(StoryboardSpec, "logline"))),
            "lore": (str, Field(..., description=get_field_desc(StoryboardSpec, "lore"))),
            "narrative": (str, Field(..., description=get_field_desc(StoryboardSpec, "narrative")))
        }
        ContextModel = create_model("ScenesContext", **fields, __base__=StrictModel)
        data = kwargs.copy()
        data["logline"] = artifact.logline
        data["lore"] = artifact.lore
        data["narrative"] = artifact.narrative
        return ContextModel(**data)
    elif step == GenerationStep.COMPONENT_DESCRIPTIONS:
        fields = {
            "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
            "logline": (str, Field(..., description=get_field_desc(StoryboardSpec, "logline"))),
            "moodboard_path": (str, Field(..., description=get_field_desc(StoryboardSpec, "moodboard_path"))),
            "lore": (str, Field(..., description=get_field_desc(StoryboardSpec, "lore"))),
            "narrative": (str, Field(..., description=get_field_desc(StoryboardSpec, "narrative"))),
            "scene_names": (List[str], Field(..., description="List of scene names from the story structure")),
            "scene_descriptions": (List[str], Field(..., description="List of scene descriptions from the story structure"))
        }
        ContextModel = create_model("ComponentDescriptionsContext", **fields, __base__=StrictModel)
        data = kwargs.copy()
        data["logline"] = artifact.logline
        data["moodboard_path"] = artifact.moodboard_path
        data["lore"] = artifact.lore
        data["narrative"] = artifact.narrative
        data["scene_names"] = [scene.name for scene in artifact.scenes]
        data["scene_descriptions"] = [scene.description or "" for scene in artifact.scenes]
        return ContextModel(**data)
    elif step == GenerationStep.COMPONENT_IMAGES:
        fields = {
            "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
            "logline": (str, Field(..., description=get_field_desc(StoryboardSpec, "logline"))),
            "moodboard_path": (str, Field(..., description=get_field_desc(StoryboardSpec, "moodboard_path"))),
            "lore": (str, Field(..., description=get_field_desc(StoryboardSpec, "lore"))),
            "narrative": (str, Field(..., description=get_field_desc(StoryboardSpec, "narrative"))),
            "characters": (List[CharacterSpec], Field(..., description="List of characters to generate images for")),
            "locations": (List[LocationSpec], Field(..., description="List of locations to generate images for")),
            "props": (List[PropSpec], Field(..., description="List of props to generate images for"))
        }
        ContextModel = create_model("ComponentImagesContext", **fields, __base__=StrictModel)
        data = kwargs.copy()
        data["logline"] = artifact.logline
        data["moodboard_path"] = artifact.moodboard_path
        data["lore"] = artifact.lore
        data["narrative"] = artifact.narrative
        data["characters"] = artifact.characters or []
        data["locations"] = artifact.locations or []
        data["props"] = artifact.props or []
        return ContextModel(**data)
    elif step == GenerationStep.STAGE_SHOT_DESCRIPTIONS:
        # Note: This DTO is no longer used by generate_batch_stage_shots (refactored to not use DTOs)
        # Kept for backward compatibility in case other code paths use it
        fields = {
            "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
            "scene_name": (str, Field(default="", description=get_field_desc(SceneSpec, "name"))),
            "scene_description": (str, Field(default="", description=get_field_desc(SceneSpec, "description"))),
            "narrative": (str, Field(..., description=get_field_desc(StoryboardSpec, "narrative"))),
            "prop_descriptions": (Dict[str, str], Field(..., description="Dictionary of prop names to descriptions"))
        }
        ContextModel = create_model("StageShotDescriptionsContext", **fields, __base__=StrictModel)
        data = kwargs.copy()
        data["narrative"] = artifact.narrative
        data["prop_descriptions"] = {prop.name: prop.description or "" for prop in artifact.props}
        return ContextModel(**data)
    elif step == GenerationStep.STAGE_SHOT_IMAGES:
        # Note: This DTO is no longer used by generate_stage_shot_images (refactored to not use DTOs)
        # Kept for backward compatibility in case other code paths use it
        fields = {
            "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
            "stage_shot_description": (str, Field(default="", description="Generated description of the stage setting shot")),
            "scene_description": (str, Field(default="", description=get_field_desc(SceneSpec, "description"))),
            "component_image_paths": (List[str], Field(default_factory=list, description="URLs of component images"))
        }
        ContextModel = create_model("StageShotImagesContext", **fields, __base__=StrictModel)
        data = kwargs.copy()
        return ContextModel(**data)
    elif step == GenerationStep.SHOT_DESCRIPTIONS:
        fields = {
            "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
            "scene_name": (str, Field(..., description=get_field_desc(SceneSpec, "name"))),
            "scene_description": (str, Field(..., description=get_field_desc(SceneSpec, "description"))),
            "narrative": (str, Field(..., description=get_field_desc(StoryboardSpec, "narrative"))),
            "character_descriptions": (Dict[str, str], Field(..., description="Dictionary of character names to descriptions")),
            "location_descriptions": (Dict[str, str], Field(..., description="Dictionary of location names to descriptions")),
            "prop_descriptions": (Dict[str, str], Field(..., description="Dictionary of prop names to descriptions"))
        }
        ContextModel = create_model("ShotDescriptionsContext", **fields, __base__=StrictModel)
        data = kwargs.copy()
        data["narrative"] = artifact.narrative
        data["character_descriptions"] = {char.name: char.description or "" for char in artifact.characters}
        data["location_descriptions"] = {loc.name: loc.description or "" for loc in artifact.locations}
        data["prop_descriptions"] = {prop.name: prop.description or "" for prop in artifact.props}
        return ContextModel(**data)
    elif step == GenerationStep.SHOT_IMAGES:
        fields = {
            "prompt_input": (str, Field(..., description="User's creative input for this generation step")),
            "shot_description": (str, Field(..., description="Generated description of the shot")),
            "stage_setting_image_path": (str, Field(..., description="Path to the stage setting image"))
        }
        ContextModel = create_model("ShotImagesContext", **fields, __base__=StrictModel)
        data = kwargs.copy()
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
        # Format the value based on its type
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


def create_output_dto(step) -> Type[StrictModel] | None:
    """Create output DTO using descriptions from the core artifact.

    Args:
        step: GenerationStep enum value

    Returns:
        Pydantic model class or None for image generation steps
    """
    # Import here to avoid circular dependency
    from .pipeline import GenerationStep

    fields = {}

    if step == GenerationStep.LOGLINE:
        base_desc = get_field_desc(StoryboardSpec, "logline")
        fields["logline_option_1"] = (str, Field(..., description=f"First option: {base_desc}"))
        fields["logline_option_2"] = (str, Field(..., description=f"Second option: {base_desc}"))
        fields["logline_option_3"] = (str, Field(..., description=f"Third option: {base_desc}"))
        fields["logline_option_4"] = (str, Field(..., description=f"Fourth option: {base_desc}"))
        fields["logline_option_5"] = (str, Field(..., description=f"Fifth option: {base_desc}"))

    elif step == GenerationStep.LORE:
        base_desc = get_field_desc(StoryboardSpec, "lore")
        fields["lore_option_1"] = (str, Field(..., description=f"First option: {base_desc}"))
        fields["lore_option_2"] = (str, Field(..., description=f"Second option: {base_desc}"))
        fields["lore_option_3"] = (str, Field(..., description=f"Third option: {base_desc}"))

    elif step == GenerationStep.NARRATIVE:
        base_desc = get_field_desc(StoryboardSpec, "narrative")
        fields["narrative_option_1"] = (str, Field(..., description=f"First option: {base_desc}"))
        fields["narrative_option_2"] = (str, Field(..., description=f"Second option: {base_desc}"))
        fields["narrative_option_3"] = (str, Field(..., description=f"Third option: {base_desc}"))

    elif step == GenerationStep.SCENES:
        fields["scene_names"] = (List[str], Field(..., description=f"List of {get_field_desc(SceneSpec, 'name').lower()}"))
        fields["scene_descriptions"] = (List[str], Field(..., description=f"List of {get_field_desc(SceneSpec, 'description').lower()}"))
        # fields["key_shot_sketches"] = (List[str], Field(..., description="List of key visual moments for each scene"))

    elif step == GenerationStep.COMPONENT_DESCRIPTIONS:
        fields["character_descriptions"] = (Dict[str, str], Field(..., description=f"Dictionary of {get_field_desc(CharacterSpec, 'name').lower()} to {get_field_desc(CharacterSpec, 'description').lower()}"))
        fields["character_reference_images"] = (Dict[str, str], Field(..., description="Dictionary of character names to reference image filenames. CRITICAL: Use the EXACT filename from the provided reference images list. Each filename is unique. Match the filename precisely character-by-character. Choose the image that best matches the visual style for each character."))
        fields["location_descriptions"] = (Dict[str, str], Field(..., description=f"Dictionary of {get_field_desc(LocationSpec, 'name').lower()} to {get_field_desc(LocationSpec, 'description').lower()}"))
        fields["location_reference_images"] = (Dict[str, str], Field(..., description="Dictionary of location names to reference image filenames. CRITICAL: Use the EXACT filename from the provided reference images list. Each filename is unique. Match the filename precisely character-by-character. Choose the image that best matches the visual style for each location."))
        fields["prop_descriptions"] = (Dict[str, str], Field(..., description=f"Dictionary of {get_field_desc(PropSpec, 'name').lower()} to {get_field_desc(PropSpec, 'description').lower()}"))
        fields["prop_reference_images"] = (Dict[str, str], Field(..., description="Dictionary of prop names to reference image filenames. CRITICAL: Use the EXACT filename from the provided reference images list. Each filename is unique. Match the filename precisely character-by-character. Choose the image that best matches the visual style for each prop."))
        fields["scene_to_characters"] = (Dict[str, List[str]], Field(..., description="Mapping of scene names to required characters"))
        fields["scene_to_locations"] = (Dict[str, List[str]], Field(..., description="Mapping of scene names to required locations"))
        fields["scene_to_props"] = (Dict[str, List[str]], Field(..., description="Mapping of scene names to required props"))

    elif step == GenerationStep.STAGE_SHOT_DESCRIPTIONS:
        fields["shot_descriptions"] = (List[str], Field(..., description="List of stage setting shot descriptions"))

    elif step == GenerationStep.SHOT_DESCRIPTIONS:
        fields["shot_descriptions"] = (List[str], Field(..., description="List of individual shot descriptions for this scene"))

    # Image steps don't need structured outputs
    else:
        return None

    return create_model(
        f"{step.value.title().replace('_', '')}Output",
        **fields,
        __base__=StrictModel
    )


# ---------- Patchers (Update Artifact from LLM Outputs) ----------

def patch_artifact(artifact: StoryboardSpec, step, output: BaseModel, user_choice: int = 0) -> StoryboardSpec:
    """Update the artifact with LLM output. For multi-option outputs, user_choice selects which option.

    Args:
        artifact: The current storyboard artifact
        step: GenerationStep enum value
        output: LLM output as Pydantic model
        user_choice: Index of option to select for multi-option outputs (0-based)

    Returns:
        Updated artifact
    """
    # Import here to avoid circular dependency
    from .pipeline import GenerationStep
    from .utils import save_multi_option_output

    if step == GenerationStep.LOGLINE:
        options = [output.logline_option_1, output.logline_option_2, output.logline_option_3,
                   output.logline_option_4, output.logline_option_5]
        save_multi_option_output(artifact.name, step, options, user_choice)
        artifact.logline = options[user_choice]

    elif step == GenerationStep.LORE:
        options = [output.lore_option_1, output.lore_option_2, output.lore_option_3]
        save_multi_option_output(artifact.name, step, options, user_choice)
        artifact.lore = options[user_choice]

    elif step == GenerationStep.NARRATIVE:
        options = [output.narrative_option_1, output.narrative_option_2, output.narrative_option_3]
        save_multi_option_output(artifact.name, step, options, user_choice)
        artifact.narrative = options[user_choice]

    elif step == GenerationStep.SCENES:
        scenes = []
        for i, (name, desc) in enumerate(zip(output.scene_names, output.scene_descriptions)):
            scene = SceneSpec(name=name, description=desc)
            scenes.append(scene)
        artifact.scenes = scenes

    elif step == GenerationStep.COMPONENT_DESCRIPTIONS:
        from .utils import validate_moodboard_filename

        # Create characters with moodboard references
        characters = []
        for name, desc in output.character_descriptions.items():
            # Convert filename to full path if available, with validation
            moodboard_ref = None
            if hasattr(output, 'character_reference_images') and name in output.character_reference_images:
                filename = output.character_reference_images[name]
                # Validate the filename
                moodboard_ref = validate_moodboard_filename(
                    filename,
                    component_name=name,
                    component_type="character"
                )

            character = CharacterSpec(
                name=name,
                description=desc,
                image_reference_from_moodboard=moodboard_ref
            )
            characters.append(character)
        artifact.characters = characters

        # Create locations with moodboard references
        locations = []
        for name, desc in output.location_descriptions.items():
            # Convert filename to full path if available, with validation
            moodboard_ref = None
            if hasattr(output, 'location_reference_images') and name in output.location_reference_images:
                filename = output.location_reference_images[name]
                # Validate the filename
                moodboard_ref = validate_moodboard_filename(
                    filename,
                    component_name=name,
                    component_type="location"
                )

            location = LocationSpec(
                name=name,
                description=desc,
                image_reference_from_moodboard=moodboard_ref
            )
            locations.append(location)
        artifact.locations = locations

        # Create props with moodboard references
        props = []
        for name, desc in output.prop_descriptions.items():
            # Convert filename to full path if available, with validation
            moodboard_ref = None
            if hasattr(output, 'prop_reference_images') and name in output.prop_reference_images:
                filename = output.prop_reference_images[name]
                # Validate the filename
                moodboard_ref = validate_moodboard_filename(
                    filename,
                    component_name=name,
                    component_type="prop"
                )

            prop = PropSpec(
                name=name,
                description=desc,
                image_reference_from_moodboard=moodboard_ref
            )
            props.append(prop)
        artifact.props = props

        # Update scenes with component references
        if artifact.scenes:
            for scene in artifact.scenes:
                if scene.name in output.scene_to_characters:
                    scene.character_names = output.scene_to_characters[scene.name]
                if scene.name in output.scene_to_locations:
                    scene.location_names = output.scene_to_locations[scene.name]
                if scene.name in output.scene_to_props:
                    scene.prop_names = output.scene_to_props[scene.name]

    elif step == GenerationStep.STAGE_SHOT_DESCRIPTIONS:
        # This step patches each scene individually
        if hasattr(output, 'shot_descriptions') and artifact.scenes:
            for i, scene in enumerate(artifact.scenes):
                if i < len(output.shot_descriptions):
                    scene = patch_scene_stage_shot(scene, [output.shot_descriptions[i]])
                    artifact.scenes[i] = scene

    elif step == GenerationStep.SHOT_DESCRIPTIONS:
        # This step patches scene shots - handled by generate_batch_shot_descriptions
        pass

    # Image generation steps update URLs in existing objects
    elif step == GenerationStep.COMPONENT_IMAGES:
        # Update specific prop with image URL
        pass  # Implementation depends on how we track which prop was generated

    return artifact


def patch_scene_stage_shot(scene: SceneSpec, shot_descriptions: List[str]) -> SceneSpec:
    """Patch a specific scene with stage setting shot descriptions.

    Args:
        scene: Scene to update
        shot_descriptions: List of shot descriptions (first one will be used)

    Returns:
        Updated scene
    """
    if shot_descriptions:
        # Create stage setting shot from first description
        scene.stage_setting_shot = ShotSpec(
            name=f"{scene.name} Stage Setting",
            description=shot_descriptions[0]
        )
    return scene


def patch_component_image_path(artifact: StoryboardSpec, component_type: str, component_name: str, image_path: str) -> StoryboardSpec:
    """Update a specific component with its generated image file path.

    Args:
        artifact: The current storyboard artifact
        component_type: Type of component ('character', 'location', 'prop')
        component_name: Name of the component
        image_path: Local file path to the generated image

    Returns:
        Updated artifact
    """
    if component_type == "character" and artifact.characters:
        for char in artifact.characters:
            if char.name == component_name:
                char.image_path = image_path
                break
    elif component_type == "location" and artifact.locations:
        for loc in artifact.locations:
            if loc.name == component_name:
                loc.image_path = image_path
                break
    elif component_type == "prop" and artifact.props:
        for prop in artifact.props:
            if prop.name == component_name:
                prop.image_path = image_path
                break
    return artifact


# Keep backwards compatibility
def patch_prop_image_path(artifact: StoryboardSpec, prop_name: str, image_path: str) -> StoryboardSpec:
    """Update a specific prop with its generated image file path.

    Backward compatibility wrapper for patch_component_image_path.
    """
    return patch_component_image_path(artifact, "prop", prop_name, image_path)


# ---------- Dependencies ----------

def get_dependencies():
    """Get step dependency graph.

    Returns a dictionary mapping each GenerationStep to its required dependencies.
    """
    # Import here to avoid circular dependency
    from .pipeline import GenerationStep

    return {
        GenerationStep.LOGLINE: [],
        GenerationStep.LORE: [GenerationStep.LOGLINE],
        GenerationStep.NARRATIVE: [GenerationStep.LORE],
        GenerationStep.SCENES: [GenerationStep.LORE, GenerationStep.NARRATIVE],
        GenerationStep.COMPONENT_DESCRIPTIONS: [GenerationStep.SCENES],
        GenerationStep.COMPONENT_IMAGES: [GenerationStep.COMPONENT_DESCRIPTIONS],
        GenerationStep.STAGE_SHOT_DESCRIPTIONS: [GenerationStep.SCENES, GenerationStep.COMPONENT_DESCRIPTIONS],
        GenerationStep.STAGE_SHOT_IMAGES: [GenerationStep.STAGE_SHOT_DESCRIPTIONS, GenerationStep.COMPONENT_IMAGES],
        GenerationStep.SHOT_DESCRIPTIONS: [GenerationStep.STAGE_SHOT_IMAGES, GenerationStep.COMPONENT_DESCRIPTIONS],
        GenerationStep.SHOT_IMAGES: [GenerationStep.SHOT_DESCRIPTIONS],
    }


# ---------- State Evaluation ----------

def get_completion_status(artifact: StoryboardSpec) -> Dict:
    """Check which generation steps have been completed.

    Args:
        artifact: The current storyboard artifact

    Returns:
        Dictionary mapping GenerationStep to completion status (bool)
    """
    # Import here to avoid circular dependency
    from .pipeline import GenerationStep

    return {
        GenerationStep.LOGLINE: artifact.logline is not None,
        GenerationStep.LORE: artifact.lore is not None,
        GenerationStep.NARRATIVE: artifact.narrative is not None,
        GenerationStep.SCENES: artifact.scenes is not None and len(artifact.scenes) > 0,
        GenerationStep.COMPONENT_DESCRIPTIONS: (
            artifact.characters is not None and len(artifact.characters) > 0 and
            artifact.locations is not None and len(artifact.locations) > 0 and
            artifact.props is not None and len(artifact.props) > 0
        ),
        GenerationStep.COMPONENT_IMAGES: (
            artifact.characters is not None and len(artifact.characters) > 0 and
            artifact.locations is not None and len(artifact.locations) > 0 and
            artifact.props is not None and len(artifact.props) > 0 and
            all(char.image_path is not None for char in artifact.characters) and
            all(loc.image_path is not None for loc in artifact.locations) and
            all(prop.image_path is not None for prop in artifact.props)
        ),
        GenerationStep.STAGE_SHOT_DESCRIPTIONS: (
            artifact.scenes is not None and
            len(artifact.scenes) > 0 and
            all(scene.stage_setting_shot is not None for scene in artifact.scenes)
        ),
        GenerationStep.STAGE_SHOT_IMAGES: (
            artifact.scenes is not None and
            len(artifact.scenes) > 0 and
            all(scene.stage_setting_shot is not None and
                scene.stage_setting_shot.image_path is not None
                for scene in artifact.scenes)
        ),
        GenerationStep.SHOT_DESCRIPTIONS: (
            artifact.scenes is not None and
            len(artifact.scenes) > 0 and
            all(scene.shots is not None and len(scene.shots) > 0 and
                all(shot.description is not None for shot in scene.shots)
                for scene in artifact.scenes)
        ),
        GenerationStep.SHOT_IMAGES: (
            artifact.scenes is not None and
            len(artifact.scenes) > 0 and
            all(scene.shots is not None and len(scene.shots) > 0 and
                all(shot.image_path is not None for shot in scene.shots)
                for scene in artifact.scenes)
        ),
    }


def get_next_steps(artifact: StoryboardSpec) -> List:
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