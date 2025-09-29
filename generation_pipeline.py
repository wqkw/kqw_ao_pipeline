from __future__ import annotations

from typing import List, Dict, Type, Any, get_origin, get_args
from pydantic import BaseModel, Field, ConfigDict, create_model
from pydantic.fields import FieldInfo
from enum import Enum

from prompts.storyboard_artifact import StoryboardSpec, SceneSpec, CharacterSpec, LocationSpec, PropSpec, SoundCue, MusicCue


# ---------- Base (forbid unknown keys for LLM outputs) ----------

class StrictModel(BaseModel):
    """Base model that rejects unknown fields to keep outputs clean."""
    model_config = ConfigDict(extra="forbid")


# ---------- Generation Steps ----------

class GenerationStep(Enum):
    LORE = "lore"
    NARRATIVE = "narrative"
    SCENES = "scenes"
    COMPONENT_DESCRIPTIONS = "component_descriptions"
    COMPONENT_IMAGES = "component_images"
    STAGE_SHOT_DESCRIPTIONS = "stage_shot_descriptions"
    STAGE_SHOT_IMAGES = "stage_shot_images"
    SHOT_IMAGES = "shot_images"


# ---------- Field Extraction Helper ----------

def get_field_desc(model_class: Type[BaseModel], field_name: str) -> str:
    """Extract field description from a Pydantic model."""
    field_info = model_class.model_fields[field_name]
    return field_info.description or f"{field_name} from {model_class.__name__}"


# ---------- Dynamic DTO Creation ----------

def create_context_dto(step: GenerationStep) -> Type[StrictModel]:
    """Create context DTO using descriptions from the core artifact."""
    fields = {"prompt_input": (str, Field(..., description="User's creative input for this generation step"))}

    if step == GenerationStep.LORE:
        fields["moodboard_path"] = (str, Field(..., description=get_field_desc(StoryboardSpec, "moodboard_path")))

    elif step == GenerationStep.NARRATIVE:
        fields["lore"] = (str, Field(..., description=get_field_desc(StoryboardSpec, "lore")))

    elif step == GenerationStep.SCENES:
        fields["lore"] = (str, Field(..., description=get_field_desc(StoryboardSpec, "lore")))
        fields["narrative"] = (str, Field(..., description=get_field_desc(StoryboardSpec, "narrative")))

    elif step == GenerationStep.COMPONENT_DESCRIPTIONS:
        fields["moodboard_path"] = (str, Field(..., description=get_field_desc(StoryboardSpec, "moodboard_path")))
        fields["lore"] = (str, Field(..., description=get_field_desc(StoryboardSpec, "lore")))
        fields["narrative"] = (str, Field(..., description=get_field_desc(StoryboardSpec, "narrative")))
        fields["scene_names"] = (List[str], Field(..., description="List of scene names from the story structure"))
        fields["scene_descriptions"] = (List[str], Field(..., description="List of scene descriptions from the story structure"))

    elif step == GenerationStep.COMPONENT_IMAGES:
        fields["moodboard_path"] = (str, Field(..., description=get_field_desc(StoryboardSpec, "moodboard_path")))
        fields["prop_name"] = (str, Field(..., description=get_field_desc(PropSpec, "name")))
        fields["prop_description"] = (str, Field(..., description=get_field_desc(PropSpec, "description")))

    elif step == GenerationStep.STAGE_SHOT_DESCRIPTIONS:
        fields["scene_name"] = (str, Field(..., description=get_field_desc(SceneSpec, "name")))
        fields["scene_description"] = (str, Field(..., description=get_field_desc(SceneSpec, "description")))
        fields["narrative"] = (str, Field(..., description=get_field_desc(StoryboardSpec, "narrative")))
        fields["prop_descriptions"] = (Dict[str, str], Field(..., description="Dictionary of prop names to descriptions"))

    elif step == GenerationStep.STAGE_SHOT_IMAGES:
        fields["stage_shot_description"] = (str, Field(..., description="Generated description of the stage setting shot"))
        fields["scene_description"] = (str, Field(..., description=get_field_desc(SceneSpec, "description")))
        fields["component_image_paths"] = (List[str], Field(..., description="URLs of component images"))

    elif step == GenerationStep.SHOT_IMAGES:
        fields["shot_description"] = (str, Field(..., description="Generated description of the shot"))
        fields["stage_setting_image_path"] = (str, Field(..., description="URL of the stage setting image"))

    return create_model(
        f"{step.value.title().replace('_', '')}Context",
        **fields,
        __base__=StrictModel
    )


def create_output_dto(step: GenerationStep) -> Type[StrictModel]:
    """Create output DTO using descriptions from the core artifact."""
    fields = {}

    if step == GenerationStep.LORE:
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
        fields["key_shot_sketches"] = (List[str], Field(..., description="List of key visual moments for each scene"))

    elif step == GenerationStep.COMPONENT_DESCRIPTIONS:
        fields["character_descriptions"] = (Dict[str, str], Field(..., description=f"Dictionary of {get_field_desc(CharacterSpec, 'name').lower()} to {get_field_desc(CharacterSpec, 'description').lower()}"))
        fields["location_descriptions"] = (Dict[str, str], Field(..., description=f"Dictionary of {get_field_desc(LocationSpec, 'name').lower()} to {get_field_desc(LocationSpec, 'description').lower()}"))
        fields["prop_descriptions"] = (Dict[str, str], Field(..., description=f"Dictionary of {get_field_desc(PropSpec, 'name').lower()} to {get_field_desc(PropSpec, 'description').lower()}"))
        fields["scene_to_characters"] = (Dict[str, List[str]], Field(..., description="Mapping of scene names to required characters"))
        fields["scene_to_locations"] = (Dict[str, List[str]], Field(..., description="Mapping of scene names to required locations"))
        fields["scene_to_props"] = (Dict[str, List[str]], Field(..., description="Mapping of scene names to required props"))

    elif step == GenerationStep.STAGE_SHOT_DESCRIPTIONS:
        fields["shot_descriptions"] = (List[str], Field(..., description="List of stage setting shot descriptions"))

    # Image steps don't need structured outputs
    else:
        return None

    return create_model(
        f"{step.value.title().replace('_', '')}Output",
        **fields,
        __base__=StrictModel
    )


# ---------- Patchers (Update Artifact from LLM Outputs) ----------

def patch_artifact(artifact: StoryboardSpec, step: GenerationStep, output: BaseModel, user_choice: int = 0) -> StoryboardSpec:
    """Update the artifact with LLM output. For multi-option outputs, user_choice selects which option."""

    if step == GenerationStep.LORE:
        options = [output.lore_option_1, output.lore_option_2, output.lore_option_3]
        artifact.lore = options[user_choice]

    elif step == GenerationStep.NARRATIVE:
        options = [output.narrative_option_1, output.narrative_option_2, output.narrative_option_3]
        artifact.narrative = options[user_choice]

    elif step == GenerationStep.SCENES:
        scenes = []
        for i, (name, desc) in enumerate(zip(output.scene_names, output.scene_descriptions)):
            scene = SceneSpec(name=name, description=desc)
            scenes.append(scene)
        artifact.scenes = scenes

    elif step == GenerationStep.COMPONENT_DESCRIPTIONS:
        # Create characters
        characters = []
        for name, desc in output.character_descriptions.items():
            character = CharacterSpec(name=name, description=desc)
            characters.append(character)
        artifact.characters = characters

        # Create locations
        locations = []
        for name, desc in output.location_descriptions.items():
            location = LocationSpec(name=name, description=desc)
            locations.append(location)
        artifact.locations = locations

        # Create props
        props = []
        for name, desc in output.prop_descriptions.items():
            prop = PropSpec(name=name, description=desc)
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

    # Image generation steps update URLs in existing objects
    elif step == GenerationStep.COMPONENT_IMAGES:
        # Update specific prop with image URL
        pass  # Implementation depends on how we track which prop was generated

    return artifact


def patch_scene_stage_shot(scene: SceneSpec, shot_descriptions: List[str]) -> SceneSpec:
    """Patch a specific scene with stage setting shot descriptions."""
    if shot_descriptions:
        # Create stage setting shot from first description
        from prompts.storyboard_artifact import ShotSpec
        scene.stage_setting_shot = ShotSpec(
            name=f"{scene.name} Stage Setting",
            description=shot_descriptions[0]
        )
    return scene


def patch_component_image_path(artifact: StoryboardSpec, component_type: str, component_name: str, image_path: str) -> StoryboardSpec:
    """Update a specific component with its generated image file path."""
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
    """Update a specific prop with its generated image file path."""
    return patch_component_image_path(artifact, "prop", prop_name, image_path)


# ---------- Dependencies ----------

DEPENDENCIES = {
    GenerationStep.LORE: [],
    GenerationStep.NARRATIVE: [GenerationStep.LORE],
    GenerationStep.SCENES: [GenerationStep.LORE, GenerationStep.NARRATIVE],
    GenerationStep.COMPONENT_DESCRIPTIONS: [GenerationStep.SCENES],
    GenerationStep.COMPONENT_IMAGES: [GenerationStep.COMPONENT_DESCRIPTIONS],
    GenerationStep.STAGE_SHOT_DESCRIPTIONS: [GenerationStep.SCENES, GenerationStep.COMPONENT_DESCRIPTIONS],
    GenerationStep.STAGE_SHOT_IMAGES: [GenerationStep.STAGE_SHOT_DESCRIPTIONS, GenerationStep.COMPONENT_IMAGES],
    GenerationStep.SHOT_IMAGES: [GenerationStep.STAGE_SHOT_IMAGES],
}


# ---------- State Evaluation ----------

def get_completion_status(artifact: StoryboardSpec) -> Dict[GenerationStep, bool]:
    """Check which generation steps have been completed."""
    return {
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
        GenerationStep.SHOT_IMAGES: (
            artifact.scenes is not None and
            len(artifact.scenes) > 0 and
            all(scene.shots is not None and len(scene.shots) > 0 and
                all(shot.image_path is not None for shot in scene.shots)
                for scene in artifact.scenes)
        ),
    }


def get_next_steps(artifact: StoryboardSpec) -> List[GenerationStep]:
    """Get the next available generation steps based on completed dependencies."""
    completion_status = get_completion_status(artifact)
    available_steps = []

    for step, deps in DEPENDENCIES.items():
        if not completion_status[step]:  # Step not completed
            if all(completion_status[dep] for dep in deps):  # All dependencies completed
                available_steps.append(step)

    return available_steps


# ---------- Generation Pipeline Orchestrator ----------

async def generate_step(artifact: StoryboardSpec, step: GenerationStep, prompt_input: str, **kwargs) -> tuple[StoryboardSpec, BaseModel]:
    """Execute a single generation step."""
    import asyncio
    from openrouter_wrapper import batch_llm

    # Create context DTO and populate with data
    ContextModel = create_context_dto(step)
    context_data = extract_context_data(artifact, step, prompt_input=prompt_input, **kwargs)
    context = ContextModel(**context_data)

    # Create output DTO
    OutputModel = create_output_dto(step)

    # Handle image generation steps
    if OutputModel is None:
        if step == GenerationStep.COMPONENT_IMAGES:
            responses, _, _ = await batch_llm(
                model="google/gemini-2.5-flash",  # Fast image generation
                texts=[f"Generate component: {context.prop_description}"],
                context="Create a high-quality component image for storyboarding",
                output_is_image=True
            )
            image_path = save_image_to_data(responses[0], artifact.name, "component", context.prop_name)
            artifact = patch_prop_image_path(artifact, context.prop_name, image_path)
            return artifact, None
        return artifact, None

    # For structured output steps - USE FAST MODEL WITHOUT REASONING
    responses, _, _ = await batch_llm(
        model="google/gemini-2.5-flash",  # Fast model instead of GPT-5
        texts=[prompt_input],
        context=f"Use this context: {context.model_dump_json()}",
        response_format=OutputModel,
        reasoning_effort="minimal"  # Minimal reasoning for speed
    )

    return artifact, responses[0]


def extract_context_data(artifact: StoryboardSpec, step: GenerationStep, **kwargs) -> Dict[str, Any]:
    """Extract context data from artifact based on step requirements."""
    data = kwargs.copy()  # Start with any provided kwargs

    # Add data from artifact based on what each step needs
    if step in [GenerationStep.LORE, GenerationStep.COMPONENT_DESCRIPTIONS, GenerationStep.COMPONENT_IMAGES]:
        if artifact.moodboard_path:
            data["moodboard_path"] = artifact.moodboard_path

    if step in [GenerationStep.NARRATIVE, GenerationStep.SCENES, GenerationStep.COMPONENT_DESCRIPTIONS]:
        if artifact.lore:
            data["lore"] = artifact.lore

    if step in [GenerationStep.SCENES, GenerationStep.COMPONENT_DESCRIPTIONS, GenerationStep.STAGE_SHOT_DESCRIPTIONS]:
        if artifact.narrative:
            data["narrative"] = artifact.narrative

    if step == GenerationStep.COMPONENT_DESCRIPTIONS:
        if artifact.scenes:
            data["scene_names"] = [scene.name for scene in artifact.scenes]
            data["scene_descriptions"] = [scene.description or "" for scene in artifact.scenes]

    if step == GenerationStep.STAGE_SHOT_DESCRIPTIONS:
        if artifact.props:
            data["prop_descriptions"] = {prop.name: prop.description or "" for prop in artifact.props}

    return data


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
    import os
    import time
    import re
    from datetime import datetime

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


def save_artifact_checkpoint(artifact: StoryboardSpec, step: GenerationStep) -> None:
    """Save intermediate artifact state for pipeline resumption."""
    import json
    import os

    # Create data directory for project
    checkpoint_dir = os.path.join("data", artifact.name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save artifact with step name
    checkpoint_path = os.path.join(checkpoint_dir, f"artifact_after_{step.value}.json")
    with open(checkpoint_path, "w") as f:
        json.dump(artifact.model_dump(), f, indent=2)

    print(f"Checkpoint saved: {checkpoint_path}")


async def generate_batch_stage_shots(artifact: StoryboardSpec, prompt_input: str) -> StoryboardSpec:
    """Generate stage setting shot descriptions for all scenes in parallel."""
    if not artifact.scenes:
        return artifact

    from openrouter_wrapper import batch_llm

    # Create context for each scene
    texts = []
    contexts = []
    for scene in artifact.scenes:
        context_data = extract_context_data(artifact, GenerationStep.STAGE_SHOT_DESCRIPTIONS,
                                          scene_name=scene.name, scene_description=scene.description)

        ContextModel = create_context_dto(GenerationStep.STAGE_SHOT_DESCRIPTIONS)
        context = ContextModel(**context_data, prompt_input=prompt_input)

        texts.append(f"Generate stage setting shot description for scene: {scene.name}")
        contexts.append(f"Context: {context.model_dump_json()}")

    # Generate descriptions using structured output
    OutputModel = create_output_dto(GenerationStep.STAGE_SHOT_DESCRIPTIONS)

    # Batch all scene stage shots together
    print(f"🎬 Generating stage setting shots for {len(artifact.scenes)} scenes")

    # Prepare batch data
    texts = []
    scene_contexts = []

    for scene in artifact.scenes:
        context_data = extract_context_data(artifact, GenerationStep.STAGE_SHOT_DESCRIPTIONS,
                                          scene_name=scene.name, scene_description=scene.description)
        ContextModel = create_context_dto(GenerationStep.STAGE_SHOT_DESCRIPTIONS)
        context = ContextModel(**context_data, prompt_input=prompt_input)

        texts.append(f"Generate a single stage setting shot description for scene: {scene.name}")
        scene_contexts.append(context.model_dump_json())

    print(f"  📍 Scenes: {[scene.name for scene in artifact.scenes]}")

    try:
        # Use single context string since all scenes share similar context structure
        batch_context = f"Generate one stage setting shot description for each scene. Context format: scene data including narrative, props, etc."

        responses, _, _ = await batch_llm(
            model="google/gemini-2.5-flash",
            texts=texts,
            context=batch_context,
            response_format=OutputModel,
            reasoning_effort="minimal"
        )

        # Process responses and update scenes
        from prompts.storyboard_artifact import ShotSpec
        successful_scenes = 0

        for i, (scene, response) in enumerate(zip(artifact.scenes, responses)):
            try:
                if hasattr(response, 'shot_descriptions') and response.shot_descriptions:
                    shot_descriptions = response.shot_descriptions
                    if shot_descriptions and len(shot_descriptions) > 0:
                        scene.stage_setting_shot = ShotSpec(
                            name=f"{scene.name} Stage Setting",
                            description=shot_descriptions[0]
                        )
                        print(f"    ✅ Scene {i+1}: {shot_descriptions[0][:50]}...")
                        successful_scenes += 1
                    else:
                        print(f"    ❌ Scene {i+1}: No descriptions in response")
                else:
                    print(f"    ❌ Scene {i+1}: Invalid response format")
            except Exception as e:
                print(f"    ❌ Scene {i+1}: Processing failed - {str(e)[:50]}")

        print(f"  📊 Stage shots created: {successful_scenes}/{len(artifact.scenes)}")

    except Exception as e:
        print(f"  ❌ Batch generation failed: {str(e)[:100]}")
        print("  Falling back to individual processing...")

        # Fallback: process individually if batch fails
        for i, scene in enumerate(artifact.scenes):
            try:
                context_data = extract_context_data(artifact, GenerationStep.STAGE_SHOT_DESCRIPTIONS,
                                                  scene_name=scene.name, scene_description=scene.description)
                ContextModel = create_context_dto(GenerationStep.STAGE_SHOT_DESCRIPTIONS)
                context = ContextModel(**context_data, prompt_input=prompt_input)

                response, _, _ = await batch_llm(
                    model="google/gemini-2.5-flash",
                    texts=[f"Generate stage setting shot description for scene: {scene.name}"],
                    context=f"Context: {context.model_dump_json()}",
                    response_format=OutputModel,
                    reasoning_effort="minimal"
                )

                if response and len(response) > 0 and hasattr(response[0], 'shot_descriptions'):
                    shot_descriptions = response[0].shot_descriptions
                    if shot_descriptions and len(shot_descriptions) > 0:
                        from prompts.storyboard_artifact import ShotSpec
                        scene.stage_setting_shot = ShotSpec(
                            name=f"{scene.name} Stage Setting",
                            description=shot_descriptions[0]
                        )
                        print(f"    ✅ Fallback - Scene {i+1}: Created stage shot")

            except Exception as fallback_e:
                print(f"    ❌ Fallback - Scene {i+1}: {str(fallback_e)[:50]}")
                continue

    return artifact


async def generate_stage_shot_images(artifact: StoryboardSpec, prompt_input: str) -> StoryboardSpec:
    """Generate stage setting images for all scenes in parallel."""
    if not artifact.scenes:
        return artifact

    from openrouter_wrapper import batch_llm

    # Create context for each scene that has a stage_setting_shot description
    texts = []
    contexts = []
    scene_indices = []

    for i, scene in enumerate(artifact.scenes):
        if scene.stage_setting_shot and scene.stage_setting_shot.description:
            # Get component image URLs for this scene
            component_image_paths = []
            if scene.prop_names and artifact.props:
                for prop_name in scene.prop_names:
                    for prop in artifact.props:
                        if prop.name == prop_name and prop.image_path:
                            component_image_paths.append(prop.image_path)

            # Create context data
            context_data = {
                "stage_shot_description": scene.stage_setting_shot.description,
                "scene_description": scene.description or "",
                "component_image_paths": component_image_paths
            }

            ContextModel = create_context_dto(GenerationStep.STAGE_SHOT_IMAGES)
            context = ContextModel(**context_data, prompt_input=prompt_input)

            texts.append(f"Generate stage setting image for scene: {scene.name}")
            contexts.append(f"Context: {context.model_dump_json()}")
            scene_indices.append(i)

    if not texts:
        return artifact

    # Generate images using Gemini 2.5 Flash in batch
    print(f"🎨 Generating {len(texts)} stage setting images in batch...")
    print(f"  📍 Scenes: {[artifact.scenes[i].name for i in scene_indices]}")

    try:
        responses, _, _ = await batch_llm(
            model="google/gemini-2.5-flash-image-preview",
            texts=texts,
            context="Generate stage setting images for fantasy scenes. Create detailed environmental illustrations.",
            output_is_image=True,
            image_generation_retries=2
        )
        print(f"  ✅ Batch generation completed successfully")

    except Exception as e:
        print(f"  ❌ Batch generation failed: {str(e)[:100]}")
        print(f"  📝 Debug info should be saved to image_generation_debug.txt")
        # Return empty responses to maintain alignment
        responses = [None] * len(texts)

    # Save images and update scenes
    successful_images = 0
    failed_images = 0

    for scene_idx, image_bytes in zip(scene_indices, responses):
        scene = artifact.scenes[scene_idx]
        if scene.stage_setting_shot and image_bytes is not None:
            try:
                image_path = save_image_to_data(image_bytes, artifact.name, "stage_setting", scene.name)
                scene.stage_setting_shot.image_path = image_path
                print(f"  ✅ Saved: {scene.name}")
                successful_images += 1
            except Exception as e:
                print(f"  ❌ Save failed for {scene.name}: {str(e)[:50]}")
                failed_images += 1
        else:
            print(f"  ❌ No image for {scene.name}")
            failed_images += 1

    print(f"\n📊 Stage shot images: {successful_images} success, {failed_images} failed")

    return artifact


async def generate_batch_components(artifact: StoryboardSpec, prompt_input: str) -> StoryboardSpec:
    """Generate all component images (characters, locations, props) in parallel."""
    from openrouter_wrapper import batch_llm

    # Collect all components to generate
    components = []

    # Add characters (only if they have descriptions)
    if artifact.characters:
        for char in artifact.characters:
            if char.description and char.description.strip():
                components.append(("character", char.name, char.description))

    # Add locations (only if they have descriptions)
    if artifact.locations:
        for loc in artifact.locations:
            if loc.description and loc.description.strip():
                components.append(("location", loc.name, loc.description))

    # Add props (only if they have descriptions)
    if artifact.props:
        for prop in artifact.props:
            if prop.description and prop.description.strip():
                components.append(("prop", prop.name, prop.description))

    if not components:
        return artifact

    # Create texts for batch generation
    texts = []
    for comp_type, name, description in components:
        # Make the prompt more explicit and visual-focused for image generation
        if comp_type == "character":
            # Extract visual elements from character descriptions
            prompt = f"Draw a fantasy character: {name}. Visual appearance: {description}. Style: concept art, character design, clear details, storyboard illustration."
        elif comp_type == "location":
            # Focus on environmental visuals
            prompt = f"Draw a fantasy location: {name}. Environment: {description}. Style: concept art, environmental design, clear details, storyboard illustration."
        else:  # prop
            # Focus on object visuals
            prompt = f"Draw a fantasy object: {name}. Item appearance: {description}. Style: concept art, prop design, clear details, storyboard illustration."
        texts.append(prompt)

    print(f"🎨 Generating {len(components)} component images:")
    for i, (comp_type, name, description) in enumerate(components):
        print(f"  {i+1}. {comp_type}: {name}")
        print(f"     Description: {description[:100]}...")

    context_str = "Generate visual concept art images. Create detailed illustrations based on the fantasy descriptions. Output: image only, no text or descriptions."

    # Split into 3 batches to avoid overwhelming the API
    batch_size = len(components) // 3 + (1 if len(components) % 3 > 0 else 0)

    all_responses = []

    for batch_num in range(3):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(components))

        if start_idx >= len(components):
            break

        batch_components = components[start_idx:end_idx]
        batch_texts = texts[start_idx:end_idx]

        print(f"🎨 Processing batch {batch_num + 1}/3 ({len(batch_components)} components)")
        print(f"  Components: {[f'{c[0]}-{c[1]}' for c in batch_components]}")

        try:
            responses, full_responses, _ = await batch_llm(
                model="google/gemini-2.5-flash-image-preview",
                texts=batch_texts,
                context=context_str,
                output_is_image=True,
                image_generation_retries=2
            )
            all_responses.extend(responses)
            print(f"  ✅ Batch {batch_num + 1} completed successfully")

        except Exception as e:
            print(f"  ❌ Batch {batch_num + 1} failed: {str(e)[:200]}")
            print(f"  Batch texts preview:")
            for i, text in enumerate(batch_texts[:3]):  # Show first 3 prompts
                print(f"    {i+1}. {text[:100]}...")

            # Debug: Try to get the actual LLM responses for failed batch
            try:
                debug_responses, debug_full_responses, _ = await batch_llm(
                    model="google/gemini-2.5-flash-image-preview",
                    texts=batch_texts,
                    context=context_str,
                    output_is_image=False  # Get text response to see what model returned
                )

                # Write debug info to file
                import json
                import datetime

                debug_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "batch_number": batch_num + 1,
                    "error": str(e),
                    "prompts": batch_texts,
                    "context": context_str,
                    "components": [f"{c[0]}-{c[1]}" for c in batch_components],
                    "text_responses": debug_responses,
                    "full_responses": debug_full_responses
                }

                with open("debug_log.txt", "a") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"FAILED BATCH DEBUG - {debug_data['timestamp']}\n")
                    f.write(f"{'='*80}\n")
                    f.write(json.dumps(debug_data, indent=2))
                    f.write(f"\n{'='*80}\n")

                print(f"  📝 Debug info written to debug_log.txt")

            except Exception as debug_e:
                print(f"  ⚠️  Debug logging failed: {str(debug_e)[:100]}")

            # Add empty responses for failed batch to maintain alignment
            all_responses.extend([None] * len(batch_components))

    # Save images and update components
    successful_generations = 0
    failed_generations = 0

    for (comp_type, name, description), image_bytes in zip(components, all_responses):
        if image_bytes is not None:
            try:
                image_path = save_image_to_data(image_bytes, artifact.name, comp_type, name)
                artifact = patch_component_image_path(artifact, comp_type, name, image_path)
                successful_generations += 1
                print(f"  ✅ Saved: {comp_type} - {name}")
            except Exception as e:
                print(f"  ❌ Save failed for {comp_type} - {name}: {str(e)[:50]}")
                failed_generations += 1
        else:
            failed_generations += 1
            print(f"  ❌ No image for {comp_type} - {name}")

    print(f"\n📊 Final result: {successful_generations} success, {failed_generations} failed")

    return artifact


async def run_generation_pipeline(artifact: StoryboardSpec, user_inputs: Dict[GenerationStep, str]) -> StoryboardSpec:
    """Run the complete generation pipeline."""
    while True:
        next_steps = get_next_steps(artifact)
        if not next_steps:
            break

        for step in next_steps:
            if step not in user_inputs:
                print(f"Waiting for user input for step: {step.value}")
                continue

            print(f"Generating {step.value}...")

            # Handle batch generation steps
            if step == GenerationStep.COMPONENT_IMAGES:
                artifact = await generate_batch_components(artifact, user_inputs[step])
            elif step == GenerationStep.STAGE_SHOT_DESCRIPTIONS:
                artifact = await generate_batch_stage_shots(artifact, user_inputs[step])
            elif step == GenerationStep.STAGE_SHOT_IMAGES:
                artifact = await generate_stage_shot_images(artifact, user_inputs[step])

            # Handle single-generation steps
            else:
                artifact, output = await generate_step(artifact, step, user_inputs[step])
                if output:
                    artifact = patch_artifact(artifact, step, output, user_choice=0)

            print(f"Completed {step.value}")

            # Save checkpoint
            save_artifact_checkpoint(artifact, step)

    return artifact