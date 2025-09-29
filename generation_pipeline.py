from __future__ import annotations

from typing import List, Dict, Type, get_origin, get_args
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
        fields["component_image_urls"] = (List[str], Field(..., description="URLs of component images"))

    elif step == GenerationStep.SHOT_IMAGES:
        fields["shot_description"] = (str, Field(..., description="Generated description of the shot"))
        fields["stage_setting_image_url"] = (str, Field(..., description="URL of the stage setting image"))

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
        fields["prop_descriptions"] = (Dict[str, str], Field(..., description=f"Dictionary of {get_field_desc(PropSpec, 'name').lower()} to {get_field_desc(PropSpec, 'description').lower()}"))
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
        props = []
        for name, desc in output.prop_descriptions.items():
            prop = PropSpec(name=name, description=desc)
            props.append(prop)
        artifact.props = props

        # Update scenes with prop references
        if artifact.scenes:
            for scene in artifact.scenes:
                if scene.name in output.scene_to_props:
                    scene.prop_names = output.scene_to_props[scene.name]

    elif step == GenerationStep.STAGE_SHOT_DESCRIPTIONS:
        # This is per-scene, so we need scene context
        pass  # Implementation depends on how we handle per-scene processing

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


def patch_prop_image_url(artifact: StoryboardSpec, prop_name: str, image_url: str) -> StoryboardSpec:
    """Update a specific prop with its generated image URL."""
    if artifact.props:
        for prop in artifact.props:
            if prop.name == prop_name:
                prop.image_url = image_url
                break
    return artifact


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
        GenerationStep.COMPONENT_DESCRIPTIONS: artifact.props is not None and len(artifact.props) > 0,
        GenerationStep.COMPONENT_IMAGES: (
            artifact.props is not None and
            len(artifact.props) > 0 and
            all(prop.image_url is not None for prop in artifact.props)
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
                scene.stage_setting_shot.image_url is not None
                for scene in artifact.scenes)
        ),
        GenerationStep.SHOT_IMAGES: (
            artifact.scenes is not None and
            len(artifact.scenes) > 0 and
            all(scene.shots is not None and len(scene.shots) > 0 and
                all(shot.image_url is not None for shot in scene.shots)
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
                model="google/gemini-2.5-flash",
                texts=[f"Generate component: {context.prop_description}"],
                context=["Create a high-quality component image for storyboarding"],
                output_is_image=True
            )
            image_url = save_image_and_get_url(responses[0], f"{context.prop_name}.png")
            artifact = patch_prop_image_url(artifact, context.prop_name, image_url)
            return artifact, None
        return artifact, None

    # For structured output steps
    responses, _, _ = await batch_llm(
        model="openai/gpt-5",
        texts=[prompt_input],
        context=[f"Use this context: {context.model_dump_json()}"],
        response_format=OutputModel
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


def save_image_and_get_url(image_bytes: bytes, filename: str) -> str:
    """Save image and return URL. Implementation depends on storage choice."""
    # Placeholder - implement based on whether using local storage, S3, etc.
    import os
    os.makedirs("generated_images", exist_ok=True)
    filepath = f"generated_images/{filename}"
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return filepath


async def generate_batch_components(artifact: StoryboardSpec, prompt_input: str) -> StoryboardSpec:
    """Generate all component images in parallel."""
    if not artifact.props:
        return artifact

    from openrouter_wrapper import batch_llm

    texts = [f"Generate component: {prop.description}" for prop in artifact.props]
    contexts = ["Create a high-quality component image for storyboarding"] * len(texts)

    responses, _, _ = await batch_llm(
        model="google/gemini-2.5-flash",
        texts=texts,
        context=contexts,
        output_is_image=True
    )

    for prop, image_bytes in zip(artifact.props, responses):
        image_url = save_image_and_get_url(image_bytes, f"{prop.name}.png")
        artifact = patch_prop_image_url(artifact, prop.name, image_url)

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

            # Handle single-generation steps
            else:
                artifact, output = await generate_step(artifact, step, user_inputs[step])
                if output:
                    artifact = patch_artifact(artifact, step, output, user_choice=0)

            print(f"Completed {step.value}")

    return artifact