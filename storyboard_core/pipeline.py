"""
Storyboard Generation Pipeline

This module orchestrates the multi-step generation pipeline for creating storyboards,
including text generation (logline, lore, narrative, scenes) and image generation
(components, stage shots, individual shots).
"""

from __future__ import annotations

import os
from typing import List, Dict, Tuple
from pydantic import BaseModel, ConfigDict  
from enum import Enum

from openrouter_wrapper import llm

from .artifact import StoryboardSpec, SceneSpec, ShotSpec, StrictModel
from .artifact_adapters import (
    extract_context_dto,
    context_dto_to_string,
    create_output_dto,
    patch_artifact,
    patch_component_image_path,
    get_next_steps
)
from .utils import save_image_to_data, save_multi_option_output, save_artifact_checkpoint


# ---------- Generation Steps ----------

class GenerationStep(Enum):
    LOGLINE = "logline"
    LORE = "lore"
    NARRATIVE = "narrative"
    SCENES = "scenes"
    COMPONENT_DESCRIPTIONS = "component_descriptions"
    COMPONENT_IMAGES = "component_images"
    STAGE_SHOT_DESCRIPTIONS = "stage_shot_descriptions"
    STAGE_SHOT_IMAGES = "stage_shot_images"
    SHOT_DESCRIPTIONS = "shot_descriptions"
    SHOT_IMAGES = "shot_images"


# ---------- Model Selection ----------

def _select_model_for_step(step: GenerationStep) -> Tuple[str, str]:
    """Select model and reasoning effort based on step requirements.

    Currently uses fast models for all steps:
    - Image generation: gemini-2.5-flash-image-preview
    - Text generation: gemini-2.5-flash with minimal reasoning

    Can be easily updated to use gpt-5 + medium reasoning for lore/narrative/scenes.

    Args:
        step: The generation step to select a model for

    Returns:
        Tuple of (model_name, reasoning_effort)
        Note: reasoning_effort is ignored for image generation steps
    """
    # Image generation steps use image-preview model
    if step in [GenerationStep.COMPONENT_IMAGES, GenerationStep.STAGE_SHOT_IMAGES, GenerationStep.SHOT_IMAGES]:
        return "google/gemini-2.5-flash-image-preview", "minimal"

    # Lore generation uses GPT-5 with medium reasoning
    if step in [GenerationStep.LORE, GenerationStep.NARRATIVE]:
        return "openai/gpt-5", "minimal"

    # Other text generation steps use fast model
    # To use GPT-5 for narrative/scenes as well, uncomment this:
    # if step in [GenerationStep.NARRATIVE, GenerationStep.SCENES]:
    #     return "openai/gpt-5", "medium"
    return "google/gemini-2.5-flash", "minimal"


# ---------- Guide Readers ----------

def _read_guide(guide_path: str) -> str:
    """Read a generation guide from the prompts directory.

    Args:
        guide_path: Path to the guide file (e.g., "prompts/lore_guide.md")

    Returns:
        Content of the guide, or empty string if file not found
    """
    try:
        with open(guide_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Could not find {guide_path}")
        return ""


# ---------- Generation Pipeline Orchestrator ----------

async def generate_step(artifact: StoryboardSpec, step: GenerationStep, prompt_input: str, **kwargs) -> Tuple[StoryboardSpec, BaseModel]:
    """Execute a single generation step.

    Args:
        artifact: Current storyboard artifact
        step: Generation step to execute
        prompt_input: User's creative input for this step
        **kwargs: Additional context data

    Returns:
        Tuple of (updated_artifact, llm_output)
    """
    # Extract context DTO with populated data
    context = extract_context_dto(artifact, step, prompt_input=prompt_input, **kwargs)

    # Create output DTO
    OutputModel = create_output_dto(step)

    # Select model and reasoning effort based on step
    model, reasoning = _select_model_for_step(step)

    # Special handling for LOGLINE step: include moodboard image and logline guide
    if step == GenerationStep.LOGLINE:
        moodboard_path = context.moodboard_path
        guide = _read_guide("prompts/logline_guide.md")

        response, _, _ = llm(
            model=model,
            text=prompt_input,
            context=guide,
            input_image_path=moodboard_path,
            response_format=OutputModel,
            reasoning_effort=reasoning
        )

    # Special handling for LORE step: include moodboard image and lore guide
    elif step == GenerationStep.LORE:
        moodboard_path = context.moodboard_path
        context_str = f"logline: {context.logline}\n\n{_read_guide("prompts/lore_guide.md")}"

        response, _, _ = llm(
            model=model,
            text=prompt_input,
            context=context_str,
            input_image_path=moodboard_path,
            response_format=OutputModel,
            reasoning_effort=reasoning
        )

    # Special handling for NARRATIVE step: use narrative guide
    elif step == GenerationStep.NARRATIVE:
        context_str = f"logline: {context.logline}\n\nlore: {context.lore}\n\n{_read_guide('prompts/narrative_guide.md')}"

        response, _, _ = llm(
            model=model,
            text=prompt_input,
            context=context_str,
            response_format=OutputModel,
            reasoning_effort=reasoning
        )

    # Special handling for SCENES step: use scenes guide
    elif step == GenerationStep.SCENES:
        context_str = f"logline: {context.logline}\n\nlore: {context.lore}\n\nnarrative: {context.narrative}\n\n{_read_guide('prompts/scenes_guide.md')}"

        response, _, _ = llm(
            model=model,
            text=prompt_input,
            context=context_str,
            response_format=OutputModel,
            reasoning_effort=reasoning
        )

    # Special handling for COMPONENT_DESCRIPTIONS step: use components guide with all moodboard images
    elif step == GenerationStep.COMPONENT_DESCRIPTIONS:
        from pathlib import Path

        scene_names_str = "\n  - ".join(context.scene_names)
        scene_descriptions_str = "\n".join([f"  - {name}: {desc}" for name, desc in zip(context.scene_names, context.scene_descriptions)])

        # Collect all moodboard images (excluding moodboard_tile.png)
        moodboard_dir = Path("data/ref_moodboard")
        moodboard_images = sorted([
            str(img) for img in moodboard_dir.glob("*.png")
            if img.name != "moodboard_tile.png"
        ])

        # Build prompt with filename identifiers (like test_gemini_filename_recognition.py)
        image_metadata_str = "Reference images with filenames:\n"
        for img_path in moodboard_images:
            filename = Path(img_path).name
            image_metadata_str += f"- {filename}\n"

        context_str = f"logline: {context.logline}\n\nlore: {context.lore}\n\nnarrative: {context.narrative}\n\nscene_names:\n  - {scene_names_str}\n\nscene_descriptions:\n{scene_descriptions_str}\n\n{image_metadata_str}\n\nIMPORTANT: When selecting reference images, you must use the EXACT filename from the list above. Copy the filename character-by-character precisely. Each filename is unique. Do not modify or invent new filenames.\n\n{_read_guide('prompts/components_guide.md')}"

        response, _, _ = llm(
            model=model,
            text=prompt_input,
            context=context_str,
            input_image_path=moodboard_images,
            response_format=OutputModel,
            reasoning_effort=reasoning
        )

    else:
        # Standard generation for other steps
        context_str = context_dto_to_string(context)
        response, _, _ = llm(
            model=model,
            text=prompt_input,
            context=context_str,
            response_format=OutputModel,
            reasoning_effort=reasoning
        )

    return artifact, response


async def generate_batch_stage_shots(artifact: StoryboardSpec, prompt_input: str) -> StoryboardSpec:
    """Generate stage setting shot descriptions for all scenes in parallel.

    Args:
        artifact: Current storyboard artifact
        prompt_input: User's creative input

    Returns:
        Updated artifact with stage setting shots
    """
    if not artifact.scenes:
        return artifact

    from openrouter_wrapper import batch_llm

    # Get model and reasoning effort
    model, reasoning = _select_model_for_step(GenerationStep.STAGE_SHOT_DESCRIPTIONS)
    OutputModel = create_output_dto(GenerationStep.STAGE_SHOT_DESCRIPTIONS)

    print(f"🎬 Generating stage setting shots for {len(artifact.scenes)} scenes")
    print(f"  📍 Scenes: {[scene.name for scene in artifact.scenes]}")

    # Prepare batch data with image references
    texts = []
    image_paths_list = []

    for scene in artifact.scenes:
        # Collect character and prop images for this scene
        scene_images = []

        # Add character images
        if scene.character_names and artifact.characters:
            for char_name in scene.character_names:
                for char in artifact.characters:
                    if char.name == char_name and char.image_path:
                        scene_images.append(char.image_path)
                        break

        # Add prop images
        if scene.prop_names and artifact.props:
            for prop_name in scene.prop_names:
                for prop in artifact.props:
                    if prop.name == prop_name and prop.image_path:
                        scene_images.append(prop.image_path)
                        break

        texts.append(f"Generate a single stage setting shot description for scene: {scene.name}\n\nScene: {scene.description}")
        image_paths_list.append(scene_images if scene_images else None)

    # Generate stage shots
    batch_context = "Generate one stage setting shot description for each scene. Use the provided character and prop images as visual reference. Focus on establishing the environment and atmosphere."

    responses, _, _ = await batch_llm(
        model=model,
        texts=texts,
        context=batch_context,
        image_paths=image_paths_list,
        response_format=OutputModel,
        reasoning_effort=reasoning
    )

    # Process responses and update scenes
    successful_scenes = 0
    for i, (scene, response) in enumerate(zip(artifact.scenes, responses)):
        if response and hasattr(response, 'shot_descriptions') and response.shot_descriptions:
            scene.stage_setting_shot = ShotSpec(
                name=f"{scene.name} Stage Setting",
                description=response.shot_descriptions[0]
            )
            print(f"    ✅ Scene {i+1}: {response.shot_descriptions[0][:50]}...")
            successful_scenes += 1
        else:
            print(f"    ❌ Scene {i+1}: No description generated")

    print(f"  📊 Stage shots created: {successful_scenes}/{len(artifact.scenes)}")

    return artifact

async def generate_batch_shot_descriptions(artifact: StoryboardSpec, prompt_input: str) -> StoryboardSpec:
    """Generate individual shot descriptions for all scenes in parallel.

    Args:
        artifact: Current storyboard artifact
        prompt_input: User's creative input

    Returns:
        Updated artifact with shot descriptions
    """
    if not artifact.scenes:
        return artifact

    from openrouter_wrapper import batch_llm

    # Get model and reasoning effort
    model, reasoning = _select_model_for_step(GenerationStep.SHOT_DESCRIPTIONS)
    OutputModel = create_output_dto(GenerationStep.SHOT_DESCRIPTIONS)

    print(f"🎬 Generating shot descriptions for {len(artifact.scenes)} scenes")
    print(f"  📍 Scenes: {[scene.name for scene in artifact.scenes]}")

    # Prepare batch data
    texts = []
    for scene in artifact.scenes:
        texts.append(f"Generate exactly 3 individual shot descriptions for scene: {scene.name}\n\nScene: {scene.description}")

    # Generate shot descriptions
    batch_context = "Generate exactly 3 shot descriptions for each scene. Each shot should break down a key moment in the scene with specific camera angles and framing."

    responses, _, _ = await batch_llm(
        model=model,
        texts=texts,
        context=batch_context,
        response_format=OutputModel,
        reasoning_effort=reasoning
    )

    # Process responses and update scenes
    successful_scenes = 0
    for i, (scene, response) in enumerate(zip(artifact.scenes, responses)):
        if response and hasattr(response, 'shot_descriptions') and response.shot_descriptions:
            scene.shots = []
            for j, shot_desc in enumerate(response.shot_descriptions):
                scene.shots.append(ShotSpec(
                    name=f"{scene.name} Shot {j+1}",
                    description=shot_desc
                ))
            print(f"    ✅ Scene {i+1}: {len(response.shot_descriptions)} shots created")
            successful_scenes += 1
        else:
            print(f"    ❌ Scene {i+1}: No descriptions generated")

    print(f"  📊 Shot descriptions created: {successful_scenes}/{len(artifact.scenes)}")

    return artifact


async def generate_stage_shot_images(artifact: StoryboardSpec, prompt_input: str) -> StoryboardSpec:
    """Generate stage setting images for all scenes in parallel.

    Args:
        artifact: Current storyboard artifact
        prompt_input: User's creative input

    Returns:
        Updated artifact with stage setting images
    """
    if not artifact.scenes:
        return artifact

    from openrouter_wrapper import batch_llm

    # Get model
    model, _ = _select_model_for_step(GenerationStep.STAGE_SHOT_IMAGES)

    # Prepare batch data with image references
    texts = []
    image_paths_list = []
    scene_indices = []

    for i, scene in enumerate(artifact.scenes):
        if scene.stage_setting_shot and scene.stage_setting_shot.description:
            # Collect character and prop images for this scene
            component_images = []

            # Add character images
            if scene.character_names and artifact.characters:
                for char_name in scene.character_names:
                    for char in artifact.characters:
                        if char.name == char_name and char.image_path:
                            component_images.append(char.image_path)
                            break

            # Add prop images
            if scene.prop_names and artifact.props:
                for prop_name in scene.prop_names:
                    for prop in artifact.props:
                        if prop.name == prop_name and prop.image_path:
                            component_images.append(prop.image_path)
                            break

            texts.append(f"{scene.stage_setting_shot.description}")
            image_paths_list.append(component_images if component_images else None)
            scene_indices.append(i)

    if not texts:
        return artifact

    print(f"🎨 Generating {len(texts)} stage setting images in batch...")
    print(f"  📍 Scenes: {[artifact.scenes[i].name for i in scene_indices]}")

    responses, _, _ = await batch_llm(
        model=model,
        texts=texts,
        context="Generate stage setting images. Use the provided character and prop images as visual reference for style and composition. Create detailed environmental illustrations that establish the scene's atmosphere.",
        image_paths=image_paths_list,
        output_is_image=True,
        image_generation_retries=2
    )

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
    """Generate all component images (characters, locations, props) in parallel.

    Args:
        artifact: Current storyboard artifact
        prompt_input: User's creative input

    Returns:
        Updated artifact with component images
    """
    from openrouter_wrapper import batch_llm

    # Extract context using DTO
    context = extract_context_dto(artifact, GenerationStep.COMPONENT_IMAGES, prompt_input=prompt_input)
    moodboard_path = context.moodboard_path

    # Collect all components to generate with their moodboard references
    components = []

    # Add characters (only if they have descriptions)
    if context.characters:
        for char in context.characters:
            if char.description and char.description.strip():
                moodboard_ref = getattr(char, 'image_reference_from_moodboard', None) or moodboard_path
                components.append(("character", char.name, char.description, moodboard_ref))

    # Add locations (only if they have descriptions)
    if context.locations:
        for loc in context.locations:
            if loc.description and loc.description.strip():
                moodboard_ref = getattr(loc, 'image_reference_from_moodboard', None) or moodboard_path
                components.append(("location", loc.name, loc.description, moodboard_ref))

    # Add props (only if they have descriptions)
    if context.props:
        for prop in context.props:
            if prop.description and prop.description.strip():
                moodboard_ref = getattr(prop, 'image_reference_from_moodboard', None) or moodboard_path
                components.append(("prop", prop.name, prop.description, moodboard_ref))

    if not components:
        return artifact

    # Create texts and image paths for batch generation
    texts = []
    image_paths = []
    for comp_type, name, description, moodboard_ref in components:
        # Make the prompt more explicit and visual-focused for image generation
        if comp_type == "character":
            # Extract visual elements from character descriptions
            prompt = f"Generate image for a character: {name}. Visual appearance: {description}. Style: copy the aesthetics of the attached reference image."
        elif comp_type == "location":
            # Focus on environmental visuals
            prompt = f"Generate image for a location: {name}. Environment: {description}. Style: copy the aesthetics of the attached reference image."
        else:  # prop
            # Focus on object visuals
            prompt = f"Generate image for an object: {name}. Item appearance: {description}. Style: copy the aesthetics of the attached reference image."
        texts.append(prompt)
        image_paths.append(moodboard_ref)

    print(f"🎨 Generating {len(components)} component images:")
    for i, (comp_type, name, description, moodboard_ref) in enumerate(components):
        print(f"  {i+1}. {comp_type}: {name}")
        print(f"     Description: {description[:100]}...")
        print(f"     Reference: {moodboard_ref}")

    # Build context string with guide
    context_str = f"logline: {context.logline}\n\nlore: {context.lore}\n\nnarrative: {context.narrative}\n\n{_read_guide('prompts/components_guide.md')}\n\nGenerate visual concept art images. Create detailed illustrations based on the descriptions. Match the style of the reference moodboard image. Output: image only, no text or descriptions."

    # Select model for image generation
    model, _ = _select_model_for_step(GenerationStep.COMPONENT_IMAGES)

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
        batch_image_paths = image_paths[start_idx:end_idx]

        print(f"🎨 Processing batch {batch_num + 1}/3 ({len(batch_components)} components)")
        print(f"  Components: {[f'{c[0]}-{c[1]}' for c in batch_components]}")

        try:
            responses, full_responses, _ = await batch_llm(
                model=model,
                texts=batch_texts,
                context=context_str,
                image_paths=batch_image_paths,
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
                    model=model,
                    texts=batch_texts,
                    context=context_str,
                    image_paths=batch_image_paths,
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

    for (comp_type, name, description, moodboard_ref), image_bytes in zip(components, all_responses):
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


async def generate_shot_images(artifact: StoryboardSpec, prompt_input: str) -> StoryboardSpec:
    """Generate individual shot images for all scenes in parallel.

    Args:
        artifact: Current storyboard artifact
        prompt_input: User's creative input

    Returns:
        Updated artifact with shot images
    """
    if not artifact.scenes:
        return artifact

    from openrouter_wrapper import batch_llm

    # Collect all shots that need images
    shots_to_generate = []
    shot_metadata = []  # (scene_index, shot_index) for tracking

    for scene_idx, scene in enumerate(artifact.scenes):
        if scene.shots:
            for shot_idx, shot in enumerate(scene.shots):
                if shot.description:
                    # Get stage setting image path for this scene
                    stage_image_path = None
                    if scene.stage_setting_shot and scene.stage_setting_shot.image_path:
                        stage_image_path = scene.stage_setting_shot.image_path

                    shots_to_generate.append({
                        "shot_description": shot.description,
                        "stage_setting_image_path": stage_image_path,
                        "scene_name": scene.name,
                        "shot_name": shot.name
                    })
                    shot_metadata.append((scene_idx, shot_idx))

    if not shots_to_generate:
        return artifact

    print(f"🎨 Generating {len(shots_to_generate)} individual shot images")

    # Create texts for batch generation
    texts = []
    image_paths = []

    for shot_data in shots_to_generate:
        # Create detailed prompt that references stage setting
        prompt = f"Generate shot image: {shot_data['shot_description']}. This shot is part of the scene shown in the reference image. Style: detailed storyboard illustration, cinematic framing."
        texts.append(prompt)

        # Add stage setting image as reference if available
        image_paths.append(shot_data['stage_setting_image_path'])

    context_str = "Generate cinematic shot images based on the shot descriptions. Use the reference stage setting image to maintain visual consistency. Output: image only."

    # Select model for image generation
    model, _ = _select_model_for_step(GenerationStep.SHOT_IMAGES)

    # Split into batches to avoid overwhelming the API
    batch_size = 10
    all_responses = []

    num_batches = (len(shots_to_generate) + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(shots_to_generate))

        batch_shots = shots_to_generate[start_idx:end_idx]
        batch_texts = texts[start_idx:end_idx]
        batch_images = image_paths[start_idx:end_idx]

        print(f"🎨 Processing batch {batch_num + 1}/{num_batches} ({len(batch_shots)} shots)")
        print(f"  Shots: {[s['shot_name'] for s in batch_shots[:3]]}{'...' if len(batch_shots) > 3 else ''}")

        try:
            responses, _, _ = await batch_llm(
                model=model,
                texts=batch_texts,
                context=context_str,
                image_paths=batch_images,
                output_is_image=True,
                image_generation_retries=2
            )
            all_responses.extend(responses)
            print(f"  ✅ Batch {batch_num + 1} completed successfully")

        except Exception as e:
            print(f"  ❌ Batch {batch_num + 1} failed: {str(e)[:100]}")
            # Add empty responses for failed batch to maintain alignment
            all_responses.extend([None] * len(batch_shots))

    # Save images and update shots
    successful_images = 0
    failed_images = 0

    for (scene_idx, shot_idx), image_bytes in zip(shot_metadata, all_responses):
        scene = artifact.scenes[scene_idx]
        shot = scene.shots[shot_idx]

        if image_bytes is not None:
            try:
                image_path = save_image_to_data(image_bytes, artifact.name, "shot", shot.name)
                shot.image_path = image_path
                print(f"  ✅ Saved: {scene.name} - {shot.name}")
                successful_images += 1
            except Exception as e:
                print(f"  ❌ Save failed for {shot.name}: {str(e)[:50]}")
                failed_images += 1
        else:
            print(f"  ❌ No image for {shot.name}")
            failed_images += 1

    print(f"\n📊 Shot images: {successful_images} success, {failed_images} failed")

    return artifact


async def run_generation_pipeline(artifact: StoryboardSpec, user_inputs: Dict[GenerationStep, str]) -> StoryboardSpec:
    """Run the complete generation pipeline.

    Args:
        artifact: Starting storyboard artifact
        user_inputs: Dictionary mapping GenerationStep to user prompts

    Returns:
        Completed storyboard artifact
    """
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
            elif step == GenerationStep.SHOT_DESCRIPTIONS:
                artifact = await generate_batch_shot_descriptions(artifact, user_inputs[step])
            elif step == GenerationStep.SHOT_IMAGES:
                artifact = await generate_shot_images(artifact, user_inputs[step])

            # Handle single-generation steps
            else:
                artifact, output = await generate_step(artifact, step, user_inputs[step])
                if output:
                    artifact = patch_artifact(artifact, step, output, user_choice=0)

            print(f"Completed {step.value}")

            # Save checkpoint
            save_artifact_checkpoint(artifact, step)

    return artifact
