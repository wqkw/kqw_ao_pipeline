"""
Short Storyboard Generation Pipeline

Simplified pipeline for generating quick scene sequences from 1-3 input images.
All generation guides are embedded directly in this file.
"""

from __future__ import annotations

import asyncio
from typing import Tuple
from pydantic import BaseModel
from enum import Enum

from openrouter_wrapper import llm, batch_llm

from .artifact import ShortStoryboardSpec, SceneSpec, ShotSpec, StrictModel
from .artifact_adapters import (
    extract_context_dto,
    context_dto_to_string,
    create_output_dto,
    patch_artifact
)
from .utils import save_image_to_data, save_artifact_checkpoint, create_shots_collage


# ---------- Generation Steps ----------

class GenerationStep(Enum):
    IMAGE_RECOGNITION = "image_recognition"
    SCENES = "scenes"
    SHOTS = "shots"
    SHOT_IMAGES = "shot_images"
    FINAL_IMAGE = "final_image"


# ---------- Embedded Generation Guides ----------

IMAGE_RECOGNITION_GUIDE = """
# Image Recognition Guide

Analyze each input image and provide detailed descriptions.

For each image, describe:
- Main subjects (people, objects, environments)
- Visual style (photography, illustration, 3D render, etc.)
- Color palette and mood
- Composition and framing
- Notable details or elements
- Suggested narrative or emotional tone

Be specific and visual. Focus on what you see, not what might be happening.
"""

SCENES_GUIDE = """
# Scene Generation Guide

Create 1 highly dynamic and action-packed scenes based on the input images.

Each scene should:
- Feature intense motion, action, or dramatic activity
- Include multiple things happening simultaneously
- Show energy, movement, and visual dynamism
- NOT follow a story progression (standalone action moments)
- Draw visual inspiration from the input images
- Have a clear mood or atmosphere with HIGH energy

Focus on:
- Dynamic action and movement (what's actively happening)
- Multiple elements in motion or interaction
- Physical energy and visual intensity
- Dramatic moments with clear activity

IMPORTANT: Scenes should feel alive with action, not static. Emphasize verbs and movement.

Keep scenes concise but evocative.

Make sure not to do anything too violent as to trigger content moderation.
"""

SHOTS_GUIDE = """
# Shot Generation Guide

For each scene, create exactly 10 camera shots that capture DYNAMIC ACTION and MOVEMENT.

Each shot should:
- Emphasize motion, action, and things actively happening
- Show movement within the frame (people moving, objects in motion, environmental action)
- Have a specific camera angle and framing
- Show a different aspect or perspective of the action
- Build visual variety (wide, medium, close-up mix)
- Include camera movement to enhance the sense of motion (tracking, following action, etc.)

IMPORTANT: Every shot should capture something MOVING or HAPPENING. Avoid static poses.

Camera angles to consider:
- Eye level, high angle, low angle, overhead, canted
- Wide shot, medium shot, close-up, extreme close-up
- Dynamic camera movements: tracking, following, whip pan, dolly, crane, handheld

What to capture in each shot:
- Active verbs: running, jumping, splashing, throwing, falling, spinning, etc.
- Environmental motion: wind, water, particles, light changes
- Multiple elements moving simultaneously
- Peak action moments and dramatic movement

Be specific about what the camera sees, what's moving, and how the camera follows the action.

Make sure not to do anything too violent as to trigger content moderation.
"""


# ---------- Model Selection ----------

def _select_model_for_step(step: GenerationStep) -> Tuple[str, str]:
    """Select model and reasoning effort based on step requirements.

    Args:
        step: The generation step

    Returns:
        Tuple of (model_name, reasoning_effort)
    """
    # Image generation uses image-preview model (SHOT_IMAGES only)
    if step == GenerationStep.SHOT_IMAGES:
        return "google/gemini-2.5-flash-image-preview", "minimal"

    # Image recognition uses vision model
    if step == GenerationStep.IMAGE_RECOGNITION:
        return "google/gemini-2.5-flash", "minimal"

    # Scene and shot generation use fast model
    return "google/gemini-2.5-flash", "minimal"


# ---------- Generation Pipeline ----------

async def generate_step(artifact: ShortStoryboardSpec, step: GenerationStep, prompt_input: str, **kwargs) -> Tuple[ShortStoryboardSpec, BaseModel]:
    """Execute a single generation step (IMAGE_RECOGNITION or SCENES only).

    Other steps use specialized batch functions:
    - SHOTS: generate_batch_shots()
    - SHOT_IMAGES: generate_shot_images()
    - FINAL_IMAGE: create_shots_collage()

    Args:
        artifact: Current storyboard artifact
        step: Generation step to execute
        prompt_input: User's creative input for this step
        **kwargs: Additional context data

    Returns:
        Tuple of (updated_artifact, llm_output)
    """
    # Extract context DTO
    context = extract_context_dto(artifact, step, prompt_input=prompt_input, **kwargs)

    # Create output DTO
    OutputModel = create_output_dto(step)

    # Select model and reasoning effort
    model, reasoning = _select_model_for_step(step)

    # IMAGE_RECOGNITION: Analyze all input images
    if step == GenerationStep.IMAGE_RECOGNITION:
        image_paths = artifact.image_references

        response, _, _ = llm(
            model=model,
            text=prompt_input,
            context=IMAGE_RECOGNITION_GUIDE,
            input_image_path=image_paths,
            response_format=OutputModel,
            reasoning_effort=reasoning
        )

    # SCENES: Generate 1-2 scenes from image descriptions
    elif step == GenerationStep.SCENES:
        context_str = f"Image descriptions:\n"
        for i, desc in enumerate(context.image_descriptions):
            context_str += f"{i+1}. {desc}\n"
        context_str += f"\n{SCENES_GUIDE}"

        response, _, _ = llm(
            model=model,
            text=prompt_input,
            context=context_str,
            response_format=OutputModel,
            reasoning_effort=reasoning
        )

    else:
        raise ValueError(f"generate_step should only be called for IMAGE_RECOGNITION or SCENES. Use specialized functions for {step.value}")

    return artifact, response


async def generate_batch_shots(artifact: ShortStoryboardSpec, prompt_input: str) -> ShortStoryboardSpec:
    """Generate shots for all scenes in parallel.

    Args:
        artifact: Current storyboard artifact
        prompt_input: User's creative input

    Returns:
        Updated artifact with shots
    """
    if not artifact.scenes:
        return artifact

    # Get model and reasoning effort
    model, reasoning = _select_model_for_step(GenerationStep.SHOTS)
    OutputModel = create_output_dto(GenerationStep.SHOTS)

    print(f"🎬 Generating shots for {len(artifact.scenes)} scenes")

    # Prepare batch data
    texts = []
    for scene in artifact.scenes:
        texts.append(f"Generate exactly 10 dynamic camera shots with lots of action and movement for scene: {scene.name}\n\nScene: {scene.description}")

    # Generate shots
    batch_context = f"Generate exactly 10 action-packed camera shots for each scene. Focus on motion, dynamics, and things actively happening.\n\n{SHOTS_GUIDE}"

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
        if response and hasattr(response, 'shots') and response.shots:
            # Convert ShotDescription objects to ShotSpec objects (with image_path=None)
            scene.shots = [
                ShotSpec(
                    name=shot.name,
                    description=shot.description,
                    camera_angle=shot.camera_angle,
                    image_path=None  # Will be filled in by SHOT_IMAGES step
                )
                for shot in response.shots
            ]
            print(f"  ✅ Scene {i+1}: {len(response.shots)} shots created")
            successful_scenes += 1
        else:
            print(f"  ❌ Scene {i+1}: No shots generated")

    print(f"  📊 Shots created: {successful_scenes}/{len(artifact.scenes)}")

    return artifact


async def generate_shot_images(artifact: ShortStoryboardSpec, prompt_input: str) -> ShortStoryboardSpec:
    """Generate images for all shots in parallel.

    Args:
        artifact: Current storyboard artifact
        prompt_input: User's creative input

    Returns:
        Updated artifact with shot images
    """
    if not artifact.scenes:
        return artifact

    # Collect all shots that need images
    shots_to_generate = []
    shot_metadata = []  # (scene_index, shot_index) for tracking

    for scene_idx, scene in enumerate(artifact.scenes):
        if scene.shots:
            for shot_idx, shot in enumerate(scene.shots):
                if shot.description:
                    shots_to_generate.append({
                        "shot_description": shot.description,
                        "scene_name": scene.name,
                        "shot_name": shot.name
                    })
                    shot_metadata.append((scene_idx, shot_idx))

    if not shots_to_generate:
        return artifact

    print(f"🎨 Generating {len(shots_to_generate)} shot images")

    # Get model
    model, _ = _select_model_for_step(GenerationStep.SHOT_IMAGES)

    # Create texts for batch generation
    texts = []
    image_paths = []

    for shot_data in shots_to_generate:
        # Create detailed prompt with emphasis on action and motion
        prompt = f"Generate dynamic action shot: {shot_data['shot_description']}. IMPORTANT: Capture motion, movement, and things actively happening. Style: cinematic storyboard illustration matching the reference images."
        texts.append(prompt)
        # Use original input images as reference
        image_paths.append(artifact.image_references)

    context_str = "Generate highly dynamic, action-packed cinematic shot images. Emphasize motion, movement, and active events. Use the reference images to match visual style. Output: image only."

    # Generate all shots in batches
    batch_size = 10
    all_responses = []

    num_batches = (len(shots_to_generate) + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(shots_to_generate))

        batch_shots = shots_to_generate[start_idx:end_idx]
        batch_texts = texts[start_idx:end_idx]
        batch_images = image_paths[start_idx:end_idx]

        print(f"  🎨 Processing batch {batch_num + 1}/{num_batches} ({len(batch_shots)} shots)")

        try:
            responses, _, _ = await batch_llm(
                model=model,
                texts=batch_texts,
                context=context_str,
                image_paths=batch_images,
                output_is_image=True,
                image_generation_retries=3
            )
            all_responses.extend(responses)
            print(f"    ✅ Batch {batch_num + 1} completed")

        except Exception as e:
            print(f"    ❌ Batch {batch_num + 1} failed: {str(e)[:100]}")
            all_responses.extend([None] * len(batch_shots))

        # Small delay between batches to avoid overwhelming API
        if batch_num < num_batches - 1:
            await asyncio.sleep(1)

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
                print(f"    ✅ Saved: {scene.name} - {shot.name}")
                successful_images += 1
            except Exception as e:
                print(f"    ❌ Save failed for {shot.name}: {str(e)[:50]}")
                failed_images += 1
        else:
            print(f"    ❌ No image for {shot.name}")
            failed_images += 1

    print(f"\n  📊 Shot images: {successful_images} success, {failed_images} failed")

    return artifact


async def run_generation_pipeline(artifact: ShortStoryboardSpec, user_inputs: dict) -> ShortStoryboardSpec:
    """Run the complete generation pipeline.

    Args:
        artifact: Starting storyboard artifact
        user_inputs: Dictionary mapping GenerationStep to user prompts

    Returns:
        Completed storyboard artifact
    """
    from .artifact_adapters import get_next_steps

    while True:
        next_steps = get_next_steps(artifact)
        if not next_steps:
            break

        for step in next_steps:
            if step not in user_inputs:
                print(f"Waiting for user input for step: {step.value}")
                continue

            print(f"Generating {step.value}...")

            # Handle batch generation for shots
            if step == GenerationStep.SHOTS:
                artifact = await generate_batch_shots(artifact, user_inputs[step])

            # Handle shot image generation
            elif step == GenerationStep.SHOT_IMAGES:
                artifact = await generate_shot_images(artifact, user_inputs[step])

            # Handle final image generation (create collage)
            elif step == GenerationStep.FINAL_IMAGE:
                print(f"🎨 Creating final collage from all shots...")
                try:
                    collage_path = create_shots_collage(artifact)
                    artifact.final_image_path = collage_path
                    print(f"  ✅ Final collage saved: {collage_path}")
                except ValueError as e:
                    print(f"  ❌ Failed to create collage: {e}")

            # Handle single-generation steps
            else:
                artifact, output = await generate_step(artifact, step, user_inputs[step])
                if output:
                    artifact = patch_artifact(artifact, step, output)

            print(f"Completed {step.value}")

            # Save checkpoint
            save_artifact_checkpoint(artifact, step)

    return artifact
