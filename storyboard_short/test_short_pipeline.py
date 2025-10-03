#!/usr/bin/env python3

import asyncio
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

from storyscene_short import (
    ShortStoryboardSpec,
    GenerationStep, get_completion_status, get_next_steps,
    extract_context_dto, context_dto_to_string, create_output_dto,
    patch_artifact, generate_step, generate_batch_shots, generate_shot_images,
    save_artifact_checkpoint
)

load_dotenv()


def print_step_info(step: GenerationStep, artifact: ShortStoryboardSpec, user_input: str):
    """Print detailed information about the current step."""
    print("\n" + "="*80)
    print(f"STEP: {step.value.upper()}")
    print("="*80)

    # Show context DTO
    try:
        context = extract_context_dto(artifact, step, prompt_input=user_input)
        print(f"\nCONTEXT:")
        context_str = context_dto_to_string(context)
        if len(context_str) > 500:
            print(f"  {context_str[:500]}...")
        else:
            print(f"  {context_str}")
    except Exception as e:
        print(f"\nCONTEXT ERROR: {e}")

    # Show output DTO structure
    OutputModel = create_output_dto(step)
    if OutputModel:
        print(f"\nOUTPUT FIELDS:")
        for field_name, field_info in OutputModel.model_fields.items():
            print(f"  - {field_name}: {field_info.annotation}")

    # Show completion status
    completion_status = get_completion_status(artifact)
    print(f"\nCOMPLETION STATUS:")
    for step_name, completed in completion_status.items():
        status = "✅" if completed else "❌"
        print(f"  {status} {step_name.value}")

    print(f"\nUSER INPUT: {user_input}")
    print("\n" + "-"*80)


async def run_single_step(artifact: ShortStoryboardSpec, step: GenerationStep, user_input: str):
    """Run a single step with detailed logging."""
    print_step_info(step, artifact, user_input)

    input(f"\nPress Enter to execute {step.value}...")

    print(f"\n🚀 Executing {step.value}...")

    # Handle batch generation for shots
    if step == GenerationStep.SHOTS:
        artifact = await generate_batch_shots(artifact, user_input)

    # Handle shot image generation
    elif step == GenerationStep.SHOT_IMAGES:
        artifact = await generate_shot_images(artifact, user_input)

    # Handle final image generation (create collage)
    elif step == GenerationStep.FINAL_IMAGE:
        from storyscene_short.utils import create_shots_collage
        print(f"🎨 Creating final collage from all shots...")
        try:
            collage_path = create_shots_collage(artifact)
            artifact.final_image_path = collage_path
            print(f"  ✅ Final collage saved: {collage_path}")
        except ValueError as e:
            print(f"  ❌ Failed to create collage: {e}")

    # Handle single-generation steps
    else:
        artifact, output = await generate_step(artifact, step, user_input)
        if output:
            print(f"\n📋 LLM OUTPUT:")
            output_dict = output.model_dump()
            print(json.dumps(output_dict, indent=2))

            print(f"\n🔧 PATCHING ARTIFACT...")
            artifact = patch_artifact(artifact, step, output)

            # Debug: Show what was patched
            if step == GenerationStep.IMAGE_RECOGNITION:
                print(f"   Image descriptions: {len(artifact.image_descriptions or [])} images")
            elif step == GenerationStep.SCENES:
                print(f"   Scenes created: {len(artifact.scenes or [])} scenes")
                if artifact.scenes:
                    for i, scene in enumerate(artifact.scenes):
                        print(f"     {i+1}. {scene.name}")
        else:
            print(f"\n⚠️  No output returned from LLM")

    print(f"\n✅ Completed {step.value}")

    # Save checkpoint
    save_artifact_checkpoint(artifact, step)

    # Show updated artifact summary
    print(f"\n📊 ARTIFACT SUMMARY:")
    print(f"  - Input images: {len(artifact.image_references)}")
    if artifact.image_descriptions:
        print(f"  - Image descriptions: {len(artifact.image_descriptions)}")
    if artifact.scenes:
        print(f"  - Scenes: {len(artifact.scenes)}")
        for i, scene in enumerate(artifact.scenes):
            shots_count = len(scene.shots) if scene.shots else 0
            if scene.shots:
                shots_with_images = sum(1 for shot in scene.shots if shot.image_path)
                print(f"    {i+1}. {scene.name} ({shots_count} shots, {shots_with_images} with images)")
            else:
                print(f"    {i+1}. {scene.name} ({shots_count} shots)")
    if artifact.final_image_path:
        print(f"  - Final image: {artifact.final_image_path}")

    return artifact


def load_latest_artifact(run_name: str) -> ShortStoryboardSpec:
    """Load the latest artifact checkpoint for a specific run.

    Args:
        run_name: Name of the run (e.g., 'test_short_gen_20251001_143022')

    Returns:
        ShortStoryboardSpec or None if no checkpoint found
    """
    import os
    import glob

    checkpoint_dir = os.path.join("data", run_name)
    if not os.path.exists(checkpoint_dir):
        return None

    # Find latest checkpoint
    pattern = os.path.join(checkpoint_dir, "artifact_after_*.json")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None

    # Sort by step order
    step_order = ["image_recognition", "scenes", "shots", "shot_images", "final_image"]

    def step_priority(path):
        filename = os.path.basename(path)
        for i, step in enumerate(step_order):
            if f"artifact_after_{step}.json" in filename:
                return i
        return -1

    latest_checkpoint = max(checkpoints, key=step_priority)

    print(f"📂 Loading checkpoint: {latest_checkpoint}")
    with open(latest_checkpoint, "r") as f:
        data = json.load(f)

    return ShortStoryboardSpec(**data)


async def main():
    # Generate unique run name with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"test_short_gen_{timestamp}"

    # Get input images from command line or use defaults
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        # Default to some moodboard images
        image_paths = [
            # "data/ref_moodboard3/05_01.png",
            "data/ref_moodboard4/magnifics_upscale-yXoYVTtgtRWqJhcapL47-nickfloats_Close-up_of_two_cars_racing_on_the_road_one_car_in_5c7756dd-65fe-416e-ac81-90536741a5d0_3.png",
            # "data/ref_moodboard2/01_04.png",
        ]

    # Validate image paths
    valid_images = []
    for img_path in image_paths:
        if Path(img_path).exists():
            valid_images.append(img_path)
        else:
            print(f"⚠️  Warning: Image not found: {img_path}")

    if not valid_images:
        print("❌ No valid images provided!")
        print("\nUsage: python test_short_pipeline.py [image1.png] [image2.png] [image3.png]")
        print("   Or: python test_short_pipeline.py  (uses default images)")
        return

    if len(valid_images) > 3:
        print(f"⚠️  Warning: More than 3 images provided, using first 3")
        valid_images = valid_images[:3]

    print(f"\n📸 Input images ({len(valid_images)}):")
    for i, img in enumerate(valid_images):
        print(f"  {i+1}. {img}")

    # Try to load existing artifact (for resume functionality)
    artifact = load_latest_artifact(run_name)

    if artifact:
        print(f"\n✅ Found existing checkpoint: {artifact.name}")
        completion_status = get_completion_status(artifact)
        completed_steps = [s.value for s, done in completion_status.items() if done]
        if completed_steps:
            print(f"   Completed steps: {', '.join(completed_steps)}")

        choice = input("\nContinue from checkpoint? [Y/n/fresh]: ").lower()
        if choice == 'fresh':
            print("🆕 Starting fresh...")
            artifact = None
        elif choice == 'n':
            print("👋 Exiting...")
            return
        else:
            print(f"🔄 Continuing from checkpoint...")

    # Initialize new artifact if needed
    if not artifact:
        artifact = ShortStoryboardSpec(
            name=run_name,
            image_references=valid_images
        )
        print(f"🆕 Starting new artifact: {artifact.name}")
        print(f"📁 Output folder: data/{run_name}")

    # Define user inputs for each step
    user_inputs = {
        GenerationStep.IMAGE_RECOGNITION: "Analyze and describe these images in detail",
        GenerationStep.SCENES: "Generate 1 highly dynamic, action-packed scene with lots of motion and things happening",
        GenerationStep.SHOTS: "Generate 10 camera shots capturing intense action, movement, and dramatic activity for each scene",
        GenerationStep.SHOT_IMAGES: "Generate images for all shots",
        GenerationStep.FINAL_IMAGE: "Create shots collage"
    }

    print(f"\n🎬 SHORT STORYBOARD PIPELINE TEST")
    print("="*80)

    # Run each step individually
    step_count = 0
    max_iterations = 10

    while step_count < max_iterations:
        next_steps = get_next_steps(artifact)
        print(f"\n🔍 Available next steps: {[s.value for s in next_steps]}")

        if not next_steps:
            print("\n🎉 All steps completed!")
            break

        for step in next_steps:
            if step not in user_inputs:
                print(f"⚠️  No user input defined for step: {step.value}")
                continue

            step_count += 1
            print(f"\n🔄 STEP {step_count}: {step.value}")

            artifact = await run_single_step(artifact, step, user_inputs[step])

            # Ask if user wants to continue
            continue_choice = input(f"\nContinue to next step? (y/n/q): ").lower()
            if continue_choice == 'q':
                print("👋 Quitting...")
                return artifact
            elif continue_choice == 'n':
                print("⏸️  Pausing pipeline...")
                return artifact

    # Final save
    print("\n💾 Saving final artifact...")
    final_path = f"data/{artifact.name}/final_artifact.json"
    with open(final_path, "w") as f:
        json.dump(artifact.model_dump(), f, indent=2)

    print("✨ Test pipeline completed!")
    print(f"📄 Final artifact saved to: {final_path}")
    print(f"📁 All outputs in: data/{artifact.name}/")

    return artifact


if __name__ == "__main__":
    asyncio.run(main())
