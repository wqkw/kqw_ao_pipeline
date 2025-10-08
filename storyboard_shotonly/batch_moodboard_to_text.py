#!/usr/bin/env python3

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from storyboard_shotonly import (
    ShortStoryboardSpec,
    GenerationStep,
    generate_step, generate_shots,
    patch_artifact
)
from storyboard_shotonly.utils import artifact_to_shot_list_string

load_dotenv()


async def process_single_image(image_path: str, output_dir: Path) -> None:
    """Process a single image through the pipeline up to shots generation."""

    # Create unique run name based on image filename
    image_name = Path(image_path).stem
    run_name = f"shotonly_{image_name}"

    print(f"\n{'='*80}")
    print(f"Processing: {image_name}")
    print(f"{'='*80}")

    # Initialize artifact
    artifact = ShortStoryboardSpec(
        name=run_name,
        image_references=[image_path]
    )

    # User inputs for each step
    user_inputs = {
        GenerationStep.IMAGE_RECOGNITION: "Analyze and describe this image in detail",
        GenerationStep.MOTION_NARRATIVE_ANALYSIS: "Based on the image, determine the appropriate motion level (low/medium/high) and narrative level (low/medium/high)",
        GenerationStep.SHOTS: "Generate 4-6 camera shots that match the determined motion and narrative levels",
    }

    # Step 1: IMAGE_RECOGNITION
    print(f"\n🔍 Step 1: Image Recognition")
    artifact, output = await generate_step(
        artifact,
        GenerationStep.IMAGE_RECOGNITION,
        user_inputs[GenerationStep.IMAGE_RECOGNITION]
    )
    if output:
        artifact = patch_artifact(artifact, GenerationStep.IMAGE_RECOGNITION, output)
        print(f"  ✅ Image descriptions: {len(artifact.image_descriptions or [])} images")

    # Step 2: MOTION_NARRATIVE_ANALYSIS
    print(f"\n📊 Step 2: Motion & Narrative Analysis")
    artifact, output = await generate_step(
        artifact,
        GenerationStep.MOTION_NARRATIVE_ANALYSIS,
        user_inputs[GenerationStep.MOTION_NARRATIVE_ANALYSIS]
    )
    if output:
        artifact = patch_artifact(artifact, GenerationStep.MOTION_NARRATIVE_ANALYSIS, output)
        print(f"  ✅ Motion level: {artifact.motion_level}")
        print(f"  ✅ Narrative level: {artifact.narrative_level}")

    # Step 3: SHOTS
    print(f"\n🎥 Step 3: Shot Generation")
    artifact = await generate_shots(
        artifact,
        user_inputs[GenerationStep.SHOTS]
    )

    # Count shots
    total_shots = 0
    if artifact.scene and artifact.scene.shots:
        total_shots = len(artifact.scene.shots)
    print(f"  ✅ Shots generated: {total_shots} total shots")

    # Convert to text string
    shot_list_text = artifact_to_shot_list_string(artifact)

    # Save to text file
    output_file = output_dir / f"{image_name}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(shot_list_text)

    print(f"  💾 Saved to: {output_file}")
    print(f"\n✅ Completed: {image_name}\n")


async def main():
    # Input and output directories
    input_dir = Path("data/ref_moodboard4")
    output_dir = Path("data/ref_moodboard4_shotonly_text")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return

    print(f"\n📸 Found {len(image_files)} images in {input_dir}")
    print(f"📁 Output directory: {output_dir}")

    # Process each image
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {image_file.name}...")
        try:
            await process_single_image(str(image_file), output_dir)
        except Exception as e:
            print(f"❌ Error processing {image_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✨ Batch processing complete!")
    print(f"📁 All text files saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
