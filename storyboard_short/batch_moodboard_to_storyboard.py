#!/usr/bin/env python3

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
import shutil

from storyboard_short import (
    ShortStoryboardSpec,
    GenerationStep,
    generate_step, generate_batch_shots, generate_shot_images,
    patch_artifact
)
from storyboard_short.utils import create_shots_collage, artifact_to_shot_list_string

load_dotenv()


async def process_single_image(image_path: str, output_base_dir: Path) -> tuple[str | None, ShortStoryboardSpec | None]:
    """Process a single image through the full pipeline including image generation and collage.

    Args:
        image_path: Path to the input image
        output_base_dir: Base directory for outputs

    Returns:
        Tuple of (path to final collage image or None, artifact or None)
    """

    # Create unique run name based on image filename
    image_name = Path(image_path).stem
    run_name = f"moodboard_storyboard_{image_name}"

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
        GenerationStep.SCENES: "Generate 1 highly dynamic, action-packed scene with lots of motion and things happening",
        GenerationStep.SHOTS: "Generate 4-6 camera shots capturing intense action, movement, and dramatic activity for each scene",
        GenerationStep.SHOT_IMAGES: "Generate images for all shots",
        GenerationStep.FINAL_IMAGE: "Create shots collage"
    }

    try:
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

        # Step 2: SCENES
        print(f"\n🎬 Step 2: Scene Generation")
        artifact, output = await generate_step(
            artifact,
            GenerationStep.SCENES,
            user_inputs[GenerationStep.SCENES]
        )
        if output:
            artifact = patch_artifact(artifact, GenerationStep.SCENES, output)
            print(f"  ✅ Scenes created: {len(artifact.scenes or [])} scenes")
            if artifact.scenes:
                for i, scene in enumerate(artifact.scenes):
                    print(f"    {i+1}. {scene.name}")

        # Step 3: SHOTS
        print(f"\n🎥 Step 3: Shot Generation")
        artifact = await generate_batch_shots(
            artifact,
            user_inputs[GenerationStep.SHOTS]
        )

        # Count shots
        total_shots = 0
        if artifact.scenes:
            for scene in artifact.scenes:
                if scene.shots:
                    total_shots += len(scene.shots)
        print(f"  ✅ Shots generated: {total_shots} total shots")

        # Step 4: SHOT_IMAGES
        print(f"\n🎨 Step 4: Shot Image Generation")
        artifact = await generate_shot_images(
            artifact,
            user_inputs[GenerationStep.SHOT_IMAGES]
        )

        # Count shots with images
        shots_with_images = 0
        if artifact.scenes:
            for scene in artifact.scenes:
                if scene.shots:
                    for shot in scene.shots:
                        if shot.image_path:
                            shots_with_images += 1
        print(f"  ✅ Shot images created: {shots_with_images}/{total_shots}")

        # Step 5: FINAL_IMAGE (create collage)
        print(f"\n🖼️  Step 5: Creating Final Collage")
        try:
            collage_path = create_shots_collage(artifact)
            artifact.final_image_path = collage_path
            print(f"  ✅ Final collage saved: {collage_path}")

            print(f"\n✅ Completed: {image_name}\n")
            return collage_path, artifact

        except ValueError as e:
            print(f"  ❌ Failed to create collage: {e}")
            return None, artifact

    except Exception as e:
        print(f"❌ Error processing {image_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


async def main():
    # Input and output directories
    input_dir = Path("data/ref_moodboard5_small")
    output_base_dir = Path("data")
    storyboards_collection_dir = Path("data/storyboards_collection")

    # Create storyboards collection directory
    storyboards_collection_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"📁 Output base directory: {output_base_dir}")
    print(f"📁 Storyboards collection: {storyboards_collection_dir}")

    # Process each image
    completed_storyboards = []

    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {image_file.name}...")

        try:
            collage_path, artifact = await process_single_image(str(image_file), output_base_dir)

            if collage_path and artifact:
                completed_storyboards.append(collage_path)
                image_name = image_file.stem

                # Copy collage to collection directory
                collage_source = Path(collage_path)
                if collage_source.exists():
                    collection_filename = f"storyboard_{image_name}.png"
                    collection_path = storyboards_collection_dir / collection_filename
                    shutil.copy2(collage_source, collection_path)
                    print(f"  📋 Copied storyboard to collection: {collection_path}")

                # Copy original image to collection directory
                original_collection_path = storyboards_collection_dir / f"original_{image_name}{image_file.suffix}"
                shutil.copy2(image_file, original_collection_path)
                print(f"  📋 Copied original image to collection: {original_collection_path}")

                # Save text description to collection directory
                shot_list_text = artifact_to_shot_list_string(artifact)
                text_collection_path = storyboards_collection_dir / f"description_{image_name}.txt"
                with open(text_collection_path, 'w', encoding='utf-8') as f:
                    f.write(shot_list_text)
                print(f"  📋 Saved text description to collection: {text_collection_path}")

        except Exception as e:
            print(f"❌ Error processing {image_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✨ Batch processing complete!")
    print(f"📊 Successfully processed: {len(completed_storyboards)}/{len(image_files)} images")
    print(f"📁 Individual outputs in: {output_base_dir}/moodboard_storyboard_*/")
    print(f"📁 All storyboards collected in: {storyboards_collection_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
