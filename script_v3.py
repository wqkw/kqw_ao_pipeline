#!/usr/bin/env python3

import asyncio
from dotenv import load_dotenv

from prompts.storyboard_artifact import StoryboardSpec
from generation_pipeline import GenerationStep, run_generation_pipeline

load_dotenv()


async def main():
    # Initialize artifact
    artifact = StoryboardSpec(
        name="crimson_rebellion",
        title="The Crimson Rebellion",
        moodboard_path="data/moodboard_tile.png"  # Assume this exists
    )

    # Define user inputs for each step
    user_inputs = {
        GenerationStep.LORE: "Create dark fantasy lore with ancient magic systems, warring kingdoms, and forgotten technologies from a lost civilization",
        GenerationStep.NARRATIVE: "Epic quest story where a reluctant hero must unite fractured kingdoms against an ancient evil, featuring betrayal, redemption, and sacrifice",
        GenerationStep.SCENES: "Break the narrative into dramatic scenes with clear emotional arcs and visual storytelling moments",
        GenerationStep.COMPONENT_DESCRIPTIONS: "Identify all props, weapons, artifacts, and environmental elements needed for the scenes",
        GenerationStep.COMPONENT_IMAGES: "Generate high-quality component images for storyboarding",
        GenerationStep.STAGE_SHOT_DESCRIPTIONS: "Create establishing shots that set the visual context for each scene",
        GenerationStep.STAGE_SHOT_IMAGES: "Generate stage setting images that establish the environment",
        GenerationStep.SHOT_DESCRIPTIONS: "Break down each scene into individual shots with detailed descriptions of camera angles, framing, and action",
        GenerationStep.SHOT_IMAGES: "Generate final shot images for the complete storyboard"
    }

    print(f"Starting generation pipeline for: {artifact.title}")
    print("=" * 50)

    # Run the pipeline
    completed_artifact = await run_generation_pipeline(artifact, user_inputs)

    print("=" * 50)
    print("Generation pipeline completed!")
    print(f"Generated {len(completed_artifact.scenes or [])} scenes")
    print(f"Generated {len(completed_artifact.props or [])} props")

    # Save the completed artifact
    import json
    with open("completed_storyboard.json", "w") as f:
        json.dump(completed_artifact.model_dump(), f, indent=2)

    print("Storyboard saved to completed_storyboard.json")


if __name__ == "__main__":
    asyncio.run(main())