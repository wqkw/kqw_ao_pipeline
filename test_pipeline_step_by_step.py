#!/usr/bin/env python3

import asyncio
import json
from dotenv import load_dotenv

from prompts.storyboard_artifact import StoryboardSpec
from generation_pipeline import (
    GenerationStep, get_completion_status, get_next_steps,
    create_context_dto, create_output_dto, extract_context_data,
    patch_artifact, generate_step, generate_batch_components,
    generate_batch_stage_shots, generate_stage_shot_images,
    generate_batch_shot_descriptions, generate_shot_images,
    save_artifact_checkpoint
)

load_dotenv()


def print_step_info(step: GenerationStep, artifact: StoryboardSpec, user_input: str):
    """Print detailed information about the current step."""
    print("\n" + "="*80)
    print(f"STEP: {step.value.upper()}")
    print("="*80)

    # Show context DTO structure
    ContextModel = create_context_dto(step)
    print(f"\nCONTEXT DTO FIELDS:")
    for field_name, field_info in ContextModel.model_fields.items():
        print(f"  - {field_name}: {field_info.annotation} - {field_info.description}")

    # Show output DTO structure
    OutputModel = create_output_dto(step)
    if OutputModel:
        print(f"\nOUTPUT DTO FIELDS:")
        for field_name, field_info in OutputModel.model_fields.items():
            print(f"  - {field_name}: {field_info.annotation} - {field_info.description}")
    else:
        print(f"\nOUTPUT: Image generation step (no structured output)")

    # Show extracted context data
    try:
        context_data = extract_context_data(artifact, step, prompt_input=user_input)
        print(f"\nEXTRACTED CONTEXT DATA:")
        for key, value in context_data.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  - {key}: {value[:100]}...")
            elif isinstance(value, list) and len(value) > 5:
                print(f"  - {key}: [{len(value)} items] {value[:3]}...")
            else:
                print(f"  - {key}: {value}")
    except Exception as e:
        print(f"\nEXTRACTED CONTEXT DATA: Error - {e}")

    # Show current artifact state
    completion_status = get_completion_status(artifact)
    print(f"\nCURRENT COMPLETION STATUS:")
    for step_name, completed in completion_status.items():
        status = "✅" if completed else "❌"
        print(f"  {status} {step_name.value}")

    # Debug: Show stage shot status for scenes
    if step == GenerationStep.STAGE_SHOT_DESCRIPTIONS or step == GenerationStep.STAGE_SHOT_IMAGES:
        print(f"\nSTAGE SHOT DEBUG:")
        if artifact.scenes:
            for i, scene in enumerate(artifact.scenes):
                stage_shot_status = "✅" if scene.stage_setting_shot is not None else "❌"
                image_status = "🖼️" if (scene.stage_setting_shot and scene.stage_setting_shot.image_path) else "📝"
                print(f"  Scene {i+1}: {scene.name}")
                print(f"    Stage shot: {stage_shot_status}")
                print(f"    Image: {image_status}")
                if scene.stage_setting_shot:
                    print(f"    Description: {scene.stage_setting_shot.description[:50] if scene.stage_setting_shot.description else 'None'}...")
        else:
            print(f"  No scenes found")

    print(f"\nUSER INPUT: {user_input}")
    print("\n" + "-"*80)


async def run_single_step(artifact: StoryboardSpec, step: GenerationStep, user_input: str):
    """Run a single step with detailed logging."""
    print_step_info(step, artifact, user_input)

    input(f"\nPress Enter to execute {step.value}...")

    print(f"\n🚀 Executing {step.value}...")

    # Handle batch generation steps
    if step == GenerationStep.COMPONENT_IMAGES:
        artifact = await generate_batch_components(artifact, user_input)
    elif step == GenerationStep.STAGE_SHOT_DESCRIPTIONS:
        artifact = await generate_batch_stage_shots(artifact, user_input)
    elif step == GenerationStep.STAGE_SHOT_IMAGES:
        artifact = await generate_stage_shot_images(artifact, user_input)
    elif step == GenerationStep.SHOT_DESCRIPTIONS:
        artifact = await generate_batch_shot_descriptions(artifact, user_input)
    elif step == GenerationStep.SHOT_IMAGES:
        artifact = await generate_shot_images(artifact, user_input)
    # Handle single-generation steps
    else:
        artifact, output = await generate_step(artifact, step, user_input)
        if output:
            print(f"\n📋 LLM OUTPUT:")
            output_dict = output.model_dump()
            print(json.dumps(output_dict, indent=2))

            print(f"\n🔧 PATCHING ARTIFACT...")
            artifact = patch_artifact(artifact, step, output, user_choice=0)

            # Debug: Show what was patched
            if step == GenerationStep.LORE:
                print(f"   Lore set: {len(artifact.lore)} characters")
            elif step == GenerationStep.NARRATIVE:
                print(f"   Narrative set: {len(artifact.narrative)} characters")
            elif step == GenerationStep.SCENES:
                print(f"   Scenes created: {len(artifact.scenes or [])} scenes")
                if artifact.scenes:
                    for i, scene in enumerate(artifact.scenes):
                        print(f"     {i+1}. {scene.name}")
            elif step == GenerationStep.COMPONENT_DESCRIPTIONS:
                print(f"   Characters created: {len(artifact.characters or [])} characters")
                if artifact.characters:
                    for char in artifact.characters:
                        print(f"     - {char.name}")
                print(f"   Locations created: {len(artifact.locations or [])} locations")
                if artifact.locations:
                    for loc in artifact.locations:
                        print(f"     - {loc.name}")
                print(f"   Props created: {len(artifact.props or [])} props")
                if artifact.props:
                    for prop in artifact.props:
                        print(f"     - {prop.name}")
        else:
            print(f"\n⚠️  No output returned from LLM")

    print(f"\n✅ Completed {step.value}")

    # Save checkpoint
    save_artifact_checkpoint(artifact, step)

    # Show updated artifact summary
    print(f"\n📊 ARTIFACT SUMMARY AFTER {step.value}:")
    if artifact.lore:
        print(f"  - Lore: {len(artifact.lore)} chars")
    if artifact.narrative:
        print(f"  - Narrative: {len(artifact.narrative)} chars")
    if artifact.scenes:
        print(f"  - Scenes: {len(artifact.scenes)} scenes")
        for i, scene in enumerate(artifact.scenes):
            print(f"    {i+1}. {scene.name}")
    if artifact.characters:
        print(f"  - Characters: {len(artifact.characters)} characters")
        for char in artifact.characters:
            status = "🖼️" if char.image_path else "📝"
            print(f"    {status} {char.name}")
    if artifact.locations:
        print(f"  - Locations: {len(artifact.locations)} locations")
        for loc in artifact.locations:
            status = "🖼️" if loc.image_path else "📝"
            print(f"    {status} {loc.name}")
    if artifact.props:
        print(f"  - Props: {len(artifact.props)} props")
        for prop in artifact.props:
            status = "🖼️" if prop.image_path else "📝"
            print(f"    {status} {prop.name}")

    return artifact


def load_latest_artifact(project_name: str) -> StoryboardSpec:
    """Load the latest artifact checkpoint or create new one."""
    import os
    import glob

    checkpoint_dir = os.path.join("data", project_name)
    if not os.path.exists(checkpoint_dir):
        return None

    # Find latest checkpoint
    pattern = os.path.join(checkpoint_dir, "artifact_after_*.json")
    checkpoints = glob.glob(pattern)

    # Also check for emergency save
    emergency_save = os.path.join(checkpoint_dir, "emergency_save.json")
    if os.path.exists(emergency_save):
        checkpoints.append(emergency_save)

    if not checkpoints:
        return None

    # Sort by step order (lore, narrative, scenes, etc.)
    step_order = ["lore", "narrative", "scenes", "component_descriptions", "component_images", "stage_shot_descriptions", "stage_shot_images", "shot_descriptions", "shot_images"]

    def step_priority(path):
        filename = os.path.basename(path)
        if "emergency_save.json" in filename:
            return 999  # Emergency saves take highest priority
        for i, step in enumerate(step_order):
            if f"artifact_after_{step}.json" in filename:
                return i
        return -1

    latest_checkpoint = max(checkpoints, key=step_priority)

    print(f"📂 Loading checkpoint: {latest_checkpoint}")
    with open(latest_checkpoint, "r") as f:
        data = json.load(f)

    return StoryboardSpec(**data)


async def main():
    project_name = "test_cobalt"

    # Try to load existing artifact first
    artifact = load_latest_artifact(project_name)

    if artifact:
        print(f"🔄 Resuming from checkpoint: {artifact.name}")
        choice = input("Continue from checkpoint? (y/n): ").lower()
        if choice != 'y':
            artifact = None

    # Initialize new artifact if needed
    if not artifact:
        artifact = StoryboardSpec(
            name=project_name,
            title="Test Cobalt Rebellion",
            moodboard_path="data/moodboard_tile.png"
        )
        print(f"🆕 Starting new artifact: {artifact.title}")

    # Define user inputs for each step
    user_inputs = {
        GenerationStep.LORE: "Dark fantasy with ancient magic and warring kingdoms",
        GenerationStep.NARRATIVE: "Hero's journey with betrayal and redemption",
        GenerationStep.SCENES: "Break into dramatic scenes with clear emotional arcs",
        GenerationStep.COMPONENT_DESCRIPTIONS: "Identify all props and environmental elements",
        GenerationStep.COMPONENT_IMAGES: "Generate component images",
        GenerationStep.STAGE_SHOT_DESCRIPTIONS: "Create establishing shots for each scene",
        GenerationStep.STAGE_SHOT_IMAGES: "Generate stage setting images",
        GenerationStep.SHOT_DESCRIPTIONS: "Break down each scene into individual shots with camera angles and framing",
        GenerationStep.SHOT_IMAGES: "Generate final shot images"
    }

    print(f"🎬 STEP-BY-STEP PIPELINE TEST")
    print(f"Project: {artifact.title}")
    print("="*80)

    # Run each step individually
    step_count = 0
    max_iterations = 20  # Prevent infinite loops

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
            continue_choice = input(f"\nContinue to next step? (y/n/q/d for debug): ").lower()
            if continue_choice == 'd':
                # Debug mode - show detailed status
                print(f"\n🔍 DEBUG INFO:")
                completion_status = get_completion_status(artifact)
                for step_name, completed in completion_status.items():
                    print(f"  {step_name.value}: {completed}")
                if artifact.scenes:
                    print(f"\nScene stage shot status:")
                    for i, scene in enumerate(artifact.scenes):
                        has_stage_shot = scene.stage_setting_shot is not None
                        print(f"  Scene {i+1} ({scene.name}): stage_setting_shot = {has_stage_shot}")
                continue
            elif continue_choice == 'q':
                print("👋 Quitting...")
                # Save emergency checkpoint
                with open(f"data/{artifact.name}/emergency_save.json", "w") as f:
                    json.dump(artifact.model_dump(), f, indent=2)
                print(f"💾 Emergency save: data/{artifact.name}/emergency_save.json")
                return artifact
            elif continue_choice == 'n':
                print("⏸️  Pausing pipeline...")
                # Save emergency checkpoint
                import os
                os.makedirs(f"data/{artifact.name}", exist_ok=True)
                with open(f"data/{artifact.name}/emergency_save.json", "w") as f:
                    json.dump(artifact.model_dump(), f, indent=2)
                print(f"💾 Emergency save: data/{artifact.name}/emergency_save.json")
                print(f"📝 To resume: run script again and select 'y' when prompted to load checkpoint")
                return artifact

    # Final save
    print("\n💾 Saving final artifact...")
    with open("test_completed_storyboard.json", "w") as f:
        json.dump(artifact.model_dump(), f, indent=2)

    print("✨ Test pipeline completed!")
    print(f"📄 Final artifact saved to: test_completed_storyboard.json")

    return artifact


if __name__ == "__main__":
    asyncio.run(main())