from openrouter_wrapper import llm, llm_async, batch_llm
import base64
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, Field
from typing import List
import json
import asyncio
import os
load_dotenv()
import utils as utils



if True:
    with open("data/moodboard_tile.png", "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode()

    RESP1, _, history = llm(
        model="openai/gpt-5",
        text='\n\n'.join(["Based on the attached moodboard, generate a detailed DND dungeon-master style description of lore for this world", utils.get_prompt("lore_guide.md")]),
        reasoning_effort='medium',
        input_image_url=f"data:image/png;base64,{img_b64}"
    )
    print(RESP1)

if True:
    PROMPT2 = f"""
    Based on the lore for a given world, generate a story. The story should fit in an approximately 1 minute long short form video. 

    Outline the scenes (should have 5-10), which each should have 1-4 shots. 

    LORE: 
    {RESP1}

    STORY GUIDE:
    {utils.get_prompt("narrative_guide.md")}
    """

    RESP2, _, history = llm(
        model="openai/gpt-5",
        text=PROMPT2,
        # context=history,
    )
    print(RESP2)



if True:
    PROMPT1 = """
    Based on this flow that we've outlined here, take this initial input of the base world components from this image (this is a mood board of that world that includes some characters, some locations, and some props all anchored to a consistent style). I want you to analyze this mood board and create the more detailed context for each of the base world components, where you add descriptions for:
    - the style and aesthetic
    - the characters
    - the locations
    - the props and objects
    to build out our world building components. Based on this mood board, create an outline of our world building components
    """

    with open("data/moodboard_tile.png", "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode()

    RESP1, _, history = llm(
        model="openai/gpt-5",
        text=PROMPT1,
        context=METAPLAN_PROMPT,
        reasoning_effort='minimal',
        input_image_url=f"data:image/png;base64,{img_b64}"
    )
    print(RESP1)

if False:
    with open("data/moodboard_tile.png", "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode()

    RESP1, _, history = llm(
        model="openai/gpt-5",
        text='Based on the aesthetics of this moodboard, generate a detailed DND dungeon-master style description of lore for this world',
        input_image_url=f"data:image/png;base64,{img_b64}"
    )
    print(RESP1)

if True:
    PROMPT2 = """
    Ok now take these components and draft 3 potential storylines and narrative arcs.
    These storylines should be approximately short story length
    """

    RESP2, _, history = llm(
        model="openai/gpt-5",
        text=PROMPT2,
        context=history,
    )
    print(RESP2)

if True:
    PROMPT3 = """
    Let's move forward with Storyline 1. Go ahead and take Storyline 1 and move on to Step 3 of our flow here, Generate Shot List & Scene Breakdown
    """

    try:
        RESP3, full_resp3, history = llm(
            model="openai/gpt-5",
            text=PROMPT3,
            context=history,
        )
        print("RESP3:")
        print(RESP3)
    except Exception as e:
        print(f"Error in step 3: {e}")
        print(f"Error type: {type(e)}")
        # Try to get the raw response if available
        try:
            import traceback
            traceback.print_exc()
        except:
            pass


class ScenePrompt(BaseModel):
    model_config = {"extra": "forbid"}

    scene_name: str = Field(description="Name of the scene")
    prompt: str = Field(description="Detailed image generation prompt for the scene setting shot. Should include specific visual details, lighting, composition, characters, props, and atmosphere. Example: 'A dimly lit Victorian library with mahogany bookshelves reaching to the ceiling, warm amber lighting from brass lamps, leather armchair in foreground, mysterious figure in dark coat examining ancient tome, gothic atmosphere with dust motes floating in shafts of light'")

class ScenePromptsOutput(BaseModel):
    model_config = {"extra": "forbid"}

    scene_prompts: List[ScenePrompt] = Field(description="List of scene setting shot prompts for image generation")


if True:
    PROMPT4 = """

    Ok move on to step 4. Create the scene setting shot prompts.
    """

    try:
        RESP4, full_response, history = llm(
            model="openai/gpt-5",
            text=PROMPT4,
            context=history,
            response_format=ScenePromptsOutput
        )
    except Exception as e:
        print(f"Error in step 4: {e}")
        print(f"Error type: {type(e)}")
        # Fallback: try without full history context
        try:
            print("Trying with reduced context...")
            RESP4, full_response, history = llm(
                model="openai/gpt-5",
                text=PROMPT4,
                context=METAPLAN_PROMPT + '\n\nShot list and scene breakdown: ' + (RESP3 if 'RESP3' in locals() else ""),
                response_format=ScenePromptsOutput
            )
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            RESP4 = ""
            full_response = {}

    # Debug: print the full response to see what's happening
    sss_prompts = []
    if RESP4 == "":
        print("Empty response. Full response:")
        print_truncated(full_response)
    else:
        # RESP4 is now automatically parsed into ScenePromptsOutput object
        if isinstance(RESP4, ScenePromptsOutput):
            print('AAA')
            for i, scene in enumerate(RESP4.scene_prompts, 1):
                print(f"\n{i}. {scene.scene_name}")
                print(f"   Prompt: {scene.prompt}")
                sss_prompts.append(scene.prompt)
        else:
            # Fallback if structured parsing failed
            print("Raw response:")
            print(RESP4)


    async def generate_images():
        responses, full_responses, combined_history = await batch_llm(
            model="google/gemini-2.5-flash-image-preview",
            texts=sss_prompts,
            output_is_image=True
        )

        # Save images to data/outputs/
        for i, (image_bytes, prompt) in enumerate(zip(responses, sss_prompts)):
            filename = f"data/outputs/scene_{i+1}.png"
            with open(filename, "wb") as f:
                f.write(image_bytes)
            print(f"Saved image {i+1} to {filename}")

    asyncio.run(generate_images())

