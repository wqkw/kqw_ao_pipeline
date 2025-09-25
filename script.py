from openrouter_wrapper import llm
import base64
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, Field
from typing import List
import json
load_dotenv()

def print_truncated(data):
    """Print nested lists/dicts with items truncated at 200 characters"""
    def truncate_item(item):
        if isinstance(item, str) and len(item) > 200:
            return item[:200] + "..."
        elif isinstance(item, dict):
            return {k: truncate_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [truncate_item(x) for x in item]
        else:
            return item

    truncated = truncate_item(data)
    if isinstance(data, list):
        for item in truncated:
            print(item)
    else:
        print(truncated)

class ScenePrompt(BaseModel):
    model_config = {"extra": "forbid"}

    scene_name: str = Field(description="Name of the scene")
    prompt: str = Field(description="Detailed image generation prompt for the scene setting shot. Should include specific visual details, lighting, composition, characters, props, and atmosphere. Example: 'A dimly lit Victorian library with mahogany bookshelves reaching to the ceiling, warm amber lighting from brass lamps, leather armchair in foreground, mysterious figure in dark coat examining ancient tome, gothic atmosphere with dust motes floating in shafts of light'")

class ScenePromptsOutput(BaseModel):
    model_config = {"extra": "forbid"}

    scene_prompts: List[ScenePrompt] = Field(description="List of scene setting shot prompts for image generation")

METAPLAN_PROMPT = """
1. Make Base World Components
	•	Style & Aesthetic: global rules around lighting, textures, palettes.
	•	Characters: personality, dialogue style, relationships, motivations.
	•	Locations: distinct places with their own look/feel tied to the world.
	•	Props & Objects: context-rich items that carry meaning or drive the plot.

These are essentially your worldbuilding “components.”

⸻

2. Generate Storylines
	•	AI ingests those components and proposes multiple narrative arcs (3-5 options).
	•	Each storyline should:
	•	Use the world rules consistently.
	•	Highlight character motivations and conflicts.
	•	Incorporate props and locations naturally into the arc.

User picks one storyline to proceed with.

⸻

3. Generate Shot List & Scene Breakdown
	•	Once a storyline is chosen, AI produces:
	•	Shot list → camera framing, composition, focal points.
	•	Scene breakdown → sequence of beats, what needs to happen, what's shown.
	•	This merges narrative logic with visual planning.

A scene is composed of multiple shots. All shots in a scene should use a (mostly) shared set of components.

⸻

4. Generate prompts for scene stage setting shots
	•	Each scene starts with a canonical “stage setting shot”:
	•	The stage that carries all contextual DNA of the world, characters, and props.
	•	Acts as a visual anchor point.
	•	AI generates stage setting shots → human does a context check (world consistency, aesthetic match, continuity).
Create prompts for making the stage setting shots. These will be passed into an image generation model.

⸻

5. Shot Generation from Stage Setting Shots
	•	Individual frames/shots are generated from the stage setting shot.
	•	Each shot is checked for:
	•	Continuity (matches the stage setting shots/world).
	•	Contextual fidelity (characters/props behave as expected).

⸻

6. Iterative Correction Loop
	•	If a shot fails:
	1.	Human analyzes the mismatch.
	2.	Writes edit instructions.
	3.	AI regenerates the shot with corrections.
	4.	Repeat until the shot passes checks.

Approved shots flow into the storyboard.

⸻

7. End Product
	•	A storyboard with continuity + narrative alignment.
	•	Each shot linked back to its base scene and world DNA.
	•	Flexible enough to spin off multiple narrative arcs from the same world setup.

⸻

Where AI adds the most value here:
	•	Storyline ideation from raw world components.
	•	Shot/scene breakdowns structured for production.
	•	Visual anchor generation (base scenes).
	•	Automated continuity checks between shots and the world.

⸻

"""

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
        image_url=f"data:image/png;base64,{img_b64}"
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

    RESP3, _, history = llm(
        model="openai/gpt-5",
        text=PROMPT3,
        context=history,
    )
    print(RESP3)


if True:
    PROMPT4 = """

    Ok move on to step 4. Create the scene setting shot prompts.
    """

    RESP4, full_response, history = llm(
        model="openai/gpt-5",
        text=PROMPT4,
        # context=METAPLAN_PROMPT + '\n\n Shot list and scene breakdown: ' + RESP3,
        context=history,
        response_format=ScenePromptsOutput
        )

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

    








    # Generate narratives from the image
    response = llm(
        model="google/gemini-2.5-flash-image-preview",
        text="given the attached image of a world, generate some narratives from it",
        image_url=f"data:image/png;base64,{img_b64}"
    )

    print(response)

    image_url = response['choices'][0]['message']['images'][0]['image_url']['url']
    img_base64 = image_url.split("data:image/png;base64,")[1]
    img_data = base64.b64decode(img_base64)
    with open("data/test_out.png", "wb") as img_file:
        img_file.write(img_data)
