from openrouter_wrapper import llm
import base64
from dotenv import load_dotenv
import requests
load_dotenv()

METAPLAN_PROMPT = """
We are following this overall workflow. 

1. **World Components**
* Style/aesthetic (lighting, textures, palettes)
* Characters (personality, dialogue, relationships, motivations)
* Locations (distinct looks tied to the world)
* Props/objects (context-rich items driving meaning/plot)
  These are the worldbuilding “atoms.”

2. **Storyline Generation**
* AI proposes 3-5 narrative arcs using the atoms
* Arcs follow world rules, surface motivations/conflicts, and use props/locations naturally
* User selects one arc

3. **Shot List & Scene Breakdown**
* AI produces a shot list (framing, composition, focal points)
* AI outlines scene beats (sequence, actions, visuals)

4. **Base Scene Creation**
* Create a canonical base scene carrying world/character/prop context
* AI generates; human checks for consistency, aesthetic match, and continuity

5. **Shot Generation**
* Generate shots from the base scene
* Check continuity with the world and contextual fidelity

6. **Iterative Correction**
* If a shot fails: analyze mismatch → write edit instructions → AI regenerates → repeat
* Approved shots move to the storyboard

7. **End Product**
* A storyboard with continuity and narrative alignment
* Each shot linked to its base scene and world DNA
* Supports branching into additional arcs

**Where AI adds the most value**
* Storyline ideation from world components
* Structured shot/scene breakdowns
* Base scene (visual anchor) generation
* Automated continuity checks across shots and world rules
"""

if True:
    PROMPT1 = """
    Based on this flow that we've outlined here, take this initial input of the base world components from this image (this is a mood board of that world that includes some characters, some locations, and some props all anchored to a consistent style). I want you to analyze this mood board and create the more detailed context for each of the base world components, where you add descriptions for:
    - the style and aesthetic
    - the characters
    - the locations
    - the props and objects
    to build out our world building atoms. Based on this mood board, create an outline of our world building atoms
    """


if True:
    # Encode the moodboard image
    with open("data/moodboard_tile.png", "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode()

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
