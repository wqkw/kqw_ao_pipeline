"""
Test Gemini 2.5 Flash's ability to identify images based on filename metadata.

This test passes multiple images with id/filename metadata and asks the model
to identify which image has a specific property.
"""

import os
import sys
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

from openrouter_wrapper import llm

load_dotenv()


class ImageIdentification(BaseModel):
    """Response format for image identification."""
    model_config = {"extra": "forbid"}

    image_id: str
    reasoning: str


def test_gemini_filename_recognition():
    """Test if Gemini 2.5 Flash can see filename metadata in image prompts."""

    # Select a few images from the moodboard
    moodboard_dir = Path("data/ref_moodboard")

    # Pick some images to test with
    test_images = [
        {"id": "img_01", "filename": "00_00.png", "path": str(moodboard_dir / "00_00.png")},
        {"id": "img_02", "filename": "00_01.png", "path": str(moodboard_dir / "00_01.png")},
        {"id": "img_03", "filename": "00_02.png", "path": str(moodboard_dir / "00_02.png")},
        {"id": "img_04", "filename": "01_00.png", "path": str(moodboard_dir / "01_00.png")},
    ]

    # Verify all images exist
    for img in test_images:
        if not Path(img["path"]).exists():
            print(f"Error: Image not found: {img['path']}")
            return

    # Build the prompt with metadata
    prompt = "I'm showing you multiple images with their metadata:\n\n"
    for img in test_images:
        prompt += f"- {img['id']}: filename=\"{img['filename']}\"\n"

    prompt += "\nPlease analyze these images and tell me which image ID corresponds to the FIRST image in the sequence (based on the order I'm sending them to you).\n\n"
    prompt += "Respond with the image_id and your reasoning."

    # Prepare image paths for the API call
    image_paths = [img["path"] for img in test_images]

    print("Testing Gemini 2.5 Flash filename recognition...")
    print(f"Sending {len(test_images)} images with metadata")
    print(f"\nPrompt:\n{prompt}\n")
    print("="*80)

    # Call the LLM with structured output
    response, full_response, _ = llm(
        model="google/gemini-2.5-flash",
        text=prompt,
        input_image_path=image_paths,
        reasoning_effort="low",
        reasoning_exclude=True,
        response_format=ImageIdentification,
        logging=True
    )

    # Display results
    print("\n" + "="*80)
    print("RESPONSE:")
    print("="*80)

    if isinstance(response, ImageIdentification):
        print(f"Identified Image ID: {response.image_id}")
        print(f"Reasoning: {response.reasoning}")

        # Verify the response
        print("\n" + "="*80)
        print("VERIFICATION:")
        print("="*80)
        expected_id = test_images[0]["id"]  # First image should be img_01
        if response.image_id == expected_id:
            print(f"✓ CORRECT! Model correctly identified {expected_id} as the first image")
        else:
            print(f"✗ INCORRECT. Expected {expected_id}, got {response.image_id}")
    else:
        print("Response is not structured as expected:")
        print(response)


def test_content_based_identification():
    """Test identifying images based on visual content rather than just metadata."""

    moodboard_dir = Path('data/ref_moodboard')

    test_images = [
        {"id": "img_01", "filename": "00_00.png", "path": str(moodboard_dir / "00_00.png")},
        {"id": "img_02", "filename": "00_01.png", "path": str(moodboard_dir / "00_01.png")},
        {"id": "img_03", "filename": "00_02.png", "path": str(moodboard_dir / "00_02.png")},
    ]

    # Verify all images exist
    for img in test_images:
        if not Path(img["path"]).exists():
            print(f"Error: Image not found: {img['path']}")
            return

    # Build the prompt asking about content
    prompt = "I'm showing you multiple images with IDs:\n\n"
    for img in test_images:
        prompt += f"- {img['id']}: {img['filename']}\n"

    prompt += "\nAnalyze the visual content of these images and tell me which image has the warmest color palette or contains the most warm tones (reds, oranges, yellows).\n\n"
    prompt += "Respond with the image_id and your reasoning about the visual content."

    image_paths = [img["path"] for img in test_images]

    print("\n\n")
    print("="*80)
    print("Testing content-based image identification...")
    print(f"Sending {len(test_images)} images")
    print(f"\nPrompt:\n{prompt}\n")
    print("="*80)

    response, full_response, _ = llm(
        model="google/gemini-2.5-flash",
        text=prompt,
        input_image_path=image_paths,
        reasoning_effort="medium",
        reasoning_exclude=True,
        response_format=ImageIdentification,
        logging=True
    )

    print("\n" + "="*80)
    print("RESPONSE:")
    print("="*80)

    if isinstance(response, ImageIdentification):
        print(f"Identified Image ID: {response.image_id}")
        print(f"Reasoning: {response.reasoning}")
    else:
        print("Response is not structured as expected:")
        print(response)


if __name__ == "__main__":
    # Run both tests
    test_gemini_filename_recognition()
    test_content_based_identification()
