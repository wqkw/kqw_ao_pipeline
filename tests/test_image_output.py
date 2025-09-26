#!/usr/bin/env python3
"""
Test the output_is_image functionality of the llm() function using Gemini 2.5 Flash.
"""

import os
from openrouter_wrapper import llm
from dotenv import load_dotenv
load_dotenv()

def test_image_generation():
    """Test image generation with Gemini 2.5 Flash and output_is_image flag."""

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Use Gemini 2.5 Flash for image generation
    model = "google/gemini-2.5-flash-image-preview"

    # Prompt for image generation
    prompt = "Generate a simple, colorful abstract geometric pattern with circles and triangles in bright colors like blue, red, yellow, and green on a white background."

    try:
        print("Generating image with Gemini 2.5 Flash...")

        # Get the response
        response_content, full_response, message_history = llm(
            model=model,
            text=prompt,
            reasoning_effort="low",
            output_is_image=True
        )

        # The response_content should now be bytes (image data)
        image_data = response_content

        # Save to data folder
        output_path = "data/img_gen_test.png"
        with open(output_path, "wb") as f:
            f.write(image_data)

        print(f"Image saved to {output_path}")
        print(f"Image size: {len(image_data)} bytes")

    except Exception as e:
        print(f"Error during test: {e}")
        raise

if __name__ == "__main__":
    test_image_generation()