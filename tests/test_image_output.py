#!/usr/bin/env python3
"""
Test the image_output functionality of the llm() function using Gemini 2.5 Flash.
"""

from openrouter_wrapper import llm
from dotenv import load_dotenv
load_dotenv()

def test_image_generation():
    """Test image generation with Gemini 2.5 Flash and image_output flag."""

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
            reasoning_effort="low"
        )

        # Extract image using standard format
        image_url = full_response['choices'][0]['message']['images'][0]['image_url']['url']

        # Test the image_output flag
        image_data = llm(
            model="dummy",
            text="dummy",
            image_url=image_url,
            image_output=True
        )

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