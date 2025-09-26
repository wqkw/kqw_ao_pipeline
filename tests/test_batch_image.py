#!/usr/bin/env python3

import asyncio
import os
import time
from dotenv import load_dotenv
load_dotenv()

from openrouter_wrapper import batch_llm

async def test_batch_image_generation():
    """Test the batch_llm function for image generation and write images to data/ folder"""

    print("Testing batch image generation...")

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Test data - 3 different image generation prompts
    model = "google/gemini-2.5-flash-image-preview"
    texts = [
        "A serene mountain landscape at sunset with golden light reflecting on a lake",
        "A futuristic city skyline with flying cars and neon lights at night",
        "A cozy coffee shop interior with warm lighting and people reading books"
    ]

    print(f"Generating {len(texts)} images...")

    # Time the batch image generation
    start_time = time.time()
    responses, full_responses, combined_history = await batch_llm(
        model=model,
        texts=texts,
        output_is_image=True
    )
    batch_time = time.time() - start_time

    print(f"✓ Batch image generation completed in {batch_time:.2f} seconds")

    # Debug: Print response structure to understand the format
    print("\nDebugging response structure...")
    for i, full_response in enumerate(full_responses[:1]):  # Just show first response
        print(f"Full response keys: {list(full_response.keys())}")
        if 'choices' in full_response:
            choice = full_response['choices'][0]
            print(f"Choice keys: {list(choice.keys())}")
            if 'message' in choice:
                message = choice['message']
                print(f"Message keys: {list(message.keys())}")
                print(f"Message content preview: {str(message.get('content', ''))[:100]}...")
                if 'images' in message:
                    print(f"Images found: {len(message['images'])}")
                    for j, img in enumerate(message['images']):
                        print(f"  Image {j+1}: {list(img.keys())}")
        break

    # Save images to data/ folder
    print("\nSaving images to data/ folder...")
    for i, img_bytes in enumerate(responses):  # responses now contains bytes
        filename = f"data/generated_image_{i+1}.png"
        with open(filename, "wb") as f:
            f.write(img_bytes)
        print(f"✓ Saved {filename} ({len(img_bytes)} bytes)")

    print(f"\n{'='*60}")
    print("IMAGE GENERATION RESULTS:")
    print(f"{'='*60}")
    print(f"Generation time: {batch_time:.2f}s")
    print(f"Images generated: {len(responses)}")
    print(f"Images saved to: data/ folder")

    # Print prompts and corresponding filenames
    print(f"\nGenerated images:")
    for i, prompt in enumerate(texts):
        print(f"  {i+1}. data/generated_image_{i+1}.png - \"{prompt}\"")

    return responses, full_responses, combined_history

if __name__ == "__main__":
    asyncio.run(test_batch_image_generation())