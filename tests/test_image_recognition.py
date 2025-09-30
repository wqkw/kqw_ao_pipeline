"""Test image recognition with Gemini 2.5 Flash and GPT-5"""
from dotenv import load_dotenv
from openrouter_wrapper import llm

load_dotenv()


def test_image_recognition(model: str):
    """Test that model can describe moodboard.png"""

    image_path = "data/moodboard.png"
    prompt = "Describe what you see in this image in detail."

    print(f"\nTesting image recognition with {model}...")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}\n")

    response, full_response, history = llm(
        model=model,
        text=prompt,
        input_image_path=image_path
    )

    print("Response:")
    print("-" * 80)
    print(response)
    print("-" * 80)

    assert response, "Response should not be empty"
    assert len(response) > 50, "Response should be detailed"

    print(f"\n✅ Image recognition test passed for {model}!")
    return response


if __name__ == "__main__":
    # Test Gemini 2.5 Flash
    print("=" * 80)
    print("Testing Gemini 2.5 Flash")
    print("=" * 80)
    test_image_recognition("google/gemini-2.5-flash")

    # Test GPT-5
    print("\n" + "=" * 80)
    print("Testing GPT-5")
    print("=" * 80)
    test_image_recognition("openai/gpt-5")