- openrouter_wrapper.py provides llm() function for API calls with image support, reasoning control, and structured outputs. All llm calls should use this.

llm(model, text, context=None, input_image_url=None, reasoning_effort="medium", reasoning_exclude=True, response_format=None, output_is_image=False)
- model: model name (e.g. "openai/gpt-5")
- text: main prompt
- context: system message
- input_image_url: single URL or list of URLs for input images
- reasoning_effort: "minimal"|"low"|"medium"|"high"
- reasoning_exclude: exclude reasoning from response
- response_format: Pydantic model for structured output
- output_is_image: If True, expects response to contain image data and returns bytes in first element of tuple

For batch/parallel processing:
batch_llm(model, texts, context=None, image_urls=None, reasoning_effort="medium", reasoning_exclude=True, response_format=None, output_is_image=False)
- texts: List of prompts to process concurrently
- image_urls: Optional list of input image URLs (one per text, or None)
- output_is_image: If True, decode base64 image data from responses
- Returns: (responses, full_responses, combined_history) tuple where responses contains strings or bytes depending on output_is_image flag

Usage: responses, full_responses, combined_history = await batch_llm("openai/gpt-5", ["prompt1", "prompt2"])

IMPORTANT: For structured outputs, all Pydantic models MUST include:
model_config = {"extra": "forbid"}
This ensures additionalProperties: false in the JSON schema, which is required by Azure OpenAI.

For default models, never use gpt 4o or 4o-mini, always use gpt 5, 5-mini, etc.
Don't use __file__ for the most part

creating new files that need credentials, remember to do:
from dotenv import load_dotenv
load_dotenv()

for all python commands use uv run 
