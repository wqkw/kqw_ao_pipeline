- openrouter_wrapper.py provides llm() function for API calls with image support, reasoning control, and structured outputs. All llm calls should use this.

llm(model, text, context=None, image_url=None, reasoning_effort="medium", reasoning_exclude=True, response_format=None)
- model: model name (e.g. "openai/gpt-5")
- text: main prompt
- context: system message
- image_url: single URL or list of URLs
- reasoning_effort: "minimal"|"low"|"medium"|"high"
- reasoning_exclude: exclude reasoning from response
- response_format: Pydantic model for structured output

IMPORTANT: For structured outputs, all Pydantic models MUST include:
model_config = {"extra": "forbid"}
This ensures additionalProperties: false in the JSON schema, which is required by Azure OpenAI.

For images, 

for all python commands use uv run 
