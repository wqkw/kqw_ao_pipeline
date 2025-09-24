import requests
import json
import os
from typing import Optional, Union, List, Type
from pydantic import BaseModel

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

ReasoningEffort = Literal["minimal", "low", "medium", "high"]

def llm(
    model: str,
    text: str,
    context: Optional[str] = None,
    image_url: Optional[Union[str, List[str]]] = None,
    reasoning_effort: ReasoningEffort = "medium",  # default
    reasoning_exclude: bool = True,
    response_format: Optional[Type[BaseModel]] = None,
) -> dict:
    """
    Simple OpenRouter API wrapper function

    Args:
        model: The model to use (e.g., "openai/gpt-5" or "google/gemini-2.5-flash-image-preview")
        text: The main text prompt
        context: Optional context/system message
        image_url: Optional image URL(s) - can be a single URL string or list of URLs
        reasoning_effort: How much the model "thinks": "minimal" | "low" | "medium" | "high".
                          Use None to omit the 'reasoning' field entirely.
        reasoning_exclude: If True (default), request that intermediate reasoning not be returned.
        response_format: Optional Pydantic model class for structured output

    Returns:
        Response from OpenRouter API as dict
    """
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not found")

    messages = []

    if context:
        messages.append({"role": "system", "content": context})

    content = [{"type": "text", "text": text}]

    if image_url:
        if isinstance(image_url, str):
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        elif isinstance(image_url, list):
            for url in image_url:
                content.append({"type": "image_url", "image_url": {"url": url}})

    messages.append({"role": "user", "content": content})

    payload = {"model": model, "messages": messages}

    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort, "exclude": reasoning_exclude}

    if response_format is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": response_format.model_json_schema()
            }
        }

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
    )

    return response.json()
