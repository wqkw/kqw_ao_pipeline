import requests
import json
import os
import base64
from typing import Optional, Union, List, Type, Tuple, Dict, Any
from pydantic import BaseModel

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

ReasoningEffort = Literal["minimal", "low", "medium", "high"]

def llm(
    model: str,
    text: str,
    context: Optional[Union[str, List[Dict[str, Any]]]] = None,
    image_url: Optional[Union[str, List[str]]] = None,
    reasoning_effort: ReasoningEffort = "medium",  # default
    reasoning_exclude: bool = True,
    response_format: Optional[Type[BaseModel]] = None,
    image_output: bool = False,
) -> Union[Tuple[str, dict, List[Dict[str, Any]]], bytes]:
    """
    OpenRouter API wrapper function

    Args:
        model: The model to use (e.g., "openai/gpt-5" or "google/gemini-2.5-flash-image-preview")
        text: The main text prompt
        context: Optional context/system message (string) or list of message history
        image_url: Optional image URL(s) - can be a single URL string or list of URLs
        reasoning_effort: How much the model "thinks": "minimal" | "low" | "medium" | "high".
                          Use None to omit the 'reasoning' field entirely.
        reasoning_exclude: If True (default), request that intermediate reasoning not be returned.
        response_format: Optional Pydantic model class for structured output
        image_output: If True, looks for base64 data in image_url and returns decoded bytes

    Returns:
        If image_output is True: bytes (decoded base64 image data)
        Otherwise: Tuple of (message_content: str, full_response: dict, message_history: List[Dict[str, Any]])
    """
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not found")

    # Handle image_output mode
    if image_output:
        if not image_url:
            raise ValueError("image_output is True but no image_url provided")

        # Handle single URL or list of URLs
        url_to_decode = image_url[0] if isinstance(image_url, list) else image_url

        # Check if it's a base64 data URL
        if url_to_decode.startswith('data:image/'):
            # Extract the base64 part after the comma
            base64_data = url_to_decode.split(',', 1)[1]
            return base64.b64decode(base64_data)
        else:
            raise ValueError("image_output requires a base64 data URL (data:image/...)")

    messages = []

    # Handle context parameter - either string or list of messages
    if context:
        if isinstance(context, str):
            # Traditional string context - add as system message
            messages.append({"role": "system", "content": context})
        elif isinstance(context, list):
            # List of messages - add them all to maintain conversation history
            messages.extend(context)

    # Prepare the user message content
    content = [{"type": "text", "text": text}]

    if image_url:
        if isinstance(image_url, str):
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        elif isinstance(image_url, list):
            for url in image_url:
                content.append({"type": "image_url", "image_url": {"url": url}})

    # Add the new user message
    new_user_message = {"role": "user", "content": content}
    messages.append(new_user_message)

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

    try:
        full_response = response.json()
    except json.JSONDecodeError:
        print(f"Error: Non-JSON response from API. Status: {response.status_code}")
        print(f"Response text: {response.text}")
        raise

    # Extract message content from the response
    try:
        message_content = full_response['choices'][0]['message']['content']
        assistant_message = full_response['choices'][0]['message']
    except (KeyError, IndexError):
        message_content = ""
        assistant_message = {"role": "assistant", "content": ""}

    # If structured output was requested, parse the JSON response into the Pydantic model
    if response_format is not None and message_content:
        try:
            parsed_content = response_format.model_validate_json(message_content)
            message_content = parsed_content
        except (json.JSONDecodeError, Exception):
            # If parsing fails, return the raw string content
            pass

    # Build the updated message history including the new assistant response
    updated_messages = messages.copy()
    updated_messages.append(assistant_message)

    return message_content, full_response, updated_messages
