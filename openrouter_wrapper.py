import requests
import json
import os
import base64
import asyncio
import aiohttp
import time
import random
from datetime import datetime
from typing import Optional, Union, List, Type, Tuple, Dict, Any
from pydantic import BaseModel

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

ReasoningEffort = Literal["minimal", "low", "medium", "high"]

def _log_llm_call(start_time: datetime, end_time: datetime, tokens_in: int, tokens_out: int, function_name: str, prompt_preview: str):
    """Log LLM call information to log.txt"""
    duration = (end_time - start_time).total_seconds()
    log_line = f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} | {function_name} | Duration: {duration:.2f}s | Tokens In: {tokens_in} | Tokens Out: {tokens_out} | Prompt: {prompt_preview}\n"

    with open("llm_log.txt", "a", encoding="utf-8") as f:
        f.write(log_line)

def _count_tokens_in_messages(messages: List[Dict[str, Any]]) -> int:
    """Rough token count estimation for input messages"""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    total_chars += len(item.get("text", ""))
    # Rough estimation: ~4 characters per token
    return total_chars // 4


def _build_messages(
    context: Optional[Union[str, List[Dict[str, Any]]]],
    text: str,
    input_image_path: Optional[Union[str, List[str]]] = None
) -> List[Dict[str, Any]]:
    """Build message list for API request.

    Args:
        context: System message (string) or conversation history (list)
        text: User's text prompt
        input_image_path: Optional image URL(s) to include

    Returns:
        List of message dicts ready for API
    """
    messages = []

    # Handle context parameter - either string or list of messages
    if context:
        if isinstance(context, str):
            messages.append({"role": "system", "content": context})
        elif isinstance(context, list):
            messages.extend(context)

    # Prepare the user message content
    content = [{"type": "text", "text": text}]

    if input_image_path:
        if isinstance(input_image_path, str):
            content.append({"type": "image_path", "image_path": {"url": input_image_path}})
        elif isinstance(input_image_path, list):
            for url in input_image_path:
                content.append({"type": "image_path", "image_path": {"url": url}})

    messages.append({"role": "user", "content": content})
    return messages


def _build_payload(
    model: str,
    messages: List[Dict[str, Any]],
    reasoning_effort: Optional[ReasoningEffort],
    reasoning_exclude: bool,
    response_format: Optional[Type[BaseModel]]
) -> Dict[str, Any]:
    """Build API request payload.

    Args:
        model: Model identifier
        messages: Message list from _build_messages()
        reasoning_effort: Reasoning level or None
        reasoning_exclude: Whether to exclude reasoning from response
        response_format: Optional Pydantic model for structured output

    Returns:
        Payload dict ready for API request
    """
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

    return payload


def _extract_image_url(full_response: Dict[str, Any]) -> Optional[str]:
    """Extract image URL from API response.

    Handles multiple response formats:
    - {"images": [{"image_url": {"url": "data:image/..."}}]}
    - {"images": [{"image_url": "data:image/..."}]}
    - Content field with data URL

    Args:
        full_response: Full API response dict

    Returns:
        Image data URL string or None if not found
    """
    # Method 1: Check for images array in message
    try:
        images = full_response.get('choices', [{}])[0].get('message', {}).get('images', [])
        if images and len(images) > 0:
            image_url_field = images[0].get('image_url')
            if isinstance(image_url_field, dict):
                return image_url_field.get('url')
            elif isinstance(image_url_field, str):
                return image_url_field
    except (KeyError, IndexError, TypeError):
        pass

    # Method 2: Check if message content itself is a data URL
    try:
        content = full_response['choices'][0]['message']['content']
        if content and isinstance(content, str) and content.startswith('data:image/'):
            return content
    except (KeyError, IndexError, TypeError):
        pass

    return None


def _decode_image_data(image_data_url: str) -> bytes:
    """Decode base64 image data from data URL.

    Args:
        image_data_url: Data URL string (e.g., "data:image/png;base64,...")

    Returns:
        Decoded image bytes

    Raises:
        ValueError: If decoding fails
    """
    try:
        # Extract the base64 part after the comma
        base64_data = image_data_url.split(',', 1)[1]
        return base64.b64decode(base64_data)
    except (IndexError, base64.binascii.Error) as e:
        raise ValueError(f"Failed to decode base64 image data: {str(e)}")


def _save_image_debug_info(
    model: str,
    text: str,
    full_response: Dict[str, Any],
    image_data_url: Optional[str] = None
) -> None:
    """Save debug information when image generation fails.

    Args:
        model: Model identifier
        text: Original prompt
        full_response: Full API response
        image_data_url: Extracted image URL (if any)
    """
    try:
        images = full_response.get('choices', [{}])[0].get('message', {}).get('images', [])
        message_content = full_response.get('choices', [{}])[0].get('message', {}).get('content', '')
    except (KeyError, IndexError):
        images = None
        message_content = None

    debug_info = {
        "timestamp": datetime.now().isoformat(),
        "error": "No base64 image data found in response",
        "model": model,
        "prompt": text[:200] if text else "None",
        "images_array": images if images else 'None',
        "image_data_url": image_data_url if image_data_url else 'Not set',
        "message_content": str(message_content)[:500] if message_content else 'None',
        "full_response_keys": list(full_response.keys()) if isinstance(full_response, dict) else 'Not a dict',
        "message_structure": {
            "has_choices": 'choices' in full_response if isinstance(full_response, dict) else False,
            "message_keys": list(full_response.get('choices', [{}])[0].get('message', {}).keys()) if isinstance(full_response, dict) and 'choices' in full_response else 'None'
        }
    }

    try:
        with open("image_generation_debug.txt", "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"IMAGE GENERATION FAILURE - {debug_info['timestamp']}\n")
            f.write(f"{'='*80}\n")
            f.write(json.dumps(debug_info, indent=2))
            f.write(f"\n{'='*80}\n")
    except Exception:
        pass  # Don't fail if debug logging fails


def _parse_structured_response(
    message_content: Union[str, bytes],
    response_format: Optional[Type[BaseModel]]
) -> Union[str, bytes, BaseModel]:
    """Parse structured output if requested.

    Args:
        message_content: Raw message content (string or bytes)
        response_format: Optional Pydantic model class

    Returns:
        Parsed Pydantic model or original content if parsing fails
    """
    if response_format is not None and message_content and isinstance(message_content, str):
        try:
            return response_format.model_validate_json(message_content)
        except (json.JSONDecodeError, Exception):
            pass  # Return original content if parsing fails
    return message_content


def _process_image_response(
    full_response: Dict[str, Any],
    model: str,
    text: str
) -> bytes:
    """Process and decode image from API response.

    Args:
        full_response: Full API response dict
        model: Model identifier (for debug)
        text: Original prompt (for debug)

    Returns:
        Decoded image bytes

    Raises:
        ValueError: If no valid image data found
    """
    image_data_url = _extract_image_url(full_response)

    if image_data_url and image_data_url.startswith('data:image/'):
        return _decode_image_data(image_data_url)
    else:
        _save_image_debug_info(model, text, full_response, image_data_url)
        raise ValueError(
            f"No base64 image data found in response. "
            f"[Debug info saved to image_generation_debug.txt]"
        )


def llm(
    model: str,
    text: str,
    context: Optional[Union[str, List[Dict[str, Any]]]] = None,
    input_image_path: Optional[Union[str, List[str]]] = None,
    reasoning_effort: ReasoningEffort = "medium",  # default
    reasoning_exclude: bool = True,
    response_format: Optional[Type[BaseModel]] = None,
    output_is_image: bool = False,
    logging: bool = True,
    image_generation_retries: int = 1,
) -> Tuple[Union[str, bytes], dict, List[Dict[str, Any]]]:
    """
    OpenRouter API wrapper function

    Args:
        model: The model to use (e.g., "openai/gpt-5" or "google/gemini-2.5-flash-image-preview")
        text: The main text prompt
        context: Optional context/system message (string) or list of message history
        input_image_path: Optional input image URL(s) - can be a single URL string or list of URLs
        reasoning_effort: How much the model "thinks": "minimal" | "low" | "medium" | "high".
                          Use None to omit the 'reasoning' field entirely.
        reasoning_exclude: If True (default), request that intermediate reasoning not be returned.
        response_format: Optional Pydantic model class for structured output
        output_is_image: If True, expects the response to contain base64 image data and returns decoded bytes
        logging: If True (default), log call details to log.txt
        image_generation_retries: Number of retries for failed image generation (default 1)

    Returns:
        Tuple of (message_content: Union[str, bytes], full_response: dict, message_history: List[Dict[str, Any]])
        If output_is_image is True, message_content will be bytes (decoded base64 image data)
    """
    start_time = datetime.now() if logging else None

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not found")

    # Build messages and payload using helpers
    messages = _build_messages(context, text, input_image_path)
    payload = _build_payload(model, messages, reasoning_effort, reasoning_exclude, response_format)

    # Retry loop for image generation or JSON decode errors
    max_json_retries = 3
    max_image_retries = image_generation_retries if output_is_image else 0
    total_attempts = max_json_retries + max_image_retries

    for attempt in range(total_attempts):
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
            if attempt < max_json_retries - 1:
                delay = 0.5 + random.uniform(0, 0.5)
                print(f"JSON decode error on attempt {attempt + 1}, retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            else:
                print(f"Error: Non-JSON response from API after {max_json_retries} attempts. Status: {response.status_code}")
                print(f"Response text: {response.text}")
                raise

        # Extract message content from response
        try:
            message_content = full_response['choices'][0]['message']['content']
            assistant_message = full_response['choices'][0]['message']
        except (KeyError, IndexError):
            message_content = ""
            assistant_message = {"role": "assistant", "content": ""}

        # For image generation, check if we got valid image data
        if output_is_image:
            image_data_url = _extract_image_url(full_response)

            if not image_data_url or not image_data_url.startswith('data:image/'):
                if attempt < total_attempts - 1:
                    delay = 1.0 + random.uniform(0, 1.0)
                    print(f"No image data in response (attempt {attempt + 1}/{total_attempts}), retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                # Last attempt failed, will raise error in post-processing
            else:
                # Success! Break out of retry loop
                break
        else:
            # Not image generation, we're done
            break

    # Post-processing: handle images and structured output
    if output_is_image:
        message_content = _process_image_response(full_response, model, text)

    message_content = _parse_structured_response(message_content, response_format)

    # Build updated message history
    updated_messages = messages.copy()
    updated_messages.append(assistant_message)

    # Log the call if logging is enabled
    if logging and start_time:
        end_time = datetime.now()
        usage = full_response.get('usage', {})
        tokens_in = usage.get('prompt_tokens') or _count_tokens_in_messages(messages)
        tokens_out = usage.get('completion_tokens', 0)
        prompt_preview = text[:20] + "..." if len(text) > 20 else text
        _log_llm_call(start_time, end_time, tokens_in, tokens_out, "llm", prompt_preview)

    return message_content, full_response, updated_messages


async def llm_async(
    model: str,
    text: str,
    context: Optional[Union[str, List[Dict[str, Any]]]] = None,
    input_image_path: Optional[Union[str, List[str]]] = None,
    reasoning_effort: ReasoningEffort = "medium",
    reasoning_exclude: bool = True,
    response_format: Optional[Type[BaseModel]] = None,
    output_is_image: bool = False,
    logging: bool = True,
    _caller: str = "async",
    image_generation_retries: int = 1,
) -> Tuple[Union[str, bytes], dict, List[Dict[str, Any]]]:
    """
    Async version of the OpenRouter API wrapper function
    """
    start_time = datetime.now() if logging else None

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not found")

    # Build messages and payload using helpers
    messages = _build_messages(context, text, input_image_path)
    payload = _build_payload(model, messages, reasoning_effort, reasoning_exclude, response_format)

    # Retry loop for image generation or JSON decode errors
    max_json_retries = 3
    max_image_retries = image_generation_retries if output_is_image else 0
    total_attempts = max_json_retries + max_image_retries

    for attempt in range(total_attempts):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
            ) as response:
                try:
                    full_response = await response.json()
                except json.JSONDecodeError:
                    if attempt < max_json_retries - 1:
                        delay = 0.5 + random.uniform(0, 0.5)
                        print(f"JSON decode error on attempt {attempt + 1}, retrying in {delay:.1f}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        response_text = await response.text()
                        print(f"Error: Non-JSON response from API after {max_json_retries} attempts. Status: {response.status}")
                        print(f"Response text: {response_text}")
                        raise

        # Extract message content from response
        try:
            message_content = full_response['choices'][0]['message']['content']
            assistant_message = full_response['choices'][0]['message']
        except (KeyError, IndexError):
            message_content = ""
            assistant_message = {"role": "assistant", "content": ""}

        # For image generation, check if we got valid image data
        if output_is_image:
            image_data_url = _extract_image_url(full_response)

            if not image_data_url or not image_data_url.startswith('data:image/'):
                if attempt < total_attempts - 1:
                    delay = 1.0 + random.uniform(0, 1.0)
                    print(f"No image data in response (attempt {attempt + 1}/{total_attempts}), retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                # Last attempt failed, will raise error in post-processing
            else:
                # Success! Break out of retry loop
                break
        else:
            # Not image generation, we're done
            break

    # Post-processing: handle images and structured output
    if output_is_image:
        message_content = _process_image_response(full_response, model, text)

    message_content = _parse_structured_response(message_content, response_format)

    # Build updated message history
    updated_messages = messages.copy()
    updated_messages.append(assistant_message)

    # Log the call if logging is enabled
    if logging and start_time:
        end_time = datetime.now()
        usage = full_response.get('usage', {})
        tokens_in = usage.get('prompt_tokens') or _count_tokens_in_messages(messages)
        tokens_out = usage.get('completion_tokens', 0)
        prompt_preview = text[:20] + "..." if len(text) > 20 else text
        _log_llm_call(start_time, end_time, tokens_in, tokens_out, _caller, prompt_preview)

    return message_content, full_response, updated_messages


async def batch_llm(
    model: str,
    texts: List[str],
    context: Optional[Union[str, List[Dict[str, Any]]]] = None,
    image_paths: Optional[List[Optional[Union[str, List[str]]]]] = None,
    reasoning_effort: ReasoningEffort = "medium",
    reasoning_exclude: bool = True,
    response_format: Optional[Type[BaseModel]] = None,
    output_is_image: bool = False,
    logging: bool = True,
    image_generation_retries: int = 1,
) -> Tuple[List[Union[str, BaseModel, bytes]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process multiple LLM requests asynchronously

    Args:
        model: The model to use for all requests
        texts: List of text prompts
        context: Shared context for all requests
        image_paths: Optional list of image URLs (one per text, or None)
        reasoning_effort: Shared reasoning effort for all requests
        reasoning_exclude: Shared reasoning exclude setting
        response_format: Shared response format for all requests
        output_is_image: If True, decode base64 image data from responses
        logging: If True (default), log call details to log.txt for each request
        image_generation_retries: Number of retries for failed image generation (default 1)

    Returns:
        Tuple of (responses: List[Union[str, BaseModel, bytes]], full_responses: List[Dict[str, Any]], combined_history: List[Dict[str, Any]])
        If output_is_image is True, responses will contain bytes instead of strings
    """
    # Prepare image URLs list if not provided
    if image_paths is None:
        image_paths = [None] * len(texts)
    elif len(image_paths) != len(texts):
        raise ValueError("Length of image_paths must match length of texts")

    # Create tasks for all requests
    tasks = []
    for i, text in enumerate(texts):
        task = llm_async(
            model=model,
            text=text,
            context=context,
            input_image_path=image_paths[i],
            reasoning_effort=reasoning_effort,
            reasoning_exclude=reasoning_exclude,
            response_format=response_format,
            output_is_image=output_is_image,
            logging=logging,
            _caller="batch",
            image_generation_retries=image_generation_retries,
        )
        tasks.append(task)

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)


    # Extract responses, full responses, and build combined history
    responses = []
    full_responses = []
    combined_history = []

    # Start with the shared context if provided
    if context:
        if isinstance(context, str):
            combined_history.append({"role": "system", "content": context})
        elif isinstance(context, list):
            combined_history.extend(context)

    for i, (message_content, full_response, message_history) in enumerate(results):
        responses.append(message_content)
        full_responses.append(full_response)

        # Add this conversation's user and assistant messages to combined history
        # Skip the context/system messages since we already added them
        context_offset = 1 if context and isinstance(context, str) else (len(context) if context and isinstance(context, list) else 0)

        # Add user message and assistant response from this conversation
        if len(message_history) > context_offset:
            combined_history.extend(message_history[context_offset:])

    return responses, full_responses, combined_history
