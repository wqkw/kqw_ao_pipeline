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

def llm(
    model: str,
    text: str,
    context: Optional[Union[str, List[Dict[str, Any]]]] = None,
    input_image_url: Optional[Union[str, List[str]]] = None,
    reasoning_effort: ReasoningEffort = "medium",  # default
    reasoning_exclude: bool = True,
    response_format: Optional[Type[BaseModel]] = None,
    output_is_image: bool = False,
    logging: bool = True,
) -> Tuple[Union[str, bytes], dict, List[Dict[str, Any]]]:
    """
    OpenRouter API wrapper function

    Args:
        model: The model to use (e.g., "openai/gpt-5" or "google/gemini-2.5-flash-image-preview")
        text: The main text prompt
        context: Optional context/system message (string) or list of message history
        input_image_url: Optional input image URL(s) - can be a single URL string or list of URLs
        reasoning_effort: How much the model "thinks": "minimal" | "low" | "medium" | "high".
                          Use None to omit the 'reasoning' field entirely.
        reasoning_exclude: If True (default), request that intermediate reasoning not be returned.
        response_format: Optional Pydantic model class for structured output
        output_is_image: If True, expects the response to contain base64 image data and returns decoded bytes
        logging: If True (default), log call details to log.txt

    Returns:
        Tuple of (message_content: Union[str, bytes], full_response: dict, message_history: List[Dict[str, Any]])
        If output_is_image is True, message_content will be bytes (decoded base64 image data)
    """
    start_time = datetime.now() if logging else None

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not found")


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

    if input_image_url:
        if isinstance(input_image_url, str):
            content.append({"type": "image_url", "image_url": {"url": input_image_url}})
        elif isinstance(input_image_url, list):
            for url in input_image_url:
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

    max_retries = 3
    for attempt in range(max_retries):
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
            break  # Success, exit retry loop
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                delay = 0.5 + random.uniform(0, 0.5)  # 0.5-1.0 second delay
                print(f"JSON decode error on attempt {attempt + 1}, retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            else:
                print(f"Error: Non-JSON response from API after {max_retries} attempts. Status: {response.status_code}")
                print(f"Response text: {response.text}")
                raise

    # Extract message content from the response
    try:
        message_content = full_response['choices'][0]['message']['content']
        assistant_message = full_response['choices'][0]['message']
    except (KeyError, IndexError):
        message_content = ""
        assistant_message = {"role": "assistant", "content": ""}

    # Handle image output processing (when model generates images)
    if output_is_image:
        # Look for image data in the response structure
        image_data_url = None

        # Method 1: Check for structured response with image URLs (Gemini format)
        try:
            images = full_response.get('choices', [{}])[0].get('message', {}).get('images', [])
            if images and len(images) > 0:
                image_data_url = images[0].get('image_url', {}).get('url')
        except (KeyError, IndexError, TypeError):
            pass

        # Method 2: Check if message content itself is a base64 data URL
        if not image_data_url and message_content and isinstance(message_content, str) and message_content.startswith('data:image/'):
            image_data_url = message_content

        # Decode the base64 data URL
        if image_data_url and image_data_url.startswith('data:image/'):
            try:
                # Extract the base64 part after the comma
                base64_data = image_data_url.split(',', 1)[1]
                message_content = base64.b64decode(base64_data)
            except (IndexError, base64.binascii.Error) as e:
                raise ValueError(f"Failed to decode base64 image data: {str(e)}")
        else:
            raise ValueError(f"No base64 image data found in response. Images: {images if 'images' in locals() else 'None'}, Content: {str(message_content)[:100]}...")

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
    input_image_url: Optional[Union[str, List[str]]] = None,
    reasoning_effort: ReasoningEffort = "medium",
    reasoning_exclude: bool = True,
    response_format: Optional[Type[BaseModel]] = None,
    output_is_image: bool = False,
    logging: bool = True,
    _caller: str = "async",
) -> Tuple[Union[str, bytes], dict, List[Dict[str, Any]]]:
    """
    Async version of the OpenRouter API wrapper function
    """
    start_time = datetime.now() if logging else None

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not found")


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

    if input_image_url:
        if isinstance(input_image_url, str):
            content.append({"type": "image_url", "image_url": {"url": input_image_url}})
        elif isinstance(input_image_url, list):
            for url in input_image_url:
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

    max_retries = 3
    for attempt in range(max_retries):
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
                    break  # Success, exit retry loop
                except json.JSONDecodeError:
                    if attempt < max_retries - 1:
                        delay = 0.5 + random.uniform(0, 0.5)  # 0.5-1.0 second delay
                        print(f"JSON decode error on attempt {attempt + 1}, retrying in {delay:.1f}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        response_text = await response.text()
                        print(f"Error: Non-JSON response from API after {max_retries} attempts. Status: {response.status}")
                        print(f"Response text: {response_text}")
                        raise

    # Extract message content from the response
    try:
        message_content = full_response['choices'][0]['message']['content']
        assistant_message = full_response['choices'][0]['message']
    except (KeyError, IndexError):
        message_content = ""
        assistant_message = {"role": "assistant", "content": ""}

    # Handle image output processing (when model generates images)
    if output_is_image:
        # Look for image data in the response structure
        image_data_url = None

        # Method 1: Check for structured response with image URLs (Gemini format)
        try:
            images = full_response.get('choices', [{}])[0].get('message', {}).get('images', [])
            if images and len(images) > 0:
                image_data_url = images[0].get('image_url', {}).get('url')
        except (KeyError, IndexError, TypeError):
            pass

        # Method 2: Check if message content itself is a base64 data URL
        if not image_data_url and message_content and isinstance(message_content, str) and message_content.startswith('data:image/'):
            image_data_url = message_content

        # Decode the base64 data URL
        if image_data_url and image_data_url.startswith('data:image/'):
            try:
                # Extract the base64 part after the comma
                base64_data = image_data_url.split(',', 1)[1]
                message_content = base64.b64decode(base64_data)
            except (IndexError, base64.binascii.Error) as e:
                raise ValueError(f"Failed to decode base64 image data: {str(e)}")
        else:
            raise ValueError(f"No base64 image data found in response. Images: {images if 'images' in locals() else 'None'}, Content: {str(message_content)[:100]}...")

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
    image_urls: Optional[List[Optional[Union[str, List[str]]]]] = None,
    reasoning_effort: ReasoningEffort = "medium",
    reasoning_exclude: bool = True,
    response_format: Optional[Type[BaseModel]] = None,
    output_is_image: bool = False,
    logging: bool = True,
) -> Tuple[List[Union[str, BaseModel, bytes]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process multiple LLM requests asynchronously

    Args:
        model: The model to use for all requests
        texts: List of text prompts
        context: Shared context for all requests
        image_urls: Optional list of image URLs (one per text, or None)
        reasoning_effort: Shared reasoning effort for all requests
        reasoning_exclude: Shared reasoning exclude setting
        response_format: Shared response format for all requests
        output_is_image: If True, decode base64 image data from responses
        logging: If True (default), log call details to log.txt for each request

    Returns:
        Tuple of (responses: List[Union[str, BaseModel, bytes]], full_responses: List[Dict[str, Any]], combined_history: List[Dict[str, Any]])
        If output_is_image is True, responses will contain bytes instead of strings
    """
    # Prepare image URLs list if not provided
    if image_urls is None:
        image_urls = [None] * len(texts)
    elif len(image_urls) != len(texts):
        raise ValueError("Length of image_urls must match length of texts")

    # Create tasks for all requests
    tasks = []
    for i, text in enumerate(texts):
        task = llm_async(
            model=model,
            text=text,
            context=context,
            input_image_url=image_urls[i],
            reasoning_effort=reasoning_effort,
            reasoning_exclude=reasoning_exclude,
            response_format=response_format,
            output_is_image=output_is_image,
            logging=logging,
            _caller="batch",
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
