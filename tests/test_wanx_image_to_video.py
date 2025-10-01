#!/usr/bin/env python3
"""
Test the Wanx image-to-video generation API from Alibaba Cloud Model Studio.
"""

import os
import time
import requests
from dotenv import load_dotenv
load_dotenv()


def test_wanx_image_to_video():
    """Test Wanx image-to-video generation with polling for results."""

    # Get API key from environment
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY environment variable not set")

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # API endpoint (using international/Singapore endpoint)
    endpoint = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"

    # Request payload
    payload = {
        "model": "wan2.5-i2v-preview",
        "input": {
            "prompt": "A cat running on the grass, smooth motion, natural lighting",
            # Use a sample image URL or provide your own
            "img_url": "https://dashscope.oss-cn-beijing.aliyuncs.com/samples/video/video_generation/i2v_cat.png"
        },
        "parameters": {
            "resolution": "720p",
            "duration": 5  # 5 seconds
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        print("Submitting image-to-video generation task...")

        # Submit the task
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        print(f"Response: {result}")

        # Check if we got a task_id (async) or direct output
        if "output" in result and "task_id" in result["output"]:
            task_id = result["output"]["task_id"]
            print(f"Task submitted successfully. Task ID: {task_id}")

            # Poll for results
            query_endpoint = f"https://dashscope-intl.aliyuncs.com/api/v1/tasks/{task_id}"
            max_attempts = 60  # Max 5 minutes (60 * 5 seconds)
            attempt = 0

            while attempt < max_attempts:
                attempt += 1
                print(f"Polling for results (attempt {attempt}/{max_attempts})...")

                time.sleep(5)  # Wait 5 seconds between polls

                query_response = requests.get(query_endpoint, headers=headers)
                query_response.raise_for_status()

                query_result = query_response.json()
                status = query_result.get("output", {}).get("task_status")

                print(f"Task status: {status}")

                if status == "SUCCEEDED":
                    video_url = query_result.get("output", {}).get("video_url")
                    print(f"\nVideo generation successful!")
                    print(f"Video URL (valid for 24 hours): {video_url}")

                    # Optionally download the video
                    if video_url:
                        print("\nDownloading video...")
                        video_response = requests.get(video_url)
                        video_response.raise_for_status()

                        output_path = "data/wanx_test_video.mp4"
                        with open(output_path, "wb") as f:
                            f.write(video_response.content)

                        print(f"Video saved to {output_path}")
                        print(f"Video size: {len(video_response.content)} bytes")

                    return

                elif status == "FAILED":
                    error_msg = query_result.get("output", {}).get("message", "Unknown error")
                    raise Exception(f"Video generation failed: {error_msg}")

            raise TimeoutError("Video generation timed out after 5 minutes")

        elif "output" in result and "video_url" in result["output"]:
            # Direct response with video URL
            video_url = result["output"]["video_url"]
            print(f"Video URL (valid for 24 hours): {video_url}")

            # Download the video
            print("\nDownloading video...")
            video_response = requests.get(video_url)
            video_response.raise_for_status()

            output_path = "data/wanx_test_video.mp4"
            with open(output_path, "wb") as f:
                f.write(video_response.content)

            print(f"Video saved to {output_path}")
            print(f"Video size: {len(video_response.content)} bytes")
        else:
            raise Exception(f"Unexpected response format: {result}")

    except requests.exceptions.RequestException as e:
        print(f"HTTP error during test: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        raise
    except Exception as e:
        print(f"Error during test: {e}")
        raise


if __name__ == "__main__":
    test_wanx_image_to_video()
