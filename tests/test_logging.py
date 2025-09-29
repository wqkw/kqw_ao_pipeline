#!/usr/bin/env python3

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if we have the API key
if not os.getenv('OPENROUTER_API_KEY'):
    print("OPENROUTER_API_KEY not found in environment variables")
    print("This test will fail, but you can see the logging structure.")
    sys.exit(1)

from openrouter_wrapper import llm, batch_llm

async def test_logging():
    print("Testing logging functionality...")

    try:
        # Test 1: Single call with logging enabled (default)
        print("\n1. Testing single LLM call...")
        response, full_response, history = llm(
            model="openai/gpt-5-mini",
            text="Say hello in exactly 3 words",
            logging=True
        )
        print("Success! Single call response:", response)

        # Test 2: Batch call with logging enabled
        print("\n2. Testing batch LLM call...")
        responses, full_responses, combined_history = await batch_llm(
            model="openai/gpt-5-mini",
            texts=[
                "Count to 3",
                "Name one color",
                "Say goodbye"
            ],
            logging=True
        )
        print("Success! Batch responses:", responses)

        # Test 3: Single call with logging disabled
        print("\n3. Testing with logging disabled...")
        response_no_log, _, _ = llm(
            model="openai/gpt-5-mini",
            text="This should not be logged",
            logging=False
        )
        print("Success! No-log response:", response_no_log)

        # Read and display the log file
        try:
            with open("llm_log.txt", "r") as f:
                print("\nLog file contents:")
                print(f.read())
        except FileNotFoundError:
            print("llm_log.txt file was not created")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_logging())