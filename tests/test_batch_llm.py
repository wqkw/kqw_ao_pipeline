#!/usr/bin/env python3

import asyncio
import time
from dotenv import load_dotenv
load_dotenv()

from openrouter_wrapper import llm, batch_llm

async def test_batch_llm():
    """Test the batch_llm function with multiple simple prompts and verify performance"""

    print("Testing batch_llm performance with multiple requests...")

    # Test data - using more requests to better show parallelization benefits
    model = "openai/gpt-5-mini"
    texts = [
        "What is 2+2?",
        "What is the capital of France?",
        "What color is the sky?",
        "Name one planet in our solar system.",
        "What is 5*3?",
        "What language is spoken in Brazil?",
        "How many sides does a triangle have?",
        "What is the largest ocean on Earth?"
    ]
    context = "You are a helpful assistant. Answer very briefly in one sentence."

    print(f"Testing with {len(texts)} requests...")

    # Time the batch request
    print("\n1. Running batch requests (async/parallel)...")
    start_time = time.time()
    responses, full_responses, combined_history = await batch_llm(
        model=model,
        texts=texts,
        context=context,
    )
    batch_time = time.time() - start_time

    print(f"✓ Batch requests completed in {batch_time:.2f} seconds")

    # Compare with sequential requests
    print("\n2. Running sequential requests...")
    start_time = time.time()
    sequential_responses = []
    for i, text in enumerate(texts):
        print(f"  Request {i+1}/{len(texts)}: {text}")
        response, _, _ = llm(
            model=model,
            text=text,
            context=context,
        )
        sequential_responses.append(response)
    sequential_time = time.time() - start_time

    print(f"✓ Sequential requests completed in {sequential_time:.2f} seconds")

    # Performance analysis
    speedup = sequential_time / batch_time
    efficiency = speedup / len(texts) * 100  # How much of theoretical max speedup we achieved

    print(f"\n{'='*60}")
    print("PERFORMANCE RESULTS:")
    print(f"{'='*60}")
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Batch time:      {batch_time:.2f}s")
    print(f"Speedup:         {speedup:.2f}x")
    print(f"Efficiency:      {efficiency:.1f}%")

    if speedup > 2.0:
        print("🚀 Excellent parallelization performance!")
    elif speedup > 1.5:
        print("✅ Good parallelization performance")
    elif speedup > 1.2:
        print("⚠️  Moderate parallelization performance")
    else:
        print("❌ Poor parallelization - check network/API limits")

    # Verify responses are reasonable
    print(f"\n{'='*60}")
    print("SAMPLE RESPONSES:")
    print(f"{'='*60}")
    for i, (question, batch_resp, seq_resp) in enumerate(zip(texts[:3], responses[:3], sequential_responses[:3])):
        print(f"\nQ: {question}")
        print(f"Batch:      {batch_resp}")
        print(f"Sequential: {seq_resp}")

    # Print some statistics about the new return values
    print(f"\n{'='*60}")
    print("NEW RETURN VALUE ANALYSIS:")
    print(f"{'='*60}")
    print(f"Responses: {len(responses)} items")
    print(f"Full responses: {len(full_responses)} items")
    print(f"Combined history: {len(combined_history)} messages")
    print(f"History message types: {[msg.get('role') for msg in combined_history]}")

    return responses, full_responses, combined_history

if __name__ == "__main__":
    asyncio.run(test_batch_llm())