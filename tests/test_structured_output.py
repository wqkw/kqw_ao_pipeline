#!/usr/bin/env python3

from pydantic import BaseModel, Field
from typing import List
from openrouter_wrapper import llm
from dotenv import load_dotenv

load_dotenv()

class PersonInfo(BaseModel):
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job or profession")
    skills: List[str] = Field(description="List of key skills or abilities")
    location: str = Field(description="City and country where they live")

def test_structured_output():
    """Test the structured output functionality with GPT-5"""

    prompt = """
    Tell me about a fictional software engineer named Alex who works at a tech startup.
    Include their age, occupation, key programming skills, and location.
    """

    print("Testing structured output with GPT-5...")
    print(f"Prompt: {prompt}")
    print("\nRequesting structured response as PersonInfo model...")

    try:
        response = llm(
            model="openai/gpt-5",
            text=prompt,
            reasoning_effort="low",
            response_format=PersonInfo
        )

        print("\nRaw API Response:")
        print(response)

        # Extract the structured content
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            print(f"\nStructured Content:")
            print(content)

            # Try to parse it as our Pydantic model
            import json
            parsed_data = json.loads(content)
            person = PersonInfo(**parsed_data)

            print(f"\nParsed PersonInfo object:")
            print(f"Name: {person.name}")
            print(f"Age: {person.age}")
            print(f"Occupation: {person.occupation}")
            print(f"Skills: {', '.join(person.skills)}")
            print(f"Location: {person.location}")

        else:
            print("No choices found in response")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_structured_output()