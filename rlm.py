"""
Use dspy.RLM to process a file with a request.

Usage: python rlm.py <file_path> <request>
"""

import os
import sys

import dspy
import openai
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

MODEL = "gpt-5-mini"


def validate_setup():
    """Check API key and model availability before running."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment")
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)
    try:
        models = [m.id for m in client.models.list()]
        if MODEL not in models:
            print(f"Error: Model '{MODEL}' not available. Available models:")
            for m in sorted(models):
                if "gpt" in m or "o1" in m or "o3" in m:
                    print(f"  {m}")
            sys.exit(1)
    except openai.AuthenticationError:
        print("Error: Invalid OPENAI_API_KEY")
        sys.exit(1)

    return api_key


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file_path> <request>")
        sys.exit(1)

    api_key = validate_setup()

    file_path = sys.argv[1]
    request = sys.argv[2]

    with open(file_path) as f:
        content = f.read()

    lm = dspy.LM(f"openai/{MODEL}", api_key=api_key)
    dspy.configure(lm=lm)

    rlm = dspy.RLM("content, request -> ideas: list[str]")

    output = rlm(content=content, request=request)

    for i, idea in enumerate(output.ideas, 1):
        print(f"{i}) {idea}")


if __name__ == "__main__":
    main()
