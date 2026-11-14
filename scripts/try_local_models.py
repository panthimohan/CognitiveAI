"""
Live smoke test for your local Ollama models.

Run from the project root, with `ollama serve` running:

    python scripts/try_local_models.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.models.ollama_provider import OllamaProvider
from backend.models.providers import ModelRequest
from config.settings import get_settings


QUESTIONS = {
    "general": "In one sentence, what is working memory?",
    "coding": "Write a Python function that reverses a string. Code only.",
    "reasoning": "A bat and ball cost $1.10 total. The bat costs $1.00 "
                 "more than the ball. How much is the ball? Be brief.",
}


async def main() -> None:

    settings = get_settings()
    print(f"Ollama server: {settings.ollama_base_url}\n")

    for role, question in QUESTIONS.items():

        model = settings.local_models[role]
        provider = OllamaProvider(model=model)

        print(f"=== {role}: {model} ===")

        if not provider.is_available():
            print("NOT AVAILABLE (server down or model not pulled)\n")
            continue

        started = time.time()
        response = await provider.generate(
            ModelRequest(prompt=question, max_tokens=400)
        )
        seconds = time.time() - started

        if response.success:
            print(f"({seconds:.1f}s) {response.content}\n")
        else:
            print(f"FAILED: {response.error}\n")


asyncio.run(main())
