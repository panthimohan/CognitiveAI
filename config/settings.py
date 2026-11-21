
"""
CognitiveAI - Settings

Reads configuration from environment variables (and a local .env
file, which is git-ignored). Defaults match the models installed
through Ollama.

Override examples (in .env):
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL_CODING=qwen2.5-coder:7b
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


DEFAULT_LOCAL_MODELS = {
    "general": "llama3.2",
    "coding": "qwen2.5-coder:7b",
    "reasoning": "qwen3:8b",
    "embedding": "nomic-embed-text",
}

# Router task type -> which local model role handles it
TASK_TO_ROLE = {
    "coding": "coding",
    "reasoning": "reasoning",
    "research": "general",
    "conversation": "general",
}


@dataclass(frozen=True)
class Settings:

    ollama_base_url: str
    local_models: dict
    request_timeout: float

    def local_model_for(self, task_type: str) -> str:
        role = TASK_TO_ROLE.get(task_type, "general")
        return self.local_models[role]


def get_settings() -> Settings:

    models = {
        role: os.getenv(f"OLLAMA_MODEL_{role.upper()}", default)
        for role, default in DEFAULT_LOCAL_MODELS.items()
    }

    return Settings(
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        ).rstrip("/"),
        local_models=models,
        request_timeout=float(os.getenv("OLLAMA_TIMEOUT", "180")),
    )
