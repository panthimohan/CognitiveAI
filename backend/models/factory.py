"""
CognitiveAI - Provider Factory

Creates a provider by name, so the rest of the system never needs
to know which class talks to which service.

    create_provider("local", "qwen2.5-coder:7b")
    create_provider("openai")

Anthropic and Google providers are registered once they exist.
"""

from typing import Callable

from backend.models.openai_provider import OpenAIProvider
from backend.models.ollama_provider import OllamaProvider
from backend.models.providers import AIProvider


def _make_local(model: str | None) -> AIProvider:
    return OllamaProvider(model=model)


def _make_openai(model: str | None) -> AIProvider:

    provider = OpenAIProvider()

    if model:
        provider.model = model

    return provider


PROVIDER_FACTORIES: dict[str, Callable[[str | None], AIProvider]] = {
    "local": _make_local,
    "ollama": _make_local,
    "openai": _make_openai,
}


def register_provider(
    name: str,
    factory: Callable[[str | None], AIProvider]
) -> None:
    PROVIDER_FACTORIES[name.lower()] = factory


def create_provider(name: str, model: str | None = None) -> AIProvider:

    key = name.lower()

    if key not in PROVIDER_FACTORIES:
        raise ValueError(
            f"Provider '{name}' is not available yet. "
            f"Available: {sorted(PROVIDER_FACTORIES)}"
        )

    return PROVIDER_FACTORIES[key](model)
