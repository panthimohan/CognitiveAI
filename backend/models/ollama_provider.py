
"""
CognitiveAI - Ollama Provider (local models)

Connects to a locally running Ollama server.
Private requests remain on the local machine.
"""

import re

import httpx

from backend.models.providers import (
    AIProvider,
    ModelRequest,
    ModelResponse,
)
from config.settings import get_settings


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove internal reasoning blocks from generated content."""
    return THINK_BLOCK.sub("", text).strip()


class OllamaProvider(AIProvider):

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        settings = get_settings()

        self.model = model or settings.local_models["general"]
        self.base_url = (
            base_url or settings.ollama_base_url
        ).rstrip("/")
        self.timeout = timeout or settings.request_timeout
        self._transport = transport

    def get_provider_name(self) -> str:
        return "local"

    def is_available(self) -> bool:
        try:
            with httpx.Client(
                timeout=3.0,
                transport=self._transport,
            ) as client:
                response = client.get(
                    f"{self.base_url}/api/tags"
                )
                response.raise_for_status()

                names = [
                    entry.get("name", "")
                    for entry in response.json().get("models", [])
                ]

        except Exception:
            return False

        return self._is_installed(names)

    def _is_installed(self, names: list[str]) -> bool:
        if ":" in self.model:
            return self.model in names

        return any(
            name == self.model
            or name.startswith(self.model + ":")
            for name in names
        )

    async def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:

        messages = []

        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt,
            })

        messages.append({
            "role": "user",
            "content": request.prompt,
        })

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,

            # Qwen3 may otherwise return reasoning separately
            # and leave message.content empty.
            "think": False,

            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                transport=self._transport,
            ) as client:

                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            message = data.get("message") or {}
            raw_content = message.get("content") or ""

            if not isinstance(raw_content, str):
                return self._failure(
                    "Ollama returned an invalid content format."
                )

            content = strip_reasoning(raw_content)

            if not content:
                return self._failure(
                    "Model returned an empty response. "
                    "Check the model output and Ollama configuration."
                )

            return ModelResponse(
                provider="local",
                model=self.model,
                content=content,
                success=True,
            )

        except httpx.ConnectError:
            return self._failure(
                f"Cannot reach Ollama at {self.base_url}. "
                "Make sure the Ollama application is running. "
                "If needed, start it with: ollama serve"
            )

        except httpx.TimeoutException:
            return self._failure(
                "Ollama request timed out. "
                "The model may need more time to respond."
            )

        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                return self._failure(
                    f"Model '{self.model}' not found. "
                    f"Pull it using: ollama pull {self.model}"
                )

            return self._failure(
                f"Ollama returned HTTP "
                f"{error.response.status_code}."
            )

        except Exception as error:
            return self._failure(
                f"Ollama provider error: {error}"
            )

    def _failure(self, message: str) -> ModelResponse:
        return ModelResponse(
            provider="local",
            model=self.model,
            content="",
            success=False,
            error=message,
        )
