"""
CognitiveAI - OpenAI Provider
"""

import os
import httpx

from backend.models.providers import (
    AIProvider,
    ModelRequest,
    ModelResponse
)


class OpenAIProvider(AIProvider):

    def __init__(self):

        self.api_key = os.getenv("OPENAI_API_KEY")

        self.model = os.getenv(
            "OPENAI_MODEL",
            "gpt-4.1"
        )

        self.base_url = "https://api.openai.com/v1"

    def get_provider_name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def generate(
        self,
        request: ModelRequest
    ) -> ModelResponse:

        if not self.is_available():

            return ModelResponse(
                provider="openai",
                model=self.model,
                content="",
                success=False,
                error="OpenAI API key is not configured."
            )

        messages = []

        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })

        messages.append({
            "role": "user",
            "content": request.prompt
        })

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:

            async with httpx.AsyncClient(timeout=60.0) as client:

                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )

                response.raise_for_status()

                data = response.json()

                content = data["choices"][0]["message"]["content"]

                return ModelResponse(
                    provider="openai",
                    model=self.model,
                    content=content,
                    success=True
                )

        except Exception as error:

            return ModelResponse(
                provider="openai",
                model=self.model,
                content="",
                success=False,
                error=str(error)
            )