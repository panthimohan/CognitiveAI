"""
CognitiveAI - Fake Provider (for tests and demos)

Returns scripted answers so the supervisor pipeline can be tested
without API keys or a running model. When the script runs out,
the last answer is repeated.
"""

from backend.models.providers import (
    AIProvider,
    ModelRequest,
    ModelResponse
)


class FakeProvider(AIProvider):

    def __init__(
        self,
        responses: list[str] | None = None,
        name: str = "fake",
        model: str = "fake-model",
        available: bool = True
    ):

        self.responses = list(responses or ["fake answer"])
        self.name = name
        self.model = model
        self.available = available
        self.requests: list[ModelRequest] = []

    def get_provider_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        return self.available

    async def generate(self, request: ModelRequest) -> ModelResponse:

        self.requests.append(request)

        index = min(len(self.requests) - 1, len(self.responses) - 1)

        return ModelResponse(
            provider=self.name,
            model=self.model,
            content=self.responses[index],
            success=True
        )
