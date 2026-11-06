"""
CognitiveAI - Universal AI Provider Interface
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel


class ModelRequest(BaseModel):
    prompt: str
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2000


class ModelResponse(BaseModel):
    provider: str
    model: str
    content: str
    success: bool
    error: str | None = None


class AIProvider(ABC):

    @abstractmethod
    async def generate(
        self,
        request: ModelRequest
    ) -> ModelResponse:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        pass