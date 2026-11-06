"""
CognitiveAI - Model Intelligence Registry

Central registry for AI model providers.
Supports model registration, capability matching,
privacy-aware selection and provider discovery.
"""

from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


# ==========================================
# 1. MODEL SPECIFICATION
# ==========================================

class ModelSpecification(BaseModel):

    model_id: str
    provider: str
    model_name: str

    capabilities: list[str] = Field(default_factory=list)

    privacy_level: str = "external"

    supports_streaming: bool = False
    supports_tools: bool = False

    context_window: Optional[int] = None

    active: bool = True

    registered_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ==========================================
# 2. MODEL REGISTRY
# ==========================================

class ModelRegistry:

    def __init__(self):

        self.models: dict[str, ModelSpecification] = {}

        self._register_default_models()

    # --------------------------------------
    # Register available models
    # --------------------------------------

    def _register_default_models(self):

        defaults = [

            ModelSpecification(
                model_id="openai-gpt",
                provider="openai",
                model_name="GPT",
                capabilities=[
                    "reasoning",
                    "coding",
                    "conversation",
                    "analysis",
                    "research"
                ],
                privacy_level="external",
                supports_streaming=True,
                supports_tools=True
            ),

            ModelSpecification(
                model_id="anthropic-claude",
                provider="anthropic",
                model_name="Claude",
                capabilities=[
                    "reasoning",
                    "coding",
                    "analysis",
                    "writing",
                    "architecture"
                ],
                privacy_level="external",
                supports_streaming=True,
                supports_tools=True
            ),

            ModelSpecification(
                model_id="google-gemini",
                provider="google",
                model_name="Gemini",
                capabilities=[
                    "reasoning",
                    "research",
                    "multimodal",
                    "analysis",
                    "coding"
                ],
                privacy_level="external",
                supports_streaming=True,
                supports_tools=True
            ),

            ModelSpecification(
                model_id="meta-llama-local",
                provider="local",
                model_name="Llama",
                capabilities=[
                    "reasoning",
                    "coding",
                    "conversation",
                    "analysis"
                ],
                privacy_level="local",
                supports_streaming=True,
                supports_tools=False
            )
        ]

        for model in defaults:
            self.register_model(model)

    # --------------------------------------
    # Register a model
    # --------------------------------------

    def register_model(self, model: ModelSpecification):

        self.models[model.model_id] = model

    # --------------------------------------
    # Remove a model
    # --------------------------------------

    def remove_model(self, model_id: str):

        self.models.pop(model_id, None)

    # --------------------------------------
    # Get a model
    # --------------------------------------

    def get_model(self, model_id: str):

        return self.models.get(model_id)

    # --------------------------------------
    # List available models
    # --------------------------------------

    def list_models(self):

        return [
            model
            for model in self.models.values()
            if model.active
        ]

    # --------------------------------------
    # Capability matching
    # --------------------------------------

    def find_models(
        self,
        required_capabilities: list[str],
        privacy: str = "private"
    ):

        matches = []

        for model in self.list_models():

            if privacy == "private" and model.privacy_level != "local":
                continue

            if not required_capabilities:
                matches.append(model)
                continue

            if all(
                capability in model.capabilities
                for capability in required_capabilities
            ):
                matches.append(model)

        return matches

    # --------------------------------------
    # Select the best available model
    # --------------------------------------

    def select_model(
        self,
        required_capabilities: list[str],
        privacy: str = "private",
        preferred_provider: Optional[str] = None
    ):

        matches = self.find_models(
            required_capabilities=required_capabilities,
            privacy=privacy
        )

        if preferred_provider:

            preferred = [
                model
                for model in matches
                if model.provider == preferred_provider
            ]

            if preferred:
                return preferred[0]

        if matches:
            return matches[0]

        return None


# ==========================================
# 3. GLOBAL REGISTRY INSTANCE
# ==========================================

model_registry = ModelRegistry()