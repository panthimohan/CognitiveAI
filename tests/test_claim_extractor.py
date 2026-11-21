
import asyncio
import json
from types import SimpleNamespace

import pytest

from backend.learning.claim_extractor import (
    ClaimExtractionError,
    ClaimExtractor,
)
from backend.learning.contracts import (
    AuthorizationBasis,
    ExperienceSource,
    LearningDomain,
)


class FakeProvider:
    def __init__(self, content=None, success=True, error=None):
        self.content = content or '{"claims": []}'
        self.success = success
        self.error = error
        self.last_request = None

    async def generate(self, request):
        self.last_request = request

        return SimpleNamespace(
            success=self.success,
            content=self.content,
            error=self.error,
        )


def test_extracts_private_claims():
    provider = FakeProvider(
        json.dumps({
            "claims": [
                {
                    "key": "career_goal",
                    "value": "Wants to become an AI engineer.",
                    "confidence": 0.95,
                },
                {
                    "key": "learning_preference",
                    "value": "Prefers complete replacement files.",
                    "confidence": 0.99,
                },
            ]
        })
    )

    extractor = ClaimExtractor(provider)

    observations = asyncio.run(
        extractor.extract(
            "User: I want to become an AI engineer.",
            owner="test-user",
            source_ref="conversation-123",
        )
    )

    assert len(observations) == 2

    observation = observations[0]

    assert observation.key == "career_goal"
    assert observation.value == "Wants to become an AI engineer."
    assert observation.confidence == 0.95

    assert observation.experience.domain == LearningDomain.PRIVATE
    assert observation.experience.owner == "test-user"
    assert (
        observation.experience.source_type
        == ExperienceSource.PRIVATE_CONVERSATION
    )
    assert (
        observation.experience.authorization_basis
        == AuthorizationBasis.PRIVATE_CONTEXT
    )
    assert observation.experience.source_ref == "conversation-123"


def test_ignores_invalid_claims():
    provider = FakeProvider(
        json.dumps({
            "claims": [
                {
                    "key": "career_goal",
                    "value": "Wants to become an AI engineer.",
                    "confidence": 0.95,
                },
                {
                    "key": "Invalid Key!",
                    "value": "Invalid key.",
                    "confidence": 0.9,
                },
                {
                    "key": "missing_value",
                    "value": "",
                    "confidence": 0.9,
                },
                {
                    "key": "invalid_confidence",
                    "value": "Invalid confidence.",
                    "confidence": 2.0,
                },
            ]
        })
    )

    extractor = ClaimExtractor(provider)

    observations = asyncio.run(
        extractor.extract(
            "User: I want to become an AI engineer.",
            owner="test-user",
            source_ref="conversation-123",
        )
    )

    assert len(observations) == 1
    assert observations[0].key == "career_goal"


def test_empty_conversation_does_not_call_provider():
    provider = FakeProvider()
    extractor = ClaimExtractor(provider)

    observations = asyncio.run(
        extractor.extract(
            "   ",
            owner="test-user",
            source_ref="conversation-123",
        )
    )

    assert observations == []
    assert provider.last_request is None


def test_provider_failure_raises_error():
    provider = FakeProvider(
        success=False,
        error="Provider unavailable",
    )

    extractor = ClaimExtractor(provider)

    with pytest.raises(ClaimExtractionError):
        asyncio.run(
            extractor.extract(
                "User: I want to learn AI.",
                owner="test-user",
                source_ref="conversation-123",
            )
        )


def test_rejects_missing_owner():
    extractor = ClaimExtractor(FakeProvider())

    with pytest.raises(ValueError):
        asyncio.run(
            extractor.extract(
                "User: I want to learn AI.",
                owner="",
                source_ref="conversation-123",
            )
        )
