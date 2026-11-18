
import asyncio
import json
from types import SimpleNamespace

from backend.memory.extractor import (
    MemoryExtractor,
    MemoryExtractionError,
)


class FakeProvider:
    def __init__(self, content, success=True):
        self.content = content
        self.success = success

    async def generate(self, request):
        return SimpleNamespace(
            content=self.content,
            success=self.success,
            error=None if self.success else "Provider failed",
        )


def test_extracts_valid_memories():
    response = json.dumps({
        "memories": [
            {
                "content": "The user wants to become an AI engineer.",
                "category": "goal",
                "importance": 0.9,
                "confidence": 0.95,
            }
        ]
    })

    extractor = MemoryExtractor(FakeProvider(response))

    candidates = asyncio.run(
        extractor.extract("I want to become an AI engineer.")
    )

    assert len(candidates) == 1
    assert candidates[0].category == "goal"
    assert candidates[0].importance == 0.9


def test_ignores_invalid_categories():
    response = json.dumps({
        "memories": [
            {
                "content": "Some information.",
                "category": "invalid_category",
                "importance": 0.8,
                "confidence": 0.9,
            }
        ]
    })

    extractor = MemoryExtractor(FakeProvider(response))

    candidates = asyncio.run(
        extractor.extract("Some conversation.")
    )

    assert candidates == []


def test_empty_conversation_returns_empty_list():
    extractor = MemoryExtractor(FakeProvider("{}"))

    candidates = asyncio.run(extractor.extract("   "))

    assert candidates == []


def test_invalid_json_raises_error():
    extractor = MemoryExtractor(
        FakeProvider("This is not valid JSON")
    )

    try:
        asyncio.run(extractor.extract("I like learning AI."))
        assert False, "Expected MemoryExtractionError"
    except MemoryExtractionError:
        pass


def test_provider_failure_raises_error():
    extractor = MemoryExtractor(
        FakeProvider("{}", success=False)
    )

    try:
        asyncio.run(extractor.extract("Some conversation."))
        assert False, "Expected MemoryExtractionError"
    except MemoryExtractionError:
        pass
