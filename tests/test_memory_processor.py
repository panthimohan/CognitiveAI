
from backend.memory.manager import MemoryManager
from backend.memory.processor import (
    MemoryCandidate,
    MemoryProcessor,
)


def test_processor_saves_personal_memory(tmp_path):
    manager = MemoryManager(
        owner="test-user",
        base_dir=tmp_path,
    )
    processor = MemoryProcessor(manager)

    candidate = MemoryCandidate(
        content="The user wants to become an AI engineer.",
        category="goal",
        importance=0.9,
        confidence=0.95,
    )

    result = processor.process(candidate)

    assert result.status == "saved"
    assert result.memory_id is not None

    stored = manager.personal_bg.all()

    assert any(
        item.id == result.memory_id
        for item in stored
    )


def test_processor_detects_duplicate(tmp_path):
    manager = MemoryManager(
        owner="test-user",
        base_dir=tmp_path,
    )
    processor = MemoryProcessor(manager)

    candidate = MemoryCandidate(
        content="The user wants to become an AI engineer.",
        category="goal",
        importance=0.9,
        confidence=0.95,
    )

    first = processor.process(candidate)
    second = processor.process(candidate)

    assert first.status == "saved"
    assert second.status == "duplicate"


def test_processor_rejects_empty_memory(tmp_path):
    manager = MemoryManager(
        owner="test-user",
        base_dir=tmp_path,
    )
    processor = MemoryProcessor(manager)

    candidate = MemoryCandidate(
        content="   ",
        category="profile",
    )

    result = processor.process(candidate)

    assert result.status == "rejected"


def test_processor_skips_low_confidence_memory(tmp_path):
    manager = MemoryManager(
        owner="test-user",
        base_dir=tmp_path,
    )
    processor = MemoryProcessor(manager)

    candidate = MemoryCandidate(
        content="Possibly interested in robotics.",
        category="interest" if False else "knowledge",
        importance=0.8,
        confidence=0.2,
    )

    result = processor.process(candidate)

    assert result.status == "skipped"
