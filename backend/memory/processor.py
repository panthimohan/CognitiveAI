
"""
CognitiveAI - Automatic Memory Processor

Classifies, validates, checks and persists memory candidates.
Personal memory is the default. Public sharing is not automatic.
"""

from dataclasses import dataclass
from typing import Literal
import re

from .models import MemoryKind, MemoryScope


MemoryCategory = Literal[
    "profile",
    "experience",
    "goal",
    "decision",
    "outcome",
    "preference",
    "knowledge",
    "skill",
    "feedback",
    "relationship",
]


CATEGORY_TO_KIND = {
    "profile": MemoryKind.SEMANTIC,
    "experience": MemoryKind.EPISODIC,
    "goal": MemoryKind.SEMANTIC,
    "decision": MemoryKind.EPISODIC,
    "outcome": MemoryKind.EPISODIC,
    "preference": MemoryKind.SEMANTIC,
    "knowledge": MemoryKind.SEMANTIC,
    "skill": MemoryKind.PROCEDURAL,
    "feedback": MemoryKind.PROCEDURAL,
    "relationship": MemoryKind.SEMANTIC,
}


@dataclass
class MemoryCandidate:
    content: str
    category: MemoryCategory
    importance: float = 0.7
    confidence: float = 0.8


@dataclass
class MemoryProcessResult:
    status: str
    category: str
    content: str
    memory_id: str | None = None
    reason: str | None = None


class MemoryProcessor:
    """Processes candidates and saves useful personal memories."""

    def __init__(
        self,
        memory_manager,
        duplicate_threshold: float = 0.82,
        minimum_importance: float = 0.4,
        minimum_confidence: float = 0.6,
    ):
        self.memory_manager = memory_manager
        self.duplicate_threshold = duplicate_threshold
        self.minimum_importance = minimum_importance
        self.minimum_confidence = minimum_confidence

    @staticmethod
    def _normalize(content: str) -> set[str]:
        words = re.findall(r"\b[a-z0-9]+\b", content.lower())

        stopwords = {
            "the", "a", "an", "and", "or", "but", "is", "are",
            "was", "were", "be", "been", "to", "of", "in",
            "on", "at", "for", "with", "from", "by", "my",
            "i", "me", "we", "our", "you", "your",
        }

        return {
            word for word in words
            if word not in stopwords
        }

    @classmethod
    def _similarity(cls, first: str, second: str) -> float:
        first_words = cls._normalize(first)
        second_words = cls._normalize(second)

        if not first_words or not second_words:
            return 0.0

        intersection = first_words & second_words
        union = first_words | second_words

        return len(intersection) / len(union)

    def _find_duplicate(self, content: str):
        memories = self.memory_manager.recall(
            query=content,
            privacy="private",
            limit=10,
        )

        for memory in memories:
            if memory.scope != MemoryScope.PERSONAL:
                continue

            similarity = self._similarity(
                content,
                memory.content,
            )

            if similarity >= self.duplicate_threshold:
                return memory

        return None

    def process(
        self,
        candidate: MemoryCandidate,
    ) -> MemoryProcessResult:

        content = candidate.content.strip()
        category = candidate.category

        if not content:
            return MemoryProcessResult(
                status="rejected",
                category=str(category),
                content=content,
                reason="Memory content cannot be empty.",
            )

        if category not in CATEGORY_TO_KIND:
            return MemoryProcessResult(
                status="rejected",
                category=str(category),
                content=content,
                reason="Unsupported memory category.",
            )

        if not 0 <= candidate.importance <= 1:
            return MemoryProcessResult(
                status="rejected",
                category=category,
                content=content,
                reason="Importance must be between 0 and 1.",
            )

        if not 0 <= candidate.confidence <= 1:
            return MemoryProcessResult(
                status="rejected",
                category=category,
                content=content,
                reason="Confidence must be between 0 and 1.",
            )

        if candidate.importance < self.minimum_importance:
            return MemoryProcessResult(
                status="skipped",
                category=category,
                content=content,
                reason="Importance is below the storage threshold.",
            )

        if candidate.confidence < self.minimum_confidence:
            return MemoryProcessResult(
                status="skipped",
                category=category,
                content=content,
                reason="Confidence is below the storage threshold.",
            )

        stored_content = f"[{category}] {content}"

        duplicate = self._find_duplicate(stored_content)

        if duplicate is not None:
            return MemoryProcessResult(
                status="duplicate",
                category=category,
                content=content,
                memory_id=duplicate.id,
                reason="A similar personal memory already exists.",
            )

        memory = self.memory_manager.remember(
            content=stored_content,
            scope=MemoryScope.PERSONAL,
            kind=CATEGORY_TO_KIND[category],
            importance=candidate.importance,
        )

        # Persist immediately in long-term personal memory.
        self.memory_manager.personal_bg.add(memory)

        # Remove the temporary foreground copy.
        self.memory_manager.personal_fg.remove(memory.id)

        return MemoryProcessResult(
            status="saved",
            category=category,
            content=content,
            memory_id=memory.id,
        )

    def process_many(
        self,
        candidates: list[MemoryCandidate],
    ) -> list[MemoryProcessResult]:

        return [
            self.process(candidate)
            for candidate in candidates
        ]
