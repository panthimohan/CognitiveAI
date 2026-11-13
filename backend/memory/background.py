"""
CognitiveAI - Background (Long-Term) Memory

A large, persistent store for everything not currently in focus.
One instance serves one scope:

    - PERSONAL: one owner, stored at data/personal/<owner>/memory.json
    - PUBLIC:   shared,    stored at data/global/memory.json

Features:
    - Persistence to JSON (human-readable, atomic writes)
    - Keyword search ranked by relevance and strength
    - Decay: memories fade unless they are used or important
    - Retrieval strengthens memories (each recall counts as access)
    - Strict scope and owner checks; owner names cannot escape
      their own directory

Strength:
    retention = 0.5 ** (days_since_last_access / half_life_days)
    boost     = 1 + 0.1 * min(access_count, 10)
    strength  = min(1, importance * boost) * retention

Search score:
    score = 0.7 * relevance + 0.3 * strength
    relevance = fraction of query words found in the item
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Callable

from backend.memory.models import (
    MemoryItem,
    MemoryKind,
    MemoryScope,
    utc_now
)


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

OWNER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,99}$")

STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "of", "and",
    "in", "on", "for", "it", "i", "my", "me"
}

RELEVANCE_WEIGHT = 0.7
STRENGTH_WEIGHT = 0.3


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class BackgroundMemory:

    def __init__(
        self,
        scope: MemoryScope,
        owner: str | None = None,
        base_dir: Path | str | None = None,
        half_life_days: float = 30.0,
        clock: Callable[[], datetime] = utc_now
    ):

        if half_life_days <= 0:
            raise ValueError("Half-life must be positive.")

        if scope == MemoryScope.PERSONAL:

            if not owner:
                raise ValueError("Personal memory requires an owner.")

            if not OWNER_PATTERN.match(owner):
                raise ValueError(
                    "Invalid owner name: use letters, digits, "
                    "and _ . @ - only."
                )

        if scope == MemoryScope.PUBLIC and owner is not None:
            raise ValueError("Public memory must not have an owner.")

        self.scope = scope
        self.owner = owner
        self.half_life_days = half_life_days
        self._clock = clock

        base = Path(base_dir) if base_dir else DEFAULT_DATA_DIR

        if scope == MemoryScope.PERSONAL:
            self.path = base / "personal" / owner / "memory.json"
        else:
            self.path = base / "global" / "memory.json"

        self._items: dict[str, MemoryItem] = {}
        self._load()

    # --------------------------------------
    # Persistence
    # --------------------------------------

    def _load(self) -> None:

        if not self.path.exists():
            return

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            items = [
                MemoryItem.model_validate(raw)
                for raw in data["items"]
            ]
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(
                f"Memory file is corrupt: {self.path} ({error})"
            ) from error

        for item in items:
            self._check_belongs(item)

        self._items = {item.id: item for item in items}

    def _save(self) -> None:

        self.path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": 1,
            "items": [
                item.model_dump(mode="json")
                for item in self._items.values()
            ]
        }

        temp_path = self.path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8"
        )
        os.replace(temp_path, self.path)

    def _check_belongs(self, item: MemoryItem) -> None:

        if item.scope != self.scope or item.owner != self.owner:
            raise ValueError(
                "Item does not belong to this memory "
                "(scope or owner mismatch)."
            )

    # --------------------------------------
    # Strength and decay
    # --------------------------------------

    def strength(self, item: MemoryItem) -> float:

        days = (
            self._clock() - item.last_accessed_at
        ).total_seconds() / 86400
        days = max(days, 0.0)

        retention = 0.5 ** (days / self.half_life_days)
        boost = 1 + 0.1 * min(item.access_count, 10)

        return min(1.0, item.importance * boost) * retention

    def forget_weak(self, threshold: float = 0.05) -> list[MemoryItem]:
        """Remove memories that have faded below the threshold."""

        weak = [
            item for item in self._items.values()
            if self.strength(item) < threshold
        ]

        for item in weak:
            del self._items[item.id]

        if weak:
            self._save()

        return weak

    # --------------------------------------
    # Write
    # --------------------------------------

    def add(self, item: MemoryItem) -> None:
        self.add_many([item])

    def add_many(self, items: list[MemoryItem]) -> None:
        """All-or-nothing: one bad item means nothing is stored."""

        for item in items:
            self._check_belongs(item)

        for item in items:
            self._items[item.id] = item

        if items:
            self._save()

    def remove(self, item_id: str) -> MemoryItem | None:

        item = self._items.pop(item_id, None)

        if item is not None:
            self._save()

        return item

    # --------------------------------------
    # Read
    # --------------------------------------

    def get(self, item_id: str) -> MemoryItem | None:

        item = self._items.get(item_id)

        if item is not None:
            item.touch(self._clock())
            self._save()

        return item

    def all(self) -> list[MemoryItem]:
        """All items, strongest first. Does not count as access."""

        return sorted(
            self._items.values(),
            key=self.strength,
            reverse=True
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        kind: MemoryKind | None = None,
        touch: bool = True
    ) -> list[MemoryItem]:
        """
        Keyword search, best match first. Recalled items are
        strengthened (touch=True), like real memory.
        """

        words = tokenize(query)
        meaningful = words - STOPWORDS

        query_words = meaningful or words

        if not query_words:
            return []

        scored = []

        for item in self._items.values():

            if kind is not None and item.kind != kind:
                continue

            overlap = query_words & tokenize(item.content)

            if not overlap:
                continue

            relevance = len(overlap) / len(query_words)

            score = (
                RELEVANCE_WEIGHT * relevance
                + STRENGTH_WEIGHT * self.strength(item)
            )

            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        results = [item for _, item in scored[:limit]]

        if touch and results:
            now = self._clock()
            for item in results:
                item.touch(now)
            self._save()

        return results

    # --------------------------------------
    # Dunder helpers
    # --------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items
