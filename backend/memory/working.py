"""
CognitiveAI - Foreground (Working) Memory

A small, fast, capacity-limited buffer holding what the system
is actively using right now. One instance serves one scope:

    - PERSONAL: bound to a single owner (a user)
    - PUBLIC:   shared, no owner

When the buffer is full, the least salient item is evicted and
RETURNED to the caller, so nothing is lost: the caller (the
memory manager) can hand evicted items to background memory.

Salience combines importance with recency:

    salience = 0.7 * importance + 0.3 * recency
    recency  = 0.5 ** (seconds_since_last_access / half_life)
"""

from datetime import datetime
from typing import Callable

from backend.memory.models import MemoryItem, MemoryScope, utc_now


IMPORTANCE_WEIGHT = 0.7
RECENCY_WEIGHT = 0.3


class ForegroundMemory:

    def __init__(
        self,
        scope: MemoryScope,
        owner: str | None = None,
        capacity: int = 7,
        half_life_seconds: float = 300.0,
        clock: Callable[[], datetime] = utc_now
    ):

        if capacity < 1:
            raise ValueError("Capacity must be at least 1.")

        if half_life_seconds <= 0:
            raise ValueError("Half-life must be positive.")

        if scope == MemoryScope.PERSONAL and not owner:
            raise ValueError("Personal memory requires an owner.")

        if scope == MemoryScope.PUBLIC and owner is not None:
            raise ValueError("Public memory must not have an owner.")

        self.scope = scope
        self.owner = owner
        self.capacity = capacity
        self.half_life_seconds = half_life_seconds
        self._clock = clock
        self._items: dict[str, MemoryItem] = {}

    # --------------------------------------
    # Salience
    # --------------------------------------

    def salience(self, item: MemoryItem) -> float:

        age = (self._clock() - item.last_accessed_at).total_seconds()
        age = max(age, 0.0)

        recency = 0.5 ** (age / self.half_life_seconds)

        return (
            IMPORTANCE_WEIGHT * item.importance
            + RECENCY_WEIGHT * recency
        )

    # --------------------------------------
    # Add an item (returns evicted items)
    # --------------------------------------

    def add(self, item: MemoryItem) -> list[MemoryItem]:

        if item.scope != self.scope or item.owner != self.owner:
            raise ValueError(
                "Item does not belong to this memory "
                "(scope or owner mismatch)."
            )

        if item.id in self._items:
            # Already present: treat as a fresh access.
            self._items[item.id].touch(self._clock())
            return []

        item.last_accessed_at = self._clock()
        self._items[item.id] = item

        evicted = []

        while len(self._items) > self.capacity:

            weakest = min(
                self._items.values(),
                key=self.salience
            )

            # Never evict the item that was just added.
            if weakest.id == item.id:
                others = [
                    i for i in self._items.values()
                    if i.id != item.id
                ]
                weakest = min(others, key=self.salience)

            evicted.append(self._items.pop(weakest.id))

        return evicted

    # --------------------------------------
    # Read
    # --------------------------------------

    def get(self, item_id: str) -> MemoryItem | None:

        item = self._items.get(item_id)

        if item is not None:
            item.touch(self._clock())

        return item

    def items(self) -> list[MemoryItem]:
        """All items, most salient first. Does not count as access."""

        return sorted(
            self._items.values(),
            key=self.salience,
            reverse=True
        )

    # --------------------------------------
    # Remove
    # --------------------------------------

    def remove(self, item_id: str) -> MemoryItem | None:

        return self._items.pop(item_id, None)

    def clear(self) -> list[MemoryItem]:

        removed = list(self._items.values())
        self._items.clear()

        return removed

    # --------------------------------------
    # Promotion to background memory
    # --------------------------------------

    def promotion_candidates(
        self,
        min_importance: float = 0.6,
        min_access_count: int = 2
    ) -> list[MemoryItem]:
        """
        Items worth keeping long-term: important, or used repeatedly.
        Does not remove them; the memory manager decides what to do.
        """

        return [
            item for item in self.items()
            if item.importance >= min_importance
            or item.access_count >= min_access_count
        ]

    # --------------------------------------
    # Dunder helpers
    # --------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items
