
"""
CognitiveAI - Memory Manager

Connects foreground and background memory for one user.

PERSONAL:
    Private memories belonging to this owner.

PUBLIC:
    Shared memories available to public tasks.

Recall behavior:
    1. Search foreground memories by keyword relevance.
    2. Search background memories by keyword relevance.
    3. For broad personal-profile questions, fall back to the
       strongest personal memories.
    4. Preserve strict privacy boundaries.

Personal memories enter public memory only through share().
"""

from pathlib import Path
from typing import Callable
from datetime import datetime

from pydantic import BaseModel

from backend.memory.background import (
    BackgroundMemory,
    STOPWORDS,
    tokenize
)
from backend.memory.models import (
    MemoryItem,
    MemoryKind,
    MemoryScope,
    utc_now
)
from backend.memory.working import ForegroundMemory


class ConsolidationReport(BaseModel):
    promoted: int
    forgotten: int


class MemoryManager:

    PROFILE_QUERY_TERMS = {
        "about", "know", "remember", "profile",
        "myself", "me", "personal", "who"
    }

    def __init__(
        self,
        owner: str,
        base_dir: Path | str | None = None,
        foreground_capacity: int = 7,
        half_life_days: float = 30.0,
        clock: Callable[[], datetime] = utc_now
    ):

        self.owner = owner

        self.personal_fg = ForegroundMemory(
            MemoryScope.PERSONAL,
            owner=owner,
            capacity=foreground_capacity,
            clock=clock
        )

        self.personal_bg = BackgroundMemory(
            MemoryScope.PERSONAL,
            owner=owner,
            base_dir=base_dir,
            half_life_days=half_life_days,
            clock=clock
        )

        self.public_fg = ForegroundMemory(
            MemoryScope.PUBLIC,
            capacity=foreground_capacity,
            clock=clock
        )

        self.public_bg = BackgroundMemory(
            MemoryScope.PUBLIC,
            base_dir=base_dir,
            half_life_days=half_life_days,
            clock=clock
        )

    # --------------------------------------
    # Helpers
    # --------------------------------------

    def _stores(self, scope: MemoryScope):

        if scope == MemoryScope.PERSONAL:
            return self.personal_fg, self.personal_bg

        return self.public_fg, self.public_bg

    def _add_to_foreground(
        self,
        scope,
        item: MemoryItem
    ) -> None:

        fg, bg = self._stores(scope)
        evicted = fg.add(item)

        if evicted:
            bg.add_many(evicted)

    @staticmethod
    def _foreground_matches(
        fg,
        query,
        kind
    ) -> list[MemoryItem]:

        words = tokenize(query)
        query_words = (words - STOPWORDS) or words

        matches = []

        for item in fg.items():

            if kind is not None and item.kind != kind:
                continue

            overlap = len(
                query_words & tokenize(item.content)
            )

            if overlap:
                matches.append((overlap, item))

        matches.sort(
            key=lambda pair: pair[0],
            reverse=True
        )

        return [item for _, item in matches]

    @staticmethod
    def _is_profile_query(query: str) -> bool:
        """
        Detect broad questions asking what the system knows
        or remembers about the current user.
        """

        words = tokenize(query)

        return bool(
            words & MemoryManager.PROFILE_QUERY_TERMS
        )

    @staticmethod
    def _unique(
        items: list[MemoryItem]
    ) -> list[MemoryItem]:

        result = []
        seen = set()

        for item in items:
            if item.id not in seen:
                seen.add(item.id)
                result.append(item)

        return result

    def _strongest_personal_memories(
        self,
        limit: int,
        kind: MemoryKind | None = None
    ) -> list[MemoryItem]:
        """
        Return the strongest personal memories regardless of
        keyword overlap. Used only for private profile queries.
        """

        foreground = [
            item
            for item in self.personal_fg.items()
            if kind is None or item.kind == kind
        ]

        background = [
            item
            for item in self.personal_bg.all()
            if kind is None or item.kind == kind
        ]

        combined = self._unique(
            foreground + background
        )

        # Foreground order represents salience; background order
        # represents calculated memory strength.
        foreground_ids = {
            item.id: index
            for index, item in enumerate(foreground)
        }

        combined.sort(
            key=lambda item: (
                item.id in foreground_ids,
                -foreground_ids.get(item.id, 0)
                if item.id in foreground_ids
                else 0
            ),
            reverse=True
        )

        return combined[:limit]

    # --------------------------------------
    # Remember
    # --------------------------------------

    def remember(
        self,
        content: str,
        scope: MemoryScope = MemoryScope.PERSONAL,
        kind: MemoryKind = MemoryKind.EPISODIC,
        importance: float = 0.5
    ) -> MemoryItem:

        item = MemoryItem(
            content=content,
            scope=scope,
            owner=(
                self.owner
                if scope == MemoryScope.PERSONAL
                else None
            ),
            kind=kind,
            importance=importance
        )

        self._add_to_foreground(scope, item)

        return item

    # --------------------------------------
    # Recall
    # --------------------------------------

    def recall(
        self,
        query: str,
        privacy: str = "private",
        limit: int = 5,
        kind: MemoryKind | None = None
    ) -> list[MemoryItem]:

        if privacy not in {"private", "public"}:
            raise ValueError(
                "Privacy must be private or public"
            )

        if limit <= 0:
            return []

        if privacy == "public":
            scopes = [MemoryScope.PUBLIC]
        else:
            scopes = [
                MemoryScope.PERSONAL,
                MemoryScope.PUBLIC
            ]

        results: list[MemoryItem] = []
        seen: set[str] = set()

        # 1. Normal keyword-based retrieval.
        for scope in scopes:

            fg, bg = self._stores(scope)

            for item in self._foreground_matches(
                fg, query, kind
            ):

                if len(results) >= limit:
                    break

                if item.id not in seen:
                    seen.add(item.id)
                    fg.get(item.id)
                    results.append(item)

            if len(results) < limit:

                for item in bg.search(
                    query,
                    limit=limit,
                    kind=kind
                ):

                    if len(results) >= limit:
                        break

                    if item.id in seen:
                        continue

                    seen.add(item.id)

                    focused = item.model_copy()

                    self._add_to_foreground(
                        scope,
                        focused
                    )

                    results.append(focused)

        # 2. Profile fallback.
        # Never run this for public requests.
        if (
            privacy == "private"
            and len(results) < limit
            and self._is_profile_query(query)
        ):

            personal = self._strongest_personal_memories(
                limit=limit - len(results),
                kind=kind
            )

            for item in personal:

                if item.id in seen:
                    continue

                seen.add(item.id)

                # Count this retrieval as an access.
                if self.personal_fg.get(item.id) is None:
                    self.personal_bg.get(item.id)

                results.append(item)

                if len(results) >= limit:
                    break

        return results[:limit]

    # --------------------------------------
    # Consolidate
    # --------------------------------------

    def consolidate(
        self,
        forget_threshold: float = 0.05
    ) -> ConsolidationReport:

        promoted = 0
        forgotten = 0

        for scope in (
            MemoryScope.PERSONAL,
            MemoryScope.PUBLIC
        ):

            fg, bg = self._stores(scope)

            forgotten += len(
                bg.forget_weak(forget_threshold)
            )

            candidates = [
                item.model_copy()
                for item in fg.promotion_candidates()
            ]

            bg.add_many(candidates)
            promoted += len(candidates)

        return ConsolidationReport(
            promoted=promoted,
            forgotten=forgotten
        )

    # --------------------------------------
    # Share and forget
    # --------------------------------------

    def _find_personal(
        self,
        item_id: str
    ) -> MemoryItem | None:

        for item in self.personal_fg.items():
            if item.id == item_id:
                return item

        for item in self.personal_bg.all():
            if item.id == item_id:
                return item

        return None

    def share(self, item_id: str) -> MemoryItem:
        """Explicitly copy personal memory into public memory."""

        original = self._find_personal(item_id)

        if original is None:
            raise KeyError(
                f"No personal memory with id {item_id}"
            )

        return self.remember(
            content=original.content,
            scope=MemoryScope.PUBLIC,
            kind=original.kind,
            importance=original.importance
        )

    def forget(self, item_id: str) -> bool:
        """Delete personal memory from foreground and background."""

        removed_fg = self.personal_fg.remove(item_id)
        removed_bg = self.personal_bg.remove(item_id)

        return (
            removed_fg is not None
            or removed_bg is not None
        )
