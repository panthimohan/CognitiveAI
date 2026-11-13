"""
CognitiveAI - Memory Manager

Connects foreground and background memory for one user, across
two scopes:

    PERSONAL  foreground + background, belongs to the user
    PUBLIC    foreground + background, shared knowledge

Flow:
    remember()    -> new item enters FOREGROUND
                     (items pushed out by capacity go to BACKGROUND,
                      so nothing is silently lost)
    recall()      -> searches foreground, then BACKGROUND; background
                     hits are pulled into the foreground
    consolidate() -> forgets faded memories, then saves important or
                     frequently used foreground items to BACKGROUND
    share()       -> explicitly copies a personal memory to public
    forget()      -> deletes a personal memory everywhere

Privacy rules (what a task may SEE):
    privacy="public"  -> public memory only. A public task may be
                         sent to an external AI provider, so personal
                         memories must never be included.
    privacy="private" -> personal + public memory. Private tasks run
                         locally, so nothing leaves the machine.

Personal memory only reaches public memory through share().
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
            MemoryScope.PERSONAL, owner=owner,
            capacity=foreground_capacity, clock=clock
        )
        self.personal_bg = BackgroundMemory(
            MemoryScope.PERSONAL, owner=owner, base_dir=base_dir,
            half_life_days=half_life_days, clock=clock
        )
        self.public_fg = ForegroundMemory(
            MemoryScope.PUBLIC,
            capacity=foreground_capacity, clock=clock
        )
        self.public_bg = BackgroundMemory(
            MemoryScope.PUBLIC, base_dir=base_dir,
            half_life_days=half_life_days, clock=clock
        )

    # --------------------------------------
    # Helpers
    # --------------------------------------

    def _stores(self, scope: MemoryScope):

        if scope == MemoryScope.PERSONAL:
            return self.personal_fg, self.personal_bg

        return self.public_fg, self.public_bg

    def _add_to_foreground(self, scope, item: MemoryItem) -> None:
        """Add to foreground; anything evicted moves to background."""

        fg, bg = self._stores(scope)
        evicted = fg.add(item)

        if evicted:
            bg.add_many(evicted)

    @staticmethod
    def _foreground_matches(fg, query, kind) -> list[MemoryItem]:

        words = tokenize(query)
        query_words = (words - STOPWORDS) or words

        matches = []

        for item in fg.items():

            if kind is not None and item.kind != kind:
                continue

            overlap = len(query_words & tokenize(item.content))

            if overlap:
                matches.append((overlap, item))

        # fg.items() is already ordered by salience; sort is stable
        matches.sort(key=lambda pair: pair[0], reverse=True)

        return [item for _, item in matches]

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
            owner=self.owner if scope == MemoryScope.PERSONAL else None,
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
            raise ValueError("Privacy must be private or public")

        if privacy == "public":
            scopes = [MemoryScope.PUBLIC]
        else:
            scopes = [MemoryScope.PERSONAL, MemoryScope.PUBLIC]

        results: list[MemoryItem] = []
        seen: set[str] = set()

        for scope in scopes:

            fg, bg = self._stores(scope)

            # 1. Foreground first (what we're already focused on)
            for item in self._foreground_matches(fg, query, kind):

                if len(results) >= limit:
                    break

                if item.id not in seen:
                    seen.add(item.id)
                    fg.get(item.id)          # counts as access
                    results.append(item)

            # 2. Then background; pull hits into the foreground
            if len(results) < limit:

                for item in bg.search(query, limit=limit, kind=kind):

                    if len(results) >= limit:
                        break

                    if item.id in seen:
                        continue

                    seen.add(item.id)

                    focused = item.model_copy()
                    self._add_to_foreground(scope, focused)
                    results.append(focused)

        return results

    # --------------------------------------
    # Consolidate
    # --------------------------------------

    def consolidate(
        self,
        forget_threshold: float = 0.05
    ) -> ConsolidationReport:

        promoted = 0
        forgotten = 0

        for scope in (MemoryScope.PERSONAL, MemoryScope.PUBLIC):

            fg, bg = self._stores(scope)

            forgotten += len(bg.forget_weak(forget_threshold))

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
    # Share and forget (personal memory only)
    # --------------------------------------

    def _find_personal(self, item_id: str) -> MemoryItem | None:

        for item in self.personal_fg.items():
            if item.id == item_id:
                return item

        for item in self.personal_bg.all():
            if item.id == item_id:
                return item

        return None

    def share(self, item_id: str) -> MemoryItem:
        """
        Explicitly copy a personal memory into public memory.
        The copy is a new item with no owner; the original stays.
        """

        original = self._find_personal(item_id)

        if original is None:
            raise KeyError(f"No personal memory with id {item_id}")

        return self.remember(
            content=original.content,
            scope=MemoryScope.PUBLIC,
            kind=original.kind,
            importance=original.importance
        )

    def forget(self, item_id: str) -> bool:
        """Delete a personal memory from foreground and background."""

        removed_fg = self.personal_fg.remove(item_id)
        removed_bg = self.personal_bg.remove(item_id)

        return removed_fg is not None or removed_bg is not None
