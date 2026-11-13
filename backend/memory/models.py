"""
CognitiveAI - Memory Data Models

Shared data structures for foreground and background memory.

Privacy rule enforced at the model level:
    - PERSONAL items must belong to an owner.
    - PUBLIC items must NOT have an owner, so personal data
      cannot be stored in public memory by mistake.
"""

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ==========================================
# 1. ENUMS
# ==========================================

class MemoryScope(str, Enum):
    PERSONAL = "personal"
    PUBLIC = "public"


class MemoryKind(str, Enum):
    EPISODIC = "episodic"        # events and experiences
    SEMANTIC = "semantic"        # facts and knowledge
    PROCEDURAL = "procedural"    # how to do things


# ==========================================
# 2. MEMORY ITEM
# ==========================================

class MemoryItem(BaseModel):

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(min_length=1)

    scope: MemoryScope
    owner: str | None = None
    kind: MemoryKind = MemoryKind.EPISODIC

    importance: float = Field(default=0.5, ge=0, le=1)

    created_at: datetime = Field(default_factory=utc_now)
    last_accessed_at: datetime = Field(default_factory=utc_now)
    access_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_scope_owner(self):

        if self.scope == MemoryScope.PERSONAL and not self.owner:
            raise ValueError(
                "Personal memory requires an owner."
            )

        if self.scope == MemoryScope.PUBLIC and self.owner is not None:
            raise ValueError(
                "Public memory must not have an owner."
            )

        return self

    def touch(self, now: datetime | None = None) -> None:
        """Record that this item was accessed."""

        self.access_count += 1
        self.last_accessed_at = now or utc_now()
