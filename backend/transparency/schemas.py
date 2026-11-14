from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from backend.transparency.events import EventStatus, EventType


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TransparencyEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    interaction_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)

    event_type: EventType
    sequence: int = Field(ge=1)

    timestamp: datetime = Field(default_factory=utc_now)
    source: str = Field(min_length=1)
    status: EventStatus = EventStatus.RECORDED

    details: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = 1