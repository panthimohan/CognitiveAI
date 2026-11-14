from threading import Lock
from typing import Any

from backend.transparency.events import EventStatus, EventType
from backend.transparency.event_store import InMemoryEventStore
from backend.transparency.schemas import TransparencyEvent


class TransparencyRecorder:

    def __init__(self, store: InMemoryEventStore):
        self.store = store
        self._sequences: dict[tuple[str, str], int] = {}
        self._lock = Lock()

    def record(
        self,
        user_id: str,
        interaction_id: str,
        event_type: EventType,
        source: str,
        status: EventStatus = EventStatus.RECORDED,
        details: dict[str, Any] | None = None
    ) -> TransparencyEvent:

        key = (user_id, interaction_id)

        with self._lock:
            sequence = self._sequences.get(key, 0) + 1
            self._sequences[key] = sequence

            event = TransparencyEvent(
                user_id=user_id,
                interaction_id=interaction_id,
                event_type=event_type,
                sequence=sequence,
                source=source,
                status=status,
                details=details or {}
            )

            self.store.append(event)

        return event