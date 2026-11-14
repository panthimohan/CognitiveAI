from threading import RLock

from backend.transparency.schemas import TransparencyEvent


class InMemoryEventStore:
    """
    Development event store.

    Events are retained only while the application is running.
    A PostgreSQL implementation can replace this later.
    """

    def __init__(self):
        self._events: list[TransparencyEvent] = []
        self._lock = RLock()

    def append(self, event: TransparencyEvent) -> None:
        with self._lock:
            self._events.append(event)

    def get_interaction(
        self,
        user_id: str,
        interaction_id: str
    ) -> list[TransparencyEvent]:

        with self._lock:
            return [
                event
                for event in self._events
                if (
                    event.user_id == user_id
                    and event.interaction_id == interaction_id
                )
            ]

    def list_user(
        self,
        user_id: str,
        limit: int = 100
    ) -> list[TransparencyEvent]:

        if limit < 1:
            raise ValueError("Limit must be at least 1.")

        with self._lock:
            matching = [
                event
                for event in self._events
                if event.user_id == user_id
            ]

            return matching[-limit:]