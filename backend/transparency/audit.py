from backend.transparency.event_store import InMemoryEventStore
from backend.transparency.explanations import (
    build_interaction_explanation
)


class TransparencyAudit:

    def __init__(self, store: InMemoryEventStore):
        self.store = store

    def get_interaction(
        self,
        user_id: str,
        interaction_id: str
    ) -> dict | None:

        events = self.store.get_interaction(
            user_id=user_id,
            interaction_id=interaction_id
        )

        if not events:
            return None

        return build_interaction_explanation(events)

    def list_events(
        self,
        user_id: str,
        limit: int = 100
    ) -> list[dict]:

        events = self.store.list_user(
            user_id=user_id,
            limit=limit
        )

        return [
            event.model_dump(mode="json")
            for event in events
        ]