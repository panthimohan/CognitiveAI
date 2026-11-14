import pytest
from pydantic import ValidationError

from backend.transparency.audit import TransparencyAudit
from backend.transparency.events import EventStatus, EventType
from backend.transparency.event_store import InMemoryEventStore
from backend.transparency.recorder import TransparencyRecorder
from backend.transparency.schemas import TransparencyEvent


@pytest.fixture
def transparency():
    store = InMemoryEventStore()
    recorder = TransparencyRecorder(store)
    audit = TransparencyAudit(store)

    return store, recorder, audit


def test_events_receive_sequential_numbers(transparency):
    _, recorder, _ = transparency

    first = recorder.record(
        user_id="user-1",
        interaction_id="interaction-1",
        event_type=EventType.INTERACTION_STARTED,
        source="api"
    )

    second = recorder.record(
        user_id="user-1",
        interaction_id="interaction-1",
        event_type=EventType.TASK_CLASSIFIED,
        source="supervisor"
    )

    assert first.sequence == 1
    assert second.sequence == 2


def test_different_interactions_have_separate_sequences(transparency):
    _, recorder, _ = transparency

    first = recorder.record(
        user_id="user-1",
        interaction_id="interaction-1",
        event_type=EventType.INTERACTION_STARTED,
        source="api"
    )

    second = recorder.record(
        user_id="user-1",
        interaction_id="interaction-2",
        event_type=EventType.INTERACTION_STARTED,
        source="api"
    )

    assert first.sequence == 1
    assert second.sequence == 1


def test_users_cannot_read_each_others_events(transparency):
    _, recorder, audit = transparency

    recorder.record(
        user_id="user-1",
        interaction_id="interaction-1",
        event_type=EventType.INTERACTION_STARTED,
        source="api"
    )

    assert audit.get_interaction(
        user_id="user-2",
        interaction_id="interaction-1"
    ) is None


def test_audit_returns_recorded_timeline(transparency):
    _, recorder, audit = transparency

    recorder.record(
        user_id="user-1",
        interaction_id="interaction-1",
        event_type=EventType.INTERACTION_STARTED,
        source="api"
    )

    recorder.record(
        user_id="user-1",
        interaction_id="interaction-1",
        event_type=EventType.INTERACTION_COMPLETED,
        source="api",
        status=EventStatus.COMPLETED
    )

    report = audit.get_interaction(
        user_id="user-1",
        interaction_id="interaction-1"
    )

    assert report is not None
    assert report["status"] == "completed"
    assert report["event_count"] == 2
    assert len(report["timeline"]) == 2


def test_model_information_comes_from_recorded_event(transparency):
    _, recorder, audit = transparency

    recorder.record(
        user_id="user-1",
        interaction_id="interaction-1",
        event_type=EventType.MODEL_SELECTED,
        source="model_router",
        details={
            "provider": "ollama",
            "model": "deepseek-r1:7b",
            "destination": "local"
        }
    )

    report = audit.get_interaction(
        user_id="user-1",
        interaction_id="interaction-1"
    )

    assert report["model"]["model"] == "deepseek-r1:7b"
    assert report["model"]["destination"] == "local"


def test_event_rejects_invalid_sequence():
    with pytest.raises(ValidationError):
        TransparencyEvent(
            interaction_id="interaction-1",
            user_id="user-1",
            event_type=EventType.INTERACTION_STARTED,
            sequence=0,
            source="api"
        )


def test_event_store_rejects_invalid_limit(transparency):
    store, _, _ = transparency

    with pytest.raises(ValueError):
        store.list_user("user-1", limit=0)