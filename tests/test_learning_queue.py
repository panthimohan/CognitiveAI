
from datetime import datetime, timedelta, timezone

import pytest

from backend.conversations.archive import ConversationArchive
from backend.learning.queue import LearningQueue


def test_enqueue_is_persistent_across_restart(tmp_path):
    database = tmp_path / "learning_queue.sqlite3"

    first_queue = LearningQueue(database)
    job = first_queue.enqueue(
        owner="test-user",
        conversation_id="conversation-1",
    )

    second_queue = LearningQueue(database)
    restored = second_queue.get_job(job.id)

    assert restored is not None
    assert restored.id == job.id
    assert restored.status == "pending"
    assert restored.attempts == 0


def test_duplicate_enqueue_does_not_create_duplicate_jobs(tmp_path):
    queue = LearningQueue(tmp_path / "learning_queue.sqlite3")

    first = queue.enqueue(
        owner="test-user",
        conversation_id="conversation-1",
    )
    second = queue.enqueue(
        owner="test-user",
        conversation_id="conversation-1",
    )

    assert first.id == second.id
    assert len(queue.list_jobs(owner="test-user")) == 1


def test_public_conversations_cannot_enter_personal_learning(tmp_path):
    queue = LearningQueue(tmp_path / "learning_queue.sqlite3")

    with pytest.raises(ValueError):
        queue.enqueue(
            owner="test-user",
            conversation_id="public-conversation",
            privacy="public",
        )

    assert queue.list_jobs() == []


def test_queue_discovers_existing_private_archived_conversations(tmp_path):
    archive = ConversationArchive(tmp_path / "archive.sqlite3")
    queue = LearningQueue(tmp_path / "learning_queue.sqlite3")

    archive.save_message(
        owner="test-user",
        conversation_id="private-1",
        interaction_id="interaction-1",
        role="user",
        content="I am learning Python.",
        privacy="private",
    )
    archive.save_message(
        owner="test-user",
        conversation_id="public-1",
        interaction_id="interaction-2",
        role="user",
        content="Public conversation.",
        privacy="public",
    )
    archive.save_message(
        owner="test-user",
        conversation_id="private-2",
        interaction_id="interaction-3",
        role="user",
        content="I want to become an AI engineer.",
        privacy="private",
    )

    jobs = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )

    assert {job.conversation_id for job in jobs} == {
        "private-1",
        "private-2",
    }
    assert all(job.privacy == "private" for job in jobs)

    # Scanning again must not duplicate jobs.
    queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )

    assert len(queue.list_jobs(owner="test-user")) == 2


def test_claim_checkpoint_and_complete_job(tmp_path):
    queue = LearningQueue(tmp_path / "learning_queue.sqlite3")

    created = queue.enqueue(
        owner="test-user",
        conversation_id="conversation-1",
    )

    claimed = queue.claim_next()

    assert claimed is not None
    assert claimed.id == created.id
    assert claimed.status == "running"
    assert claimed.attempts == 1

    updated = queue.update_checkpoint(
        job_id=claimed.id,
        checkpoint_offset=25,
    )

    assert updated.checkpoint_offset == 25

    with pytest.raises(ValueError):
        queue.update_checkpoint(
            job_id=claimed.id,
            checkpoint_offset=10,
        )

    completed = queue.complete(job_id=claimed.id)

    assert completed.status == "completed"
    assert completed.lease_until is None
    assert queue.claim_next() is None


def test_failed_job_can_be_retried(tmp_path):
    queue = LearningQueue(tmp_path / "learning_queue.sqlite3")

    created = queue.enqueue(
        owner="test-user",
        conversation_id="conversation-1",
    )

    claimed = queue.claim_next()
    assert claimed is not None

    failed = queue.fail(
        job_id=claimed.id,
        error="Temporary model provider error.",
        retry_delay_seconds=0,
        max_attempts=3,
    )

    assert failed.status == "failed"
    assert failed.last_error == "Temporary model provider error."

    retried = queue.claim_next()

    assert retried is not None
    assert retried.id == created.id
    assert retried.status == "running"
    assert retried.attempts == 2


def test_job_becomes_dead_after_maximum_attempts(tmp_path):
    queue = LearningQueue(tmp_path / "learning_queue.sqlite3")

    queue.enqueue(
        owner="test-user",
        conversation_id="conversation-1",
    )

    claimed = queue.claim_next(max_attempts=2)
    assert claimed is not None

    failed = queue.fail(
        job_id=claimed.id,
        error="First failure.",
        retry_delay_seconds=0,
        max_attempts=2,
    )
    assert failed.status == "failed"

    claimed_again = queue.claim_next(max_attempts=2)
    assert claimed_again is not None
    assert claimed_again.attempts == 2

    dead = queue.fail(
        job_id=claimed_again.id,
        error="Second failure.",
        retry_delay_seconds=0,
        max_attempts=2,
    )

    assert dead.status == "dead"
    assert queue.claim_next(max_attempts=2) is None


def test_expired_running_job_can_be_reclaimed_after_restart(tmp_path):
    database = tmp_path / "learning_queue.sqlite3"
    queue = LearningQueue(database)

    original = queue.enqueue(
        owner="test-user",
        conversation_id="conversation-1",
    )

    claimed = queue.claim_next(lease_seconds=300)
    assert claimed is not None
    assert claimed.id == original.id

    expired_time = (
        datetime.now(timezone.utc) - timedelta(minutes=5)
    ).isoformat()

    # Simulate a worker crash followed by an expired lease.
    with queue._connect() as connection:
        connection.execute(
            """
            UPDATE learning_jobs
            SET lease_until = ?
            WHERE id = ?
            """,
            (expired_time, claimed.id),
        )

    restarted_queue = LearningQueue(database)
    recovered = restarted_queue.claim_next()

    assert recovered is not None
    assert recovered.id == original.id
    assert recovered.status == "running"
    assert recovered.attempts == 2


def test_new_messages_reopen_completed_job_from_previous_checkpoint(tmp_path):
    archive = ConversationArchive(tmp_path / "archive.sqlite3")
    queue = LearningQueue(tmp_path / "learning_queue.sqlite3")

    for number in range(2):
        archive.save_message(
            owner="test-user",
            conversation_id="conversation-1",
            interaction_id=f"interaction-{number}",
            role="user",
            content=f"Original message {number}.",
        )

    original_jobs = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )
    original_job = original_jobs[0]

    claimed = queue.claim_next()
    assert claimed is not None

    queue.update_checkpoint(
        job_id=claimed.id,
        checkpoint_offset=2,
    )
    queue.complete(job_id=claimed.id)

    # A new interaction adds another archived message.
    archive.save_message(
        owner="test-user",
        conversation_id="conversation-1",
        interaction_id="interaction-3",
        role="user",
        content="I have started building an AI learning system.",
    )

    refreshed_jobs = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )

    refreshed = refreshed_jobs[0]

    assert refreshed.id == original_job.id
    assert refreshed.status == "pending"
    assert refreshed.checkpoint_offset == 2
    assert refreshed.attempts == 0
    assert refreshed.last_error is None

    # The worker can claim the same job and resume from message 2.
    resumed = queue.claim_next()

    assert resumed is not None
    assert resumed.id == original_job.id
    assert resumed.status == "running"
    assert resumed.checkpoint_offset == 2


def test_completed_job_stays_completed_without_new_messages(tmp_path):
    archive = ConversationArchive(tmp_path / "archive.sqlite3")
    queue = LearningQueue(tmp_path / "learning_queue.sqlite3")

    archive.save_message(
        owner="test-user",
        conversation_id="conversation-1",
        interaction_id="interaction-1",
        role="user",
        content="Original message.",
    )

    job = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )[0]

    claimed = queue.claim_next()
    assert claimed is not None

    queue.update_checkpoint(
        job_id=claimed.id,
        checkpoint_offset=1,
    )
    queue.complete(job_id=claimed.id)

    refreshed = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )[0]

    assert refreshed.id == job.id
    assert refreshed.status == "completed"
    assert refreshed.checkpoint_offset == 1


def test_archive_discovery_does_not_reset_dead_jobs(tmp_path):
    archive = ConversationArchive(tmp_path / "archive.sqlite3")
    queue = LearningQueue(tmp_path / "learning_queue.sqlite3")

    archive.save_message(
        owner="test-user",
        conversation_id="conversation-1",
        interaction_id="interaction-1",
        role="user",
        content="Original message.",
    )

    job = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )[0]

    claimed = queue.claim_next(max_attempts=1)
    assert claimed is not None

    dead = queue.fail(
        job_id=claimed.id,
        error="Permanent failure.",
        retry_delay_seconds=0,
        max_attempts=1,
    )

    assert dead.status == "dead"

    # Even new messages must not silently erase the dead-job decision.
    archive.save_message(
        owner="test-user",
        conversation_id="conversation-1",
        interaction_id="interaction-2",
        role="user",
        content="A new message.",
    )

    refreshed = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )[0]

    assert refreshed.id == job.id
    assert refreshed.status == "dead"
