
import asyncio

from backend.conversations.archive import ConversationArchive
from backend.learning.background import PrivateLearningWorker
from backend.learning.queue import LearningQueue
from backend.memory.manager import MemoryManager
from backend.memory.processor import MemoryCandidate, MemoryProcessor


class FakeExtractor:
    """Asynchronous deterministic extractor for worker tests."""

    def __init__(
        self,
        max_input_characters=12000,
        fail_once=False,
        fail_on_call=None,
    ):
        self.max_input_characters = max_input_characters
        self.fail_once = fail_once
        self.fail_on_call = fail_on_call
        self.calls = []

    async def extract(self, conversation_text):
        self.calls.append(conversation_text)

        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("Simulated extraction failure.")

        if self.fail_on_call == len(self.calls):
            raise RuntimeError("Simulated later-page failure.")

        return [
            MemoryCandidate(
                content="The user wants to become an AI engineer.",
                category="goal",
                importance=0.9,
                confidence=0.95,
            )
        ]


def make_worker(
    tmp_path,
    extractor=None,
    page_size=100,
    max_batch_characters=None,
):
    archive = ConversationArchive(tmp_path / "archive.sqlite3")
    queue = LearningQueue(tmp_path / "learning_queue.sqlite3")

    manager = MemoryManager(
        owner="test-user",
        base_dir=tmp_path / "memory",
    )

    processor = MemoryProcessor(manager)
    extractor = extractor or FakeExtractor()

    worker = PrivateLearningWorker(
        archive=archive,
        queue=queue,
        extractor=extractor,
        processor=processor,
        page_size=page_size,
        retry_delay_seconds=0,
        max_batch_characters=max_batch_characters,
    )

    return archive, queue, manager, extractor, worker


def add_message(
    archive,
    *,
    conversation_id="conversation-1",
    interaction_id="interaction-1",
    role="user",
    content="I want to become an AI engineer.",
    privacy="private",
):
    return archive.save_message(
        owner="test-user",
        conversation_id=conversation_id,
        interaction_id=interaction_id,
        role=role,
        content=content,
        privacy=privacy,
    )


def test_worker_processes_private_conversation(tmp_path):
    archive, queue, manager, extractor, worker = make_worker(tmp_path)

    add_message(
        archive,
        interaction_id="interaction-1",
        role="user",
        content="I want to become an AI engineer.",
    )

    add_message(
        archive,
        interaction_id="interaction-1",
        role="assistant",
        content="Let's build your machine learning skills.",
    )

    jobs = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )

    assert len(jobs) == 1

    result = asyncio.run(worker.process_next())

    assert result is not None
    assert result.status == "completed"
    assert result.messages_processed == 2
    assert result.candidates_extracted == 1
    assert result.memories_saved == 1

    memories = manager.personal_bg.all()

    assert any(
        "AI engineer" in item.content
        for item in memories
    )

    assert queue.get_job(jobs[0].id).checkpoint_offset == 2


def test_worker_processes_large_message_without_truncation(tmp_path):
    extractor = FakeExtractor(max_input_characters=300)

    archive, queue, manager, extractor, worker = make_worker(
        tmp_path,
        extractor=extractor,
    )

    long_content = "x" * 2000

    add_message(
        archive,
        content=long_content,
    )

    jobs = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )

    assert len(jobs) == 1

    result = asyncio.run(worker.process_next())

    assert result is not None
    assert result.status == "completed"
    assert result.messages_processed == 1
    assert len(extractor.calls) > 1

    extracted_x_count = sum(
        call.count("x")
        for call in extractor.calls
    )

    assert extracted_x_count == len(long_content)

    assert all(
        len(call) <= 300
        for call in extractor.calls
    )

    assert queue.get_job(jobs[0].id).checkpoint_offset == 1


def test_worker_resumes_from_checkpoint_after_failure(tmp_path):
    extractor = FakeExtractor(fail_once=True)

    archive, queue, manager, extractor, worker = make_worker(
        tmp_path,
        extractor=extractor,
    )

    add_message(
        archive,
        interaction_id="interaction-1",
        content="First archived message.",
    )

    add_message(
        archive,
        interaction_id="interaction-2",
        content="Second archived message.",
    )

    jobs = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )

    job_id = jobs[0].id

    first_result = asyncio.run(worker.process_next())

    assert first_result is not None
    assert first_result.status == "failed"
    assert "Simulated extraction failure" in first_result.error

    failed_job = queue.get_job(job_id)

    assert failed_job.status == "failed"
    assert failed_job.checkpoint_offset == 0

    second_result = asyncio.run(worker.process_next())

    assert second_result is not None
    assert second_result.status == "completed"
    assert second_result.messages_processed == 2

    completed_job = queue.get_job(job_id)

    assert completed_job.status == "completed"
    assert completed_job.checkpoint_offset == 2


def test_worker_processes_conversations_in_multiple_pages(tmp_path):
    archive, queue, manager, extractor, worker = make_worker(
        tmp_path,
        page_size=2,
    )

    for number in range(5):
        add_message(
            archive,
            interaction_id=f"interaction-{number}",
            content=f"Archived message number {number}.",
        )

    jobs = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )

    assert len(jobs) == 1

    result = asyncio.run(worker.process_next())

    assert result is not None
    assert result.status == "completed"
    assert result.messages_processed == 5

    job = queue.get_job(jobs[0].id)

    assert job.checkpoint_offset == 5
    assert job.status == "completed"


def test_public_conversation_is_not_enqueued_for_private_learning(tmp_path):
    archive, queue, manager, extractor, worker = make_worker(tmp_path)

    add_message(
        archive,
        conversation_id="public-conversation",
        interaction_id="public-interaction",
        content="This is public information.",
        privacy="public",
    )

    jobs = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )

    assert jobs == []
    assert queue.list_jobs(owner="test-user") == []
    assert manager.personal_bg.all() == []


def test_worker_awaits_async_extractor(tmp_path):
    archive, queue, manager, extractor, worker = make_worker(tmp_path)

    add_message(
        archive,
        content="I want to become an AI engineer.",
    )

    queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )

    result = asyncio.run(worker.process_next())

    assert result.status == "completed"
    assert len(extractor.calls) == 1
    assert result.candidates_extracted == 1


def test_worker_completes_empty_enqueued_conversation(tmp_path):
    archive, queue, manager, extractor, worker = make_worker(tmp_path)

    job = queue.enqueue(
        owner="test-user",
        conversation_id="empty-conversation",
    )

    result = asyncio.run(worker.process_next())

    assert result is not None
    assert result.job_id == job.id
    assert result.status == "completed"
    assert result.messages_processed == 0
    assert result.candidates_extracted == 0

    completed = queue.get_job(job.id)

    assert completed.status == "completed"
    assert completed.checkpoint_offset == 0
    assert extractor.calls == []


def test_worker_preserves_checkpoint_when_later_page_fails(tmp_path):
    extractor = FakeExtractor(fail_on_call=2)

    archive, queue, manager, extractor, worker = make_worker(
        tmp_path,
        extractor=extractor,
        page_size=1,
    )

    add_message(
        archive,
        interaction_id="interaction-1",
        content="First message.",
    )

    add_message(
        archive,
        interaction_id="interaction-2",
        content="Second message.",
    )

    jobs = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )

    job_id = jobs[0].id

    first_result = asyncio.run(worker.process_next())

    assert first_result.status == "failed"
    assert "later-page failure" in first_result.error

    failed_job = queue.get_job(job_id)

    assert failed_job.status == "failed"
    assert failed_job.checkpoint_offset == 1
    assert first_result.messages_processed == 1

    retry_result = asyncio.run(worker.process_next())

    assert retry_result.status == "completed"
    assert retry_result.messages_processed == 1

    completed_job = queue.get_job(job_id)

    assert completed_job.status == "completed"
    assert completed_job.checkpoint_offset == 2

    # First page was not extracted again.
    assert len(extractor.calls) == 3


def test_worker_processes_messages_added_during_learning(tmp_path):
    archive, queue, manager, extractor, worker = make_worker(
        tmp_path,
        page_size=1,
    )

    add_message(
        archive,
        interaction_id="initial-interaction",
        content="Initial archived message.",
    )

    class AddingMessageExtractor(FakeExtractor):
        def __init__(self):
            super().__init__()
            self.message_added = False

        async def extract(self, conversation_text):
            result = await super().extract(conversation_text)

            if not self.message_added:
                self.message_added = True

                add_message(
                    archive,
                    interaction_id="new-interaction",
                    content="A message added during learning.",
                )

            return result

    extractor = AddingMessageExtractor()

    # Replace the worker's extractor with the message-injecting test double.
    worker.extractor = extractor

    jobs = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )

    result = asyncio.run(worker.process_next())

    assert result.status == "completed"
    assert result.messages_processed == 2
    assert len(extractor.calls) == 2

    completed_job = queue.get_job(jobs[0].id)

    assert completed_job.status == "completed"
    assert completed_job.checkpoint_offset == 2


def test_worker_keeps_extraction_batches_within_character_limit(tmp_path):
    extractor = FakeExtractor(max_input_characters=300)

    archive, queue, manager, extractor, worker = make_worker(
        tmp_path,
        extractor=extractor,
        page_size=10,
    )

    add_message(
        archive,
        interaction_id="interaction-1",
        content="a" * 100,
    )

    add_message(
        archive,
        interaction_id="interaction-2",
        content="b" * 100,
    )

    jobs = queue.enqueue_archived_conversations(
        archive,
        owner="test-user",
    )

    result = asyncio.run(worker.process_next())

    assert result.status == "completed"
    assert result.messages_processed == 2
    assert len(extractor.calls) >= 2

    assert all(
        len(call) <= 300
        for call in extractor.calls
    )

    assert queue.get_job(jobs[0].id).checkpoint_offset == 2
