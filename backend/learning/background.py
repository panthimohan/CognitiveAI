
"""
CognitiveAI - Private Background Learning Worker

Processes archived private conversations using the asynchronous
AI provider interface.

Responsibilities:
- Claim durable learning jobs.
- Read conversation history in chronological pages.
- Split large messages without truncating their contents.
- Extract useful personal memories asynchronously.
- Persist memories through MemoryProcessor.
- Advance checkpoints only after successful processing.
- Retry failed jobs through LearningQueue.

Private conversations are never sent to public learning.
"""

from dataclasses import dataclass
from typing import Any

from backend.conversations.archive import ConversationArchive
from backend.learning.queue import LearningJob, LearningQueue
from backend.memory.extractor import MemoryExtractor
from backend.memory.processor import MemoryProcessor


@dataclass
class LearningRunResult:
    job_id: str
    status: str
    messages_processed: int = 0
    candidates_extracted: int = 0
    memories_saved: int = 0
    memories_duplicate: int = 0
    memories_skipped: int = 0
    memories_rejected: int = 0
    error: str | None = None


class PrivateLearningWorker:
    """Processes private archived conversations into personal memory."""

    def __init__(
        self,
        *,
        archive: ConversationArchive,
        queue: LearningQueue,
        extractor: MemoryExtractor,
        processor: MemoryProcessor,
        page_size: int = 100,
        lease_seconds: int = 300,
        max_attempts: int = 5,
        retry_delay_seconds: int = 60,
        max_batch_characters: int | None = None,
    ):
        if not 1 <= page_size <= 10000:
            raise ValueError("Page size must be between 1 and 10000.")

        if lease_seconds < 1:
            raise ValueError("Lease duration must be positive.")

        if max_attempts < 1:
            raise ValueError("Maximum attempts must be positive.")

        if retry_delay_seconds < 0:
            raise ValueError("Retry delay cannot be negative.")

        extractor_limit = getattr(
            extractor,
            "max_input_characters",
            12000,
        )

        self.max_batch_characters = (
            max_batch_characters or extractor_limit
        )

        if self.max_batch_characters < 1:
            raise ValueError("Batch character limit must be positive.")

        self.archive = archive
        self.queue = queue
        self.extractor = extractor
        self.processor = processor
        self.page_size = page_size
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds

    async def process_next(self) -> LearningRunResult | None:
        """Claim and asynchronously process one available job."""

        job = self.queue.claim_next(
            lease_seconds=self.lease_seconds,
            max_attempts=self.max_attempts,
        )

        if job is None:
            return None

        return await self.process_job(job)

    async def process_job(
        self,
        job: LearningJob,
    ) -> LearningRunResult:
        """Process a previously claimed private learning job."""

        result = LearningRunResult(
            job_id=job.id,
            status="running",
        )

        if job.privacy != "private":
            error = (
                "Private learning worker cannot process public jobs."
            )

            failed = self.queue.fail(
                job_id=job.id,
                error=error,
                retry_delay_seconds=self.retry_delay_seconds,
                max_attempts=self.max_attempts,
            )

            result.status = failed.status
            result.error = error
            return result

        try:
            while True:
                messages = self.archive.get_conversation_page(
                    owner=job.owner,
                    conversation_id=job.conversation_id,
                    offset=job.checkpoint_offset,
                    limit=self.page_size,
                    privacy="private",
                )

                if not messages:
                    break

                for message in messages:
                    if (
                        message.owner != job.owner
                        or message.conversation_id != job.conversation_id
                        or message.privacy != "private"
                    ):
                        raise ValueError(
                            "Archive returned a message outside the "
                            "claimed private conversation."
                        )

                await self._process_page(
                    messages,
                    job,
                    result,
                )

                # Every message in this page has been checkpointed.
                job.checkpoint_offset += len(messages)

            self.queue.complete(job_id=job.id)

            result.status = "completed"
            return result

        except Exception as error:
            failed = self.queue.fail(
                job_id=job.id,
                error=str(error),
                retry_delay_seconds=self.retry_delay_seconds,
                max_attempts=self.max_attempts,
            )

            result.status = failed.status
            result.error = str(error)[:4000]
            return result

    async def _extract_and_store(
        self,
        text: str,
        result: LearningRunResult,
    ) -> None:
        """Extract candidates asynchronously and store them."""

        candidates = await self.extractor.extract(text)

        processed = self.processor.process_many(candidates)

        result.candidates_extracted += len(candidates)

        for item in processed:
            if item.status == "saved":
                result.memories_saved += 1

            elif item.status == "duplicate":
                result.memories_duplicate += 1

            elif item.status == "skipped":
                result.memories_skipped += 1

            elif item.status == "rejected":
                result.memories_rejected += 1

    async def _process_page(
        self,
        messages: list[Any],
        job: LearningJob,
        result: LearningRunResult,
    ) -> None:
        """Process a page and checkpoint each successful batch."""

        pending: list[str] = []
        pending_characters = 0
        pending_end_offset = job.checkpoint_offset
        pending_message_count = 0

        async def flush_pending() -> None:
            nonlocal pending
            nonlocal pending_characters
            nonlocal pending_end_offset
            nonlocal pending_message_count

            if not pending:
                return

            await self._extract_and_store(
                "\n\n".join(pending),
                result,
            )

            self.queue.update_checkpoint(
                job_id=job.id,
                checkpoint_offset=pending_end_offset,
            )

            result.messages_processed += pending_message_count

            pending = []
            pending_characters = 0
            pending_message_count = 0

        for index, message in enumerate(messages):
            message_offset = job.checkpoint_offset + index + 1

            prefix = (
                f"Role: {message.role.upper()}\n"
                f"Timestamp: {message.created_at}\n"
                f"Interaction: {message.interaction_id}\n"
                "Content:\n"
            )

            available = (
                self.max_batch_characters - len(prefix) - 30
            )

            if available < 1:
                raise ValueError(
                    "Batch character limit is too small "
                    "for message metadata."
                )

            content = message.content

            if len(content) > available:
                await flush_pending()

                parts = [
                    content[start:start + available]
                    for start in range(0, len(content), available)
                ]

                # Do not checkpoint this message until every part succeeds.
                for number, part in enumerate(parts, start=1):
                    part_text = (
                        prefix
                        + f"Part: {number}/{len(parts)}\n"
                        + part
                    )

                    await self._extract_and_store(
                        part_text,
                        result,
                    )

                self.queue.update_checkpoint(
                    job_id=job.id,
                    checkpoint_offset=message_offset,
                )

                result.messages_processed += 1
                continue

            rendered = prefix + content

            if (
                pending
                and pending_characters + len(rendered) + 2
                > self.max_batch_characters
            ):
                await flush_pending()

            pending.append(rendered)
            pending_characters += len(rendered) + 2
            pending_end_offset = message_offset
            pending_message_count += 1

        await flush_pending()
