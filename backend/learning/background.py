
"""
CognitiveAI - Private Background Learning Worker

Processes archived private conversations through two independent
learning paths:

1. Memory extraction and persistence.
2. Structured claim extraction and active learning.

Responsibilities:
- Claim durable learning jobs.
- Read conversation history in chronological pages.
- Split large messages without truncating their contents.
- Extract and persist personal memories.
- Extract and record structured private claims.
- Run the persistent insight lifecycle after processing.
- Advance checkpoints only after successful batch processing.
- Retry failed jobs safely.

Private conversations are never sent to public learning.
"""

from dataclasses import dataclass
from typing import Any

from backend.conversations.archive import ConversationArchive
from backend.learning.active import ActiveLearningEngine
from backend.learning.claim_extractor import ClaimExtractor
from backend.learning.contracts import LearningDomain
from backend.learning.lifecycle import LearningLifecycle
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

    claims_extracted: int = 0
    claims_recorded: int = 0
    claims_duplicate: int = 0
    insights_created: int = 0

    error: str | None = None


class PrivateLearningWorker:
    """
    Processes private archived conversations into memory and
    structured learning evidence.
    """

    def __init__(
        self,
        *,
        archive: ConversationArchive,
        queue: LearningQueue,
        extractor: MemoryExtractor,
        processor: MemoryProcessor,
        claim_extractor: ClaimExtractor | None = None,
        active_engine: ActiveLearningEngine | None = None,
        lifecycle: LearningLifecycle | None = None,
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

        learning_components = (
            claim_extractor,
            active_engine,
            lifecycle,
        )

        if any(item is not None for item in learning_components):
            if not all(item is not None for item in learning_components):
                raise ValueError(
                    "Claim learning requires claim_extractor, "
                    "active_engine, and lifecycle together."
                )

        if (
            lifecycle is not None
            and lifecycle.active_engine is not active_engine
        ):
            raise ValueError(
                "The lifecycle and worker must use the same "
                "ActiveLearningEngine instance."
            )

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

        self.claim_extractor = claim_extractor
        self.active_engine = active_engine
        self.lifecycle = lifecycle

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

                # Each successful batch advances the persistent queue.
                job.checkpoint_offset += len(messages)

            # Run insight discovery after all available conversation
            # evidence has been recorded. If this fails, the job retries.
            if self.lifecycle is not None:
                learning_result = self.lifecycle.run(
                    domain=LearningDomain.PRIVATE,
                    owner=job.owner,
                )

                result.insights_created += (
                    learning_result.insights_created
                )

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
        *,
        owner: str,
        source_ref: str,
    ) -> None:
        """Run memory extraction and optional structured claim learning."""

        # Existing memory pipeline.
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

        # Structured claim pipeline.
        if self.claim_extractor is None:
            return

        observations = await self.claim_extractor.extract(
            text,
            owner=owner,
            source_ref=source_ref,
        )

        result.claims_extracted += len(observations)

        for observation in observations:
            recorded = self.active_engine.record_claim(observation)

            if recorded:
                result.claims_recorded += 1
            else:
                result.claims_duplicate += 1

    async def _process_page(
        self,
        messages: list[Any],
        job: LearningJob,
        result: LearningRunResult,
    ) -> None:
        """Process a page and checkpoint each successful batch."""

        pending: list[str] = []
        pending_characters = 0
        pending_start_offset = job.checkpoint_offset + 1
        pending_end_offset = job.checkpoint_offset
        pending_message_count = 0

        async def flush_pending() -> None:
            nonlocal pending
            nonlocal pending_characters
            nonlocal pending_start_offset
            nonlocal pending_end_offset
            nonlocal pending_message_count

            if not pending:
                return

            source_ref = (
                f"conversation:{job.conversation_id}:"
                f"messages:{pending_start_offset}-{pending_end_offset}"
            )

            await self._extract_and_store(
                "\n\n".join(pending),
                result,
                owner=job.owner,
                source_ref=source_ref,
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

                # Each part has a stable source reference. The experience
                # ID also includes the claim key and value.
                for number, part in enumerate(parts, start=1):
                    part_text = (
                        prefix
                        + f"Part: {number}/{len(parts)}\n"
                        + part
                    )

                    source_ref = (
                        f"conversation:{job.conversation_id}:"
                        f"message:{message_offset}"
                    )

                    await self._extract_and_store(
                        part_text,
                        result,
                        owner=job.owner,
                        source_ref=source_ref,
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

            if not pending:
                pending_start_offset = message_offset

            pending.append(rendered)
            pending_characters += len(rendered) + 2
            pending_end_offset = message_offset
            pending_message_count += 1

        await flush_pending()
