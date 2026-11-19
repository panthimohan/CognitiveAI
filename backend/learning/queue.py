
"""
CognitiveAI - Persistent Background Learning Queue

Tracks archived conversations that still need to be processed.

The queue is independent of the conversation archive:
- Original messages are never deleted.
- Jobs survive application restarts.
- Duplicate enqueue operations are safe.
- New messages reopen completed jobs.
- Checkpoints allow processing to resume.
- Failed jobs can be retried.
- Expired running jobs can be reclaimed.

This version queues private conversations only.
Public conversations must never enter personal learning automatically.
"""

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4


JobStatus = Literal[
    "pending",
    "running",
    "failed",
    "completed",
    "dead",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


@dataclass
class LearningJob:
    id: str
    owner: str
    conversation_id: str
    privacy: str
    status: str
    attempts: int
    checkpoint_offset: int
    last_error: str | None
    next_attempt_at: str
    lease_until: str | None
    created_at: str
    updated_at: str


class LearningQueue:
    """Durable SQLite queue for background learning jobs."""

    def __init__(self, database_path: str | Path | None = None):
        if database_path is None:
            database_path = os.getenv(
                "COGNITIVEAI_LEARNING_QUEUE_PATH",
                "data/learning_queue.sqlite3",
            )

        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_jobs (
                    id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    privacy TEXT NOT NULL
                        CHECK(privacy = 'private'),
                    status TEXT NOT NULL
                        CHECK(status IN (
                            'pending', 'running', 'failed',
                            'completed', 'dead'
                        )),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    checkpoint_offset INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    next_attempt_at TEXT NOT NULL,
                    lease_until TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(owner, conversation_id, privacy)
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_learning_jobs_ready
                ON learning_jobs (status, next_attempt_at, created_at)
                """
            )

    def enqueue(
        self,
        *,
        owner: str,
        conversation_id: str,
        privacy: str = "private",
    ) -> LearningJob:
        """
        Add a conversation to the queue.

        Repeated calls return the existing job rather than creating
        duplicates. Archive discovery handles new-message detection.
        """

        if not owner or not owner.strip():
            raise ValueError("Owner cannot be empty.")
        if not conversation_id or not conversation_id.strip():
            raise ValueError("Conversation ID cannot be empty.")
        if privacy != "private":
            raise ValueError(
                "Only private conversations can enter personal learning."
            )

        now = _timestamp()
        job_id = str(uuid4())

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO learning_jobs (
                    id, owner, conversation_id, privacy,
                    status, attempts, checkpoint_offset,
                    last_error, next_attempt_at, lease_until,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, 'private', 'pending', 0, 0,
                        NULL, ?, NULL, ?, ?)
                """,
                (job_id, owner, conversation_id, now, now, now),
            )

            row = connection.execute(
                """
                SELECT *
                FROM learning_jobs
                WHERE owner = ?
                  AND conversation_id = ?
                  AND privacy = 'private'
                """,
                (owner, conversation_id),
            ).fetchone()

        return self._to_job(row)

    def enqueue_archived_conversations(
        self,
        archive,
        *,
        owner: str,
    ) -> list[LearningJob]:
        """
        Discover private archived conversations.

        If a completed job has fewer processed messages than the archive
        currently contains, reopen that same job from its checkpoint.

        Running, failed, and dead jobs are not reset by discovery.
        """

        if not owner or not owner.strip():
            raise ValueError("Owner cannot be empty.")

        summaries = archive.list_conversations(
            owner=owner,
            privacy="private",
        )

        jobs = []

        for summary in summaries:
            message_count = int(summary["message_count"])

            if message_count <= 0:
                continue

            conversation_id = summary["conversation_id"]

            job = self._enqueue_archived_conversation(
                owner=owner,
                conversation_id=conversation_id,
                message_count=message_count,
            )

            jobs.append(job)

        return jobs

    def _enqueue_archived_conversation(
        self,
        *,
        owner: str,
        conversation_id: str,
        message_count: int,
    ) -> LearningJob:
        """Create a job or reopen a completed job when new messages exist."""

        now = _timestamp()
        job_id = str(uuid4())

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            connection.execute(
                """
                INSERT OR IGNORE INTO learning_jobs (
                    id, owner, conversation_id, privacy,
                    status, attempts, checkpoint_offset,
                    last_error, next_attempt_at, lease_until,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, 'private', 'pending', 0, 0,
                        NULL, ?, NULL, ?, ?)
                """,
                (job_id, owner, conversation_id, now, now, now),
            )

            row = connection.execute(
                """
                SELECT *
                FROM learning_jobs
                WHERE owner = ?
                  AND conversation_id = ?
                  AND privacy = 'private'
                """,
                (owner, conversation_id),
            ).fetchone()

            if (
                row["status"] == "completed"
                and message_count > row["checkpoint_offset"]
            ):
                connection.execute(
                    """
                    UPDATE learning_jobs
                    SET status = 'pending',
                        attempts = 0,
                        last_error = NULL,
                        next_attempt_at = ?,
                        lease_until = NULL,
                        updated_at = ?
                    WHERE id = ?
                      AND status = 'completed'
                    """,
                    (now, now, row["id"]),
                )

                row = connection.execute(
                    """
                    SELECT *
                    FROM learning_jobs
                    WHERE id = ?
                    """,
                    (row["id"],),
                ).fetchone()

            connection.commit()

        return self._to_job(row)

    def claim_next(
        self,
        *,
        lease_seconds: int = 300,
        max_attempts: int = 5,
    ) -> LearningJob | None:
        """Atomically claim the next available job."""

        if lease_seconds < 1:
            raise ValueError("Lease duration must be positive.")
        if max_attempts < 1:
            raise ValueError("Maximum attempts must be positive.")

        now = _now()
        now_text = _timestamp(now)
        lease_text = _timestamp(now + timedelta(seconds=lease_seconds))

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            connection.execute(
                """
                UPDATE learning_jobs
                SET status = 'dead',
                    lease_until = NULL,
                    updated_at = ?
                WHERE status = 'running'
                  AND lease_until <= ?
                  AND attempts >= ?
                """,
                (now_text, now_text, max_attempts),
            )

            row = connection.execute(
                """
                SELECT *
                FROM learning_jobs
                WHERE attempts < ?
                  AND (
                    (status IN ('pending', 'failed')
                     AND next_attempt_at <= ?)
                    OR
                    (status = 'running' AND lease_until <= ?)
                  )
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (max_attempts, now_text, now_text),
            ).fetchone()

            if row is None:
                connection.commit()
                return None

            connection.execute(
                """
                UPDATE learning_jobs
                SET status = 'running',
                    attempts = attempts + 1,
                    lease_until = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (lease_text, now_text, row["id"]),
            )

            updated = connection.execute(
                "SELECT * FROM learning_jobs WHERE id = ?",
                (row["id"],),
            ).fetchone()

            connection.commit()

        return self._to_job(updated)

    def update_checkpoint(
        self,
        *,
        job_id: str,
        checkpoint_offset: int,
    ) -> LearningJob:
        """Advance a running job's chronological message offset."""

        if checkpoint_offset < 0:
            raise ValueError("Checkpoint cannot be negative.")

        now = _timestamp()

        with self._connect() as connection:
            result = connection.execute(
                """
                UPDATE learning_jobs
                SET checkpoint_offset = ?,
                    updated_at = ?
                WHERE id = ?
                  AND status = 'running'
                  AND checkpoint_offset <= ?
                """,
                (checkpoint_offset, now, job_id, checkpoint_offset),
            )

            if result.rowcount != 1:
                raise ValueError(
                    "Job is not running, does not exist, or checkpoint "
                    "would move backwards."
                )

            row = connection.execute(
                "SELECT * FROM learning_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()

        return self._to_job(row)

    def complete(self, *, job_id: str) -> LearningJob:
        """Mark a running job as completed."""

        now = _timestamp()

        with self._connect() as connection:
            result = connection.execute(
                """
                UPDATE learning_jobs
                SET status = 'completed',
                    lease_until = NULL,
                    last_error = NULL,
                    updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (now, job_id),
            )

            if result.rowcount != 1:
                raise ValueError("Only a running job can be completed.")

            row = connection.execute(
                "SELECT * FROM learning_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()

        return self._to_job(row)

    def fail(
        self,
        *,
        job_id: str,
        error: str,
        retry_delay_seconds: int = 60,
        max_attempts: int = 5,
    ) -> LearningJob:
        """Record a failure and schedule a retry or mark the job dead."""

        if retry_delay_seconds < 0:
            raise ValueError("Retry delay cannot be negative.")
        if max_attempts < 1:
            raise ValueError("Maximum attempts must be positive.")

        now = _now()
        now_text = _timestamp(now)

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM learning_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()

            if row is None or row["status"] != "running":
                raise ValueError("Only a running job can be failed.")

            status = (
                "dead" if row["attempts"] >= max_attempts else "failed"
            )
            next_attempt = _timestamp(
                now + timedelta(seconds=retry_delay_seconds)
            )

            connection.execute(
                """
                UPDATE learning_jobs
                SET status = ?,
                    last_error = ?,
                    next_attempt_at = ?,
                    lease_until = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    str(error)[:4000],
                    next_attempt,
                    now_text,
                    job_id,
                ),
            )

            updated = connection.execute(
                "SELECT * FROM learning_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()

        return self._to_job(updated)

    def get_job(self, job_id: str) -> LearningJob | None:
        """Retrieve one job by ID."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM learning_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()

        return self._to_job(row) if row else None

    def list_jobs(
        self,
        *,
        owner: str | None = None,
        status: str | None = None,
    ) -> list[LearningJob]:
        """List jobs, optionally filtered by owner and status."""

        query = "SELECT * FROM learning_jobs WHERE 1 = 1"
        parameters: list = []

        if owner is not None:
            query += " AND owner = ?"
            parameters.append(owner)

        if status is not None:
            query += " AND status = ?"
            parameters.append(status)

        query += " ORDER BY created_at ASC"

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        return [self._to_job(row) for row in rows]

    @staticmethod
    def _to_job(row: sqlite3.Row) -> LearningJob:
        return LearningJob(
            id=row["id"],
            owner=row["owner"],
            conversation_id=row["conversation_id"],
            privacy=row["privacy"],
            status=row["status"],
            attempts=row["attempts"],
            checkpoint_offset=row["checkpoint_offset"],
            last_error=row["last_error"],
            next_attempt_at=row["next_attempt_at"],
            lease_until=row["lease_until"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
