
"""
CognitiveAI - Persistent Learning Repository

Durable SQLite storage for private and public learning.

Responsibilities:
- Persist authorized learning experiences.
- Persist structured claims and their evidence.
- Preserve historical insights.
- Preserve consolidation decision history.
- Prevent duplicate observations.
- Enforce private-owner isolation.
- Keep public and private learning strictly separated.

This repository does not automatically promote insights
or decisions into permanent memory.
"""

import json
import os
import sqlite3

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.learning.active import (
    ClaimObservation,
    LearningInsight,
)
from backend.learning.contracts import (
    LearningDomain,
    LearningExperience,
)
from backend.learning.consolidation import (
    ConsolidationDecision,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _owner_key(
    domain: LearningDomain,
    owner: str | None,
) -> str:
    """Return a consistent database key for a learning scope."""

    if domain == LearningDomain.PRIVATE:
        if not owner or not owner.strip():
            raise ValueError(
                "Private learning requires a valid owner."
            )
        return owner.strip()

    if domain == LearningDomain.PUBLIC:
        if owner is not None:
            raise ValueError(
                "Public learning cannot have a private owner."
            )
        return ""

    raise ValueError("Unsupported learning domain.")


def _normalize(value: str) -> str:
    """Normalize text for duplicate detection."""

    return " ".join(value.strip().casefold().split())


class LearningRepository:
    """
    Persistent storage for learning experiences, claims,
    insights, and consolidation decisions.

    The repository is independent of:
    - ConversationArchive
    - LearningQueue
    - BackgroundMemory
    - MemoryProcessor
    """

    def __init__(
        self,
        database_path: str | Path | None = None,
    ):
        configured_path = (
            database_path
            or os.getenv("COGNITIVEAI_LEARNING_REPOSITORY_PATH")
            or "data/learning_repository.sqlite3"
        )

        self.database_path = Path(configured_path)

        if str(self.database_path) != ":memory:":
            self.database_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._initialize_database()

    # ==========================================
    # DATABASE CONNECTION
    # ==========================================

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=30,
        )

        connection.row_factory = sqlite3.Row

        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")

        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS learning_experiences (
                    id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    owner TEXT,
                    owner_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    authorization_basis TEXT NOT NULL,
                    created_at TEXT NOT NULL,

                    CHECK (
                        (domain = 'private' AND owner IS NOT NULL)
                        OR
                        (domain = 'public' AND owner IS NULL)
                    )
                );

                CREATE INDEX IF NOT EXISTS
                    idx_learning_experiences_scope
                ON learning_experiences (
                    domain,
                    owner_key,
                    created_at
                );

                CREATE TABLE IF NOT EXISTS learning_claims (
                    id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    owner TEXT,
                    owner_key TEXT NOT NULL,
                    experience_id TEXT NOT NULL,
                    claim_key TEXT NOT NULL,
                    normalized_key TEXT NOT NULL,
                    claim_value TEXT NOT NULL,
                    normalized_value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,

                    FOREIGN KEY (experience_id)
                        REFERENCES learning_experiences(id),

                    UNIQUE (
                        domain,
                        owner_key,
                        experience_id,
                        normalized_key,
                        normalized_value
                    )
                );

                CREATE INDEX IF NOT EXISTS
                    idx_learning_claims_scope
                ON learning_claims (
                    domain,
                    owner_key,
                    normalized_key,
                    created_at
                );

                CREATE TABLE IF NOT EXISTS learning_insights (
                    id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    owner TEXT,
                    owner_key TEXT NOT NULL,
                    claim_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS
                    idx_learning_insights_scope
                ON learning_insights (
                    domain,
                    owner_key,
                    claim_key,
                    created_at
                );

                CREATE TABLE IF NOT EXISTS consolidation_decisions (
                    id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    owner TEXT,
                    owner_key TEXT NOT NULL,
                    claim_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    evaluated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS
                    idx_consolidation_decisions_scope
                ON consolidation_decisions (
                    domain,
                    owner_key,
                    claim_key,
                    evaluated_at
                );
                """
            )

    # ==========================================
    # VALIDATION
    # ==========================================

    @staticmethod
    def _validate_experience_scope(
        experience: LearningExperience,
    ) -> str:
        """Validate the experience and return its owner key."""

        return _owner_key(
            experience.domain,
            experience.owner,
        )

    def _verify_existing_experience(
        self,
        connection: sqlite3.Connection,
        experience: LearningExperience,
        owner_key: str,
    ) -> None:
        """Reject reuse of an experience ID with different data."""

        row = connection.execute(
            """
            SELECT *
            FROM learning_experiences
            WHERE id = ?
            """,
            (experience.id,),
        ).fetchone()

        if row is None:
            return

        expected = (
            experience.domain.value,
            experience.owner,
            owner_key,
            experience.content,
            experience.source_type.value,
            experience.source_ref,
            experience.authorization_basis.value,
            experience.created_at.isoformat(),
        )

        actual = (
            row["domain"],
            row["owner"],
            row["owner_key"],
            row["content"],
            row["source_type"],
            row["source_ref"],
            row["authorization_basis"],
            row["created_at"],
        )

        if actual != expected:
            raise ValueError(
                "An existing experience ID cannot be reused "
                "with different experience data."
            )

    # ==========================================
    # EXPERIENCE STORAGE
    # ==========================================

    def record_experience(
        self,
        experience: LearningExperience,
    ) -> bool:
        """
        Persist one authorized learning experience.

        Returns:
            True if inserted.
            False if the exact experience already exists.
        """

        owner_key = self._validate_experience_scope(
            experience
        )

        with self._connect() as connection:
            self._verify_existing_experience(
                connection,
                experience,
                owner_key,
            )

            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO learning_experiences (
                    id,
                    domain,
                    owner,
                    owner_key,
                    content,
                    source_type,
                    source_ref,
                    authorization_basis,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experience.id,
                    experience.domain.value,
                    experience.owner,
                    owner_key,
                    experience.content,
                    experience.source_type.value,
                    experience.source_ref,
                    experience.authorization_basis.value,
                    experience.created_at.isoformat(),
                ),
            )

            return cursor.rowcount == 1

    def get_experience(
        self,
        experience_id: str,
        *,
        domain: LearningDomain,
        owner: str | None = None,
    ) -> LearningExperience | None:
        """Retrieve an experience within its authorized scope."""

        owner_key = _owner_key(domain, owner)

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM learning_experiences
                WHERE id = ?
                  AND domain = ?
                  AND owner_key = ?
                """,
                (
                    experience_id,
                    domain.value,
                    owner_key,
                ),
            ).fetchone()

        if row is None:
            return None

        return LearningExperience(
            id=row["id"],
            domain=row["domain"],
            owner=row["owner"],
            content=row["content"],
            source_type=row["source_type"],
            source_ref=row["source_ref"],
            authorization_basis=row["authorization_basis"],
            created_at=datetime.fromisoformat(
                row["created_at"]
            ),
        )

    def list_experiences(
        self,
        *,
        domain: LearningDomain,
        owner: str | None = None,
        limit: int = 100,
    ) -> list[LearningExperience]:
        """List experiences without crossing learning boundaries."""

        if limit < 1:
            raise ValueError("Limit must be at least 1.")

        owner_key = _owner_key(domain, owner)

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM learning_experiences
                WHERE domain = ?
                  AND owner_key = ?
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (
                    domain.value,
                    owner_key,
                    limit,
                ),
            ).fetchall()

        return [
            LearningExperience(
                id=row["id"],
                domain=row["domain"],
                owner=row["owner"],
                content=row["content"],
                source_type=row["source_type"],
                source_ref=row["source_ref"],
                authorization_basis=row["authorization_basis"],
                created_at=datetime.fromisoformat(
                    row["created_at"]
                ),
            )
            for row in rows
        ]

    # ==========================================
    # CLAIM STORAGE
    # ==========================================

    def record_claim(
        self,
        observation: ClaimObservation,
    ) -> bool:
        """
        Persist an observation and its experience atomically.

        Returns:
            True if the claim was inserted.
            False if the same claim already exists.
        """

        experience = observation.experience

        owner_key = self._validate_experience_scope(
            experience
        )

        claim_key = observation.key.strip()
        claim_value = observation.value.strip()

        if not claim_key:
            raise ValueError("Claim key cannot be empty.")

        if not claim_value:
            raise ValueError("Claim value cannot be empty.")

        if not 0 <= observation.confidence <= 1:
            raise ValueError(
                "Claim confidence must be between 0 and 1."
            )

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            self._verify_existing_experience(
                connection,
                experience,
                owner_key,
            )

            connection.execute(
                """
                INSERT OR IGNORE INTO learning_experiences (
                    id,
                    domain,
                    owner,
                    owner_key,
                    content,
                    source_type,
                    source_ref,
                    authorization_basis,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experience.id,
                    experience.domain.value,
                    experience.owner,
                    owner_key,
                    experience.content,
                    experience.source_type.value,
                    experience.source_ref,
                    experience.authorization_basis.value,
                    experience.created_at.isoformat(),
                ),
            )

            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO learning_claims (
                    id,
                    domain,
                    owner,
                    owner_key,
                    experience_id,
                    claim_key,
                    normalized_key,
                    claim_value,
                    normalized_value,
                    confidence,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    experience.domain.value,
                    experience.owner,
                    owner_key,
                    experience.id,
                    claim_key,
                    _normalize(claim_key),
                    claim_value,
                    _normalize(claim_value),
                    observation.confidence,
                    utc_now().isoformat(),
                ),
            )

            return cursor.rowcount == 1

    def get_claim_history(
        self,
        *,
        domain: LearningDomain,
        owner: str | None = None,
        claim_key: str | None = None,
        limit: int = 500,
    ) -> list[ClaimObservation]:
        """
        Retrieve claims and their original experiences.

        Results are always restricted to the requested scope.
        """

        if limit < 1:
            raise ValueError("Limit must be at least 1.")

        owner_key = _owner_key(domain, owner)

        query = """
            SELECT
                c.claim_key,
                c.claim_value,
                c.confidence,
                e.id AS experience_id,
                e.domain,
                e.owner,
                e.content,
                e.source_type,
                e.source_ref,
                e.authorization_basis,
                e.created_at
            FROM learning_claims AS c
            JOIN learning_experiences AS e
                ON e.id = c.experience_id
            WHERE c.domain = ?
              AND c.owner_key = ?
        """

        parameters: list[Any] = [
            domain.value,
            owner_key,
        ]

        if claim_key is not None:
            query += " AND c.normalized_key = ?"
            parameters.append(_normalize(claim_key))

        query += """
            ORDER BY c.created_at ASC, c.id ASC
            LIMIT ?
        """

        parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        observations = []

        for row in rows:
            experience = LearningExperience(
                id=row["experience_id"],
                domain=row["domain"],
                owner=row["owner"],
                content=row["content"],
                source_type=row["source_type"],
                source_ref=row["source_ref"],
                authorization_basis=row["authorization_basis"],
                created_at=datetime.fromisoformat(
                    row["created_at"]
                ),
            )

            observations.append(
                ClaimObservation(
                    experience=experience,
                    key=row["claim_key"],
                    value=row["claim_value"],
                    confidence=row["confidence"],
                )
            )

        return observations

    # ==========================================
    # INSIGHT STORAGE
    # ==========================================

    def _validate_insight_evidence(
        self,
        connection: sqlite3.Connection,
        insight: LearningInsight,
        owner_key: str,
    ) -> None:
        """Ensure every evidence reference belongs to this scope."""

        if insight.domain not in (
            LearningDomain.PRIVATE,
            LearningDomain.PUBLIC,
        ):
            raise ValueError("Unsupported insight domain.")

        if insight.domain == LearningDomain.PRIVATE:
            if not insight.owner or not insight.owner.strip():
                raise ValueError(
                    "Private insights must have an owner."
                )

        elif insight.owner is not None:
            raise ValueError(
                "Public insights cannot have a private owner."
            )

        if not insight.claim_key.strip():
            raise ValueError("Insight claim key cannot be empty.")

        if not insight.values:
            raise ValueError("Insight must contain values.")

        seen_experience_ids = set()
        total_evidence = 0

        for value in insight.values:
            if not value.value.strip():
                raise ValueError("Insight values cannot be empty.")

            if not value.evidence:
                raise ValueError(
                    "Every insight value requires evidence."
                )

            if value.support_count != len(value.evidence):
                raise ValueError(
                    "Value support count does not match its evidence."
                )

            for reference in value.evidence:
                if (
                    not reference.experience_id.strip()
                    or not reference.source_ref.strip()
                ):
                    raise ValueError(
                        "Evidence references require an experience "
                        "ID and source reference."
                    )

                if reference.experience_id in seen_experience_ids:
                    raise ValueError(
                        "An experience cannot be counted more "
                        "than once in an insight."
                    )

                seen_experience_ids.add(
                    reference.experience_id
                )
                total_evidence += 1

                row = connection.execute(
                    """
                    SELECT source_ref
                    FROM learning_experiences
                    WHERE id = ?
                      AND domain = ?
                      AND owner_key = ?
                    """,
                    (
                        reference.experience_id,
                        insight.domain.value,
                        owner_key,
                    ),
                ).fetchone()

                if row is None:
                    raise ValueError(
                        "Insight evidence references an experience "
                        "outside its authorized learning scope."
                    )

                if row["source_ref"] != reference.source_ref:
                    raise ValueError(
                        "Evidence source reference does not match "
                        "the stored experience."
                    )

        if total_evidence != insight.evidence_count:
            raise ValueError(
                "Insight evidence count does not match "
                "its supporting references."
            )

    def save_insight(
        self,
        insight: LearningInsight,
    ) -> str:
        """
        Persist an insight snapshot.

        Each save creates a new historical record.
        Existing insight history is never overwritten.
        """

        owner_key = _owner_key(
            insight.domain,
            insight.owner,
        )

        insight_id = str(uuid4())
        payload = insight.model_dump_json()

        with self._connect() as connection:
            self._validate_insight_evidence(
                connection,
                insight,
                owner_key,
            )

            connection.execute(
                """
                INSERT INTO learning_insights (
                    id,
                    domain,
                    owner,
                    owner_key,
                    claim_key,
                    payload,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    insight_id,
                    insight.domain.value,
                    insight.owner,
                    owner_key,
                    insight.claim_key,
                    payload,
                    utc_now().isoformat(),
                ),
            )

        return insight_id

    def get_insight(
        self,
        insight_id: str,
        *,
        domain: LearningDomain,
        owner: str | None = None,
    ) -> LearningInsight | None:
        """Retrieve one insight within its authorized scope."""

        owner_key = _owner_key(domain, owner)

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload
                FROM learning_insights
                WHERE id = ?
                  AND domain = ?
                  AND owner_key = ?
                """,
                (
                    insight_id,
                    domain.value,
                    owner_key,
                ),
            ).fetchone()

        if row is None:
            return None

        return LearningInsight.model_validate_json(
            row["payload"]
        )

    def list_insights(
        self,
        *,
        domain: LearningDomain,
        owner: str | None = None,
        claim_key: str | None = None,
        limit: int = 100,
    ) -> list[LearningInsight]:
        """List historical insight snapshots."""

        if limit < 1:
            raise ValueError("Limit must be at least 1.")

        owner_key = _owner_key(domain, owner)

        query = """
            SELECT payload
            FROM learning_insights
            WHERE domain = ?
              AND owner_key = ?
        """

        parameters: list[Any] = [
            domain.value,
            owner_key,
        ]

        if claim_key is not None:
            query += " AND claim_key = ?"
            parameters.append(claim_key)

        query += """
            ORDER BY created_at ASC, id ASC
            LIMIT ?
        """

        parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return [
            LearningInsight.model_validate_json(
                row["payload"]
            )
            for row in rows
        ]

    # ==========================================
    # CONSOLIDATION DECISION STORAGE
    # ==========================================

    def save_decision(
        self,
        decision: ConsolidationDecision,
    ) -> str:
        """
        Persist a consolidation decision.

        Decisions are append-only historical records.
        Saving a decision never writes to permanent memory.
        """

        owner_key = _owner_key(
            decision.domain,
            decision.owner,
        )

        decision_id = str(uuid4())

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO consolidation_decisions (
                    id,
                    domain,
                    owner,
                    owner_key,
                    claim_key,
                    status,
                    payload,
                    evaluated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    decision.domain.value,
                    decision.owner,
                    owner_key,
                    decision.claim_key,
                    decision.status.value,
                    decision.model_dump_json(),
                    decision.evaluated_at.isoformat(),
                ),
            )

        return decision_id

    def list_decisions(
        self,
        *,
        domain: LearningDomain,
        owner: str | None = None,
        claim_key: str | None = None,
        limit: int = 100,
    ) -> list[ConsolidationDecision]:
        """Retrieve consolidation decision history."""

        if limit < 1:
            raise ValueError("Limit must be at least 1.")

        owner_key = _owner_key(domain, owner)

        query = """
            SELECT payload
            FROM consolidation_decisions
            WHERE domain = ?
              AND owner_key = ?
        """

        parameters: list[Any] = [
            domain.value,
            owner_key,
        ]

        if claim_key is not None:
            query += " AND claim_key = ?"
            parameters.append(claim_key)

        query += """
            ORDER BY evaluated_at ASC, id ASC
            LIMIT ?
        """

        parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return [
            ConsolidationDecision.model_validate_json(
                row["payload"]
            )
            for row in rows
        ]

    # ==========================================
    # REPOSITORY STATISTICS
    # ==========================================

    def get_statistics(
        self,
        *,
        domain: LearningDomain,
        owner: str | None = None,
    ) -> dict[str, int]:
        """Return counts for one isolated learning scope."""

        owner_key = _owner_key(domain, owner)

        with self._connect() as connection:
            experiences = connection.execute(
                """
                SELECT COUNT(*)
                FROM learning_experiences
                WHERE domain = ? AND owner_key = ?
                """,
                (domain.value, owner_key),
            ).fetchone()[0]

            claims = connection.execute(
                """
                SELECT COUNT(*)
                FROM learning_claims
                WHERE domain = ? AND owner_key = ?
                """,
                (domain.value, owner_key),
            ).fetchone()[0]

            insights = connection.execute(
                """
                SELECT COUNT(*)
                FROM learning_insights
                WHERE domain = ? AND owner_key = ?
                """,
                (domain.value, owner_key),
            ).fetchone()[0]

            decisions = connection.execute(
                """
                SELECT COUNT(*)
                FROM consolidation_decisions
                WHERE domain = ? AND owner_key = ?
                """,
                (domain.value, owner_key),
            ).fetchone()[0]

        return {
            "experiences": experiences,
            "claims": claims,
            "insights": insights,
            "decisions": decisions,
        }
