
"""
CognitiveAI - Persistent Conversation Archive

Stores original user messages and AI responses in SQLite.
The archive is independent of the memory extraction system.

A saved message remains available even if background learning fails.
"""

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4


MessageRole = Literal["user", "assistant"]
PrivacyScope = Literal["private", "public"]


@dataclass
class ConversationMessage:
    id: str
    owner: str
    conversation_id: str
    interaction_id: str
    role: str
    content: str
    privacy: str
    created_at: str


class ConversationArchive:
    """Persistent SQLite storage for complete conversation history."""

    def __init__(self, database_path: str | Path | None = None):
        if database_path is None:
            database_path = os.getenv(
                "COGNITIVEAI_ARCHIVE_PATH",
                "data/conversations.sqlite3",
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
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    interaction_id TEXT NOT NULL,
                    role TEXT NOT NULL
                        CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    privacy TEXT NOT NULL
                        CHECK(privacy IN ('private', 'public')),
                    created_at TEXT NOT NULL,
                    UNIQUE(owner, interaction_id, role)
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_history
                ON conversation_messages (
                    owner, conversation_id, created_at
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_privacy
                ON conversation_messages (owner, privacy, created_at)
                """
            )

    def save_message(
        self,
        *,
        owner: str,
        conversation_id: str,
        interaction_id: str,
        role: MessageRole,
        content: str,
        privacy: PrivacyScope = "private",
    ) -> ConversationMessage:
        """Save one message. Duplicate interaction/role pairs are ignored."""

        if not owner.strip():
            raise ValueError("Owner cannot be empty.")
        if not conversation_id.strip():
            raise ValueError("Conversation ID cannot be empty.")
        if not interaction_id.strip():
            raise ValueError("Interaction ID cannot be empty.")
        if role not in {"user", "assistant"}:
            raise ValueError("Role must be user or assistant.")
        if privacy not in {"private", "public"}:
            raise ValueError("Privacy must be private or public.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message content cannot be empty.")

        message_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO conversation_messages (
                    id, owner, conversation_id, interaction_id,
                    role, content, privacy, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    owner,
                    conversation_id,
                    interaction_id,
                    role,
                    content,
                    privacy,
                    created_at,
                ),
            )

            row = connection.execute(
                """
                SELECT *
                FROM conversation_messages
                WHERE owner = ? AND interaction_id = ? AND role = ?
                """,
                (owner, interaction_id, role),
            ).fetchone()

        return self._to_message(row)

    def get_conversation(
        self,
        *,
        owner: str,
        conversation_id: str,
        limit: int = 1000,
        privacy: PrivacyScope | None = None,
    ) -> list[ConversationMessage]:
        """Return messages chronologically, optionally filtered by privacy."""

        if not owner.strip():
            raise ValueError("Owner cannot be empty.")
        if not conversation_id.strip():
            raise ValueError("Conversation ID cannot be empty.")
        if limit < 1 or limit > 10000:
            raise ValueError("Limit must be between 1 and 10000.")
        if privacy is not None and privacy not in {"private", "public"}:
            raise ValueError("Invalid privacy scope.")

        query = """
            SELECT *
            FROM conversation_messages
            WHERE owner = ? AND conversation_id = ?
        """
        parameters: list = [owner, conversation_id]

        if privacy is not None:
            query += " AND privacy = ?"
            parameters.append(privacy)

        query += " ORDER BY created_at ASC, rowid ASC LIMIT ?"
        parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        return [self._to_message(row) for row in rows]

    def get_conversation_page(
        self,
        *,
        owner: str,
        conversation_id: str,
        offset: int = 0,
        limit: int = 500,
        privacy: PrivacyScope | None = None,
    ) -> list[ConversationMessage]:
        """
        Return one chronological page of conversation history.

        Append-only archive plus offset pagination allows background
        learning to process conversations larger than 10,000 messages.
        """

        if not owner or not owner.strip():
            raise ValueError("Owner cannot be empty.")
        if not conversation_id or not conversation_id.strip():
            raise ValueError("Conversation ID cannot be empty.")
        if offset < 0:
            raise ValueError("Offset cannot be negative.")
        if limit < 1 or limit > 10000:
            raise ValueError("Limit must be between 1 and 10000.")
        if privacy is not None and privacy not in {"private", "public"}:
            raise ValueError("Invalid privacy scope.")

        query = """
            SELECT *
            FROM conversation_messages
            WHERE owner = ? AND conversation_id = ?
        """
        parameters: list = [owner, conversation_id]

        if privacy is not None:
            query += " AND privacy = ?"
            parameters.append(privacy)

        query += """
            ORDER BY created_at ASC, rowid ASC
            LIMIT ? OFFSET ?
        """
        parameters.extend([limit, offset])

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        return [self._to_message(row) for row in rows]

    def list_conversations(
        self,
        *,
        owner: str,
        privacy: PrivacyScope | None = None,
    ) -> list[dict]:
        """Return one summary per conversation."""

        if not owner.strip():
            raise ValueError("Owner cannot be empty.")
        if privacy is not None and privacy not in {"private", "public"}:
            raise ValueError("Invalid privacy scope.")

        query = """
            SELECT
                conversation_id,
                privacy,
                COUNT(*) AS message_count,
                MIN(created_at) AS first_message_at,
                MAX(created_at) AS last_message_at
            FROM conversation_messages
            WHERE owner = ?
        """
        parameters: list = [owner]

        if privacy is not None:
            query += " AND privacy = ?"
            parameters.append(privacy)

        query += """
            GROUP BY conversation_id, privacy
            ORDER BY last_message_at ASC
        """

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        return [dict(row) for row in rows]

    def list_messages(
        self,
        *,
        owner: str,
        privacy: PrivacyScope | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ConversationMessage]:
        """List archived messages, newest first."""

        if limit < 1 or limit > 10000:
            raise ValueError("Limit must be between 1 and 10000.")
        if offset < 0:
            raise ValueError("Offset cannot be negative.")
        if privacy is not None and privacy not in {"private", "public"}:
            raise ValueError("Invalid privacy scope.")

        query = """
            SELECT *
            FROM conversation_messages
            WHERE owner = ?
        """
        parameters: list = [owner]

        if privacy is not None:
            query += " AND privacy = ?"
            parameters.append(privacy)

        query += """
            ORDER BY created_at DESC, rowid DESC
            LIMIT ? OFFSET ?
        """
        parameters.extend([limit, offset])

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        return [self._to_message(row) for row in rows]

    def count_messages(
        self,
        *,
        owner: str,
        privacy: PrivacyScope | None = None,
    ) -> int:
        """Count archived messages."""

        query = """
            SELECT COUNT(*)
            FROM conversation_messages
            WHERE owner = ?
        """
        parameters: list = [owner]

        if privacy is not None:
            if privacy not in {"private", "public"}:
                raise ValueError("Invalid privacy scope.")
            query += " AND privacy = ?"
            parameters.append(privacy)

        with self._connect() as connection:
            result = connection.execute(query, parameters).fetchone()

        return int(result[0])

    @staticmethod
    def _to_message(row: sqlite3.Row) -> ConversationMessage:
        return ConversationMessage(
            id=row["id"],
            owner=row["owner"],
            conversation_id=row["conversation_id"],
            interaction_id=row["interaction_id"],
            role=row["role"],
            content=row["content"],
            privacy=row["privacy"],
            created_at=row["created_at"],
        )
