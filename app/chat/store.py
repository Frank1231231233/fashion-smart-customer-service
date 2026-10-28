"""会话注册表:记录 thread_id 与 user_id 的归属关系,支持按用户列历史会话。"""

import sqlite3
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

CHAT_DB_PATH = Path("data/chats.db")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ChatSession(TypedDict):
    thread_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


class ChatStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def create_session(self, user_id: str, thread_id: str, title: str) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions (thread_id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (thread_id, user_id, title.strip() or "新会话", now, now),
            )

    def get_session(self, thread_id: str) -> ChatSession | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT thread_id, user_id, title, created_at, updated_at
                FROM chat_sessions
                WHERE thread_id = ?
                """,
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        return ChatSession(
            thread_id=row[0],
            user_id=row[1],
            title=row[2],
            created_at=row[3],
            updated_at=row[4],
        )

    def touch_session(self, thread_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE chat_sessions
                SET updated_at = ?
                WHERE thread_id = ?
                """,
                (_now(), thread_id),
            )

    def list_sessions(self, user_id: str) -> list[ChatSession]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT thread_id, user_id, title, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [
            ChatSession(
                thread_id=row[0],
                user_id=row[1],
                title=row[2],
                created_at=row[3],
                updated_at=row[4],
            )
            for row in rows
        ]


@lru_cache
def get_chat_store() -> ChatStore:
    return ChatStore(CHAT_DB_PATH)
