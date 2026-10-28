"""人工审核存储:SQLite 持久化待审任务(review_tasks)与售后工单(tickets)。

状态机:pending -(resume)→ accepted_for_manual_processing / rejected / need_more_info,
处理动作写入 operator_id 与 processed_at,便于追责"哪个人、什么时候批的"。
"""

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

REVIEW_DB_PATH = Path("data/reviews.db")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ReviewStore:
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
                CREATE TABLE IF NOT EXISTS review_tasks (
                    task_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    trace_id TEXT,
                    user_id TEXT,
                    order_id TEXT,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    note TEXT,
                    operator_id TEXT,
                    created_at TEXT NOT NULL,
                    processed_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'created',
                    created_at TEXT NOT NULL
                )
                """
            )

    def create_pending_task(
        self,
        *,
        thread_id: str,
        trace_id: str,
        payload: dict[str, object],
        order_id: str | None = None,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "type": "pending",
            "task_id": f"TSK-{uuid.uuid4().hex[:8].upper()}",
            "thread_id": thread_id,
            "trace_id": trace_id,
            "user_id": payload.get("user_id"),
            "order_id": order_id,
            "payload": payload,
            "status": "pending",
            "note": None,
            "operator_id": None,
            "created_at": _now(),
            "processed_at": None,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO review_tasks (
                    task_id, thread_id, trace_id, user_id, order_id,
                    payload_json, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    record["task_id"],
                    thread_id,
                    trace_id,
                    record["user_id"],
                    order_id,
                    json.dumps(payload, ensure_ascii=False),
                    record["created_at"],
                ),
            )
        return record

    def resolve_pending_task(
        self,
        *,
        thread_id: str,
        status: str,
        note: str | None = None,
        operator_id: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE review_tasks
                SET status = ?, note = ?, operator_id = ?, processed_at = ?
                WHERE thread_id = ? AND status = 'pending'
                """,
                (status, note, operator_id, _now(), thread_id),
            )

    def create_ticket(
        self,
        *,
        thread_id: str,
        user_id: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "type": "ticket",
            "ticket_id": f"TKT-{uuid.uuid4().hex[:8].upper()}",
            "thread_id": thread_id,
            "user_id": user_id,
            "payload": payload,
            "status": "created",
            "created_at": _now(),
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tickets (
                    ticket_id, thread_id, user_id, payload_json, status, created_at
                )
                VALUES (?, ?, ?, ?, 'created', ?)
                """,
                (
                    record["ticket_id"],
                    thread_id,
                    user_id,
                    json.dumps(payload, ensure_ascii=False),
                    record["created_at"],
                ),
            )
        return record

    def list_records(
        self,
        *,
        record_type: str | None = None,
        thread_id: str | None = None,
    ) -> list[dict[str, object]]:
        if record_type == "ticket":
            return self._list_tickets(thread_id)
        return self._list_tasks(record_type, thread_id)

    def _list_tasks(
        self,
        record_type: str | None,
        thread_id: str | None,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        params: list[str] = []
        if record_type == "pending":
            clauses.append("status = 'pending'")
        elif record_type == "resolution":
            clauses.append("processed_at IS NOT NULL")
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT task_id, thread_id, trace_id, user_id, order_id, payload_json,
                       status, note, operator_id, created_at, processed_at
                FROM review_tasks
                {where}
                ORDER BY created_at
                """,
                params,
            ).fetchall()
        return [
            {
                "type": "resolution" if row[10] is not None else "pending",
                "task_id": row[0],
                "thread_id": row[1],
                "trace_id": row[2],
                "user_id": row[3],
                "order_id": row[4],
                "payload": json.loads(row[5]),
                "status": row[6],
                "note": row[7],
                "operator_id": row[8],
                "created_at": row[9],
                "processed_at": row[10],
            }
            for row in rows
        ]

    def _list_tickets(self, thread_id: str | None) -> list[dict[str, object]]:
        where = "WHERE thread_id = ?" if thread_id is not None else ""
        params = [thread_id] if thread_id is not None else []
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT ticket_id, thread_id, user_id, payload_json, status, created_at
                FROM tickets
                {where}
                ORDER BY created_at
                """,
                params,
            ).fetchall()
        return [
            {
                "type": "ticket",
                "ticket_id": row[0],
                "thread_id": row[1],
                "user_id": row[2],
                "payload": json.loads(row[3]),
                "status": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]


@lru_cache
def get_review_store() -> ReviewStore:
    return ReviewStore(REVIEW_DB_PATH)


def create_pending_task(
    *,
    thread_id: str,
    trace_id: str,
    payload: dict[str, object],
    order_id: str | None = None,
) -> dict[str, object]:
    return get_review_store().create_pending_task(
        thread_id=thread_id,
        trace_id=trace_id,
        payload=payload,
        order_id=order_id,
    )


def resolve_pending_task(
    *,
    thread_id: str,
    status: str,
    note: str | None = None,
    operator_id: str | None = None,
) -> None:
    get_review_store().resolve_pending_task(
        thread_id=thread_id,
        status=status,
        note=note,
        operator_id=operator_id,
    )


def create_ticket(
    *,
    thread_id: str,
    user_id: str,
    payload: dict[str, object],
) -> dict[str, object]:
    return get_review_store().create_ticket(
        thread_id=thread_id,
        user_id=user_id,
        payload=payload,
    )


def list_records(
    *,
    record_type: str | None = None,
    thread_id: str | None = None,
) -> list[dict[str, object]]:
    return get_review_store().list_records(
        record_type=record_type,
        thread_id=thread_id,
    )
