"""用户体系:SQLite 存储用户与访问令牌,提供注册/登录/认证能力。

安全纪律:user_id 只允许来自服务端 token 解析,禁止从请求参数中读取。
"""

import hashlib
import hmac
import secrets
import sqlite3
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from fastapi import Depends, Header, HTTPException

USERS_DB_PATH = Path("data/users.db")

PBKDF2_ITERATIONS = 100_000

Role = Literal["customer", "operator"]


class User(TypedDict):
    user_id: str
    username: str
    role: Role
    created_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PBKDF2_ITERATIONS,
    )
    return digest.hex()


class UserStore:
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
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'customer',
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tokens (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def create_user(self, username: str, password: str, role: Role = "customer") -> User:
        salt = secrets.token_hex(16)
        password_hash = _hash_password(password, salt)
        now = _now()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT 1 FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if existing is not None:
                raise ValueError("username already exists")
            row = connection.execute("SELECT COUNT(*) FROM users").fetchone()
            user_id = f"user-{row[0] + 1:03d}"
            connection.execute(
                """
                INSERT INTO users (user_id, username, password_hash, salt, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, username, password_hash, salt, role, now),
            )
        return User(user_id=user_id, username=username, role=role, created_at=now)

    def authenticate(self, username: str, password: str) -> User | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, username, password_hash, salt, role, created_at
                FROM users
                WHERE username = ?
                """,
                (username,),
            ).fetchone()
        if row is None:
            return None
        if not hmac.compare_digest(row[2], _hash_password(password, row[3])):
            return None
        return User(user_id=row[0], username=row[1], role=row[4], created_at=row[5])

    def create_token(self, user_id: str) -> str:
        token = secrets.token_hex(16)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tokens (token, user_id, created_at)
                VALUES (?, ?, ?)
                """,
                (token, user_id, _now()),
            )
        return token

    def get_user_by_token(self, token: str) -> User | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT u.user_id, u.username, u.role, u.created_at
                FROM tokens t
                JOIN users u ON u.user_id = t.user_id
                WHERE t.token = ?
                """,
                (token,),
            ).fetchone()
        if row is None:
            return None
        return User(user_id=row[0], username=row[1], role=row[2], created_at=row[3])


@lru_cache
def get_user_store() -> UserStore:
    return UserStore(USERS_DB_PATH)


def require_user(authorization: str | None = Header(default=None)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="missing bearer token")
    user = get_user_store().get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid token")
    return user


def require_operator(user: Annotated[User, Depends(require_user)]) -> User:
    if user["role"] != "operator":
        raise HTTPException(status_code=403, detail="operator role required")
    return user
