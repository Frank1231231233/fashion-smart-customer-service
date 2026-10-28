import pytest
from fastapi.testclient import TestClient

from app.auth import UserStore
from app.chat import store as chat_store_module
from app.graph import checkpoint as checkpoint_module
from app.main import app


@pytest.fixture
def isolated_stores(monkeypatch, tmp_path):
    monkeypatch.setattr(
        checkpoint_module,
        "CHECKPOINT_DB_PATH",
        tmp_path / "checkpoints.db",
    )
    checkpoint_module.get_checkpointer.cache_clear()
    monkeypatch.setattr(
        chat_store_module,
        "CHAT_DB_PATH",
        tmp_path / "chats.db",
    )
    chat_store_module.get_chat_store.cache_clear()
    yield
    checkpoint_module.get_checkpointer.cache_clear()
    chat_store_module.get_chat_store.cache_clear()


def test_user_store_assigns_sequential_ids_and_authenticates(tmp_path):
    store = UserStore(tmp_path / "users.db")

    first = store.create_user("alice", "secret123")
    second = store.create_user("bob", "secret456", role="operator")

    assert first["user_id"] == "user-001"
    assert first["role"] == "customer"
    assert second["user_id"] == "user-002"
    assert second["role"] == "operator"

    assert store.authenticate("alice", "secret123")["user_id"] == "user-001"
    assert store.authenticate("alice", "wrong-password") is None
    assert store.authenticate("nobody", "secret123") is None

    with pytest.raises(ValueError, match="username already exists"):
        store.create_user("alice", "another-secret")


def test_user_store_token_roundtrip(tmp_path):
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("alice", "secret123")

    token = store.create_token(user["user_id"])

    resolved = store.get_user_by_token(token)
    assert resolved is not None
    assert resolved["user_id"] == "user-001"
    assert store.get_user_by_token("no-such-token") is None


def test_password_hash_is_not_plaintext(tmp_path):
    store = UserStore(tmp_path / "users.db")
    store.create_user("alice", "secret123")

    with store._connect() as connection:
        row = connection.execute(
            "SELECT password_hash, salt FROM users WHERE username = 'alice'"
        ).fetchone()

    assert "secret123" not in row[0]
    assert row[1]  # salt 存在


def test_register_login_and_chat_roundtrip(isolated_stores):
    client = TestClient(app)

    register = client.post(
        "/auth/register",
        json={"username": "alice", "password": "secret123"},
    )
    assert register.status_code == 201
    assert register.json()["user"]["user_id"] == "user-001"
    assert register.json()["user"]["role"] == "customer"

    login = client.post(
        "/auth/login",
        json={"username": "alice", "password": "secret123"},
    )
    assert login.status_code == 200
    token = login.json()["token"]

    chat = client.post(
        "/chat",
        json={"message": "你好"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert chat.status_code == 200


def test_register_duplicate_username_returns_409():
    client = TestClient(app)
    payload = {"username": "alice", "password": "secret123"}
    assert client.post("/auth/register", json=payload).status_code == 201

    duplicate = client.post(
        "/auth/register",
        json={"username": "alice", "password": "other-secret"},
    )

    assert duplicate.status_code == 409


def test_register_rejects_short_password():
    client = TestClient(app)
    response = client.post(
        "/auth/register",
        json={"username": "alice", "password": "123"},
    )

    assert response.status_code == 422


def test_login_wrong_password_returns_401():
    client = TestClient(app)
    client.post("/auth/register", json={"username": "alice", "password": "secret123"})

    response = client.post(
        "/auth/login",
        json={"username": "alice", "password": "wrong-password"},
    )

    assert response.status_code == 401


def test_chat_rejects_missing_or_invalid_token():
    client = TestClient(app)

    missing = client.post("/chat", json={"message": "你好"})
    assert missing.status_code == 401

    invalid = client.post(
        "/chat",
        json={"message": "你好"},
        headers={"Authorization": "Bearer no-such-token"},
    )
    assert invalid.status_code == 401


def test_register_operator_role():
    client = TestClient(app)
    response = client.post(
        "/auth/register",
        json={"username": "op", "password": "secret123", "role": "operator"},
    )

    assert response.status_code == 201
    assert response.json()["user"]["role"] == "operator"
