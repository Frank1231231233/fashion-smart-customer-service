import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app.chat import store as chat_store_module
from app.chat.store import ChatStore
from app.graph import checkpoint as checkpoint_module
from app.graph.builder import build_graph
from app.main import app
from app.tools import order_client as order_client_module
from app.tools.order_client import OrderServiceClient


def _make_order_client(handler) -> OrderServiceClient:
    return OrderServiceClient(
        httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://order-service",
        )
    )


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


def test_chat_store_create_get_and_list_by_user(tmp_path):
    store = ChatStore(tmp_path / "chats.db")
    store.create_session("user-001", "t-1", "第一个会话")
    store.create_session("user-001", "t-2", "第二个会话")
    store.create_session("user-002", "t-3", "别人的会话")

    session = store.get_session("t-1")
    assert session is not None
    assert session["user_id"] == "user-001"
    assert session["title"] == "第一个会话"
    assert store.get_session("no-such") is None

    mine = store.list_sessions("user-001")
    assert {item["thread_id"] for item in mine} == {"t-1", "t-2"}
    assert [item["thread_id"] for item in store.list_sessions("user-002")] == ["t-3"]


def test_chat_requires_user_header(isolated_stores):
    response = TestClient(app).post("/chat", json={"message": "你好"})

    assert response.status_code == 401


def test_new_chat_registers_session(isolated_stores, auth_headers):
    client = TestClient(app)
    headers = auth_headers(client, "alice")
    body = client.post("/chat", json={"message": "你好"}, headers=headers).json()

    session = chat_store_module.get_chat_store().get_session(body["thread_id"])
    assert session is not None
    assert session["user_id"] == "user-001"
    assert session["title"] == "你好"


def test_multi_turn_restores_history_from_checkpointer(isolated_stores, auth_headers):
    client = TestClient(app)
    headers = auth_headers(client, "alice")
    first = client.post("/chat", json={"message": "你好"}, headers=headers).json()

    second = client.post(
        "/chat",
        json={"message": "在吗", "thread_id": first["thread_id"]},
        headers=headers,
    ).json()

    assert second["thread_id"] == first["thread_id"]

    graph = build_graph(checkpoint_module.get_checkpointer())
    state = graph.get_state({"configurable": {"thread_id": first["thread_id"]}})
    human_contents = [
        message.content
        for message in state.values["messages"]
        if isinstance(message, HumanMessage)
    ]
    assert human_contents == ["你好", "在吗"]


def test_continue_chat_rejects_foreign_thread(isolated_stores, auth_headers):
    client = TestClient(app)
    headers = auth_headers(client, "alice")
    other_headers = auth_headers(client, "bob")
    first = client.post("/chat", json={"message": "你好"}, headers=headers).json()

    response = client.post(
        "/chat",
        json={"message": "在吗", "thread_id": first["thread_id"]},
        headers=other_headers,
    )

    assert response.status_code == 404


def test_continue_unknown_thread_returns_404(isolated_stores, auth_headers):
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"message": "在吗", "thread_id": "no-such-thread"},
        headers=auth_headers(client, "alice"),
    )

    assert response.status_code == 404


def test_list_chats_returns_own_sessions_ordered_by_recent(
    isolated_stores, auth_headers
):
    client = TestClient(app)
    headers = auth_headers(client, "alice")
    other_headers = auth_headers(client, "bob")
    first = client.post("/chat", json={"message": "会话一"}, headers=headers).json()
    client.post("/chat", json={"message": "会话二"}, headers=headers)
    client.post("/chat", json={"message": "别人"}, headers=other_headers)

    client.post(
        "/chat",
        json={"message": "继续", "thread_id": first["thread_id"]},
        headers=headers,
    )

    body = client.get("/chats", headers=headers).json()
    titles = [item["title"] for item in body["chats"]]
    assert titles == ["会话一", "会话二"]

    other = client.get("/chats", headers=other_headers).json()
    assert [item["title"] for item in other["chats"]] == ["别人"]


def test_list_chats_requires_user_header(isolated_stores):
    response = TestClient(app).get("/chats")

    assert response.status_code == 401


def test_get_chat_messages_returns_history(isolated_stores, auth_headers):
    client = TestClient(app)
    headers = auth_headers(client, "alice")
    first = client.post(
        "/chat",
        json={"message": "查一下我的订单"},
        headers=headers,
    ).json()

    response = client.get(
        f"/chats/{first['thread_id']}/messages",
        headers=headers,
    )

    assert response.status_code == 200
    messages = response.json()["messages"]
    assert messages[0] == {"role": "user", "content": "查一下我的订单"}
    assert messages[-1]["role"] == "assistant"


def test_get_chat_messages_rejects_foreign_thread(isolated_stores, auth_headers):
    client = TestClient(app)
    headers = auth_headers(client, "alice")
    other_headers = auth_headers(client, "bob")
    first = client.post("/chat", json={"message": "你好"}, headers=headers).json()

    response = client.get(
        f"/chats/{first['thread_id']}/messages",
        headers=other_headers,
    )

    assert response.status_code == 404


def test_agent_steps_reset_between_turns(
    isolated_stores,
    install_fake_agent,
    auth_headers,
    monkeypatch,
):
    def mock_order_service(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"order_id": "ORD-001", "status": "shipped", "amount": "299.00"},
        )

    monkeypatch.setattr(
        order_client_module,
        "get_order_service_client",
        lambda: _make_order_client(mock_order_service),
    )
    install_fake_agent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_order_detail", "args": {"order_id": "ORD-001"}, "id": "c1"}
                ],
            ),
            AIMessage(content="订单已发货,299 元"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_order_detail", "args": {"order_id": "ORD-001"}, "id": "c2"}
                ],
            ),
            AIMessage(content="还是 299 元"),
        ]
    )

    client = TestClient(app)
    headers = auth_headers(client, "alice")
    first = client.post(
        "/chat",
        json={"message": "订单 ORD-001 到哪了"},
        headers=headers,
    ).json()
    assert first["final_answer"] == "订单已发货,299 元"

    second = client.post(
        "/chat",
        json={"message": "这个订单多少钱", "thread_id": first["thread_id"]},
        headers=headers,
    ).json()

    assert second["final_answer"] == "还是 299 元"
    assert "无法完成" not in second["final_answer"]
