import sqlite3

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.graph import checkpoint as checkpoint_module
from app.graph.builder import build_graph
from app.main import app
from app.nodes.human_review import mask_sensitive
from app.review import store as review_store
from app.tools.registry import TOOL_REGISTRY
from app.trace import service as trace_service

_CONFIG = {"configurable": {"thread_id": "thread-refund"}}


def _initial_state(query: str) -> dict[str, object]:
    return {
        "trace_id": "trace-refund",
        "thread_id": "thread-refund",
        "user_id": "user-001",
        "raw_query": query,
        "messages": [],
    }


def _build_paused_graph(query: str = "我要退款,手机 13800000000 别泄露"):
    graph = build_graph(InMemorySaver())
    result = graph.invoke(_initial_state(query), config=_CONFIG)
    return graph, result


def test_refund_pauses_with_masked_payload():
    _, result = _build_paused_graph()

    assert result["intent"] == "refund_request"
    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    payload = interrupts[0].value
    assert payload["user_id"] == "user-001"
    assert "13800000000" not in str(payload)
    assert "***" in payload["reason_summary"]
    assert "final_answer" not in result


def test_resume_accepted_creates_ticket_without_refund(monkeypatch, tmp_path):
    monkeypatch.setattr(
        review_store,
        "REVIEW_DB_PATH",
        tmp_path / "reviews.db",
    )
    review_store.get_review_store.cache_clear()
    graph, _ = _build_paused_graph()

    result = graph.invoke(
        Command(
            resume={"status": "accepted_for_manual_processing", "note": None}
        ),
        config=_CONFIG,
    )

    assert result["review_status"] == "accepted_for_manual_processing"
    final_answer = result["final_answer"]
    assert "已转人工处理" in final_answer
    assert "退款成功" not in final_answer

    tickets = review_store.list_records(record_type="ticket")
    assert len(tickets) == 1
    assert tickets[0]["user_id"] == "user-001"
    assert tickets[0]["ticket_id"] in final_answer
    assert "execute_refund" not in TOOL_REGISTRY


def test_resume_rejected_returns_human_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(
        review_store,
        "REVIEW_DB_PATH",
        tmp_path / "reviews.db",
    )
    review_store.get_review_store.cache_clear()
    graph, _ = _build_paused_graph()

    result = graph.invoke(
        Command(resume={"status": "rejected", "note": "缺少退款凭证"}),
        config=_CONFIG,
    )

    assert result["review_status"] == "rejected"
    assert "缺少退款凭证" in result["final_answer"]
    assert review_store.list_records(record_type="ticket") == []


def test_resume_need_more_info_returns_human_note(monkeypatch, tmp_path):
    monkeypatch.setattr(
        review_store,
        "REVIEW_DB_PATH",
        tmp_path / "reviews.db",
    )
    review_store.get_review_store.cache_clear()
    graph, _ = _build_paused_graph()

    result = graph.invoke(
        Command(resume={"status": "need_more_info", "note": "请提供订单截图"}),
        config=_CONFIG,
    )

    assert result["review_status"] == "need_more_info"
    assert "请提供订单截图" in result["final_answer"]
    assert review_store.list_records(record_type="ticket") == []


def test_resume_invalid_decision_returns_error():
    graph, _ = _build_paused_graph()

    result = graph.invoke(
        Command(resume={"status": "refunded"}),
        config=_CONFIG,
    )

    assert result["error_type"] == "ValidationError"
    assert "final_answer" not in result


def test_restart_recovers_paused_thread(tmp_path):
    db_path = tmp_path / "checkpoints.db"

    first_connection = sqlite3.connect(db_path, check_same_thread=False)
    first_saver = SqliteSaver(first_connection)
    first_saver.setup()
    build_graph(first_saver).invoke(
        _initial_state("我要退款"),
        config=_CONFIG,
    )
    first_connection.close()

    second_connection = sqlite3.connect(db_path, check_same_thread=False)
    second_saver = SqliteSaver(second_connection)
    second_saver.setup()
    second_graph = build_graph(second_saver)
    snapshot = second_graph.get_state(_CONFIG)
    assert snapshot.next
    pending_interrupts = [
        interrupt
        for task in snapshot.tasks
        for interrupt in task.interrupts
    ]
    assert len(pending_interrupts) == 1

    result = second_graph.invoke(
        Command(resume={"status": "rejected", "note": "凭证不足"}),
        config=_CONFIG,
    )
    second_connection.close()

    assert result["review_status"] == "rejected"


def test_interrupt_event_recorded_in_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_service, "TRACE_FILE", tmp_path / "trace.jsonl")

    _build_paused_graph()

    events = trace_service.query_trace_events("trace-refund")
    assert [(event["node"], event["event"]) for event in events] == [
        ("intent_node", "start"),
        ("intent_node", "end"),
        ("human_review_node", "start"),
        ("human_review_node", "interrupt"),
    ]
    assert events[-1]["intent"] == "refund_request"


def test_mask_sensitive_hides_phone_and_long_digits():
    masked = mask_sensitive("手机 13800000000,卡号 6222020200112233,单号 ORD-001")

    assert "13800000000" not in masked
    assert "6222020200112233" not in masked
    assert "ORD-001" in masked


@pytest.fixture
def isolated_stores(monkeypatch, tmp_path):
    monkeypatch.setattr(
        checkpoint_module,
        "CHECKPOINT_DB_PATH",
        tmp_path / "checkpoints.db",
    )
    checkpoint_module.get_checkpointer.cache_clear()
    monkeypatch.setattr(
        review_store,
        "REVIEW_DB_PATH",
        tmp_path / "reviews.db",
    )
    review_store.get_review_store.cache_clear()
    yield
    checkpoint_module.get_checkpointer.cache_clear()
    review_store.get_review_store.cache_clear()


def test_chat_refund_registers_pending_task(isolated_stores, auth_headers):
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"message": "我要退款"},
        headers=auth_headers(client, "alice"),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "refund_request"
    assert body["interrupted"] is True
    assert body["final_answer"] is None

    pending = review_store.list_records(record_type="pending")
    assert len(pending) == 1
    assert pending[0]["thread_id"] == body["thread_id"]


def test_resume_endpoint_accepted_flow(isolated_stores, auth_headers):
    client = TestClient(app)
    chat_body = client.post(
        "/chat",
        json={"message": "我要退款"},
        headers=auth_headers(client, "alice"),
    ).json()
    thread_id = chat_body["thread_id"]
    operator_headers = auth_headers(client, "op", role="operator")

    response = client.post(
        "/reviews/resume",
        json={
            "thread_id": thread_id,
            "status": "accepted_for_manual_processing",
        },
        headers=operator_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["review_status"] == "accepted_for_manual_processing"
    assert "已转人工处理" in body["final_answer"]

    tickets = review_store.list_records(record_type="ticket")
    assert len(tickets) == 1
    resolutions = review_store.list_records(record_type="resolution")
    assert resolutions[-1]["thread_id"] == thread_id


def test_resume_unknown_thread_returns_404(isolated_stores, auth_headers):
    client = TestClient(app)
    response = client.post(
        "/reviews/resume",
        json={"thread_id": "no-such-thread", "status": "rejected"},
        headers=auth_headers(client, "op", role="operator"),
    )

    assert response.status_code == 404


def test_pending_list_requires_operator(isolated_stores, auth_headers):
    client = TestClient(app)
    customer_headers = auth_headers(client, "alice")

    assert client.get("/reviews/pending").status_code == 401
    assert (
        client.get("/reviews/pending", headers=customer_headers).status_code == 403
    )

    operator_headers = auth_headers(client, "op", role="operator")
    response = client.get("/reviews/pending", headers=operator_headers)
    assert response.status_code == 200
    assert response.json()["pending"] == []


def test_resume_requires_operator(isolated_stores, auth_headers):
    client = TestClient(app)
    chat_body = client.post(
        "/chat",
        json={"message": "我要退款"},
        headers=auth_headers(client, "alice"),
    ).json()

    payload = {"thread_id": chat_body["thread_id"], "status": "rejected"}
    assert client.post("/reviews/resume", json=payload).status_code == 401

    customer_headers = auth_headers(client, "bob")
    assert (
        client.post(
            "/reviews/resume",
            json=payload,
            headers=customer_headers,
        ).status_code
        == 403
    )


def test_resume_writes_operator_id_and_trace_event(
    isolated_stores,
    auth_headers,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(trace_service, "TRACE_FILE", tmp_path / "trace.jsonl")
    client = TestClient(app)
    chat_body = client.post(
        "/chat",
        json={"message": "我要退款"},
        headers=auth_headers(client, "alice"),
    ).json()
    operator_headers = auth_headers(client, "op", role="operator")

    response = client.post(
        "/reviews/resume",
        json={
            "thread_id": chat_body["thread_id"],
            "status": "accepted_for_manual_processing",
        },
        headers=operator_headers,
    )
    assert response.status_code == 200

    resolutions = review_store.list_records(record_type="resolution")
    assert len(resolutions) == 1
    assert resolutions[0]["operator_id"] == "user-002"
    assert resolutions[0]["processed_at"] is not None

    resume_events = [
        event
        for event in trace_service.query_trace_events(chat_body["trace_id"])
        if event["event"] == "resume"
    ]
    assert len(resume_events) == 1
    assert resume_events[0]["operator_id"] == "user-002"
    assert resume_events[0]["status"] == "accepted_for_manual_processing"
