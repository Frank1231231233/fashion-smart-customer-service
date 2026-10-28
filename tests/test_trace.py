from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.chat import store as chat_store_module
from app.main import app
from app.trace import service as trace_service


@pytest.fixture(autouse=True)
def isolate_chat_store(monkeypatch, tmp_path):
    monkeypatch.setattr(
        chat_store_module,
        "CHAT_DB_PATH",
        tmp_path / "chats.db",
    )
    chat_store_module.get_chat_store.cache_clear()
    yield
    chat_store_module.get_chat_store.cache_clear()


def test_chat_trace_can_be_queried(monkeypatch, tmp_path, auth_headers):
    monkeypatch.setattr(trace_service, "TRACE_FILE", tmp_path / "trace.jsonl")
    client = TestClient(app)
    headers = auth_headers(client, "alice")

    chat_response = client.post("/chat", json={"message": "你好"}, headers=headers)
    trace_id = chat_response.json()["trace_id"]
    trace_response = client.get(f"/traces/{trace_id}")

    assert chat_response.status_code == 200
    assert trace_response.status_code == 200
    events = trace_response.json()["events"]
    assert [event["event"] for event in events] == ["start", "end"]
    assert all(event["trace_id"] == trace_id for event in events)


def test_concurrent_requests_do_not_mix_traces(monkeypatch, tmp_path, auth_headers):
    monkeypatch.setattr(trace_service, "TRACE_FILE", tmp_path / "trace.jsonl")
    headers = auth_headers(TestClient(app), "alice")

    def send_chat(message: str) -> dict[str, object]:
        response = TestClient(app).post(
            "/chat",
            json={"message": message},
            headers=headers,
        )
        assert response.status_code == 200
        return response.json()

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(send_chat, ["请求一", "请求二"]))

    trace_ids = [response["trace_id"] for response in responses]
    assert len(set(trace_ids)) == 2

    for trace_id in trace_ids:
        events = trace_service.query_trace_events(trace_id)
        assert [event["event"] for event in events] == ["start", "end"]
        assert all(event["trace_id"] == trace_id for event in events)
