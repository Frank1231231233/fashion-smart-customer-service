from collections.abc import Callable

import httpx
from langchain_core.messages import AIMessage

from app.graph.state import AgentState
from app.nodes.tool import tool_node
from app.tools import order_client as order_client_module
from app.tools.order_client import OrderServiceClient
from app.trace import service as trace_service
from app.trace.wrapper import traced_node


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OrderServiceClient:
    transport = httpx.MockTransport(handler)
    return OrderServiceClient(
        httpx.Client(transport=transport, base_url="http://order-service")
    )


def _state(tool_calls: list[dict], **overrides: object) -> AgentState:
    state: dict[str, object] = {
        "trace_id": "trace-test",
        "thread_id": "thread-test",
        "user_id": "user-001",
        "raw_query": "查一下我的订单",
        "messages": [AIMessage(content="", tool_calls=tool_calls)],
        "intent": "order_query",
    }
    state.update(overrides)
    return state  # type: ignore[return-value]


def _forbid_factory_call(monkeypatch) -> list[bool]:
    called = []

    def fake_factory() -> OrderServiceClient:
        called.append(True)
        raise AssertionError("order service must not be called")

    monkeypatch.setattr(
        order_client_module,
        "get_order_service_client",
        fake_factory,
    )
    return called


def test_tool_node_rejects_unknown_tool(monkeypatch):
    called = _forbid_factory_call(monkeypatch)

    result = tool_node(
        _state([{"name": "execute_refund", "args": {}, "id": "call-1"}])
    )

    assert result["error_type"] == "UnknownToolError"
    assert "execute_refund" in result["error"]
    assert result["tool_name"] == "execute_refund"
    assert "whitelist" in result["messages"][-1].content
    assert not called


def test_tool_node_get_order_detail_success(monkeypatch):
    def mock_order_service(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/orders/ORD-001"
        assert request.headers["X-User-Id"] == "user-001"
        return httpx.Response(
            200,
            json={
                "order_id": "ORD-001",
                "status": "shipped",
                "amount": "299.00",
                "phone": "13800000000",
                "address": "测试地址",
            },
        )

    monkeypatch.setattr(
        order_client_module,
        "get_order_service_client",
        lambda: _make_client(mock_order_service),
    )

    result = tool_node(
        _state([
            {
                "name": "get_order_detail",
                "args": {"order_id": "ORD-001"},
                "id": "call-1",
            }
        ])
    )

    assert result["tool_name"] == "get_order_detail"
    assert result["tool_results"] == [
        {"order_id": "ORD-001", "status": "shipped", "amount": "299.00"}
    ]
    assert "ORD-001" in result["messages"][-1].content
    assert "phone" not in str(result["tool_results"])


def test_tool_node_rejects_forged_user_id(monkeypatch):
    called = _forbid_factory_call(monkeypatch)

    result = tool_node(
        _state([
            {
                "name": "get_order_detail",
                "args": {"order_id": "ORD-001", "user_id": "user-002"},
                "id": "call-1",
            }
        ])
    )

    assert result["error_type"] == "ValidationError"
    assert result["tool_name"] == "get_order_detail"
    assert not called


def test_tool_node_order_not_found(monkeypatch):
    def mock_order_service(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    monkeypatch.setattr(
        order_client_module,
        "get_order_service_client",
        lambda: _make_client(mock_order_service),
    )

    result = tool_node(
        _state([
            {
                "name": "get_order_detail",
                "args": {"order_id": "ORD-404"},
                "id": "call-1",
            }
        ])
    )

    assert result["error_type"] == "OrderNotFoundError"
    assert "ORD-404" in result["error"]


def test_tool_node_http_error(monkeypatch):
    def mock_order_service(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    monkeypatch.setattr(
        order_client_module,
        "get_order_service_client",
        lambda: _make_client(mock_order_service),
    )

    result = tool_node(
        _state([
            {
                "name": "get_order_detail",
                "args": {"order_id": "ORD-001"},
                "id": "call-1",
            }
        ])
    )

    assert result["error_type"] == "HTTPStatusError"
    assert result["tool_name"] == "get_order_detail"
    assert "error" in result


def test_tool_node_list_orders(monkeypatch):
    def mock_order_service(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/orders"
        assert request.headers["X-User-Id"] == "user-001"
        return httpx.Response(
            200,
            json=[
                {
                    "order_id": "ORD-001",
                    "status": "shipped",
                    "amount": "299.00",
                    "phone": "13800000000",
                },
                {
                    "order_id": "ORD-002",
                    "status": "paid",
                    "amount": "129.50",
                },
            ],
        )

    monkeypatch.setattr(
        order_client_module,
        "get_order_service_client",
        lambda: _make_client(mock_order_service),
    )

    result = tool_node(
        _state([{"name": "list_orders", "args": {}, "id": "call-1"}])
    )

    assert result["tool_name"] == "list_orders"
    assert [item["order_id"] for item in result["tool_results"]] == [
        "ORD-001",
        "ORD-002",
    ]


def test_tool_node_requires_agent_tool_calls():
    result = tool_node(
        _state([], messages=[AIMessage(content="直接回答")])
    )

    assert result["error_type"] == "InvalidToolCallError"


def test_tool_node_trace_records_tool_name(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_service, "TRACE_FILE", tmp_path / "trace.jsonl")

    def mock_order_service(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"order_id": "ORD-001", "status": "shipped", "amount": "299.00"},
        )

    monkeypatch.setattr(
        order_client_module,
        "get_order_service_client",
        lambda: _make_client(mock_order_service),
    )

    traced_node("tool_node", tool_node)(
        _state([
            {
                "name": "get_order_detail",
                "args": {"order_id": "ORD-001"},
                "id": "call-1",
            }
        ])
    )

    events = trace_service.query_trace_events("trace-test")
    assert [event["event"] for event in events] == ["start", "end"]
    assert events[-1]["tool_name"] == "get_order_detail"
    assert events[-1]["error_type"] is None


def test_tool_node_trace_records_error_event(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_service, "TRACE_FILE", tmp_path / "trace.jsonl")
    _forbid_factory_call(monkeypatch)

    traced_node("tool_node", tool_node)(
        _state([{"name": "execute_refund", "args": {}, "id": "call-1"}])
    )

    events = trace_service.query_trace_events("trace-test")
    assert [event["event"] for event in events] == ["start", "error"]
    assert events[-1]["tool_name"] == "execute_refund"
    assert events[-1]["error_type"] == "UnknownToolError"
