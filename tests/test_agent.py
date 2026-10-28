from collections.abc import Callable

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_graph
from app.nodes import retrieval as retrieval_module
from app.tools import order_client as order_client_module
from app.tools.order_client import OrderServiceClient
from app.tools.product import ProductEvidence

_CONFIG = {"configurable": {"thread_id": "react-thread"}}


def _initial_state(query: str) -> dict[str, object]:
    return {
        "trace_id": "trace-react",
        "thread_id": "react-thread",
        "user_id": "user-001",
        "raw_query": query,
        "messages": [HumanMessage(content=query)],
    }


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OrderServiceClient:
    transport = httpx.MockTransport(handler)
    return OrderServiceClient(
        httpx.Client(transport=transport, base_url="http://order-service")
    )


def test_greeting_does_not_call_tools():
    graph = build_graph(InMemorySaver())
    result = graph.invoke(_initial_state("你好"), config=_CONFIG)

    assert result["intent"] == "other"
    assert "agent_steps" not in result
    assert "tool_results" not in result


def test_order_question_calls_tool_once_and_answers(
    install_fake_agent,
    monkeypatch,
):
    install_fake_agent(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "list_orders", "args": {}, "id": "call-1"}],
            ),
            AIMessage(content="您有两笔订单:ORD-001 已发货,金额 299.00 元。"),
        ]
    )

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
                {"order_id": "ORD-002", "status": "paid", "amount": "129.50"},
            ],
        )

    monkeypatch.setattr(
        order_client_module,
        "get_order_service_client",
        lambda: _make_client(mock_order_service),
    )

    graph = build_graph(InMemorySaver())
    result = graph.invoke(_initial_state("查一下我的订单"), config=_CONFIG)

    assert result["intent"] == "order_query"
    assert result["agent_steps"] == 2
    assert [item["order_id"] for item in result["tool_results"]] == [
        "ORD-001",
        "ORD-002",
    ]
    assert "ORD-001" in result["final_answer"]
    assert "299.00" in result["final_answer"]
    assert "phone" not in str(result["tool_results"])


def test_invalid_tool_name_is_rejected(install_fake_agent, monkeypatch):
    called = []

    def fake_factory() -> OrderServiceClient:
        called.append(True)
        raise AssertionError("order service must not be called")

    monkeypatch.setattr(
        order_client_module,
        "get_order_service_client",
        fake_factory,
    )
    install_fake_agent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "execute_refund", "args": {}, "id": "call-1"}
                ],
            ),
            AIMessage(content="抱歉,该操作无法完成。"),
        ]
    )

    graph = build_graph(InMemorySaver())
    result = graph.invoke(_initial_state("查一下我的订单"), config=_CONFIG)

    assert result["error_type"] == "UnknownToolError"
    assert not called
    assert result["final_answer"] == "抱歉,该操作无法完成。"
    assert "退款成功" not in result["final_answer"]


def test_step_limit_stops_tool_loop(install_fake_agent, monkeypatch):
    calls = []

    def mock_order_service(request: httpx.Request) -> httpx.Response:
        calls.append(True)
        return httpx.Response(200, json=[])

    monkeypatch.setattr(
        order_client_module,
        "get_order_service_client",
        lambda: _make_client(mock_order_service),
    )
    install_fake_agent(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "list_orders", "args": {}, "id": "call-1"}],
            ),
        ]
    )

    graph = build_graph(InMemorySaver())
    result = graph.invoke(_initial_state("查一下我的订单"), config=_CONFIG)

    assert result["agent_steps"] == 3
    assert len(calls) == 2
    assert "无法完成" in result["final_answer"]


def test_product_search_answers_from_tool_evidence(
    install_fake_agent,
    monkeypatch,
):
    evidence_calls: list[str] = []

    def fake_evidence(
        query: str,
        *,
        top_k: int = 5,
        category: str | None = None,
        color: str | None = None,
        price_min: object = None,
        price_max: object = None,
        size: str | None = None,
    ) -> list[ProductEvidence]:
        evidence_calls.append(query)
        return [
            ProductEvidence(
                product_id="SKU-001",
                title="黑色通勤连衣裙",
                category="连衣裙",
                color="黑色",
                sizes=["M", "L"],
                price="399.00",
                style="通勤",
                description="修身 面试",
            )
        ]

    monkeypatch.setattr(
        retrieval_module,
        "search_product_evidence",
        fake_evidence,
    )
    install_fake_agent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_product",
                        "args": {"query": "黑色连衣裙"},
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage(content="为您找到黑色通勤连衣裙,价格 399.00 元,有 M 码。"),
        ]
    )

    graph = build_graph(InMemorySaver())
    result = graph.invoke(_initial_state("有没有黑色连衣裙"), config=_CONFIG)

    assert result["intent"] == "product_search"
    assert evidence_calls == ["黑色连衣裙"]
    assert result["tool_results"][0]["product_id"] == "SKU-001"
    assert result["tool_results"][0]["price"] == "399.00"
    assert "399.00" in result["final_answer"]
    assert "未找到" not in result["final_answer"]


def test_tool_error_is_returned_to_agent(install_fake_agent, monkeypatch):
    def mock_order_service(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    monkeypatch.setattr(
        order_client_module,
        "get_order_service_client",
        lambda: _make_client(mock_order_service),
    )
    install_fake_agent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_order_detail",
                        "args": {"order_id": "ORD-001"},
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage(content="订单查询服务暂时不可用,请稍后再试。"),
        ]
    )

    graph = build_graph(InMemorySaver())
    result = graph.invoke(
        _initial_state("我的订单 ORD-001 到哪了"),
        config=_CONFIG,
    )

    assert result["error_type"] == "HTTPStatusError"
    assert "暂时不可用" in result["final_answer"]
