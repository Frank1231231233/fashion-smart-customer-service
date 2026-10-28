import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_graph
from app.nodes import retrieval as retrieval_module


@pytest.fixture(autouse=True)
def stub_product_documents(monkeypatch):
    monkeypatch.setattr(
        retrieval_module,
        "load_product_documents",
        list,
    )


def _invoke_graph(query: str):
    graph = build_graph(InMemorySaver())
    return graph.invoke(
        {
            "trace_id": "123",
            "thread_id": "456",
            "user_id": "789",
            "raw_query": query,
            "messages": [],
        },
        config={"configurable": {"thread_id": "456"}},
    )


def test_graph_entry_to_end():
    result = _invoke_graph("你好")

    assert result["trace_id"] == "123"
    assert result["intent"] == "other"


def test_product_search_routes_through_rewrite_node():
    result = _invoke_graph("有没有黑色连衣裙")

    assert result["intent"] == "product_search"
    assert result["rewritten_query"] == "有没有黑色连衣裙"
    assert result["final_answer"] == "测试回复"
    assert "retrieval_results" not in result


@pytest.mark.parametrize(
    ("query", "expected_intent"),
    [
        ("查一下我的订单", "order_query"),
        ("我要退款", "refund_request"),
        ("你好", "other"),
    ],
)
def test_non_product_intents_skip_rewrite_and_retrieval(
    query,
    expected_intent,
):
    result = _invoke_graph(query)

    assert result["intent"] == expected_intent
    assert "rewritten_query" not in result
    assert "retrieval_results" not in result
