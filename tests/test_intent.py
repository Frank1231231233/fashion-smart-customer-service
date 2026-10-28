import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.state import AgentState
from app.nodes import intent as intent_module
from app.nodes.intent import IntentDecision, intent_node


@pytest.mark.parametrize(
    ("query", "expected_intent"),
    [
        ("有没有黑色连衣裙", "product_search"),
        ("查一下我上周的订单", "order_query"),
        ("这件不合适，我要退款", "refund_request"),
        ("你们几点下班", "other"),
        ("有 M 码吗", "product_search"),
    ],
)
def test_intent_node_l1_rules(query, expected_intent):
    state: AgentState = {
        "trace_id": "trace-test",
        "thread_id": "thread-test",
        "user_id": "user-test",
        "raw_query": query,
        "messages": [],
    }

    result = intent_node(state)

    assert result["intent"] == expected_intent


def test_refund_rule_has_priority_over_order_rule():
    state: AgentState = {
        "trace_id": "trace-test",
        "thread_id": "thread-test",
        "user_id": "user-test",
        "raw_query": "查询订单后申请退款",
        "messages": [],
    }

    result = intent_node(state)

    assert result["intent"] == "refund_request"


def test_l1_hit_does_not_call_l2(monkeypatch):
    def fail_if_called(query, messages):
        raise AssertionError("L1 命中时不应调用 L2")

    monkeypatch.setattr(intent_module, "classify_intent_l2", fail_if_called)
    state: AgentState = {
        "trace_id": "trace-test",
        "thread_id": "thread-test",
        "user_id": "user-test",
        "raw_query": "我要退款",
        "messages": [],
    }

    result = intent_node(state)

    assert result["intent"] == "refund_request"


def test_l1_miss_uses_l2(monkeypatch):
    calls: list[str] = []

    def fake_l2(query, messages):
        calls.append(query)
        return IntentDecision(
            intent="product_search",
            confidence=0.8,
        )

    monkeypatch.setattr(intent_module, "classify_intent_l2", fake_l2)
    state: AgentState = {
        "trace_id": "trace-test",
        "thread_id": "thread-test",
        "user_id": "user-test",
        "raw_query": "我想找适合面试穿的",
        "messages": [],
    }

    result = intent_node(state)

    assert calls == ["我想找适合面试穿的"]
    assert result["intent"] == "product_search"
    assert result["confidence"] == 0.8


def test_l2_receives_history_without_duplicate_current_message(monkeypatch):
    captured_messages = []

    def fake_l2(query, messages):
        captured_messages.extend(messages)
        return IntentDecision(
            intent="product_search",
            confidence=0.8,
        )

    monkeypatch.setattr(intent_module, "classify_intent_l2", fake_l2)
    state: AgentState = {
        "trace_id": "trace-test",
        "thread_id": "thread-test",
        "user_id": "user-test",
        "raw_query": "我想找适合面试穿的",
        "messages": [
            HumanMessage(content="之前看过黑色卫衣"),
            AIMessage(content="我来帮你查询。"),
            HumanMessage(content="我想找适合面试穿的"),
        ],
    }

    intent_node(state)

    assert [message.content for message in captured_messages] == [
        "之前看过黑色卫衣",
        "我来帮你查询。",
    ]
