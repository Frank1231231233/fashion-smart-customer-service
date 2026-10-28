from langchain_core.messages import AIMessage, HumanMessage

from app.graph.state import AgentState
from app.nodes import rewrite as rewrite_module
from app.nodes.rewrite import RewriteDecision, rewrite_node


def test_short_attribute_follow_up_keeps_product_context():
    state: AgentState = {
        "trace_id": "trace-test",
        "thread_id": "thread-test",
        "user_id": "user-test",
        "raw_query": "有 M 码吗",
        "messages": [
            HumanMessage(content="黑色卫衣"),
            AIMessage(content="我来帮你查询。"),
            HumanMessage(content="有 M 码吗"),
        ],
        "intent": "product_search",
        "intent_source": "l1",
    }

    result = rewrite_node(state)

    assert result["rewritten_query"] == "黑色卫衣，有 M 码吗"


def test_explicit_reference_is_replaced_with_product_context():
    state: AgentState = {
        "trace_id": "trace-test",
        "thread_id": "thread-test",
        "user_id": "user-test",
        "raw_query": "这件有 M 码吗",
        "messages": [
            HumanMessage(content="黑色卫衣"),
            HumanMessage(content="这件有 M 码吗"),
        ],
        "intent": "product_search",
        "intent_source": "l1",
    }

    result = rewrite_node(state)

    assert result["rewritten_query"] == "黑色卫衣有 M 码吗"


def test_reference_without_history_is_not_invented():
    state: AgentState = {
        "trace_id": "trace-test",
        "thread_id": "thread-test",
        "user_id": "user-test",
        "raw_query": "这件有 M 码吗",
        "messages": [HumanMessage(content="这件有 M 码吗")],
        "intent": "product_search",
        "intent_source": "l1",
    }

    result = rewrite_node(state)

    assert result["rewritten_query"] == "这件有 M 码吗"


def test_l1_product_query_does_not_call_l2_rewriter(monkeypatch):
    def fail_if_called(query, messages):
        raise AssertionError("L1 商品问题不应调用 L2 改写")

    monkeypatch.setattr(rewrite_module, "rewrite_query_l2", fail_if_called)
    state: AgentState = {
        "trace_id": "trace-test",
        "thread_id": "thread-test",
        "user_id": "user-test",
        "raw_query": "有没有黑色连衣裙",
        "messages": [HumanMessage(content="有没有黑色连衣裙")],
        "intent": "product_search",
        "intent_source": "l1",
    }

    result = rewrite_node(state)

    assert result["rewritten_query"] == "有没有黑色连衣裙"


def test_l2_product_intent_uses_l2_rewriter(monkeypatch):
    captured_history = []

    def fake_l2(query, messages):
        captured_history.extend(messages)
        return RewriteDecision(rewritten_query="适合面试穿的服装")

    monkeypatch.setattr(rewrite_module, "rewrite_query_l2", fake_l2)
    state: AgentState = {
        "trace_id": "trace-test",
        "thread_id": "thread-test",
        "user_id": "user-test",
        "raw_query": "我想找适合面试穿的",
        "messages": [
            HumanMessage(content="之前看过通勤款"),
            AIMessage(content="好的。"),
            HumanMessage(content="我想找适合面试穿的"),
        ],
        "intent": "product_search",
        "intent_source": "l2",
    }

    result = rewrite_node(state)

    assert result["rewritten_query"] == "适合面试穿的服装"
    assert [message.content for message in captured_history] == [
        "之前看过通勤款",
        "好的。",
    ]
