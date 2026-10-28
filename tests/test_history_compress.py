from langchain_core.messages import AIMessage, HumanMessage

from app.nodes import agent as agent_module
from app.nodes import history as history_module
from app.nodes.agent import agent_node
from app.nodes.history import (
    compress_history_node,
    merge_tool_profile,
)
from app.nodes.tool import tool_node
from app.tools.product import ProductEvidence, SearchProductInput
from app.tools.registry import TOOL_REGISTRY, ToolSpec


def _humans(count: int) -> list[HumanMessage]:
    return [HumanMessage(content=f"m{i}") for i in range(count)]


def test_compress_skips_short_history(monkeypatch):
    def fail_if_called():
        raise AssertionError("历史未超阈值时不应调用 LLM")

    monkeypatch.setattr(history_module, "get_chat_model", fail_if_called)

    assert compress_history_node({"messages": _humans(5)}) == {}


def test_compress_summarizes_only_messages_beyond_recent_window(monkeypatch):
    captured: list[str] = []

    class FakeModel:
        def invoke(self, messages):
            captured.append(messages[-1].content)
            return AIMessage(content="合并后的摘要")

    monkeypatch.setattr(history_module, "get_chat_model", FakeModel)

    result = compress_history_node(
        {
            "messages": _humans(9),
            "summary": "旧摘要",
        }
    )

    assert result == {"summary": "合并后的摘要"}
    rendered = captured[0]
    assert "已有摘要:旧摘要" in rendered
    for i in range(5):
        assert f"m{i}" in rendered
    for i in range(5, 9):
        assert f"m{i}" not in rendered


def test_merge_tool_profile_records_order_and_viewed_products():
    class Args:
        order_id = "ORD-001"

    profile = merge_tool_profile({}, "get_order_detail", Args(), [])
    assert profile["order_ids"] == ["ORD-001"]

    dumped = [
        {"product_id": "SKU-001", "title": "黑色修身连衣裙"},
        {"product_id": "SKU-003", "title": "红色法式连衣裙"},
    ]
    profile = merge_tool_profile(profile, "search_product", object(), dumped)
    assert profile["viewed_products"] == dumped

    profile = merge_tool_profile(profile, "get_order_detail", Args(), [])
    assert profile["order_ids"] == ["ORD-001"]


def test_tool_node_writes_user_profile_on_successful_search(monkeypatch):
    spec = ToolSpec(
        SearchProductInput,
        lambda _user_id, _args: [
            ProductEvidence(
                product_id="SKU-001",
                title="黑色修身连衣裙",
                category="连衣裙",
                color="黑色",
                sizes=["M"],
                price="299",
            )
        ],
        "检索服装商品",
    )
    monkeypatch.setitem(TOOL_REGISTRY, "search_product", spec)

    result = tool_node(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_product", "args": {"query": "连衣裙"}, "id": "c1"}
                    ],
                )
            ],
            "user_id": "user-001",
            "user_profile": {"order_ids": ["ORD-001"], "viewed_products": []},
        }
    )

    profile = result["user_profile"]
    assert profile["order_ids"] == ["ORD-001"]
    assert profile["viewed_products"] == [
        {"product_id": "SKU-001", "title": "黑色修身连衣裙"}
    ]


def test_agent_prompt_injects_profile_summary_and_recent_window(monkeypatch):
    captured: list[object] = []

    class FakeRunnable:
        def invoke(self, messages):
            captured.extend(messages)
            return AIMessage(content="ok")

    monkeypatch.setattr(agent_module, "get_agent_runnable", FakeRunnable)

    agent_node(
        {
            "messages": _humans(9),
            "summary": "用户想退款 ORD-001",
            "user_profile": {"order_ids": ["ORD-001"], "viewed_products": []},
            "agent_steps": 1,
            "raw_query": "m8",
        }
    )

    human_contents = [m.content for m in captured if isinstance(m, HumanMessage)]
    assert any("用户想退款 ORD-001" in content for content in human_contents)
    assert any("ORD-001" in content and "档案" in content for content in human_contents)
    history_in_prompt = [c for c in human_contents if c.startswith("m")]
    assert history_in_prompt == ["m5", "m6", "m7", "m8"]


def test_agent_prompt_uses_full_window_without_summary(monkeypatch):
    captured: list[object] = []

    class FakeRunnable:
        def invoke(self, messages):
            captured.extend(messages)
            return AIMessage(content="ok")

    monkeypatch.setattr(agent_module, "get_agent_runnable", FakeRunnable)

    agent_node(
        {
            "messages": _humans(9),
            "agent_steps": 1,
            "raw_query": "m8",
        }
    )

    history_in_prompt = [
        m.content
        for m in captured
        if isinstance(m, HumanMessage) and m.content.startswith("m")
    ]
    assert history_in_prompt == [f"m{i}" for i in range(1, 9)]
