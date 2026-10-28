from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.graph.state import AgentState
from app.llm import get_chat_model

COMPRESS_THRESHOLD = 8
KEEP_RECENT = 4
MAX_VIEWED_PRODUCTS = 5

_SUMMARY_SYSTEM_PROMPT = """
请总结以下客服对话历史。必须严格保留以下信息:
1. 用户提到的具体订单号、尺码、商品名称。
2. 用户的明确诉求(如退款、换货、比价)。
3. 用户拒绝过的方案。
去除:所有寒暄(你好、谢谢)和重复的问答。
若输入包含"已有摘要",请把它与新内容合并成一段,总长控制在 200 字以内。
""".strip()


def _render_messages(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"用户: {message.content}")
        elif isinstance(message, AIMessage):
            tool_note = ""
            if message.tool_calls:
                names = ",".join(call["name"] for call in message.tool_calls)
                tool_note = f" [调用工具: {names}]"
            lines.append(f"客服: {message.content}{tool_note}")
        elif isinstance(message, ToolMessage):
            lines.append(f"工具结果: {str(message.content)[:200]}")
    return "\n".join(lines)


def compress_history_node(state: AgentState) -> dict[str, object]:
    messages = state["messages"]
    if len(messages) <= COMPRESS_THRESHOLD:
        return {}

    parts: list[str] = []
    previous_summary = state.get("summary")
    if previous_summary:
        parts.append(f"已有摘要:{previous_summary}")
    parts.append(_render_messages(messages[:-KEEP_RECENT]))

    response = get_chat_model().invoke(
        [
            SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(parts)),
        ]
    )
    return {"summary": str(response.content)}


def merge_tool_profile(
    profile: dict[str, object],
    tool_name: str,
    args: object,
    dumped_results: list[dict[str, object]],
) -> dict[str, object]:
    merged: dict[str, object] = {
        **profile,
        "order_ids": list(profile.get("order_ids", [])),
        "viewed_products": list(profile.get("viewed_products", [])),
    }

    if tool_name == "get_order_detail":
        order_id = getattr(args, "order_id", None)
        if order_id and order_id not in merged["order_ids"]:
            merged["order_ids"].append(order_id)

    if tool_name == "search_product":
        viewed: list[dict[str, object]] = merged["viewed_products"]
        seen = {item["product_id"] for item in viewed}
        for item in dumped_results:
            if item["product_id"] not in seen:
                viewed.append(
                    {"product_id": item["product_id"], "title": item["title"]}
                )
                seen.add(item["product_id"])
        merged["viewed_products"] = viewed[-MAX_VIEWED_PRODUCTS:]

    return merged
