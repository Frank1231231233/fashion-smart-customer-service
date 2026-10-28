import re
from functools import lru_cache

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.graph.state import AgentState
from app.llm import get_chat_model
from app.nodes.message_utils import (
    history_before_current_message,
    latest_human_context,
)

_REFERENCE_PATTERN = re.compile(r"这件|那件|这个|那个|同款")
_ATTRIBUTE_FOLLOW_UP_PATTERN = re.compile(
    r"[XSML]{1,3}\s*码|尺码|码数|颜色|库存|有货|价格|多少钱",
    re.IGNORECASE,
)
_REWRITE_SYSTEM_PROMPT = """
你是服装商品检索 Query 改写器，只输出结构化结果。

任务：
1. 根据对话历史补全当前商品问题中必要的商品上下文。
2. 保留用户原本表达的品类、颜色、尺码、价格和风格条件。
3. 不得添加用户没有表达的任何商品属性。
4. 问题已经完整时，保持原意并尽量保留原文。
5. 只负责商品检索 Query，不回答用户问题。
""".strip()


class RewriteDecision(BaseModel):
    rewritten_query: str = Field(min_length=1)


@lru_cache
def _get_query_rewriter():
    return get_chat_model().with_structured_output(
        RewriteDecision,
        method="function_calling",
    )


def rewrite_query_l1(raw_query: str, messages: list[AnyMessage]) -> str:
    current_query = raw_query.strip()
    previous_context = latest_human_context(messages, current_query)
    if previous_context is None:
        return current_query

    if _REFERENCE_PATTERN.search(current_query):
        return _REFERENCE_PATTERN.sub(previous_context, current_query)

    is_short_follow_up = len(current_query) <= 20
    if is_short_follow_up and _ATTRIBUTE_FOLLOW_UP_PATTERN.search(current_query):
        return f"{previous_context}，{current_query}"

    return current_query


def rewrite_query_l2(
    query: str,
    messages: list[AnyMessage],
) -> RewriteDecision:
    recent_history = [
        message
        for message in messages[-6:]
        if isinstance(message, HumanMessage | AIMessage)
    ]
    prompt_messages = [
        SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
        *recent_history,
        HumanMessage(content=f"当前商品问题：{query}"),
    ]
    decision = _get_query_rewriter().invoke(prompt_messages)
    return RewriteDecision.model_validate(decision)


def rewrite_node(state: AgentState) -> dict[str, object]:
    current_query = state["raw_query"].strip()
    rewritten_query = rewrite_query_l1(current_query, state["messages"])

    l1_resolved = rewritten_query != current_query
    if l1_resolved or state.get("intent_source") == "l1":
        decision = RewriteDecision(rewritten_query=rewritten_query)
    else:
        history = history_before_current_message(
            state["messages"],
            current_query,
        )
        decision = rewrite_query_l2(current_query, history)

    return decision.model_dump()
