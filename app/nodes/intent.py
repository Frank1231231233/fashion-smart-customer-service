import re
from functools import lru_cache

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.graph.state import AgentState, Intent, IntentSource
from app.llm import get_chat_model
from app.nodes.message_utils import history_before_current_message

_INTENT_PATTERNS: tuple[tuple[Intent, re.Pattern[str]], ...] = (
    (
        "refund_request",
        re.compile(r"退款|退货|退钱|退掉|退款进度", re.IGNORECASE),
    ),
    (
        "order_query",
        re.compile(r"订单|订单号|购买记录|下单记录|物流|快递|发货|收货", re.IGNORECASE),
    ),
    (
        "product_search",
        re.compile(
            r"连衣裙|卫衣|衬衫|T恤|短袖|裤子|牛仔裤|外套|大衣|羽绒服|"
            r"裙子|尺码|码数|库存|有货|颜色|款式|价格|多少钱|推荐|"
            r"[XSML]{1,3}\s*码",
            re.IGNORECASE,
        ),
    ),
)
_INTENT_SYSTEM_PROMPT = """
你是服装商城客服的意图识别器，只输出结构化结果。

允许的意图：
- product_search：查询、挑选或推荐服装商品。
- order_query：查询当前用户自己的订单、物流或发货状态。
- refund_request：退款、退货退款或退款进度诉求。
- other：无法确定或不属于以上业务。

规则：
1. 只能基于对话历史和当前问题判断，不得补充用户没有表达的商品属性。
2. 不确定时必须选择 other，不要猜测。
3. 退款相关诉求必须选择 refund_request。
4. confidence 必须在 0 到 1 之间。
""".strip()


class IntentDecision(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)


@lru_cache
def _get_intent_classifier():
    return get_chat_model().with_structured_output(
        IntentDecision,
        method="function_calling",
    )


def classify_intent_l1(query: str) -> Intent | None:
    normalized_query = query.strip()
    if not normalized_query:
        return "other"

    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(normalized_query):
            return intent

    return None


def classify_intent_l2(
    query: str,
    messages: list[AnyMessage],
) -> IntentDecision:
    recent_history = [
        message
        for message in messages[-6:]
        if isinstance(message, HumanMessage | AIMessage)
    ]
    prompt_messages = [
        SystemMessage(content=_INTENT_SYSTEM_PROMPT),
        *recent_history,
        HumanMessage(content=f"当前问题：{query}"),
    ]
    decision = _get_intent_classifier().invoke(prompt_messages)
    return IntentDecision.model_validate(decision)


def intent_node(state: AgentState) -> dict[str, object]:
    current_query = state["raw_query"].strip()
    intent = classify_intent_l1(current_query)

    if intent is not None:
        decision = IntentDecision(
            intent=intent,
            confidence=1.0,
        )
        source: IntentSource = "l1"
    else:
        history = history_before_current_message(
            state["messages"],
            current_query,
        )
        decision = classify_intent_l2(current_query, history)
        source = "l2"

    return {
        **decision.model_dump(),
        "intent_source": source,
    }
