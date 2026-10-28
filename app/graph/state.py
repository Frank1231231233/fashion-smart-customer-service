from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.retrieval.models import RankedProduct

Intent = Literal[
    "product_search",
    "order_query",
    "refund_request",
    "other",
]
IntentSource = Literal["l1", "l2"]
ReviewStatus = Literal[
    "accepted_for_manual_processing",
    "rejected",
    "need_more_info",
]


class ToolCall(TypedDict):
    name: str
    arguments: dict[str, object]


class AgentState(TypedDict):
    trace_id: str
    thread_id: str
    user_id: str

    raw_query: str
    messages: Annotated[list[AnyMessage], add_messages]

    intent: NotRequired[Intent]
    intent_source: NotRequired[IntentSource]
    rewritten_query: NotRequired[str]
    confidence: NotRequired[float]
    retrieval_results: NotRequired[list[RankedProduct]]

    tool_calls: NotRequired[list[ToolCall]]
    tool_results: NotRequired[list[dict[str, object]]]
    tool_name: NotRequired[str]
    agent_steps: NotRequired[int]

    summary: NotRequired[str]
    user_profile: NotRequired[dict[str, object]]

    review_payload: NotRequired[dict[str, object]]
    review_status: NotRequired[ReviewStatus]

    final_answer: NotRequired[str]
    error: NotRequired[str]
    error_type: NotRequired[str]
