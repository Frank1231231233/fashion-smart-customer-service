import re

from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from app.graph.state import AgentState, ReviewStatus
from app.review.store import create_ticket

_PHONE_PATTERN = re.compile(r"1[3-9]\d{9}")
_LONG_DIGIT_PATTERN = re.compile(r"\d{6,}")


def mask_sensitive(text: str) -> str:
    masked = _PHONE_PATTERN.sub("***", text)
    return _LONG_DIGIT_PATTERN.sub("***", masked)


class ReviewDecision(BaseModel):
    status: ReviewStatus
    note: str | None = Field(default=None, max_length=200)


def human_review_node(state: AgentState) -> dict[str, object]:
    profile = state.get("user_profile") or {}
    order_ids = profile.get("order_ids") or []
    payload = {
        "user_id": state["user_id"],
        "trace_id": state["trace_id"],
        "order_id": order_ids[-1] if order_ids else None,
        "reason_summary": mask_sensitive(state["raw_query"]),
    }
    resume_value = interrupt(payload)

    try:
        decision = ReviewDecision.model_validate(resume_value)
    except ValidationError:
        return {
            "error": "invalid human review decision",
            "error_type": "ValidationError",
        }

    update: dict[str, object] = {
        "review_payload": payload,
        "review_status": decision.status,
    }
    answer: str
    if decision.status == "accepted_for_manual_processing":
        ticket = create_ticket(
            thread_id=state["thread_id"],
            user_id=state["user_id"],
            payload=payload,
        )
        answer = (
            f"已转人工处理,售后工单 {ticket['ticket_id']} 已创建,"
            "请等待人工客服联系。"
        )
    elif decision.status == "rejected":
        reason = decision.note or "人工未填写原因"
        answer = f"人工客服暂未接管:{reason}"
    else:
        reason = decision.note or "请补充订单号与退款原因"
        answer = f"人工客服需要您补充信息:{reason}"
    update["final_answer"] = answer
    update["messages"] = [AIMessage(content=answer)]
    return update
