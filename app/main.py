import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from app.auth import Role, User, get_user_store, require_operator, require_user
from app.chat.store import get_chat_store
from app.graph.builder import build_graph
from app.graph.checkpoint import get_checkpointer
from app.graph.state import ReviewStatus
from app.review.store import (
    create_pending_task,
    list_records,
    resolve_pending_task,
)
from app.trace.service import append_trace_event, query_trace_events

app = FastAPI(
    title="Fashion Smart Customer Service",
    version="1.0.0",
)


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=6, max_length=64)
    role: Role = "customer"


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=6, max_length=64)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1)
    thread_id: str | None = None


class ReviewResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    thread_id: str = Field(min_length=1)
    status: ReviewStatus
    note: str | None = Field(default=None, max_length=200)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register", status_code=201)
def register(request: RegisterRequest):
    store = get_user_store()
    try:
        user = store.create_user(
            username=request.username,
            password=request.password,
            role=request.role,
        )
    except ValueError:
        raise HTTPException(status_code=409, detail="username already exists") from None
    return {"user": user}


@app.post("/auth/login")
def login(request: LoginRequest):
    store = get_user_store()
    user = store.authenticate(request.username, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid username or password")
    token = store.create_token(user["user_id"])
    return {"token": token, "user": user}


@app.post("/chat")
def chat(request: ChatRequest, user: Annotated[User, Depends(require_user)]):
    user_id = user["user_id"]
    trace_id = str(uuid.uuid4())
    store = get_chat_store()
    graph = build_graph(get_checkpointer())

    if request.thread_id is None:
        thread_id = str(uuid.uuid4())
        store.create_session(
            user_id=user_id,
            thread_id=thread_id,
            title=request.message[:20],
        )
        initial_state: dict[str, object] = {
            "trace_id": trace_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "raw_query": request.message,
            "messages": [HumanMessage(content=request.message)],
        }
    else:
        thread_id = request.thread_id
        session = store.get_session(thread_id)
        if session is None or session["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="unknown thread_id")
        store.touch_session(thread_id)
        initial_state = {
            "trace_id": trace_id,
            "raw_query": request.message,
            "messages": [HumanMessage(content=request.message)],
            "agent_steps": 0,
            "final_answer": None,
            "tool_results": None,
            "review_status": None,
            "error": None,
            "error_type": None,
        }

    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(initial_state, config=config)

    interrupts = result.get("__interrupt__") or []
    if interrupts:
        payload = interrupts[0].value
        create_pending_task(
            thread_id=thread_id,
            trace_id=trace_id,
            payload=payload,
            order_id=payload.get("order_id"),
        )

    return {
        "trace_id": trace_id,
        "thread_id": thread_id,
        "intent": result.get("intent"),
        "rewritten_query": result.get("rewritten_query"),
        "confidence": result.get("confidence"),
        "tool_results": result.get("tool_results"),
        "review_status": result.get("review_status"),
        "final_answer": result.get("final_answer"),
        "interrupted": bool(interrupts),
        "error": result.get("error"),
    }


@app.get("/chats")
def list_chats(user: Annotated[User, Depends(require_user)]):
    return {"chats": get_chat_store().list_sessions(user["user_id"])}


@app.get("/chats/{thread_id}/messages")
def get_chat_messages(thread_id: str, user: Annotated[User, Depends(require_user)]):
    session = get_chat_store().get_session(thread_id)
    if session is None or session["user_id"] != user["user_id"]:
        raise HTTPException(status_code=404, detail="unknown thread_id")

    graph = build_graph(get_checkpointer())
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    messages = [
        {
            "role": "user" if isinstance(message, HumanMessage) else "assistant",
            "content": str(message.content),
        }
        for message in state.values.get("messages", [])
        if isinstance(message, (HumanMessage, AIMessage)) and message.content
    ]
    return {"thread_id": thread_id, "messages": messages}


@app.get("/reviews/pending")
def list_pending_reviews(operator: Annotated[User, Depends(require_operator)]):
    return {"pending": list_records(record_type="pending")}


@app.post("/reviews/resume")
def resume_review(
    request: ReviewResumeRequest,
    operator: Annotated[User, Depends(require_operator)],
):
    graph = build_graph(get_checkpointer())
    config = {"configurable": {"thread_id": request.thread_id}}
    if not graph.get_state(config).values:
        raise HTTPException(status_code=404, detail="unknown thread_id")

    pending = list_records(record_type="pending", thread_id=request.thread_id)
    result = graph.invoke(
        Command(resume={"status": request.status, "note": request.note}),
        config=config,
    )
    resolve_pending_task(
        thread_id=request.thread_id,
        status=request.status,
        note=request.note,
        operator_id=operator["user_id"],
    )
    append_trace_event(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "trace_id": pending[0]["trace_id"] if pending else None,
            "thread_id": request.thread_id,
            "node": "human_review_node",
            "event": "resume",
            "duration_ms": 0.0,
            "intent": None,
            "tool_name": None,
            "error_type": None,
            "operator_id": operator["user_id"],
            "status": request.status,
        }
    )
    return {
        "thread_id": request.thread_id,
        "review_status": result.get("review_status"),
        "final_answer": result.get("final_answer"),
        "error": result.get("error"),
    }


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str):
    events = query_trace_events(trace_id)

    return {
        "trace_id": trace_id,
        "events": events,
    }
