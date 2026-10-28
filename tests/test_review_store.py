from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_graph
from app.review.store import ReviewStore


def _make_store(tmp_path) -> ReviewStore:
    return ReviewStore(tmp_path / "reviews.db")


def test_create_pending_task_stores_order_id(tmp_path):
    store = _make_store(tmp_path)

    record = store.create_pending_task(
        thread_id="thread-1",
        trace_id="trace-1",
        payload={"user_id": "user-001", "reason_summary": "我要退款"},
        order_id="ORD-001",
    )

    assert record["task_id"].startswith("TSK-")
    assert record["status"] == "pending"
    assert record["processed_at"] is None

    pending = store.list_records(record_type="pending", thread_id="thread-1")
    assert len(pending) == 1
    assert pending[0]["order_id"] == "ORD-001"
    assert pending[0]["user_id"] == "user-001"
    assert pending[0]["payload"]["reason_summary"] == "我要退款"


def test_pending_task_without_order_id_is_null(tmp_path):
    store = _make_store(tmp_path)
    store.create_pending_task(
        thread_id="thread-1",
        trace_id="trace-1",
        payload={"user_id": "user-001"},
    )

    pending = store.list_records(record_type="pending")
    assert pending[0]["order_id"] is None


def test_resolve_writes_operator_and_processed_at(tmp_path):
    store = _make_store(tmp_path)
    store.create_pending_task(
        thread_id="thread-1",
        trace_id="trace-1",
        payload={"user_id": "user-001"},
        order_id="ORD-001",
    )

    store.resolve_pending_task(
        thread_id="thread-1",
        status="accepted_for_manual_processing",
        note=None,
        operator_id="user-900",
    )

    assert store.list_records(record_type="pending") == []
    resolutions = store.list_records(record_type="resolution")
    assert len(resolutions) == 1
    assert resolutions[0]["operator_id"] == "user-900"
    assert resolutions[0]["processed_at"] is not None
    assert resolutions[0]["status"] == "accepted_for_manual_processing"
    assert resolutions[0]["order_id"] == "ORD-001"


def test_list_records_filters_by_status_and_thread(tmp_path):
    store = _make_store(tmp_path)
    for thread_id in ("thread-1", "thread-2"):
        store.create_pending_task(
            thread_id=thread_id,
            trace_id="trace",
            payload={"user_id": "user-001"},
        )
    store.resolve_pending_task(thread_id="thread-1", status="rejected", note="凭证不足")

    assert len(store.list_records(record_type="pending")) == 1
    assert len(store.list_records(record_type="resolution")) == 1
    assert len(store.list_records(thread_id="thread-2")) == 1
    assert len(store.list_records(record_type="resolution", thread_id="thread-2")) == 0


def test_create_ticket_roundtrip(tmp_path):
    store = _make_store(tmp_path)

    ticket = store.create_ticket(
        thread_id="thread-1",
        user_id="user-001",
        payload={"reason_summary": "我要退款"},
    )

    assert ticket["ticket_id"].startswith("TKT-")
    tickets = store.list_records(record_type="ticket", thread_id="thread-1")
    assert len(tickets) == 1
    assert tickets[0]["user_id"] == "user-001"
    assert tickets[0]["status"] == "created"


_CONFIG = {"configurable": {"thread_id": "thread-refund-profile"}}


def _initial_state_with_profile(order_ids: list[str] | None) -> dict[str, object]:
    state: dict[str, object] = {
        "trace_id": "trace-profile",
        "thread_id": "thread-refund-profile",
        "user_id": "user-001",
        "raw_query": "我要退款",
        "messages": [],
    }
    if order_ids is not None:
        state["user_profile"] = {"order_ids": order_ids}
    return state


def test_interrupt_payload_carries_latest_order_id():
    graph = build_graph(InMemorySaver())

    result = graph.invoke(
        _initial_state_with_profile(["ORD-001", "ORD-002"]),
        config=_CONFIG,
    )

    payload = result["__interrupt__"][0].value
    assert payload["order_id"] == "ORD-002"


def test_interrupt_payload_order_id_null_without_profile():
    graph = build_graph(InMemorySaver())

    result = graph.invoke(
        _initial_state_with_profile(None),
        config=_CONFIG,
    )

    payload = result["__interrupt__"][0].value
    assert payload["order_id"] is None
