from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter

from langgraph.errors import GraphInterrupt

from app.graph.state import AgentState
from app.trace.service import append_trace_event

NodeFunction = Callable[[AgentState], dict[str, object]]


def traced_node(node_name: str, node_func: NodeFunction) -> NodeFunction:
    def wrapper(state: AgentState) -> dict[str, object]:
        trace_id = state["trace_id"]
        start_time = perf_counter()
        print(f"[trace_id={trace_id}] node={node_name} event=start")
        append_trace_event(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "trace_id": trace_id,
                "thread_id": state["thread_id"],
                "node": node_name,
                "event": "start",
                "duration_ms": 0.0,
                "intent": state.get("intent"),
                "tool_name": None,
                "error_type": None,
            }
        )
        try:
            result = node_func(state)
            duration_ms = round((perf_counter() - start_time) * 1000, 2)
            print(
                f"[trace_id={trace_id}] node={node_name} "
                f"event=end duration_ms={duration_ms}"
            )
            append_trace_event(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "trace_id": trace_id,
                    "thread_id": state["thread_id"],
                    "node": node_name,
                    "event": "error" if result.get("error") else "end",
                    "duration_ms": duration_ms,
                    "intent": result.get("intent"),
                    "tool_name": result.get("tool_name"),
                    "error_type": result.get("error_type"),
                }
            )
            return result
        except GraphInterrupt:
            duration_ms = round((perf_counter() - start_time) * 1000, 2)
            print(
                f"[trace_id={trace_id}] node={node_name} "
                f"event=interrupt duration_ms={duration_ms}"
            )
            append_trace_event(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "trace_id": trace_id,
                    "thread_id": state["thread_id"],
                    "node": node_name,
                    "event": "interrupt",
                    "duration_ms": duration_ms,
                    "intent": state.get("intent"),
                    "tool_name": None,
                    "error_type": None,
                }
            )
            raise
        except Exception as exc:
            duration_ms = round((perf_counter() - start_time) * 1000, 2)
            error_type = type(exc).__name__
            print(
                f"[trace_id={trace_id}] node={node_name} event=error "
                f"duration_ms={duration_ms} error_type={error_type}"
            )
            append_trace_event(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "trace_id": trace_id,
                    "thread_id": state["thread_id"],
                    "node": node_name,
                    "event": "error",
                    "duration_ms": duration_ms,
                    "intent": state.get("intent"),
                    "tool_name": None,
                    "error_type": error_type,
                }
            )
            raise

    return wrapper
