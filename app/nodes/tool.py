import json

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from app.graph.state import AgentState
from app.nodes.history import merge_tool_profile
from app.tools.registry import TOOL_REGISTRY


def _dump_output(output: BaseModel | list[BaseModel]) -> list[dict[str, object]]:
    items = output if isinstance(output, list) else [output]
    return [item.model_dump(mode="json") for item in items]


def tool_node(state: AgentState) -> dict[str, object]:
    last_message = state["messages"][-1] if state["messages"] else None
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {
            "error": "tool_node requires tool calls from the agent",
            "error_type": "InvalidToolCallError",
        }

    results: list[dict[str, object]] = []
    tool_messages: list[ToolMessage] = []
    tool_name = ""
    error: str | None = None
    error_type: str | None = None
    profile = dict(state.get("user_profile") or {})
    profile_changed = False

    for call in last_message.tool_calls:
        tool_name = call["name"]
        spec = TOOL_REGISTRY.get(tool_name)
        if spec is None:
            payload: object = {
                "error": f"tool '{tool_name}' is not in the whitelist",
                "error_type": "UnknownToolError",
            }
            error = error or f"tool '{tool_name}' is not in the whitelist"
            error_type = error_type or "UnknownToolError"
        else:
            try:
                args = spec.input_model.model_validate(call["args"])
                output = spec.execute(state["user_id"], args)
                dumped = _dump_output(output)
                results.extend(dumped)
                payload = dumped
                profile = merge_tool_profile(profile, tool_name, args, dumped)
                profile_changed = True
            except Exception as exc:  # noqa: BLE001  # 任务书要求:单点捕获工具异常写入 error
                payload = {"error": str(exc), "error_type": type(exc).__name__}
                error = error or str(exc)
                error_type = error_type or type(exc).__name__
        tool_messages.append(
            ToolMessage(
                content=json.dumps(payload, ensure_ascii=False),
                tool_call_id=call["id"],
            )
        )

    update: dict[str, object] = {
        "messages": tool_messages,
        "tool_results": results,
        "tool_name": tool_name,
    }
    if profile_changed:
        update["user_profile"] = profile
    if error is not None:
        update["error"] = error
        update["error_type"] = error_type
    return update
