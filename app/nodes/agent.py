import json

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from app.graph.state import AgentState
from app.llm import get_chat_model
from app.nodes.history import COMPRESS_THRESHOLD, KEEP_RECENT
from app.tools.registry import build_agent_tool_schemas

MAX_AGENT_STEPS = 3

_AGENT_SYSTEM_PROMPT = """
你是服装商城智能客服。你只能基于当前对话和已授权工具结果回答。

规则:
1. 商品名称、尺码、颜色、价格、库存、面料、订单号、金额、订单状态等业务事实,
   只能引用工具返回结果,不得编造。
2. 没有证据时明确说明未查到,不得猜测、补全或承诺。
3. 只能调用系统提供的白名单工具,不得虚构工具或参数。
4. 当前用户身份由服务端注入,不得索要、采信或使用用户声明的 user_id。
5. 工具报错时如实告知用户查询失败,不要反复重试。
""".strip()


def get_agent_runnable():
    return get_chat_model().bind_tools(build_agent_tool_schemas())


def _build_prompt_messages(state: AgentState) -> list[AnyMessage]:
    prompt: list[AnyMessage] = [SystemMessage(content=_AGENT_SYSTEM_PROMPT)]
    profile = state.get("user_profile")
    if profile:
        prompt.append(
            HumanMessage(
                content="用户档案(系统记录,权威信息,长期有效):"
                + json.dumps(profile, ensure_ascii=False)
            )
        )
    summary = state.get("summary")
    if summary:
        prompt.append(HumanMessage(content=f"早前对话摘要:{summary}"))
    window = KEEP_RECENT if summary else COMPRESS_THRESHOLD
    prompt.extend(state["messages"][-window:])
    return prompt


def agent_node(state: AgentState) -> dict[str, object]:
    steps = state.get("agent_steps", 0)
    prompt_messages = _build_prompt_messages(state)

    if steps == 0:
        rewritten = state.get("rewritten_query")
        if rewritten and rewritten != state["raw_query"]:
            prompt_messages.append(HumanMessage(content=f"当前问题:{rewritten}"))

    response = get_agent_runnable().invoke(prompt_messages)

    update: dict[str, object] = {
        "messages": [response],
        "agent_steps": steps + 1,
    }
    if not response.tool_calls:
        update["final_answer"] = response.content
    return update


def route_after_agent(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        if state.get("agent_steps", 0) < MAX_AGENT_STEPS:
            return "tool_node"
        return "fallback_node"
    return "end"


def fallback_node(state: AgentState) -> dict[str, object]:
    return {
        "final_answer": "抱歉,这个问题暂时无法完成,请换个问法或联系人工客服。",
    }
