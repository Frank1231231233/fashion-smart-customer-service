from langgraph.graph import END, START, StateGraph

from app.graph.state import AgentState, Intent
from app.nodes.agent import (
    agent_node,
    fallback_node,
    route_after_agent,
)
from app.nodes.history import compress_history_node
from app.nodes.human_review import human_review_node
from app.nodes.intent import intent_node
from app.nodes.rewrite import rewrite_node
from app.nodes.tool import tool_node
from app.trace.wrapper import traced_node


def route_by_intent(state: AgentState) -> Intent:
    return state["intent"]


def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("intent_node", traced_node("intent_node", intent_node))
    builder.add_node("rewrite_node", traced_node("rewrite_node", rewrite_node))
    builder.add_node("agent_node", traced_node("agent_node", agent_node))
    builder.add_node(
        "compress_history_node",
        traced_node("compress_history_node", compress_history_node),
    )
    builder.add_node("tool_node", traced_node("tool_node", tool_node))
    builder.add_node("fallback_node", traced_node("fallback_node", fallback_node))
    builder.add_node(
        "human_review_node",
        traced_node("human_review_node", human_review_node),
    )

    builder.add_edge(START, "intent_node")
    builder.add_conditional_edges(
        "intent_node",
        route_by_intent,
        {
            "product_search": "rewrite_node",
            "order_query": "compress_history_node",
            "refund_request": "human_review_node",
            "other": END,
        },
    )
    builder.add_edge("rewrite_node", "compress_history_node")
    builder.add_edge("compress_history_node", "agent_node")
    builder.add_conditional_edges(
        "agent_node",
        route_after_agent,
        {
            "tool_node": "tool_node",
            "fallback_node": "fallback_node",
            "end": END,
        },
    )
    builder.add_edge("tool_node", "agent_node")
    builder.add_edge("fallback_node", END)
    builder.add_edge("human_review_node", END)

    return builder.compile(checkpointer=checkpointer)
