from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.nodes import retrieval as product_retrieval
from app.tools import order_client
from app.tools.order import (
    GetOrderDetailInput,
    ListOrdersInput,
    OrderDetail,
)
from app.tools.product import ProductEvidence, SearchProductInput


@dataclass(frozen=True)
class ToolSpec:
    input_model: type[BaseModel]
    execute: Callable[[str, Any], Any]
    description: str


def _search_product(
    user_id: str,
    args: SearchProductInput,
) -> list[ProductEvidence]:
    return product_retrieval.search_product_evidence(
        args.query,
        category=args.category,
        color=args.color,
        price_min=args.price_min,
        price_max=args.price_max,
        size=args.size,
    )


def _list_orders(user_id: str, args: ListOrdersInput) -> list[OrderDetail]:
    return order_client.get_order_service_client().list_orders(user_id=user_id)


def _get_order_detail(user_id: str, args: GetOrderDetailInput) -> OrderDetail:
    return order_client.get_order_service_client().get_order_detail(
        user_id=user_id,
        order_id=args.order_id,
    )


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "search_product": ToolSpec(
        SearchProductInput,
        _search_product,
        "检索服装商品,可查询品类、颜色、尺码、价格、库存、面料和风格;"
        "用户明确限定品类/颜色/价格范围/尺码时,同时填写对应硬过滤字段",
    ),
    "list_orders": ToolSpec(
        ListOrdersInput,
        _list_orders,
        "查询当前登录用户自己的订单列表",
    ),
    "get_order_detail": ToolSpec(
        GetOrderDetailInput,
        _get_order_detail,
        "按订单号查询当前登录用户自己的订单详情",
    ),
}


def build_agent_tool_schemas() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec.description,
                "parameters": spec.input_model.model_json_schema(),
            },
        }
        for name, spec in TOOL_REGISTRY.items()
    ]
