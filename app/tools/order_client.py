from functools import lru_cache

import httpx

from app.config import get_settings
from app.tools.order import OrderDetail


class OrderNotFoundError(Exception):
    pass


class OrderServiceClient:
    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def list_orders(self, *, user_id: str) -> list[OrderDetail]:
        response = self._client.get(
            "/internal/orders",
            headers={"X-User-Id": user_id},
        )

        response.raise_for_status()
        return [OrderDetail.model_validate(item) for item in response.json()]

    def get_order_detail(
        self,
        *,
        user_id: str,
        order_id: str,
    ) -> OrderDetail:
        response = self._client.get(
            f"/internal/orders/{order_id}",
            headers={"X-User-Id": user_id},
        )

        if response.status_code == 404:
            raise OrderNotFoundError(order_id)

        response.raise_for_status()
        return OrderDetail.model_validate(response.json())


@lru_cache
def get_order_service_client() -> OrderServiceClient:
    settings = get_settings()
    return OrderServiceClient(
        httpx.Client(
            base_url=settings.order_service_base_url,
            timeout=5.0,
        )
    )
