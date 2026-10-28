from decimal import Decimal

import httpx
import pytest

from app.tools.order_client import OrderNotFoundError, OrderServiceClient


def test_get_order_detail_uses_trusted_user_id():
    def mock_order_service(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/orders/ORD-001"
        assert request.headers["X-User-Id"] == "user-001"

        return httpx.Response(
            200,
            json={
                "order_id": "ORD-001",
                "status": "shipped",
                "amount": "299.00",
                "phone": "13800000000",
            },
        )

    transport = httpx.MockTransport(mock_order_service)

    with httpx.Client(
        transport=transport,
        base_url="http://order-service",
    ) as http_client:
        client = OrderServiceClient(http_client)
        result = client.get_order_detail(
            user_id="user-001",
            order_id="ORD-001",
        )

    assert result.order_id == "ORD-001"
    assert result.status == "shipped"
    assert result.amount == Decimal("299.00")
    assert "phone" not in result.model_dump()


def test_get_order_detail_hides_missing_or_unauthorized_order():
    def mock_order_service(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-User-Id"] == "user-001"
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_order_service)

    with httpx.Client(
        transport=transport,
        base_url="http://order-service",
    ) as http_client:
        client = OrderServiceClient(http_client)

        with pytest.raises(OrderNotFoundError) as error_info:
            client.get_order_detail(
                user_id="user-001",
                order_id="ORD-OTHER-USER",
            )

    assert error_info.value.args == ("ORD-OTHER-USER",)
