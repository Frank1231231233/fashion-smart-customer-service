import pytest
from pydantic import ValidationError

from app.tools.order import GetOrderDetailInput, OrderDetail


def test_order_detail_input_rejects_blank_order_id():
    with pytest.raises(ValidationError):
        GetOrderDetailInput.model_validate({"order_id": "   "})


def test_order_detail_input_rejects_forged_user_id():
    with pytest.raises(ValidationError) as error_info:
        GetOrderDetailInput.model_validate(
            {
                "order_id": "ORD-001",
                "user_id": "other-user",
            }
        )

    errors = error_info.value.errors()
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == ("user_id",)


def test_order_detail_input_accepts_valid_order_id():
    tool_input = GetOrderDetailInput.model_validate({"order_id": "  ORD-001  "})

    assert tool_input.order_id == "ORD-001"


def test_order_detail_removes_sensitive_fields():
    detail = OrderDetail.model_validate(
        {
            "order_id": "ORD-001",
            "status": "shipped",
            "amount": "299.00",
            "phone": "13800000000",
            "address": "测试地址",
        }
    )

    result = detail.model_dump()

    assert "phone" not in result
    assert "address" not in result
