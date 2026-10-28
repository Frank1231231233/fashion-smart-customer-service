from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ListOrdersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetOrderDetailInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    order_id: str = Field(min_length=1)


class OrderDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: str
    status: Literal[
        "pending",
        "paid",
        "shipped",
        "completed",
        "cancelled",
    ]
    amount: Decimal = Field(ge=0)
