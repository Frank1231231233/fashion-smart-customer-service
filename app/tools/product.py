from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class SearchProductInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(min_length=1, description="语义检索词,如 黑色连衣裙、面试穿搭")
    category: str | None = Field(
        default=None,
        description="硬过滤:品类,如 连衣裙/衬衫;用户明确限定时才填",
    )
    color: str | None = Field(
        default=None,
        description="硬过滤:颜色,如 黑色/红色;用户明确限定时才填",
    )
    price_min: Decimal | None = Field(
        default=None, ge=0,
        description="硬过滤:价格下限(元),如 用户说500以上时填 500",
    )
    price_max: Decimal | None = Field(
        default=None, ge=0,
        description="硬过滤:价格上限(元),如 用户说500元以下时填 500",
    )
    size: str | None = Field(
        default=None,
        description="硬过滤:必须有的尺码,如 M;用户明确限定尺码时才填",
    )


class ProductEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    product_id: str
    title: str
    category: str
    color: str
    sizes: list[str]
    price: Decimal = Field(ge=0)
    style: str = ""
    description: str = ""
