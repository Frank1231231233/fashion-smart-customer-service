from decimal import Decimal
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from app.llm import get_chat_model
from app.retrieval.ingestion import ProductInput

_EXTRACT_SYSTEM_PROMPT = """
你是服装商品信息抽取器,把非结构化商品文本抽取为结构化字段,只输出结构化结果。

规则:
1. title/category/color/price 是入库必填:文本明确给出才填写,没有就给 null,禁止编造。
2. product_id 仅当文本明确出现货号/SKU 编号时填写,否则给 null。
3. sizes 只保留文本中出现的尺码,如 S/M/L 或 26/27。
4. style 用 2-4 个字概括风格(如 通勤、复古、宴会),推断不出就给空字符串。
5. description 保留原文的面料、工艺、卖点,可精简,不得添加原文没有的信息。

示例输入:
"上新:法式碎花连衣裙,颜色是奶黄色,有S和M两个码,售价329元,雪纺面料,收腰设计,很法式复古"

示例输出:
{"product_id": null, "title": "法式碎花连衣裙", "category": "连衣裙", "color": "奶黄色",
 "sizes": ["S", "M"], "price": "329", "style": "法式复古",
 "description": "雪纺面料,收腰设计,法式复古风格。"}
""".strip()

REQUIRED_FIELDS = ("category", "color", "price")


class ProductDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    product_id: str | None = None
    title: str
    category: str | None = None
    color: str | None = None
    sizes: list[str] = []
    price: Decimal | None = None
    style: str = ""
    description: str = ""


@lru_cache
def _get_product_extractor():
    return get_chat_model().with_structured_output(
        ProductDraft,
        method="function_calling",
    )


def extract_product_draft(text: str) -> ProductDraft:
    prompt_messages = [
        SystemMessage(content=_EXTRACT_SYSTEM_PROMPT),
        HumanMessage(content=text.strip()),
    ]
    draft = _get_product_extractor().invoke(prompt_messages)
    return ProductDraft.model_validate(draft)


def draft_to_product(draft: ProductDraft, product_id: str) -> ProductInput:
    missing = [name for name in REQUIRED_FIELDS if getattr(draft, name) is None]
    if missing:
        raise ValueError(f"文本缺少必填信息: {', '.join(missing)}")
    return ProductInput(
        product_id=product_id,
        title=draft.title,
        category=draft.category,
        color=draft.color,
        sizes=draft.sizes,
        price=draft.price,
        style=draft.style,
        description=draft.description,
    )
