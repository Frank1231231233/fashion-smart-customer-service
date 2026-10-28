import pytest

from app.retrieval import extractor as extractor_module
from app.retrieval.extractor import (
    ProductDraft,
    draft_to_product,
    extract_product_draft,
)


def _complete_draft() -> ProductDraft:
    return ProductDraft(
        title="法式碎花连衣裙",
        category="连衣裙",
        color="奶黄色",
        sizes=["S", "M"],
        price="329",
        style="法式复古",
        description="雪纺面料,收腰设计。",
    )


def test_draft_to_product_uses_given_product_id():
    product = draft_to_product(_complete_draft(), "SKU-011")

    assert product.product_id == "SKU-011"
    assert product.title == "法式碎花连衣裙"
    assert str(product.price) == "329"


def test_draft_to_product_rejects_missing_required_fields():
    draft = ProductDraft(title="碎花连衣裙", color="奶黄色")

    with pytest.raises(ValueError, match="category, price"):
        draft_to_product(draft, "SKU-011")


def test_extract_product_draft_calls_llm_with_few_shot_prompt(monkeypatch):
    captured: list[object] = []

    class FakeExtractor:
        def invoke(self, messages):
            captured.extend(messages)
            return _complete_draft()

    monkeypatch.setattr(extractor_module, "_get_product_extractor", FakeExtractor)

    draft = extract_product_draft("奶黄色连衣裙 329 元")

    assert draft.title == "法式碎花连衣裙"
    system_prompt = captured[0].content
    assert "示例输入" in system_prompt
    assert "示例输出" in system_prompt
    assert captured[-1].content == "奶黄色连衣裙 329 元"
