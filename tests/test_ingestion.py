from app.retrieval.ingestion import (
    ProductInput,
    calculate_product_md5,
    normalize_product,
)


def test_equivalent_products_have_same_normalized_text():
    first = ProductInput(
        product_id="SKU-001",
        title=" 通勤Ｔ恤 ",
        category=" 上衣 ",
        color=" BLACK ",
        sizes=["L", "m", "L"],
        price="299.00",
        style=" Office ",
        description="适合   日常通勤",
    )
    second = ProductInput(
        product_id="SKU-002",
        title="通勤T恤",
        category="上衣",
        color="black",
        sizes=["M", "l"],
        price="299",
        style="office",
        description="适合 日常通勤",
    )

    assert normalize_product(first) == normalize_product(second)
    assert calculate_product_md5(first) == calculate_product_md5(second)
    assert len(calculate_product_md5(first)) == 32


def test_substantive_product_change_changes_normalized_text():
    black_product = ProductInput(
        product_id="SKU-001",
        title="通勤连衣裙",
        category="连衣裙",
        color="黑色",
        sizes=["M", "L"],
        price="299",
    )
    white_product = black_product.model_copy(update={"color": "白色"})

    assert normalize_product(black_product) != normalize_product(white_product)
    assert calculate_product_md5(black_product) != calculate_product_md5(white_product)


def test_normalized_text_uses_fixed_field_order():
    product = ProductInput(
        product_id="SKU-001",
        title="通勤连衣裙",
        category="连衣裙",
        color="黑色",
        sizes=["M", "L"],
        price="299.00",
        style="通勤",
        description="修身版型",
    )

    assert normalize_product(product).splitlines() == [
        "title=通勤连衣裙",
        "category=连衣裙",
        "color=黑色",
        "sizes=L,M",
        "price=299",
        "style=通勤",
        "description=修身版型",
    ]
