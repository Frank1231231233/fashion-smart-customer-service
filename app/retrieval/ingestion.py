import hashlib
from decimal import Decimal
from unicodedata import normalize

from pydantic import BaseModel, Field


class ProductInput(BaseModel):
    product_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    category: str = Field(min_length=1)
    color: str = Field(min_length=1)
    sizes: list[str]
    price: Decimal = Field(ge=0)
    style: str = ""
    description: str = ""


def _normalize_text(value: str) -> str:
    normalized_value = normalize("NFKC", value)
    return " ".join(normalized_value.strip().casefold().split())


def _normalize_sizes(sizes: list[str]) -> str:
    normalized_sizes = {
        _normalize_text(size).upper() for size in sizes if _normalize_text(size)
    }
    return ",".join(sorted(normalized_sizes))


def _normalize_price(price: Decimal) -> str:
    return format(price.normalize(), "f")


def normalize_product(product: ProductInput) -> str:
    fields = (
        ("title", _normalize_text(product.title)),
        ("category", _normalize_text(product.category)),
        ("color", _normalize_text(product.color)),
        ("sizes", _normalize_sizes(product.sizes)),
        ("price", _normalize_price(product.price)),
        ("style", _normalize_text(product.style)),
        ("description", _normalize_text(product.description)),
    )
    return "\n".join(f"{name}={value}" for name, value in fields)


def calculate_product_md5(product: ProductInput) -> str:
    normalized_text = normalize_product(product)
    return hashlib.md5(
        normalized_text.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
