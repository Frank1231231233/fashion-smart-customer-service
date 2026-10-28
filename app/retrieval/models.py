from typing import Protocol, TypedDict


class ProductDocument(TypedDict):
    product_id: str
    text: str


class RankedProduct(TypedDict):
    product_id: str
    rank: int
    score: float


class ProductDocumentIndex(Protocol):
    def upsert_documents(self, documents: list[ProductDocument]) -> None: ...
