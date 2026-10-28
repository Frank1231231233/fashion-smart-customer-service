from langchain_core.embeddings import Embeddings

from app.retrieval.models import ProductDocument
from app.retrieval.vector import VectorRetriever


class FakeEmbeddings(Embeddings):
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectors[text]


def test_vector_search_returns_semantically_ranked_products():
    documents: list[ProductDocument] = [
        {
            "product_id": "SKU-001",
            "text": "黑色通勤连衣裙",
        },
        {
            "product_id": "SKU-002",
            "text": "休闲运动外套",
        },
        {
            "product_id": "SKU-003",
            "text": "度假碎花长裙",
        },
    ]
    embeddings = FakeEmbeddings(
        {
            "黑色通勤连衣裙": [1.0, 0.0],
            "休闲运动外套": [0.0, 1.0],
            "度假碎花长裙": [0.6, 0.4],
            "适合面试的正式穿搭": [0.9, 0.1],
        }
    )
    retriever = VectorRetriever(documents, embeddings)

    results = retriever.search("适合面试的正式穿搭", top_k=2)

    assert [result["product_id"] for result in results] == [
        "SKU-001",
        "SKU-003",
    ]
    assert [result["rank"] for result in results] == [1, 2]
    assert results[0]["score"] > results[1]["score"]


def test_vector_search_handles_empty_inputs():
    embeddings = FakeEmbeddings({})
    retriever = VectorRetriever([], embeddings)

    assert retriever.search("连衣裙") == []
    assert retriever.search("") == []
