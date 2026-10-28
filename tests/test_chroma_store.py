from langchain_core.embeddings import Embeddings

from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.models import ProductDocument


class FakeEmbeddings(Embeddings):
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectors[text]


def test_chroma_persists_products_across_store_instances(tmp_path):
    embeddings = FakeEmbeddings(
        {
            "黑色通勤连衣裙": [1.0, 0.0],
            "休闲运动外套": [0.0, 1.0],
            "适合面试的衣服": [0.9, 0.1],
        }
    )
    documents: list[ProductDocument] = [
        {
            "product_id": "SKU-001",
            "text": "黑色通勤连衣裙",
        },
        {
            "product_id": "SKU-002",
            "text": "休闲运动外套",
        },
    ]
    path = tmp_path / "chroma"

    first_store = ChromaVectorStore(path, embeddings)
    first_store.upsert_documents(documents)
    reopened_store = ChromaVectorStore(path, embeddings)

    results = reopened_store.search("适合面试的衣服", top_k=2)

    assert [result["product_id"] for result in results] == [
        "SKU-001",
        "SKU-002",
    ]
    assert [result["rank"] for result in results] == [1, 2]


def test_chroma_upsert_replaces_existing_product_vector(tmp_path):
    embeddings = FakeEmbeddings(
        {
            "通勤连衣裙": [1.0, 0.0],
            "运动连衣裙": [0.0, 1.0],
            "运动外套": [0.2, 0.8],
            "运动风": [0.0, 1.0],
        }
    )
    store = ChromaVectorStore(tmp_path / "chroma", embeddings)
    store.upsert_documents(
        [
            {"product_id": "SKU-001", "text": "通勤连衣裙"},
            {"product_id": "SKU-002", "text": "运动外套"},
        ]
    )

    store.upsert_documents(
        [{"product_id": "SKU-001", "text": "运动连衣裙"}],
    )
    results = store.search("运动风", top_k=2)

    assert results[0]["product_id"] == "SKU-001"
    assert len({result["product_id"] for result in results}) == 2


def test_chroma_search_handles_empty_inputs(tmp_path):
    store = ChromaVectorStore(
        tmp_path / "chroma",
        FakeEmbeddings({}),
    )

    assert store.search("连衣裙") == []
    assert store.search("") == []
