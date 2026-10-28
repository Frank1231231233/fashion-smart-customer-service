from langchain_core.embeddings import Embeddings

from app.nodes import retrieval as retrieval_module
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.models import ProductDocument
from app.retrieval.vector import VectorRetriever


class FakeEmbeddings(Embeddings):
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectors[text]


def _documents() -> list[ProductDocument]:
    return [
        {
            "product_id": "SKU-001",
            "text": "黑色 通勤 连衣裙 修身 面试",
        },
        {
            "product_id": "SKU-002",
            "text": "职业 面试 西装 套装",
        },
        {
            "product_id": "SKU-003",
            "text": "黑色 通勤 衬衫 正装",
        },
    ]


def _embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(
        {
            "黑色 通勤 连衣裙 修身 面试": [1.0, 0.0],
            "职业 面试 西装 套装": [0.9, 0.1],
            "黑色 通勤 衬衫 正装": [0.4, 0.6],
            "黑色通勤连衣裙": [1.0, 0.0],
        }
    )


def test_hybrid_search_fuses_keyword_and_vector_rankings(monkeypatch):
    documents = _documents()
    embeddings = _embeddings()

    bm25_results = BM25Retriever(documents).search("黑色通勤连衣裙")
    vector_results = VectorRetriever(documents, embeddings).search("黑色通勤连衣裙")

    class FakeVectorStore:
        def search(
            self,
            query: str,
            top_k: int = 5,
            candidate_ids: list[str] | None = None,
        ):
            return vector_results[:top_k]

    monkeypatch.setattr(
        retrieval_module,
        "get_product_vector_store",
        FakeVectorStore,
    )
    results = retrieval_module.hybrid_search("黑色通勤连衣裙", documents)

    assert bm25_results[0]["product_id"] == "SKU-001"
    assert vector_results[0]["product_id"] == "SKU-001"
    assert results[0]["product_id"] == "SKU-001"
    assert [result["rank"] for result in results] == [1, 2, 3]
    assert len({result["product_id"] for result in results}) == len(results)


def test_retrieval_node_uses_rewritten_query(monkeypatch):
    captured_queries: list[str] = []

    monkeypatch.setattr(
        retrieval_module,
        "load_product_documents",
        lambda: _documents(),
    )

    def fake_hybrid_search(
        query: str,
        documents: list[ProductDocument],
        *,
        top_k: int = 5,
    ):
        captured_queries.append(query)
        return []

    monkeypatch.setattr(
        retrieval_module,
        "hybrid_search",
        fake_hybrid_search,
    )

    result = retrieval_module.retrieval_node(
        {
            "trace_id": "trace-001",
            "thread_id": "thread-001",
            "user_id": "user-001",
            "raw_query": "有 M 码吗",
            "rewritten_query": "黑色通勤连衣裙有 M 码吗",
            "messages": [],
        }
    )

    assert captured_queries == ["黑色通勤连衣裙有 M 码吗"]
    assert result == {"retrieval_results": []}
