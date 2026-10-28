from decimal import Decimal

from langchain_core.embeddings import Embeddings

from app.nodes import retrieval as retrieval_module
from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.store import ProductStore

_PRODUCTS = [
    {
        "product_id": "SKU-001",
        "title": "黑色修身连衣裙",
        "category": "连衣裙",
        "color": "黑色",
        "sizes": ["S", "M", "L"],
        "price": "299",
        "style": "通勤",
        "description": "修身显瘦",
    },
    {
        "product_id": "SKU-002",
        "title": "黑色真丝晚礼连衣裙",
        "category": "连衣裙",
        "color": "黑色",
        "sizes": ["M"],
        "price": "829",
        "style": "宴会",
        "description": "真丝面料",
    },
    {
        "product_id": "SKU-003",
        "title": "红色法式连衣裙",
        "category": "连衣裙",
        "color": "红色",
        "sizes": ["S", "M"],
        "price": "459",
        "style": "复古",
        "description": "方领碎花",
    },
]


def _store(tmp_path) -> ProductStore:
    store = ProductStore(tmp_path / "products.db")
    store.import_products(_PRODUCTS)
    return store


def _ids(documents) -> list[str]:
    return [document["product_id"] for document in documents]


def test_price_max_filter_excludes_overpriced(tmp_path):
    documents = _store(tmp_path).list_documents_filtered(price_max=Decimal(500))

    assert _ids(documents) == ["SKU-001", "SKU-003"]


def test_color_and_category_filter(tmp_path):
    store = _store(tmp_path)

    assert _ids(store.list_documents_filtered(color="红色")) == ["SKU-003"]
    assert _ids(store.list_documents_filtered(category="衬衫")) == []


def test_size_filter_matches_sizes_json(tmp_path):
    store = _store(tmp_path)

    assert _ids(store.list_documents_filtered(size="L")) == ["SKU-001"]
    assert _ids(store.list_documents_filtered(size="m")) == [
        "SKU-001",
        "SKU-002",
        "SKU-003",
    ]


def test_search_product_evidence_filters_before_ranking(tmp_path, monkeypatch):
    store = _store(tmp_path)
    captured: list[list[str]] = []

    def fake_hybrid_search(query, documents, *, top_k=5):
        captured.append(_ids(documents))
        return [
            {"product_id": product_id, "rank": rank, "score": 1.0}
            for rank, product_id in enumerate(_ids(documents), start=1)
        ]

    monkeypatch.setattr(retrieval_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(
        retrieval_module,
        "ProductStore",
        lambda _path: store,
    )

    evidence = retrieval_module.search_product_evidence(
        "连衣裙",
        category="连衣裙",
        price_max=Decimal(500),
    )

    assert captured == [["SKU-001", "SKU-003"]]
    assert [item.product_id for item in evidence] == ["SKU-001", "SKU-003"]
    assert all(item.price <= Decimal(500) for item in evidence)


def test_search_product_evidence_empty_candidates_returns_empty(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("候选集为空时不应执行检索排序")

    monkeypatch.setattr(retrieval_module, "hybrid_search", fail_if_called)
    monkeypatch.setattr(retrieval_module, "ProductStore", lambda _path: store)

    assert (
        retrieval_module.search_product_evidence("连衣裙", price_max=Decimal(50))
        == []
    )


def test_chroma_search_restricts_to_candidate_ids(tmp_path):
    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    vector_store = ChromaVectorStore(tmp_path / "chroma", FakeEmbeddings())
    vector_store.upsert_documents(
        [
            {"product_id": "SKU-001", "text": "黑色 连衣裙"},
            {"product_id": "SKU-002", "text": "黑色 连衣裙"},
        ]
    )

    results = vector_store.search("黑色连衣裙", candidate_ids=["SKU-001"])

    assert [result["product_id"] for result in results] == ["SKU-001"]
    assert vector_store.search("黑色连衣裙", candidate_ids=[]) == []
