from app.retrieval.bm25 import BM25Retriever
from app.retrieval.models import ProductDocument


def _documents() -> list[ProductDocument]:
    return [
        {
            "product_id": "SKU-001",
            "text": "黑色 通勤 连衣裙 修身 面试",
        },
        {
            "product_id": "SKU-002",
            "text": "白色 休闲 连衣裙 宽松 度假",
        },
        {
            "product_id": "SKU-003",
            "text": "黑色 通勤 衬衫 正装",
        },
        {
            "product_id": "SKU-004",
            "text": "蓝色 牛仔裤 直筒",
        },
        {
            "product_id": "SKU-005",
            "text": "红色 运动 外套 防风",
        },
    ]


def test_bm25_returns_ranked_product_ids():
    retriever = BM25Retriever(_documents())

    results = retriever.search("黑色通勤连衣裙", top_k=3)

    assert results[0]["product_id"] == "SKU-001"
    assert [result["rank"] for result in results] == [1, 2, 3]
    assert all(isinstance(result["score"], float) for result in results)
    assert len({result["product_id"] for result in results}) == len(results)


def test_bm25_respects_top_k():
    retriever = BM25Retriever(_documents())

    results = retriever.search("连衣裙", top_k=1)

    assert len(results) == 1
    assert results[0]["rank"] == 1


def test_bm25_returns_empty_for_no_match_or_no_documents():
    retriever = BM25Retriever(_documents())

    assert retriever.search("羽绒服") == []
    assert BM25Retriever([]).search("连衣裙") == []
