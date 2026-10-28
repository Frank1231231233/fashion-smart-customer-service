from app.retrieval.models import RankedProduct
from app.retrieval.rrf import reciprocal_rank_fusion


def _result(product_id: str, rank: int) -> RankedProduct:
    return {
        "product_id": product_id,
        "rank": rank,
        "score": 1.0,
    }


def test_rrf_merges_duplicate_products_without_duplicate_output():
    bm25 = [_result("SKU-001", 1), _result("SKU-002", 2)]
    vector = [_result("SKU-002", 1), _result("SKU-001", 2)]

    results = reciprocal_rank_fusion([bm25, vector])

    assert [result["product_id"] for result in results] == [
        "SKU-001",
        "SKU-002",
    ]
    assert len({result["product_id"] for result in results}) == 2


def test_rrf_keeps_products_found_by_only_one_retriever():
    bm25 = [_result("SKU-001", 1)]
    vector = [_result("SKU-002", 1)]

    results = reciprocal_rank_fusion([bm25, vector])

    assert {result["product_id"] for result in results} == {
        "SKU-001",
        "SKU-002",
    }


def test_rrf_breaks_score_ties_by_product_id():
    bm25 = [_result("SKU-002", 1)]
    vector = [_result("SKU-001", 1)]

    results = reciprocal_rank_fusion([bm25, vector])

    assert [result["product_id"] for result in results] == [
        "SKU-001",
        "SKU-002",
    ]
    assert results[0]["score"] == results[1]["score"]


def test_rrf_handles_empty_rankings():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
