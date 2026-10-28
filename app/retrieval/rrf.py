from collections.abc import Sequence

from app.retrieval.models import RankedProduct


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankedProduct]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[RankedProduct]:
    if k <= 0:
        raise ValueError("RRF k must be greater than zero")
    if top_k is not None and top_k <= 0:
        return []

    fused_scores: dict[str, float] = {}
    for ranking in rankings:
        best_ranks: dict[str, int] = {}
        for result in ranking:
            product_id = result["product_id"]
            rank = result["rank"]
            if rank <= 0:
                raise ValueError("Product rank must be greater than zero")
            best_ranks[product_id] = min(
                rank,
                best_ranks.get(product_id, rank),
            )

        for product_id, rank in best_ranks.items():
            fused_scores[product_id] = fused_scores.get(
                product_id,
                0.0,
            ) + 1 / (k + rank)

    ordered_results = sorted(
        fused_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    if top_k is not None:
        ordered_results = ordered_results[:top_k]

    return [
        {
            "product_id": product_id,
            "rank": rank,
            "score": score,
        }
        for rank, (product_id, score) in enumerate(
            ordered_results,
            start=1,
        )
    ]
