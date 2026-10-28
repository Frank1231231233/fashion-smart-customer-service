import math

from langchain_core.embeddings import Embeddings

from app.retrieval.models import ProductDocument, RankedProduct


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise ValueError("Embedding dimensions do not match")

    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0

    dot_product = sum(left * right for left, right in zip(first, second, strict=True))
    return dot_product / (first_norm * second_norm)


class VectorRetriever:
    def __init__(
        self,
        documents: list[ProductDocument],
        embeddings: Embeddings,
    ) -> None:
        self._documents = documents
        self._embeddings = embeddings
        self._vectors = (
            embeddings.embed_documents([document["text"] for document in documents])
            if documents
            else []
        )

        if len(self._vectors) != len(documents):
            raise ValueError("Embedding count does not match document count")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RankedProduct]:
        if not query.strip() or not self._documents or top_k <= 0:
            return []

        query_vector = self._embeddings.embed_query(query)
        candidates = [
            (
                document["product_id"],
                _cosine_similarity(query_vector, document_vector),
            )
            for document, document_vector in zip(
                self._documents,
                self._vectors,
                strict=True,
            )
        ]
        candidates = [
            candidate for candidate in candidates if candidate[1] >= min_score
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))

        return [
            {
                "product_id": product_id,
                "rank": rank,
                "score": score,
            }
            for rank, (product_id, score) in enumerate(
                candidates[:top_k],
                start=1,
            )
        ]
