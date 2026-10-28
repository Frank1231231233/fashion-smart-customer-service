import re

import jieba
from rank_bm25 import BM25Okapi

from app.retrieval.models import ProductDocument, RankedProduct

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    normalized_text = text.casefold()
    return [
        token
        for segment in _TOKEN_PATTERN.findall(normalized_text)
        for token in jieba.lcut(segment)
        if token.strip()
    ]


class BM25Retriever:
    def __init__(self, documents: list[ProductDocument]) -> None:
        self._documents = documents
        self._tokenized_documents = [
            tokenize(document["text"]) for document in documents
        ]
        self._index = (
            BM25Okapi(self._tokenized_documents) if self._tokenized_documents else None
        )

    def search(self, query: str, top_k: int = 5) -> list[RankedProduct]:
        query_tokens = tokenize(query)
        if self._index is None or not query_tokens or top_k <= 0:
            return []

        query_token_set = set(query_tokens)
        scores = self._index.get_scores(query_tokens)
        candidates = [
            (document["product_id"], float(score))
            for document, document_tokens, score in zip(
                self._documents,
                self._tokenized_documents,
                scores,
                strict=True,
            )
            if query_token_set.intersection(document_tokens)
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
