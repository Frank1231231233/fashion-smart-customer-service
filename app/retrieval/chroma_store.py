from pathlib import Path

import chromadb
from langchain_core.embeddings import Embeddings

from app.retrieval.models import ProductDocument, RankedProduct


class ChromaVectorStore:
    def __init__(
        self,
        path: str | Path,
        embeddings: Embeddings,
        *,
        collection_name: str = "products",
    ) -> None:
        self._embeddings = embeddings
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_documents(self, documents: list[ProductDocument]) -> None:
        if not documents:
            return

        texts = [document["text"] for document in documents]
        vectors = self._embeddings.embed_documents(texts)
        if len(vectors) != len(documents):
            raise ValueError("Embedding count does not match document count")

        self._collection.upsert(
            ids=[document["product_id"] for document in documents],
            documents=texts,
            metadatas=[
                {"product_id": document["product_id"]} for document in documents
            ],
            embeddings=vectors,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_ids: list[str] | None = None,
    ) -> list[RankedProduct]:
        if (
            not query.strip()
            or top_k <= 0
            or self._collection.count() == 0
            or (candidate_ids is not None and not candidate_ids)
        ):
            return []

        where = (
            {"product_id": {"$in": candidate_ids}}
            if candidate_ids is not None
            else None
        )
        query_vector = self._embeddings.embed_query(query)
        response = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self._collection.count()),
            where=where,
            include=["distances"],
        )
        ids = response["ids"][0]
        distances = response["distances"][0] if response["distances"] else []

        return [
            {
                "product_id": product_id,
                "rank": rank,
                "score": 1.0 - float(distance),
            }
            for rank, (product_id, distance) in enumerate(
                zip(ids, distances, strict=True),
                start=1,
            )
        ]
