from decimal import Decimal
from functools import lru_cache

from app.config import get_settings
from app.graph.state import AgentState
from app.llm import get_embedding_model
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.chroma_store import ChromaVectorStore
from app.retrieval.models import ProductDocument, RankedProduct
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.store import ProductStore
from app.tools.product import ProductEvidence


def load_product_documents() -> list[ProductDocument]:
    settings = get_settings()
    return ProductStore(settings.product_db_path).list_documents()


@lru_cache
def get_product_vector_store() -> ChromaVectorStore:
    settings = get_settings()
    return ChromaVectorStore(
        settings.chroma_path,
        get_embedding_model(),
        collection_name=settings.chroma_collection,
    )


def hybrid_search(
    query: str,
    documents: list[ProductDocument],
    *,
    top_k: int = 5,
) -> list[RankedProduct]:
    if not documents:
        return []

    bm25_results = BM25Retriever(documents).search(query, top_k=top_k)
    candidate_ids = [document["product_id"] for document in documents]
    vector_results = get_product_vector_store().search(
        query,
        top_k=top_k,
        candidate_ids=candidate_ids,
    )
    return reciprocal_rank_fusion(
        [bm25_results, vector_results],
        top_k=top_k,
    )


def retrieval_node(state: AgentState) -> dict[str, object]:
    query = state.get("rewritten_query", state["raw_query"])
    results = hybrid_search(query, load_product_documents())
    return {"retrieval_results": results}


def search_product_evidence(
    query: str,
    *,
    top_k: int = 5,
    category: str | None = None,
    color: str | None = None,
    price_min: Decimal | None = None,
    price_max: Decimal | None = None,
    size: str | None = None,
) -> list[ProductEvidence]:
    settings = get_settings()
    store = ProductStore(settings.product_db_path)

    has_filters = any(
        value is not None
        for value in (category, color, price_min, price_max, size)
    )
    documents = (
        store.list_documents_filtered(
            category=category,
            color=color,
            price_min=price_min,
            price_max=price_max,
            size=size,
        )
        if has_filters
        else store.list_documents()
    )
    if not documents:
        return []

    ranked = hybrid_search(query, documents, top_k=top_k)
    product_ids = [result["product_id"] for result in ranked]
    return [
        ProductEvidence.model_validate(product)
        for product in store.get_products(product_ids)
    ]
