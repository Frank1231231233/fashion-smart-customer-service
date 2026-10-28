"""把一段商品文本结构化后同时写入 SQLite 与 Chroma 向量库。

用法: uv run python -m scripts.ingest_text "商品描述文本..."
缺必填信息(类目/颜色/价格)时拒绝入库,不编造。
"""

import sys

from app.config import get_settings
from app.nodes.retrieval import get_product_vector_store
from app.retrieval.extractor import draft_to_product, extract_product_draft
from app.retrieval.store import ProductStore


def _next_product_id(store: ProductStore) -> str:
    numbers = [
        int(document["product_id"].split("-", 1)[1])
        for document in store.list_documents()
        if document["product_id"].startswith("SKU-")
        and document["product_id"].split("-", 1)[1].isdigit()
    ]
    return f"SKU-{(max(numbers) + 1) if numbers else 1:03d}"


def main() -> None:
    text = " ".join(sys.argv[1:]).strip() or sys.stdin.read().strip()
    if not text:
        raise SystemExit("用法: uv run python -m scripts.ingest_text <商品文本>")

    settings = get_settings()
    store = ProductStore(
        settings.product_db_path,
        document_index=get_product_vector_store(),
    )

    draft = extract_product_draft(text)
    product_id = draft.product_id or _next_product_id(store)
    product = draft_to_product(draft, product_id)

    status = store.save_product(product)
    print(f"{status}: {product.model_dump_json()}")


if __name__ == "__main__":
    main()
