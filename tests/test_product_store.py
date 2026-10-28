import sqlite3
from decimal import Decimal

from app.retrieval.ingestion import ProductInput, calculate_product_md5
from app.retrieval.models import ProductDocument
from app.retrieval.store import ProductStore


def test_product_store_creates_products_table(tmp_path):
    db_path = tmp_path / "products.db"

    ProductStore(db_path)
    ProductStore(db_path)

    assert db_path.exists()
    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(products)").fetchall()
        }

    assert columns == {
        "product_id",
        "content_md5",
        "title",
        "category",
        "color",
        "sizes_json",
        "price",
        "style",
        "description",
        "normalized_text",
        "updated_at",
    }


def _product(product_id: str = "SKU-001") -> ProductInput:
    return ProductInput(
        product_id=product_id,
        title="通勤连衣裙",
        category="连衣裙",
        color="黑色",
        sizes=["M", "L"],
        price="299.00",
        style="通勤",
        description="修身版型",
    )


def test_save_product_adds_skips_and_updates(tmp_path):
    db_path = tmp_path / "products.db"
    store = ProductStore(db_path)
    original = _product()

    assert store.save_product(original) == "added"
    assert store.save_product(original) == "skipped"

    updated = original.model_copy(update={"color": "白色"})
    assert store.save_product(updated) == "updated"

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT color, content_md5, normalized_text
            FROM products
            WHERE product_id = ?
            """,
            ("SKU-001",),
        ).fetchone()

    assert row is not None
    assert row[0] == "白色"
    assert row[1] == calculate_product_md5(updated)
    assert "color=白色" in row[2]


def test_save_product_skips_duplicate_content_under_another_id(tmp_path):
    db_path = tmp_path / "products.db"
    store = ProductStore(db_path)

    assert store.save_product(_product("SKU-001")) == "added"
    assert store.save_product(_product("SKU-002")) == "skipped"

    with sqlite3.connect(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]

    assert count == 1


def test_import_products_reports_each_result(tmp_path):
    store = ProductStore(tmp_path / "products.db")
    original = _product("SKU-001")
    changed = original.model_copy(update={"color": "白色"})
    invalid = {
        "product_id": "SKU-BAD",
        "title": "",
        "category": "连衣裙",
        "color": "黑色",
        "sizes": ["M"],
        "price": "-1",
    }

    report = store.import_products(
        [
            original,
            original,
            changed,
            invalid,
        ]
    )

    assert report == {
        "added": 1,
        "skipped": 1,
        "updated": 1,
        "failed": 1,
    }


def test_importing_same_batch_twice_adds_nothing_second_time(tmp_path):
    store = ProductStore(tmp_path / "products.db")
    products = [
        _product("SKU-001"),
        _product("SKU-002").model_copy(
            update={
                "title": "休闲卫衣",
                "category": "上衣",
                "price": Decimal("199.00"),
            }
        ),
    ]

    first_report = store.import_products(products)
    second_report = store.import_products(products)

    assert first_report == {
        "added": 2,
        "skipped": 0,
        "updated": 0,
        "failed": 0,
    }
    assert second_report == {
        "added": 0,
        "skipped": 2,
        "updated": 0,
        "failed": 0,
    }


def test_list_documents_returns_search_text(tmp_path):
    store = ProductStore(tmp_path / "products.db")
    product = _product()
    store.save_product(product)

    documents = store.list_documents()

    assert documents == [
        {
            "product_id": "SKU-001",
            "text": (
                "title=通勤连衣裙\n"
                "category=连衣裙\n"
                "color=黑色\n"
                "sizes=L,M\n"
                "price=299\n"
                "style=通勤\n"
                "description=修身版型"
            ),
        }
    ]


def test_product_store_indexes_added_and_updated_products(tmp_path):
    indexed_batches: list[list[ProductDocument]] = []

    class SpyDocumentIndex:
        def upsert_documents(self, documents: list[ProductDocument]) -> None:
            indexed_batches.append(documents)

    store = ProductStore(
        tmp_path / "products.db",
        document_index=SpyDocumentIndex(),
    )
    original = _product()
    updated = original.model_copy(update={"color": "白色"})

    assert store.save_product(original) == "added"
    assert store.save_product(original) == "skipped"
    assert store.save_product(updated) == "updated"

    assert len(indexed_batches) == 2
    assert indexed_batches[0][0]["product_id"] == "SKU-001"
    assert "color=黑色" in indexed_batches[0][0]["text"]
    assert "color=白色" in indexed_batches[1][0]["text"]
