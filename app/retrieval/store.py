import json
import sqlite3
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import ValidationError

from app.retrieval.ingestion import (
    ProductInput,
    calculate_product_md5,
    normalize_product,
)
from app.retrieval.models import ProductDocument, ProductDocumentIndex

SaveStatus = Literal["added", "skipped", "updated"]


class ImportReport(TypedDict):
    added: int
    skipped: int
    updated: int
    failed: int


class ProductStore:
    def __init__(
        self,
        db_path: str | Path,
        *,
        document_index: ProductDocumentIndex | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self._document_index = document_index
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS products (
                    product_id TEXT PRIMARY KEY,
                    content_md5 TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    color TEXT NOT NULL,
                    sizes_json TEXT NOT NULL,
                    price TEXT NOT NULL,
                    style TEXT NOT NULL,
                    description TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save_product(self, product: ProductInput) -> SaveStatus:
        product_id = product.product_id.strip()
        content_md5 = calculate_product_md5(product)
        normalized_text = normalize_product(product)

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT content_md5 FROM products WHERE product_id = ?",
                (product_id,),
            ).fetchone()
            if existing is not None and existing[0] == content_md5:
                return "skipped"

            duplicate = connection.execute(
                """
                SELECT product_id
                FROM products
                WHERE content_md5 = ? AND product_id != ?
                """,
                (content_md5, product_id),
            ).fetchone()
            if duplicate is not None:
                return "skipped"

            values = (
                content_md5,
                product.title.strip(),
                product.category.strip(),
                product.color.strip(),
                json.dumps(product.sizes, ensure_ascii=False),
                format(product.price, "f"),
                product.style.strip(),
                product.description.strip(),
                normalized_text,
            )

            if existing is not None:
                connection.execute(
                    """
                    UPDATE products
                    SET content_md5 = ?,
                        title = ?,
                        category = ?,
                        color = ?,
                        sizes_json = ?,
                        price = ?,
                        style = ?,
                        description = ?,
                        normalized_text = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE product_id = ?
                    """,
                    (*values, product_id),
                )
                status: SaveStatus = "updated"
            else:
                connection.execute(
                    """
                    INSERT INTO products (
                        product_id,
                        content_md5,
                        title,
                        category,
                        color,
                        sizes_json,
                        price,
                        style,
                        description,
                        normalized_text
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (product_id, *values),
                )
                status = "added"

        if self._document_index is not None:
            self._document_index.upsert_documents(
                [
                    {
                        "product_id": product_id,
                        "text": normalized_text,
                    }
                ]
            )
        return status

    def import_products(
        self,
        products: Iterable[ProductInput | Mapping[str, object]],
    ) -> ImportReport:
        report: ImportReport = {
            "added": 0,
            "skipped": 0,
            "updated": 0,
            "failed": 0,
        }

        for item in products:
            try:
                product = (
                    item
                    if isinstance(item, ProductInput)
                    else ProductInput.model_validate(item)
                )
            except ValidationError:
                report["failed"] += 1
                continue

            status = self.save_product(product)
            report[status] += 1

        return report

    def list_documents(self) -> list[ProductDocument]:
        return self.list_documents_filtered()

    def list_documents_filtered(
        self,
        *,
        category: str | None = None,
        color: str | None = None,
        price_min: Decimal | None = None,
        price_max: Decimal | None = None,
        size: str | None = None,
    ) -> list[ProductDocument]:
        clauses: list[str] = []
        params: list[object] = []
        if category:
            clauses.append("category = ?")
            params.append(category)
        if color:
            clauses.append("color = ?")
            params.append(color)
        if price_min is not None:
            clauses.append("CAST(price AS REAL) >= ?")
            params.append(float(price_min))
        if price_max is not None:
            clauses.append("CAST(price AS REAL) <= ?")
            params.append(float(price_max))
        if size:
            clauses.append("sizes_json LIKE ?")
            params.append(f'%"{size.upper()}"%')

        sql = "SELECT product_id, normalized_text FROM products"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY product_id"

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()

        return [
            {
                "product_id": product_id,
                "text": normalized_text,
            }
            for product_id, normalized_text in rows
        ]

    def get_products(self, product_ids: list[str]) -> list[dict[str, object]]:
        if not product_ids:
            return []

        placeholders = ",".join("?" for _ in product_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT product_id, title, category, color, sizes_json,
                       price, style, description
                FROM products
                WHERE product_id IN ({placeholders})
                """,
                product_ids,
            ).fetchall()

        products_by_id = {
            product_id: {
                "product_id": product_id,
                "title": title,
                "category": category,
                "color": color,
                "sizes": json.loads(sizes_json),
                "price": price,
                "style": style,
                "description": description,
            }
            for (
                product_id,
                title,
                category,
                color,
                sizes_json,
                price,
                style,
                description,
            ) in rows
        }
        return [
            products_by_id[product_id]
            for product_id in product_ids
            if product_id in products_by_id
        ]
