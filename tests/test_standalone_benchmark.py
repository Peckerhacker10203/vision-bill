"""Run benchmark SQL against a test-owned, in-memory relational schema."""

import re
import sqlite3

import pytest

from scripts.standalone_benchmark import fetch_receipt_details, fetch_verified_receipts


class LocalQueryPool:
    """Async boundary adapter; SQLite executes the script's actual SQL and bindings."""

    def __init__(self, connection):
        self.connection = connection

    async def fetch(self, query, *params):
        return self.connection.execute(re.sub(r"\$(\d+)", r"?\1", query), params).fetchall()

    async def fetchrow(self, query, *params):
        return self.connection.execute(re.sub(r"\$(\d+)", r"?\1", query), params).fetchone()


@pytest.fixture
def query_pool():
    # Names match the app's migration-owned schema. No application DB is opened.
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE receipts (
            id TEXT, merchant_name TEXT, date TEXT, confidence INTEGER, category TEXT,
            subtotal NUMERIC, tax_total NUMERIC, total NUMERIC, currency TEXT,
            payment_method TEXT, merchant_address TEXT, receipt_number TEXT,
            time TEXT, discount_total NUMERIC, tip NUMERIC, verified BOOLEAN
        );
        CREATE TABLE images (receipt_id TEXT, image_path TEXT, original_filename TEXT);
        CREATE TABLE line_items (
            id TEXT, receipt_id TEXT, description TEXT, quantity NUMERIC,
            unit_price NUMERIC, total_price NUMERIC, tags TEXT, spending_category TEXT
        );
        CREATE TABLE taxes (id TEXT, receipt_id TEXT, name TEXT, rate NUMERIC, amount NUMERIC);
        INSERT INTO receipts VALUES
            ('groceries-old', 'Old Shop', '2026-11-01', 95, 'grocery',
             10, 1, 11, 'EUR', 'cash', NULL, NULL, NULL, 0, 0, true),
            ('groceries-new', 'New Shop', '2026-11-02', 96, 'grocery',
             20, 2, 22, 'EUR', 'cash', NULL, NULL, NULL, 0, 0, true),
            ('other-new', 'Other Shop', '2026-11-03', 97, 'other',
             30, 3, 33, 'EUR', 'cash', NULL, NULL, NULL, 0, 0, true),
            ('unverified', 'Draft Shop', '2026-11-04', 90, 'grocery',
             40, 4, 44, 'EUR', 'cash', NULL, NULL, NULL, 0, 0, false);
        INSERT INTO images VALUES
            ('groceries-old', '/app/uploads/old.png', 'old.png'),
            ('groceries-new', '/app/uploads/new.png', 'new.png'),
            ('other-new', '/app/uploads/other.png', 'other.png'),
            ('unverified', '/app/uploads/draft.png', 'draft.png');
        INSERT INTO line_items VALUES
            ('item', 'groceries-new', 'Test product', 2, 10, 20, NULL, 'grocery');
        INSERT INTO taxes VALUES ('tax', 'groceries-new', 'VAT', 0.1, 2);
    """)
    try:
        yield LocalQueryPool(connection)
    finally:
        connection.close()


async def test_verified_receipts_use_stored_image_path(query_pool):
    rows = await fetch_verified_receipts(query_pool)

    assert [row["id"] for row in rows] == ["other-new", "groceries-new", "groceries-old"]
    assert rows[1]["image_path"] == "/app/uploads/new.png"
    assert rows[1]["original_filename"] == "new.png"


async def test_receipt_details_load_persisted_taxes(query_pool):
    receipt = await fetch_receipt_details(query_pool, "groceries-new")

    assert receipt is not None
    assert receipt.merchant_name == "New Shop"
    assert receipt.line_items[0].description == "Test product"
    assert receipt.line_items[0].total_price == 20
    assert receipt.taxes[0].name == "VAT"
    assert receipt.taxes[0].amount == 2
    assert receipt.total == 22


async def test_missing_receipt_details_return_none(query_pool):
    assert await fetch_receipt_details(query_pool, "missing") is None


@pytest.mark.parametrize(
    ("category", "limit", "expected_ids"),
    [
        ("grocery", None, ["groceries-new", "groceries-old"]),
        (None, 1, ["other-new"]),
        ("grocery", 1, ["groceries-new"]),
    ],
)
async def test_category_and_limit_bind_independently(query_pool, category, limit, expected_ids):
    rows = await fetch_verified_receipts(query_pool, category=category, limit=limit)

    assert [row["id"] for row in rows] == expected_ids
