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
        CREATE TABLE images (receipt_id TEXT, image_path TEXT, original_filename TEXT, additional_images TEXT);
        CREATE TABLE line_items (
            id TEXT, receipt_id TEXT, description TEXT, quantity NUMERIC,
            unit_price NUMERIC, total_price NUMERIC, tags TEXT, spending_category TEXT, position INTEGER DEFAULT 0
        );
        CREATE TABLE taxes (id TEXT, receipt_id TEXT, name TEXT, rate NUMERIC, amount NUMERIC, position INTEGER DEFAULT 0);
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
            ('groceries-old', '/app/uploads/old.png', 'old.png', '[]'),
            ('groceries-new', '/app/uploads/new.png', 'new.png', '[]'),
            ('other-new', '/app/uploads/other.png', 'other.png', '[]'),
            ('unverified', '/app/uploads/draft.png', 'draft.png', '[]');
        INSERT INTO line_items VALUES
            ('item', 'groceries-new', 'Test product', 2, 10, 20, NULL, 'grocery', 0);
        INSERT INTO taxes VALUES ('tax', 'groceries-new', 'VAT', 0.1, 2, 0);
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


async def test_details_keep_tax_rate_and_position_order(query_pool):
    query_pool.connection.execute("INSERT INTO line_items VALUES ('aaa','groceries-new','Later product',1,2,2,NULL,'unknown',1)")
    result = await fetch_receipt_details(query_pool, 'groceries-new')
    assert result.taxes[0].rate == .1
    assert [item.description for item in result.line_items] == ['Test product','Later product']


async def test_standalone_uses_all_ordered_photos_in_common_pipeline(settings, tmp_path):
    from scripts import standalone_benchmark as script
    from test_extraction_pipeline import formatter, sidecar
    from PIL import Image
    paths = [tmp_path / f'{i}.png' for i in range(2)]
    for p in paths:
        Image.new('RGB',(3,4)).save(p)
    ocr, llm = sidecar(), formatter(False)
    assert hasattr(script, 'execute_photos'), 'common complete submission adapter missing'
    result = await script.execute_photos(llm, ocr, 'fixture', paths, settings.pg)
    assert result.receipt and result.metadata['source_image_count'] == 2
    assert [call.kwargs['photo_index'] for call in ocr.extract_page.await_args_list] == [0,1]


@pytest.mark.parametrize("failure", ["inference", "reference", "source"])
async def test_standalone_failure_denominator_and_logs_exclude_receipt_content(settings,tmp_path,monkeypatch,caplog,failure):
    import sys
    import logging
    import hashlib
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from uuid import uuid4
    from scripts import standalone_benchmark as script
    from test_experiment_scoring import receipt
    from vision_bill.service.experiment_execution import ExperimentResult
    from vision_bill.service.experiment_scoring import OBSERVABLE_FIELDS
    owner=uuid4()
    monkeypatch.setattr(sys,'argv',['benchmark','--model','fixture','--user-id',str(owner)])
    args=script.parse_args()
    path=tmp_path/'source';path.write_bytes(b'image')
    digest=hashlib.sha256(b'image').hexdigest()
    pool=SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(script,'get_pg_settings',lambda args:settings.pg)
    monkeypatch.setattr(script,'get_llm_settings',lambda args:settings.llm.model_copy(update={'model_name':'fixture'}))
    monkeypatch.setattr(script,'get_ocr_settings',lambda args:settings.ocr)
    provider=SimpleNamespace(check_connection=AsyncMock(return_value=True))
    provider.for_experiment=lambda options: SimpleNamespace(build_prompt=lambda *a,**k: "prompt")
    monkeypatch.setattr(script,'get_llm_provider',lambda settings:provider)
    monkeypatch.setattr(script,'get_ocr_provider',lambda settings:provider)
    monkeypatch.setattr(script.asyncpg,'create_pool',AsyncMock(return_value=pool))
    actual=receipt().model_copy(update={'tip':__import__('decimal').Decimal('0')})
    good=actual.model_dump(mode='json')
    def reference(expected):
        data={'receipt_identity':str(uuid4()),'expected':expected,'sources':[{'path':str(path),'sha256':digest}], 'scored_fields':list(OBSERVABLE_FIELDS)}
        return SimpleNamespace(**data,model_dump=lambda **kwargs:data)
    refs=[reference(good.copy()),reference(good.copy())]
    if failure=='reference': refs[0].expected['date']='malformed'
    if failure=='source': refs[0].sources[0]['sha256']='0'*64
    monkeypatch.setattr(script,'freeze_references',AsyncMock(return_value=refs))
    identity={'requested_model':{'id':'fixture','digest':'d'},'pipeline':{'ocr_runtime':{'profile_id':'p','runtime_fingerprint':'r'}}}
    monkeypatch.setattr(script,'execution_identity',AsyncMock(return_value=identity))
    metadata={'model_id':'fixture','model_digest':'d','ordered_source_sha256':[digest],'used_pipeline':'ocr','profile_id':'p','runtime_fingerprint':'r'}
    result=ExperimentResult(actual,1,12,metadata)
    execute=AsyncMock(side_effect=[ValueError('PRIVATE_PROVIDER_TEXT'),result] if failure=='inference' else [result])
    monkeypatch.setattr(script,'execute_photos',execute)
    with caplog.at_level(logging.INFO): summary=await script.run_benchmark(args)
    assert summary.total_receipts==2 and summary.failed==1 and summary.successful==1
    assert summary.avg_score==1 and summary.all_case_average_score==.5
    assert summary.full_receipt_correct_rate==.5
    assert summary.results[1].metadata==metadata
    assert summary.manifest['scope']=={'user_id':str(owner),'can_see_all':False}
    assert len(summary.manifest_sha256)==64
    assert summary.item_omissions==(0 if failure=='reference' else len(good['line_items']))
    assert 'PRIVATE_PROVIDER_TEXT' not in caplog.text
    # Extraction receives only ordered images/settings, never reference data.
    assert execute.call_args.args[3]==[path]
    assert set(execute.call_args.kwargs)=={'ocr_settings'}


async def test_actual_asyncpg_time_type_preserves_subminute_precision(query_pool,monkeypatch):
    from datetime import time
    fetch=query_pool.fetchrow
    async def pg_row(*args):
        row=dict(await fetch(*args));row['time']=time(14,30,15,1);return row
    monkeypatch.setattr(query_pool,'fetchrow',pg_row)
    result=await fetch_receipt_details(query_pool,'groceries-new')
    assert result.time == '14:30:15.000001'
