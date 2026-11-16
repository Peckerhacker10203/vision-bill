"""Locked review gate and exact storage on disposable PostgreSQL 18."""

import asyncio
import json
from decimal import Decimal
from uuid import uuid4

import pytest

from tests.e2e import test_workflow_db_e2e as workflow_tests
from vision_bill.api.receipts import ReceiptUpdate
from vision_bill.model.review import AmountPolicy, FindingAcknowledgement
from vision_bill.model.workflow import ReviewRequest

db = workflow_tests.db
postgres = workflow_tests.postgres
workflow_db = workflow_tests.workflow_db
migrate = workflow_tests.migrate

pytestmark = pytest.mark.e2e
POLICY = AmountPolicy(
    tax_basis="gross",
    discount_treatment="included_in_lines",
    tip_treatment="none",
    subtotal_origin="printed",
    tax_information="explicit_zero",
)


async def seed(w, scope, tmp_path, *, total="10", verified=False):
    w._receipts._pool = w.pool
    original = tmp_path / str(uuid4())
    original.write_bytes(b"original receipt bytes")
    payload = {
        "confidence": 90,
        "merchant_name": "Shop",
        "date": "2026-01-01",
        "currency": "EUR",
        "subtotal": total,
        "total": total,
        "line_items": [
            {"description": "Item", "quantity": 1, "unit_price": total, "total_price": total}
        ],
        "amount_policy": POLICY.model_dump(),
        "expected_operation_revision": 1,
    }
    body = ReceiptUpdate.model_validate(payload)
    async with w.pool.acquire() as conn:
        image = await conn.fetchval(
            "INSERT INTO images(image_path,user_id) VALUES($1,$2) RETURNING id",
            str(original),
            scope.user_id,
        )
        row = await w._receipts.insert_receipt(
            conn,
            body,
            image_id=image,
            user_id=scope.user_id,
            status="verified" if verified else "unverified",
            verified=verified,
        )
        await conn.execute(
            "UPDATE receipts SET amount_policy=$2::jsonb WHERE id=$1",
            row.id,
            POLICY.model_dump_json(),
        )
    return row, body, original


async def test_subcent_storage_and_guarded_downgrade(workflow_db, tmp_path):
    w, scope = workflow_db
    row, _body, _ = await seed(w, scope, tmp_path, total="10.000001")
    detail = await w._receipts.get_receipt_with_details(row.id)
    assert detail.receipt.total == Decimal("10.000001")
    assert detail.line_items[0].unit_price == Decimal("10.000001")
    refusal = migrate(w._settings, "0013", "downgrade", check=False)
    assert refusal.returncode != 0


async def test_clean_no_body_verify_records_human_history(workflow_db, tmp_path):
    w, scope = workflow_db
    row, _, path = await seed(w, scope, tmp_path)
    saved = await w._receipts.verify_receipt(row.id, ReviewRequest(), scope=scope)
    assert saved.verified and saved.verification_source == "human"
    assert saved.operation_revision == 2
    assert path.read_bytes() == b"original receipt bytes"
    async with w.pool.acquire() as conn:
        history = await conn.fetch(
            "SELECT * FROM receipt_review_revisions WHERE receipt_id=$1", row.id
        )
    assert len(history) == 1 and history[0]["event_kind"] == "verified"
    with pytest.raises(ValueError, match="already_verified"):
        await w._receipts.verify_receipt(row.id, ReviewRequest(), scope=scope)


async def test_material_cas_revokes_verification_and_invalidates_acks(workflow_db, tmp_path):
    w, scope = workflow_db
    row, body, _ = await seed(w, scope, tmp_path, verified=True)
    body.total = Decimal(12)
    saved = await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    assert saved.operation_revision == 2 and saved.review_revision == 2
    assert not saved.verified and saved.status == "unverified"
    assert [f.code for f in saved.findings] == ["amount_discrepancy"]
    with pytest.raises(ValueError, match="revision"):
        await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    request = ReviewRequest(
        expected_operation_revision=2,
        expected_review_revision=2,
        acknowledgements=[
            FindingAcknowledgement(
                finding_id=saved.findings[0].id, comment="Printed amount checked"
            )
        ],
    )
    verified = await w._receipts.verify_receipt(row.id, request, scope=scope)
    assert verified.verified and verified.operation_revision == 3
    body.expected_operation_revision = 3
    body.merchant_name = "Other"
    edited = await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    assert not edited.verified and edited.review_revision == 3 and edited.acknowledgements == []
    request.expected_operation_revision = 4
    request.expected_review_revision = 3
    with pytest.raises(ValueError, match="acknowledgements"):
        await w._receipts.verify_receipt(row.id, request, scope=scope)


async def test_missing_additional_legacy_original_cannot_be_acknowledged(workflow_db, tmp_path):
    w, scope = workflow_db
    row, _, original = await seed(w, scope, tmp_path)
    async with w.pool.acquire() as conn:
        await conn.execute(
            "UPDATE images SET additional_images=$2::jsonb WHERE id=$1",
            row.image_id,
            json.dumps([{"image_path": str(tmp_path / "missing")}]),
        )
    with pytest.raises(ValueError, match="integrity"):
        await w._receipts.verify_receipt(row.id, ReviewRequest(), scope=scope)
    assert original.exists()


@pytest.mark.parametrize("competitor", ["verify", "put"])
async def test_competing_put_verify_only_one_revision_wins(workflow_db, tmp_path, competitor):
    w, scope = workflow_db
    row, body, _ = await seed(w, scope, tmp_path)
    body.total = Decimal(12)
    competing_body = body.model_copy(deep=True)
    competing_body.merchant_name = "Competing edit"
    second = (
        w._receipts.verify_receipt(
            row.id,
            ReviewRequest(expected_operation_revision=1, expected_review_revision=1),
            scope=scope,
        )
        if competitor == "verify"
        else w._receipts.update_receipt(row.id, competing_body, user_id=scope.user_id)
    )
    results = await asyncio.gather(
        w._receipts.update_receipt(row.id, body, user_id=scope.user_id),
        second,
        return_exceptions=True,
    )
    assert sum(isinstance(result, ValueError) for result in results) == 1
    detail = await w._receipts.get_receipt_with_details(row.id)
    assert detail.receipt.operation_revision == 2
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM receipt_review_revisions WHERE receipt_id=$1", row.id
            )
            == 1
        )


async def test_foreign_collection_rolls_back_money_children_and_owner_learning(
    workflow_db, tmp_path
):
    w, scope = workflow_db
    row, body, _ = await seed(w, scope, tmp_path)
    async with w.pool.acquire() as conn:
        outsider = await conn.fetchval(
            "INSERT INTO users(username,hashed_password) VALUES($1,'test') RETURNING id",
            str(uuid4()),
        )
        foreign = await conn.fetchval(
            "INSERT INTO collections(user_id,name) VALUES($1,'private') RETURNING id", outsider
        )
    body.total = Decimal(12)
    body.collection_ids = [foreign]
    body.line_items[0].spending_category = "milk"
    body.line_items[0].remember_category = True
    with pytest.raises(ValueError, match="collection_not_found"):
        await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    detail = await w._receipts.get_receipt_with_details(row.id)
    assert detail.receipt.total == Decimal(10) and detail.receipt.operation_revision == 1
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM item_category_aliases") == 0
        assert await conn.fetchval("SELECT count(*) FROM receipt_review_revisions") == 0


async def test_stored_minute_time_does_not_become_integrity_failure(workflow_db, tmp_path):
    w, scope = workflow_db
    row, _, _ = await seed(w, scope, tmp_path)
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE receipts SET time='14:30' WHERE id=$1", row.id)
    saved = await w._receipts.verify_receipt(row.id, ReviewRequest(), scope=scope)
    assert saved.verified


async def test_classification_collection_intent_and_admin_owner_learning(workflow_db, tmp_path):
    w, scope = workflow_db
    row, body, _ = await seed(w, scope, tmp_path, verified=True)
    detail = await w._receipts.get_receipt_with_details(row.id)
    item_id = detail.line_items[0].id
    body.line_items[0].id = item_id
    body.line_items[0].spending_category = "milk"
    body.line_items[0].remember_category = True
    body.line_items[0].tags = ["Tag"]
    async with w.pool.acquire() as conn:
        admin = await conn.fetchval(
            "INSERT INTO users(username,hashed_password) VALUES($1,'test') RETURNING id",
            str(uuid4()),
        )
        collection = await conn.fetchval(
            "INSERT INTO collections(user_id,name) VALUES($1,'mine') RETURNING id", scope.user_id
        )
    body.collection_ids = [collection]
    saved = await w._receipts.update_receipt(row.id, body, user_id=admin, can_see_all=True)
    assert saved.verified and saved.review_revision == 1 and saved.operation_revision == 2
    detail = await w._receipts.get_receipt_with_details(row.id)
    assert (
        detail.line_items[0].id == item_id and detail.line_items[0].original_description == "Item"
    )
    assert detail.line_items[0].spending_category == "milk" and detail.line_items[0].tags == ["Tag"]
    assert detail.collection_ids == [collection]
    async with w.pool.acquire() as conn:
        aliases = await conn.fetch("SELECT * FROM item_category_aliases")
    assert len(aliases) == 1 and aliases[0]["user_id"] == scope.user_id
    body.expected_operation_revision = 2
    body.collection_ids = None
    await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    assert (await w._receipts.get_receipt_with_details(row.id)).collection_ids == [collection]
    body.expected_operation_revision = 3
    body.collection_ids = []
    await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    assert (await w._receipts.get_receipt_with_details(row.id)).collection_ids == []


async def test_gate_reads_original_after_waiting_for_lock(workflow_db, tmp_path):
    w, scope = workflow_db
    row, _, path = await seed(w, scope, tmp_path)
    import asyncpg

    from vision_bill.service.image_service import ImageService

    reached, proceed = asyncio.Event(), asyncio.Event()

    async def checker(parts):
        async with w.pool.acquire() as conn, conn.transaction():
            with pytest.raises(asyncpg.LockNotAvailableError):
                await conn.execute(
                    "SELECT id FROM images WHERE id=$1 FOR UPDATE NOWAIT", row.image_id
                )
        reached.set()
        await proceed.wait()
        return await ImageService.check_originals(parts)

    w._receipts._original_checker = checker
    verification = asyncio.create_task(
        w._receipts.verify_receipt(row.id, ReviewRequest(), scope=scope)
    )
    await asyncio.wait_for(reached.wait(), 5)
    path.unlink()
    proceed.set()
    with pytest.raises(ValueError, match="integrity"):
        await verification
    assert not (await w._receipts.get_receipt_with_details(row.id)).receipt.verified


async def test_child_and_category_writes_roll_back_when_history_insert_fails(workflow_db, tmp_path):
    w, scope = workflow_db
    row, body, _ = await seed(w, scope, tmp_path)
    before = await w._receipts.get_receipt_with_details(row.id)
    body.line_items[0].description = "Renamed"
    body.line_items[0].id = before.line_items[0].id
    body.line_items[0].spending_category = "milk"
    body.line_items[0].remember_category = True
    async with w.pool.acquire() as conn:
        await conn.execute(
            "ALTER TABLE receipt_review_revisions ADD CONSTRAINT test_reject CHECK (event_kind <> 'material_edit')"
        )
    import asyncpg

    with pytest.raises(asyncpg.CheckViolationError):
        await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    after = await w._receipts.get_receipt_with_details(row.id)
    assert after == before
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM item_category_aliases") == 0


async def test_reordered_duplicate_items_keep_identity_and_reject_foreign_ids(
    workflow_db, tmp_path
):
    w, scope = workflow_db
    row, body, _ = await seed(w, scope, tmp_path)
    body.line_items.append(body.line_items[0].model_copy(deep=True))
    body.line_items[1].total_price = body.line_items[1].unit_price = Decimal(-10)
    body.total = body.subtotal = Decimal(0)
    await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    before = await w._receipts.get_receipt_with_details(row.id)
    body.expected_operation_revision = 2
    for item, old in zip(body.line_items, before.line_items, strict=True):
        item.id = old.id
    body.line_items.reverse()
    await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    after = await w._receipts.get_receipt_with_details(row.id)
    assert [i.id for i in after.line_items] == [i.id for i in reversed(before.line_items)]
    assert [i.total_price for i in after.line_items] == [Decimal(-10), Decimal(10)]
    body.expected_operation_revision = 3
    body.line_items[1].id = body.line_items[0].id
    with pytest.raises(ValueError, match="Duplicate"):
        await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    body.line_items[1].id = uuid4()
    with pytest.raises(ValueError, match="does not belong"):
        await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    assert await w._receipts.get_receipt_with_details(row.id) == after


async def test_automatic_bypass_requires_clean_finances_and_fresh_originals(workflow_db, tmp_path):
    from hashlib import sha256

    from tests.e2e.test_workflow_db_e2e import complete, submission

    w, scope = workflow_db
    for amount, remove, expected in [
        ("10", False, True),
        ("12", False, False),
        ("10", True, False),
    ]:
        request = submission()
        path = tmp_path / str(request.id)
        path.write_bytes(b"original")
        digest = sha256(path.read_bytes()).hexdigest()
        request.parts[0].image_path = str(path)
        request.source_manifest[0]["source_sha256"] = digest
        request.bypass_review = True
        image = (await w.enqueue_submission(request, scope=scope, idempotency_key=None)).submission
        outcome = complete(total="10")
        outcome.envelope.provenance.ordered_source_sha256 = [digest]
        outcome.envelope.amount_policy = POLICY
        outcome.envelope.receipt.taxes = []
        outcome.envelope.receipt.total = Decimal(amount)
        outcome.review.can_auto_verify = True  # Consumer must not trust this flag alone.
        if remove:
            path.unlink()
        await w.finish_attempt(await w.claim_next(), outcome)
        async with w.pool.acquire() as conn:
            saved = await conn.fetchrow("SELECT * FROM receipts WHERE image_id=$1", image.id)
        assert saved["verified"] is expected
        assert saved["verification_source"] == ("automatic_bypass" if expected else None)


async def test_draft_finalize_rechecks_real_original_under_lock(workflow_db, tmp_path):
    import asyncpg

    from tests.e2e.test_workflow_db_e2e import complete, partial_draft
    from vision_bill.model.review import OriginalAvailability
    from vision_bill.model.workflow import DraftFinalize, DraftPatch
    from vision_bill.service.image_service import ImageService

    w, scope = workflow_db
    path = tmp_path / "original.png"
    image = await partial_draft(w, scope, path)
    parts = await w.get_source_parts(image.id, scope=scope)
    attestations = [
        OriginalAvailability(part_id=p.part_id, available=True, sha256_matches=True) for p in parts
    ]
    await w.patch_draft(
        image.id,
        DraftPatch(
            expected_revision=1, changes=complete().envelope.receipt.model_dump(mode="json")
        ),
        scope=scope,
    )
    reached, proceed = asyncio.Event(), asyncio.Event()

    async def checker(parts):
        async with w.pool.acquire() as conn, conn.transaction():
            with pytest.raises(asyncpg.LockNotAvailableError):
                await conn.execute(
                    "SELECT image_id FROM receipt_drafts WHERE image_id=$1 FOR UPDATE NOWAIT",
                    image.id,
                )
        reached.set()
        await proceed.wait()
        return await ImageService.check_originals(parts)

    w._receipts._original_checker = checker
    finalization = asyncio.create_task(
        w.finalize_draft(
            image.id,
            DraftFinalize(expected_revision=2, request_key="final"),
            scope=scope,
            originals=attestations,
        )
    )
    await asyncio.wait_for(reached.wait(), 5)
    path.unlink()
    proceed.set()
    with pytest.raises(ValueError, match="original_integrity"):
        await finalization
    async with w.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM receipts") == 0


async def test_direct_membership_changes_version_receipts_and_stale_put_conflicts(
    workflow_db, tmp_path
):
    from vision_bill.provider.db.collection_db import CollectionDB

    w, scope = workflow_db
    row, body, _ = await seed(w, scope, tmp_path, verified=True)
    collections = CollectionDB(w.pool)
    async with w.pool.acquire() as conn:
        target = await conn.fetchval(
            "INSERT INTO collections(user_id,name) VALUES($1,'mine') RETURNING id", scope.user_id
        )
    await collections.assign(target, row.id)
    after = await w._receipts.get_receipt_with_details(row.id)
    assert after.receipt.operation_revision == 2
    assert after.receipt.review_revision == 1 and after.receipt.verified
    body.collection_ids = []
    with pytest.raises(ValueError, match="revision"):
        await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    await collections.assign(target, row.id)  # A genuine no-op does not create a revision.
    assert (await w._receipts.get_receipt_with_details(row.id)).receipt.operation_revision == 2
    assert await collections.unassign(target, row.id)
    assert (await w._receipts.get_receipt_with_details(row.id)).receipt.operation_revision == 3
    await collections.assign(target, row.id)
    assert await collections.delete(target, scope.user_id, False)
    after = await w._receipts.get_receipt_with_details(row.id)
    assert after.receipt.operation_revision == 5 and after.collection_ids == []
    assert after.receipt.review_revision == 1 and after.receipt.verified
    async with w.pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM receipt_review_revisions WHERE receipt_id=$1", row.id
            )
            == 4
        )


async def test_source_decisions_must_bind_saved_hash_and_view(workflow_db, tmp_path):
    from hashlib import sha256

    from vision_bill.model.ocr import SourceRef

    w, scope = workflow_db
    row, body, path = await seed(w, scope, tmp_path)
    digest = sha256(path.read_bytes()).hexdigest()
    async with w.pool.acquire() as conn:
        part = await conn.fetchval(
            "INSERT INTO image_source_parts(image_id,photo_index,source_sha256,image_path,metadata) VALUES($1,0,$2,$3,'{}') RETURNING part_id",
            row.image_id,
            digest,
            str(path),
        )
    for ref in [
        SourceRef(part_id=part, photo_index=0, source_sha256="f" * 64, coordinate_space="original"),
        SourceRef(
            part_id=part,
            photo_index=0,
            source_sha256=digest,
            coordinate_space="working",
            view_id=uuid4(),
        ),
    ]:
        body.field_sources = {"total": [ref]}
        with pytest.raises(ValueError, match="source reference"):
            await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    body.field_sources = None
    body.overlap_decisions = [{"verified": True, "view_id": str(uuid4())}]
    with pytest.raises(ValueError, match="source reference"):
        await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    assert (await w._receipts.get_receipt_with_details(row.id)).receipt.operation_revision == 1


async def test_http_review_gate_with_real_postgres(workflow_db, tmp_path):
    import httpx
    from fastapi import FastAPI

    from vision_bill.api.helper.helper import get_receipt_service
    from vision_bill.api.receipts import router
    from vision_bill.security.dependencies import get_current_user
    from vision_bill.security.models import User
    from vision_bill.service.receipt_service import ReceiptService

    w, scope = workflow_db
    row, body, original = await seed(w, scope, tmp_path)
    service = ReceiptService.__new__(ReceiptService)
    service._db = w._receipts
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/receipts")
    app.dependency_overrides[get_receipt_service] = lambda: service
    app.dependency_overrides[get_current_user] = lambda: User(
        id=scope.user_id, username="owner", is_admin=False, can_see_all=False
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        body.total = Decimal(12)
        payload = body.model_dump(mode="json", exclude_unset=True)
        payload.pop("expected_operation_revision")
        path = f"/api/v1/receipts/{row.id}"
        assert (await client.put(path, json=payload)).status_code == 409
        payload["expected_operation_revision"] = 1
        result = await client.put(path, json=payload)
        assert result.status_code == 200
        finding = result.json()["findings"][0]
        assert finding["code"] == "amount_discrepancy"
        request = {
            "expected_operation_revision": 2,
            "expected_review_revision": 2,
            "acknowledgements": [{"finding_id": finding["id"], "comment": "  "}],
        }
        assert (await client.post(path + "/verify", json=request)).status_code == 422
        request["acknowledgements"][0]["comment"] = "Confirmed printed exception"
        verified = await client.post(path + "/verify", json=request)
        assert verified.status_code == 200
        assert verified.json()["verification_source"] == "human"
        detail = await client.get(path)
        assert detail.json()["receipt"]["acknowledgements"] == request["acknowledgements"]
        assert (await client.post(path + "/verify")).status_code == 409
    assert original.read_bytes() == b"original receipt bytes"


@pytest.mark.parametrize("stored_hash,verified", [(None, True), ("f" * 64, False)])
async def test_legacy_unknown_identity_is_readable_but_known_mismatch_blocks(
    workflow_db, tmp_path, stored_hash, verified
):
    w, scope = workflow_db
    row, _, path = await seed(w, scope, tmp_path)
    async with w.pool.acquire() as conn:
        if stored_hash is not None:
            await conn.execute(
                "INSERT INTO image_source_parts(image_id,photo_index,source_sha256,image_path,metadata) VALUES($1,0,$2,$3,'{}')",
                row.image_id,
                stored_hash,
                str(path),
            )
        else:
            second = tmp_path / "legacy-additional.png"
            second.write_bytes(b"additional original")
            await conn.execute(
                "UPDATE images SET additional_images=$2::jsonb WHERE id=$1",
                row.image_id,
                json.dumps([{"image_path": str(second), "source_sha256": None}]),
            )
    if verified:
        result = await w._receipts.verify_receipt(row.id, ReviewRequest(), scope=scope)
        assert result.verified and result.verification_source == "human"
        async with w.pool.acquire() as conn:
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM image_source_parts WHERE image_id=$1", row.image_id
                )
                == 0
            )
        assert second.read_bytes() == b"additional original"
    else:
        with pytest.raises(ValueError, match="integrity"):
            await w._receipts.verify_receipt(row.id, ReviewRequest(), scope=scope)
    assert path.read_bytes() == b"original receipt bytes"
