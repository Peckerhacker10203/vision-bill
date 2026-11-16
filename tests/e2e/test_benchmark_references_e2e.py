"""Frozen Gold consent, current source identity, isolation and deletion on private PG18."""
import json
from uuid import uuid4

import asyncpg
import pytest

from tests.e2e import test_receipt_review_e2e as reviews
from vision_bill.model.workflow import AccessScope, ReviewRequest
from vision_bill.provider.db.benchmark_db import BenchmarkDB

postgres, db, workflow_db = reviews.postgres, reviews.db, reviews.workflow_db
pytestmark = pytest.mark.e2e


def bench(w):
    result = BenchmarkDB(w.pool)
    result._receipts = w._receipts
    return result


async def human(w, scope, tmp_path):
    row, body, path = await reviews.seed(w, scope, tmp_path)
    await w._receipts.verify_receipt(row.id, ReviewRequest(), scope=scope)
    return row, body, path


async def test_gold_requires_separate_current_human_confirmation(workflow_db, tmp_path):
    w, scope = workflow_db
    row, body, path = await human(w, scope, tmp_path)
    b = bench(w)
    assert hasattr(b, 'reference_preview'), 'scoped current confirmation preview missing'
    preview = await b.reference_preview(row.id, scope=scope)
    assert preview.review_revision == 1
    assert preview.sources[0]['part_id'] is None
    assert preview.sources[0]['photo_index'] == 0
    with pytest.raises(ValueError, match='qualified'):
        await b.current_reference(row.id, scope=scope)
    with pytest.raises(ValueError, match='confirmation'):
        await b.qualify_reference(row.id, 1, '0' * 64, scope=scope)
    frozen = await b.qualify_reference(row.id, 1, preview.document_sha256, scope=scope)
    assert frozen.qualified_by == scope.user_id and frozen.owner_id == scope.user_id
    assert frozen.expected['total'] == '10.00' or frozen.expected['total'] == '10'
    assert 'confidence' not in frozen.scored_fields and 'category' not in frozen.scored_fields
    body.expected_operation_revision = 2
    body.merchant_name = 'Clean edit'
    edited = await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    assert not edited.verified  # legacy identity still requires ordinary review after editing
    await w._receipts.verify_receipt(row.id, ReviewRequest(), scope=scope)
    with pytest.raises(ValueError, match='qualified'):
        await b.current_reference(row.id, scope=scope)
    fresh = await b.reference_preview(row.id, scope=scope)
    with pytest.raises(ValueError, match='confirmation'):
        await b.qualify_reference(row.id, 1, preview.document_sha256, scope=scope)
    second = await b.qualify_reference(row.id, 2, fresh.document_sha256, scope=scope)
    assert second.id != frozen.id and frozen.expected['merchant_name'] == 'Shop'
    path.write_bytes(b'replaced legacy original')
    with pytest.raises(ValueError, match='confirmation'):
        await b.qualify_reference(row.id, 2, fresh.document_sha256, scope=scope)
    with pytest.raises(ValueError, match='qualified'):
        await b.current_reference(row.id, scope=scope)


@pytest.mark.parametrize('source', ['legacy', 'automatic_bypass', None])
async def test_verified_flag_never_qualifies_gold(workflow_db, tmp_path, source):
    w, scope = workflow_db
    row, _, _ = await reviews.seed(w, scope, tmp_path, verified=True)
    async with w.pool.acquire() as conn:
        await conn.execute('UPDATE receipts SET verification_source=$2 WHERE id=$1', row.id, source)
    b = bench(w)
    assert hasattr(b, 'reference_preview'), 'human provenance gate missing'
    with pytest.raises(ValueError, match='human'):
        await b.reference_preview(row.id, scope=scope)


async def test_gold_owner_scope_immutable_history_survives_receipt_and_user_delete(workflow_db, tmp_path):
    w, scope = workflow_db
    row, _, _ = await human(w, scope, tmp_path)
    b = bench(w)
    assert hasattr(b, 'reference_preview'), 'reference storage missing'
    other = uuid4()
    async with w.pool.acquire() as conn:
        await conn.execute("INSERT INTO users(id,username,hashed_password,is_admin) VALUES($1,$2,'test',true)", other, str(other))
    with pytest.raises(ValueError, match='not found'):
        await b.reference_preview(row.id, scope=AccessScope(user_id=other))
    admin_scope = AccessScope(user_id=other, can_see_all=True)
    preview = await b.reference_preview(row.id, scope=admin_scope)
    frozen = await b.qualify_reference(row.id, 1, preview.document_sha256, scope=admin_scope)
    async with w.pool.acquire() as conn:
        with pytest.raises(asyncpg.RaiseError, match='immutable'):
            await conn.execute("UPDATE benchmark_references SET expected='{}' WHERE id=$1", frozen.id)
        with pytest.raises(asyncpg.RaiseError, match='immutable'):
            await conn.execute('DELETE FROM benchmark_references WHERE id=$1', frozen.id)
        await conn.execute('DELETE FROM receipts WHERE id=$1', row.id)
        await conn.execute('DELETE FROM users WHERE id=ANY($1::uuid[])', [other, scope.user_id])
        saved = await conn.fetchrow('SELECT * FROM benchmark_references WHERE id=$1', frozen.id)
    assert saved['receipt_id'] is None and saved['owner_id'] is None and saved['qualified_by'] is None
    assert json.loads(saved['expected']) == frozen.expected
    assert saved['receipt_identity'] == row.id and saved['owner_identity'] == scope.user_id
    assert saved['actor_identity'] == other


async def test_clean_durable_edit_can_attest_current_gold_without_second_verify(workflow_db, tmp_path):
    import hashlib
    w, scope = workflow_db
    row, body, path = await human(w, scope, tmp_path)
    async with w.pool.acquire() as conn:
        await conn.execute("INSERT INTO image_source_parts(part_id,image_id,photo_index,source_sha256,image_path,metadata) VALUES($1,$2,0,$3,$4,'{}')",
                           uuid4(), row.image_id, hashlib.sha256(path.read_bytes()).hexdigest(), str(path))
    body.expected_operation_revision = 2
    body.merchant_name = 'Current human attestation'
    edited = await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    assert edited.verified and edited.verification_source == 'human'
    b = bench(w)
    preview = await b.reference_preview(row.id, scope=scope)
    gold = await b.qualify_reference(row.id, 2, preview.document_sha256, scope=scope)
    assert gold.review_revision == 2 and gold.expected['merchant_name'] == body.merchant_name


async def test_new_runs_scope_owner_and_preserve_unknown_historical_ownership(workflow_db, tmp_path):
    from vision_bill.model.benchmark import CouncilPolicy
    w, scope = workflow_db
    row, _, _ = await human(w, scope, tmp_path)
    b = bench(w)
    kwargs = dict(model_ids=['m'], receipt_ids=[row.id], fingerprint='f', model_digests={}, timeout=600,
                  policy=CouncilPolicy.ALL, absolute=None, relative=None, apply_flags=False,
                  prompt_version='4', scoring_version='4')
    run = await b.create_run(**kwargs, owner_id=scope.user_id)
    stranger = AccessScope(user_id=uuid4())
    assert await b.run(run.id, scope=stranger) is None
    assert await b.runs(scope=stranger) == []
    assert (await b.run(run.id, scope=scope)).id == run.id
    async with w.pool.acquire() as conn:
        await conn.execute('UPDATE benchmark_runs SET owner_id=NULL WHERE id=$1', run.id)
    assert await b.run(run.id, scope=scope) is None
    assert (await b.run(run.id, scope=AccessScope(user_id=scope.user_id,can_see_all=True))).id == run.id
    # Active tasks retain their existing temporary deletion refusal.
    async with w.pool.acquire() as conn:
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await conn.execute('DELETE FROM receipts WHERE id=$1', row.id)


async def test_two_tax_roundtrip_keeps_printed_positions_and_source_paths(workflow_db, tmp_path):
    from uuid import UUID
    from vision_bill.model.receipt import TaxLine
    from vision_bill.model.ocr import SourceRef
    import hashlib
    w, scope = workflow_db
    row, body, path = await reviews.seed(w, scope, tmp_path)
    part_id = uuid4()
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    async with w.pool.acquire() as conn:
        await conn.execute("INSERT INTO image_source_parts(part_id,image_id,photo_index,source_sha256,image_path,metadata) VALUES($1,$2,0,$3,$4,'{}')", part_id, row.image_id, source_hash, str(path))
    body.taxes = [TaxLine(name='VAT seven', rate=7, amount='1'), TaxLine(name='VAT nineteen', rate=19, amount='2')]
    from decimal import Decimal
    body.tax_total = Decimal(3)
    body.amount_policy = reviews.POLICY.model_copy(update={'tax_information': 'present'})
    body.field_sources = {f'taxes.{i}.amount': [SourceRef(part_id=part_id,photo_index=0,source_sha256=source_hash,block_id=f'printed-tax-{i}')] for i in range(2)}
    saved = await w._receipts.update_receipt(row.id, body, user_id=scope.user_id)
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE taxes SET id=$2 WHERE receipt_id=$1 AND name='VAT seven'",row.id,UUID('ffffffff-ffff-ffff-ffff-ffffffffffff'))
        await conn.execute("UPDATE taxes SET id=$2 WHERE receipt_id=$1 AND name='VAT nineteen'",row.id,UUID('00000000-0000-0000-0000-000000000001'))
    details = await w._receipts.get_receipt_with_details(row.id)
    assert [tax.name for tax in details.taxes] == ['VAT seven','VAT nineteen']
    assert [tax.id for tax in details.taxes] == [UUID('ffffffff-ffff-ffff-ffff-ffffffffffff'), UUID('00000000-0000-0000-0000-000000000001')]
    for index in range(2):
        source = details.receipt.field_sources[f'taxes.{index}.amount'][0]
        assert source.block_id == f'printed-tax-{index}'
        assert source.part_id == part_id and source.source_sha256 == source_hash
    async with w.pool.acquire() as conn:
        reread = await w._receipts._review_document(conn, await conn.fetchrow('SELECT * FROM receipts WHERE id=$1',row.id))
    from vision_bill.service.receipt_review import material_document
    assert material_document(reread) == material_document(body)
    assert saved.findings == details.receipt.findings


async def test_run_uses_explicit_frozen_reference_and_never_feeds_expected_to_model(workflow_db,tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from ollama import ChatResponse
    from vision_bill.config import ImageSettings, OcrSettings
    from vision_bill.model.benchmark import BenchmarkCreate
    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.provider.llm.ollama import OllamaProvider
    from vision_bill.provider.llm.base import ModelInfo
    from vision_bill.service.receipt_service import ReceiptService
    from vision_bill.service.benchmark_service import BenchmarkService
    from vision_bill.model.ocr import OcrCapabilities, OcrResponse
    from tests.test_ocr_provider import wire_page
    w, scope = workflow_db
    row, body, path = await human(w,scope,tmp_path)
    # Two ordered originals exercise the same service path as live uploads.
    second = tmp_path / 'second'
    second.write_bytes(b'second original')
    async with w.pool.acquire() as conn:
        await conn.execute('UPDATE images SET additional_images=$2 WHERE id=$1',row.image_id,json.dumps([{'image_path':str(second)}]))
    provider = OllamaProvider('http://local')
    model = ModelInfo('fixture'); model.capabilities=['completion','vision']
    provider.get_available_models = AsyncMock(return_value=[model])
    from vision_bill.model.receipt import Receipt
    payload = Receipt.model_validate(body.model_dump()).model_dump(mode='json')
    payload['merchant_name'] = 'Observed from provider'
    payload['line_items'][0]['source_image'] = 0
    payload['duplicate_candidates'] = []
    provider._client.chat = AsyncMock(return_value=ChatResponse(message={'role':'assistant','content':json.dumps(payload)}))
    async def page(path,*,photo_index,source_sha256,**kwargs):
        return OcrResponse.model_validate(wire_page(photo_index,source_sha256,text=f'Only observed photo {photo_index}'))
    ocr = SimpleNamespace(get_capabilities=AsyncMock(return_value=OcrCapabilities(enabled=True,configured=True,reachable=True,ready=True,profile_id='fixed-profile',runtime_fingerprint='runtime')),extract_page=page)
    receipts = ReceiptService(ImageSettings(save_dir=str(tmp_path),tmp_dir=str(tmp_path)),w._settings,provider,ocr_provider=ocr,ocr_settings=OcrSettings(enabled=True))
    receipts._db = w._receipts
    service = BenchmarkService(provider,receipts)
    request = BenchmarkCreate(model_ids=['fixture'],receipt_ids=[row.id],experiment=ExperimentOptions(pipeline='ocr',include_classification=False))
    with pytest.raises(ValueError,match='qualified'):
        await service.create_run(request,scope=scope)
    preview = await service.reference_preview(row.id,scope=scope)
    gold = await service.qualify_reference(row.id,1,preview.document_sha256,scope=scope)
    run = await service.create_run(request,scope=scope)
    manifest = await service._db.manifest(run.id)
    assert manifest['inputs'][0]['reference']['id'] == str(gold.id)
    assert len(manifest['inputs'][0]['images']) == 2
    async with w.pool.acquire() as conn:
        await conn.execute("UPDATE receipts SET merchant_name='REFERENCE_SENTINEL' WHERE id=$1",row.id)
    task = await service._db.lease_task()
    await service._execute(dict(task))
    result = (await service.results(run.id,limit=10,offset=0,scope=scope))[0]
    assert result['success'] and result['expected']['merchant_name']=='Shop', result
    assert result['metadata']['source_image_count']==2
    assert len(result['metadata']['ocr_pages'])==2
    prompt = provider._client.chat.call_args.kwargs['messages'][0]['content']
    assert 'REFERENCE_SENTINEL' not in prompt and 'Shop' not in prompt
    before = await service._db.results(run.id)
    reevaluated = await service.reevaluate(run.id,row.id,'fixture',scope=scope)
    assert reevaluated['expected'] == gold.expected
    assert await service._db.results(run.id)==before
    original_execute = receipts.analyse_submission_with_metadata
    async def changed_sources(*args, **kwargs):
        result = await original_execute(*args, **kwargs)
        result.metadata['ordered_source_sha256'] = list(reversed(result.metadata['ordered_source_sha256']))
        return result
    receipts.analyse_submission_with_metadata = changed_sources
    with pytest.raises(ValueError, match='source identity'):
        await service.reevaluate(run.id,row.id,'fixture',scope=scope)
    assert await service._db.results(run.id)==before


async def test_runtime_role_enforces_reference_immutability_and_current_scope(workflow_db,tmp_path):
    w,scope=workflow_db
    row,_,_=await human(w,scope,tmp_path)
    async with w.pool.acquire() as conn:
        await conn.execute('CREATE ROLE benchmark_runtime NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE')
        await conn.execute('GRANT USAGE ON SCHEMA public TO benchmark_runtime')
        await conn.execute('GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA public TO benchmark_runtime')
    async def init(conn): await conn.execute('SET ROLE benchmark_runtime')
    pool=await asyncpg.create_pool(w._settings.pg_dsn,init=init,min_size=1,max_size=2)
    try:
        b=BenchmarkDB(pool,w._receipts)
        preview=await b.reference_preview(row.id,scope=scope)
        gold=await b.qualify_reference(row.id,1,preview.document_sha256,scope=scope)
        async with pool.acquire() as conn:
            assert not await conn.fetchval("SELECT rolsuper FROM pg_roles WHERE rolname=current_user")
            with pytest.raises(asyncpg.RaiseError,match='immutable'):
                await conn.execute('UPDATE benchmark_references SET owner_id=NULL WHERE id=$1',gold.id)
        with pytest.raises(ValueError,match='not found'):
            await b.reference_preview(row.id,scope=AccessScope(user_id=uuid4()))
    finally: await pool.close()


async def test_qualification_race_before_run_creation_is_rejected(workflow_db,tmp_path):
    from vision_bill.model.benchmark import CouncilPolicy
    w,scope=workflow_db
    row,_,path=await human(w,scope,tmp_path)
    b=bench(w)
    preview=await b.reference_preview(row.id,scope=scope)
    gold=await b.qualify_reference(row.id,1,preview.document_sha256,scope=scope)
    manifest={'inputs':[{'receipt_id':str(row.id),'reference':gold.model_dump(mode='json')} ]}
    path.write_bytes(b'changed after selection before insertion')
    with pytest.raises(ValueError,match='reference'):
        await b.create_run(model_ids=['m'],receipt_ids=[row.id],fingerprint='f',model_digests={},timeout=600,
            policy=CouncilPolicy.ALL,absolute=None,relative=None,apply_flags=False,prompt_version='4',scoring_version='4',
            manifest=manifest,owner_id=scope.user_id,scope=scope)


async def test_upgrade_keeps_legacy_tax_order_and_unknown_run_owner(db):
    from tests.e2e.test_workflow_db_e2e import migrate
    from uuid import UUID
    migrate(db,'0015')
    conn=await asyncpg.connect(db.pg_dsn)
    try:
        row=await conn.fetchval("INSERT INTO receipts(merchant_name,date,confidence,subtotal,total) VALUES('legacy','2026-01-01',90,10,10) RETURNING id")
        ids=[UUID('ffffffff-ffff-ffff-ffff-ffffffffffff'),UUID('00000000-0000-0000-0000-000000000001')]
        await conn.execute("INSERT INTO taxes(id,receipt_id,name,rate,amount) VALUES($1,$3,'first inserted',19,1.234567),($2,$3,'second inserted',7,2.345678)",*ids,row)
        before=await conn.fetch('SELECT id,rate,amount FROM taxes ORDER BY id')
        migrate(db)
        after=await conn.fetch('SELECT id,rate,amount FROM taxes ORDER BY position,id')
        assert before==after
        assert await conn.fetchval('SELECT version_num FROM alembic_version')=='0016'
        assert migrate(db,'0015','downgrade',check=False).returncode==0
    finally: await conn.close()


async def test_new_summary_counts_extraction_failures_without_redefining_success_average(workflow_db,tmp_path):
    from vision_bill.model.benchmark import CouncilPolicy
    w,scope=workflow_db
    first,_,_=await human(w,scope,tmp_path)
    second,_,_=await human(w,scope,tmp_path)
    b=bench(w)
    run=await b.create_run(model_ids=['m'],receipt_ids=[first.id,second.id],fingerprint='f',model_digests={},timeout=600,
        policy=CouncilPolicy.ALL,absolute=None,relative=None,apply_flags=False,prompt_version='4',scoring_version='4',owner_id=scope.user_id)
    await b.finish_task({'run_id':run.id,'receipt_id':first.id,'model_id':'m'},success=True,score=1,result={'version':'4','full_receipt_correct':True})
    await b.finish_task({'run_id':run.id,'receipt_id':second.id,'model_id':'m'},success=False,result={'version':'4','full_receipt_correct':False})
    summary=(await b.summaries(run.id))[0]
    assert summary.average_score==1
    assert summary.all_case_average_score==.5
    assert summary.full_receipt_correct_rate==.5
    assert summary.completed==2 and summary.failed==1
