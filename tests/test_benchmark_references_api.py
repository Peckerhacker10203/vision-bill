from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from vision_bill.api.benchmarks import router
from vision_bill.api.helper.helper import get_benchmark_service
from vision_bill.security.dependencies import get_current_user
from vision_bill.security.models import User


@pytest.mark.parametrize('method,suffix,name,body', [
    ('GET', '', 'list_runs', None),
    ('GET', '/references/{id}', 'reference_preview', None),
    ('POST', '/references/{id}', 'qualify_reference', {'expected_review_revision': 1, 'expected_document_sha256': 'a'*64}),
    ('GET', '/{id}', 'get_status', None),
    ('GET', '/{id}/results', 'results', None),
    ('POST', '/{id}/reevaluate', 'reevaluate', {'receipt_id': str(uuid4()), 'model_id': 'm'}),
    ('POST', '', 'create_run', {'model_ids': ['m']}),
])
async def test_every_benchmark_read_write_receives_real_admin_scope(method, suffix, name, body):
    user = User(id=uuid4(), username='restricted-admin', is_admin=True, can_see_all=False)
    app = FastAPI()
    app.include_router(router, prefix='/benchmarks')
    app.dependency_overrides[get_current_user] = lambda: user
    service = SimpleNamespace(**{name: AsyncMock(side_effect=ValueError('test boundary'))})
    app.dependency_overrides[get_benchmark_service] = lambda: service
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url='http://test') as client:
        await client.request(method, '/benchmarks' + suffix.replace('{id}', str(uuid4())), json=body)
    call = getattr(service, name)
    assert call.await_count == 1
    scope = call.call_args.kwargs['scope']
    assert scope.user_id == user.id and scope.can_see_all is False


@pytest.mark.parametrize('message', ['Explicit receipt is unavailable or not verified', 'No current qualified references match the selection'])
async def test_create_rejects_ineligible_selection_as_unprocessable(message):
    user = User(id=uuid4(), username='restricted-admin', is_admin=True, can_see_all=False)
    app = FastAPI()
    app.include_router(router, prefix='/benchmarks')
    app.dependency_overrides[get_current_user] = lambda: user
    service = SimpleNamespace(create_run=AsyncMock(side_effect=ValueError(message)))
    app.dependency_overrides[get_benchmark_service] = lambda: service
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post(
            '/benchmarks', json={'model_ids': ['m'], 'receipt_ids': [str(uuid4()), str(uuid4())]}
        )
    assert response.status_code == 422 and response.json()['detail'] == message
    assert len(service.create_run.call_args.args[0].receipt_ids) == 2
    assert service.create_run.call_args.kwargs['scope'].user_id == user.id
