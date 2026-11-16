"""Routes consistently scope all new resources through their submission."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from vision_bill.security.dependencies import get_current_user
from vision_bill.security.models import User


@pytest.fixture
def setup_api():
    from vision_bill.api.drafts import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/images")
    user = User(id=uuid4(), username="owner", is_admin=False)
    app.dependency_overrides[get_current_user] = lambda: user
    db = SimpleNamespace(
        is_ready=True,
        get_extraction=AsyncMock(return_value=None),
        get_working_view=AsyncMock(return_value=None),
        retry_submission=AsyncMock(return_value=None),
    )
    service = SimpleNamespace(
        db=db,
        get_draft=AsyncMock(return_value=None),
        patch_draft=AsyncMock(return_value=None),
        finalize_draft=AsyncMock(return_value=None),
    )
    app.state.submission_service = service
    app.state.analysis_scheduler = SimpleNamespace(trigger=lambda: None)
    return app, service, user


@pytest.mark.parametrize(
    "method,suffix,body",
    [
        ("GET", "/draft", None),
        ("PATCH", "/draft", {"expected_revision": 1, "changes": {"total": None}}),
        ("POST", "/draft/finalize", {"expected_revision": 1, "request_key": "key"}),
        ("POST", "/retry", {"expected_generation": 1, "request_key": "key"}),
        ("GET", "/extractions/" + str(uuid4()), None),
        ("GET", "/views/" + str(uuid4()), None),
    ],
)
async def test_cross_user_missing_resources_are_404_and_always_scoped(
    setup_api, method, suffix, body
):
    app, service, user = setup_api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.request(
            method, "/api/v1/images/" + str(uuid4()) + suffix, json=body
        )
    assert response.status_code == 404
    calls = [
        m
        for m in (
            service.get_draft,
            service.patch_draft,
            service.finalize_draft,
            service.db.retry_submission,
            service.db.get_extraction,
            service.db.get_working_view,
        )
        if m.await_count
    ]
    assert len(calls) == 1
    assert calls[0].call_args.kwargs["scope"].user_id == user.id


async def test_stale_draft_conflict_and_incomplete_finalize(setup_api):
    app, service, _user = setup_api
    service.patch_draft.side_effect = ValueError("draft_revision_conflict")
    service.finalize_draft.side_effect = ValueError("draft_incomplete")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        path = "/api/v1/images/" + str(uuid4()) + "/draft"
        assert (
            await client.patch(path, json={"expected_revision": 1, "changes": {}})
        ).status_code == 409
        assert (
            await client.post(
                path + "/finalize", json={"expected_revision": 1, "request_key": "key"}
            )
        ).status_code == 422
