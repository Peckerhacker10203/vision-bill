"""Dedicated OCR can be requested only by an administrator."""
import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vision_bill.api.system.main import router
from vision_bill.security.dependencies import get_current_user, require_admin


def app_with_system() -> FastAPI:
    app = FastAPI()
    app.state.user_db = SimpleNamespace(is_ready=True)
    app.include_router(router, prefix="/system")
    return app


def test_activation_requires_admin() -> None:
    client = TestClient(app_with_system())
    assert client.get("/system/dedicated-ocr").status_code == 401
    assert client.post("/system/dedicated-ocr", json={}).status_code == 401


def test_app_without_a_gpu_profile_reports_unavailable() -> None:
    app = app_with_system()
    app.dependency_overrides[require_admin] = lambda: None
    client = TestClient(app)
    response = client.get("/system/dedicated-ocr")
    assert response.status_code == 200
    assert response.json() == {"state": "unavailable", "reason": "gpu_profile_required"}
    response = client.post("/system/dedicated-ocr", json={})
    assert response.status_code == 409
    assert response.json()["detail"] == "gpu_profile_required"


def test_a_regular_user_cannot_request_activation() -> None:
    app = app_with_system()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(is_admin=False)
    assert TestClient(app).post(
        "/system/dedicated-ocr", json={"confirm_handoff": True}
    ).status_code == 403


def test_activation_does_not_accept_browser_supplied_download_sources() -> None:
    app = app_with_system()
    app.dependency_overrides[require_admin] = lambda: None
    response = TestClient(app).post(
        "/system/dedicated-ocr", json={"url": "https://example.invalid/models"}
    )
    assert response.status_code == 422


def test_activation_request_is_idempotent_at_the_app_boundary(tmp_path, monkeypatch) -> None:
    from ocr_contract.activation import ActivationPhase, ActivationStatus, ActivationStore

    from vision_bill.config import settings
    from vision_bill.service.dedicated_ocr import DedicatedOcrActivation

    store = ActivationStore(
        tmp_path / "request", tmp_path / "status", tmp_path / "lock"
    )
    config = settings.model_copy(deep=True)
    config.ocr.enabled = False
    config.ocr.coordination_path = tmp_path / "legacy.lock"
    route = []

    class Provider:
        supports_model_release = True
        lifecycle_identity = "openai:test"
        handoff = None

        async def resident_model_ids(self, **_kwargs) -> frozenset[str]:
            return frozenset()

    activation = DedicatedOcrActivation(
        config, Provider(), lambda ocr, settings: route.append((ocr, settings)),
        lambda: None, store=store, handoff_path=tmp_path / "gpu.lock",
    )

    async def controller_is_live() -> bool:
        return True

    monkeypatch.setattr(activation, "_controller_is_live", controller_is_live)
    monkeypatch.setattr(activation, "_runtime_is_serving", controller_is_live, raising=False)

    async def run() -> None:
        assert (await activation.status()).state == "disabled"
        assert store.read_request() is None
        assert (await activation.submit(confirm_handoff=False)).reason == "ocr_handoff_confirmation_required"
        assert store.read_request() is None
        assert (await activation.submit(confirm_handoff=True)).state == "requested"
        request = store.read_request()
        assert request
        assert (await activation.submit(confirm_handoff=True)).state == "requested"
        assert store.read_request() == request
        assert not config.ocr.enabled
        assert route == []
        store.publish_status(ActivationStatus(request.generation, ActivationPhase.READY))
        await activation.reconcile()
        assert (await activation.status()).state == "active"
        assert config.ocr.enabled and config.ocr.manage_model_lifecycle
        assert len(route) == 1
        await activation.reconcile()
        assert len(route) == 1

        async def offline() -> bool:
            return False

        monkeypatch.setattr(activation, "_runtime_is_serving", offline)
        assert (await activation.status()).reason == "ocr_runtime_unavailable"
        assert config.ocr.enabled

    asyncio.run(run())


def test_startup_request_can_wait_for_an_idle_controller(tmp_path, monkeypatch) -> None:
    import pytest
    from ocr_contract.activation import ActivationPhase, ActivationStatus, ActivationStore

    from vision_bill.config import settings
    from vision_bill.service.dedicated_ocr import DedicatedOcrActivation

    store = ActivationStore(tmp_path / "request", tmp_path / "status", tmp_path / "lock")
    config = settings.model_copy(deep=True)
    config.ocr.enabled = False
    config.ocr.coordination_path = tmp_path / "legacy.lock"

    async def idle_models(**_kwargs) -> frozenset[str]:
        return frozenset()

    provider = SimpleNamespace(
        supports_model_release=True, lifecycle_identity="openai:test", resident_model_ids=idle_models,
    )
    activation = DedicatedOcrActivation(
        config, provider, lambda *_args: None, lambda: None,
        store=store, handoff_path=tmp_path / "gpu.lock",
    )
    checks = iter((False, True))

    async def controller_is_live() -> bool:
        return next(checks)

    monkeypatch.setattr(activation, "_controller_is_live", controller_is_live)
    monkeypatch.setenv("OCR_ACTIVATION_ON_STARTUP", "true")
    monkeypatch.setenv("OCR_ACTIVATION_CONFIRM_HANDOFF", "true")

    async def run() -> None:
        task = asyncio.create_task(activation.run())
        try:
            async with asyncio.timeout(4):
                while (request := store.read_request()) is None:
                    await asyncio.sleep(0.05)
            store.publish_status(ActivationStatus(request.generation, ActivationPhase.FAILED))
            await asyncio.sleep(1)
            assert store.read_request() == request
            assert (await activation.status()).state == "failed"
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
