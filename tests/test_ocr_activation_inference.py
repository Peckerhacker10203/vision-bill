import asyncio
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from ocr_contract.activation import (
    ActivationBusy,
    ActivationPhase,
    ActivationStatus,
    ActivationStore,
    MaintenanceIntent,
)
from openai import APITimeoutError

from scripts.context_budget import Budget
from vision_bill.model.handoff import HandoffBinding, HandoffWait
from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.provider.llm.openai import OpenAIProvider
from vision_bill.provider.ocr.paddleocr import PaddleOcrProvider
from vision_bill.service import activation_admission
from vision_bill.service.activation_admission import inference_admission, unsettled_direct_calls
from vision_bill.service.gpu_handoff import ModelHandoff

_MESSAGE = [{"role": "user", "content": "synthetic"}]


@pytest.fixture
def activation_store(tmp_path, monkeypatch):
    paths = [tmp_path / part for part in ("request", "status", "lock")]
    store = ActivationStore(*paths)
    monkeypatch.setattr(activation_admission, "ActivationStore", lambda: store)
    monkeypatch.setenv("VISION_BILL_OCR_ACTIVATION", "true")
    return store


@pytest.fixture(params=["openai", "ollama"])
async def connect_provider(request):
    provider_type = request.param
    host = "http://local-provider.test"
    if provider_type == "openai":
        provider = OpenAIProvider(f"{host}/v1", "none", model_lifecycle="llamacpp")
    else:
        provider = OllamaProvider(host)
    await provider._client._client.aclose()

    def connect(handle):
        async def respond(request):
            data = json.loads(request.content)
            assert data["model"] == "fixture"
            content = await handle()
            if provider_type == "openai":
                assert request.url.path == "/v1/chat/completions"
                return httpx.Response(200, json={
                    "id": "fixture", "model": "fixture", "created": 1, "object": "chat.completion",
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": content,
                    }}],
                })
            assert request.url.path == "/api/chat"
            return httpx.Response(200, json={
                "model": "fixture", "done": True, "message": {
                    "role": "assistant", "content": content,
                },
            })

        provider._client._client = httpx.AsyncClient(
            base_url=f"{host}/v1" if provider_type == "openai" else host,
            transport=httpx.MockTransport(respond),
        )
        return provider

    yield connect
    await provider._client._client.aclose()


@pytest.mark.parametrize("phase", [None, ActivationPhase.FAILED, ActivationPhase.READY])
async def test_uncoordinated_provider_cannot_dispatch_after_activation_request(
    activation_store, connect_provider, phase,
):
    async def respond():
        pytest.fail("An uncoordinated request reached the provider")

    provider = connect_provider(respond)
    request = activation_store.submit_request(MaintenanceIntent.EXTERNAL_LLM_QUIESCED)
    if phase is not None:
        assert activation_store.publish_status(ActivationStatus(request.generation, phase))
    with pytest.raises(HandoffWait):
        await provider.send_message("fixture", _MESSAGE)


async def test_exclusive_maintenance_blocks_sdk_dispatch(activation_store, connect_provider):
    async def respond():
        pytest.fail("An SDK request reached the provider during exclusive maintenance")

    provider = connect_provider(respond)
    with activation_store.admission(exclusive=True):
        with pytest.raises(HandoffWait, match="dedicated_ocr_activation_pending"):
            await provider.send_message("fixture", _MESSAGE)
        assert unsettled_direct_calls(activation_store) == ()


@pytest.mark.parametrize("mismatch", ["path", "endpoint"])
async def test_ready_activation_rejects_a_stale_handoff(
    activation_store, connect_provider, tmp_path, monkeypatch, mismatch,
):
    async def respond():
        pytest.fail("A provider belonging to another OCR deployment dispatched work")

    provider = connect_provider(respond)
    endpoint = "http://dedicated-ocr.test"
    managed_path = tmp_path / "gpu-handoff.lock"
    monkeypatch.setenv("OCR_ACTIVATION_ENDPOINT", endpoint)
    monkeypatch.setenv("OCR_ACTIVATION_HANDOFF_LOCK_PATH", str(managed_path))
    ocr = PaddleOcrProvider("http://old-ocr.test" if mismatch == "endpoint" else endpoint)
    provider.handoff = ModelHandoff(
        tmp_path / "stale.lock" if mismatch == "path" else managed_path, provider, ocr,
    )
    request = activation_store.submit_request(MaintenanceIntent.EXTERNAL_LLM_QUIESCED)
    assert activation_store.publish_status(ActivationStatus(request.generation, ActivationPhase.READY))

    with pytest.raises(HandoffWait, match="execution_policy_changed"):
        await provider.send_message("fixture", _MESSAGE)


async def test_ready_activation_dispatches_through_the_adopted_coordinator(
    activation_store, connect_provider, tmp_path, monkeypatch,
):
    from vision_bill.config import settings

    endpoint = "http://dedicated-ocr.test"
    managed_path = tmp_path / "gpu-handoff.lock"
    monkeypatch.setenv("OCR_ACTIVATION_ENDPOINT", endpoint)
    monkeypatch.setenv("OCR_ACTIVATION_HANDOFF_LOCK_PATH", str(managed_path))
    request = activation_store.submit_request(MaintenanceIntent.EXTERNAL_LLM_QUIESCED)
    assert activation_store.publish_status(ActivationStatus(request.generation, ActivationPhase.READY))
    events = []

    async def respond():
        events.append("dispatch")
        state = handoff.read_journal()
        assert state.unsettled is True and state.models == ["fixture"]
        assert unsettled_direct_calls(activation_store) == ()
        with pytest.raises(ActivationBusy), activation_store.admission(exclusive=True):
            pytest.fail("Native qualification overlapped a coordinated request")
        return "coordinated response"

    async def resident_model_ids(*, deadline):
        return frozenset()

    async def release_model(model):
        assert model == "fixture"
        events.append("release")

    async def claim(token, deadline):
        # The guard is held before entering the coordinator, even before SDK dispatch.
        with pytest.raises(ActivationBusy), activation_store.admission(exclusive=True):
            pytest.fail("Handoff was claimed before inference admission")
        events.append("claim")

    async def park(token, deadline):
        events.append("park")

    provider = connect_provider(respond)
    ocr = PaddleOcrProvider(endpoint)
    provider.handoff = handoff = ModelHandoff(managed_path, provider, ocr)
    handoff.binding_path.write_text(HandoffBinding(
        coordination_path=str(managed_path.resolve()), resources=handoff.resources,
        model_ids=["fixture"], dedicated_coordinated_requests=True,
        external_eager_loading_disabled=True,
    ).model_dump_json())
    monkeypatch.setattr(settings.ocr, "enabled", True)
    monkeypatch.setattr(settings.ocr, "manage_model_lifecycle", True)
    monkeypatch.setattr(provider, "resident_model_ids", resident_model_ids)
    monkeypatch.setattr(provider, "release_model", release_model)
    monkeypatch.setattr(ocr, "claim", claim)
    monkeypatch.setattr(ocr, "park", park)

    assert await provider.send_message("fixture", _MESSAGE) == "coordinated response"
    assert events == ["claim", "dispatch", "park", "release"]
    assert not handoff.read_journal().unsettled


async def test_provider_holds_admission_and_creates_private_marker_before_dispatch(
    activation_store, connect_provider,
):
    markers = activation_store.request_dir / "direct-calls"

    async def respond():
        with pytest.raises(ActivationBusy), activation_store.admission(exclusive=True):
            pytest.fail("Qualification overlapped an external request")
        [marker] = markers.iterdir()
        assert stat.S_IMODE(marker.stat().st_mode) == 0o600
        assert marker.read_bytes() == b""
        return "synthetic response"

    assert await connect_provider(respond).send_message("fixture", _MESSAGE) == "synthetic response"
    assert list(markers.iterdir()) == []
    with activation_store.admission(exclusive=True):
        pass


async def test_cancelled_sdk_request_leaves_its_own_marker(activation_store, connect_provider):
    dispatch_started = asyncio.Event()
    markers = activation_store.request_dir / "direct-calls"

    async def respond():
        dispatch_started.set()
        await asyncio.Future()

    provider = connect_provider(respond)
    task = asyncio.create_task(provider.send_message("fixture", _MESSAGE))
    await asyncio.wait_for(dispatch_started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with activation_store.admission(exclusive=True):
        [marker] = unsettled_direct_calls(activation_store)
        assert marker.parent == markers
        assert stat.S_IMODE(marker.stat().st_mode) == 0o600


async def test_sdk_timeout_preserves_marker_after_next_call_completes(
    activation_store, connect_provider,
):
    calls = 0

    async def respond():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("remote request may still be running")
        return "confirmed response"

    provider = connect_provider(respond)
    with pytest.raises((APITimeoutError, httpx.ReadTimeout)):
        await provider.send_message("fixture", _MESSAGE)
    with activation_store.admission(exclusive=True):
        [unsettled] = unsettled_direct_calls(activation_store)
    assert await provider.send_message("fixture", _MESSAGE) == "confirmed response"
    with activation_store.admission(exclusive=True):
        assert unsettled_direct_calls(activation_store) == (unsettled,)


def test_process_exit_releases_admission_and_keeps_ambiguous_dispatch(activation_store):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", """
import os
import sys
from pathlib import Path
from ocr_contract.activation import ActivationStore
from vision_bill.service.activation_admission import direct_inference_call, inference_admission

store = ActivationStore(*(Path(arg) for arg in sys.argv[1:]))
with inference_admission(store=store), direct_inference_call():
    os._exit(27)
""", *(str(path) for path in (
            activation_store.request_dir, activation_store.status_dir, activation_store.lock_dir,
        ))],
        env={**os.environ, "PYTHONPATH": os.pathsep.join([
            str(root / "src"), str(root / "paddleocr-service"),
        ])},
        check=False,
        timeout=5,
    )
    assert result.returncode == 27
    with activation_store.admission(exclusive=True):
        [marker] = unsettled_direct_calls(activation_store)
    request = activation_store.submit_request(MaintenanceIntent.EXTERNAL_LLM_QUIESCED)
    assert activation_store.publish_status(ActivationStatus(request.generation, ActivationPhase.FAILED))
    with activation_store.admission(exclusive=True):
        assert unsettled_direct_calls(activation_store) == (marker,)


async def test_admitted_direct_attempt_keeps_lock_through_response_repair(
    activation_store, connect_provider,
):
    calls = 0

    async def respond():
        nonlocal calls
        calls += 1
        if calls == 1:
            activation_store.submit_request(MaintenanceIntent.EXTERNAL_LLM_QUIESCED)
        with pytest.raises(ActivationBusy), activation_store.admission(exclusive=True):
            pytest.fail("Native qualification overlapped a repair attempt")
        return "synthetic invalid receipt"

    provider = connect_provider(respond)
    with pytest.raises(ValueError, match="Failed to get a valid response after 3 attempts"):
        await provider._legacy_analysis("fixture", _MESSAGE)
    assert calls == 3
    with pytest.raises(HandoffWait):
        await provider.send_message("fixture", _MESSAGE)


async def test_spawned_task_checks_queued_activation_independently(activation_store, connect_provider):
    async def respond():
        pytest.fail("A spawned task inherited the parent's direct admission")

    provider = connect_provider(respond)
    with inference_admission(store=activation_store):
        activation_store.submit_request(MaintenanceIntent.EXTERNAL_LLM_QUIESCED)
        with pytest.raises(HandoffWait, match="dedicated_ocr_activation_pending"):
            await asyncio.create_task(provider.send_message("fixture", _MESSAGE))


def test_diagnostic_generate_obeys_activation_admission(activation_store, monkeypatch):
    budget = Budget("http://local-provider.test", "fixture", 1024, 10)
    budget.ctx = 1024

    def post(path, payload):
        assert path == "/api/generate"
        with pytest.raises(ActivationBusy), activation_store.admission(exclusive=True):
            pytest.fail("Native qualification overlapped diagnostic inference")
        return {"prompt_eval_count": 17}

    monkeypatch.setattr(budget, "_post", post)
    assert budget.count("hello") == 17
    activation_store.submit_request(MaintenanceIntent.EXTERNAL_LLM_QUIESCED)
    with pytest.raises(HandoffWait):
        budget.count("hello")


async def test_activation_environment_opt_out_preserves_direct_transport(
    activation_store, connect_provider, monkeypatch,
):
    monkeypatch.delenv("VISION_BILL_OCR_ACTIVATION")

    async def respond():
        return "normal response"

    with activation_store.admission(exclusive=True):
        assert await connect_provider(respond).send_message("fixture", _MESSAGE) == "normal response"
    assert not activation_store.request_dir.exists()


def test_ready_runtime_toggle_holds_stale_worker_route(activation_store, monkeypatch):
    from vision_bill.config import settings

    request = activation_store.submit_request(MaintenanceIntent.EXTERNAL_LLM_QUIESCED)
    activation_store.publish_status(ActivationStatus(request.generation, ActivationPhase.READY))
    activation_store.set_runtime_enabled(request, False)
    monkeypatch.setattr(settings.ocr, "enabled", True)
    with pytest.raises(HandoffWait, match="execution_policy_changed"), inference_admission():
        pytest.fail("An OCR-enabled worker entered after the durable OFF intent")
