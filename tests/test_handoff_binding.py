"""External dedication and residency barriers, with owned software doubles only."""
import json
import stat
from time import monotonic

import pytest
from test_gpu_handoff import Llm, Ocr

from vision_bill.service.gpu_handoff import ModelHandoff


class ResidentLlm(Llm):
    def __init__(self, events):
        super().__init__(events)
        self.resident = frozenset()

    async def resident_model_ids(self, *, deadline):
        self.events.append("residency")
        return self.resident


async def adopt(handoff, models=("own",), **kwargs):
    from vision_bill.distribution.handoff import adopt_binding
    return await adopt_binding(
        handoff, models, dedicated_coordinated_requests=True,
        external_eager_loading_disabled=True, deadline=monotonic() + 1, **kwargs,
    )


@pytest.mark.asyncio
async def test_fresh_empty_journal_cannot_load_ocr_without_binding(tmp_path):
    from vision_bill.model.handoff import HandoffWait
    events = []
    llm = ResidentLlm(events)
    llm.resident = frozenset({"foreign"})
    handoff = ModelHandoff(tmp_path / "handoff.lock", llm, Ocr(events))
    with pytest.raises(HandoffWait) as caught:
        async with handoff.session(monotonic() + 1) as owner:
            await owner.ocr_phase()
    assert caught.value.status.state == "configuration_required"
    assert "load_ocr" not in events
    assert not handoff.journal.exists()
    with pytest.raises(HandoffWait):
        await adopt(handoff)
    assert all(not isinstance(event, tuple) for event in events)
    assert not handoff.binding_path.exists()


@pytest.mark.asyncio
async def test_adoption_persists_exact_secret_free_binding_and_inspection_is_read_only(tmp_path):
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    binding = await adopt(handoff)
    assert binding.model_ids == ["own"]
    assert binding.coordination_path == str(handoff.path.resolve())
    assert stat.S_IMODE(handoff.binding_path.stat().st_mode) == 0o600
    assert handoff.binding_path == tmp_path / "handoff.binding.json"
    assert not handoff.journal.exists()
    before = handoff.binding_path.read_bytes()
    status = await handoff.inspect(model_id="own", deadline=monotonic() + 1)
    assert status.state == "ready"
    assert handoff.binding_path.read_bytes() == before
    assert events == ["residency", "residency"]
    assert not handoff.journal.exists()


@pytest.mark.asyncio
async def test_model_outside_explicit_binding_is_held_before_tracking_or_dispatch(tmp_path):
    from vision_bill.model.handoff import HandoffWait
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    async with handoff.session(monotonic() + 1) as owner:
        with pytest.raises(HandoffWait) as caught:
            await owner.llm_phase("other")
        assert caught.value.status.state == "configuration_required"
        assert owner.models == []
    assert ("unload", "other") not in events


@pytest.mark.asyncio
async def test_foreign_residency_after_adoption_never_claims_or_unloads(tmp_path):
    from vision_bill.model.handoff import HandoffWait
    events = []
    llm = ResidentLlm(events)
    handoff = ModelHandoff(tmp_path / "handoff.lock", llm, Ocr(events))
    await adopt(handoff)
    async with handoff.session(monotonic() + 1):
        pass
    events.clear()
    llm.resident = frozenset({"foreign"})
    before = handoff.binding_path.read_bytes()
    journal_before = handoff.journal.read_bytes()
    with pytest.raises(HandoffWait) as caught:
        async with handoff.session(monotonic() + 1) as owner:
            await owner.ocr_phase()
    assert caught.value.status.state == "recovery_required"
    assert events == ["residency"]
    assert handoff.binding_path.read_bytes() == before
    assert handoff.journal.read_bytes() == journal_before


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["legacy", "unsettled", "ocr_pending", "models", "mismatch", "malformed"])
async def test_adoption_preserves_unresolved_journal_bytes(tmp_path, kind):
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    state = {"version": 2, "coordinator": "vision-bill-model-handoff-v2", "resources": handoff.resources.model_dump(),
             "models": [], "generation": 8, "unsettled": False, "ocr_pending": False}
    if kind in ("unsettled", "ocr_pending"):
        state[kind] = True
    elif kind == "models":
        state["models"] = ["own"]
    elif kind == "mismatch":
        state["resources"]["llm"] = "old"
    raw = b'{"models":[]}' if kind == "legacy" else b'{' if kind == "malformed" else json.dumps(state).encode()
    handoff.journal.write_bytes(raw)
    with pytest.raises((ValueError, RuntimeError)):
        await adopt(handoff)
    assert handoff.journal.read_bytes() == raw
    assert not handoff.binding_path.exists()
    assert events == []


@pytest.mark.asyncio
async def test_live_owner_cannot_be_adopted_and_binding_is_preserved(tmp_path):
    from vision_bill.model.handoff import HandoffWait
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    async with handoff.session(monotonic() + 1):
        original = handoff.binding_path.read_bytes()
        journal_before = handoff.journal.read_bytes()
        with pytest.raises(HandoffWait) as caught:
            await adopt(handoff)
        assert caught.value.status.state == "busy"
        assert handoff.binding_path.read_bytes() == original
        assert handoff.journal.read_bytes() == journal_before


@pytest.mark.asyncio
@pytest.mark.parametrize("models,dedicated,eager", [([], True, True), (["own"], False, True), (["own"], True, False), ([" "], True, True)])
async def test_invalid_attestations_and_ids_reject_before_effects(tmp_path, models, dedicated, eager):
    from vision_bill.distribution.handoff import adopt_binding
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    with pytest.raises(ValueError):
        await adopt_binding(handoff, models, dedicated_coordinated_requests=dedicated,
                            external_eager_loading_disabled=eager, deadline=monotonic() + 1)
    assert events == []
    assert not handoff.path.exists()


@pytest.mark.asyncio
async def test_lost_chat_reply_keeps_original_error_and_fences_journal(tmp_path):
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    async def lost():
        raise ConnectionError("original chat reply lost")
    with pytest.raises(ConnectionError, match="original chat reply lost"):
        async with handoff.session(monotonic() + 1) as owner:
            await owner.llm_phase("own")
            await owner.model_call(lost())
    state = json.loads(handoff.journal.read_bytes())
    assert state["unsettled"] is True
    assert state["models"] == ["own"]
    assert ("unload", "own") not in events


def test_generic_inference_identity_is_independent_of_lifecycle_and_credentials():
    from vision_bill.provider.llm.ollama import OllamaProvider
    from vision_bill.provider.llm.openai import OpenAIProvider
    first = OpenAIProvider("HTTP://user:secret@EXAMPLE.local:80/v1/?token=private", "key-one")
    second = OpenAIProvider("http://example.local/v1", "key-two")
    assert first.lifecycle_identity == "unsupported"
    assert first.inference_identity == second.inference_identity
    assert first.inference_identity != OpenAIProvider("http://other.local/v1", "key-two").inference_identity
    assert all(secret not in first.inference_identity for secret in ("secret", "private", "key-one", "example"))
    assert OllamaProvider("http://EXAMPLE.local:80/").inference_identity == OllamaProvider("http://example.local").inference_identity


@pytest.mark.asyncio
async def test_initial_router_status_failure_is_retryable_without_unsettled(monkeypatch, tmp_path):
    import httpx
    from test_llamacpp_lifecycle import configured
    events = []
    fail = False
    def handler(request):
        assert request.method == "GET"
        if fail:
            return httpx.Response(503, json={"error": "temporarily unavailable"})
        return httpx.Response(200, json={"data": [{"id": "own", "status": {"value": "unloaded"}}]})
    async with configured(monkeypatch, handler) as provider:
        handoff = ModelHandoff(tmp_path / "handoff.lock", provider, Ocr(events))
        await adopt(handoff)
        async with handoff.session(monotonic() + 1, model_id="own") as owner:
            owner.track("own")
            fail = True
            with pytest.raises(httpx.HTTPStatusError):
                await provider.release_model("own")
            assert json.loads(handoff.journal.read_bytes())["unsettled"] is False
            assert owner.dispatch_disposition == "not_dispatched"
            fail = False
            await owner.release_models()
            assert owner.models == []


@pytest.mark.asyncio
async def test_selected_model_is_checked_before_claim_and_inspection_preserves_generation(tmp_path):
    from vision_bill.model.handoff import HandoffWait
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    events.clear()
    with pytest.raises(HandoffWait):
        async with handoff.session(monotonic() + 1, model_id="other"):
            pytest.fail("unadopted selection claimed resources")
    assert events == []
    async with handoff.session(monotonic() + 1, model_id="own"):
        pass
    before = handoff.journal.read_bytes()
    events.clear()
    assert (await handoff.inspect(model_id="own", deadline=monotonic() + 1)).state == "ready"
    assert (await handoff.inspect(model_id="other", deadline=monotonic() + 1)).state == "configuration_required"
    assert handoff.journal.read_bytes() == before
    assert events == ["residency"]


@pytest.mark.asyncio
async def test_clean_journal_readoption_changes_only_explicit_model_set(tmp_path):
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    async with handoff.session(monotonic() + 1):
        pass
    before = handoff.journal.read_bytes()
    binding = await adopt(handoff, models=("new-exact-id",))
    assert binding.model_ids == ["new-exact-id"]
    assert handoff.journal.read_bytes() == before
    assert (await handoff.inspect(model_id="own", deadline=monotonic() + 1)).state == "configuration_required"


@pytest.mark.asyncio
async def test_busy_inspect_and_adoption_do_not_change_existing_artifacts(tmp_path):
    import fcntl

    from vision_bill.model.handoff import HandoffWait
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm([]), Ocr([]))
    await adopt(handoff)
    before = handoff.binding_path.read_bytes()
    with handoff.path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (await handoff.inspect(model_id="own", deadline=monotonic() + 1)).state == "busy"
        with pytest.raises(HandoffWait) as caught:
            await adopt(handoff)
        assert caught.value.status.state == "busy"
    assert handoff.binding_path.read_bytes() == before
    assert not handoff.journal.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    {}, {"models": [None]}, {"models": [{"name": "foreign"}]},
    {"models": [{"model": "foreign"}, {"model": "foreign"}]},
])
async def test_malformed_ollama_residency_cannot_prove_empty(body):
    import httpx
    from ollama import AsyncClient

    from vision_bill.provider.llm.ollama import OllamaProvider
    provider = OllamaProvider("http://local")
    provider._client = AsyncClient(host="http://local", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=body),
    ))
    try:
        with pytest.raises((ValueError, RuntimeError)):
            await provider.resident_model_ids(deadline=monotonic() + 1)
    finally:
        await provider._client._client.aclose()


@pytest.mark.asyncio
async def test_lost_router_unload_reply_preserves_owned_unknown_request(monkeypatch, tmp_path):
    import httpx
    from test_llamacpp_lifecycle import configured
    state = "unloaded"
    posts = []
    def handler(request):
        if request.method == "POST":
            posts.append(json.loads(request.content))
            raise httpx.ReadError("original unload reply lost")
        return httpx.Response(200, json={"data": [{"id": "own", "status": {"value": state}}]})
    async with configured(monkeypatch, handler) as provider:
        handoff = ModelHandoff(tmp_path / "handoff.lock", provider, Ocr([]))
        await adopt(handoff)
        with pytest.raises(httpx.ReadError, match="original unload reply lost"):
            async with handoff.session(monotonic() + 1, model_id="own") as owner:
                owner.track("own")
                state = "loaded"
                await provider.release_model("own")
        assert owner.dispatch_disposition == "unknown"
        saved = json.loads(handoff.journal.read_bytes())
        assert saved["unsettled"] is True
        assert saved["models"] == ["own"]
        assert posts == [{"model": "own"}]


@pytest.mark.asyncio
async def test_cli_status_and_adopt_use_settings_and_do_not_disclose_credentials(monkeypatch, tmp_path, capsys):
    import argparse

    from vision_bill.distribution.handoff import _run
    events = []
    llm, ocr = ResidentLlm(events), Ocr(events)
    ocr.managed_lifecycle = True
    monkeypatch.setattr("vision_bill.provider.factory.get_llm_provider", lambda config: llm)
    monkeypatch.setattr("vision_bill.provider.factory.get_ocr_provider", lambda config: ocr)
    args = argparse.Namespace(command="adopt", handoff_lock=tmp_path / "handoff.lock",
                              model_id=["own"], dedicated_coordinated_requests=True,
                              external_eager_loading_disabled=True)
    assert await _run(args) == 0
    args.command, args.model_id = "status", "own"
    assert await _run(args) == 0
    assert capsys.readouterr().out == '{"state":"ready","reason_code":null}\n' * 2
    assert events == ["residency", "residency"]


@pytest.mark.asyncio
async def test_provider_cannot_release_a_different_endpoint_under_another_owner(tmp_path):
    from unittest.mock import AsyncMock

    from ollama import ProcessResponse

    from vision_bill.provider.llm.ollama import OllamaProvider
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm([]), Ocr([]))
    await adopt(handoff)
    other = OllamaProvider("http://different-endpoint.local")
    other._client.generate = AsyncMock()
    other._client.ps = AsyncMock(return_value=ProcessResponse(models=[]))
    async with handoff.session(monotonic() + 1) as owner:
        owner.track("own")
        with pytest.raises(ValueError, match="model_handoff_identity_changed"):
            await other.release_model("own")
        other._client.generate.assert_not_awaited()


@pytest.mark.parametrize("field", ["version", "dedicated_coordinated_requests", "external_eager_loading_disabled"])
def test_binding_rejects_numeric_attestations_and_boolean_version(field):
    from pydantic import ValidationError

    from vision_bill.model.handoff import HandoffBinding
    data = {"version": 1, "coordination_path": "/tmp/test.lock", "resources": {"llm": "test", "ocr": "test"},
            "model_ids": ["own"], "dedicated_coordinated_requests": True, "external_eager_loading_disabled": True}
    data[field] = True if field == "version" else 1
    with pytest.raises(ValidationError):
        HandoffBinding.model_validate(data)


@pytest.mark.asyncio
async def test_cli_foreign_residency_asks_operator_to_quiesce_without_unloading(monkeypatch, tmp_path, capsys):
    import argparse

    from vision_bill.distribution.handoff import _run
    events = []
    llm, ocr = ResidentLlm(events), Ocr(events)
    llm.resident = frozenset({"foreign"})
    ocr.managed_lifecycle = True
    monkeypatch.setattr("vision_bill.provider.factory.get_llm_provider", lambda config: llm)
    monkeypatch.setattr("vision_bill.provider.factory.get_ocr_provider", lambda config: ocr)
    args = argparse.Namespace(command="adopt", handoff_lock=tmp_path / "handoff.lock", model_id=["own"],
                              dedicated_coordinated_requests=True, external_eager_loading_disabled=True)
    assert await _run(args) == 1
    output = capsys.readouterr()
    assert "quiesce" in output.err.lower()
    assert json.loads(output.out)["reason_code"] == "model_handoff_foreign_residency"
    assert events == ["residency"]


@pytest.mark.asyncio
async def test_later_unadopted_model_after_ocr_load_is_not_an_admission_wait(tmp_path):
    from vision_bill.model.handoff import HandoffWait
    events = []
    handoff = ModelHandoff(tmp_path / "handoff.lock", ResidentLlm(events), Ocr(events))
    await adopt(handoff)
    async with handoff.session(monotonic() + 1, model_id="own") as owner:
        await owner.ocr_phase()
        before = handoff.journal.read_bytes()
        with pytest.raises(RuntimeError, match="model_handoff_model_not_adopted") as caught:
            await owner.llm_phase("unadopted")
        assert not isinstance(caught.value, HandoffWait)
        assert handoff.journal.read_bytes() == before
        assert events.count("load_ocr") == 1
        assert ("unload", "unadopted") not in events


@pytest.mark.asyncio
@pytest.mark.parametrize("evidence", ["failed", "foreign"])
async def test_residency_failure_after_ocr_load_is_not_an_admission_wait(tmp_path, evidence):
    from vision_bill.model.handoff import HandoffWait
    events = []
    llm = ResidentLlm(events)
    handoff = ModelHandoff(tmp_path / "handoff.lock", llm, Ocr(events))
    await adopt(handoff)
    async with handoff.session(monotonic() + 1, model_id="own") as owner:
        await owner.ocr_phase()
        before = handoff.journal.read_bytes()
        if evidence == "failed":
            async def failed_residency(*, deadline):
                raise ConnectionError("read-only evidence unavailable")
            llm.resident_model_ids = failed_residency
        else:
            llm.resident = frozenset({"foreign"})
        with pytest.raises(RuntimeError, match="model_handoff_.*residency") as caught:
            await owner.ocr_phase()
        assert not isinstance(caught.value, HandoffWait)
        assert handoff.journal.read_bytes() == before
        assert events.count("load_ocr") == 1
        assert all(not isinstance(event, tuple) for event in events)
