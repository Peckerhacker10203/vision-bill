from unittest.mock import AsyncMock
from vision_bill.model.workflow import AccessScope
from uuid import uuid4

import pytest

from vision_bill.model.experiment import ExperimentOptions
from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.provider.ocr.base import compact_ocr_text


def test_empty_primary_text_uses_recognized_blocks():
    text = compact_ocr_text({"text": "", "blocks": [{"text": "Mate 12.90"}]})
    assert text == "Mate 12.90"


def test_layout_pixels_are_not_sent_to_llm():
    with pytest.raises(ValueError, match="recognized text"):
        compact_ocr_text({"layout_info": {"input_img": [[[255, 255, 255]]]}})


def test_experiment_provider_settings_do_not_mutate_live_provider():
    live = OllamaProvider("http://localhost:11434", temperature=0.7)
    probe = live.for_experiment(ExperimentOptions(temperature=0, include_classification=False))
    assert live._temperature == 0.7
    assert probe._temperature == 0
    prompt = probe.build_prompt(["deposit", "food"])
    assert "spending_category" not in prompt
    assert "Tag every deposit" not in prompt
    assert "spending_category" in live.build_prompt(["deposit", "food"])


async def test_thinking_override_is_sent_as_top_level_option():
    live = OllamaProvider("http://localhost:11434", temperature=0.7)
    probe = live.for_experiment(ExperimentOptions(think=False))
    response = AsyncMock()
    response.message.content = "{}"
    probe._client.chat = AsyncMock(return_value=response)
    assert await probe.send_message("model", [{"role": "user", "content": "read"}]) == "{}"
    kwargs = probe._client.chat.call_args.kwargs
    assert kwargs["think"] is False
    assert "think" not in kwargs["options"]


async def test_upload_execution_preserves_all_photo_paths_and_frozen_tags(settings, tmp_path):
    import json

    from ollama import ChatResponse
    from test_experiment_scoring import receipt

    from vision_bill.service.experiment_execution import execute_submission
    from vision_bill.service.receipt_service import ReceiptService

    first, second = tmp_path / "first.jpg", tmp_path / "second.jpg"
    from PIL import Image

    from vision_bill.provider.llm.base import ModelInfo
    for path, color in ((first, "red"), (second, "blue")):
        Image.new("RGB", (5, 7), color).save(path, format="JPEG")
    provider = OllamaProvider("http://localhost:11434", temperature=0.7)
    available = ModelInfo("model")
    available.capabilities = ["vision", "completion"]
    provider.get_available_models = AsyncMock(return_value=[available])
    payload = receipt(("Milk",)).model_dump(mode="json")
    payload["line_items"][0]["source_image"] = 0
    payload["duplicate_candidates"] = []
    provider._client.chat = AsyncMock(side_effect=[ChatResponse(
        message={"role": "assistant", "content": json.dumps(payload)},
        eval_count=10, eval_duration=20_000_000), ChatResponse(
        message={"role": "assistant", "content": json.dumps({"category": "grocery", "line_items": [
            {"spending_category": "food", "tags": ["frozen_tag"]}]})})])
    service = ReceiptService(settings.images, settings.pg, provider)
    result = await execute_submission(service, "model", [first, second],
                                      ExperimentOptions(temperature=0), ["frozen_tag"])
    assert result.receipt.line_items[0].description == "Milk"
    calls = provider._client.chat.call_args_list
    request = calls[0].kwargs
    assert len(request["messages"][0]["images"]) == 2
    assert "frozen_tag" not in request["messages"][0]["content"]
    assert "frozen_tag" in calls[1].kwargs["messages"][0]["content"]
    assert result.receipt.line_items[0].tags == ["frozen_tag"]
    assert request["options"]["temperature"] == 0
    assert provider._temperature == 0.7
    assert result.metadata["image_count"] == 2
    assert result.metadata["calls"][0]["eval_ms"] == 20


async def test_forced_ocr_is_not_silently_replaced_with_vision(settings, tmp_path, monkeypatch):
    from vision_bill.service.experiment_execution import execute_submission
    from vision_bill.service.receipt_service import ReceiptService
    path = tmp_path / "receipt.jpg"
    from PIL import Image
    from test_extraction_pipeline import formatter, sidecar
    from vision_bill.provider import factory
    Image.new('RGB',(3,4)).save(path)
    ocr = sidecar()
    ocr.get_capabilities.return_value.ready = False
    monkeypatch.setattr(factory, 'get_ocr_provider', lambda settings: ocr)
    service = ReceiptService(settings.images, settings.pg, formatter())
    with pytest.raises(ValueError, match="ocr_unavailable"):
        await execute_submission(service, "model", [path], ExperimentOptions(pipeline="ocr"), [])


def test_ocr_fingerprint_detects_endpoint_and_timeout_changes():
    from types import SimpleNamespace

    from vision_bill.service.experiment_execution import ocr_fingerprint
    provider = SimpleNamespace(service_url="http://ocr-one:8080", timeout=300)
    original = ocr_fingerprint(provider)
    assert original == ocr_fingerprint(provider)
    provider.service_url = "http://ocr-two:8080"
    assert ocr_fingerprint(provider) != original
    provider.service_url = "http://ocr-one:8080"
    provider.timeout = 600
    assert ocr_fingerprint(provider) != original
    assert ocr_fingerprint(None) is None


@pytest.mark.parametrize("missing", [False, True])
async def test_reevaluation_rejects_changed_or_missing_model(missing):
    from types import SimpleNamespace
    from uuid import uuid4

    from vision_bill.service.benchmark_service import BenchmarkService
    run_id, receipt_id = uuid4(), uuid4()
    run = SimpleNamespace(receipt_ids=[receipt_id], model_ids=["model"])
    service = object.__new__(BenchmarkService)
    service._db = SimpleNamespace(run=AsyncMock(return_value=run),
                                  manifest=AsyncMock(return_value={"options": {}}))
    current = SimpleNamespace(id="model", digest="replacement")
    service._provider = SimpleNamespace(get_available_models=AsyncMock(return_value=[] if missing else [current]))
    service._run_digests = AsyncMock(return_value={"model": "original"})
    service._run_experiment = AsyncMock()
    with pytest.raises(ValueError, match="digest changed"):
        await service.reevaluate(run_id, receipt_id, "model", scope=AccessScope(user_id=uuid4()))
    service._run_experiment.assert_not_awaited()


async def test_reevaluation_enforces_saved_run_deadline():
    import asyncio
    from types import SimpleNamespace
    from uuid import uuid4

    from vision_bill.service.benchmark_service import BenchmarkService
    run_id, receipt_id = uuid4(), uuid4()
    run = SimpleNamespace(receipt_ids=[receipt_id], model_ids=["model"])
    service = object.__new__(BenchmarkService)
    service._db = SimpleNamespace(run=AsyncMock(return_value=run),
                                  manifest=AsyncMock(return_value={"options": {}}))
    current = SimpleNamespace(id="model", digest="original")
    service._provider = SimpleNamespace(get_available_models=AsyncMock(return_value=[current]))
    service._run_digests = AsyncMock(return_value={"model": "original"})
    service._timeout = AsyncMock(return_value=0.01)

    async def slow(*args):
        await asyncio.Event().wait()

    service._run_experiment = slow
    with pytest.raises(TimeoutError):
        await service.reevaluate(run_id, receipt_id, "model", scope=AccessScope(user_id=uuid4()))


async def test_untyped_replay_cannot_bypass_complete_envelope(settings, tmp_path):
    from types import SimpleNamespace
    from PIL import Image
    from vision_bill.service.experiment_execution import ExperimentFailure, execute_submission
    from vision_bill.service.receipt_service import ReceiptService
    path = tmp_path / "receipt.png"
    Image.new("RGB", (3, 4)).save(path)
    provider = OllamaProvider("http://localhost:11434")
    provider._client.chat = AsyncMock()
    service = ReceiptService(settings.images, settings.pg, provider)
    service._ocr_provider = SimpleNamespace(extract_text=AsyncMock(return_value={"text": "old"}))
    with pytest.raises(ExperimentFailure):
        await execute_submission(service, "model", [path], ExperimentOptions(pipeline="ocr"), [])
    provider._client.chat.assert_not_awaited()


async def test_typed_experiment_ocr_uses_complete_photo_pipeline(settings, tmp_path):
    import time

    from PIL import Image
    from test_extraction_pipeline import formatter, sidecar

    from vision_bill.service.experiment_execution import execute_submission
    from vision_bill.service.receipt_service import ReceiptService
    paths = [tmp_path / f"part-{i}.png" for i in range(2)]
    for path in paths:
        Image.new("RGB", (4, 5)).save(path)
    llm, ocr = formatter(False), sidecar()
    service = ReceiptService(settings.images, settings.pg, llm, ocr_provider=ocr)
    result = await execute_submission(service, "fixture", paths,
        ExperimentOptions(pipeline="ocr", include_classification=False), [])
    assert result.metadata["used_pipeline"] == "ocr"
    assert result.metadata["image_count"] == 2
    assert ocr.extract_page.await_count == 2
    assert [call.kwargs["photo_index"] for call in ocr.extract_page.await_args_list] == [0, 1]
    assert all(call.kwargs["deadline"] > time.monotonic()-1 for call in ocr.extract_page.await_args_list)


async def test_effective_pipeline_identity_binds_profile_runtime_budgets_and_fallback(settings):
    from test_extraction_pipeline import formatter, sidecar
    from vision_bill.service.receipt_service import ReceiptService
    from vision_bill.service import experiment_execution as execution
    service = ReceiptService(settings.images, settings.pg, formatter(), ocr_provider=sidecar())
    assert hasattr(execution, 'pipeline_identity'), 'effective pipeline identity missing'
    options = ExperimentOptions(pipeline='ocr', include_classification=False)
    original = await execution.pipeline_identity(service, options)
    caps = service._ocr_provider.get_capabilities.return_value
    caps.profile_id = 'different'
    assert await execution.pipeline_identity(service, options) != original
    caps.profile_id = 'fixed-profile'
    caps.runtime_fingerprint = 'changed'
    assert await execution.pipeline_identity(service, options) != original
    caps.runtime_fingerprint = None
    service._provider.max_output_tokens += 1
    assert await execution.pipeline_identity(service, options) != original
    assert await execution.pipeline_identity(service, options.model_copy(update={'allow_ocr_fallback':True})) != original


async def test_run_manifest_preserves_upload_policy_instead_of_forcing_configured_ocr(settings):
    from types import SimpleNamespace
    from test_experiment_scoring import receipt
    from test_extraction_pipeline import formatter, sidecar
    from vision_bill.config import OcrSettings
    from vision_bill.service.benchmark_service import BenchmarkService
    from vision_bill.model.benchmark import BenchmarkCreate
    service=object.__new__(BenchmarkService)
    service._provider=formatter()
    service._receipts=SimpleNamespace(_provider=service._provider,_ocr_provider=sidecar(),_ocr_settings=OcrSettings(enabled=False),list_tags=AsyncMock(return_value=[]))
    rid=uuid4()
    service._select_receipts=AsyncMock(return_value=[rid])
    service._capture_input=AsyncMock(return_value={'receipt_id':str(rid),'expected':receipt().model_dump(mode='json'),'images':[{'path':'source','sha256':'a'*64}]})
    service._db=SimpleNamespace(create_run=AsyncMock())
    await service.create_run(BenchmarkCreate(model_ids=['fixture']),scope=AccessScope(user_id=uuid4()))
    manifest=service._db.create_run.call_args.kwargs['manifest']
    assert manifest['inputs'][0]['pipeline']=='upload'
    assert manifest['identity']['ocr_settings']['enabled'] is False


async def test_ocr_benchmark_discovers_text_only_formatter_models(settings):
    from types import SimpleNamespace
    from test_experiment_scoring import receipt
    from test_extraction_pipeline import formatter, sidecar
    from vision_bill.config import OcrSettings
    from vision_bill.service.benchmark_service import BenchmarkService
    from vision_bill.model.benchmark import BenchmarkCreate
    service=object.__new__(BenchmarkService)
    service._provider=formatter(False)
    model=(await service._provider.get_available_models())[0]
    async def models(*,capability='vision'):
        return [model] if capability=='text' else []
    service._provider.get_available_models=models
    service._receipts=SimpleNamespace(_provider=service._provider,_ocr_provider=sidecar(),_ocr_settings=OcrSettings(enabled=True),list_tags=AsyncMock(return_value=[]))
    rid=uuid4()
    service._select_receipts=AsyncMock(return_value=[rid])
    service._capture_input=AsyncMock(return_value={'receipt_id':str(rid),'expected':receipt().model_dump(mode='json'),'images':[{'path':'source','sha256':'a'*64}]})
    service._db=SimpleNamespace(create_run=AsyncMock())
    await service.create_run(BenchmarkCreate(model_ids=['fixture'],experiment=ExperimentOptions(pipeline='ocr')),scope=AccessScope(user_id=uuid4()))
    assert service._db.create_run.call_args.kwargs['model_ids']==['fixture']


@pytest.mark.parametrize('intent,enabled,fail_second,expected', [
    ('upload',False,False,'direct_vision'),
    ('upload',True,False,'ocr'),
    ('upload',True,True,'direct_vision'),
    ('direct_vision',True,False,'direct_vision'),
    ('ocr',False,False,'ocr'),
    ('ocr',False,True,None),
])
async def test_serialized_experiment_intent_preserves_effective_ocr_policy(settings,tmp_path,monkeypatch,intent,enabled,fail_second,expected):
    from PIL import Image
    from test_extraction_pipeline import formatter, sidecar
    from vision_bill.config import OcrSettings
    from vision_bill.service.receipt_service import ReceiptService
    from vision_bill.service.experiment_execution import execute_submission, ExperimentFailure, pipeline_identity
    from vision_bill.provider import factory
    paths=[tmp_path/f'{i}.png' for i in range(2)]
    for path in paths: Image.new('RGB',(3,4)).save(path)
    ocr=sidecar(1 if fail_second else None)
    monkeypatch.setattr(factory,'get_ocr_provider',lambda settings:ocr)
    service=ReceiptService(settings.images,settings.pg,formatter(),ocr_provider=ocr if enabled else None,
        ocr_settings=OcrSettings(enabled=enabled,required=False,allow_vision_fallback=True))
    # Exactly the JSON options serialized in a manifest and reconstructed for replay.
    original=ExperimentOptions(pipeline=intent,include_classification=False)
    options=ExperimentOptions.model_validate_json(original.model_dump_json())
    identity=await pipeline_identity(service,options)
    if expected is None:
        with pytest.raises(ExperimentFailure): await execute_submission(service,'fixture',paths,options,[])
        service._provider._client.chat.assert_not_awaited()
    else:
        result=await execute_submission(service,'fixture',paths,options,[])
        assert result.metadata['used_pipeline']==expected
        assert result.metadata['fallback_used'] is (intent=='upload' and fail_second)
    if intent=='direct_vision' or (intent=='upload' and not enabled):
        ocr.extract_page.assert_not_awaited()
    if intent=='ocr':
        assert identity['ocr_settings']['enabled'] and identity['ocr_settings']['required']
        assert not identity['ocr_settings']['allow_vision_fallback']
    assert service._ocr_settings.enabled is enabled and not service._ocr_settings.required


@pytest.mark.parametrize('failure', [ConnectionRefusedError('database reconnect failed'), TimeoutError('database acquisition timed out')])
async def test_reference_selection_propagates_infrastructure_errors(failure):
    from types import SimpleNamespace
    from vision_bill.model.benchmark import BenchmarkCreate
    from vision_bill.service.benchmark_service import BenchmarkService
    candidates = [uuid4(),uuid4()]
    service = object.__new__(BenchmarkService)
    service._db = SimpleNamespace(reference_candidates=AsyncMock(return_value=candidates),
        current_reference=AsyncMock(side_effect=[failure,object()]))
    with pytest.raises(type(failure),match=str(failure)):
        await service._select_receipts(BenchmarkCreate(limit=1),scope=AccessScope(user_id=uuid4()))
    assert service._db.current_reference.await_count == 1
