"""The shared parser and repair path keeps nullable values and private attempts."""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from ollama import ChatResponse, ListResponse, ShowResponse
from test_ocr_provider import wire_page

from vision_bill.model.extraction import ReceiptInput
from vision_bill.model.ocr import OcrResponse, SourceRef
from vision_bill.provider.llm.ollama import OllamaProvider
from vision_bill.provider.llm.openai import OpenAIProvider


def ocr_input():
    page = OcrResponse.model_validate(wire_page())
    ref = SourceRef(
        part_id=uuid4(),
        photo_index=0,
        source_sha256=page.source_sha256,
        view_id=page.working_image.view_id,
        coordinate_space="working",
    )
    return ReceiptInput(
        mode="ocr", image_paths=["original.gif"], ocr_pages=[page], source_refs=[ref]
    )


@pytest.mark.parametrize("backend", ["ollama", "openai"])
async def test_partial_null_draft_retains_actual_model_finish_and_bounded_attempts(backend, caplog):
    provider = (
        OllamaProvider("http://local")
        if backend == "ollama"
        else OpenAIProvider("http://local", "local")
    )
    payload = json.dumps({"merchant_name": "Private Shop", "total": None, "line_items": None})
    if backend == "ollama":
        provider._client.chat = AsyncMock(
            return_value=ChatResponse(
                model="actual",
                done_reason="length",
                message={"role": "assistant", "content": payload},
            )
        )
    else:
        provider._client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                model="actual",
                choices=[
                    SimpleNamespace(
                        finish_reason="length", message=SimpleNamespace(content=payload)
                    )
                ],
                usage=None,
            )
        )
    assert hasattr(provider, "extract_receipt_input"), "shared typed provider path missing"
    output = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=[], deadline=time.monotonic() + 2
    )
    assert output.receipt is None and output.draft.merchant_name == "Private Shop"
    assert output.draft.total is None and output.draft.discount_total is None
    assert output.telemetry.calls[-1]["actual_model"] == "actual"
    assert output.telemetry.finish_reason == "length"
    assert output.telemetry.output_tokens is None
    assert len(output.telemetry.calls) == 3
    assert "Private Shop" not in caplog.text
    call = (
        provider._client.chat.call_args
        if backend == "ollama"
        else provider._client.chat.completions.create.call_args
    )
    assert (
        call.kwargs["options"]["num_predict"] if backend == "ollama" else call.kwargs["max_tokens"]
    ) == 8192


async def test_ollama_text_discovery_includes_completion_without_changing_default():
    provider = OllamaProvider("http://local")
    provider._client.list = AsyncMock(
        return_value=ListResponse(models=[{"model": "text"}, {"model": "vision"}])
    )
    provider._client.show = AsyncMock(
        side_effect=lambda name: ShowResponse(
            capabilities=["completion"] + (["vision"] if name == "vision" else []),
            model_info={"llama.context_length": 32768},
        )
    )
    assert [m.id for m in await provider.get_available_models()] == ["vision"]
    assert (
        "capability" in __import__("inspect").signature(provider.get_available_models).parameters
    ), "explicit discovery filter missing"
    models = await provider.get_available_models(capability="text")
    assert [m.id for m in models] == ["text", "vision"]
    assert models[0].context_window_tokens == 32768


async def test_openai_text_discovery_keeps_unknown_text_models():
    provider = OpenAIProvider("http://local", "local")
    provider._client.models.with_raw_response.list = AsyncMock(
        return_value=SimpleNamespace(
            http_response=SimpleNamespace(
                content=json.dumps(
                    {
                        "data": [
                            {"id": "text"},
                            {
                                "id": "vision",
                                "architecture": {"input_modalities": ["image", "text"]},
                            },
                        ]
                    }
                ).encode()
            )
        )
    )
    assert [m.id for m in await provider.get_available_models()] == ["vision"]
    assert (
        "capability" in __import__("inspect").signature(provider.get_available_models).parameters
    ), "explicit discovery filter missing"
    assert [m.id for m in await provider.get_available_models(capability="text")] == [
        "text",
        "vision",
    ]


async def test_source_aware_overlap_keeps_original_draft_and_verified_decisions():
    from test_multi_image_extraction import extraction, item

    from vision_bill.model.experiment import ExperimentOptions

    inp = ocr_input()
    page = OcrResponse.model_validate(wire_page(1, "b" * 64))
    ref = SourceRef(
        part_id=uuid4(),
        photo_index=1,
        source_sha256="b" * 64,
        view_id=page.working_image.view_id,
        coordinate_space="working",
    )
    inp = ReceiptInput(
        mode="ocr",
        image_paths=["a", "b"],
        ocr_pages=[*inp.ocr_pages, page],
        source_refs=[*inp.source_refs, ref],
    )
    payload = extraction([item(), item(), item(source=1), item("Bread", source=1)], [(1, 2)])
    payload["overlap_decisions"] = [{"reason": "printed suffix and prefix"}]
    provider = OllamaProvider("http://local").for_experiment(
        ExperimentOptions(include_classification=False)
    )
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    assert hasattr(provider, "extract_receipt_input"), "typed path missing"
    result = await provider.extract_receipt_input(
        "requested", inp, tags=[], deadline=time.monotonic() + 2
    )
    assert len(result.draft.line_items) == 4
    assert len(result.receipt.line_items) == 3
    assert result.field_sources["line_items.2"] == [ref]
    assert result.overlap_decisions[-1]["verified"] is True
    assert result.overlap_decisions[-1]["removed_index"] == 2


async def test_unknown_source_ref_cannot_survive_as_valid_draft():
    from test_experiment_scoring import receipt

    inp = ocr_input()
    invalid = inp.source_refs[0].model_copy(update={"source_sha256": "c" * 64})
    payload = receipt().model_dump(mode="json")
    payload["field_sources"] = {"merchant_name": [invalid.model_dump(mode="json")]}
    provider = OllamaProvider("http://local")
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    result = await provider.extract_receipt_input(
        "requested", inp, tags=[], deadline=time.monotonic() + 2
    )
    assert result.receipt is None and result.draft is None
    assert any(f.code == "formatter_invalid_response" for f in result.findings)


async def test_classification_is_separate_cannot_change_observations_and_is_in_telemetry():
    from test_experiment_scoring import receipt

    provider = OllamaProvider("http://local")
    provider._client.chat = AsyncMock(
        side_effect=[
            ChatResponse(
                model="extractor",
                message={"role": "assistant", "content": receipt().model_dump_json()},
            ),
            ChatResponse(
                model="classifier",
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "total": "99999",
                            "category": "grocery",
                            "line_items": [
                                {
                                    "description": "changed",
                                    "tags": ["food"],
                                    "spending_category": "food",
                                }
                                for _ in receipt().line_items
                            ],
                        }
                    ),
                },
            ),
        ]
    )
    result = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=["food"], deadline=time.monotonic() + 2
    )
    assert result.receipt.total == receipt().total
    assert [r.description for r in result.receipt.line_items] == [
        r.description for r in receipt().line_items
    ]
    assert len(result.telemetry.calls) == 2
    assert result.telemetry.calls[1]["stage"] == "classification"


async def test_supplied_and_explicit_null_fields_are_distinguished():
    provider = OllamaProvider("http://local")
    provider.send_message = AsyncMock(return_value='{"merchant_name":"Shop","total":null}')
    result = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=[], deadline=time.monotonic() + 2
    )
    assert result.telemetry.calls[0]["supplied_fields"] == ["merchant_name", "total"]
    assert result.telemetry.calls[0]["null_fields"] == ["total"]
    assert result.draft.discount_total is None


async def test_openai_discovery_records_only_declared_context_capacity():
    provider = OpenAIProvider("http://local", "local")
    provider._client.models.with_raw_response.list = AsyncMock(
        return_value=SimpleNamespace(
            http_response=SimpleNamespace(
                content=json.dumps(
                    {"data": [{"id": "text", "context_length": 16384}, {"id": "unknown"}]}
                ).encode()
            )
        )
    )
    models = await provider.get_available_models(capability="text")
    assert models[0].context_window_tokens == 16384
    assert models[1].context_window_tokens is None


@pytest.mark.parametrize("sources", [{"total": 5}, {"total": None}])
async def test_malformed_field_sources_trigger_bounded_repairs(sources):
    from test_experiment_scoring import receipt

    provider = OllamaProvider("http://local")
    payload = receipt().model_dump(mode="json")
    payload["field_sources"] = sources
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    result = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=[], deadline=time.monotonic() + 2
    )
    assert result.receipt is None
    assert len(result.telemetry.calls) == 3


async def test_formatter_does_not_receive_wire_telemetry_or_image_transform_payloads():
    from test_experiment_scoring import receipt

    inp = ocr_input()
    inp.ocr_pages[0].telemetry = {"png_base64": "PRIVATE_IMAGE_PAYLOAD", "pixels": [[[255]]]}
    inp.ocr_pages[0].working_image.transforms = [{"raw_image": "PRIVATE_TRANSFORM_PAYLOAD"}]
    from vision_bill.model.experiment import ExperimentOptions

    provider = OllamaProvider("http://local").for_experiment(
        ExperimentOptions(include_classification=False)
    )
    provider.send_message = AsyncMock(return_value=receipt().model_dump_json())
    await provider.extract_receipt_input("requested", inp, tags=[], deadline=time.monotonic() + 2)
    content = provider.send_message.call_args.args[1][0]["content"]
    assert "PRIVATE_IMAGE_PAYLOAD" not in content
    assert "PRIVATE_TRANSFORM_PAYLOAD" not in content
    assert "pixels" not in content


@pytest.mark.parametrize("backend", ["ollama", "openai"])
async def test_formatter_discovery_excludes_embedding_only_models(backend):
    if backend == "ollama":
        provider = OllamaProvider("http://local")
        provider._client.list = AsyncMock(
            return_value=ListResponse(models=[{"model": "embed"}, {"model": "text"}])
        )
        provider._client.show = AsyncMock(
            side_effect=lambda name: ShowResponse(
                capabilities=["embedding"] if name == "embed" else ["completion"], model_info={}
            )
        )
    else:
        provider = OpenAIProvider("http://local", "local")
        provider._client.models.with_raw_response.list = AsyncMock(
            return_value=SimpleNamespace(
                http_response=SimpleNamespace(
                    content=json.dumps(
                        {
                            "data": [
                                {"id": "embed", "capabilities": ["embedding"]},
                                {"id": "text", "capabilities": ["completion"]},
                            ]
                        }
                    ).encode()
                )
            )
        )
    assert [model.id for model in await provider.get_available_models(capability="text")] == [
        "text"
    ]


async def test_openai_data_capabilities_are_honored_for_direct_discovery():
    provider = OpenAIProvider("http://local", "local")
    provider._client.models.with_raw_response.list = AsyncMock(
        return_value=SimpleNamespace(
            http_response=SimpleNamespace(
                content=json.dumps(
                    {
                        "data": [
                            {"id": "vision", "capabilities": ["vision"]},
                            {"id": "text", "capabilities": ["completion"]},
                        ]
                    }
                ).encode()
            )
        )
    )
    assert [model.id for model in await provider.get_available_models()] == ["vision"]


async def test_null_overlap_candidates_are_repaired_without_crashing():
    from test_experiment_scoring import receipt

    provider = OllamaProvider("http://local")
    payload = receipt().model_dump(mode="json")
    payload["duplicate_candidates"] = None
    provider.send_message = AsyncMock(return_value=json.dumps(payload))
    result = await provider.extract_receipt_input(
        "requested", ocr_input(), tags=[], deadline=time.monotonic() + 2
    )
    assert result.receipt is None and len(result.telemetry.calls) == 3
