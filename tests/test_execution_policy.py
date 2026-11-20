"""Canonical configuration and durable strict processing policy."""

import pytest

from vision_bill.config import OcrSettings
from vision_bill.service.execution_policy import snapshot_ocr


@pytest.mark.parametrize(
    "endpoint,enabled,url",
    [
        ("", False, ""),
        ("  ", False, ""),
        (" http://local-ocr:8082 ", True, "http://local-ocr:8082"),
    ],
)
def test_explicit_endpoint_selects_ocr_without_legacy_activation(endpoint, enabled, url):
    settings = OcrSettings.model_validate({"endpoint": endpoint})
    assert settings.enabled is enabled
    assert settings.service_url == url
    assert "endpoint" not in settings.model_dump(mode="json")


def test_legacy_default_url_is_not_activation():
    assert not OcrSettings().enabled
    assert not OcrSettings(enabled=False, service_url="http://old-ocr").enabled
    assert OcrSettings(enabled=True).enabled


@pytest.mark.parametrize(
    "endpoint",
    [
        None,
        True,
        123,
        [],
        "ftp://host",
        "file:///tmp/ocr",
        "ocr",
        "http://",
        "http:host",
        "http:///host",
    ],
)
def test_canonical_endpoint_rejects_non_http_urls(endpoint):
    with pytest.raises(ValueError):
        OcrSettings.model_validate({"endpoint": endpoint})


@pytest.mark.parametrize("endpoint", ["http://127.0.0.1:8082/api/ocr", "https://ocr.local/prefix"])
def test_canonical_endpoint_preserves_http_path_prefix(endpoint):
    settings = OcrSettings.model_validate(
        {"endpoint": endpoint, "enabled": False, "service_url": "http://legacy"}
    )
    assert settings.enabled and settings.service_url == endpoint


def test_blank_endpoint_does_not_silently_rewrite_required_legacy_policy():
    with pytest.raises(ValueError):
        OcrSettings.model_validate({"endpoint": "", "enabled": True, "required": True})


def test_new_snapshot_disables_effective_fallback_and_is_accepted(settings):
    from vision_bill.service.execution_policy import current_snapshot

    settings.ocr = OcrSettings(
        enabled=True, allow_vision_fallback=True, vision_fallback_model_id="old-alternate"
    )
    snapshot = current_snapshot(settings)
    assert snapshot["execution_policy_version"] == 3
    assert snapshot["managed_binding_sha256"] is None
    assert snapshot["allow_ocr_fallback"] is False
    assert snapshot["ocr"]["allow_vision_fallback"] is False
    assert snapshot["ocr"]["vision_fallback_model_id"] is None
    assert snapshot_ocr(snapshot).allow_vision_fallback is False
    assert settings.ocr.allow_vision_fallback  # building the snapshot never mutates live settings


@pytest.mark.parametrize(
    "version,permission", [(1, False), (4, False), (True, False), (2, True), (3, True)]
)
def test_versioned_snapshot_rejects_unknown_or_permissive_policy(settings, version, permission):
    from vision_bill.service.execution_policy import current_snapshot

    snapshot = current_snapshot(settings)
    snapshot.update(execution_policy_version=version, allow_ocr_fallback=permission)
    with pytest.raises(ValueError, match="execution_snapshot_invalid"):
        snapshot_ocr(snapshot)


@pytest.mark.parametrize("version", [None, 2, 3])
@pytest.mark.parametrize("change", ["enable", "disable", "endpoint"])
def test_policy_holds_every_snapshot_version_on_operator_change(settings, version, change):
    from copy import deepcopy

    from vision_bill.service import execution_policy

    settings.ocr = OcrSettings(enabled=change != "enable", service_url="http://old")
    snapshot = execution_policy.current_snapshot(settings)
    if version != 3:
        snapshot.pop("managed_binding_sha256", None)
        snapshot["execution_policy_version"] = version
    if version is None:
        snapshot.pop("execution_policy_version")
    before = deepcopy(snapshot)
    current = OcrSettings(enabled=change != "disable", service_url="http://new")
    assert execution_policy.policy_hold_reason(snapshot, current) == "execution_policy_changed"
    assert snapshot == before


@pytest.mark.parametrize(
    "permission,required,grant,held",
    [
        (True, False, True, True),
        (False, False, True, False),
        (True, True, True, False),
        (True, False, False, False),
    ],
)
def test_legacy_policy_holds_only_effective_image_fallback_grants(
    settings, permission, required, grant, held
):
    from vision_bill.service import execution_policy

    settings.ocr = OcrSettings(
        enabled=True,
        required=required,
        allow_vision_fallback=grant,
        manage_model_lifecycle=False,
    )
    snapshot = {
        "ocr": settings.ocr.model_dump(mode="json"),
        "temperature": 0.2,
        "max_output_tokens": 8192,
        "allow_ocr_fallback": permission,
    }
    reason = execution_policy.policy_hold_reason(snapshot, settings.ocr)
    assert reason == ("execution_policy_changed" if held else None)


def test_compatible_frozen_formatter_and_inactive_endpoint_are_retained(settings):
    from vision_bill.service import execution_policy

    snapshot = execution_policy.current_snapshot(settings)
    current = settings.ocr.model_copy(update={"service_url": "http://other", "timeout_seconds": 99})
    assert execution_policy.policy_hold_reason(snapshot, current) is None


def test_old_shared_policy_holds_after_switch_to_separate_gpu(settings):
    from vision_bill.service.execution_policy import current_snapshot, policy_hold_reason

    settings.ocr = OcrSettings(enabled=True)
    snapshot = current_snapshot(settings)
    snapshot["ocr"].pop("shared_gpu", None)
    assert policy_hold_reason(snapshot, settings.ocr) is None
    separate_gpu = settings.ocr.model_copy(update={"shared_gpu": False})
    assert policy_hold_reason(snapshot, separate_gpu) == "execution_policy_changed"


@pytest.mark.parametrize("version", [True, 1, 4, "2"])
def test_policy_hold_parser_rejects_invalid_versions(settings, version):
    from vision_bill.service import execution_policy

    snapshot = execution_policy.current_snapshot(settings)
    snapshot["execution_policy_version"] = version
    with pytest.raises(ValueError, match="execution_snapshot_invalid"):
        execution_policy.policy_hold_reason(snapshot, settings.ocr)


@pytest.mark.parametrize("version", [None, 2])
def test_legacy_direct_remains_compatible_without_binding(settings, version):
    from vision_bill.service.execution_policy import policy_hold_reason

    snapshot = {
        "ocr": settings.ocr.model_dump(mode="json"),
        "temperature": 0.13,
        "max_output_tokens": 1234,
    }
    if version is not None:
        snapshot.update(execution_policy_version=version, allow_ocr_fallback=False)
    assert policy_hold_reason(snapshot, settings.ocr) is None


@pytest.mark.parametrize("version", [None, 2])
def test_historical_managed_work_requires_explicit_adoption(settings, version):
    from vision_bill.service.execution_policy import policy_hold_reason

    settings.ocr = OcrSettings(enabled=True, manage_model_lifecycle=True)
    snapshot = {
        "ocr": settings.ocr.model_dump(mode="json"),
        "temperature": 0.13,
        "max_output_tokens": 1234,
    }
    if version is not None:
        snapshot.update(execution_policy_version=version, allow_ocr_fallback=False)
    assert policy_hold_reason(snapshot, settings.ocr) == "execution_policy_changed"


@pytest.mark.parametrize("digest", ["", "A" * 64, "z" * 64, "a" * 63, False, 0])
def test_invalid_binding_digest_is_rejected(settings, digest):
    from vision_bill.service.execution_policy import current_snapshot, snapshot_ocr

    snapshot = current_snapshot(settings)
    snapshot["managed_binding_sha256"] = digest
    with pytest.raises(ValueError, match="execution_snapshot_invalid"):
        snapshot_ocr(snapshot)
