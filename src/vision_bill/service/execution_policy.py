"""The server-selected policy frozen for newly accepted receipt submissions."""

from typing import Any

from ..config import OcrSettings, Settings

EXECUTION_POLICY_VERSION = 2


def snapshot_ocr(snapshot: dict[str, Any]) -> OcrSettings:
    """Validate current and historical shapes without rewriting frozen intent."""
    old = {"ocr", "temperature", "max_output_tokens"}
    current = old | {"allow_ocr_fallback", "execution_policy_version"}
    permission = snapshot.get("allow_ocr_fallback", False)
    if set(snapshot) not in (old, old | {"allow_ocr_fallback"}, current) or type(permission) is not bool:
        raise ValueError("execution_snapshot_invalid")
    if "execution_policy_version" in snapshot and (
        type(snapshot["execution_policy_version"]) is not int
        or snapshot["execution_policy_version"] != EXECUTION_POLICY_VERSION
        or permission
    ):
        raise ValueError("execution_snapshot_invalid")
    return OcrSettings.model_validate(snapshot["ocr"])


def policy_hold_reason(snapshot: dict[str, Any], current_ocr: OcrSettings) -> str | None:
    """Hold obsolete routing or an effective historical image fallback grant."""
    ocr = snapshot_ocr(snapshot)
    if (
        ocr.enabled != current_ocr.enabled
        or (ocr.enabled and ocr.service_url != current_ocr.service_url)
        or (
            "execution_policy_version" not in snapshot
            and ocr.enabled and not ocr.required and ocr.allow_vision_fallback
            and snapshot.get("allow_ocr_fallback", False)
        )
    ):
        return "execution_policy_changed"
    return None


def current_snapshot(settings: Settings) -> dict[str, Any]:
    ocr = settings.ocr.model_copy(update={
        "allow_vision_fallback": False,
        "vision_fallback_model_id": None,
    })
    return {
        "execution_policy_version": EXECUTION_POLICY_VERSION,
        "allow_ocr_fallback": False,
        "ocr": ocr.model_dump(mode="json"),
        "temperature": settings.llm.temperature,
        "max_output_tokens": settings.llm.max_output_tokens,
    }
