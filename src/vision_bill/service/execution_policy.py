"""The server-selected policy frozen for newly accepted receipt submissions."""

import re
from typing import TYPE_CHECKING, Any

from ..config import OcrSettings, Settings
from ..model.handoff import HandoffWait

if TYPE_CHECKING:
    from .gpu_handoff import ModelHandoff

EXECUTION_POLICY_VERSION = 3


def snapshot_ocr(snapshot: dict[str, Any]) -> OcrSettings:
    """Validate current and historical shapes without rewriting frozen intent."""
    old = {"ocr", "temperature", "max_output_tokens"}
    versioned = old | {"allow_ocr_fallback", "execution_policy_version"}
    shapes = {2: versioned, 3: versioned | {"managed_binding_sha256"}}
    permission = snapshot.get("allow_ocr_fallback", False)
    version = snapshot.get("execution_policy_version")
    binding = snapshot.get("managed_binding_sha256")
    if type(permission) is not bool:
        raise ValueError("execution_snapshot_invalid")
    if "execution_policy_version" in snapshot:
        if type(version) is not int or permission or set(snapshot) != shapes.get(version):
            raise ValueError("execution_snapshot_invalid")
    elif set(snapshot) not in (old, old | {"allow_ocr_fallback"}):
        raise ValueError("execution_snapshot_invalid")
    if binding is not None and (
        not isinstance(binding, str) or not re.fullmatch("[0-9a-f]{64}", binding)
    ):
        raise ValueError("execution_snapshot_invalid")
    return OcrSettings.model_validate(snapshot["ocr"])


def policy_hold_reason(snapshot: dict[str, Any], current_ocr: OcrSettings) -> str | None:
    """Hold obsolete routing or an effective historical image fallback grant."""
    ocr = snapshot_ocr(snapshot)
    if (
        ocr.enabled != current_ocr.enabled
        or (
            ocr.enabled
            and (
                ocr.provider != current_ocr.provider
                or ocr.service_url.rstrip("/") != current_ocr.service_url.rstrip("/")
                or ocr.manage_model_lifecycle != current_ocr.manage_model_lifecycle
            )
        )
        or (
            managed_ocr(ocr)
            and (
                ocr.coordination_path.resolve() != current_ocr.coordination_path.resolve()
                or "managed_binding_sha256" not in snapshot
            )
        )
        or (
            "execution_policy_version" not in snapshot
            and ocr.enabled
            and not ocr.required
            and ocr.allow_vision_fallback
            and snapshot.get("allow_ocr_fallback", False)
        )
    ):
        return "execution_policy_changed"
    return None


def managed_ocr(settings: OcrSettings) -> bool:
    return settings.enabled and settings.manage_model_lifecycle


def current_snapshot(
    settings: Settings, *, handoff: "ModelHandoff | None" = None
) -> dict[str, Any]:
    binding = None
    if handoff is not None:
        try:
            # Adoption publishes the binding atomically. Execution still
            # validates its current contents under the lifecycle lock.
            binding = handoff.binding_sha256()
        except (HandoffWait, OSError, ValueError):
            # A job accepted while dedication is unavailable needs explicit adoption.
            pass
    ocr = settings.ocr.model_copy(
        update={
            "allow_vision_fallback": False,
            "vision_fallback_model_id": None,
        }
    )
    return {
        "execution_policy_version": EXECUTION_POLICY_VERSION,
        "managed_binding_sha256": binding,
        "allow_ocr_fallback": False,
        "ocr": ocr.model_dump(mode="json"),
        "temperature": settings.llm.temperature,
        "max_output_tokens": settings.llm.max_output_tokens,
    }
