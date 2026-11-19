"""Admin opt-in for the project's idle, device-specific OCR service."""
import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from uuid import UUID

import httpx
from ocr_contract.activation import (
    ActivationBusy,
    ActivationConflict,
    ActivationPhase,
    ActivationStateError,
    ActivationStore,
    MaintenanceIntent,
)

from ..config import OcrSettings, Settings, runtime_restart_required
from ..distribution.handoff import adopt_binding
from ..model.dedicated_ocr import DedicatedOcrStatus
from ..model.handoff import HandoffWait
from ..provider.llm.base import LLMProvider
from ..provider.ocr.paddleocr import PaddleOcrProvider
from .activation_admission import activation_handoff_path, unsettled_direct_calls
from .gpu_handoff import ModelHandoff


class DedicatedOcrActivation:
    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider,
        apply_route: Callable[[PaddleOcrProvider, OcrSettings], None],
        wake_receipts: Callable[[], None],
        *,
        store: ActivationStore | None = None,
        handoff_path: Path | None = None,
    ):
        self.settings = settings
        self.provider = provider
        self.apply_route = apply_route
        self.wake_receipts = wake_receipts
        self.store = store if store is not None else ActivationStore()
        self.external_mode = settings.ocr.enabled or not settings.ocr.service_url
        self.prior_handoff = settings.ocr.coordination_path
        self.ocr_settings = settings.ocr.model_copy(update={
            "enabled": True,
            "allow_vision_fallback": False,
            "manage_model_lifecycle": True,
            "service_url": os.environ.get("OCR_ACTIVATION_ENDPOINT", settings.ocr.service_url),
            "coordination_path": handoff_path if handoff_path is not None else activation_handoff_path(),
        })
        self.ocr = PaddleOcrProvider(
            self.ocr_settings.service_url, self.ocr_settings.timeout_seconds
        )
        self.handoff = ModelHandoff(self.ocr_settings.coordination_path, provider, self.ocr)
        self._applied: UUID | None = None
        self._blocker: str | None = None

    async def _controller_is_live(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                response = await client.get(f"{self.ocr.service_url}/health")
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def _runtime_is_serving(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                response = await client.get(f"{self.ocr.service_url}/info")
                response.raise_for_status()
                info = response.json()
                return isinstance(info, dict) and info.get("service_mode") == "managed"
        except (httpx.HTTPError, ValueError):
            return False

    def _lifecycle_blocker(self) -> str | None:
        if not self.provider.supports_model_release or runtime_restart_required(self.settings):
            return "llm_lifecycle_required"
        return None

    async def status(self) -> DedicatedOcrStatus:
        if self.external_mode:
            return DedicatedOcrStatus(state="unavailable", reason="ocr_config_managed_externally")
        try:
            request = self.store.read_request()
            current = self.store.read_status(request) if request else None
        except ActivationStateError:
            return DedicatedOcrStatus(state="action_required", reason="ocr_activation_state_invalid")
        if current and current.phase == ActivationPhase.READY:
            if self._applied == current.generation:
                if await self._runtime_is_serving():
                    return DedicatedOcrStatus(state="active")
                return DedicatedOcrStatus(state="action_required", reason="ocr_runtime_unavailable")
            if self._blocker:
                return DedicatedOcrStatus(state="action_required", reason=self._blocker)
            return DedicatedOcrStatus(state="installing")
        if current and current.phase == ActivationPhase.FAILED:
            return DedicatedOcrStatus(state="failed", reason=current.reason_code or "ocr_setup_failed")
        if request:
            return DedicatedOcrStatus(state="installing" if current else "requested")
        if not await self._controller_is_live():
            return DedicatedOcrStatus(state="unavailable", reason="gpu_profile_required")
        if blocker := self._blocker or self._lifecycle_blocker():
            return DedicatedOcrStatus(state="action_required", reason=blocker)
        return DedicatedOcrStatus(state="disabled")

    async def submit(self, *, confirm_handoff: bool) -> DedicatedOcrStatus:
        if self.external_mode:
            return await self.status()
        if not confirm_handoff:
            self._blocker = "ocr_handoff_confirmation_required"
            return DedicatedOcrStatus(state="action_required", reason=self._blocker)
        if blocker := self._lifecycle_blocker():
            self._blocker = blocker
            return DedicatedOcrStatus(state="action_required", reason=blocker)

        try:
            async with asyncio.timeout(6):
                with self.store.admission(exclusive=True):
                    current = self.store.read_request()
                    phase = self.store.read_status(current) if current else None
                    if current and (phase is None or phase.phase != ActivationPhase.FAILED):
                        return await self.status()
                    if not await self._controller_is_live():
                        self._blocker = "gpu_profile_required"
                        return DedicatedOcrStatus(state="action_required", reason=self._blocker)
                    if unsettled_direct_calls(self.store):
                        self._blocker = "llm_request_recovery_required"
                        return DedicatedOcrStatus(state="action_required", reason=self._blocker)
                    if self.prior_handoff != self.handoff.path and (
                        self.prior_handoff.with_suffix(".json").exists()
                        or self.prior_handoff.with_suffix(".binding.json").exists()
                    ):
                        self._blocker = "ocr_existing_handoff"
                        return DedicatedOcrStatus(state="action_required", reason=self._blocker)
                    intent = MaintenanceIntent(os.environ.get(
                        "OCR_ACTIVATION_MAINTENANCE_INTENT",
                        MaintenanceIntent.EXTERNAL_LLM_QUIESCED.value,
                    ))
                    await adopt_binding(
                        self.handoff, [self.settings.llm.model_name],
                        dedicated_coordinated_requests=True,
                        external_eager_loading_disabled=True,
                        deadline=monotonic() + 3,
                    )
                    self.store.submit_request(
                        intent, retry_generation=current.generation if current else None,
                    )
                    self._blocker = None
                    return DedicatedOcrStatus(state="requested")
        except (ActivationBusy, ActivationConflict):
            self._blocker = "llm_quiescence_required"
        except (ActivationStateError, OSError, ValueError):
            self._blocker = "ocr_activation_state_invalid"
        except (HandoffWait, httpx.HTTPError, TimeoutError):
            self._blocker = "llm_quiescence_required"
        return DedicatedOcrStatus(state="action_required", reason=self._blocker)

    async def reconcile(self) -> None:
        if self.external_mode:
            return
        try:
            request = self.store.read_request()
            if request is None or request.generation == self._applied:
                return
            current = self.store.read_status(request)
            if current is None or current.phase != ActivationPhase.READY:
                return
            with self.store.admission(exclusive=True):
                binding = self.handoff.read_binding()
                if self._lifecycle_blocker() or self.settings.llm.model_name not in binding.model_ids:
                    self._blocker = "llm_lifecycle_required"
                    return
                # Existing services keep the same provider and receipt store. All
                # route-facing references are replaced together under admission.
                self.provider.handoff = self.handoff
                self.settings.ocr = self.ocr_settings
                self.apply_route(self.ocr, self.ocr_settings)
                self._applied = request.generation
                self._blocker = None
                self.wake_receipts()
        except ActivationBusy:
            return
        except (ActivationStateError, HandoffWait, OSError, ValueError):
            self._blocker = "ocr_activation_state_invalid"

    async def run(self) -> None:
        setup_on_startup = os.environ.get("OCR_ACTIVATION_ON_STARTUP") == "true"
        confirm_handoff = os.environ.get("OCR_ACTIVATION_CONFIRM_HANDOFF") == "true"
        while self._applied is None and not self.external_mode:
            if setup_on_startup:
                try:
                    if self.store.read_request() is None:
                        await self.submit(confirm_handoff=confirm_handoff)
                    else:
                        setup_on_startup = False
                except ActivationStateError:
                    self._blocker = "ocr_activation_state_invalid"
            await self.reconcile()
            await asyncio.sleep(1)
