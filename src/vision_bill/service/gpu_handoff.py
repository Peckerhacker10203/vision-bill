"""Linux process-shared, internal OCR/LLM ownership (never device exclusivity).

The journal survives a cancelled/crashed client. A successor must park the OCR
worker and confirm release of every recorded model before it can load anything.
Only explicit Vision Bill model IDs are recorded; discovery is never ownership.
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from time import monotonic
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..model.handoff import (
    DispatchDisposition,
    HandoffBinding,
    HandoffResources,
    HandoffStatus,
    HandoffWait,
    ResourceIdentity,
)

_current: ContextVar["HandoffOwner | None"] = ContextVar("model_handoff", default=None)


def _handoff_error(status: HandoffStatus, *, admission: bool) -> RuntimeError:
    """Only a pre-work admission rejection can be treated as a wait."""
    return HandoffWait(status) if admission else RuntimeError(status.reason_code or status.state)


def _unique_json_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    members: dict[str, object] = {}
    for key, value in pairs:
        if key in members:
            raise ValueError("model_handoff_duplicate_member")
        members[key] = value
    return members


class HandoffJournal(BaseModel):
    """No legacy endpoint key: old coordinators must fail before dispatch."""
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[2]
    coordinator: Literal["vision-bill-model-handoff-v2"]
    resources: HandoffResources
    models: list[ResourceIdentity]
    generation: int = Field(ge=1)
    unsettled: bool
    ocr_pending: bool


def current_owner() -> "HandoffOwner | None":
    owner = _current.get()
    # ContextVars propagate to spawned tasks; that does not transfer ownership.
    return owner if owner is not None and owner.task is asyncio.current_task() else None


def configure_handoff(provider: Any) -> None:
    """Also protect supported legacy callers that construct providers directly."""
    from ..config import settings
    from ..provider.factory import get_ocr_provider
    ocr = get_ocr_provider(settings.ocr)
    if ocr is not None and getattr(ocr, "managed_lifecycle", False):
        provider.handoff = ModelHandoff(settings.ocr.coordination_path, provider, ocr)


def managed_ocr_enabled() -> bool:
    """Ordinary LLM providers must participate before any managed OCR can load."""
    from ..config import settings
    return settings.ocr.enabled and settings.ocr.manage_model_lifecycle


def handoff_resource_identity(resource: Any, name: str) -> str:
    """Validate a stable binding without contacting or acquiring the resource."""
    identity = getattr(resource, "lifecycle_identity", None)
    if not isinstance(identity, str) or not identity.strip() or identity == "unsupported":
        raise ValueError(f"model_handoff_{name}_identity_required: provide an explicit stable resource identity")
    return identity


class HandoffOwner:
    def __init__(self, coordinator: "ModelHandoff", deadline: float):
        self.coordinator = coordinator
        self.resources = coordinator.resources
        self.binding: HandoffBinding | None = None
        self.deadline = deadline
        self.token = uuid4().hex
        self.task = asyncio.current_task()
        self.models: list[str] = []
        self.generation = 1
        self.unsettled = False
        self.dispatch_disposition: DispatchDisposition = "not_dispatched"
        self.model_work_started = False
        self.ocr_loaded = False
        self.ocr_phase_closed = False
        self.ocr_pending = True  # Claim has not yet confirmed an abandoned worker parked.

    def save(self) -> None:
        path = self.coordinator.journal
        temporary = path.with_suffix(".pending")
        with temporary.open("w") as stream:
            os.chmod(temporary, 0o600)
            json.dump(HandoffJournal(
                version=2, coordinator="vision-bill-model-handoff-v2",
                resources=self.resources, models=self.models,
                generation=self.generation, unsettled=self.unsettled,
                ocr_pending=self.ocr_pending,
            ).model_dump(), stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    async def release_models(self) -> None:
        if self.unsettled:
            raise RuntimeError("model_request_unsettled: reconcile the dedicated endpoint before recovery")
        for model in tuple(self.models):
            self.require_model(model)
            await self.coordinator.provider.release_model(model)
            self.models.remove(model)
            self.save()

    async def ocr_phase(self) -> None:
        if self.ocr_phase_closed:
            raise RuntimeError("model_handoff_ocr_phase_closed: start a fresh session for further OCR")
        await self.release_models()
        await self.coordinator.require_residency(
            self.models, self.deadline, admission=not self.model_work_started,
        )
        if not self.ocr_loaded:
            self.ocr_pending = True
            self.save()  # Durable before load HTTP can create native residency.
            self.model_work_started = True
            await self.coordinator.ocr.load(self.token, self.deadline)
            self.ocr_loaded = True

    async def llm_phase(self, model: str) -> None:
        self.require_model(model)
        if self.unsettled:
            raise RuntimeError("model_request_unsettled")
        self.ocr_phase_closed = True
        if self.ocr_pending:
            await self.coordinator.ocr.park(self.token, self.deadline)
            self.ocr_loaded = False
            self.ocr_pending = False
            self.save()  # Only a confirmed park clears possible OCR residency.
        self.track(model)

    def track(self, model: str) -> None:
        self.require_model(model)
        if model not in self.models:
            self.models.append(model)
            self.save()  # Durable before the request can reach Ollama.

    def require_model(self, model: str) -> None:
        if self.binding is None or model not in self.binding.model_ids:
            raise _handoff_error(HandoffStatus(
                state="configuration_required", reason_code="model_handoff_model_not_adopted",
            ), admission=not self.model_work_started)

    def require_release(self, model: str) -> None:
        self.require_model(model)
        if self.unsettled:
            raise RuntimeError("model_request_unsettled")
        if model not in self.models:
            raise ValueError("model_handoff_model_not_recorded")
        self.dispatch_disposition = "not_dispatched"

    async def model_call(self, call: Coroutine[Any, Any, Any]) -> Any:
        self.model_work_started = True
        self.dispatch_disposition = "unknown"
        self.unsettled = True
        self.save()
        task = asyncio.create_task(call)
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                async with asyncio.timeout(min(30, max(0, self.deadline - monotonic()))):
                    await asyncio.shield(task)
            except (Exception, asyncio.CancelledError):
                # Never let a detached call mutate a successor's journal.
                task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
                raise
            self.unsettled = False
            self.dispatch_disposition = "settled"
            self.save()
            raise
        self.unsettled = False
        self.dispatch_disposition = "settled"
        self.save()
        return result


class ModelHandoff:
    def __init__(self, path: Path, provider: Any, ocr: Any):
        self.path = Path(path)
        self.journal = self.path.with_suffix(".json")
        self.binding_path = self.path.with_suffix(".binding.json")
        self.provider = provider
        self.ocr = ocr

    @property
    def resources(self) -> HandoffResources:
        return HandoffResources(
            llm=handoff_resource_identity(self.provider, "llm"),
            ocr=handoff_resource_identity(self.ocr, "ocr"),
        )

    @asynccontextmanager
    async def exclusion(self, deadline: float, *, wait: bool = True) -> AsyncIterator[None]:
        """Use the original lock inode for inspection, adoption and sessions."""
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        path_identity = self.path.resolve()
        resources = self.resources
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        acquired = False
        try:
            while not acquired:
                if monotonic() >= deadline:
                    raise HandoffWait(HandoffStatus(state="busy", reason_code="model_handoff_busy"))
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    if not wait:
                        raise HandoffWait(HandoffStatus(state="busy", reason_code="model_handoff_busy")) from None
                    await asyncio.sleep(min(.05, max(0, deadline - monotonic())))
            current, opened = self.path.stat(), os.fstat(descriptor)
            if ((current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
                    or self.path.resolve() != path_identity or self.resources != resources):
                raise ValueError("model_handoff_identity_changed")
            yield
        finally:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def read_journal(self) -> HandoffJournal | None:
        if not self.journal.exists():
            return None
        try:
            state = json.loads(self.journal.read_text(), object_pairs_hook=_unique_json_members)
        except (ValueError, UnicodeError):
            raise ValueError("model_handoff_invalid_journal: preserve journal and recover managed resources") from None
        if isinstance(state, dict) and "version" not in state:
            raise ValueError(
                "model_handoff_legacy_journal: stop/drain all workers; recover and confirm "
                "old OCR parked and recorded LLM models released at their old endpoints "
                "before manually archiving the journal"
            )
        try:
            saved = HandoffJournal.model_validate(state)
        except ValidationError:
            raise ValueError("model_handoff_invalid_journal: preserve journal and recover managed resources") from None
        for name in ("llm", "ocr"):
            if getattr(saved.resources, name) != getattr(self.resources, name):
                raise ValueError(f"model_handoff_{name}_resource_changed: recover old resources before switching")
        return saved

    def read_binding(self) -> HandoffBinding:
        try:
            data = json.loads(self.binding_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_members)
            binding = HandoffBinding.model_validate(data)
        except FileNotFoundError:
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="model_handoff_binding_required",
            )) from None
        except (ValueError, UnicodeError):
            raise HandoffWait(HandoffStatus(
                state="recovery_required", reason_code="model_handoff_invalid_binding",
            )) from None
        if binding.coordination_path != str(self.path.resolve()) or binding.resources != self.resources:
            raise HandoffWait(HandoffStatus(
                state="recovery_required", reason_code="model_handoff_binding_changed",
            ))
        return binding

    async def require_residency(self, recorded: list[str], deadline: float, *, admission: bool = True) -> None:
        try:
            async with asyncio.timeout(max(0, deadline - monotonic())):
                resident = await self.provider.resident_model_ids(deadline=deadline)
            if not isinstance(resident, frozenset) or any(
                not isinstance(model, str) or not model.strip() or model != model.strip() for model in resident
            ):
                raise ValueError("invalid_residency")
        except Exception:  # noqa: BLE001 - unknown provider evidence must remain fenced, without URL/credential output.
            raise _handoff_error(HandoffStatus(
                state="recovery_required", reason_code="model_handoff_residency_unknown",
            ), admission=admission) from None
        allowed = set(recorded)
        residency_ids_for = getattr(self.provider, "residency_ids_for", None)
        if residency_ids_for is not None:
            allowed.update(resident_id for model in recorded for resident_id in residency_ids_for(model))
        if resident.difference(allowed):
            raise _handoff_error(HandoffStatus(
                state="recovery_required", reason_code="model_handoff_foreign_residency",
            ), admission=admission)

    async def inspect(self, *, model_id: str, deadline: float) -> HandoffStatus:
        """Read-only admission evidence; never claim, release, or advance a journal."""
        if not self.provider.supports_model_release:
            return HandoffStatus(state="configuration_required", reason_code="model_release_unsupported")
        try:
            async with self.exclusion(deadline, wait=False):
                saved = self.read_journal()
                binding = self.read_binding()
                if model_id not in binding.model_ids:
                    return HandoffStatus(state="configuration_required", reason_code="model_handoff_model_not_adopted")
                if saved is not None and saved.unsettled:
                    return HandoffStatus(state="recovery_required", reason_code="model_request_unsettled")
                recorded = saved.models if saved else []
                if any(model not in binding.model_ids for model in recorded):
                    return HandoffStatus(state="recovery_required", reason_code="model_handoff_recorded_model_not_adopted")
                await self.require_residency(recorded, deadline)
                return HandoffStatus(state="ready")
        except HandoffWait as exc:
            return exc.status
        except (ValueError, OSError):
            return HandoffStatus(state="recovery_required", reason_code="model_handoff_invalid_configuration")

    @asynccontextmanager
    async def session(self, deadline: float, *, model_id: str | None = None) -> AsyncIterator[HandoffOwner]:
        # Exclusion owns its retry deadline. Activate cancellation only once we own the lock.
        async with asyncio.timeout(None) as timeout:
            async with self._session(deadline, model_id=model_id, timeout=timeout) as owner:
                yield owner
            if monotonic() >= deadline:
                raise TimeoutError("model_handoff_deadline")

    @asynccontextmanager
    async def _session(
        self, deadline: float, *, model_id: str | None, timeout: asyncio.Timeout,
    ) -> AsyncIterator[HandoffOwner]:
        expires_at = asyncio.get_running_loop().time() + (deadline - monotonic())
        if getattr(self.ocr, "managed_lifecycle", False) and not managed_ocr_enabled():
            raise ValueError(
                "managed_ocr_requires_production_ocr: enable OCR__ENABLED and "
                "OCR__MANAGE_MODEL_LIFECYCLE for every application worker"
            )
        resources = self.resources  # Validate explicit bindings before any resource calls.
        existing = current_owner()
        if existing is not None:
            timeout.reschedule(expires_at)
            if (existing.coordinator.path != self.path
                    or existing.coordinator.resources != resources):
                raise ValueError("model_handoff_identity_changed")
            if model_id is not None:
                existing.require_model(model_id)
            yield existing
            return
        if not self.provider.supports_model_release:
            raise ValueError("model_release_unsupported: managed OCR requires an unload-capable provider")
        token = None
        owner = HandoffOwner(self, deadline)
        async with self.exclusion(deadline):
            timeout.reschedule(expires_at)
            if self.resources != resources:
                raise ValueError("model_handoff_identity_changed")
            saved = self.read_journal()
            owner.binding = self.read_binding()
            if model_id is not None:
                owner.require_model(model_id)
            if saved is not None:
                owner.models = saved.models
                owner.generation = saved.generation + 1
                owner.unsettled = saved.unsettled
                owner.ocr_pending = saved.ocr_pending
            if owner.unsettled:
                raise HandoffWait(HandoffStatus(state="recovery_required", reason_code="model_request_unsettled"))
            for model in owner.models:
                owner.require_model(model)
            await self.require_residency(owner.models, deadline)
            owner.token = f"{owner.generation}:{owner.token}"
            owner.save()
            token = _current.set(owner)
            try:
                # Claim parks abandoned native workers and fences old HTTP requests.
                await self.ocr.claim(owner.token, deadline)
                owner.ocr_pending = False
                owner.save()
                await owner.release_models()
                owner.save()
                try:
                    yield owner
                except BaseException as body_error:
                    # Preserve ordinary body failures. A wait must account for cleanup dispatch.
                    cause = body_error
                    try:
                        await self._cleanup(owner, deadline)
                    except BaseException as cleanup_error:
                        if isinstance(body_error, HandoffWait) and isinstance(cleanup_error, asyncio.CancelledError):
                            raise
                        cause = cleanup_error
                    if isinstance(body_error, HandoffWait) and owner.model_work_started:
                        raise _handoff_error(body_error.status, admission=False) from cause
                    raise
                else:
                    await self._cleanup(owner, deadline)
            finally:
                if token is not None:
                    _current.reset(token)

    async def _cleanup(self, owner: HandoffOwner, deadline: float) -> None:
        # Cancellation interrupts inference, not the ownership barrier.
        cleanup_deadline = min(deadline, monotonic() + 30)
        if monotonic() < cleanup_deadline:
            owner.deadline = cleanup_deadline
            async with asyncio.timeout(max(0, cleanup_deadline - monotonic())):
                await self.ocr.park(owner.token, cleanup_deadline)
                owner.ocr_loaded = False
                owner.ocr_pending = False
                owner.save()
                await owner.release_models()
