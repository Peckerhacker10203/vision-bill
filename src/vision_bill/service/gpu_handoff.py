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
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_current: ContextVar["HandoffOwner | None"] = ContextVar("model_handoff", default=None)

ResourceIdentity = Annotated[str, Field(min_length=1)]


class HandoffResources(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    llm: ResourceIdentity
    ocr: ResourceIdentity


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
        self.deadline = deadline
        self.token = uuid4().hex
        self.task = asyncio.current_task()
        self.models: list[str] = []
        self.generation = 1
        self.unsettled = False
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
                resources=self.coordinator.resources, models=self.models,
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
            await self.coordinator.provider.release_model(model)
            self.models.remove(model)
            self.save()

    async def ocr_phase(self) -> None:
        if self.ocr_phase_closed:
            raise RuntimeError("model_handoff_ocr_phase_closed: start a fresh session for further OCR")
        await self.release_models()
        if not self.ocr_loaded:
            self.ocr_pending = True
            self.save()  # Durable before load HTTP can create native residency.
            await self.coordinator.ocr.load(self.token, self.deadline)
            self.ocr_loaded = True

    async def llm_phase(self, model: str) -> None:
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
        if model not in self.models:
            self.models.append(model)
            self.save()  # Durable before the request can reach Ollama.

    async def model_call(self, call: Coroutine[Any, Any, Any]) -> Any:
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
            self.save()
            raise
        self.unsettled = False
        self.save()
        return result


class ModelHandoff:
    def __init__(self, path: Path, provider: Any, ocr: Any):
        self.path = Path(path)
        self.journal = self.path.with_suffix(".json")
        self.provider = provider
        self.ocr = ocr

    @property
    def resources(self) -> HandoffResources:
        return HandoffResources(
            llm=handoff_resource_identity(self.provider, "llm"),
            ocr=handoff_resource_identity(self.ocr, "ocr"),
        )

    @asynccontextmanager
    async def session(self, deadline: float) -> AsyncIterator[HandoffOwner]:
        async with asyncio.timeout(max(0, deadline - monotonic())):
            async with self._session(deadline) as owner:
                yield owner
            if monotonic() >= deadline:
                raise TimeoutError("model_handoff_deadline")

    @asynccontextmanager
    async def _session(self, deadline: float) -> AsyncIterator[HandoffOwner]:
        if getattr(self.ocr, "managed_lifecycle", False) and not managed_ocr_enabled():
            raise ValueError(
                "managed_ocr_requires_production_ocr: enable OCR__ENABLED and "
                "OCR__MANAGE_MODEL_LIFECYCLE for every application worker"
            )
        resources = self.resources  # Validate explicit bindings before any resource calls.
        existing = current_owner()
        if existing is not None:
            if (existing.coordinator.path != self.path
                    or existing.coordinator.resources != resources):
                raise ValueError("model_handoff_identity_changed")
            yield existing
            return
        if not self.provider.supports_model_release:
            raise ValueError("model_release_unsupported: managed OCR requires an unload-capable provider")
        import fcntl  # Linux app/standalone contract; no GPU device-wide locks.

        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        acquired = False
        token = None
        owner = HandoffOwner(self, deadline)
        try:
            while not acquired:
                if monotonic() >= deadline:
                    raise TimeoutError("model_handoff_deadline")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    await asyncio.sleep(min(.05, max(0, deadline - monotonic())))
            if self.journal.exists():
                try:
                    state = json.loads(self.journal.read_text())
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
                    if getattr(saved.resources, name) != getattr(resources, name):
                        raise ValueError(
                            f"model_handoff_{name}_resource_changed: recover old resources before switching"
                        )
                owner.models = saved.models
                owner.generation = saved.generation + 1
                owner.unsettled = saved.unsettled
                owner.ocr_pending = saved.ocr_pending
            owner.token = f"{owner.generation}:{owner.token}"
            owner.save()
            token = _current.set(owner)
            # Claim parks any abandoned native worker and fences old HTTP requests.
            await self.ocr.claim(owner.token, deadline)
            owner.ocr_pending = False
            owner.save()
            await owner.release_models()
            owner.save()
            try:
                yield owner
            finally:
                # Cancellation interrupts inference, not the ownership barrier. A
                # second cancellation can abort cleanup, but the durable journal
                # and next claim still prevent an unsafe successor.
                cleanup_deadline = min(deadline, monotonic() + 30)
                if monotonic() < cleanup_deadline:
                    owner.deadline = cleanup_deadline
                    async with asyncio.timeout(max(0, cleanup_deadline - monotonic())):
                        await self.ocr.park(owner.token, cleanup_deadline)
                        owner.ocr_loaded = False
                        owner.ocr_pending = False
                        owner.save()
                        await owner.release_models()
        finally:
            if token is not None:
                _current.reset(token)
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
