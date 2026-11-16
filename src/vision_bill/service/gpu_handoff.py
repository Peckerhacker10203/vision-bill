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
from typing import Any
from uuid import uuid4

_current: ContextVar["HandoffOwner | None"] = ContextVar("model_handoff", default=None)


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

    def save(self) -> None:
        path = self.coordinator.journal
        temporary = path.with_suffix(".pending")
        with temporary.open("w") as stream:
            os.chmod(temporary, 0o600)
            json.dump({"endpoint": self.coordinator.provider.lifecycle_identity,
                       "models": self.models, "generation": self.generation,
                       "unsettled": self.unsettled}, stream)
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
        await self.release_models()
        if not self.ocr_loaded:
            await self.coordinator.ocr.load(self.token, self.deadline)
            self.ocr_loaded = True

    async def llm_phase(self, model: str) -> None:
        if self.unsettled:
            raise RuntimeError("model_request_unsettled")
        if self.ocr_loaded:
            await self.coordinator.ocr.park(self.token, self.deadline)
            self.ocr_loaded = False
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
        existing = current_owner()
        if existing is not None:
            if (existing.coordinator.path != self.path
                    or existing.coordinator.provider.lifecycle_identity != self.provider.lifecycle_identity):
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
                state = json.loads(self.journal.read_text())
                if state["endpoint"] != self.provider.lifecycle_identity:
                    raise ValueError("model_handoff_endpoint_changed: recover old endpoint before switching")
                models = state["models"]
                if not isinstance(models, list) or any(not isinstance(m, str) or not m for m in models):
                    raise ValueError("model_handoff_invalid_journal")
                owner.models = models
                owner.generation = int(state["generation"]) + 1
                owner.unsettled = bool(state.get("unsettled", False))
            owner.token = f"{owner.generation}:{owner.token}"
            owner.save()
            token = _current.set(owner)
            # Claim parks any abandoned native worker and fences old HTTP requests.
            await self.ocr.claim(owner.token, deadline)
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
                        await owner.release_models()
        finally:
            if token is not None:
                _current.reset(token)
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
