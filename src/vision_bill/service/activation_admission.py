"""Keep inference and opt-in native OCR qualification mutually exclusive."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import get_ident
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ocr_contract.activation import (
    ActivationBusy,
    ActivationPhase,
    ActivationStateError,
    ActivationStore,
)

from ..model.handoff import HandoffStatus, HandoffWait
from .gpu_handoff import ModelHandoff, current_owner

if TYPE_CHECKING:
    from ..provider.llm.base import LLMProvider

_MARKERS = "direct-calls"
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


@dataclass(frozen=True)
class _Admission:
    store: ActivationStore
    requires_handoff: bool
    task: asyncio.Task[Any] | None
    thread: int


_current: ContextVar[_Admission | None] = ContextVar("ocr_activation_admission", default=None)


def activation_enabled() -> bool:
    return os.environ.get("VISION_BILL_OCR_ACTIVATION") == "true"


def activation_handoff_path() -> Path:
    return Path(os.environ.get("OCR_ACTIVATION_HANDOFF_LOCK_PATH", "/coord/gpu-handoff.lock"))


def _task() -> asyncio.Task[Any] | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def _current_admission() -> _Admission | None:
    admitted = _current.get()
    if admitted is not None and admitted.task is _task() and admitted.thread == get_ident():
        return admitted
    return None


def _require_provider(provider: LLMProvider | None) -> None:
    if provider is None:
        return
    owner = current_owner() if _task() is not None else None
    handoff = owner.coordinator if owner is not None else provider.handoff
    endpoint = os.environ.get("OCR_ACTIVATION_ENDPOINT")
    if (
        isinstance(handoff, ModelHandoff)
        and provider.supports_model_release
        and getattr(handoff.ocr, "managed_lifecycle", False)
        and handoff.path.resolve() == activation_handoff_path().resolve()
        and (endpoint is None or getattr(handoff.ocr, "service_url", None) == endpoint.rstrip("/"))
        and handoff.resources.llm == provider.lifecycle_identity
        and (
            owner is None or provider.handoff is None
            or isinstance(provider.handoff, ModelHandoff) and handoff.matches(provider.handoff)
        )
    ):
        return
    raise HandoffWait(HandoffStatus(state="configuration_required", reason_code="execution_policy_changed"))


@contextmanager
def inference_admission(
    provider: LLMProvider | None = None, *, store: ActivationStore | None = None,
) -> Iterator[None]:
    """Hold a shared lock for an attempt, reusing it for that task's nested calls."""
    if not activation_enabled():
        yield
        return
    admitted = _current_admission()
    if admitted is not None and (store is None or store == admitted.store):
        if admitted.requires_handoff:
            _require_provider(provider)
        yield
        return

    selected = store if store is not None else ActivationStore()
    with ExitStack() as lock:
        try:
            lock.enter_context(selected.admission(exclusive=False, blocking=False))
            request = selected.read_request()
            status = selected.read_status(request) if request is not None else None
        except ActivationBusy:
            raise HandoffWait(HandoffStatus(state="busy", reason_code="dedicated_ocr_activation_pending")) from None
        except (ActivationStateError, OSError):
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="dedicated_ocr_activation_state_invalid",
            )) from None
        if request is not None and (status is None or status.phase != ActivationPhase.READY):
            raise HandoffWait(HandoffStatus(state="busy", reason_code="dedicated_ocr_activation_pending"))

        if request is not None:
            _require_provider(provider)
        token = _current.set(_Admission(selected, request is not None, _task(), get_ident()))
        try:
            yield
        finally:
            _current.reset(token)


@contextmanager
def _marker_directory(store: ActivationStore, *, create: bool) -> Iterator[int]:
    if create:
        store.request_dir.mkdir(mode=0o700, exist_ok=True)
    directory = os.open(store.request_dir, _DIRECTORY_FLAGS)
    try:
        if create:
            try:
                os.mkdir(_MARKERS, 0o700, dir_fd=directory)
            except FileExistsError:
                pass
            os.fsync(directory)
            # The request directory may have been created by another first call.
            parent = os.open(store.request_dir.parent, _DIRECTORY_FLAGS)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        markers = os.open(_MARKERS, _DIRECTORY_FLAGS, dir_fd=directory)
        try:
            info = os.fstat(markers)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ActivationStateError("Unsafe direct inference marker directory")
            yield markers
        finally:
            os.close(markers)
    finally:
        os.close(directory)


def unsettled_direct_calls(store: ActivationStore) -> tuple[Path, ...]:
    """Read markers while holding exclusive admission, including after a client crash."""
    try:
        with _marker_directory(store, create=False) as directory:
            return tuple(store.request_dir / _MARKERS / name for name in sorted(os.listdir(directory)))
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise ActivationStateError("Cannot read direct inference markers") from exc


@contextmanager
def direct_inference_call(*, coordinated: bool = False) -> Iterator[None]:
    """Record dispatch before uncoordinated I/O and retain uncertain remote calls."""
    if not activation_enabled():
        yield
        return
    admitted = _current_admission()
    if admitted is None or (admitted.requires_handoff and not coordinated):
        raise HandoffWait(HandoffStatus(state="configuration_required", reason_code="execution_policy_changed"))
    if coordinated:
        # The ModelHandoff journal records dispatch and settlement under its lock.
        yield
        return

    with ExitStack() as marker:
        try:
            directory = marker.enter_context(_marker_directory(admitted.store, create=True))
            name = uuid4().hex
            descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(directory)
        except (ActivationStateError, OSError):
            raise HandoffWait(HandoffStatus(
                state="configuration_required", reason_code="dedicated_ocr_activation_state_invalid",
            )) from None
        yield
        os.unlink(name, dir_fd=directory)
        os.fsync(directory)
