"""Single native process with bounded admission and request-owned deadlines."""

import math
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from vision_bill.model.ocr import OcrResponse, WorkerReadiness

from .profile import RuntimeSettings, load_profile


@dataclass(frozen=True)
class OcrRequest:
    request_id: UUID
    original_path: Path
    photo_index: int
    source_sha256: str


class OcrWorkerError(Exception):
    def __init__(self, code: str, status_code: int, *, cleanup_safe: bool = True):
        super().__init__(code)
        self.code = code
        self.status_code = status_code
        self.cleanup_safe = cleanup_safe


def _worker_main(
    input_channel: Any,
    output_channel: Any,
    generation: int,
    profile_path: Path,
    runtime: RuntimeSettings | None,
    smoke_path: Path | None,
) -> None:
    # No exception text or receipt content crosses the general error/log boundary.
    try:
        from .pipeline import build_pipeline

        if runtime is None or smoke_path is None:
            raise ValueError("Explicit runtime and sealed smoke image are required")
        adapter = build_pipeline(load_profile(profile_path), runtime)
        adapter.qualify(smoke_path)
        output_channel.put(
            {"kind": "ready", "generation": generation, "profile_id": adapter.profile_id}
        )
    except Exception:  # noqa: BLE001 - native boundary must hide exception details
        output_channel.put({"kind": "init_failed", "generation": generation})
        return
    while True:
        request = input_channel.get()
        if request is None:
            return
        message: dict[str, Any] = {
            "generation": generation,
            "request_id": request.request_id,
        }
        try:
            result = adapter.predict(
                request.original_path,
                request_id=request.request_id,
                photo_index=request.photo_index,
                source_sha256=request.source_sha256,
            )
            message.update(kind="result", result=result)
        except Exception:  # noqa: BLE001 - native boundary must hide exception details
            message.update(kind="error", code="ocr_inference_failed")
        output_channel.put(message)


class OcrWorkerManager:
    def __init__(
        self,
        *,
        runtime: RuntimeSettings | None = None,
        smoke_path: Path | None = None,
        profile_path: Path = Path(__file__).parents[1] / "profile.json",
        startup_timeout: float = 600,
        mp_context: Any = None,
    ):
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("Startup timeout must be finite and positive")
        self._ctx = mp_context if mp_context is not None else mp.get_context("spawn")
        self._runtime = runtime
        self._smoke_path = smoke_path
        self._profile_path = profile_path
        self._startup_timeout = startup_timeout
        self._control = threading.RLock()
        self._state = threading.Lock()
        self._active = threading.Lock()
        self._stopped = threading.Event()
        self._generation = 0
        self._reason: str | None = "ocr_starting"
        self._profile_id: str | None = None
        self._ready = False
        self._admitted = 0
        self._proc: Any = None
        self._in_q: Any = None
        self._out_q: Any = None

    @property
    def pending_count(self) -> int:
        with self._state:
            return max(0, self._admitted - 1)

    def readiness(self) -> WorkerReadiness:
        with self._state:
            alive = self._ready and self._proc is not None and self._proc.is_alive()
            ready = self._ready and alive and not self._stopped.is_set()
            reason = self._reason if not ready else None
            if self._stopped.is_set():
                reason = "ocr_stopped"
            elif self._ready and not alive:
                reason = "ocr_worker_crashed"
            return WorkerReadiness(
                ready=ready,
                generation=max(1, self._generation),
                profile_id=self._profile_id,
                reason_code=reason,
            )

    def _set_unready(self, reason: str) -> None:
        with self._state:
            self._ready, self._reason, self._profile_id = False, reason, None

    def start(self) -> None:
        """Bounded initialization, called off the HTTP event loop."""
        with self._control:
            if self._stopped.is_set() or self.readiness().ready:
                return
            self._set_unready("ocr_starting")
            if self._proc is not None:
                self._teardown()
            self._generation += 1
            try:
                self._in_q = self._ctx.Queue(maxsize=1)
                self._out_q = self._ctx.Queue(maxsize=2)
                self._proc = self._ctx.Process(
                    target=_worker_main,
                    args=(
                        self._in_q,
                        self._out_q,
                        self._generation,
                        self._profile_path,
                        self._runtime,
                        self._smoke_path,
                    ),
                    daemon=True,
                )
                self._proc.start()
                deadline = time.monotonic() + self._startup_timeout
                while not self._stopped.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._set_unready("ocr_startup_timeout")
                        break
                    try:
                        message = self._out_q.get(timeout=min(0.05, remaining))
                    except queue.Empty:
                        if not self._proc.is_alive():
                            self._set_unready("ocr_init_failed")
                            break
                        continue
                    if message.get("generation") != self._generation:
                        continue
                    if message.get("kind") == "ready":
                        with self._state:
                            self._ready = not self._stopped.is_set()
                            self._reason = None if self._ready else "ocr_stopped"
                            self._profile_id = message["profile_id"]
                        if self._ready:
                            return
                    self._set_unready("ocr_init_failed")
                    break
            except Exception:  # noqa: BLE001 - native boundary must hide exception details
                self._set_unready("ocr_init_failed")
            self._teardown()

    def _teardown(self) -> None:
        """Never release a process's input ownership until death is confirmed."""
        proc = self._proc
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
                proc.join(timeout=0.1)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1)
                if proc.is_alive():
                    raise OSError("Worker death could not be confirmed")
            except (OSError, ValueError):
                self._set_unready("ocr_worker_unstoppable")
                raise OcrWorkerError("ocr_worker_unstoppable", 503, cleanup_safe=False) from None
            proc.close()
        for channel in (self._in_q, self._out_q):
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()
        self._proc = self._in_q = self._out_q = None

    def _recover(self, reason: str) -> None:
        with self._control:
            self._set_unready(reason)
            self._teardown()
        # A new generation is qualified separately; never replay the failed request.
        threading.Thread(target=self.start, name="ocr-worker-recovery", daemon=True).start()

    def infer(self, request: OcrRequest, *, deadline_monotonic: float) -> OcrResponse:
        if time.monotonic() >= deadline_monotonic:
            raise OcrWorkerError("ocr_deadline_exceeded", 504)
        with self._state:
            if self._admitted >= 5:
                raise OcrWorkerError("ocr_queue_full", 429)
            self._admitted += 1
        acquired = False
        try:
            acquired = self._active.acquire(timeout=max(0, deadline_monotonic - time.monotonic()))
            if not acquired or time.monotonic() >= deadline_monotonic:
                # Waiting callers do not own the worker and cannot terminate its owner.
                raise OcrWorkerError("ocr_deadline_exceeded", 504)
            ready = self.readiness()
            if not ready.ready:
                raise OcrWorkerError(ready.reason_code or "ocr_not_ready", 503)
            generation, proc, in_q, out_q = (self._generation, self._proc, self._in_q, self._out_q)
            in_q.put(request)
            while True:
                if self._stopped.is_set():
                    with self._control:
                        self._teardown()
                    raise OcrWorkerError("ocr_stopped", 503)
                remaining = deadline_monotonic - time.monotonic()
                if remaining <= 0:
                    self._recover("ocr_deadline_exceeded")
                    raise OcrWorkerError("ocr_deadline_exceeded", 504)
                try:
                    message = out_q.get(timeout=min(0.05, remaining))
                except queue.Empty:
                    if self._stopped.is_set():
                        with self._control:
                            self._teardown()
                        raise OcrWorkerError("ocr_stopped", 503) from None
                    if not proc.is_alive():
                        self._recover("ocr_worker_crashed")
                        raise OcrWorkerError("ocr_worker_crashed", 503) from None
                    continue
                except (OSError, ValueError):
                    if self._stopped.is_set():
                        with self._control:
                            self._teardown()
                        raise OcrWorkerError("ocr_stopped", 503) from None
                    self._recover("ocr_worker_crashed")
                    raise OcrWorkerError("ocr_worker_crashed", 503) from None
                if (
                    message.get("generation") != generation
                    or message.get("request_id") != request.request_id
                ):
                    continue
                if message.get("kind") != "result":
                    raise OcrWorkerError("ocr_inference_failed", 502)
                try:
                    result = OcrResponse.model_validate(message["result"])
                    if (
                        result.request_id != request.request_id
                        or result.photo_index != request.photo_index
                        or result.source_sha256 != request.source_sha256
                    ):
                        raise ValueError("Response identity mismatch")
                except (ValueError, TypeError, KeyError):
                    raise OcrWorkerError("ocr_invalid_result", 502) from None
                return result
        finally:
            if acquired:
                self._active.release()
            with self._state:
                self._admitted -= 1

    def stop(self) -> None:
        self._stopped.set()
        with self._control:
            self._set_unready("ocr_stopped")
            self._teardown()
