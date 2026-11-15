"""Deterministic process-boundary simulations; these do not qualify GPU execution."""

import base64
import hashlib
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image


def response(request):
    from vision_bill.model.ocr import OcrResponse, WorkingImage

    output = BytesIO()
    Image.new("RGB", (2, 2)).save(output, "PNG")
    raw = output.getvalue()
    return OcrResponse(
        request_id=request.request_id,
        photo_index=request.photo_index,
        source_sha256=request.source_sha256,
        text="SHOP",
        blocks=[],
        working_image=WorkingImage(
            view_id=uuid4(),
            width=2,
            height=2,
            sha256=hashlib.sha256(raw).hexdigest(),
            png_base64=base64.b64encode(raw).decode(),
        ),
        profile_id="profile",
        runtime_fingerprint="runtime",
    )


class Channel(queue.Queue):
    def __init__(self):
        super().__init__()
        self.closed = False

    def close(self):
        self.closed = True

    def cancel_join_thread(self):
        pass


class FakeProcess:
    def __init__(self, context, args):
        self.context, self.args = context, args
        self.dead = threading.Event()
        self.active_request = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def run(self):
        in_q, out_q, generation, *_ = self.args
        self.context.started.set()
        while not self.context.init_release.wait(0.005):
            if self.dead.is_set():
                return
        if self.context.init_fail:
            out_q.put({"generation": generation, "kind": "init_failed"})
            self.dead.set()
            return
        out_q.put({"generation": generation, "kind": "ready", "profile_id": "profile"})
        while not self.dead.is_set():
            try:
                request = in_q.get(timeout=0.005)
            except queue.Empty:
                continue
            self.active_request = request
            self.context.requests.append(request)
            self.context.entered.set()
            if self.context.mode == "crash":
                self.dead.set()
                return
            while self.context.mode == "hang" and not self.context.release.wait(0.005):
                if self.dead.is_set():
                    return
            if self.context.stale:
                stale_result = response(request).model_copy(update={"text": "STALE"})
                out_q.put(
                    {
                        "generation": generation - 1,
                        "request_id": request.request_id,
                        "kind": "result",
                        "result": stale_result,
                    }
                )
                out_q.put(
                    {
                        "generation": generation,
                        "request_id": uuid4(),
                        "kind": "result",
                        "result": stale_result,
                    }
                )
            self.active_request = None
            out_q.put(
                {
                    "generation": generation,
                    "request_id": request.request_id,
                    "kind": "result",
                    "result": response(request),
                }
            )

    def is_alive(self):
        return self.thread.is_alive() and not self.dead.is_set()

    def terminate(self):
        self.context.events.append("terminate")
        if self.active_request is not None:
            assert self.active_request.original_path.exists(), (
                "input removed before confirmed death"
            )
        if not self.context.ignore_terminate:
            self.dead.set()

    def kill(self):
        self.context.events.append("kill")
        self.dead.set()

    def join(self, timeout=None):
        self.context.events.append("join")
        self.thread.join(timeout)

    def close(self):
        pass


class FakeContext:
    def __init__(self, mode="ok", *, init_fail=False, init_blocked=False):
        self.mode, self.init_fail = mode, init_fail
        self.started, self.entered, self.release = (threading.Event() for _ in range(3))
        self.init_release = threading.Event()
        if not init_blocked:
            self.init_release.set()
        self.stale = False
        self.ignore_terminate = False
        self.channels, self.processes, self.events, self.requests = [], [], [], []

    def Queue(self, maxsize=0):
        channel = Channel()
        self.channels.append(channel)
        return channel

    def Process(self, *, target, args, daemon):
        process = FakeProcess(self, args)
        self.processes.append(process)
        return process


def manager(context, **kwargs):
    from ocr_service.worker import OcrWorkerManager

    result = OcrWorkerManager(mp_context=context, startup_timeout=0.2, **kwargs)
    result.start()
    return result


def request(tmp_path):
    from ocr_service.worker import OcrRequest

    path = tmp_path / (str(uuid4()) + ".png")
    path.write_bytes(b"original")
    return OcrRequest(
        request_id=uuid4(),
        original_path=path,
        photo_index=0,
        source_sha256=hashlib.sha256(b"original").hexdigest(),
    )


def wait_ready(worker):
    end = time.monotonic() + 1
    while not worker.readiness().ready:
        assert time.monotonic() < end
        threading.Event().wait(0.005)


def test_feature_requires_bounded_worker():
    import importlib.util

    assert importlib.util.find_spec("ocr_service.worker") is not None, "bounded manager missing"


def test_timeout_kills_joins_and_replaces_channels_before_next_request(tmp_path):
    from ocr_service.worker import OcrWorkerError

    context = FakeContext("hang")
    context.ignore_terminate = True
    worker = manager(context)
    try:
        for _ in range(2):
            wait_ready(worker)
            old = context.channels[-2:]
            with pytest.raises(OcrWorkerError, match="ocr_deadline_exceeded"):
                worker.infer(request(tmp_path), deadline_monotonic=time.monotonic() + 0.03)
            assert context.events[-3:] == ["join", "kill", "join"]
            assert all(channel.closed for channel in old)
        context.mode = "ok"
        wait_ready(worker)
        item = request(tmp_path)
        assert worker.infer(item, deadline_monotonic=time.monotonic() + 1).text == "SHOP"
    finally:
        worker.stop()


def test_pending_deadline_does_not_kill_active_owner(tmp_path):
    from ocr_service.worker import OcrWorkerError

    context = FakeContext("hang")
    worker = manager(context)
    with ThreadPoolExecutor(2) as pool:
        active = pool.submit(
            worker.infer, request(tmp_path), deadline_monotonic=time.monotonic() + 1
        )
        assert context.entered.wait(1)
        assert worker.readiness().ready
        with pytest.raises(OcrWorkerError, match="ocr_deadline_exceeded"):
            worker.infer(request(tmp_path), deadline_monotonic=time.monotonic() + 0.03)
        assert "terminate" not in context.events
        context.release.set()
        assert active.result(1).text == "SHOP"
    worker.stop()


def test_one_active_four_pending_then_capacity_rejects(tmp_path):
    from ocr_service.worker import OcrWorkerError

    context = FakeContext("hang")
    worker = manager(context)
    with ThreadPoolExecutor(5) as pool:
        jobs = [
            pool.submit(worker.infer, request(tmp_path), deadline_monotonic=time.monotonic() + 2)
            for _ in range(5)
        ]
        assert context.entered.wait(1)
        # Event-controlled jobs plus observable admission count, no timing assumption.
        end = time.monotonic() + 1
        while worker.pending_count != 4:
            assert time.monotonic() < end
            threading.Event().wait(0.005)
        with pytest.raises(OcrWorkerError, match="ocr_queue_full"):
            worker.infer(request(tmp_path), deadline_monotonic=time.monotonic() + 1)
        context.release.set()
        assert all(job.result(1).text == "SHOP" for job in jobs)
        assert len(context.processes) == 1
    worker.stop()


def test_stale_generation_and_uuid_cannot_satisfy_request(tmp_path):
    context = FakeContext()
    context.stale = True
    worker = manager(context)
    item = request(tmp_path)
    try:
        assert worker.infer(item, deadline_monotonic=time.monotonic() + 1).text == "SHOP"
    finally:
        worker.stop()


def test_crash_fails_once_without_replaying_request(tmp_path):
    from ocr_service.worker import OcrWorkerError

    context = FakeContext("crash")
    worker = manager(context)
    try:
        with pytest.raises(OcrWorkerError, match="ocr_worker_crashed"):
            worker.infer(request(tmp_path), deadline_monotonic=time.monotonic() + 1)
        assert len(context.requests) == 1
    finally:
        worker.stop()


@pytest.mark.parametrize("failure", ["init", "timeout"])
def test_startup_failure_never_reports_ready(failure):
    context = FakeContext(init_fail=failure == "init", init_blocked=failure == "timeout")
    worker = manager(context)
    assert not worker.readiness().ready
    assert worker.readiness().reason_code in {"ocr_init_failed", "ocr_startup_timeout"}
    worker.stop()


def test_stop_during_blocked_init_prevents_orphan():
    from ocr_service.worker import OcrWorkerManager

    context = FakeContext(init_blocked=True)
    worker = OcrWorkerManager(mp_context=context, startup_timeout=1)
    with ThreadPoolExecutor(2) as pool:
        start = pool.submit(worker.start)
        assert context.started.wait(1)
        stop = pool.submit(worker.stop)
        stop.result(1)
        context.init_release.set()
        start.result(1)
    assert not worker.readiness().ready
    assert all(not process.is_alive() for process in context.processes)
    assert all(channel.closed for channel in context.channels)


def test_shutdown_waits_for_active_input_ownership_before_return(tmp_path):
    from ocr_service.worker import OcrWorkerError

    context = FakeContext("hang")
    worker = manager(context)
    with ThreadPoolExecutor(1) as pool:
        job = pool.submit(worker.infer, request(tmp_path), deadline_monotonic=time.monotonic() + 1)
        assert context.entered.wait(1)
        worker.stop()
        with pytest.raises(OcrWorkerError, match="ocr_stopped"):
            job.result(1)
        assert all(not process.is_alive() for process in context.processes)


def test_native_entrypoint_qualifies_before_predict(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ocr_service import pipeline
    from ocr_service.worker import _worker_main

    calls = []
    adapter = SimpleNamespace(
        profile_id="qualified",
        qualify=lambda image: calls.append("qualify"),
        predict=lambda *args, **kwargs: calls.append("predict"),
    )
    monkeypatch.setattr(pipeline, "build_pipeline", lambda profile, runtime: adapter)
    incoming, outgoing = queue.Queue(), queue.Queue()
    incoming.put(request(tmp_path))
    incoming.put(None)
    _worker_main(
        incoming,
        outgoing,
        1,
        Path(__file__).parents[1] / "paddleocr-service/profile.json",
        object(),
        tmp_path,
    )
    assert calls == ["qualify", "predict"]
    assert outgoing.get()["kind"] == "ready"


@pytest.mark.parametrize("seconds", [float("inf"), float("nan"), 0, -1])
def test_startup_budget_must_be_finite_positive(seconds):
    from ocr_service.worker import OcrWorkerManager

    with pytest.raises(ValueError, match="finite and positive"):
        OcrWorkerManager(mp_context=FakeContext(), startup_timeout=seconds)


def test_failed_termination_reports_safe_error_and_preserves_input(tmp_path, monkeypatch):
    from ocr_service.worker import OcrWorkerError

    context = FakeContext("hang")
    worker = manager(context)
    item = request(tmp_path)
    process = context.processes[0]
    original_terminate = process.terminate

    def fail_terminate():
        raise OSError("private native details")

    monkeypatch.setattr(process, "terminate", fail_terminate)
    try:
        with pytest.raises(OcrWorkerError) as error:
            worker.infer(item, deadline_monotonic=time.monotonic() + 0.025)
        assert error.value.code == "ocr_worker_unstoppable"
        assert not error.value.cleanup_safe
        assert item.original_path.exists()
        assert not worker.readiness().ready
    finally:
        monkeypatch.setattr(process, "terminate", original_terminate)
        worker.stop()
