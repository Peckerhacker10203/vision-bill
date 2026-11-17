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


def artifact_reply(directory, generation, request_id, result):
    destination = directory / (uuid4().hex + ".json")
    temporary = destination.with_suffix(".part")
    temporary.write_text(result.model_dump_json())
    temporary.replace(destination)
    return {
        "generation": generation,
        "request_id": str(request_id),
        "kind": "result",
        "artifact": destination.name,
    }


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
        out_q.put(
            {
                "generation": generation,
                "kind": "ready",
                "profile_id": "profile",
                "runtime_fingerprint": "runtime",
            }
        )
        while not self.dead.is_set():
            try:
                request = in_q.get(timeout=0.005)
            except queue.Empty:
                continue
            self.active_request = request
            self.context.requests.append(request)
            self.context.entered.set()
            if self.context.mode == "fatal":
                self.context.init_release.clear()
                out_q.put({"generation":generation, "request_id":str(request.request_id), "kind":"qualification_invalidated"})
                continue
            if self.context.mode == "crash":
                self.dead.set()
                return
            while self.context.mode == "hang" and not self.context.release.wait(0.005):
                if self.dead.is_set():
                    return
            if self.context.stale:
                stale_result = response(request).model_copy(update={"text": "STALE"})
                out_q.put(
                    artifact_reply(self.args[-1], generation - 1, request.request_id, stale_result)
                )
                out_q.put(artifact_reply(self.args[-1], generation, uuid4(), stale_result))
            self.active_request = None
            out_q.put(
                artifact_reply(self.args[-1], generation, request.request_id, response(request))
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
            old = [worker._in_q, worker._out_q, worker._reply_sender]
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
    from ocr_service.worker import _worker_main, read_result

    calls = []
    item = request(tmp_path)

    def predict(*args, **kwargs):
        calls.append("predict")
        return response(item)

    adapter = SimpleNamespace(
        profile_id="qualified",
        runtime_fingerprint="qualified-runtime",
        qualify=lambda image: calls.append("qualify"),
        predict=predict,
    )
    monkeypatch.setattr(pipeline, "build_pipeline", lambda profile, runtime: adapter)
    incoming, outgoing = queue.Queue(), queue.Queue()
    incoming.put(item)
    incoming.put(None)
    _worker_main(
        incoming,
        outgoing,
        1,
        Path(__file__).parents[1] / "paddleocr-service/profile.json",
        object(),
        tmp_path,
        tmp_path,
    )
    assert calls == ["qualify", "predict"]
    ready = outgoing.get()
    assert ready["kind"] == "ready"
    assert ready["runtime_fingerprint"] == "qualified-runtime"
    completion = outgoing.get()
    assert completion["kind"] == "result"
    assert read_result(tmp_path, completion["artifact"]).request_id == item.request_id


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


def _real_partial_reply_worker(args, failure_mode):
    """Actual OS process: partial publication/death, with old stream proof support."""
    import os
    import struct

    in_channel, out_channel, generation, *_ = args
    legacy_stream = hasattr(out_channel, "_writer")
    ready = {"kind": "ready", "generation": generation, "profile_id": "real-process"}
    out_channel.put(ready)
    item = in_channel.get()
    if generation == 1:
        if legacy_stream:
            # Exact reviewer counterexample; only the pre-fix transport takes this path.
            out_channel._sem.acquire()
            os.write(out_channel._writer.fileno(), struct.pack("!i", 100000) + b"partial")
        elif failure_mode == "partial_artifact":
            (args[-1] / "unfinished.part").write_bytes(b'{"working_image":')
        elif failure_mode == "oversized_notification":
            out_channel.socket.send(b"x" * 5000)
        elif failure_mode == "oversized_artifact":
            from ocr_contract.limits import MAX_OCR_RESPONSE_BYTES

            result_path = args[-1] / (uuid4().hex + ".json")
            with result_path.open("wb") as output:
                output.truncate(MAX_OCR_RESPONSE_BYTES + 1)
            out_channel.put(
                {
                    "kind": "result",
                    "generation": generation,
                    "request_id": str(item.request_id),
                    "artifact": result_path.name,
                }
            )
        else:
            out_channel.socket.send(b'{"kind":"result",')
        os._exit(7)
    result = response(item)
    if legacy_stream:
        out_channel.put(
            {
                "kind": "result",
                "generation": generation,
                "request_id": item.request_id,
                "result": result,
            }
        )
    else:
        result_path = args[-1] / (uuid4().hex + ".json")
        temporary = result_path.with_suffix(".part")
        temporary.write_text(result.model_dump_json())
        temporary.replace(result_path)
        out_channel.put(
            {
                "kind": "result",
                "generation": generation,
                "request_id": str(item.request_id),
                "artifact": result_path.name,
            }
        )
    # Remain alive until the manager retires this real process.
    in_channel.get()


class RealReplyContext:
    def __init__(self, failure_mode):
        import multiprocessing as mp

        self.context = mp.get_context("spawn")
        self.failure_mode = failure_mode
        self.processes = []

    def Queue(self, maxsize=0):
        return self.context.Queue(maxsize=maxsize)

    def Process(self, *, target, args, daemon):
        process = self.context.Process(
            target=_real_partial_reply_worker, args=(args, self.failure_mode), daemon=daemon
        )
        self.processes.append(process)
        return process


@pytest.mark.parametrize(
    "failure_mode",
    [
        "truncated_notification",
        "partial_artifact",
        "oversized_notification",
        "oversized_artifact",
    ],
)
def test_real_process_partial_reply_is_bounded_and_next_generation_works(tmp_path, failure_mode):
    from ocr_service.worker import OcrWorkerError, OcrWorkerManager

    context = RealReplyContext(failure_mode)
    worker = OcrWorkerManager(mp_context=context, startup_timeout=2)
    worker.start()
    assert worker.readiness().ready
    old_reply, old_process = worker._out_q, worker._proc
    old_directory, old_sender = Path(worker._reply_directory.name), worker._reply_sender
    outcome = []
    finished = threading.Event()

    def infer():
        try:
            worker.infer(request(tmp_path), deadline_monotonic=time.monotonic() + 0.1)
        except Exception as error:  # noqa: BLE001 - detect unclassified thread failures
            outcome.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=infer, daemon=True)
    thread.start()
    bounded = finished.wait(0.7)
    try:
        if not bounded and hasattr(old_reply, "_writer"):
            # Force EOF only to clean up the old implementation after observing RED.
            old_reply._writer.close()
        assert finished.wait(1), "supervisor remained stuck after cleanup"
        thread.join(0.2)
        assert bounded, "partial native reply defeated the 100ms deadline at 700ms"
        assert len(outcome) == 1 and isinstance(outcome[0], OcrWorkerError)
        assert str(outcome[0]) in {
            "ocr_worker_crashed",
            "ocr_deadline_exceeded",
            "ocr_invalid_result",
        }
        assert old_process._closed
        assert old_reply.closed
        assert old_sender.closed
        assert not old_directory.exists()
        wait_ready(worker)
        item = request(tmp_path)
        result = worker.infer(item, deadline_monotonic=time.monotonic() + 1)
        assert result.request_id == item.request_id and result.text == "SHOP"
    finally:
        worker.stop()
        assert all(process._closed for process in context.processes)


def test_stop_completed_after_readiness_cannot_dispatch(tmp_path, monkeypatch):
    from ocr_service.worker import OcrWorkerError

    context = FakeContext()
    worker = manager(context)
    observed, release = threading.Event(), threading.Event()
    original = worker.readiness

    def pause_readiness():
        result = original()
        if threading.current_thread().name == "inference-dispatch":
            observed.set()
            assert release.wait(1)
        return result

    monkeypatch.setattr(worker, "readiness", pause_readiness)
    outcome = []

    def infer():
        try:
            worker.infer(request(tmp_path), deadline_monotonic=time.monotonic() + 1)
        except Exception as error:  # noqa: BLE001 - detect unclassified thread failures
            outcome.append(error)

    thread = threading.Thread(target=infer, name="inference-dispatch", daemon=True)
    thread.start()
    try:
        assert observed.wait(1)
        worker.stop()  # Teardown finishes before the ready observation is released.
        release.set()
        thread.join(1)
        assert not thread.is_alive()
        assert len(outcome) == 1 and isinstance(outcome[0], OcrWorkerError)
        assert str(outcome[0]) == "ocr_stopped"
        assert not context.requests
    finally:
        release.set()
        thread.join(1)
        worker.stop()


def test_active_process_probe_holds_lifecycle_ownership_against_stop(tmp_path, monkeypatch):
    from ocr_service.worker import OcrWorkerError

    context = FakeContext("hang")
    worker = manager(context)
    process = context.processes[0]
    entered, release, stop_called, stopped = (threading.Event() for _ in range(4))
    original = process.is_alive

    def paused_probe():
        if (
            threading.current_thread().name == "inference-probe"
            and context.entered.is_set()
            and not entered.is_set()
        ):
            entered.set()
            assert release.wait(1)
        return original()

    monkeypatch.setattr(process, "is_alive", paused_probe)
    outcome = []

    def infer():
        try:
            worker.infer(request(tmp_path), deadline_monotonic=time.monotonic() + 1)
        except Exception as error:  # noqa: BLE001 - detect unclassified thread failures
            outcome.append(error)

    def stop():
        stop_called.set()
        worker.stop()
        stopped.set()

    inference = threading.Thread(target=infer, name="inference-probe", daemon=True)
    shutdown = threading.Thread(target=stop, daemon=True)
    inference.start()
    try:
        assert entered.wait(1)
        shutdown.start()
        assert stop_called.wait(1)
        assert not stopped.wait(0.05), "stop closed the process during its live probe"
        release.set()
        inference.join(1)
        shutdown.join(1)
        assert not inference.is_alive() and stopped.is_set()
        assert len(outcome) == 1 and isinstance(outcome[0], OcrWorkerError)
        assert str(outcome[0]) == "ocr_stopped"
    finally:
        release.set()
        inference.join(1)
        if shutdown.ident is not None:
            shutdown.join(1)
        worker.stop()


def test_runtime_readiness_belongs_to_live_generation_only():
    worker = manager(FakeContext())
    assert worker.readiness().runtime_fingerprint == "runtime"
    worker.stop()
    assert worker.readiness().runtime_fingerprint is None


def test_native_invalidated_qualification_exits_child_instead_of_serving_again(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ocr_service import pipeline
    from ocr_service.worker import _worker_main

    item = request(tmp_path)
    calls = []
    adapter = SimpleNamespace(qualified=True, profile_id='p', runtime_fingerprint='r', qualify=lambda p: None)
    def predict(*args, **kwargs):
        calls.append(kwargs['request_id'])
        adapter.qualified = False
        raise ValueError('private native failure')
    adapter.predict = predict
    monkeypatch.setattr(pipeline, 'build_pipeline', lambda *a: adapter)
    incoming, outgoing = queue.Queue(), queue.Queue()
    incoming.put(item)
    incoming.put(item)
    incoming.put(None)
    _worker_main(incoming, outgoing, 1, Path(__file__).parents[1] / 'paddleocr-service/profile.json', object(), tmp_path, tmp_path)
    assert outgoing.get()['kind'] == 'ready'
    assert outgoing.get()['kind'] == 'qualification_invalidated'
    assert calls == [item.request_id]


def test_fatal_live_worker_retires_owned_generation_and_waits_for_fresh_qualification(tmp_path):
    from ocr_service.worker import OcrWorkerError
    context = FakeContext('fatal')
    worker = manager(context)
    first = request(tmp_path)
    old_proc, old_in, old_out = worker._proc, worker._in_q, worker._out_q
    old_directory = Path(worker._reply_directory.name)
    try:
        with pytest.raises(OcrWorkerError, match='ocr_qualification_invalidated'):
            worker.infer(first, deadline_monotonic=time.monotonic()+1)
        assert not worker.readiness().ready
        assert not old_proc.is_alive() and old_in.closed and old_out.closed
        assert not old_directory.exists()
        context.mode = 'ok'
        context.init_release.set()
        for _ in range(100):
            if worker.readiness().ready:
                break
            time.sleep(.01)
        assert worker.readiness().ready
        second = request(tmp_path)
        assert worker.infer(second, deadline_monotonic=time.monotonic()+1).request_id == second.request_id
        assert context.requests == [first, second]
    finally:
        worker.stop()


def test_idle_death_recovers_from_readiness_only_and_qualifies_new_request(tmp_path):
    context = FakeContext()
    worker = manager(context)
    old = context.processes[0]
    old_channels = [worker._in_q, worker._out_q, worker._reply_sender]
    old_dir = Path(worker._reply_directory.name)
    try:
        old.dead.set()
        old.thread.join(1)
        context.started.clear()
        context.init_release.clear()
        assert not worker.readiness().ready
        assert context.started.wait(1), "readiness did not schedule replacement"
        assert worker.readiness().generation == 2
        assert not worker.readiness().ready, "replacement must freshly qualify"
        assert not context.requests
        assert all(channel.closed for channel in old_channels)
        assert not old_dir.exists()
        context.init_release.set()
        wait_ready(worker)
        item = request(tmp_path)
        assert worker.infer(item, deadline_monotonic=time.monotonic() + 1).request_id == item.request_id
        assert context.requests == [item]
        assert len(context.processes) == 2
    finally:
        worker.stop()


def test_concurrent_idle_observers_schedule_one_replacement_without_blocking_http(tmp_path):
    context = FakeContext()
    worker = manager(context)
    try:
        context.processes[0].dead.set()
        context.processes[0].thread.join(1)
        # Hold inference ownership to make background recovery observably pending.
        worker._active.acquire()
        try:
            with ThreadPoolExecutor(12) as pool:
                observations = list(pool.map(lambda _: worker.readiness(), range(24)))
            assert all(not item.ready for item in observations)
            assert len(context.processes) == 1
        finally:
            worker._active.release()
        wait_ready(worker)
        assert len(context.processes) == 2
        item = request(tmp_path)
        assert worker.infer(item, deadline_monotonic=time.monotonic() + 1).request_id == item.request_id
    finally:
        worker.stop()


def test_stop_wins_over_scheduled_idle_recovery():
    context = FakeContext()
    worker = manager(context)
    worker._active.acquire()
    try:
        context.processes[0].dead.set()
        context.processes[0].thread.join(1)
        assert not worker.readiness().ready
        worker.stop()
    finally:
        worker._active.release()
    # A scheduled recovery must finish without starting another process.
    for thread in threading.enumerate():
        if thread.name == 'ocr-idle-recovery':
            thread.join(1)
            assert not thread.is_alive()
    assert len(context.processes) == 1
    assert worker.readiness().reason_code == 'ocr_stopped'


def test_failed_idle_replacement_is_not_an_infinite_configuration_restart_loop():
    context = FakeContext()
    worker = manager(context)
    try:
        context.init_fail = True
        context.processes[0].dead.set()
        context.processes[0].thread.join(1)
        context.started.clear()
        worker.readiness()
        assert context.started.wait(1), "no replacement was attempted"
        for thread in threading.enumerate():
            if thread.name == 'ocr-idle-recovery':
                thread.join(1)
        for _ in range(20):
            assert not worker.readiness().ready
        assert len(context.processes) == 2
        assert worker.readiness().reason_code == 'ocr_init_failed'
    finally:
        worker.stop()


@pytest.mark.parametrize("delayed_name", ["ocr-worker-recovery", "ocr-idle-recovery"])
def test_poll_and_infer_share_one_failed_replacement_when_recovery_is_delayed(tmp_path, monkeypatch, delayed_name):
    from ocr_service import worker as worker_module
    from ocr_service.worker import OcrWorkerError

    context = FakeContext('hang')
    worker = manager(context)
    item = request(tmp_path)
    real_thread = threading.Thread
    release_delayed_recovery = threading.Event()
    delayed_recovery_scheduled = threading.Event()
    recovery_threads = []
    errors = []

    def schedule(*args, **kwargs):
        if kwargs.get('name') == delayed_name:
            target, target_args = kwargs['target'], kwargs.get('args', ())

            def delayed():
                delayed_recovery_scheduled.set()
                assert release_delayed_recovery.wait(2)
                target(*target_args)

            kwargs = {**kwargs, 'target': delayed, 'args': ()}
        thread = real_thread(*args, **kwargs)
        if kwargs.get('name') in {'ocr-worker-recovery', 'ocr-idle-recovery'}:
            recovery_threads.append(thread)
        return thread

    monkeypatch.setattr(worker_module.threading, 'Thread', schedule)

    def infer():
        try:
            worker.infer(item, deadline_monotonic=time.monotonic() + 3)
        except OcrWorkerError as exc:
            errors.append(str(exc))

    owner = real_thread(target=infer)
    owner.start()
    try:
        assert context.entered.wait(1), 'actual inference did not dispatch'
        # Arrange detection only: poll observes death while the real request
        # owner is waiting to enter its recovery critical section.
        with worker._control:
            context.init_fail = True
            context.processes[0].dead.set()
            context.processes[0].thread.join(1)
            assert not worker.readiness().ready
        assert delayed_recovery_scheduled.wait(1)
        owner.join(1)
        assert not owner.is_alive()
        first = next(t for t in recovery_threads if t.name != delayed_name)
        first.join(1)
        assert not first.is_alive()
        assert worker.readiness().generation == 2
        assert worker.readiness().reason_code == 'ocr_init_failed'
        assert len(context.processes) == 2
        # The delayed callback must not treat the failed replacement as another
        # recovery opportunity for the original death.
        release_delayed_recovery.set()
        for thread in recovery_threads:
            thread.join(1)
            assert not thread.is_alive()
        assert worker.readiness().generation == 2
        assert len(context.processes) == 2
        assert worker.readiness().reason_code == 'ocr_init_failed'
        assert errors == ['ocr_worker_crashed']
        assert context.requests == [item], 'failed request was replayed'
    finally:
        release_delayed_recovery.set()
        worker.stop()
        owner.join(1)
        for thread in recovery_threads:
            thread.join(1)


def test_result_artifact_write_and_read_reject_metadata_over_the_wire_cap(tmp_path, monkeypatch):
    from ocr_contract import limits
    from ocr_service.worker import publish_result, read_result

    result = response(request(tmp_path))
    result.telemetry = {"model_observations": "€" * 200}
    artifact = tmp_path / (uuid4().hex + ".json")
    artifact.write_bytes(result.model_dump_json().encode("utf-8"))
    monkeypatch.setattr(limits, "MAX_RESPONSE_METADATA_BYTES", 500)

    with pytest.raises(ValueError, match="ocr_response_metadata_too_large"):
        publish_result(result, tmp_path)
    assert list(tmp_path.glob("*.json")) == [artifact]
    with pytest.raises(ValueError, match="ocr_response_metadata_too_large"):
        read_result(tmp_path, artifact.name)
