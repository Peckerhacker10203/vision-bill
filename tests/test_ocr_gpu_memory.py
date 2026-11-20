"""Exercise selected-device driver calls with CPU doubles."""

from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
@pytest.mark.parametrize("failed", [False, True])
def test_preload_probe_reads_selected_device_and_restores_context(monkeypatch, backend, failed):
    from ocr_service import gpu_memory

    driver = Mock()
    initial_device = 1
    selected_device = 4
    cuda_context = 124
    current_device = initial_device
    current_context = 42

    def set_output(out, value):
        out._obj.value = value
        return 0

    def select_hip_device(index):
        nonlocal current_device
        current_device = index
        return 0

    def push_cuda_context(context):
        nonlocal current_context
        current_context = context.value
        return 0

    def pop_cuda_context(context):
        nonlocal current_context
        set_output(context, current_context)
        current_context = 42
        return 0

    def read(free, total):
        if backend == "cuda":
            assert current_context == cuda_context
        else:
            assert current_device == selected_device
        if failed:
            return 2
        set_output(free, 123)
        return set_output(total, 456)

    def cuda_device(out, ordinal):
        assert ordinal == selected_device
        return set_output(out, 9)

    def retain(context, native_device):
        assert native_device.value == 9
        return set_output(context, cuda_context)

    driver.cuInit.return_value = 0
    driver.cuDeviceGet.side_effect = cuda_device
    driver.cuDevicePrimaryCtxRetain.side_effect = retain
    driver.cuCtxPushCurrent_v2.side_effect = push_cuda_context
    driver.cuCtxPopCurrent_v2.side_effect = pop_cuda_context
    driver.cuDevicePrimaryCtxRelease_v2.return_value = 0
    driver.cuMemGetInfo_v2.side_effect = read
    driver.hipGetDevice.side_effect = lambda out: set_output(out, current_device)
    driver.hipSetDevice.side_effect = select_hip_device
    driver.hipMemGetInfo.side_effect = read

    def load_library(library):
        assert library == {"cuda": "libcuda.so.1", "rocm": "libamdhip64.so"}[backend]
        return driver

    monkeypatch.setattr(gpu_memory.ctypes, "CDLL", load_library)
    reading = gpu_memory.probe_gpu_memory(backend, selected_device)
    if failed:
        assert reading is None
    else:
        assert reading.model_dump() == {
            "accelerator": backend, "gpu_index": 4, "free_bytes": 123, "total_bytes": 456
        }
    assert current_device == initial_device
    assert current_context == 42
    if backend == "cuda":
        assert driver.cuDevicePrimaryCtxRelease_v2.call_args.args[0].value == 9
