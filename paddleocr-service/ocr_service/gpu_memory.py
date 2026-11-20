"""Advisory free device memory and classification of native allocation errors."""

import ctypes
import re
from typing import Any, Literal

from ocr_contract import GpuMemorySnapshot
from pydantic import ValidationError

GPU_OOM_CODE = "ocr_gpu_out_of_memory"
GPU_OOM_EXIT = 42  # Reserved between the supervised qualification processes.

# Paddle's shared CUDA/HIP allocator explicitly names device memory exhaustion.
# Bare runtime allocation errors also occur for page-locked host memory.
_GPU_ALLOCATION_ERROR = re.compile(
    r"^\s*(?:\(ResourceExhausted\)\s*)?Out of memory error on GPU \d+\b",
    re.MULTILINE,
)


class GpuOutOfMemory(RuntimeError):
    def __init__(self) -> None:
        super().__init__(GPU_OOM_CODE)


def is_gpu_out_of_memory(error: BaseException) -> bool:
    """Recognize explicit GPU allocation evidence, including wrapped native errors."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, GpuOutOfMemory) or _GPU_ALLOCATION_ERROR.search(str(current)):
            return True
        seen.add(id(current))
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return False


def _call(library: ctypes.CDLL, symbol: str, types: list[Any], *args: Any) -> None:
    function = getattr(library, symbol)
    function.argtypes = types
    function.restype = ctypes.c_int
    if function(*args) != 0:
        raise OSError(f"GPU memory query failed in {symbol}")


def _memory_info(library: ctypes.CDLL, symbol: str) -> tuple[int, int]:
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    size_pointer = ctypes.POINTER(ctypes.c_size_t)
    _call(library, symbol, [size_pointer, size_pointer], ctypes.byref(free), ctypes.byref(total))
    return free.value, total.value


def _cuda_memory(index: int) -> tuple[int, int]:
    driver = ctypes.CDLL("libcuda.so.1")
    device, context = ctypes.c_int(), ctypes.c_void_p()
    _call(driver, "cuInit", [ctypes.c_uint], 0)
    _call(
        driver, "cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        ctypes.byref(device), index,
    )
    _call(
        driver, "cuDevicePrimaryCtxRetain", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
        ctypes.byref(context), device,
    )
    try:
        _call(driver, "cuCtxPushCurrent_v2", [ctypes.c_void_p], context)
        try:
            return _memory_info(driver, "cuMemGetInfo_v2")
        finally:
            _call(
                driver, "cuCtxPopCurrent_v2", [ctypes.POINTER(ctypes.c_void_p)],
                ctypes.byref(context),
            )
    finally:
        _call(driver, "cuDevicePrimaryCtxRelease_v2", [ctypes.c_int], device)


def _hip_memory(index: int) -> tuple[int, int]:
    runtime = ctypes.CDLL("libamdhip64.so")
    previous = ctypes.c_int()
    _call(runtime, "hipGetDevice", [ctypes.POINTER(ctypes.c_int)], ctypes.byref(previous))
    _call(runtime, "hipSetDevice", [ctypes.c_int], index)
    try:
        return _memory_info(runtime, "hipMemGetInfo")
    finally:
        _call(runtime, "hipSetDevice", [ctypes.c_int], previous.value)


def probe_gpu_memory(
    accelerator: Literal["cuda", "rocm"], gpu_index: int
) -> GpuMemorySnapshot | None:
    """Sample the visible ordinal inside a disposable native process before model load."""
    try:
        free, total = (_cuda_memory if accelerator == "cuda" else _hip_memory)(gpu_index)
        return GpuMemorySnapshot(
            accelerator=accelerator, gpu_index=gpu_index, free_bytes=free, total_bytes=total
        )
    except (OSError, AttributeError, ValidationError):
        return None
