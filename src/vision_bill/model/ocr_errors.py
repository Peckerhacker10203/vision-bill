"""Safe application errors reported by the native OCR worker."""

OCR_GPU_OUT_OF_MEMORY = "ocr_gpu_out_of_memory"


class OcrGpuOutOfMemoryError(RuntimeError):
    """The native worker reported a device allocation failure."""

    def __init__(self) -> None:
        super().__init__(OCR_GPU_OUT_OF_MEMORY)
