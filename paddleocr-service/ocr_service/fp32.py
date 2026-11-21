"""Native VLM substitution with an explicit loader dtype and parameter audit."""

import platform
import sys
from collections.abc import Callable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any


def floating_dtype(dtype: Any) -> str | None:
    normalized = str(dtype).lower().rsplit(".", 1)[-1]
    floating = "float" in normalized or normalized in {"fp16", "bf16", "fp32", "fp64"}
    return normalized if floating else None


def require_gpu(tensor: Any, gpu_index: int) -> None:
    location = tensor.place
    if not location.is_gpu_place():
        raise ValueError("Tensor is outside the requested GPU")
    if location.gpu_device_id() != gpu_index:
        raise ValueError("Tensor is on a different GPU index")


def audit_dynamic_model(model: Any, gpu_index: int) -> int:
    inspected = 0
    for weight in model.parameters():
        precision = floating_dtype(weight.dtype)
        if precision is None:
            continue
        if precision != "float32":
            raise ValueError("Native dynamic floating parameter precision must be FP32")
        require_gpu(weight, gpu_index)
        inspected += 1
    if inspected == 0:
        raise ValueError("Native dynamic model has no verified floating parameters")
    return inspected


def requires_host_assembly(accelerator: str, capability: Sequence[int] | None) -> bool:
    if accelerator != "cuda" or capability is None or tuple(capability) != (12, 0):
        return False
    release = platform.release().casefold()
    return ("microsoft" in release and "wsl2" in release) or Path("/dev/dxg").exists()


def fp32_predictor_class(
    upstream: type[Any],
    gpu_index: int,
    *,
    assemble_on_host: bool = False,
) -> type[Any]:
    class ExplicitFp32Predictor(upstream):  # type: ignore[misc]
        def _build(self, **kwargs: Any) -> Any:
            # The pinned upstream constructor chooses BF16 immediately before _build.
            self.dtype = "float32"
            path = "host_fp32" if assemble_on_host else "native_gpu"
            print(
                f"ocr_vlm_checkpoint_load path={path} stage=assemble", file=sys.stderr, flush=True
            )
            if assemble_on_host:
                model_type = import_module(
                    "paddlex.inference.models.doc_vlm.modeling"
                ).PaddleOCRVLForConditionalGeneration
                device_guard = import_module("paddlex.utils.device").TemporaryDeviceChanger
                processor = self.build_processor()
                # RMSNorm and other configuration choices depend on the execution device.
                with device_guard(self.device):
                    config = model_type.config_class.from_pretrained(self.model_dir)
                    with device_guard("cpu"):
                        model = model_type.from_pretrained(
                            self.model_dir,
                            config=config,
                            dtype=self.dtype,
                            convert_from_hf=True,
                        )
                    print(
                        f"ocr_vlm_checkpoint_load path={path} stage=transfer",
                        file=sys.stderr,
                        flush=True,
                    )
                    # Paddle Layer.to moves existing parameter storage and all registered
                    # buffers in-place, including shared objects and nonpersistent buffers.
                    model.to(device=self.device, blocking=True)
                for param in model.parameters():
                    require_gpu(param, gpu_index)
                for buffer in model.buffers():
                    require_gpu(buffer, gpu_index)
                    if floating_dtype(buffer.dtype) not in (None, "float32"):
                        raise ValueError("Native floating buffer precision must be FP32")
                built = model, processor
            else:
                built = super()._build(**kwargs)
            audit_dynamic_model(built[0], gpu_index)
            print(f"ocr_vlm_checkpoint_load path={path} stage=ready", file=sys.stderr, flush=True)
            return built

    return ExplicitFp32Predictor


def native_pipeline_class(
    upstream: type[Any], predictor: Callable[..., Any], normalize: Callable[..., Any]
) -> type[Any]:
    class ExplicitNativePipeline(upstream):  # type: ignore[misc]
        def create_model(self, config: dict[str, Any], **kwargs: Any) -> Any:
            if config.get("module_name") == "vl_recognition":
                name = config["model_name"]
                directory = config.get("model_dir")
                if name != "PaddleOCR-VL-1.6-0.9B" or not directory:
                    raise ValueError("Native VLM requires its pinned name and explicit model path")
                normalized = normalize(
                    "paddle_dynamic",
                    config.get("engine_config", {}),
                    model_name=name,
                    device=self.device,
                )
                return predictor(
                    model_name=name,
                    model_dir=directory,
                    batch_size=1,
                    engine_config=normalized,
                    **kwargs,
                )
            return super().create_model(config, **kwargs)

    return ExplicitNativePipeline
