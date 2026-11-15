"""Small source-bound substitution: set dtype before the sole native loader call."""

from collections.abc import Callable
from typing import Any


def floating_dtype(dtype: Any) -> str | None:
    name = str(dtype).removeprefix("paddle.").removeprefix("DataType.").lower()
    if "float" in name or name in ("fp16", "fp32", "fp64", "bf16"):
        return name
    return None


def require_gpu(tensor: Any, gpu_index: int) -> None:
    place = tensor.place
    if not place.is_gpu_place() or place.gpu_device_id() != gpu_index:
        raise ValueError("Floating tensor is not on the requested native GPU")


def audit_dynamic_model(model: Any, gpu_index: int) -> int:
    floating = 0
    for parameter in model.parameters():
        dtype = floating_dtype(parameter.dtype)
        if dtype is not None:
            floating += 1
            if dtype != "float32":
                raise ValueError("Native VLM has a non-FP32 floating parameter")
            require_gpu(parameter, gpu_index)
    if not floating:
        raise ValueError("Native VLM has no audited floating parameters")
    return floating


def fp32_predictor_class(upstream: type[Any], gpu_index: int) -> type[Any]:
    class Fp32DocVLMPredictor(upstream):  # type: ignore[misc]
        def _build(self, **kwargs: Any) -> Any:
            self.dtype = "float32"
            model, processor = super()._build(**kwargs)
            audit_dynamic_model(model, gpu_index)
            return model, processor

    return Fp32DocVLMPredictor


def native_pipeline_class(
    upstream: type[Any], predictor: Callable[..., Any], normalize: Callable[..., Any]
) -> type[Any]:
    class NativeFp32Pipeline(upstream):  # type: ignore[misc]
        def create_model(self, config: dict[str, Any], **kwargs: Any) -> Any:
            if config.get("module_name") != "vl_recognition":
                return super().create_model(config, **kwargs)
            if config["model_name"] != "PaddleOCR-VL-1.6-0.9B" or not config.get("model_dir"):
                raise ValueError("Explicit pinned VLM model directory is required")
            engine_config = normalize(
                "paddle_dynamic",
                config.get("engine_config", {}),
                model_name=config["model_name"],
                device=self.device,
            )
            return predictor(
                model_name=config["model_name"],
                model_dir=config["model_dir"],
                batch_size=1,
                engine_config=engine_config,
                **kwargs,
            )

    return NativeFp32Pipeline
