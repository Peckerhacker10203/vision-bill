"""Fixed effective OCR policy and explicit native runtime selection."""

import hashlib
import importlib.metadata
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ADAPTER_VERSION = "native-fp32-v1"
NORMALIZER_VERSION = "rgb-source-view-v1"
MODEL_NAMES = ("PP-DocLayoutV3", "PP-LCNet_x1_0_doc_ori", "UVDoc", "PaddleOCR-VL-1.6-0.9B")


class OcrProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pipeline_version: Literal["v1.6"]
    model_name: Literal["PaddleOCR-VL-1.6-0.9B"]
    layout_model: Literal["PP-DocLayoutV3"]
    backend: Literal["native"]
    precision: Literal["float32"]
    batch_size: Literal[1]
    use_queues: Literal[False]
    use_doc_orientation_classify: Literal[True]
    use_doc_unwarping: Literal[True]
    use_chart_recognition: Literal[False]
    use_seal_recognition: Literal[False]
    min_pixels: Literal[112896]
    max_pixels: Literal[2000000]
    max_new_tokens: Literal[4096]


class RuntimeSettings(BaseModel):
    """Image-build supplied identities; no autodetection or CPU selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    accelerator: Literal["cuda", "rocm"]
    gpu_index: int = Field(ge=0)
    model_dirs: dict[str, Path]
    cache_dir: Path
    qualification_manifest: Path
    expected_packages: dict[str, str]
    expected_paddle_commit: str = Field(min_length=7)
    image_identity: str = Field(min_length=1)

    @property
    def device(self) -> str:
        # Native HIP and CUDA both use GPUPlace; `dcu` is a different backend.
        return f"gpu:{self.gpu_index}"


def load_profile(path: Path) -> OcrProfile:
    return OcrProfile.model_validate_json(path.read_text())


def content_identity(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dependency_identity(package: str) -> str:
    distribution = importlib.metadata.distribution(package)
    inventory = {}
    for relative in distribution.files or ():
        if relative.suffix == ".pyc" or relative.name == "RECORD":
            continue
        path = Path(str(distribution.locate_file(relative)))
        if path.is_file():
            inventory[str(relative)] = file_digest(path)
    if not inventory:
        raise ValueError("Installed dependency has no auditable content inventory")
    return content_identity(inventory)


def runtime_identity(profile_id: str, runtime: dict[str, Any]) -> str:
    return content_identity({"profile_id": profile_id, "runtime": runtime})


def check_accelerator(paddle: Any, accelerator: str, gpu_index: int) -> None:
    rocm = bool(paddle.is_compiled_with_rocm())
    cuda = bool(paddle.is_compiled_with_cuda())
    if (accelerator == "rocm" and not rocm) or (accelerator == "cuda" and (rocm or not cuda)):
        raise ValueError("Paddle build does not match requested native accelerator")
    if (
        accelerator not in ("cuda", "rocm")
        or not 0 <= gpu_index < paddle.device.cuda.device_count()
    ):
        raise ValueError("Requested GPU is unavailable; CPU fallback is forbidden")


# Complete PaddleX v3.7.2 v1.6 YAML, with policy overrides explicit.
_BASE_CONFIG = {
    "pipeline_name": "PaddleOCR-VL-1.6",
    "batch_size": 1,
    "use_queues": False,
    "use_doc_preprocessor": True,
    "use_layout_detection": True,
    "use_chart_recognition": False,
    "use_seal_recognition": False,
    "format_block_content": False,
    "merge_layout_blocks": True,
    "markdown_ignore_labels": [
        "number",
        "footnote",
        "header",
        "header_image",
        "footer",
        "footer_image",
        "aside_text",
    ],
    "SubModules": {
        "LayoutDetection": {
            "module_name": "layout_detection",
            "model_name": "PP-DocLayoutV3",
            "model_dir": None,
            "batch_size": 1,
            "threshold": 0.3,
            "layout_nms": True,
            "layout_unclip_ratio": [1.0, 1.0],
            "layout_merge_bboxes_mode": {
                0: "union",
                1: "union",
                2: "union",
                3: "large",
                4: "union",
                5: "large",
                6: "large",
                7: "union",
                8: "union",
                9: "union",
                10: "union",
                11: "union",
                12: "union",
                13: "union",
                14: "union",
                15: "large",
                16: "union",
                17: "large",
                18: "union",
                19: "union",
                20: "union",
                21: "union",
                22: "union",
                23: "union",
                24: "union",
            },
        },
        "VLRecognition": {
            "module_name": "vl_recognition",
            "model_name": "PaddleOCR-VL-1.6-0.9B",
            "model_dir": None,
            "batch_size": 1,
            "genai_config": {"backend": "native"},
        },
    },
    "SubPipelines": {
        "DocPreprocessor": {
            "pipeline_name": "doc_preprocessor",
            "batch_size": 1,
            "use_doc_orientation_classify": True,
            "use_doc_unwarping": True,
            "SubModules": {
                "DocOrientationClassify": {
                    "module_name": "doc_text_orientation",
                    "model_name": "PP-LCNet_x1_0_doc_ori",
                    "model_dir": None,
                    "batch_size": 1,
                },
                "DocUnwarping": {
                    "module_name": "image_unwarping",
                    "model_name": "UVDoc",
                    "model_dir": None,
                    "batch_size": 1,
                },
            },
        }
    },
    "layout_prep_cpu_workers": 0,
}


def effective_pipeline_config(profile: OcrProfile) -> dict[str, Any]:
    OcrProfile.model_validate(profile.model_dump())
    return deepcopy(_BASE_CONFIG)
