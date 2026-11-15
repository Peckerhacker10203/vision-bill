"""Validated OCR policy, native accelerator selection, and content identities."""

import hashlib
import importlib.metadata
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ADAPTER_VERSION = "native-fp32-v1"
NORMALIZER_VERSION = "rgb-source-view-v1"
MODEL_NAMES = (
    "PP-DocLayoutV3",
    "PP-LCNet_x1_0_doc_ori",
    "UVDoc",
    "PaddleOCR-VL-1.6-0.9B",
)


class OcrProfile(BaseModel):
    """The approved policy admits no platform-dependent substitutions."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    pipeline_version: Literal["v1.6"]
    backend: Literal["native"]
    precision: Literal["float32"]
    model_name: Literal["PaddleOCR-VL-1.6-0.9B"]
    layout_model: Literal["PP-DocLayoutV3"]
    min_pixels: Literal[112896]
    max_pixels: Literal[2000000]
    max_new_tokens: Literal[4096]
    batch_size: Literal[1]
    use_doc_orientation_classify: Literal[True]
    use_doc_unwarping: Literal[True]
    use_queues: Literal[False]
    use_chart_recognition: Literal[False]
    use_seal_recognition: Literal[False]


class RuntimeSettings(BaseModel):
    """Explicit identities and local paths supplied by the native image build."""

    model_config = ConfigDict(frozen=True, extra="forbid")
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
        # Paddle's native CUDA and HIP builds both expose GPUPlace.
        return "gpu:" + str(self.gpu_index)


def load_profile(path: Path) -> OcrProfile:
    return OcrProfile.model_validate_json(path.read_bytes())


def check_accelerator(paddle: Any, accelerator: str, gpu_index: int) -> None:
    if accelerator == "rocm":
        matches = paddle.is_compiled_with_rocm()
    elif accelerator == "cuda":
        matches = paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm()
    else:
        matches = False
    if not matches:
        raise ValueError("Requested accelerator does not match the native Paddle build")
    visible_count = paddle.device.cuda.device_count()
    if gpu_index < 0 or gpu_index >= visible_count:
        raise ValueError("Requested GPU index is unavailable; CPU fallback is forbidden")


def file_digest(path: Path) -> str:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256")
    return digest.hexdigest()


def content_identity(value: Any) -> str:
    encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def dependency_identity(package: str) -> str:
    installed = importlib.metadata.distribution(package)
    content: dict[str, str] = {}
    for entry in installed.files or []:
        # Bytecode and pip's rewritten RECORD are installation outputs.
        if entry.suffix == ".pyc" or entry.name == "RECORD":
            continue
        installed_file = Path(str(installed.locate_file(entry)))
        if installed_file.is_file():
            content[str(entry)] = file_digest(installed_file)
    if not content:
        raise ValueError("Dependency content inventory is empty")
    return content_identity(content)


def runtime_identity(profile_id: str, runtime: dict[str, Any]) -> str:
    return content_identity({"profile_id": profile_id, "runtime": runtime})


# Regenerated from the official PaddleX v3.7.2 PaddleOCR-VL-1.6 YAML.
# Runtime model directories and engine configuration are resolved by the factory.
_PIPELINE_TEMPLATE: dict[str, Any] = {
    "pipeline_name": "PaddleOCR-VL-1.6",
    "batch_size": 64,
    "use_queues": True,
    "use_doc_preprocessor": False,
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
            "batch_size": 8,
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
            "batch_size": -1,
            "genai_config": {"backend": "native"},
        },
    },
    "SubPipelines": {
        "DocPreprocessor": {
            "pipeline_name": "doc_preprocessor",
            "batch_size": 8,
            "use_doc_orientation_classify": True,
            "use_doc_unwarping": True,
            "SubModules": {
                "DocOrientationClassify": {
                    "module_name": "doc_text_orientation",
                    "model_name": "PP-LCNet_x1_0_doc_ori",
                    "model_dir": None,
                    "batch_size": 8,
                },
                "DocUnwarping": {
                    "module_name": "image_unwarping",
                    "model_name": "UVDoc",
                    "model_dir": None,
                },
            },
        }
    },
}


def effective_pipeline_config(profile: OcrProfile) -> dict[str, Any]:
    validated = OcrProfile.model_validate(profile.model_dump())
    config = deepcopy(_PIPELINE_TEMPLATE)

    def set_batch_one(section: Any) -> None:
        if isinstance(section, dict):
            if "batch_size" in section or "model_name" in section or "pipeline_name" in section:
                section["batch_size"] = validated.batch_size
            for child in section.values():
                set_batch_one(child)
        elif isinstance(section, list):
            for child in section:
                set_batch_one(child)

    set_batch_one(config)
    config.update(
        use_doc_preprocessor=True,
        use_queues=validated.use_queues,
        use_chart_recognition=validated.use_chart_recognition,
        use_seal_recognition=validated.use_seal_recognition,
        layout_prep_cpu_workers=0,
    )
    preprocessor = config["SubPipelines"]["DocPreprocessor"]
    preprocessor["use_doc_orientation_classify"] = validated.use_doc_orientation_classify
    preprocessor["use_doc_unwarping"] = validated.use_doc_unwarping
    return config
