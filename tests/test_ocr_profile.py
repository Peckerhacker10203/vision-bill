import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

PROFILE = Path(__file__).parents[1] / "paddleocr-service/profile.json"


def test_profile_is_exact():
    from ocr_service.profile import effective_pipeline_config, load_profile

    profile = load_profile(PROFILE)
    assert (profile.min_pixels, profile.max_pixels, profile.max_new_tokens) == (
        112896,
        2000000,
        4096,
    )
    assert profile.use_doc_orientation_classify is True
    assert profile.use_doc_unwarping is True
    assert profile.use_queues is False
    config = effective_pipeline_config(profile)
    assert config["use_doc_preprocessor"] is True
    assert config["SubPipelines"]["DocPreprocessor"]["use_doc_unwarping"] is True
    assert config["SubPipelines"]["DocPreprocessor"]["use_doc_orientation_classify"] is True
    assert config["SubModules"]["VLRecognition"]["genai_config"] == {"backend": "native"}
    assert config["SubModules"]["LayoutDetection"]["model_name"] == "PP-DocLayoutV3"

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "batch_size":
                    assert item == 1
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(config)
    config["SubModules"]["LayoutDetection"]["batch_size"] = 100
    assert effective_pipeline_config(profile)["SubModules"]["LayoutDetection"]["batch_size"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_pixels", 1605632),
        ("precision", "bfloat16"),
        ("use_doc_unwarping", False),
        ("use_doc_orientation_classify", False),
        ("use_queues", True),
    ],
)
def test_profile_rejects_adaptive_changes(tmp_path, field, value):
    from ocr_service.profile import load_profile

    data = json.loads(PROFILE.read_text())
    data[field] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError):
        load_profile(path)


def test_runtime_only_supports_explicit_native_accelerators(tmp_path):
    from ocr_service.profile import RuntimeSettings

    with pytest.raises(ValidationError):
        RuntimeSettings(accelerator="cpu")


def test_identity_separates_effective_content_from_device():
    from ocr_service.profile import content_identity, runtime_identity

    config = {"a": 1, "b": {"c": 2}}
    assert content_identity(config) == content_identity({"b": {"c": 2}, "a": 1})
    changed = copy.deepcopy(config)
    changed["b"]["c"] = 3
    assert content_identity(config) != content_identity(changed)
    assert runtime_identity("profile", {"device": "a"}) != runtime_identity(
        "profile", {"device": "b"}
    )


def test_dependency_identity_hashes_installed_content_and_ignores_bytecode(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ocr_service import profile

    code = tmp_path / "module.py"
    code.write_text("version = 1")
    bytecode = tmp_path / "module.pyc"
    bytecode.write_bytes(b"generated")
    distribution = SimpleNamespace(
        files=[Path("module.py"), Path("module.pyc")], locate_file=lambda path: tmp_path / path
    )
    monkeypatch.setattr(profile.importlib.metadata, "distribution", lambda name: distribution)
    first = profile.dependency_identity("package")
    bytecode.write_bytes(b"changed")
    assert profile.dependency_identity("package") == first
    code.write_text("version = 2")
    assert profile.dependency_identity("package") != first
