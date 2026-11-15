"""Static recipe checks do not assert a working GPU binary."""

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "paddleocr-service"


def test_separate_digest_pinned_native_images_and_cuda_lock():
    for accelerator in ("cuda", "rocm"):
        path = SERVICE / f"Dockerfile.{accelerator}"
        assert path.is_file(), f"{accelerator} image missing"
        text = path.read_text()
        assert re.search(r"FROM \S+@sha256:[0-9a-f]{64}", text)
        assert "COPY paddleocr-service/ocr_service/" in text
        assert "COPY src/vision_bill/model/ocr.py" in text
        assert "/ready" in text
    cuda = (SERVICE / "requirements.cuda.lock").read_text()
    assert "paddlepaddle_gpu-3.3.1" in cuda and "cu126" in cuda
    if "UNRESOLVED" in cuda:
        import pytest

        with pytest.raises(ValueError, match="unresolved"):
            load_build_module("resolve_cuda").validate_lock(SERVICE / "requirements.cuda.lock")
    else:
        load_build_module("resolve_cuda").validate_lock(SERVICE / "requirements.cuda.lock")
    assert "requirements.cuda.lock" not in (SERVICE / "Dockerfile.rocm").read_text()


def test_rocm_exact_pins_full_sdk_targets_and_no_spoofing():
    path = SERVICE / "build/build_rocm.sh"
    assert path.is_file(), "native source build missing"
    script = path.read_text()
    lock = json.loads((SERVICE / "runtime-lock.json").read_text())
    assert lock["sources"]["paddle"] == "7688495538f4d6c1893f084dd238a402e8f68ab6"
    assert lock["sources"]["therock"] == "b324869fb3116353ac7eec5e6926aa3d452ac987"
    assert lock["rdna2_targets"] == ["gfx1030", "gfx1031", "gfx1032", "gfx1034"]
    assert "THEROCK_ENABLE_ALL=ON" in script and "--target dist-rocm" in script
    assert "HSA_OVERRIDE_GFX_VERSION" not in script
    assert "WITH_FLASHATTN=OFF" in script
    assert "git apply --check" in script
    patch = (SERVICE / "build/hip-rdna2-v1.patch").read_text()
    assert patch.count("+        --offload-arch=gfx1034") == 2
    assert "WITH_ROCM AND WITH_FLASHATTN" in patch


def test_sidecar_root_context_cache_and_internal_readiness():
    base = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    service = base["services"]["paddleocr-service"]
    assert service["build"]["context"] == "."
    assert not service.get("ports")
    assert service["environment"]["PADDLE_PDX_CACHE_HOME"] == "/var/cache/paddlex"
    assert any("/var/cache/paddlex" in volume for volume in service["volumes"])
    assert "/ready" in str(service["healthcheck"])
    ignore = (ROOT / ".dockerignore").read_text().splitlines()
    assert "*" in ignore and "!paddleocr-service/ocr_service/*.py" in ignore
    assert "!paddleocr-service/**" not in ignore
    for accelerator in ("cuda", "rocm"):
        overlay = yaml.safe_load((ROOT / f"docker-compose.{accelerator}.yml").read_text())
        ocr = overlay["services"]["paddleocr-service"]
        assert ocr["build"]["dockerfile"] == f"paddleocr-service/Dockerfile.{accelerator}"
        assert not ocr.get("ports") and not ocr.get("privileged")
    rocm = yaml.safe_load((ROOT / "docker-compose.rocm.yml").read_text())["services"][
        "paddleocr-service"
    ]
    assert "/dev/kfd:/dev/kfd" in rocm["devices"]
    assert all("/dev/dri:" not in device for device in rocm["devices"])


def load_build_module(name):
    import importlib.util

    path = SERVICE / "build" / f"{name}.py"
    assert path.is_file(), f"{name} tool missing"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_resolver_selects_only_exact_official_cp312_artifact():
    module = load_build_module("resolve_cuda")
    official = "https://paddle-whl.bj.bcebos.com/stable/cu126/paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl"
    html = f'<a href="{official}">wheel</a><a href="https://evil.invalid/paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl">other</a>'
    assert module.select_wheel(html) == official
    import pytest

    with pytest.raises(ValueError):
        module.select_wheel(html + f'<a href="{official}?other">duplicate</a>')
    with pytest.raises(ValueError):
        module.select_wheel(html.replace("3.3.1", "3.3.0"))


def test_unresolved_cuda_lock_is_explicit_and_build_validation_rejects_it(tmp_path):
    module = load_build_module("resolve_cuda")
    import pytest

    lock = tmp_path / "requirements.cuda.lock"
    lock.write_text("# UNRESOLVED: official wheel access required\n")
    with pytest.raises(ValueError, match="unresolved"):
        module.validate_lock(lock)


def test_kernel_inspector_rejects_missing_required_targets():
    module = load_build_module("inspect_kernels")
    import pytest

    with pytest.raises(ValueError, match="missing"):
        module.require_targets({"gfx1030"}, {"gfx1030", "gfx1031"})


def test_opaque_kernel_payload_stays_unverified_not_missing_or_supported(tmp_path, monkeypatch):
    module = load_build_module("inspect_kernels")
    from types import SimpleNamespace

    lib = tmp_path / "librocblas.so"
    lib.write_bytes(b"\x7fELFopaque-packed-library")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    evidence = module.inventory([lib], tmp_path / "llvm-objdump")
    assert evidence["offline_status"] == "unverified"
    assert evidence["targets"] == [] and str(lib) in evidence["files"]
    import pytest

    with pytest.raises(ValueError, match="library"):
        module.inventory([], tmp_path / "llvm-objdump")


def test_native_compile_inventory_rejects_extra_or_missing_targets(tmp_path):
    module = load_build_module("inspect_kernels")
    import pytest

    commands = tmp_path / "compile_commands.json"
    commands.write_text(json.dumps([{"command": "hipcc -c x --offload-arch=gfx1030"}]))
    with pytest.raises(ValueError, match="target"):
        module.validate_compile_targets(commands)
    commands.write_text(
        json.dumps(
            [
                {
                    "command": "hipcc -c x "
                    + " ".join("--offload-arch=" + target for target in sorted(module.TARGETS))
                }
            ]
        )
    )
    module.validate_compile_targets(commands)


def test_build_bootstrap_tools_are_hashed_before_no_isolation_source_builds():
    bootstrap = SERVICE / "requirements.bootstrap.lock"
    assert bootstrap.is_file(), "locked build bootstrap missing"
    assert "setuptools==" in bootstrap.read_text() and "--hash=sha256:" in bootstrap.read_text()
    for accelerator in ("cuda", "rocm"):
        text = (SERVICE / f"Dockerfile.{accelerator}").read_text()
        assert "--only-binary=:all: -r /opt/locks/requirements.bootstrap.lock" in text
        assert "--no-build-isolation -r /opt/locks/requirements.build.lock" in text


def test_runtime_lock_hashes_match_committed_inputs():
    import hashlib

    lock = json.loads((SERVICE / "runtime-lock.json").read_text())
    for name, digest in lock["input_sha256"].items():
        assert hashlib.sha256((SERVICE / name).read_bytes()).hexdigest() == digest, name


def test_sidecar_directory_negations_do_not_reinclude_private_descendants():
    lines = (ROOT / ".dockerignore").read_text().splitlines()
    for directory in (
        "paddleocr-service",
        "paddleocr-service/ocr_service",
        "paddleocr-service/build",
    ):
        index = lines.index("!" + directory + "/")
        assert lines[index + 1] == directory + "/**", (
            "Docker negated directory otherwise re-includes all descendants"
        )


def test_kernel_targets_come_from_metadata_not_object_filenames(tmp_path, monkeypatch):
    module = load_build_module("inspect_kernels")
    from types import SimpleNamespace

    lib = tmp_path / "gfx1030.co"
    lib.write_bytes(b"\x7fELF")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=str(lib) + ": file format elf64-amdgpu\n"),
    )
    assert module.inventory([lib], tmp_path / "llvm-objdump")["offline_status"] == "unverified"
    output = "\n".join("arch            " + target for target in sorted(module.TARGETS))
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=output))
    assert module.inventory([lib], tmp_path / "llvm-objdump")["offline_status"] == "verified"


def cuda_lock_fixture(tmp_path):
    import hashlib

    module = load_build_module("resolve_cuda")
    lock = tmp_path / "requirements.cuda.lock"
    (tmp_path / "requirements.runtime.lock").write_bytes(
        (SERVICE / "requirements.runtime.lock").read_bytes()
    )
    url = "https://paddle-whl.bj.bcebos.com/stable/cu126/paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl"
    digest = hashlib.sha256(b"synthetic wheel test bytes").hexdigest()
    native = f"paddlepaddle-gpu @ {url} \\\n    --hash=sha256:{digest}\n"
    text = (SERVICE / "requirements.runtime.lock").read_text() + "\n" + native
    data = {
        "index": module.INDEX,
        "index_sha256": hashlib.sha256(b"test index").hexdigest(),
        "wheel_url": url,
        "wheel_sha256": digest,
        "metadata_sha256": hashlib.sha256(b"test metadata").hexdigest(),
        "metadata_name": "paddlepaddle-gpu",
        "metadata_version": "3.3.1",
        "requires_dist": ["numpy>=1.21", "opt_einsum==3.3.0"],
        "platform": "linux-x86_64-cp312",
    }

    def write(contents=text, fields=None):
        lock.write_text(contents)
        attested = {**data, "lock_sha256": hashlib.sha256(contents.encode()).hexdigest()}
        if fields is not None:
            attested = fields(attested)
        lock.with_suffix(".attestation.json").write_text(json.dumps(attested))
        return lock

    return module, write, text, native


def test_cuda_validator_accepts_complete_resolver_shaped_executable_lock(tmp_path):
    module, write, _, _ = cuda_lock_fixture(tmp_path)
    assert module.validate_lock(write())["metadata_version"] == "3.3.1"


def test_cuda_validator_rejects_comment_only_and_shadowed_native_requirements(tmp_path):
    import pytest

    module, write, text, native = cuda_lock_fixture(tmp_path)
    commented = "\n".join("# " + line for line in native.splitlines()) + "\n"
    variants = [
        commented,
        text.replace(native, commented),
        text.replace(native, commented + "paddlepaddle==3.3.1 --hash=sha256:" + "b" * 64 + "\n"),
        text.replace(native, commented + native.replace("cu126/", "cu128/")),
        text + native.replace("paddlepaddle-gpu", "PaddlePaddle_GPU"),
        text.replace("paddlepaddle-gpu @", "paddlepaddle-gpu[other] @"),
        text.replace(native, native.replace(" \\\n", ' ; python_version < "1" \\\n')),
    ]
    for contents in variants:
        with pytest.raises(ValueError):
            module.validate_lock(write(contents))


def test_cuda_validator_rejects_incomplete_attestation_or_unclosed_lock(tmp_path):
    import pytest

    module, write, text, _ = cuda_lock_fixture(tmp_path)
    for field in ("index_sha256", "metadata_sha256", "requires_dist", "platform"):
        with pytest.raises(ValueError):
            module.validate_lock(
                write(
                    fields=lambda data, field=field: {k: v for k, v in data.items() if k != field}
                )
            )
    for field, value in [
        ("index_sha256", "not-a-hash"),
        ("requires_dist", "numpy"),
        ("platform", "windows"),
        ("requires_dist", ["absent-gpu-dependency==1"]),
    ]:
        with pytest.raises(ValueError):
            module.validate_lock(
                write(fields=lambda data, field=field, value=value: {**data, field: value})
            )
    variants = [
        text.replace("numpy==2.2.6", "numpy==1.0.0"),
        text.replace("pillow-heif==1.7.0", "unrelated==1.7.0"),
        text + "\n-r another.lock\n",
        text + "\n--index-url https://evil.invalid\n",
        text + "\nunpinned-package\n",
        text + "\nno-hash==1\n",
    ]
    for contents in variants:
        with pytest.raises(ValueError):
            module.validate_lock(write(contents))
