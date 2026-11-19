import hashlib
import io
import json
import os
import platform
import runpy
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
from test_ocr_provision import release as ocr_release  # type: ignore[import-not-found] # noqa: F401

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ocr_source_up.py"


@pytest.mark.parametrize("accelerator", ["cuda", "rocm"])
def test_source_command_prepares_default_inputs(
    accelerator: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import ocr_source_up

    calls = []

    def bootstrap(chosen: str, inputs: Path | None, *, restart: bool) -> None:
        calls.append((chosen, inputs, restart))

    monkeypatch.setattr(sys, "argv", [str(SCRIPT), accelerator])
    monkeypatch.setattr(ocr_source_up, "bootstrap", bootstrap)
    assert ocr_source_up.main() == 0
    assert calls == [(accelerator, None, False)]


def test_source_command_rejects_supplied_inputs_with_restart() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "cuda", "approved", "--restart"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "APPROVED_DIRECTORY" in result.stderr
    assert "--restart" in result.stderr


def test_staging_keeps_approved_metadata_and_checks_fixture_bytes(tmp_path: Path) -> None:
    from scripts.ocr_source_up import stage_approved_inputs

    approved = tmp_path / "approved"
    approved.mkdir()
    fixture = approved / "assets" / "fixtures.tar"
    fixture.parent.mkdir()
    fixture.write_bytes(b"sealed fixture archive")
    manifest = {
        "accelerator": "rocm",
        "gpu_index": 0,
        "models": {"model": {"source": "https://example.org/model.tar"}},
        "fixtures": {
            "source": "assets/fixtures.tar",
            "size_bytes": 22,
            "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        },
    }
    (approved / "bundle.json").write_text(json.dumps(manifest))
    destination = tmp_path / "staged"
    destination.mkdir()
    assert stage_approved_inputs(approved, destination, "rocm") == manifest
    assert (destination / "assets/fixtures.tar").read_bytes() == b"sealed fixture archive"
    assert json.loads((approved / "bundle.json").read_text()) == manifest

    fixture.write_bytes(b"changed archive bytes!")
    with pytest.raises(ValueError, match="fixture archive"):
        stage_approved_inputs(approved, destination, "rocm")


def test_managed_overlay_pins_native_image_and_disables_startup_enable(tmp_path: Path) -> None:
    from scripts.ocr_source_up import managed_overlay

    image_id = "sha256:" + "d" * 64
    app_id = "sha256:" + "e" * 64
    result = managed_overlay(
        "cuda", tmp_path, image_id, app_id, "test", {"OCR_CUDA_DEVICE_ID": "0"}, "a" * 64
    )
    app = result["services"]["vision_bill"]
    native = result["services"]["paddleocr-service"]
    assert app["image"] == "vision-bill:source-" + "e" * 64
    assert "/api/v1/system/ui-config" in app["healthcheck"]["test"][-1]
    assert app["environment"] == {"OCR_ACTIVATION_ON_STARTUP": "false"}
    assert native["image"] == "vision-bill-ocr-cuda:source-" + "d" * 64
    assert native["pull_policy"] == "never"
    assert native["environment"]["OCR_IMAGE_ID"] == image_id
    assert native["volumes"] == [
        {
            "type": "bind",
            "source": str(tmp_path),
            "target": "/release",
            "read_only": True,
            "bind": {"create_host_path": False},
        }
    ]


@pytest.mark.parametrize(
    ("automatic", "requested_channel", "selected_channel"),
    [(False, None, "cu126"), (True, None, "cu129"), (True, "cu126", "cu126")],
)
def test_failed_candidate_cannot_create_restart_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    automatic: bool,
    requested_channel: str | None,
    selected_channel: str,
) -> None:
    from scripts import ocr_source_up

    approved = tmp_path / "approved"
    approved.mkdir()
    (approved / "bundle.json").write_text(
        json.dumps(
            {
                "accelerator": "cuda",
                "gpu_index": 0,
                "models": {"model": {"source": "https://example.org/model.tar"}},
                "fixtures": {"source": "https://example.org/fixtures.tar"},
            }
        )
    )
    state = tmp_path / "state"
    state.mkdir()
    managed = state / "compose.local.json"
    config = {
        "name": "test",
        "services": {
            "paddleocr-service": {
                "image": "vision-bill-ocr-cuda:candidate",
                "deploy": {"resources": {"reservations": {"devices": [{"device_ids": ["2"]}]}}},
            }
        },
    }
    image_config = b'{"architecture":"amd64"}'
    image_id = "sha256:" + hashlib.sha256(image_config).hexdigest()
    prepare_paths: list[Path] = []

    def docker_boundary(*args: str, capture: bool = False) -> str:
        if "config" in args:
            return json.dumps(config)
        if "ps" in args or args[1:3] == ("volume", "ls"):
            return ""
        if args[0] == "uv" and any(arg.endswith("/prepare_ocr_inputs.py") for arg in args):
            assert automatic
            assert {"pydantic==2.13.4", "packaging==26.2", "pillow==12.3.0"} <= set(args)
            assert args[args.index("--models") + 1].endswith("/build/paddle-model-snapshots.json")
            assert args[args.index("--fixtures") + 1].endswith("/first-run-fixtures/fixtures.json")
            output = Path(args[args.index("--output") + 1])
            assert output.parent.parent == state
            assert output.parent.stat().st_mode & 0o077 == 0
            prepare_paths.append(output.parent)
            prepared = subprocess.run(
                [sys.executable, *args[args.index("python") + 1 :]],
                cwd=SCRIPT.parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
            assert prepared.returncode == 0, prepared.stderr
            return prepared.stdout
        if "--check" in args:
            assert args[-2:] == ("--channel", selected_channel)
            return ""
        if args[0] == "make" or args[1:3] == ("image", "tag"):
            return ""
        if args[1:3] == ("image", "inspect"):
            return image_id
        if args[1:3] == ("image", "save"):
            with tarfile.open(args[4], "w") as output:
                records = {
                    "manifest.json": b'[{"Config":"config.json"}]',
                    "config.json": image_config,
                }
                for name, content in records.items():
                    entry = tarfile.TarInfo(name)
                    entry.size = len(content)
                    output.addfile(entry, io.BytesIO(content))
            return ""
        if args[1] == "run":
            assert args[2:7] == ("--rm", "--pull", "never", "--network", "none")
            assert args[7:9] == ("--gpus", "device=2")
            if automatic:
                release = Path(args[args.index("--volume") + 1].removesuffix(":/release:ro"))
                assert {path.name for path in release.iterdir()} == {
                    "bundle.json",
                    "fixtures.tar",
                    "image-config.json",
                }
                with tarfile.open(release / "fixtures.tar") as archive:
                    assert set(archive.getnames()) == {
                        "LICENSE.txt",
                        "upright.png",
                        "rotated.png",
                        "warped.png",
                        "table.png",
                        "formula.png",
                    }
                assert all(not path.exists() for path in prepare_paths)
            raise subprocess.CalledProcessError(1, args)
        pytest.fail(f"Unexpected subprocess command: {args}")

    for key in ("OCR_CUDA_DEVICE_ID", "OCR_CUDA_CHANNEL", "VB_IMAGE_TAG", "OCR_IMAGE_TAG"):
        monkeypatch.delenv(key, raising=False)
    if requested_channel is not None:
        monkeypatch.setenv("OCR_CUDA_CHANNEL", requested_channel)
    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    with pytest.raises(subprocess.CalledProcessError):
        ocr_source_up.deploy(("docker", "compose"), state, "cuda", None if automatic else approved)
    assert len(prepare_paths) == int(automatic)
    assert not managed.exists()
    assert not list(state.iterdir())


@pytest.mark.parametrize("altered_lock", [False, True])
def test_native_binding_uses_full_exact_package_closure(
    ocr_release: tuple[Path, dict[str, Any], Path],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    altered_lock: bool,
) -> None:
    from importlib import metadata
    from types import SimpleNamespace

    from ocr_service import bootstrap, device_inventory

    from scripts.ocr_source_up import BIND_BUNDLE

    path, manifest, image_config = ocr_release
    approved = manifest["expected_packages"]
    closure = {**approved, "numpy": "2.4.0", "pydantic": "2.12.0"}
    image_id = "sha256:" + hashlib.sha256(image_config.read_bytes()).hexdigest()
    lock = path.parent / "requirements.cuda.lock"
    lock.write_text("expected CUDA lock")
    resolver = path.parent / "resolve_cuda.py"
    resolver.write_text(
        "import sys; assert sys.argv[1:] == " + repr(["--check", str(lock), "--channel", "cu129"])
    )
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    if altered_lock:
        lock.write_text("wrong CUDA lock")
    monkeypatch.setattr(
        sys, "argv", ["-c", image_id, str(path.parent), "cu129", digest, str(lock), str(resolver)]
    )
    monkeypatch.setattr(
        metadata, "distributions", lambda: [SimpleNamespace(metadata={"Name": "paddlepaddle-gpu"})]
    )
    monkeypatch.setattr(bootstrap, "seal_dependencies", lambda name, **kwargs: (closure, {}))
    monkeypatch.setattr(device_inventory, "inspect_device_identity", lambda *args: {})
    binder = path.parent / "bind.py"
    binder.write_text(BIND_BUNDLE)
    if altered_lock:
        with pytest.raises(ValueError, match="Built CUDA lock differs"):
            runpy.run_path(str(binder))
        return
    runpy.run_path(str(binder))
    bound = json.loads(capsys.readouterr().out)
    assert bound["expected_packages"] == closure
    assert bound["image_identity"] == image_id
    assert bound["models"] == manifest["models"]


@pytest.fixture
def managed_release(tmp_path: Path) -> tuple[Path, str, str]:
    release = tmp_path / "release-validated"
    release.mkdir(mode=0o700)
    image_config = b'{"architecture":"amd64"}'
    (release / "image-config.json").write_bytes(image_config)
    manifest = json.dumps(
        {
            "accelerator": "cuda",
            "gpu_index": 0,
            "models": {"model": {"source": "https://example.org/model.tar"}},
            "fixtures": {"source": "https://example.org/fixtures.tar"},
        }
    ).encode()
    (release / "bundle.json").write_bytes(manifest)
    return (
        release,
        "sha256:" + hashlib.sha256(image_config).hexdigest(),
        hashlib.sha256(manifest).hexdigest(),
    )


@pytest.mark.parametrize("bundle_altered", [False, True])
def test_restart_checks_pinned_images_and_waits_on_original_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    managed_release: tuple[Path, str, str],
    bundle_altered: bool,
) -> None:
    from scripts import ocr_source_up

    release, native_id, bundle_digest = managed_release
    app_id = "sha256:" + "e" * 64
    overlay = ocr_source_up.managed_overlay(
        "cuda", release, native_id, app_id, "original", {"OCR_CUDA_DEVICE_ID": "2"}, bundle_digest
    )
    (tmp_path / "compose.local.json").write_text(json.dumps(overlay))
    calls = []

    def docker_boundary(*args: str, capture: bool = False) -> str:
        calls.append(args)
        if args[1:3] == ("image", "inspect"):
            return app_id if args[-1].startswith("vision-bill:") else native_id
        if "ps" in args or "up" in args or args[1:3] == ("volume", "ls"):
            return ""
        pytest.fail(f"Unexpected subprocess command: {args}")

    monkeypatch.setenv("OCR_CUDA_DEVICE_ID", "0")
    monkeypatch.setattr(ocr_source_up, "acquire_project_claim", lambda *args: None)
    monkeypatch.setattr(ocr_source_up, "has_project_claim", lambda *args: False)
    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    if bundle_altered:
        (release / "bundle.json").write_text("altered")
        with pytest.raises(ValueError, match="Prepared release file identity changed"):
            ocr_source_up.deploy(("docker", "compose"), tmp_path, "cuda", None, restart=True)
        assert not calls
        return
    ocr_source_up.deploy(("docker", "compose"), tmp_path, "cuda", None, restart=True)
    assert calls[-1] == (
        "docker",
        "compose",
        "--project-name",
        "original",
        "-f",
        str(tmp_path / "compose.local.json"),
        "up",
        "--detach",
        "--no-build",
        "--wait",
        "--wait-timeout",
        "300",
    )
    assert os.environ["OCR_CUDA_DEVICE_ID"] == "2"


def test_fixture_inputs_reject_writable_ancestor(tmp_path: Path) -> None:
    from scripts.ocr_source_up import stage_approved_inputs

    approved = tmp_path / "approved"
    approved.mkdir()
    approved.chmod(0o770)
    (approved / "bundle.json").write_text("{}")
    with pytest.raises(ValueError, match="trusted directory"):
        stage_approved_inputs(approved, tmp_path / "staged", "cuda")


def test_restart_rejects_a_different_container_image_in_its_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    managed_release: tuple[Path, str, str],
) -> None:
    from scripts import ocr_source_up

    release, native_id, bundle_digest = managed_release
    app_id = "sha256:" + "e" * 64
    overlay = ocr_source_up.managed_overlay(
        "cuda", release, native_id, app_id, "original", {"OCR_CUDA_DEVICE_ID": "2"}, bundle_digest
    )
    (tmp_path / "compose.local.json").write_text(json.dumps(overlay))

    def docker_boundary(*args: str, capture: bool = False) -> str:
        if args[1:3] == ("image", "inspect"):
            return app_id if args[-1].startswith("vision-bill:") else native_id
        if "ps" in args:
            return "active-ocr-id" if args[-1] == "paddleocr-service" else ""
        if args[1:3] == ("volume", "ls"):
            return ""
        if args[1:3] == ("container", "inspect"):
            return json.dumps([{"Image": "sha256:" + "a" * 64}])
        pytest.fail(f"Unexpected subprocess command: {args}")

    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    monkeypatch.setattr(ocr_source_up, "acquire_project_claim", lambda *args: None)
    monkeypatch.setattr(ocr_source_up, "has_project_claim", lambda *args: False)
    with pytest.raises(ValueError, match="different OCR image"):
        ocr_source_up.deploy(("docker", "compose"), tmp_path, "cuda", None, restart=True)


def test_prepare_rejects_the_other_accelerator_record_in_its_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import ocr_source_up

    prepared = tmp_path / "cuda"
    prepared.mkdir()
    overlay = {"x-vision-bill-source": {"project": "original"}}
    (prepared / "compose.local.json").write_text(json.dumps(overlay))
    state = tmp_path / "rocm"
    state.mkdir()

    def docker_boundary(*args: str, capture: bool = False) -> str:
        if "config" in args:
            return json.dumps({"name": "original"})
        pytest.fail(f"Unexpected subprocess command: {args}")

    for key in ("VB_IMAGE_TAG", "OCR_IMAGE_TAG"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    with pytest.raises(ValueError, match="another accelerator"):
        ocr_source_up.deploy(("docker", "compose"), state, "rocm", tmp_path / "approved")


def test_project_claim_cannot_be_relabelled_by_another_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import ocr_source_up

    volumes: dict[str, dict[str, Any]] = {}

    def docker_boundary(*args: str, capture: bool = False) -> str:
        if args[:3] == ("docker", "volume", "create"):
            key, value = args[args.index("--label") + 1].split("=", 1)
            volumes.setdefault(args[-1], {"Labels": {key: value}})
            return args[-1]
        if args[:3] == ("docker", "volume", "inspect"):
            return json.dumps([volumes[args[-1]]])
        pytest.fail(f"Unexpected Docker call: {args}")

    identity = {"project": "old", "bundle_sha256": "a" * 64, "image_ids": {}}
    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    monkeypatch.setattr(ocr_source_up, "ROOT", tmp_path / "first")
    ocr_source_up.acquire_project_claim(identity, "cuda", tmp_path / "first" / "release")
    ocr_source_up.acquire_project_claim(identity, "cuda", tmp_path / "first" / "release")
    monkeypatch.setattr(ocr_source_up, "ROOT", tmp_path / "second")
    with pytest.raises(ValueError, match="already claimed"):
        ocr_source_up.acquire_project_claim(identity, "rocm", tmp_path / "second" / "release")
    assert len(volumes) == 1


def test_existing_app_checks_actual_data_binds_even_with_matching_compose_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import ocr_source_up

    monkeypatch.setattr(ocr_source_up, "ROOT", tmp_path)
    targets = ["logs", "uploads", "uploads_tmp", "config"]
    mounts = [
        {
            "Type": "bind",
            "Source": str(tmp_path / "server_data" / name),
            "Destination": f"/app/{name}",
            "RW": True,
        }
        for name in targets
    ]
    app = {
        "Config": {"Labels": {"com.docker.compose.project.working_dir": str(tmp_path)}},
        "Mounts": mounts,
    }

    def docker_boundary(*args: str, capture: bool = False) -> str:
        if "ps" in args:
            return "old-app"
        if args[:3] == ("docker", "container", "inspect"):
            return json.dumps([app])
        if "config" in args:
            volumes = [
                {
                    "type": "bind",
                    "source": str(tmp_path / "server_data" / name),
                    "target": f"/app/{name}",
                }
                for name in targets
            ]
            return json.dumps({"services": {"vision_bill": {"volumes": volumes}}})
        pytest.fail(f"Unexpected Docker call: {args}")

    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    ocr_source_up.verify_existing_app(("docker", "compose"), "original")
    mounts[1]["Source"] = "/mnt/c/original/server_data/uploads"
    with pytest.raises(ValueError, match="data bind origins"):
        ocr_source_up.verify_existing_app(("docker", "compose"), "original")


@pytest.mark.parametrize("has_db_container", [False, True])
def test_orphan_database_requires_known_prior_ownership(
    monkeypatch: pytest.MonkeyPatch, has_db_container: bool
) -> None:
    from scripts import ocr_source_up

    def docker_boundary(*args: str, capture: bool = False) -> str:
        if "ps" in args:
            return "existing-db" if has_db_container and args[-1] == "db" else ""
        if args[:3] == ("docker", "container", "inspect"):
            return json.dumps(
                [
                    {
                        "Config": {
                            "Labels": {
                                "com.docker.compose.project.working_dir": str(ocr_source_up.ROOT)
                            }
                        }
                    }
                ]
            )
        if args[:3] == ("docker", "volume", "ls"):
            assert "label=com.docker.compose.project=original" in args
            assert "label=com.docker.compose.volume=db_data" in args
            return "original_db_data"
        pytest.fail(f"Unexpected Docker call: {args}")

    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    with pytest.raises(ValueError, match="existing database volume"):
        ocr_source_up.verify_existing_app(("docker", "compose"), "original")
    ocr_source_up.verify_existing_app(("docker", "compose"), "original", allow_orphan=True)


def test_wsl_bind_translation_uses_unstarted_probe_and_removes_only_its_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import ocr_source_up

    source = "/home/alex/vision-bill/server_data/config"
    observed = "/run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/known"
    container_id = "a" * 64
    calls: list[tuple[str, ...]] = []

    def docker_boundary(*args: str, capture: bool = False) -> str:
        calls.append(args)
        if args[:2] == ("docker", "create"):
            assert "--pull" in args and "never" in args
            assert "--network" in args and "none" in args
            assert f"type=bind,source={source},target=/app/config" in args
            return container_id
        if args[:3] == ("docker", "container", "inspect"):
            assert args[-1] == container_id
            return json.dumps(
                [
                    {
                        "Mounts": [
                            {
                                "Destination": "/app/config",
                                "Source": observed,
                                "Type": "bind",
                                "RW": True,
                            }
                        ]
                    }
                ]
            )
        if args[:3] == ("docker", "container", "rm"):
            assert args[3:] == (container_id,)
            return ""
        pytest.fail(f"Unexpected Docker call: {args}")

    monkeypatch.setattr(platform, "release", lambda: "6.6.0-microsoft-standard-WSL2")
    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    expected = {"/app/config": source}
    assert ocr_source_up.bind_sources_match(
        expected, {"/app/config": observed}, "sha256:" + "c" * 64
    )
    assert not ocr_source_up.bind_sources_match(
        expected, {"/app/config": observed + "other"}, "sha256:" + "c" * 64
    )
    assert len(calls) == 6


def test_reserved_claim_cannot_adopt_database_before_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, managed_release: tuple[Path, str, str]
) -> None:
    from scripts import ocr_source_up

    release, native_id, bundle_digest = managed_release
    app_id = "sha256:" + "e" * 64
    overlay = ocr_source_up.managed_overlay(
        "cuda", release, native_id, app_id, "original", {"OCR_CUDA_DEVICE_ID": "2"}, bundle_digest
    )
    (tmp_path / "compose.local.json").write_text(json.dumps(overlay))

    def docker_boundary(*args: str, capture: bool = False) -> str:
        if args[1:3] == ("image", "inspect"):
            return app_id if args[-1].startswith("vision-bill:") else native_id
        if "ps" in args:
            return ""
        if args[1:3] == ("volume", "ls"):
            return "original_db_data"
        pytest.fail(f"Unexpected Docker call: {args}")

    monkeypatch.setattr(ocr_source_up, "has_project_claim", lambda *args: True)
    monkeypatch.setattr(ocr_source_up, "acquire_project_claim", lambda *args: None)
    monkeypatch.setattr(ocr_source_up, "command", docker_boundary)
    with pytest.raises(ValueError, match="existing database volume"):
        ocr_source_up.deploy(("docker", "compose"), tmp_path, "cuda", None, restart=True)
