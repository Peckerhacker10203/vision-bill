#!/usr/bin/env python3
"""Build a local GPU OCR candidate and start source Compose with an idle controller."""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
NATIVE = "paddleocr-service"
RECORD_LIMIT = 32 * 1024**2
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
NATIVE_PLATFORM = "linux/amd64"
MANIFEST_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}

# Use the verifiers shipped in the exact built image. Neither provisioning nor
# qualification can run during this offline metadata-only invocation.
BIND_BUNDLE = """
import json, subprocess, sys
from pathlib import Path
from ocr_service.bootstrap import bind_approved_dependencies
from ocr_service.device_inventory import inspect_device_identity
from ocr_service.initialize import verify_image
from ocr_service.provision import ReleaseBundle, file_digest, source_identities, verify_sources

image_id, config_id, release, expected_sources, *verification = sys.argv[1:]
root = Path(release)
manifest = json.loads((root / 'bundle.json').read_bytes())
sources = source_identities()
if expected_sources and sources != json.loads(expected_sources):
    raise ValueError('Reused OCR image sources, profile or runtime lock differ from this checkout')
manifest.update(image_identity=image_id, image_config_sha256=config_id[7:],
                profile_sha256=sources['profile.json'], source_files=sources)
# Validate in the bind before emitting the final manifest, without touching assets.
bundle = ReleaseBundle.model_validate(manifest)
verify_sources(bundle)
verify_image(bundle, root / 'image-config.json', config_id)
if bundle.accelerator == 'cuda':
    channel, digest, lock, resolver = verification
    subprocess.run([sys.executable, resolver, '--check', lock, '--channel', channel], check=True)
    if file_digest(Path(lock)) != digest:
        raise ValueError('Built CUDA lock differs from the selected checkout lock')
manifest['expected_packages'] = bind_approved_dependencies(
    bundle.expected_packages, accelerator=bundle.accelerator,
)
bundle = ReleaseBundle.model_validate(manifest)
device = inspect_device_identity(bundle.accelerator, bundle.gpu_index)
if bundle.accelerator == 'rocm':
    from ocr_service.targets import validate_profile
    kernels = json.loads(Path('/opt/ocr/build-evidence/kernels.json').read_bytes())
    targets = validate_profile(kernels['target_profile'])['candidate_targets']
    if device['gfx_target'] not in targets:
        raise ValueError('Render device GFX target missing from OCR_AMDGPU_TARGETS')
print(json.dumps(bundle.model_dump(mode='json'), sort_keys=True, indent=2))
"""


def unique_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in items:
        if key in record:
            raise ValueError("Duplicate key in approved bundle")
        record[key] = value
    return record


def trusted_file(path: Path) -> os.stat_result:
    details = path.lstat()
    system_owner = Path("/").stat().st_uid
    trusted_owners = {system_owner, os.getuid()}
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_uid not in trusted_owners
        or details.st_mode & 0o022
        or path.resolve() != path
    ):
        raise ValueError(f"Approved input must be a trusted regular file: {path}")
    for parent in path.parents:
        owner = parent.lstat()
        shared_sticky = owner.st_uid == system_owner and owner.st_mode & stat.S_ISVTX
        if not stat.S_ISDIR(owner.st_mode) or (
            not shared_sticky and (owner.st_uid not in trusted_owners or owner.st_mode & 0o022)
        ):
            raise ValueError(f"Approved input parent must be a trusted directory: {parent}")
    return details


def https_asset(source: str) -> bool:
    url = urlsplit(source)
    if not url.scheme:
        return False
    if (
        not source.startswith("https://")
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or any(char.isspace() or char == "\\" for char in source)
        or any(
            re.search(r"secret|token|signature|password|api.?key", key, re.IGNORECASE)
            for key, _ in parse_qsl(url.query)
        )
    ):
        raise ValueError("Approved remote assets require an HTTPS URL without credentials")
    return True


def path_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkout_sources() -> dict[str, str]:
    root = ROOT / NATIVE
    paths = [
        path
        for directory in ("ocr_service", "ocr_contract")
        for path in (root / directory).rglob("*.py")
    ]
    paths.extend(root / name for name in ("server.py", "profile.json", "runtime-lock.json"))
    return {path.relative_to(root).as_posix(): path_digest(path) for path in paths}


def stage_approved_inputs(
    source: Path, destination: Path | None, accelerator: str
) -> dict[str, Any]:
    manifest_path = source / "bundle.json"
    if trusted_file(manifest_path).st_size > RECORD_LIMIT:
        raise ValueError("Approved bundle exceeds its size limit")
    manifest: dict[str, Any] = json.loads(manifest_path.read_bytes(), object_pairs_hook=unique_keys)
    if manifest["accelerator"] != accelerator or manifest["gpu_index"] != 0:
        raise ValueError(
            "Approved bundle must select this accelerator and container device index 0"
        )
    models = manifest["models"]
    if not isinstance(models, dict) or not all(
        https_asset(asset["source"]) for asset in models.values()
    ):
        raise ValueError("Browser activation requires approved HTTPS model archives")
    fixtures = manifest["fixtures"]
    fixture_source = fixtures["source"]
    if not https_asset(fixture_source):
        parts = PurePosixPath(fixture_source).parts
        if (
            not parts
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", fixture_source)
            or any(not part or part.startswith(".") for part in fixture_source.split("/"))
            or parts[0] in {"bundle.json", "image-config.json", "compose.local.json"}
        ):
            raise ValueError("Unsafe approved fixture archive path")
        archive = source / fixture_source
        size = fixtures["size_bytes"]
        if type(size) is not int or not 0 < size <= 20 * 1024**3:
            raise ValueError("Invalid approved fixture archive size")
        if trusted_file(archive).st_size != size:
            raise ValueError("Approved fixture archive size mismatch")
        staged = archive
        if destination is not None:
            staged = destination / fixture_source
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive, staged)
        if path_digest(staged) != fixtures["sha256"] or staged.stat().st_size != size:
            raise ValueError("Approved fixture archive identity mismatch")
    if destination is not None:
        (destination / "bundle.json").write_text(json.dumps(manifest))
    return manifest


def image_tag(accelerator: str, image_id: str) -> str:
    if not IMAGE_ID.fullmatch(image_id):
        raise ValueError("Invalid Docker image configuration ID")
    return f"vision-bill-ocr-{accelerator}:source-{image_id[7:]}"


def managed_overlay(
    accelerator: str,
    release: Path,
    image_id: str,
    app_id: str,
    project: str,
    device_environment: dict[str, str],
    bundle_digest: str,
    *,
    config_id: str,
) -> dict[str, Any]:
    if not IMAGE_ID.fullmatch(app_id):
        raise ValueError("Invalid application Docker image configuration ID")
    if not IMAGE_ID.fullmatch(config_id):
        raise ValueError("Invalid native Docker image configuration ID")
    return {
        "x-vision-bill-source": {
            "project": project,
            "device_environment": device_environment,
            "bundle_sha256": bundle_digest,
            "image_ids": {"vision_bill": app_id, NATIVE: image_id},
            "image_config_sha256": config_id[7:],
        },
        "services": {
            "vision_bill": {
                "image": f"vision-bill:source-{app_id[7:]}",
                "pull_policy": "never",
                "healthcheck": {
                    "test": [
                        "CMD",
                        "python",
                        "-c",
                        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/v1/system/ui-config', timeout=5)",
                    ],
                    "interval": "3s",
                    "timeout": "6s",
                    "retries": 10,
                    "start_period": "20s",
                },
                "environment": {"OCR_ACTIVATION_ON_STARTUP": "false"},
            },
            NATIVE: {
                "image": image_tag(accelerator, image_id),
                "platform": NATIVE_PLATFORM,
                "pull_policy": "never",
                "environment": {"OCR_IMAGE_ID": config_id},
                "volumes": [
                    {
                        "type": "bind",
                        "source": str(release),
                        "target": "/release",
                        "read_only": True,
                        "bind": {"create_host_path": False},
                    }
                ],
            },
        },
    }


def command(*args: str, capture: bool = False) -> str:
    return (
        subprocess.run(args, cwd=ROOT, check=True, capture_output=capture, text=True).stdout or ""
    )


def stage_first_run_inputs(release: Path, accelerator: str) -> None:
    with tempfile.TemporaryDirectory(prefix=".inputs-", dir=release.parent) as temporary:
        prepared = Path(temporary) / "approved"
        command(
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "--with",
            "pydantic==2.13.4",
            "--with",
            "packaging==26.2",
            "--with",
            "pillow==12.3.0",
            "python",
            f"{NATIVE}/build/prepare_ocr_inputs.py",
            accelerator,
            "--models",
            f"{NATIVE}/build/paddle-model-snapshots.json",
            "--fixtures",
            f"{NATIVE}/first-run-fixtures/fixtures.json",
            "--output",
            str(prepared),
            capture=True,
        )
        stage_approved_inputs(prepared, release, accelerator)


def build_history_blob(digest: str) -> bytes:
    # An empty build ref selects the builder's latest record. The explicit
    # digest addresses a blob on that node independently of the selected record.
    args = ("docker", "buildx", "history", "inspect", "attachment", "", digest)
    with subprocess.Popen(
        args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    ) as process:
        assert process.stdout is not None
        record = process.stdout.read(RECORD_LIMIT + 1)
        if len(record) > RECORD_LIMIT:
            process.kill()
            raise ValueError("Docker image metadata exceeds its size limit")
        if status := process.wait():
            raise subprocess.CalledProcessError(status, args)
    return record


def resolve_image_config(image_id: str, read_blob: Callable[[str], bytes]) -> tuple[str, bytes]:
    def read_verified(digest: str) -> tuple[dict[str, Any], bytes]:
        if not isinstance(digest, str) or not IMAGE_ID.fullmatch(digest):
            raise ValueError("Invalid Docker image metadata digest")
        raw = read_blob(digest)
        if not 0 < len(raw) <= RECORD_LIMIT or hashlib.sha256(raw).hexdigest() != digest[7:]:
            raise ValueError("Docker image metadata digest does not match the exact bytes")
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise ValueError("Invalid Docker image metadata record")
        return record, raw

    digest = image_id
    record, raw = read_verified(digest)
    if "manifests" in record:
        native = [
            entry["digest"]
            for entry in record["manifests"]
            if entry.get("mediaType") in MANIFEST_TYPES
            and entry.get("platform", {}).get("os") == "linux"
            and entry.get("platform", {}).get("architecture") == "amd64"
            and entry.get("annotations", {}).get("vnd.docker.reference.type") != "attestation-manifest"
        ]
        if record.get("schemaVersion") != 2 or len(native) != 1:
            raise ValueError("Docker image index must include one runnable linux/amd64 manifest")
        digest = native[0]
        record, raw = read_verified(digest)
        if record.get("schemaVersion") != 2 or "layers" not in record:
            raise ValueError("Invalid Docker platform image manifest")
    if record.get("schemaVersion") == 2 and "layers" in record:
        digest = record["config"]["digest"]
        record, raw = read_verified(digest)
    if (
        record.get("os") != "linux"
        or record.get("architecture") != "amd64"
        or record.get("rootfs", {}).get("type") != "layers"
        or not isinstance(record["rootfs"].get("diff_ids"), list)
    ):
        raise ValueError("Docker image configuration must describe a linux/amd64 root filesystem")
    return digest, raw


def export_config(release: Path, image_id: str) -> str:
    print("Verifying the native Docker image configuration…", flush=True)
    try:
        config_id, config = resolve_image_config(image_id, build_history_blob)
    except subprocess.CalledProcessError:
        print(
            "Buildx metadata is unavailable. Exporting the Docker image for verification; "
            "large images may take several minutes and require temporary disk space.",
            flush=True,
        )
        with tempfile.TemporaryDirectory(dir=release.parent) as temporary:
            archive = Path(temporary) / "image.tar"
            command("docker", "image", "save", "--output", str(archive), image_id)
            with tarfile.open(archive, "r:") as saved:

                def archive_blob(digest: str) -> bytes:
                    try:
                        return read_tar_record(saved, f"blobs/sha256/{digest[7:]}")
                    except KeyError:
                        if digest != image_id:
                            raise
                    # Older classic stores write the config at the path named in manifest.json.
                    entries = json.loads(read_tar_record(saved, "manifest.json"))
                    if len(entries) != 1:
                        raise ValueError("Docker export did not select one image")
                    return read_tar_record(saved, entries[0]["Config"])

                config_id, config = resolve_image_config(image_id, archive_blob)
    (release / "image-config.json").write_bytes(config)
    return config_id


def read_tar_record(archive: tarfile.TarFile, name: str) -> bytes:
    entry = archive.getmember(name)
    if not entry.isreg() or not 0 < entry.size <= RECORD_LIMIT:
        raise ValueError("Invalid Docker image export record")
    stream = archive.extractfile(entry)
    if stream is None:
        raise ValueError("Missing Docker image export record")
    with stream:
        return stream.read()


def need_host(accelerator: str, *, build: bool, automatic_inputs: bool) -> None:
    if sys.version_info < (3, 12):
        raise ValueError("Python 3.12 or later is required on the host")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Run this command on Linux x86_64 or in Windows WSL 2 x86_64 for CUDA")
    if accelerator == "rocm" and "microsoft" in platform.release().lower():
        raise ValueError("The ROCm source overlay requires native Linux")
    tools = ["docker"]
    if build and accelerator == "cuda":
        tools.append("make")
    if automatic_inputs or (build and accelerator == "cuda"):
        tools.append("uv")
    for tool in tools:
        if not shutil.which(tool):
            raise ValueError(f"Required executable is missing: {tool}")
    if not (ROOT / ".env").is_file():
        raise ValueError("Set up the stable configuration file .env before running this command")
    engine = command(
        "docker", "info", "--format", "{{.OSType}}/{{.Architecture}}", capture=True
    ).strip()
    if engine not in {"linux/x86_64", "linux/amd64"}:
        raise ValueError("Docker must use a Linux x86_64 engine")
    command("docker", "compose", "version", capture=True)
    if build:
        command("docker", "buildx", "version", capture=True)
    context = command(
        "docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}", capture=True
    ).strip()
    if not context.startswith("unix://") or not (
        os.environ.get("DOCKER_HOST") or "unix://"
    ).startswith("unix://"):
        raise ValueError("Use a local Docker engine with host bind mounts and GPU access")


def bootstrap(accelerator: str, inputs: Path | None, *, restart: bool = False) -> None:
    need_host(accelerator, build=not restart, automatic_inputs=not restart and inputs is None)
    deployment = ROOT / "server_data" / "ocr-source"
    state = deployment / accelerator
    for directory in (deployment, state):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        details = directory.lstat()
        if (
            directory.resolve() != directory
            or details.st_uid != os.getuid()
            or details.st_mode & 0o077
        ):
            raise ValueError("Source deployment state must be a private operator-owned directory")
    compose = (
        "docker",
        "compose",
        "--profile",
        "ocr",
        "-f",
        "docker-compose.yml",
        "-f",
        f"docker-compose.{accelerator}.yml",
    )
    lock = deployment / "deploy.lock"
    with lock.open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deploy(compose, state, accelerator, inputs, restart=restart)


def gpu_args(accelerator: str, native: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    if accelerator == "cuda":
        reservation = native["deploy"]["resources"]["reservations"]["devices"][0]
        ids = reservation["device_ids"]
        if len(ids) != 1 or not re.fullmatch(r"(?:[0-9]+|GPU-[0-9A-Fa-f-]+)", ids[0]):
            raise ValueError("Select one CUDA device by numeric index or GPU UUID")
        return ["--gpus", f"device={ids[0]}"], {"OCR_CUDA_DEVICE_ID": ids[0]}

    devices = {entry["target"]: entry["source"] for entry in native["devices"]}
    render = Path(devices["/dev/dri/renderD128"])
    kfd = Path(devices["/dev/kfd"])
    if not re.fullmatch(r"/dev/dri/renderD[0-9]+", str(render)) or kfd != Path("/dev/kfd"):
        raise ValueError("ROCm requires /dev/kfd and one selected DRI render node")
    arguments: list[str] = []
    for target, device in (("/dev/kfd", kfd), ("/dev/dri/renderD128", render)):
        details = device.stat()
        if not stat.S_ISCHR(details.st_mode):
            raise ValueError(f"ROCm device is not a character device: {device}")
        arguments.extend(("--device", f"{device}:{target}", "--group-add", str(details.st_gid)))
    return arguments, {
        "OCR_RENDER_NODE": str(render),
        "OCR_VIDEO_GID": str(kfd.stat().st_gid),
        "OCR_RENDER_GID": str(render.stat().st_gid),
    }


def reject_other_profile(state: Path, accelerator: str, project: str) -> None:
    other = "cuda" if accelerator == "rocm" else "rocm"
    record = state.parent / other / "compose.local.json"
    if record.exists():
        trusted_file(record)
        identity = json.loads(record.read_bytes())["x-vision-bill-source"]
        if identity["project"] == project:
            raise ValueError("This Compose project is already prepared for another accelerator")


def existing_controllers(compose: tuple[str, ...]) -> list[str]:
    return command(*compose, "ps", "--all", "--quiet", NATIVE, capture=True).split()


def project_claim(
    identity: dict[str, Any], accelerator: str, release: Path
) -> tuple[str, str, str]:
    project: str = identity["project"]
    name = "vision-bill-source-claim-" + hashlib.sha256(project.encode()).hexdigest()
    label = "io.vision-bill.source-owner"
    generation = {
        "checkout": str(ROOT),
        "accelerator": accelerator,
        "release": str(release),
        "bundle_sha256": identity["bundle_sha256"],
        "image_ids": identity["image_ids"],
    }
    owner = hashlib.sha256(json.dumps(generation, sort_keys=True).encode()).hexdigest()
    return name, label, owner


def has_project_claim(identity: dict[str, Any], accelerator: str, release: Path) -> bool:
    name, label, owner = project_claim(identity, accelerator, release)
    try:
        records = json.loads(command("docker", "volume", "inspect", name, capture=True))
    except subprocess.CalledProcessError:
        return False
    return len(records) == 1 and (records[0].get("Labels") or {}).get(label) == owner


def acquire_project_claim(identity: dict[str, Any], accelerator: str, release: Path) -> None:
    name, label, owner = project_claim(identity, accelerator, release)
    # Docker serializes creation by name and reuses an existing local volume
    # without replacing its creation labels, including during concurrent calls.
    command(
        "docker",
        "volume",
        "create",
        "--driver",
        "local",
        "--label",
        f"{label}={owner}",
        name,
        capture=True,
    )
    if not has_project_claim(identity, accelerator, release):
        raise ValueError(
            f"Compose project {identity['project']!r} is already claimed by another source deployment"
        )


def bind_sources(
    record: dict[str, Any], targets: dict[str, str], *, writable: bool
) -> dict[str, str]:
    return {
        mount["Destination"]: mount["Source"]
        for mount in record["Mounts"]
        if mount.get("Type") == "bind"
        and mount.get("RW") is writable
        and mount["Destination"] in targets
    }


def bind_sources_match(expected: dict[str, str], observed: dict[str, str], image: str) -> bool:
    if expected == observed:
        return True
    if (
        "microsoft" not in platform.release().lower()
        or expected.keys() != observed.keys()
        or not IMAGE_ID.fullmatch(image)
        or any(
            observed[target] != source
            and not observed[target].startswith(
                "/run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/"
            )
            for target, source in expected.items()
        )
    ):
        return False
    mounts: list[str] = []
    for target, source in expected.items():
        if any(char in source for char in ',"\n\r'):
            return False
        mounts.extend(("--mount", f"type=bind,source={source},target={target}"))
    # Docker Desktop translates WSL source paths at the daemon boundary. An
    # unstarted container proves the exact translation without running the image.
    probe = command(
        "docker",
        "create",
        "--pull",
        "never",
        "--network",
        "none",
        *mounts,
        "--entrypoint",
        "/bin/true",
        image,
        capture=True,
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", probe):
        raise ValueError("Cannot verify the Docker Desktop bind-probe container ID")
    try:
        records = json.loads(command("docker", "container", "inspect", probe, capture=True))
        return len(records) == 1 and bind_sources(records[0], expected, writable=True) == observed
    finally:
        command("docker", "container", "rm", probe, capture=True)


def compose_volume_exists(project: str, volume: str) -> bool:
    return bool(
        command(
            "docker",
            "volume",
            "ls",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--filter",
            f"label=com.docker.compose.volume={volume}",
            capture=True,
        ).strip()
    )


def verify_existing_app(
    compose: tuple[str, ...], project: str, *, allow_orphan: bool = False
) -> None:
    containers = command(*compose, "ps", "--all", "--quiet", "vision_bill", capture=True).split()
    if not containers:
        db = command(*compose, "ps", "--all", "--quiet", "db", capture=True).split()
        if not allow_orphan and (db or compose_volume_exists(project, "db_data")):
            raise ValueError(
                "Cannot verify ownership of the existing database volume. Start the base app in its original checkout or use a fresh project"
            )
        if db:
            records = json.loads(command("docker", "container", "inspect", *db, capture=True))
            if len(records) != 1 or (records[0]["Config"].get("Labels") or {}).get(
                "com.docker.compose.project.working_dir"
            ) != str(ROOT):
                raise ValueError(
                    "Cannot verify existing database checkout; keep its original deployment"
                )
        return
    records = json.loads(command("docker", "container", "inspect", *containers, capture=True))
    config = json.loads(
        command(*compose, "config", "--no-env-resolution", "--format", "json", capture=True)
    )
    configured = config["services"]["vision_bill"].get("volumes", [])
    expected = {
        f"/app/{name}": str(ROOT / "server_data" / name)
        for name in ("logs", "uploads", "uploads_tmp", "config")
    }
    current = {
        mount["target"]: mount["source"]
        for mount in configured
        if mount.get("type") == "bind"
        and not mount.get("read_only", False)
        and mount["target"] in expected
    }
    if (
        len(records) != 1
        or current != expected
        or any(Path(source).resolve() != Path(source) for source in expected.values())
    ):
        raise ValueError(
            "Cannot verify existing app checkout and data bind origins; keep its original deployment"
        )
    app = records[0]
    labels = app["Config"].get("Labels") or {}
    actual = bind_sources(app, expected, writable=True)
    if labels.get("com.docker.compose.project.working_dir") != str(ROOT) or not bind_sources_match(
        expected, actual, app.get("Image", "")
    ):
        raise ValueError(
            "Cannot verify existing app checkout and data bind origins; keep its original deployment"
        )


def reject_existing_ocr(compose: tuple[str, ...], project: str) -> None:
    if existing_controllers(compose) or compose_volume_exists(project, "ocr-activation-request"):
        raise ValueError("Existing OCR activation state must stay with its original deployment")


def verify_prepared_release(overlay: dict[str, Any], state: Path, accelerator: str) -> None:
    release = Path(overlay["services"][NATIVE]["volumes"][0]["source"])
    details = release.lstat()
    if not stat.S_ISDIR(details.st_mode) or release.parent != state or details.st_mode & 0o077:
        raise ValueError("Prepared release must be a private directory inside its deployment")
    identity = overlay["x-vision-bill-source"]
    for filename, digest in (
        ("bundle.json", identity["bundle_sha256"]),
        ("image-config.json", identity["image_ids"][NATIVE][7:]),
    ):
        path = release / filename
        if trusted_file(path).st_size > RECORD_LIMIT or path_digest(path) != digest:
            raise ValueError(f"Prepared release file identity changed: {filename}")
    stage_approved_inputs(release, None, accelerator)


def was_admitted(managed: Path) -> bool:
    marker = managed.with_name("admitted.sha256")
    if not marker.exists():
        return False
    return trusted_file(marker).st_size == 64 and marker.read_text() == path_digest(managed)


def record_admission(managed: Path) -> None:
    marker = managed.with_name("admitted.sha256")
    pending = marker.with_suffix(".pending")
    pending.write_text(path_digest(managed))
    pending.replace(marker)


def deploy(
    compose: tuple[str, ...],
    state: Path,
    accelerator: str,
    inputs: Path | None,
    *,
    restart: bool = False,
) -> None:
    managed = state / "compose.local.json"
    if restart:
        trusted_file(managed)
        overlay = json.loads(managed.read_bytes())
        identity = overlay["x-vision-bill-source"]
        verify_prepared_release(overlay, state, accelerator)
        for service, expected_id in identity["image_ids"].items():
            actual_id = inspect_image(overlay["services"][service]["image"])
            if actual_id != expected_id:
                raise ValueError(f"Managed source image changed for {service}")
        reject_other_profile(state, accelerator, identity["project"])
        os.environ.update(identity["device_environment"])
        selected = (*compose, "--project-name", identity["project"], "-f", str(managed))
        release = Path(overlay["services"][NATIVE]["volumes"][0]["source"])
        project = identity["project"]
        admitted = was_admitted(managed) and has_project_claim(identity, accelerator, release)
        verify_existing_app(selected, project, allow_orphan=admitted)
        acquire_project_claim(identity, accelerator, release)
        verify_existing_app(selected, project, allow_orphan=admitted)
        containers = existing_controllers(selected)
        if containers:
            records = json.loads(
                command("docker", "container", "inspect", *containers, capture=True)
            )
            if len(records) != 1 or records[0]["Image"] != identity["image_ids"][NATIVE]:
                raise ValueError("The original Compose project contains a different OCR image")
            expected = {"/release": str(release)}
            actual = bind_sources(records[0], expected, writable=False)
            if not bind_sources_match(expected, actual, identity["image_ids"][NATIVE]):
                raise ValueError(
                    "The original Compose project contains a different OCR release bind"
                )
        record_admission(managed)
        start(*selected)
        return
    if managed.exists():
        raise ValueError(
            "Source OCR is already prepared. Use --restart to retain its activation state"
        )
    build_tag = f"source-build-{uuid4().hex}"
    os.environ.update({"VB_IMAGE_TAG": build_tag, "OCR_IMAGE_TAG": build_tag})
    config = json.loads(
        command(*compose, "config", "--no-env-resolution", "--format", "json", capture=True)
    )
    project = config["name"]
    reject_other_profile(state, accelerator, project)
    compose = (*compose, "--project-name", project)
    reject_existing_ocr(compose, project)
    verify_existing_app(compose, project)
    candidate = config["services"][NATIVE]
    devices, device_environment = gpu_args(accelerator, candidate)
    os.environ.update(device_environment)
    release = Path(tempfile.mkdtemp(prefix="release-", dir=state))
    pending = release / "compose.local.json"
    try:
        if inputs is None:
            stage_first_run_inputs(release, accelerator)
        else:
            stage_approved_inputs(inputs, release, accelerator)
        cuda_verification: list[str] = []
        if accelerator == "cuda":
            lock = [
                "uv",
                "run",
                "--no-project",
                "--with",
                "packaging==26.2",
                "python",
                "paddleocr-service/build/resolve_cuda.py",
                "--check",
                "paddleocr-service/requirements.cuda.lock",
            ]
            channel = os.environ.get("OCR_CUDA_CHANNEL") or ("cu129" if inputs is None else "cu126")
            if channel not in {"cu126", "cu129"}:
                raise ValueError("OCR_CUDA_CHANNEL must be cu126 or cu129")
            os.environ["OCR_CUDA_CHANNEL"] = channel
            lock.extend(("--channel", channel))
            try:
                command(*lock, capture=True)
            except subprocess.CalledProcessError:
                command("make", "ocr-lock-cuda")
                command(*lock, capture=True)
            digest = path_digest(ROOT / "paddleocr-service" / "requirements.cuda.lock")
            cuda_verification = [
                channel,
                digest,
                "/opt/locks/requirements.cuda.lock",
                "/opt/ocr/build/resolve_cuda.py",
            ]
            command("make", "ocr-build-cuda")
        else:
            command(*compose, "build", NATIVE)
        image_id = inspect_image(candidate["image"])
        export_config(release, image_id)
        # The metadata probe has neither model cache/state mounts nor a network.
        bound = command(
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            *devices,
            "--volume",
            f"{release}:/release:ro",
            "--entrypoint",
            "python",
            image_id,
            "-c",
            BIND_BUNDLE,
            image_id,
            "/release",
            *cuda_verification,
            capture=True,
        )
        (release / "bundle.json").write_text(json.dumps(json.loads(bound), indent=2) + "\n")
        command("docker", "image", "tag", image_id, image_tag(accelerator, image_id))
        command(*compose, "build", "vision_bill")
        app_id = inspect_image(config["services"]["vision_bill"]["image"])
        overlay = managed_overlay(
            accelerator,
            release,
            image_id,
            app_id,
            project,
            device_environment,
            path_digest(release / "bundle.json"),
        )
        command("docker", "image", "tag", app_id, overlay["services"]["vision_bill"]["image"])
        pending.write_text(json.dumps(overlay))
        pending.replace(managed)
    except BaseException:
        if not managed.exists():
            shutil.rmtree(release)
        raise
    # A failed start can leave containers. Preserve the validated record so
    # restart can retry this exact image and the device selection.
    selected = (*compose, "-f", str(managed))
    acquire_project_claim(overlay["x-vision-bill-source"], accelerator, release)
    reject_existing_ocr(selected, project)
    verify_existing_app(selected, project)
    record_admission(managed)
    start(*selected)


def inspect_image(image: str) -> str:
    return command("docker", "image", "inspect", "--format", "{{.Id}}", image, capture=True).strip()


def start(*compose: str) -> None:
    command(*compose, "up", "--detach", "--no-build", "--wait", "--wait-timeout", "300")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accelerator", choices=("cuda", "rocm"))
    parser.add_argument(
        "approved_inputs",
        nargs="?",
        type=Path,
        metavar="APPROVED_DIRECTORY",
        help="use a custom approved directory in place of the repository snapshots and fixtures",
    )
    parser.add_argument(
        "--restart", action="store_true", help="start the last prepared source deployment"
    )
    args = parser.parse_args()
    if args.approved_inputs is not None and args.restart:
        parser.error("Choose either APPROVED_DIRECTORY or --restart")
    try:
        bootstrap(
            args.accelerator,
            Path(os.path.abspath(args.approved_inputs)) if args.approved_inputs else None,
            restart=args.restart,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"Source OCR preparation failed in {exc.cmd[0]} (exit {exc.returncode}).",
            file=sys.stderr,
        )
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        return 1
    except (OSError, ValueError, TypeError, KeyError, tarfile.TarError) as exc:
        print(f"Source OCR preparation failed: {exc}", file=sys.stderr)
        return 1
    print("Source stack started. Open Settings → Dedicated OCR to enable setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
