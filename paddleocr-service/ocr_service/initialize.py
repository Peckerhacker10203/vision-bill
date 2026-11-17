"""Owned, offline-verifying coordinator for the unchanged full native release gate.

The operator supplies the actual raw OCI image config and Docker inspect image ID;
no Docker socket or guessed self-image identity is used inside this container.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from .initialization_lock import lock_path
from .provision import (
    KINDS,
    ProvisionedRuntime,
    ReleaseBundle,
    file_digest,
    load_release_bundle,
    provision_bundle,
    safe_relative,
    source_identities,
    verify_cases,
    verify_sources,
    verify_tree,
)
from .supervision import ManagedInterrupted as InitializationInterrupted
from .supervision import managed_installation, run_owned


class InitializationFailed(RuntimeError):
    """Native qualification did not commit a complete release gate result."""


def owned_qualification_process(command: list[str]) -> int:
    return run_owned(command, quiet=True)


def verify_image(bundle: ReleaseBundle, config: Path, image_id: str) -> None:
    if config.stat().st_size > 32 * 1024**2:
        raise ValueError("Image configuration exceeds bounds")
    digest = file_digest(config)
    if image_id != "sha256:" + digest or digest != bundle.image_config_sha256:
        raise ValueError("Actual image config bytes/inspect ID differ from release identity")
    raw = json.loads(config.read_bytes())
    if raw.get("architecture") != "amd64" or raw.get("os") != "linux":
        raise ValueError("Only Linux x86_64 native images are supported")
    if raw.get("rootfs", {}).get("type") != "layers" or not isinstance(
        raw["rootfs"].get("diff_ids"), list
    ):
        raise ValueError("Actual OCI image configuration with rootfs identity required")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _asset_paths(bundle: ReleaseBundle, state: Path) -> ProvisionedRuntime:
    root = state / "bundles" / bundle.identity
    for name, asset in bundle.models.items():
        verify_tree(root / "models" / name, asset)
    verify_tree(root / "fixtures", bundle.fixtures)
    verify_cases(root / "cases.toml", bundle.fixtures)
    smoke = next(case for case in bundle.fixtures.cases if case.kind == "upright")
    return ProvisionedRuntime(
        root,
        {name: root / "models" / name for name in bundle.models},
        root / "cases.toml",
        root / "fixtures" / smoke.path,
        bundle.identity,
    )


def verify_qualification(root: Path, bundle: ReleaseBundle, assets: ProvisionedRuntime) -> None:
    report = json.loads((root / "qualification.json").read_bytes())
    runtime = json.loads((root / "runtime.json").read_bytes())
    manifest = root / "manifest.json"
    if (
        report.get("qualified") is not True
        or report.get("stage") != "complete"
        or report.get("image_identity") != bundle.image_identity
        or report.get("accelerator") != bundle.accelerator
    ):
        raise ValueError("Native release gate did not qualify the selected identity")
    if (
        not report.get("profile_id")
        or not report.get("runtime_fingerprint")
        or report.get("manifest_sha256") != file_digest(manifest)
    ):
        raise ValueError("Incomplete native qualification publication")
    cases = report.get("cases", [])
    if len(cases) != 10 or {(row.get("kind"), row.get("fresh_worker")) for row in cases} != {
        (kind, repetition) for kind in KINDS for repetition in range(2)
    }:
        raise ValueError("Full five-case/two-fresh-worker release gate is required")
    expected = {
        "accelerator": bundle.accelerator,
        "gpu_index": bundle.gpu_index,
        "model_dirs": {name: str(path) for name, path in assets.model_dirs.items()},
        "cache_dir": str(root / "cache"),
        "qualification_manifest": str(manifest),
        "expected_packages": bundle.expected_packages,
        "expected_paddle_commit": bundle.expected_paddle_commit,
        "image_identity": bundle.image_identity,
    }
    if (
        runtime != expected
        or json.loads((root / "qualification.runtime.json").read_bytes()) != expected
    ):
        raise ValueError("Published native runtime differs from selected release settings")
    selection = json.loads((root / "selection.json").read_bytes())
    if selection != {"bundle_id": bundle.identity, "smoke_path": str(assets.smoke_path)}:
        raise ValueError("Selected smoke fixture differs from qualified release")


def cached_installation(state: Path, bundle: ReleaseBundle, image_id: str) -> Path | None:
    selected = state / "selected"
    if not selected.is_symlink():
        if selected.exists():
            raise ValueError("Selected installation must be an atomic pointer")
        return None
    root = selected.resolve(strict=True)
    if root.parent != (state / "installations").resolve():
        raise ValueError("Selected installation escapes owned state")
    completion = json.loads((root / "complete.json").read_bytes())
    if completion.get("bundle_id") != bundle.identity or completion.get("image_id") != image_id:
        return None
    files = completion.get("files", {})
    required = {
        "bundle.json",
        "candidate.runtime.json",
        "runtime.json",
        "qualification.json",
        "qualification.runtime.json",
        "manifest.json",
        "selection.json",
    }
    if set(files) != required:
        raise ValueError("Selected completion inventory is incomplete")
    for name, digest in files.items():
        path = root / safe_relative(name)
        if path.is_symlink() or not path.is_file() or file_digest(path) != digest:
            raise ValueError("Selected installation content changed")
    if json.loads((root / "bundle.json").read_bytes()) != bundle.model_dump(mode="json"):
        raise ValueError("Selected bundle identity changed")
    assets = _asset_paths(bundle, state)
    verify_qualification(root, bundle, assets)
    return root


def initialize(
    bundle_path: Path, state_dir: Path, image_config: Path, image_id: str, *, force: bool = False
) -> Path:
    bundle = load_release_bundle(bundle_path)
    verify_image(bundle, image_config, image_id)
    state = state_dir.resolve()
    if not force and (cached := cached_installation(state, bundle, image_id)) is not None:
        return cached  # CPU-verified success: no native work or selected-state writes.
    with managed_installation("initializer", lock_path()):
        # Another initializer may have finished before this exclusive acquisition.
        if not force and (cached := cached_installation(state, bundle, image_id)) is not None:
            return cached
        assets = provision_bundle(bundle, state)
        root = state / "installations" / uuid4().hex
        root.mkdir(parents=True, mode=0o700)
        try:
            runtime = {
                "accelerator": bundle.accelerator,
                "gpu_index": bundle.gpu_index,
                "model_dirs": {name: str(path) for name, path in assets.model_dirs.items()},
                "cache_dir": str(root / "cache"),
                "qualification_manifest": str(root / "manifest.json"),
                "expected_packages": bundle.expected_packages,
                "expected_paddle_commit": bundle.expected_paddle_commit,
                "image_identity": bundle.image_identity,
            }
            write_json(root / "bundle.json", bundle.model_dump(mode="json"))
            write_json(root / "candidate.runtime.json", runtime)
            output = root / "qualification.json"
            command = [
                sys.executable,
                "-m",
                "ocr_service.qualify",
                "--accelerator",
                bundle.accelerator,
                "--runtime",
                str(root / "candidate.runtime.json"),
                "--cases",
                str(assets.cases_path),
                "--output",
                str(output),
            ]
            if owned_qualification_process(command):
                raise InitializationFailed(
                    "Full native qualification failed; selected settings were not published"
                )
            shutil.copyfile(root / "qualification.runtime.json", root / "runtime.json")
            write_json(
                root / "selection.json",
                {"bundle_id": bundle.identity, "smoke_path": str(assets.smoke_path)},
            )
            verify_sources(bundle)
            _asset_paths(bundle, state)
            verify_qualification(root, bundle, assets)
            files = {path.name: file_digest(path) for path in root.iterdir() if path.is_file()}
            write_json(
                root / "complete.json",
                {"bundle_id": bundle.identity, "image_id": image_id, "files": files},
            )
            temporary = state / (".selected-" + uuid4().hex)
            try:
                temporary.symlink_to(root.relative_to(state), target_is_directory=True)
                temporary.replace(state / "selected")
            finally:
                temporary.unlink(missing_ok=True)
            return root
        except BaseException:
            shutil.rmtree(root)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument(
        "--state-dir", type=Path, default=Path(os.environ.get("OCR_STATE_DIR", "/var/lib/ocr"))
    )
    parser.add_argument("--image-config", type=Path)
    parser.add_argument("--image-id")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Requalify only after both managed serving processes have stopped",
    )
    parser.add_argument("--print-source-identities", action="store_true")
    args = parser.parse_args()
    if args.print_source_identities:
        print(json.dumps(source_identities(), sort_keys=True, indent=2))
        return
    if not args.bundle or not args.image_config or not args.image_id:
        parser.error("--bundle, --image-config and actual --image-id are required")
    try:
        initialize(args.bundle, args.state_dir, args.image_config, args.image_id, force=args.force)
    except InitializationInterrupted:
        raise SystemExit(143) from None


if __name__ == "__main__":
    main()
