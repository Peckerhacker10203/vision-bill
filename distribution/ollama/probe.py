"""Inspect the selected Ollama GPU package without model loading or inference."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def inspect_backend(accelerator: str, targets: list[str]) -> dict[str, object]:
    if os.geteuid() != 0:
        raise ValueError("Prepared Ollama image must run as root")
    if accelerator not in {"cuda", "rocm"} or (accelerator == "cuda" and targets):
        raise ValueError("Invalid accelerator target selection")
    backend = "libggml-" + ("cuda" if accelerator == "cuda" else "hip") + ".so"
    libraries: dict[str, str] = {}
    target_files: dict[str, str] = {}
    found: set[str] = set()
    for path in Path("/usr/lib/ollama").rglob("*.so*"):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        markers: set[str] = set()
        tail = b""
        with path.open("rb") as source:
            if source.read(4) != b"\x7fELF":
                continue
            source.seek(0)
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                data = tail + chunk
                markers.update(target for target in targets if target.encode() in data)
                tail = data[-32:]
        identity = digest.hexdigest()
        if path.name == backend or path.name.startswith(backend + "."):
            linked = subprocess.run(
                ["ldd", str(path)], capture_output=True, text=True, timeout=60, check=False
            )
            if linked.returncode or "not found" in linked.stdout + linked.stderr:
                raise ValueError("Selected GPU backend dependencies missing")
            libraries[str(path)] = identity
            if markers:
                found.update(markers)
                target_files[str(path)] = identity
    if not libraries or not set(targets) <= found:
        raise ValueError("Selected accelerator or AMD target absent from native package")
    return {
        "scope": "backend-package-only",
        "accelerator": accelerator,
        "backend_libraries": libraries,
        "dependencies_resolved": True,
        "target_markers": sorted(found),
        "target_files": target_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", choices=("cuda", "rocm"), required=True)
    parser.add_argument("--target", action="append", default=[])
    args = parser.parse_args()
    try:
        print(json.dumps(inspect_backend(args.accelerator, args.target), sort_keys=True))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        raise SystemExit("Ollama backend package verification failed") from None


if __name__ == "__main__":
    main()
