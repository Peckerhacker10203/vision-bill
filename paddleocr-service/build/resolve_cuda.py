#!/usr/bin/env python3
"""Resolve the official cu126 wheel BEFORE build; never infer its hash or metadata.

Requires Python 3.12, uv and packaging (provided by the Make targets). Network failures leave the committed unresolved gate
intact. Output lock and attestation belong together and are verified by Docker.
"""

import argparse
import email
import hashlib
import html.parser
import json
import os
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

INDEX = "https://www.paddlepaddle.org.cn/packages/stable/cu126/paddlepaddle-gpu/"
HOSTS = {"www.paddlepaddle.org.cn", "paddle-whl.bj.bcebos.com"}
WHEEL = re.compile(
    r"paddlepaddle_gpu-3\.3\.1-cp312-cp312-(?:linux_x86_64|manylinux[^/]*x86_64)\.whl$"
)


class Links(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.urls.add(urllib.parse.urljoin(INDEX, value))


def select_wheel(document: str) -> str:
    links = Links()
    links.feed(document)
    candidates = []
    for url in links.urls:
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme == "https"
            and parsed.hostname in HOSTS
            and "/cu126/" in parsed.path
            and WHEEL.fullmatch(parsed.path.rsplit("/", 1)[-1])
        ):
            candidates.append(url.split("#", 1)[0])
    if len(set(candidates)) != 1:
        raise ValueError("Official index must identify exactly one cu126 cp312 x86_64 wheel")
    return candidates[0]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def executable_requirements(text: str) -> dict[str, tuple[Requirement, frozenset[str]]]:
    """Accept only closed, unconditional, hash-pinned uv requirements output."""
    entries: dict[str, tuple[Requirement, frozenset[str]]] = {}
    pending = ""
    for raw in text.splitlines():
        line = re.sub(r"(^|\s)#.*$", "", raw).strip()
        if not line:
            continue
        continued = line.endswith("\\")
        pending += " " + (line[:-1].rstrip() if continued else line)
        if continued:
            continue
        statement, pending = pending.strip(), ""
        hash_tokens = re.findall(r"(?:^|\s)--hash=sha256:([0-9a-f]{64})(?=\s|$)", statement)
        requirement_text = re.sub(r"\s+--hash=sha256:[0-9a-f]{64}(?=\s|$)", "", statement)
        if not hash_tokens or " --" in requirement_text or requirement_text.startswith("-"):
            raise ValueError("CUDA lock must contain only executable hash-pinned requirements")
        requirement = Requirement(requirement_text)
        name = str(canonicalize_name(requirement.name))
        if name in entries or requirement.marker or requirement.extras:
            raise ValueError("Duplicate, conditional or extra-bearing CUDA lock requirement")
        if not requirement.url:
            pins = list(requirement.specifier)
            if len(pins) != 1 or pins[0].operator != "==" or "*" in pins[0].version:
                raise ValueError("CUDA lock versions must be exact")
        entries[name] = (requirement, frozenset(hash_tokens))
    if pending or not entries:
        raise ValueError("CUDA executable lock is empty or incomplete")
    return entries


def validate_lock(lock: Path, runtime_lock: Path | None = None) -> dict[str, Any]:
    text = lock.read_text()
    attestation = lock.with_suffix(".attestation.json")
    if "UNRESOLVED" in text or not text.strip() or not attestation.is_file():
        raise ValueError("CUDA lock unresolved: run official-index resolver before building")
    data: dict[str, Any] = json.loads(attestation.read_text())
    fields = {
        "index",
        "index_sha256",
        "wheel_url",
        "wheel_sha256",
        "metadata_sha256",
        "metadata_name",
        "metadata_version",
        "requires_dist",
        "lock_sha256",
        "platform",
    }
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError("CUDA attestation schema is incomplete or unsupported")
    for key in ("index_sha256", "wheel_sha256", "metadata_sha256", "lock_sha256"):
        if not isinstance(data[key], str) or not re.fullmatch(r"[0-9a-f]{64}", data[key]):
            raise ValueError("CUDA attestation digest is invalid")
    if (
        not isinstance(data["requires_dist"], list)
        or not data["requires_dist"]
        or not all(isinstance(value, str) for value in data["requires_dist"])
        or not isinstance(data["wheel_url"], str)
    ):
        raise ValueError("CUDA attestation metadata is invalid")
    parsed = urllib.parse.urlsplit(data["wheel_url"])
    if (
        data["index"] != INDEX
        or parsed.scheme != "https"
        or parsed.hostname not in HOSTS
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
        or "/cu126/" not in parsed.path
        or not WHEEL.fullmatch(parsed.path.rsplit("/", 1)[-1])
        or data["lock_sha256"] != sha256(lock)
        or data["metadata_name"] != "paddlepaddle-gpu"
        or data["metadata_version"] != "3.3.1"
        or data["platform"] != "linux-x86_64-cp312"
    ):
        raise ValueError("CUDA lock or official input attestation mismatch")
    entries = executable_requirements(text)
    natives = {name for name in entries if name.startswith("paddlepaddle")}
    if natives != {"paddlepaddle-gpu"}:
        raise ValueError("Exactly one intended native Paddle requirement is required")
    native, hashes = entries["paddlepaddle-gpu"]
    if native.url != data["wheel_url"] or hashes != {data["wheel_sha256"]}:
        raise ValueError("Executable native requirement differs from attested official wheel")
    common_path = runtime_lock or lock.with_name("requirements.runtime.lock")
    common = executable_requirements(common_path.read_text())
    for name, (requirement, hashes) in common.items():
        if (
            name not in entries
            or str(entries[name][0]) != str(requirement)
            or entries[name][1] != hashes
        ):
            raise ValueError("CUDA lock omits or changes a pinned runtime input")
    # Closed common runtime inputs are preserved verbatim. Extra wheel dependencies
    # must also be unconditional exact/hash pins; the resolver closes their graph,
    # and pip check verifies installed transitive metadata after installation.
    for name, (requirement, _) in entries.items():
        if requirement.url and name not in common and name != "paddlepaddle-gpu":
            raise ValueError("Unexpected direct URL in CUDA lock")
    environment = {
        "python_version": "3.12",
        "python_full_version": "3.12.0",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "extra": "",
    }
    for specification in data["requires_dist"]:
        required = Requirement(specification)
        if required.marker and not required.marker.evaluate(environment):
            continue
        name = str(canonicalize_name(required.name))
        if name not in entries:
            raise ValueError("CUDA lock omits an active native wheel dependency")
        actual = entries[name][0]
        if required.url:
            if required.url != actual.url:
                raise ValueError("Native dependency URL differs from wheel metadata")
        elif actual.url or not required.specifier.contains(
            next(iter(actual.specifier)).version, prereleases=True
        ):
            raise ValueError("Native dependency pin violates wheel metadata")
    return data


def resolve(service: Path) -> None:
    os.environ["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PADDLEX"] = "3.7.2"
    os.environ["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PADDLEOCR"] = "3.7.0"
    with urllib.request.urlopen(INDEX, timeout=60) as response:
        document = response.read().decode()
    url = select_wheel(document)
    with tempfile.TemporaryDirectory(prefix="cuda-lock-") as scratch:
        directory = Path(scratch)
        wheel = directory / urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]
        with urllib.request.urlopen(url, timeout=600) as response, wheel.open("wb") as target:
            if urllib.parse.urlsplit(response.geturl()).hostname not in HOSTS:
                raise ValueError("Official wheel redirected to an unapproved host")
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        digest = sha256(wheel)
        with zipfile.ZipFile(wheel) as archive:
            paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(paths) != 1:
                raise ValueError("Wheel must have one distribution metadata record")
            metadata_bytes = archive.read(paths[0])
        metadata = email.message_from_bytes(metadata_bytes)
        if (
            metadata["Name"].lower().replace("_", "-") != "paddlepaddle-gpu"
            or metadata["Version"] != "3.3.1"
        ):
            raise ValueError("Official wheel metadata does not match requested build")
        # Use actual wheel METADATA to seed all active Requires-Dist requirements.
        # Native wheel is installed no-deps only after this complete lock is installed.
        incoming = directory / "cuda.in"
        incoming.write_text(
            (service / "build/runtime.in").read_text()
            + "\n"
            + "\n".join(metadata.get_all("Requires-Dist", []))
            + "\n"
        )
        output = directory / "requirements.cuda.lock"
        subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                str(incoming),
                "--python-version",
                "3.12",
                "--python-platform",
                "x86_64-manylinux_2_39",
                "--generate-hashes",
                "--constraint",
                str(service / "requirements.runtime.lock"),
                "--output-file",
                str(output),
            ],
            check=True,
        )
        with output.open("a") as stream:
            stream.write(f"\npaddlepaddle-gpu @ {url} \\\n    --hash=sha256:{digest}\n")
        data = {
            "index": INDEX,
            "index_sha256": hashlib.sha256(document.encode()).hexdigest(),
            "wheel_url": url,
            "wheel_sha256": digest,
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "metadata_name": "paddlepaddle-gpu",
            "metadata_version": "3.3.1",
            "requires_dist": metadata.get_all("Requires-Dist", []),
            "lock_sha256": sha256(output),
            "platform": "linux-x86_64-cp312",
        }
        attestation = output.with_suffix(".attestation.json")
        attestation.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
        validate_lock(output, service / "requirements.runtime.lock")
        (service / output.name).write_bytes(output.read_bytes())
        (service / attestation.name).write_bytes(attestation.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--service", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.check:
        validate_lock(args.check)
    else:
        resolve(args.service)
