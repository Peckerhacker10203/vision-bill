"""CPU-only, bounded verification and immutable staging of release assets.

The release bundle is external to the image. No native framework is imported here.
Local sources are resolved relative to that bundle; remote sources require HTTPS.
"""

import ctypes
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import tomllib
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .initialization_lock import installation_lock
from .paddle_distribution import native_paddle_distribution

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
MODEL_NAMES = ("PP-DocLayoutV3", "PP-LCNet_x1_0_doc_ori", "UVDoc", "PaddleOCR-VL-1.6-0.9B")
KINDS = {"upright", "rotated", "warped", "table", "formula"}
SERVICE = Path(__file__).resolve().parents[1]


def file_digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def publish_directory(staged: Path, output: Path) -> None:
    # Linux renameat2 supports a no-replace directory rename. Path.rename could
    # replace an empty directory created between staging and publication.
    at_fdcwd, rename_noreplace = -100, 1
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        rename = libc.renameat2
    except AttributeError as exc:
        raise OSError("Atomic output publication requires Linux renameat2 support") from exc
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(at_fdcwd, os.fsencode(staged), at_fdcwd, os.fsencode(output), rename_noreplace) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), output)


def content_identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError("Asset paths must be canonical safe relative paths")
    return value


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class License(Record):
    spdx: str = Field(min_length=1)
    text_file: str = Field(min_length=1)
    redistribution_allowed: Literal[True]


class Provenance(Record):
    url: str = Field(pattern=r"^https://[^\s]+$")
    revision: str = Field(min_length=1)


class FileRecord(Record):
    sha256: Digest
    size_bytes: int = Field(ge=0, le=50 * 1024**3)


class Archive(Record):
    source: str = Field(min_length=1)
    sha256: Digest
    size_bytes: int = Field(gt=0, le=20 * 1024**3)
    unpacked_bytes: int = Field(gt=0, le=50 * 1024**3)
    files: dict[str, FileRecord] = Field(min_length=1, max_length=100_000)
    tree_sha256: Digest
    license: License
    provenance: Provenance

    @model_validator(mode="after")
    def inventory(self) -> Self:
        for name in self.files:
            safe_relative(name)
            if any(
                parent.as_posix() in self.files
                for parent in PurePosixPath(name).parents
                if parent.as_posix() != "."
            ):
                raise ValueError("File/directory collision in archive inventory")
        license_record = self.files.get(safe_relative(self.license.text_file))
        if license_record is None or not license_record.size_bytes:
            raise ValueError("License text must be included and sealed")
        if sum(row.size_bytes for row in self.files.values()) != self.unpacked_bytes:
            raise ValueError("Unpacked archive size disagrees with complete inventory")
        if (
            content_identity({name: row.sha256 for name, row in self.files.items()})
            != self.tree_sha256
        ):
            raise ValueError("Archive tree identity mismatch")
        scheme = urllib.parse.urlparse(self.source).scheme
        if scheme and scheme != "https":
            raise ValueError("Remote assets require HTTPS")
        return self


class Case(Record):
    kind: Literal["upright", "rotated", "warped", "table", "formula"]
    path: str
    sha256: Digest
    required_labels: list[str] = Field(default_factory=list)
    license: License
    provenance: Provenance


class Fixtures(Archive):
    cases: list[Case] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def case_inventory(self) -> Self:
        if {case.kind for case in self.cases} != KINDS or len(
            {case.sha256 for case in self.cases}
        ) != 5:
            raise ValueError("Exactly five distinct qualification kinds are required")
        for case in self.cases:
            record = self.files.get(safe_relative(case.path))
            license_record = self.files.get(safe_relative(case.license.text_file))
            if (
                not record
                or record.sha256 != case.sha256
                or not record.size_bytes
                or not license_record
                or not license_record.size_bytes
            ):
                raise ValueError("Fixture or license identity omitted from archive")
            if case.kind in {"table", "formula"} and case.kind not in case.required_labels:
                raise ValueError("Table/formula qualification labels are required")
        return self


class ReleaseBundle(Record):
    schema_version: Literal[1]
    accelerator: Literal["cuda", "rocm"]
    gpu_index: int = Field(ge=0)
    image_identity: str = Field(pattern=r"^(?:[^\s]+@)?sha256:[0-9a-f]{64}$")
    image_config_sha256: Digest
    profile_sha256: Digest
    source_files: dict[str, Digest] = Field(min_length=1)
    expected_paddle_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    expected_packages: dict[str, str] = Field(min_length=3)
    models: dict[str, Archive]
    fixtures: Fixtures
    _base: Path = PrivateAttr(default=Path("."))

    @model_validator(mode="after")
    def complete(self) -> Self:
        if set(self.models) != set(MODEL_NAMES):
            raise ValueError("Exactly the four approved model identities are required")
        if (
            self.expected_packages.get("paddleocr") != "3.7.0"
            or self.expected_packages.get("paddlex") != "3.7.2"
        ):
            raise ValueError("Pinned OCR/PaddleX package identities required")
        paddle = native_paddle_distribution(self.expected_packages, accelerator=self.accelerator)
        if self.expected_packages.get(paddle, "").split("+", 1)[0] != "3.3.1":
            raise ValueError("One pinned native Paddle identity required")
        return self

    @property
    def identity(self) -> str:
        return content_identity(self.model_dump(mode="json"))


def source_identities() -> dict[str, str]:
    paths = [
        path
        for directory in ("ocr_service", "ocr_contract")
        for path in (SERVICE / directory).rglob("*.py")
    ]
    paths.extend([SERVICE / "server.py", SERVICE / "profile.json"])
    records = {path.relative_to(SERVICE).as_posix(): file_digest(path) for path in paths}
    records["runtime-lock.json"] = file_digest(
        Path(os.environ.get("OCR_SOURCE_LOCK_PATH", str(SERVICE / "runtime-lock.json")))
    )
    return records


def verify_sources(bundle: ReleaseBundle) -> None:
    actual = source_identities()
    if bundle.source_files != actual or bundle.profile_sha256 != actual["profile.json"]:
        raise ValueError("Release profile/source locks differ from installed image sources")
    lock = Path(os.environ.get("OCR_SOURCE_LOCK_PATH", str(SERVICE / "runtime-lock.json")))
    if json.loads(lock.read_bytes())["sources"]["paddle"] != bundle.expected_paddle_commit:
        raise ValueError("Paddle source commit differs from installed source lock")


def load_release_bundle(path: Path) -> ReleaseBundle:
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError("Release manifest exceeds bounds")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate release manifest key")
            result[key] = value
        return result

    bundle = ReleaseBundle.model_validate(
        json.loads(path.read_bytes(), object_pairs_hook=unique_pairs)
    )
    bundle._base = path.resolve().parent
    verify_sources(bundle)
    return bundle


def verify_tree(root: Path, asset: Archive) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Staged asset root must be a real directory")
    found = {}
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Nonregular staged asset")
        if path.is_file():
            found[path.relative_to(root).as_posix()] = FileRecord(
                sha256=file_digest(path), size_bytes=path.stat().st_size
            )
    if found != asset.files:
        raise ValueError("Staged tree differs from complete archive inventory")


def fetch_archive(asset: Archive, base: Path, cache: Path) -> Path:
    target = cache / asset.sha256
    if target.exists():
        if (
            target.is_symlink()
            or target.stat().st_size != asset.size_bytes
            or file_digest(target) != asset.sha256
        ):
            raise ValueError("Cached archive identity mismatch")
        return target
    source_url = urllib.parse.urlparse(asset.source)
    source: Any = (
        urllib.request.urlopen(asset.source, timeout=60)
        if source_url.scheme
        else (base / asset.source).open("rb")
    )
    temporary = target.with_suffix(".partial")
    try:
        with source, temporary.open("wb") as output:
            if source_url.scheme and urllib.parse.urlparse(source.geturl()).scheme != "https":
                raise ValueError("Asset redirect must preserve HTTPS")
            remaining = asset.size_bytes
            while chunk := source.read(min(1024**2, remaining + 1)):
                remaining -= len(chunk)
                if remaining < 0:
                    raise ValueError("Archive exceeds declared bound")
                output.write(chunk)
        if remaining or file_digest(temporary) != asset.sha256:
            raise ValueError("Archive hash/size mismatch")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def extract_archive(path: Path, destination: Path, asset: Archive) -> None:
    seen: set[str] = set()
    total = 0
    directories = {
        parent.as_posix()
        for name in asset.files
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    }
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            name = safe_relative(member.name)
            if name in seen:
                raise ValueError("Archive contains duplicate entries")
            seen.add(name)
            if member.isdir() and name in directories and member.size == 0:
                (destination / name).mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or name not in asset.files:
                raise ValueError("Archive contains unsafe or unexpected entry")
            record = asset.files[name]
            total += member.size
            if member.size != record.size_bytes or total > asset.unpacked_bytes:
                raise ValueError("Archive expands beyond declared file bounds")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            incoming = archive.extractfile(member)
            if incoming is None:
                raise ValueError("Missing archive file")
            with incoming, target.open("xb") as output:
                shutil.copyfileobj(incoming, output, length=1024**2)
            if file_digest(target) != record.sha256:
                raise ValueError("Extracted file hash mismatch")
    verify_tree(destination, asset)


@dataclass(frozen=True)
class ProvisionedRuntime:
    root: Path
    model_dirs: dict[str, Path]
    cases_path: Path
    smoke_path: Path
    bundle_id: str


def provision_bundle(bundle: ReleaseBundle, state_dir: Path) -> ProvisionedRuntime:
    verify_sources(bundle)
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with installation_lock(state_dir / ".provision.lock", exclusive=True):
        cache, bundles = state_dir / "archives", state_dir / "bundles"
        cache.mkdir(exist_ok=True)
        bundles.mkdir(exist_ok=True)
        root = bundles / bundle.identity
        assets: dict[str, Archive] = {
            **{"models/" + name: asset for name, asset in bundle.models.items()},
            "fixtures": bundle.fixtures,
        }
        if root.exists():
            for name, asset in assets.items():
                verify_tree(root / name, asset)
        else:
            with tempfile.TemporaryDirectory(prefix=".staging-", dir=state_dir) as scratch:
                staging = Path(scratch)
                for name, asset in assets.items():
                    target = staging / name
                    target.mkdir(parents=True)
                    extract_archive(fetch_archive(asset, bundle._base, cache), target, asset)
                # Case paths remain relative to this generated TOML after atomic promotion.
                lines = []
                for case in bundle.fixtures.cases:
                    lines.extend(
                        [
                            "[[cases]]",
                            "kind = " + json.dumps(case.kind),
                            "path = " + json.dumps("fixtures/" + case.path),
                            "sha256 = " + json.dumps(case.sha256),
                            "required_labels = " + json.dumps(case.required_labels),
                            "",
                        ]
                    )
                (staging / "cases.toml").write_text("\n".join(lines))
                staging.replace(root)
        cases_path = root / "cases.toml"
        verify_cases(cases_path, bundle.fixtures)
        smoke = next(case for case in bundle.fixtures.cases if case.kind == "upright")
        return ProvisionedRuntime(
            root,
            {name: root / "models" / name for name in MODEL_NAMES},
            cases_path,
            root / "fixtures" / smoke.path,
            bundle.identity,
        )


def verify_cases(cases_path: Path, fixtures: Fixtures) -> None:
    if cases_path.is_symlink():
        raise ValueError("Qualification cases must be a regular owned file")
    raw = tomllib.loads(cases_path.read_text())
    expected = [
        {
            "kind": case.kind,
            "path": "fixtures/" + case.path,
            "sha256": case.sha256,
            "required_labels": case.required_labels,
        }
        for case in fixtures.cases
    ]
    if raw != {"cases": expected}:
        raise ValueError("Provisioned qualification cases changed")
