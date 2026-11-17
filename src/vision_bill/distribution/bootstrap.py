"""Create persistent installation secrets before application settings are imported."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import secrets
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL = "llama3.2-vision:11b"


class BootstrapError(ValueError):
    """Unsafe or incomplete persistent installation state."""


def private_directory(path: Path) -> None:
    """Create a private directory, refusing symlinks or another owner's state."""
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise BootstrapError("Secret directory must not traverse a symlink")
    if not path.parent.exists():
        private_directory(path.parent)
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise BootstrapError("Secret directory has an unexpected owner or type")
    path.chmod(0o700)


def private_file(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
        raise BootstrapError("Secret file has an unexpected owner or type")
    path.chmod(0o600)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_private(path: Path, content: str, *, replace: bool = False) -> None:
    """Publish fully written 0600 files atomically; never clobber by default."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            if path.exists() or path.is_symlink():
                private_file(path)
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_mapping(path: Path) -> dict[str, Any]:
    private_file(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BootstrapError("Persistent YAML must be a mapping")
    return value


def read_config(path: Path) -> dict[str, Any]:
    """Read existing credentials without importing config's settings singleton."""
    data = read_mapping(path)
    for section, names in {
        "llm": ("host", "model_name"),
        "pg": ("user", "password"),
        "auth": ("secret_key",),
    }.items():
        values = data.get(section)
        if not isinstance(values, dict) or any(
            not isinstance(values.get(name), str) or not values[name] for name in names
        ):
            raise BootstrapError("Persistent configuration is incomplete")
    llm = data["llm"]
    if (
        llm.get("provider", "ollama") not in ("ollama", "openai", "anthropic")
        or not isinstance(llm.get("api_key"), str)
        or not isinstance(llm.get("temperature"), (int, float))
    ):
        raise BootstrapError("Persistent LLM configuration is invalid")
    auth = data["auth"]
    username, password = auth.get("bootstrap_username"), auth.get("bootstrap_password")
    if bool(username) != bool(password) or (
        username and (not isinstance(username, str) or not isinstance(password, str))
    ):
        raise BootstrapError("Persistent bootstrap credentials are incomplete")
    return data


def digest_value(value: str) -> str:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise BootstrapError("Expected model digest must be a full sha256 digest")
    return value


def initialize(
    config_dir: Path,
    postgres_password_file: Path,
    *,
    pg_user: str = "vision_bill",
    pg_database: str = "vision_bill",
    pg_host: str = "db",
    provider: str = "ollama",
    llm_host: str = "http://ollama:11434",
    model: str = DEFAULT_MODEL,
    expected_model_digest: str | None = None,
) -> None:
    if expected_model_digest is not None:
        digest_value(expected_model_digest)
    private_directory(config_dir)
    private_directory(postgres_password_file.parent)
    config_path = config_dir / "config.yaml"
    admin_path = config_dir / "initial-admin.yaml"
    pending = config_dir / ".bootstrap-pending"
    complete = config_dir / ".bootstrap-complete.yaml"
    # Persistent flock protects independently started initializer containers.
    lock_path = config_dir / ".bootstrap.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "r+") as lock:
        private_file(lock_path)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if pending.exists() or pending.is_symlink():
            raise BootstrapError("Interrupted installation requires explicit recovery")
        existing = config_path.exists() or config_path.is_symlink()
        if not existing and any(
            p.exists() or p.is_symlink() for p in (admin_path, postgres_password_file, complete)
        ):
            raise BootstrapError("Incomplete installation requires explicit recovery")
        if existing:
            data = read_config(config_path)
        else:
            if not all((pg_user, pg_database, pg_host, llm_host, model)):
                raise BootstrapError("Installation identities must not be empty")
            data = {
                "llm": {
                    "provider": provider,
                    "host": llm_host,
                    "api_key": "none" if provider == "openai" else "",
                    "model_name": model,
                    "temperature": 0.0,
                },
                "pg": {
                    "user": pg_user,
                    "password": secrets.token_urlsafe(36),
                    "db": pg_database,
                    "host": pg_host,
                    "port": 5432,
                },
                "auth": {
                    "secret_key": secrets.token_urlsafe(48),
                    "bootstrap_username": "admin",
                    "bootstrap_password": secrets.token_urlsafe(36),
                    "allow_registration": True,
                    "admin_can_see_all": False,
                },
            }
            if expected_model_digest:
                data["distribution"] = {"expected_model_digest": expected_model_digest}
        auth = data["auth"]
        admin = (
            {"username": auth["bootstrap_username"], "password": auth["bootstrap_password"]}
            if auth.get("bootstrap_username")
            else None
        )
        has_password = postgres_password_file.exists() or postgres_password_file.is_symlink()
        if has_password:
            private_file(postgres_password_file)
            if postgres_password_file.read_text(encoding="utf-8") != data["pg"]["password"] + "\n":
                raise BootstrapError("PostgreSQL password file conflicts with configuration")
        has_admin = admin_path.exists() or admin_path.is_symlink()
        if has_admin and read_mapping(admin_path) != admin:
            raise BootstrapError("Initial admin file conflicts with configuration")
        if complete.exists() or complete.is_symlink():
            state = read_mapping(complete)
            if state != {"version": 1, "admin_credentials": admin is not None} or (
                not has_password or (admin is not None and not has_admin)
            ):
                raise BootstrapError("Completed installation is incomplete or conflicting")
            return
        write_private(pending, "Installation incomplete; do not reset secrets automatically.\n")
        if not has_password:
            write_private(postgres_password_file, data["pg"]["password"] + "\n")
        if admin is not None and not has_admin:
            write_private(admin_path, yaml.safe_dump(admin))
        if not existing:
            write_private(config_path, yaml.safe_dump(data, sort_keys=False))
        write_private(
            complete, yaml.safe_dump({"version": 1, "admin_credentials": admin is not None})
        )
        pending.unlink()
        sync_directory(config_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("/app/config"))
    parser.add_argument("--postgres-password-file", type=Path, required=True)
    parser.add_argument("--pg-user", default="vision_bill")
    parser.add_argument("--pg-database", default="vision_bill")
    parser.add_argument("--pg-host", default="db")
    parser.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--llm-host", default="http://ollama:11434")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--expected-model-digest")
    args = parser.parse_args()
    try:
        initialize(**vars(args))
    except (OSError, ValueError, yaml.YAMLError):
        print(
            "Installation initialization failed: unsafe, conflicting, or incomplete state.",
            file=sys.stderr,
        )
        return 1
    print("Private installation configuration is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
