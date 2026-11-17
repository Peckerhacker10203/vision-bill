"""Provision Ollama model data without loading a model or running inference."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from .bootstrap import (
    BootstrapError,
    digest_value,
    private_directory,
    private_file,
    read_config,
    sync_directory,
    write_private,
)


def installed_digest(client: httpx.Client, host: str, model: str) -> str | None:
    response = client.get(f"{host}/api/tags", follow_redirects=False)
    response.raise_for_status()
    entries = response.json().get("models")
    if not isinstance(entries, list):
        raise BootstrapError("Invalid installed model metadata")
    identities = [
        entry.get("digest")
        for entry in entries
        if isinstance(entry, dict) and (entry.get("name") == model or entry.get("model") == model)
    ]
    if len(identities) > 1:
        raise BootstrapError("Ambiguous installed model metadata")
    if not identities:
        return None
    value = identities[0]
    if not isinstance(value, str):
        raise BootstrapError("Invalid installed model digest")
    if re.fullmatch(r"[0-9a-f]{64}", value):
        value = "sha256:" + value
    return digest_value(value)


def pull_data(client: httpx.Client, host: str, model: str) -> None:
    # Ollama's pull endpoint resumes partial server-owned layers. This endpoint
    # only transfers data; never use show/generate/chat to warm or qualify a model.
    success = False
    with client.stream(
        "POST", f"{host}/api/pull", json={"model": model, "stream": True}, follow_redirects=False
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if not isinstance(event, dict) or event.get("error"):
                raise BootstrapError("Model data pull failed")
            success = event.get("status") == "success"
    if not success:
        raise BootstrapError("Model data pull did not finish")


def provision(
    config_dir: Path,
    *,
    expected_digest: str | None = None,
    bundled_host: str = "http://ollama:11434",
    bundled_model: str | None = None,
    client: httpx.Client | None = None,
) -> bool:
    """Verify model data; skip external endpoints and explicitly nonbundled models."""
    private_directory(config_dir)
    marker = config_dir / ".model-provisioned.yaml"
    if marker.exists() or marker.is_symlink():
        private_file(marker)
        marker.unlink()
        sync_directory(config_dir)
    if (config_dir / ".bootstrap-pending").exists():
        raise BootstrapError("Installation is incomplete")
    data = read_config(config_dir / "config.yaml")
    llm = data["llm"]
    if llm.get("provider", "ollama") != "ollama":
        return False
    if httpx.URL(llm["host"].rstrip("/")) != httpx.URL(bundled_host.rstrip("/")):
        return False
    if bundled_model is not None:
        if not bundled_model or bundled_model.strip() != bundled_model:
            raise BootstrapError("Invalid bundled model name")
        if llm["model_name"] != bundled_model:
            return False
    distribution = data.get("distribution", {})
    if not isinstance(distribution, dict):
        raise BootstrapError("Invalid distribution configuration")
    configured = distribution.get("expected_model_digest")
    if expected_digest and configured and expected_digest != configured:
        raise BootstrapError("Expected digest conflicts with persistent configuration")
    digest = expected_digest or configured
    if not isinstance(digest, str):
        raise BootstrapError("An explicit verified expected model digest is required")
    digest_value(digest)
    model, host = llm["model_name"], llm["host"].rstrip("/")
    parsed_host = httpx.URL(host)
    if parsed_host.scheme not in ("http", "https") or not parsed_host.host or parsed_host.userinfo:
        raise BootstrapError("Ollama endpoint must be an HTTP(S) URL without credentials")
    if client is None:
        with httpx.Client(timeout=httpx.Timeout(120, connect=10), trust_env=False) as owned_client:
            return _provision(owned_client, host, model, digest, marker)
    return _provision(client, host, model, digest, marker)


def _provision(client: httpx.Client, host: str, model: str, digest: str, marker: Path) -> bool:
    try:
        if installed_digest(client, host, model) != digest:
            pull_data(client, host, model)
            if installed_digest(client, host, model) != digest:
                raise BootstrapError("Installed model digest does not match expected identity")
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
        raise BootstrapError("Model data provisioning or verification failed") from exc
    metadata: dict[str, Any] = {"model": model, "digest": digest, "host": host}
    write_private(marker, yaml.safe_dump(metadata), replace=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("/app/config"))
    parser.add_argument("--expected-digest")
    parser.add_argument("--bundled-host", default="http://ollama:11434")
    parser.add_argument("--bundled-model")
    args = parser.parse_args()
    try:
        local = provision(**vars(args))
    except (OSError, ValueError, httpx.InvalidURL, yaml.YAMLError):
        print("Model data provisioning failed; installation remains incomplete.", file=sys.stderr)
        return 1
    print(
        "Model data digest verified."
        if local else "Configured endpoint or model: bundled model data skipped."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
