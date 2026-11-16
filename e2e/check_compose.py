"""Validate deployment overlays without reading operator dotenv or starting services."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # Compose config needs no service credentials. Use a fresh private dotenv for
    # interpolation AND replace the application's service-level env_file list.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG"}
    }
    with tempfile.TemporaryDirectory(prefix="vb-compose-check-") as directory:
        private = Path(directory)
        dotenv = private / "test.env"
        dotenv.write_text(
            "PG__USER=compose_test\nPG__PASSWORD=unused_test_value\nPG__DB=compose_test\nAPI__PORT=58089\n"
        )
        override = private / "override.yaml"
        override.write_text(
            "services:\n  vision_bill:\n    env_file: !override\n      - "
            + json.dumps(str(dotenv))
            + "\n"
        )
        for accelerator in ("cuda", "rocm"):
            command = [
                "docker",
                "compose",
                "--env-file",
                str(dotenv),
                "--project-name",
                "vb-private-compose-check",
                "--project-directory",
                str(ROOT),
                "--profile",
                "ocr",
                "-f",
                str(ROOT / "docker-compose.yml"),
                "-f",
                str(ROOT / f"docker-compose.{accelerator}.yml"),
                "-f",
                str(override),
                "config",
                "--format",
                "json",
            ]
            result = subprocess.run(
                command, cwd=private, env=environment, capture_output=True, text=True, check=False
            )
            if result.returncode:
                # Never print expanded config/environment, including on failure.
                raise RuntimeError(
                    f"{accelerator} Compose configuration failed (exit {result.returncode})"
                )
            config = json.loads(result.stdout)
            services = config["services"]
            assert set(services) == {"vision_bill", "db", "paddleocr-service"}
            assert services["db"]["image"] == "postgres:18"
            assert services["db"]["environment"]["POSTGRES_USER"] == "compose_test"
            assert services["vision_bill"]["environment"]["PG__PASSWORD"] == "unused_test_value"
            ocr = services["paddleocr-service"]
            assert ocr["build"]["dockerfile"] == f"paddleocr-service/Dockerfile.{accelerator}"
            assert ocr["profiles"] == ["ocr"]
            assert not ocr.get("privileged", False)
            if accelerator == "cuda":
                device = ocr["deploy"]["resources"]["reservations"]["devices"][0]
                assert device["driver"] == "nvidia" and device["device_ids"] == ["0"]
            else:
                assert {device["source"] for device in ocr["devices"]} == {
                    "/dev/kfd",
                    "/dev/dri/renderD128",
                }
            print(f"main + {accelerator}: configuration valid (no services started)")


if __name__ == "__main__":
    main()
