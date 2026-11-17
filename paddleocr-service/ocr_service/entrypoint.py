"""Native serving entrypoint: hold the installation's shared lock for its lifetime."""

import json
import os
import sys
from pathlib import Path

from .supervision import ManagedInterrupted, lock_path, managed_installation, run_owned


def configure_selected() -> None:
    if os.environ.get("OCR_RUNTIME_SETTINGS_PATH") and os.environ.get(
        "OCR_QUALIFICATION_IMAGE_PATH"
    ):
        return  # Preserve the existing explicitly supplied manual qualifier route.
    selected = Path(os.environ.get("OCR_STATE_DIR", "/var/lib/ocr")) / "selected"
    root = selected.resolve(strict=True)
    if not (root / "complete.json").is_file():
        raise ValueError("Native initialization has not completed")
    selection = json.loads((root / "selection.json").read_bytes())
    os.environ.setdefault("OCR_RUNTIME_SETTINGS_PATH", str(root / "runtime.json"))
    os.environ.setdefault("OCR_QUALIFICATION_IMAGE_PATH", selection["smoke_path"])


def main() -> None:
    try:
        with managed_installation("ocr", lock_path()):
            configure_selected()
            command = sys.argv[1:]
            if command and command[0] == "--":
                command = command[1:]
            if not command:
                command = [sys.executable, str(Path(__file__).resolve().parents[1] / "server.py")]
            code = run_owned(command)
        raise SystemExit(code if code >= 0 else 128 - code)
    except ManagedInterrupted as exc:
        raise SystemExit(128 + exc.signum) from None


if __name__ == "__main__":
    main()
