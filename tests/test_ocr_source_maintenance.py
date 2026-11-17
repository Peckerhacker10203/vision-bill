"""Explicit source qualification shares the serving fences before touching evidence."""

from pathlib import Path
from unittest.mock import Mock

import pytest


def arguments(tmp_path: Path) -> list[str]:
    return [
        "--accelerator", "cuda",
        "--runtime", str(tmp_path / "candidate.json"),
        "--cases", str(tmp_path / "cases.toml"),
        "--output", str(tmp_path / "qualification.json"),
    ]


@pytest.mark.parametrize("intent", [[], ["--maintenance-intent", "external-llm-quiesced"]])
def test_requires_intent_and_original_pairing_mode_before_qualification(
    tmp_path, monkeypatch, intent
):
    from ocr_service import source_maintenance

    launch = Mock()
    monkeypatch.setattr(source_maintenance, "run_owned", launch)
    with pytest.raises(SystemExit, match="2"):
        source_maintenance.main([*arguments(tmp_path), *intent])
    launch.assert_not_called()
    assert not (tmp_path / "qualification.json").exists()


@pytest.mark.parametrize("fence", ["running_ocr", "crashed_ocr", "outstanding_app", "running_app"])
def test_source_qualification_leaves_previous_outputs_untouched_while_fenced(
    tmp_path, monkeypatch, fence
):
    from contextlib import ExitStack

    from ocr_service import source_maintenance, supervision

    native = tmp_path / "installation.lock"
    app = tmp_path / "config" / "gpu-handoff.lock"
    output = tmp_path / "qualification.json"
    runtime = tmp_path / "qualification.runtime.json"
    output.write_bytes(b"previous report")
    runtime.write_bytes(b"previous verified runtime")
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(native))
    monkeypatch.setattr(source_maintenance, "APP_CONFIG", app.parent)
    launch = Mock()
    monkeypatch.setattr(source_maintenance, "run_owned", launch)
    with ExitStack() as held:
        if fence == "running_ocr":
            held.enter_context(supervision.installation_lock(native, exclusive=False))
        elif fence == "crashed_ocr":
            supervision.marker_path(native, "ocr").write_text("{}")
        elif fence == "running_app":
            held.enter_context(supervision.installation_lock(app, exclusive=True))
        else:
            app.parent.mkdir()
            journal = app.with_suffix(".json")
            journal.write_text("{}")
            journal.chmod(0o600)
        with pytest.raises(SystemExit, match="1"):
            source_maintenance.main([
                *arguments(tmp_path),
                "--maintenance-intent", "external-llm-quiesced",
                "--handoff-lock", str(app),
            ])
    launch.assert_not_called()
    assert output.read_bytes() == b"previous report"
    assert runtime.read_bytes() == b"previous verified runtime"


def test_source_qualifier_holds_both_original_locks_through_its_child(tmp_path, monkeypatch):
    from ocr_service import source_maintenance, supervision

    native = tmp_path / "installation.lock"
    app = tmp_path / "config" / "same-original-app.lock"
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(native))
    monkeypatch.setattr(source_maintenance, "APP_CONFIG", app.parent)
    monkeypatch.setattr(supervision.os, "getpid", lambda: 1)
    monkeypatch.setattr(supervision, "_descendants", lambda: set())

    def child(command):
        assert command[1:3] == ["-m", "ocr_service.qualify"]
        assert supervision.marker_path(native, "initializer").is_file()
        with pytest.raises(supervision.InstallationBusy), supervision.installation_lock(
            native, exclusive=False
        ):
            pass
        with pytest.raises(supervision.InstallationBusy), supervision.installation_lock(
            app, exclusive=True
        ):
            pass
        return 0

    launch = Mock(side_effect=child)
    monkeypatch.setattr(source_maintenance, "run_owned", launch)
    with pytest.raises(SystemExit, match="0"):
        source_maintenance.main([
            *arguments(tmp_path),
            "--maintenance-intent", "separate-ocr-device",
            "--handoff-lock", str(app),
        ])
    launch.assert_called_once()
    assert not supervision.marker_path(native, "initializer").exists()
    with (
        supervision.installation_lock(native, exclusive=True),
        supervision.installation_lock(app, exclusive=True),
    ):
        pass
