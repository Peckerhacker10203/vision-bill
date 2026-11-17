"""Initializer software evidence; the native process boundary is simulated."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_ocr_provision import NAMES, sha
from test_ocr_provision import release as _release

release = _release


@pytest.fixture
def native_pid1():
    if os.getpid() != 1:
        pytest.skip(
            "Native coordinator lifecycle is verified as PID 1 in the owned CPU test container"
        )


def external_qualifier(monkeypatch, *, success=True):
    """Replace only the external native executable, keeping coordinator behavior real."""
    from ocr_service import supervision

    real_popen = subprocess.Popen

    def spawn(command, **kwargs):
        assert command[1:3] == ["-m", "ocr_service.qualify"]
        runtime = Path(command[command.index("--runtime") + 1])
        cases = Path(command[command.index("--cases") + 1])
        output = Path(command[command.index("--output") + 1])
        script = """
import json,sys,pathlib
runtime,cases,output=map(pathlib.Path,sys.argv[1:4])
settings=json.loads(runtime.read_text())
success=sys.argv[4]=='yes'
manifest=pathlib.Path(settings['qualification_manifest'])
manifest.write_text(json.dumps({'image_identity':settings['image_identity'],'model_content':{},'qualified_target':{'name':'CPU-TEST-ONLY'}}))
report={'qualified':success,'stage':'complete' if success else 'native_collection','accelerator':settings['accelerator'],'image_identity':settings['image_identity'],'runtime_fingerprint':'cpu-test-only','profile_id':'cpu-test-only','manifest_sha256':__import__('hashlib').sha256(manifest.read_bytes()).hexdigest(),'cases':[{'kind':kind,'fresh_worker':repetition} for kind in ['upright','rotated','warped','table','formula'] for repetition in range(2)]}
output.write_text(json.dumps(report))
if success: output.with_suffix('.runtime.json').write_text(json.dumps(settings))
sys.exit(0 if success else 1)
"""
        return real_popen(
            [
                sys.executable,
                "-c",
                script,
                str(runtime),
                str(cases),
                str(output),
                "yes" if success else "no",
            ],
            **kwargs,
        )

    monkeypatch.setattr(supervision.subprocess, "Popen", spawn)


def initialize_release(tmp_path, release, **kwargs):
    from ocr_service.initialize import initialize

    bundle, _, config = release
    return initialize(
        bundle, tmp_path / "state", config, "sha256:" + sha(config.read_bytes()), **kwargs
    )


@pytest.mark.usefixtures("native_pid1")
def test_failed_native_process_cannot_publish_settings(tmp_path, release, monkeypatch):
    from ocr_service.initialize import InitializationFailed

    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    external_qualifier(monkeypatch, success=False)
    with pytest.raises(InitializationFailed):
        initialize_release(tmp_path, release)
    assert not (tmp_path / "state/selected").exists()


@pytest.mark.usefixtures("native_pid1")
def test_success_publishes_verified_settings_then_cached_start_never_spawns_or_rewrites(
    tmp_path, release, monkeypatch
):
    from ocr_service.initialization_lock import installation_lock

    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    external_qualifier(monkeypatch)
    result = initialize_release(tmp_path, release)
    selected = tmp_path / "state/selected"
    assert selected.resolve() == result
    runtime = json.loads((selected / "runtime.json").read_bytes())
    assert set(runtime["model_dirs"]) == set(NAMES)
    smoke = Path(json.loads((selected / "selection.json").read_bytes())["smoke_path"])
    assert smoke.read_bytes() == b"tiny-private-test-upright"
    before = {
        path: path.lstat().st_mtime_ns
        for path in (selected, selected / "runtime.json", selected / "complete.json")
    }

    def forbidden(*args, **kwargs):
        pytest.fail("Cached complete initialization must not run native work")

    from ocr_service import supervision

    monkeypatch.setattr(supervision.subprocess, "Popen", forbidden)
    with installation_lock(tmp_path / "installation.lock", exclusive=False):
        assert initialize_release(tmp_path, release) == result
    assert before == {path: path.lstat().st_mtime_ns for path in before}


@pytest.mark.usefixtures("native_pid1")
def test_live_runtime_prevents_first_and_forced_native_initialization(
    tmp_path, release, monkeypatch
):
    from ocr_service.initialization_lock import InstallationBusy, installation_lock

    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    with (
        installation_lock(tmp_path / "installation.lock", exclusive=False),
        pytest.raises(InstallationBusy),
    ):
        initialize_release(tmp_path, release)
    external_qualifier(monkeypatch)
    initialize_release(tmp_path, release)
    before = (tmp_path / "state/selected").readlink()
    with (
        installation_lock(tmp_path / "installation.lock", exclusive=False),
        pytest.raises(InstallationBusy),
    ):
        initialize_release(tmp_path, release, force=True)
    assert (tmp_path / "state/selected").readlink() == before


@pytest.mark.parametrize(
    "alter", ["runtime", "report", "model", "smoke", "manifest", "completion", "cases"]
)
@pytest.mark.usefixtures("native_pid1")
def test_cached_installation_integrity_is_fully_rechecked(tmp_path, release, monkeypatch, alter):
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    external_qualifier(monkeypatch)
    result = initialize_release(tmp_path, release)
    selected = tmp_path / "state/selected"
    settings = json.loads((selected / "runtime.json").read_bytes())
    paths = {
        "cases": Path(settings["model_dirs"][NAMES[0]]).parents[1] / "cases.toml",
        "runtime": result / "runtime.json",
        "report": result / "qualification.json",
        "manifest": Path(settings["qualification_manifest"]),
        "model": Path(settings["model_dirs"][NAMES[0]]) / "weights",
        "smoke": Path(json.loads((selected / "selection.json").read_bytes())["smoke_path"]),
        "completion": result / "complete.json",
    }
    paths[alter].write_bytes(b"tampered")
    with pytest.raises(ValueError):
        initialize_release(tmp_path, release)


@pytest.mark.parametrize("identity", ["id", "config", "architecture"])
def test_actual_image_configuration_identity_required_before_native(
    tmp_path, release, monkeypatch, identity
):
    from ocr_service.initialize import initialize

    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(tmp_path / "installation.lock"))
    bundle, data, config = release
    image_id = "sha256:" + sha(config.read_bytes())
    if identity == "id":
        image_id = "sha256:" + "0" * 64
    elif identity == "config":
        config.write_text("{}")
    else:
        config.write_text('{"architecture":"arm64","os":"linux"}')
        image_id = "sha256:" + sha(config.read_bytes())
        data["image_config_sha256"] = sha(config.read_bytes())
        bundle.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        initialize(bundle, tmp_path / "state", config, image_id)
    assert not (tmp_path / "state/selected").exists()


def test_entrypoint_preserves_explicit_manual_paths(monkeypatch):
    from ocr_service.entrypoint import configure_selected

    monkeypatch.setenv("OCR_RUNTIME_SETTINGS_PATH", "/manual/runtime.json")
    monkeypatch.setenv("OCR_QUALIFICATION_IMAGE_PATH", "/manual/smoke.png")
    configure_selected()
    assert os.environ["OCR_RUNTIME_SETTINGS_PATH"] == "/manual/runtime.json"
    assert os.environ["OCR_QUALIFICATION_IMAGE_PATH"] == "/manual/smoke.png"


def test_entrypoint_resolves_selected_paths_without_native_import(tmp_path, monkeypatch):
    from ocr_service.entrypoint import configure_selected

    root = tmp_path / "installations" / "private-test"
    root.mkdir(parents=True)
    (root / "complete.json").write_text("{}")
    smoke = tmp_path / "smoke.png"
    smoke.write_bytes(b"CPU-test-only")
    (root / "selection.json").write_text(json.dumps({"smoke_path": str(smoke)}))
    (tmp_path / "selected").symlink_to(root, target_is_directory=True)
    monkeypatch.setenv("OCR_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("OCR_RUNTIME_SETTINGS_PATH", raising=False)
    monkeypatch.delenv("OCR_QUALIFICATION_IMAGE_PATH", raising=False)
    configure_selected()
    assert os.environ["OCR_RUNTIME_SETTINGS_PATH"] == str(root / "runtime.json")
    assert os.environ["OCR_QUALIFICATION_IMAGE_PATH"] == str(smoke)


@pytest.mark.parametrize("role", ["initializer", "ocr", "ollama"])
@pytest.mark.parametrize("force", [False, True])
def test_uncleared_marker_blocks_actual_initialization_including_force(
    tmp_path, release, monkeypatch, role, force
):
    from ocr_service.supervision import UncleanInstallation

    lock = tmp_path / "installation.lock"
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(lock))
    marker = tmp_path / f"installation.active-{role}.json"
    marker.write_text('{"run_id":"hard-dead-test"}')
    before = marker.read_bytes()
    with pytest.raises(UncleanInstallation):
        initialize_release(tmp_path, release, force=force)
    assert marker.read_bytes() == before
    assert not (tmp_path / "state/selected").exists()


@pytest.mark.usefixtures("native_pid1")
def test_cpu_cache_read_survives_unclean_marker_without_clearing_it(tmp_path, release, monkeypatch):
    from ocr_service.supervision import UncleanInstallation

    lock = tmp_path / "installation.lock"
    monkeypatch.setenv("OCR_INSTALLATION_LOCK_PATH", str(lock))
    external_qualifier(monkeypatch)
    root = initialize_release(tmp_path, release)
    marker = tmp_path / "installation.active-initializer.json"
    marker.write_text('{"run_id":"unclean-test"}')
    before = marker.read_bytes(), marker.stat().st_mtime_ns
    assert initialize_release(tmp_path, release) == root
    assert before == (marker.read_bytes(), marker.stat().st_mtime_ns)
    with pytest.raises(UncleanInstallation):
        initialize_release(tmp_path, release, force=True)
    # A non-PID-1 process can still verify this complete cache without native work.
    monkeypatch.undo()
    bundle, _, config = release
    code = "import sys;from pathlib import Path;from ocr_service.initialize import initialize;print(initialize(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4]))"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(bundle),
            str(tmp_path / "state"),
            str(config),
            "sha256:" + sha(config.read_bytes()),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(root)
    assert before == (marker.read_bytes(), marker.stat().st_mtime_ns)
