import json
import os
from uuid import uuid4

import pytest
from ocr_contract.activation import (
    ActivationPhase,
    ActivationStatus,
    ActivationStore,
    MaintenanceIntent,
)

from scripts import ocr_stopped_recovery as recovery


@pytest.fixture
def stopped(tmp_path):
    coord = tmp_path / "coord"
    coord.mkdir(mode=0o700)
    store = ActivationStore(*(tmp_path / part for part in ("request", "status", "locks")))
    with store.controller(), store.admission(exclusive=True):
        request = store.submit_request(MaintenanceIntent.SEPARATE_OCR_DEVICE)
        store.publish_status(ActivationStatus(request.generation, ActivationPhase.READY))
        store.set_shared_gpu(False)
    for name in ("installation.lock", "gpu-handoff.lock"):
        (coord / name).touch(mode=0o600)
    journal = coord / "gpu-handoff.ocr-only.json"
    journal.touch(mode=0o600)
    saved = {
        "version": 2,
        "coordinator": "vision-bill-model-handoff-v2",
        "resources": {"llm": "external-unmanaged", "ocr": recovery.OCR_RESOURCE},
        "generation": 39,
        "unsettled": False,
        "models": [],
        "ocr_pending": True,
    }
    journal.write_text(json.dumps(saved, indent=4))
    return coord, store, journal, saved


def test_stopped_recovery_preserves_generation_bytes_and_locks(stopped):
    coord, store, journal, saved = stopped
    before = journal.read_bytes()
    inode = (coord / "gpu-handoff.lock").stat().st_ino
    backup = recovery.settle_stopped_owner(coord, store, lambda: None)
    assert backup.parent == coord and backup.read_bytes() == before
    assert backup.stat().st_mode & 0o777 == 0o600
    assert json.loads(journal.read_bytes()) == {**saved, "ocr_pending": False}
    assert (coord / "gpu-handoff.lock").stat().st_ino == inode
    assert not store.shared_gpu()


@pytest.mark.parametrize(
    "issue",
    [
        "other-ocr",
        "other-llm",
        "owned-model",
        "unsettled",
        "corrupt",
        "shared",
        "pending-peer",
        "foreign-peer",
        "marker",
        "direct",
        "unsafe-direct",
        "lock",
        "missing-lock",
        "unsafe-journal",
        "Docker owner",
        "status",
        "hardlink",
    ],
)
def test_stopped_recovery_refuses_ambiguous_state(stopped, issue):
    import fcntl

    coord, store, journal, saved = stopped
    if issue == "other-ocr":
        saved["resources"]["ocr"] = "paddleocr:foreign"
    if issue == "other-llm":
        saved["resources"]["llm"] = "external"
    if issue == "owned-model":
        saved["models"] = ["private-model"]
    if issue == "unsettled":
        saved["unsettled"] = True
    journal.write_text(json.dumps(saved))
    if issue == "corrupt":
        journal.write_text("{")
    if issue == "shared":
        store.set_shared_gpu(True)
    if issue in ("pending-peer", "foreign-peer"):
        peer = coord / "gpu-handoff.json"
        peer.touch(mode=0o600)
        peer.write_text(
            json.dumps(
                {
                    **saved,
                    "ocr_pending": issue == "pending-peer",
                    "resources": {
                        "llm": "ollama:abc",
                        "ocr": recovery.OCR_RESOURCE if issue == "pending-peer" else "foreign",
                    },
                }
            )
        )
    if issue == "marker":
        (coord / "installation.active-ocr.json").touch()
    if issue == "direct":
        (store.request_dir / "direct-calls").mkdir(mode=0o700)
        (store.request_dir / "direct-calls" / uuid4().hex).touch()
    if issue == "unsafe-direct":
        (store.request_dir / "direct-calls").mkdir(mode=0o755)
        (store.request_dir / "direct-calls").chmod(0o755)
    if issue == "missing-lock":
        (coord / "gpu-handoff.lock").unlink()
    if issue == "unsafe-journal":
        journal.chmod(0o644)
    if issue == "status":
        (store.status_dir / "status.json").unlink()
    if issue == "hardlink":
        os.link(journal, coord / "alias")
    before = journal.read_bytes()
    with (coord / "installation.lock").open("rb") as stream:
        if issue == "lock":
            fcntl.flock(stream, fcntl.LOCK_SH)

        def confirm():
            if issue == "Docker owner":
                raise ValueError("owner remains")

        with pytest.raises((ValueError, OSError, RuntimeError)):
            recovery.settle_stopped_owner(coord, store, confirm)
    assert journal.read_bytes() == before
    assert not list(coord.glob("*.recovered-*.json"))


def test_inspection_program_uses_only_transported_source(tmp_path):
    import subprocess
    import sys

    program = recovery.inspection_program(
        "from ocr_service.initialization_lock import OutstandingHandoff\nprint(OutstandingHandoff.__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "OutstandingHandoff"


def stock_config(project="sample"):
    res = {
        "services": {},
        "networks": {
            "default": {"name": project + "_default", "ipam": {}},
            "postgres_net": {"name": "postgres_net", "driver": "bridge"},
        },
        "volumes": {k: {"name": project + "_" + k} for k in recovery.MOUNTS},
    }
    for name in ("vision_bill", recovery.SERVING):
        res["services"][name] = {
            "volumes": [
                {"type": "volume", "source": k, "target": v}
                for k, v in recovery.MOUNTS.items()
                if name == recovery.SERVING or k not in ("ocr-cache", "ocr-state")
            ]
        }
    res["services"][recovery.SERVING].update(
        environment={
            "OCR_HANDOFF_LOCK_PATH": "/coord/gpu-handoff.lock",
            "OCR_INSTALLATION_LOCK_PATH": "/coord/installation.lock",
            "OCR_PORT": "8082",
            "OCR_STATE_DIR": "/var/lib/ocr",
        },
        command=["python", "-m", "ocr_service.activation"],
        networks={"default": {}},
    )
    res["services"]["vision_bill"].update(
        environment={
            "OCR_ACTIVATION_ENDPOINT": recovery.ENDPOINT,
            "VISION_BILL_OCR_ACTIVATION": "true",
            "OCR_ACTIVATION_HANDOFF_LOCK_PATH": "/coord/gpu-handoff.lock",
        },
        networks={"default": None, "postgres_net": None},
        extra_hosts=["host.docker.internal=host-gateway"],
    )
    return res


@pytest.mark.parametrize(
    "override",
    [
        "extra_hosts",
        "external-network",
        "alias",
        "host-mode",
        "port",
        "install-lock",
        "nonlocal",
        "redirect",
    ],
)
def test_recovery_validates_original_route(override):
    config = stock_config()
    assert set(recovery._volume_names(config, "sample")) == set(recovery.MOUNTS)
    services = config["services"]
    app, native = services["vision_bill"], services[recovery.SERVING]
    if override == "extra_hosts":
        app["extra_hosts"] = ["paddleocr-service=127.9.9.9"]
    if override == "external-network":
        config["networks"]["default"]["external"] = True
    if override == "alias":
        native["networks"]["default"] = {"aliases": ["other"]}
    if override == "host-mode":
        app["network_mode"] = "host"
    if override == "port":
        native["environment"]["OCR_PORT"] = "9999"
    if override == "install-lock":
        native["environment"]["OCR_INSTALLATION_LOCK_PATH"] = "/other/lock"
    if override == "nonlocal":
        app["environment"]["OCR_ACTIVATION_ENDPOINT"] = "http://external.invalid"
    if override == "redirect":
        app["volumes"].append({"type": "bind", "source": "/foreign/hosts", "target": "/etc/hosts"})
    with pytest.raises(ValueError):
        recovery._volume_names(config, "sample")


@pytest.mark.parametrize("foreign", ["network", "mount", "bind", "alias-volume"])
def test_recovery_checks_foreign_container_access(foreign):
    names = {"ocr-state": "sample_ocr-state"}
    roots = "/var/lib/docker/volumes/"
    records = {
        name: {"Name": name, "Mountpoint": roots + name + "/_data"} for name in names.values()
    }
    original = records["sample_ocr-state"]["Mountpoint"]
    mount = {"Type": "volume", "Name": "sample_ocr-state"}
    if foreign == "bind":
        mount = {"Type": "bind", "Source": original + "/models"}
    if foreign == "alias-volume":
        mount = {"Type": "volume", "Name": "foreign"}
        records["foreign"] = {
            "Name": "foreign",
            "Mountpoint": roots + "foreign/_data",
            "Driver": "local",
            "Options": {"o": "bind", "device": original},
        }

    def run(*args, capture):
        if args[1:3] == ("network", "ls"):
            return "sample_default" if foreign == "network" else ""
        if args[1] == "ps":
            return "helper\nstranger"
        if args[1:3] == ("volume", "inspect"):
            return json.dumps([records[n] for n in args[3:]])
        if args[1:3] == ("container", "inspect"):
            return json.dumps({"mounts": [mount], "labels": {}})
        pytest.fail(args)

    with pytest.raises(ValueError, match="network|volume"):
        recovery._verify_stopped(names, "sample", "helper", run)
