"""Admin saves preserve private persisted state and reject unsupported changes."""

import stat

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vision_bill import config
from vision_bill.api.system import main as system
from vision_bill.security.dependencies import require_admin


@pytest.fixture
def saved_settings(settings, tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    data = settings.model_dump(mode="json", exclude={"pg": {"database_url", "pg_dsn"}})
    data["llm"].update(
        host="http://ollama:11434",
        api_key="persisted-api-secret",
        metadata={"future": [1, {"x": "y"}]},
    )
    data["pg"]["password"] = "persisted-db-secret"
    data["auth"].update(secret_key="persisted-auth-secret", pepper="persisted-pepper")
    data["distribution"] = {"expected_model_digest": "legacy-identity", "other": {"retain": True}}
    path.write_text(yaml.safe_dump(data))
    for name in config.EDITABLE_ENV_KEYS.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))
    settings.llm.host = data["llm"]["host"]
    # Effective private overrides must never overwrite unrelated persisted keys.
    settings.llm.api_key = "environment-api-secret"
    settings.pg.password = "environment-db-secret"
    settings.auth.secret_key = "environment-auth-secret"
    monkeypatch.setattr(system, "settings", settings)
    monkeypatch.setattr(config, "_startup_llm_identity", (settings.llm.provider, settings.llm.host))
    app = FastAPI()
    app.include_router(system.router, prefix="/system")
    app.dependency_overrides[require_admin] = lambda: None
    with TestClient(app) as client:
        yield path, data, settings, client


def payload(settings, **llm):
    return {
        "llm": {
            "provider": settings.llm.provider,
            "host": settings.llm.host,
            "model_name": settings.llm.model_name,
            "temperature": settings.llm.temperature,
            **llm,
        },
        "allow_registration": settings.auth.allow_registration,
    }


def test_settings_save_preserves_nested_unknown_metadata(saved_settings):
    path, original, settings, client = saved_settings
    response = client.put("/system/settings", json=payload(settings, model_name="new-selection"))
    assert response.status_code == 200
    saved = yaml.safe_load(path.read_text())
    original["llm"]["model_name"] = "new-selection"
    assert saved == original
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert response.json()["sources"]["llm.model_name"] == "config"


def test_unsupported_provider_update_is_atomic(saved_settings):
    path, _, settings, client = saved_settings
    before_update = path.read_bytes()
    previous_runtime = settings.model_dump()
    unsupported_update = client.put(
        "/system/settings", json=payload(settings, provider="anthropic")
    )
    assert unsupported_update.status_code == 422
    assert path.read_bytes() == before_update
    assert settings.model_dump() == previous_runtime


@pytest.mark.parametrize("field", ["host", "model_name"])
def test_blank_provider_identity_update_is_atomic(saved_settings, field):
    path, _, settings, client = saved_settings
    before = path.read_bytes()
    previous_runtime = settings.model_dump()
    response = client.put("/system/settings", json=payload(settings, **{field: " "}))
    assert response.status_code == 422
    assert path.read_bytes() == before
    assert settings.model_dump() == previous_runtime


@pytest.mark.parametrize(
    "content", ["auth: [broken]\n", "- not-a-mapping\n", "llm: [\n", "llm: {}\nllm: {}\n"]
)
def test_malformed_existing_document_is_not_replaced(saved_settings, content):
    path, _, settings, client = saved_settings
    path.write_text(content)
    previous_runtime = settings.model_dump()
    response = client.put("/system/settings", json=payload(settings, model_name="changed"))
    assert response.status_code == 503
    assert path.read_text() == content
    assert settings.model_dump() == previous_runtime


def test_environment_locked_setting_stays_effective_without_being_persisted(
    saved_settings, monkeypatch
):
    path, original, settings, client = saved_settings
    monkeypatch.setenv("LLM__HOST", "http://effective.example")
    settings.llm.host = "http://effective.example"
    result = client.put("/system/settings", json=payload(settings, model_name="new-selection"))
    assert result.status_code == 200
    saved = yaml.safe_load(path.read_text())
    assert saved["llm"]["host"] == original["llm"]["host"]
    assert result.json()["llm"]["host"] == "http://effective.example"
    assert result.json()["sources"]["llm.host"] == "environment"
    assert result.json()["restart_required"] is True
    before = path.read_bytes()
    assert (
        client.put("/system/settings", json=payload(settings, host="http://other")).status_code
        == 409
    )
    assert path.read_bytes() == before


def test_full_save_merges_unknown_nested_keys(saved_settings):
    path, original, settings, _ = saved_settings
    config.write_config_file(settings)
    saved = yaml.safe_load(path.read_text())
    assert saved["distribution"] == original["distribution"]
    assert saved["llm"]["metadata"] == {"future": [1, {"x": "y"}]}


@pytest.mark.parametrize(
    "artifact", [".bootstrap-pending", ".bootstrap-complete.yaml", "initial-admin.yaml"]
)
def test_settings_cannot_recreate_partial_bootstrap_state(saved_settings, artifact):
    path, _, settings, client = saved_settings
    path.unlink()
    (path.parent / artifact).write_text("retain-existing-state")
    response = client.put("/system/settings", json=payload(settings, model_name="changed"))
    assert response.status_code == 503
    assert not path.exists()
    assert (path.parent / artifact).read_text() == "retain-existing-state"


def test_settings_publication_uses_bootstrap_lock(saved_settings):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event

    from vision_bill.distribution.bootstrap import config_lock

    path, _, settings, _ = saved_settings
    before = path.read_bytes()
    started = Event()

    def save():
        started.set()
        config.write_config_file(settings, changed_paths=frozenset({"llm.model_name"}))

    with ThreadPoolExecutor(max_workers=1) as pool:
        with config_lock(path.parent):
            future = pool.submit(save)
            assert started.wait(timeout=2)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
            assert path.read_bytes() == before
        future.result(timeout=2)


def test_invalid_save_diagnostics_do_not_log_secret_lines(saved_settings, caplog):
    path, _, settings, client = saved_settings
    path.write_text("private-secret: [malformed-secret-value")
    response = client.put("/system/settings", json=payload(settings, model_name="changed"))
    assert response.status_code == 503
    assert "malformed-secret-value" not in caplog.text + response.text
