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


@pytest.mark.parametrize(
    ("saved_key", "runtime_key", "expected_key"),
    [("", None, "none"), ("saved-key", None, "saved-key"), ("", "runtime-key", "runtime-key")],
)
async def test_ollama_to_openai_update_constructs_fresh_provider(
    saved_settings, monkeypatch, saved_key, runtime_key, expected_key
):
    from vision_bill.provider.factory import get_llm_provider

    path, original, _, client = saved_settings
    original["llm"].update(api_key=saved_key, model_lifecycle="llamacpp")
    path.write_text(yaml.safe_dump(original))
    monkeypatch.setitem(config.Settings.model_config, "yaml_file", str(path))
    monkeypatch.delenv("LLM__API_KEY", raising=False)
    if runtime_key is not None:
        monkeypatch.setenv("LLM__API_KEY", runtime_key)
    before_restart = config.Settings()
    monkeypatch.setattr(system, "settings", before_restart)

    response = client.put(
        "/system/settings",
        json=payload(before_restart, provider="openai", host="http://router.example/v1"),
    )
    assert response.status_code == 200
    original["llm"].update(provider="openai", host="http://router.example/v1")
    assert yaml.safe_load(path.read_text()) == original
    after_restart = config.Settings()
    provider = get_llm_provider(after_restart.llm)
    try:
        assert provider._client.api_key == expected_key
        async with provider._lifecycle._client() as lifecycle:
            assert lifecycle.headers["Authorization"] == f"Bearer {expected_key}"
    finally:
        await provider._client.close()


@pytest.mark.parametrize("source", ["lowercase env", "JSON env", "JSON dotenv", "dotenv local"])
def test_actual_settings_sources_lock_overrides(saved_settings, monkeypatch, source):
    path, _, _, client = saved_settings
    monkeypatch.setitem(config.Settings.model_config, "yaml_file", str(path))
    if source == "lowercase env":
        monkeypatch.setenv("llm__host", "http://effective.example")
    elif source == "JSON env":
        monkeypatch.setenv("LLM", '{"HOST":"http://effective.example"}')
    elif source == "JSON dotenv":
        (path.parent / ".env").write_text('llm=\'{"HOST":"http://effective.example"}\'\n')
    else:
        (path.parent / ".env").write_text("LLM__HOST=http://overridden.example\n")
        (path.parent / ".env.local").write_text("export llm__host=http://effective.example\n")
    settings = config.Settings()
    monkeypatch.setattr(system, "settings", settings)
    before = path.read_bytes()

    response = client.get("/system/settings")
    assert response.status_code == 200
    assert response.json()["llm"]["host"] == "http://effective.example"
    assert response.json()["sources"]["llm.host"] == "environment"
    assert response.json()["sources"]["llm.model_name"] == "config"
    blocked = client.put("/system/settings", json=payload(settings, host="http://other"))
    assert blocked.status_code == 409
    assert path.read_bytes() == before


def test_settings_save_accepts_yaml_merge_with_explicit_override(saved_settings):
    path, original, settings, client = saved_settings
    inherited = {**original.pop("llm"), "model_name": "inherited"}
    path.write_text(
        yaml.safe_dump(original)
        + "llm_profile: &profile\n"
        + "".join(f"  {line}\n" for line in yaml.safe_dump(inherited).splitlines())
        + f"llm:\n  <<: *profile\n  model_name: {settings.llm.model_name!r}\n"
    )
    expected = yaml.safe_load(path.read_text())
    expected["llm"]["model_name"] = "new-selection"

    response = client.put("/system/settings", json=payload(settings, model_name="new-selection"))

    assert response.status_code == 200
    assert yaml.safe_load(path.read_text()) == expected


def test_first_start_does_not_overwrite_locked_winner(saved_settings, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from vision_bill.distribution import bootstrap

    path, original, settings, _ = saved_settings
    path.unlink()
    real_lock = bootstrap.config_lock
    at_lock = Event()

    def observe_lock(directory):
        at_lock.set()
        return real_lock(directory)

    monkeypatch.setattr(bootstrap, "config_lock", observe_lock)
    winner = yaml.safe_dump(original)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with real_lock(path.parent):
            loser = pool.submit(config.initialize_config_file, settings)
            assert at_lock.wait(timeout=2)
            bootstrap.write_private(path, winner)
        loser.result(timeout=2)

    preserved = path.read_text() == winner
    assert preserved


@pytest.mark.parametrize(
    ("saved_ocr", "status"),
    [
        ({"required": True}, 200),
        ({"required": True, "enabled": "invalid-boolean"}, 503),
        ({"timeout_seconds": 0}, 503),
    ],
)
def test_settings_save_preserves_partial_ocr_with_canonical_runtime_endpoint(
    saved_settings, monkeypatch, saved_ocr, status
):
    path, original, _, client = saved_settings
    original["ocr"] = saved_ocr
    path.write_text(yaml.safe_dump(original))
    monkeypatch.setitem(config.Settings.model_config, "yaml_file", str(path))
    monkeypatch.setenv("OCR__ENDPOINT", "http://ocr:8082")
    monkeypatch.setenv("OCR__TIMEOUT_SECONDS", "600")
    active = config.Settings()
    monkeypatch.setattr(system, "settings", active)
    assert active.ocr.enabled and active.ocr.timeout_seconds == 600
    before = path.read_bytes()

    response = client.put("/system/settings", json=payload(active, model_name="new-selection"))

    assert response.status_code == status
    if status == 200:
        original["llm"]["model_name"] = "new-selection"
        assert yaml.safe_load(path.read_text()) == original
    else:
        assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("persisted", "prepared", "status"),
    [
        ({"temperature": 0.3, "metadata": {"retain": True}}, False, 200),
        ({"temperature": 0.3, "host": ["invalid"]}, False, 503),
        ({"temperature": 0.3, "api_key": 42}, False, 503),
        ({"temperature": 0.3}, True, 503),
    ],
)
def test_edits_sparse_runtime_config_without_saving_environment_credentials(
    saved_settings, monkeypatch, persisted, prepared, status
):
    path, _, settings, client = saved_settings
    original = {"llm": persisted, "extension": {"keep": "private metadata"}}
    path.write_text(yaml.safe_dump(original))
    before = path.read_bytes()
    for field in ("provider", "host", "model_name", "api_key"):
        monkeypatch.setenv(f"LLM__{field.upper()}", str(settings.llm.model_dump(mode="json")[field]))
    monkeypatch.setenv("PG__PASSWORD", settings.pg.password)
    monkeypatch.setenv("AUTH__SECRET_KEY", settings.auth.secret_key)
    if prepared:
        (path.parent / ".bootstrap-complete.yaml").write_text("existing: true")

    response = client.put("/system/settings", json=payload(settings, temperature=0.61))

    assert response.status_code == status
    if status == 200:
        original["llm"]["temperature"] = 0.61
        assert yaml.safe_load(path.read_text()) == original
        assert response.json()["sources"]["llm.host"] == "environment"
        assert response.json()["sources"]["llm.temperature"] == "config"
    else:
        assert path.read_bytes() == before
