"""Provisioning only transfers model data and checks installed metadata."""

import importlib
import json

import httpx
import pytest
import yaml

DIGEST = "sha256:" + "1" * 64
OTHER = "sha256:" + "2" * 64
MODEL = "llama3.2-vision:11b"


def config(tmp_path, *, provider="ollama", digest=DIGEST, model=MODEL):
    data = {
        "llm": {
            "provider": provider,
            "host": "http://ollama:11434",
            "api_key": "",
            "model_name": model,
            "temperature": 0,
        },
        "auth": {"secret_key": "synthetic-auth-secret"},
        "pg": {"user": "synthetic-user", "password": "synthetic-password"},
    }
    if digest:
        data["distribution"] = {"expected_model_digest": digest}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data))


def module():
    try:
        return importlib.import_module("vision_bill.distribution.models")
    except ModuleNotFoundError:
        pytest.fail("Data-only model provisioning module is missing")


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("wire_digest", ["1" * 64, DIGEST])
@pytest.mark.parametrize("bundled_model", [None, MODEL])
def test_verified_model_transfers_data_only_and_cached_metadata_is_fast(
    tmp_path, cached, wire_digest, bundled_model
):
    config(tmp_path)
    requests = []
    pulled = False

    def server(request):
        nonlocal pulled
        requests.append((request.method, request.url.path))
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": ([{"name": MODEL, "digest": wire_digest}] if cached or pulled else [])
                },
            )
        assert request.url.path == "/api/pull"
        assert json.loads(request.content) == {"model": MODEL, "stream": True}
        pulled = True
        return httpx.Response(200, text='{"status":"pulling manifest"}\n{"status":"success"}\n')

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        assert module().provision(tmp_path, client=client, bundled_model=bundled_model) is True
    assert requests == (
        [("GET", "/api/tags")]
        if cached
        else [("GET", "/api/tags"), ("POST", "/api/pull"), ("GET", "/api/tags")]
    )
    marker = yaml.safe_load((tmp_path / ".model-provisioned.yaml").read_text())
    assert marker == {"model": MODEL, "digest": DIGEST, "host": "http://ollama:11434"}


@pytest.mark.parametrize("failure", ["http", "stream", "truncated", "digest"])
def test_failed_pull_or_digest_mismatch_stays_incomplete_and_can_retry(tmp_path, failure):
    config(tmp_path)
    calls = []

    def server(request):
        calls.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": MODEL, "digest": OTHER}]})
        assert request.url.path == "/api/pull"
        if failure == "http":
            return httpx.Response(503, text="private-upstream-secret")
        if failure == "stream":
            return httpx.Response(200, text='{"error":"private-upstream-secret"}\n')
        return httpx.Response(
            200,
            text='{"status":"'
            + ("pulling manifest" if failure == "truncated" else "success")
            + '"}\n',
        )

    with (
        httpx.Client(transport=httpx.MockTransport(server)) as client,
        pytest.raises(module().BootstrapError),
    ):
        module().provision(tmp_path, client=client)
    assert not (tmp_path / ".model-provisioned.yaml").exists()
    assert set(calls) <= {"/api/tags", "/api/pull"}
    # The server owns partial layer resumption. Re-running makes the same pull.
    if failure == "http":
        with (
            httpx.Client(transport=httpx.MockTransport(server)) as client,
            pytest.raises(module().BootstrapError),
        ):
            module().provision(tmp_path, client=client)
        assert calls.count("/api/pull") == 2


def test_external_provider_skips_all_local_model_calls(tmp_path):
    config(tmp_path, provider="openai", digest=None)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: pytest.fail("External mode must not contact local Ollama")
        )
    ) as client:
        assert module().provision(tmp_path, client=client) is False
    assert not (tmp_path / ".model-provisioned.yaml").exists()


def test_expected_digest_is_required_and_cannot_be_overridden_with_conflict(tmp_path):
    config(tmp_path, digest=None)
    with pytest.raises(module().BootstrapError):
        module().provision(tmp_path)
    config(tmp_path)
    with pytest.raises(module().BootstrapError):
        module().provision(tmp_path, expected_digest=OTHER)


@pytest.mark.parametrize("expected", ["1" * 64, "sha256:" + "1" * 63, "sha256:" + "g" * 64])
@pytest.mark.parametrize("persistent", [False, True])
def test_expected_release_digest_stays_strict_before_any_model_call(tmp_path, expected, persistent):
    config(tmp_path, digest=expected if persistent else None)
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: pytest.fail("Invalid release identity contacted Ollama")
            )
        ) as client,
        pytest.raises(module().BootstrapError),
    ):
        module().provision(
            tmp_path, expected_digest=None if persistent else expected, client=client
        )
    assert not (tmp_path / ".model-provisioned.yaml").exists()


@pytest.mark.parametrize(
    "wire_digest", [None, "", "1" * 63, "1" * 65, "g" * 64, "sha256:" + "1" * 63]
)
def test_invalid_installed_wire_digest_stays_incomplete_without_pull(tmp_path, wire_digest):
    config(tmp_path)
    requests = []

    def server(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(200, json={"models": [{"name": MODEL, "digest": wire_digest}]})

    with (
        httpx.Client(transport=httpx.MockTransport(server)) as client,
        pytest.raises(module().BootstrapError),
    ):
        module().provision(tmp_path, client=client)
    assert requests == [("GET", "/api/tags")]
    assert not (tmp_path / ".model-provisioned.yaml").exists()


def test_conflicting_installed_identities_stay_incomplete_without_pull(tmp_path):
    config(tmp_path)
    requests = []

    def server(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(
            200,
            json={
                "models": [{"name": MODEL, "digest": "1" * 64}, {"model": MODEL, "digest": OTHER}]
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(server)) as client,
        pytest.raises(module().BootstrapError),
    ):
        module().provision(tmp_path, client=client)
    assert requests == [("GET", "/api/tags")]
    assert not (tmp_path / ".model-provisioned.yaml").exists()


def test_full_bare_digest_mismatch_is_rechecked_after_pull_and_never_published(tmp_path):
    config(tmp_path)
    requests = []

    def server(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": MODEL, "digest": "2" * 64}]})
        assert request.url.path == "/api/pull"
        return httpx.Response(200, text='{"status":"success"}\n')

    with (
        httpx.Client(transport=httpx.MockTransport(server)) as client,
        pytest.raises(module().BootstrapError),
    ):
        module().provision(tmp_path, client=client, bundled_model=MODEL)
    assert requests == [("GET", "/api/tags"), ("POST", "/api/pull"), ("GET", "/api/tags")]
    assert not (tmp_path / ".model-provisioned.yaml").exists()


@pytest.mark.parametrize("release_digest", [None, DIGEST, "invalid-release-digest"])
@pytest.mark.parametrize("expected_digest", [None, OTHER])
def test_custom_model_on_bundled_host_skips_before_digest_validation_or_model_calls(
    tmp_path, release_digest, expected_digest
):
    config(tmp_path, model="operator-selected:custom", digest=release_digest)
    path = tmp_path / "config.yaml"
    original = path.read_bytes()
    marker = tmp_path / ".model-provisioned.yaml"
    marker.write_text(yaml.safe_dump({"model": MODEL, "digest": DIGEST}))
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: pytest.fail("Custom model must not be contacted"))
    ) as client:
        assert (
            module().provision(
                tmp_path, bundled_model=MODEL, expected_digest=expected_digest, client=client
            )
            is False
        )
    assert path.read_bytes() == original
    assert not marker.exists()


def test_legacy_manual_provision_without_bundled_identity_can_verify_custom_model(tmp_path):
    custom_model = "operator-selected:custom"
    config(tmp_path, model=custom_model, digest=OTHER)
    path = tmp_path / "config.yaml"
    original = path.read_bytes()
    requests = []

    def server(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(200, json={"models": [{"name": custom_model, "digest": OTHER}]})

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        assert module().provision(tmp_path, client=client) is True
    assert requests == [("GET", "/api/tags")]
    assert path.read_bytes() == original
    assert yaml.safe_load((tmp_path / ".model-provisioned.yaml").read_text()) == {
        "model": custom_model,
        "digest": OTHER,
        "host": "http://ollama:11434",
    }


@pytest.mark.parametrize("bundled_model", ["", " ", " " + MODEL])
def test_invalid_bundled_model_cannot_silently_skip_default_verification(tmp_path, bundled_model):
    config(tmp_path)
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: pytest.fail("Invalid bundled identity contacted Ollama")
            )
        ) as client,
        pytest.raises(module().BootstrapError),
    ):
        module().provision(tmp_path, bundled_model=bundled_model, client=client)
    assert not (tmp_path / ".model-provisioned.yaml").exists()


def test_model_init_cli_custom_model_skip_needs_no_release_digest(tmp_path, monkeypatch):
    config(tmp_path, model="operator-selected:custom", digest=None)
    path = tmp_path / "config.yaml"
    original = path.read_bytes()
    monkeypatch.setattr(
        "sys.argv", ["model-init", "--config-dir", str(tmp_path), "--bundled-model", MODEL]
    )
    monkeypatch.setattr(
        httpx, "Client", lambda **_: pytest.fail("Custom model CLI made a network client")
    )
    assert module().main() == 0
    assert path.read_bytes() == original
    assert not (tmp_path / ".model-provisioned.yaml").exists()


def test_old_completion_marker_never_hides_removed_or_changed_model(tmp_path):
    config(tmp_path)
    marker = tmp_path / ".model-provisioned.yaml"
    marker.write_text(
        yaml.safe_dump({"model": MODEL, "digest": DIGEST, "host": "http://ollama:11434"})
    )
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))) as client,
        pytest.raises(module().BootstrapError),
    ):
        module().provision(tmp_path, client=client)
    assert not marker.exists()


def test_advanced_external_ollama_endpoint_skips_bundled_provisioning(tmp_path):
    config(tmp_path, digest=None)
    path = tmp_path / "config.yaml"
    data = yaml.safe_load(path.read_text())
    data["llm"]["host"] = "https://operator-managed.example"
    path.write_text(yaml.safe_dump(data))
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: pytest.fail("External Ollama must not be managed by bundled provisioning")
        )
    ) as client:
        assert module().provision(tmp_path, client=client) is False
