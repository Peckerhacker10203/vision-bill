from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import quote, urlsplit

import yaml
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    Field,
    PostgresDsn,
    TypeAdapter,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

logger = logging.getLogger(__name__)

CONFIG_PATH_ENV = "VISION_BILL_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path("/app/config/config.yaml")

EDITABLE_ENV_KEYS = {
    "llm.provider": "LLM__PROVIDER",
    "llm.host": "LLM__HOST",
    "llm.model_name": "LLM__MODEL_NAME",
    "llm.temperature": "LLM__TEMPERATURE",
    "auth.allow_registration": "AUTH__ALLOW_REGISTRATION",
}

_startup_llm_identity: tuple[LLMProviderEnum, str] | None = None


def get_config_path() -> Path:
    """Return the external YAML path used for persistent application settings."""
    return Path(os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH)).expanduser()


class LLMProviderEnum(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"


class OcrProviderEnum(str, Enum):
    PADDLEOCR = "paddleocr"


class LLMSettings(BaseModel):
    provider: LLMProviderEnum = LLMProviderEnum.OLLAMA
    host: str  # Used for local LLMs
    api_key: str  # Used for external LLMs
    model_name: str
    temperature: float
    # Keep the complete extraction deadline below the 15-minute image claim lease.
    analysis_timeout_seconds: float = Field(default=600, gt=0, le=600)
    max_output_tokens: int = Field(default=8192, gt=0)
    vision_model_ids: list[str] = Field(default_factory=list)
    # OpenAI-only option; other providers keep their own lifecycle after a switch.
    model_lifecycle: Literal["none", "llamacpp"] = "none"

    @field_validator("provider", mode="before")
    @classmethod
    def supported_provider(cls, value: Any) -> Any:
        if value not in ("ollama", "openai"):
            raise ValueError(
                "Unsupported LLM provider; select ollama or openai in LLM__PROVIDER or llm.provider"
            )
        return value

    @field_validator("host", "model_name")
    @classmethod
    def explicit_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("LLM host and model_name must be explicitly selected and nonblank")
        return value


class ApiSettings(BaseModel):
    port: int = 8080
    log_level: str = "INFO"


class ImageSettings(BaseModel):
    save_dir: str = "/app/uploads/"
    tmp_dir: str = "/app/uploads_tmp/"
    bypass_review_default: bool = False
    thumbnail_max_edge: int = Field(default=512, ge=64)
    thumbnail_quality: int = Field(default=80, ge=1, le=100)


class WorkerSettings(BaseModel):
    """Background analysis worker tuning."""

    check_interval_seconds: int = Field(default=300, ge=1)
    attempt_timeout_seconds: int = Field(default=600, gt=0, le=600)
    max_attempts: int = Field(default=3, ge=1, le=3)
    retry_backoff_seconds: tuple[Annotated[int, Field(gt=0)], ...] = (30, 60)
    retry_backoff_cap_seconds: int = Field(default=300, gt=0)
    claim_lease_seconds: int = Field(default=900, gt=0)

    @model_validator(mode="after")
    def validate_budgets(self) -> Self:
        if self.claim_lease_seconds <= self.attempt_timeout_seconds:
            raise ValueError("claim lease must exceed the complete attempt deadline")
        if len(self.retry_backoff_seconds) < self.max_attempts - 1:
            raise ValueError("a backoff is required for each automatic retry")
        return self


class OcrSettings(BaseModel):
    """OCR service settings for the two-stage extraction pipeline."""

    provider: OcrProviderEnum = OcrProviderEnum.PADDLEOCR
    enabled: bool = False
    service_url: str = "http://paddleocr-service:8082"
    timeout_seconds: int = Field(default=300, ge=1)
    required: bool = False
    allow_vision_fallback: bool = False
    vision_fallback_model_id: str | None = None
    coordination_path: Path = Field(
        default_factory=lambda: get_config_path().parent / "gpu-handoff.lock"
    )
    manage_model_lifecycle: bool = True

    @model_validator(mode="before")
    @classmethod
    def canonical_endpoint(cls, value: Any) -> Any:
        """An explicitly supplied endpoint selects OCR; legacy URLs do not."""
        if isinstance(value, dict) and "endpoint" in value:
            value = dict(value)
            endpoint = value.pop("endpoint")
            if not isinstance(endpoint, str):
                raise ValueError("OCR endpoint must be a string")
            value["service_url"] = endpoint.strip()
            if value["service_url"]:
                parsed = urlsplit(value["service_url"])
                if parsed.scheme not in ("http", "https") or not parsed.netloc:
                    raise ValueError("OCR endpoint must be an absolute HTTP(S) URL")
                AnyHttpUrl(value["service_url"])
            value["enabled"] = bool(value["service_url"])
        return value

    @model_validator(mode="after")
    def required_needs_enabled(self) -> Self:
        if self.required and not self.enabled:
            raise ValueError("required OCR must also be enabled")
        return self


class PGSettings(BaseModel):
    user: str
    password: str
    db: str = "vision_bill"
    host: str = "localhost"
    port: int = 5432

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> PostgresDsn:
        return PostgresDsn(self.pg_dsn.replace("postgresql://", "postgresql+asyncpg://", 1))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pg_dsn(self) -> str:
        """Plain DSN suitable for asyncpg.create_pool."""
        host = self.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"postgresql://{quote(self.user, safe='')}:{quote(self.password, safe='')}@{host}:{self.port}/{quote(self.db, safe='')}"


class AuthSettings(BaseModel):
    """Multi-user authentication settings.

    ``secret_key`` is required and signs the stateless session cookie. The
    optional ``pepper`` salts passwords before hashing; when unset the
    ``secret_key`` is used as the pepper, so a pepper is always in play.
    """

    secret_key: str
    session_cookie_name: str = "vb_session"
    session_max_age_seconds: int = 1_209_600  # 14 days
    session_secure: bool = False
    bootstrap_username: str | None = None
    bootstrap_password: str | None = None
    allow_registration: bool = True
    admin_can_see_all: bool = False
    pepper: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_nested_delimiter="__",
        yaml_file=str(get_config_path()),
        extra="ignore",
    )

    api: ApiSettings = ApiSettings()
    images: ImageSettings = ImageSettings()
    llm: LLMSettings
    pg: PGSettings
    worker: WorkerSettings = WorkerSettings()
    auth: AuthSettings
    ocr: OcrSettings = OcrSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load YAML before dotenv/environment values.

        Pydantic Settings gives earlier sources precedence, so environment
        variables and dotenv files override the persistent YAML file.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def _read_yaml_config(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, returning an empty mapping when it is absent."""
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError):
        logger.warning("Could not read YAML config %s", path)
        return {}
    return data if isinstance(data, dict) else {}


def _dotenv_names(path: Path) -> set[str]:
    """Return setting names present in a dotenv file without reading values."""
    if not path.is_file():
        return set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()

    names: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name = stripped.removeprefix("export ").split("=", 1)[0].strip()
        if name in EDITABLE_ENV_KEYS.values():
            names.add(name)
    return names


def editable_setting_sources() -> dict[str, str]:
    """Return the source of each setting shown in the admin UI."""
    environment_names = set(os.environ)
    environment_names.update(_dotenv_names(Path(".env")))
    environment_names.update(_dotenv_names(Path(".env.local")))
    config_data = _read_yaml_config(get_config_path())
    sources: dict[str, str] = {}

    for field_name, environment_name in EDITABLE_ENV_KEYS.items():
        if environment_name in environment_names:
            sources[field_name] = "environment"
            continue
        section, key = field_name.split(".", 1)
        section_data = config_data.get(section)
        sources[field_name] = (
            "config" if isinstance(section_data, dict) and key in section_data else "default"
        )
    return sources


def write_config_file(config: Settings, *, changed_paths: frozenset[str] | None = None) -> None:
    """Merge a save under the bootstrap lock without discarding private metadata."""
    from .distribution.bootstrap import BootstrapError, config_lock, read_config, write_private

    path = get_config_path()
    resolved = config.model_dump(mode="json", exclude={"pg": {"database_url", "pg_dsn"}})
    if changed_paths is not None and not changed_paths <= EDITABLE_ENV_KEYS.keys():
        raise BootstrapError("Only editable setting paths may be selectively saved")

    def merge(original: dict[str, Any], values: dict[str, Any]) -> None:
        for key, value in values.items():
            if isinstance(value, dict):
                if key not in original:
                    original[key] = {}
                if not isinstance(original[key], dict):
                    raise BootstrapError("Persistent configuration has a conflicting section type")
                merge(original[key], value)
            else:
                original[key] = value

    with config_lock(path.parent):
        pending = path.parent / ".bootstrap-pending"
        if pending.exists() or pending.is_symlink():
            raise BootstrapError("Interrupted installation requires explicit recovery")
        existing = path.exists() or path.is_symlink()
        if not existing and any(
            marker.exists() or marker.is_symlink()
            for marker in (
                path.parent / ".bootstrap-complete.yaml",
                path.parent / "initial-admin.yaml",
            )
        ):
            raise BootstrapError("Incomplete installation requires explicit recovery")
        data = read_config(path) if existing else resolved
        if existing:
            # Validate the persisted document itself, independent of environment overrides.
            for section, model_field in Settings.model_fields.items():
                if section in data:
                    TypeAdapter(model_field.annotation).validate_python(data[section])
            if changed_paths is None:
                merge(data, resolved)
            else:
                for field in sorted(changed_paths):
                    section, key = field.split(".", 1)
                    section_data = data.setdefault(section, {})
                    if not isinstance(section_data, dict):
                        raise BootstrapError(
                            "Persistent configuration has a conflicting section type"
                        )
                    section_data[key] = resolved[section][key]
        write_private(path, yaml.safe_dump(data, sort_keys=False), replace=existing)


def initialize_config_file(config: Settings) -> None:
    """Create the external config on first start from resolved env/default values."""
    path = get_config_path()
    if path.exists():
        return
    try:
        write_config_file(config)
        logger.info("Created initial YAML config at %s", path)
    except OSError as exc:
        # Unit tests and read-only deployments may intentionally omit the
        # config volume; the application can still run from environment values.
        logger.warning("Could not create YAML config at %s: %s", path, exc)


def mark_startup_settings(config: Settings) -> None:
    """Record which provider/host the running process actually initialized."""
    global _startup_llm_identity
    _startup_llm_identity = (config.llm.provider, config.llm.host)


def runtime_restart_required(config: Settings) -> bool:
    """Whether the editable LLM provider/host differs from the live provider."""
    return _startup_llm_identity is not None and _startup_llm_identity != (
        config.llm.provider,
        config.llm.host,
    )


# Required fields (llm, pg and auth) are populated from YAML, dotenv files, or
# environment variables. The first start writes the resolved values to YAML.
settings = Settings()  # type: ignore[call-arg]
initialize_config_file(settings)
