from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import AnyHttpUrl, BaseModel, Field, TypeAdapter, ValidationError

from ...config import (
    EDITABLE_ENV_KEYS,
    LLMProviderEnum,
    editable_setting_sources,
    has_real_api_key,
    llm_origin,
    runtime_restart_required,
    settings,
    write_config_file,
)
from ...model.dedicated_ocr import DedicatedOcrRequest, DedicatedOcrStatus
from ...security.dependencies import require_admin

router = APIRouter(tags=["System"])


class LLMSettingsUpdate(BaseModel):
    provider: LLMProviderEnum
    host: str = Field(min_length=1, pattern=r"\S")
    model_name: str = Field(min_length=1, pattern=r"\S")
    temperature: float = Field(ge=0, le=2)


class SettingsUpdate(BaseModel):
    llm: LLMSettingsUpdate
    allow_registration: bool
    changed_paths: frozenset[str] | None = Field(default=None, exclude=True)


class SettingsView(SettingsUpdate):
    sources: dict[str, str]
    restart_required: bool


@router.get(
    "/dedicated-ocr", response_model=DedicatedOcrStatus, dependencies=[Depends(require_admin)]
)
async def dedicated_ocr_status(request: Request) -> DedicatedOcrStatus:
    activation = getattr(request.app.state, "dedicated_ocr", None)
    if activation is None:
        return DedicatedOcrStatus(state="unavailable", reason="gpu_profile_required")
    result: DedicatedOcrStatus = await activation.status()
    return result


@router.post(
    "/dedicated-ocr", status_code=202, response_model=DedicatedOcrStatus,
    dependencies=[Depends(require_admin)],
)
async def activate_dedicated_ocr(
    request: Request, body: DedicatedOcrRequest,
) -> DedicatedOcrStatus:
    activation = getattr(request.app.state, "dedicated_ocr", None)
    if activation is None:
        raise HTTPException(status_code=409, detail="gpu_profile_required")
    result: DedicatedOcrStatus = await activation.submit(
        confirm_handoff=body.confirm_handoff, enabled=body.enabled,
    )
    if result.state in {"unavailable", "action_required"}:
        raise HTTPException(status_code=409, detail=result.reason)
    return result


@router.get("/ui-config")
async def get_ui_config() -> dict[str, bool | int]:
    """Return safe runtime defaults needed by the browser UI (public, pre-auth)."""
    return {
        "bypass_review_default": settings.images.bypass_review_default,
        "registration_open": settings.auth.allow_registration,
        "ocr_enabled": settings.ocr.enabled,
        "ocr_required": settings.ocr.required,
        "ocr_allow_vision_fallback": settings.ocr.allow_vision_fallback,
        "formatter_max_output_tokens": settings.llm.max_output_tokens,
    }


def _settings_view() -> SettingsView:
    return SettingsView(
        llm=LLMSettingsUpdate(
            provider=settings.llm.provider,
            host=settings.llm.host,
            model_name=settings.llm.model_name,
            temperature=settings.llm.temperature,
        ),
        allow_registration=settings.auth.allow_registration,
        sources=editable_setting_sources(),
        restart_required=runtime_restart_required(settings),
    )


@router.get("/settings", response_model=SettingsView, dependencies=[Depends(require_admin)])
async def get_settings() -> SettingsView:
    """Return editable settings without exposing credentials or secrets."""
    return _settings_view()


@router.put("/settings", response_model=SettingsView, dependencies=[Depends(require_admin)])
async def update_settings(request: Request, update: SettingsUpdate) -> SettingsView:
    """Persist admin-editable settings and apply safe changes immediately."""
    requested = {
        "llm.provider": update.llm.provider,
        "llm.host": update.llm.host,
        "llm.model_name": update.llm.model_name,
        "llm.temperature": update.llm.temperature,
        "auth.allow_registration": update.allow_registration,
    }
    current = {
        "llm.provider": settings.llm.provider,
        "llm.host": settings.llm.host,
        "llm.model_name": settings.llm.model_name,
        "llm.temperature": settings.llm.temperature,
        "auth.allow_registration": settings.auth.allow_registration,
    }
    selected = update.changed_paths if update.changed_paths is not None else frozenset(requested)
    if not selected <= EDITABLE_ENV_KEYS.keys():
        raise HTTPException(status_code=422, detail="Unsupported setting path")
    changed = frozenset(field for field in selected if requested[field] != current[field])
    if "auth.allow_registration" in changed and editable_setting_sources()["auth.allow_registration"] == "environment":
        raise HTTPException(status_code=409, detail="Registration is controlled by environment configuration")
    credential_origin = None
    if "llm.host" in changed:
        try:
            parsed = TypeAdapter(AnyHttpUrl).validate_python(update.llm.host)
        except ValidationError:
            raise HTTPException(status_code=422, detail="Host must be an HTTP(S) URL") from None
        if parsed.username or parsed.password:
            raise HTTPException(status_code=422, detail="Host must not contain credentials")
        credential_origin = llm_origin(settings.llm.host)
        if has_real_api_key(settings.llm.api_key) and llm_origin(update.llm.host) != credential_origin:
            raise HTTPException(status_code=409, detail="Endpoint origin is bound to private credentials; ask your operator to approve the new endpoint")

    candidate = settings.model_copy(deep=True)
    for field in changed:
        section, key = field.split(".", 1)
        setattr(getattr(candidate, section), key, requested[field])
    try:
        write_config_file(
            candidate,
            changed_paths=changed,
            credential_origin=credential_origin,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Could not write application config") from exc

    settings.llm.provider = candidate.llm.provider
    settings.llm.host = candidate.llm.host
    settings.llm.model_name = candidate.llm.model_name
    settings.llm.temperature = candidate.llm.temperature
    settings.auth.allow_registration = candidate.auth.allow_registration
    provider = getattr(request.app.state, "llm_provider", None)
    if provider is not None:
        provider.update_runtime_settings(temperature=settings.llm.temperature)
    return _settings_view()


# @router.post("/compare-models")
# async def compare_models_endpoint(prompt: str, models: list[str] | None = None):
#    results = await compare_models(prompt, models)
#    return {"results": [r.__dict__ for r in results]}
