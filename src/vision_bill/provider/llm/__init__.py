from typing import TypeGuard

OPENAI_CREDENTIALLESS_API_KEY = "none"
DEFAULT_FORMATTER_MAX_OUTPUT_TOKENS = 32_768
MAX_MODEL_ID_LENGTH = 1024


def valid_model_id(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= MAX_MODEL_ID_LENGTH and value.isprintable()
