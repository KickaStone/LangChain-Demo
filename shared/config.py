"""Load LLM settings from the project root .env file."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from shared.providers import DEFAULT_PROVIDER_ID, Provider, get_provider

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LLMSettings:
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.7
    provider_id: str = DEFAULT_PROVIDER_ID


def load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def parse_model_list(raw: str | None, fallback: str) -> list[str]:
    if not raw:
        return [fallback]
    models = [m.strip() for m in raw.split(",") if m.strip()]
    return models or [fallback]


def resolve_api_key(provider: Provider) -> str:
    candidates = [provider.api_key_env, "OPENAI_API_KEY", "LLM_API_KEY"]
    seen: set[str] = set()
    for env_name in candidates:
        if env_name in seen:
            continue
        seen.add(env_name)
        value = os.getenv(env_name)
        if value:
            return value

    if provider.api_key_optional:
        return "ollama"

    print(
        f"Error: API key not found for provider '{provider.id}'.\n"
        f"Set {provider.api_key_env} (or OPENAI_API_KEY / LLM_API_KEY) in {PROJECT_ROOT / '.env'}",
        file=sys.stderr,
    )
    sys.exit(1)


def build_settings(provider: Provider) -> tuple[LLMSettings, list[str]]:
    base_url = os.getenv("OPENAI_BASE_URL") or provider.base_url
    default_model = os.getenv("OPENAI_MODEL") or provider.default_model

    if os.getenv("OPENAI_MODELS"):
        available_models = parse_model_list(os.getenv("OPENAI_MODELS"), default_model)
    else:
        available_models = list(provider.models)

    if default_model not in available_models:
        available_models.insert(0, default_model)

    settings = LLMSettings(
        model=default_model,
        api_key=resolve_api_key(provider),
        base_url=base_url,
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        provider_id=provider.id,
    )
    return settings, available_models


def load_settings() -> tuple[LLMSettings, list[str], Provider]:
    load_env()
    provider_id = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER_ID)
    try:
        provider = get_provider(provider_id)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    settings, available_models = build_settings(provider)
    return settings, available_models, provider


def create_chat_model(settings: LLMSettings):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=settings.temperature,
    )
