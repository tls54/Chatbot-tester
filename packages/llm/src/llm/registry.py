"""Build a client from config. Adapters are imported lazily so unused SDKs aren't required."""

from __future__ import annotations

import os
from collections.abc import Callable

from dotenv import load_dotenv

from llm.base import LLMClient, LLMError
from llm.types import LLMConfig

Factory = Callable[[LLMConfig], LLMClient]
_FACTORIES: dict[str, Factory] = {}


def register(provider: str, factory: Factory) -> None:
    """Register (or override) a provider. New adapters plug in here."""
    _FACTORIES[provider] = factory


def _groq(config: LLMConfig) -> LLMClient:
    from llm.groq_client import GroqClient

    return GroqClient(config)


register("groq", _groq)


def create_client(config: LLMConfig) -> LLMClient:
    try:
        factory = _FACTORIES[config.provider]
    except KeyError:
        raise LLMError(
            f"Unknown LLM provider {config.provider!r}. Known: {sorted(_FACTORIES)}"
        ) from None
    return factory(config)


def config_from_env() -> LLMConfig:
    """Load .env, then build a config from LLM_PROVIDER / LLM_MODEL."""
    load_dotenv()
    provider, model = os.environ.get("LLM_PROVIDER", ""), os.environ.get("LLM_MODEL", "")
    if not provider or not model:
        raise LLMError("Set LLM_PROVIDER and LLM_MODEL in .env")
    return LLMConfig(provider=provider, model=model)
