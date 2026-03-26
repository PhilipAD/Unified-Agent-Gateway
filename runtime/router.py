from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Type

from config.settings import AgentProfile, GatewaySettings, ProviderSettings
from providers.anthropic import AnthropicProvider
from providers.base import BaseProvider
from providers.deepseek import DeepSeekProvider
from providers.gemini import GeminiProvider
from providers.groq import GroqProvider
from providers.mistral import MistralProvider
from providers.openai_compatible import OpenAICompatibleProvider
from providers.openai_responses import OpenAIResponsesProvider
from providers.xai import XAIProvider

PROVIDERS: Dict[str, Type[BaseProvider]] = {
    "openai_compatible": OpenAICompatibleProvider,
    "openai_responses": OpenAIResponsesProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "deepseek": DeepSeekProvider,
    "mistral": MistralProvider,
    "xai": XAIProvider,
}


@dataclass
class ProviderConfig:
    provider_name: str
    api_key: str
    model: str
    base_url: Optional[str] = None


def resolve_agent_profile(
    gateway_settings: GatewaySettings,
    agent_id: str = "default",
    profile: str = "default",
) -> AgentProfile:
    """Return the ``AgentProfile`` for a given *agent_id* / *profile* pair.

    Lookup order:
    1. ``{agent_id}:{profile}`` — e.g. ``"mybot:fast"``
    2. ``{profile}``
    3. ``"default"``
    """
    for key in (f"{agent_id}:{profile}", profile, "default"):
        if key in gateway_settings.AGENT_PROFILES:
            return gateway_settings.AGENT_PROFILES[key]
    return AgentProfile()


def resolve_provider_config(
    provider_settings: ProviderSettings,
    gateway_settings: GatewaySettings,
    agent_id: str = "default",
    profile: str = "default",
) -> ProviderConfig:
    """Resolve a ``ProviderConfig`` from agent profile and env-based settings.

    Supports:
    - Built-in providers: ``openai_compatible``, ``anthropic``, ``gemini``
    - Named OAI-compatible providers defined in ``OPENAI_COMPATIBLE_PROVIDERS``
      (e.g. ``groq``, ``deepseek``, ``together``)
    """
    agent_profile = resolve_agent_profile(gateway_settings, agent_id, profile)
    provider_name = agent_profile.provider_name

    # Check named OAI-compatible presets first
    if provider_name in gateway_settings.OPENAI_COMPATIBLE_PROVIDERS:
        preset = gateway_settings.OPENAI_COMPATIBLE_PROVIDERS[provider_name]
        return ProviderConfig(
            provider_name="openai_compatible",
            api_key=agent_profile.api_key or preset.api_key,
            model=agent_profile.model or preset.model,
            base_url=agent_profile.base_url or preset.base_url,
        )

    defaults: Dict[str, dict] = {
        "openai_compatible": {
            "api_key": provider_settings.OPENAI_API_KEY or "",
            "model": provider_settings.DEFAULT_OPENAI_MODEL,
            "base_url": provider_settings.OPENAI_BASE_URL,
        },
        "openai_responses": {
            "api_key": provider_settings.OPENAI_API_KEY or "",
            "model": provider_settings.DEFAULT_OPENAI_MODEL,
            "base_url": provider_settings.OPENAI_BASE_URL,
        },
        "anthropic": {
            "api_key": provider_settings.ANTHROPIC_API_KEY or "",
            "model": provider_settings.DEFAULT_ANTHROPIC_MODEL,
            "base_url": None,
        },
        "gemini": {
            "api_key": provider_settings.GOOGLE_API_KEY or "",
            "model": provider_settings.DEFAULT_GEMINI_MODEL,
            "base_url": None,
        },
        "groq": {
            "api_key": provider_settings.GROQ_API_KEY or "",
            "model": provider_settings.DEFAULT_GROQ_MODEL,
            "base_url": provider_settings.GROQ_BASE_URL,
        },
        "deepseek": {
            "api_key": provider_settings.DEEPSEEK_API_KEY or "",
            "model": provider_settings.DEFAULT_DEEPSEEK_MODEL,
            "base_url": provider_settings.DEEPSEEK_BASE_URL,
        },
        "mistral": {
            "api_key": provider_settings.MISTRAL_API_KEY or "",
            "model": provider_settings.DEFAULT_MISTRAL_MODEL,
            "base_url": provider_settings.MISTRAL_BASE_URL,
        },
        "xai": {
            "api_key": provider_settings.XAI_API_KEY or "",
            "model": provider_settings.DEFAULT_XAI_MODEL,
            "base_url": provider_settings.XAI_BASE_URL,
        },
    }

    cfg = defaults.get(provider_name, {})
    return ProviderConfig(
        provider_name=provider_name,
        api_key=agent_profile.api_key or cfg.get("api_key", ""),
        model=agent_profile.model or cfg.get("model", ""),
        base_url=agent_profile.base_url or cfg.get("base_url"),
    )


def merge_provider_config_overrides(
    cfg: ProviderConfig,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ProviderConfig:
    """Apply optional per-request overrides. ``None`` means keep *cfg* value."""
    return ProviderConfig(
        provider_name=cfg.provider_name,
        api_key=api_key if api_key is not None else cfg.api_key,
        model=model if model is not None else cfg.model,
        base_url=base_url if base_url is not None else cfg.base_url,
    )


def create_provider(cfg: ProviderConfig) -> BaseProvider:
    """Instantiate a provider from its config."""
    if cfg.provider_name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{cfg.provider_name}'. Available: {list(PROVIDERS.keys())}"
        )
    cls = PROVIDERS[cfg.provider_name]
    return cls(api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url)
