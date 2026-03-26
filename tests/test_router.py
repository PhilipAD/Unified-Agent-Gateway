import pytest

from config.settings import (
    GatewaySettings,
    OAICompatibleProviderPreset,
    ProviderSettings,
)
from providers.anthropic import AnthropicProvider
from providers.gemini import GeminiProvider
from providers.openai_compatible import OpenAICompatibleProvider
from runtime.router import (
    ProviderConfig,
    create_provider,
    merge_provider_config_overrides,
    resolve_agent_profile,
    resolve_provider_config,
)


def test_create_provider_openai():
    cfg = ProviderConfig(provider_name="openai_compatible", api_key="k", model="m")
    p = create_provider(cfg)
    assert isinstance(p, OpenAICompatibleProvider)


def test_create_provider_anthropic():
    cfg = ProviderConfig(provider_name="anthropic", api_key="k", model="m")
    p = create_provider(cfg)
    assert isinstance(p, AnthropicProvider)


def test_create_provider_gemini():
    cfg = ProviderConfig(provider_name="gemini", api_key="k", model="m")
    p = create_provider(cfg)
    assert isinstance(p, GeminiProvider)


def test_create_provider_unknown():
    cfg = ProviderConfig(provider_name="nope", api_key="k", model="m")
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider(cfg)


def test_resolve_default_profile():
    ps = ProviderSettings(OPENAI_API_KEY="sk-test")
    gs = GatewaySettings()
    cfg = resolve_provider_config(ps, gs)
    assert cfg.provider_name == "openai_compatible"
    assert cfg.api_key == "sk-test"
    assert cfg.model == "gpt-4o"


def test_resolve_custom_profile():
    ps = ProviderSettings(ANTHROPIC_API_KEY="ant-key")
    gs = GatewaySettings(
        AGENT_PROFILES={
            "default": {"provider_name": "openai_compatible"},
            "deep": {"provider_name": "anthropic", "model": "claude-sonnet-4-20250514"},
        }
    )
    cfg = resolve_provider_config(ps, gs, profile="deep")
    assert cfg.provider_name == "anthropic"
    assert cfg.model == "claude-sonnet-4-20250514"
    assert cfg.api_key == "ant-key"


def test_resolve_named_oai_compatible_provider():
    ps = ProviderSettings()
    gs = GatewaySettings(
        OPENAI_COMPATIBLE_PROVIDERS={
            "groq": OAICompatibleProviderPreset(
                api_key="gsk-123",
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.3-70b-versatile",
            )
        },
        AGENT_PROFILES={"default": {"provider_name": "groq"}},
    )
    cfg = resolve_provider_config(ps, gs)
    assert cfg.provider_name == "openai_compatible"
    assert cfg.api_key == "gsk-123"
    assert cfg.base_url == "https://api.groq.com/openai/v1"
    assert cfg.model == "llama-3.3-70b-versatile"


def test_profile_api_key_overrides_preset():
    ps = ProviderSettings()
    gs = GatewaySettings(
        OPENAI_COMPATIBLE_PROVIDERS={
            "groq": OAICompatibleProviderPreset(
                api_key="gsk-default",
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.3-70b-versatile",
            )
        },
        AGENT_PROFILES={"default": {"provider_name": "groq", "api_key": "gsk-override"}},
    )
    cfg = resolve_provider_config(ps, gs)
    assert cfg.api_key == "gsk-override"


def test_resolve_agent_profile_agent_id_lookup():
    gs = GatewaySettings(
        AGENT_PROFILES={
            "default": {"provider_name": "openai_compatible"},
            "mybot:fast": {"provider_name": "anthropic", "model": "claude-haiku-3-5"},
        }
    )
    profile = resolve_agent_profile(gs, agent_id="mybot", profile="fast")
    assert profile.provider_name == "anthropic"
    assert profile.model == "claude-haiku-3-5"


def test_resolve_agent_profile_falls_back_to_default():
    gs = GatewaySettings(AGENT_PROFILES={"default": {"provider_name": "gemini"}})
    profile = resolve_agent_profile(gs, agent_id="unknown", profile="nonexistent")
    assert profile.provider_name == "gemini"


def test_merge_provider_config_overrides_partial():
    base = ProviderConfig(
        provider_name="openai_compatible",
        api_key="old",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
    )
    only_key = merge_provider_config_overrides(base, api_key="new-key")
    assert only_key.api_key == "new-key"
    assert only_key.model == "gpt-4o"
    assert only_key.base_url == "https://api.openai.com/v1"

    only_model = merge_provider_config_overrides(base, model="gpt-4o-mini")
    assert only_model.api_key == "old"
    assert only_model.model == "gpt-4o-mini"


def test_resolve_agent_profile_mcp_namespaces():
    gs = GatewaySettings(
        AGENT_PROFILES={
            "researcher": {
                "provider_name": "openai_compatible",
                "mcp_namespaces": ["search", "files"],
                "context_names": ["company_info"],
            }
        }
    )
    profile = resolve_agent_profile(gs, profile="researcher")
    assert profile.mcp_namespaces == ["search", "files"]
    assert profile.context_names == ["company_info"]
