"""Tests for config/settings.py models and env-var parsing."""

import json

from config.settings import (
    AgentProfile,
    GatewaySettings,
    MCPServerPreset,
    NamedContextPreset,
    OAICompatibleProviderPreset,
)

# ---------------------------------------------------------------------------
# AgentProfile
# ---------------------------------------------------------------------------


def test_agent_profile_defaults():
    p = AgentProfile()
    assert p.provider_name == "openai_compatible"
    assert p.model is None
    assert p.mcp_namespaces == []
    assert p.context_names == []


def test_agent_profile_with_presets():
    p = AgentProfile(
        provider_name="anthropic",
        model="claude-opus-4-5",
        mcp_namespaces=["search", "files"],
        context_names=["company_info"],
    )
    assert p.provider_name == "anthropic"
    assert p.mcp_namespaces == ["search", "files"]
    assert p.context_names == ["company_info"]


# ---------------------------------------------------------------------------
# MCPServerPreset
# ---------------------------------------------------------------------------


def test_mcp_server_preset_defaults():
    p = MCPServerPreset(url="http://mcp/mcp")
    assert p.transport == "streamable_http"
    assert p.headers == {}
    assert p.timeout_seconds == 30.0


def test_mcp_server_preset_sse():
    p = MCPServerPreset(url="http://mcp/sse", transport="sse", timeout_seconds=60)
    assert p.transport == "sse"
    assert p.timeout_seconds == 60.0


# ---------------------------------------------------------------------------
# NamedContextPreset
# ---------------------------------------------------------------------------


def test_named_context_preset_static():
    p = NamedContextPreset(mode="static", text="Hello world")
    assert p.text == "Hello world"
    assert p.url is None


def test_named_context_preset_http():
    p = NamedContextPreset(
        mode="http",
        url="http://kb/search",
        payload_template={"query": "{input}"},
        required=True,
        max_chars=2000,
    )
    assert p.url == "http://kb/search"
    assert p.required is True
    assert p.max_chars == 2000


# ---------------------------------------------------------------------------
# OAICompatibleProviderPreset
# ---------------------------------------------------------------------------


def test_oai_preset():
    p = OAICompatibleProviderPreset(
        api_key="gsk-123",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
    )
    assert p.base_url == "https://api.groq.com/openai/v1"


# ---------------------------------------------------------------------------
# GatewaySettings — JSON env var parsing
# ---------------------------------------------------------------------------


def test_gateway_settings_mcp_servers_from_json(monkeypatch):
    payload = json.dumps(
        {
            "search": {"url": "http://search-mcp/mcp", "transport": "streamable_http"},
            "files": {"url": "http://files-mcp/sse", "transport": "sse"},
        }
    )
    monkeypatch.setenv("MCP_SERVERS", payload)
    gs = GatewaySettings()
    assert "search" in gs.MCP_SERVERS
    assert gs.MCP_SERVERS["search"].url == "http://search-mcp/mcp"
    assert gs.MCP_SERVERS["files"].transport == "sse"


def test_gateway_settings_named_contexts_from_json(monkeypatch):
    payload = json.dumps(
        {
            "company_info": {"mode": "static", "text": "We are Acme Corp"},
            "faq": {"mode": "http", "url": "http://kb/search"},
        }
    )
    monkeypatch.setenv("NAMED_CONTEXTS", payload)
    gs = GatewaySettings()
    assert gs.NAMED_CONTEXTS["company_info"].text == "We are Acme Corp"
    assert gs.NAMED_CONTEXTS["faq"].url == "http://kb/search"


def test_gateway_settings_oai_providers_from_json(monkeypatch):
    payload = json.dumps(
        {
            "groq": {
                "api_key": "gsk-abc",
                "base_url": "https://api.groq.com/openai/v1",
                "model": "llama-3.3-70b-versatile",
            }
        }
    )
    monkeypatch.setenv("OPENAI_COMPATIBLE_PROVIDERS", payload)
    gs = GatewaySettings()
    assert "groq" in gs.OPENAI_COMPATIBLE_PROVIDERS
    assert gs.OPENAI_COMPATIBLE_PROVIDERS["groq"].model == "llama-3.3-70b-versatile"


def test_gateway_settings_agent_profiles_from_json(monkeypatch):
    payload = json.dumps(
        {
            "default": {"provider_name": "openai_compatible"},
            "fast": {"provider_name": "groq", "mcp_namespaces": ["search"]},
        }
    )
    monkeypatch.setenv("AGENT_PROFILES", payload)
    gs = GatewaySettings()
    assert isinstance(gs.AGENT_PROFILES["fast"], AgentProfile)
    assert gs.AGENT_PROFILES["fast"].mcp_namespaces == ["search"]


def test_gateway_settings_dict_coercion():
    # Dicts should be coerced to typed models without explicit JSON env var
    gs = GatewaySettings(
        MCP_SERVERS={"ns": {"url": "http://x/mcp"}},
        NAMED_CONTEXTS={"ctx": {"mode": "static", "text": "hi"}},
        AGENT_PROFILES={"default": {"provider_name": "anthropic"}},
    )
    assert isinstance(gs.MCP_SERVERS["ns"], MCPServerPreset)
    assert isinstance(gs.NAMED_CONTEXTS["ctx"], NamedContextPreset)
    assert isinstance(gs.AGENT_PROFILES["default"], AgentProfile)
